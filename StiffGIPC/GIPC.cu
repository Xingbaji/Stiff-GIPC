//
// GIPC.cu
// GIPC
//
// created by Kemeng Huang on 2022/12/01
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#include "GIPC.cuh"
#include <gipc/gipc.h>
#include "cuda_tools/cuda_tools.h"
#include "GIPC_PDerivative.cuh"
#include "fem_parameters.h"
#include "ACCD.cuh"
#include "femEnergy.cuh"
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/device_ptr.h>
#include "FrictionUtils.cuh"
#include <fstream>
#include "Eigen/Eigen"
#include <gipc/statistics.h>
#include <gipc_path.h>
#include <gipc/utils/timer.h>
#include "ipc_collision.cuh"              // Collision & friction kernel declarations
#include "barrier_gradient_hessian.cuh"   // Barrier gradient/hessian kernel declarations

#include <tbb/parallel_for.h>

#include <muda/cub/device/device_radix_sort.h>
using namespace Eigen;
#define RANK 2
#define NEWF

template <typename Scalar, int size>
__device__ __host__ void makePDGeneral(Eigen::Matrix<Scalar, size, size>& symMtr)
{
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Scalar, size, size>> eigen_solver;

    if constexpr(size <= 3)
        eigen_solver.computeDirect(symMtr);
    else
        eigen_solver.compute(symMtr);
    Eigen::Vector<Scalar, size> eigen_values = eigen_solver.eigenvalues();
    Eigen::Matrix<Scalar, size, size> eigen_vectors = eigen_solver.eigenvectors();


    if(eigen_values[0] >= 0.0)
    {
        return;
    }

    for(int i = 0; i < size; ++i)
    {
        if(eigen_values(i) < 0)
        {
            eigen_values(i) = 0;
        }
    }
    symMtr = eigen_vectors * eigen_values.asDiagonal() * eigen_vectors.transpose();
}

template <typename Scalar, int size>
__device__ __host__ void makePD(Eigen::Matrix<Scalar, size, size>& symMtr)
{
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Scalar, size, size>> eigenSolver(symMtr);
    if(eigenSolver.eigenvalues()[0] >= 0.0)
    {
        return;
    }
    Eigen::Matrix<Scalar, size, size> D;  //(eigenSolver.eigenvalues());
    D.setZero();
    int rows = size;  //((size == Eigen::Dynamic) ? symMtr.rows() : size);
    for(int i = 0; i < rows; i++)
    {
        if(eigenSolver.eigenvalues()[i] > 0.0)
        {
            D(i, i) = eigenSolver.eigenvalues()[i];
        }
    }
    symMtr = eigenSolver.eigenvectors() * D * eigenSolver.eigenvectors().transpose();
}

template <int ROWS, int COLS>
__device__ inline void write_triplet(Eigen::Matrix3d*    triplet_value,
                                     int*                row_ids,
                                     int*                col_ids,
                                     const unsigned int* index,
                                     const double        input[ROWS][COLS],
                                     const int&          offset)
{
    int rown = ROWS / 3;
    int coln = COLS / 3;
    for(int ii = 0; ii < rown; ii++)
    {
#ifdef SymGH
        int start = ii;
#else
        int start = 0;
#endif
        for(int jj = start; jj < coln; jj++)
        {

#ifdef SymGH
            int kk = ii * rown + jj - ii * (ii + 1) / 2;
#else
            int kk = ii * rown + jj;  // - ii * (ii + 1) / 2;
#endif
            int row = index[ii];
            int col = index[jj];
#ifdef SymGH
            if(row > col)
            {
                row_ids[offset + kk] = col;
                col_ids[offset + kk] = row;
                for(int iii = 0; iii < 3; iii++)
                {
                    for(int jjj = 0; jjj < 3; jjj++)
                    {
                        triplet_value[offset + kk](iii, jjj) =
                            input[jj * 3 + iii][ii * 3 + jjj];
                    }
                }
            }
            else
#endif
            {
                row_ids[offset + kk] = row;
                col_ids[offset + kk] = col;
                for(int iii = 0; iii < 3; iii++)
                {
                    for(int jjj = 0; jjj < 3; jjj++)
                    {
                        triplet_value[offset + kk](iii, jjj) =
                            input[ii * 3 + iii][jj * 3 + jjj];
                    }
                }
            }
        }
    }
}


__device__ __host__ inline uint32_t expand_bits(std::uint32_t v) noexcept
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

__device__ __host__ inline uint32_t hash_code(
    int type, double x, double y, double z, double resolution = 1024) noexcept
{
    x = std::min(std::max(x * resolution, 0.0), resolution - 1.0);
    y = std::min(std::max(y * resolution, 0.0), resolution - 1.0);
    z = std::min(std::max(z * resolution, 0.0), resolution - 1.0);


    //
    if(type == -1)
    {
        const uint32_t xx     = expand_bits(static_cast<uint32_t>(x));
        const uint32_t yy     = expand_bits(static_cast<uint32_t>(y));
        const uint32_t zz     = expand_bits(static_cast<uint32_t>(z));
        std::uint32_t  mchash = ((xx << 2) + (yy << 1) + zz);

        return mchash;
    }
    else if(type == 0)
    {
        return (((static_cast<uint32_t>(z) * 1024) + static_cast<uint32_t>(y)) * 1024)
               + static_cast<uint32_t>(x);
    }
    else if(type == 1)
    {
        return (((static_cast<uint32_t>(y) * 1024) + static_cast<uint32_t>(z)) * 1024)
               + static_cast<uint32_t>(x);
    }
    else if(type == 2)
    {
        return (((static_cast<uint32_t>(x) * 1024) + static_cast<uint32_t>(z)) * 1024)
               + static_cast<uint32_t>(y);
    }
    else if(type == 3)
    {
        return (((static_cast<uint32_t>(z) * 1024) + static_cast<uint32_t>(x)) * 1024)
               + static_cast<uint32_t>(y);
    }
    else if(type == 4)
    {
        return (((static_cast<uint32_t>(y) * 1024) + static_cast<uint32_t>(x)) * 1024)
               + static_cast<uint32_t>(z);
    }
    else
    {
        return (((static_cast<uint32_t>(x) * 1024) + static_cast<uint32_t>(y)) * 1024)
               + static_cast<uint32_t>(z);
    }
    //std::uint32_t mchash = (((static_cast<std::uint32_t>(z) * 1024) + static_cast<std::uint32_t>(y)) * 1024) + static_cast<std::uint32_t>(x);//((xx << 2) + (yy << 1) + zz);
    //return mchash;
}

__global__ void _partition_collision_triplets(const uint64_t* sort_hash,
                                              int*            abd_abd_offset,
                                              int*            abd_fem_offset,
                                              int*            fem_abd_offset,
                                              int*            fem_fem_offset,
                                              int             number)
{
    extern __shared__ int shared_hash[];
    unsigned int          idx = threadIdx.x + (blockDim.x * blockIdx.x);
    //if(idx == 0)
    //{
    //    *abd_abd_offset = -1;
    //    *abd_fem_offset = -1;
    //    *fem_abd_offset = -1;
    //    *fem_fem_offset = -1;
    //}
    int self_hash;
    if(idx < number)
    {
        self_hash                    = sort_hash[idx];
        shared_hash[threadIdx.x + 1] = self_hash;
        if(idx > 0 && threadIdx.x == 0)
        {
            shared_hash[0] = sort_hash[idx - 1];
        }
    }
    __syncthreads();
    if(idx < number)
    {
        int prior_hash = idx == 0 ? -1 : shared_hash[threadIdx.x];
        if(self_hash != prior_hash)
        {
            if(self_hash == 3)
            {
                *abd_abd_offset = idx;
            }
            else if(self_hash == 1)
            {
                *abd_fem_offset = idx;
            }
            else if(self_hash == 2)
            {
                *fem_abd_offset = idx;
            }
            else if(self_hash == 0)
            {
                *fem_fem_offset = idx;
            }
        }
    }
}

__global__ void _reorder_triplets(int*             row_ids_input,
                                  int*             col_ids_input,
                                  Eigen::Matrix3d* triplet_value_inpuit,
                                  int*             row_ids,
                                  int*             col_ids,
                                  Eigen::Matrix3d* triplet_value,
                                  const uint32_t*  sort_index,
                                  int              number)
{
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= number)
        return;
    row_ids[idx]       = row_ids_input[sort_index[idx]];
    col_ids[idx]       = col_ids_input[sort_index[idx]];
    triplet_value[idx] = triplet_value_inpuit[sort_index[idx]];
}


uint64_t GIPC::getHashCode(double3 p, uint32_t i)
{
    uint64_t code = hash_code(-1, p.x, p.y, p.z);
    return (code << 32) | i;
}

__global__ void _calcTetMChash(uint64_t*       _MChash,
                               const double3*  _vertexes,
                               uint4*          tets,
                               const AABB*     _MaxBv,
                               const uint32_t* sortMapVertIndex,
                               int             number)
{
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= number)
        return;

    tets[idx].x = sortMapVertIndex[tets[idx].x];
    tets[idx].y = sortMapVertIndex[tets[idx].y];
    tets[idx].z = sortMapVertIndex[tets[idx].z];
    tets[idx].w = sortMapVertIndex[tets[idx].w];

    double3 SceneSize = make_double3((*_MaxBv).upper.x - (*_MaxBv).lower.x,
                                     (*_MaxBv).upper.y - (*_MaxBv).lower.y,
                                     (*_MaxBv).upper.z - (*_MaxBv).lower.z);
    double3 centerP   = __GEIGEN__::__s_vec_multiply(
        __GEIGEN__::__add(
            __GEIGEN__::__add(_vertexes[tets[idx].x], _vertexes[tets[idx].y]),
            __GEIGEN__::__add(_vertexes[tets[idx].z], _vertexes[tets[idx].w])),
        0.25);
    double3 offset = make_double3(centerP.x - (*_MaxBv).lower.x,
                                  centerP.y - (*_MaxBv).lower.y,
                                  centerP.z - (*_MaxBv).lower.z);

    int type = 0;
    if(SceneSize.x > SceneSize.y && SceneSize.y > SceneSize.z)
    {
        type = 0;
    }
    else if(SceneSize.x > SceneSize.z && SceneSize.z > SceneSize.y)
    {
        type = 1;
    }
    else if(SceneSize.y > SceneSize.z && SceneSize.z > SceneSize.x)
    {
        type = 2;
    }
    else if(SceneSize.y > SceneSize.x && SceneSize.x > SceneSize.z)
    {
        type = 3;
    }
    else if(SceneSize.z > SceneSize.x && SceneSize.x > SceneSize.y)
    {
        type = 4;
    }
    else
    {
        type = 5;
    }

    //printf("%d   %f     %f     %f\n", offset.x, offset.y, offset.z);
    uint64_t mc32 = hash_code(type,
                              offset.x / SceneSize.x,
                              offset.y / SceneSize.y,
                              offset.z / SceneSize.z);
    uint64_t mc64 = ((mc32 << 32) | idx);
    //printf("morton code %d\n", mc64);
    _MChash[idx] = mc64;
}

__global__ void _updateTopology(uint4*          tets,
                                uint3*          tris,
                                const uint32_t* sortMapVertIndex,
                                int             traNumber,
                                int             triNumber)
{
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < traNumber)
    {

        tets[idx].x = sortMapVertIndex[tets[idx].x];
        tets[idx].y = sortMapVertIndex[tets[idx].y];
        tets[idx].z = sortMapVertIndex[tets[idx].z];
        tets[idx].w = sortMapVertIndex[tets[idx].w];
    }
    if(idx < triNumber)
    {
        tris[idx].x = sortMapVertIndex[tris[idx].x];
        tris[idx].y = sortMapVertIndex[tris[idx].y];
        tris[idx].z = sortMapVertIndex[tris[idx].z];
    }
}


__global__ void _updateVertexes(double3*                      o_vertexes,
                                const double3*                _vertexes,
                                double*                       tempM,
                                const double*                 mass,
                                __GEIGEN__::Matrix3x3d*       tempCons,
                                int*                          tempBtype,
                                const __GEIGEN__::Matrix3x3d* cons,
                                const int*                    bType,
                                const uint32_t*               sortIndex,
                                uint32_t*                     sortMapIndex,
                                int                           number)
{
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= number)
        return;
    o_vertexes[idx]              = _vertexes[sortIndex[idx]];
    tempM[idx]                   = mass[sortIndex[idx]];
    tempCons[idx]                = cons[sortIndex[idx]];
    sortMapIndex[sortIndex[idx]] = idx;
    tempBtype[idx]               = bType[sortIndex[idx]];
    //printf("original idx: %d        new idx: %d\n", sortIndex[idx], idx);
}

__global__ void _updateTetrahedras(uint4*                        o_tetrahedras,
                                   uint4*                        tetrahedras,
                                   double*                       tempV,
                                   const double*                 volum,
                                   __GEIGEN__::Matrix3x3d*       tempDmInverse,
                                   const __GEIGEN__::Matrix3x3d* dmInverse,
                                   const uint32_t*               sortTetIndex,
                                   const uint32_t* sortMapVertIndex,
                                   int             number)
{
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= number)
        return;
    //tetrahedras[idx].x = sortMapVertIndex[tetrahedras[idx].x];
    //tetrahedras[idx].y = sortMapVertIndex[tetrahedras[idx].y];
    //tetrahedras[idx].z = sortMapVertIndex[tetrahedras[idx].z];
    //tetrahedras[idx].w = sortMapVertIndex[tetrahedras[idx].w];
    o_tetrahedras[idx] = tetrahedras[sortTetIndex[idx]];
    tempV[idx]         = volum[sortTetIndex[idx]];
    tempDmInverse[idx] = dmInverse[sortTetIndex[idx]];
}

__global__ void _calcVertMChash(uint64_t* _MChash, const double3* _vertexes, const AABB* _MaxBv, int number)
{
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= number)
        return;
    double3 SceneSize = make_double3((*_MaxBv).upper.x - (*_MaxBv).lower.x,
                                     (*_MaxBv).upper.y - (*_MaxBv).lower.y,
                                     (*_MaxBv).upper.z - (*_MaxBv).lower.z);
    double3 centerP   = _vertexes[idx];
    double3 offset    = make_double3(centerP.x - (*_MaxBv).lower.x,
                                  centerP.y - (*_MaxBv).lower.y,
                                  centerP.z - (*_MaxBv).lower.z);
    int     type      = -1;
    if(type >= 0)
    {
        if(SceneSize.x > SceneSize.y && SceneSize.y > SceneSize.z)
        {
            type = 0;
        }
        else if(SceneSize.x > SceneSize.z && SceneSize.z > SceneSize.y)
        {
            type = 1;
        }
        else if(SceneSize.y > SceneSize.z && SceneSize.z > SceneSize.x)
        {
            type = 2;
        }
        else if(SceneSize.y > SceneSize.x && SceneSize.x > SceneSize.z)
        {
            type = 3;
        }
        else if(SceneSize.z > SceneSize.x && SceneSize.x > SceneSize.y)
        {
            type = 4;
        }
        else
        {
            type = 5;
        }
    }

    //printf("minSize %f     %f     %f\n", SceneSize.x, SceneSize.y, SceneSize.z);
    uint64_t mc32 = hash_code(type,
                              offset.x / SceneSize.x,
                              offset.y / SceneSize.y,
                              offset.z / SceneSize.z);
    uint64_t mc64 = ((mc32 << 32) | idx);
    //printf("morton code %lld\n", mc64);
    _MChash[idx] = mc64;
}

__global__ void _reduct_max_double3_to_double(const double3* _double3Dim, double* _double1Dim, int number)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];

    if(idx >= number)
        return;
    //int cfid = tid + CONFLICT_FREE_OFFSET(tid);
    double3 tempMove = _double3Dim[idx];

    double temp =
        std::max(std::max(abs(tempMove.x), abs(tempMove.y)), abs(tempMove.z));

    int warpTid = threadIdx.x % 32;
    int warpId  = (threadIdx.x >> 5);
    int warpNum;
    //int tidNum = 32;
    if(blockIdx.x == gridDim.x - 1)
    {
        //tidNum = numbers - idof;
        warpNum = ((number - idof + 31) >> 5);
    }
    else
    {
        warpNum = ((blockDim.x) >> 5);
    }
    for(int i = 1; i < 32; i = (i << 1))
    {
        double tempMin = __shfl_down_sync(0xffffffff, temp, i);
        temp           = std::max(temp, tempMin);
    }
    if(warpTid == 0)
    {
        tep[warpId] = temp;
    }
    __syncthreads();
    if(threadIdx.x >= warpNum)
        return;
    if(warpNum > 1)
    {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];

        //	warpNum = ((tidNum + 31) >> 5);
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            double tempMin = __shfl_down_sync(0xffffffff, temp, i);
            temp           = std::max(temp, tempMin);
        }
    }
    if(threadIdx.x == 0)
    {
        _double1Dim[blockIdx.x] = temp;
    }
}

__global__ void _reduct_min_double(double* _double1Dim, int number)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];

    if(idx >= number)
        return;
    //int cfid = tid + CONFLICT_FREE_OFFSET(tid);
    double temp = _double1Dim[idx];

    __threadfence();


    int warpTid = threadIdx.x % 32;
    int warpId  = (threadIdx.x >> 5);
    int warpNum;
    //int tidNum = 32;
    if(blockIdx.x == gridDim.x - 1)
    {
        //tidNum = numbers - idof;
        warpNum = ((number - idof + 31) >> 5);
    }
    else
    {
        warpNum = ((blockDim.x) >> 5);
    }
    for(int i = 1; i < 32; i = (i << 1))
    {
        double tempMin = __shfl_down_sync(0xffffffff, temp, i);
        temp           = std::min(temp, tempMin);
    }
    if(warpTid == 0)
    {
        tep[warpId] = temp;
    }
    __syncthreads();
    if(threadIdx.x >= warpNum)
        return;
    if(warpNum > 1)
    {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];

        //	warpNum = ((tidNum + 31) >> 5);
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            double tempMin = __shfl_down_sync(0xffffffff, temp, i);
            temp           = std::min(temp, tempMin);
        }
    }
    if(threadIdx.x == 0)
    {
        _double1Dim[blockIdx.x] = temp;
    }
}

__global__ void _reduct_M_double2(double2* _double2Dim, int number)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double2 sdata[];

    if(idx >= number)
        return;
    //int cfid = tid + CONFLICT_FREE_OFFSET(tid);
    double2 temp = _double2Dim[idx];

    __threadfence();


    int warpTid = threadIdx.x % 32;
    int warpId  = (threadIdx.x >> 5);
    int warpNum;
    //int tidNum = 32;
    if(blockIdx.x == gridDim.x - 1)
    {
        //tidNum = numbers - idof;
        warpNum = ((number - idof + 31) >> 5);
    }
    else
    {
        warpNum = ((blockDim.x) >> 5);
    }
    for(int i = 1; i < 32; i = (i << 1))
    {
        double tempMin = __shfl_down_sync(0xffffffff, temp.x, i);
        double tempMax = __shfl_down_sync(0xffffffff, temp.y, i);
        temp.x         = std::max(temp.x, tempMin);
        temp.y         = std::max(temp.y, tempMax);
    }
    if(warpTid == 0)
    {
        sdata[warpId] = temp;
    }
    __syncthreads();
    if(threadIdx.x >= warpNum)
        return;
    if(warpNum > 1)
    {
        //	tidNum = warpNum;
        temp = sdata[threadIdx.x];

        //	warpNum = ((tidNum + 31) >> 5);
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            double tempMin = __shfl_down_sync(0xffffffff, temp.x, i);
            double tempMax = __shfl_down_sync(0xffffffff, temp.y, i);
            temp.x         = std::max(temp.x, tempMin);
            temp.y         = std::max(temp.y, tempMax);
        }
    }
    if(threadIdx.x == 0)
    {
        _double2Dim[blockIdx.x] = temp;
    }
}

__global__ void _reduct_max_double(double* _double1Dim, int number)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];

    if(idx >= number)
        return;
    //int cfid = tid + CONFLICT_FREE_OFFSET(tid);
    double temp = _double1Dim[idx];

    __threadfence();


    int warpTid = threadIdx.x % 32;
    int warpId  = (threadIdx.x >> 5);
    int warpNum;
    //int tidNum = 32;
    if(blockIdx.x == gridDim.x - 1)
    {
        //tidNum = numbers - idof;
        warpNum = ((number - idof + 31) >> 5);
    }
    else
    {
        warpNum = ((blockDim.x) >> 5);
    }
    for(int i = 1; i < 32; i = (i << 1))
    {
        double tempMax = __shfl_down_sync(0xffffffff, temp, i);
        temp           = std::max(temp, tempMax);
    }
    if(warpTid == 0)
    {
        tep[warpId] = temp;
    }
    __syncthreads();
    if(threadIdx.x >= warpNum)
        return;
    if(warpNum > 1)
    {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];

        //	warpNum = ((tidNum + 31) >> 5);
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            double tempMax = __shfl_down_sync(0xffffffff, temp, i);
            temp           = std::max(temp, tempMax);
        }
    }
    if(threadIdx.x == 0)
    {
        _double1Dim[blockIdx.x] = temp;
    }
}

__device__ double __cal_Barrier_energy(const double3* _vertexes,
                                       const double3* _rest_vertexes,
                                       int4           MMCVIDI,
                                       double         _Kappa,
                                       double         _dHat)
{
    double dHat_sqrt = sqrt(_dHat);
    double dHat      = _dHat;
    double Kappa     = _Kappa;
    if(MMCVIDI.x >= 0)
    {
        if(MMCVIDI.w >= 0)
        {
            double dis;
            _d_EE(_vertexes[MMCVIDI.x],
                  _vertexes[MMCVIDI.y],
                  _vertexes[MMCVIDI.z],
                  _vertexes[MMCVIDI.w],
                  dis);
            double I5 = dis / dHat;

            double lenE = (dis - dHat);
#if (RANK == 1)
            return -Kappa * lenE * lenE * log(I5);
#elif (RANK == 2)
            return Kappa * lenE * lenE * log(I5) * log(I5);
#elif (RANK == 3)
            return -Kappa * lenE * lenE * log(I5) * log(I5) * log(I5);
#elif (RANK == 4)
            return Kappa * lenE * lenE * log(I5) * log(I5) * log(I5) * log(I5);
#elif (RANK == 5)
            return -Kappa * lenE * lenE * log(I5) * log(I5) * log(I5) * log(I5) * log(I5);
#elif (RANK == 6)
            return Kappa * lenE * lenE * log(I5) * log(I5) * log(I5) * log(I5)
                   * log(I5) * log(I5);
#endif
        }
        else
        {
            //return 0;
            MMCVIDI.w = -MMCVIDI.w - 1;
            double3 v0 =
                __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _vertexes[MMCVIDI.x]);
            double3 v1 =
                __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.z]);
            double c = __GEIGEN__::__norm(__GEIGEN__::__v_vec_cross(v0, v1)) /*/ __GEIGEN__::__norm(v0)*/;
            double I1 = c * c;
            if(I1 == 0)
                return 0;
            double dis;
            _d_EE(_vertexes[MMCVIDI.x],
                  _vertexes[MMCVIDI.y],
                  _vertexes[MMCVIDI.z],
                  _vertexes[MMCVIDI.w],
                  dis);
            double I2    = dis / dHat;
            double eps_x = _compute_epx(_rest_vertexes[MMCVIDI.x],
                                        _rest_vertexes[MMCVIDI.y],
                                        _rest_vertexes[MMCVIDI.z],
                                        _rest_vertexes[MMCVIDI.w]);
#if (RANK == 1)
            double Energy = Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1)
                            * -(dHat - dHat * I2) * (dHat - dHat * I2) * log(I2);
#elif (RANK == 2)
            double Energy =
                Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1)
                * (dHat - dHat * I2) * (dHat - dHat * I2) * log(I2) * log(I2);
#elif (RANK == 4)
            double Energy = Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1)
                            * (dHat - dHat * I2) * (dHat - dHat * I2) * log(I2)
                            * log(I2) * log(I2) * log(I2);
#elif (RANK == 6)
            double Energy = Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1)
                            * (dHat - dHat * I2) * (dHat - dHat * I2) * log(I2)
                            * log(I2) * log(I2) * log(I2) * log(I2) * log(I2);
#endif
            if(Energy < 0)
                printf("I am pee\n");
            return Energy;
        }
    }
    else
    {
        int v0I = -MMCVIDI.x - 1;
        if(MMCVIDI.z < 0)
        {
            if(MMCVIDI.y < 0)
            {
                MMCVIDI.y = -MMCVIDI.y - 1;
                MMCVIDI.z = -MMCVIDI.z - 1;
                MMCVIDI.w = -MMCVIDI.w - 1;
                MMCVIDI.x = v0I;

                double3 v0 =
                    __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.x]);
                double3 v1 =
                    __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.y]);
                double c = __GEIGEN__::__norm(__GEIGEN__::__v_vec_cross(v0, v1)) /*/ __GEIGEN__::__norm(v0)*/;
                double I1 = c * c;
                if(I1 == 0)
                    return 0;
                double dis;
                _d_PP(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], dis);
                double I2    = dis / dHat;
                double eps_x = _compute_epx(_rest_vertexes[MMCVIDI.x],
                                            _rest_vertexes[MMCVIDI.z],
                                            _rest_vertexes[MMCVIDI.y],
                                            _rest_vertexes[MMCVIDI.w]);
#if (RANK == 1)
                double Energy =
                    Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1)
                    * -(dHat - dHat * I2) * (dHat - dHat * I2) * log(I2);
#elif (RANK == 2)
                double Energy =
                    Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1)
                    * (dHat - dHat * I2) * (dHat - dHat * I2) * log(I2) * log(I2);
#elif (RANK == 4)
                double Energy =
                    Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1)
                    * (dHat - dHat * I2) * (dHat - dHat * I2) * log(I2)
                    * log(I2) * log(I2) * log(I2);
#elif (RANK == 6)
                double Energy =
                    Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1)
                    * (dHat - dHat * I2) * (dHat - dHat * I2) * log(I2)
                    * log(I2) * log(I2) * log(I2) * log(I2) * log(I2);
#endif
                if(Energy < 0)
                    printf("I am pp\n");
                return Energy;
            }
            else
            {
                double dis;
                _d_PP(_vertexes[v0I], _vertexes[MMCVIDI.y], dis);
                double I5 = dis / dHat;

                double lenE = (dis - dHat);
#if (RANK == 1)
                return -Kappa * lenE * lenE * log(I5);
#elif (RANK == 2)
                return Kappa * lenE * lenE * log(I5) * log(I5);
#elif (RANK == 3)
                return -Kappa * lenE * lenE * log(I5) * log(I5) * log(I5);
#elif (RANK == 4)
                return Kappa * lenE * lenE * log(I5) * log(I5) * log(I5) * log(I5);
#elif (RANK == 5)
                return -Kappa * lenE * lenE * log(I5) * log(I5) * log(I5)
                       * log(I5) * log(I5);
#elif (RANK == 6)
                return Kappa * lenE * lenE * log(I5) * log(I5) * log(I5)
                       * log(I5) * log(I5) * log(I5);
#endif
            }
        }
        else if(MMCVIDI.w < 0)
        {
            if(MMCVIDI.y < 0)
            {
                MMCVIDI.y = -MMCVIDI.y - 1;
                //MMCVIDI.z = -MMCVIDI.z - 1;
                MMCVIDI.w = -MMCVIDI.w - 1;
                MMCVIDI.x = v0I;

                double3 v0 =
                    __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.x]);
                double3 v1 =
                    __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.y]);
                double c = __GEIGEN__::__norm(__GEIGEN__::__v_vec_cross(v0, v1)) /*/ __GEIGEN__::__norm(v0)*/;
                double I1 = c * c;
                if(I1 == 0)
                    return 0;
                double dis;
                _d_PE(_vertexes[MMCVIDI.x],
                      _vertexes[MMCVIDI.y],
                      _vertexes[MMCVIDI.z],
                      dis);
                double I2    = dis / dHat;
                double eps_x = _compute_epx(_rest_vertexes[MMCVIDI.x],
                                            _rest_vertexes[MMCVIDI.w],
                                            _rest_vertexes[MMCVIDI.y],
                                            _rest_vertexes[MMCVIDI.z]);
#if (RANK == 1)
                double Energy =
                    Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1)
                    * -(dHat - dHat * I2) * (dHat - dHat * I2) * log(I2);
#elif (RANK == 2)
                double Energy =
                    Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1)
                    * (dHat - dHat * I2) * (dHat - dHat * I2) * log(I2) * log(I2);
#elif (RANK == 4)
                double Energy =
                    Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1)
                    * (dHat - dHat * I2) * (dHat - dHat * I2) * log(I2)
                    * log(I2) * log(I2) * log(I2);
#elif (RANK == 6)
                double Energy =
                    Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1)
                    * (dHat - dHat * I2) * (dHat - dHat * I2) * log(I2)
                    * log(I2) * log(I2) * log(I2) * log(I2) * log(I2);
#endif
                if(Energy < 0)
                    printf("I am ppe\n");
                return Energy;
            }
            else
            {
                double dis;
                _d_PE(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], dis);
                double I5 = dis / dHat;

                double lenE = (dis - dHat);
#if (RANK == 1)
                return -Kappa * lenE * lenE * log(I5);
#elif (RANK == 2)
                return Kappa * lenE * lenE * log(I5) * log(I5);
#elif (RANK == 3)
                return -Kappa * lenE * lenE * log(I5) * log(I5) * log(I5);
#elif (RANK == 4)
                return Kappa * lenE * lenE * log(I5) * log(I5) * log(I5) * log(I5);
#elif (RANK == 5)
                return -Kappa * lenE * lenE * log(I5) * log(I5) * log(I5)
                       * log(I5) * log(I5);
#elif (RANK == 6)
                return Kappa * lenE * lenE * log(I5) * log(I5) * log(I5)
                       * log(I5) * log(I5) * log(I5);
#endif
            }
        }
        else
        {
            double dis;
            _d_PT(_vertexes[v0I],
                  _vertexes[MMCVIDI.y],
                  _vertexes[MMCVIDI.z],
                  _vertexes[MMCVIDI.w],
                  dis);
            double I5 = dis / dHat;

            double lenE = (dis - dHat);
#if (RANK == 1)
            return -Kappa * lenE * lenE * log(I5);
#elif (RANK == 2)
            return Kappa * lenE * lenE * log(I5) * log(I5);
#elif (RANK == 3)
            return -Kappa * lenE * lenE * log(I5) * log(I5) * log(I5);
#elif (RANK == 4)
            return Kappa * lenE * lenE * log(I5) * log(I5) * log(I5) * log(I5);
#elif (RANK == 5)
            return -Kappa * lenE * lenE * log(I5) * log(I5) * log(I5) * log(I5) * log(I5);
#elif (RANK == 6)
            return Kappa * lenE * lenE * log(I5) * log(I5) * log(I5) * log(I5)
                   * log(I5) * log(I5);
#endif
        }
    }
}

__device__ bool segTriIntersect(const double3& ve0,
                                const double3& ve1,
                                const double3& vt0,
                                const double3& vt1,
                                const double3& vt2)
{

    //printf("check for tri and lines\n");

    __GEIGEN__::Matrix3x3d coefMtr;
    double3                col0 = __GEIGEN__::__minus(vt1, vt0);
    double3                col1 = __GEIGEN__::__minus(vt2, vt0);
    double3                col2 = __GEIGEN__::__minus(ve0, ve1);

    __GEIGEN__::__set_Mat_val_column(coefMtr, col0, col1, col2);

    double3 n = __GEIGEN__::__v_vec_cross(col0, col1);
    if(__GEIGEN__::__v_vec_dot(n, __GEIGEN__::__minus(ve0, vt0))
           * __GEIGEN__::__v_vec_dot(n, __GEIGEN__::__minus(ve1, vt0))
       > 0)
    {
        return false;
    }

    double det = __GEIGEN__::__Determiant(coefMtr);

    if(abs(det) < 1e-20)
    {
        return false;
    }

    __GEIGEN__::Matrix3x3d D1, D2, D3;
    double3                b = __GEIGEN__::__minus(ve0, vt0);

    __GEIGEN__::__set_Mat_val_column(D1, b, col1, col2);
    __GEIGEN__::__set_Mat_val_column(D2, col0, b, col2);
    __GEIGEN__::__set_Mat_val_column(D3, col0, col1, b);

    double uvt[3];
    uvt[0] = __GEIGEN__::__Determiant(D1) / det;
    uvt[1] = __GEIGEN__::__Determiant(D2) / det;
    uvt[2] = __GEIGEN__::__Determiant(D3) / det;

    if(uvt[0] >= 0.0 && uvt[1] >= 0.0 && uvt[0] + uvt[1] <= 1.0 && uvt[2] >= 0.0
       && uvt[2] <= 1.0)
    {
        return true;
    }
    else
    {
        return false;
    }
}

__device__ __host__ inline bool _overlap(const AABB& lhs, const AABB& rhs, const double& gapL) noexcept
{
    if((rhs.lower.x - lhs.upper.x) >= gapL || (lhs.lower.x - rhs.upper.x) >= gapL)
        return false;
    if((rhs.lower.y - lhs.upper.y) >= gapL || (lhs.lower.y - rhs.upper.y) >= gapL)
        return false;
    if((rhs.lower.z - lhs.upper.z) >= gapL || (lhs.lower.z - rhs.upper.z) >= gapL)
        return false;
    return true;
}

// _selfConstraintVal moved to ipc_barrier.cu

__device__ double _computeInjectiveStepSize_3d(const double3*  verts,
                                               const double3*  mv,
                                               const uint32_t& v0,
                                               const uint32_t& v1,
                                               const uint32_t& v2,
                                               const uint32_t& v3,
                                               double          ratio,
                                               double          errorRate)
{

    double x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4;
    double p1, p2, p3, p4, q1, q2, q3, q4, r1, r2, r3, r4;
    double a, b, c, d, t;


    x1 = verts[v0].x;
    x2 = verts[v1].x;
    x3 = verts[v2].x;
    x4 = verts[v3].x;

    y1 = verts[v0].y;
    y2 = verts[v1].y;
    y3 = verts[v2].y;
    y4 = verts[v3].y;

    z1 = verts[v0].z;
    z2 = verts[v1].z;
    z3 = verts[v2].z;
    z4 = verts[v3].z;

    p1 = -mv[v0].x;
    p2 = -mv[v1].x;
    p3 = -mv[v2].x;
    p4 = -mv[v3].x;

    q1 = -mv[v0].y;
    q2 = -mv[v1].y;
    q3 = -mv[v2].y;
    q4 = -mv[v3].y;

    r1 = -mv[v0].z;
    r2 = -mv[v1].z;
    r3 = -mv[v2].z;
    r4 = -mv[v3].z;

    a = -p1 * q2 * r3 + p1 * r2 * q3 + q1 * p2 * r3 - q1 * r2 * p3 - r1 * p2 * q3
        + r1 * q2 * p3 + p1 * q2 * r4 - p1 * r2 * q4 - q1 * p2 * r4 + q1 * r2 * p4
        + r1 * p2 * q4 - r1 * q2 * p4 - p1 * q3 * r4 + p1 * r3 * q4 + q1 * p3 * r4
        - q1 * r3 * p4 - r1 * p3 * q4 + r1 * q3 * p4 + p2 * q3 * r4 - p2 * r3 * q4
        - q2 * p3 * r4 + q2 * r3 * p4 + r2 * p3 * q4 - r2 * q3 * p4;
    b = -x1 * q2 * r3 + x1 * r2 * q3 + y1 * p2 * r3 - y1 * r2 * p3 - z1 * p2 * q3
        + z1 * q2 * p3 + x2 * q1 * r3 - x2 * r1 * q3 - y2 * p1 * r3
        + y2 * r1 * p3 + z2 * p1 * q3 - z2 * q1 * p3 - x3 * q1 * r2
        + x3 * r1 * q2 + y3 * p1 * r2 - y3 * r1 * p2 - z3 * p1 * q2 + z3 * q1 * p2
        + x1 * q2 * r4 - x1 * r2 * q4 - y1 * p2 * r4 + y1 * r2 * p4 + z1 * p2 * q4
        - z1 * q2 * p4 - x2 * q1 * r4 + x2 * r1 * q4 + y2 * p1 * r4 - y2 * r1 * p4
        - z2 * p1 * q4 + z2 * q1 * p4 + x4 * q1 * r2 - x4 * r1 * q2 - y4 * p1 * r2
        + y4 * r1 * p2 + z4 * p1 * q2 - z4 * q1 * p2 - x1 * q3 * r4 + x1 * r3 * q4
        + y1 * p3 * r4 - y1 * r3 * p4 - z1 * p3 * q4 + z1 * q3 * p4 + x3 * q1 * r4
        - x3 * r1 * q4 - y3 * p1 * r4 + y3 * r1 * p4 + z3 * p1 * q4 - z3 * q1 * p4
        - x4 * q1 * r3 + x4 * r1 * q3 + y4 * p1 * r3 - y4 * r1 * p3 - z4 * p1 * q3
        + z4 * q1 * p3 + x2 * q3 * r4 - x2 * r3 * q4 - y2 * p3 * r4 + y2 * r3 * p4
        + z2 * p3 * q4 - z2 * q3 * p4 - x3 * q2 * r4 + x3 * r2 * q4 + y3 * p2 * r4
        - y3 * r2 * p4 - z3 * p2 * q4 + z3 * q2 * p4 + x4 * q2 * r3 - x4 * r2 * q3
        - y4 * p2 * r3 + y4 * r2 * p3 + z4 * p2 * q3 - z4 * q2 * p3;
    c = -x1 * y2 * r3 + x1 * z2 * q3 + x1 * y3 * r2 - x1 * z3 * q2 + y1 * x2 * r3
        - y1 * z2 * p3 - y1 * x3 * r2 + y1 * z3 * p2 - z1 * x2 * q3
        + z1 * y2 * p3 + z1 * x3 * q2 - z1 * y3 * p2 - x2 * y3 * r1
        + x2 * z3 * q1 + y2 * x3 * r1 - y2 * z3 * p1 - z2 * x3 * q1 + z2 * y3 * p1
        + x1 * y2 * r4 - x1 * z2 * q4 - x1 * y4 * r2 + x1 * z4 * q2 - y1 * x2 * r4
        + y1 * z2 * p4 + y1 * x4 * r2 - y1 * z4 * p2 + z1 * x2 * q4 - z1 * y2 * p4
        - z1 * x4 * q2 + z1 * y4 * p2 + x2 * y4 * r1 - x2 * z4 * q1 - y2 * x4 * r1
        + y2 * z4 * p1 + z2 * x4 * q1 - z2 * y4 * p1 - x1 * y3 * r4 + x1 * z3 * q4
        + x1 * y4 * r3 - x1 * z4 * q3 + y1 * x3 * r4 - y1 * z3 * p4 - y1 * x4 * r3
        + y1 * z4 * p3 - z1 * x3 * q4 + z1 * y3 * p4 + z1 * x4 * q3 - z1 * y4 * p3
        - x3 * y4 * r1 + x3 * z4 * q1 + y3 * x4 * r1 - y3 * z4 * p1 - z3 * x4 * q1
        + z3 * y4 * p1 + x2 * y3 * r4 - x2 * z3 * q4 - x2 * y4 * r3 + x2 * z4 * q3
        - y2 * x3 * r4 + y2 * z3 * p4 + y2 * x4 * r3 - y2 * z4 * p3 + z2 * x3 * q4
        - z2 * y3 * p4 - z2 * x4 * q3 + z2 * y4 * p3 + x3 * y4 * r2 - x3 * z4 * q2
        - y3 * x4 * r2 + y3 * z4 * p2 + z3 * x4 * q2 - z3 * y4 * p2;
    d = (ratio)
        * (x1 * z2 * y3 - x1 * y2 * z3 + y1 * x2 * z3 - y1 * z2 * x3 - z1 * x2 * y3
           + z1 * y2 * x3 + x1 * y2 * z4 - x1 * z2 * y4 - y1 * x2 * z4 + y1 * z2 * x4
           + z1 * x2 * y4 - z1 * y2 * x4 - x1 * y3 * z4 + x1 * z3 * y4 + y1 * x3 * z4
           - y1 * z3 * x4 - z1 * x3 * y4 + z1 * y3 * x4 + x2 * y3 * z4 - x2 * z3 * y4
           - y2 * x3 * z4 + y2 * z3 * x4 + z2 * x3 * y4 - z2 * y3 * x4);


    //printf("a b c d:   %f  %f  %f  %f     %f     %f,    id0, id1, id2, id3:  %d  %d  %d  %d\n", a, b, c, d, ratio, errorRate, v0, v1, v2, v3);
    if(abs(a) <= errorRate /** errorRate*/)
    {
        if(abs(b) <= errorRate /** errorRate*/)
        {
            if(false && abs(c) <= errorRate)
            {
                t = 1;
            }
            else
            {
                t = -d / c;
            }
        }
        else
        {
            double desc = c * c - 4 * b * d;
            if(desc > 0)
            {
                t = (-c - sqrt(desc)) / (2 * b);
                if(t < 0)
                    t = (-c + sqrt(desc)) / (2 * b);
            }
            else
                t = 1;
        }
    }
    else
    {
        //double results[3];
        //int number = 0;
        //__GEIGEN__::__NewtonSolverForCubicEquation(a, b, c, d, results, number, errorRate);

        //t = 1;
        //for (int index = 0;index < number;index++) {
        //    if (results[index] > 0 && results[index] < t) {
        //        t = results[index];
        //    }
        //}
        //zs::complex<double> i(0, 1);
        //zs::complex<double> delta0(b * b - 3 * a * c, 0);
        //zs::complex<double> delta1(2 * b * b * b - 9 * a * b * c + 27 * a * a * d, 0);
        //zs::complex<double> C =
        //    pow((delta1 + sqrt(delta1 * delta1 - 4.0 * delta0 * delta0 * delta0)) / 2.0,
        //        1.0 / 3.0);
        //if(abs(C) == 0.0)
        //{
        //    // a corner case listed by wikipedia found by our collaborate from another project
        //    C = pow((delta1 - sqrt(delta1 * delta1 - 4.0 * delta0 * delta0 * delta0)) / 2.0,
        //            1.0 / 3.0);
        //}

        //zs::complex<double> u2 = (-1.0 + sqrt(3.0) * i) / 2.0;
        //zs::complex<double> u3 = (-1.0 - sqrt(3.0) * i) / 2.0;

        //zs::complex<double> t1 = (b + C + delta0 / C) / (-3.0 * a);
        //zs::complex<double> t2 = (b + u2 * C + delta0 / (u2 * C)) / (-3.0 * a);
        //zs::complex<double> t3 = (b + u3 * C + delta0 / (u3 * C)) / (-3.0 * a);
        //t                      = -1;
        //if((abs(imag(t1)) < errorRate /** errorRate*/) && (real(t1) > 0))
        //    t = real(t1);
        //if((abs(imag(t2)) < errorRate /** errorRate*/) && (real(t2) > 0)
        //   && ((real(t2) < t) || (t < 0)))
        //    t = real(t2);
        //if((abs(imag(t3)) < errorRate /** errorRate*/) && (real(t3) > 0)
        //   && ((real(t3) < t) || (t < 0)))
        //    t = real(t3);
    }
    if(t <= 0)
        t = 1;
    return t;
}

__device__ double __cal_Friction_gd_energy(const double3* _vertexes,
                                           const double3* _o_vertexes,
                                           const double3* _normal,
                                           uint32_t       gidx,
                                           double         dt,
                                           double         lastH,
                                           double         eps)
{

    double3 normal = *_normal;
    double3 Vdiff  = __GEIGEN__::__minus(_vertexes[gidx], _o_vertexes[gidx]);
    double3 VProj  = __GEIGEN__::__minus(
        Vdiff, __GEIGEN__::__s_vec_multiply(normal, __GEIGEN__::__v_vec_dot(Vdiff, normal)));
    double VProjMag2 = __GEIGEN__::__squaredNorm(VProj);
    if(VProjMag2 > eps * eps)
    {
        return lastH * (sqrt(VProjMag2) - eps * 0.5);
    }
    else
    {
        return lastH * VProjMag2 / eps * 0.5;
    }
}


__device__ double __cal_Friction_energy(const double3*         _vertexes,
                                        const double3*         _o_vertexes,
                                        int4                   MMCVIDI,
                                        double                 dt,
                                        double2                distCoord,
                                        __GEIGEN__::Matrix3x2d tanBasis,
                                        double                 lastH,
                                        double                 fricDHat,
                                        double                 eps)
{
    double3 relDX3D;
    if(MMCVIDI.x >= 0)
    {
        if(MMCVIDI.w >= 0)
        {
            Friction::computeRelDX_EE(
                __GEIGEN__::__minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _o_vertexes[MMCVIDI.w]),
                distCoord.x,
                distCoord.y,
                relDX3D);
        }
    }
    else
    {
        int v0I = -MMCVIDI.x - 1;
        if(MMCVIDI.z < 0)
        {
            if(MMCVIDI.y >= 0)
            {
                Friction::computeRelDX_PP(
                    __GEIGEN__::__minus(_vertexes[v0I], _o_vertexes[v0I]),
                    __GEIGEN__::__minus(_vertexes[MMCVIDI.y],
                                        _o_vertexes[MMCVIDI.y]),
                    relDX3D);
            }
        }
        else if(MMCVIDI.w < 0)
        {
            if(MMCVIDI.y >= 0)
            {
                Friction::computeRelDX_PE(
                    __GEIGEN__::__minus(_vertexes[v0I], _o_vertexes[v0I]),
                    __GEIGEN__::__minus(_vertexes[MMCVIDI.y],
                                        _o_vertexes[MMCVIDI.y]),
                    __GEIGEN__::__minus(_vertexes[MMCVIDI.z],
                                        _o_vertexes[MMCVIDI.z]),
                    distCoord.x,
                    relDX3D);
            }
        }
        else
        {
            Friction::computeRelDX_PT(
                __GEIGEN__::__minus(_vertexes[v0I], _o_vertexes[v0I]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _o_vertexes[MMCVIDI.w]),
                distCoord.x,
                distCoord.y,
                relDX3D);
        }
    }
    __GEIGEN__::Matrix2x3d tB_T = __GEIGEN__::__Transpose3x2(tanBasis);
    double                 relDXSqNorm =
        __GEIGEN__::__squaredNorm(__GEIGEN__::__M2x3_v3_multiply(tB_T, relDX3D));
    if(relDXSqNorm > fricDHat)
    {
        return lastH * sqrt(relDXSqNorm);
    }
    else
    {
        double f0;
        Friction::f0_SF(relDXSqNorm, eps, f0);
        return lastH * f0;
    }
}

// _calFrictionHessian_gd moved to ipc_barrier.cu

__global__ void _calFrictionHessian(const double3*          _vertexes,
                                    const double3*          _o_vertexes,
                                    const int4*             _last_collisionPair,
                                    Eigen::Matrix3d*        triplet_values,
                                    int*                    row_ids,
                                    int*                    col_ids,
                                    uint32_t*               _cpNum,
                                    int                     number,
                                    double                  dt,
                                    double2*                distCoord,
                                    __GEIGEN__::Matrix3x2d* tanBasis,
                                    double                  eps2,
                                    double*                 lastH,
                                    double                  coef,
                                    int                     cd_offset4,
                                    int                     cd_offset3,
                                    int                     cd_offset2,
                                    int                     f_offset4,
                                    int                     f_offset3,
                                    int                     f_offset2)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    int4    MMCVIDI = _last_collisionPair[idx];
    double  eps     = sqrt(eps2);
    double3 relDX3D;
    int global_offset = cd_offset4 * M12_Off + cd_offset3 * M9_Off + cd_offset2 * M6_Off;
    if(MMCVIDI.x >= 0)
    {
        Friction::computeRelDX_EE(
            __GEIGEN__::__minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
            __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
            __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
            __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _o_vertexes[MMCVIDI.w]),
            distCoord[idx].x,
            distCoord[idx].y,
            relDX3D);


        __GEIGEN__::Matrix2x3d tB_T = __GEIGEN__::__Transpose3x2(tanBasis[idx]);
        double2 relDX       = __GEIGEN__::__M2x3_v3_multiply(tB_T, relDX3D);
        double  relDXSqNorm = __GEIGEN__::__squaredNorm(relDX);
        double  relDXNorm   = sqrt(relDXSqNorm);
        __GEIGEN__::Matrix12x2d T;
        Friction::computeT_EE(tanBasis[idx], distCoord[idx].x, distCoord[idx].y, T);
        __GEIGEN__::Matrix2x2d M2;
        if(relDXSqNorm > eps2)
        {
            __GEIGEN__::__set_Mat_identity(M2);
            M2.m[0][0] /= relDXNorm;
            M2.m[1][1] /= relDXNorm;
            M2 = __GEIGEN__::__Mat2x2_minus(
                M2,
                __GEIGEN__::__s_Mat2x2_multiply(__GEIGEN__::__v2_vec2_toMat2x2(relDX, relDX),
                                                1 / (relDXSqNorm * relDXNorm)));
        }
        else
        {
            double f1_div_relDXNorm;
            Friction::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
            double f2;
            Friction::f2_SF(relDXSqNorm, eps, f2);
            if(f2 != f1_div_relDXNorm && relDXSqNorm)
            {

                __GEIGEN__::__set_Mat_identity(M2);
                M2.m[0][0] *= f1_div_relDXNorm;
                M2.m[1][1] *= f1_div_relDXNorm;
                M2 = __GEIGEN__::__Mat2x2_minus(
                    M2,
                    __GEIGEN__::__s_Mat2x2_multiply(__GEIGEN__::__v2_vec2_toMat2x2(relDX, relDX),
                                                    (f1_div_relDXNorm - f2) / relDXSqNorm));
            }
            else
            {
                __GEIGEN__::__set_Mat_identity(M2);
                M2.m[0][0] *= f1_div_relDXNorm;
                M2.m[1][1] *= f1_div_relDXNorm;
            }
        }

        __GEIGEN__::Matrix2x2d projH;

        Matrix2d F_mat2;
        F_mat2 << M2.m[0][0], M2.m[0][1], M2.m[1][0], M2.m[1][1];
        makePDGeneral<double, 2>(F_mat2);
        projH.m[0][0] = F_mat2(0, 0);
        projH.m[0][1] = F_mat2(0, 1);
        projH.m[1][0] = F_mat2(1, 0);
        projH.m[1][1] = F_mat2(1, 1);


        __GEIGEN__::Matrix12x2d TM2 = __GEIGEN__::__M12x2_M2x2_Multiply(T, projH);

        __GEIGEN__::Matrix12x12d HessianBlock =
            __GEIGEN__::__s_M12x12_Multiply(__M12x2_M12x2T_Multiply(TM2, T),
                                            coef * lastH[idx]);
        int Hidx   = atomicAdd(_cpNum + 4, 1);
        int offset = global_offset + Hidx * M12_Off;
        //Hidx += cd_offset4;
        //H12x12[Hidx]  = HessianBlock;
        uint4 global_index = make_uint4(MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);
        //D4Index[Hidx] = global_index;

        write_triplet<12, 12>(
            triplet_values, row_ids, col_ids, &(global_index.x), HessianBlock.m, offset);
    }
    else
    {
        int v0I = -MMCVIDI.x - 1;
        if(MMCVIDI.z < 0)
        {

            MMCVIDI.x = v0I;
            Friction::computeRelDX_PP(
                __GEIGEN__::__minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
                relDX3D);

            __GEIGEN__::Matrix2x3d tB_T = __GEIGEN__::__Transpose3x2(tanBasis[idx]);
            double2 relDX       = __GEIGEN__::__M2x3_v3_multiply(tB_T, relDX3D);
            double  relDXSqNorm = __GEIGEN__::__squaredNorm(relDX);
            double  relDXNorm   = sqrt(relDXSqNorm);
            __GEIGEN__::Matrix6x2d T;
            Friction::computeT_PP(tanBasis[idx], T);
            __GEIGEN__::Matrix2x2d M2;
            if(relDXSqNorm > eps2)
            {
                __GEIGEN__::__set_Mat_identity(M2);
                M2.m[0][0] /= relDXNorm;
                M2.m[1][1] /= relDXNorm;
                M2 = __GEIGEN__::__Mat2x2_minus(
                    M2,
                    __GEIGEN__::__s_Mat2x2_multiply(__GEIGEN__::__v2_vec2_toMat2x2(relDX, relDX),
                                                    1 / (relDXSqNorm * relDXNorm)));
            }
            else
            {
                double f1_div_relDXNorm;
                Friction::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
                double f2;
                Friction::f2_SF(relDXSqNorm, eps, f2);
                if(f2 != f1_div_relDXNorm && relDXSqNorm)
                {

                    __GEIGEN__::__set_Mat_identity(M2);
                    M2.m[0][0] *= f1_div_relDXNorm;
                    M2.m[1][1] *= f1_div_relDXNorm;
                    M2 = __GEIGEN__::__Mat2x2_minus(
                        M2,
                        __GEIGEN__::__s_Mat2x2_multiply(
                            __GEIGEN__::__v2_vec2_toMat2x2(relDX, relDX),
                            (f1_div_relDXNorm - f2) / relDXSqNorm));
                }
                else
                {
                    __GEIGEN__::__set_Mat_identity(M2);
                    M2.m[0][0] *= f1_div_relDXNorm;
                    M2.m[1][1] *= f1_div_relDXNorm;
                }
            }
            __GEIGEN__::Matrix2x2d projH;
            Matrix2d               F_mat2;
            F_mat2 << M2.m[0][0], M2.m[0][1], M2.m[1][0], M2.m[1][1];
            makePDGeneral<double, 2>(F_mat2);
            projH.m[0][0] = F_mat2(0, 0);
            projH.m[0][1] = F_mat2(0, 1);
            projH.m[1][0] = F_mat2(1, 0);
            projH.m[1][1] = F_mat2(1, 1);

            __GEIGEN__::Matrix6x2d TM2 = __GEIGEN__::__M6x2_M2x2_Multiply(T, projH);

            __GEIGEN__::Matrix6x6d HessianBlock =
                __GEIGEN__::__s_M6x6_Multiply(__M6x2_M6x2T_Multiply(TM2, T),
                                              coef * lastH[idx]);

            int Hidx   = atomicAdd(_cpNum + 2, 1);
            int offset = global_offset + f_offset4 * M12_Off
                         + f_offset3 * M9_Off + Hidx * M6_Off;
            //Hidx += cd_offset2;
            //H6x6[Hidx]    = HessianBlock;
            uint2 global_index = make_uint2(MMCVIDI.x, MMCVIDI.y);
            //D2Index[Hidx]      = global_index;


            write_triplet<6, 6>(triplet_values,
                                row_ids,
                                col_ids,
                                &(global_index.x),
                                HessianBlock.m,
                                offset);
        }
        else if(MMCVIDI.w < 0)
        {

            MMCVIDI.x = v0I;
            Friction::computeRelDX_PE(
                __GEIGEN__::__minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
                distCoord[idx].x,
                relDX3D);

            __GEIGEN__::Matrix2x3d tB_T = __GEIGEN__::__Transpose3x2(tanBasis[idx]);
            double2 relDX       = __GEIGEN__::__M2x3_v3_multiply(tB_T, relDX3D);
            double  relDXSqNorm = __GEIGEN__::__squaredNorm(relDX);
            double  relDXNorm   = sqrt(relDXSqNorm);
            __GEIGEN__::Matrix9x2d T;
            Friction::computeT_PE(tanBasis[idx], distCoord[idx].x, T);
            __GEIGEN__::Matrix2x2d M2;
            if(relDXSqNorm > eps2)
            {
                __GEIGEN__::__set_Mat_identity(M2);
                M2.m[0][0] /= relDXNorm;
                M2.m[1][1] /= relDXNorm;
                M2 = __GEIGEN__::__Mat2x2_minus(
                    M2,
                    __GEIGEN__::__s_Mat2x2_multiply(__GEIGEN__::__v2_vec2_toMat2x2(relDX, relDX),
                                                    1 / (relDXSqNorm * relDXNorm)));
            }
            else
            {
                double f1_div_relDXNorm;
                Friction::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
                double f2;
                Friction::f2_SF(relDXSqNorm, eps, f2);
                if(f2 != f1_div_relDXNorm && relDXSqNorm)
                {

                    __GEIGEN__::__set_Mat_identity(M2);
                    M2.m[0][0] *= f1_div_relDXNorm;
                    M2.m[1][1] *= f1_div_relDXNorm;
                    M2 = __GEIGEN__::__Mat2x2_minus(
                        M2,
                        __GEIGEN__::__s_Mat2x2_multiply(
                            __GEIGEN__::__v2_vec2_toMat2x2(relDX, relDX),
                            (f1_div_relDXNorm - f2) / relDXSqNorm));
                }
                else
                {
                    __GEIGEN__::__set_Mat_identity(M2);
                    M2.m[0][0] *= f1_div_relDXNorm;
                    M2.m[1][1] *= f1_div_relDXNorm;
                }
            }
            __GEIGEN__::Matrix2x2d projH;
            Matrix2d               F_mat2;
            F_mat2 << M2.m[0][0], M2.m[0][1], M2.m[1][0], M2.m[1][1];
            makePDGeneral<double, 2>(F_mat2);
            projH.m[0][0] = F_mat2(0, 0);
            projH.m[0][1] = F_mat2(0, 1);
            projH.m[1][0] = F_mat2(1, 0);
            projH.m[1][1] = F_mat2(1, 1);

            __GEIGEN__::Matrix9x2d TM2 = __GEIGEN__::__M9x2_M2x2_Multiply(T, projH);

            __GEIGEN__::Matrix9x9d HessianBlock =
                __GEIGEN__::__s_M9x9_Multiply(__M9x2_M9x2T_Multiply(TM2, T),
                                              coef * lastH[idx]);
            int Hidx   = atomicAdd(_cpNum + 3, 1);
            int offset = global_offset + f_offset4 * M12_Off + Hidx * M9_Off;
            //Hidx += cd_offset3;
            //H9x9[Hidx]    = HessianBlock;
            uint3 global_index = make_uint3(v0I, MMCVIDI.y, MMCVIDI.z);
            //D3Index[Hidx]      = global_index;


            write_triplet<9, 9>(triplet_values,
                                row_ids,
                                col_ids,
                                &(global_index.x),
                                HessianBlock.m,
                                offset);
        }
        else
        {
            MMCVIDI.x = v0I;
            Friction::computeRelDX_PT(
                __GEIGEN__::__minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _o_vertexes[MMCVIDI.w]),
                distCoord[idx].x,
                distCoord[idx].y,
                relDX3D);


            __GEIGEN__::Matrix2x3d tB_T = __GEIGEN__::__Transpose3x2(tanBasis[idx]);
            double2 relDX       = __GEIGEN__::__M2x3_v3_multiply(tB_T, relDX3D);
            double  relDXSqNorm = __GEIGEN__::__squaredNorm(relDX);
            double  relDXNorm   = sqrt(relDXSqNorm);
            __GEIGEN__::Matrix12x2d T;
            Friction::computeT_PT(
                tanBasis[idx], distCoord[idx].x, distCoord[idx].y, T);
            __GEIGEN__::Matrix2x2d M2;
            if(relDXSqNorm > eps2)
            {
                __GEIGEN__::__set_Mat_identity(M2);
                M2.m[0][0] /= relDXNorm;
                M2.m[1][1] /= relDXNorm;
                M2 = __GEIGEN__::__Mat2x2_minus(
                    M2,
                    __GEIGEN__::__s_Mat2x2_multiply(__GEIGEN__::__v2_vec2_toMat2x2(relDX, relDX),
                                                    1 / (relDXSqNorm * relDXNorm)));
            }
            else
            {
                double f1_div_relDXNorm;
                Friction::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
                double f2;
                Friction::f2_SF(relDXSqNorm, eps, f2);
                if(f2 != f1_div_relDXNorm && relDXSqNorm)
                {

                    __GEIGEN__::__set_Mat_identity(M2);
                    M2.m[0][0] *= f1_div_relDXNorm;
                    M2.m[1][1] *= f1_div_relDXNorm;
                    M2 = __GEIGEN__::__Mat2x2_minus(
                        M2,
                        __GEIGEN__::__s_Mat2x2_multiply(
                            __GEIGEN__::__v2_vec2_toMat2x2(relDX, relDX),
                            (f1_div_relDXNorm - f2) / relDXSqNorm));
                }
                else
                {
                    __GEIGEN__::__set_Mat_identity(M2);
                    M2.m[0][0] *= f1_div_relDXNorm;
                    M2.m[1][1] *= f1_div_relDXNorm;
                }
            }
            __GEIGEN__::Matrix2x2d projH;
            Matrix2d               F_mat2;
            F_mat2 << M2.m[0][0], M2.m[0][1], M2.m[1][0], M2.m[1][1];
            makePDGeneral<double, 2>(F_mat2);
            projH.m[0][0] = F_mat2(0, 0);
            projH.m[0][1] = F_mat2(0, 1);
            projH.m[1][0] = F_mat2(1, 0);
            projH.m[1][1] = F_mat2(1, 1);

            __GEIGEN__::Matrix12x2d TM2 = __GEIGEN__::__M12x2_M2x2_Multiply(T, projH);

            __GEIGEN__::Matrix12x12d HessianBlock =
                __GEIGEN__::__s_M12x12_Multiply(__M12x2_M12x2T_Multiply(TM2, T),
                                                coef * lastH[idx]);
            int Hidx   = atomicAdd(_cpNum + 4, 1);
            int offset = global_offset + Hidx * M12_Off;
            //Hidx += cd_offset4;
            //H12x12[Hidx]  = HessianBlock;
            uint4 global_index = make_uint4(v0I, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);
            //D4Index[Hidx] = global_index;


            write_triplet<12, 12>(triplet_values,
                                  row_ids,
                                  col_ids,
                                  &(global_index.x),
                                  HessianBlock.m,
                                  offset);
        }
    }
}

template <typename T>
__global__ inline void moveMemory_1(T* data, int output_start, int input_start, int length)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= length)
        return;
    data[output_start + idx] = data[input_start + idx];
}

// _calBarrierHessian moved to ipc_barrier.cu

// _calBarrierGradientAndHessian moved to ipc_barrier.cu

// _calSelfCloseVal, _checkSelfCloseVal, _reduct_MSelfDist, _calFrictionGradient_gd moved to ipc_barrier.cu

__global__ void _calFrictionGradient(const double3*    _vertexes,
                                     const double3*    _o_vertexes,
                                     const int4* _last_collisionPair,
                                     double3*          _gradient,
                                     int               number,
                                     double            dt,
                                     double2*          distCoord,
                                     __GEIGEN__::Matrix3x2d* tanBasis,
                                     double                  eps2,
                                     double*                 lastH,
                                     double                  coef)
{
    double eps = std::sqrt(eps2);
    int    idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    int4    MMCVIDI = _last_collisionPair[idx];
    double3 relDX3D;
    if(MMCVIDI.x >= 0)
    {
        Friction::computeRelDX_EE(
            __GEIGEN__::__minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
            __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
            __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
            __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _o_vertexes[MMCVIDI.w]),
            distCoord[idx].x,
            distCoord[idx].y,
            relDX3D);

        __GEIGEN__::Matrix2x3d tB_T = __GEIGEN__::__Transpose3x2(tanBasis[idx]);
        double2 relDX       = __GEIGEN__::__M2x3_v3_multiply(tB_T, relDX3D);
        double  relDXSqNorm = __GEIGEN__::__squaredNorm(relDX);
        if(relDXSqNorm > eps2)
        {
            relDX = __GEIGEN__::__s_vec_multiply(relDX, 1.0 / sqrt(relDXSqNorm));
        }
        else
        {
            double f1_div_relDXNorm;
            Friction::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
            relDX = __GEIGEN__::__s_vec_multiply(relDX, f1_div_relDXNorm);
        }
        __GEIGEN__::Vector12 TTTDX;
        Friction::liftRelDXTanToMesh_EE(
            relDX, tanBasis[idx], distCoord[idx].x, distCoord[idx].y, TTTDX);
        TTTDX = __GEIGEN__::__s_vec12_multiply(TTTDX, lastH[idx] * coef);
        {
            atomicAdd(&(_gradient[MMCVIDI.x].x), TTTDX.v[0]);
            atomicAdd(&(_gradient[MMCVIDI.x].y), TTTDX.v[1]);
            atomicAdd(&(_gradient[MMCVIDI.x].z), TTTDX.v[2]);
            atomicAdd(&(_gradient[MMCVIDI.y].x), TTTDX.v[3]);
            atomicAdd(&(_gradient[MMCVIDI.y].y), TTTDX.v[4]);
            atomicAdd(&(_gradient[MMCVIDI.y].z), TTTDX.v[5]);
            atomicAdd(&(_gradient[MMCVIDI.z].x), TTTDX.v[6]);
            atomicAdd(&(_gradient[MMCVIDI.z].y), TTTDX.v[7]);
            atomicAdd(&(_gradient[MMCVIDI.z].z), TTTDX.v[8]);
            atomicAdd(&(_gradient[MMCVIDI.w].x), TTTDX.v[9]);
            atomicAdd(&(_gradient[MMCVIDI.w].y), TTTDX.v[10]);
            atomicAdd(&(_gradient[MMCVIDI.w].z), TTTDX.v[11]);
        }
    }
    else
    {
        int v0I = -MMCVIDI.x - 1;
        if(MMCVIDI.z < 0)
        {
            MMCVIDI.x = v0I;

            Friction::computeRelDX_PP(
                __GEIGEN__::__minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
                relDX3D);

            __GEIGEN__::Matrix2x3d tB_T = __GEIGEN__::__Transpose3x2(tanBasis[idx]);
            double2 relDX       = __GEIGEN__::__M2x3_v3_multiply(tB_T, relDX3D);
            double  relDXSqNorm = __GEIGEN__::__squaredNorm(relDX);
            if(relDXSqNorm > eps2)
            {
                relDX = __GEIGEN__::__s_vec_multiply(relDX, 1.0 / sqrt(relDXSqNorm));
            }
            else
            {
                double f1_div_relDXNorm;
                Friction::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
                relDX = __GEIGEN__::__s_vec_multiply(relDX, f1_div_relDXNorm);
            }

            __GEIGEN__::Vector6 TTTDX;
            Friction::liftRelDXTanToMesh_PP(relDX, tanBasis[idx], TTTDX);
            TTTDX = __GEIGEN__::__s_vec6_multiply(TTTDX, lastH[idx] * coef);
            {
                atomicAdd(&(_gradient[MMCVIDI.x].x), TTTDX.v[0]);
                atomicAdd(&(_gradient[MMCVIDI.x].y), TTTDX.v[1]);
                atomicAdd(&(_gradient[MMCVIDI.x].z), TTTDX.v[2]);
                atomicAdd(&(_gradient[MMCVIDI.y].x), TTTDX.v[3]);
                atomicAdd(&(_gradient[MMCVIDI.y].y), TTTDX.v[4]);
                atomicAdd(&(_gradient[MMCVIDI.y].z), TTTDX.v[5]);
            }
        }
        else if(MMCVIDI.w < 0)
        {
            MMCVIDI.x = v0I;
            Friction::computeRelDX_PE(
                __GEIGEN__::__minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
                distCoord[idx].x,
                relDX3D);

            __GEIGEN__::Matrix2x3d tB_T = __GEIGEN__::__Transpose3x2(tanBasis[idx]);
            double2 relDX       = __GEIGEN__::__M2x3_v3_multiply(tB_T, relDX3D);
            double  relDXSqNorm = __GEIGEN__::__squaredNorm(relDX);
            if(relDXSqNorm > eps2)
            {
                relDX = __GEIGEN__::__s_vec_multiply(relDX, 1.0 / sqrt(relDXSqNorm));
            }
            else
            {
                double f1_div_relDXNorm;
                Friction::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
                relDX = __GEIGEN__::__s_vec_multiply(relDX, f1_div_relDXNorm);
            }
            __GEIGEN__::Vector9 TTTDX;
            Friction::liftRelDXTanToMesh_PE(relDX, tanBasis[idx], distCoord[idx].x, TTTDX);
            TTTDX = __GEIGEN__::__s_vec9_multiply(TTTDX, lastH[idx] * coef);
            {
                atomicAdd(&(_gradient[MMCVIDI.x].x), TTTDX.v[0]);
                atomicAdd(&(_gradient[MMCVIDI.x].y), TTTDX.v[1]);
                atomicAdd(&(_gradient[MMCVIDI.x].z), TTTDX.v[2]);
                atomicAdd(&(_gradient[MMCVIDI.y].x), TTTDX.v[3]);
                atomicAdd(&(_gradient[MMCVIDI.y].y), TTTDX.v[4]);
                atomicAdd(&(_gradient[MMCVIDI.y].z), TTTDX.v[5]);
                atomicAdd(&(_gradient[MMCVIDI.z].x), TTTDX.v[6]);
                atomicAdd(&(_gradient[MMCVIDI.z].y), TTTDX.v[7]);
                atomicAdd(&(_gradient[MMCVIDI.z].z), TTTDX.v[8]);
            }
        }
        else
        {
            MMCVIDI.x = v0I;
            Friction::computeRelDX_PT(
                __GEIGEN__::__minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _o_vertexes[MMCVIDI.w]),
                distCoord[idx].x,
                distCoord[idx].y,
                relDX3D);

            __GEIGEN__::Matrix2x3d tB_T = __GEIGEN__::__Transpose3x2(tanBasis[idx]);
            double2 relDX = __GEIGEN__::__M2x3_v3_multiply(tB_T, relDX3D);

            double relDXSqNorm = __GEIGEN__::__squaredNorm(relDX);
            if(relDXSqNorm > eps2)
            {
                relDX = __GEIGEN__::__s_vec_multiply(relDX, 1.0 / sqrt(relDXSqNorm));
            }
            else
            {
                double f1_div_relDXNorm;
                Friction::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
                relDX = __GEIGEN__::__s_vec_multiply(relDX, f1_div_relDXNorm);
            }
            __GEIGEN__::Vector12 TTTDX;
            Friction::liftRelDXTanToMesh_PT(
                relDX, tanBasis[idx], distCoord[idx].x, distCoord[idx].y, TTTDX);
            TTTDX = __GEIGEN__::__s_vec12_multiply(TTTDX, lastH[idx] * coef);

            atomicAdd(&(_gradient[MMCVIDI.x].x), TTTDX.v[0]);
            atomicAdd(&(_gradient[MMCVIDI.x].y), TTTDX.v[1]);
            atomicAdd(&(_gradient[MMCVIDI.x].z), TTTDX.v[2]);
            atomicAdd(&(_gradient[MMCVIDI.y].x), TTTDX.v[3]);
            atomicAdd(&(_gradient[MMCVIDI.y].y), TTTDX.v[4]);
            atomicAdd(&(_gradient[MMCVIDI.y].z), TTTDX.v[5]);
            atomicAdd(&(_gradient[MMCVIDI.z].x), TTTDX.v[6]);
            atomicAdd(&(_gradient[MMCVIDI.z].y), TTTDX.v[7]);
            atomicAdd(&(_gradient[MMCVIDI.z].z), TTTDX.v[8]);
            atomicAdd(&(_gradient[MMCVIDI.w].x), TTTDX.v[9]);
            atomicAdd(&(_gradient[MMCVIDI.w].y), TTTDX.v[10]);
            atomicAdd(&(_gradient[MMCVIDI.w].z), TTTDX.v[11]);
        }
    }
}


// _calBarrierGradient moved to ipc_barrier.cu


__global__ void _calKineticGradient(
    double3* vertexes, double3* xTilta, double3* gradient, double* masses, int numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;
    double3 deltaX = __GEIGEN__::__minus(vertexes[idx], xTilta[idx]);
    //masses[idx] = 1;
    gradient[idx] = make_double3(
        deltaX.x * masses[idx], deltaX.y * masses[idx], deltaX.z * masses[idx]);
    //printf("%f  %f  %f\n", gradient[idx].x, gradient[idx].y, gradient[idx].z);
}

__global__ void _calKineticEnergy(
    double3* vertexes, double3* xTilta, double3* gradient, double* masses, int numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;
    double3 deltaX = __GEIGEN__::__minus(vertexes[idx], xTilta[idx]);
    gradient[idx]  = make_double3(
        deltaX.x * masses[idx], deltaX.y * masses[idx], deltaX.z * masses[idx]);
}

__global__ void _computeSoftConstraintGradientAndHessian(const double3* vertexes,
                                                         const double3* targetVert,
                                                         const uint32_t* targetInd,
                                                         double3*  gradient,
                                                         uint32_t* _gpNum,
                                                         Eigen::Matrix3d* triplet_values,
                                                         int*   row_ids,
                                                         int*   col_ids,
                                                         double motionRate,
                                                         double rate,
                                                         int    global_offset,
                                                         int global_hessian_fem_offset,
                                                         int number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    uint32_t vInd = targetInd[idx];
    double   x = vertexes[vInd].x, y = vertexes[vInd].y, z = vertexes[vInd].z,
           a = targetVert[idx].x, b = targetVert[idx].y, c = targetVert[idx].z;
    //double dis = __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(vertexes[vInd], targetVert[idx]));
    //printf("%f\n", dis);
    double d = motionRate;
    {
        atomicAdd(&(gradient[vInd].x), d * rate * rate * (x - a));
        atomicAdd(&(gradient[vInd].y), d * rate * rate * (y - b));
        atomicAdd(&(gradient[vInd].z), d * rate * rate * (z - c));
    }
    __GEIGEN__::Matrix3x3d Hpg;
    Hpg.m[0][0] = rate * rate * d;
    Hpg.m[0][1] = 0;
    Hpg.m[0][2] = 0;
    Hpg.m[1][0] = 0;
    Hpg.m[1][1] = rate * rate * d;
    Hpg.m[1][2] = 0;
    Hpg.m[2][0] = 0;
    Hpg.m[2][1] = 0;
    Hpg.m[2][2] = rate * rate * d;
    int pidx    = atomicAdd(_gpNum, 1);
    //H3x3[pidx]    = Hpg;
    //D1Index[pidx] = vInd;
    vInd += global_hessian_fem_offset;
    write_triplet<3, 3>(triplet_values, row_ids, col_ids, &vInd, Hpg.m, global_offset + idx);
    //_environment_collisionPair[atomicAdd(_gpNum, 1)] = surfVertIds[idx];
}

__global__ void _computeSoftConstraintGradient(const double3*  vertexes,
                                               const double3*  targetVert,
                                               const uint32_t* targetInd,
                                               double3*        gradient,
                                               double          motionRate,
                                               double          rate,
                                               int             number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    uint32_t vInd = targetInd[idx];
    double   x = vertexes[vInd].x, y = vertexes[vInd].y, z = vertexes[vInd].z,
           a = targetVert[idx].x, b = targetVert[idx].y, c = targetVert[idx].z;
    //double dis = __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(vertexes[vInd], targetVert[idx]));
    //printf("%f\n", dis);
    double d = motionRate;
    {
        atomicAdd(&(gradient[vInd].x), d * rate * rate * (x - a));
        atomicAdd(&(gradient[vInd].y), d * rate * rate * (y - b));
        atomicAdd(&(gradient[vInd].z), d * rate * rate * (z - c));
    }
}

// _GroundCollisionDetect moved to ipc_barrier.cu

__global__ void _getTotalForce(const double3* _force0, double3* _force, int number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    _force[idx].x += _force0[idx].x;
    _force[idx].y += _force0[idx].y;
    _force[idx].z += _force0[idx].z;
}


// _computeGroundGradientAndHessian, _computeGroundGradient, _computeGroundCloseVal,
// _checkGroundCloseVal, _reduct_MGroundDist, _computeSelfCloseVal, _checkGroundIntersection
// moved to ipc_barrier.cu

__global__ void _getFrictionEnergy_Reduction_3D(double*        squeue,
                                                const double3* vertexes,
                                                const double3* o_vertexes,
                                                const int4*    _collisionPair,
                                                int            cpNum,
                                                double         dt,
                                                const double2* distCoord,
                                                const __GEIGEN__::Matrix3x2d* tanBasis,
                                                const double* lastH,
                                                double        fricDHat,
                                                double        eps

)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];
    int                      numbers = cpNum;
    if(idx >= numbers)
        return;

    double temp = __cal_Friction_energy(
        vertexes, o_vertexes, _collisionPair[idx], dt, distCoord[idx], tanBasis[idx], lastH[idx], fricDHat, eps);

    int warpTid = threadIdx.x % 32;
    int warpId  = (threadIdx.x >> 5);
    int warpNum;
    //int tidNum = 32;
    if(blockIdx.x == gridDim.x - 1)
    {
        //tidNum = numbers - idof;
        warpNum = ((numbers - idof + 31) >> 5);
    }
    else
    {
        warpNum = ((blockDim.x) >> 5);
    }
    for(int i = 1; i < 32; i = (i << 1))
    {
        temp += __shfl_down_sync(0xffffffff, temp, i);
    }
    if(warpTid == 0)
    {
        tep[warpId] = temp;
    }
    __syncthreads();
    if(threadIdx.x >= warpNum)
        return;
    if(warpNum > 1)
    {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];
        //	warpNum = ((tidNum + 31) >> 5);
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            temp += __shfl_down_sync(0xffffffff, temp, i);
        }
    }
    if(threadIdx.x == 0)
    {
        squeue[blockIdx.x] = temp;
    }
}

__global__ void _getFrictionEnergy_gd_Reduction_3D(double*        squeue,
                                                   const double3* vertexes,
                                                   const double3* o_vertexes,
                                                   const double3* _normal,
                                                   const uint32_t* _collisionPair_gd,
                                                   int           gpNum,
                                                   double        dt,
                                                   const double* lastH,
                                                   double        eps

)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];
    int                      numbers = gpNum;
    if(idx >= numbers)
        return;

    double temp = __cal_Friction_gd_energy(
        vertexes, o_vertexes, _normal, _collisionPair_gd[idx], dt, lastH[idx], eps);

    int warpTid = threadIdx.x % 32;
    int warpId  = (threadIdx.x >> 5);
    int warpNum;
    //int tidNum = 32;
    if(blockIdx.x == gridDim.x - 1)
    {
        //tidNum = numbers - idof;
        warpNum = ((numbers - idof + 31) >> 5);
    }
    else
    {
        warpNum = ((blockDim.x) >> 5);
    }
    for(int i = 1; i < 32; i = (i << 1))
    {
        temp += __shfl_down_sync(0xffffffff, temp, i);
    }
    if(warpTid == 0)
    {
        tep[warpId] = temp;
    }
    __syncthreads();
    if(threadIdx.x >= warpNum)
        return;
    if(warpNum > 1)
    {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];
        //	warpNum = ((tidNum + 31) >> 5);
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            temp += __shfl_down_sync(0xffffffff, temp, i);
        }
    }
    if(threadIdx.x == 0)
    {
        squeue[blockIdx.x] = temp;
    }
}

__global__ void _computeGroundEnergy_Reduction(double*        squeue,
                                               const double3* vertexes,
                                               const double*  g_offset,
                                               const double3* g_normal,
                                               const uint32_t* _environment_collisionPair,
                                               double dHat,
                                               double Kappa,
                                               int    number)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];

    if(idx >= number)
        return;

    double3 normal = *g_normal;
    int     gidx   = _environment_collisionPair[idx];
    double  dist  = __GEIGEN__::__v_vec_dot(normal, vertexes[gidx]) - *g_offset;
    double  dist2 = dist * dist;
    double  temp  = -(dist2 - dHat) * (dist2 - dHat) * log(dist2 / dHat);

    int warpTid = threadIdx.x % 32;
    int warpId  = (threadIdx.x >> 5);
    int warpNum;
    //int tidNum = 32;
    if(blockIdx.x == gridDim.x - 1)
    {
        //tidNum = numbers - idof;
        warpNum = ((number - idof + 31) >> 5);
    }
    else
    {
        warpNum = ((blockDim.x) >> 5);
    }
    for(int i = 1; i < 32; i = (i << 1))
    {
        temp += __shfl_down_sync(0xffffffff, temp, i);
    }
    if(warpTid == 0)
    {
        tep[warpId] = temp;
    }
    __syncthreads();
    if(threadIdx.x >= warpNum)
        return;
    if(warpNum > 1)
    {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];

        //	warpNum = ((tidNum + 31) >> 5);
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            temp += __shfl_down_sync(0xffffffff, temp, i);
        }
    }
    if(threadIdx.x == 0)
    {
        squeue[blockIdx.x] = temp;
    }
}

__global__ void _reduct_min_groundTimeStep_to_double(const double3* vertexes,
                                                     const uint32_t* surfVertIds,
                                                     const double*  g_offset,
                                                     const double3* g_normal,
                                                     const double3* moveDir,
                                                     double* minStepSizes,
                                                     double  slackness,
                                                     int     number)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];

    if(idx >= number)
        return;
    int     svI    = surfVertIds[idx];
    double  temp   = 1.0;
    double3 normal = *g_normal;
    double  coef   = __GEIGEN__::__v_vec_dot(normal, moveDir[svI]);
    if(coef > 0.0)
    {
        double dist = __GEIGEN__::__v_vec_dot(normal, vertexes[svI]) - *g_offset;  //normal
        temp = coef / (dist * slackness);
        //printf("%f\n", temp);
    }
    /*if (blockIdx.x == 4) {
        printf("%f\n", temp);
    }
    __syncthreads();*/
    //printf("%f\n", temp);
    int warpTid = threadIdx.x % 32;
    int warpId  = (threadIdx.x >> 5);
    int warpNum;
    //int tidNum = 32;
    if(blockIdx.x == gridDim.x - 1)
    {
        //tidNum = numbers - idof;
        warpNum = ((number - idof + 31) >> 5);
        //printf("warpNum %d\n", warpNum);
    }
    else
    {
        warpNum = ((blockDim.x) >> 5);
    }
    for(int i = 1; i < 32; i = (i << 1))
    {
        double tempMin = __shfl_down_sync(0xffffffff, temp, i);
        temp           = std::max(temp, tempMin);
    }
    if(warpTid == 0)
    {
        tep[warpId] = temp;
    }
    __syncthreads();
    if(threadIdx.x >= warpNum)
        return;
    if(warpNum > 1)
    {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];

        //	warpNum = ((tidNum + 31) >> 5);
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            double tempMin = __shfl_down_sync(0xffffffff, temp, i);
            temp           = std::max(temp, tempMin);
        }
    }
    if(threadIdx.x == 0)
    {
        minStepSizes[blockIdx.x] = temp;
        //printf("%f   %d\n", temp, blockIdx.x);
    }
}

__global__ void _reduct_min_InjectiveTimeStep_to_double(const double3* vertexes,
                                                        const uint4* tetrahedra,
                                                        const double3* moveDir,
                                                        double* minStepSizes,
                                                        double  slackness,
                                                        double  errorRate,
                                                        int     number)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];

    if(idx >= number)
        return;
    double ratio = 1 - slackness;

    double temp = 1.0
                  / _computeInjectiveStepSize_3d(vertexes,
                                                 moveDir,
                                                 tetrahedra[idx].x,
                                                 tetrahedra[idx].y,
                                                 tetrahedra[idx].z,
                                                 tetrahedra[idx].w,
                                                 ratio,
                                                 errorRate);

    int warpTid = threadIdx.x % 32;
    int warpId  = (threadIdx.x >> 5);
    int warpNum;
    //int tidNum = 32;
    if(blockIdx.x == gridDim.x - 1)
    {
        //tidNum = numbers - idof;
        warpNum = ((number - idof + 31) >> 5);
        //printf("warpNum %d\n", warpNum);
    }
    else
    {
        warpNum = ((blockDim.x) >> 5);
    }
    for(int i = 1; i < 32; i = (i << 1))
    {
        double tempMin = __shfl_down_sync(0xffffffff, temp, i);
        temp           = std::max(temp, tempMin);
    }
    if(warpTid == 0)
    {
        tep[warpId] = temp;
    }
    __syncthreads();
    if(threadIdx.x >= warpNum)
        return;
    if(warpNum > 1)
    {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];

        //	warpNum = ((tidNum + 31) >> 5);
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            double tempMin = __shfl_down_sync(0xffffffff, temp, i);
            temp           = std::max(temp, tempMin);
        }
    }
    if(threadIdx.x == 0)
    {
        minStepSizes[blockIdx.x] = temp;
        //printf("%f   %d\n", temp, blockIdx.x);
    }
}

__global__ void _reduct_min_selfTimeStep_to_double(const double3* vertexes,
                                                   const int4* _ccd_collitionPairs,
                                                   const double3* moveDir,
                                                   double*        minStepSizes,
                                                   double         slackness,
                                                   int            number)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];

    if(idx >= number)
        return;
    double temp         = 1.0;
    double CCDDistRatio = 1.0 - slackness;

    int4 MMCVIDI = _ccd_collitionPairs[idx];

    if(MMCVIDI.x < 0)
    {
        MMCVIDI.x = -MMCVIDI.x - 1;

        double temp1 =
            point_triangle_ccd(vertexes[MMCVIDI.x],
                               vertexes[MMCVIDI.y],
                               vertexes[MMCVIDI.z],
                               vertexes[MMCVIDI.w],
                               __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.x], -1),
                               __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.y], -1),
                               __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.z], -1),
                               __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.w], -1),
                               CCDDistRatio,
                               0);

        //double temp2 = doCCDVF(vertexes[MMCVIDI.x],
        //    vertexes[MMCVIDI.y],
        //    vertexes[MMCVIDI.z],
        //    vertexes[MMCVIDI.w],
        //    __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.x], -1),
        //    __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.y], -1),
        //    __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.z], -1),
        //    __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.w], -1), 1e-9, 0.2);

        temp = 1.0 / temp1;
    }
    else
    {
        temp = 1.0
               / edge_edge_ccd(vertexes[MMCVIDI.x],
                               vertexes[MMCVIDI.y],
                               vertexes[MMCVIDI.z],
                               vertexes[MMCVIDI.w],
                               __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.x], -1),
                               __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.y], -1),
                               __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.z], -1),
                               __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.w], -1),
                               CCDDistRatio,
                               0);
    }

    int warpTid = threadIdx.x % 32;
    int warpId  = (threadIdx.x >> 5);
    int warpNum;
    //int tidNum = 32;
    if(blockIdx.x == gridDim.x - 1)
    {
        //tidNum = numbers - idof;
        warpNum = ((number - idof + 31) >> 5);
    }
    else
    {
        warpNum = ((blockDim.x) >> 5);
    }
    for(int i = 1; i < 32; i = (i << 1))
    {
        double tempMin = __shfl_down_sync(0xffffffff, temp, i);
        temp           = std::max(temp, tempMin);
    }
    if(warpTid == 0)
    {
        tep[warpId] = temp;
    }
    __syncthreads();
    if(threadIdx.x >= warpNum)
        return;
    if(warpNum > 1)
    {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];

        //	warpNum = ((tidNum + 31) >> 5);
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            double tempMin = __shfl_down_sync(0xffffffff, temp, i);
            temp           = std::max(temp, tempMin);
        }
    }
    if(threadIdx.x == 0)
    {
        minStepSizes[blockIdx.x] = temp;
    }
}

__global__ void _reduct_max_cfl_to_double(const double3* moveDir,
                                          double*        max_double_val,
                                          uint32_t*      mSVI,
                                          int            number)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];

    if(idx >= number)
        return;

    double temp = __GEIGEN__::__norm(moveDir[mSVI[idx]]);


    int warpTid = threadIdx.x % 32;
    int warpId  = (threadIdx.x >> 5);
    int warpNum;
    //int tidNum = 32;
    if(blockIdx.x == gridDim.x - 1)
    {
        //tidNum = numbers - idof;
        warpNum = ((number - idof + 31) >> 5);
    }
    else
    {
        warpNum = ((blockDim.x) >> 5);
    }
    for(int i = 1; i < 32; i = (i << 1))
    {
        double tempMax = __shfl_down_sync(0xffffffff, temp, i);
        temp           = std::max(temp, tempMax);
    }
    if(warpTid == 0)
    {
        tep[warpId] = temp;
    }
    __syncthreads();
    if(threadIdx.x >= warpNum)
        return;
    if(warpNum > 1)
    {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];

        //	warpNum = ((tidNum + 31) >> 5);
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            double tempMax = __shfl_down_sync(0xffffffff, temp, i);
            temp           = std::max(temp, tempMax);
        }
    }
    if(threadIdx.x == 0)
    {
        max_double_val[blockIdx.x] = temp;
    }
}

__global__ void _reduct_double3Sqn_to_double(const double3* A, double* D, int number)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];

    if(idx >= number)
        return;

    double temp = __GEIGEN__::__squaredNorm(A[idx]);


    int warpTid = threadIdx.x % 32;
    int warpId  = (threadIdx.x >> 5);
    int warpNum;
    //int tidNum = 32;
    if(blockIdx.x == gridDim.x - 1)
    {
        //tidNum = numbers - idof;
        warpNum = ((number - idof + 31) >> 5);
    }
    else
    {
        warpNum = ((blockDim.x) >> 5);
    }
    for(int i = 1; i < 32; i = (i << 1))
    {
        //double tempMax = __shfl_down_sync(0xffffffff, temp, i);
        temp += __shfl_down_sync(0xffffffff, temp, i);
    }
    if(warpTid == 0)
    {
        tep[warpId] = temp;
    }
    __syncthreads();
    if(threadIdx.x >= warpNum)
        return;
    if(warpNum > 1)
    {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];

        //	warpNum = ((tidNum + 31) >> 5);
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            temp += __shfl_down_sync(0xffffffff, temp, i);
        }
    }
    if(threadIdx.x == 0)
    {
        D[blockIdx.x] = temp;
    }
}

__global__ void _reduct_double3Dot_to_double(const double3* A, const double3* B, double* D, int number)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];

    if(idx >= number)
        return;

    double temp = __GEIGEN__::__v_vec_dot(A[idx], B[idx]);


    int warpTid = threadIdx.x % 32;
    int warpId  = (threadIdx.x >> 5);
    int warpNum;
    //int tidNum = 32;
    if(blockIdx.x == gridDim.x - 1)
    {
        //tidNum = numbers - idof;
        warpNum = ((number - idof + 31) >> 5);
    }
    else
    {
        warpNum = ((blockDim.x) >> 5);
    }
    for(int i = 1; i < 32; i = (i << 1))
    {
        //double tempMax = __shfl_down_sync(0xffffffff, temp, i);
        temp += __shfl_down_sync(0xffffffff, temp, i);
    }
    if(warpTid == 0)
    {
        tep[warpId] = temp;
    }
    __syncthreads();
    if(threadIdx.x >= warpNum)
        return;
    if(warpNum > 1)
    {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];

        //	warpNum = ((tidNum + 31) >> 5);
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            temp += __shfl_down_sync(0xffffffff, temp, i);
        }
    }
    if(threadIdx.x == 0)
    {
        D[blockIdx.x] = temp;
    }
}


__global__ void _getKineticEnergy_Reduction_3D(
    double3* _vertexes, double3* _xTilta, double* _energy, double* _masses, int number)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];

    if(idx >= number)
        return;

    double temp =
        __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(_vertexes[idx], _xTilta[idx]))
        * _masses[idx] * 0.5;

    int warpTid = threadIdx.x % 32;
    int warpId  = (threadIdx.x >> 5);
    int warpNum;
    //int tidNum = 32;
    if(blockIdx.x == gridDim.x - 1)
    {
        //tidNum = numbers - idof;
        warpNum = ((number - idof + 31) >> 5);
    }
    else
    {
        warpNum = ((blockDim.x) >> 5);
    }
    for(int i = 1; i < 32; i = (i << 1))
    {
        temp += __shfl_down_sync(0xffffffff, temp, i);
    }
    if(warpTid == 0)
    {
        tep[warpId] = temp;
    }
    __syncthreads();
    if(threadIdx.x >= warpNum)
        return;
    if(warpNum > 1)
    {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];

        //	warpNum = ((tidNum + 31) >> 5);
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            temp += __shfl_down_sync(0xffffffff, temp, i);
        }
    }
    if(threadIdx.x == 0)
    {
        _energy[blockIdx.x] = temp;
    }
}



__global__ void _getBendingEnergy_Reduction(double*        squeue,
                                            const double3* vertexes,
                                            const double3* rest_vertexex,
                                            const uint2*   edges,
                                            const uint2*   edge_adj_vertex,
                                            int            edgesNum,
                                            double         bendStiff)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];
    int                      numbers = edgesNum;
    if(idx >= numbers)
        return;

    //double temp = __cal_BaraffWitkinStretch_energy(vertexes, triangles[idx], triDmInverses[idx], area[idx], stretchStiff, shearStiff);
    // double temp = __cal_hc_cloth_energy(vertexes, triangles[idx], triDmInverses[idx], area[idx], stretchStiff, shearStiff);
    uint2   adj     = edge_adj_vertex[idx];
    double3 rest_x0 = rest_vertexex[edges[idx].x];
    double3 rest_x1 = rest_vertexex[edges[idx].y];
    double  length  = __GEIGEN__::__norm(__GEIGEN__::__minus(rest_x0, rest_x1));
    double  temp =
        __cal_bending_energy(vertexes, rest_vertexex, edges[idx], adj, length, bendStiff);
    //double temp = 0;
    //printf("%f    %f\n\n\n", lenRate, volRate);
    int warpTid = threadIdx.x % 32;
    int warpId  = (threadIdx.x >> 5);
    int warpNum;
    //int tidNum = 32;
    if(blockIdx.x == gridDim.x - 1)
    {
        //tidNum = numbers - idof;
        warpNum = ((numbers - idof + 31) >> 5);
    }
    else
    {
        warpNum = ((blockDim.x) >> 5);
    }
    for(int i = 1; i < 32; i = (i << 1))
    {
        temp += __shfl_down_sync(0xffffffff, temp, i);
    }
    if(warpTid == 0)
    {
        tep[warpId] = temp;
    }
    __syncthreads();
    if(threadIdx.x >= warpNum)
        return;
    if(warpNum > 1)
    {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];
        //	warpNum = ((tidNum + 31) >> 5);
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            temp += __shfl_down_sync(0xffffffff, temp, i);
        }
    }
    if(threadIdx.x == 0)
    {
        squeue[blockIdx.x] = temp;
    }
}


__global__ void _getFEMEnergy_Reduction_3D(double*        squeue,
                                           const double3* vertexes,
                                           const uint4*   tetrahedras,
                                           const __GEIGEN__::Matrix3x3d* DmInverses,
                                           const double* volume,
                                           int           tetrahedraNum,
                                           double*       lenRate,
                                           double*       volRate)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];
    int                      numbers = tetrahedraNum;
    if(idx >= numbers)
        return;

#ifdef USE_SNK1
    double temp = __cal_StabbleNHK_energy1_3D(
        vertexes, tetrahedras[idx], DmInverses[idx], volume[idx], lenRate[idx], volRate[idx]);
#elif USE_SNK2
    double temp = __cal_StabbleNHK_energy2_3D(
        vertexes, tetrahedras[idx], DmInverses[idx], volume[idx], lenRate[idx], volRate[idx]);
#else
    double temp = __cal_ARAP_energy_3D(
        vertexes, tetrahedras[idx], DmInverses[idx], volume[idx], lenRate[idx]);
#endif

    //printf("%f    %f\n\n\n", lenRate, volRate);
    int warpTid = threadIdx.x % 32;
    int warpId  = (threadIdx.x >> 5);
    int warpNum;
    //int tidNum = 32;
    if(blockIdx.x == gridDim.x - 1)
    {
        //tidNum = numbers - idof;
        warpNum = ((numbers - idof + 31) >> 5);
    }
    else
    {
        warpNum = ((blockDim.x) >> 5);
    }
    for(int i = 1; i < 32; i = (i << 1))
    {
        temp += __shfl_down_sync(0xffffffff, temp, i);
    }
    if(warpTid == 0)
    {
        tep[warpId] = temp;
    }
    __syncthreads();
    if(threadIdx.x >= warpNum)
        return;
    if(warpNum > 1)
    {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];
        //	warpNum = ((tidNum + 31) >> 5);
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            temp += __shfl_down_sync(0xffffffff, temp, i);
        }
    }
    if(threadIdx.x == 0)
    {
        squeue[blockIdx.x] = temp;
    }
}
__global__ void _computeSoftConstraintEnergy_Reduction(double*        squeue,
                                                       const double3* vertexes,
                                                       const double3* targetVert,
                                                       const uint32_t* targetInd,
                                                       double motionRate,
                                                       double rate,
                                                       int    number)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];

    if(idx >= number)
        return;
    uint32_t vInd = targetInd[idx];
    double   dis  = __GEIGEN__::__squaredNorm(__GEIGEN__::__s_vec_multiply(
        __GEIGEN__::__minus(vertexes[vInd], targetVert[idx]), rate));
    double   d    = motionRate;
    double   temp = d * dis * 0.5;

    int warpTid = threadIdx.x % 32;
    int warpId  = (threadIdx.x >> 5);
    int warpNum;
    //int tidNum = 32;
    if(blockIdx.x == gridDim.x - 1)
    {
        //tidNum = numbers - idof;
        warpNum = ((number - idof + 31) >> 5);
    }
    else
    {
        warpNum = ((blockDim.x) >> 5);
    }
    for(int i = 1; i < 32; i = (i << 1))
    {
        temp += __shfl_down_sync(0xffffffff, temp, i);
    }
    if(warpTid == 0)
    {
        tep[warpId] = temp;
    }
    __syncthreads();
    if(threadIdx.x >= warpNum)
        return;
    if(warpNum > 1)
    {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];

        //	warpNum = ((tidNum + 31) >> 5);
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            temp += __shfl_down_sync(0xffffffff, temp, i);
        }
    }
    if(threadIdx.x == 0)
    {
        squeue[blockIdx.x] = temp;
    }
}
__global__ void _get_triangleFEMEnergy_Reduction_3D(double*        squeue,
                                                    const double3* vertexes,
                                                    const uint3*   triangles,
                                                    const __GEIGEN__::Matrix2x2d* triDmInverses,
                                                    const double* area,
                                                    int           trianglesNum,
                                                    double        stretchStiff,
                                                    double        shearStiff,
                                                    double        strainRate)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];
    int                      numbers = trianglesNum;
    if(idx >= numbers)
        return;

    double temp = __cal_BaraffWitkinStretch_energy(
        vertexes, triangles[idx], triDmInverses[idx], area[idx], stretchStiff, shearStiff, strainRate);


    //printf("%f    %f\n\n\n", lenRate, volRate);
    int warpTid = threadIdx.x % 32;
    int warpId  = (threadIdx.x >> 5);
    int warpNum;
    //int tidNum = 32;
    if(blockIdx.x == gridDim.x - 1)
    {
        //tidNum = numbers - idof;
        warpNum = ((numbers - idof + 31) >> 5);
    }
    else
    {
        warpNum = ((blockDim.x) >> 5);
    }
    for(int i = 1; i < 32; i = (i << 1))
    {
        temp += __shfl_down_sync(0xffffffff, temp, i);
    }
    if(warpTid == 0)
    {
        tep[warpId] = temp;
    }
    __syncthreads();
    if(threadIdx.x >= warpNum)
        return;
    if(warpNum > 1)
    {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];
        //	warpNum = ((tidNum + 31) >> 5);
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            temp += __shfl_down_sync(0xffffffff, temp, i);
        }
    }
    if(threadIdx.x == 0)
    {
        squeue[blockIdx.x] = temp;
    }
}
__global__ void _getRestStableNHKEnergy_Reduction_3D(double*       squeue,
                                                     const double* volume,
                                                     int    tetrahedraNum,
                                                     double lenRate,
                                                     double volRate)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];
    int                      numbers = tetrahedraNum;
    if(idx >= numbers)
        return;

    double temp = ((0.5 * volRate * (3 * lenRate / 4 / volRate) * (3 * lenRate / 4 / volRate)
                    - 0.5 * lenRate * log(4.0)))
                  * volume[idx];

    int warpTid = threadIdx.x % 32;
    int warpId  = (threadIdx.x >> 5);
    int warpNum;
    //int tidNum = 32;
    if(blockIdx.x == gridDim.x - 1)
    {
        //tidNum = numbers - idof;
        warpNum = ((numbers - idof + 31) >> 5);
    }
    else
    {
        warpNum = ((blockDim.x) >> 5);
    }
    for(int i = 1; i < 32; i = (i << 1))
    {
        temp += __shfl_down_sync(0xffffffff, temp, i);
    }
    if(warpTid == 0)
    {
        tep[warpId] = temp;
    }
    __syncthreads();
    if(threadIdx.x >= warpNum)
        return;
    if(warpNum > 1)
    {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];
        //	warpNum = ((tidNum + 31) >> 5);
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            temp += __shfl_down_sync(0xffffffff, temp, i);
        }
    }
    if(threadIdx.x == 0)
    {
        squeue[blockIdx.x] = temp;
    }
}

__global__ void _getBarrierEnergy_Reduction_3D(double*        squeue,
                                               const double3* vertexes,
                                               const double3* rest_vertexes,
                                               int4*          _collisionPair,
                                               double         _Kappa,
                                               double         _dHat,
                                               int            cpNum)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];
    int                      numbers = cpNum;
    if(idx >= numbers)
        return;

    double temp =
        __cal_Barrier_energy(vertexes, rest_vertexes, _collisionPair[idx], _Kappa, _dHat);

    int warpTid = threadIdx.x % 32;
    int warpId  = (threadIdx.x >> 5);
    int warpNum;
    //int tidNum = 32;
    if(blockIdx.x == gridDim.x - 1)
    {
        //tidNum = numbers - idof;
        warpNum = ((numbers - idof + 31) >> 5);
    }
    else
    {
        warpNum = ((blockDim.x) >> 5);
    }
    for(int i = 1; i < 32; i = (i << 1))
    {
        temp += __shfl_down_sync(0xffffffff, temp, i);
    }
    if(warpTid == 0)
    {
        tep[warpId] = temp;
    }
    __syncthreads();
    if(threadIdx.x >= warpNum)
        return;
    if(warpNum > 1)
    {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];
        //	warpNum = ((tidNum + 31) >> 5);
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            temp += __shfl_down_sync(0xffffffff, temp, i);
        }
    }
    if(threadIdx.x == 0)
    {
        squeue[blockIdx.x] = temp;
    }
}

__global__ void _getDeltaEnergy_Reduction(double* squeue, const double3* b, const double3* dx, int vertexNum)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];
    int                      numbers = vertexNum;
    if(idx >= numbers)
        return;
    //int cfid = tid + CONFLICT_FREE_OFFSET(tid);

    double temp = __GEIGEN__::__v_vec_dot(b[idx], dx[idx]);

    int warpTid = threadIdx.x % 32;
    int warpId  = (threadIdx.x >> 5);
    int warpNum;
    //int tidNum = 32;
    if(blockIdx.x == gridDim.x - 1)
    {
        //tidNum = numbers - idof;
        warpNum = ((numbers - idof + 31) >> 5);
    }
    else
    {
        warpNum = ((blockDim.x) >> 5);
    }
    for(int i = 1; i < 32; i = (i << 1))
    {
        temp += __shfl_down_sync(0xffffffff, temp, i);
    }
    if(warpTid == 0)
    {
        tep[warpId] = temp;
    }
    __syncthreads();
    if(threadIdx.x >= warpNum)
        return;
    if(warpNum > 1)
    {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];
        //	warpNum = ((tidNum + 31) >> 5);
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            temp += __shfl_down_sync(0xffffffff, temp, i);
        }
    }
    if(threadIdx.x == 0)
    {
        squeue[blockIdx.x] = temp;
    }
}

__global__ void __add_reduction(double* mem, int numbers)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];

    if(idx >= numbers)
        return;
    //int cfid = tid + CONFLICT_FREE_OFFSET(tid);
    double temp = mem[idx];

    __threadfence();

    int warpTid = threadIdx.x % 32;
    int warpId  = (threadIdx.x >> 5);
    int warpNum;
    //int tidNum = 32;
    if(blockIdx.x == gridDim.x - 1)
    {
        //tidNum = numbers - idof;
        warpNum = ((numbers - idof + 31) >> 5);
    }
    else
    {
        warpNum = ((blockDim.x) >> 5);
    }
    for(int i = 1; i < 32; i = (i << 1))
    {
        temp += __shfl_down_sync(0xffffffff, temp, i);
    }
    if(warpTid == 0)
    {
        tep[warpId] = temp;
    }
    __syncthreads();
    if(threadIdx.x >= warpNum)
        return;
    if(warpNum > 1)
    {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];
        //	warpNum = ((tidNum + 31) >> 5);
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            temp += __shfl_down_sync(0xffffffff, temp, i);
        }
    }
    if(threadIdx.x == 0)
    {
        mem[blockIdx.x] = temp;
    }
}

__global__ void _stepForward(double3* _vertexes,
                             double3* _vertexesTemp,
                             double3* _moveDir,
                             int*     bType,
                             double   alpha,
                             bool     moveBoundary,
                             int      numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;
    if(abs(bType[idx]) == 0 || moveBoundary)
    {
        _vertexes[idx] =
            __GEIGEN__::__minus(_vertexesTemp[idx],
                                __GEIGEN__::__s_vec_multiply(_moveDir[idx], alpha));
    }
}

__global__ void _updateVelocities(double3* _vertexes,
                                  double3* _o_vertexes,
                                  double3* _velocities,
                                  int*     btype,
                                  double   ipc_dt,
                                  int      numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;
    if(btype[idx] == 0)
    {
        _velocities[idx] = __GEIGEN__::__s_vec_multiply(
            __GEIGEN__::__minus(_vertexes[idx], _o_vertexes[idx]), 1 / ipc_dt);
        //_velocities[idx] = make_double3(0, 0, 0);
        _o_vertexes[idx] = _vertexes[idx];
    }
    else
    {
        _velocities[idx] = make_double3(0, 0, 0);
        _o_vertexes[idx] = _vertexes[idx];
    }
}

__global__ void _updateBoundary(double3* _vertexes, int* _btype, double3* _moveDir, double ipc_dt, int numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;

    if((_btype[idx]) == -1 || (_btype[idx]) == 1)
    {
        _vertexes[idx] = __GEIGEN__::__add(_vertexes[idx], _moveDir[idx]);
    }
}

__global__ void _updateBoundary2(int* _btype, __GEIGEN__::Matrix3x3d* _constraints, int numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;

    if((_btype[idx]) == 1)
    {
        _btype[idx] = 0;
        __GEIGEN__::__set_Mat_val(_constraints[idx], 1, 0, 0, 0, 1, 0, 0, 0, 1);
    }
}


__global__ void _updateBoundaryMoveDir(double3* _vertexes,
                                       int*     _btype,
                                       double3* _moveDir,
                                       double   ipc_dt,
                                       double   PI,
                                       double   alpha,
                                       int      numbers,
                                       int      frameid)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;

    double                 angleX  = PI / 2.5 * ipc_dt * alpha;
    __GEIGEN__::Matrix3x3d rotationL, rotationR;
    __GEIGEN__::__set_Mat_val(
        rotationL, 1, 0, 0, 0, cos(angleX), sin(angleX), 0, -sin(angleX), cos(angleX));
    __GEIGEN__::__set_Mat_val(
        rotationR, 1, 0, 0, 0, cos(angleX), -sin(angleX), 0, sin(angleX), cos(angleX));

    //_moveDir[idx] = make_double3(0, 0, 0);
    double mvl = -0.3 * ipc_dt * alpha;
    //if((_btype[idx]) == 1)
    //{
    //    _moveDir[idx] = make_double3(mvl, 0, 0);  //__GEIGEN__::__minus(__GEIGEN__::__M_v_multiply(rotationL, _vertexes[idx]), _vertexes[idx]);
    //}
    if((_btype[idx]) > 0)
    {
        if(frameid < 32)
        {
            if(_vertexes[idx].y > 0.01)
            {
                _moveDir[idx] = make_double3(0, -mvl, 0);
            }
            else if(_vertexes[idx].y < -0.01)
            {
                _moveDir[idx] = make_double3(0, mvl, 0);
            }
        }
        else
        {
            _moveDir[idx] = __GEIGEN__::__minus(
                __GEIGEN__::__M_v_multiply(rotationL, _vertexes[idx]), _vertexes[idx]);
        }
    }
    if((_btype[idx]) < 0)
    {
        if(frameid < 32)
        {
            if(_vertexes[idx].y > 0.01)
            {
                _moveDir[idx] = make_double3(0, -mvl, 0);
            }
            else if(_vertexes[idx].y < -0.01)
            {
                _moveDir[idx] = make_double3(0, mvl, 0);
            }
        }
        else
        {
            _moveDir[idx] = __GEIGEN__::__minus(
                __GEIGEN__::__M_v_multiply(rotationR, _vertexes[idx]), _vertexes[idx]);
        }
    }
}

__global__ void _computeXTilta(int*     _btype,
                               double3* _velocities,
                               double3* _o_vertexes,
                               double3* _xTilta,
                               double   ipc_dt,
                               double   rate,
                               int      numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;

    double3 gravityDtSq = make_double3(0, 0, 0);  //__GEIGEN__::__s_vec_multiply(make_double3(0, -9.8, 0), ipc_dt * ipc_dt);//Vector3d(0, gravity, 0) * IPC_dt * IPC_dt;
    if(_btype[idx] == 0)
    {
        gravityDtSq =
            __GEIGEN__::__s_vec_multiply(make_double3(0, -9.8, 0), ipc_dt * ipc_dt);
    }
    _xTilta[idx] = __GEIGEN__::__add(
        _o_vertexes[idx],
        __GEIGEN__::__add(__GEIGEN__::__s_vec_multiply(_velocities[idx], ipc_dt),
                          gravityDtSq));  //(mesh.V_prev[vI] + (mesh.velocities[vI] * IPC_dt + gravityDtSq));
}

__global__ void _updateSurfaces(uint32_t* sortIndex, uint3* _faces, int _offset_num, int numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;
    if(_faces[idx].x < _offset_num)
    {
        _faces[idx].x = sortIndex[_faces[idx].x];
    }
    else
    {
        _faces[idx].x = _faces[idx].x;
    }
    if(_faces[idx].y < _offset_num)
    {
        _faces[idx].y = sortIndex[_faces[idx].y];
    }
    else
    {
        _faces[idx].y = _faces[idx].y;
    }
    if(_faces[idx].z < _offset_num)
    {
        _faces[idx].z = sortIndex[_faces[idx].z];
    }
    else
    {
        _faces[idx].z = _faces[idx].z;
    }
    //printf("sorted face: %d  %d  %d\n", _faces[idx].x, _faces[idx].y, _faces[idx].z);
}

__global__ void _updateNeighborNum(unsigned int*   _neighborNumInit,
                                   unsigned int*   _neighborNum,
                                   const uint32_t* sortMapVertIndex,
                                   int             numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;

    _neighborNum[idx] = _neighborNumInit[sortMapVertIndex[idx]];
}

__global__ void _updateNeighborList(unsigned int*   _neighborListInit,
                                    unsigned int*   _neighborList,
                                    unsigned int*   _neighborNum,
                                    unsigned int*   _neighborStart,
                                    unsigned int*   _neighborStartTemp,
                                    const uint32_t* sortIndex,
                                    const uint32_t* sortMapVertIndex,
                                    int             numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;

    int startId   = _neighborStartTemp[idx];
    int o_startId = _neighborStart[sortIndex[idx]];
    int neiNum    = _neighborNum[idx];
    for(int i = 0; i < neiNum; i++)
    {
        _neighborList[startId + i] = sortMapVertIndex[_neighborListInit[o_startId + i]];
    }
    //_neighborStart[sortMapVertIndex[idx]] = startId;
    //_neighborNum[idx] = _neighborNum[sortMapVertIndex[idx]];
}

__global__ void _updateEdges(uint32_t* sortIndex, uint2* _edges, int _offset_num, int numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;
    if(_edges[idx].x < _offset_num)
    {
        _edges[idx].x = sortIndex[_edges[idx].x];
    }
    else
    {
        _edges[idx].x = _edges[idx].x;
    }
    if(_edges[idx].y < _offset_num)
    {
        _edges[idx].y = sortIndex[_edges[idx].y];
    }
    else
    {
        _edges[idx].y = _edges[idx].y;
    }
}

__global__ void _updateTriEdges_adjVerts(
    uint32_t* sortIndex, uint2* _edges, uint2* _adj_verts, int _offset_num, int numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;
    if(_edges[idx].x < _offset_num)
    {
        _edges[idx].x = sortIndex[_edges[idx].x];
    }
    else
    {
        _edges[idx].x = _edges[idx].x;
    }
    if(_edges[idx].y < _offset_num)
    {
        _edges[idx].y = sortIndex[_edges[idx].y];
    }
    else
    {
        _edges[idx].y = _edges[idx].y;
    }


    if(_adj_verts[idx].x < _offset_num)
    {
        _adj_verts[idx].x = sortIndex[_adj_verts[idx].x];
    }
    else
    {
        _adj_verts[idx].x = _adj_verts[idx].x;
    }
    if(_adj_verts[idx].y < _offset_num)
    {
        _adj_verts[idx].y = sortIndex[_adj_verts[idx].y];
    }
    else
    {
        _adj_verts[idx].y = _adj_verts[idx].y;
    }
}

__global__ void _updateSurfVerts(uint32_t* sortIndex, uint32_t* _sVerts, int _offset_num, int numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;
    if(_sVerts[idx] < _offset_num)
    {
        _sVerts[idx] = sortIndex[_sVerts[idx]];
    }
    else
    {
        _sVerts[idx] = _sVerts[idx];
    }
}

__global__ void _edgeTriIntersectionQuery(const int*     _btype,
                                          const double3* _vertexes,
                                          const uint2*   _edges,
                                          const uint3*   _faces,
                                          const AABB*    _edge_bvs,
                                          const Node*    _edge_nodes,
                                          int*           _isIntesect,
                                          double         dHat,
                                          int            number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;

    uint32_t  stack[64];
    uint32_t* stack_ptr = stack;
    *stack_ptr++        = 0;

    uint3 face = _faces[idx];
    //idx = idx + number - 1;


    AABB _bv;

    double3 _v = _vertexes[face.x];
    _bv.combines(_v.x, _v.y, _v.z);
    _v = _vertexes[face.y];
    _bv.combines(_v.x, _v.y, _v.z);
    _v = _vertexes[face.z];
    _bv.combines(_v.x, _v.y, _v.z);

    //uint32_t self_eid = _edge_nodes[idx].element_idx;
    //double bboxDiagSize2 = __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(_edge_bvs[0].upper, _edge_bvs[0].lower));
    //printf("%f\n", bboxDiagSize2);
    double gapl = 0;  //sqrt(dHat);
    //double dHat = gapl * gapl;// *bboxDiagSize2;
    do
    {
        const uint32_t node_id = *--stack_ptr;
        const uint32_t L_idx   = _edge_nodes[node_id].left_idx;
        const uint32_t R_idx   = _edge_nodes[node_id].right_idx;

        if(_overlap(_bv, _edge_bvs[L_idx], gapl))
        {
            const auto obj_idx = _edge_nodes[L_idx].element_idx;
            if(obj_idx != 0xFFFFFFFF)
            {
                if(!(face.x == _edges[obj_idx].x || face.x == _edges[obj_idx].y
                     || face.y == _edges[obj_idx].x || face.y == _edges[obj_idx].y
                     || face.z == _edges[obj_idx].x || face.z == _edges[obj_idx].y))
                {
                    if(!(_btype[face.x] >= 2 && _btype[face.y] >= 2
                         && _btype[face.z] >= 2 && _btype[_edges[obj_idx].x] >= 2
                         && _btype[_edges[obj_idx].y] >= 2))
                        if(segTriIntersect(_vertexes[_edges[obj_idx].x],
                                           _vertexes[_edges[obj_idx].y],
                                           _vertexes[face.x],
                                           _vertexes[face.y],
                                           _vertexes[face.z]))
                        {
                            //atomicAdd(_isIntesect, -1);
                            *_isIntesect = -1;
                            //printf("tri: %d %d %d,  edge: %d  %d\n",
                            //       face.x,
                            //       face.y,
                            //       face.z,
                            //       _edges[obj_idx].x,
                            //       _edges[obj_idx].y);
                            return;
                        }
                }
            }
            else  // the node is not a leaf.
            {
                *stack_ptr++ = L_idx;
            }
        }
        if(_overlap(_bv, _edge_bvs[R_idx], gapl))
        {
            const auto obj_idx = _edge_nodes[R_idx].element_idx;
            if(obj_idx != 0xFFFFFFFF)
            {
                if(!(face.x == _edges[obj_idx].x || face.x == _edges[obj_idx].y
                     || face.y == _edges[obj_idx].x || face.y == _edges[obj_idx].y
                     || face.z == _edges[obj_idx].x || face.z == _edges[obj_idx].y))
                {
                    if(!(_btype[face.x] >= 2 && _btype[face.y] >= 2
                         && _btype[face.z] >= 2 && _btype[_edges[obj_idx].x] >= 2
                         && _btype[_edges[obj_idx].y] >= 2))
                        if(segTriIntersect(_vertexes[_edges[obj_idx].x],
                                           _vertexes[_edges[obj_idx].y],
                                           _vertexes[face.x],
                                           _vertexes[face.y],
                                           _vertexes[face.z]))
                        {
                            //atomicAdd(_isIntesect, -1);
                            *_isIntesect = -1;
                            //printf("tri: %d %d %d,  edge: %d  %d\n",
                            //       face.x,
                            //       face.y,
                            //       face.z,
                            //       _edges[obj_idx].x,
                            //       _edges[obj_idx].y);
                            return;
                        }
                }
            }
            else  // the node is not a leaf.
            {
                *stack_ptr++ = R_idx;
            }
        }
    } while(stack < stack_ptr);
}

__global__ void _calFrictionLastH_gd(const double3* _vertexes,
                                     const double*  g_offset,
                                     const double3* g_normal,
                                     const uint32_t* _collisionPair_environment,
                                     double*   lambda_lastH_gd,
                                     uint32_t* _collisionPair_last_gd,
                                     double    dHat,
                                     double    Kappa,
                                     int       number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;

    double3 normal = *g_normal;
    int     gidx   = _collisionPair_environment[idx];
    double  dist = __GEIGEN__::__v_vec_dot(normal, _vertexes[gidx]) - *g_offset;
    double  dist2 = dist * dist;

    double t   = dist2 - dHat;
    double g_b = t * log(dist2 / dHat) * -2.0 - (t * t) / dist2;

    lambda_lastH_gd[idx]        = -Kappa * 2.0 * sqrt(dist2) * g_b;
    _collisionPair_last_gd[idx] = gidx;
}

__global__ void _calFrictionLastH_DistAndTan(const double3*    _vertexes,
                                             const int4* _collisionPair,
                                             double*           lambda_lastH,
                                             double2*          distCoord,
                                             __GEIGEN__::Matrix3x2d* tanBasis,
                                             int4*     _collisionPair_last,
                                             double    dHat,
                                             double    Kappa,
                                             uint32_t* _cpNum_last,
                                             int       number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    int4   MMCVIDI = _collisionPair[idx];
    double dis;
    int    last_index = -1;
    if(MMCVIDI.x >= 0)
    {
        if(MMCVIDI.w >= 0)
        {
            last_index = atomicAdd(_cpNum_last, 1);
            atomicAdd(_cpNum_last + 4, 1);
            _d_EE(_vertexes[MMCVIDI.x],
                  _vertexes[MMCVIDI.y],
                  _vertexes[MMCVIDI.z],
                  _vertexes[MMCVIDI.w],
                  dis);
            Friction::computeClosestPoint_EE(_vertexes[MMCVIDI.x],
                                             _vertexes[MMCVIDI.y],
                                             _vertexes[MMCVIDI.z],
                                             _vertexes[MMCVIDI.w],
                                             distCoord[last_index]);
            Friction::computeTangentBasis_EE(_vertexes[MMCVIDI.x],
                                             _vertexes[MMCVIDI.y],
                                             _vertexes[MMCVIDI.z],
                                             _vertexes[MMCVIDI.w],
                                             tanBasis[last_index]);
        }
    }
    else
    {
        int v0I = -MMCVIDI.x - 1;
        if(MMCVIDI.z < 0)
        {
            if(MMCVIDI.y >= 0)
            {
                last_index = atomicAdd(_cpNum_last, 1);
                atomicAdd(_cpNum_last + 2, 1);
                _d_PP(_vertexes[v0I], _vertexes[MMCVIDI.y], dis);
                distCoord[last_index].x = 0;
                distCoord[last_index].y = 0;
                Friction::computeTangentBasis_PP(
                    _vertexes[v0I], _vertexes[MMCVIDI.y], tanBasis[last_index]);
            }
        }
        else if(MMCVIDI.w < 0)
        {
            if(MMCVIDI.y >= 0)
            {
                last_index = atomicAdd(_cpNum_last, 1);
                atomicAdd(_cpNum_last + 3, 1);
                _d_PE(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], dis);
                Friction::computeClosestPoint_PE(_vertexes[v0I],
                                                 _vertexes[MMCVIDI.y],
                                                 _vertexes[MMCVIDI.z],
                                                 distCoord[last_index].x);
                distCoord[last_index].y = 0;
                Friction::computeTangentBasis_PE(_vertexes[v0I],
                                                 _vertexes[MMCVIDI.y],
                                                 _vertexes[MMCVIDI.z],
                                                 tanBasis[last_index]);
            }
        }
        else
        {
            last_index = atomicAdd(_cpNum_last, 1);
            atomicAdd(_cpNum_last + 4, 1);
            _d_PT(_vertexes[v0I],
                  _vertexes[MMCVIDI.y],
                  _vertexes[MMCVIDI.z],
                  _vertexes[MMCVIDI.w],
                  dis);
            Friction::computeClosestPoint_PT(_vertexes[v0I],
                                             _vertexes[MMCVIDI.y],
                                             _vertexes[MMCVIDI.z],
                                             _vertexes[MMCVIDI.w],
                                             distCoord[last_index]);
            Friction::computeTangentBasis_PT(_vertexes[v0I],
                                             _vertexes[MMCVIDI.y],
                                             _vertexes[MMCVIDI.z],
                                             _vertexes[MMCVIDI.w],
                                             tanBasis[last_index]);
        }
    }
    if(last_index >= 0)
    {
//        double t = dis - dHat;
//        lambda_lastH[last_index] = -Kappa * 2.0 * std::sqrt(dis) * (t * std::log(dis / dHat) * -2.0 - (t * t) / dis);
#if (RANK == 1)
        double t = dis - dHat;
        lambda_lastH[last_index] =
            -Kappa * 2.0 * sqrt(dis) * (t * log(dis / dHat) * -2.0 - (t * t) / dis);
#elif (RANK == 2)
        lambda_lastH[last_index] =
            -Kappa * 2.0 * sqrt(dis)
            * (log(dis / dHat) * log(dis / dHat) * (2 * dis - 2 * dHat)
               + (2 * log(dis / dHat) * (dis - dHat) * (dis - dHat)) / dis);
#endif
        _collisionPair_last[last_index] = _collisionPair[idx];
    }
}

/// <summary>
///  host code
/// </summary>
void GIPC::FREE_DEVICE_MEM()
{
    CUDA_SAFE_CALL(cudaFree(_MatIndex));
    CUDA_SAFE_CALL(cudaFree(_collisonPairs));
    CUDA_SAFE_CALL(cudaFree(_ccd_collisonPairs));
    CUDA_SAFE_CALL(cudaFree(_cpNum));
    CUDA_SAFE_CALL(cudaFree(_close_cpNum));
    CUDA_SAFE_CALL(cudaFree(_close_gpNum));
    CUDA_SAFE_CALL(cudaFree(_environment_collisionPair));
    CUDA_SAFE_CALL(cudaFree(_gpNum));
    CUDA_SAFE_CALL(cudaFree(_groundNormal));
    CUDA_SAFE_CALL(cudaFree(_groundOffset));

    CUDA_SAFE_CALL(cudaFree(_faces));
    CUDA_SAFE_CALL(cudaFree(_edges));
    CUDA_SAFE_CALL(cudaFree(_surfVerts));

    pcg_data.FREE_DEVICE_MEM();

    bvh_e.FREE_DEVICE_MEM();
    bvh_f.FREE_DEVICE_MEM();
}

void GIPC::MALLOC_DEVICE_MEM()
{
    CUDA_SAFE_CALL(cudaMalloc((void**)&_MatIndex, MAX_COLLITION_PAIRS_NUM * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&_collisonPairs,
                              MAX_COLLITION_PAIRS_NUM * sizeof(int4)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&_ccd_collisonPairs,
                              MAX_CCD_COLLITION_PAIRS_NUM * sizeof(int4)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&_environment_collisionPair,
                              surf_vertexNum * sizeof(int)));
    //CUDA_SAFE_CALL(cudaMalloc((void**)&_moveDir, vertexNum * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&_cpNum, 5 * sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&_gpNum, sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&_groundNormal, 5 * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&_groundOffset, 5 * sizeof(double)));
    double  h_offset[5] = {-1, -1, 1, -1, 1};
    double3 H_normal[5];  // = { make_double3(0, 1, 0);
    H_normal[0] = make_double3(0, 1, 0);
    H_normal[1] = make_double3(1, 0, 0);
    H_normal[2] = make_double3(-1, 0, 0);
    H_normal[3] = make_double3(0, 0, 1);
    H_normal[4] = make_double3(0, 0, -1);
    CUDA_SAFE_CALL(cudaMemcpy(_groundOffset, &h_offset, 5 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(_groundNormal, &H_normal, 5 * sizeof(double3), cudaMemcpyHostToDevice));


    CUDA_SAFE_CALL(cudaMalloc((void**)&_faces, surface_Num * sizeof(uint3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&_edges, edge_Num * sizeof(uint2)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&_surfVerts, surf_vertexNum * sizeof(uint32_t)));

    CUDA_SAFE_CALL(cudaMalloc((void**)&_close_cpNum, sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&_close_gpNum, sizeof(uint32_t)));

    CUDA_SAFE_CALL(cudaMemset(_close_cpNum, 0, sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMemset(_close_gpNum, 0, sizeof(uint32_t)));

    pcg_data.Malloc_DEVICE_MEM(vertexNum, tetrahedraNum);
}


void GIPC::initBVH(int* _btype, int* _bodyId)
{

    bvh_e.init(_bodyId,
               _btype,
               _vertexes,
               _rest_vertexes,
               _edges,
               _collisonPairs,
               _ccd_collisonPairs,
               _cpNum,
               _MatIndex,
               edge_Num,
               surf_vertexNum);
    bvh_f.init(_bodyId,
               _btype,
               _vertexes,
               _faces,
               _surfVerts,
               _collisonPairs,
               _ccd_collisonPairs,
               _cpNum,
               _MatIndex,
               surface_Num,
               surf_vertexNum);
}

void GIPC::init(double m_meanMass, double m_meanVolumn, double3 minConer, double3 maxConer)
{
    SceneSize     = bvh_f.scene;
    bboxDiagSize2 = __GEIGEN__::__squaredNorm(
        __GEIGEN__::__minus(SceneSize.upper, SceneSize.lower));  //(maxConer - minConer).squaredNorm();
    dTol         = 1e-18 * bboxDiagSize2;
    minKappaCoef = 1e11;
    meanMass     = m_meanMass;
    meanVolumn   = m_meanVolumn;
    dHat = relative_dhat * relative_dhat * bboxDiagSize2;  //__GEIGEN__::__squaredNorm(__GEIGEN__::__minus(maxConer, minConer));
    fDhat = 1e-4 * bboxDiagSize2;


    int global_matrix_block3_size =
        abd_fem_count_info.abd_body_num * 4 + abd_fem_count_info.fem_point_num;


    uint32_t Minimum = 100000;
    int minCollisionBuffer4 = std::max(2 * (surf_vertexNum + edge_Num), Minimum);
    int minCollisionBuffer3 = std::max(2 * (surf_vertexNum + edge_Num), Minimum);
    int minCollisionBuffer2 = std::max(2 * (surf_vertexNum + edge_Num), Minimum);
    int minCollisionBuffer1 = 2 * surf_vertexNum;

    long long unsigned total_internal_triplet_num =
        ((abd_fem_count_info.fem_tet_num + tri_edge_num) * 10 + triangleNum * 6)
        + abd_fem_count_info.abd_body_num * 10;
    long long unsigned total_max_collision_triplet_num =
        minCollisionBuffer4 * 16 + minCollisionBuffer3 * 9
        + minCollisionBuffer2 * 4 + minCollisionBuffer1;
    long long unsigned total_max_global_triplet_num =
        total_internal_triplet_num * 2 + total_max_collision_triplet_num;

    gipc_global_triplet.init_var();

    gipc_global_triplet.resize(global_matrix_block3_size,
                               global_matrix_block3_size,
                               total_max_global_triplet_num);

    gipc_global_triplet.global_external_max_capcity =
        total_internal_triplet_num + total_max_collision_triplet_num;
    gipc_global_triplet.resize_collision_hash_size(gipc_global_triplet.global_external_max_capcity);


    m_global_linear_system->gipc_global_triplet = &(gipc_global_triplet);
    m_abd_system->global_triplet                = &(gipc_global_triplet);
    init_abd_system();
}

GIPC::~GIPC()
{
    FREE_DEVICE_MEM();
}

GIPC::GIPC()
{
    IPC_dt            = 0.01;
    animation_subRate = 1.0;
    animation         = false;

    h_cpNum_last[0] = 0;
    h_cpNum_last[1] = 0;
    h_cpNum_last[2] = 0;
    h_cpNum_last[3] = 0;
    h_cpNum_last[4] = 0;
}

void GIPC::buildFrictionSets()
{
    CUDA_SAFE_CALL(cudaMemset(_cpNum, 0, 5 * sizeof(uint32_t)));
    int                numbers   = h_cpNum[0];
    const unsigned int threadNum = 256;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    if(numbers > 0)
    {
        _calFrictionLastH_DistAndTan<<<blockNum, threadNum>>>(_vertexes,
                                                              _collisonPairs,
                                                              lambda_lastH_scalar,
                                                              distCoord,
                                                              tanBasis,
                                                              _collisonPairs_lastH,
                                                              dHat,
                                                              Kappa,
                                                              _cpNum,
                                                              h_cpNum[0]);
    }
    CUDA_SAFE_CALL(cudaMemcpy(h_cpNum_last, _cpNum, 5 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    numbers = h_gpNum;
    if(numbers > 0)
    {

        blockNum = (numbers + threadNum - 1) / threadNum;
        _calFrictionLastH_gd<<<blockNum, threadNum>>>(_vertexes,
                                                      _groundOffset,
                                                      _groundNormal,
                                                      _environment_collisionPair,
                                                      lambda_lastH_scalar_gd,
                                                      _collisonPairs_lastH_gd,
                                                      dHat,
                                                      Kappa,
                                                      h_gpNum);
    }
    h_gpNum_last = h_gpNum;
}


void GIPC::GroundCollisionDetect()
{
    int numbers = surf_vertexNum;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    _GroundCollisionDetect<<<blockNum, threadNum>>>(
        _vertexes, _surfVerts, _groundOffset, _groundNormal, _environment_collisionPair, _gpNum, dHat, numbers);
}

void GIPC::computeSoftConstraintGradientAndHessian(double3* _gradient, int global_hessian_fem_offset)
{
    int numbers = softNum;
    if(numbers < 1)
    {
        return;
    }
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    // offset
    _computeSoftConstraintGradientAndHessian<<<blockNum, threadNum>>>(
        _vertexes,
        targetVert,
        targetInd,
        _gradient,
        _gpNum,
        gipc_global_triplet.block_values(),
        gipc_global_triplet.block_row_indices(),
        gipc_global_triplet.block_col_indices(),
        softMotionRate,
        animation_fullRate,
        gipc_global_triplet.global_triplet_offset,
        global_hessian_fem_offset,
        softNum);
}

void GIPC::getTotalForce(double3* _gradient0, double3* _gradient1)
{

    int numbers = vertexNum;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    _getTotalForce<<<blockNum, threadNum>>>(_gradient0, _gradient1, numbers);
}


void GIPC::computeGroundGradientAndHessian(double3* _gradient)
{
#ifndef USE_FRICTION
    CUDA_SAFE_CALL(cudaMemset(_gpNum, 0, sizeof(uint32_t)));
#endif
    int numbers = h_gpNum;
    if(numbers < 1)
    {
        return;
    }
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    _computeGroundGradientAndHessian<<<blockNum, threadNum>>>(
        _vertexes,
        _groundOffset,
        _groundNormal,
        _environment_collisionPair,
        _gradient,
        _gpNum,
        gipc_global_triplet.block_values(),
        gipc_global_triplet.block_row_indices(),
        gipc_global_triplet.block_col_indices(),
        dHat,
        Kappa,
        gipc_global_triplet.global_triplet_offset,
        numbers);
}

void GIPC::computeCloseGroundVal()
{
    int numbers = h_gpNum;
    if(h_gpNum <= 0)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    _computeGroundCloseVal<<<blockNum, threadNum>>>(_vertexes,
                                                    _groundOffset,
                                                    _groundNormal,
                                                    _environment_collisionPair,
                                                    dTol,
                                                    _closeConstraintID,
                                                    _closeConstraintVal,
                                                    _close_gpNum,
                                                    numbers);
}

bool GIPC::checkCloseGroundVal()
{
    int numbers = h_close_gpNum;
    if(numbers < 1)
        return false;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    int*               _isChange;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_isChange, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(_isChange, 0, sizeof(int)));
    _checkGroundCloseVal<<<blockNum, threadNum>>>(
        _vertexes, _groundOffset, _groundNormal, _isChange, _closeConstraintID, _closeConstraintVal, numbers);
    int isChange;
    CUDA_SAFE_CALL(cudaMemcpy(&isChange, _isChange, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaFree(_isChange));

    return (isChange == 1);
}

double2 GIPC::minMaxGroundDist()
{
    //_reduct_minGroundDist << <blockNum, threadNum >> > (_vertexes, _groundOffset, _groundNormal, _isChange, _closeConstraintID, _closeConstraintVal, numbers);

    int numbers = h_gpNum;
    if(numbers < 1)
        return make_double2(1e32, 0);
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double2) * (threadNum >> 5);

    double2* _queue;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_queue, numbers * sizeof(double2)));
    //CUDA_SAFE_CALL(cudaMemcpy(_tempMinMovement, _moveDir, number * sizeof(AABB), cudaMemcpyDeviceToDevice));
    _reduct_MGroundDist<<<blockNum, threadNum, sharedMsize>>>(
        _vertexes, _groundOffset, _groundNormal, _environment_collisionPair, _queue, numbers);
    //_reduct_min_double3_to_double << <blockNum, threadNum, sharedMsize >> > (_moveDir, _tempMinMovement, numbers);

    numbers  = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while(numbers > 1)
    {
        //_reduct_max_box << <blockNum, threadNum, sharedMsize >> > (_tempLeafBox, numbers);
        _reduct_M_double2<<<blockNum, threadNum, sharedMsize>>>(_queue, numbers);
        numbers  = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }
    //cudaMemcpy(_leafBoxes, _tempLeafBox, sizeof(AABB), cudaMemcpyDeviceToDevice);
    double2 minMaxValue;
    cudaMemcpy(&minMaxValue, _queue, sizeof(double2), cudaMemcpyDeviceToHost);
    CUDA_SAFE_CALL(cudaFree(_queue));
    minMaxValue.x = 1.0 / minMaxValue.x;
    return minMaxValue;
}

void GIPC::computeGroundGradient(double3* _gradient, double mKappa)
{
    int numbers = h_gpNum;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    _computeGroundGradient<<<blockNum, threadNum>>>(_vertexes,
                                                    _groundOffset,
                                                    _groundNormal,
                                                    _environment_collisionPair,
                                                    _gradient,
                                                    _gpNum,
                                                    dHat,
                                                    mKappa,
                                                    numbers);
}

void GIPC::computeSoftConstraintGradient(double3* _gradient)
{
    int numbers = softNum;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    // offset
    _computeSoftConstraintGradient<<<blockNum, threadNum>>>(
        _vertexes, targetVert, targetInd, _gradient, softMotionRate, animation_fullRate, softNum);
}

double GIPC::self_largestFeasibleStepSize(double slackness, double* mqueue, int numbers)
{
    //slackness = 0.9;
    //int numbers = h_cpNum[0];
    if(numbers < 1)
        return 1;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);

    //double* _minSteps;
    //CUDA_SAFE_CALL(cudaMalloc((void**)&_minSteps, numbers * sizeof(double)));
    //CUDA_SAFE_CALL(cudaMemcpy(_tempMinMovement, _moveDir, number * sizeof(AABB), cudaMemcpyDeviceToDevice));
    _reduct_min_selfTimeStep_to_double<<<blockNum, threadNum, sharedMsize>>>(
        _vertexes, _ccd_collisonPairs, _moveDir, mqueue, slackness, numbers);
    //_reduct_min_double3_to_double << <blockNum, threadNum, sharedMsize >> > (_moveDir, _tempMinMovement, numbers);

    numbers  = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while(numbers > 1)
    {
        //_reduct_max_box << <blockNum, threadNum, sharedMsize >> > (_tempLeafBox, numbers);
        _reduct_max_double<<<blockNum, threadNum, sharedMsize>>>(mqueue, numbers);
        numbers  = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }
    //cudaMemcpy(_leafBoxes, _tempLeafBox, sizeof(AABB), cudaMemcpyDeviceToDevice);
    double minValue;
    cudaMemcpy(&minValue, mqueue, sizeof(double), cudaMemcpyDeviceToHost);
    //printf("                 full ccd time step:  %f\n", 1.0 / minValue);
    //CUDA_SAFE_CALL(cudaFree(_minSteps));
    return 1.0 / minValue;
}

double GIPC::cfl_largestSpeed(double* mqueue)
{
    int                numbers   = surf_vertexNum;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);

    /*double* _maxV;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_maxV, numbers * sizeof(double)));*/
    //CUDA_SAFE_CALL(cudaMemcpy(_tempMinMovement, _moveDir, number * sizeof(AABB), cudaMemcpyDeviceToDevice));
    _reduct_max_cfl_to_double<<<blockNum, threadNum, sharedMsize>>>(
        _moveDir, mqueue, _surfVerts, numbers);
    //_reduct_min_double3_to_double << <blockNum, threadNum, sharedMsize >> > (_moveDir, _tempMinMovement, numbers);

    numbers  = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while(numbers > 1)
    {
        //_reduct_max_box << <blockNum, threadNum, sharedMsize >> > (_tempLeafBox, numbers);
        _reduct_max_double<<<blockNum, threadNum, sharedMsize>>>(mqueue, numbers);
        numbers  = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }
    //cudaMemcpy(_leafBoxes, _tempLeafBox, sizeof(AABB), cudaMemcpyDeviceToDevice);
    double minValue;
    cudaMemcpy(&minValue, mqueue, sizeof(double), cudaMemcpyDeviceToHost);
    //CUDA_SAFE_CALL(cudaFree(_maxV));
    return minValue;
}

double reduction2Kappa(int type, const double3* A, const double3* B, double* _queue, int vertexNum)
{
    int                numbers   = vertexNum;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);

    /*double* _queue;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_queue, numbers * sizeof(double)));*/
    if(type == 0)
    {
        //CUDA_SAFE_CALL(cudaMemcpy(_tempMinMovement, _moveDir, number * sizeof(AABB), cudaMemcpyDeviceToDevice));
        _reduct_double3Dot_to_double<<<blockNum, threadNum, sharedMsize>>>(A, B, _queue, numbers);
    }
    else if(type == 1)
    {
        _reduct_double3Sqn_to_double<<<blockNum, threadNum, sharedMsize>>>(A, _queue, numbers);
    }
    //_reduct_min_double3_to_double << <blockNum, threadNum, sharedMsize >> > (_moveDir, _tempMinMovement, numbers);

    numbers  = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while(numbers > 1)
    {
        //_reduct_max_box << <blockNum, threadNum, sharedMsize >> > (_tempLeafBox, numbers);
        __add_reduction<<<blockNum, threadNum, sharedMsize>>>(_queue, numbers);
        numbers  = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }
    //cudaMemcpy(_leafBoxes, _tempLeafBox, sizeof(AABB), cudaMemcpyDeviceToDevice);
    double dotValue;
    cudaMemcpy(&dotValue, _queue, sizeof(double), cudaMemcpyDeviceToHost);
    //CUDA_SAFE_CALL(cudaFree(_queue));
    return dotValue;
}

double GIPC::ground_largestFeasibleStepSize(double slackness, double* mqueue)
{

    int numbers = surf_vertexNum;
    if(numbers < 1)
        return 1;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);

    //double* _minSteps;
    //CUDA_SAFE_CALL(cudaMalloc((void**)&_minSteps, numbers * sizeof(double)));

    //if (h_cpNum[0] > 0) {
    //    double3* mvd = new double3[vertexNum];
    //    cudaMemcpy(mvd, _moveDir, sizeof(double3) * vertexNum, cudaMemcpyDeviceToHost);
    //    for (int i = 0;i < vertexNum;i++) {
    //        printf("%f  %f  %f\n", mvd[i].x, mvd[i].y, mvd[i].z);
    //    }
    //    delete[] mvd;
    //}
    _reduct_min_groundTimeStep_to_double<<<blockNum, threadNum, sharedMsize>>>(
        _vertexes, _surfVerts, _groundOffset, _groundNormal, _moveDir, mqueue, slackness, numbers);


    numbers  = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while(numbers > 1)
    {
        //_reduct_max_box << <blockNum, threadNum, sharedMsize >> > (_tempLeafBox, numbers);
        _reduct_max_double<<<blockNum, threadNum, sharedMsize>>>(mqueue, numbers);
        numbers  = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }
    //cudaMemcpy(_leafBoxes, _tempLeafBox, sizeof(AABB), cudaMemcpyDeviceToDevice);
    double minValue;
    cudaMemcpy(&minValue, mqueue, sizeof(double), cudaMemcpyDeviceToHost);
    //CUDA_SAFE_CALL(cudaFree(_minSteps));
    return 1.0 / minValue;
}

double GIPC::InjectiveStepSize(double slackness, double errorRate, double* mqueue, uint4* tets)
{

    int numbers = tetrahedraNum;
    if(numbers < 1)
        return 1;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);

    _reduct_min_InjectiveTimeStep_to_double<<<blockNum, threadNum, sharedMsize>>>(
        _vertexes, tets, _moveDir, mqueue, slackness, errorRate, numbers);


    numbers  = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while(numbers > 1)
    {
        //_reduct_max_box << <blockNum, threadNum, sharedMsize >> > (_tempLeafBox, numbers);
        _reduct_max_double<<<blockNum, threadNum, sharedMsize>>>(mqueue, numbers);
        numbers  = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }
    //cudaMemcpy(_leafBoxes, _tempLeafBox, sizeof(AABB), cudaMemcpyDeviceToDevice);
    double minValue;
    cudaMemcpy(&minValue, mqueue, sizeof(double), cudaMemcpyDeviceToHost);
    //printf("Injective Time step:   %f\n", 1.0 / minValue);
    //if (1.0 / minValue < 1) {
    //    system("pause");
    //}
    //CUDA_SAFE_CALL(cudaFree(_minSteps));
    return 1.0 / minValue;
}

void GIPC::buildCP()
{

    CUDA_SAFE_CALL(cudaMemset(_cpNum, 0, 5 * sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMemset(_gpNum, 0, sizeof(uint32_t)));
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    //bvh_f.Construct();
    bvh_f.SelfCollitionDetect(dHat);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    //bvh_e.Construct();
    bvh_e.SelfCollitionDetect(dHat);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    GroundCollisionDetect();
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    CUDA_SAFE_CALL(cudaMemcpy(&h_cpNum, _cpNum, 5 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(&h_gpNum, _gpNum, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    /*CUDA_SAFE_CALL(cudaMemset(_cpNum, 0, 5 * sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMemset(_gpNum, 0, sizeof(uint32_t)));*/
}

void GIPC::buildFullCP(const double& alpha)
{
    CUDA_SAFE_CALL(cudaMemset(_cpNum, 0, sizeof(uint32_t)));

    bvh_f.SelfCollitionFullDetect(dHat, _moveDir, alpha);
    bvh_e.SelfCollitionFullDetect(dHat, _moveDir, alpha);
    CUDA_SAFE_CALL(cudaMemcpy(&h_ccd_cpNum, _cpNum, sizeof(uint32_t), cudaMemcpyDeviceToHost));
}


void GIPC::buildBVH()
{
    bvh_f.Construct();
    bvh_e.Construct();
}

AABB* GIPC::calcuMaxSceneSize()
{
    return bvh_f.getSceneSize();
}

void GIPC::buildBVH_FULLCCD(const double& alpha)
{
    bvh_f.ConstructFullCCD(_moveDir, alpha);
    bvh_e.ConstructFullCCD(_moveDir, alpha);
}

// calBarrierGradientAndHessian, calBarrierHessian, calFrictionHessian moved to ipc_barrier.cu

void GIPC::computeSelfCloseVal()
{
    int numbers = h_cpNum[0];
    if(numbers <= 0)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    _calSelfCloseVal<<<blockNum, threadNum>>>(
        _vertexes, _collisonPairs, _closeMConstraintID, _closeMConstraintVal, _close_cpNum, dTol, numbers);
}

bool GIPC::checkSelfCloseVal()
{
    int numbers = h_close_cpNum;
    if(numbers < 1)
        return false;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    int*               _isChange;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_isChange, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(_isChange, 0, sizeof(int)));
    _checkSelfCloseVal<<<blockNum, threadNum>>>(
        _vertexes, _isChange, _closeMConstraintID, _closeMConstraintVal, numbers);
    int isChange;
    CUDA_SAFE_CALL(cudaMemcpy(&isChange, _isChange, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaFree(_isChange));

    return (isChange == 1);
}

double2 GIPC::minMaxSelfDist()
{
    int numbers = h_cpNum[0];
    if(numbers < 1)
        return make_double2(1e32, 0);
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double2) * (threadNum >> 5);

    double2* _queue;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_queue, numbers * sizeof(double2)));
    //CUDA_SAFE_CALL(cudaMemcpy(_tempMinMovement, _moveDir, number * sizeof(AABB), cudaMemcpyDeviceToDevice));
    _reduct_MSelfDist<<<blockNum, threadNum, sharedMsize>>>(
        _vertexes, _collisonPairs, _queue, numbers);
    //_reduct_min_double3_to_double << <blockNum, threadNum, sharedMsize >> > (_moveDir, _tempMinMovement, numbers);

    numbers  = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while(numbers > 1)
    {
        //_reduct_max_box << <blockNum, threadNum, sharedMsize >> > (_tempLeafBox, numbers);
        _reduct_M_double2<<<blockNum, threadNum, sharedMsize>>>(_queue, numbers);
        numbers  = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }
    //cudaMemcpy(_leafBoxes, _tempLeafBox, sizeof(AABB), cudaMemcpyDeviceToDevice);
    double2 minValue;
    cudaMemcpy(&minValue, _queue, sizeof(double2), cudaMemcpyDeviceToHost);
    CUDA_SAFE_CALL(cudaFree(_queue));
    minValue.x = 1.0 / minValue.x;
    return minValue;
}

// calBarrierGradient, calFrictionGradient moved to ipc_barrier.cu

void calKineticGradient(double3* _vertexes, double3* _xTilta, double3* _gradient, double* _masses, int numbers)
{
    const unsigned int threadNum = default_threads;
    if(numbers < 1)
        return;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    _calKineticGradient<<<blockNum, threadNum>>>(_vertexes, _xTilta, _gradient, _masses, numbers);
}


void calculate_fem_gradient_hessian(__GEIGEN__::Matrix3x3d* DmInverses,
                                    const double3*          vertexes,
                                    const uint4*            tetrahedras,
                                    const double*           volume,
                                    double3*                gradient,
                                    int                     tetrahedraNum_FEM,
                                    int                     tetrahedraNum_ABD,
                                    const double*           lenRate,
                                    const double*           volRate,
                                    int                     global_offset,
                                    Eigen::Matrix3d*        triplet_values,
                                    int*                    row_ids,
                                    int*                    col_ids,
                                    double                  IPC_dt,
                                    int global_hessian_fem_offset)
{
    int numbers = tetrahedraNum_FEM;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    _calculate_fem_gradient_hessian<<<blockNum, threadNum>>>(
        DmInverses + tetrahedraNum_ABD,
        vertexes,
        tetrahedras + tetrahedraNum_ABD,
        volume + tetrahedraNum_ABD,
        gradient,
        numbers,
        lenRate + tetrahedraNum_ABD,
        volRate + tetrahedraNum_ABD,
        //tet_ids,
        global_offset,
        triplet_values,
        row_ids,
        col_ids,
        IPC_dt,
        global_hessian_fem_offset);
}

void calculate_triangle_fem_gradient_hessian(__GEIGEN__::Matrix2x2d* triDmInverses,
                                             const double3*   vertexes,
                                             const uint3*     triangles,
                                             const double*    area,
                                             double3*         gradient,
                                             int              triangleNum,
                                             double           stretchStiff,
                                             double           shearStiff,
                                             double           strainRate,
                                             int              global_offset,
                                             Eigen::Matrix3d* triplet_values,
                                             int*             row_ids,
                                             int*             col_ids,
                                             double           IPC_dt,
                                             int global_hessian_fem_offset)
{
    int numbers = triangleNum;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    _calculate_triangle_fem_gradient_hessian<<<blockNum, threadNum>>>(triDmInverses,
                                                                      vertexes,
                                                                      triangles,
                                                                      area,
                                                                      gradient,
                                                                      triangleNum,
                                                                      stretchStiff,
                                                                      shearStiff,
                                                                      IPC_dt,
                                                                      global_offset,
                                                                      triplet_values,
                                                                      row_ids,
                                                                      col_ids,
                                                                      strainRate,
                                                                      global_hessian_fem_offset);
}


void calculate_triangle_fem_strain_limiting_gradient_hessian(__GEIGEN__::Matrix2x2d* triDmInverses,
                                                             const double3* vertexes,
                                                             const uint3* triangles,
                                                             __GEIGEN__::Matrix9x9d* Hessians,
                                                             const uint32_t& offset,
                                                             const double* area,
                                                             double3* gradient,
                                                             int triangleNum,
                                                             Eigen::Matrix3d* U3x2,
                                                             Eigen::Matrix2d* V3x2,
                                                             Eigen::Vector2d* S3x2,
                                                             double IPC_dt)
{
    int numbers = triangleNum;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    _calculate_triangle_fem_strain_limiting_gradient_hessian<<<blockNum, threadNum>>>(
        triDmInverses, vertexes, triangles, Hessians, offset, area, gradient, triangleNum, U3x2, V3x2, S3x2, IPC_dt);
}


void calculate_triangle_fem_deformationF(__GEIGEN__::Matrix2x2d* triDmInverses,
                                         const double3*          vertexes,
                                         const uint3*            triangles,
                                         int                     triangleNum,
                                         Eigen::Matrix<double, 3, 2>* F3x2)
{
    int numbers = triangleNum;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    _calculate_triangle_fem_deformationF<<<blockNum, threadNum>>>(triDmInverses,
                                                                  vertexes,
                                                                  triangles,

                                                                  triangleNum,
                                                                  F3x2);
}

void calculate_bending_gradient_hessian(const double3*   vertexes,
                                        const double3*   rest_vertexes,
                                        const uint2*     edges,
                                        const uint2*     edges_adj_vertex,
                                        double3*         gradient,
                                        int              edgeNum,
                                        double           bendStiff,
                                        int              global_offset,
                                        Eigen::Matrix3d* triplet_values,
                                        int*             row_ids,
                                        int*             col_ids,
                                        double           IPC_dt,
                                        int global_hessian_fem_offset)
{
    int numbers = edgeNum;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    _calculate_bending_gradient_hessian<<<blockNum, threadNum>>>(vertexes,
                                                                 rest_vertexes,
                                                                 edges,
                                                                 edges_adj_vertex,
                                                                 gradient,
                                                                 edgeNum,
                                                                 bendStiff,
                                                                 global_offset,
                                                                 triplet_values,
                                                                 row_ids,
                                                                 col_ids,
                                                                 IPC_dt,
                                                                 global_hessian_fem_offset);
}

void calculate_fem_gradient(__GEIGEN__::Matrix3x3d* DmInverses,
                            const double3*          vertexes,
                            const uint4*            tetrahedras,
                            const double*           volume,
                            double3*                gradient,
                            int                     tetrahedraNum,
                            double*                 lenRate,
                            double*                 volRate,
                            double                  dt)
{
    int numbers = tetrahedraNum;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    _calculate_fem_gradient<<<blockNum, threadNum>>>(
        DmInverses, vertexes, tetrahedras, volume, gradient, tetrahedraNum, lenRate, volRate, dt);
}

void calculate_triangle_fem_gradient(__GEIGEN__::Matrix2x2d* triDmInverses,
                                     const double3*          vertexes,
                                     const uint3*            triangles,
                                     const double*           area,
                                     double3*                gradient,
                                     int                     triangleNum,
                                     double                  stretchStiff,
                                     double                  shearStiff,
                                     double                  IPC_dt,
                                     double                  strainRate)
{
    int numbers = triangleNum;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    _calculate_triangle_fem_gradient<<<blockNum, threadNum>>>(
        triDmInverses, vertexes, triangles, area, gradient, triangleNum, stretchStiff, shearStiff, IPC_dt, strainRate);
}

double calcMinMovement(const double3* _moveDir, double* _queue, const int& number)
{

    int numbers = number;
    if(numbers < 1)
        return 0;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);

    /*double* _tempMinMovement;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_tempMinMovement, numbers * sizeof(double)));*/
    //CUDA_SAFE_CALL(cudaMemcpy(_tempMinMovement, _moveDir, number * sizeof(AABB), cudaMemcpyDeviceToDevice));

    _reduct_max_double3_to_double<<<blockNum, threadNum, sharedMsize>>>(_moveDir, _queue, numbers);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());

    numbers  = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while(numbers > 1)
    {
        //_reduct_max_box << <blockNum, threadNum, sharedMsize >> > (_tempLeafBox, numbers);
        _reduct_max_double<<<blockNum, threadNum, sharedMsize>>>(_queue, numbers);
        numbers  = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }
    //cudaMemcpy(_leafBoxes, _tempLeafBox, sizeof(AABB), cudaMemcpyDeviceToDevice);
    double minValue;
    cudaMemcpy(&minValue, _queue, sizeof(double), cudaMemcpyDeviceToHost);
    //CUDA_SAFE_CALL(cudaFree(_tempMinMovement));
    return minValue;
}

void stepForward(double3* _vertexes,
                 double3* _vertexesTemp,
                 double3* _moveDir,
                 int*     bType,
                 double   alpha,
                 bool     moveBoundary,
                 int      numbers)
{
    const unsigned int threadNum = default_threads;
    if(numbers < 1)
        return;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    _stepForward<<<blockNum, threadNum>>>(
        _vertexes, _vertexesTemp, _moveDir, bType, alpha, moveBoundary, numbers);
}

void GIPC::step_forward(device_TetraData& TetMesh, double alpha, bool move_boundary)
{
    auto vertexes = muda::BufferView<double3>{TetMesh.vertexes, vertexNum};
    auto vertexes_temp = muda::BufferView<double3>{TetMesh.temp_double3Mem, vertexNum};
    auto move_dir = muda::BufferView<double3>{_moveDir, vertexNum};
    if(abd_fem_count_info.fem_point_num > 0)
    {
        auto fem_vertexes = vertexes.subview(abd_fem_count_info.fem_point_offset,
                                             abd_fem_count_info.fem_point_num);
        auto fem_vertexes_temp =
            vertexes_temp.subview(abd_fem_count_info.fem_point_offset,
                                  abd_fem_count_info.fem_point_num);

        auto fem_move_dir = move_dir.subview(abd_fem_count_info.fem_point_offset,
                                             abd_fem_count_info.fem_point_num);

        auto btype = muda::BufferView<int>{TetMesh.BoundaryType, vertexNum}.subview(
            abd_fem_count_info.fem_point_offset, abd_fem_count_info.fem_point_num);


        stepForward(fem_vertexes.data(),
                    fem_vertexes_temp.data(),
                    fem_move_dir.data(),
                    btype.data(),
                    alpha,
                    move_boundary,
                    fem_vertexes.size());
    }
    if(abd_fem_count_info.abd_point_num <= 0)
        return;

    auto abd_vertexes = muda::BufferView<double3>{TetMesh.vertexes, vertexNum}.subview(
        abd_fem_count_info.abd_point_offset, abd_fem_count_info.abd_point_num);

    m_abd_system->step_forward(*m_abd_sim_data, abd_vertexes, alpha);
}

void updateSurfaces(uint32_t* sortIndex, uint3* _faces, const int& offset_num, const int& numbers)
{
    const unsigned int threadNum = default_threads;
    if(numbers < 1)
        return;
    int blockNum = (numbers + threadNum - 1) / threadNum;  //
    _updateSurfaces<<<blockNum, threadNum>>>(sortIndex, _faces, offset_num, numbers);
}

void updateSurfaceEdges(uint32_t* sortIndex, uint2* _edges, const int& offset_num, const int& numbers)
{
    const unsigned int threadNum = default_threads;
    if(numbers < 1)
        return;
    int blockNum = (numbers + threadNum - 1) / threadNum;  //
    _updateEdges<<<blockNum, threadNum>>>(sortIndex, _edges, offset_num, numbers);
}

void updateTriEdges_adjVerts(uint32_t*  sortIndex,
                             uint2*     _tri_edges,
                             uint2*     _adj_verts,
                             const int& offset_num,
                             const int& numbers)
{
    const unsigned int threadNum = default_threads;
    if(numbers < 1)
        return;
    int blockNum = (numbers + threadNum - 1) / threadNum;  //
    _updateTriEdges_adjVerts<<<blockNum, threadNum>>>(
        sortIndex, _tri_edges, _adj_verts, offset_num, numbers);
}


void updateSurfaceVerts(uint32_t* sortIndex, uint32_t* _sVerts, const int& offset_num, const int& numbers)
{
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    _updateSurfVerts<<<blockNum, threadNum>>>(sortIndex, _sVerts, offset_num, numbers);
}

void updateNeighborInfo(unsigned int*   _neighborList,
                        unsigned int*   d_neighborListInit,
                        unsigned int*   _neighborNum,
                        unsigned int*   _neighborNumInit,
                        unsigned int*   _neighborStart,
                        unsigned int*   _neighborStartTemp,
                        const uint32_t* sortIndex,
                        const uint32_t* sortMapVertIndex,
                        const int&      numbers,
                        const int&      neighborListSize)
{
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    _updateNeighborNum<<<blockNum, threadNum>>>(_neighborNumInit, _neighborNum, sortIndex, numbers);
    thrust::exclusive_scan(thrust::device_ptr<unsigned int>(_neighborNum),
                           thrust::device_ptr<unsigned int>(_neighborNum) + numbers,
                           thrust::device_ptr<unsigned int>(_neighborStartTemp));
    _updateNeighborList<<<blockNum, threadNum>>>(d_neighborListInit,
                                                 _neighborList,
                                                 _neighborNum,
                                                 _neighborStart,
                                                 _neighborStartTemp,
                                                 sortIndex,
                                                 sortMapVertIndex,
                                                 numbers);
    CUDA_SAFE_CALL(cudaMemcpy(d_neighborListInit,
                              _neighborList,
                              neighborListSize * sizeof(unsigned int),
                              cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(_neighborStart,
                              _neighborStartTemp,
                              numbers * sizeof(unsigned int),
                              cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(
        _neighborNumInit, _neighborNum, numbers * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
}

void calcTetMChash(uint64_t*         _MChash,
                   const double3*    _vertexes,
                   uint4*            tets,
                   const AABB* _MaxBv,
                   const uint32_t*   sortMapVertIndex,
                   int               number)
{
    int numbers = number;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    _calcTetMChash<<<blockNum, threadNum>>>(
        _MChash, _vertexes, tets, _MaxBv, sortMapVertIndex, number);
}

void updateTopology(uint4* tets, uint3* tris, const uint32_t* sortMapVertIndex, int traNumber, int triNumber)
{
    int numbers = std::max(traNumber, triNumber);
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    _updateTopology<<<blockNum, threadNum>>>(tets, tris, sortMapVertIndex, traNumber, triNumber);
}

void updateVertexes(double3*                      o_vertexes,
                    const double3*                _vertexes,
                    double*                       tempM,
                    const double*                 mass,
                    __GEIGEN__::Matrix3x3d*       tempCons,
                    int*                          tempBtype,
                    const __GEIGEN__::Matrix3x3d* cons,
                    const int*                    bType,
                    const uint32_t*               sortIndex,
                    uint32_t*                     sortMapIndex,
                    int                           number)
{
    int numbers = number;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    _updateVertexes<<<blockNum, threadNum>>>(
        o_vertexes, _vertexes, tempM, mass, tempCons, tempBtype, cons, bType, sortIndex, sortMapIndex, numbers);
}

void updateTetrahedras(uint4*                        o_tetrahedras,
                       uint4*                        tetrahedras,
                       double*                       tempV,
                       const double*                 volum,
                       __GEIGEN__::Matrix3x3d*       tempDmInverse,
                       const __GEIGEN__::Matrix3x3d* dmInverse,
                       const uint32_t*               sortTetIndex,
                       const uint32_t*               sortMapVertIndex,
                       int                           number)
{
    int numbers = number;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    _updateTetrahedras<<<blockNum, threadNum>>>(
        o_tetrahedras, tetrahedras, tempV, volum, tempDmInverse, dmInverse, sortTetIndex, sortMapVertIndex, number);
}

void calcVertMChash(uint64_t* _MChash, const double3* _vertexes, const AABB* _MaxBv, int number)
{
    int numbers = number;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    _calcVertMChash<<<blockNum, threadNum>>>(_MChash, _vertexes, _MaxBv, number);
}

void sortGeometry(device_TetraData& TetMesh,
                  const AABB*       _MaxBv,
                  const int&        vertex_num,
                  const int&        tetradedra_num,
                  const int&        triangle_num)
{


}

////////////////////////TO DO LATER/////////////////////////////////////////


void compute_H_b(double d, double dHat, double& H)
{
    double t = d - dHat;
    H = (std::log(d / dHat) * -2.0 - t * 4.0 / d) + 1.0 / (d * d) * (t * t);
}

void GIPC::suggestKappa(double& kappa)
{
    double H_b;
    //double bboxDiagSize2 = __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(bvh_f.scene.upper, bvh_f.scene.lower));
    compute_H_b(1.0e-16 * bboxDiagSize2, dHat, H_b);
    if(meanMass == 0.0)
    {
        kappa = minKappaCoef / (4.0e-16 * bboxDiagSize2 * H_b);
    }
    else
    {
        kappa = minKappaCoef * meanMass / (4.0e-16 * bboxDiagSize2 * H_b);
    }
    //    printf("bboxDiagSize2: %f\n", bboxDiagSize2);
    //    printf("H_b: %f\n", H_b);
    //    printf("sug Kappa: %f\n", kappa);
}

void GIPC::upperBoundKappa(double& kappa)
{
    double H_b;
    //double bboxDiagSize2 = __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(bvh_f.scene.upper, bvh_f.scene.lower));//(maxConer - minConer).squaredNorm();
    compute_H_b(1.0e-16 * bboxDiagSize2, dHat, H_b);
    double kappaMax = 100 * minKappaCoef * meanMass / (4.0e-16 * bboxDiagSize2 * H_b);
    //printf("max Kappa: %f\n", kappaMax);
    if(meanMass == 0.0)
    {
        kappaMax = 100 * minKappaCoef / (4.0e-16 * bboxDiagSize2 * H_b);
    }

    if(kappa > kappaMax)
    {
        kappa = kappaMax;
    }
}


void GIPC::initKappa(device_TetraData& TetMesh)
{
    if(h_cpNum[0] > 0)
    {
        double3* _GE = TetMesh.fb;
        double3* _gc = TetMesh.temp_double3Mem;
        //CUDA_SAFE_CALL(cudaMalloc((void**)&_gc, vertexNum * sizeof(double3)));
        //CUDA_SAFE_CALL(cudaMalloc((void**)&_GE, vertexNum * sizeof(double3)));
        CUDA_SAFE_CALL(cudaMemset(_gc, 0, vertexNum * sizeof(double3)));
        CUDA_SAFE_CALL(cudaMemset(_GE, 0, vertexNum * sizeof(double3)));
        calKineticGradient(TetMesh.vertexes, TetMesh.xTilta, _GE, TetMesh.masses, vertexNum);
        calculate_fem_gradient(TetMesh.DmInverses,
                               TetMesh.vertexes,
                               TetMesh.tetrahedras,
                               TetMesh.volum,
                               _GE,
                               tetrahedraNum,
                               TetMesh.lengthRate,
                               TetMesh.volumeRate,
                               IPC_dt);
        //calculate_triangle_fem_gradient(TetMesh.triDmInverses, TetMesh.vertexes, TetMesh.triangles, TetMesh.area, _GE, triangleNum, stretchStiff, shearStiff, IPC_dt);
        computeSoftConstraintGradient(_GE);
        computeGroundGradient(_gc, 1);
        calBarrierGradient(_gc, 1);
        double gsum = reduction2Kappa(0, _gc, _GE, pcg_data.squeue, vertexNum);
        double gsnorm = reduction2Kappa(1, _gc, _GE, pcg_data.squeue, vertexNum);
        //CUDA_SAFE_CALL(cudaFree(_gc));
        //CUDA_SAFE_CALL(cudaFree(_GE));
        double minKappa = -gsum / gsnorm;
        if(minKappa > 0.0)
        {
            Kappa = minKappa;
        }
        suggestKappa(minKappa);
        if(Kappa < minKappa)
        {
            Kappa = minKappa;
        }
        upperBoundKappa(Kappa);
    }

    //printf("Kappa ====== %f\n", Kappa);
}


void GIPC::partitionContactHessian()
{

    muda::DeviceRadixSort().SortPairs(gipc_global_triplet.block_hash_value(),
                                      gipc_global_triplet.block_sort_hash_value(),
                                      gipc_global_triplet.block_index(),
                                      gipc_global_triplet.block_sort_index(),
                                      gipc_global_triplet.global_collision_triplet_offset);

    int threadNum = 256;

    LaunchCudaKernal_default(
        gipc_global_triplet.global_collision_triplet_offset,
        threadNum,
        0,
        _reorder_triplets,
        gipc_global_triplet.block_row_indices(),
        gipc_global_triplet.block_col_indices(),
        gipc_global_triplet.block_values(),
        gipc_global_triplet.block_row_indices(gipc_global_triplet.global_collision_triplet_offset),
        gipc_global_triplet.block_col_indices(gipc_global_triplet.global_collision_triplet_offset),
        gipc_global_triplet.block_values(gipc_global_triplet.global_collision_triplet_offset),
        (const uint32_t*)gipc_global_triplet.block_sort_index(),
        gipc_global_triplet.global_collision_triplet_offset);

    //gipc_global_triplet.d_abd_abd_contact_start_id = -1;
    //gipc_global_triplet.d_abd_fem_contact_start_id = -1;
    //gipc_global_triplet.d_fem_abd_contact_start_id = -1;
    //gipc_global_triplet.d_fem_fem_contact_start_id = -1;

    CUDA_SAFE_CALL(cudaMemset(gipc_global_triplet.d_abd_abd_contact_start_id, -1, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(gipc_global_triplet.d_abd_fem_contact_start_id, -1, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(gipc_global_triplet.d_fem_abd_contact_start_id, -1, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(gipc_global_triplet.d_fem_fem_contact_start_id, -1, sizeof(int)));

    size_t shareMem = (threadNum + 1) * sizeof(int);
    LaunchCudaKernal_default(gipc_global_triplet.global_collision_triplet_offset,
                             threadNum,
                             shareMem,
                             _partition_collision_triplets,
                             (const uint64_t*)gipc_global_triplet.block_sort_hash_value(),
                             gipc_global_triplet.d_abd_abd_contact_start_id,
                             gipc_global_triplet.d_abd_fem_contact_start_id,
                             gipc_global_triplet.d_fem_abd_contact_start_id,
                             gipc_global_triplet.d_fem_fem_contact_start_id,
                             //abd_fem_count_info.abd_point_num,
                             gipc_global_triplet.global_collision_triplet_offset);


    //gipc_global_triplet.h_abd_abd_contact_start_id =
    //    gipc_global_triplet.d_abd_abd_contact_start_id;
    //gipc_global_triplet.h_abd_fem_contact_start_id =
    //    gipc_global_triplet.d_abd_fem_contact_start_id;
    //gipc_global_triplet.h_fem_abd_contact_start_id =
    //    gipc_global_triplet.d_fem_abd_contact_start_id;
    //gipc_global_triplet.h_fem_fem_contact_start_id =
    //    gipc_global_triplet.d_fem_fem_contact_start_id;

    CUDA_SAFE_CALL(cudaMemcpy(&(gipc_global_triplet.h_abd_abd_contact_start_id),
                              gipc_global_triplet.d_abd_abd_contact_start_id,
                              sizeof(int),
                              cudaMemcpyDeviceToHost));

    CUDA_SAFE_CALL(cudaMemcpy(&(gipc_global_triplet.h_abd_fem_contact_start_id),
                              gipc_global_triplet.d_abd_fem_contact_start_id,
                              sizeof(int),
                              cudaMemcpyDeviceToHost));

    CUDA_SAFE_CALL(cudaMemcpy(&(gipc_global_triplet.h_fem_abd_contact_start_id),
                              gipc_global_triplet.d_fem_abd_contact_start_id,
                              sizeof(int),
                              cudaMemcpyDeviceToHost));

    CUDA_SAFE_CALL(cudaMemcpy(&(gipc_global_triplet.h_fem_fem_contact_start_id),
                              gipc_global_triplet.d_fem_fem_contact_start_id,
                              sizeof(int),
                              cudaMemcpyDeviceToHost));


    if(gipc_global_triplet.h_fem_fem_contact_start_id >= 0)
    {
        if(gipc_global_triplet.h_abd_fem_contact_start_id > 0)
        {
            gipc_global_triplet.fem_fem_contact_num =
                gipc_global_triplet.h_abd_fem_contact_start_id
                - gipc_global_triplet.h_fem_fem_contact_start_id;
            if(gipc_global_triplet.h_fem_abd_contact_start_id > 0)
            {
                gipc_global_triplet.abd_fem_contact_num =
                    gipc_global_triplet.h_fem_abd_contact_start_id
                    - gipc_global_triplet.h_abd_fem_contact_start_id;

                gipc_global_triplet.fem_abd_contact_num =
                    gipc_global_triplet.h_abd_abd_contact_start_id
                    - gipc_global_triplet.h_fem_abd_contact_start_id;
            }
            else
            {
                gipc_global_triplet.abd_fem_contact_num =
                    gipc_global_triplet.h_abd_abd_contact_start_id
                    - gipc_global_triplet.h_abd_fem_contact_start_id;

                gipc_global_triplet.fem_abd_contact_num = 0;
            }
            gipc_global_triplet.abd_abd_contact_num =
                gipc_global_triplet.global_collision_triplet_offset
                - gipc_global_triplet.h_abd_abd_contact_start_id;
        }
        else if(gipc_global_triplet.h_abd_abd_contact_start_id > 0)
        {
            gipc_global_triplet.fem_fem_contact_num =
                gipc_global_triplet.h_abd_abd_contact_start_id
                - gipc_global_triplet.h_fem_fem_contact_start_id;
            gipc_global_triplet.abd_abd_contact_num =
                gipc_global_triplet.global_collision_triplet_offset
                - gipc_global_triplet.h_abd_abd_contact_start_id;

            gipc_global_triplet.abd_fem_contact_num = 0;
            gipc_global_triplet.fem_abd_contact_num = 0;
        }
        else
        {
            gipc_global_triplet.fem_fem_contact_num =
                gipc_global_triplet.global_collision_triplet_offset;

            gipc_global_triplet.abd_abd_contact_num = 0;

            gipc_global_triplet.abd_fem_contact_num = 0;
            gipc_global_triplet.fem_abd_contact_num = 0;
        }
    }
    else if(gipc_global_triplet.h_abd_abd_contact_start_id >= 0)
    {
        gipc_global_triplet.abd_abd_contact_num =
            gipc_global_triplet.global_collision_triplet_offset;

        gipc_global_triplet.fem_fem_contact_num = 0;
        gipc_global_triplet.abd_fem_contact_num = 0;
        gipc_global_triplet.fem_abd_contact_num = 0;
    }
    else
    {
        gipc_global_triplet.abd_abd_contact_num = 0;
        gipc_global_triplet.fem_fem_contact_num = 0;
        gipc_global_triplet.abd_fem_contact_num = 0;
        gipc_global_triplet.fem_abd_contact_num = 0;
    }

    gipc_global_triplet.h_fem_fem_contact_start_id = 0;
    gipc_global_triplet.h_abd_fem_contact_start_id =
        gipc_global_triplet.h_fem_fem_contact_start_id + gipc_global_triplet.fem_fem_contact_num;
    gipc_global_triplet.h_fem_abd_contact_start_id =
        gipc_global_triplet.h_abd_fem_contact_start_id + gipc_global_triplet.abd_fem_contact_num;
    gipc_global_triplet.h_abd_abd_contact_start_id =
        gipc_global_triplet.h_fem_abd_contact_start_id + gipc_global_triplet.fem_abd_contact_num;

    CUDA_SAFE_CALL(
        cudaMemcpy(gipc_global_triplet.block_row_indices(),
                   gipc_global_triplet.block_row_indices() + gipc_global_triplet.global_collision_triplet_offset,
                   gipc_global_triplet.global_collision_triplet_offset * sizeof(int),
                   cudaMemcpyDeviceToDevice));

    CUDA_SAFE_CALL(
        cudaMemcpy(gipc_global_triplet.block_col_indices(),
                   gipc_global_triplet.block_col_indices() + gipc_global_triplet.global_collision_triplet_offset,
                   gipc_global_triplet.global_collision_triplet_offset * sizeof(int),
                   cudaMemcpyDeviceToDevice));

    CUDA_SAFE_CALL(cudaMemcpy(
        gipc_global_triplet.block_values(),
        gipc_global_triplet.block_values() + gipc_global_triplet.global_collision_triplet_offset,
        gipc_global_triplet.global_collision_triplet_offset * sizeof(Eigen::Matrix3d),
        cudaMemcpyDeviceToDevice));
}

float GIPC::computeGradientAndHessian(device_TetraData& TetMesh)
{
    gipc::Timer timer{"cal_gradient_hessian"};

    CUDA_SAFE_CALL(cudaMemset(TetMesh.fb, 0, vertexNum * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMemset(TetMesh.shape_grads, 0, vertexNum * sizeof(double3)));

    //muda::BufferView<double3>{TetMesh.shape_grads, vertexNum}.fill(double3{0, 0, 0});


    auto shape_grads   = TetMesh.shape_grads;
    auto contact_grads = TetMesh.fb;
    {
        gipc::Timer timer{"cal_kinetic_gradient"};
        calKineticGradient(
            TetMesh.vertexes, TetMesh.xTilta, shape_grads, TetMesh.masses, vertexNum);
    }

    gipc_global_triplet.global_triplet_offset = 0;

    {
        gipc::Timer timer{"cal_barrier_gradient_hessian"};
        CUDA_SAFE_CALL(cudaMemset(_cpNum, 0, 5 * sizeof(uint32_t)));
        //calBarrierHessian();
        //calBarrierGradient(contact_grads, Kappa);

        calBarrierGradientAndHessian(contact_grads, Kappa);
        gipc_global_triplet.global_triplet_offset +=
            h_cpNum[4] * M12_Off + h_cpNum[3] * M9_Off + h_cpNum[2] * M6_Off;
    }

    float time00 = 0;

#ifdef USE_FRICTION
    {

        gipc::Timer timer{"cal_friction_gradient_hessian"};
        calFrictionGradient(contact_grads, TetMesh);
        //CUDA_SAFE_CALL(cudaDeviceSynchronize());
        calFrictionHessian(TetMesh);
        gipc_global_triplet.global_triplet_offset +=
            h_cpNum_last[4] * M12_Off + h_cpNum_last[3] * M9_Off
            + h_cpNum_last[2] * M6_Off + h_gpNum_last;
        //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }
#endif

    computeGroundGradientAndHessian(contact_grads);
    gipc_global_triplet.global_triplet_offset += h_gpNum;
    gipc_global_triplet.global_collision_triplet_offset =
        gipc_global_triplet.global_triplet_offset;

    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    gipc_global_triplet.update_hash_value(abd_fem_count_info.abd_point_num);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    partitionContactHessian();
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());


    {
        gipc::Timer timer{"setup_abd_system_gradient_hessian"};

        m_abd_system->setup_abd_system_gradient_hessian(
            *m_abd_sim_data,
            TetMesh.BoundaryType,
            muda::BufferView<double3>{TetMesh.fb, vertexNum}.subview(
                abd_fem_count_info.abd_point_offset, abd_fem_count_info.abd_point_num),
            gipc_global_triplet);
    }

    int fem_global_hessian_index_offset =
        -abd_fem_count_info.abd_point_num + abd_fem_count_info.abd_body_num * 4;
    {
        muda::ParallelFor(256)
            .kernel_name(__FUNCTION__)
            .apply(gipc_global_triplet.fem_fem_contact_num,
                   [cfem_rows = gipc_global_triplet.block_row_indices(
                        gipc_global_triplet.h_fem_fem_contact_start_id),
                    cfem_cols = gipc_global_triplet.block_col_indices(
                        gipc_global_triplet.h_fem_fem_contact_start_id),
                    cfem_vals = gipc_global_triplet.block_values(
                        gipc_global_triplet.h_fem_fem_contact_start_id),
                    fem_global_hessian_index_offset] __device__(int i) mutable
                   {
                       int row = cfem_rows[i];
                       int col = cfem_cols[i];
                       if(row <= col)
                       {
                           cfem_rows[i] = row + fem_global_hessian_index_offset;
                           cfem_cols[i] = col + fem_global_hessian_index_offset;
                       }
                       else
                       {
                           cfem_rows[i] = col + fem_global_hessian_index_offset;
                           cfem_cols[i] = row + fem_global_hessian_index_offset;
                           cfem_vals[i].setZero();
                       }
                   });
    }

    {
        gipc::Timer timer{"cal_fem_gradient_hessian"};

        //CUDA_SAFE_CALL(cudaDeviceSynchronize());
        calculate_fem_gradient_hessian(TetMesh.DmInverses,
                                       TetMesh.vertexes,
                                       TetMesh.tetrahedras,
                                       TetMesh.volum,
                                       shape_grads,
                                       abd_fem_count_info.fem_tet_num,
                                       abd_fem_count_info.abd_tet_num,
                                       TetMesh.lengthRate,
                                       TetMesh.volumeRate,
                                       gipc_global_triplet.global_triplet_offset,
                                       gipc_global_triplet.block_values(),
                                       gipc_global_triplet.block_row_indices(),
                                       gipc_global_triplet.block_col_indices(),
                                       IPC_dt,
                                       fem_global_hessian_index_offset);
        gipc_global_triplet.global_triplet_offset += abd_fem_count_info.fem_tet_num * 10;


        calculate_bending_gradient_hessian(TetMesh.vertexes,
                                           TetMesh.rest_vertexes,
                                           TetMesh.tri_edges,
                                           TetMesh.tri_edge_adj_vertex,
                                           shape_grads,
                                           tri_edge_num,
                                           bendStiff,
                                           gipc_global_triplet.global_triplet_offset,
                                           gipc_global_triplet.block_values(),
                                           gipc_global_triplet.block_row_indices(),
                                           gipc_global_triplet.block_col_indices(),
                                           IPC_dt,
                                           fem_global_hessian_index_offset);
        gipc_global_triplet.global_triplet_offset += tri_edge_num * 10;
        //CUDA_SAFE_CALL(cudaDeviceSynchronize());

        calculate_triangle_fem_gradient_hessian(TetMesh.triDmInverses,
                                                TetMesh.vertexes,
                                                TetMesh.triangles,
                                                TetMesh.area,
                                                shape_grads,
                                                triangleNum,
                                                stretchStiff,
                                                shearStiff,
                                                strainRate,
                                                gipc_global_triplet.global_triplet_offset,
                                                gipc_global_triplet.block_values(),
                                                gipc_global_triplet.block_row_indices(),
                                                gipc_global_triplet.block_col_indices(),
                                                IPC_dt,
                                                fem_global_hessian_index_offset);

        gipc_global_triplet.global_triplet_offset += triangleNum * 6;


        computeSoftConstraintGradientAndHessian(shape_grads, fem_global_hessian_index_offset);
        gipc_global_triplet.global_triplet_offset += softNum;

        //int massNum =
        muda::ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(abd_fem_count_info.fem_point_num,
                   [mass      = TetMesh.masses,
                    cfem_rows = gipc_global_triplet.block_row_indices(
                        gipc_global_triplet.global_triplet_offset),
                    cfem_cols = gipc_global_triplet.block_col_indices(
                        gipc_global_triplet.global_triplet_offset),
                    triplet_fem = gipc_global_triplet.block_values(
                        gipc_global_triplet.global_triplet_offset),
                    fem_global_hessian_index_offset,
                    fem_pint_start = abd_fem_count_info.abd_point_num,
                    abd_num = abd_fem_count_info.abd_body_num] __device__(int i) mutable
                   {
                       triplet_fem[i] =
                           mass[i + fem_pint_start] * gipc::Matrix3x3::Identity();
                       cfem_rows[i] = i + abd_num * 4;
                       cfem_cols[i] = i + abd_num * 4;
                   });
        gipc_global_triplet.global_triplet_offset += abd_fem_count_info.fem_point_num;

        //cudaMemcpy(TetMesh.totalForce, contact_grads, vertexNum * sizeof(double3), cudaMemcpyDeviceToDevice);
        //getTotalForce(shape_grads, TetMesh.totalForce);
    }

    return time00;
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
}


double GIPC::Energy_Add_Reduction_Algorithm(int type, device_TetraData& TetMesh)
{
    int tet_offset   = abd_fem_count_info.fem_tet_offset;
    int tet_count    = abd_fem_count_info.fem_tet_num;
    int point_offset = abd_fem_count_info.fem_point_offset;
    int point_count  = abd_fem_count_info.fem_point_num;

    int numbers = tet_count;

    if(type == 0 || type == 3)
    {
        numbers = point_count;
    }
    else if(type == 2)
    {
        numbers = h_cpNum[0];
    }
    else if(type == 4)
    {
        numbers = h_gpNum;
    }
    else if(type == 5)
    {
        numbers = h_cpNum_last[0];
    }
    else if(type == 6)
    {
        numbers = h_gpNum_last;
    }
    else if(type == 7 || type == 1)
    {
        numbers = tet_count;
    }
    else if(type == 8 || type == 11)
    {
        numbers = triangleNum;
    }
    else if(type == 9)
    {
        numbers = softNum;
    }
    else if(type == 10)
    {
        numbers = tri_edge_num;
    }
    if(numbers == 0)
        return 0;
    double* queue = pcg_data.squeue;
    //CUDA_SAFE_CALL(cudaMalloc((void**)&queue, numbers * sizeof(double)));*/

    const unsigned int threadNum = 256;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);
    switch(type)
    {
        case 0:
            _getKineticEnergy_Reduction_3D<<<blockNum, threadNum, sharedMsize>>>(
                TetMesh.vertexes + point_offset,
                TetMesh.xTilta + point_offset,
                queue,
                TetMesh.masses + point_offset,
                numbers);
            break;
        case 1:
            _getFEMEnergy_Reduction_3D<<<blockNum, threadNum, sharedMsize>>>(
                queue,
                TetMesh.vertexes,
                TetMesh.tetrahedras + tet_offset,
                TetMesh.DmInverses + tet_offset,
                TetMesh.volum + tet_offset,
                numbers,
                TetMesh.lengthRate + tet_offset,
                TetMesh.volumeRate + tet_offset);
            break;
        case 2:
            _getBarrierEnergy_Reduction_3D<<<blockNum, threadNum, sharedMsize>>>(
                queue, TetMesh.vertexes, TetMesh.rest_vertexes, _collisonPairs, Kappa, dHat, numbers);
            break;
        case 3:
            _getDeltaEnergy_Reduction<<<blockNum, threadNum, sharedMsize>>>(
                queue, TetMesh.fb + point_offset, _moveDir + point_offset, numbers);
            break;
        case 4:
            _computeGroundEnergy_Reduction<<<blockNum, threadNum, sharedMsize>>>(
                queue, TetMesh.vertexes, _groundOffset, _groundNormal, _environment_collisionPair, dHat, Kappa, numbers);
            break;
        case 5:
            _getFrictionEnergy_Reduction_3D<<<blockNum, threadNum, sharedMsize>>>(
                queue,
                TetMesh.vertexes,
                TetMesh.o_vertexes,
                _collisonPairs_lastH,
                numbers,
                IPC_dt,
                distCoord,
                tanBasis,
                lambda_lastH_scalar,
                fDhat * IPC_dt * IPC_dt,
                sqrt(fDhat) * IPC_dt);
            break;
        case 6:
            _getFrictionEnergy_gd_Reduction_3D<<<blockNum, threadNum, sharedMsize>>>(
                queue,
                TetMesh.vertexes,
                TetMesh.o_vertexes,
                _groundNormal,
                _collisonPairs_lastH_gd,
                numbers,
                IPC_dt,
                lambda_lastH_scalar_gd,
                sqrt(fDhat) * IPC_dt);
            break;
        case 7:
            _getRestStableNHKEnergy_Reduction_3D<<<blockNum, threadNum, sharedMsize>>>(
                queue, TetMesh.volum + tet_offset, numbers, lengthRate, volumeRate);
            break;
        case 8:
            _get_triangleFEMEnergy_Reduction_3D<<<blockNum, threadNum, sharedMsize>>>(
                queue,
                TetMesh.vertexes,
                TetMesh.triangles,
                TetMesh.triDmInverses,
                TetMesh.area,
                numbers,
                stretchStiff,
                shearStiff,
                strainRate);
            break;
        case 9:
            _computeSoftConstraintEnergy_Reduction<<<blockNum, threadNum, sharedMsize>>>(
                queue, TetMesh.vertexes, TetMesh.targetVert, TetMesh.targetIndex, softMotionRate, animation_fullRate, numbers);
            break;
        case 10:
            _getBendingEnergy_Reduction<<<blockNum, threadNum, sharedMsize>>>(
                queue,
                TetMesh.vertexes,
                TetMesh.rest_vertexes,
                TetMesh.tri_edges,
                TetMesh.tri_edge_adj_vertex,
                numbers,
                bendStiff);
            break;
    }
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    numbers  = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while(numbers > 1)
    {
        __add_reduction<<<blockNum, threadNum, sharedMsize>>>(queue, numbers);
        numbers  = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }
    double result;
    cudaMemcpy(&result, queue, sizeof(double), cudaMemcpyDeviceToHost);
    //CUDA_SAFE_CALL(cudaFree(queue));
    return result;
}


double GIPC::computeEnergy(device_TetraData& TetMesh)
{
    double Energy      = 0.0;
    auto   fem_kinetic = Energy_Add_Reduction_Algorithm(0, TetMesh);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    Energy += fem_kinetic;

    auto abd_kinetic = m_abd_system->cal_abd_kinetic_energy(*m_abd_sim_data);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    Energy += abd_kinetic;

    auto abd_shape = m_abd_system->cal_abd_shape_energy(*m_abd_sim_data);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    Energy += abd_shape;

    auto fem = IPC_dt * IPC_dt * Energy_Add_Reduction_Algorithm(1, TetMesh);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    Energy += fem;

    auto tri_fem = IPC_dt * IPC_dt * Energy_Add_Reduction_Algorithm(8, TetMesh);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    Energy += tri_fem;

    auto bend = IPC_dt * IPC_dt * Energy_Add_Reduction_Algorithm(10, TetMesh);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    Energy += bend;

    auto constraint = Energy_Add_Reduction_Algorithm(9, TetMesh);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    Energy += constraint;

    auto barrier = Energy_Add_Reduction_Algorithm(2, TetMesh);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    Energy += barrier;

    auto ground = Kappa * Energy_Add_Reduction_Algorithm(4, TetMesh);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    Energy += ground;

    //std::cout << "fem_kinetic: " << fem_kinetic << std::endl;
    //std::cout << "abd_kinetic: " << abd_kinetic << std::endl;
    //std::cout << "abd_shape: " << abd_shape << std::endl;
    //std::cout << "fem: " << fem << std::endl;
    //std::cout << "tri_fem: " << tri_fem << std::endl;
    //std::cout << "bend: " << bend << std::endl;
    //std::cout << "constraint: " << constraint << std::endl;
    //std::cout << "barrier: " << barrier << std::endl;
    //std::cout << "ground: " << ground << std::endl;

#ifdef USE_FRICTION
    auto fric = frictionRate * Energy_Add_Reduction_Algorithm(5, TetMesh);
    Energy += fric;
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    auto fric_ground = gd_frictionRate * Energy_Add_Reduction_Algorithm(6, TetMesh);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    Energy += fric_ground;
#endif

    return Energy;
}

int GIPC::calculateMovingDirection(device_TetraData& TetMesh, int cpNum, int preconditioner_type)
{
    gipc::Timer timer{"solve_linear_system"};
    auto        iter = 0;

    iter = m_global_linear_system->solve_linear_system();


    auto& json = gipc::Statistics::instance().at_current_frame();
    json["newton"].back()["pcg"]["iterations"] = iter;
    return iter;
}


bool edgeTriIntersectionQuery(const int*     _btype,
                              const double3* _vertexes,
                              const uint2*   _edges,
                              const uint3*   _faces,
                              const AABB*    _edge_bvs,
                              const Node*    _edge_nodes,
                              double         dHat,
                              int            number)
{
    int numbers = number;
    if(numbers <= 0)
        return false;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    int*               _isIntersect;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_isIntersect, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(_isIntersect, 0, sizeof(int)));

    _edgeTriIntersectionQuery<<<blockNum, threadNum>>>(
        _btype, _vertexes, _edges, _faces, _edge_bvs, _edge_nodes, _isIntersect, dHat, numbers);

    int h_isITST;
    cudaMemcpy(&h_isITST, _isIntersect, sizeof(int), cudaMemcpyDeviceToHost);
    CUDA_SAFE_CALL(cudaFree(_isIntersect));
    if(h_isITST < 0)
    {
        return true;
    }
    return false;
}

bool GIPC::checkEdgeTriIntersectionIfAny(device_TetraData& TetMesh)
{
    return edgeTriIntersectionQuery(bvh_e._btype,
                                    TetMesh.vertexes,
                                    bvh_e._edges,
                                    bvh_f._faces,
                                    bvh_e._bvs,
                                    bvh_e._nodes,
                                    dHat,
                                    bvh_f.face_number);
}

bool GIPC::checkGroundIntersection()
{
    int numbers = h_gpNum;
    if(numbers <= 0)
        return false;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //

    int* _isIntersect;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_isIntersect, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(_isIntersect, 0, sizeof(int)));
    _checkGroundIntersection<<<blockNum, threadNum>>>(
        _vertexes, _groundOffset, _groundNormal, _environment_collisionPair, _isIntersect, numbers);

    int h_isITST;
    cudaMemcpy(&h_isITST, _isIntersect, sizeof(int), cudaMemcpyDeviceToHost);
    CUDA_SAFE_CALL(cudaFree(_isIntersect));
    if(h_isITST < 0)
    {
        return true;
    }
    return false;
}

bool GIPC::isIntersected(device_TetraData& TetMesh)
{
    if(checkGroundIntersection())
    {
        return true;
    }

    if(checkEdgeTriIntersectionIfAny(TetMesh))
    {
        std::cout << "is edge triangle\n";
        return true;
    }
    return false;
}


bool is_strain_limit_violated(Eigen::Vector2d* Sigma, int triangleNum)
{
    VectorXi violated(triangleNum);
    violated.setZero();
    double slimit = 1.1;
    tbb::parallel_for(0,
                      triangleNum,
                      1,
                      [&](int ii)
                      {
                          for(int i = 0; i < 2; i++)
                          {
                              double s = Sigma[ii][i];
                              if(s > slimit * (1))
                              {
                                  violated[ii]++;
                              }
                          }
                      });

    return (violated.array() != 0).any();
}

bool GIPC::lineSearch(device_TetraData& TetMesh, double& alpha, const double& cfl_alpha)
{
    muda::wait_device();
    bool   stopped       = false;
    double lastEnergyVal = computeEnergy(TetMesh);

    double c1m         = 0.0;
    double armijoParam = 0;
    if(armijoParam > 0.0)
    {
        c1m += armijoParam * Energy_Add_Reduction_Algorithm(3, TetMesh);
    }

    CUDA_SAFE_CALL(cudaMemcpy(TetMesh.temp_double3Mem,
                              TetMesh.vertexes,
                              vertexNum * sizeof(double3),
                              cudaMemcpyDeviceToDevice));

    m_abd_system->copy_q_to_q_temp(*m_abd_sim_data);

    step_forward(TetMesh, alpha, false);

    buildBVH();

    int numOfIntersect = 0;
    int insectNum      = 0;

    bool checkInterset = true;

    while(checkInterset && isIntersected(TetMesh))
    {
        printf("type 0 intersection happened 0:  %d\n", insectNum);
        insectNum++;
        alpha /= 2.0;
        numOfIntersect++;
        alpha = std::min(cfl_alpha, alpha);
        step_forward(TetMesh, alpha, false);
        buildBVH();
        //break;
    }

    buildCP();

    double testingE = computeEnergy(TetMesh);

    int    numOfLineSearch = 0;
    double LFStepSize      = alpha;

    std::cout.precision(18);
    constexpr int report_line_search_threshold = 8;

    while((testingE > lastEnergyVal + c1m * alpha) && numOfLineSearch <= report_line_search_threshold)
    {
        //std::cout << "[" << numOfLineSearch << "]   testE:    " << testingE
        //          << "      lastEnergyVal:        " << lastEnergyVal << std::endl;
        alpha /= 2.0;
        ++numOfLineSearch;

        step_forward(TetMesh, alpha, false);
        buildBVH();
        buildCP();
        testingE = computeEnergy(TetMesh);
    }
    if(numOfLineSearch > report_line_search_threshold)
        printf("!!!!!!!!!!!!!!!!!!!linesearch number is a bit high, lineSearchCount=%d !!!!!!!!!!!!!!!!!!!!!!\n",
               numOfLineSearch);


    if(alpha < LFStepSize)
    {
        bool needRecomputeCS = false;
        while(checkInterset && isIntersected(TetMesh))
        {
            printf("type 1 intersection happened 1:  %d\n", insectNum);
            insectNum++;
            alpha /= 2.0;
            numOfIntersect++;
            alpha = std::min(cfl_alpha, alpha);

            step_forward(TetMesh, alpha, false);
            buildBVH();
            needRecomputeCS = true;
        }
        if(needRecomputeCS)
        {
            buildCP();
        }
    }

    return stopped;
}


void GIPC::postLineSearch(device_TetraData& TetMesh, double alpha)
{
    if(Kappa == 0.0)
    {
        initKappa(TetMesh);
    }
    else
    {

        bool updateKappa = checkCloseGroundVal();
        if(!updateKappa)
        {
            updateKappa = checkSelfCloseVal();
        }
        if(updateKappa)
        {
            Kappa *= 2.0;
            upperBoundKappa(Kappa);
        }
        tempFree_closeConstraint();
        tempMalloc_closeConstraint();
        CUDA_SAFE_CALL(cudaMemset(_close_cpNum, 0, sizeof(uint32_t)));
        CUDA_SAFE_CALL(cudaMemset(_close_gpNum, 0, sizeof(uint32_t)));

        computeCloseGroundVal();

        computeSelfCloseVal();
    }
    //printf("------------------------------------------Kappa: %f\n", Kappa);
}

void GIPC::tempMalloc_closeConstraint()
{
    CUDA_SAFE_CALL(cudaMalloc((void**)&_closeConstraintID, h_gpNum * sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&_closeConstraintVal, h_gpNum * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&_closeMConstraintID, h_cpNum[0] * sizeof(int4)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&_closeMConstraintVal, h_cpNum[0] * sizeof(double)));
}

void GIPC::tempFree_closeConstraint()
{
    CUDA_SAFE_CALL(cudaFree(_closeConstraintID));
    CUDA_SAFE_CALL(cudaFree(_closeConstraintVal));
    CUDA_SAFE_CALL(cudaFree(_closeMConstraintID));
    CUDA_SAFE_CALL(cudaFree(_closeMConstraintVal));
}
double maxCOllisionPairNum = 0;
double totalCollisionPairs = 0;
double total_Cg_count      = 0;
double timemakePd          = 0;
#include <vector>
#include <fstream>
std::vector<int> iterV;
int              GIPC::solve_subIP(device_TetraData& TetMesh,
                      double&           time0,
                      double&           time1,
                      double&           time2,
                      double&           time3,
                      double&           time4)
{
    auto& stats_at_current_frame = gipc::Statistics::instance().at_current_frame();
    std::cout << "solve_subIP >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
              << std::endl;

    stats_at_current_frame["newton"] = gipc::Json::array();

    int iterCap = 10000, k = 0;

    CUDA_SAFE_CALL(cudaMemset(_moveDir, 0, vertexNum * sizeof(double3)));
    double totalTimeStep = 0;
    for(; k < iterCap; ++k)
    {
        stats_at_current_frame["newton"].push_back(gipc::Json::object());

        totalCollisionPairs += h_cpNum[0];
        maxCOllisionPairNum =
            (maxCOllisionPairNum > h_cpNum[0]) ? maxCOllisionPairNum : h_cpNum[0];
        cudaEvent_t start, end0, end1, end2, end3, end4;
        cudaEventCreate(&start);
        cudaEventCreate(&end0);
        cudaEventCreate(&end1);
        cudaEventCreate(&end2);
        cudaEventCreate(&end3);
        cudaEventCreate(&end4);

        //printf("\n\n\ncollision num  %d\n\n\n", h_cpNum[0]+h_gpNum);

        cudaEventRecord(start);
        timemakePd += computeGradientAndHessian(TetMesh);


        double distToOpt_PN = calcMinMovement(_moveDir, pcg_data.squeue, vertexNum);

        bool gradVanish = (distToOpt_PN < sqrt(Newton_solver_threshold * Newton_solver_threshold
                                               * bboxDiagSize2 * IPC_dt * IPC_dt));

        //double distToOpt_PN = calcMinMovement(TetMesh.totalForce, pcg_data.squeue, vertexNum);
        //printf("disToopt:  %f        %f\n",
        //       distToOpt_PN,
        //       2 * sqrt(Newton_solver_threshold * Newton_solver_threshold * bboxDiagSize2)
        //           * IPC_dt * IPC_dt);

        //bool gradVanish =
        //    (distToOpt_PN < 1
        //                        * sqrt(Newton_solver_threshold * Newton_solver_threshold * bboxDiagSize2)
        //                        * IPC_dt * IPC_dt);

        if(k && gradVanish)
        {
            break;
        }
        cudaEventRecord(end0);

        auto cg_count = calculateMovingDirection(TetMesh, h_cpNum[0], pcg_data.P_type);
        //std::cout << "[" << k << "]"
        //          << "cg_count = " << cg_count << std::endl;
        total_Cg_count += cg_count;
        cudaEventRecord(end1);
        double alpha = 1.0, slackness_a = 0.8, slackness_m = 0.8;

        alpha =
            std::min(alpha, ground_largestFeasibleStepSize(slackness_a, pcg_data.squeue));
        //alpha = std::min(alpha, InjectiveStepSize(0.2, 1e-6, pcg_data.squeue, TetMesh.tetrahedras));
        alpha = std::min(
            alpha, self_largestFeasibleStepSize(slackness_m, pcg_data.squeue, h_cpNum[0]));
        double temp_alpha = alpha;
        double alpha_CFL  = alpha;

        double ccd_size = 1.0;
        //#ifdef USE_FRICTION
        //        ccd_size = 0.6;
        //#endif

        buildBVH_FULLCCD(temp_alpha);
        buildFullCP(temp_alpha);
        if(h_ccd_cpNum > 0)
        {
            double maxSpeed = cfl_largestSpeed(pcg_data.squeue);
            alpha_CFL       = sqrt(dHat) / maxSpeed * 0.5;
            alpha           = std::min(alpha, alpha_CFL);
            if(temp_alpha > 2 * alpha_CFL)
            {
                /*buildBVH_FULLCCD(temp_alpha);
                buildFullCP(temp_alpha);*/
                alpha =
                    std::min(temp_alpha,
                             self_largestFeasibleStepSize(slackness_m, pcg_data.squeue, h_ccd_cpNum)
                                 * ccd_size);
                alpha = std::max(alpha, alpha_CFL);
            }
        }

        cudaEventRecord(end2);
        //printf("alpha:  %f\n", alpha);

        bool isStop = lineSearch(TetMesh, alpha, alpha_CFL);

        cudaEventRecord(end3);
        postLineSearch(TetMesh, alpha);
        //computeGradientAndHessian(TetMesh);
        cudaEventRecord(end4);

        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        float time00, time11, time22, time33, time44;
        cudaEventElapsedTime(&time00, start, end0);
        cudaEventElapsedTime(&time11, end0, end1);
        //total_Cg_time += time1;
        cudaEventElapsedTime(&time22, end1, end2);
        cudaEventElapsedTime(&time33, end2, end3);
        cudaEventElapsedTime(&time44, end3, end4);
        time0 += time00;
        time1 += time11;
        time2 += time22;
        time3 += time33;
        time4 += time44;
        ////*cflTime = ptime;
        //printf("time0 = %f,  time1 = %f,  time2 = %f,  time3 = %f,  time4 = %f\n",
        //       time00,
        //       time11,
        //       time22,
        //       time33,
        //       time44);
        (cudaEventDestroy(start));
        (cudaEventDestroy(end0));
        (cudaEventDestroy(end1));
        (cudaEventDestroy(end2));
        (cudaEventDestroy(end3));
        (cudaEventDestroy(end4));
        totalTimeStep += alpha;
    }
    //iterV.push_back(k);
    //std::ofstream outiter("iterCount.txt");
    //for(int ii = 0; ii < iterV.size(); ii++)
    //{
    //    outiter << iterV[ii] << std::endl;
    //}
    //outiter.close();
    printf("\n\n      Kappa: %f                               iteration k:  %d\n", Kappa, k);
    return k;
}

void GIPC::updateVelocities(device_TetraData& TetMesh)
{
    int numbers = vertexNum;
    if(numbers <= 0)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    _updateVelocities<<<blockNum, threadNum>>>(
        TetMesh.vertexes, TetMesh.o_vertexes, TetMesh.velocities, TetMesh.BoundaryType, IPC_dt, numbers);

    m_abd_system->update_velocity(*m_abd_sim_data);
}

void GIPC::updateBoundary(device_TetraData& TetMesh, double alpha)
{
    int numbers = vertexNum;
    if(numbers <= 0)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    _updateBoundary<<<blockNum, threadNum>>>(
        TetMesh.vertexes, TetMesh.BoundaryType, _moveDir, alpha, numbers);
}

void GIPC::updateBoundaryMoveDir(device_TetraData& TetMesh, double alpha, int fid)
{
    int numbers = vertexNum;
    if(numbers <= 0)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    _updateBoundaryMoveDir<<<blockNum, threadNum>>>(
        TetMesh.vertexes, TetMesh.BoundaryType, _moveDir, IPC_dt, FEM::PI, alpha, numbers, fid);
}


void GIPC::computeXTilta(device_TetraData& TetMesh, const double& rate)
{
    int numbers = vertexNum;
    if(numbers <= 0)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    _computeXTilta<<<blockNum, threadNum>>>(
        TetMesh.BoundaryType, TetMesh.velocities, TetMesh.o_vertexes, TetMesh.xTilta, IPC_dt, rate, numbers);

    m_abd_system->cal_q_tilde(*m_abd_sim_data);
}


int    totalNT          = 0;
double totalTime        = 0;
int    total_Frames     = 0;
double ttime0           = 0;
double ttime1           = 0;
double ttime2           = 0;
double ttime3           = 0;
double ttime4           = 0;
bool   isUpdateBoundary = false;
void   GIPC::IPC_Solver(device_TetraData& TetMesh)
{
    //double animation_fullRate = 0;
    cudaEvent_t start, end0;
    cudaEventCreate(&start);
    cudaEventCreate(&end0);
    double alpha = 1;
    cudaEventRecord(start);
    //    if(isRotate&&total_Frames*IPC_dt>=2.2){
    //        isRotate = false;
    //        updateBoundary2(TetMesh);
    //    }
    if(isUpdateBoundary)
    {
        updateBoundaryMoveDir(TetMesh, alpha, total_Frames);
        buildBVH_FULLCCD(alpha);
        buildFullCP(alpha);
        if(h_ccd_cpNum > 0)
        {
            double slackness_m = 0.8;
            alpha              = std::min(alpha,
                             self_largestFeasibleStepSize(slackness_m, pcg_data.squeue, h_ccd_cpNum));
        }
        //updateBoundary(TetMesh, alpha);

        CUDA_SAFE_CALL(cudaMemcpy(TetMesh.temp_double3Mem,
                                  TetMesh.vertexes,
                                  vertexNum * sizeof(double3),
                                  cudaMemcpyDeviceToDevice));
        updateBoundaryMoveDir(TetMesh, alpha, total_Frames);
        stepForward(TetMesh.vertexes, TetMesh.temp_double3Mem, _moveDir, TetMesh.BoundaryType, 1, true, vertexNum);
        //step_forward(TetMesh, 1, true);

        buildBVH();
        int numOfIntersect = 0;
        while(isIntersected(TetMesh))
        {
            printf("type 6 intersection happened:    %f\n", alpha);
            alpha /= 2.0;
            updateBoundaryMoveDir(TetMesh, alpha, total_Frames);
            numOfIntersect++;
            stepForward(TetMesh.vertexes,
                        TetMesh.temp_double3Mem,
                        _moveDir,
                        TetMesh.BoundaryType,
                        1,
                        true,
                        vertexNum);
            //step_forward(TetMesh, 1, true);
            buildBVH();
        }

        buildCP();
        printf("boundary alpha: %f\n  finished a step\n", alpha);
    }

    //suggestKappa(Kappa);
    upperBoundKappa(Kappa);
    if(Kappa < 1e-16)
    {
        suggestKappa(Kappa);
    }
    initKappa(TetMesh);
    //Kappa = 1e4;
#ifdef USE_FRICTION
    CUDA_SAFE_CALL(cudaMalloc((void**)&lambda_lastH_scalar, h_cpNum[0] * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&distCoord, h_cpNum[0] * sizeof(double2)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&tanBasis, h_cpNum[0] * sizeof(__GEIGEN__::Matrix3x2d)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&_collisonPairs_lastH, h_cpNum[0] * sizeof(int4)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&_MatIndex_last, h_cpNum[0] * sizeof(int)));

    CUDA_SAFE_CALL(cudaMalloc((void**)&lambda_lastH_scalar_gd, h_gpNum * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&_collisonPairs_lastH_gd, h_gpNum * sizeof(uint32_t)));
    buildFrictionSets();
#endif
    animation_fullRate = animation_subRate;
    double time0       = 0;
    double time1       = 0;
    double time2       = 0;
    double time3       = 0;
    double time4       = 0;
    while(true)
    {
        //if (h_cpNum[0] > 0) return;
        tempMalloc_closeConstraint();
        CUDA_SAFE_CALL(cudaMemset(_close_cpNum, 0, sizeof(uint32_t)));
        CUDA_SAFE_CALL(cudaMemset(_close_gpNum, 0, sizeof(uint32_t)));

        totalNT += solve_subIP(TetMesh, time0, time1, time2, time3, time4);

        double2 minMaxDist1 = minMaxGroundDist();
        double2 minMaxDist2 = minMaxSelfDist();

        double minDist = std::min(minMaxDist1.x, minMaxDist2.x);
        double maxDist = std::max(minMaxDist1.y, minMaxDist2.y);


        bool finishMotion = animation_fullRate > 0.99 ? true : false;

        if(finishMotion)
        {
            tempFree_closeConstraint();
            break;
            //}
        }
        else
        {
            tempFree_closeConstraint();
        }

        animation_fullRate += animation_subRate;
        //updateVelocities(TetMesh);

        //computeXTilta(TetMesh, 1);
#ifdef USE_FRICTION
        CUDA_SAFE_CALL(cudaFree(lambda_lastH_scalar));
        CUDA_SAFE_CALL(cudaFree(distCoord));
        CUDA_SAFE_CALL(cudaFree(tanBasis));
        CUDA_SAFE_CALL(cudaFree(_collisonPairs_lastH));
        CUDA_SAFE_CALL(cudaFree(_MatIndex_last));

        CUDA_SAFE_CALL(cudaFree(lambda_lastH_scalar_gd));
        CUDA_SAFE_CALL(cudaFree(_collisonPairs_lastH_gd));

        CUDA_SAFE_CALL(cudaMalloc((void**)&lambda_lastH_scalar, h_cpNum[0] * sizeof(double)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&distCoord, h_cpNum[0] * sizeof(double2)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&tanBasis,
                                  h_cpNum[0] * sizeof(__GEIGEN__::Matrix3x2d)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&_collisonPairs_lastH, h_cpNum[0] * sizeof(int4)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&_MatIndex_last, h_cpNum[0] * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&lambda_lastH_scalar_gd, h_gpNum * sizeof(double)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&_collisonPairs_lastH_gd,
                                  h_gpNum * sizeof(uint32_t)));
        buildFrictionSets();
#endif
    }

#ifdef USE_FRICTION
    CUDA_SAFE_CALL(cudaFree(lambda_lastH_scalar));
    CUDA_SAFE_CALL(cudaFree(distCoord));
    CUDA_SAFE_CALL(cudaFree(tanBasis));
    CUDA_SAFE_CALL(cudaFree(_collisonPairs_lastH));
    CUDA_SAFE_CALL(cudaFree(_MatIndex_last));

    CUDA_SAFE_CALL(cudaFree(lambda_lastH_scalar_gd));
    CUDA_SAFE_CALL(cudaFree(_collisonPairs_lastH_gd));
#endif

    updateVelocities(TetMesh);

    computeXTilta(TetMesh, 1);
    cudaEventRecord(end0);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    float tttime;
    cudaEventElapsedTime(&tttime, start, end0);
    totalTime += tttime;
    total_Frames++;
    printf("average time cost:     %f,    frame id:   %d\n", totalTime / totalNT, total_Frames);


    ttime0 += time0;
    ttime1 += time1;
    ttime2 += time2;
    ttime3 += time3;
    ttime4 += time4;


    std::ofstream outTime("timeCost.txt");

    outTime << "time0: " << ttime0 / 1000.0 << std::endl;
    outTime << "time1: " << ttime1 / 1000.0 << std::endl;
    outTime << "time2: " << ttime2 / 1000.0 << std::endl;
    outTime << "time3: " << ttime3 / 1000.0 << std::endl;
    outTime << "time4: " << ttime4 / 1000.0 << std::endl;
    outTime << "time_makePD: " << timemakePd / 1000.0 << std::endl;

    outTime << "totalTime: " << totalTime / 1000.0 << std::endl;
    outTime << "total iter: " << totalNT << std::endl;
    outTime << "frames: " << total_Frames << std::endl;
    outTime << "totalCollisionNum: " << totalCollisionPairs << std::endl;
    outTime << "averageCollision: " << totalCollisionPairs / totalNT << std::endl;
    outTime << "maxCOllisionPairNum: " << maxCOllisionPairNum << std::endl;
    outTime << "totalCgTime: " << total_Cg_count << std::endl;
    outTime.close();


    auto& stats = gipc::Statistics::instance();

    stats.at_current_frame()["timer"] =
        gipc::GlobalTimer::current()->report_merged_as_json();
    gipc::GlobalTimer::current()->print_merged_timings();
    gipc::GlobalTimer::current()->clear();
    stats.write_to_file(std::string{gipc::output_dir()} + "/stats.json");

    auto f = stats.frame();
    stats.frame(f + 1);
}