//
// GIPC.cu
// GIPC - GPU-based Incremental Potential Contact
//
// created by Kemeng Huang on 2022/12/01
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//
// ============================================================================
// FILE STRUCTURE (for refactoring reference)
// ============================================================================
// 
// Section 1: Includes and Utility Functions (Lines 1-500)
//   - Headers and includes
//   - makePD, makePDGeneral templates
//   - write_triplet, expand_bits, hash_code utilities
//   - Mesh sorting/hashing kernels
//
// Section 2: Collision Detection Helpers (Lines 500-1300)
//   - Distance functions (_d_PP, _d_PE, _d_PT, _d_EE)
//   - _selfConstraintVal device function
//   - _computeInjectiveStepSize_3d
//   - Friction energy device functions
//
// Section 3: Barrier (IPC) Kernels -> MOVED TO gipc_impl/gipc_barrier_kernels.cu
//
// Section 4: Friction Kernels -> MOVED TO gipc_impl/gipc_friction_kernels.cu
//
// Section 5: Reduction Kernels -> MOVED TO gipc_impl/gipc_reduction_kernels.cu
//
// Section 6: Energy Kernels -> MOVED TO gipc_impl/gipc_energy_kernels.cu
//
// Section 7: Ground Collision Kernels -> MOVED TO gipc_impl/gipc_ground_kernels.cu
//
// Section 8: Simulation Step Kernels -> MOVED TO gipc_impl/gipc_simulation_kernels.cu
//
// Section 9: GIPC Class Member Functions (Remaining in this file)
//   - Constructor/Destructor
//   - Memory management (MALLOC_DEVICE_MEM, FREE_DEVICE_MEM)
//   - Collision detection (buildCP, buildBVH)
//   - Gradient/Hessian computation
//   - solve_subIP (Newton iteration)
//   - IPC_Solver (main solver entry)
//   - Line search and step size computation
//
// ============================================================================

#include "GIPC.cuh"
#include "gipc_impl/gipc_kernels.cuh"
#include "gipc_impl/gipc_device_functions.cuh"
// Note: gipc_impl/gipc_kernels.cuh contains kernel declarations
// Kernel implementations are in separate gipc_impl/*.cu files
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


// ============================================================================
// Kernel implementations have been moved to separate files in gipc_impl/
// See gipc_kernels.cuh for declarations and the following files for definitions:
//   - gipc_barrier_kernels.cu
//   - gipc_friction_kernels.cu
//   - gipc_reduction_kernels.cu
//   - gipc_energy_kernels.cu
//   - gipc_ground_kernels.cu
//   - gipc_simulation_kernels.cu
// ============================================================================

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

void GIPC::calBarrierGradientAndHessian(double3* _gradient, double mKappa)
{
    int numbers = h_cpNum[0];
    if(numbers < 1)
        return;
    const unsigned int threadNum = 256;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;

    _calBarrierGradientAndHessian<<<blockNum, threadNum>>>(
        _vertexes,
        _rest_vertexes,
        _collisonPairs,
        _gradient,
        gipc_global_triplet.block_values(),
        gipc_global_triplet.block_row_indices(),
        gipc_global_triplet.block_col_indices(),
        _cpNum,
        _MatIndex,
        dHat,
        mKappa,
        h_cpNum[4],
        h_cpNum[3],
        h_cpNum[2],
        numbers);
}


void GIPC::calBarrierHessian()
{

    int numbers = h_cpNum[0];
    if(numbers < 1)
        return;
    const unsigned int threadNum = 32;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //

    _calBarrierHessian<<<blockNum, threadNum>>>(_vertexes,
                                                _rest_vertexes,
                                                _collisonPairs,
                                                gipc_global_triplet.block_values(),
                                                gipc_global_triplet.block_row_indices(),
                                                gipc_global_triplet.block_col_indices(),
                                                _cpNum,
                                                _MatIndex,
                                                dHat,
                                                Kappa,
                                                h_cpNum[4],
                                                h_cpNum[3],
                                                h_cpNum[2],
                                                numbers);
}

void GIPC::calFrictionHessian(device_TetraData& TetMesh)
{
    int numbers = h_cpNum_last[0];
    //if (numbers < 1) return;
    const unsigned int threadNum = 256;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    if(numbers > 0)
    {
        _calFrictionHessian<<<blockNum, threadNum>>>(
            _vertexes,
            TetMesh.o_vertexes,
            _collisonPairs_lastH,
            gipc_global_triplet.block_values(),
            gipc_global_triplet.block_row_indices(),
            gipc_global_triplet.block_col_indices(),
            _cpNum,
            numbers,
            IPC_dt,
            distCoord,
            tanBasis,
            fDhat * IPC_dt * IPC_dt,
            lambda_lastH_scalar,
            frictionRate,
            h_cpNum[4],
            h_cpNum[3],
            h_cpNum[2],
            h_cpNum_last[4],
            h_cpNum_last[3],
            h_cpNum_last[2]);
    }

    numbers = h_gpNum_last;
    CUDA_SAFE_CALL(cudaMemcpy(_gpNum, &h_gpNum_last, sizeof(uint32_t), cudaMemcpyHostToDevice));
    if(numbers < 1)
        return;

    blockNum = (numbers + threadNum - 1) / threadNum;
    int global_offset = gipc_global_triplet.global_triplet_offset + h_cpNum_last[4] * M12_Off
                        + h_cpNum_last[3] * M9_Off + h_cpNum_last[2] * M6_Off;
    _calFrictionHessian_gd<<<blockNum, threadNum>>>(
        _vertexes,
        TetMesh.o_vertexes,
        _groundNormal,
        _collisonPairs_lastH_gd,
        gipc_global_triplet.block_values(),
        gipc_global_triplet.block_row_indices(),
        gipc_global_triplet.block_col_indices(),
        numbers,
        IPC_dt,
        fDhat * IPC_dt * IPC_dt,
        lambda_lastH_scalar_gd,
        global_offset,
        gd_frictionRate);
}

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

// void GIPC::calBarrierGradient(double3* _gradient, double mKappa) {
//     int numbers = h_cpNum[0];
//     if (numbers < 1)return;
//     const unsigned int threadNum = 256;
//     int blockNum = (numbers + threadNum - 1) / threadNum;
//     _calBarrierGradient << <blockNum, threadNum >> > (_vertexes, _rest_vertexes, _collisonPairs, _gradient, dHat, mKappa, numbers);
// }

void GIPC::calBarrierGradient(double3* _gradient, double mKappa)
{
    int numbers = h_cpNum[0];
    if(numbers < 1)
        return;
    const unsigned int threadNum = 256;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;


    _calBarrierGradient<<<blockNum, threadNum>>>(
        _vertexes, _rest_vertexes, _collisonPairs, _gradient, dHat, mKappa, numbers);
}

void GIPC::calFrictionGradient(double3* _gradient, device_TetraData& TetMesh)
{
    int                numbers   = h_cpNum_last[0];
    const unsigned int threadNum = 256;
    int                blockNum  = 0;
    if(numbers > 0)
    {
        blockNum = (numbers + threadNum - 1) / threadNum;
        _calFrictionGradient<<<blockNum, threadNum>>>(_vertexes,
                                                      TetMesh.o_vertexes,
                                                      _collisonPairs_lastH,
                                                      _gradient,
                                                      numbers,
                                                      IPC_dt,
                                                      distCoord,
                                                      tanBasis,
                                                      fDhat * IPC_dt * IPC_dt,
                                                      lambda_lastH_scalar,
                                                      frictionRate);
    }
    numbers = h_gpNum_last;
    if(numbers < 1)
        return;
    blockNum = (numbers + threadNum - 1) / threadNum;

    _calFrictionGradient_gd<<<blockNum, threadNum>>>(_vertexes,
                                                     TetMesh.o_vertexes,
                                                     _groundNormal,
                                                     _collisonPairs_lastH_gd,
                                                     _gradient,
                                                     numbers,
                                                     IPC_dt,
                                                     fDhat * IPC_dt * IPC_dt,
                                                     lambda_lastH_scalar_gd,
                                                     gd_frictionRate);
}


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
                   const const AABB* _MaxBv,
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


    int number = gipc_global_triplet.global_collision_triplet_offset;

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


    double alpha_SL = alpha;

    step_forward(TetMesh, alpha, false);

    bool rehash = true;

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

        bool rehash = true;

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
    int    k           = 0;
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