//
// gipc_energy_kernels.cu
// GIPC - Energy Computation Kernels
//
// This file contains CUDA kernels for energy computation:
// - Kinetic energy
// - FEM energy
// - Barrier energy
// - Soft constraint energy
// - Bending energy
//
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#include <cuda_runtime.h>
#include "../gpu_eigen_libs.cuh"
#include "../femEnergy.cuh"
#include "gipc_device_functions.cuh"

// ============================================================================
// Kinetic Energy Kernels
// ============================================================================

__global__ void _calKineticGradient(
    double3* _vertexes, double3* _xTilta, double3* _gradient, double* _masses, int numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;
    double3 Tg   = __GEIGEN__::__s_vec_multiply(__GEIGEN__::__minus(_vertexes[idx], _xTilta[idx]),
                                              _masses[idx]);
    _gradient[idx] = __GEIGEN__::__add(_gradient[idx], Tg);
}

__global__ void _calKineticEnergy(
    double3* _vertexes, double3* _xTilta, double* _energy, double* _masses, int numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;
    _energy[idx] = __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(_vertexes[idx], _xTilta[idx]))
                   * _masses[idx] * 0.5;
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

    int    warpTid = threadIdx.x % 32;
    int    warpId  = (threadIdx.x >> 5);
    double nextTp;
    int    warpNum;
    if(blockIdx.x == gridDim.x - 1)
    {
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
        temp = tep[threadIdx.x];
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

// ============================================================================
// FEM Energy Kernels
// ============================================================================

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

    int    warpTid = threadIdx.x % 32;
    int    warpId  = (threadIdx.x >> 5);
    double nextTp;
    int    warpNum;
    if(blockIdx.x == gridDim.x - 1)
    {
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
        temp = tep[threadIdx.x];
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

    int    warpTid = threadIdx.x % 32;
    int    warpId  = (threadIdx.x >> 5);
    double nextTp;
    int    warpNum;
    if(blockIdx.x == gridDim.x - 1)
    {
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
        temp = tep[threadIdx.x];
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

    int    warpTid = threadIdx.x % 32;
    int    warpId  = (threadIdx.x >> 5);
    double nextTp;
    int    warpNum;
    if(blockIdx.x == gridDim.x - 1)
    {
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
        temp = tep[threadIdx.x];
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

    uint2   adj     = edge_adj_vertex[idx];
    double3 rest_x0 = rest_vertexex[edges[idx].x];
    double3 rest_x1 = rest_vertexex[edges[idx].y];
    double  length  = __GEIGEN__::__norm(__GEIGEN__::__minus(rest_x0, rest_x1));
    double  temp =
        __cal_bending_energy(vertexes, rest_vertexex, edges[idx], adj, length, bendStiff);

    int    warpTid = threadIdx.x % 32;
    int    warpId  = (threadIdx.x >> 5);
    double nextTp;
    int    warpNum;
    if(blockIdx.x == gridDim.x - 1)
    {
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
        temp = tep[threadIdx.x];
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

// ============================================================================
// Barrier Energy Kernels
// ============================================================================

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

    int    warpTid = threadIdx.x % 32;
    int    warpId  = (threadIdx.x >> 5);
    double nextTp;
    int    warpNum;
    if(blockIdx.x == gridDim.x - 1)
    {
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
        temp = tep[threadIdx.x];
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

// ============================================================================
// Soft Constraint Energy Kernels
// ============================================================================

// Helper template for writing triplets (local to this file)
template <int ROWS, int COLS>
__device__ inline void write_triplet_local(Eigen::Matrix3d*    triplet_value,
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
            int kk = ii * rown + jj;
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
                                                         int    global_hessian_fem_offset,
                                                         int    number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    uint32_t vInd = targetInd[idx];
    double   x = vertexes[vInd].x, y = vertexes[vInd].y, z = vertexes[vInd].z,
           a = targetVert[idx].x, b = targetVert[idx].y, c = targetVert[idx].z;
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
    vInd += global_hessian_fem_offset;
    write_triplet_local<3, 3>(triplet_values, row_ids, col_ids, &vInd, Hpg.m, global_offset + idx);
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
    double d = motionRate;
    {
        atomicAdd(&(gradient[vInd].x), d * rate * rate * (x - a));
        atomicAdd(&(gradient[vInd].y), d * rate * rate * (y - b));
        atomicAdd(&(gradient[vInd].z), d * rate * rate * (z - c));
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

    int    warpTid = threadIdx.x % 32;
    int    warpId  = (threadIdx.x >> 5);
    double nextTp;
    int    warpNum;
    if(blockIdx.x == gridDim.x - 1)
    {
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
        temp = tep[threadIdx.x];
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

// ============================================================================
// Delta Energy and General Reduction Kernels
// ============================================================================

__global__ void _getDeltaEnergy_Reduction(double* squeue, const double3* b, const double3* dx, int vertexNum)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];
    int                      numbers = vertexNum;
    if(idx >= numbers)
        return;

    double temp = __GEIGEN__::__v_vec_dot(b[idx], dx[idx]);

    int    warpTid = threadIdx.x % 32;
    int    warpId  = (threadIdx.x >> 5);
    double nextTp;
    int    warpNum;
    if(blockIdx.x == gridDim.x - 1)
    {
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
        temp = tep[threadIdx.x];
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
    double temp = mem[idx];

    __threadfence();

    int    warpTid = threadIdx.x % 32;
    int    warpId  = (threadIdx.x >> 5);
    double nextTp;
    int    warpNum;
    if(blockIdx.x == gridDim.x - 1)
    {
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
        temp = tep[threadIdx.x];
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

