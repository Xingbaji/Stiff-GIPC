//
// gipc_reduction_kernels.cu
// GIPC - Parallel Reduction Kernels
//
// This file contains CUDA kernels for parallel reduction operations:
// - Max/min reductions
// - Dot product reductions
// - CFL and timestep reductions
// - Self-collision distance reductions
//
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#include <cuda_runtime.h>
#include "../gpu_eigen_libs.cuh"
#include "../mlbvh.cuh"
#include "../ACCD.cuh"
#include "gipc_device_functions.cuh"

// ============================================================================
// Basic Reduction Kernels
// ============================================================================

__global__ void _reduct_max_double3_to_double(const double3* _double3Dim, double* _double1Dim, int number)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];

    if(idx >= number)
        return;
    double3 tempMove = _double3Dim[idx];

    double temp =
        std::max(std::max(abs(tempMove.x), abs(tempMove.y)), abs(tempMove.z));

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
        temp = tep[threadIdx.x];
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
    double temp = _double1Dim[idx];

    __threadfence();

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
        temp = tep[threadIdx.x];
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
    double2 temp = _double2Dim[idx];

    __threadfence();

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
        temp = sdata[threadIdx.x];
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
    double temp = _double1Dim[idx];

    __threadfence();

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
        temp = tep[threadIdx.x];
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

// ============================================================================
// Vector Reduction Kernels
// ============================================================================

__global__ void _reduct_double3Sqn_to_double(const double3* A, double* D, int number)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];

    if(idx >= number)
        return;

    double temp = __GEIGEN__::__squaredNorm(A[idx]);

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
        D[blockIdx.x] = temp;
    }
}

// ============================================================================
// Self-Collision Distance Reduction
// ============================================================================

__global__ void _reduct_MSelfDist(const double3* _vertexes,
                                  int4*          _collisionPairs,
                                  double2*       _queue,
                                  int            number)
{
    int                       idof = blockIdx.x * blockDim.x;
    int                       idx  = threadIdx.x + idof;
    extern __shared__ double2 sdata[];

    if(idx >= number)
        return;
    int4    MMCVIDI = _collisionPairs[idx];
    double  tempv   = _selfConstraintVal(_vertexes, MMCVIDI);
    double2 temp    = make_double2(1.0 / tempv, tempv);
    int     warpTid = threadIdx.x % 32;
    int     warpId  = (threadIdx.x >> 5);
    double  nextTp;
    int     warpNum;
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
        temp = sdata[threadIdx.x];
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
        _queue[blockIdx.x] = temp;
    }
}

// ============================================================================
// Self-Collision Close Value Kernels
// ============================================================================

__global__ void _calSelfCloseVal(const double3* _vertexes,
                                 const int4*    _collisionPair,
                                 int4*          _close_collisionPair,
                                 double*        _close_collisionVal,
                                 uint32_t*      _close_cpNum,
                                 double         dTol,
                                 int            number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    int4   MMCVIDI = _collisionPair[idx];
    double dist2   = _selfConstraintVal(_vertexes, MMCVIDI);
    if(dist2 < dTol)
    {
        int tidx                   = atomicAdd(_close_cpNum, 1);
        _close_collisionPair[tidx] = MMCVIDI;
        _close_collisionVal[tidx]  = dist2;
    }
}

__global__ void _checkSelfCloseVal(const double3* _vertexes,
                                   int*           _isChange,
                                   int4*          _close_collisionPair,
                                   double*        _close_collisionVal,
                                   int            number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    int4   MMCVIDI = _close_collisionPair[idx];
    double dist2   = _selfConstraintVal(_vertexes, MMCVIDI);
    if(dist2 < _close_collisionVal[idx])
    {
        *_isChange = 1;
    }
}

// ============================================================================
// CFL Reduction Kernel
// ============================================================================

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
    double3 v  = moveDir[mSVI[idx]];
    double  temp = std::max(std::max(abs(v.x), abs(v.y)), abs(v.z));

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
        temp = tep[threadIdx.x];
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

// ============================================================================
// Time Step Reduction Kernels
// ============================================================================

__global__ void _reduct_min_InjectiveTimeStep_to_double(const double3* vertexes,
                                                        const uint4*   tetrahedras,
                                                        const double3* moveDir,
                                                        double*        minStepSizes,
                                                        double         slackness,
                                                        double         errorRate,
                                                        int            number)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];

    if(idx >= number)
        return;

    uint4 indices = tetrahedras[idx];

    double temp = 1.0;
    double x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4;
    double p1, p2, p3, p4, q1, q2, q3, q4, r1, r2, r3, r4;
    double a, b, c, d, t;

    x1 = vertexes[indices.x].x;
    x2 = vertexes[indices.y].x;
    x3 = vertexes[indices.z].x;
    x4 = vertexes[indices.w].x;

    y1 = vertexes[indices.x].y;
    y2 = vertexes[indices.y].y;
    y3 = vertexes[indices.z].y;
    y4 = vertexes[indices.w].y;

    z1 = vertexes[indices.x].z;
    z2 = vertexes[indices.y].z;
    z3 = vertexes[indices.z].z;
    z4 = vertexes[indices.w].z;

    p1 = -moveDir[indices.x].x;
    p2 = -moveDir[indices.y].x;
    p3 = -moveDir[indices.z].x;
    p4 = -moveDir[indices.w].x;

    q1 = -moveDir[indices.x].y;
    q2 = -moveDir[indices.y].y;
    q3 = -moveDir[indices.z].y;
    q4 = -moveDir[indices.w].y;

    r1 = -moveDir[indices.x].z;
    r2 = -moveDir[indices.y].z;
    r3 = -moveDir[indices.z].z;
    r4 = -moveDir[indices.w].z;

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

    d = (slackness)
        * (x1 * z2 * y3 - x1 * y2 * z3 + y1 * x2 * z3 - y1 * z2 * x3 - z1 * x2 * y3
           + z1 * y2 * x3 + x1 * y2 * z4 - x1 * z2 * y4 - y1 * x2 * z4 + y1 * z2 * x4
           + z1 * x2 * y4 - z1 * y2 * x4 - x1 * y3 * z4 + x1 * z3 * y4 + y1 * x3 * z4
           - y1 * z3 * x4 - z1 * x3 * y4 + z1 * y3 * x4 + x2 * y3 * z4 - x2 * z3 * y4
           - y2 * x3 * z4 + y2 * z3 * x4 + z2 * x3 * y4 - z2 * y3 * x4);

    if(abs(a) <= errorRate)
    {
        if(abs(b) <= errorRate)
        {
            if(abs(c) <= errorRate)
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
        t = 1;
    }
    if(t <= 0)
        t = 1;
    temp = t;

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
        temp = tep[threadIdx.x];
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            double tempMin = __shfl_down_sync(0xffffffff, temp, i);
            temp           = std::min(temp, tempMin);
        }
    }
    if(threadIdx.x == 0)
    {
        minStepSizes[blockIdx.x] = temp;
    }
}

__global__ void _reduct_min_selfTimeStep_to_double(const double3* vertexes,
                                                   const int4*    _ccd_collisionPairs,
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
    int4   MMCVIDI = _ccd_collisionPairs[idx];
    double temp    = 1.0;

    if(MMCVIDI.x >= 0)
    {
        if(MMCVIDI.w >= 0)
        {
            edge_edge_ccd(vertexes[MMCVIDI.x],
                          vertexes[MMCVIDI.y],
                          vertexes[MMCVIDI.z],
                          vertexes[MMCVIDI.w],
                          __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.x], -1),
                          __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.y], -1),
                          __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.z], -1),
                          __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.w], -1),
                          slackness,
                          temp);
        }
        else
        {
            MMCVIDI.w = -MMCVIDI.w - 1;
            edge_edge_ccd(vertexes[MMCVIDI.x],
                          vertexes[MMCVIDI.y],
                          vertexes[MMCVIDI.z],
                          vertexes[MMCVIDI.w],
                          __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.x], -1),
                          __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.y], -1),
                          __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.z], -1),
                          __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.w], -1),
                          slackness,
                          temp);
        }
    }
    else
    {
        MMCVIDI.x = -MMCVIDI.x - 1;
        if(MMCVIDI.z < 0)
        {
            if(MMCVIDI.y < 0)
            {
                MMCVIDI.y = -MMCVIDI.y - 1;
                temp      = 1;
            }
            else
            {
                temp = 1;
            }
        }
        else if(MMCVIDI.w < 0)
        {
            if(MMCVIDI.y < 0)
            {
                MMCVIDI.y = -MMCVIDI.y - 1;
                temp      = 1;
            }
            else
            {
                temp = 1;
            }
        }
        else
        {
            point_triangle_ccd(vertexes[MMCVIDI.x],
                               vertexes[MMCVIDI.y],
                               vertexes[MMCVIDI.z],
                               vertexes[MMCVIDI.w],
                               __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.x], -1),
                               __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.y], -1),
                               __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.z], -1),
                               __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.w], -1),
                               slackness,
                               temp);
        }
    }

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
        temp = tep[threadIdx.x];
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            double tempMin = __shfl_down_sync(0xffffffff, temp, i);
            temp           = std::min(temp, tempMin);
        }
    }
    if(threadIdx.x == 0)
    {
        minStepSizes[blockIdx.x] = temp;
    }
}

