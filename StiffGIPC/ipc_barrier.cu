//
// ipc_barrier.cu
// GIPC - IPC Barrier Functions Implementation (Ground/Self Collision Kernels)
//
// Refactored from GIPC.cu
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#include "ipc_barrier.cuh"
#include "GIPC.cuh"
#include "mlbvh.cuh"  // Contains _d_EE, _d_PP, _d_PE, _d_PT, _compute_epx functions
#include <cuda_runtime.h>
#include <cmath>

//=============================================================================
// Triplet Write Helper
//=============================================================================

template <int ROWS, int COLS>
__device__ inline void write_triplet_ipc(Eigen::Matrix3d*    triplet_value,
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
        int start = 0;
        for(int jj = start; jj < coln; jj++)
        {
            int kk = ii * rown + jj;
            int row = index[ii];
            int col = index[jj];
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

//=============================================================================
// Self Constraint Value Device Function
//=============================================================================

__device__ double _selfConstraintVal(const double3* _vertexes, const int4& active)
{
    double val;
    if(active.x >= 0)
    {
        if(active.w >= 0)
        {
            _d_EE(_vertexes[active.x],
                  _vertexes[active.y],
                  _vertexes[active.z],
                  _vertexes[active.w],
                  val);
        }
        else
        {
            _d_EE(_vertexes[active.x],
                  _vertexes[active.y],
                  _vertexes[active.z],
                  _vertexes[-active.w - 1],
                  val);
        }
    }
    else
    {
        if(active.z < 0)
        {
            if(active.y < 0)
            {
                _d_PP(_vertexes[-active.x - 1], _vertexes[-active.y - 1], val);
            }
            else
            {
                _d_PP(_vertexes[-active.x - 1], _vertexes[active.y], val);
            }
        }
        else if(active.w < 0)
        {
            if(active.y < 0)
            {
                _d_PE(_vertexes[-active.x - 1],
                      _vertexes[-active.y - 1],
                      _vertexes[active.z],
                      val);
            }
            else
            {
                _d_PE(
                    _vertexes[-active.x - 1], _vertexes[active.y], _vertexes[active.z], val);
            }
        }
        else
        {
            _d_PT(_vertexes[-active.x - 1],
                  _vertexes[active.y],
                  _vertexes[active.z],
                  _vertexes[active.w],
                  val);
        }
    }
    return val;
}

//=============================================================================
// Ground Collision Kernels
//=============================================================================

__global__ void _GroundCollisionDetect(const double3*  vertexes,
                                       const uint32_t* surfVertIds,
                                       const double*   g_offset,
                                       const double3*  g_normal,
                                       uint32_t* _environment_collisionPair,
                                       uint32_t* _gpNum,
                                       double    dHat,
                                       int       number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    double dist = __GEIGEN__::__v_vec_dot(*g_normal, vertexes[surfVertIds[idx]]) - *g_offset;
    if(dist * dist > dHat)
        return;

    _environment_collisionPair[atomicAdd(_gpNum, 1)] = surfVertIds[idx];
}

__global__ void _computeGroundGradientAndHessian(const double3*  vertexes,
                                                 const double*   g_offset,
                                                 const double3*  g_normal,
                                                 const uint32_t* _environment_collisionPair,
                                                 double3*        gradient,
                                                 uint32_t*       _gpNum,
                                                 Eigen::Matrix3d* triplet_values,
                                                 int*             row_ids,
                                                 int*             col_ids,
                                                 double           dHat,
                                                 double           Kappa,
                                                 int              global_offset,
                                                 int              number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    double3      normal = *g_normal;
    unsigned int gidx   = _environment_collisionPair[idx];
    double dist  = __GEIGEN__::__v_vec_dot(normal, vertexes[gidx]) - *g_offset;
    double dist2 = dist * dist;

    double t   = dist2 - dHat;
    double g_b = t * log(dist2 / dHat) * -2.0 - (t * t) / dist2;

    double H_b = (log(dist2 / dHat) * -2.0 - t * 4.0 / dist2)
                 + 1.0 / (dist2 * dist2) * (t * t);

    double3 grad = __GEIGEN__::__s_vec_multiply(normal, Kappa * g_b * 2 * dist);

    {
        atomicAdd(&(gradient[gidx].x), grad.x);
        atomicAdd(&(gradient[gidx].y), grad.y);
        atomicAdd(&(gradient[gidx].z), grad.z);
    }

    double param = 4.0 * H_b * dist2 + 2.0 * g_b;
    {
        __GEIGEN__::Matrix3x3d nn = __GEIGEN__::__v_vec_toMat(normal, normal);
        __GEIGEN__::Matrix3x3d Hpg = __GEIGEN__::__S_Mat_multiply(nn, Kappa * param);

        int pidx = atomicAdd(_gpNum, 1);
        write_triplet_ipc<3, 3>(triplet_values, row_ids, col_ids, &gidx, Hpg.m, global_offset + idx);
    }
}

__global__ void _computeGroundGradient(const double3*  vertexes,
                                       const double*   g_offset,
                                       const double3*  g_normal,
                                       const uint32_t* _environment_collisionPair,
                                       double3*        gradient,
                                       uint32_t*       _gpNum,
                                       double          dHat,
                                       double          Kappa,
                                       int             number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    double3 normal = *g_normal;
    int     gidx   = _environment_collisionPair[idx];
    double  dist  = __GEIGEN__::__v_vec_dot(normal, vertexes[gidx]) - *g_offset;
    double  dist2 = dist * dist;

    double t   = dist2 - dHat;
    double g_b = t * std::log(dist2 / dHat) * -2.0 - (t * t) / dist2;

    double3 grad = __GEIGEN__::__s_vec_multiply(normal, Kappa * g_b * 2 * dist);

    {
        atomicAdd(&(gradient[gidx].x), grad.x);
        atomicAdd(&(gradient[gidx].y), grad.y);
        atomicAdd(&(gradient[gidx].z), grad.z);
    }
}

__global__ void _computeGroundCloseVal(const double3*  vertexes,
                                       const double*   g_offset,
                                       const double3*  g_normal,
                                       const uint32_t* _environment_collisionPair,
                                       double          dTol,
                                       uint32_t*       _closeConstraintID,
                                       double*         _closeConstraintVal,
                                       uint32_t*       _close_gpNum,
                                       int             number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    double3 normal = *g_normal;
    int     gidx   = _environment_collisionPair[idx];
    double  dist  = __GEIGEN__::__v_vec_dot(normal, vertexes[gidx]) - *g_offset;
    double  dist2 = dist * dist;

    if(dist2 < dTol)
    {
        int tidx                  = atomicAdd(_close_gpNum, 1);
        _closeConstraintID[tidx]  = gidx;
        _closeConstraintVal[tidx] = dist2;
    }
}

__global__ void _checkGroundCloseVal(const double3* vertexes,
                                     const double*  g_offset,
                                     const double3* g_normal,
                                     int*           _isChange,
                                     uint32_t*      _closeConstraintID,
                                     double*        _closeConstraintVal,
                                     int            number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    double3 normal = *g_normal;
    int     gidx   = _closeConstraintID[idx];
    double  dist  = __GEIGEN__::__v_vec_dot(normal, vertexes[gidx]) - *g_offset;
    double  dist2 = dist * dist;

    if(dist2 < _closeConstraintVal[gidx])
    {
        *_isChange = 1;
    }
}

__global__ void _reduct_MGroundDist(const double3* vertexes,
                                    const double*  g_offset,
                                    const double3* g_normal,
                                    uint32_t*      _environment_collisionPair,
                                    double2*       _queue,
                                    int            number)
{
    int                       idof = blockIdx.x * blockDim.x;
    int                       idx  = threadIdx.x + idof;
    extern __shared__ double2 sdata[];

    if(idx >= number)
        return;
    double3 normal = *g_normal;
    int     gidx   = _environment_collisionPair[idx];
    double  dist  = __GEIGEN__::__v_vec_dot(normal, vertexes[gidx]) - *g_offset;
    double  tempv = dist * dist;
    double2 temp  = make_double2(1.0 / tempv, tempv);

    int warpTid = threadIdx.x % 32;
    int warpId  = (threadIdx.x >> 5);
    int warpNum;
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

__global__ void _checkGroundIntersection(const double3*  vertexes,
                                         const double*   g_offset,
                                         const double3*  g_normal,
                                         const uint32_t* _environment_collisionPair,
                                         int*            _isIntersect,
                                         int             number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    double3 normal = *g_normal;
    int     gidx   = _environment_collisionPair[idx];
    double  dist = __GEIGEN__::__v_vec_dot(normal, vertexes[gidx]) - *g_offset;
    if(dist < 0)
        *_isIntersect = -1;
}

//=============================================================================
// Self Collision Kernels
//=============================================================================

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
    int warpTid = threadIdx.x % 32;
    int warpId  = (threadIdx.x >> 5);
    int warpNum;
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

__global__ void _computeSelfCloseVal(const double3*  vertexes,
                                     const double*   g_offset,
                                     const double3*  g_normal,
                                     const uint32_t* _environment_collisionPair,
                                     double          dTol,
                                     uint32_t*       _closeConstraintID,
                                     double*         _closeConstraintVal,
                                     uint32_t*       _close_gpNum,
                                     int             number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    double3 normal = *g_normal;
    int     gidx   = _environment_collisionPair[idx];
    double  dist  = __GEIGEN__::__v_vec_dot(normal, vertexes[gidx]) - *g_offset;
    double  dist2 = dist * dist;

    if(dist2 < dTol)
    {
        int tidx                  = atomicAdd(_close_gpNum, 1);
        _closeConstraintID[tidx]  = gidx;
        _closeConstraintVal[tidx] = dist2;
    }
}

//=============================================================================
// GIPC Member Function Implementations
//=============================================================================

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
    int                blockNum  = (numbers + threadNum - 1) / threadNum;

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
    const unsigned int threadNum = 256;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
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
