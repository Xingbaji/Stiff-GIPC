//
// ipc_collision.cu
// GIPC - IPC Collision & Friction Kernels Implementation
//
// This file contains implementations for:
// - Friction kernels (gradient and hessian)
// - Ground collision kernels (detection, gradient, hessian)
// - Self collision kernels (close value, distance reduction)
// - GIPC class collision-related member functions
//
// Barrier gradient/hessian kernels are in barrier_gradient_hessian.cu
//
// Refactored from GIPC.cu
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#include "ipc_collision.cuh"
#include "barrier_gradient_hessian.cuh"  // Barrier gradient/hessian kernel declarations
#include "ipc_common.cuh"  // Common helper functions (makePDGeneral_ipc, write_triplet)
#include "GIPC.cuh"
#include "mlbvh.cuh"  // Contains _d_EE, _d_PP, _d_PE, _d_PT, _compute_epx functions
#include "FrictionUtils.cuh"
#include "GIPC_PDerivative.cuh"
#include <cuda_runtime.h>
#include <cmath>

using namespace Eigen;


//=============================================================================
// Friction Hessian Kernels
//=============================================================================

__global__ void _calFrictionHessian_gd(const double3*   _vertexes,
                                       const double3*   _o_vertexes,
                                       const double3*   _normal,
                                       const uint32_t*  _last_collisionPair_gd,
                                       Eigen::Matrix3d* triplet_values,
                                       int*             row_ids,
                                       int*             col_ids,
                                       int              number,
                                       double           dt,
                                       double           eps2,
                                       double*          lastH,
                                       int              global_offset,
                                       double           coef)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    double                 eps           = sqrt(eps2);
    unsigned int           gidx          = _last_collisionPair_gd[idx];
    double                 multiplier_vI = coef * lastH[idx];
    __GEIGEN__::Matrix3x3d H_vI;

    double3 Vdiff  = __GEIGEN__::__minus(_vertexes[gidx], _o_vertexes[gidx]);
    double3 normal = *_normal;
    double3 VProj  = __GEIGEN__::__minus(
        Vdiff, __GEIGEN__::__s_vec_multiply(normal, __GEIGEN__::__v_vec_dot(Vdiff, normal)));
    double VProjMag2 = __GEIGEN__::__squaredNorm(VProj);

    if(VProjMag2 > eps2)
    {
        double VProjMag = sqrt(VProjMag2);

        __GEIGEN__::Matrix2x2d projH;
        __GEIGEN__::__set_Mat2x2_val_column(projH, make_double2(0, 0), make_double2(0, 0));

        double  eigenValues[2];
        int     eigenNum = 0;
        double2 eigenVecs[2];
        __GEIGEN__::__makePD2x2(VProj.x * VProj.x * -multiplier_vI / VProjMag2 / VProjMag
                                    + (multiplier_vI / VProjMag),
                                VProj.x * VProj.z * -multiplier_vI / VProjMag2 / VProjMag,
                                VProj.x * VProj.z * -multiplier_vI / VProjMag2 / VProjMag,
                                VProj.z * VProj.z * -multiplier_vI / VProjMag2 / VProjMag
                                    + (multiplier_vI / VProjMag),
                                eigenValues,
                                eigenNum,
                                eigenVecs);
        for(int i = 0; i < eigenNum; i++)
        {
            if(eigenValues[i] > 0)
            {
                __GEIGEN__::Matrix2x2d eigenMatrix =
                    __GEIGEN__::__v2_vec2_toMat2x2(eigenVecs[i], eigenVecs[i]);
                eigenMatrix =
                    __GEIGEN__::__s_Mat2x2_multiply(eigenMatrix, eigenValues[i]);
                projH = __GEIGEN__::__Mat2x2_add(projH, eigenMatrix);
            }
        }

        __GEIGEN__::__set_Mat_val(H_vI,
                                  projH.m[0][0],
                                  0,
                                  projH.m[0][1],
                                  0,
                                  0,
                                  0,
                                  projH.m[1][0],
                                  0,
                                  projH.m[1][1]);
    }
    else
    {
        __GEIGEN__::__set_Mat_val(
            H_vI, (multiplier_vI / eps), 0, 0, 0, 0, 0, 0, 0, (multiplier_vI / eps));
    }

    write_triplet<3, 3>(triplet_values, row_ids, col_ids, &gidx, H_vI.m, global_offset + idx);
}

//=============================================================================
// Friction Gradient Kernels  
//=============================================================================

__global__ void _calFrictionGradient_gd(const double3* _vertexes,
                                        const double3* _o_vertexes,
                                        const double3* _normal,
                                        const uint32_t* _last_collisionPair_gd,
                                        double3* _gradient,
                                        int      number,
                                        double   dt,
                                        double   eps2,
                                        double*  lastH,
                                        double   coef)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    double   eps    = sqrt(eps2);
    double3  normal = *_normal;
    uint32_t gidx   = _last_collisionPair_gd[idx];
    double3  Vdiff  = __GEIGEN__::__minus(_vertexes[gidx], _o_vertexes[gidx]);
    double3  VProj  = __GEIGEN__::__minus(
        Vdiff, __GEIGEN__::__s_vec_multiply(normal, __GEIGEN__::__v_vec_dot(Vdiff, normal)));
    double VProjMag2 = __GEIGEN__::__squaredNorm(VProj);
    if(VProjMag2 > eps2)
    {
        double3 gdf =
            __GEIGEN__::__s_vec_multiply(VProj, coef * lastH[idx] / sqrt(VProjMag2));
        _gradient[gidx] = __GEIGEN__::__add(_gradient[gidx], gdf);
    }
    else
    {
        double3 gdf = __GEIGEN__::__s_vec_multiply(VProj, coef * lastH[idx] / eps);
        _gradient[gidx] = __GEIGEN__::__add(_gradient[gidx], gdf);
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
        write_triplet<3, 3>(triplet_values, row_ids, col_ids, &gidx, Hpg.m, global_offset + idx);
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



//=============================================================================
// Barrier Hessian Kernel
//=============================================================================


// Barrier kernel implementations moved to barrier_gradient_hessian.cu


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

// Adaptive Stiffness Implementation

__global__ void _reduct_MaxDouble(double* _data, int number)
{
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;
    
    double myVal = (i < number) ? _data[i] : 0.0;
    sdata[tid] = myVal;
    __syncthreads();
    
    for(unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if(tid < s) {
            if(sdata[tid + s] > sdata[tid]) {
                sdata[tid] = sdata[tid + s];
            }
        }
        __syncthreads();
    }
    
    if(tid == 0) {
        _data[blockIdx.x] = sdata[0];
    }
}

double GIPC::calAdaptiveStiffness(device_TetraData& TetMesh)
{
    int numbers = h_cpNum[0];
    if(numbers < 1) return 0.0;
    
    double* d_stiffness;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_stiffness, numbers * sizeof(double)));
    CUDA_SAFE_CALL(cudaMemset(d_stiffness, 0, numbers * sizeof(double)));
    
    const unsigned int threadNum = 256;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    
    _calAdaptiveStiffness<<<blockNum, threadNum>>>(
        _vertexes,
        TetMesh.masses,
        _collisonPairs,
        d_stiffness,
        dHat,
        numbers);
        
    // Reduction
    int n = numbers;
    while(n > 1) {
        int blocks = (n + threadNum - 1) / threadNum;
        _reduct_MaxDouble<<<blocks, threadNum, threadNum * sizeof(double)>>>(d_stiffness, n);
        n = blocks;
    }
    
    double maxVal;
    CUDA_SAFE_CALL(cudaMemcpy(&maxVal, d_stiffness, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaFree(d_stiffness));
    
    return maxVal;
}
