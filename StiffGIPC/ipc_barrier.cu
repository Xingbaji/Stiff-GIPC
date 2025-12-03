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
#include "FrictionUtils.cuh"
#include "GIPC_PDerivative.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include "Eigen/Eigen"

using namespace Eigen;
#define RANK 2
#define NEWF

//=============================================================================
// Helper Functions (from GIPC.cu)
//=============================================================================

template <typename Scalar, int size>
__device__ __host__ void makePDGeneral_ipc(Eigen::Matrix<Scalar, size, size>& symMtr)
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

    Eigen::Matrix<Scalar, size, size> D = eigen_values.asDiagonal();
    symMtr = eigen_solver.eigenvectors() * D * eigen_solver.eigenvectors().transpose();
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

__global__ void _calBarrierHessian(const double3*   _vertexes,
                                   const double3*   _rest_vertexes,
                                   const int4*      _collisionPair,
                                   Eigen::Matrix3d* triplet_values,
                                   int*             row_ids,
                                   int*             col_ids,
                                   uint32_t*        _cpNum,
                                   int*             matIndex,
                                   double           dHat,
                                   double           Kappa,
                                   int              offset4,
                                   int              offset3,
                                   int              offset2,
                                   int              number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    int4   MMCVIDI   = _collisionPair[idx];
    double dHat_sqrt = sqrt(dHat);

    double gassThreshold = 1e-6;
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
            dis = sqrt(dis);
            __GEIGEN__::Matrix12x9d PFPxT;
            pFpx_ee2(_vertexes[MMCVIDI.x],
                     _vertexes[MMCVIDI.y],
                     _vertexes[MMCVIDI.z],
                     _vertexes[MMCVIDI.w],
                     dHat_sqrt,
                     PFPxT);
            double              I5 = pow(dis / dHat_sqrt, 2);
            __GEIGEN__::Vector9 q0;
            q0.v[0] = q0.v[1] = q0.v[2] = q0.v[3] = q0.v[4] = q0.v[5] =
                q0.v[6] = q0.v[7] = 0;
            q0.v[8]               = 1;
            __GEIGEN__::Matrix9x9d H;
            __GEIGEN__::__init_Mat9x9(H, 0);

#if (RANK == 1)
            double lambda0 =
                Kappa
                * (2 * dHat * dHat
                   * (6 * I5 + 2 * I5 * log(I5) - 7 * I5 * I5 - 6 * I5 * I5 * log(I5) + 1))
                / I5;
            if(dis * dis < gassThreshold * dHat)
            {
                double lambda1 =
                    Kappa
                    * (2 * dHat * dHat
                       * (6 * gassThreshold + 2 * gassThreshold * log(gassThreshold)
                          - 7 * gassThreshold * gassThreshold
                          - 6 * gassThreshold * gassThreshold * log(gassThreshold) + 1))
                    / gassThreshold;
                lambda0 = lambda1;
            }
#elif (RANK == 2)
            double lambda0 =
                -(4 * Kappa * dHat * dHat
                  * (4 * I5 + log(I5) - 3 * I5 * I5 * log(I5) * log(I5) + 6 * I5 * log(I5)
                     - 2 * I5 * I5 + I5 * log(I5) * log(I5) - 7 * I5 * I5 * log(I5) - 2))
                / I5;
            if(dis * dis < gassThreshold * dHat)
            {
                double lambda1 =
                    -(4 * Kappa * dHat * dHat
                      * (4 * gassThreshold + log(gassThreshold)
                         - 3 * gassThreshold * gassThreshold * log(gassThreshold) * log(gassThreshold)
                         + 6 * gassThreshold * log(gassThreshold) - 2 * gassThreshold * gassThreshold
                         + gassThreshold * log(gassThreshold) * log(gassThreshold)
                         - 7 * gassThreshold * gassThreshold * log(gassThreshold) - 2))
                    / gassThreshold;
                lambda0 = lambda1;
            }
#elif (RANK == 3)
            double lambda0 =
                (2 * Kappa * dHat * dHat * log(I5)
                 * (24 * I5 + 3 * log(I5) - 6 * I5 * I5 * log(I5) * log(I5)
                    + 18 * I5 * log(I5) - 12 * I5 * I5
                    + 2 * I5 * log(I5) * log(I5) - 21 * I5 * I5 * log(I5) - 12))
                / I5;
#elif (RANK == 4)
            double lambda0 =
                -(4 * Kappa * dHat * dHat * log(I5) * log(I5)
                  * (24 * I5 + 2 * log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                     + 12 * I5 * log(I5) - 12 * I5 * I5 + I5 * log(I5) * log(I5)
                     - 14 * I5 * I5 * log(I5) - 12))
                / I5;
#elif (RANK == 5)
            double lambda0 =
                (2 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                 * (80 * I5 + 5 * log(I5) - 6 * I5 * I5 * log(I5) * log(I5)
                    + 30 * I5 * log(I5) - 40 * I5 * I5
                    + 2 * I5 * log(I5) * log(I5) - 35 * I5 * I5 * log(I5) - 40))
                / I5;
#elif (RANK == 6)
            double lambda0 =
                -(4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5)
                  * (60 * I5 + 3 * log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                     + 18 * I5 * log(I5) - 30 * I5 * I5 + I5 * log(I5) * log(I5)
                     - 21 * I5 * I5 * log(I5) - 30))
                / I5;
#endif

            H = __GEIGEN__::__S_Mat9x9_multiply(__GEIGEN__::__v9_vec9_toMat9x9(q0, q0), lambda0);

            __GEIGEN__::Matrix12x12d Hessian;  // = __GEIGEN__::__M12x9_M9x12_Multiply(__GEIGEN__::__M12x9_M9x9_Multiply(PFPxT, H), __GEIGEN__::__Transpose12x9(PFPxT));

            __GEIGEN__::__M12x9_S9x9_MT9x12_Multiply(PFPxT, H, Hessian);

            int Hidx = matIndex[idx];

            uint4 global_index =
                make_uint4(MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);

            int triplet_id_offset = Hidx * 16;
            write_triplet<12, 12>(
                triplet_values, row_ids, col_ids, &(global_index.x), Hessian.m, triplet_id_offset);
        }
        else
        {
            MMCVIDI.w = -MMCVIDI.w - 1;
            double3 v0 =
                __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _vertexes[MMCVIDI.x]);
            double3 v1 =
                __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.z]);
            double c  = __GEIGEN__::__norm(__GEIGEN__::__v_vec_cross(v0, v1));
            double I1 = c * c;
            if(I1 == 0)
                return;
            __GEIGEN__::Matrix12x9d PFPx;
            pFpx_pee(_vertexes[MMCVIDI.x],
                     _vertexes[MMCVIDI.y],
                     _vertexes[MMCVIDI.z],
                     _vertexes[MMCVIDI.w],
                     dHat_sqrt,
                     PFPx);

            double dis;
            _d_EE(_vertexes[MMCVIDI.x],
                  _vertexes[MMCVIDI.y],
                  _vertexes[MMCVIDI.z],
                  _vertexes[MMCVIDI.w],
                  dis);
            double I2 = dis / dHat;
            dis       = sqrt(dis);

            __GEIGEN__::Matrix3x3d F;
            __GEIGEN__::__set_Mat_val(F, 1, 0, 0, 0, c, 0, 0, 0, dis / dHat_sqrt);
            double3 n1 = make_double3(0, 1, 0);
            double3 n2 = make_double3(0, 0, 1);

            double eps_x = _compute_epx(_rest_vertexes[MMCVIDI.x],
                                        _rest_vertexes[MMCVIDI.y],
                                        _rest_vertexes[MMCVIDI.z],
                                        _rest_vertexes[MMCVIDI.w]);

#if (RANK == 1)
            double lambda10 =
                Kappa * (4 * dHat * dHat * log(I2) * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x))
                / (eps_x * eps_x);
            double lambda11 =
                Kappa * 2
                * (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                / (eps_x * eps_x);
            double lambda12 =
                Kappa * 2
                * (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                / (eps_x * eps_x);
#elif (RANK == 2)
            double lambda10 = -Kappa
                              * (4 * dHat * dHat * log(I2) * log(I2) * (I2 - 1)
                                 * (I2 - 1) * (3 * I1 - eps_x))
                              / (eps_x * eps_x);
            double lambda11 = -Kappa
                              * (4 * dHat * dHat * log(I2) * log(I2)
                                 * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                              / (eps_x * eps_x);
            double lambda12 = -Kappa
                              * (4 * dHat * dHat * log(I2) * log(I2)
                                 * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                              / (eps_x * eps_x);
#elif (RANK == 4)
            double lambda10 = -Kappa
                              * (4 * dHat * dHat * pow(log(I2), 4) * (I2 - 1)
                                 * (I2 - 1) * (3 * I1 - eps_x))
                              / (eps_x * eps_x);
            double lambda11 = -Kappa
                              * (4 * dHat * dHat * pow(log(I2), 4)
                                 * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                              / (eps_x * eps_x);
            double lambda12 = -Kappa
                              * (4 * dHat * dHat * pow(log(I2), 4)
                                 * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                              / (eps_x * eps_x);
#elif (RANK == 6)
            double lambda10 = -Kappa
                              * (4 * dHat * dHat * pow(log(I2), 6) * (I2 - 1)
                                 * (I2 - 1) * (3 * I1 - eps_x))
                              / (eps_x * eps_x);
            double lambda11 = -Kappa
                              * (4 * dHat * dHat * pow(log(I2), 6)
                                 * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                              / (eps_x * eps_x);
            double lambda12 = -Kappa
                              * (4 * dHat * dHat * pow(log(I2), 6)
                                 * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                              / (eps_x * eps_x);
#endif
            __GEIGEN__::Matrix3x3d fnn;
            __GEIGEN__::Matrix3x3d nn = __GEIGEN__::__v_vec_toMat(n1, n1);
            __GEIGEN__::__M_Mat_multiply(F, nn, fnn);
            __GEIGEN__::Vector9 q10 = __GEIGEN__::__Mat3x3_to_vec9_double(fnn);
            q10 = __GEIGEN__::__s_vec9_multiply(q10, 1.0 / sqrt(I1));

            __GEIGEN__::Matrix3x3d Tx, Ty, Tz;
            __GEIGEN__::__set_Mat_val(Tx, 0, 0, 0, 0, 0, 1, 0, -1, 0);
            __GEIGEN__::__set_Mat_val(Ty, 0, 0, -1, 0, 0, 0, 1, 0, 0);
            __GEIGEN__::__set_Mat_val(Tz, 0, 1, 0, -1, 0, 0, 0, 0, 0);

            double ratio = 1.f / sqrt(2.f);
            Tx           = __S_Mat_multiply(Tx, ratio);
            Ty           = __S_Mat_multiply(Ty, ratio);
            Tz           = __S_Mat_multiply(Tz, ratio);

            __GEIGEN__::Vector9 q11 = __GEIGEN__::__Mat3x3_to_vec9_double(
                __GEIGEN__::__M_Mat_multiply(Tx, fnn));
            __GEIGEN__::__normalized_vec9_double(q11);
            __GEIGEN__::Vector9 q12 = __GEIGEN__::__Mat3x3_to_vec9_double(
                __GEIGEN__::__M_Mat_multiply(Tz, fnn));
            //__GEIGEN__::__s_vec9_multiply(q12, c);
            __GEIGEN__::__normalized_vec9_double(q12);

            __GEIGEN__::Matrix9x9d projectedH;
            __GEIGEN__::__init_Mat9x9(projectedH, 0);

            __GEIGEN__::Matrix9x9d M9_temp = __GEIGEN__::__v9_vec9_toMat9x9(q11, q11);
            M9_temp    = __GEIGEN__::__S_Mat9x9_multiply(M9_temp, lambda11);
            projectedH = __GEIGEN__::__Mat9x9_add(projectedH, M9_temp);

            M9_temp    = __GEIGEN__::__v9_vec9_toMat9x9(q12, q12);
            M9_temp    = __GEIGEN__::__S_Mat9x9_multiply(M9_temp, lambda12);
            projectedH = __GEIGEN__::__Mat9x9_add(projectedH, M9_temp);

#if (RANK == 1)
            double lambda20 =
                -Kappa
                * (2 * I1 * dHat * dHat * (I1 - 2 * eps_x)
                   * (6 * I2 + 2 * I2 * log(I2) - 7 * I2 * I2 - 6 * I2 * I2 * log(I2) + 1))
                / (I2 * eps_x * eps_x);
#elif (RANK == 2)
            double lambda20 =
                Kappa
                * (4 * I1 * dHat * dHat * (I1 - 2 * eps_x)
                   * (4 * I2 + log(I2) - 3 * I2 * I2 * log(I2) * log(I2)
                      + 6 * I2 * log(I2) - 2 * I2 * I2 + I2 * log(I2) * log(I2)
                      - 7 * I2 * I2 * log(I2) - 2))
                / (I2 * (eps_x * eps_x));
#elif (RANK == 4)
            double lambda20 =
                Kappa
                * (4 * I1 * dHat * dHat * log(I2) * log(I2) * (I1 - 2 * eps_x)
                   * (24 * I2 + 2 * log(I2) - 3 * I2 * I2 * log(I2) * log(I2)
                      + 12 * I2 * log(I2) - 12 * I2 * I2
                      + I2 * log(I2) * log(I2) - 14 * I2 * I2 * log(I2) - 12))
                / (I2 * (eps_x * eps_x));
#elif (RANK == 6)
            double lambda20 =
                Kappa
                * (4 * I1 * dHat * dHat * pow(log(I2), 4) * (I1 - 2 * eps_x)
                   * (60 * I2 + 3 * log(I2) - 3 * I2 * I2 * log(I2) * log(I2)
                      + 18 * I2 * log(I2) - 30 * I2 * I2
                      + I2 * log(I2) * log(I2) - 21 * I2 * I2 * log(I2) - 30))
                / (I2 * (eps_x * eps_x));
#endif
            nn = __GEIGEN__::__v_vec_toMat(n2, n2);
            __GEIGEN__::__M_Mat_multiply(F, nn, fnn);
            __GEIGEN__::Vector9 q20 = __GEIGEN__::__Mat3x3_to_vec9_double(fnn);
            q20 = __GEIGEN__::__s_vec9_multiply(q20, 1.0 / sqrt(I2));


#if (RANK == 1)
            double lambdag1g = Kappa * 4 * c * F.m[2][2]
                               * ((2 * dHat * dHat * (I1 - eps_x) * (I2 - 1)
                                   * (I2 + 2 * I2 * log(I2) - 1))
                                  / (I2 * eps_x * eps_x));
#elif (RANK == 2)
            double lambdag1g = -Kappa * 4 * c * F.m[2][2]
                               * (4 * dHat * dHat * log(I2) * (I1 - eps_x)
                                  * (I2 - 1) * (I2 + I2 * log(I2) - 1))
                               / (I2 * (eps_x * eps_x));
#elif (RANK == 4)
            double lambdag1g = -Kappa * 4 * c * F.m[2][2]
                               * (4 * dHat * dHat * pow(log(I2), 3) * (I1 - eps_x)
                                  * (I2 - 1) * (2 * I2 + I2 * log(I2) - 2))
                               / (I2 * (eps_x * eps_x));
#elif (RANK == 6)
            double lambdag1g = -Kappa * 4 * c * F.m[2][2]
                               * (4 * dHat * dHat * pow(log(I2), 5) * (I1 - eps_x)
                                  * (I2 - 1) * (3 * I2 + I2 * log(I2) - 3))
                               / (I2 * (eps_x * eps_x));
#endif

            Eigen::Matrix2d FMat2;
            FMat2 << lambda10, lambdag1g, lambdag1g, lambda20;
            makePDGeneral_ipc<double, 2>(FMat2);
            projectedH.m[4][4] += FMat2(0, 0);
            projectedH.m[4][8] += FMat2(0, 1);
            projectedH.m[8][4] += FMat2(1, 0);
            projectedH.m[8][8] += FMat2(1, 1);

            __GEIGEN__::Matrix12x12d Hessian;
            __GEIGEN__::__M12x9_S9x9_MT9x12_Multiply(PFPx, projectedH, Hessian);
            int Hidx = matIndex[idx];  //int Hidx = atomicAdd(_cpNum + 4, 1);

            uint4 global_index =
                make_uint4(MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);
            int triplet_id_offset = Hidx * 16;
            write_triplet<12, 12>(
                triplet_values, row_ids, col_ids, &(global_index.x), Hessian.m, triplet_id_offset);
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
                //printf("ppp condition  ***************************************\n: %d  %d  %d  %d\n***************************************\n", MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);
                double3 v0 =
                    __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.x]);
                double3 v1 =
                    __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.y]);
                double c = __GEIGEN__::__norm(__GEIGEN__::__v_vec_cross(v0, v1)) /*/ __GEIGEN__::__norm(v0)*/;
                double I1 = c * c;
                if(I1 == 0)
                    return;
                __GEIGEN__::Matrix12x9d PFPx;
                pFpx_ppp(_vertexes[MMCVIDI.x],
                         _vertexes[MMCVIDI.y],
                         _vertexes[MMCVIDI.z],
                         _vertexes[MMCVIDI.w],
                         dHat_sqrt,
                         PFPx);

                double dis;
                _d_PP(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], dis);
                double I2 = dis / dHat;
                dis       = sqrt(dis);

                __GEIGEN__::Matrix3x3d F;
                __GEIGEN__::__set_Mat_val(F, 1, 0, 0, 0, c, 0, 0, 0, dis / dHat_sqrt);
                double3 n1 = make_double3(0, 1, 0);
                double3 n2 = make_double3(0, 0, 1);

                double eps_x = _compute_epx(_rest_vertexes[MMCVIDI.x],
                                            _rest_vertexes[MMCVIDI.z],
                                            _rest_vertexes[MMCVIDI.y],
                                            _rest_vertexes[MMCVIDI.w]);

#if (RANK == 1)
                double lambda10 =
                    Kappa * (4 * dHat * dHat * log(I2) * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x))
                    / (eps_x * eps_x);
                double lambda11 =
                    Kappa * 2
                    * (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                    / (eps_x * eps_x);
                double lambda12 =
                    Kappa * 2
                    * (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                    / (eps_x * eps_x);
#elif (RANK == 2)
                double lambda10 = -Kappa
                                  * (4 * dHat * dHat * log(I2) * log(I2)
                                     * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x))
                                  / (eps_x * eps_x);
                double lambda11 = -Kappa
                                  * (4 * dHat * dHat * log(I2) * log(I2)
                                     * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                                  / (eps_x * eps_x);
                double lambda12 = -Kappa
                                  * (4 * dHat * dHat * log(I2) * log(I2)
                                     * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                                  / (eps_x * eps_x);
#elif (RANK == 4)
                double lambda10 = -Kappa
                                  * (4 * dHat * dHat * pow(log(I2), 4)
                                     * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x))
                                  / (eps_x * eps_x);
                double lambda11 = -Kappa
                                  * (4 * dHat * dHat * pow(log(I2), 4)
                                     * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                                  / (eps_x * eps_x);
                double lambda12 = -Kappa
                                  * (4 * dHat * dHat * pow(log(I2), 4)
                                     * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                                  / (eps_x * eps_x);
#elif (RANK == 6)
                double lambda10 = -Kappa
                                  * (4 * dHat * dHat * pow(log(I2), 6)
                                     * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x))
                                  / (eps_x * eps_x);
                double lambda11 = -Kappa
                                  * (4 * dHat * dHat * pow(log(I2), 6)
                                     * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                                  / (eps_x * eps_x);
                double lambda12 = -Kappa
                                  * (4 * dHat * dHat * pow(log(I2), 6)
                                     * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                                  / (eps_x * eps_x);
#endif
                __GEIGEN__::Matrix3x3d fnn;
                __GEIGEN__::Matrix3x3d nn = __GEIGEN__::__v_vec_toMat(n1, n1);
                __GEIGEN__::__M_Mat_multiply(F, nn, fnn);
                __GEIGEN__::Vector9 q10 = __GEIGEN__::__Mat3x3_to_vec9_double(fnn);
                q10 = __GEIGEN__::__s_vec9_multiply(q10, 1.0 / sqrt(I1));

                __GEIGEN__::Matrix3x3d Tx, Ty, Tz;
                __GEIGEN__::__set_Mat_val(Tx, 0, 0, 0, 0, 0, 1, 0, -1, 0);
                __GEIGEN__::__set_Mat_val(Ty, 0, 0, -1, 0, 0, 0, 1, 0, 0);
                __GEIGEN__::__set_Mat_val(Tz, 0, 1, 0, -1, 0, 0, 0, 0, 0);

                double ratio = 1.f / sqrt(2.f);
                Tx           = __S_Mat_multiply(Tx, ratio);
                Ty           = __S_Mat_multiply(Ty, ratio);
                Tz           = __S_Mat_multiply(Tz, ratio);

                __GEIGEN__::Vector9 q11 = __GEIGEN__::__Mat3x3_to_vec9_double(
                    __GEIGEN__::__M_Mat_multiply(Tx, fnn));
                __GEIGEN__::__normalized_vec9_double(q11);
                __GEIGEN__::Vector9 q12 = __GEIGEN__::__Mat3x3_to_vec9_double(
                    __GEIGEN__::__M_Mat_multiply(Tz, fnn));
                //__GEIGEN__::__s_vec9_multiply(q12, c);
                __GEIGEN__::__normalized_vec9_double(q12);

                __GEIGEN__::Matrix9x9d projectedH;
                __GEIGEN__::__init_Mat9x9(projectedH, 0);

                __GEIGEN__::Matrix9x9d M9_temp = __GEIGEN__::__v9_vec9_toMat9x9(q11, q11);
                M9_temp    = __GEIGEN__::__S_Mat9x9_multiply(M9_temp, lambda11);
                projectedH = __GEIGEN__::__Mat9x9_add(projectedH, M9_temp);

                M9_temp    = __GEIGEN__::__v9_vec9_toMat9x9(q12, q12);
                M9_temp    = __GEIGEN__::__S_Mat9x9_multiply(M9_temp, lambda12);
                projectedH = __GEIGEN__::__Mat9x9_add(projectedH, M9_temp);

#if (RANK == 1)
                double lambda20 = -Kappa
                                  * (2 * I1 * dHat * dHat * (I1 - 2 * eps_x)
                                     * (6 * I2 + 2 * I2 * log(I2) - 7 * I2 * I2
                                        - 6 * I2 * I2 * log(I2) + 1))
                                  / (I2 * eps_x * eps_x);
#elif (RANK == 2)
                double lambda20 =
                    Kappa
                    * (4 * I1 * dHat * dHat * (I1 - 2 * eps_x)
                       * (4 * I2 + log(I2) - 3 * I2 * I2 * log(I2) * log(I2)
                          + 6 * I2 * log(I2) - 2 * I2 * I2
                          + I2 * log(I2) * log(I2) - 7 * I2 * I2 * log(I2) - 2))
                    / (I2 * (eps_x * eps_x));
#elif (RANK == 4)
                double lambda20 =
                    Kappa
                    * (4 * I1 * dHat * dHat * log(I2) * log(I2) * (I1 - 2 * eps_x)
                       * (24 * I2 + 2 * log(I2) - 3 * I2 * I2 * log(I2) * log(I2)
                          + 12 * I2 * log(I2) - 12 * I2 * I2
                          + I2 * log(I2) * log(I2) - 14 * I2 * I2 * log(I2) - 12))
                    / (I2 * (eps_x * eps_x));
#elif (RANK == 6)
                double lambda20 =
                    Kappa
                    * (4 * I1 * dHat * dHat * pow(log(I2), 4) * (I1 - 2 * eps_x)
                       * (60 * I2 + 3 * log(I2) - 3 * I2 * I2 * log(I2) * log(I2)
                          + 18 * I2 * log(I2) - 30 * I2 * I2
                          + I2 * log(I2) * log(I2) - 21 * I2 * I2 * log(I2) - 30))
                    / (I2 * (eps_x * eps_x));
#endif
                nn = __GEIGEN__::__v_vec_toMat(n2, n2);
                __GEIGEN__::__M_Mat_multiply(F, nn, fnn);
                __GEIGEN__::Vector9 q20 = __GEIGEN__::__Mat3x3_to_vec9_double(fnn);
                q20 = __GEIGEN__::__s_vec9_multiply(q20, 1.0 / sqrt(I2));


#if (RANK == 1)
                double lambdag1g = Kappa * 4 * c * F.m[2][2]
                                   * ((2 * dHat * dHat * (I1 - eps_x) * (I2 - 1)
                                       * (I2 + 2 * I2 * log(I2) - 1))
                                      / (I2 * eps_x * eps_x));
#elif (RANK == 2)
                double lambdag1g = -Kappa * 4 * c * F.m[2][2]
                                   * (4 * dHat * dHat * log(I2) * (I1 - eps_x)
                                      * (I2 - 1) * (I2 + I2 * log(I2) - 1))
                                   / (I2 * (eps_x * eps_x));
#elif (RANK == 4)
                double lambdag1g = -Kappa * 4 * c * F.m[2][2]
                                   * (4 * dHat * dHat * pow(log(I2), 3) * (I1 - eps_x)
                                      * (I2 - 1) * (2 * I2 + I2 * log(I2) - 2))
                                   / (I2 * (eps_x * eps_x));
#elif (RANK == 6)
                double lambdag1g = -Kappa * 4 * c * F.m[2][2]
                                   * (4 * dHat * dHat * pow(log(I2), 5) * (I1 - eps_x)
                                      * (I2 - 1) * (3 * I2 + I2 * log(I2) - 3))
                                   / (I2 * (eps_x * eps_x));
#endif
                Eigen::Matrix2d FMat2;
                FMat2 << lambda10, lambdag1g, lambdag1g, lambda20;
                makePDGeneral_ipc<double, 2>(FMat2);
                projectedH.m[4][4] += FMat2(0, 0);
                projectedH.m[4][8] += FMat2(0, 1);
                projectedH.m[8][4] += FMat2(1, 0);
                projectedH.m[8][8] += FMat2(1, 1);

                //__GEIGEN__::Matrix9x12d PFPxTransPos = __GEIGEN__::__Transpose12x9(PFPx);
                __GEIGEN__::Matrix12x12d Hessian;  // = __GEIGEN__::__M12x9_M9x12_Multiply(__GEIGEN__::__M12x9_M9x9_Multiply(PFPx, projectedH), PFPxTransPos);
                __GEIGEN__::__M12x9_S9x9_MT9x12_Multiply(PFPx, projectedH, Hessian);
                int Hidx = matIndex[idx];  //atomicAdd(_cpNum + 4, 1);

                uint4 global_index =
                    make_uint4(MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);
                //D4Index[Hidx] = global_index;


                int triplet_id_offset = Hidx * 16;
                write_triplet<12, 12>(
                    triplet_values, row_ids, col_ids, &(global_index.x), Hessian.m, triplet_id_offset);
            }
            else
            {
#ifdef NEWF
                double dis;
                _d_PP(_vertexes[v0I], _vertexes[MMCVIDI.y], dis);
                dis                            = sqrt(dis);
                double              d_hat_sqrt = sqrt(dHat);
                __GEIGEN__::Vector6 PFPxT;
                pFpx_pp2(_vertexes[v0I], _vertexes[MMCVIDI.y], d_hat_sqrt, PFPxT);
                double I5 = pow(dis / d_hat_sqrt, 2);
                //double q0 = 1;
#else
                double3 v0 = __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _vertexes[v0I]);
                double3 Ds  = v0;
                double  dis = __GEIGEN__::__norm(v0);
                //if (dis > dHat_sqrt) return;
                double3 vec_normal =
                    __GEIGEN__::__normalized(make_double3(-v0.x, -v0.y, -v0.z));
                double3 target = make_double3(0, 1, 0);
                double3 vec    = __GEIGEN__::__v_vec_cross(vec_normal, target);
                double  cos    = __GEIGEN__::__v_vec_dot(vec_normal, target);
                __GEIGEN__::Matrix3x3d rotation;
                __GEIGEN__::__set_Mat_val(rotation, 1, 0, 0, 0, 1, 0, 0, 0, 1);
                if(cos + 1 == 0)
                {
                    rotation.m[0][0] = -1;
                    rotation.m[1][1] = -1;
                }
                else
                {
                    //pDmpx_pp(_vertexes[v0I], _vertexes[MMCVIDI.y], dHat_sqrt, PDmPx);
                    __GEIGEN__::Matrix3x3d cross_vec;
                    __GEIGEN__::__set_Mat_val(
                        cross_vec, 0, -vec.z, vec.y, vec.z, 0, -vec.x, -vec.y, vec.x, 0);

                    rotation = __GEIGEN__::__Mat_add(
                        rotation,
                        __GEIGEN__::__Mat_add(cross_vec,
                                              __GEIGEN__::__S_Mat_multiply(
                                                  __GEIGEN__::__M_Mat_multiply(cross_vec, cross_vec),
                                                  1.0 / (1 + cos))));
                }

                double3 pos0 = __GEIGEN__::__add(
                    _vertexes[v0I],
                    __GEIGEN__::__s_vec_multiply(vec_normal, dHat_sqrt - dis));
                double3 rotate_uv0 = __GEIGEN__::__M_v_multiply(rotation, pos0);
                double3 rotate_uv1 =
                    __GEIGEN__::__M_v_multiply(rotation, _vertexes[MMCVIDI.y]);

                double uv0 = rotate_uv0.y;
                double uv1 = rotate_uv1.y;

                double u0    = uv1 - uv0;
                double Dm    = u0;
                double DmInv = 1 / u0;

                double3 F  = __GEIGEN__::__s_vec_multiply(Ds, DmInv);
                double  I5 = __GEIGEN__::__squaredNorm(F);

                double3 fnn = F;

                __GEIGEN__::Matrix3x6d PFPx = __computePFDsPX3D_3x6_double(DmInv);
#endif


#if (RANK == 1)
                double lambda0 = Kappa
                                 * (2 * dHat * dHat
                                    * (6 * I5 + 2 * I5 * log(I5) - 7 * I5 * I5
                                       - 6 * I5 * I5 * log(I5) + 1))
                                 / I5;
                if(dis * dis < gassThreshold * dHat)
                {
                    double lambda1 =
                        Kappa
                        * (2 * dHat * dHat
                           * (6 * gassThreshold + 2 * gassThreshold * log(gassThreshold)
                              - 7 * gassThreshold * gassThreshold
                              - 6 * gassThreshold * gassThreshold * log(gassThreshold) + 1))
                        / gassThreshold;
                    lambda0 = lambda1;
                }
#elif (RANK == 2)
                double lambda0 =
                    -(4 * Kappa * dHat * dHat
                      * (4 * I5 + log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                         + 6 * I5 * log(I5) - 2 * I5 * I5
                         + I5 * log(I5) * log(I5) - 7 * I5 * I5 * log(I5) - 2))
                    / I5;
                if(dis * dis < gassThreshold * dHat)
                {
                    double lambda1 =
                        -(4 * Kappa * dHat * dHat
                          * (4 * gassThreshold + log(gassThreshold)
                             - 3 * gassThreshold * gassThreshold
                                   * log(gassThreshold) * log(gassThreshold)
                             + 6 * gassThreshold * log(gassThreshold) - 2 * gassThreshold * gassThreshold
                             + gassThreshold * log(gassThreshold) * log(gassThreshold)
                             - 7 * gassThreshold * gassThreshold * log(gassThreshold) - 2))
                        / gassThreshold;
                    lambda0 = lambda1;
                }
#elif (RANK == 3)
                double lambda0 =
                    (2 * Kappa * dHat * dHat * log(I5)
                     * (24 * I5 + 3 * log(I5) - 6 * I5 * I5 * log(I5) * log(I5)
                        + 18 * I5 * log(I5) - 12 * I5 * I5
                        + 2 * I5 * log(I5) * log(I5) - 21 * I5 * I5 * log(I5) - 12))
                    / I5;
#elif (RANK == 4)
                double lambda0 =
                    -(4 * Kappa * dHat * dHat * log(I5) * log(I5)
                      * (24 * I5 + 2 * log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                         + 12 * I5 * log(I5) - 12 * I5 * I5
                         + I5 * log(I5) * log(I5) - 14 * I5 * I5 * log(I5) - 12))
                    / I5;
#elif (RANK == 5)
                double lambda0 =
                    (2 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                     * (80 * I5 + 5 * log(I5) - 6 * I5 * I5 * log(I5) * log(I5)
                        + 30 * I5 * log(I5) - 40 * I5 * I5
                        + 2 * I5 * log(I5) * log(I5) - 35 * I5 * I5 * log(I5) - 40))
                    / I5;
#elif (RANK == 6)
                double lambda0 =
                    -(4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5)
                      * (60 * I5 + 3 * log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                         + 18 * I5 * log(I5) - 30 * I5 * I5
                         + I5 * log(I5) * log(I5) - 21 * I5 * I5 * log(I5) - 30))
                    / I5;
#endif

#ifdef NEWF
                double                 H       = lambda0;
                __GEIGEN__::Matrix6x6d Hessian = __GEIGEN__::__s_M6x6_Multiply(
                    __GEIGEN__::__v6_vec6_toMat6x6(PFPxT, PFPxT), H);
#else
                double3 q0 = __GEIGEN__::__s_vec_multiply(F, 1 / sqrt(I5));

                __GEIGEN__::Matrix3x3d H =
                    __GEIGEN__::__S_Mat_multiply(__GEIGEN__::__v_vec_toMat(q0, q0),
                                                 lambda0);  //lambda0 * q0 * q0.transpose();

                __GEIGEN__::Matrix6x3d PFPxTransPos = __GEIGEN__::__Transpose3x6(PFPx);
                __GEIGEN__::Matrix6x6d Hessian = __GEIGEN__::__M6x3_M3x6_Multiply(
                    __GEIGEN__::__M6x3_M3x3_Multiply(PFPxTransPos, H), PFPx);
#endif
                int Hidx = matIndex[idx];  //atomicAdd(_cpNum + 4, 1);

                uint2 global_index = make_uint2(v0I, MMCVIDI.y);
                //D2Index[Hidx]      = global_index;

                int triplet_id_offset = Hidx * 4 + offset3 * 9 + offset4 * 16;
                write_triplet<6, 6>(
                    triplet_values, row_ids, col_ids, &(global_index.x), Hessian.m, triplet_id_offset);
            }
        }
        else if(MMCVIDI.w < 0)
        {
            if(MMCVIDI.y < 0)
            {
                MMCVIDI.y = -MMCVIDI.y - 1;
                MMCVIDI.w = -MMCVIDI.w - 1;
                MMCVIDI.x = v0I;
                //printf("ppe condition  ***************************************\n: %d  %d  %d  %d\n***************************************\n", MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);
                double3 v0 =
                    __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.x]);
                double3 v1 =
                    __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.y]);
                double c = __GEIGEN__::__norm(__GEIGEN__::__v_vec_cross(v0, v1)) /*/ __GEIGEN__::__norm(v0)*/;
                double I1 = c * c;
                if(I1 == 0)
                    return;
                __GEIGEN__::Matrix12x9d PFPx;
                pFpx_ppe(_vertexes[MMCVIDI.x],
                         _vertexes[MMCVIDI.y],
                         _vertexes[MMCVIDI.z],
                         _vertexes[MMCVIDI.w],
                         dHat_sqrt,
                         PFPx);

                double dis;
                _d_PE(_vertexes[MMCVIDI.x],
                      _vertexes[MMCVIDI.y],
                      _vertexes[MMCVIDI.z],
                      dis);
                double I2 = dis / dHat;
                dis       = sqrt(dis);

                __GEIGEN__::Matrix3x3d F;
                __GEIGEN__::__set_Mat_val(F, 1, 0, 0, 0, c, 0, 0, 0, dis / dHat_sqrt);
                double3 n1 = make_double3(0, 1, 0);
                double3 n2 = make_double3(0, 0, 1);

                double eps_x = _compute_epx(_rest_vertexes[MMCVIDI.x],
                                            _rest_vertexes[MMCVIDI.w],
                                            _rest_vertexes[MMCVIDI.y],
                                            _rest_vertexes[MMCVIDI.z]);

#if (RANK == 1)
                double lambda10 =
                    Kappa * (4 * dHat * dHat * log(I2) * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x))
                    / (eps_x * eps_x);
                double lambda11 =
                    Kappa * 2
                    * (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                    / (eps_x * eps_x);
                double lambda12 =
                    Kappa * 2
                    * (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                    / (eps_x * eps_x);
#elif (RANK == 2)
                double lambda10 = -Kappa
                                  * (4 * dHat * dHat * log(I2) * log(I2)
                                     * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x))
                                  / (eps_x * eps_x);
                double lambda11 = -Kappa
                                  * (4 * dHat * dHat * log(I2) * log(I2)
                                     * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                                  / (eps_x * eps_x);
                double lambda12 = -Kappa
                                  * (4 * dHat * dHat * log(I2) * log(I2)
                                     * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                                  / (eps_x * eps_x);
#elif (RANK == 4)
                double lambda10 = -Kappa
                                  * (4 * dHat * dHat * pow(log(I2), 4)
                                     * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x))
                                  / (eps_x * eps_x);
                double lambda11 = -Kappa
                                  * (4 * dHat * dHat * pow(log(I2), 4)
                                     * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                                  / (eps_x * eps_x);
                double lambda12 = -Kappa
                                  * (4 * dHat * dHat * pow(log(I2), 4)
                                     * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                                  / (eps_x * eps_x);
#elif (RANK == 6)
                double lambda10 = -Kappa
                                  * (4 * dHat * dHat * pow(log(I2), 6)
                                     * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x))
                                  / (eps_x * eps_x);
                double lambda11 = -Kappa
                                  * (4 * dHat * dHat * pow(log(I2), 6)
                                     * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                                  / (eps_x * eps_x);
                double lambda12 = -Kappa
                                  * (4 * dHat * dHat * pow(log(I2), 6)
                                     * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                                  / (eps_x * eps_x);
#endif
                __GEIGEN__::Matrix3x3d fnn;
                __GEIGEN__::Matrix3x3d nn = __GEIGEN__::__v_vec_toMat(n1, n1);
                __GEIGEN__::__M_Mat_multiply(F, nn, fnn);
                __GEIGEN__::Vector9 q10 = __GEIGEN__::__Mat3x3_to_vec9_double(fnn);
                q10 = __GEIGEN__::__s_vec9_multiply(q10, 1.0 / sqrt(I1));

                __GEIGEN__::Matrix3x3d Tx, Ty, Tz;
                __GEIGEN__::__set_Mat_val(Tx, 0, 0, 0, 0, 0, 1, 0, -1, 0);
                __GEIGEN__::__set_Mat_val(Ty, 0, 0, -1, 0, 0, 0, 1, 0, 0);
                __GEIGEN__::__set_Mat_val(Tz, 0, 1, 0, -1, 0, 0, 0, 0, 0);

                double ratio = 1.f / sqrt(2.f);
                Tx           = __S_Mat_multiply(Tx, ratio);
                Ty           = __S_Mat_multiply(Ty, ratio);
                Tz           = __S_Mat_multiply(Tz, ratio);

                __GEIGEN__::Vector9 q11 = __GEIGEN__::__Mat3x3_to_vec9_double(
                    __GEIGEN__::__M_Mat_multiply(Tx, fnn));
                __GEIGEN__::__normalized_vec9_double(q11);
                __GEIGEN__::Vector9 q12 = __GEIGEN__::__Mat3x3_to_vec9_double(
                    __GEIGEN__::__M_Mat_multiply(Tz, fnn));
                //__GEIGEN__::__s_vec9_multiply(q12, c);
                __GEIGEN__::__normalized_vec9_double(q12);

                __GEIGEN__::Matrix9x9d projectedH;
                __GEIGEN__::__init_Mat9x9(projectedH, 0);

                __GEIGEN__::Matrix9x9d M9_temp = __GEIGEN__::__v9_vec9_toMat9x9(q11, q11);
                M9_temp    = __GEIGEN__::__S_Mat9x9_multiply(M9_temp, lambda11);
                projectedH = __GEIGEN__::__Mat9x9_add(projectedH, M9_temp);

                M9_temp    = __GEIGEN__::__v9_vec9_toMat9x9(q12, q12);
                M9_temp    = __GEIGEN__::__S_Mat9x9_multiply(M9_temp, lambda12);
                projectedH = __GEIGEN__::__Mat9x9_add(projectedH, M9_temp);

#if (RANK == 1)
                double lambda20 = -Kappa
                                  * (2 * I1 * dHat * dHat * (I1 - 2 * eps_x)
                                     * (6 * I2 + 2 * I2 * log(I2) - 7 * I2 * I2
                                        - 6 * I2 * I2 * log(I2) + 1))
                                  / (I2 * eps_x * eps_x);
#elif (RANK == 2)
                double lambda20 =
                    Kappa
                    * (4 * I1 * dHat * dHat * (I1 - 2 * eps_x)
                       * (4 * I2 + log(I2) - 3 * I2 * I2 * log(I2) * log(I2)
                          + 6 * I2 * log(I2) - 2 * I2 * I2
                          + I2 * log(I2) * log(I2) - 7 * I2 * I2 * log(I2) - 2))
                    / (I2 * (eps_x * eps_x));
#elif (RANK == 4)
                double lambda20 =
                    Kappa
                    * (4 * I1 * dHat * dHat * log(I2) * log(I2) * (I1 - 2 * eps_x)
                       * (24 * I2 + 2 * log(I2) - 3 * I2 * I2 * log(I2) * log(I2)
                          + 12 * I2 * log(I2) - 12 * I2 * I2
                          + I2 * log(I2) * log(I2) - 14 * I2 * I2 * log(I2) - 12))
                    / (I2 * (eps_x * eps_x));
#elif (RANK == 6)
                double lambda20 =
                    Kappa
                    * (4 * I1 * dHat * dHat * pow(log(I2), 4) * (I1 - 2 * eps_x)
                       * (60 * I2 + 3 * log(I2) - 3 * I2 * I2 * log(I2) * log(I2)
                          + 18 * I2 * log(I2) - 30 * I2 * I2
                          + I2 * log(I2) * log(I2) - 21 * I2 * I2 * log(I2) - 30))
                    / (I2 * (eps_x * eps_x));
#endif
                nn = __GEIGEN__::__v_vec_toMat(n2, n2);
                __GEIGEN__::__M_Mat_multiply(F, nn, fnn);
                __GEIGEN__::Vector9 q20 = __GEIGEN__::__Mat3x3_to_vec9_double(fnn);
                q20 = __GEIGEN__::__s_vec9_multiply(q20, 1.0 / sqrt(I2));


#if (RANK == 1)
                double lambdag1g = Kappa * 4 * c * F.m[2][2]
                                   * ((2 * dHat * dHat * (I1 - eps_x) * (I2 - 1)
                                       * (I2 + 2 * I2 * log(I2) - 1))
                                      / (I2 * eps_x * eps_x));
#elif (RANK == 2)
                double lambdag1g = -Kappa * 4 * c * F.m[2][2]
                                   * (4 * dHat * dHat * log(I2) * (I1 - eps_x)
                                      * (I2 - 1) * (I2 + I2 * log(I2) - 1))
                                   / (I2 * (eps_x * eps_x));
#elif (RANK == 4)
                double lambdag1g = -Kappa * 4 * c * F.m[2][2]
                                   * (4 * dHat * dHat * pow(log(I2), 3) * (I1 - eps_x)
                                      * (I2 - 1) * (2 * I2 + I2 * log(I2) - 2))
                                   / (I2 * (eps_x * eps_x));
#elif (RANK == 6)
                double lambdag1g = -Kappa * 4 * c * F.m[2][2]
                                   * (4 * dHat * dHat * pow(log(I2), 5) * (I1 - eps_x)
                                      * (I2 - 1) * (3 * I2 + I2 * log(I2) - 3))
                                   / (I2 * (eps_x * eps_x));
#endif
                Eigen::Matrix2d FMat2;
                FMat2 << lambda10, lambdag1g, lambdag1g, lambda20;
                makePDGeneral_ipc<double, 2>(FMat2);
                projectedH.m[4][4] += FMat2(0, 0);
                projectedH.m[4][8] += FMat2(0, 1);
                projectedH.m[8][4] += FMat2(1, 0);
                projectedH.m[8][8] += FMat2(1, 1);
                //__GEIGEN__::Matrix9x12d PFPxTransPos = __GEIGEN__::__Transpose12x9(PFPx);
                __GEIGEN__::Matrix12x12d Hessian;  // = __GEIGEN__::__M12x9_M9x12_Multiply(__GEIGEN__::__M12x9_M9x9_Multiply(PFPx, projectedH), PFPxTransPos);
                __GEIGEN__::__M12x9_S9x9_MT9x12_Multiply(PFPx, projectedH, Hessian);
                int Hidx = matIndex[idx];  //atomicAdd(_cpNum + 4, 1);

                uint4 global_index =
                    make_uint4(MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);

                //D4Index[Hidx] = global_index;


                int triplet_id_offset = Hidx * 16;
                write_triplet<12, 12>(
                    triplet_values, row_ids, col_ids, &(global_index.x), Hessian.m, triplet_id_offset);
            }
            else
            {
#ifdef NEWF
                double dis;
                _d_PE(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], dis);
                dis                               = sqrt(dis);
                double                 d_hat_sqrt = sqrt(dHat);
                __GEIGEN__::Matrix9x4d PFPxT;
                pFpx_pe2(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], d_hat_sqrt, PFPxT);
                double              I5 = pow(dis / d_hat_sqrt, 2);
                __GEIGEN__::Vector4 q0;
                q0.v[0] = q0.v[1] = q0.v[2] = 0;
                q0.v[3]                     = 1;

                __GEIGEN__::Matrix4x4d H;
                //__GEIGEN__::__init_Mat4x4_val(H, 0);
#else
                double3 v0 = __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _vertexes[v0I]);
                double3 v1 = __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _vertexes[v0I]);


                __GEIGEN__::Matrix3x2d Ds;
                __GEIGEN__::__set_Mat3x2_val_column(Ds, v0, v1);

                double3 triangle_normal =
                    __GEIGEN__::__normalized(__GEIGEN__::__v_vec_cross(v0, v1));
                double3 target = make_double3(0, 1, 0);

                double3 vec = __GEIGEN__::__v_vec_cross(triangle_normal, target);
                double cos = __GEIGEN__::__v_vec_dot(triangle_normal, target);

                double3 edge_normal = __GEIGEN__::__normalized(__GEIGEN__::__v_vec_cross(
                    __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z]),
                    triangle_normal));
                double dis = __GEIGEN__::__v_vec_dot(
                    __GEIGEN__::__minus(_vertexes[v0I], _vertexes[MMCVIDI.y]), edge_normal);

                //if (dis > dHat_sqrt) return;

                __GEIGEN__::Matrix3x3d rotation;
                __GEIGEN__::__set_Mat_val(rotation, 1, 0, 0, 0, 1, 0, 0, 0, 1);

                __GEIGEN__::Matrix9x4d PDmPx;

                if(cos + 1 == 0)
                {
                    rotation.m[0][0] = -1;
                    rotation.m[1][1] = -1;
                }
                else
                {
                    //pDmpx_pe(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], dHat_sqrt, PDmPx);
                    __GEIGEN__::Matrix3x3d cross_vec;
                    __GEIGEN__::__set_Mat_val(
                        cross_vec, 0, -vec.z, vec.y, vec.z, 0, -vec.x, -vec.y, vec.x, 0);

                    rotation = __GEIGEN__::__Mat_add(
                        rotation,
                        __GEIGEN__::__Mat_add(cross_vec,
                                              __GEIGEN__::__S_Mat_multiply(
                                                  __GEIGEN__::__M_Mat_multiply(cross_vec, cross_vec),
                                                  1.0 / (1 + cos))));
                }

                double3 pos0 = __GEIGEN__::__add(
                    _vertexes[v0I],
                    __GEIGEN__::__s_vec_multiply(edge_normal, dHat_sqrt - dis));

                double3 rotate_uv0 = __GEIGEN__::__M_v_multiply(rotation, pos0);
                double3 rotate_uv1 =
                    __GEIGEN__::__M_v_multiply(rotation, _vertexes[MMCVIDI.y]);
                double3 rotate_uv2 =
                    __GEIGEN__::__M_v_multiply(rotation, _vertexes[MMCVIDI.z]);
                double3 rotate_normal = __GEIGEN__::__M_v_multiply(rotation, edge_normal);

                double2 uv0    = make_double2(rotate_uv0.x, rotate_uv0.z);
                double2 uv1    = make_double2(rotate_uv1.x, rotate_uv1.z);
                double2 uv2    = make_double2(rotate_uv2.x, rotate_uv2.z);
                double2 normal = make_double2(rotate_normal.x, rotate_normal.z);

                double2 u0 = __GEIGEN__::__minus_v2(uv1, uv0);
                double2 u1 = __GEIGEN__::__minus_v2(uv2, uv0);

                __GEIGEN__::Matrix2x2d Dm;

                __GEIGEN__::__set_Mat2x2_val_column(Dm, u0, u1);

                __GEIGEN__::Matrix2x2d DmInv;
                __GEIGEN__::__Inverse2x2(Dm, DmInv);

                __GEIGEN__::Matrix3x2d F = __GEIGEN__::__M3x2_M2x2_Multiply(Ds, DmInv);

                double3 FxN = __GEIGEN__::__M3x2_v2_multiply(F, normal);
                double  I5  = __GEIGEN__::__squaredNorm(FxN);

                __GEIGEN__::Matrix3x2d fnn;

                __GEIGEN__::Matrix2x2d nn = __GEIGEN__::__v2_vec2_toMat2x2(normal, normal);

                fnn = __GEIGEN__::__M3x2_M2x2_Multiply(F, nn);

                __GEIGEN__::Matrix6x9d PFPx = __computePFDsPX3D_6x9_double(DmInv);
#endif

#if (RANK == 1)
                double lambda0 = Kappa
                                 * (2 * dHat * dHat
                                    * (6 * I5 + 2 * I5 * log(I5) - 7 * I5 * I5
                                       - 6 * I5 * I5 * log(I5) + 1))
                                 / I5;
                if(dis * dis < gassThreshold * dHat)
                {
                    double lambda1 =
                        Kappa
                        * (2 * dHat * dHat
                           * (6 * gassThreshold + 2 * gassThreshold * log(gassThreshold)
                              - 7 * gassThreshold * gassThreshold
                              - 6 * gassThreshold * gassThreshold * log(gassThreshold) + 1))
                        / gassThreshold;
                    lambda0 = lambda1;
                }
#elif (RANK == 2)
                double lambda0 =
                    -(4 * Kappa * dHat * dHat
                      * (4 * I5 + log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                         + 6 * I5 * log(I5) - 2 * I5 * I5
                         + I5 * log(I5) * log(I5) - 7 * I5 * I5 * log(I5) - 2))
                    / I5;
                if(dis * dis < gassThreshold * dHat)
                {
                    double lambda1 =
                        -(4 * Kappa * dHat * dHat
                          * (4 * gassThreshold + log(gassThreshold)
                             - 3 * gassThreshold * gassThreshold
                                   * log(gassThreshold) * log(gassThreshold)
                             + 6 * gassThreshold * log(gassThreshold) - 2 * gassThreshold * gassThreshold
                             + gassThreshold * log(gassThreshold) * log(gassThreshold)
                             - 7 * gassThreshold * gassThreshold * log(gassThreshold) - 2))
                        / gassThreshold;
                    lambda0 = lambda1;
                }
#elif (RANK == 3)
                double lambda0 =
                    (2 * Kappa * dHat * dHat * log(I5)
                     * (24 * I5 + 3 * log(I5) - 6 * I5 * I5 * log(I5) * log(I5)
                        + 18 * I5 * log(I5) - 12 * I5 * I5
                        + 2 * I5 * log(I5) * log(I5) - 21 * I5 * I5 * log(I5) - 12))
                    / I5;
#elif (RANK == 4)
                double lambda0 =
                    -(4 * Kappa * dHat * dHat * log(I5) * log(I5)
                      * (24 * I5 + 2 * log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                         + 12 * I5 * log(I5) - 12 * I5 * I5
                         + I5 * log(I5) * log(I5) - 14 * I5 * I5 * log(I5) - 12))
                    / I5;
#elif (RANK == 5)
                double lambda0 =
                    (2 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                     * (80 * I5 + 5 * log(I5) - 6 * I5 * I5 * log(I5) * log(I5)
                        + 30 * I5 * log(I5) - 40 * I5 * I5
                        + 2 * I5 * log(I5) * log(I5) - 35 * I5 * I5 * log(I5) - 40))
                    / I5;
#elif (RANK == 6)
                double lambda0 =
                    -(4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5)
                      * (60 * I5 + 3 * log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                         + 18 * I5 * log(I5) - 30 * I5 * I5
                         + I5 * log(I5) * log(I5) - 21 * I5 * I5 * log(I5) - 30))
                    / I5;
#endif

#ifdef NEWF
                H = __GEIGEN__::__S_Mat4x4_multiply(
                    __GEIGEN__::__v4_vec4_toMat4x4(q0, q0), lambda0);

                __GEIGEN__::Matrix9x9d Hessian;  // = __GEIGEN__::__M9x4_M4x9_Multiply(__GEIGEN__::__M9x4_M4x4_Multiply(PFPxT, H), __GEIGEN__::__Transpose9x4(PFPxT));
                __M9x4_S4x4_MT4x9_Multiply(PFPxT, H, Hessian);
#else

                __GEIGEN__::Vector6 q0 = __GEIGEN__::__Mat3x2_to_vec6_double(fnn);

                q0 = __GEIGEN__::__s_vec6_multiply(q0, 1.0 / sqrt(I5));

                __GEIGEN__::Matrix6x6d H;
                __GEIGEN__::__init_Mat6x6(H, 0);

                H = __GEIGEN__::__S_Mat6x6_multiply(
                    __GEIGEN__::__v6_vec6_toMat6x6(q0, q0), lambda0);

                __GEIGEN__::Matrix9x6d PFPxTransPos = __GEIGEN__::__Transpose6x9(PFPx);
                __GEIGEN__::Matrix9x9d Hessian = __GEIGEN__::__M9x6_M6x9_Multiply(
                    __GEIGEN__::__M9x6_M6x6_Multiply(PFPxTransPos, H), PFPx);
#endif
                int Hidx = matIndex[idx];  //atomicAdd(_cpNum + 4, 1);

                uint3 global_index = make_uint3(v0I, MMCVIDI.y, MMCVIDI.z);

                //D3Index[Hidx] = global_index;

                int triplet_id_offset = Hidx * 9 + offset4 * 16;
                write_triplet<9, 9>(
                    triplet_values, row_ids, col_ids, &(global_index.x), Hessian.m, triplet_id_offset);
            }
        }
        else
        {
#ifdef NEWF
            double dis;
            //printf("PT: %d %d %d %d\n", v0I, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);
            _d_PT(_vertexes[v0I],
                  _vertexes[MMCVIDI.y],
                  _vertexes[MMCVIDI.z],
                  _vertexes[MMCVIDI.w],
                  dis);
            double I5                          = dis / dHat;
            dis                                = sqrt(dis);
            double                  d_hat_sqrt = sqrt(dHat);
            __GEIGEN__::Matrix12x9d PFPxT;
            pFpx_pt2(_vertexes[v0I],
                     _vertexes[MMCVIDI.y],
                     _vertexes[MMCVIDI.z],
                     _vertexes[MMCVIDI.w],
                     d_hat_sqrt,
                     PFPxT);

            __GEIGEN__::Vector9 q0;
            q0.v[0] = q0.v[1] = q0.v[2] = q0.v[3] = q0.v[4] = q0.v[5] =
                q0.v[6] = q0.v[7] = 0;
            q0.v[8]               = 1;


#else
            double3 v0 = __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _vertexes[v0I]);
            double3 v1 = __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _vertexes[v0I]);
            double3 v2 = __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[v0I]);

            __GEIGEN__::Matrix3x3d Ds;
            __GEIGEN__::__set_Mat_val_column(Ds, v0, v1, v2);

            double3 normal = __GEIGEN__::__normalized(__GEIGEN__::__v_vec_cross(
                __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.y]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.y])));
            double  dis    = __GEIGEN__::__v_vec_dot(v0, normal);

            if(dis > 0)
            {
                normal = make_double3(-normal.x, -normal.y, -normal.z);
            }
            else
            {
                dis = -dis;
            }

            double3 pos0 = __GEIGEN__::__add(
                _vertexes[v0I], __GEIGEN__::__s_vec_multiply(normal, dHat_sqrt - dis));


            double3 u0 = __GEIGEN__::__minus(_vertexes[MMCVIDI.y], pos0);
            double3 u1 = __GEIGEN__::__minus(_vertexes[MMCVIDI.z], pos0);
            double3 u2 = __GEIGEN__::__minus(_vertexes[MMCVIDI.w], pos0);

            __GEIGEN__::Matrix3x3d Dm, DmInv;
            __GEIGEN__::__set_Mat_val_column(Dm, u0, u1, u2);

            __GEIGEN__::__Inverse(Dm, DmInv);

            __GEIGEN__::Matrix3x3d F;
            __GEIGEN__::__M_Mat_multiply(Ds, DmInv, F);
            __GEIGEN__::Matrix3x3d uu, vv, ss;
            __GEIGEN__::SVD(F, uu, vv, ss);
            double values = ss.m[0][0] + ss.m[1][1] + ss.m[2][2];
            values        = (values - 2) * (values - 2);
            double3 FxN   = __GEIGEN__::__M_v_multiply(F, normal);
            double  I5    = __GEIGEN__::__squaredNorm(FxN);

            __GEIGEN__::Matrix9x12d PFPx = __computePFDsPX3D_double(DmInv);
#endif

#if (RANK == 1)
            double lambda0 =
                Kappa
                * (2 * dHat * dHat
                   * (6 * I5 + 2 * I5 * log(I5) - 7 * I5 * I5 - 6 * I5 * I5 * log(I5) + 1))
                / I5;
            if(dis * dis < gassThreshold * dHat)
            {
                double lambda1 =
                    Kappa
                    * (2 * dHat * dHat
                       * (6 * gassThreshold + 2 * gassThreshold * log(gassThreshold)
                          - 7 * gassThreshold * gassThreshold
                          - 6 * gassThreshold * gassThreshold * log(gassThreshold) + 1))
                    / gassThreshold;
                lambda0 = lambda1;
            }
#elif (RANK == 2)
            double lambda0 =
                -(4 * Kappa * dHat * dHat
                  * (4 * I5 + log(I5) - 3 * I5 * I5 * log(I5) * log(I5) + 6 * I5 * log(I5)
                     - 2 * I5 * I5 + I5 * log(I5) * log(I5) - 7 * I5 * I5 * log(I5) - 2))
                / I5;
            if(dis * dis < gassThreshold * dHat)
            {
                double lambda1 =
                    -(4 * Kappa * dHat * dHat
                      * (4 * gassThreshold + log(gassThreshold)
                         - 3 * gassThreshold * gassThreshold * log(gassThreshold) * log(gassThreshold)
                         + 6 * gassThreshold * log(gassThreshold) - 2 * gassThreshold * gassThreshold
                         + gassThreshold * log(gassThreshold) * log(gassThreshold)
                         - 7 * gassThreshold * gassThreshold * log(gassThreshold) - 2))
                    / gassThreshold;
                lambda0 = lambda1;
            }
#elif (RANK == 3)
            double lambda0 =
                (2 * Kappa * dHat * dHat * log(I5)
                 * (24 * I5 + 3 * log(I5) - 6 * I5 * I5 * log(I5) * log(I5)
                    + 18 * I5 * log(I5) - 12 * I5 * I5
                    + 2 * I5 * log(I5) * log(I5) - 21 * I5 * I5 * log(I5) - 12))
                / I5;
#elif (RANK == 4)
            double lambda0 =
                -(4 * Kappa * dHat * dHat * log(I5) * log(I5)
                  * (24 * I5 + 2 * log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                     + 12 * I5 * log(I5) - 12 * I5 * I5 + I5 * log(I5) * log(I5)
                     - 14 * I5 * I5 * log(I5) - 12))
                / I5;
#elif (RANK == 5)
            double lambda0 =
                (2 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                 * (80 * I5 + 5 * log(I5) - 6 * I5 * I5 * log(I5) * log(I5)
                    + 30 * I5 * log(I5) - 40 * I5 * I5
                    + 2 * I5 * log(I5) * log(I5) - 35 * I5 * I5 * log(I5) - 40))
                / I5;
#elif (RANK == 6)
            double lambda0 =
                -(4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5)
                  * (60 * I5 + 3 * log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                     + 18 * I5 * log(I5) - 30 * I5 * I5 + I5 * log(I5) * log(I5)
                     - 21 * I5 * I5 * log(I5) - 30))
                / I5;
#endif

#ifdef NEWF
            //printf("lamdba0:    %f\n", lambda0*1e6);
            //__GEIGEN__::__v9_vec9_toMat9x9(H,q0, q0, lambda0); //__GEIGEN__::__S_Mat9x9_multiply(__GEIGEN__::__v9_vec9_toMat9x9(q0, q0), lambda0);
            //__GEIGEN__::Matrix9x9d H;
            //__GEIGEN__::__init_Mat9x9(H, 0);
            //H.m[8][8] = lambda0;
            __GEIGEN__::Matrix9x9d H = __GEIGEN__::__S_Mat9x9_multiply(
                __GEIGEN__::__v9_vec9_toMat9x9(q0, q0), lambda0);  //__GEIGEN__::__v9_vec9_toMat9x9(q0, q0, lambda0);
            __GEIGEN__::Matrix12x12d Hessian;  // = __GEIGEN__::__M12x9_M9x12_Multiply(__GEIGEN__::__M12x9_M9x9_Multiply(PFPxT, H), __GEIGEN__::__Transpose12x9(PFPxT));
            __GEIGEN__::__M12x9_S9x9_MT9x12_Multiply(PFPxT, H, Hessian);

#else

            __GEIGEN__::Matrix3x3d Q0;

            __GEIGEN__::Matrix3x3d fnn;

            __GEIGEN__::Matrix3x3d nn = __GEIGEN__::__v_vec_toMat(normal, normal);

            __GEIGEN__::__M_Mat_multiply(F, nn, fnn);

            __GEIGEN__::Vector9 q0 = __GEIGEN__::__Mat3x3_to_vec9_double(fnn);

            q0 = __GEIGEN__::__s_vec9_multiply(q0, 1.0 / sqrt(I5));

            __GEIGEN__::Matrix9x9d H = __GEIGEN__::__S_Mat9x9_multiply(
                __GEIGEN__::__v9_vec9_toMat9x9(q0, q0), lambda0);

            __GEIGEN__::Matrix12x9d PFPxTransPos = __GEIGEN__::__Transpose9x12(PFPx);
            __GEIGEN__::Matrix12x12d H2 = __GEIGEN__::__M12x9_M9x12_Multiply(
                __GEIGEN__::__M12x9_M9x9_Multiply(PFPxTransPos, H), PFPx);
#endif

            int Hidx = matIndex[idx];  //atomicAdd(_cpNum + 4, 1);

            uint4 global_index = make_uint4(v0I, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);
            //D4Index[Hidx]         = global_index;
            int triplet_id_offset = Hidx * 16;
            write_triplet<12, 12>(
                triplet_values, row_ids, col_ids, &(global_index.x), Hessian.m, triplet_id_offset);
        }
    }
}



//=============================================================================
// Barrier Gradient And Hessian Kernel
//=============================================================================

__global__ void _calBarrierGradientAndHessian(const double3*   _vertexes,
                                              const double3*   _rest_vertexes,
                                              const int4*      _collisionPair,
                                              double3*         _gradient,
                                              Eigen::Matrix3d* triplet_values,
                                              int*             row_ids,
                                              int*             col_ids,
                                              uint32_t*        _cpNum,
                                              int*             matIndex,
                                              double           dHat,
                                              double           Kappa,
                                              int              offset4,
                                              int              offset3,
                                              int              offset2,
                                              int              number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    int4   MMCVIDI   = _collisionPair[idx];
    double dHat_sqrt = sqrt(dHat);
    //double dHat = dHat_sqrt * dHat_sqrt;
    //double Kappa = 1;
    double gassThreshold = 1e-6;
    if(MMCVIDI.x >= 0)
    {
        if(MMCVIDI.w >= 0)
        {
#ifdef NEWF
            double dis;
            _d_EE(_vertexes[MMCVIDI.x],
                  _vertexes[MMCVIDI.y],
                  _vertexes[MMCVIDI.z],
                  _vertexes[MMCVIDI.w],
                  dis);
            dis                                = sqrt(dis);
            double                  d_hat_sqrt = sqrt(dHat);
            __GEIGEN__::Matrix12x9d PFPxT;
            pFpx_ee2(_vertexes[MMCVIDI.x],
                     _vertexes[MMCVIDI.y],
                     _vertexes[MMCVIDI.z],
                     _vertexes[MMCVIDI.w],
                     d_hat_sqrt,
                     PFPxT);
            double              I5 = pow(dis / d_hat_sqrt, 2);
            __GEIGEN__::Vector9 tmp;
            tmp.v[0] = tmp.v[1] = tmp.v[2] = tmp.v[3] = tmp.v[4] = tmp.v[5] =
                tmp.v[6] = tmp.v[7] = 0;
            tmp.v[8]                = dis / d_hat_sqrt;

            __GEIGEN__::Vector9 q0;
            q0.v[0] = q0.v[1] = q0.v[2] = q0.v[3] = q0.v[4] = q0.v[5] =
                q0.v[6] = q0.v[7] = 0;
            q0.v[8]               = 1;
            //q0 = __GEIGEN__::__s_vec9_multiply(q0, 1.0 / sqrt(I5));

            __GEIGEN__::Matrix9x9d H;
            //__GEIGEN__::__init_Mat9x9(H, 0);
#else

            double3 v0 =
                __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _vertexes[MMCVIDI.x]);
            double3 v1 =
                __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.x]);
            double3 v2 =
                __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.x]);
            __GEIGEN__::Matrix3x3d Ds;
            __GEIGEN__::__set_Mat_val_column(Ds, v0, v1, v2);
            double3 normal = __GEIGEN__::__normalized(__GEIGEN__::__v_vec_cross(
                v0, __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.z])));
            double  dis    = __GEIGEN__::__v_vec_dot(v1, normal);
            if(dis < 0)
            {
                normal = make_double3(-normal.x, -normal.y, -normal.z);
                dis    = -dis;
            }

            double3 pos2 =
                __GEIGEN__::__add(_vertexes[MMCVIDI.z],
                                  __GEIGEN__::__s_vec_multiply(normal, dHat_sqrt - dis));
            double3 pos3 =
                __GEIGEN__::__add(_vertexes[MMCVIDI.w],
                                  __GEIGEN__::__s_vec_multiply(normal, dHat_sqrt - dis));

            double3 u0 = v0;
            double3 u1 = __GEIGEN__::__minus(pos2, _vertexes[MMCVIDI.x]);
            double3 u2 = __GEIGEN__::__minus(pos3, _vertexes[MMCVIDI.x]);

            __GEIGEN__::Matrix3x3d Dm, DmInv;
            __GEIGEN__::__set_Mat_val_column(Dm, u0, u1, u2);

            __GEIGEN__::__Inverse(Dm, DmInv);

            __GEIGEN__::Matrix3x3d F;
            __GEIGEN__::__M_Mat_multiply(Ds, DmInv, F);

            double3 FxN = __GEIGEN__::__M_v_multiply(F, normal);
            double  I5  = __GEIGEN__::__squaredNorm(FxN);

            __GEIGEN__::Matrix9x12d PFPx = __computePFDsPX3D_double(DmInv);

            __GEIGEN__::Matrix3x3d fnn;

            __GEIGEN__::Matrix3x3d nn = __GEIGEN__::__v_vec_toMat(normal, normal);

            __GEIGEN__::__M_Mat_multiply(F, nn, fnn);

            __GEIGEN__::Vector9 tmp = __GEIGEN__::__Mat3x3_to_vec9_double(fnn);

#endif

#if (RANK == 1)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);
#elif (RANK == 2)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5);
#elif (RANK == 3)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                -2
                    * (Kappa * dHat * dHat * log(I5) * log(I5) * (I5 - 1)
                       * (3 * I5 + 2 * I5 * log(I5) - 3))
                    / I5);
#elif (RANK == 4)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                 * (I5 - 1) * (2 * I5 + I5 * log(I5) - 2))
                    / I5);
#elif (RANK == 5)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                -2
                    * (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                       * log(I5) * (I5 - 1) * (5 * I5 + 2 * I5 * log(I5) - 5))
                    / I5);
#elif (RANK == 6)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5)
                 * log(I5) * (I5 - 1) * (3 * I5 + I5 * log(I5) - 3))
                    / I5);
#endif


#if (RANK == 1)
            double lambda0 =
                Kappa
                * (2 * dHat * dHat
                   * (6 * I5 + 2 * I5 * log(I5) - 7 * I5 * I5 - 6 * I5 * I5 * log(I5) + 1))
                / I5;
            if(dis * dis < gassThreshold * dHat)
            {
                double lambda1 =
                    Kappa
                    * (2 * dHat * dHat
                       * (6 * gassThreshold + 2 * gassThreshold * log(gassThreshold)
                          - 7 * gassThreshold * gassThreshold
                          - 6 * gassThreshold * gassThreshold * log(gassThreshold) + 1))
                    / gassThreshold;
                lambda0 = lambda1;
            }
#elif (RANK == 2)
            double lambda0 =
                -(4 * Kappa * dHat * dHat
                  * (4 * I5 + log(I5) - 3 * I5 * I5 * log(I5) * log(I5) + 6 * I5 * log(I5)
                     - 2 * I5 * I5 + I5 * log(I5) * log(I5) - 7 * I5 * I5 * log(I5) - 2))
                / I5;
            if(dis * dis < gassThreshold * dHat)
            {
                double lambda1 =
                    -(4 * Kappa * dHat * dHat
                      * (4 * gassThreshold + log(gassThreshold)
                         - 3 * gassThreshold * gassThreshold * log(gassThreshold) * log(gassThreshold)
                         + 6 * gassThreshold * log(gassThreshold) - 2 * gassThreshold * gassThreshold
                         + gassThreshold * log(gassThreshold) * log(gassThreshold)
                         - 7 * gassThreshold * gassThreshold * log(gassThreshold) - 2))
                    / gassThreshold;
                lambda0 = lambda1;
            }
#elif (RANK == 3)
            double lambda0 =
                (2 * Kappa * dHat * dHat * log(I5)
                 * (24 * I5 + 3 * log(I5) - 6 * I5 * I5 * log(I5) * log(I5)
                    + 18 * I5 * log(I5) - 12 * I5 * I5
                    + 2 * I5 * log(I5) * log(I5) - 21 * I5 * I5 * log(I5) - 12))
                / I5;
#elif (RANK == 4)
            double lambda0 =
                -(4 * Kappa * dHat * dHat * log(I5) * log(I5)
                  * (24 * I5 + 2 * log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                     + 12 * I5 * log(I5) - 12 * I5 * I5 + I5 * log(I5) * log(I5)
                     - 14 * I5 * I5 * log(I5) - 12))
                / I5;
#elif (RANK == 5)
            double lambda0 =
                (2 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                 * (80 * I5 + 5 * log(I5) - 6 * I5 * I5 * log(I5) * log(I5)
                    + 30 * I5 * log(I5) - 40 * I5 * I5
                    + 2 * I5 * log(I5) * log(I5) - 35 * I5 * I5 * log(I5) - 40))
                / I5;
#elif (RANK == 6)
            double lambda0 =
                -(4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5)
                  * (60 * I5 + 3 * log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                     + 18 * I5 * log(I5) - 30 * I5 * I5 + I5 * log(I5) * log(I5)
                     - 21 * I5 * I5 * log(I5) - 30))
                / I5;
#endif


#ifdef NEWF
            __GEIGEN__::Vector12 gradient_vec =
                __GEIGEN__::__M12x9_v9_multiply((PFPxT), flatten_pk1);
            H = __GEIGEN__::__S_Mat9x9_multiply(__GEIGEN__::__v9_vec9_toMat9x9(q0, q0), lambda0);

            __GEIGEN__::Matrix12x12d Hessian;  // = __GEIGEN__::__M12x9_M9x12_Multiply(__GEIGEN__::__M12x9_M9x9_Multiply(PFPxT, H), __GEIGEN__::__Transpose12x9(PFPxT));
            __GEIGEN__::__M12x9_S9x9_MT9x12_Multiply(PFPxT, H, Hessian);
#else

            __GEIGEN__::Vector12 gradient_vec =
                __GEIGEN__::__M12x9_v9_multiply(__GEIGEN__::__Transpose9x12(PFPx), flatten_pk1);
            //__GEIGEN__::Matrix3x3d Q0;

            //            __GEIGEN__::Matrix3x3d fnn;

            //           __GEIGEN__::Matrix3x3d nn = __GEIGEN__::__v_vec_toMat(normal, normal);

            //            __GEIGEN__::__M_Mat_multiply(F, nn, fnn);

            __GEIGEN__::Vector9 q0 = __GEIGEN__::__Mat3x3_to_vec9_double(fnn);

            q0 = __GEIGEN__::__s_vec9_multiply(q0, 1.0 / sqrt(I5));

            __GEIGEN__::Matrix9x9d H;
            __GEIGEN__::__init_Mat9x9(H, 0);

            H = __GEIGEN__::__S_Mat9x9_multiply(__GEIGEN__::__v9_vec9_toMat9x9(q0, q0), lambda0);

            __GEIGEN__::Matrix12x9d PFPxTransPos = __GEIGEN__::__Transpose9x12(PFPx);
            __GEIGEN__::Matrix12x12d Hessian = __GEIGEN__::__M12x9_M9x12_Multiply(
                __GEIGEN__::__M12x9_M9x9_Multiply(PFPxTransPos, H), PFPx);
#endif

            {
                atomicAdd(&(_gradient[MMCVIDI.x].x), gradient_vec.v[0]);
                atomicAdd(&(_gradient[MMCVIDI.x].y), gradient_vec.v[1]);
                atomicAdd(&(_gradient[MMCVIDI.x].z), gradient_vec.v[2]);
                atomicAdd(&(_gradient[MMCVIDI.y].x), gradient_vec.v[3]);
                atomicAdd(&(_gradient[MMCVIDI.y].y), gradient_vec.v[4]);
                atomicAdd(&(_gradient[MMCVIDI.y].z), gradient_vec.v[5]);
                atomicAdd(&(_gradient[MMCVIDI.z].x), gradient_vec.v[6]);
                atomicAdd(&(_gradient[MMCVIDI.z].y), gradient_vec.v[7]);
                atomicAdd(&(_gradient[MMCVIDI.z].z), gradient_vec.v[8]);
                atomicAdd(&(_gradient[MMCVIDI.w].x), gradient_vec.v[9]);
                atomicAdd(&(_gradient[MMCVIDI.w].y), gradient_vec.v[10]);
                atomicAdd(&(_gradient[MMCVIDI.w].z), gradient_vec.v[11]);
            }
            int Hidx = matIndex[idx];  //atomicAdd(_cpNum + 4, 1);

            uint4 global_index =
                make_uint4(MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);

            int triplet_id_offset = Hidx * M12_Off;
            write_triplet<12, 12>(
                triplet_values, row_ids, col_ids, &(global_index.x), Hessian.m, triplet_id_offset);
        }
        else
        {
            //return;
            MMCVIDI.w = -MMCVIDI.w - 1;
            double3 v0 =
                __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _vertexes[MMCVIDI.x]);
            double3 v1 =
                __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.z]);
            double c = __GEIGEN__::__norm(__GEIGEN__::__v_vec_cross(v0, v1)) /*/ __GEIGEN__::__norm(v0)*/;
            double I1 = c * c;
            if(I1 == 0)
                return;
            double dis;
            _d_EE(_vertexes[MMCVIDI.x],
                  _vertexes[MMCVIDI.y],
                  _vertexes[MMCVIDI.z],
                  _vertexes[MMCVIDI.w],
                  dis);
            double I2 = dis / dHat;
            dis       = sqrt(dis);

            __GEIGEN__::Matrix3x3d F;
            __GEIGEN__::__set_Mat_val(F, 1, 0, 0, 0, c, 0, 0, 0, dis / dHat_sqrt);
            double3 n1 = make_double3(0, 1, 0);
            double3 n2 = make_double3(0, 0, 1);

            double eps_x = _compute_epx(_rest_vertexes[MMCVIDI.x],
                                        _rest_vertexes[MMCVIDI.y],
                                        _rest_vertexes[MMCVIDI.z],
                                        _rest_vertexes[MMCVIDI.w]);

            __GEIGEN__::Matrix3x3d g1, g2;

            __GEIGEN__::Matrix3x3d nn = __GEIGEN__::__v_vec_toMat(n1, n1);
            __GEIGEN__::__M_Mat_multiply(F, nn, g1);
            nn = __GEIGEN__::__v_vec_toMat(n2, n2);
            __GEIGEN__::__M_Mat_multiply(F, nn, g2);

            __GEIGEN__::Vector9 flatten_g1 = __GEIGEN__::__Mat3x3_to_vec9_double(g1);
            __GEIGEN__::Vector9 flatten_g2 = __GEIGEN__::__Mat3x3_to_vec9_double(g2);

            __GEIGEN__::Matrix12x9d PFPx;
            pFpx_pee(_vertexes[MMCVIDI.x],
                     _vertexes[MMCVIDI.y],
                     _vertexes[MMCVIDI.z],
                     _vertexes[MMCVIDI.w],
                     dHat_sqrt,
                     PFPx);

#if (RANK == 1)
            double p1 = Kappa * 2
                        * (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                        / (eps_x * eps_x);
            double p2 = Kappa * 2
                        * (I1 * dHat * dHat * (I1 - 2 * eps_x) * (I2 - 1)
                           * (I2 + 2 * I2 * log(I2) - 1))
                        / (I2 * eps_x * eps_x);
#elif (RANK == 2)
            double p1 = -Kappa * 2
                        * (2 * dHat * dHat * log(I2) * log(I2) * (I1 - eps_x)
                           * (I2 - 1) * (I2 - 1))
                        / (eps_x * eps_x);
            double p2 = -Kappa * 2
                        * (2 * I1 * dHat * dHat * log(I2) * (I1 - 2 * eps_x)
                           * (I2 - 1) * (I2 + I2 * log(I2) - 1))
                        / (I2 * (eps_x * eps_x));
#elif (RANK == 4)
            double p1 = -Kappa * 2
                        * (2 * dHat * dHat * pow(log(I2), 4) * (I1 - eps_x)
                           * (I2 - 1) * (I2 - 1))
                        / (eps_x * eps_x);
            double p2 = -Kappa * 2
                        * (2 * I1 * dHat * dHat * pow(log(I2), 3) * (I1 - 2 * eps_x)
                           * (I2 - 1) * (2 * I2 + I2 * log(I2) - 2))
                        / (I2 * (eps_x * eps_x));
#elif (RANK == 6)
            double p1 = -Kappa * 2
                        * (2 * dHat * dHat * pow(log(I2), 6) * (I1 - eps_x)
                           * (I2 - 1) * (I2 - 1))
                        / (eps_x * eps_x);
            double p2 = -Kappa * 2
                        * (2 * I1 * dHat * dHat * pow(log(I2), 5) * (I1 - 2 * eps_x)
                           * (I2 - 1) * (3 * I2 + I2 * log(I2) - 3))
                        / (I2 * (eps_x * eps_x));
#endif
            __GEIGEN__::Vector9 flatten_pk1 =
                __GEIGEN__::__add9(__GEIGEN__::__s_vec9_multiply(flatten_g1, p1),
                                   __GEIGEN__::__s_vec9_multiply(flatten_g2, p2));
            __GEIGEN__::Vector12 gradient_vec =
                __GEIGEN__::__M12x9_v9_multiply(PFPx, flatten_pk1);

            {
                atomicAdd(&(_gradient[MMCVIDI.x].x), gradient_vec.v[0]);
                atomicAdd(&(_gradient[MMCVIDI.x].y), gradient_vec.v[1]);
                atomicAdd(&(_gradient[MMCVIDI.x].z), gradient_vec.v[2]);
                atomicAdd(&(_gradient[MMCVIDI.y].x), gradient_vec.v[3]);
                atomicAdd(&(_gradient[MMCVIDI.y].y), gradient_vec.v[4]);
                atomicAdd(&(_gradient[MMCVIDI.y].z), gradient_vec.v[5]);
                atomicAdd(&(_gradient[MMCVIDI.z].x), gradient_vec.v[6]);
                atomicAdd(&(_gradient[MMCVIDI.z].y), gradient_vec.v[7]);
                atomicAdd(&(_gradient[MMCVIDI.z].z), gradient_vec.v[8]);
                atomicAdd(&(_gradient[MMCVIDI.w].x), gradient_vec.v[9]);
                atomicAdd(&(_gradient[MMCVIDI.w].y), gradient_vec.v[10]);
                atomicAdd(&(_gradient[MMCVIDI.w].z), gradient_vec.v[11]);
            }

#if (RANK == 1)
            double lambda10 =
                Kappa * (4 * dHat * dHat * log(I2) * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x))
                / (eps_x * eps_x);
            double lambda11 =
                Kappa * 2
                * (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                / (eps_x * eps_x);
            double lambda12 =
                Kappa * 2
                * (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                / (eps_x * eps_x);
#elif (RANK == 2)
            double lambda10 = -Kappa
                              * (4 * dHat * dHat * log(I2) * log(I2) * (I2 - 1)
                                 * (I2 - 1) * (3 * I1 - eps_x))
                              / (eps_x * eps_x);
            double lambda11 = -Kappa
                              * (4 * dHat * dHat * log(I2) * log(I2)
                                 * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                              / (eps_x * eps_x);
            double lambda12 = -Kappa
                              * (4 * dHat * dHat * log(I2) * log(I2)
                                 * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                              / (eps_x * eps_x);
#elif (RANK == 4)
            double lambda10 = -Kappa
                              * (4 * dHat * dHat * pow(log(I2), 4) * (I2 - 1)
                                 * (I2 - 1) * (3 * I1 - eps_x))
                              / (eps_x * eps_x);
            double lambda11 = -Kappa
                              * (4 * dHat * dHat * pow(log(I2), 4)
                                 * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                              / (eps_x * eps_x);
            double lambda12 = -Kappa
                              * (4 * dHat * dHat * pow(log(I2), 4)
                                 * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                              / (eps_x * eps_x);
#elif (RANK == 6)
            double lambda10 = -Kappa
                              * (4 * dHat * dHat * pow(log(I2), 6) * (I2 - 1)
                                 * (I2 - 1) * (3 * I1 - eps_x))
                              / (eps_x * eps_x);
            double lambda11 = -Kappa
                              * (4 * dHat * dHat * pow(log(I2), 6)
                                 * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                              / (eps_x * eps_x);
            double lambda12 = -Kappa
                              * (4 * dHat * dHat * pow(log(I2), 6)
                                 * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                              / (eps_x * eps_x);
#endif
            __GEIGEN__::Matrix3x3d Tx, Ty, Tz;
            __GEIGEN__::__set_Mat_val(Tx, 0, 0, 0, 0, 0, 1, 0, -1, 0);
            __GEIGEN__::__set_Mat_val(Ty, 0, 0, -1, 0, 0, 0, 1, 0, 0);
            __GEIGEN__::__set_Mat_val(Tz, 0, 1, 0, -1, 0, 0, 0, 0, 0);

            __GEIGEN__::Vector9 q11 = __GEIGEN__::__Mat3x3_to_vec9_double(
                __GEIGEN__::__M_Mat_multiply(Tx, g1));
            __GEIGEN__::__normalized_vec9_double(q11);
            __GEIGEN__::Vector9 q12 = __GEIGEN__::__Mat3x3_to_vec9_double(
                __GEIGEN__::__M_Mat_multiply(Tz, g1));
            __GEIGEN__::__normalized_vec9_double(q12);

            __GEIGEN__::Matrix9x9d projectedH;
            __GEIGEN__::__init_Mat9x9(projectedH, 0);

            __GEIGEN__::Matrix9x9d M9_temp = __GEIGEN__::__v9_vec9_toMat9x9(q11, q11);
            M9_temp    = __GEIGEN__::__S_Mat9x9_multiply(M9_temp, lambda11);
            projectedH = __GEIGEN__::__Mat9x9_add(projectedH, M9_temp);

            M9_temp    = __GEIGEN__::__v9_vec9_toMat9x9(q12, q12);
            M9_temp    = __GEIGEN__::__S_Mat9x9_multiply(M9_temp, lambda12);
            projectedH = __GEIGEN__::__Mat9x9_add(projectedH, M9_temp);

#if (RANK == 1)
            double lambda20 =
                -Kappa
                * (2 * I1 * dHat * dHat * (I1 - 2 * eps_x)
                   * (6 * I2 + 2 * I2 * log(I2) - 7 * I2 * I2 - 6 * I2 * I2 * log(I2) + 1))
                / (I2 * eps_x * eps_x);
#elif (RANK == 2)
            double lambda20 =
                Kappa
                * (4 * I1 * dHat * dHat * (I1 - 2 * eps_x)
                   * (4 * I2 + log(I2) - 3 * I2 * I2 * log(I2) * log(I2)
                      + 6 * I2 * log(I2) - 2 * I2 * I2 + I2 * log(I2) * log(I2)
                      - 7 * I2 * I2 * log(I2) - 2))
                / (I2 * (eps_x * eps_x));
#elif (RANK == 4)
            double lambda20 =
                Kappa
                * (4 * I1 * dHat * dHat * log(I2) * log(I2) * (I1 - 2 * eps_x)
                   * (24 * I2 + 2 * log(I2) - 3 * I2 * I2 * log(I2) * log(I2)
                      + 12 * I2 * log(I2) - 12 * I2 * I2
                      + I2 * log(I2) * log(I2) - 14 * I2 * I2 * log(I2) - 12))
                / (I2 * (eps_x * eps_x));
#elif (RANK == 6)
            double lambda20 =
                Kappa
                * (4 * I1 * dHat * dHat * pow(log(I2), 4) * (I1 - 2 * eps_x)
                   * (60 * I2 + 3 * log(I2) - 3 * I2 * I2 * log(I2) * log(I2)
                      + 18 * I2 * log(I2) - 30 * I2 * I2
                      + I2 * log(I2) * log(I2) - 21 * I2 * I2 * log(I2) - 30))
                / (I2 * (eps_x * eps_x));
#endif

#if (RANK == 1)
            double lambdag1g = Kappa * 4 * c * F.m[2][2]
                               * ((2 * dHat * dHat * (I1 - eps_x) * (I2 - 1)
                                   * (I2 + 2 * I2 * log(I2) - 1))
                                  / (I2 * eps_x * eps_x));
#elif (RANK == 2)
            double lambdag1g = -Kappa * 4 * c * F.m[2][2]
                               * (4 * dHat * dHat * log(I2) * (I1 - eps_x)
                                  * (I2 - 1) * (I2 + I2 * log(I2) - 1))
                               / (I2 * (eps_x * eps_x));
#elif (RANK == 4)
            double lambdag1g = -Kappa * 4 * c * F.m[2][2]
                               * (4 * dHat * dHat * pow(log(I2), 3) * (I1 - eps_x)
                                  * (I2 - 1) * (2 * I2 + I2 * log(I2) - 2))
                               / (I2 * (eps_x * eps_x));
#elif (RANK == 6)
            double lambdag1g = -Kappa * 4 * c * F.m[2][2]
                               * (4 * dHat * dHat * pow(log(I2), 5) * (I1 - eps_x)
                                  * (I2 - 1) * (3 * I2 + I2 * log(I2) - 3))
                               / (I2 * (eps_x * eps_x));
#endif
            Eigen::Matrix2d FMat2;
            FMat2 << lambda10, lambdag1g, lambdag1g, lambda20;
            makePDGeneral_ipc<double, 2>(FMat2);
            projectedH.m[4][4] += FMat2(0, 0);
            projectedH.m[4][8] += FMat2(0, 1);
            projectedH.m[8][4] += FMat2(1, 0);
            projectedH.m[8][8] += FMat2(1, 1);

            //__GEIGEN__::Matrix9x12d PFPxTransPos = __GEIGEN__::__Transpose12x9(PFPx);
            __GEIGEN__::Matrix12x12d Hessian;  // = __GEIGEN__::__M12x9_M9x12_Multiply(__GEIGEN__::__M12x9_M9x9_Multiply(PFPx, projectedH), PFPxTransPos);
            __GEIGEN__::__M12x9_S9x9_MT9x12_Multiply(PFPx, projectedH, Hessian);
            int Hidx = matIndex[idx];  //int Hidx = atomicAdd(_cpNum + 4, 1);

            uint4 global_index =
                make_uint4(MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);

            int triplet_id_offset = Hidx * M12_Off;
            write_triplet<12, 12>(
                triplet_values, row_ids, col_ids, &(global_index.x), Hessian.m, triplet_id_offset);
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
                    return;
                double dis;
                _d_PP(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], dis);
                double I2 = dis / dHat;
                dis       = sqrt(dis);

                __GEIGEN__::Matrix3x3d F;
                __GEIGEN__::__set_Mat_val(F, 1, 0, 0, 0, c, 0, 0, 0, dis / dHat_sqrt);
                double3 n1 = make_double3(0, 1, 0);
                double3 n2 = make_double3(0, 0, 1);

                double eps_x = _compute_epx(_rest_vertexes[MMCVIDI.x],
                                            _rest_vertexes[MMCVIDI.z],
                                            _rest_vertexes[MMCVIDI.y],
                                            _rest_vertexes[MMCVIDI.w]);

                __GEIGEN__::Matrix3x3d g1, g2;

                __GEIGEN__::Matrix3x3d nn = __GEIGEN__::__v_vec_toMat(n1, n1);
                __GEIGEN__::__M_Mat_multiply(F, nn, g1);
                nn = __GEIGEN__::__v_vec_toMat(n2, n2);
                __GEIGEN__::__M_Mat_multiply(F, nn, g2);

                __GEIGEN__::Vector9 flatten_g1 = __GEIGEN__::__Mat3x3_to_vec9_double(g1);
                __GEIGEN__::Vector9 flatten_g2 = __GEIGEN__::__Mat3x3_to_vec9_double(g2);

                __GEIGEN__::Matrix12x9d PFPx;
                pFpx_ppp(_vertexes[MMCVIDI.x],
                         _vertexes[MMCVIDI.y],
                         _vertexes[MMCVIDI.z],
                         _vertexes[MMCVIDI.w],
                         dHat_sqrt,
                         PFPx);

#if (RANK == 1)
                double p1 =
                    Kappa * 2
                    * (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                    / (eps_x * eps_x);
                double p2 = Kappa * 2
                            * (I1 * dHat * dHat * (I1 - 2 * eps_x) * (I2 - 1)
                               * (I2 + 2 * I2 * log(I2) - 1))
                            / (I2 * eps_x * eps_x);
#elif (RANK == 2)
                double p1 = -Kappa * 2
                            * (2 * dHat * dHat * log(I2) * log(I2)
                               * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                            / (eps_x * eps_x);
                double p2 = -Kappa * 2
                            * (2 * I1 * dHat * dHat * log(I2) * (I1 - 2 * eps_x)
                               * (I2 - 1) * (I2 + I2 * log(I2) - 1))
                            / (I2 * (eps_x * eps_x));
#elif (RANK == 4)
                double p1 = -Kappa * 2
                            * (2 * dHat * dHat * pow(log(I2), 4) * (I1 - eps_x)
                               * (I2 - 1) * (I2 - 1))
                            / (eps_x * eps_x);
                double p2 = -Kappa * 2
                            * (2 * I1 * dHat * dHat * pow(log(I2), 3) * (I1 - 2 * eps_x)
                               * (I2 - 1) * (2 * I2 + I2 * log(I2) - 2))
                            / (I2 * (eps_x * eps_x));
#elif (RANK == 6)
                double p1 = -Kappa * 2
                            * (2 * dHat * dHat * pow(log(I2), 6) * (I1 - eps_x)
                               * (I2 - 1) * (I2 - 1))
                            / (eps_x * eps_x);
                double p2 = -Kappa * 2
                            * (2 * I1 * dHat * dHat * pow(log(I2), 5) * (I1 - 2 * eps_x)
                               * (I2 - 1) * (3 * I2 + I2 * log(I2) - 3))
                            / (I2 * (eps_x * eps_x));
#endif
                __GEIGEN__::Vector9 flatten_pk1 =
                    __GEIGEN__::__add9(__GEIGEN__::__s_vec9_multiply(flatten_g1, p1),
                                       __GEIGEN__::__s_vec9_multiply(flatten_g2, p2));
                __GEIGEN__::Vector12 gradient_vec =
                    __GEIGEN__::__M12x9_v9_multiply(PFPx, flatten_pk1);

                {
                    atomicAdd(&(_gradient[MMCVIDI.x].x), gradient_vec.v[0]);
                    atomicAdd(&(_gradient[MMCVIDI.x].y), gradient_vec.v[1]);
                    atomicAdd(&(_gradient[MMCVIDI.x].z), gradient_vec.v[2]);
                    atomicAdd(&(_gradient[MMCVIDI.y].x), gradient_vec.v[3]);
                    atomicAdd(&(_gradient[MMCVIDI.y].y), gradient_vec.v[4]);
                    atomicAdd(&(_gradient[MMCVIDI.y].z), gradient_vec.v[5]);
                    atomicAdd(&(_gradient[MMCVIDI.z].x), gradient_vec.v[6]);
                    atomicAdd(&(_gradient[MMCVIDI.z].y), gradient_vec.v[7]);
                    atomicAdd(&(_gradient[MMCVIDI.z].z), gradient_vec.v[8]);
                    atomicAdd(&(_gradient[MMCVIDI.w].x), gradient_vec.v[9]);
                    atomicAdd(&(_gradient[MMCVIDI.w].y), gradient_vec.v[10]);
                    atomicAdd(&(_gradient[MMCVIDI.w].z), gradient_vec.v[11]);
                }

#if (RANK == 1)
                double lambda10 =
                    Kappa * (4 * dHat * dHat * log(I2) * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x))
                    / (eps_x * eps_x);
                double lambda11 =
                    Kappa * 2
                    * (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                    / (eps_x * eps_x);
                double lambda12 =
                    Kappa * 2
                    * (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                    / (eps_x * eps_x);
#elif (RANK == 2)
                double lambda10 = -Kappa
                                  * (4 * dHat * dHat * log(I2) * log(I2)
                                     * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x))
                                  / (eps_x * eps_x);
                double lambda11 = -Kappa
                                  * (4 * dHat * dHat * log(I2) * log(I2)
                                     * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                                  / (eps_x * eps_x);
                double lambda12 = -Kappa
                                  * (4 * dHat * dHat * log(I2) * log(I2)
                                     * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                                  / (eps_x * eps_x);
#elif (RANK == 4)
                double lambda10 = -Kappa
                                  * (4 * dHat * dHat * pow(log(I2), 4)
                                     * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x))
                                  / (eps_x * eps_x);
                double lambda11 = -Kappa
                                  * (4 * dHat * dHat * pow(log(I2), 4)
                                     * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                                  / (eps_x * eps_x);
                double lambda12 = -Kappa
                                  * (4 * dHat * dHat * pow(log(I2), 4)
                                     * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                                  / (eps_x * eps_x);
#elif (RANK == 6)
                double lambda10 = -Kappa
                                  * (4 * dHat * dHat * pow(log(I2), 6)
                                     * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x))
                                  / (eps_x * eps_x);
                double lambda11 = -Kappa
                                  * (4 * dHat * dHat * pow(log(I2), 6)
                                     * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                                  / (eps_x * eps_x);
                double lambda12 = -Kappa
                                  * (4 * dHat * dHat * pow(log(I2), 6)
                                     * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                                  / (eps_x * eps_x);
#endif
                __GEIGEN__::Matrix3x3d Tx, Ty, Tz;
                __GEIGEN__::__set_Mat_val(Tx, 0, 0, 0, 0, 0, 1, 0, -1, 0);
                __GEIGEN__::__set_Mat_val(Ty, 0, 0, -1, 0, 0, 0, 1, 0, 0);
                __GEIGEN__::__set_Mat_val(Tz, 0, 1, 0, -1, 0, 0, 0, 0, 0);

                __GEIGEN__::Vector9 q11 = __GEIGEN__::__Mat3x3_to_vec9_double(
                    __GEIGEN__::__M_Mat_multiply(Tx, g1));
                __GEIGEN__::__normalized_vec9_double(q11);
                __GEIGEN__::Vector9 q12 = __GEIGEN__::__Mat3x3_to_vec9_double(
                    __GEIGEN__::__M_Mat_multiply(Tz, g1));
                __GEIGEN__::__normalized_vec9_double(q12);

                __GEIGEN__::Matrix9x9d projectedH;
                __GEIGEN__::__init_Mat9x9(projectedH, 0);

                __GEIGEN__::Matrix9x9d M9_temp = __GEIGEN__::__v9_vec9_toMat9x9(q11, q11);
                M9_temp    = __GEIGEN__::__S_Mat9x9_multiply(M9_temp, lambda11);
                projectedH = __GEIGEN__::__Mat9x9_add(projectedH, M9_temp);

                M9_temp    = __GEIGEN__::__v9_vec9_toMat9x9(q12, q12);
                M9_temp    = __GEIGEN__::__S_Mat9x9_multiply(M9_temp, lambda12);
                projectedH = __GEIGEN__::__Mat9x9_add(projectedH, M9_temp);

#if (RANK == 1)
                double lambda20 = -Kappa
                                  * (2 * I1 * dHat * dHat * (I1 - 2 * eps_x)
                                     * (6 * I2 + 2 * I2 * log(I2) - 7 * I2 * I2
                                        - 6 * I2 * I2 * log(I2) + 1))
                                  / (I2 * eps_x * eps_x);
#elif (RANK == 2)
                double lambda20 =
                    Kappa
                    * (4 * I1 * dHat * dHat * (I1 - 2 * eps_x)
                       * (4 * I2 + log(I2) - 3 * I2 * I2 * log(I2) * log(I2)
                          + 6 * I2 * log(I2) - 2 * I2 * I2
                          + I2 * log(I2) * log(I2) - 7 * I2 * I2 * log(I2) - 2))
                    / (I2 * (eps_x * eps_x));
#elif (RANK == 4)
                double lambda20 =
                    Kappa
                    * (4 * I1 * dHat * dHat * log(I2) * log(I2) * (I1 - 2 * eps_x)
                       * (24 * I2 + 2 * log(I2) - 3 * I2 * I2 * log(I2) * log(I2)
                          + 12 * I2 * log(I2) - 12 * I2 * I2
                          + I2 * log(I2) * log(I2) - 14 * I2 * I2 * log(I2) - 12))
                    / (I2 * (eps_x * eps_x));
#elif (RANK == 6)
                double lambda20 =
                    Kappa
                    * (4 * I1 * dHat * dHat * pow(log(I2), 4) * (I1 - 2 * eps_x)
                       * (60 * I2 + 3 * log(I2) - 3 * I2 * I2 * log(I2) * log(I2)
                          + 18 * I2 * log(I2) - 30 * I2 * I2
                          + I2 * log(I2) * log(I2) - 21 * I2 * I2 * log(I2) - 30))
                    / (I2 * (eps_x * eps_x));
#endif

#if (RANK == 1)
                double lambdag1g = Kappa * 4 * c * F.m[2][2]
                                   * ((2 * dHat * dHat * (I1 - eps_x) * (I2 - 1)
                                       * (I2 + 2 * I2 * log(I2) - 1))
                                      / (I2 * eps_x * eps_x));
#elif (RANK == 2)
                double lambdag1g = -Kappa * 4 * c * F.m[2][2]
                                   * (4 * dHat * dHat * log(I2) * (I1 - eps_x)
                                      * (I2 - 1) * (I2 + I2 * log(I2) - 1))
                                   / (I2 * (eps_x * eps_x));
#elif (RANK == 4)
                double lambdag1g = -Kappa * 4 * c * F.m[2][2]
                                   * (4 * dHat * dHat * pow(log(I2), 3) * (I1 - eps_x)
                                      * (I2 - 1) * (2 * I2 + I2 * log(I2) - 2))
                                   / (I2 * (eps_x * eps_x));
#elif (RANK == 6)
                double lambdag1g = -Kappa * 4 * c * F.m[2][2]
                                   * (4 * dHat * dHat * pow(log(I2), 5) * (I1 - eps_x)
                                      * (I2 - 1) * (3 * I2 + I2 * log(I2) - 3))
                                   / (I2 * (eps_x * eps_x));
#endif
                Eigen::Matrix2d FMat2;
                FMat2 << lambda10, lambdag1g, lambdag1g, lambda20;
                makePDGeneral_ipc<double, 2>(FMat2);
                projectedH.m[4][4] += FMat2(0, 0);
                projectedH.m[4][8] += FMat2(0, 1);
                projectedH.m[8][4] += FMat2(1, 0);
                projectedH.m[8][8] += FMat2(1, 1);

                //__GEIGEN__::Matrix9x12d PFPxTransPos = __GEIGEN__::__Transpose12x9(PFPx);
                __GEIGEN__::Matrix12x12d Hessian;  // = __GEIGEN__::__M12x9_M9x12_Multiply(__GEIGEN__::__M12x9_M9x9_Multiply(PFPx, projectedH), PFPxTransPos);
                __GEIGEN__::__M12x9_S9x9_MT9x12_Multiply(PFPx, projectedH, Hessian);
                int Hidx = matIndex[idx];  //int Hidx = atomicAdd(_cpNum + 4, 1);

                uint4 global_index =
                    make_uint4(MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);

                int triplet_id_offset = Hidx * M12_Off;
                write_triplet<12, 12>(
                    triplet_values, row_ids, col_ids, &(global_index.x), Hessian.m, triplet_id_offset);
            }
            else
            {
#ifdef NEWF
                double dis;
                _d_PP(_vertexes[v0I], _vertexes[MMCVIDI.y], dis);
                dis                            = sqrt(dis);
                double              d_hat_sqrt = sqrt(dHat);
                __GEIGEN__::Vector6 PFPxT;
                pFpx_pp2(_vertexes[v0I], _vertexes[MMCVIDI.y], d_hat_sqrt, PFPxT);
                double I5  = pow(dis / d_hat_sqrt, 2);
                double fnn = dis / d_hat_sqrt;

#if (RANK == 1)
                double flatten_pk1 =
                    fnn * 2 * Kappa
                    * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5;
#elif (RANK == 2)
                double flatten_pk1 = fnn * 2
                                     * (2 * Kappa * dHat * dHat * log(I5)
                                        * (I5 - 1) * (I5 + I5 * log(I5) - 1))
                                     / I5;
#elif (RANK == 3)
                double flatten_pk1 = fnn * -2
                                     * (Kappa * dHat * dHat * log(I5) * log(I5)
                                        * (I5 - 1) * (3 * I5 + 2 * I5 * log(I5) - 3))
                                     / I5;
#elif (RANK == 4)
                double flatten_pk1 =
                    fnn
                    * (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                       * (I5 - 1) * (2 * I5 + I5 * log(I5) - 2))
                    / I5;
#elif (RANK == 5)
                double flatten_pk1 =
                    fnn * -2
                    * (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                       * log(I5) * (I5 - 1) * (5 * I5 + 2 * I5 * log(I5) - 5))
                    / I5;
#elif (RANK == 6)
                double flatten_pk1 =
                    fnn
                    * (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                       * log(I5) * log(I5) * (I5 - 1) * (3 * I5 + I5 * log(I5) - 3))
                    / I5;
#endif

                __GEIGEN__::Vector6 gradient_vec =
                    __GEIGEN__::__s_vec6_multiply(PFPxT, flatten_pk1);

#else
                double3 v0 = __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _vertexes[v0I]);
                double3 Ds  = v0;
                double  dis = __GEIGEN__::__norm(v0);
                //if (dis > dHat_sqrt) return;
                double3 vec_normal =
                    __GEIGEN__::__normalized(make_double3(-v0.x, -v0.y, -v0.z));
                double3 target = make_double3(0, 1, 0);
                double3 vec    = __GEIGEN__::__v_vec_cross(vec_normal, target);
                double  cos    = __GEIGEN__::__v_vec_dot(vec_normal, target);
                __GEIGEN__::Matrix3x3d rotation;
                __GEIGEN__::__set_Mat_val(rotation, 1, 0, 0, 0, 1, 0, 0, 0, 1);
                __GEIGEN__::Vector6 PDmPx;
                if(cos + 1 == 0)
                {
                    rotation.m[0][0] = -1;
                    rotation.m[1][1] = -1;
                }
                else
                {
                    __GEIGEN__::Matrix3x3d cross_vec;
                    __GEIGEN__::__set_Mat_val(
                        cross_vec, 0, -vec.z, vec.y, vec.z, 0, -vec.x, -vec.y, vec.x, 0);

                    rotation = __GEIGEN__::__Mat_add(
                        rotation,
                        __GEIGEN__::__Mat_add(cross_vec,
                                              __GEIGEN__::__S_Mat_multiply(
                                                  __GEIGEN__::__M_Mat_multiply(cross_vec, cross_vec),
                                                  1.0 / (1 + cos))));
                }

                double3 pos0 = __GEIGEN__::__add(
                    _vertexes[v0I],
                    __GEIGEN__::__s_vec_multiply(vec_normal, dHat_sqrt - dis));
                double3 rotate_uv0 = __GEIGEN__::__M_v_multiply(rotation, pos0);
                double3 rotate_uv1 =
                    __GEIGEN__::__M_v_multiply(rotation, _vertexes[MMCVIDI.y]);

                double uv0 = rotate_uv0.y;
                double uv1 = rotate_uv1.y;

                double u0    = uv1 - uv0;
                double Dm    = u0;  //PFPx
                double DmInv = 1 / u0;

                double3 F  = __GEIGEN__::__s_vec_multiply(Ds, DmInv);
                double  I5 = __GEIGEN__::__squaredNorm(F);

                double3 tmp = F;

#if (RANK == 1)
                double3 flatten_pk1 = __GEIGEN__::__s_vec_multiply(
                    tmp, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);
#elif (RANK == 2)
                double3 flatten_pk1 = __GEIGEN__::__s_vec_multiply(
                    tmp,
                    2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1))
                        / I5);

#elif (RANK == 3)
                double3 flatten_pk1 = __GEIGEN__::__s_vec_multiply(
                    tmp,
                    -2
                        * (Kappa * dHat * dHat * log(I5) * log(I5) * (I5 - 1)
                           * (3 * I5 + 2 * I5 * log(I5) - 3))
                        / I5);
#elif (RANK == 4)
                double3 flatten_pk1 = __GEIGEN__::__s_vec_multiply(
                    tmp,
                    (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                     * (I5 - 1) * (2 * I5 + I5 * log(I5) - 2))
                        / I5);
#elif (RANK == 5)
                double3 flatten_pk1 = __GEIGEN__::__s_vec_multiply(
                    tmp,
                    -2
                        * (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                           * log(I5) * (I5 - 1) * (5 * I5 + 2 * I5 * log(I5) - 5))
                        / I5);
#elif (RANK == 6)
                double3 flatten_pk1 = __GEIGEN__::__s_vec_multiply(
                    tmp,
                    (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                     * log(I5) * log(I5) * (I5 - 1) * (3 * I5 + I5 * log(I5) - 3))
                        / I5);
#endif
                __GEIGEN__::Matrix3x6d PFPx = __computePFDsPX3D_3x6_double(DmInv);

                __GEIGEN__::Vector6 gradient_vec =
                    __GEIGEN__::__M6x3_v3_multiply(__GEIGEN__::__Transpose3x6(PFPx), flatten_pk1);
#endif


                {
                    atomicAdd(&(_gradient[v0I].x), gradient_vec.v[0]);
                    atomicAdd(&(_gradient[v0I].y), gradient_vec.v[1]);
                    atomicAdd(&(_gradient[v0I].z), gradient_vec.v[2]);
                    atomicAdd(&(_gradient[MMCVIDI.y].x), gradient_vec.v[3]);
                    atomicAdd(&(_gradient[MMCVIDI.y].y), gradient_vec.v[4]);
                    atomicAdd(&(_gradient[MMCVIDI.y].z), gradient_vec.v[5]);
                }

#if (RANK == 1)
                double lambda0 = Kappa
                                 * (2 * dHat * dHat
                                    * (6 * I5 + 2 * I5 * log(I5) - 7 * I5 * I5
                                       - 6 * I5 * I5 * log(I5) + 1))
                                 / I5;
                if(dis * dis < gassThreshold * dHat)
                {
                    double lambda1 =
                        Kappa
                        * (2 * dHat * dHat
                           * (6 * gassThreshold + 2 * gassThreshold * log(gassThreshold)
                              - 7 * gassThreshold * gassThreshold
                              - 6 * gassThreshold * gassThreshold * log(gassThreshold) + 1))
                        / gassThreshold;
                    lambda0 = lambda1;
                }
#elif (RANK == 2)
                double lambda0 =
                    -(4 * Kappa * dHat * dHat
                      * (4 * I5 + log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                         + 6 * I5 * log(I5) - 2 * I5 * I5
                         + I5 * log(I5) * log(I5) - 7 * I5 * I5 * log(I5) - 2))
                    / I5;
                if(dis * dis < gassThreshold * dHat)
                {
                    double lambda1 =
                        -(4 * Kappa * dHat * dHat
                          * (4 * gassThreshold + log(gassThreshold)
                             - 3 * gassThreshold * gassThreshold
                                   * log(gassThreshold) * log(gassThreshold)
                             + 6 * gassThreshold * log(gassThreshold) - 2 * gassThreshold * gassThreshold
                             + gassThreshold * log(gassThreshold) * log(gassThreshold)
                             - 7 * gassThreshold * gassThreshold * log(gassThreshold) - 2))
                        / gassThreshold;
                    lambda0 = lambda1;
                }
#elif (RANK == 3)
                double lambda0 =
                    (2 * Kappa * dHat * dHat * log(I5)
                     * (24 * I5 + 3 * log(I5) - 6 * I5 * I5 * log(I5) * log(I5)
                        + 18 * I5 * log(I5) - 12 * I5 * I5
                        + 2 * I5 * log(I5) * log(I5) - 21 * I5 * I5 * log(I5) - 12))
                    / I5;
#elif (RANK == 4)
                double lambda0 =
                    -(4 * Kappa * dHat * dHat * log(I5) * log(I5)
                      * (24 * I5 + 2 * log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                         + 12 * I5 * log(I5) - 12 * I5 * I5
                         + I5 * log(I5) * log(I5) - 14 * I5 * I5 * log(I5) - 12))
                    / I5;
#elif (RANK == 5)
                double lambda0 =
                    (2 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                     * (80 * I5 + 5 * log(I5) - 6 * I5 * I5 * log(I5) * log(I5)
                        + 30 * I5 * log(I5) - 40 * I5 * I5
                        + 2 * I5 * log(I5) * log(I5) - 35 * I5 * I5 * log(I5) - 40))
                    / I5;
#elif (RANK == 6)
                double lambda0 =
                    -(4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5)
                      * (60 * I5 + 3 * log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                         + 18 * I5 * log(I5) - 30 * I5 * I5
                         + I5 * log(I5) * log(I5) - 21 * I5 * I5 * log(I5) - 30))
                    / I5;
#endif


#ifdef NEWF
                double                 H       = lambda0;
                __GEIGEN__::Matrix6x6d Hessian = __GEIGEN__::__s_M6x6_Multiply(
                    __GEIGEN__::__v6_vec6_toMat6x6(PFPxT, PFPxT), H);
#else
                double3 q0 = __GEIGEN__::__s_vec_multiply(F, 1 / sqrt(I5));

                __GEIGEN__::Matrix3x3d H =
                    __GEIGEN__::__S_Mat_multiply(__GEIGEN__::__v_vec_toMat(q0, q0),
                                                 lambda0);  //lambda0 * q0 * q0.transpose();

                __GEIGEN__::Matrix6x3d PFPxTransPos = __GEIGEN__::__Transpose3x6(PFPx);
                __GEIGEN__::Matrix6x6d Hessian = __GEIGEN__::__M6x3_M3x6_Multiply(
                    __GEIGEN__::__M6x3_M3x3_Multiply(PFPxTransPos, H), PFPx);
#endif
                int Hidx = matIndex[idx];  //int Hidx = atomicAdd(_cpNum + 2, 1);

                //H6x6[Hidx]    = Hessian;
                uint2 global_index = make_uint2(v0I, MMCVIDI.y);
                //D2Index[Hidx]      = global_index;

                int triplet_id_offset = Hidx * M6_Off + offset3 * M9_Off + offset4 * M12_Off;
                write_triplet<6, 6>(
                    triplet_values, row_ids, col_ids, &(global_index.x), Hessian.m, triplet_id_offset);
            }
        }
        else if(MMCVIDI.w < 0)
        {
            if(MMCVIDI.y < 0)
            {
                MMCVIDI.y = -MMCVIDI.y - 1;
                MMCVIDI.x = v0I;
                MMCVIDI.w = -MMCVIDI.w - 1;
                double3 v0 =
                    __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.x]);
                double3 v1 =
                    __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.y]);
                double c = __GEIGEN__::__norm(__GEIGEN__::__v_vec_cross(v0, v1)) /*/ __GEIGEN__::__norm(v0)*/;
                double I1 = c * c;
                if(I1 == 0)
                    return;
                double dis;
                _d_PE(_vertexes[MMCVIDI.x],
                      _vertexes[MMCVIDI.y],
                      _vertexes[MMCVIDI.z],
                      dis);
                double I2 = dis / dHat;
                dis       = sqrt(dis);

                __GEIGEN__::Matrix3x3d F;
                __GEIGEN__::__set_Mat_val(F, 1, 0, 0, 0, c, 0, 0, 0, dis / dHat_sqrt);
                double3 n1 = make_double3(0, 1, 0);
                double3 n2 = make_double3(0, 0, 1);

                double eps_x = _compute_epx(_rest_vertexes[MMCVIDI.x],
                                            _rest_vertexes[MMCVIDI.w],
                                            _rest_vertexes[MMCVIDI.y],
                                            _rest_vertexes[MMCVIDI.z]);

                __GEIGEN__::Matrix3x3d g1, g2;

                __GEIGEN__::Matrix3x3d nn = __GEIGEN__::__v_vec_toMat(n1, n1);
                __GEIGEN__::__M_Mat_multiply(F, nn, g1);
                nn = __GEIGEN__::__v_vec_toMat(n2, n2);
                __GEIGEN__::__M_Mat_multiply(F, nn, g2);

                __GEIGEN__::Vector9 flatten_g1 = __GEIGEN__::__Mat3x3_to_vec9_double(g1);
                __GEIGEN__::Vector9 flatten_g2 = __GEIGEN__::__Mat3x3_to_vec9_double(g2);

                __GEIGEN__::Matrix12x9d PFPx;
                pFpx_ppe(_vertexes[MMCVIDI.x],
                         _vertexes[MMCVIDI.y],
                         _vertexes[MMCVIDI.z],
                         _vertexes[MMCVIDI.w],
                         dHat_sqrt,
                         PFPx);

#if (RANK == 1)
                double p1 =
                    Kappa * 2
                    * (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                    / (eps_x * eps_x);
                double p2 = Kappa * 2
                            * (I1 * dHat * dHat * (I1 - 2 * eps_x) * (I2 - 1)
                               * (I2 + 2 * I2 * log(I2) - 1))
                            / (I2 * eps_x * eps_x);
#elif (RANK == 2)
                double p1 = -Kappa * 2
                            * (2 * dHat * dHat * log(I2) * log(I2)
                               * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                            / (eps_x * eps_x);
                double p2 = -Kappa * 2
                            * (2 * I1 * dHat * dHat * log(I2) * (I1 - 2 * eps_x)
                               * (I2 - 1) * (I2 + I2 * log(I2) - 1))
                            / (I2 * (eps_x * eps_x));
#elif (RANK == 4)
                double p1 = -Kappa * 2
                            * (2 * dHat * dHat * pow(log(I2), 4) * (I1 - eps_x)
                               * (I2 - 1) * (I2 - 1))
                            / (eps_x * eps_x);
                double p2 = -Kappa * 2
                            * (2 * I1 * dHat * dHat * pow(log(I2), 3) * (I1 - 2 * eps_x)
                               * (I2 - 1) * (2 * I2 + I2 * log(I2) - 2))
                            / (I2 * (eps_x * eps_x));
#elif (RANK == 6)
                double p1 = -Kappa * 2
                            * (2 * dHat * dHat * pow(log(I2), 6) * (I1 - eps_x)
                               * (I2 - 1) * (I2 - 1))
                            / (eps_x * eps_x);
                double p2 = -Kappa * 2
                            * (2 * I1 * dHat * dHat * pow(log(I2), 5) * (I1 - 2 * eps_x)
                               * (I2 - 1) * (3 * I2 + I2 * log(I2) - 3))
                            / (I2 * (eps_x * eps_x));
#endif
                __GEIGEN__::Vector9 flatten_pk1 =
                    __GEIGEN__::__add9(__GEIGEN__::__s_vec9_multiply(flatten_g1, p1),
                                       __GEIGEN__::__s_vec9_multiply(flatten_g2, p2));
                __GEIGEN__::Vector12 gradient_vec =
                    __GEIGEN__::__M12x9_v9_multiply(PFPx, flatten_pk1);

                {
                    atomicAdd(&(_gradient[MMCVIDI.x].x), gradient_vec.v[0]);
                    atomicAdd(&(_gradient[MMCVIDI.x].y), gradient_vec.v[1]);
                    atomicAdd(&(_gradient[MMCVIDI.x].z), gradient_vec.v[2]);
                    atomicAdd(&(_gradient[MMCVIDI.y].x), gradient_vec.v[3]);
                    atomicAdd(&(_gradient[MMCVIDI.y].y), gradient_vec.v[4]);
                    atomicAdd(&(_gradient[MMCVIDI.y].z), gradient_vec.v[5]);
                    atomicAdd(&(_gradient[MMCVIDI.z].x), gradient_vec.v[6]);
                    atomicAdd(&(_gradient[MMCVIDI.z].y), gradient_vec.v[7]);
                    atomicAdd(&(_gradient[MMCVIDI.z].z), gradient_vec.v[8]);
                    atomicAdd(&(_gradient[MMCVIDI.w].x), gradient_vec.v[9]);
                    atomicAdd(&(_gradient[MMCVIDI.w].y), gradient_vec.v[10]);
                    atomicAdd(&(_gradient[MMCVIDI.w].z), gradient_vec.v[11]);
                }

#if (RANK == 1)
                double lambda10 =
                    Kappa * (4 * dHat * dHat * log(I2) * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x))
                    / (eps_x * eps_x);
                double lambda11 =
                    Kappa * 2
                    * (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                    / (eps_x * eps_x);
                double lambda12 =
                    Kappa * 2
                    * (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                    / (eps_x * eps_x);
#elif (RANK == 2)
                double lambda10 = -Kappa
                                  * (4 * dHat * dHat * log(I2) * log(I2)
                                     * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x))
                                  / (eps_x * eps_x);
                double lambda11 = -Kappa
                                  * (4 * dHat * dHat * log(I2) * log(I2)
                                     * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                                  / (eps_x * eps_x);
                double lambda12 = -Kappa
                                  * (4 * dHat * dHat * log(I2) * log(I2)
                                     * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                                  / (eps_x * eps_x);
#elif (RANK == 4)
                double lambda10 = -Kappa
                                  * (4 * dHat * dHat * pow(log(I2), 4)
                                     * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x))
                                  / (eps_x * eps_x);
                double lambda11 = -Kappa
                                  * (4 * dHat * dHat * pow(log(I2), 4)
                                     * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                                  / (eps_x * eps_x);
                double lambda12 = -Kappa
                                  * (4 * dHat * dHat * pow(log(I2), 4)
                                     * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                                  / (eps_x * eps_x);
#elif (RANK == 6)
                double lambda10 = -Kappa
                                  * (4 * dHat * dHat * pow(log(I2), 6)
                                     * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x))
                                  / (eps_x * eps_x);
                double lambda11 = -Kappa
                                  * (4 * dHat * dHat * pow(log(I2), 6)
                                     * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                                  / (eps_x * eps_x);
                double lambda12 = -Kappa
                                  * (4 * dHat * dHat * pow(log(I2), 6)
                                     * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                                  / (eps_x * eps_x);
#endif
                __GEIGEN__::Matrix3x3d Tx, Ty, Tz;
                __GEIGEN__::__set_Mat_val(Tx, 0, 0, 0, 0, 0, 1, 0, -1, 0);
                __GEIGEN__::__set_Mat_val(Ty, 0, 0, -1, 0, 0, 0, 1, 0, 0);
                __GEIGEN__::__set_Mat_val(Tz, 0, 1, 0, -1, 0, 0, 0, 0, 0);

                __GEIGEN__::Vector9 q11 = __GEIGEN__::__Mat3x3_to_vec9_double(
                    __GEIGEN__::__M_Mat_multiply(Tx, g1));
                __GEIGEN__::__normalized_vec9_double(q11);
                __GEIGEN__::Vector9 q12 = __GEIGEN__::__Mat3x3_to_vec9_double(
                    __GEIGEN__::__M_Mat_multiply(Tz, g1));
                __GEIGEN__::__normalized_vec9_double(q12);

                __GEIGEN__::Matrix9x9d projectedH;
                __GEIGEN__::__init_Mat9x9(projectedH, 0);

                __GEIGEN__::Matrix9x9d M9_temp = __GEIGEN__::__v9_vec9_toMat9x9(q11, q11);
                M9_temp    = __GEIGEN__::__S_Mat9x9_multiply(M9_temp, lambda11);
                projectedH = __GEIGEN__::__Mat9x9_add(projectedH, M9_temp);

                M9_temp    = __GEIGEN__::__v9_vec9_toMat9x9(q12, q12);
                M9_temp    = __GEIGEN__::__S_Mat9x9_multiply(M9_temp, lambda12);
                projectedH = __GEIGEN__::__Mat9x9_add(projectedH, M9_temp);

#if (RANK == 1)
                double lambda20 = -Kappa
                                  * (2 * I1 * dHat * dHat * (I1 - 2 * eps_x)
                                     * (6 * I2 + 2 * I2 * log(I2) - 7 * I2 * I2
                                        - 6 * I2 * I2 * log(I2) + 1))
                                  / (I2 * eps_x * eps_x);
#elif (RANK == 2)
                double lambda20 =
                    Kappa
                    * (4 * I1 * dHat * dHat * (I1 - 2 * eps_x)
                       * (4 * I2 + log(I2) - 3 * I2 * I2 * log(I2) * log(I2)
                          + 6 * I2 * log(I2) - 2 * I2 * I2
                          + I2 * log(I2) * log(I2) - 7 * I2 * I2 * log(I2) - 2))
                    / (I2 * (eps_x * eps_x));
#elif (RANK == 4)
                double lambda20 =
                    Kappa
                    * (4 * I1 * dHat * dHat * log(I2) * log(I2) * (I1 - 2 * eps_x)
                       * (24 * I2 + 2 * log(I2) - 3 * I2 * I2 * log(I2) * log(I2)
                          + 12 * I2 * log(I2) - 12 * I2 * I2
                          + I2 * log(I2) * log(I2) - 14 * I2 * I2 * log(I2) - 12))
                    / (I2 * (eps_x * eps_x));
#elif (RANK == 6)
                double lambda20 =
                    Kappa
                    * (4 * I1 * dHat * dHat * pow(log(I2), 4) * (I1 - 2 * eps_x)
                       * (60 * I2 + 3 * log(I2) - 3 * I2 * I2 * log(I2) * log(I2)
                          + 18 * I2 * log(I2) - 30 * I2 * I2
                          + I2 * log(I2) * log(I2) - 21 * I2 * I2 * log(I2) - 30))
                    / (I2 * (eps_x * eps_x));
#endif

#if (RANK == 1)
                double lambdag1g = Kappa * 4 * c * F.m[2][2]
                                   * ((2 * dHat * dHat * (I1 - eps_x) * (I2 - 1)
                                       * (I2 + 2 * I2 * log(I2) - 1))
                                      / (I2 * eps_x * eps_x));
#elif (RANK == 2)
                double lambdag1g = -Kappa * 4 * c * F.m[2][2]
                                   * (4 * dHat * dHat * log(I2) * (I1 - eps_x)
                                      * (I2 - 1) * (I2 + I2 * log(I2) - 1))
                                   / (I2 * (eps_x * eps_x));
#elif (RANK == 4)
                double lambdag1g = -Kappa * 4 * c * F.m[2][2]
                                   * (4 * dHat * dHat * pow(log(I2), 3) * (I1 - eps_x)
                                      * (I2 - 1) * (2 * I2 + I2 * log(I2) - 2))
                                   / (I2 * (eps_x * eps_x));
#elif (RANK == 6)
                double lambdag1g = -Kappa * 4 * c * F.m[2][2]
                                   * (4 * dHat * dHat * pow(log(I2), 5) * (I1 - eps_x)
                                      * (I2 - 1) * (3 * I2 + I2 * log(I2) - 3))
                                   / (I2 * (eps_x * eps_x));
#endif
                Eigen::Matrix2d FMat2;
                FMat2 << lambda10, lambdag1g, lambdag1g, lambda20;
                makePDGeneral_ipc<double, 2>(FMat2);
                projectedH.m[4][4] += FMat2(0, 0);
                projectedH.m[4][8] += FMat2(0, 1);
                projectedH.m[8][4] += FMat2(1, 0);
                projectedH.m[8][8] += FMat2(1, 1);

                //__GEIGEN__::Matrix9x12d PFPxTransPos = __GEIGEN__::__Transpose12x9(PFPx);
                __GEIGEN__::Matrix12x12d Hessian;  // = __GEIGEN__::__M12x9_M9x12_Multiply(__GEIGEN__::__M12x9_M9x9_Multiply(PFPx, projectedH), PFPxTransPos);
                __GEIGEN__::__M12x9_S9x9_MT9x12_Multiply(PFPx, projectedH, Hessian);
                int Hidx = matIndex[idx];  //int Hidx = atomicAdd(_cpNum + 4, 1);

                uint4 global_index =
                    make_uint4(MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);

                int triplet_id_offset = Hidx * M12_Off;
                write_triplet<12, 12>(
                    triplet_values, row_ids, col_ids, &(global_index.x), Hessian.m, triplet_id_offset);
            }
            else
            {
#ifdef NEWF
                double dis;
                _d_PE(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], dis);
                dis                               = sqrt(dis);
                double                 d_hat_sqrt = sqrt(dHat);
                __GEIGEN__::Matrix9x4d PFPxT;
                pFpx_pe2(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], d_hat_sqrt, PFPxT);
                double              I5 = pow(dis / d_hat_sqrt, 2);
                __GEIGEN__::Vector4 fnn;
                fnn.v[0] = fnn.v[1] = fnn.v[2] = 0;  // = fnn.v[3] = fnn.v[4] = 1;
                fnn.v[3] = dis / d_hat_sqrt;
                //__GEIGEN__::Vector4 flatten_pk1 = __GEIGEN__::__s_vec4_multiply(fnn, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);
                __GEIGEN__::Vector4 q0;
                q0.v[0] = q0.v[1] = q0.v[2] = 0;
                q0.v[3]                     = 1;
                __GEIGEN__::Matrix4x4d H;
                //__GEIGEN__::__init_Mat4x4_val(H, 0);
#if (RANK == 1)
                __GEIGEN__::Vector4 flatten_pk1 = __GEIGEN__::__s_vec4_multiply(
                    fnn, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);
#elif (RANK == 2)
                __GEIGEN__::Vector4 flatten_pk1 = __GEIGEN__::__s_vec4_multiply(
                    fnn,
                    2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1))
                        / I5);
#elif (RANK == 3)
                __GEIGEN__::Vector4 flatten_pk1 = __GEIGEN__::__s_vec4_multiply(
                    fnn,
                    -2
                        * (Kappa * dHat * dHat * log(I5) * log(I5) * (I5 - 1)
                           * (3 * I5 + 2 * I5 * log(I5) - 3))
                        / I5);
#elif (RANK == 4)
                __GEIGEN__::Vector4 flatten_pk1 = __GEIGEN__::__s_vec4_multiply(
                    fnn,
                    (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                     * (I5 - 1) * (2 * I5 + I5 * log(I5) - 2))
                        / I5);
#elif (RANK == 5)
                __GEIGEN__::Vector4 flatten_pk1 = __GEIGEN__::__s_vec4_multiply(
                    fnn,
                    -2
                        * (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                           * log(I5) * (I5 - 1) * (5 * I5 + 2 * I5 * log(I5) - 5))
                        / I5);
#elif (RANK == 6)
                __GEIGEN__::Vector4 flatten_pk1 = __GEIGEN__::__s_vec4_multiply(
                    fnn,
                    (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                     * log(I5) * log(I5) * (I5 - 1) * (3 * I5 + I5 * log(I5) - 3))
                        / I5);
#endif

                __GEIGEN__::Vector9 gradient_vec =
                    __GEIGEN__::__M9x4_v4_multiply(PFPxT, flatten_pk1);
#else

                double3 v0 = __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _vertexes[v0I]);
                double3 v1 = __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _vertexes[v0I]);


                __GEIGEN__::Matrix3x2d Ds;
                __GEIGEN__::__set_Mat3x2_val_column(Ds, v0, v1);

                double3 triangle_normal =
                    __GEIGEN__::__normalized(__GEIGEN__::__v_vec_cross(v0, v1));
                double3 target = make_double3(0, 1, 0);

                double3 vec = __GEIGEN__::__v_vec_cross(triangle_normal, target);
                double cos = __GEIGEN__::__v_vec_dot(triangle_normal, target);

                double3 edge_normal = __GEIGEN__::__normalized(__GEIGEN__::__v_vec_cross(
                    __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z]),
                    triangle_normal));
                double dis = __GEIGEN__::__v_vec_dot(
                    __GEIGEN__::__minus(_vertexes[v0I], _vertexes[MMCVIDI.y]), edge_normal);

                __GEIGEN__::Matrix3x3d rotation;
                __GEIGEN__::__set_Mat_val(rotation, 1, 0, 0, 0, 1, 0, 0, 0, 1);

                __GEIGEN__::Matrix9x4d PDmPx;

                if(cos + 1 == 0)
                {
                    rotation.m[0][0] = -1;
                    rotation.m[1][1] = -1;
                }
                else
                {
                    __GEIGEN__::Matrix3x3d cross_vec;
                    __GEIGEN__::__set_Mat_val(
                        cross_vec, 0, -vec.z, vec.y, vec.z, 0, -vec.x, -vec.y, vec.x, 0);

                    rotation = __GEIGEN__::__Mat_add(
                        rotation,
                        __GEIGEN__::__Mat_add(cross_vec,
                                              __GEIGEN__::__S_Mat_multiply(
                                                  __GEIGEN__::__M_Mat_multiply(cross_vec, cross_vec),
                                                  1.0 / (1 + cos))));
                }

                double3 pos0 = __GEIGEN__::__add(
                    _vertexes[v0I],
                    __GEIGEN__::__s_vec_multiply(edge_normal, dHat_sqrt - dis));

                double3 rotate_uv0 = __GEIGEN__::__M_v_multiply(rotation, pos0);
                double3 rotate_uv1 =
                    __GEIGEN__::__M_v_multiply(rotation, _vertexes[MMCVIDI.y]);
                double3 rotate_uv2 =
                    __GEIGEN__::__M_v_multiply(rotation, _vertexes[MMCVIDI.z]);
                double3 rotate_normal = __GEIGEN__::__M_v_multiply(rotation, edge_normal);

                double2 uv0    = make_double2(rotate_uv0.x, rotate_uv0.z);
                double2 uv1    = make_double2(rotate_uv1.x, rotate_uv1.z);
                double2 uv2    = make_double2(rotate_uv2.x, rotate_uv2.z);
                double2 normal = make_double2(rotate_normal.x, rotate_normal.z);

                double2 u0 = __GEIGEN__::__minus_v2(uv1, uv0);
                double2 u1 = __GEIGEN__::__minus_v2(uv2, uv0);

                __GEIGEN__::Matrix2x2d Dm;

                __GEIGEN__::__set_Mat2x2_val_column(Dm, u0, u1);

                __GEIGEN__::Matrix2x2d DmInv;
                __GEIGEN__::__Inverse2x2(Dm, DmInv);

                __GEIGEN__::Matrix3x2d F = __GEIGEN__::__M3x2_M2x2_Multiply(Ds, DmInv);

                double3 FxN = __GEIGEN__::__M3x2_v2_multiply(F, normal);
                double  I5  = __GEIGEN__::__squaredNorm(FxN);

                __GEIGEN__::Matrix3x2d fnn;

                __GEIGEN__::Matrix2x2d nn = __GEIGEN__::__v2_vec2_toMat2x2(normal, normal);

                fnn = __GEIGEN__::__M3x2_M2x2_Multiply(F, nn);

                __GEIGEN__::Vector6 tmp = __GEIGEN__::__Mat3x2_to_vec6_double(fnn);

#if (RANK == 1)
                __GEIGEN__::Vector6 flatten_pk1 = __GEIGEN__::__s_vec6_multiply(
                    tmp, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);
#elif (RANK == 2)
                __GEIGEN__::Vector6 flatten_pk1 = __GEIGEN__::__s_vec6_multiply(
                    tmp,
                    2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1))
                        / I5);
#elif (RANK == 3)
                __GEIGEN__::Vector6 flatten_pk1 = __GEIGEN__::__s_vec6_multiply(
                    tmp,
                    -2
                        * (Kappa * dHat * dHat * log(I5) * log(I5) * (I5 - 1)
                           * (3 * I5 + 2 * I5 * log(I5) - 3))
                        / I5);
#elif (RANK == 4)
                __GEIGEN__::Vector6 flatten_pk1 = __GEIGEN__::__s_vec6_multiply(
                    tmp,
                    (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                     * (I5 - 1) * (2 * I5 + I5 * log(I5) - 2))
                        / I5);
#elif (RANK == 5)
                __GEIGEN__::Vector6 flatten_pk1 = __GEIGEN__::__s_vec6_multiply(
                    tmp,
                    -2
                        * (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                           * log(I5) * (I5 - 1) * (5 * I5 + 2 * I5 * log(I5) - 5))
                        / I5);
#elif (RANK == 6)
                __GEIGEN__::Vector6 flatten_pk1 = __GEIGEN__::__s_vec6_multiply(
                    tmp,
                    (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                     * log(I5) * log(I5) * (I5 - 1) * (3 * I5 + I5 * log(I5) - 3))
                        / I5);
#endif

                __GEIGEN__::Matrix6x9d PFPx = __computePFDsPX3D_6x9_double(DmInv);

                __GEIGEN__::Vector9 gradient_vec =
                    __GEIGEN__::__M9x6_v6_multiply(__GEIGEN__::__Transpose6x9(PFPx), flatten_pk1);
#endif

                {
                    atomicAdd(&(_gradient[v0I].x), gradient_vec.v[0]);
                    atomicAdd(&(_gradient[v0I].y), gradient_vec.v[1]);
                    atomicAdd(&(_gradient[v0I].z), gradient_vec.v[2]);
                    atomicAdd(&(_gradient[MMCVIDI.y].x), gradient_vec.v[3]);
                    atomicAdd(&(_gradient[MMCVIDI.y].y), gradient_vec.v[4]);
                    atomicAdd(&(_gradient[MMCVIDI.y].z), gradient_vec.v[5]);
                    atomicAdd(&(_gradient[MMCVIDI.z].x), gradient_vec.v[6]);
                    atomicAdd(&(_gradient[MMCVIDI.z].y), gradient_vec.v[7]);
                    atomicAdd(&(_gradient[MMCVIDI.z].z), gradient_vec.v[8]);
                }

#if (RANK == 1)
                double lambda0 = Kappa
                                 * (2 * dHat * dHat
                                    * (6 * I5 + 2 * I5 * log(I5) - 7 * I5 * I5
                                       - 6 * I5 * I5 * log(I5) + 1))
                                 / I5;
                if(dis * dis < gassThreshold * dHat)
                {
                    double lambda1 =
                        Kappa
                        * (2 * dHat * dHat
                           * (6 * gassThreshold + 2 * gassThreshold * log(gassThreshold)
                              - 7 * gassThreshold * gassThreshold
                              - 6 * gassThreshold * gassThreshold * log(gassThreshold) + 1))
                        / gassThreshold;
                    lambda0 = lambda1;
                }
#elif (RANK == 2)
                double lambda0 =
                    -(4 * Kappa * dHat * dHat
                      * (4 * I5 + log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                         + 6 * I5 * log(I5) - 2 * I5 * I5
                         + I5 * log(I5) * log(I5) - 7 * I5 * I5 * log(I5) - 2))
                    / I5;
                if(dis * dis < gassThreshold * dHat)
                {
                    double lambda1 =
                        -(4 * Kappa * dHat * dHat
                          * (4 * gassThreshold + log(gassThreshold)
                             - 3 * gassThreshold * gassThreshold
                                   * log(gassThreshold) * log(gassThreshold)
                             + 6 * gassThreshold * log(gassThreshold) - 2 * gassThreshold * gassThreshold
                             + gassThreshold * log(gassThreshold) * log(gassThreshold)
                             - 7 * gassThreshold * gassThreshold * log(gassThreshold) - 2))
                        / gassThreshold;
                    lambda0 = lambda1;
                }
#elif (RANK == 3)
                double lambda0 =
                    (2 * Kappa * dHat * dHat * log(I5)
                     * (24 * I5 + 3 * log(I5) - 6 * I5 * I5 * log(I5) * log(I5)
                        + 18 * I5 * log(I5) - 12 * I5 * I5
                        + 2 * I5 * log(I5) * log(I5) - 21 * I5 * I5 * log(I5) - 12))
                    / I5;
#elif (RANK == 4)
                double lambda0 =
                    -(4 * Kappa * dHat * dHat * log(I5) * log(I5)
                      * (24 * I5 + 2 * log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                         + 12 * I5 * log(I5) - 12 * I5 * I5
                         + I5 * log(I5) * log(I5) - 14 * I5 * I5 * log(I5) - 12))
                    / I5;
#elif (RANK == 5)
                double lambda0 =
                    (2 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                     * (80 * I5 + 5 * log(I5) - 6 * I5 * I5 * log(I5) * log(I5)
                        + 30 * I5 * log(I5) - 40 * I5 * I5
                        + 2 * I5 * log(I5) * log(I5) - 35 * I5 * I5 * log(I5) - 40))
                    / I5;
#elif (RANK == 6)
                double lambda0 =
                    -(4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5)
                      * (60 * I5 + 3 * log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                         + 18 * I5 * log(I5) - 30 * I5 * I5
                         + I5 * log(I5) * log(I5) - 21 * I5 * I5 * log(I5) - 30))
                    / I5;
#endif


#ifdef NEWF
                H = __GEIGEN__::__S_Mat4x4_multiply(
                    __GEIGEN__::__v4_vec4_toMat4x4(q0, q0), lambda0);

                __GEIGEN__::Matrix9x9d Hessian;  // = __GEIGEN__::__M9x4_M4x9_Multiply(__GEIGEN__::__M9x4_M4x4_Multiply(PFPxT, H), __GEIGEN__::__Transpose9x4(PFPxT));
                __GEIGEN__::__M9x4_S4x4_MT4x9_Multiply(PFPxT, H, Hessian);
#else

                __GEIGEN__::Vector6 q0 = __GEIGEN__::__Mat3x2_to_vec6_double(fnn);

                q0 = __GEIGEN__::__s_vec6_multiply(q0, 1.0 / sqrt(I5));

                __GEIGEN__::Matrix6x6d H;
                __GEIGEN__::__init_Mat6x6(H, 0);

                H = __GEIGEN__::__S_Mat6x6_multiply(
                    __GEIGEN__::__v6_vec6_toMat6x6(q0, q0), lambda0);

                __GEIGEN__::Matrix9x6d PFPxTransPos = __GEIGEN__::__Transpose6x9(PFPx);
                __GEIGEN__::Matrix9x9d Hessian = __GEIGEN__::__M9x6_M6x9_Multiply(
                    __GEIGEN__::__M9x6_M6x6_Multiply(PFPxTransPos, H), PFPx);
#endif
                int Hidx = matIndex[idx];  //int Hidx = atomicAdd(_cpNum + 3, 1);

                //H9x9[Hidx]    = Hessian;

                uint3 global_index = make_uint3(v0I, MMCVIDI.y, MMCVIDI.z);

                //D3Index[Hidx] = global_index;

                int triplet_id_offset = Hidx * M9_Off + offset4 * M12_Off;
                write_triplet<9, 9>(
                    triplet_values, row_ids, col_ids, &(global_index.x), Hessian.m, triplet_id_offset);
            }
        }
        else
        {
#ifdef NEWF
            double dis;
            _d_PT(_vertexes[v0I],
                  _vertexes[MMCVIDI.y],
                  _vertexes[MMCVIDI.z],
                  _vertexes[MMCVIDI.w],
                  dis);
            dis                                = sqrt(dis);
            double                  d_hat_sqrt = sqrt(dHat);
            __GEIGEN__::Matrix12x9d PFPxT;
            pFpx_pt2(_vertexes[v0I],
                     _vertexes[MMCVIDI.y],
                     _vertexes[MMCVIDI.z],
                     _vertexes[MMCVIDI.w],
                     d_hat_sqrt,
                     PFPxT);
            double              I5 = pow(dis / d_hat_sqrt, 2);
            __GEIGEN__::Vector9 tmp;
            tmp.v[0] = tmp.v[1] = tmp.v[2] = tmp.v[3] = tmp.v[4] = tmp.v[5] =
                tmp.v[6] = tmp.v[7] = 0;
            tmp.v[8]                = dis / d_hat_sqrt;

            __GEIGEN__::Vector9 q0;
            q0.v[0] = q0.v[1] = q0.v[2] = q0.v[3] = q0.v[4] = q0.v[5] =
                q0.v[6] = q0.v[7] = 0;
            q0.v[8]               = 1;

            __GEIGEN__::Matrix9x9d H;
            //__GEIGEN__::__init_Mat9x9(H, 0);
#else
            double3 v0 = __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _vertexes[v0I]);
            double3 v1 = __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _vertexes[v0I]);
            double3 v2 = __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[v0I]);

            __GEIGEN__::Matrix3x3d Ds;
            __GEIGEN__::__set_Mat_val_column(Ds, v0, v1, v2);

            double3 normal = __GEIGEN__::__normalized(__GEIGEN__::__v_vec_cross(
                __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.y]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.y])));
            double  dis    = __GEIGEN__::__v_vec_dot(v0, normal);
            //if (abs(dis) > dHat_sqrt) return;
            __GEIGEN__::Matrix12x9d PDmPx;
            //bool is_flip = false;

            if(dis > 0)
            {
                //is_flip = true;
                normal = make_double3(-normal.x, -normal.y, -normal.z);
                //pDmpx_pt_flip(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dHat_sqrt, PDmPx);
                //printf("dHat_sqrt = %f,   dis = %f\n", dHat_sqrt, dis);
            }
            else
            {
                dis = -dis;
                //pDmpx_pt(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dHat_sqrt, PDmPx);
                //printf("dHat_sqrt = %f,   dis = %f\n", dHat_sqrt, dis);
            }

            double3 pos0 = __GEIGEN__::__add(
                _vertexes[v0I], __GEIGEN__::__s_vec_multiply(normal, dHat_sqrt - dis));


            double3 u0 = __GEIGEN__::__minus(_vertexes[MMCVIDI.y], pos0);
            double3 u1 = __GEIGEN__::__minus(_vertexes[MMCVIDI.z], pos0);
            double3 u2 = __GEIGEN__::__minus(_vertexes[MMCVIDI.w], pos0);

            __GEIGEN__::Matrix3x3d Dm, DmInv;
            __GEIGEN__::__set_Mat_val_column(Dm, u0, u1, u2);

            __GEIGEN__::__Inverse(Dm, DmInv);

            __GEIGEN__::Matrix3x3d F;  //, Ftest;
            __GEIGEN__::__M_Mat_multiply(Ds, DmInv, F);
            //__GEIGEN__::__M_Mat_multiply(Dm, DmInv, Ftest);

            double3 FxN = __GEIGEN__::__M_v_multiply(F, normal);
            double  I5  = __GEIGEN__::__squaredNorm(FxN);

            //printf("I5 = %f,   dist/dHat_sqrt = %f\n", I5, (dis / dHat_sqrt)* (dis / dHat_sqrt));


            __GEIGEN__::Matrix9x12d PFPx = __computePFDsPX3D_double(DmInv);

            __GEIGEN__::Matrix3x3d fnn;

            __GEIGEN__::Matrix3x3d nn = __GEIGEN__::__v_vec_toMat(normal, normal);

            __GEIGEN__::__M_Mat_multiply(F, nn, fnn);

            __GEIGEN__::Vector9 tmp = __GEIGEN__::__Mat3x3_to_vec9_double(fnn);
#endif
#if (RANK == 1)
            double lambda0 =
                Kappa
                * (2 * dHat * dHat
                   * (6 * I5 + 2 * I5 * log(I5) - 7 * I5 * I5 - 6 * I5 * I5 * log(I5) + 1))
                / I5;
            if(dis * dis < gassThreshold * dHat)
            {
                double lambda1 =
                    Kappa
                    * (2 * dHat * dHat
                       * (6 * gassThreshold + 2 * gassThreshold * log(gassThreshold)
                          - 7 * gassThreshold * gassThreshold
                          - 6 * gassThreshold * gassThreshold * log(gassThreshold) + 1))
                    / gassThreshold;
                lambda0 = lambda1;
            }
#elif (RANK == 2)
            double lambda0 =
                -(4 * Kappa * dHat * dHat
                  * (4 * I5 + log(I5) - 3 * I5 * I5 * log(I5) * log(I5) + 6 * I5 * log(I5)
                     - 2 * I5 * I5 + I5 * log(I5) * log(I5) - 7 * I5 * I5 * log(I5) - 2))
                / I5;
            if(dis * dis < gassThreshold * dHat)
            {
                double lambda1 =
                    -(4 * Kappa * dHat * dHat
                      * (4 * gassThreshold + log(gassThreshold)
                         - 3 * gassThreshold * gassThreshold * log(gassThreshold) * log(gassThreshold)
                         + 6 * gassThreshold * log(gassThreshold) - 2 * gassThreshold * gassThreshold
                         + gassThreshold * log(gassThreshold) * log(gassThreshold)
                         - 7 * gassThreshold * gassThreshold * log(gassThreshold) - 2))
                    / gassThreshold;
                lambda0 = lambda1;
            }
#elif (RANK == 3)
            double lambda0 =
                (2 * Kappa * dHat * dHat * log(I5)
                 * (24 * I5 + 3 * log(I5) - 6 * I5 * I5 * log(I5) * log(I5)
                    + 18 * I5 * log(I5) - 12 * I5 * I5
                    + 2 * I5 * log(I5) * log(I5) - 21 * I5 * I5 * log(I5) - 12))
                / I5;
#elif (RANK == 4)
            double lambda0 =
                -(4 * Kappa * dHat * dHat * log(I5) * log(I5)
                  * (24 * I5 + 2 * log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                     + 12 * I5 * log(I5) - 12 * I5 * I5 + I5 * log(I5) * log(I5)
                     - 14 * I5 * I5 * log(I5) - 12))
                / I5;
#elif (RANK == 5)
            double lambda0 =
                (2 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                 * (80 * I5 + 5 * log(I5) - 6 * I5 * I5 * log(I5) * log(I5)
                    + 30 * I5 * log(I5) - 40 * I5 * I5
                    + 2 * I5 * log(I5) * log(I5) - 35 * I5 * I5 * log(I5) - 40))
                / I5;
#elif (RANK == 6)
            double lambda0 =
                -(4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5)
                  * (60 * I5 + 3 * log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                     + 18 * I5 * log(I5) - 30 * I5 * I5 + I5 * log(I5) * log(I5)
                     - 21 * I5 * I5 * log(I5) - 30))
                / I5;
#endif

#if (RANK == 1)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);
#elif (RANK == 2)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5);
#elif (RANK == 3)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                -2
                    * (Kappa * dHat * dHat * log(I5) * log(I5) * (I5 - 1)
                       * (3 * I5 + 2 * I5 * log(I5) - 3))
                    / I5);
#elif (RANK == 4)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                 * (I5 - 1) * (2 * I5 + I5 * log(I5) - 2))
                    / I5);
#elif (RANK == 5)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                -2
                    * (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                       * log(I5) * (I5 - 1) * (5 * I5 + 2 * I5 * log(I5) - 5))
                    / I5);
#elif (RANK == 6)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5)
                 * log(I5) * (I5 - 1) * (3 * I5 + I5 * log(I5) - 3))
                    / I5);
#endif

#ifdef NEWF
            __GEIGEN__::Vector12 gradient_vec =
                __GEIGEN__::__M12x9_v9_multiply(PFPxT, flatten_pk1);
#else
            __GEIGEN__::Vector12 gradient_vec =
                __GEIGEN__::__M12x9_v9_multiply(__GEIGEN__::__Transpose9x12(PFPx), flatten_pk1);
#endif

            atomicAdd(&(_gradient[v0I].x), gradient_vec.v[0]);
            atomicAdd(&(_gradient[v0I].y), gradient_vec.v[1]);
            atomicAdd(&(_gradient[v0I].z), gradient_vec.v[2]);
            atomicAdd(&(_gradient[MMCVIDI.y].x), gradient_vec.v[3]);
            atomicAdd(&(_gradient[MMCVIDI.y].y), gradient_vec.v[4]);
            atomicAdd(&(_gradient[MMCVIDI.y].z), gradient_vec.v[5]);
            atomicAdd(&(_gradient[MMCVIDI.z].x), gradient_vec.v[6]);
            atomicAdd(&(_gradient[MMCVIDI.z].y), gradient_vec.v[7]);
            atomicAdd(&(_gradient[MMCVIDI.z].z), gradient_vec.v[8]);
            atomicAdd(&(_gradient[MMCVIDI.w].x), gradient_vec.v[9]);
            atomicAdd(&(_gradient[MMCVIDI.w].y), gradient_vec.v[10]);
            atomicAdd(&(_gradient[MMCVIDI.w].z), gradient_vec.v[11]);

#ifdef NEWF

            H = __GEIGEN__::__S_Mat9x9_multiply(__GEIGEN__::__v9_vec9_toMat9x9(q0, q0), lambda0);

            __GEIGEN__::Matrix12x12d Hessian;  // = __GEIGEN__::__M12x9_M9x12_Multiply(__GEIGEN__::__M12x9_M9x9_Multiply(PFPxT, H), __GEIGEN__::__Transpose12x9(PFPxT));
            __GEIGEN__::__M12x9_S9x9_MT9x12_Multiply(PFPxT, H, Hessian);
#else

            //__GEIGEN__::Matrix3x3d Q0;

            //__GEIGEN__::Matrix3x3d fnn;

            //__GEIGEN__::Matrix3x3d nn = __GEIGEN__::__v_vec_toMat(normal, normal);

            //__GEIGEN__::__M_Mat_multiply(F, nn, fnn);

            __GEIGEN__::Vector9 q0 = __GEIGEN__::__Mat3x3_to_vec9_double(fnn);

            q0 = __GEIGEN__::__s_vec9_multiply(q0, 1.0 / sqrt(I5));

            __GEIGEN__::Matrix9x9d H = __GEIGEN__::__S_Mat9x9_multiply(
                __GEIGEN__::__v9_vec9_toMat9x9(q0, q0), lambda0);

            __GEIGEN__::Matrix12x9d PFPxTransPos = __GEIGEN__::__Transpose9x12(PFPx);
            __GEIGEN__::Matrix12x12d Hessian = __GEIGEN__::__M12x9_M9x12_Multiply(
                __GEIGEN__::__M12x9_M9x9_Multiply(PFPxTransPos, H), PFPx);
#endif

            int Hidx = matIndex[idx];  //int Hidx = atomicAdd(_cpNum + 4, 1);

            //H12x12[Hidx]  = Hessian;
            uint4 global_index = make_uint4(v0I, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);
            //D4Index[Hidx]         = global_index;
            int triplet_id_offset = Hidx * M12_Off;
            write_triplet<12, 12>(
                triplet_values, row_ids, col_ids, &(global_index.x), Hessian.m, triplet_id_offset);
        }
    }
}



//=============================================================================
// Barrier Gradient Kernel
//=============================================================================

__global__ void _calBarrierGradient(const double3*    _vertexes,
                                    const double3*    _rest_vertexes,
                                    const int4* _collisionPair,
                                    double3*          _gradient,
                                    double            dHat,
                                    double            Kappa,
                                    int               number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    int4   MMCVIDI   = _collisionPair[idx];
    double dHat_sqrt = sqrt(dHat);
    //double dHat = dHat_sqrt * dHat_sqrt;
    //double Kappa = 1;
    if(MMCVIDI.x >= 0)
    {
        if(MMCVIDI.w >= 0)
        {
#ifdef NEWF
            double dis;
            _d_EE(_vertexes[MMCVIDI.x],
                  _vertexes[MMCVIDI.y],
                  _vertexes[MMCVIDI.z],
                  _vertexes[MMCVIDI.w],
                  dis);
            dis                                = sqrt(dis);
            double                  d_hat_sqrt = sqrt(dHat);
            __GEIGEN__::Matrix12x9d PFPxT;
            pFpx_ee2(_vertexes[MMCVIDI.x],
                     _vertexes[MMCVIDI.y],
                     _vertexes[MMCVIDI.z],
                     _vertexes[MMCVIDI.w],
                     d_hat_sqrt,
                     PFPxT);
            double              I5 = pow(dis / d_hat_sqrt, 2);
            __GEIGEN__::Vector9 tmp;
            tmp.v[0] = tmp.v[1] = tmp.v[2] = tmp.v[3] = tmp.v[4] = tmp.v[5] =
                tmp.v[6] = tmp.v[7] = 0;
            tmp.v[8]                = dis / d_hat_sqrt;
#else

            double3 v0 =
                __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _vertexes[MMCVIDI.x]);
            double3 v1 =
                __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.x]);
            double3 v2 =
                __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.x]);
            __GEIGEN__::Matrix3x3d Ds;
            __GEIGEN__::__set_Mat_val_column(Ds, v0, v1, v2);
            double3 normal = __GEIGEN__::__normalized(__GEIGEN__::__v_vec_cross(
                v0, __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.z])));
            double  dis    = __GEIGEN__::__v_vec_dot(v1, normal);
            if(dis < 0)
            {
                normal = make_double3(-normal.x, -normal.y, -normal.z);
                dis    = -dis;
            }

            double3 pos2 =
                __GEIGEN__::__add(_vertexes[MMCVIDI.z],
                                  __GEIGEN__::__s_vec_multiply(normal, dHat_sqrt - dis));
            double3 pos3 =
                __GEIGEN__::__add(_vertexes[MMCVIDI.w],
                                  __GEIGEN__::__s_vec_multiply(normal, dHat_sqrt - dis));

            double3 u0 = v0;
            double3 u1 = __GEIGEN__::__minus(pos2, _vertexes[MMCVIDI.x]);
            double3 u2 = __GEIGEN__::__minus(pos3, _vertexes[MMCVIDI.x]);

            __GEIGEN__::Matrix3x3d Dm, DmInv;
            __GEIGEN__::__set_Mat_val_column(Dm, u0, u1, u2);

            __GEIGEN__::__Inverse(Dm, DmInv);

            __GEIGEN__::Matrix3x3d F;
            __GEIGEN__::__M_Mat_multiply(Ds, DmInv, F);

            double3 FxN = __GEIGEN__::__M_v_multiply(F, normal);
            double  I5  = __GEIGEN__::__squaredNorm(FxN);

            __GEIGEN__::Matrix9x12d PFPx = __computePFDsPX3D_double(DmInv);

            __GEIGEN__::Matrix3x3d fnn;

            __GEIGEN__::Matrix3x3d nn = __GEIGEN__::__v_vec_toMat(normal, normal);

            __GEIGEN__::__M_Mat_multiply(F, nn, fnn);

            __GEIGEN__::Vector9 tmp = __GEIGEN__::__Mat3x3_to_vec9_double(fnn);

#endif

#if (RANK == 1)
            double judge =
                (2 * dHat * dHat
                 * (6 * I5 + 2 * I5 * log(I5) - 7 * I5 * I5 - 6 * I5 * I5 * log(I5) + 1))
                / I5;
            double judge2 = 2 * (dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1))
                            / I5 * dis / d_hat_sqrt;
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);
            //if (dis*dis<1e-2*dHat)
            //flatten_pk1 = __GEIGEN__::__s_vec9_multiply(tmp, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5 / (I5) /*/ (I5) / (I5)*/);
#elif (RANK == 2)
            //__GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(tmp, 2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5);

            double judge = -(4 * dHat * dHat
                             * (4 * I5 + log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                                + 6 * I5 * log(I5) - 2 * I5 * I5
                                + I5 * log(I5) * log(I5) - 7 * I5 * I5 * log(I5) - 2))
                           / I5;
            double judge2 =
                2 * (2 * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1))
                / I5 * dis / dHat_sqrt;
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5);
            //if (dis*dis<1e-2*dHat)
            //flatten_pk1 = __GEIGEN__::__s_vec9_multiply(tmp, 2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5/I5);

#elif (RANK == 3)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                -2
                    * (Kappa * dHat * dHat * log(I5) * log(I5) * (I5 - 1)
                       * (3 * I5 + 2 * I5 * log(I5) - 3))
                    / I5);
#elif (RANK == 4)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                 * (I5 - 1) * (2 * I5 + I5 * log(I5) - 2))
                    / I5);
#elif (RANK == 5)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                -2
                    * (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                       * log(I5) * (I5 - 1) * (5 * I5 + 2 * I5 * log(I5) - 5))
                    / I5);
#elif (RANK == 6)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5)
                 * log(I5) * (I5 - 1) * (3 * I5 + I5 * log(I5) - 3))
                    / I5);
#endif

#ifdef NEWF
            __GEIGEN__::Vector12 gradient_vec =
                __GEIGEN__::__M12x9_v9_multiply((PFPxT), flatten_pk1);
#else

            __GEIGEN__::Vector12 gradient_vec =
                __GEIGEN__::__M12x9_v9_multiply(__GEIGEN__::__Transpose9x12(PFPx), flatten_pk1);
#endif

            {
                atomicAdd(&(_gradient[MMCVIDI.x].x), gradient_vec.v[0]);
                atomicAdd(&(_gradient[MMCVIDI.x].y), gradient_vec.v[1]);
                atomicAdd(&(_gradient[MMCVIDI.x].z), gradient_vec.v[2]);
                atomicAdd(&(_gradient[MMCVIDI.y].x), gradient_vec.v[3]);
                atomicAdd(&(_gradient[MMCVIDI.y].y), gradient_vec.v[4]);
                atomicAdd(&(_gradient[MMCVIDI.y].z), gradient_vec.v[5]);
                atomicAdd(&(_gradient[MMCVIDI.z].x), gradient_vec.v[6]);
                atomicAdd(&(_gradient[MMCVIDI.z].y), gradient_vec.v[7]);
                atomicAdd(&(_gradient[MMCVIDI.z].z), gradient_vec.v[8]);
                atomicAdd(&(_gradient[MMCVIDI.w].x), gradient_vec.v[9]);
                atomicAdd(&(_gradient[MMCVIDI.w].y), gradient_vec.v[10]);
                atomicAdd(&(_gradient[MMCVIDI.w].z), gradient_vec.v[11]);
            }
        }
        else
        {
            //return;
            MMCVIDI.w = -MMCVIDI.w - 1;
            double3 v0 =
                __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _vertexes[MMCVIDI.x]);
            double3 v1 =
                __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.z]);
            double c = __GEIGEN__::__norm(__GEIGEN__::__v_vec_cross(v0, v1)) /*/ __GEIGEN__::__norm(v0)*/;
            double I1 = c * c;
            if(I1 == 0)
                return;
            double dis;
            _d_EE(_vertexes[MMCVIDI.x],
                  _vertexes[MMCVIDI.y],
                  _vertexes[MMCVIDI.z],
                  _vertexes[MMCVIDI.w],
                  dis);
            double I2 = dis / dHat;
            dis       = sqrt(dis);

            __GEIGEN__::Matrix3x3d F;
            __GEIGEN__::__set_Mat_val(F, 1, 0, 0, 0, c, 0, 0, 0, dis / dHat_sqrt);
            double3 n1 = make_double3(0, 1, 0);
            double3 n2 = make_double3(0, 0, 1);

            double eps_x = _compute_epx(_rest_vertexes[MMCVIDI.x],
                                        _rest_vertexes[MMCVIDI.y],
                                        _rest_vertexes[MMCVIDI.z],
                                        _rest_vertexes[MMCVIDI.w]);

            __GEIGEN__::Matrix3x3d g1, g2;

            __GEIGEN__::Matrix3x3d nn = __GEIGEN__::__v_vec_toMat(n1, n1);
            __GEIGEN__::__M_Mat_multiply(F, nn, g1);
            nn = __GEIGEN__::__v_vec_toMat(n2, n2);
            __GEIGEN__::__M_Mat_multiply(F, nn, g2);

            __GEIGEN__::Vector9 flatten_g1 = __GEIGEN__::__Mat3x3_to_vec9_double(g1);
            __GEIGEN__::Vector9 flatten_g2 = __GEIGEN__::__Mat3x3_to_vec9_double(g2);

            __GEIGEN__::Matrix12x9d PFPx;
            pFpx_pee(_vertexes[MMCVIDI.x],
                     _vertexes[MMCVIDI.y],
                     _vertexes[MMCVIDI.z],
                     _vertexes[MMCVIDI.w],
                     dHat_sqrt,
                     PFPx);


#if (RANK == 1)
            double p1 = Kappa * 2
                        * (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                        / (eps_x * eps_x);
            double p2 = Kappa * 2
                        * (I1 * dHat * dHat * (I1 - 2 * eps_x) * (I2 - 1)
                           * (I2 + 2 * I2 * log(I2) - 1))
                        / (I2 * eps_x * eps_x);
#elif (RANK == 2)
            double p1 = -Kappa * 2
                        * (2 * dHat * dHat * log(I2) * log(I2) * (I1 - eps_x)
                           * (I2 - 1) * (I2 - 1))
                        / (eps_x * eps_x);
            double p2 = -Kappa * 2
                        * (2 * I1 * dHat * dHat * log(I2) * (I1 - 2 * eps_x)
                           * (I2 - 1) * (I2 + I2 * log(I2) - 1))
                        / (I2 * (eps_x * eps_x));
#elif (RANK == 4)
            double p1 = -Kappa * 2
                        * (2 * dHat * dHat * pow(log(I2), 4) * (I1 - eps_x)
                           * (I2 - 1) * (I2 - 1))
                        / (eps_x * eps_x);
            double p2 = -Kappa * 2
                        * (2 * I1 * dHat * dHat * pow(log(I2), 3) * (I1 - 2 * eps_x)
                           * (I2 - 1) * (2 * I2 + I2 * log(I2) - 2))
                        / (I2 * (eps_x * eps_x));
#elif (RANK == 6)
            double p1 = -Kappa * 2
                        * (2 * dHat * dHat * pow(log(I2), 6) * (I1 - eps_x)
                           * (I2 - 1) * (I2 - 1))
                        / (eps_x * eps_x);
            double p2 = -Kappa * 2
                        * (2 * I1 * dHat * dHat * pow(log(I2), 5) * (I1 - 2 * eps_x)
                           * (I2 - 1) * (3 * I2 + I2 * log(I2) - 3))
                        / (I2 * (eps_x * eps_x));
#endif


            __GEIGEN__::Vector9 flatten_pk1 =
                __GEIGEN__::__add9(__GEIGEN__::__s_vec9_multiply(flatten_g1, p1),
                                   __GEIGEN__::__s_vec9_multiply(flatten_g2, p2));
            __GEIGEN__::Vector12 gradient_vec =
                __GEIGEN__::__M12x9_v9_multiply(PFPx, flatten_pk1);

            {
                atomicAdd(&(_gradient[MMCVIDI.x].x), gradient_vec.v[0]);
                atomicAdd(&(_gradient[MMCVIDI.x].y), gradient_vec.v[1]);
                atomicAdd(&(_gradient[MMCVIDI.x].z), gradient_vec.v[2]);
                atomicAdd(&(_gradient[MMCVIDI.y].x), gradient_vec.v[3]);
                atomicAdd(&(_gradient[MMCVIDI.y].y), gradient_vec.v[4]);
                atomicAdd(&(_gradient[MMCVIDI.y].z), gradient_vec.v[5]);
                atomicAdd(&(_gradient[MMCVIDI.z].x), gradient_vec.v[6]);
                atomicAdd(&(_gradient[MMCVIDI.z].y), gradient_vec.v[7]);
                atomicAdd(&(_gradient[MMCVIDI.z].z), gradient_vec.v[8]);
                atomicAdd(&(_gradient[MMCVIDI.w].x), gradient_vec.v[9]);
                atomicAdd(&(_gradient[MMCVIDI.w].y), gradient_vec.v[10]);
                atomicAdd(&(_gradient[MMCVIDI.w].z), gradient_vec.v[11]);
            }
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
                    return;
                double dis;
                _d_PP(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], dis);
                double I2 = dis / dHat;
                dis       = sqrt(dis);

                __GEIGEN__::Matrix3x3d F;
                __GEIGEN__::__set_Mat_val(F, 1, 0, 0, 0, c, 0, 0, 0, dis / dHat_sqrt);
                double3 n1 = make_double3(0, 1, 0);
                double3 n2 = make_double3(0, 0, 1);

                double eps_x = _compute_epx(_rest_vertexes[MMCVIDI.x],
                                            _rest_vertexes[MMCVIDI.z],
                                            _rest_vertexes[MMCVIDI.y],
                                            _rest_vertexes[MMCVIDI.w]);

                __GEIGEN__::Matrix3x3d g1, g2;

                __GEIGEN__::Matrix3x3d nn = __GEIGEN__::__v_vec_toMat(n1, n1);
                __GEIGEN__::__M_Mat_multiply(F, nn, g1);
                nn = __GEIGEN__::__v_vec_toMat(n2, n2);
                __GEIGEN__::__M_Mat_multiply(F, nn, g2);

                __GEIGEN__::Vector9 flatten_g1 = __GEIGEN__::__Mat3x3_to_vec9_double(g1);
                __GEIGEN__::Vector9 flatten_g2 = __GEIGEN__::__Mat3x3_to_vec9_double(g2);

                __GEIGEN__::Matrix12x9d PFPx;
                pFpx_ppp(_vertexes[MMCVIDI.x],
                         _vertexes[MMCVIDI.y],
                         _vertexes[MMCVIDI.z],
                         _vertexes[MMCVIDI.w],
                         dHat_sqrt,
                         PFPx);
#if (RANK == 1)
                double p1 =
                    Kappa * 2
                    * (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                    / (eps_x * eps_x);
                double p2 = Kappa * 2
                            * (I1 * dHat * dHat * (I1 - 2 * eps_x) * (I2 - 1)
                               * (I2 + 2 * I2 * log(I2) - 1))
                            / (I2 * eps_x * eps_x);
#elif (RANK == 2)
                double p1 = -Kappa * 2
                            * (2 * dHat * dHat * log(I2) * log(I2)
                               * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                            / (eps_x * eps_x);
                double p2 = -Kappa * 2
                            * (2 * I1 * dHat * dHat * log(I2) * (I1 - 2 * eps_x)
                               * (I2 - 1) * (I2 + I2 * log(I2) - 1))
                            / (I2 * (eps_x * eps_x));
#elif (RANK == 4)
                double p1 = -Kappa * 2
                            * (2 * dHat * dHat * pow(log(I2), 4) * (I1 - eps_x)
                               * (I2 - 1) * (I2 - 1))
                            / (eps_x * eps_x);
                double p2 = -Kappa * 2
                            * (2 * I1 * dHat * dHat * pow(log(I2), 3) * (I1 - 2 * eps_x)
                               * (I2 - 1) * (2 * I2 + I2 * log(I2) - 2))
                            / (I2 * (eps_x * eps_x));
#elif (RANK == 6)
                double p1 = -Kappa * 2
                            * (2 * dHat * dHat * pow(log(I2), 6) * (I1 - eps_x)
                               * (I2 - 1) * (I2 - 1))
                            / (eps_x * eps_x);
                double p2 = -Kappa * 2
                            * (2 * I1 * dHat * dHat * pow(log(I2), 5) * (I1 - 2 * eps_x)
                               * (I2 - 1) * (3 * I2 + I2 * log(I2) - 3))
                            / (I2 * (eps_x * eps_x));
#endif
                __GEIGEN__::Vector9 flatten_pk1 =
                    __GEIGEN__::__add9(__GEIGEN__::__s_vec9_multiply(flatten_g1, p1),
                                       __GEIGEN__::__s_vec9_multiply(flatten_g2, p2));
                __GEIGEN__::Vector12 gradient_vec =
                    __GEIGEN__::__M12x9_v9_multiply(PFPx, flatten_pk1);

                {
                    atomicAdd(&(_gradient[MMCVIDI.x].x), gradient_vec.v[0]);
                    atomicAdd(&(_gradient[MMCVIDI.x].y), gradient_vec.v[1]);
                    atomicAdd(&(_gradient[MMCVIDI.x].z), gradient_vec.v[2]);
                    atomicAdd(&(_gradient[MMCVIDI.y].x), gradient_vec.v[3]);
                    atomicAdd(&(_gradient[MMCVIDI.y].y), gradient_vec.v[4]);
                    atomicAdd(&(_gradient[MMCVIDI.y].z), gradient_vec.v[5]);
                    atomicAdd(&(_gradient[MMCVIDI.z].x), gradient_vec.v[6]);
                    atomicAdd(&(_gradient[MMCVIDI.z].y), gradient_vec.v[7]);
                    atomicAdd(&(_gradient[MMCVIDI.z].z), gradient_vec.v[8]);
                    atomicAdd(&(_gradient[MMCVIDI.w].x), gradient_vec.v[9]);
                    atomicAdd(&(_gradient[MMCVIDI.w].y), gradient_vec.v[10]);
                    atomicAdd(&(_gradient[MMCVIDI.w].z), gradient_vec.v[11]);
                }
            }
            else
            {
#ifdef NEWF
                double dis;
                _d_PP(_vertexes[v0I], _vertexes[MMCVIDI.y], dis);
                dis                            = sqrt(dis);
                double              d_hat_sqrt = sqrt(dHat);
                __GEIGEN__::Vector6 PFPxT;
                pFpx_pp2(_vertexes[v0I], _vertexes[MMCVIDI.y], d_hat_sqrt, PFPxT);
                double I5  = pow(dis / d_hat_sqrt, 2);
                double fnn = dis / d_hat_sqrt;

#if (RANK == 1)


                double judge = (2 * dHat * dHat
                                * (6 * I5 + 2 * I5 * log(I5) - 7 * I5 * I5
                                   - 6 * I5 * I5 * log(I5) + 1))
                               / I5;
                double judge2 =
                    2 * (dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1))
                    / I5 * dis / d_hat_sqrt;
                double flatten_pk1 =
                    fnn * 2 * Kappa
                    * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5;
                //if (dis*dis<1e-2*dHat)
                //flatten_pk1 = fnn * 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5 / (I5) /*/ (I5) / (I5)*/;
#elif (RANK == 2)
                //double flatten_pk1 = fnn * 2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5;

                double judge =
                    -(4 * dHat * dHat
                      * (4 * I5 + log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                         + 6 * I5 * log(I5) - 2 * I5 * I5
                         + I5 * log(I5) * log(I5) - 7 * I5 * I5 * log(I5) - 2))
                    / I5;
                double judge2 =
                    2 * (2 * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1))
                    / I5 * dis / dHat_sqrt;
                double flatten_pk1 = fnn * 2
                                     * (2 * Kappa * dHat * dHat * log(I5)
                                        * (I5 - 1) * (I5 + I5 * log(I5) - 1))
                                     / I5;
                //if (dis*dis<1e-2*dHat)
                //flatten_pk1 = fnn * 2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5/I5;

#elif (RANK == 3)
                double flatten_pk1 = fnn * -2
                                     * (Kappa * dHat * dHat * log(I5) * log(I5)
                                        * (I5 - 1) * (3 * I5 + 2 * I5 * log(I5) - 3))
                                     / I5;
#elif (RANK == 4)
                double flatten_pk1 =
                    fnn
                    * (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                       * (I5 - 1) * (2 * I5 + I5 * log(I5) - 2))
                    / I5;
#elif (RANK == 5)
                double flatten_pk1 =
                    fnn * -2
                    * (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                       * log(I5) * (I5 - 1) * (5 * I5 + 2 * I5 * log(I5) - 5))
                    / I5;
#elif (RANK == 6)
                double flatten_pk1 =
                    fnn
                    * (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                       * log(I5) * log(I5) * (I5 - 1) * (3 * I5 + I5 * log(I5) - 3))
                    / I5;
#endif

                __GEIGEN__::Vector6 gradient_vec =
                    __GEIGEN__::__s_vec6_multiply(PFPxT, flatten_pk1);

#else
                double3 v0 = __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _vertexes[v0I]);
                double3 Ds  = v0;
                double  dis = __GEIGEN__::__norm(v0);
                //if (dis > dHat_sqrt) return;
                double3 vec_normal =
                    __GEIGEN__::__normalized(make_double3(-v0.x, -v0.y, -v0.z));
                double3 target = make_double3(0, 1, 0);
                double3 vec    = __GEIGEN__::__v_vec_cross(vec_normal, target);
                double  cos    = __GEIGEN__::__v_vec_dot(vec_normal, target);
                __GEIGEN__::Matrix3x3d rotation;
                __GEIGEN__::__set_Mat_val(rotation, 1, 0, 0, 0, 1, 0, 0, 0, 1);
                __GEIGEN__::Vector6 PDmPx;
                if(cos + 1 == 0)
                {
                    rotation.m[0][0] = -1;
                    rotation.m[1][1] = -1;
                }
                else
                {
                    __GEIGEN__::Matrix3x3d cross_vec;
                    __GEIGEN__::__set_Mat_val(
                        cross_vec, 0, -vec.z, vec.y, vec.z, 0, -vec.x, -vec.y, vec.x, 0);

                    rotation = __GEIGEN__::__Mat_add(
                        rotation,
                        __GEIGEN__::__Mat_add(cross_vec,
                                              __GEIGEN__::__S_Mat_multiply(
                                                  __GEIGEN__::__M_Mat_multiply(cross_vec, cross_vec),
                                                  1.0 / (1 + cos))));
                }

                double3 pos0 = __GEIGEN__::__add(
                    _vertexes[v0I],
                    __GEIGEN__::__s_vec_multiply(vec_normal, dHat_sqrt - dis));
                double3 rotate_uv0 = __GEIGEN__::__M_v_multiply(rotation, pos0);
                double3 rotate_uv1 =
                    __GEIGEN__::__M_v_multiply(rotation, _vertexes[MMCVIDI.y]);

                double uv0 = rotate_uv0.y;
                double uv1 = rotate_uv1.y;

                double u0    = uv1 - uv0;
                double Dm    = u0;  //PFPx
                double DmInv = 1 / u0;

                double3 F  = __GEIGEN__::__s_vec_multiply(Ds, DmInv);
                double  I5 = __GEIGEN__::__squaredNorm(F);

                double3 tmp = F;

#if (RANK == 1)
                double3 flatten_pk1 = __GEIGEN__::__s_vec_multiply(
                    tmp, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);
#elif (RANK == 2)
                double3 flatten_pk1 = __GEIGEN__::__s_vec_multiply(
                    tmp,
                    2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1))
                        / I5);
#elif (RANK == 3)
                double3 flatten_pk1 = __GEIGEN__::__s_vec_multiply(
                    tmp,
                    -2
                        * (Kappa * dHat * dHat * log(I5) * log(I5) * (I5 - 1)
                           * (3 * I5 + 2 * I5 * log(I5) - 3))
                        / I5);
#elif (RANK == 4)
                double3 flatten_pk1 = __GEIGEN__::__s_vec_multiply(
                    tmp,
                    (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                     * (I5 - 1) * (2 * I5 + I5 * log(I5) - 2))
                        / I5);
#elif (RANK == 5)
                double3 flatten_pk1 = __GEIGEN__::__s_vec_multiply(
                    tmp,
                    -2
                        * (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                           * log(I5) * (I5 - 1) * (5 * I5 + 2 * I5 * log(I5) - 5))
                        / I5);
#elif (RANK == 6)
                double3 flatten_pk1 = __GEIGEN__::__s_vec_multiply(
                    tmp,
                    (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                     * log(I5) * log(I5) * (I5 - 1) * (3 * I5 + I5 * log(I5) - 3))
                        / I5);
#endif
                __GEIGEN__::Matrix3x6d PFPx = __computePFDsPX3D_3x6_double(DmInv);

                __GEIGEN__::Vector6 gradient_vec =
                    __GEIGEN__::__M6x3_v3_multiply(__GEIGEN__::__Transpose3x6(PFPx), flatten_pk1);
#endif


                {
                    atomicAdd(&(_gradient[v0I].x), gradient_vec.v[0]);
                    atomicAdd(&(_gradient[v0I].y), gradient_vec.v[1]);
                    atomicAdd(&(_gradient[v0I].z), gradient_vec.v[2]);
                    atomicAdd(&(_gradient[MMCVIDI.y].x), gradient_vec.v[3]);
                    atomicAdd(&(_gradient[MMCVIDI.y].y), gradient_vec.v[4]);
                    atomicAdd(&(_gradient[MMCVIDI.y].z), gradient_vec.v[5]);
                }
            }
        }
        else if(MMCVIDI.w < 0)
        {
            if(MMCVIDI.y < 0)
            {
                MMCVIDI.y = -MMCVIDI.y - 1;
                MMCVIDI.x = v0I;
                MMCVIDI.w = -MMCVIDI.w - 1;
                double3 v0 =
                    __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.x]);
                double3 v1 =
                    __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.y]);
                double c = __GEIGEN__::__norm(__GEIGEN__::__v_vec_cross(v0, v1)) /*/ __GEIGEN__::__norm(v0)*/;
                double I1 = c * c;
                if(I1 == 0)
                    return;
                double dis;
                _d_PE(_vertexes[MMCVIDI.x],
                      _vertexes[MMCVIDI.y],
                      _vertexes[MMCVIDI.z],
                      dis);
                double I2 = dis / dHat;
                dis       = sqrt(dis);

                __GEIGEN__::Matrix3x3d F;
                __GEIGEN__::__set_Mat_val(F, 1, 0, 0, 0, c, 0, 0, 0, dis / dHat_sqrt);
                double3 n1 = make_double3(0, 1, 0);
                double3 n2 = make_double3(0, 0, 1);

                double eps_x = _compute_epx(_rest_vertexes[MMCVIDI.x],
                                            _rest_vertexes[MMCVIDI.w],
                                            _rest_vertexes[MMCVIDI.y],
                                            _rest_vertexes[MMCVIDI.z]);

                __GEIGEN__::Matrix3x3d g1, g2;

                __GEIGEN__::Matrix3x3d nn = __GEIGEN__::__v_vec_toMat(n1, n1);
                __GEIGEN__::__M_Mat_multiply(F, nn, g1);
                nn = __GEIGEN__::__v_vec_toMat(n2, n2);
                __GEIGEN__::__M_Mat_multiply(F, nn, g2);

                __GEIGEN__::Vector9 flatten_g1 = __GEIGEN__::__Mat3x3_to_vec9_double(g1);
                __GEIGEN__::Vector9 flatten_g2 = __GEIGEN__::__Mat3x3_to_vec9_double(g2);

                __GEIGEN__::Matrix12x9d PFPx;
                pFpx_ppe(_vertexes[MMCVIDI.x],
                         _vertexes[MMCVIDI.y],
                         _vertexes[MMCVIDI.z],
                         _vertexes[MMCVIDI.w],
                         dHat_sqrt,
                         PFPx);

#if (RANK == 1)
                double p1 =
                    Kappa * 2
                    * (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                    / (eps_x * eps_x);
                double p2 = Kappa * 2
                            * (I1 * dHat * dHat * (I1 - 2 * eps_x) * (I2 - 1)
                               * (I2 + 2 * I2 * log(I2) - 1))
                            / (I2 * eps_x * eps_x);
#elif (RANK == 2)
                double p1 = -Kappa * 2
                            * (2 * dHat * dHat * log(I2) * log(I2)
                               * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                            / (eps_x * eps_x);
                double p2 = -Kappa * 2
                            * (2 * I1 * dHat * dHat * log(I2) * (I1 - 2 * eps_x)
                               * (I2 - 1) * (I2 + I2 * log(I2) - 1))
                            / (I2 * (eps_x * eps_x));
#elif (RANK == 4)
                double p1 = -Kappa * 2
                            * (2 * dHat * dHat * pow(log(I2), 4) * (I1 - eps_x)
                               * (I2 - 1) * (I2 - 1))
                            / (eps_x * eps_x);
                double p2 = -Kappa * 2
                            * (2 * I1 * dHat * dHat * pow(log(I2), 3) * (I1 - 2 * eps_x)
                               * (I2 - 1) * (2 * I2 + I2 * log(I2) - 2))
                            / (I2 * (eps_x * eps_x));
#elif (RANK == 6)
                double p1 = -Kappa * 2
                            * (2 * dHat * dHat * pow(log(I2), 6) * (I1 - eps_x)
                               * (I2 - 1) * (I2 - 1))
                            / (eps_x * eps_x);
                double p2 = -Kappa * 2
                            * (2 * I1 * dHat * dHat * pow(log(I2), 5) * (I1 - 2 * eps_x)
                               * (I2 - 1) * (3 * I2 + I2 * log(I2) - 3))
                            / (I2 * (eps_x * eps_x));
#endif
                __GEIGEN__::Vector9 flatten_pk1 =
                    __GEIGEN__::__add9(__GEIGEN__::__s_vec9_multiply(flatten_g1, p1),
                                       __GEIGEN__::__s_vec9_multiply(flatten_g2, p2));
                __GEIGEN__::Vector12 gradient_vec =
                    __GEIGEN__::__M12x9_v9_multiply(PFPx, flatten_pk1);

                {
                    atomicAdd(&(_gradient[MMCVIDI.x].x), gradient_vec.v[0]);
                    atomicAdd(&(_gradient[MMCVIDI.x].y), gradient_vec.v[1]);
                    atomicAdd(&(_gradient[MMCVIDI.x].z), gradient_vec.v[2]);
                    atomicAdd(&(_gradient[MMCVIDI.y].x), gradient_vec.v[3]);
                    atomicAdd(&(_gradient[MMCVIDI.y].y), gradient_vec.v[4]);
                    atomicAdd(&(_gradient[MMCVIDI.y].z), gradient_vec.v[5]);
                    atomicAdd(&(_gradient[MMCVIDI.z].x), gradient_vec.v[6]);
                    atomicAdd(&(_gradient[MMCVIDI.z].y), gradient_vec.v[7]);
                    atomicAdd(&(_gradient[MMCVIDI.z].z), gradient_vec.v[8]);
                    atomicAdd(&(_gradient[MMCVIDI.w].x), gradient_vec.v[9]);
                    atomicAdd(&(_gradient[MMCVIDI.w].y), gradient_vec.v[10]);
                    atomicAdd(&(_gradient[MMCVIDI.w].z), gradient_vec.v[11]);
                }
            }
            else
            {
#ifdef NEWF
                double dis;
                _d_PE(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], dis);
                dis                               = sqrt(dis);
                double                 d_hat_sqrt = sqrt(dHat);
                __GEIGEN__::Matrix9x4d PFPxT;
                pFpx_pe2(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], d_hat_sqrt, PFPxT);
                double              I5 = pow(dis / d_hat_sqrt, 2);
                __GEIGEN__::Vector4 fnn;
                fnn.v[0] = fnn.v[1] = fnn.v[2] = 0;  // = fnn.v[3] = fnn.v[4] = 1;
                fnn.v[3] = dis / d_hat_sqrt;
                //__GEIGEN__::Vector4 flatten_pk1 = __GEIGEN__::__s_vec4_multiply(fnn, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);

#if (RANK == 1)


                double judge = (2 * dHat * dHat
                                * (6 * I5 + 2 * I5 * log(I5) - 7 * I5 * I5
                                   - 6 * I5 * I5 * log(I5) + 1))
                               / I5;
                double judge2 =
                    2 * (dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1))
                    / I5 * dis / d_hat_sqrt;
                __GEIGEN__::Vector4 flatten_pk1 = __GEIGEN__::__s_vec4_multiply(
                    fnn, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);
                //if (dis*dis<1e-2*dHat)
                //flatten_pk1 = __GEIGEN__::__s_vec4_multiply(fnn, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5 / (I5) /*/ (I5) / (I5)*/);

#elif (RANK == 2)
                //__GEIGEN__::Vector4 flatten_pk1 = __GEIGEN__::__s_vec4_multiply(fnn, 2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5);

                double judge =
                    -(4 * dHat * dHat
                      * (4 * I5 + log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                         + 6 * I5 * log(I5) - 2 * I5 * I5
                         + I5 * log(I5) * log(I5) - 7 * I5 * I5 * log(I5) - 2))
                    / I5;
                double judge2 =
                    2 * (2 * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1))
                    / I5 * dis / dHat_sqrt;
                __GEIGEN__::Vector4 flatten_pk1 = __GEIGEN__::__s_vec4_multiply(
                    fnn,
                    2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1))
                        / I5);
                //if (dis*dis<1e-2*dHat)
                //flatten_pk1 = __GEIGEN__::__s_vec4_multiply(fnn, 2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5/I5);
#elif (RANK == 3)
                __GEIGEN__::Vector4 flatten_pk1 = __GEIGEN__::__s_vec4_multiply(
                    fnn,
                    -2
                        * (Kappa * dHat * dHat * log(I5) * log(I5) * (I5 - 1)
                           * (3 * I5 + 2 * I5 * log(I5) - 3))
                        / I5);
#elif (RANK == 4)
                __GEIGEN__::Vector4 flatten_pk1 = __GEIGEN__::__s_vec4_multiply(
                    fnn,
                    (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                     * (I5 - 1) * (2 * I5 + I5 * log(I5) - 2))
                        / I5);
#elif (RANK == 5)
                __GEIGEN__::Vector4 flatten_pk1 = __GEIGEN__::__s_vec4_multiply(
                    fnn,
                    -2
                        * (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                           * log(I5) * (I5 - 1) * (5 * I5 + 2 * I5 * log(I5) - 5))
                        / I5);
#elif (RANK == 6)
                __GEIGEN__::Vector4 flatten_pk1 = __GEIGEN__::__s_vec4_multiply(
                    fnn,
                    (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                     * log(I5) * log(I5) * (I5 - 1) * (3 * I5 + I5 * log(I5) - 3))
                        / I5);
#endif

                __GEIGEN__::Vector9 gradient_vec =
                    __GEIGEN__::__M9x4_v4_multiply(PFPxT, flatten_pk1);
#else

                double3 v0 = __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _vertexes[v0I]);
                double3 v1 = __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _vertexes[v0I]);


                __GEIGEN__::Matrix3x2d Ds;
                __GEIGEN__::__set_Mat3x2_val_column(Ds, v0, v1);

                double3 triangle_normal =
                    __GEIGEN__::__normalized(__GEIGEN__::__v_vec_cross(v0, v1));
                double3 target = make_double3(0, 1, 0);

                double3 vec = __GEIGEN__::__v_vec_cross(triangle_normal, target);
                double cos = __GEIGEN__::__v_vec_dot(triangle_normal, target);

                double3 edge_normal = __GEIGEN__::__normalized(__GEIGEN__::__v_vec_cross(
                    __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z]),
                    triangle_normal));
                double dis = __GEIGEN__::__v_vec_dot(
                    __GEIGEN__::__minus(_vertexes[v0I], _vertexes[MMCVIDI.y]), edge_normal);

                __GEIGEN__::Matrix3x3d rotation;
                __GEIGEN__::__set_Mat_val(rotation, 1, 0, 0, 0, 1, 0, 0, 0, 1);

                __GEIGEN__::Matrix9x4d PDmPx;

                if(cos + 1 == 0)
                {
                    rotation.m[0][0] = -1;
                    rotation.m[1][1] = -1;
                }
                else
                {
                    __GEIGEN__::Matrix3x3d cross_vec;
                    __GEIGEN__::__set_Mat_val(
                        cross_vec, 0, -vec.z, vec.y, vec.z, 0, -vec.x, -vec.y, vec.x, 0);

                    rotation = __GEIGEN__::__Mat_add(
                        rotation,
                        __GEIGEN__::__Mat_add(cross_vec,
                                              __GEIGEN__::__S_Mat_multiply(
                                                  __GEIGEN__::__M_Mat_multiply(cross_vec, cross_vec),
                                                  1.0 / (1 + cos))));
                }

                double3 pos0 = __GEIGEN__::__add(
                    _vertexes[v0I],
                    __GEIGEN__::__s_vec_multiply(edge_normal, dHat_sqrt - dis));

                double3 rotate_uv0 = __GEIGEN__::__M_v_multiply(rotation, pos0);
                double3 rotate_uv1 =
                    __GEIGEN__::__M_v_multiply(rotation, _vertexes[MMCVIDI.y]);
                double3 rotate_uv2 =
                    __GEIGEN__::__M_v_multiply(rotation, _vertexes[MMCVIDI.z]);
                double3 rotate_normal = __GEIGEN__::__M_v_multiply(rotation, edge_normal);

                double2 uv0    = make_double2(rotate_uv0.x, rotate_uv0.z);
                double2 uv1    = make_double2(rotate_uv1.x, rotate_uv1.z);
                double2 uv2    = make_double2(rotate_uv2.x, rotate_uv2.z);
                double2 normal = make_double2(rotate_normal.x, rotate_normal.z);

                double2 u0 = __GEIGEN__::__minus_v2(uv1, uv0);
                double2 u1 = __GEIGEN__::__minus_v2(uv2, uv0);

                __GEIGEN__::Matrix2x2d Dm;

                __GEIGEN__::__set_Mat2x2_val_column(Dm, u0, u1);

                __GEIGEN__::Matrix2x2d DmInv;
                __GEIGEN__::__Inverse2x2(Dm, DmInv);

                __GEIGEN__::Matrix3x2d F = __GEIGEN__::__M3x2_M2x2_Multiply(Ds, DmInv);

                double3 FxN = __GEIGEN__::__M3x2_v2_multiply(F, normal);
                double  I5  = __GEIGEN__::__squaredNorm(FxN);

                __GEIGEN__::Matrix3x2d fnn;

                __GEIGEN__::Matrix2x2d nn = __GEIGEN__::__v2_vec2_toMat2x2(normal, normal);

                fnn = __GEIGEN__::__M3x2_M2x2_Multiply(F, nn);

                __GEIGEN__::Vector6 tmp = __GEIGEN__::__Mat3x2_to_vec6_double(fnn);


#if (RANK == 1)
                __GEIGEN__::Vector6 flatten_pk1 = __GEIGEN__::__s_vec6_multiply(
                    tmp, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);
#elif (RANK == 2)
                __GEIGEN__::Vector6 flatten_pk1 = __GEIGEN__::__s_vec6_multiply(
                    tmp,
                    2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1))
                        / I5);
#elif (RANK == 3)
                __GEIGEN__::Vector6 flatten_pk1 = __GEIGEN__::__s_vec6_multiply(
                    tmp,
                    -2
                        * (Kappa * dHat * dHat * log(I5) * log(I5) * (I5 - 1)
                           * (3 * I5 + 2 * I5 * log(I5) - 3))
                        / I5);
#elif (RANK == 4)
                __GEIGEN__::Vector6 flatten_pk1 = __GEIGEN__::__s_vec6_multiply(
                    tmp,
                    (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                     * (I5 - 1) * (2 * I5 + I5 * log(I5) - 2))
                        / I5);
#elif (RANK == 5)
                __GEIGEN__::Vector6 flatten_pk1 = __GEIGEN__::__s_vec6_multiply(
                    tmp,
                    -2
                        * (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                           * log(I5) * (I5 - 1) * (5 * I5 + 2 * I5 * log(I5) - 5))
                        / I5);
#elif (RANK == 6)
                __GEIGEN__::Vector6 flatten_pk1 = __GEIGEN__::__s_vec6_multiply(
                    tmp,
                    (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                     * log(I5) * log(I5) * (I5 - 1) * (3 * I5 + I5 * log(I5) - 3))
                        / I5);
#endif

                __GEIGEN__::Matrix6x9d PFPx = __computePFDsPX3D_6x9_double(DmInv);

                __GEIGEN__::Vector9 gradient_vec =
                    __GEIGEN__::__M9x6_v6_multiply(__GEIGEN__::__Transpose6x9(PFPx), flatten_pk1);
#endif

                {
                    atomicAdd(&(_gradient[v0I].x), gradient_vec.v[0]);
                    atomicAdd(&(_gradient[v0I].y), gradient_vec.v[1]);
                    atomicAdd(&(_gradient[v0I].z), gradient_vec.v[2]);
                    atomicAdd(&(_gradient[MMCVIDI.y].x), gradient_vec.v[3]);
                    atomicAdd(&(_gradient[MMCVIDI.y].y), gradient_vec.v[4]);
                    atomicAdd(&(_gradient[MMCVIDI.y].z), gradient_vec.v[5]);
                    atomicAdd(&(_gradient[MMCVIDI.z].x), gradient_vec.v[6]);
                    atomicAdd(&(_gradient[MMCVIDI.z].y), gradient_vec.v[7]);
                    atomicAdd(&(_gradient[MMCVIDI.z].z), gradient_vec.v[8]);
                }
            }
        }
        else
        {
#ifdef NEWF
            double dis;
            _d_PT(_vertexes[v0I],
                  _vertexes[MMCVIDI.y],
                  _vertexes[MMCVIDI.z],
                  _vertexes[MMCVIDI.w],
                  dis);
            dis                                = sqrt(dis);
            double                  d_hat_sqrt = sqrt(dHat);
            __GEIGEN__::Matrix12x9d PFPxT;
            pFpx_pt2(_vertexes[v0I],
                     _vertexes[MMCVIDI.y],
                     _vertexes[MMCVIDI.z],
                     _vertexes[MMCVIDI.w],
                     d_hat_sqrt,
                     PFPxT);
            double              I5 = pow(dis / d_hat_sqrt, 2);
            __GEIGEN__::Vector9 tmp;
            tmp.v[0] = tmp.v[1] = tmp.v[2] = tmp.v[3] = tmp.v[4] = tmp.v[5] =
                tmp.v[6] = tmp.v[7] = 0;
            tmp.v[8]                = dis / d_hat_sqrt;
#else
            double3 v0 = __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _vertexes[v0I]);
            double3 v1 = __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _vertexes[v0I]);
            double3 v2 = __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[v0I]);

            __GEIGEN__::Matrix3x3d Ds;
            __GEIGEN__::__set_Mat_val_column(Ds, v0, v1, v2);

            double3 normal = __GEIGEN__::__normalized(__GEIGEN__::__v_vec_cross(
                __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.y]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.y])));
            double  dis    = __GEIGEN__::__v_vec_dot(v0, normal);
            //if (abs(dis) > dHat_sqrt) return;
            __GEIGEN__::Matrix12x9d PDmPx;
            //bool is_flip = false;

            if(dis > 0)
            {
                //is_flip = true;
                normal = make_double3(-normal.x, -normal.y, -normal.z);
                //pDmpx_pt_flip(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dHat_sqrt, PDmPx);
                //printf("dHat_sqrt = %f,   dis = %f\n", dHat_sqrt, dis);
            }
            else
            {
                dis = -dis;
                //pDmpx_pt(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dHat_sqrt, PDmPx);
                //printf("dHat_sqrt = %f,   dis = %f\n", dHat_sqrt, dis);
            }

            double3 pos0 = __GEIGEN__::__add(
                _vertexes[v0I], __GEIGEN__::__s_vec_multiply(normal, dHat_sqrt - dis));


            double3 u0 = __GEIGEN__::__minus(_vertexes[MMCVIDI.y], pos0);
            double3 u1 = __GEIGEN__::__minus(_vertexes[MMCVIDI.z], pos0);
            double3 u2 = __GEIGEN__::__minus(_vertexes[MMCVIDI.w], pos0);

            __GEIGEN__::Matrix3x3d Dm, DmInv;
            __GEIGEN__::__set_Mat_val_column(Dm, u0, u1, u2);

            __GEIGEN__::__Inverse(Dm, DmInv);

            __GEIGEN__::Matrix3x3d F;  //, Ftest;
            __GEIGEN__::__M_Mat_multiply(Ds, DmInv, F);
            //__GEIGEN__::__M_Mat_multiply(Dm, DmInv, Ftest);

            double3 FxN = __GEIGEN__::__M_v_multiply(F, normal);
            double  I5  = __GEIGEN__::__squaredNorm(FxN);

            //printf("I5 = %f,   dist/dHat_sqrt = %f\n", I5, (dis / dHat_sqrt)* (dis / dHat_sqrt));


            __GEIGEN__::Matrix9x12d PFPx = __computePFDsPX3D_double(DmInv);

            __GEIGEN__::Matrix3x3d fnn;

            __GEIGEN__::Matrix3x3d nn = __GEIGEN__::__v_vec_toMat(normal, normal);

            __GEIGEN__::__M_Mat_multiply(F, nn, fnn);

            __GEIGEN__::Vector9 tmp = __GEIGEN__::__Mat3x3_to_vec9_double(fnn);
#endif


#if (RANK == 1)


            double judge =
                (2 * dHat * dHat
                 * (6 * I5 + 2 * I5 * log(I5) - 7 * I5 * I5 - 6 * I5 * I5 * log(I5) + 1))
                / I5;
            double judge2 = 2 * (dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1))
                            / I5 * dis / d_hat_sqrt;
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);
            //if (dis*dis<1e-2*dHat)
            //flatten_pk1 = __GEIGEN__::__s_vec9_multiply(tmp, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5 / (I5) /*/ (I5) / (I5)*/);

#elif (RANK == 2)
            //__GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(tmp, 2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5);

            double judge = -(4 * dHat * dHat
                             * (4 * I5 + log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                                + 6 * I5 * log(I5) - 2 * I5 * I5
                                + I5 * log(I5) * log(I5) - 7 * I5 * I5 * log(I5) - 2))
                           / I5;
            double judge2 =
                2 * (2 * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1))
                / I5 * dis / dHat_sqrt;
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5);
            //if (dis*dis<1e-2*dHat)
            //flatten_pk1 = __GEIGEN__::__s_vec9_multiply(tmp, 2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5/I5);
#elif (RANK == 3)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                -2
                    * (Kappa * dHat * dHat * log(I5) * log(I5) * (I5 - 1)
                       * (3 * I5 + 2 * I5 * log(I5) - 3))
                    / I5);
#elif (RANK == 4)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                 * (I5 - 1) * (2 * I5 + I5 * log(I5) - 2))
                    / I5);
#elif (RANK == 5)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                -2
                    * (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                       * log(I5) * (I5 - 1) * (5 * I5 + 2 * I5 * log(I5) - 5))
                    / I5);
#elif (RANK == 6)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5)
                 * log(I5) * (I5 - 1) * (3 * I5 + I5 * log(I5) - 3))
                    / I5);
#endif

#ifdef NEWF
            __GEIGEN__::Vector12 gradient_vec =
                __GEIGEN__::__M12x9_v9_multiply(PFPxT, flatten_pk1);
#else
            __GEIGEN__::Vector12 gradient_vec =
                __GEIGEN__::__M12x9_v9_multiply(__GEIGEN__::__Transpose9x12(PFPx), flatten_pk1);
#endif

            atomicAdd(&(_gradient[v0I].x), gradient_vec.v[0]);
            atomicAdd(&(_gradient[v0I].y), gradient_vec.v[1]);
            atomicAdd(&(_gradient[v0I].z), gradient_vec.v[2]);
            atomicAdd(&(_gradient[MMCVIDI.y].x), gradient_vec.v[3]);
            atomicAdd(&(_gradient[MMCVIDI.y].y), gradient_vec.v[4]);
            atomicAdd(&(_gradient[MMCVIDI.y].z), gradient_vec.v[5]);
            atomicAdd(&(_gradient[MMCVIDI.z].x), gradient_vec.v[6]);
            atomicAdd(&(_gradient[MMCVIDI.z].y), gradient_vec.v[7]);
            atomicAdd(&(_gradient[MMCVIDI.z].z), gradient_vec.v[8]);
            atomicAdd(&(_gradient[MMCVIDI.w].x), gradient_vec.v[9]);
            atomicAdd(&(_gradient[MMCVIDI.w].y), gradient_vec.v[10]);
            atomicAdd(&(_gradient[MMCVIDI.w].z), gradient_vec.v[11]);
        }
    }
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
