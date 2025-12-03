//
// ipc_barrier.cuh
// GIPC - IPC Barrier Functions (Ground/Self Collision Kernels)
//
// Refactored from GIPC.cu
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#pragma once
#ifndef _IPC_BARRIER_H_
#define _IPC_BARRIER_H_

#include <cuda_runtime.h>
#include "gpu_eigen_libs.cuh"
#include "Eigen/Eigen"

//=============================================================================
// Device Helper Functions
//=============================================================================

// Self constraint value calculation
__device__ double _selfConstraintVal(const double3* _vertexes, const int4& MMCVIDI);

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
                                       int       number);

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
                                                 int              number);

__global__ void _computeGroundGradient(const double3*  vertexes,
                                       const double*   g_offset,
                                       const double3*  g_normal,
                                       const uint32_t* _environment_collisionPair,
                                       double3*        gradient,
                                       uint32_t*       _gpNum,
                                       double          dHat,
                                       double          Kappa,
                                       int             number);

__global__ void _computeGroundCloseVal(const double3*  vertexes,
                                       const double*   g_offset,
                                       const double3*  g_normal,
                                       const uint32_t* _environment_collisionPair,
                                       double          dTol,
                                       uint32_t*       _closeConstraintID,
                                       double*         _closeConstraintVal,
                                       uint32_t*       _close_gpNum,
                                       int             number);

__global__ void _checkGroundCloseVal(const double3* vertexes,
                                     const double*  g_offset,
                                     const double3* g_normal,
                                     int*           _isChange,
                                     uint32_t*      _closeConstraintID,
                                     double*        _closeConstraintVal,
                                     int            number);

__global__ void _reduct_MGroundDist(const double3* vertexes,
                                    const double*  g_offset,
                                    const double3* g_normal,
                                    uint32_t*      _environment_collisionPair,
                                    double2*       _queue,
                                    int            number);

__global__ void _checkGroundIntersection(const double3*  vertexes,
                                         const double*   g_offset,
                                         const double3*  g_normal,
                                         const uint32_t* _environment_collisionPair,
                                         int*            _isIntersect,
                                         int             number);

//=============================================================================
// Self Collision Kernels
//=============================================================================

__global__ void _calSelfCloseVal(const double3* _vertexes,
                                 const int4*    _collisionPair,
                                 int4*          _close_collisionPair,
                                 double*        _close_collisionVal,
                                 uint32_t*      _close_cpNum,
                                 double         dTol,
                                 int            number);

__global__ void _checkSelfCloseVal(const double3* _vertexes,
                                   int*           _isChange,
                                   int4*          _close_collisionPair,
                                   double*        _close_collisionVal,
                                   int            number);

__global__ void _reduct_MSelfDist(const double3* _vertexes,
                                  int4*          _collisionPairs,
                                  double2*       _queue,
                                  int            number);

__global__ void _computeSelfCloseVal(const double3*  vertexes,
                                     const double*   g_offset,
                                     const double3*  g_normal,
                                     const uint32_t* _environment_collisionPair,
                                     double          dTol,
                                     uint32_t*       _closeConstraintID,
                                     double*         _closeConstraintVal,
                                     uint32_t*       _close_gpNum,
                                     int             number);

#endif // _IPC_BARRIER_H_
