//
// barrier_gradient_hessian.cuh
// GIPC - Barrier Gradient/Hessian Kernels Header
//
// Extracted from ipc_barrier.cu (lines 97-136)
// These kernels calculate barrier gradient and hessian for IPC collision handling
//
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#pragma once
#ifndef _BARRIER_GRADIENT_HESSIAN_H_
#define _BARRIER_GRADIENT_HESSIAN_H_

#include <cuda_runtime.h>
#include "gpu_eigen_libs.cuh"
#include "Eigen/Eigen"

//=============================================================================
// Barrier Gradient/Hessian Kernels
// These are the core kernels for computing IPC barrier forces
//=============================================================================

/**
 * @brief Calculate both barrier gradient and hessian for collision pairs
 * 
 * This kernel computes gradient and Hessian contributions for all collision types:
 * - Edge-Edge (EE) collisions
 * - Point-Triangle (PT) collisions  
 * - Point-Edge (PE) collisions
 * - Point-Point (PP) collisions
 * 
 * The collision type is encoded in the MMCVIDI (int4) format:
 * - x >= 0 && w >= 0: EE collision
 * - x >= 0 && w < 0:  Parallel EE collision
 * - x < 0 && z < 0 && y < 0: PPP collision  
 * - x < 0 && z < 0 && y >= 0: PP collision
 * - x < 0 && z >= 0 && w < 0 && y < 0: PPE collision
 * - x < 0 && z >= 0 && w < 0 && y >= 0: PE collision
 * - x < 0 && z >= 0 && w >= 0: PT collision
 */
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
                                              int              number);

/**
 * @brief Calculate only barrier hessian for collision pairs
 * 
 * Similar to _calBarrierGradientAndHessian but only computes Hessian.
 * Used when only the stiffness matrix is needed (e.g., in some line search methods).
 */
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
                                   int              number);

/**
 * @brief Calculate only barrier gradient for collision pairs
 * 
 * Similar to _calBarrierGradientAndHessian but only computes gradient.
 * Used when only forces are needed (e.g., in energy evaluation).
 */
__global__ void _calBarrierGradient(const double3* _vertexes,
                                    const double3* _rest_vertexes,
                                    const int4*    _collisionPair,
                                    double3*       _gradient,
                                    double         dHat,
                                    double         Kappa,
                                    int            number);

#endif // _BARRIER_GRADIENT_HESSIAN_H_

