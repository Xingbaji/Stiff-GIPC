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

//=============================================================================
// Adaptive Stiffness Barrier Kernels (ppf-contact-solver style)
//
// These kernels compute barrier forces with per-contact adaptive stiffness
// based on local Hessian and mass regularization.
// 
// The stiffness coefficient is computed as:
//   k = w^T * (H_local + M/gap²) * w
// where:
//   - H_local: local Hessian blocks from inertia + elasticity
//   - M: mass matrix diagonal
//   - gap: distance to collision threshold
//   - w: normalized proximity-weighted direction vector
//=============================================================================

/**
 * @brief Calculate barrier gradient and hessian with adaptive stiffness
 * 
 * This version computes per-contact stiffness based on local material properties
 * and mass regularization, following the approach from ppf-contact-solver.
 * 
 * @param _vertexes        Current vertex positions
 * @param _rest_vertexes   Rest vertex positions
 * @param _collisionPair   Collision pairs in int4 format
 * @param _gradient        Output gradient (forces)
 * @param triplet_values   Output Hessian values
 * @param row_ids          Output Hessian row indices
 * @param col_ids          Output Hessian column indices
 * @param _cpNum           Collision pair counts
 * @param matIndex         Matrix indices for each collision
 * @param _masses          Per-vertex mass values
 * @param _hess_diag       Per-vertex diagonal Hessian values (inertia + elasticity)
 * @param dHat             Distance threshold squared
 * @param dt               Timestep for inertia computation
 * @param offset4/3/2      Offsets for different collision types
 * @param number           Total number of collisions
 */
__global__ void _calBarrierGradientAndHessianAdaptive(
    const double3*   _vertexes,
    const double3*   _rest_vertexes,
    const int4*      _collisionPair,
    double3*         _gradient,
    Eigen::Matrix3d* triplet_values,
    int*             row_ids,
    int*             col_ids,
    uint32_t*        _cpNum,
    int*             matIndex,
    const double*    _masses,
    const double3*   _hess_diag,
    double           dHat,
    double           dt,
    int              offset4,
    int              offset3,
    int              offset2,
    int              number);

/**
 * @brief Calculate barrier gradient with adaptive stiffness
 */
__global__ void _calBarrierGradientAdaptive(
    const double3*   _vertexes,
    const double3*   _rest_vertexes,
    const int4*      _collisionPair,
    double3*         _gradient,
    const double*    _masses,
    const double3*   _hess_diag,
    double           dHat,
    double           dt,
    int              number);

/**
 * @brief Compute per-vertex diagonal Hessian from inertia
 * 
 * This kernel computes the diagonal Hessian contribution from the inertia term:
 *   H_diag[i] = (mass[i] / dt²) * I
 * 
 * This should be called before the adaptive barrier computation.
 * Additional stiffness contributions (e.g., from elasticity) can be added
 * by the caller before passing to the barrier kernels.
 */
__global__ void _computeInertiaHessDiag(
    const double* _masses,
    double3*      _hess_diag,
    double        dt,
    int           number);

#endif // _BARRIER_GRADIENT_HESSIAN_H_

