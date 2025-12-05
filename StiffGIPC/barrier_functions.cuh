//
// barrier_functions.cuh
// GIPC - Barrier Function Definitions
//
// Supports multiple barrier types:
// - LOG_BARRIER: Original IPC log barrier (default)
// - CUBIC_BARRIER: Cubic barrier from ppf-contact-solver
//
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#pragma once
#ifndef _BARRIER_FUNCTIONS_H_
#define _BARRIER_FUNCTIONS_H_

#include <cuda_runtime.h>
#include <cmath>

namespace barrier {

//=============================================================================
// Cubic Barrier Functions (from ppf-contact-solver)
// Reference: cubic.hpp
//
// For distance g and threshold ghat:
//   y = g - ghat
//   energy = -2 * y^3 / (3 * ghat)  when y < 0
//   gradient = -2 * y^2 / ghat       when y < 0  
//   curvature = 4 * (1 - g/ghat)     when y < 0
//=============================================================================

__device__ __forceinline__ double cubic_energy(double g, double ghat) {
    double y = g - ghat;
    if (y < 0.0) {
        return -2.0 * (y * y * y) / (3.0 * ghat);
    }
    return 0.0;
}

__device__ __forceinline__ double cubic_gradient(double g, double ghat) {
    double y = g - ghat;
    if (y < 0.0) {
        return -2.0 * y * y / ghat;
    }
    return 0.0;
}

__device__ __forceinline__ double cubic_curvature(double g, double ghat) {
    double y = g - ghat;
    if (y < 0.0) {
        return 4.0 * (1.0 - g / ghat);
    }
    return 0.0;
}

//=============================================================================
// Log Barrier Functions (Original IPC RANK=2 formulation)
// 
// For I5 = (dis/dHat_sqrt)^2 = dis^2/dHat:
//   gradient_coeff = 2 * dHat^2 * log(I5) * (I5-1) * (I5 + I5*log(I5) - 1) / I5
//   hessian_coeff = -4 * dHat^2 * (4*I5 + log(I5) - 3*I5^2*log(I5)^2 + 6*I5*log(I5) 
//                    - 2*I5^2 + I5*log(I5)^2 - 7*I5^2*log(I5) - 2) / I5
//=============================================================================

__device__ __forceinline__ double log_energy(double I5, double dHat) {
    // Energy = dHat^2 * (I5-1)^2 * log(I5)^2
    double t = I5 - 1.0;
    double l = log(I5);
    return dHat * dHat * t * t * l * l;
}

__device__ __forceinline__ double log_gradient_coeff(double I5, double dHat) {
    // d(energy)/dI5 = 2 * dHat^2 * log(I5) * (I5-1) * (I5 + I5*log(I5) - 1) / I5
    return 2.0 * dHat * dHat * log(I5) * (I5 - 1.0) * (I5 + I5 * log(I5) - 1.0) / I5;
}

__device__ __forceinline__ double log_hessian_coeff(double I5, double dHat) {
    // d²(energy)/dI5² (simplified form)
    return -(4.0 * dHat * dHat
             * (4.0 * I5 + log(I5) - 3.0 * I5 * I5 * log(I5) * log(I5) 
                + 6.0 * I5 * log(I5) - 2.0 * I5 * I5 
                + I5 * log(I5) * log(I5) - 7.0 * I5 * I5 * log(I5) - 2.0))
           / I5;
}

__device__ __forceinline__ double log_hessian_coeff_clamped(double I5, double dHat, 
                                                             double dis_sq, double gassThreshold) {
    double lambda0 = log_hessian_coeff(I5, dHat);
    
    if (dis_sq < gassThreshold * dHat) {
        double lambda1 = -(4.0 * dHat * dHat
                           * (4.0 * gassThreshold + log(gassThreshold)
                              - 3.0 * gassThreshold * gassThreshold * log(gassThreshold) * log(gassThreshold)
                              + 6.0 * gassThreshold * log(gassThreshold) 
                              - 2.0 * gassThreshold * gassThreshold
                              + gassThreshold * log(gassThreshold) * log(gassThreshold)
                              - 7.0 * gassThreshold * gassThreshold * log(gassThreshold) - 2.0))
                         / gassThreshold;
        lambda0 = lambda1;
    }
    return lambda0;
}

//=============================================================================
// Unified Simple Barrier Interface (for PP, PE, PT, non-parallel EE)
// 
// These compute gradient/hessian coefficients for the simple barrier case
// where the barrier depends on squared distance I5 = dis²/dHat
//=============================================================================

// Compute gradient coefficient: multiply by Kappa * (dis/dHat_sqrt) to get pk1
// Returns: scalar coefficient for the gradient direction
__device__ __forceinline__ double simple_gradient_coeff(double dis, double dHat, double Kappa,
                                                         bool use_cubic) {
    double dHat_sqrt = sqrt(dHat);
    
    if (use_cubic) {
        // Cubic barrier gradient
        double g = dis;
        double ghat = dHat_sqrt;
        double grad = cubic_gradient(g, ghat);
        // We want to return a coefficient C such that C * y = dE/dy, where y = dis/dHat_sqrt.
        // dE/dy = (dE/dg) * (dg/dy) = grad * dHat_sqrt.
        // So C = (grad * dHat_sqrt) / y = (grad * dHat_sqrt) / (dis/dHat_sqrt) = grad * dHat / dis.
        // Note: input dHat is dHat^2, dHat_sqrt is dHat.
        return Kappa * grad * dHat / dis;
    } else {
        // Log barrier gradient
        double I5 = (dis / dHat_sqrt) * (dis / dHat_sqrt);
        // Note: Log barrier code in barrier_gradient_hessian.cu applies a factor of 2
        // to dE/dI5. To unify the interface where simple_gradient_coeff returns C
        // such that Force ~ C * y (where y = dis/dHat_sqrt), we must include this factor.
        // dE/dy = dE/dI5 * dI5/dy = dE/dI5 * 2*y.
        // So C = 2 * dE/dI5.
        return 2.0 * Kappa * log_gradient_coeff(I5, dHat);
    }
}

// Compute hessian coefficient (lambda0)
__device__ __forceinline__ double simple_hessian_coeff(double dis, double dHat, double Kappa,
                                                        double gassThreshold, bool use_cubic) {
    double dHat_sqrt = sqrt(dHat);
    
    if (use_cubic) {
        // Cubic barrier curvature
        double g = dis;
        double ghat = dHat_sqrt;
        double curv = cubic_curvature(g, ghat);
        // The caller expects lambda0 = d^2E/dy^2, where y = dis/dHat_sqrt.
        // d^2E/dy^2 = d/dy (dE/dy) = d/dy (grad * dHat_sqrt) 
        //           = (d(grad)/dg * dg/dy) * dHat_sqrt 
        //           = curv * dHat_sqrt * dHat_sqrt = curv * dHat.
        // Note: input dHat is dHat^2.
        return Kappa * curv * dHat;
    } else {
        // Log barrier hessian
        double I5 = (dis / dHat_sqrt) * (dis / dHat_sqrt);
        double dis_sq = dis * dis;
        return Kappa * log_hessian_coeff_clamped(I5, dHat, dis_sq, gassThreshold);
    }
}

//=============================================================================
// Parallel/Mollified Barrier Interface (for parallel EE, PPP, PPE)
//
// These use I1 (mollifier factor from cross product) and I2 (normalized dist)
// Energy = f(I1, I2) where I1 controls mollification
//=============================================================================

// Parallel barrier gradient coefficients (p1 for I1 direction, p2 for I2 direction)
__device__ __forceinline__ void parallel_gradient_coeffs(
    double I1, double I2, double eps_x, double dHat, double Kappa,
    double& p1, double& p2, bool use_cubic) 
{
    if (use_cubic) {
        double dis = sqrt(I2 * dHat);
        double ghat = sqrt(dHat);
        double y = dis - ghat;
        
        if (y < 0.0) {
            double mollifier = (I1 - eps_x) / (eps_x * eps_x);
            double cubic_grad = -2.0 * y * y / ghat;
            // Partial derivatives of mollified energy
            // p2 = dE/dI2 = dE/dg * dg/dI2 = grad * (dHat / (2*g))
            // Note: dHat input is dHat^2.
            p1 = Kappa * 2.0 * mollifier * cubic_energy(dis, ghat) / (I1 - eps_x);
            p2 = Kappa * mollifier * cubic_grad * dHat / (2.0 * dis);
        } else {
            p1 = 0.0;
            p2 = 0.0;
        }
    } else {
        // Log barrier (original formulation)
        p1 = -Kappa * 2.0
             * (2.0 * dHat * dHat * log(I2) * log(I2) * (I1 - eps_x)
                * (I2 - 1.0) * (I2 - 1.0))
             / (eps_x * eps_x);
        p2 = -Kappa * 2.0
             * (2.0 * I1 * dHat * dHat * log(I2) * (I1 - 2.0 * eps_x)
                * (I2 - 1.0) * (I2 + I2 * log(I2) - 1.0))
             / (I2 * (eps_x * eps_x));
    }
}

// Parallel barrier hessian coefficients
__device__ __forceinline__ void parallel_hessian_coeffs(
    double I1, double I2, double c, double F22, double eps_x, double dHat, double Kappa,
    double& lambda10, double& lambda11, double& lambda12, double& lambda20, double& lambdag1g,
    bool use_cubic)
{
    if (use_cubic) {
        double dis = sqrt(I2 * dHat);
        double ghat = sqrt(dHat);
        double y = dis - ghat;
        
        if (y < 0.0) {
            double mollifier = (I1 - eps_x) / (eps_x * eps_x);
            double curv = 4.0 * (1.0 - dis / ghat);
            double energy = cubic_energy(dis, ghat);
            
            // Hessian components for mollified cubic barrier
            lambda10 = Kappa * 2.0 * energy / (eps_x * eps_x);
            lambda11 = Kappa * mollifier * curv * 0.25;
            lambda12 = lambda11;
            lambda20 = Kappa * mollifier * curv * dHat / (4.0 * I2);
            lambdag1g = Kappa * mollifier * curv * c * ghat / (2.0 * dis) * F22 / I2;
        } else {
            lambda10 = lambda11 = lambda12 = lambda20 = lambdag1g = 0.0;
        }
    } else {
        // Log barrier (original formulation)
        lambda10 = -Kappa
                   * (4.0 * dHat * dHat * log(I2) * log(I2) * (I2 - 1.0)
                      * (I2 - 1.0) * (3.0 * I1 - eps_x))
                   / (eps_x * eps_x);
        lambda11 = -Kappa
                   * (4.0 * dHat * dHat * log(I2) * log(I2)
                      * (I1 - eps_x) * (I2 - 1.0) * (I2 - 1.0))
                   / (eps_x * eps_x);
        lambda12 = lambda11;
        lambda20 = Kappa
                   * (4.0 * I1 * dHat * dHat * (I1 - 2.0 * eps_x)
                      * (4.0 * I2 + log(I2) - 3.0 * I2 * I2 * log(I2) * log(I2)
                         + 6.0 * I2 * log(I2) - 2.0 * I2 * I2 
                         + I2 * log(I2) * log(I2) - 7.0 * I2 * I2 * log(I2) - 2.0))
                   / (I2 * (eps_x * eps_x));
        lambdag1g = -Kappa * 4.0 * c * F22
                    * (4.0 * dHat * dHat * log(I2) * (I1 - eps_x)
                       * (I2 - 1.0) * (I2 + I2 * log(I2) - 1.0))
                    / (I2 * (eps_x * eps_x));
    }
}

} // namespace barrier

#endif // _BARRIER_FUNCTIONS_H_

