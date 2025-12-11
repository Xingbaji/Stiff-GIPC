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
//
// Note: These return the raw cubic barrier values. The caller is responsible
// for proper scaling to match GIPC's convention.
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
        // g = dis (linear distance), ghat = dHat_sqrt (linear threshold)
        double g = dis;
        double ghat = dHat_sqrt;
        double grad = cubic_gradient(g, ghat);  // = dE/dg = -2*(g-ghat)²/ghat
        
        // Transform from g-space to y-space where y = g/ghat = dis/dHat_sqrt
        // dE/dy = dE/dg * dg/dy = grad * ghat
        // grad_coeff * y = dE/dy => grad_coeff = dE/dy / y = grad * ghat / (g/ghat) = grad * ghat² / g
        // Since ghat² = dHat: grad_coeff = grad * dHat / g = grad * dHat / dis
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
        // Cubic barrier hessian
        double g = dis;
        double ghat = dHat_sqrt;
        double curv = cubic_curvature(g, ghat);  // = d²E/dg² = 4*(1-g/ghat)
        
        // Transform from g-space to y-space where y = g/ghat
        // d²E/dy² = d/dy(dE/dy) = d/dy(dE/dg * ghat) = d²E/dg² * ghat * ghat = curv * ghat²
        // Since ghat² = dHat: d²E/dy² = curv * dHat
        return Kappa * curv * dHat;
    } else {
        // Log barrier hessian
        double I5 = (dis / dHat_sqrt) * (dis / dHat_sqrt);
        double dis_sq = dis * dis;
        return Kappa * log_hessian_coeff_clamped(I5, dHat, dis_sq, gassThreshold);
    }
}

//=============================================================================
// Additive Stiffness Computation (from ppf-contact-solver)
//
// Computes the adaptive stiffness coefficient for barrier forces based on:
// 1. Local Hessian matrix from inertia + elasticity
// 2. Mass-based regularization term: mass / gap²
// 3. Rayleigh quotient: w^T * H * w
//
// This strategy ensures the barrier stiffness adapts to local material properties
// and prevents overshooting in stiff regions.
//=============================================================================

// Compute stiffness for PP (Point-Point) collision
// indices: vertex indices involved [v0, v1]
// prox_weights: proximity weights [w0, w1], e.g., [1, -1]
// masses: mass values for each vertex
// gap: current gap distance (dis - offset)
// global_hess_diag: diagonal Hessian block values for each vertex (inertia + elasticity)
__device__ __forceinline__ double compute_stiffness_PP(
    const double3& e,           // edge vector = v0 - v1
    double mass0, double mass1, // masses of vertices
    double gap,                 // gap = dis - offset, must be > 0
    const double3& hess_diag0,  // diagonal Hessian block for v0 (e.g., mass/dt² + stiffness)
    const double3& hess_diag1)  // diagonal Hessian block for v1
{
    // Proximity weights for PP: [1, -1]
    const double w0 = 1.0;
    const double w1 = -1.0;
    
    double gap_sq = gap * gap;
    
    // Build direction vector: w = [w0*e, w1*e] and normalize
    double3 dir0 = __GEIGEN__::__s_vec_multiply(e, w0);
    double3 dir1 = __GEIGEN__::__s_vec_multiply(e, w1);
    double norm_sq = __GEIGEN__::__squaredNorm(dir0) + __GEIGEN__::__squaredNorm(dir1);
    if (norm_sq < 1e-20) return 1.0;
    double inv_norm = 1.0 / sqrt(norm_sq);
    dir0 = __GEIGEN__::__s_vec_multiply(dir0, inv_norm);
    dir1 = __GEIGEN__::__s_vec_multiply(dir1, inv_norm);
    
    // Compute w^T * H * w where H is block-diagonal with mass regularization
    // H_ii = hess_diag_i + (mass_i / gap²) * I
    double result = 0.0;
    
    // Contribution from vertex 0
    double3 hess0_with_mass = make_double3(
        hess_diag0.x + mass0 / gap_sq,
        hess_diag0.y + mass0 / gap_sq,
        hess_diag0.z + mass0 / gap_sq);
    result += dir0.x * hess0_with_mass.x * dir0.x
            + dir0.y * hess0_with_mass.y * dir0.y
            + dir0.z * hess0_with_mass.z * dir0.z;
    
    // Contribution from vertex 1
    double3 hess1_with_mass = make_double3(
        hess_diag1.x + mass1 / gap_sq,
        hess_diag1.y + mass1 / gap_sq,
        hess_diag1.z + mass1 / gap_sq);
    result += dir1.x * hess1_with_mass.x * dir1.x
            + dir1.y * hess1_with_mass.y * dir1.y
            + dir1.z * hess1_with_mass.z * dir1.z;
    
    return result > 0.0 ? result : 1.0;
}

// Compute stiffness for PE (Point-Edge) collision
// prox_weights: [1, -c0, -c1] where c0, c1 are barycentric coords on edge
__device__ __forceinline__ double compute_stiffness_PE(
    const double3& e,                    // gap vector
    double mass0, double mass1, double mass2,  // masses
    double c0, double c1,               // barycentric coords (edge point = c0*v1 + c1*v2)
    double gap,
    const double3& hess_diag0,
    const double3& hess_diag1,
    const double3& hess_diag2)
{
    const double w0 = 1.0;
    const double w1 = -c0;
    const double w2 = -c1;
    
    double gap_sq = gap * gap;
    
    // Build and normalize direction
    double3 dir0 = __GEIGEN__::__s_vec_multiply(e, w0);
    double3 dir1 = __GEIGEN__::__s_vec_multiply(e, w1);
    double3 dir2 = __GEIGEN__::__s_vec_multiply(e, w2);
    double norm_sq = __GEIGEN__::__squaredNorm(dir0) 
                   + __GEIGEN__::__squaredNorm(dir1)
                   + __GEIGEN__::__squaredNorm(dir2);
    if (norm_sq < 1e-20) return 1.0;
    double inv_norm = 1.0 / sqrt(norm_sq);
    dir0 = __GEIGEN__::__s_vec_multiply(dir0, inv_norm);
    dir1 = __GEIGEN__::__s_vec_multiply(dir1, inv_norm);
    dir2 = __GEIGEN__::__s_vec_multiply(dir2, inv_norm);
    
    double result = 0.0;
    
    // Vertex 0
    double3 hess0 = make_double3(
        hess_diag0.x + mass0 / gap_sq,
        hess_diag0.y + mass0 / gap_sq,
        hess_diag0.z + mass0 / gap_sq);
    result += dir0.x * hess0.x * dir0.x + dir0.y * hess0.y * dir0.y + dir0.z * hess0.z * dir0.z;
    
    // Vertex 1
    double3 hess1 = make_double3(
        hess_diag1.x + mass1 / gap_sq,
        hess_diag1.y + mass1 / gap_sq,
        hess_diag1.z + mass1 / gap_sq);
    result += dir1.x * hess1.x * dir1.x + dir1.y * hess1.y * dir1.y + dir1.z * hess1.z * dir1.z;
    
    // Vertex 2
    double3 hess2 = make_double3(
        hess_diag2.x + mass2 / gap_sq,
        hess_diag2.y + mass2 / gap_sq,
        hess_diag2.z + mass2 / gap_sq);
    result += dir2.x * hess2.x * dir2.x + dir2.y * hess2.y * dir2.y + dir2.z * hess2.z * dir2.z;
    
    return result > 0.0 ? result : 1.0;
}

// Compute stiffness for PT (Point-Triangle) collision
// prox_weights: [1, -c0, -c1, -c2] where c_i are barycentric coords on triangle
__device__ __forceinline__ double compute_stiffness_PT(
    const double3& e,
    double mass0, double mass1, double mass2, double mass3,
    double c0, double c1, double c2,    // barycentric coords
    double gap,
    const double3& hess_diag0,
    const double3& hess_diag1,
    const double3& hess_diag2,
    const double3& hess_diag3)
{
    const double w0 = 1.0;
    const double w1 = -c0;
    const double w2 = -c1;
    const double w3 = -c2;
    
    double gap_sq = gap * gap;
    
    double3 dir0 = __GEIGEN__::__s_vec_multiply(e, w0);
    double3 dir1 = __GEIGEN__::__s_vec_multiply(e, w1);
    double3 dir2 = __GEIGEN__::__s_vec_multiply(e, w2);
    double3 dir3 = __GEIGEN__::__s_vec_multiply(e, w3);
    double norm_sq = __GEIGEN__::__squaredNorm(dir0) 
                   + __GEIGEN__::__squaredNorm(dir1)
                   + __GEIGEN__::__squaredNorm(dir2)
                   + __GEIGEN__::__squaredNorm(dir3);
    if (norm_sq < 1e-20) return 1.0;
    double inv_norm = 1.0 / sqrt(norm_sq);
    dir0 = __GEIGEN__::__s_vec_multiply(dir0, inv_norm);
    dir1 = __GEIGEN__::__s_vec_multiply(dir1, inv_norm);
    dir2 = __GEIGEN__::__s_vec_multiply(dir2, inv_norm);
    dir3 = __GEIGEN__::__s_vec_multiply(dir3, inv_norm);
    
    double result = 0.0;
    
    // Sum contributions from all 4 vertices
    double3 hess0 = make_double3(hess_diag0.x + mass0/gap_sq, hess_diag0.y + mass0/gap_sq, hess_diag0.z + mass0/gap_sq);
    result += dir0.x * hess0.x * dir0.x + dir0.y * hess0.y * dir0.y + dir0.z * hess0.z * dir0.z;
    
    double3 hess1 = make_double3(hess_diag1.x + mass1/gap_sq, hess_diag1.y + mass1/gap_sq, hess_diag1.z + mass1/gap_sq);
    result += dir1.x * hess1.x * dir1.x + dir1.y * hess1.y * dir1.y + dir1.z * hess1.z * dir1.z;
    
    double3 hess2 = make_double3(hess_diag2.x + mass2/gap_sq, hess_diag2.y + mass2/gap_sq, hess_diag2.z + mass2/gap_sq);
    result += dir2.x * hess2.x * dir2.x + dir2.y * hess2.y * dir2.y + dir2.z * hess2.z * dir2.z;
    
    double3 hess3 = make_double3(hess_diag3.x + mass3/gap_sq, hess_diag3.y + mass3/gap_sq, hess_diag3.z + mass3/gap_sq);
    result += dir3.x * hess3.x * dir3.x + dir3.y * hess3.y * dir3.y + dir3.z * hess3.z * dir3.z;
    
    return result > 0.0 ? result : 1.0;
}

// Compute stiffness for EE (Edge-Edge) collision
// prox_weights: [c0, c1, -c2, -c3] where c_i are barycentric coords
__device__ __forceinline__ double compute_stiffness_EE(
    const double3& e,
    double mass0, double mass1, double mass2, double mass3,
    double c0, double c1, double c2, double c3,  // barycentric coords
    double gap,
    const double3& hess_diag0,
    const double3& hess_diag1,
    const double3& hess_diag2,
    const double3& hess_diag3)
{
    const double w0 = c0;
    const double w1 = c1;
    const double w2 = -c2;
    const double w3 = -c3;
    
    double gap_sq = gap * gap;
    
    double3 dir0 = __GEIGEN__::__s_vec_multiply(e, w0);
    double3 dir1 = __GEIGEN__::__s_vec_multiply(e, w1);
    double3 dir2 = __GEIGEN__::__s_vec_multiply(e, w2);
    double3 dir3 = __GEIGEN__::__s_vec_multiply(e, w3);
    double norm_sq = __GEIGEN__::__squaredNorm(dir0) 
                   + __GEIGEN__::__squaredNorm(dir1)
                   + __GEIGEN__::__squaredNorm(dir2)
                   + __GEIGEN__::__squaredNorm(dir3);
    if (norm_sq < 1e-20) return 1.0;
    double inv_norm = 1.0 / sqrt(norm_sq);
    dir0 = __GEIGEN__::__s_vec_multiply(dir0, inv_norm);
    dir1 = __GEIGEN__::__s_vec_multiply(dir1, inv_norm);
    dir2 = __GEIGEN__::__s_vec_multiply(dir2, inv_norm);
    dir3 = __GEIGEN__::__s_vec_multiply(dir3, inv_norm);
    
    double result = 0.0;
    
    double3 hess0 = make_double3(hess_diag0.x + mass0/gap_sq, hess_diag0.y + mass0/gap_sq, hess_diag0.z + mass0/gap_sq);
    result += dir0.x * hess0.x * dir0.x + dir0.y * hess0.y * dir0.y + dir0.z * hess0.z * dir0.z;
    
    double3 hess1 = make_double3(hess_diag1.x + mass1/gap_sq, hess_diag1.y + mass1/gap_sq, hess_diag1.z + mass1/gap_sq);
    result += dir1.x * hess1.x * dir1.x + dir1.y * hess1.y * dir1.y + dir1.z * hess1.z * dir1.z;
    
    double3 hess2 = make_double3(hess_diag2.x + mass2/gap_sq, hess_diag2.y + mass2/gap_sq, hess_diag2.z + mass2/gap_sq);
    result += dir2.x * hess2.x * dir2.x + dir2.y * hess2.y * dir2.y + dir2.z * hess2.z * dir2.z;
    
    double3 hess3 = make_double3(hess_diag3.x + mass3/gap_sq, hess_diag3.y + mass3/gap_sq, hess_diag3.z + mass3/gap_sq);
    result += dir3.x * hess3.x * dir3.x + dir3.y * hess3.y * dir3.y + dir3.z * hess3.z * dir3.z;
    
    return result > 0.0 ? result : 1.0;
}

// Simplified stiffness computation using only mass and inertia (no full Hessian access)
// This is a practical approximation when full Hessian blocks are not readily available
// Uses: stiffness ≈ mass / (dt² * gap²) based on inertia dominance assumption
__device__ __forceinline__ double compute_stiffness_simple(
    double avg_mass,    // average mass of involved vertices
    double dt,          // timestep
    double gap)         // gap distance
{
    // Inertia-based stiffness: mass / dt²
    // Plus mass regularization: mass / gap²
    // Combined: mass * (1/dt² + 1/gap²)
    double gap_sq = gap * gap;
    double dt_sq = dt * dt;
    return avg_mass * (1.0 / dt_sq + 1.0 / gap_sq);
}

//=============================================================================
// Ground Collision Stiffness Computation (from ppf-contact-solver)
//
// For ground/floor collisions, the stiffness is computed as:
//   stiff_k = (normal^T * local_hess * normal) + (mass / gap²)
// where:
//   - normal: ground normal direction
//   - local_hess: local 3x3 Hessian block for the vertex (inertia + elasticity)
//   - mass: vertex mass
//   - gap: distance to ground (must be positive)
//=============================================================================

// Compute stiffness for ground collision (single vertex)
// normal: ground normal (e.g., (0,1,0) for y-up)
// mass: vertex mass
// gap: signed distance to ground (positive = above ground)
// hess_diag: diagonal Hessian block for the vertex (3D, isotropic approximation)
__device__ __forceinline__ double compute_stiffness_ground(
    const double3& normal,      // ground normal direction
    double mass,                // vertex mass
    double gap,                 // gap distance (must be > 0)
    const double3& hess_diag)   // diagonal Hessian block
{
    if (gap <= 0.0) return 1.0;  // Safety check
    
    double gap_sq = gap * gap;
    
    // Compute normal^T * H_diag * normal
    // For diagonal H, this is: sum_i (normal_i² * H_ii)
    double hess_contrib = normal.x * normal.x * hess_diag.x
                        + normal.y * normal.y * hess_diag.y
                        + normal.z * normal.z * hess_diag.z;
    
    // Add mass regularization
    double result = hess_contrib + mass / gap_sq;
    
    return result > 0.0 ? result : 1.0;
}

// Simplified ground stiffness using only mass (when local Hessian is not available)
// Based on ppf-contact-solver: stiff_k = normal^T * H * normal + mass/gap²
// Without full Hessian access, we only use the gap-dependent term
__device__ __forceinline__ double compute_stiffness_ground_simple(
    double mass,
    double dt,
    double gap)
{
    if (gap <= 0.0) return 1.0;
    
    double gap_sq = gap * gap;
    
    // Only use mass regularization term: mass / gap²
    // This is the most important term when the gap is small (near contact)
    // The full Hessian contribution would require access to the global system matrix
    return mass / gap_sq;
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
            // E = mollifier(I1) * cubic_energy(g)
            // p1 = dE/dI1 = d_mollifier/dI1 * cubic_energy
            // p2 = dE/dI2 = mollifier * dE_cubic/dI2 = mollifier * dE_cubic/dg * dg/dI2
            //    where dg/dI2 = d(sqrt(I2*dHat))/dI2 = dHat / (2*g)
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

