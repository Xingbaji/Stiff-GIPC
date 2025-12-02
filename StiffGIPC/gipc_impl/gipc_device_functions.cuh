//
// gipc_device_functions.cuh
// GIPC - Shared Device Functions
//
// This file contains device functions that are shared across multiple
// CUDA compilation units.
//
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#pragma once
#ifndef GIPC_DEVICE_FUNCTIONS_CUH
#define GIPC_DEVICE_FUNCTIONS_CUH

#include <cuda_runtime.h>
#include "../gpu_eigen_libs.cuh"
#include "../mlbvh.cuh"
#include "../ACCD.cuh"

#ifndef RANK
#define RANK 2
#endif

// __cal_Barrier_energy function
__device__ inline double __cal_Barrier_energy(const double3* _vertexes,
                                       const double3* _rest_vertexes,
                                       int4           MMCVIDI,
                                       double         _Kappa,
                                       double         _dHat)
{
    double dHat_sqrt = sqrt(_dHat);
    double dHat      = _dHat;
    double Kappa     = _Kappa;
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
            double I5 = dis / dHat;

            double lenE = (dis - dHat);
#if (RANK == 1)
            return -Kappa * lenE * lenE * log(I5);
#elif (RANK == 2)
            return Kappa * lenE * lenE * log(I5) * log(I5);
#elif (RANK == 3)
            return -Kappa * lenE * lenE * log(I5) * log(I5) * log(I5);
#elif (RANK == 4)
            return Kappa * lenE * lenE * log(I5) * log(I5) * log(I5) * log(I5);
#elif (RANK == 5)
            return -Kappa * lenE * lenE * log(I5) * log(I5) * log(I5) * log(I5) * log(I5);
#elif (RANK == 6)
            return Kappa * lenE * lenE * log(I5) * log(I5) * log(I5) * log(I5)
                   * log(I5) * log(I5);
#endif
        }
        else
        {
            //return 0;
            MMCVIDI.w = -MMCVIDI.w - 1;
            double3 v0 =
                __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _vertexes[MMCVIDI.x]);
            double3 v1 =
                __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.z]);
            double c = __GEIGEN__::__norm(__GEIGEN__::__v_vec_cross(v0, v1)) /*/ __GEIGEN__::__norm(v0)*/;
            double I1 = c * c;
            if(I1 == 0)
                return 0;
            double dis;
            _d_EE(_vertexes[MMCVIDI.x],
                  _vertexes[MMCVIDI.y],
                  _vertexes[MMCVIDI.z],
                  _vertexes[MMCVIDI.w],
                  dis);
            double I2    = dis / dHat;
            double eps_x = _compute_epx(_rest_vertexes[MMCVIDI.x],
                                        _rest_vertexes[MMCVIDI.y],
                                        _rest_vertexes[MMCVIDI.z],
                                        _rest_vertexes[MMCVIDI.w]);
#if (RANK == 1)
            double Energy = Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1)
                            * -(dHat - dHat * I2) * (dHat - dHat * I2) * log(I2);
#elif (RANK == 2)
            double Energy =
                Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1)
                * (dHat - dHat * I2) * (dHat - dHat * I2) * log(I2) * log(I2);
#elif (RANK == 4)
            double Energy = Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1)
                            * (dHat - dHat * I2) * (dHat - dHat * I2) * log(I2)
                            * log(I2) * log(I2) * log(I2);
#elif (RANK == 6)
            double Energy = Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1)
                            * (dHat - dHat * I2) * (dHat - dHat * I2) * log(I2)
                            * log(I2) * log(I2) * log(I2) * log(I2) * log(I2);
#endif
            if(Energy < 0)
                printf("I am pee\n");
            return Energy;
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
                    return 0;
                double dis;
                _d_PP(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], dis);
                double I2    = dis / dHat;
                double eps_x = _compute_epx(_rest_vertexes[MMCVIDI.x],
                                            _rest_vertexes[MMCVIDI.z],
                                            _rest_vertexes[MMCVIDI.y],
                                            _rest_vertexes[MMCVIDI.w]);
#if (RANK == 1)
                double Energy =
                    Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1)
                    * -(dHat - dHat * I2) * (dHat - dHat * I2) * log(I2);
#elif (RANK == 2)
                double Energy =
                    Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1)
                    * (dHat - dHat * I2) * (dHat - dHat * I2) * log(I2) * log(I2);
#elif (RANK == 4)
                double Energy =
                    Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1)
                    * (dHat - dHat * I2) * (dHat - dHat * I2) * log(I2)
                    * log(I2) * log(I2) * log(I2);
#elif (RANK == 6)
                double Energy =
                    Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1)
                    * (dHat - dHat * I2) * (dHat - dHat * I2) * log(I2)
                    * log(I2) * log(I2) * log(I2) * log(I2) * log(I2);
#endif
                if(Energy < 0)
                    printf("I am pp\n");
                return Energy;
            }
            else
            {
                double dis;
                _d_PP(_vertexes[v0I], _vertexes[MMCVIDI.y], dis);
                double I5 = dis / dHat;

                double lenE = (dis - dHat);
#if (RANK == 1)
                return -Kappa * lenE * lenE * log(I5);
#elif (RANK == 2)
                return Kappa * lenE * lenE * log(I5) * log(I5);
#elif (RANK == 3)
                return -Kappa * lenE * lenE * log(I5) * log(I5) * log(I5);
#elif (RANK == 4)
                return Kappa * lenE * lenE * log(I5) * log(I5) * log(I5) * log(I5);
#elif (RANK == 5)
                return -Kappa * lenE * lenE * log(I5) * log(I5) * log(I5)
                       * log(I5) * log(I5);
#elif (RANK == 6)
                return Kappa * lenE * lenE * log(I5) * log(I5) * log(I5)
                       * log(I5) * log(I5) * log(I5);
#endif
            }
        }
        else if(MMCVIDI.w < 0)
        {
            if(MMCVIDI.y < 0)
            {
                MMCVIDI.y = -MMCVIDI.y - 1;
                //MMCVIDI.z = -MMCVIDI.z - 1;
                MMCVIDI.w = -MMCVIDI.w - 1;
                MMCVIDI.x = v0I;

                double3 v0 =
                    __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.x]);
                double3 v1 =
                    __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.y]);
                double c = __GEIGEN__::__norm(__GEIGEN__::__v_vec_cross(v0, v1)) /*/ __GEIGEN__::__norm(v0)*/;
                double I1 = c * c;
                if(I1 == 0)
                    return 0;
                double dis;
                _d_PE(_vertexes[MMCVIDI.x],
                      _vertexes[MMCVIDI.y],
                      _vertexes[MMCVIDI.z],
                      dis);
                double I2    = dis / dHat;
                double eps_x = _compute_epx(_rest_vertexes[MMCVIDI.x],
                                            _rest_vertexes[MMCVIDI.w],
                                            _rest_vertexes[MMCVIDI.y],
                                            _rest_vertexes[MMCVIDI.z]);
#if (RANK == 1)
                double Energy =
                    Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1)
                    * -(dHat - dHat * I2) * (dHat - dHat * I2) * log(I2);
#elif (RANK == 2)
                double Energy =
                    Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1)
                    * (dHat - dHat * I2) * (dHat - dHat * I2) * log(I2) * log(I2);
#elif (RANK == 4)
                double Energy =
                    Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1)
                    * (dHat - dHat * I2) * (dHat - dHat * I2) * log(I2)
                    * log(I2) * log(I2) * log(I2);
#elif (RANK == 6)
                double Energy =
                    Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1)
                    * (dHat - dHat * I2) * (dHat - dHat * I2) * log(I2)
                    * log(I2) * log(I2) * log(I2) * log(I2) * log(I2);
#endif
                if(Energy < 0)
                    printf("I am ppe\n");
                return Energy;
            }
            else
            {
                double dis;
                _d_PE(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], dis);
                double I5 = dis / dHat;

                double lenE = (dis - dHat);
#if (RANK == 1)
                return -Kappa * lenE * lenE * log(I5);
#elif (RANK == 2)
                return Kappa * lenE * lenE * log(I5) * log(I5);
#elif (RANK == 3)
                return -Kappa * lenE * lenE * log(I5) * log(I5) * log(I5);
#elif (RANK == 4)
                return Kappa * lenE * lenE * log(I5) * log(I5) * log(I5) * log(I5);
#elif (RANK == 5)
                return -Kappa * lenE * lenE * log(I5) * log(I5) * log(I5)
                       * log(I5) * log(I5);
#elif (RANK == 6)
                return Kappa * lenE * lenE * log(I5) * log(I5) * log(I5)
                       * log(I5) * log(I5) * log(I5);
#endif
            }
        }
        else
        {
            double dis;
            _d_PT(_vertexes[v0I],
                  _vertexes[MMCVIDI.y],
                  _vertexes[MMCVIDI.z],
                  _vertexes[MMCVIDI.w],
                  dis);
            double I5 = dis / dHat;

            double lenE = (dis - dHat);
#if (RANK == 1)
            return -Kappa * lenE * lenE * log(I5);
#elif (RANK == 2)
            return Kappa * lenE * lenE * log(I5) * log(I5);
#elif (RANK == 3)
            return -Kappa * lenE * lenE * log(I5) * log(I5) * log(I5);
#elif (RANK == 4)
            return Kappa * lenE * lenE * log(I5) * log(I5) * log(I5) * log(I5);
#elif (RANK == 5)
            return -Kappa * lenE * lenE * log(I5) * log(I5) * log(I5) * log(I5) * log(I5);
#elif (RANK == 6)
            return Kappa * lenE * lenE * log(I5) * log(I5) * log(I5) * log(I5)
                   * log(I5) * log(I5);
#endif
        }
    }
}

__device__ inline bool segTriIntersect(const double3& ve0,
                                       const double3& ve1,
                                       const double3& vt0,
                                       const double3& vt1,
                                       const double3& vt2)
{

    //printf("check for tri and lines\n");

    __GEIGEN__::Matrix3x3d coefMtr;
    double3                col0 = __GEIGEN__::__minus(vt1, vt0);
    double3                col1 = __GEIGEN__::__minus(vt2, vt0);
    double3                col2 = __GEIGEN__::__minus(ve0, ve1);

    __GEIGEN__::__set_Mat_val_column(coefMtr, col0, col1, col2);

    double3 n = __GEIGEN__::__v_vec_cross(col0, col1);
    if(__GEIGEN__::__v_vec_dot(n, __GEIGEN__::__minus(ve0, vt0))
           * __GEIGEN__::__v_vec_dot(n, __GEIGEN__::__minus(ve1, vt0))
       > 0)
    {
        return false;
    }

    double det = __GEIGEN__::__Determiant(coefMtr);

    if(abs(det) < 1e-20)
    {
        return false;
    }

    __GEIGEN__::Matrix3x3d D1, D2, D3;
    double3                b = __GEIGEN__::__minus(ve0, vt0);

    __GEIGEN__::__set_Mat_val_column(D1, b, col1, col2);
    __GEIGEN__::__set_Mat_val_column(D2, col0, b, col2);
    __GEIGEN__::__set_Mat_val_column(D3, col0, col1, b);

    double uvt[3];
    uvt[0] = __GEIGEN__::__Determiant(D1) / det;
    uvt[1] = __GEIGEN__::__Determiant(D2) / det;
    uvt[2] = __GEIGEN__::__Determiant(D3) / det;

    if(uvt[0] >= 0.0 && uvt[1] >= 0.0 && uvt[0] + uvt[1] <= 1.0 && uvt[2] >= 0.0
       && uvt[2] <= 1.0)
    {
        return true;
    }
    else
    {
        return false;
    }
}

__device__ __host__ inline bool _overlap(const AABB& lhs, const AABB& rhs, const double& gapL) noexcept
{
    if((rhs.lower.x - lhs.upper.x) >= gapL || (lhs.lower.x - rhs.upper.x) >= gapL)
        return false;
    if((rhs.lower.y - lhs.upper.y) >= gapL || (lhs.lower.y - rhs.upper.y) >= gapL)
        return false;
    if((rhs.lower.z - lhs.upper.z) >= gapL || (lhs.lower.z - rhs.upper.z) >= gapL)
        return false;
    return true;
}

// _selfConstraintVal function
__device__ inline double _selfConstraintVal(const double3* vertexes, const int4& active)
{
    double val;
    if(active.x >= 0)
    {
        if(active.w >= 0)
        {
            _d_EE(vertexes[active.x],
                  vertexes[active.y],
                  vertexes[active.z],
                  vertexes[active.w],
                  val);
        }
        else
        {
            _d_EE(vertexes[active.x],
                  vertexes[active.y],
                  vertexes[active.z],
                  vertexes[-active.w - 1],
                  val);
        }
    }
    else
    {
        if(active.z < 0)
        {
            if(active.y < 0)
            {
                _d_PP(vertexes[-active.x - 1], vertexes[-active.y - 1], val);
            }
            else
            {
                _d_PP(vertexes[-active.x - 1], vertexes[active.y], val);
            }
        }
        else if(active.w < 0)
        {
            if(active.y < 0)
            {
                _d_PE(vertexes[-active.x - 1],
                      vertexes[-active.y - 1],
                      vertexes[active.z],
                      val);
            }
            else
            {
                _d_PE(
                    vertexes[-active.x - 1], vertexes[active.y], vertexes[active.z], val);
            }
        }
        else
        {
            _d_PT(vertexes[-active.x - 1],
                  vertexes[active.y],
                  vertexes[active.z],
                  vertexes[active.w],
                  val);
        }
    }
    return val;
}

// _computeInjectiveStepSize_3d function
__device__ inline double _computeInjectiveStepSize_3d(const double3*  verts,
                                               const double3*  mv,
                                               const uint32_t& v0,
                                               const uint32_t& v1,
                                               const uint32_t& v2,
                                               const uint32_t& v3,
                                               double          ratio,
                                               double          errorRate)
{

    double x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4;
    double p1, p2, p3, p4, q1, q2, q3, q4, r1, r2, r3, r4;
    double a, b, c, d, t;


    x1 = verts[v0].x;
    x2 = verts[v1].x;
    x3 = verts[v2].x;
    x4 = verts[v3].x;

    y1 = verts[v0].y;
    y2 = verts[v1].y;
    y3 = verts[v2].y;
    y4 = verts[v3].y;

    z1 = verts[v0].z;
    z2 = verts[v1].z;
    z3 = verts[v2].z;
    z4 = verts[v3].z;

    int _3Fii0 = v0 * 3;
    int _3Fii1 = v1 * 3;
    int _3Fii2 = v2 * 3;
    int _3Fii3 = v3 * 3;

    p1 = -mv[v0].x;
    p2 = -mv[v1].x;
    p3 = -mv[v2].x;
    p4 = -mv[v3].x;

    q1 = -mv[v0].y;
    q2 = -mv[v1].y;
    q3 = -mv[v2].y;
    q4 = -mv[v3].y;

    r1 = -mv[v0].z;
    r2 = -mv[v1].z;
    r3 = -mv[v2].z;
    r4 = -mv[v3].z;

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
    d = (ratio)
        * (x1 * z2 * y3 - x1 * y2 * z3 + y1 * x2 * z3 - y1 * z2 * x3 - z1 * x2 * y3
           + z1 * y2 * x3 + x1 * y2 * z4 - x1 * z2 * y4 - y1 * x2 * z4 + y1 * z2 * x4
           + z1 * x2 * y4 - z1 * y2 * x4 - x1 * y3 * z4 + x1 * z3 * y4 + y1 * x3 * z4
           - y1 * z3 * x4 - z1 * x3 * y4 + z1 * y3 * x4 + x2 * y3 * z4 - x2 * z3 * y4
           - y2 * x3 * z4 + y2 * z3 * x4 + z2 * x3 * y4 - z2 * y3 * x4);


    //printf("a b c d:   %f  %f  %f  %f     %f     %f,    id0, id1, id2, id3:  %d  %d  %d  %d\n", a, b, c, d, ratio, errorRate, v0, v1, v2, v3);
    if(abs(a) <= errorRate /** errorRate*/)
    {
        if(abs(b) <= errorRate /** errorRate*/)
        {
            if(false && abs(c) <= errorRate)
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
        //double results[3];
        //int number = 0;
        //__GEIGEN__::__NewtonSolverForCubicEquation(a, b, c, d, results, number, errorRate);

        //t = 1;
        //for (int index = 0;index < number;index++) {
        //    if (results[index] > 0 && results[index] < t) {
        //        t = results[index];
        //    }
        //}
        //zs::complex<double> i(0, 1);
        //zs::complex<double> delta0(b * b - 3 * a * c, 0);
        //zs::complex<double> delta1(2 * b * b * b - 9 * a * b * c + 27 * a * a * d, 0);
        //zs::complex<double> C =
        //    pow((delta1 + sqrt(delta1 * delta1 - 4.0 * delta0 * delta0 * delta0)) / 2.0,
        //        1.0 / 3.0);
        //if(abs(C) == 0.0)
        //{
        //    // a corner case listed by wikipedia found by our collaborate from another project
        //    C = pow((delta1 - sqrt(delta1 * delta1 - 4.0 * delta0 * delta0 * delta0)) / 2.0,
        //            1.0 / 3.0);
        //}

        //zs::complex<double> u2 = (-1.0 + sqrt(3.0) * i) / 2.0;
        //zs::complex<double> u3 = (-1.0 - sqrt(3.0) * i) / 2.0;

        //zs::complex<double> t1 = (b + C + delta0 / C) / (-3.0 * a);
        //zs::complex<double> t2 = (b + u2 * C + delta0 / (u2 * C)) / (-3.0 * a);
        //zs::complex<double> t3 = (b + u3 * C + delta0 / (u3 * C)) / (-3.0 * a);
        //t                      = -1;
        //if((abs(imag(t1)) < errorRate /** errorRate*/) && (real(t1) > 0))
        //    t = real(t1);
        //if((abs(imag(t2)) < errorRate /** errorRate*/) && (real(t2) > 0)
        //   && ((real(t2) < t) || (t < 0)))
        //    t = real(t2);
        //if((abs(imag(t3)) < errorRate /** errorRate*/) && (real(t3) > 0)
        //   && ((real(t3) < t) || (t < 0)))
        //    t = real(t3);
    }
    if(t <= 0)
        t = 1;
    return t;
}


#endif // GIPC_DEVICE_FUNCTIONS_CUH
