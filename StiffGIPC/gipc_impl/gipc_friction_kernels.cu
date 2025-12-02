//
// gipc_friction_kernels.cu
// GIPC - Friction Gradient and Hessian Kernels
//
// These kernels compute friction forces for contact dynamics.
//
// created by Kemeng Huang on 2022/12/01
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#include <cuda_runtime.h>
#include "gipc_utils.cuh"
#include "Eigen/Eigen"
using namespace Eigen;
#include "../gpu_eigen_libs.cuh"
#include "../FrictionUtils.cuh"
#include "../mlbvh.cuh"

// Local definitions for matrix offsets
constexpr int M12_Off = 16;
constexpr int M9_Off = 9;
constexpr int M6_Off = 4;

// ============================================================================
// Device Helper Functions
// ============================================================================

__device__ double __cal_Friction_gd_energy(const double3* _vertexes,
                                           const double3* _o_vertexes,
                                           const double3* _normal,
                                           uint32_t       gidx,
                                           double         dt,
                                           double         lastH,
                                           double         eps)
{

    double3 normal = *_normal;
    double3 Vdiff  = __GEIGEN__::__minus(_vertexes[gidx], _o_vertexes[gidx]);
    double3 VProj  = __GEIGEN__::__minus(
        Vdiff, __GEIGEN__::__s_vec_multiply(normal, __GEIGEN__::__v_vec_dot(Vdiff, normal)));
    double VProjMag2 = __GEIGEN__::__squaredNorm(VProj);
    if(VProjMag2 > eps * eps)
    {
        return lastH * (sqrt(VProjMag2) - eps * 0.5);
    }
    else
    {
        return lastH * VProjMag2 / eps * 0.5;
    }
}


__device__ double __cal_Friction_energy(const double3*         _vertexes,
                                        const double3*         _o_vertexes,
                                        int4                   MMCVIDI,
                                        double                 dt,
                                        double2                distCoord,
                                        __GEIGEN__::Matrix3x2d tanBasis,
                                        double                 lastH,
                                        double                 fricDHat,
                                        double                 eps)
{
    double3 relDX3D;
    if(MMCVIDI.x >= 0)
    {
        if(MMCVIDI.w >= 0)
        {
            Friction::computeRelDX_EE(
                __GEIGEN__::__minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _o_vertexes[MMCVIDI.w]),
                distCoord.x,
                distCoord.y,
                relDX3D);
        }
    }
    else
    {
        int v0I = -MMCVIDI.x - 1;
        if(MMCVIDI.z < 0)
        {
            if(MMCVIDI.y >= 0)
            {
                Friction::computeRelDX_PP(
                    __GEIGEN__::__minus(_vertexes[v0I], _o_vertexes[v0I]),
                    __GEIGEN__::__minus(_vertexes[MMCVIDI.y],
                                        _o_vertexes[MMCVIDI.y]),
                    relDX3D);
            }
        }
        else if(MMCVIDI.w < 0)
        {
            if(MMCVIDI.y >= 0)
            {
                Friction::computeRelDX_PE(
                    __GEIGEN__::__minus(_vertexes[v0I], _o_vertexes[v0I]),
                    __GEIGEN__::__minus(_vertexes[MMCVIDI.y],
                                        _o_vertexes[MMCVIDI.y]),
                    __GEIGEN__::__minus(_vertexes[MMCVIDI.z],
                                        _o_vertexes[MMCVIDI.z]),
                    distCoord.x,
                    relDX3D);
            }
        }
        else
        {
            Friction::computeRelDX_PT(
                __GEIGEN__::__minus(_vertexes[v0I], _o_vertexes[v0I]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _o_vertexes[MMCVIDI.w]),
                distCoord.x,
                distCoord.y,
                relDX3D);
        }
    }
    __GEIGEN__::Matrix2x3d tB_T = __GEIGEN__::__Transpose3x2(tanBasis);
    double                 relDXSqNorm =
        __GEIGEN__::__squaredNorm(__GEIGEN__::__M2x3_v3_multiply(tB_T, relDX3D));
    if(relDXSqNorm > fricDHat)
    {
        return lastH * sqrt(relDXSqNorm);
    }
    else
    {
        double f0;
        Friction::f0_SF(relDXSqNorm, eps, f0);
        return lastH * f0;
    }
}

// ============================================================================
// Friction Hessian Kernels
// ============================================================================

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

    //H3x3[idx]    = H_vI;
    //D1Index[idx] = gidx;

    write_triplet<3, 3>(triplet_values, row_ids, col_ids, &gidx, H_vI.m, global_offset + idx);
}

__global__ void _calFrictionHessian(const double3*          _vertexes,
                                    const double3*          _o_vertexes,
                                    const int4*             _last_collisionPair,
                                    Eigen::Matrix3d*        triplet_values,
                                    int*                    row_ids,
                                    int*                    col_ids,
                                    uint32_t*               _cpNum,
                                    int                     number,
                                    double                  dt,
                                    double2*                distCoord,
                                    __GEIGEN__::Matrix3x2d* tanBasis,
                                    double                  eps2,
                                    double*                 lastH,
                                    double                  coef,
                                    int                     cd_offset4,
                                    int                     cd_offset3,
                                    int                     cd_offset2,
                                    int                     f_offset4,
                                    int                     f_offset3,
                                    int                     f_offset2)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    int4    MMCVIDI = _last_collisionPair[idx];
    double  eps     = sqrt(eps2);
    double3 relDX3D;
    int global_offset = cd_offset4 * M12_Off + cd_offset3 * M9_Off + cd_offset2 * M6_Off;
    if(MMCVIDI.x >= 0)
    {
        Friction::computeRelDX_EE(
            __GEIGEN__::__minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
            __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
            __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
            __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _o_vertexes[MMCVIDI.w]),
            distCoord[idx].x,
            distCoord[idx].y,
            relDX3D);


        __GEIGEN__::Matrix2x3d tB_T = __GEIGEN__::__Transpose3x2(tanBasis[idx]);
        double2 relDX       = __GEIGEN__::__M2x3_v3_multiply(tB_T, relDX3D);
        double  relDXSqNorm = __GEIGEN__::__squaredNorm(relDX);
        double  relDXNorm   = sqrt(relDXSqNorm);
        __GEIGEN__::Matrix12x2d T;
        Friction::computeT_EE(tanBasis[idx], distCoord[idx].x, distCoord[idx].y, T);
        __GEIGEN__::Matrix2x2d M2;
        if(relDXSqNorm > eps2)
        {
            __GEIGEN__::__set_Mat_identity(M2);
            M2.m[0][0] /= relDXNorm;
            M2.m[1][1] /= relDXNorm;
            M2 = __GEIGEN__::__Mat2x2_minus(
                M2,
                __GEIGEN__::__s_Mat2x2_multiply(__GEIGEN__::__v2_vec2_toMat2x2(relDX, relDX),
                                                1 / (relDXSqNorm * relDXNorm)));
        }
        else
        {
            double f1_div_relDXNorm;
            Friction::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
            double f2;
            Friction::f2_SF(relDXSqNorm, eps, f2);
            if(f2 != f1_div_relDXNorm && relDXSqNorm)
            {

                __GEIGEN__::__set_Mat_identity(M2);
                M2.m[0][0] *= f1_div_relDXNorm;
                M2.m[1][1] *= f1_div_relDXNorm;
                M2 = __GEIGEN__::__Mat2x2_minus(
                    M2,
                    __GEIGEN__::__s_Mat2x2_multiply(__GEIGEN__::__v2_vec2_toMat2x2(relDX, relDX),
                                                    (f1_div_relDXNorm - f2) / relDXSqNorm));
            }
            else
            {
                __GEIGEN__::__set_Mat_identity(M2);
                M2.m[0][0] *= f1_div_relDXNorm;
                M2.m[1][1] *= f1_div_relDXNorm;
            }
        }

        __GEIGEN__::Matrix2x2d projH;

        Matrix2d F_mat2;
        F_mat2 << M2.m[0][0], M2.m[0][1], M2.m[1][0], M2.m[1][1];
        makePDGeneral<double, 2>(F_mat2);
        projH.m[0][0] = F_mat2(0, 0);
        projH.m[0][1] = F_mat2(0, 1);
        projH.m[1][0] = F_mat2(1, 0);
        projH.m[1][1] = F_mat2(1, 1);


        __GEIGEN__::Matrix12x2d TM2 = __GEIGEN__::__M12x2_M2x2_Multiply(T, projH);

        __GEIGEN__::Matrix12x12d HessianBlock =
            __GEIGEN__::__s_M12x12_Multiply(__M12x2_M12x2T_Multiply(TM2, T),
                                            coef * lastH[idx]);
        int Hidx   = atomicAdd(_cpNum + 4, 1);
        int offset = global_offset + Hidx * M12_Off;
        //Hidx += cd_offset4;
        //H12x12[Hidx]  = HessianBlock;
        uint4 global_index = make_uint4(MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);
        //D4Index[Hidx] = global_index;

        write_triplet<12, 12>(
            triplet_values, row_ids, col_ids, &(global_index.x), HessianBlock.m, offset);
    }
    else
    {
        int v0I = -MMCVIDI.x - 1;
        if(MMCVIDI.z < 0)
        {

            MMCVIDI.x = v0I;
            Friction::computeRelDX_PP(
                __GEIGEN__::__minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
                relDX3D);

            __GEIGEN__::Matrix2x3d tB_T = __GEIGEN__::__Transpose3x2(tanBasis[idx]);
            double2 relDX       = __GEIGEN__::__M2x3_v3_multiply(tB_T, relDX3D);
            double  relDXSqNorm = __GEIGEN__::__squaredNorm(relDX);
            double  relDXNorm   = sqrt(relDXSqNorm);
            __GEIGEN__::Matrix6x2d T;
            Friction::computeT_PP(tanBasis[idx], T);
            __GEIGEN__::Matrix2x2d M2;
            if(relDXSqNorm > eps2)
            {
                __GEIGEN__::__set_Mat_identity(M2);
                M2.m[0][0] /= relDXNorm;
                M2.m[1][1] /= relDXNorm;
                M2 = __GEIGEN__::__Mat2x2_minus(
                    M2,
                    __GEIGEN__::__s_Mat2x2_multiply(__GEIGEN__::__v2_vec2_toMat2x2(relDX, relDX),
                                                    1 / (relDXSqNorm * relDXNorm)));
            }
            else
            {
                double f1_div_relDXNorm;
                Friction::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
                double f2;
                Friction::f2_SF(relDXSqNorm, eps, f2);
                if(f2 != f1_div_relDXNorm && relDXSqNorm)
                {

                    __GEIGEN__::__set_Mat_identity(M2);
                    M2.m[0][0] *= f1_div_relDXNorm;
                    M2.m[1][1] *= f1_div_relDXNorm;
                    M2 = __GEIGEN__::__Mat2x2_minus(
                        M2,
                        __GEIGEN__::__s_Mat2x2_multiply(
                            __GEIGEN__::__v2_vec2_toMat2x2(relDX, relDX),
                            (f1_div_relDXNorm - f2) / relDXSqNorm));
                }
                else
                {
                    __GEIGEN__::__set_Mat_identity(M2);
                    M2.m[0][0] *= f1_div_relDXNorm;
                    M2.m[1][1] *= f1_div_relDXNorm;
                }
            }
            __GEIGEN__::Matrix2x2d projH;
            Matrix2d               F_mat2;
            F_mat2 << M2.m[0][0], M2.m[0][1], M2.m[1][0], M2.m[1][1];
            makePDGeneral<double, 2>(F_mat2);
            projH.m[0][0] = F_mat2(0, 0);
            projH.m[0][1] = F_mat2(0, 1);
            projH.m[1][0] = F_mat2(1, 0);
            projH.m[1][1] = F_mat2(1, 1);

            __GEIGEN__::Matrix6x2d TM2 = __GEIGEN__::__M6x2_M2x2_Multiply(T, projH);

            __GEIGEN__::Matrix6x6d HessianBlock =
                __GEIGEN__::__s_M6x6_Multiply(__M6x2_M6x2T_Multiply(TM2, T),
                                              coef * lastH[idx]);

            int Hidx   = atomicAdd(_cpNum + 2, 1);
            int offset = global_offset + f_offset4 * M12_Off
                         + f_offset3 * M9_Off + Hidx * M6_Off;
            //Hidx += cd_offset2;
            //H6x6[Hidx]    = HessianBlock;
            uint2 global_index = make_uint2(MMCVIDI.x, MMCVIDI.y);
            //D2Index[Hidx]      = global_index;


            write_triplet<6, 6>(triplet_values,
                                row_ids,
                                col_ids,
                                &(global_index.x),
                                HessianBlock.m,
                                offset);
        }
        else if(MMCVIDI.w < 0)
        {

            MMCVIDI.x = v0I;
            Friction::computeRelDX_PE(
                __GEIGEN__::__minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
                distCoord[idx].x,
                relDX3D);

            __GEIGEN__::Matrix2x3d tB_T = __GEIGEN__::__Transpose3x2(tanBasis[idx]);
            double2 relDX       = __GEIGEN__::__M2x3_v3_multiply(tB_T, relDX3D);
            double  relDXSqNorm = __GEIGEN__::__squaredNorm(relDX);
            double  relDXNorm   = sqrt(relDXSqNorm);
            __GEIGEN__::Matrix9x2d T;
            Friction::computeT_PE(tanBasis[idx], distCoord[idx].x, T);
            __GEIGEN__::Matrix2x2d M2;
            if(relDXSqNorm > eps2)
            {
                __GEIGEN__::__set_Mat_identity(M2);
                M2.m[0][0] /= relDXNorm;
                M2.m[1][1] /= relDXNorm;
                M2 = __GEIGEN__::__Mat2x2_minus(
                    M2,
                    __GEIGEN__::__s_Mat2x2_multiply(__GEIGEN__::__v2_vec2_toMat2x2(relDX, relDX),
                                                    1 / (relDXSqNorm * relDXNorm)));
            }
            else
            {
                double f1_div_relDXNorm;
                Friction::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
                double f2;
                Friction::f2_SF(relDXSqNorm, eps, f2);
                if(f2 != f1_div_relDXNorm && relDXSqNorm)
                {

                    __GEIGEN__::__set_Mat_identity(M2);
                    M2.m[0][0] *= f1_div_relDXNorm;
                    M2.m[1][1] *= f1_div_relDXNorm;
                    M2 = __GEIGEN__::__Mat2x2_minus(
                        M2,
                        __GEIGEN__::__s_Mat2x2_multiply(
                            __GEIGEN__::__v2_vec2_toMat2x2(relDX, relDX),
                            (f1_div_relDXNorm - f2) / relDXSqNorm));
                }
                else
                {
                    __GEIGEN__::__set_Mat_identity(M2);
                    M2.m[0][0] *= f1_div_relDXNorm;
                    M2.m[1][1] *= f1_div_relDXNorm;
                }
            }
            __GEIGEN__::Matrix2x2d projH;
            Matrix2d               F_mat2;
            F_mat2 << M2.m[0][0], M2.m[0][1], M2.m[1][0], M2.m[1][1];
            makePDGeneral<double, 2>(F_mat2);
            projH.m[0][0] = F_mat2(0, 0);
            projH.m[0][1] = F_mat2(0, 1);
            projH.m[1][0] = F_mat2(1, 0);
            projH.m[1][1] = F_mat2(1, 1);

            __GEIGEN__::Matrix9x2d TM2 = __GEIGEN__::__M9x2_M2x2_Multiply(T, projH);

            __GEIGEN__::Matrix9x9d HessianBlock =
                __GEIGEN__::__s_M9x9_Multiply(__M9x2_M9x2T_Multiply(TM2, T),
                                              coef * lastH[idx]);
            int Hidx   = atomicAdd(_cpNum + 3, 1);
            int offset = global_offset + f_offset4 * M12_Off + Hidx * M9_Off;
            //Hidx += cd_offset3;
            //H9x9[Hidx]    = HessianBlock;
            uint3 global_index = make_uint3(v0I, MMCVIDI.y, MMCVIDI.z);
            //D3Index[Hidx]      = global_index;


            write_triplet<9, 9>(triplet_values,
                                row_ids,
                                col_ids,
                                &(global_index.x),
                                HessianBlock.m,
                                offset);
        }
        else
        {
            MMCVIDI.x = v0I;
            Friction::computeRelDX_PT(
                __GEIGEN__::__minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _o_vertexes[MMCVIDI.w]),
                distCoord[idx].x,
                distCoord[idx].y,
                relDX3D);


            __GEIGEN__::Matrix2x3d tB_T = __GEIGEN__::__Transpose3x2(tanBasis[idx]);
            double2 relDX       = __GEIGEN__::__M2x3_v3_multiply(tB_T, relDX3D);
            double  relDXSqNorm = __GEIGEN__::__squaredNorm(relDX);
            double  relDXNorm   = sqrt(relDXSqNorm);
            __GEIGEN__::Matrix12x2d T;
            Friction::computeT_PT(
                tanBasis[idx], distCoord[idx].x, distCoord[idx].y, T);
            __GEIGEN__::Matrix2x2d M2;
            if(relDXSqNorm > eps2)
            {
                __GEIGEN__::__set_Mat_identity(M2);
                M2.m[0][0] /= relDXNorm;
                M2.m[1][1] /= relDXNorm;
                M2 = __GEIGEN__::__Mat2x2_minus(
                    M2,
                    __GEIGEN__::__s_Mat2x2_multiply(__GEIGEN__::__v2_vec2_toMat2x2(relDX, relDX),
                                                    1 / (relDXSqNorm * relDXNorm)));
            }
            else
            {
                double f1_div_relDXNorm;
                Friction::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
                double f2;
                Friction::f2_SF(relDXSqNorm, eps, f2);
                if(f2 != f1_div_relDXNorm && relDXSqNorm)
                {

                    __GEIGEN__::__set_Mat_identity(M2);
                    M2.m[0][0] *= f1_div_relDXNorm;
                    M2.m[1][1] *= f1_div_relDXNorm;
                    M2 = __GEIGEN__::__Mat2x2_minus(
                        M2,
                        __GEIGEN__::__s_Mat2x2_multiply(
                            __GEIGEN__::__v2_vec2_toMat2x2(relDX, relDX),
                            (f1_div_relDXNorm - f2) / relDXSqNorm));
                }
                else
                {
                    __GEIGEN__::__set_Mat_identity(M2);
                    M2.m[0][0] *= f1_div_relDXNorm;
                    M2.m[1][1] *= f1_div_relDXNorm;
                }
            }
            __GEIGEN__::Matrix2x2d projH;
            Matrix2d               F_mat2;
            F_mat2 << M2.m[0][0], M2.m[0][1], M2.m[1][0], M2.m[1][1];
            makePDGeneral<double, 2>(F_mat2);
            projH.m[0][0] = F_mat2(0, 0);
            projH.m[0][1] = F_mat2(0, 1);
            projH.m[1][0] = F_mat2(1, 0);
            projH.m[1][1] = F_mat2(1, 1);

            __GEIGEN__::Matrix12x2d TM2 = __GEIGEN__::__M12x2_M2x2_Multiply(T, projH);

            __GEIGEN__::Matrix12x12d HessianBlock =
                __GEIGEN__::__s_M12x12_Multiply(__M12x2_M12x2T_Multiply(TM2, T),
                                                coef * lastH[idx]);
            int Hidx   = atomicAdd(_cpNum + 4, 1);
            int offset = global_offset + Hidx * M12_Off;
            //Hidx += cd_offset4;
            //H12x12[Hidx]  = HessianBlock;
            uint4 global_index = make_uint4(v0I, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);
            //D4Index[Hidx] = global_index;


            write_triplet<12, 12>(triplet_values,
                                  row_ids,
                                  col_ids,
                                  &(global_index.x),
                                  HessianBlock.m,
                                  offset);
        }
    }
}

// ============================================================================
// Friction Gradient Kernels
// ============================================================================

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
        /*atomicAdd(&(_gradient[gidx].x), gdf.x);
        atomicAdd(&(_gradient[gidx].y), gdf.y);
        atomicAdd(&(_gradient[gidx].z), gdf.z);*/
        _gradient[gidx] = __GEIGEN__::__add(_gradient[gidx], gdf);
    }
    else
    {
        double3 gdf = __GEIGEN__::__s_vec_multiply(VProj, coef * lastH[idx] / eps);
        /*atomicAdd(&(_gradient[gidx].x), gdf.x);
        atomicAdd(&(_gradient[gidx].y), gdf.y);
        atomicAdd(&(_gradient[gidx].z), gdf.z);*/
        _gradient[gidx] = __GEIGEN__::__add(_gradient[gidx], gdf);
    }
}

__global__ void _calFrictionGradient(const double3*    _vertexes,
                                     const double3*    _o_vertexes,
                                     const int4* _last_collisionPair,
                                     double3*          _gradient,
                                     int               number,
                                     double            dt,
                                     double2*          distCoord,
                                     __GEIGEN__::Matrix3x2d* tanBasis,
                                     double                  eps2,
                                     double*                 lastH,
                                     double                  coef)
{
    double eps = std::sqrt(eps2);
    int    idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    int4    MMCVIDI = _last_collisionPair[idx];
    double3 relDX3D;
    if(MMCVIDI.x >= 0)
    {
        Friction::computeRelDX_EE(
            __GEIGEN__::__minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
            __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
            __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
            __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _o_vertexes[MMCVIDI.w]),
            distCoord[idx].x,
            distCoord[idx].y,
            relDX3D);

        __GEIGEN__::Matrix2x3d tB_T = __GEIGEN__::__Transpose3x2(tanBasis[idx]);
        double2 relDX       = __GEIGEN__::__M2x3_v3_multiply(tB_T, relDX3D);
        double  relDXSqNorm = __GEIGEN__::__squaredNorm(relDX);
        if(relDXSqNorm > eps2)
        {
            relDX = __GEIGEN__::__s_vec_multiply(relDX, 1.0 / sqrt(relDXSqNorm));
        }
        else
        {
            double f1_div_relDXNorm;
            Friction::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
            relDX = __GEIGEN__::__s_vec_multiply(relDX, f1_div_relDXNorm);
        }
        __GEIGEN__::Vector12 TTTDX;
        Friction::liftRelDXTanToMesh_EE(
            relDX, tanBasis[idx], distCoord[idx].x, distCoord[idx].y, TTTDX);
        TTTDX = __GEIGEN__::__s_vec12_multiply(TTTDX, lastH[idx] * coef);
        {
            atomicAdd(&(_gradient[MMCVIDI.x].x), TTTDX.v[0]);
            atomicAdd(&(_gradient[MMCVIDI.x].y), TTTDX.v[1]);
            atomicAdd(&(_gradient[MMCVIDI.x].z), TTTDX.v[2]);
            atomicAdd(&(_gradient[MMCVIDI.y].x), TTTDX.v[3]);
            atomicAdd(&(_gradient[MMCVIDI.y].y), TTTDX.v[4]);
            atomicAdd(&(_gradient[MMCVIDI.y].z), TTTDX.v[5]);
            atomicAdd(&(_gradient[MMCVIDI.z].x), TTTDX.v[6]);
            atomicAdd(&(_gradient[MMCVIDI.z].y), TTTDX.v[7]);
            atomicAdd(&(_gradient[MMCVIDI.z].z), TTTDX.v[8]);
            atomicAdd(&(_gradient[MMCVIDI.w].x), TTTDX.v[9]);
            atomicAdd(&(_gradient[MMCVIDI.w].y), TTTDX.v[10]);
            atomicAdd(&(_gradient[MMCVIDI.w].z), TTTDX.v[11]);
        }
    }
    else
    {
        int v0I = -MMCVIDI.x - 1;
        if(MMCVIDI.z < 0)
        {
            MMCVIDI.x = v0I;

            Friction::computeRelDX_PP(
                __GEIGEN__::__minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
                relDX3D);

            __GEIGEN__::Matrix2x3d tB_T = __GEIGEN__::__Transpose3x2(tanBasis[idx]);
            double2 relDX       = __GEIGEN__::__M2x3_v3_multiply(tB_T, relDX3D);
            double  relDXSqNorm = __GEIGEN__::__squaredNorm(relDX);
            if(relDXSqNorm > eps2)
            {
                relDX = __GEIGEN__::__s_vec_multiply(relDX, 1.0 / sqrt(relDXSqNorm));
            }
            else
            {
                double f1_div_relDXNorm;
                Friction::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
                relDX = __GEIGEN__::__s_vec_multiply(relDX, f1_div_relDXNorm);
            }

            __GEIGEN__::Vector6 TTTDX;
            Friction::liftRelDXTanToMesh_PP(relDX, tanBasis[idx], TTTDX);
            TTTDX = __GEIGEN__::__s_vec6_multiply(TTTDX, lastH[idx] * coef);
            {
                atomicAdd(&(_gradient[MMCVIDI.x].x), TTTDX.v[0]);
                atomicAdd(&(_gradient[MMCVIDI.x].y), TTTDX.v[1]);
                atomicAdd(&(_gradient[MMCVIDI.x].z), TTTDX.v[2]);
                atomicAdd(&(_gradient[MMCVIDI.y].x), TTTDX.v[3]);
                atomicAdd(&(_gradient[MMCVIDI.y].y), TTTDX.v[4]);
                atomicAdd(&(_gradient[MMCVIDI.y].z), TTTDX.v[5]);
            }
        }
        else if(MMCVIDI.w < 0)
        {
            MMCVIDI.x = v0I;
            Friction::computeRelDX_PE(
                __GEIGEN__::__minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
                distCoord[idx].x,
                relDX3D);

            __GEIGEN__::Matrix2x3d tB_T = __GEIGEN__::__Transpose3x2(tanBasis[idx]);
            double2 relDX       = __GEIGEN__::__M2x3_v3_multiply(tB_T, relDX3D);
            double  relDXSqNorm = __GEIGEN__::__squaredNorm(relDX);
            if(relDXSqNorm > eps2)
            {
                relDX = __GEIGEN__::__s_vec_multiply(relDX, 1.0 / sqrt(relDXSqNorm));
            }
            else
            {
                double f1_div_relDXNorm;
                Friction::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
                relDX = __GEIGEN__::__s_vec_multiply(relDX, f1_div_relDXNorm);
            }
            __GEIGEN__::Vector9 TTTDX;
            Friction::liftRelDXTanToMesh_PE(relDX, tanBasis[idx], distCoord[idx].x, TTTDX);
            TTTDX = __GEIGEN__::__s_vec9_multiply(TTTDX, lastH[idx] * coef);
            {
                atomicAdd(&(_gradient[MMCVIDI.x].x), TTTDX.v[0]);
                atomicAdd(&(_gradient[MMCVIDI.x].y), TTTDX.v[1]);
                atomicAdd(&(_gradient[MMCVIDI.x].z), TTTDX.v[2]);
                atomicAdd(&(_gradient[MMCVIDI.y].x), TTTDX.v[3]);
                atomicAdd(&(_gradient[MMCVIDI.y].y), TTTDX.v[4]);
                atomicAdd(&(_gradient[MMCVIDI.y].z), TTTDX.v[5]);
                atomicAdd(&(_gradient[MMCVIDI.z].x), TTTDX.v[6]);
                atomicAdd(&(_gradient[MMCVIDI.z].y), TTTDX.v[7]);
                atomicAdd(&(_gradient[MMCVIDI.z].z), TTTDX.v[8]);
            }
        }
        else
        {
            MMCVIDI.x = v0I;
            Friction::computeRelDX_PT(
                __GEIGEN__::__minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _o_vertexes[MMCVIDI.w]),
                distCoord[idx].x,
                distCoord[idx].y,
                relDX3D);

            __GEIGEN__::Matrix2x3d tB_T = __GEIGEN__::__Transpose3x2(tanBasis[idx]);
            double2 relDX = __GEIGEN__::__M2x3_v3_multiply(tB_T, relDX3D);

            double relDXSqNorm = __GEIGEN__::__squaredNorm(relDX);
            if(relDXSqNorm > eps2)
            {
                relDX = __GEIGEN__::__s_vec_multiply(relDX, 1.0 / sqrt(relDXSqNorm));
            }
            else
            {
                double f1_div_relDXNorm;
                Friction::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
                relDX = __GEIGEN__::__s_vec_multiply(relDX, f1_div_relDXNorm);
            }
            __GEIGEN__::Vector12 TTTDX;
            Friction::liftRelDXTanToMesh_PT(
                relDX, tanBasis[idx], distCoord[idx].x, distCoord[idx].y, TTTDX);
            TTTDX = __GEIGEN__::__s_vec12_multiply(TTTDX, lastH[idx] * coef);

            atomicAdd(&(_gradient[MMCVIDI.x].x), TTTDX.v[0]);
            atomicAdd(&(_gradient[MMCVIDI.x].y), TTTDX.v[1]);
            atomicAdd(&(_gradient[MMCVIDI.x].z), TTTDX.v[2]);
            atomicAdd(&(_gradient[MMCVIDI.y].x), TTTDX.v[3]);
            atomicAdd(&(_gradient[MMCVIDI.y].y), TTTDX.v[4]);
            atomicAdd(&(_gradient[MMCVIDI.y].z), TTTDX.v[5]);
            atomicAdd(&(_gradient[MMCVIDI.z].x), TTTDX.v[6]);
            atomicAdd(&(_gradient[MMCVIDI.z].y), TTTDX.v[7]);
            atomicAdd(&(_gradient[MMCVIDI.z].z), TTTDX.v[8]);
            atomicAdd(&(_gradient[MMCVIDI.w].x), TTTDX.v[9]);
            atomicAdd(&(_gradient[MMCVIDI.w].y), TTTDX.v[10]);
            atomicAdd(&(_gradient[MMCVIDI.w].z), TTTDX.v[11]);
        }
    }
}

// ============================================================================
// Friction Energy Reduction Kernels
// ============================================================================

__global__ void _getFrictionEnergy_Reduction_3D(double*        squeue,
                                                const double3* vertexes,
                                                const double3* o_vertexes,
                                                const int4*    _collisionPair,
                                                int            cpNum,
                                                double         dt,
                                                const double2* distCoord,
                                                const __GEIGEN__::Matrix3x2d* tanBasis,
                                                const double* lastH,
                                                double        fricDHat,
                                                double        eps

)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];
    int                      numbers = cpNum;
    if(idx >= numbers)
        return;

    double temp = __cal_Friction_energy(
        vertexes, o_vertexes, _collisionPair[idx], dt, distCoord[idx], tanBasis[idx], lastH[idx], fricDHat, eps);

    int    warpTid = threadIdx.x % 32;
    int    warpId  = (threadIdx.x >> 5);
    double nextTp;
    int    warpNum;
    //int tidNum = 32;
    if(blockIdx.x == gridDim.x - 1)
    {
        //tidNum = numbers - idof;
        warpNum = ((numbers - idof + 31) >> 5);
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
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];
        //	warpNum = ((tidNum + 31) >> 5);
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            temp += __shfl_down_sync(0xffffffff, temp, i);
        }
    }
    if(threadIdx.x == 0)
    {
        squeue[blockIdx.x] = temp;
    }
}

__global__ void _getFrictionEnergy_gd_Reduction_3D(double*        squeue,
                                                   const double3* vertexes,
                                                   const double3* o_vertexes,
                                                   const double3* _normal,
                                                   const uint32_t* _collisionPair_gd,
                                                   int           gpNum,
                                                   double        dt,
                                                   const double* lastH,
                                                   double        eps

)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];
    int                      numbers = gpNum;
    if(idx >= numbers)
        return;

    double temp = __cal_Friction_gd_energy(
        vertexes, o_vertexes, _normal, _collisionPair_gd[idx], dt, lastH[idx], eps);

    int    warpTid = threadIdx.x % 32;
    int    warpId  = (threadIdx.x >> 5);
    double nextTp;
    int    warpNum;
    //int tidNum = 32;
    if(blockIdx.x == gridDim.x - 1)
    {
        //tidNum = numbers - idof;
        warpNum = ((numbers - idof + 31) >> 5);
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
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];
        //	warpNum = ((tidNum + 31) >> 5);
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            temp += __shfl_down_sync(0xffffffff, temp, i);
        }
    }
    if(threadIdx.x == 0)
    {
        squeue[blockIdx.x] = temp;
    }
}

// ============================================================================
// Friction LastH Computation Kernels
// ============================================================================

__global__ void _calFrictionLastH_gd(const double3* _vertexes,
                                     const double*  g_offset,
                                     const double3* g_normal,
                                     const uint32_t* _collisionPair_environment,
                                     double*   lambda_lastH_gd,
                                     uint32_t* _collisionPair_last_gd,
                                     double    dHat,
                                     double    Kappa,
                                     int       number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;

    double3 normal = *g_normal;
    int     gidx   = _collisionPair_environment[idx];
    double  dist = __GEIGEN__::__v_vec_dot(normal, _vertexes[gidx]) - *g_offset;
    double  dist2 = dist * dist;

    double t   = dist2 - dHat;
    double g_b = t * log(dist2 / dHat) * -2.0 - (t * t) / dist2;

    lambda_lastH_gd[idx]        = -Kappa * 2.0 * sqrt(dist2) * g_b;
    _collisionPair_last_gd[idx] = gidx;
}

__global__ void _calFrictionLastH_DistAndTan(const double3*    _vertexes,
                                             const int4* _collisionPair,
                                             double*           lambda_lastH,
                                             double2*          distCoord,
                                             __GEIGEN__::Matrix3x2d* tanBasis,
                                             int4*     _collisionPair_last,
                                             double    dHat,
                                             double    Kappa,
                                             uint32_t* _cpNum_last,
                                             int       number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    int4   MMCVIDI = _collisionPair[idx];
    double dis;
    int    last_index = -1;
    if(MMCVIDI.x >= 0)
    {
        if(MMCVIDI.w >= 0)
        {
            last_index = atomicAdd(_cpNum_last, 1);
            atomicAdd(_cpNum_last + 4, 1);
            _d_EE(_vertexes[MMCVIDI.x],
                  _vertexes[MMCVIDI.y],
                  _vertexes[MMCVIDI.z],
                  _vertexes[MMCVIDI.w],
                  dis);
            Friction::computeClosestPoint_EE(_vertexes[MMCVIDI.x],
                                             _vertexes[MMCVIDI.y],
                                             _vertexes[MMCVIDI.z],
                                             _vertexes[MMCVIDI.w],
                                             distCoord[last_index]);
            Friction::computeTangentBasis_EE(_vertexes[MMCVIDI.x],
                                             _vertexes[MMCVIDI.y],
                                             _vertexes[MMCVIDI.z],
                                             _vertexes[MMCVIDI.w],
                                             tanBasis[last_index]);
        }
    }
    else
    {
        int v0I = -MMCVIDI.x - 1;
        if(MMCVIDI.z < 0)
        {
            if(MMCVIDI.y >= 0)
            {
                last_index = atomicAdd(_cpNum_last, 1);
                atomicAdd(_cpNum_last + 2, 1);
                _d_PP(_vertexes[v0I], _vertexes[MMCVIDI.y], dis);
                distCoord[last_index].x = 0;
                distCoord[last_index].y = 0;
                Friction::computeTangentBasis_PP(
                    _vertexes[v0I], _vertexes[MMCVIDI.y], tanBasis[last_index]);
            }
        }
        else if(MMCVIDI.w < 0)
        {
            if(MMCVIDI.y >= 0)
            {
                last_index = atomicAdd(_cpNum_last, 1);
                atomicAdd(_cpNum_last + 3, 1);
                _d_PE(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], dis);
                Friction::computeClosestPoint_PE(_vertexes[v0I],
                                                 _vertexes[MMCVIDI.y],
                                                 _vertexes[MMCVIDI.z],
                                                 distCoord[last_index].x);
                distCoord[last_index].y = 0;
                Friction::computeTangentBasis_PE(_vertexes[v0I],
                                                 _vertexes[MMCVIDI.y],
                                                 _vertexes[MMCVIDI.z],
                                                 tanBasis[last_index]);
            }
        }
        else
        {
            last_index = atomicAdd(_cpNum_last, 1);
            atomicAdd(_cpNum_last + 4, 1);
            _d_PT(_vertexes[v0I],
                  _vertexes[MMCVIDI.y],
                  _vertexes[MMCVIDI.z],
                  _vertexes[MMCVIDI.w],
                  dis);
            Friction::computeClosestPoint_PT(_vertexes[v0I],
                                             _vertexes[MMCVIDI.y],
                                             _vertexes[MMCVIDI.z],
                                             _vertexes[MMCVIDI.w],
                                             distCoord[last_index]);
            Friction::computeTangentBasis_PT(_vertexes[v0I],
                                             _vertexes[MMCVIDI.y],
                                             _vertexes[MMCVIDI.z],
                                             _vertexes[MMCVIDI.w],
                                             tanBasis[last_index]);
        }
    }
    if(last_index >= 0)
    {
//        double t = dis - dHat;
//        lambda_lastH[last_index] = -Kappa * 2.0 * std::sqrt(dis) * (t * std::log(dis / dHat) * -2.0 - (t * t) / dis);
#if (RANK == 1)
        double t = dis - dHat;
        lambda_lastH[last_index] =
            -Kappa * 2.0 * sqrt(dis) * (t * log(dis / dHat) * -2.0 - (t * t) / dis);
#elif (RANK == 2)
        lambda_lastH[last_index] =
            -Kappa * 2.0 * sqrt(dis)
            * (log(dis / dHat) * log(dis / dHat) * (2 * dis - 2 * dHat)
               + (2 * log(dis / dHat) * (dis - dHat) * (dis - dHat)) / dis);
#endif
        _collisionPair_last[last_index] = _collisionPair[idx];
    }
}
