//
// barrier_gradient_hessian.cu (Simplified: RANK=2, NEWF enabled)
// GIPC - Barrier Gradient/Hessian Kernels Implementation
//
// This file contains the core IPC barrier kernel implementations:
// - _calBarrierHessian
// - _calBarrierGradientAndHessian
// - _calBarrierGradient
//
// These kernels compute gradient and Hessian contributions for collision pairs
// including EE, PT, PE, PP collision types.
//
// Extracted from ipc_barrier.cu for better code organization.
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

//=============================================================================
// Barrier Type Selection
// Set USE_CUBIC_BARRIER to true to use cubic barrier, false for log barrier
//=============================================================================
#define USE_CUBIC_BARRIER true  // Change to true to use cubic barrier

#include "barrier_gradient_hessian.cuh"
#include "barrier_functions.cuh"  // Barrier function definitions
#include "ipc_common.cuh"  // Common helper functions (makePDGeneral_ipc, write_triplet)
#include "GIPC.cuh"
#include "mlbvh.cuh"  // Contains _d_EE, _d_PP, _d_PE, _d_PT, _compute_epx functions
#include "FrictionUtils.cuh"
#include "GIPC_PDerivative.cuh"
#include <cuda_runtime.h>
#include <cmath>

using namespace Eigen;

//=============================================================================
// Barrier Gradient/Hessian Kernel Implementations
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
            // EE (Edge-Edge) non-parallel collision
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
            __GEIGEN__::Vector9 q0;
            q0.v[0] = q0.v[1] = q0.v[2] = q0.v[3] = q0.v[4] = q0.v[5] =
                q0.v[6] = q0.v[7] = 0;
            q0.v[8]               = 1;
            __GEIGEN__::Matrix9x9d H;
            __GEIGEN__::__init_Mat9x9(H, 0);

            // Compute hessian coefficient using barrier function
            double lambda0 = barrier::simple_hessian_coeff(dis, dHat, Kappa, gassThreshold, USE_CUBIC_BARRIER);

            H = __GEIGEN__::__S_Mat9x9_multiply(__GEIGEN__::__v9_vec9_toMat9x9(q0, q0), lambda0);

            __GEIGEN__::Matrix12x12d Hessian;
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
            
            dis       = sqrt(dis); // Now dis is linear distance

            __GEIGEN__::Matrix3x3d F;
            __GEIGEN__::__set_Mat_val(F, 1, 0, 0, 0, c, 0, 0, 0, dis / dHat_sqrt);
            double3 n1 = make_double3(0, 1, 0);
            double3 n2 = make_double3(0, 0, 1);

            double eps_x = _compute_epx(_rest_vertexes[MMCVIDI.x],
                                        _rest_vertexes[MMCVIDI.y],
                                        _rest_vertexes[MMCVIDI.z],
                                        _rest_vertexes[MMCVIDI.w]);

            double lambda10, lambda11, lambda12, lambda20, lambdag1g;
            
            // Use unified parallel barrier hessian calculation
            
            // Note: I2 passed to parallel_hessian_coeffs should be I2 as used in barrier functions.
            // In original code: I2 = dis_sq / dHat.
            // Let's recompute I2 correctly if needed.
            // Original code used I2 (computed before sqrt) for log terms.
            
            double I2_sq = (dis * dis) / dHat; // Reconstruct squared ratio if needed, or use I2 from above.
            // Actually, I2 variable above was dis(squared)/dHat.
            
            barrier::parallel_hessian_coeffs(I1, I2_sq, c, F.m[2][2], eps_x, dHat, Kappa,
                                            lambda10, lambda11, lambda12, lambda20, lambdag1g,
                                            USE_CUBIC_BARRIER);

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

            nn = __GEIGEN__::__v_vec_toMat(n2, n2);
            __GEIGEN__::__M_Mat_multiply(F, nn, fnn);
            __GEIGEN__::Vector9 q20 = __GEIGEN__::__Mat3x3_to_vec9_double(fnn);
            q20 = __GEIGEN__::__s_vec9_multiply(q20, 1.0 / sqrt(I2_sq));

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
                dis       = sqrt(dis);

                __GEIGEN__::Matrix3x3d F;
                __GEIGEN__::__set_Mat_val(F, 1, 0, 0, 0, c, 0, 0, 0, dis / dHat_sqrt);
                double3 n1 = make_double3(0, 1, 0);
                double3 n2 = make_double3(0, 0, 1);

                double eps_x = _compute_epx(_rest_vertexes[MMCVIDI.x],
                                            _rest_vertexes[MMCVIDI.z],
                                            _rest_vertexes[MMCVIDI.y],
                                            _rest_vertexes[MMCVIDI.w]);

                double lambda10, lambda11, lambda12, lambda20, lambdag1g;
                
                double I2_sq = (dis * dis) / dHat;
                barrier::parallel_hessian_coeffs(I1, I2_sq, c, F.m[2][2], eps_x, dHat, Kappa,
                                                lambda10, lambda11, lambda12, lambda20, lambdag1g,
                                                USE_CUBIC_BARRIER);

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

                nn = __GEIGEN__::__v_vec_toMat(n2, n2);
                __GEIGEN__::__M_Mat_multiply(F, nn, fnn);
                __GEIGEN__::Vector9 q20 = __GEIGEN__::__Mat3x3_to_vec9_double(fnn);
                q20 = __GEIGEN__::__s_vec9_multiply(q20, 1.0 / sqrt(I2_sq));


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
                // PP (Point-Point) collision
                double dis;
                _d_PP(_vertexes[v0I], _vertexes[MMCVIDI.y], dis);
                dis                            = sqrt(dis);
                double              d_hat_sqrt = sqrt(dHat);
                __GEIGEN__::Vector6 PFPxT;
                pFpx_pp2(_vertexes[v0I], _vertexes[MMCVIDI.y], d_hat_sqrt, PFPxT);

                // Compute hessian coefficient using barrier function
                double lambda0 = barrier::simple_hessian_coeff(dis, dHat, Kappa, gassThreshold, USE_CUBIC_BARRIER);

                double                 H       = lambda0;
                __GEIGEN__::Matrix6x6d Hessian = __GEIGEN__::__s_M6x6_Multiply(
                    __GEIGEN__::__v6_vec6_toMat6x6(PFPxT, PFPxT), H);
                int Hidx = matIndex[idx];

                uint2 global_index = make_uint2(v0I, MMCVIDI.y);
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
                dis       = sqrt(dis);

                __GEIGEN__::Matrix3x3d F;
                __GEIGEN__::__set_Mat_val(F, 1, 0, 0, 0, c, 0, 0, 0, dis / dHat_sqrt);
                double3 n1 = make_double3(0, 1, 0);
                double3 n2 = make_double3(0, 0, 1);

                double eps_x = _compute_epx(_rest_vertexes[MMCVIDI.x],
                                            _rest_vertexes[MMCVIDI.w],
                                            _rest_vertexes[MMCVIDI.y],
                                            _rest_vertexes[MMCVIDI.z]);

                double lambda10, lambda11, lambda12, lambda20, lambdag1g;
                
                double I2_sq = (dis * dis) / dHat;
                barrier::parallel_hessian_coeffs(I1, I2_sq, c, F.m[2][2], eps_x, dHat, Kappa,
                                                lambda10, lambda11, lambda12, lambda20, lambdag1g,
                                                USE_CUBIC_BARRIER);

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

                nn = __GEIGEN__::__v_vec_toMat(n2, n2);
                __GEIGEN__::__M_Mat_multiply(F, nn, fnn);
                __GEIGEN__::Vector9 q20 = __GEIGEN__::__Mat3x3_to_vec9_double(fnn);
                q20 = __GEIGEN__::__s_vec9_multiply(q20, 1.0 / sqrt(I2_sq));


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
                // PE (Point-Edge) collision
                double dis;
                _d_PE(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], dis);
                dis                               = sqrt(dis);
                double                 d_hat_sqrt = sqrt(dHat);
                __GEIGEN__::Matrix9x4d PFPxT;
                pFpx_pe2(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], d_hat_sqrt, PFPxT);
                __GEIGEN__::Vector4 q0;
                q0.v[0] = q0.v[1] = q0.v[2] = 0;
                q0.v[3]                     = 1;

                __GEIGEN__::Matrix4x4d H;

                // Compute hessian coefficient using barrier function
                double lambda0 = barrier::simple_hessian_coeff(dis, dHat, Kappa, gassThreshold, USE_CUBIC_BARRIER);

                H = __GEIGEN__::__S_Mat4x4_multiply(
                    __GEIGEN__::__v4_vec4_toMat4x4(q0, q0), lambda0);

                __GEIGEN__::Matrix9x9d Hessian;
                __M9x4_S4x4_MT4x9_Multiply(PFPxT, H, Hessian);
                int Hidx = matIndex[idx];

                uint3 global_index = make_uint3(v0I, MMCVIDI.y, MMCVIDI.z);
                int triplet_id_offset = Hidx * 9 + offset4 * 16;
                write_triplet<9, 9>(
                    triplet_values, row_ids, col_ids, &(global_index.x), Hessian.m, triplet_id_offset);
            }
        }
        else
        {
            // PT (Point-Triangle) collision
            double dis;
            _d_PT(_vertexes[v0I],
                  _vertexes[MMCVIDI.y],
                  _vertexes[MMCVIDI.z],
                  _vertexes[MMCVIDI.w],
                  dis);
            dis                = sqrt(dis);
            double d_hat_sqrt  = sqrt(dHat);
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

            // Compute hessian coefficient using barrier function
            double lambda0 = barrier::simple_hessian_coeff(dis, dHat, Kappa, gassThreshold, USE_CUBIC_BARRIER);

            __GEIGEN__::Matrix9x9d H = __GEIGEN__::__S_Mat9x9_multiply(
                __GEIGEN__::__v9_vec9_toMat9x9(q0, q0), lambda0);
            __GEIGEN__::Matrix12x12d Hessian;
            __GEIGEN__::__M12x9_S9x9_MT9x12_Multiply(PFPxT, H, Hessian);

            int Hidx = matIndex[idx];
            uint4 global_index = make_uint4(v0I, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);
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

            // Unified Gradient Coefficient
            double grad_coeff = barrier::simple_gradient_coeff(dis, dHat, Kappa, USE_CUBIC_BARRIER);
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(tmp, grad_coeff);

            // Unified Hessian Coefficient
            double lambda0 = barrier::simple_hessian_coeff(dis, dHat, Kappa, gassThreshold, USE_CUBIC_BARRIER);

            __GEIGEN__::Vector12 gradient_vec =
                __GEIGEN__::__M12x9_v9_multiply((PFPxT), flatten_pk1);
            H = __GEIGEN__::__S_Mat9x9_multiply(__GEIGEN__::__v9_vec9_toMat9x9(q0, q0), lambda0);

            __GEIGEN__::Matrix12x12d Hessian;  // = __GEIGEN__::__M12x9_M9x12_Multiply(__GEIGEN__::__M12x9_M9x9_Multiply(PFPxT, H), __GEIGEN__::__Transpose12x9(PFPxT));
            __GEIGEN__::__M12x9_S9x9_MT9x12_Multiply(PFPxT, H, Hessian);

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

            double p1, p2;
            // Use unified parallel barrier gradient calculation
            // Note: I2 passed here should be squared ratio? 
            // parallel_gradient_coeffs expects I2 as used in barrier function.
            // In original code: I2 = dis_sq/dHat.
            double I2_sq = (dis * dis) / dHat;
            barrier::parallel_gradient_coeffs(I1, I2_sq, eps_x, dHat, Kappa, p1, p2, USE_CUBIC_BARRIER);

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

            double lambda10, lambda11, lambda12, lambda20, lambdag1g;
            barrier::parallel_hessian_coeffs(I1, I2_sq, c, F.m[2][2], eps_x, dHat, Kappa,
                                            lambda10, lambda11, lambda12, lambda20, lambdag1g,
                                            USE_CUBIC_BARRIER);

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
                         
                double p1, p2;
                double I2_sq = (dis * dis) / dHat;
                barrier::parallel_gradient_coeffs(I1, I2_sq, eps_x, dHat, Kappa, p1, p2, USE_CUBIC_BARRIER);

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

                double lambda10, lambda11, lambda12, lambda20, lambdag1g;
                barrier::parallel_hessian_coeffs(I1, I2_sq, c, F.m[2][2], eps_x, dHat, Kappa,
                                                lambda10, lambda11, lambda12, lambda20, lambdag1g,
                                                USE_CUBIC_BARRIER);

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
                double dis;
                _d_PP(_vertexes[v0I], _vertexes[MMCVIDI.y], dis);
                dis                            = sqrt(dis);
                double              d_hat_sqrt = sqrt(dHat);
                __GEIGEN__::Vector6 PFPxT;
                pFpx_pp2(_vertexes[v0I], _vertexes[MMCVIDI.y], d_hat_sqrt, PFPxT);
                double I5  = pow(dis / d_hat_sqrt, 2);
                double fnn = dis / d_hat_sqrt;

                // Unified Gradient Coefficient
                double grad_coeff = barrier::simple_gradient_coeff(dis, dHat, Kappa, USE_CUBIC_BARRIER);
                double flatten_pk1 = fnn * grad_coeff;

                __GEIGEN__::Vector6 gradient_vec =
                    __GEIGEN__::__s_vec6_multiply(PFPxT, flatten_pk1);

                {
                    atomicAdd(&(_gradient[v0I].x), gradient_vec.v[0]);
                    atomicAdd(&(_gradient[v0I].y), gradient_vec.v[1]);
                    atomicAdd(&(_gradient[v0I].z), gradient_vec.v[2]);
                    atomicAdd(&(_gradient[MMCVIDI.y].x), gradient_vec.v[3]);
                    atomicAdd(&(_gradient[MMCVIDI.y].y), gradient_vec.v[4]);
                    atomicAdd(&(_gradient[MMCVIDI.y].z), gradient_vec.v[5]);
                }

                // Unified Hessian Coefficient
                double lambda0 = barrier::simple_hessian_coeff(dis, dHat, Kappa, gassThreshold, USE_CUBIC_BARRIER);

                double                 H       = lambda0;
                __GEIGEN__::Matrix6x6d Hessian = __GEIGEN__::__s_M6x6_Multiply(
                    __GEIGEN__::__v6_vec6_toMat6x6(PFPxT, PFPxT), H);
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

                double p1, p2;
                double I2_sq = (dis * dis) / dHat;
                barrier::parallel_gradient_coeffs(I1, I2_sq, eps_x, dHat, Kappa, p1, p2, USE_CUBIC_BARRIER);

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

                double lambda10, lambda11, lambda12, lambda20, lambdag1g;
                barrier::parallel_hessian_coeffs(I1, I2_sq, c, F.m[2][2], eps_x, dHat, Kappa,
                                                lambda10, lambda11, lambda12, lambda20, lambdag1g,
                                                USE_CUBIC_BARRIER);

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
                
                // Unified Gradient Coefficient
                double grad_coeff = barrier::simple_gradient_coeff(dis, dHat, Kappa, USE_CUBIC_BARRIER);
                __GEIGEN__::Vector4 flatten_pk1 = __GEIGEN__::__s_vec4_multiply(
                    fnn, grad_coeff);

                __GEIGEN__::Vector9 gradient_vec =
                    __GEIGEN__::__M9x4_v4_multiply(PFPxT, flatten_pk1);

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

                __GEIGEN__::Vector4 q0;
                q0.v[0] = q0.v[1] = q0.v[2] = 0;
                q0.v[3]                     = 1;
                __GEIGEN__::Matrix4x4d H;

                // Unified Hessian Coefficient
                double lambda0 = barrier::simple_hessian_coeff(dis, dHat, Kappa, gassThreshold, USE_CUBIC_BARRIER);

                H = __GEIGEN__::__S_Mat4x4_multiply(
                    __GEIGEN__::__v4_vec4_toMat4x4(q0, q0), lambda0);

                __GEIGEN__::Matrix9x9d Hessian;  // = __GEIGEN__::__M9x4_M4x9_Multiply(__GEIGEN__::__M9x4_M4x4_Multiply(PFPxT, H), __GEIGEN__::__Transpose9x4(PFPxT));
                __GEIGEN__::__M9x4_S4x4_MT4x9_Multiply(PFPxT, H, Hessian);
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
            
            // Unified Gradient Coefficient
            double grad_coeff = barrier::simple_gradient_coeff(dis, dHat, Kappa, USE_CUBIC_BARRIER);
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp, grad_coeff);

            __GEIGEN__::Vector12 gradient_vec =
                __GEIGEN__::__M12x9_v9_multiply(PFPxT, flatten_pk1);

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

            // Unified Hessian Coefficient
            double lambda0 = barrier::simple_hessian_coeff(dis, dHat, Kappa, gassThreshold, USE_CUBIC_BARRIER);

            H = __GEIGEN__::__S_Mat9x9_multiply(__GEIGEN__::__v9_vec9_toMat9x9(q0, q0), lambda0);

            __GEIGEN__::Matrix12x12d Hessian;  // = __GEIGEN__::__M12x9_M9x12_Multiply(__GEIGEN__::__M12x9_M9x9_Multiply(PFPxT, H), __GEIGEN__::__Transpose12x9(PFPxT));
            __GEIGEN__::__M12x9_S9x9_MT9x12_Multiply(PFPxT, H, Hessian);

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

            // Unified Gradient Coefficient
            double grad_coeff = barrier::simple_gradient_coeff(dis, dHat, Kappa, USE_CUBIC_BARRIER);
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp, grad_coeff);

            __GEIGEN__::Vector12 gradient_vec =
                __GEIGEN__::__M12x9_v9_multiply((PFPxT), flatten_pk1);

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

            double p1, p2;
            double I2_sq = (dis * dis) / dHat;
            barrier::parallel_gradient_coeffs(I1, I2_sq, eps_x, dHat, Kappa, p1, p2, USE_CUBIC_BARRIER);

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
                
                double p1, p2;
                double I2_sq = (dis * dis) / dHat;
                barrier::parallel_gradient_coeffs(I1, I2_sq, eps_x, dHat, Kappa, p1, p2, USE_CUBIC_BARRIER);

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
                double dis;
                _d_PP(_vertexes[v0I], _vertexes[MMCVIDI.y], dis);
                dis                            = sqrt(dis);
                double              d_hat_sqrt = sqrt(dHat);
                __GEIGEN__::Vector6 PFPxT;
                pFpx_pp2(_vertexes[v0I], _vertexes[MMCVIDI.y], d_hat_sqrt, PFPxT);
                double I5  = pow(dis / d_hat_sqrt, 2);
                double fnn = dis / d_hat_sqrt;

                // Unified Gradient Coefficient
                double grad_coeff = barrier::simple_gradient_coeff(dis, dHat, Kappa, USE_CUBIC_BARRIER);
                double flatten_pk1 = fnn * grad_coeff;

                __GEIGEN__::Vector6 gradient_vec =
                    __GEIGEN__::__s_vec6_multiply(PFPxT, flatten_pk1);

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

                double p1, p2;
                double I2_sq = (dis * dis) / dHat;
                barrier::parallel_gradient_coeffs(I1, I2_sq, eps_x, dHat, Kappa, p1, p2, USE_CUBIC_BARRIER);

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
                
                // Unified Gradient Coefficient
                double grad_coeff = barrier::simple_gradient_coeff(dis, dHat, Kappa, USE_CUBIC_BARRIER);
                __GEIGEN__::Vector4 flatten_pk1 = __GEIGEN__::__s_vec4_multiply(
                    fnn, grad_coeff);

                __GEIGEN__::Vector9 gradient_vec =
                    __GEIGEN__::__M9x4_v4_multiply(PFPxT, flatten_pk1);

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

            // Unified Gradient Coefficient
            double grad_coeff = barrier::simple_gradient_coeff(dis, dHat, Kappa, USE_CUBIC_BARRIER);
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp, grad_coeff);

            __GEIGEN__::Vector12 gradient_vec =
                __GEIGEN__::__M12x9_v9_multiply(PFPxT, flatten_pk1);

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

__global__ void _calAdaptiveStiffness(const double3* _vertexes,
                                      const double*  _masses,
                                      const int4*    _collisionPair,
                                      double*        _stiffness,
                                      double         dHat,
                                      int            number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    
    int4 MMCVIDI = _collisionPair[idx];
    double dis_sq = 0.0;
    double stiffness = 0.0;
    
    // Identify vertices and compute distance
    int v[4] = {-1, -1, -1, -1};
    int v_count = 0;
    
    if(MMCVIDI.x >= 0)
    {
        if(MMCVIDI.w >= 0) // EE
        {
            v[0] = MMCVIDI.x; v[1] = MMCVIDI.y; v[2] = MMCVIDI.z; v[3] = MMCVIDI.w;
            v_count = 4;
            _d_EE(_vertexes[v[0]], _vertexes[v[1]], _vertexes[v[2]], _vertexes[v[3]], dis_sq);
        }
        else // Parallel EE or others? The logic in _calBarrierHessian is complex for parallel.
        {
             // Parallel cases (w < 0). Logic is same as in _calBarrierHessian.
             MMCVIDI.w = -MMCVIDI.w - 1;
             double3 v0 = __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _vertexes[MMCVIDI.x]);
             double3 v1 = __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.z]);
             double c = __GEIGEN__::__norm(__GEIGEN__::__v_vec_cross(v0, v1));
             double I1 = c * c;
             
             // If I1 != 0, we calculate distance.
             if (I1 != 0) {
                 _d_EE(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dis_sq);
                 v[0] = MMCVIDI.x; v[1] = MMCVIDI.y; v[2] = MMCVIDI.z; v[3] = MMCVIDI.w;
                 v_count = 4;
             }
        }
    }
    else
    {
        int v0I = -MMCVIDI.x - 1;
        if(MMCVIDI.z < 0)
        {
            if(MMCVIDI.y < 0) // PPP (Parallel Point-Triangle -> Point-Point-Point?)
            {
                 MMCVIDI.y = -MMCVIDI.y - 1;
                 MMCVIDI.z = -MMCVIDI.z - 1;
                 MMCVIDI.w = -MMCVIDI.w - 1;
                 MMCVIDI.x = v0I;
                 // Simplified parallel check omitted, just get distance
                 double3 v0 = __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.x]);
                 double3 v1 = __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.y]);
                 double c = __GEIGEN__::__norm(__GEIGEN__::__v_vec_cross(v0, v1));
                 double I1 = c * c;
                 if (I1 != 0) {
                     _d_PP(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], dis_sq);
                     v[0] = MMCVIDI.x; v[1] = MMCVIDI.y;
                     v_count = 2;
                 }
            }
            else // PP
            {
                _d_PP(_vertexes[v0I], _vertexes[MMCVIDI.y], dis_sq);
                v[0] = v0I; v[1] = MMCVIDI.y;
                v_count = 2;
            }
        }
        else if(MMCVIDI.w < 0)
        {
            if(MMCVIDI.y < 0) // PPE
            {
                 MMCVIDI.y = -MMCVIDI.y - 1;
                 MMCVIDI.x = v0I;
                 MMCVIDI.w = -MMCVIDI.w - 1;
                 double3 v0 = __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.x]);
                 double3 v1 = __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.y]);
                 double c = __GEIGEN__::__norm(__GEIGEN__::__v_vec_cross(v0, v1));
                 double I1 = c * c;
                 if (I1 != 0) {
                     _d_PE(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], dis_sq);
                     v[0] = MMCVIDI.x; v[1] = MMCVIDI.y; v[2] = MMCVIDI.z;
                     v_count = 3;
                 }
            }
            else // PE
            {
                _d_PE(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], dis_sq);
                v[0] = v0I; v[1] = MMCVIDI.y; v[2] = MMCVIDI.z;
                v_count = 3;
            }
        }
        else // PT
        {
            _d_PT(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dis_sq);
            v[0] = v0I; v[1] = MMCVIDI.y; v[2] = MMCVIDI.z; v[3] = MMCVIDI.w;
            v_count = 4;
        }
    }
    
    // Adaptive stiffness calculation
    // Only if active (distance < dHat)
    // 
    // References ppf-contact-solver/src/cpp/barrier/barrier.cu: compute_stiffness
    // Formula: S_eff = w^T * H_elastic * w + w^T * M * w / d^2
    //
    // In our current GIPC architecture, we address these two terms separately:
    // 1. The "Elastic Stiffness" (w^T * H_elastic * w) is effectively handled by 
    //    GIPC::initKappa()'s gradient-based reduction (reduction2Kappa). This ensures 
    //    force balance at the current configuration (Kappa * d ~ Force_elastic).
    //
    // 2. The "Inertial/Mass Stiffness" (w^T * M * w / d^2) is calculated here.
    //    This term dominates as d -> 0 and is critical for preventing tunneling 
    //    during high-speed impacts.
    //
    // Implementation Details:
    // - We use sum(mass) as a conservative upper bound for the projected inertial mass 
    //   (w^T * M * w). Since barycentric weights <= 1, sum(mass) >= projected mass.
    // - We compute stiffness = sum(mass) / dis_sq.
    // - We convert this to Kappa units: _stiffness = stiffness / (4.0 * dHat).
    //   (Note: dHat parameter here is squared dHat, consistent with curvature scaling).
    
    if (dis_sq < dHat && dis_sq > 1e-20) {
        double sum_mass = 0.0;
        for(int i=0; i<v_count; ++i) {
            if(v[i] >= 0) sum_mass += _masses[v[i]];
        }
        
        // stiffness = sum(mass) / dis_sq
        // This is the adaptive stiffness.
        // We might want to normalize by number of vertices? 
        // ppf uses mass[ii] / sqr_x. Summing it up makes sense for total stiffness.
        
        stiffness = sum_mass / dis_sq;
        
        // For cubic barrier: H ~ 4 * Kappa * dHat^2.
        // We want H ~ stiffness.
        // So Kappa ~ stiffness / (4 * dHat^2).
        // We will store stiffness directly and handle Kappa conversion in host code.
        
        // Or better: store the required Kappa directly.
        // required_Kappa = stiffness / (4.0 * dHat); // Note dHat passed here is dHat^2.
        // Wait, dHat input is dHat^2.
        // cubic_curvature * dHat returns curvature * dHat.
        // curvature ~ 4 / sqrt(dHat).
        // So H ~ Kappa * 4 * sqrt(dHat).
        // Let's recheck simple_hessian_coeff.
        // curv = 4 * (1 - g/ghat). Max is 4.
        // return Kappa * curv * dHat. (dHat is dHat^2).
        // So max H = 4 * Kappa * dHat^2.
        // So Kappa = stiffness / (4 * dHat).
        
        _stiffness[idx] = stiffness / (4.0 * dHat);
    } else {
        _stiffness[idx] = 0.0;
    }
}
