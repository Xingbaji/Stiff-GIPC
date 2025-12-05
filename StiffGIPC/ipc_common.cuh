//
// ipc_common.cuh
// GIPC - Common Helper Functions for IPC Kernels
//
// This header contains shared utility functions used by both
// ipc_barrier.cu and barrier_gradient_hessian.cu
//
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#pragma once
#ifndef _IPC_COMMON_H_
#define _IPC_COMMON_H_

#include <cuda_runtime.h>
#include "Eigen/Eigen"

//=============================================================================
// Common Macros
//=============================================================================

#ifndef RANK
#define RANK 2
#endif

#ifndef NEWF
#define NEWF
#endif

//=============================================================================
// Helper Functions
//=============================================================================

/**
 * @brief Make a symmetric matrix positive definite
 * 
 * Uses eigenvalue decomposition to project the matrix onto the 
 * positive semi-definite cone by clamping negative eigenvalues to zero.
 */
template <typename Scalar, int size>
__device__ __host__ inline void makePDGeneral_ipc(Eigen::Matrix<Scalar, size, size>& symMtr)
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

/**
 * @brief Write a dense matrix block to sparse triplet format
 * 
 * Converts a ROWSxCOLS dense matrix into 3x3 block triplet format
 * for efficient sparse matrix assembly.
 * 
 * @tparam ROWS Number of rows (must be multiple of 3)
 * @tparam COLS Number of columns (must be multiple of 3)
 * @param triplet_value Output array for 3x3 matrix blocks
 * @param row_ids Output array for row indices
 * @param col_ids Output array for column indices
 * @param index Vertex indices for the block
 * @param input Dense matrix values
 * @param offset Starting offset in output arrays
 */
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

#endif // _IPC_COMMON_H_

