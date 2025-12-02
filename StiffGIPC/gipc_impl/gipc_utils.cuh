#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include "Eigen/Eigen"
#include "cuda_tools/cuda_tools.h"

#ifndef RANK
#define RANK 2
#endif

#ifndef NEWF
#define NEWF
#endif

// Template function declarations
template <typename Scalar, int size>
__device__ __host__ void makePDGeneral(Eigen::Matrix<Scalar, size, size>& symMtr)
{
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Scalar, size, size>> eigen_solver;

    if constexpr(size <= 3)
        eigen_solver.computeDirect(symMtr);
    else
        eigen_solver.compute(symMtr);
    Eigen::Vector<Scalar, size>       eigen_values  = eigen_solver.eigenvalues();
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
    symMtr = eigen_vectors * eigen_values.asDiagonal() * eigen_vectors.transpose();
}

template <typename Scalar, int size>
__device__ __host__ void makePD(Eigen::Matrix<Scalar, size, size>& symMtr)
{
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Scalar, size, size>> eigenSolver(symMtr);
    if(eigenSolver.eigenvalues()[0] >= 0.0)
    {
        return;
    }
    Eigen::Matrix<Scalar, size, size> D;  //(eigenSolver.eigenvalues());
    D.setZero();
    int rows = size;  //((size == Eigen::Dynamic) ? symMtr.rows() : size);
    for(int i = 0; i < rows; i++)
    {
        if(eigenSolver.eigenvalues()[i] > 0.0)
        {
            D(i, i) = eigenSolver.eigenvalues()[i];
        }
    }
    symMtr = eigenSolver.eigenvectors() * D * eigenSolver.eigenvectors().transpose();
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
#ifdef SymGH
        int start = ii;
#else
        int start = 0;
#endif
        for(int jj = start; jj < coln; jj++)
        {

#ifdef SymGH
            int kk = ii * rown + jj - ii * (ii + 1) / 2;
#else
            int kk = ii * rown + jj;  // - ii * (ii + 1) / 2;
#endif
            int row = index[ii];
            int col = index[jj];
#ifdef SymGH
            if(row > col)
            {
                row_ids[offset + kk] = col;
                col_ids[offset + kk] = row;
                for(int iii = 0; iii < 3; iii++)
                {
                    for(int jjj = 0; jjj < 3; jjj++)
                    {
                        triplet_value[offset + kk](iii, jjj) =
                            input[jj * 3 + iii][ii * 3 + jjj];
                    }
                }
            }
            else
#endif
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

__device__ __host__ inline uint32_t expand_bits(std::uint32_t v) noexcept
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

__device__ __host__ inline uint32_t hash_code(
    int type, double x, double y, double z, double resolution = 1024) noexcept
{
    x = std::min(std::max(x * resolution, 0.0), resolution - 1.0);
    y = std::min(std::max(y * resolution, 0.0), resolution - 1.0);
    z = std::min(std::max(z * resolution, 0.0), resolution - 1.0);


    //
    if(type == -1)
    {
        const uint32_t xx     = expand_bits(static_cast<uint32_t>(x));
        const uint32_t yy     = expand_bits(static_cast<uint32_t>(y));
        const uint32_t zz     = expand_bits(static_cast<uint32_t>(z));
        std::uint32_t  mchash = ((xx << 2) + (yy << 1) + zz);

        return mchash;
    }
    else if(type == 0)
    {
        return (((static_cast<uint32_t>(z) * 1024) + static_cast<uint32_t>(y)) * 1024)
               + static_cast<uint32_t>(x);
    }
    else if(type == 1)
    {
        return (((static_cast<uint32_t>(y) * 1024) + static_cast<uint32_t>(z)) * 1024)
               + static_cast<uint32_t>(x);
    }
    else if(type == 2)
    {
        return (((static_cast<uint32_t>(x) * 1024) + static_cast<uint32_t>(z)) * 1024)
               + static_cast<uint32_t>(y);
    }
    else if(type == 3)
    {
        return (((static_cast<uint32_t>(z) * 1024) + static_cast<uint32_t>(x)) * 1024)
               + static_cast<uint32_t>(y);
    }
    else if(type == 4)
    {
        return (((static_cast<uint32_t>(y) * 1024) + static_cast<uint32_t>(x)) * 1024)
               + static_cast<uint32_t>(z);
    }
    else
    {
        return (((static_cast<uint32_t>(x) * 1024) + static_cast<uint32_t>(y)) * 1024)
               + static_cast<uint32_t>(z);
    }
}
