#include <linear_system/solver/pncg_solver.h>
#include <gipc/utils/timer.h>
#include <gipc/statistics.h>
#include <cuda_tools/cuda_tools.h>

// Helper CUDA kernels for PNCG operations
__global__ void Scale_Multiply_Inplace(double* arr, double scale, int numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;
    arr[idx] = arr[idx] * scale;
}

void Z_Scale_Multiply_Inplace(double* arr, double scale, int vertexNum)
{
    int numbers = vertexNum;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    Scale_Multiply_Inplace<<<blockNum, threadNum>>>(arr, scale, numbers);
}

__global__ void Scale_Multiply(double* dst, const double* src, double scale, int numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;
    dst[idx] = src[idx] * scale;
}

void Z_Scale_Multiply(double* dst, const double* src, double scale, int vertexNum)
{
    int numbers = vertexNum;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    Scale_Multiply<<<blockNum, threadNum>>>(dst, src, scale, numbers);
}

__global__ void Add_Inplace(double* dst, const double* src, int numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;
    dst[idx] = dst[idx] + src[idx];
}

void Z_Add_Inplace(double* dst, const double* src, int vertexNum)
{
    int numbers = vertexNum;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    Add_Inplace<<<blockNum, threadNum>>>(dst, src, numbers);
}

__global__ void Add_Scale_Mul_Vec_Inplace(double* dst, const double* src, double scale, int numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;
    dst[idx] = dst[idx] + src[idx] * scale;
}

void Z_Add_Scale_Mul_Vec_Inplace(double* dst, const double* src, double scale, int vertexNum)
{
    int numbers = vertexNum;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    Add_Scale_Mul_Vec_Inplace<<<blockNum, threadNum>>>(dst, src, scale, numbers);
}

__global__ void Subtract_Inplace(double* dst, const double* src, int numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;
    dst[idx] = dst[idx] - src[idx];
}

void Z_Subtract_Inplace(double* dst, const double* src, int vertexNum)
{
    int numbers = vertexNum;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    Subtract_Inplace<<<blockNum, threadNum>>>(dst, src, numbers);
}

__global__ void Subtract(double* dst, const double* src1, const double* src2, int numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;
    dst[idx] = src1[idx] - src2[idx];
}

void Z_Subtract(double* dst, const double* src1, const double* src2, int vertexNum)
{
    int numbers = vertexNum;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    Subtract<<<blockNum, threadNum>>>(dst, src1, src2, numbers);
}

// Reuse the reduction algorithm from pcg_solver.cu (declaration)
extern double My_PCG_General_v_v_Reduction_Algorithm(double* temp, double* A, double* B, int vertexNum);

namespace gipc
{
PNCGSolver::PNCGSolver(const PNCGSolverConfig& cfg)
    : m_config(cfg)
{
}

SizeT PNCGSolver::solve(muda::DenseVectorView<Float> x, muda::CDenseVectorView<Float> b)
{
    Timer timer{"pncg"};

    x.buffer_view().fill(0);
    z.resize(b.size());
    p.resize(b.size());
    r.resize(b.size());
    Ap.resize(b.size());
    
    // Single-step PNCG (preconditioned steepest descent)
    // This is called once per Newton iteration; the outer loop handles the iteration
    auto iter = pncg(x, b, m_config.max_iter_ratio * b.size(), 0);

    return iter;
}

// PNCG implementation - Single-step Preconditioned Steepest Descent
// This solver computes ONE search direction per call.
// The outer Newton iteration handles the full optimization loop.
//
// Solves: find x such that x ≈ -H^{-1} * g
// where b = g (gradient) and A = H (Hessian)
SizeT PNCGSolver::pncg(muda::DenseVectorView<Float> x, muda::CDenseVectorView<Float> b, SizeT max_iter, int /* unused */)
{
    // Resize working vectors
    p_k.resize(b.size());
    p_k_1.resize(b.size());
    g_k.resize(b.size());
    g_k_1.resize(b.size());
    P_g_k_1.resize(b.size());
    P_g_k.resize(b.size());
    ss.resize(b.size());

    // Copy gradient to g_k_1: g = b
    g_k_1.buffer_view().copy_from(b.buffer_view());

    // Apply preconditioner: P_g = P * g
    apply_preconditioner(P_g_k_1, g_k_1);

    // Compute search direction: p = -P * g (steepest descent direction)
    Z_Scale_Multiply(p_k_1.buffer_view().data(),
                     P_g_k_1.buffer_view().data(),
                     -1.0,
                     p_k_1.size());

    // Copy to p for spmv
    p = p_k_1;
    
    // Compute Ap = A * p (Hessian times search direction)
    spmv(p.cview(), Ap.view());

    // Compute gTp = g^T * p
    double aa = My_PCG_General_v_v_Reduction_Algorithm(
        ss.buffer_view().data(),
        g_k_1.buffer_view().data(),
        p_k_1.buffer_view().data(),
        g_k_1.size());

    // Compute pHp = p^T * A * p (curvature along search direction)
    double bb = My_PCG_General_v_v_Reduction_Algorithm(
        ss.buffer_view().data(),
        p_k_1.buffer_view().data(),
        Ap.buffer_view().data(),
        p_k_1.size());

    // Compute optimal step size: alpha = -g^T*p / p^T*A*p
    // Since p = -P*g, we have g^T*p = -g^T*P*g < 0 (assuming P is SPD)
    // So alpha = -(-g^T*P*g) / (p^T*A*p) = g^T*P*g / p^T*A*p > 0
    double alpha = -aa / bb;

    // Save values for external access (e.g., line search)
    gTp = aa;
    pHp = bb;

    // Handle negative curvature
    if(alpha < 0)
    {
        // Negative curvature indicates indefinite Hessian in this direction
        // Use a small positive step or zero
        alpha = 0;
    }

    // Compute the search direction: x = alpha * p
    // Note: The original code had x = -alpha * p, but since p = -P*g,
    // we want x = alpha * p = -alpha * P * g (descent direction scaled by alpha)
    Z_Scale_Multiply(x.buffer_view().data(),
                     p_k_1.buffer_view().data(),
                     alpha,
                     p_k_1.size());

    // Return 1 to indicate one step was taken
    return 1;
}

}  // namespace gipc
