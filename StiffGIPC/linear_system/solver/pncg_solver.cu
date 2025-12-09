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
    // Match fork behavior: always call pncg with iter=0
    auto iter = pncg(x, b, m_config.max_iter_ratio * b.size(), 0);

    return iter;
}

// PNCG implementation - matches Stiff-GIPC-fork behavior
// Note: Fork always uses iter=0, so this is effectively one-step preconditioned steepest descent
SizeT PNCGSolver::pncg(muda::DenseVectorView<Float> x, muda::CDenseVectorView<Float> b, SizeT max_iter, int iter)
{
    p_k.resize(b.size());
    p_k_1.resize(b.size());
    g_k.resize(b.size());
    g_k_1.resize(b.size());
    y.resize(b.size());
    Py.resize(b.size());
    P_g_k_1.resize(b.size());
    P_g_k.resize(b.size());
    ss.resize(b.size());
    prev_vertices.resize(b.size());

    // Copy gradient to g_k_1
    g_k_1.buffer_view().copy_from(b.buffer_view());

    double alpha = 1;
    bool restart = true;
    
    while(1) {
        if(iter == 0 || restart) {
            // m_global_preconditioner->do_assemble()
        }
        
        // P_g_k_1 = P * g_{k+1}
        apply_preconditioner(P_g_k_1, g_k_1);

        if(iter == 0) {
            // p_{k+1} = -P * g_{k+1}
            Z_Scale_Multiply(p_k_1.buffer_view().data(),
                             P_g_k_1.buffer_view().data(),
                             -1.0,
                             p_k_1.size());
        }
        else {
            // FR (Fletcher-Reeves)
            // beta = (g_{k+1}^T * P * g_{k+1}) / (g_{k}^T * P * g_{k})
            double numerator = My_PCG_General_v_v_Reduction_Algorithm(
                ss.buffer_view().data(),
                g_k_1.buffer_view().data(),
                P_g_k_1.buffer_view().data(),
                g_k_1.size());
            double denominator = My_PCG_General_v_v_Reduction_Algorithm(
                ss.buffer_view().data(),
                g_k.buffer_view().data(),
                P_g_k.buffer_view().data(),
                g_k.size());
            double beta = numerator / denominator;

            // p_{k+1} = -P * g_{k+1} + beta * p_{k}
            Z_Scale_Multiply_Inplace(p_k.buffer_view().data(), beta, p_k.size());
            Z_Add_Scale_Mul_Vec_Inplace(p_k_1.buffer_view().data(),
                                        P_g_k_1.buffer_view().data(),
                                        -1.0,
                                        p_k_1.size());
        }

        p = p_k_1;
        // Ap = A * p_{k+1}
        spmv(p.cview(), Ap.view());

        double aa = My_PCG_General_v_v_Reduction_Algorithm(
            ss.buffer_view().data(),
            g_k_1.buffer_view().data(),
            p_k_1.buffer_view().data(),
            g_k_1.size());
        double bb = My_PCG_General_v_v_Reduction_Algorithm(
            ss.buffer_view().data(),
            p_k_1.buffer_view().data(),
            Ap.buffer_view().data(),
            p_k_1.size());
        double g_alpha = -aa / bb;
        alpha = g_alpha;

        if (alpha < 0) {
            restart = true;
            std::cout << "alpha < 0: " << alpha << std::endl;
            alpha = 0;
        } else {
            gTp = aa;
            pHp = bb;
            break;
        }
        break;
    }

    // Copy the final moving direction to x
    Z_Scale_Multiply(x.buffer_view().data(),
                     p_k_1.buffer_view().data(),
                     -alpha,
                     p_k_1.size());
    
    // Copy g_{k+1} to g_{k}
    g_k.buffer_view().copy_from(g_k_1.buffer_view());
    // Copy p_{k+1} to p_{k}
    p_k.buffer_view().copy_from(p_k_1.buffer_view());

    return iter;
}

}  // namespace gipc

