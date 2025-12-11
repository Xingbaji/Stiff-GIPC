#pragma once
#include <linear_system/solver/pcg_solver.h>

namespace gipc
{

class PNCGSolverConfig
{
  public:
    /**
     * \brief the maximum number of iterations will be:
     *  dof * max_iter_ratio
     */
    Float max_iter_ratio  = 0.3;
    Float global_tol_rate = 1e-4;
    bool  use_bsr         = true;
};

class PNCGSolver : public IterativeSolver
{
    using DeviceDenseVector = muda::DeviceDenseVector<Float>;

  public:
    PNCGSolver(const PNCGSolverConfig& cfg);
    virtual ~PNCGSolver() = default;

    void config(const PNCGSolverConfig& config) { this->m_config = config; }
    const auto& config() const { return this->m_config; }

  private:
    DeviceDenseVector z;   // preconditioned residual
    DeviceDenseVector r;   // residual
    DeviceDenseVector p;   // search direction
    DeviceDenseVector Ap;  // A*p
    PNCGSolverConfig  m_config;

    // PNCG specific vectors
    DeviceDenseVector p_k;
    DeviceDenseVector p_k_1;
    DeviceDenseVector g_k;
    DeviceDenseVector g_k_1;
    DeviceDenseVector y;
    DeviceDenseVector Py;
    DeviceDenseVector P_g_k_1;
    DeviceDenseVector P_g_k;
    DeviceDenseVector ss;
    DeviceDenseVector prev_vertices;
    DeviceDenseVector local_alphas;

  public:
    double pHp;
    double gTp;
    double delta_E_init;
    
  public:
    // Reset is a no-op for PNCG (stateless iterative solver)
    void reset() override {}

  protected:
    SizeT solve(muda::DenseVectorView<Float> x, muda::CDenseVectorView<Float> b) override;

  private:
    /**
     * \brief Preconditioned Nonlinear Conjugate Gradient solver
     * 
     * Solves the linear system Ax = -b using Fletcher-Reeves PNCG.
     * 
     * \param x Output: the solution vector
     * \param b Input: the right-hand side (gradient)
     * \param max_iter Maximum number of iterations
     * \return Actual number of iterations performed
     */
    SizeT pncg(muda::DenseVectorView<Float> x, muda::CDenseVectorView<Float> b, SizeT max_iter, int unused = 0);
};
}  // namespace gipc

