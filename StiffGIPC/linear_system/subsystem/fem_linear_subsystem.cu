#include <linear_system/subsystem/fem_linear_subsystem.h>
#include <muda/ext/eigen.h>
#include <muda/cub/device/device_reduce.h>
#include <gipc/utils/host_log.h>

namespace gipc
{
FEMLinearSubsystem::FEMLinearSubsystem(GIPC&                gipc,
                                       device_TetraData&    tetra_data)
    : m_gipc(gipc)
    , m_tetra_data(tetra_data)
{
    //muda::Debug::debug_sync_all(true);
}


muda::CBufferView<int> FEMLinearSubsystem::boundary_type() const
{
    auto fem_offset = m_gipc.abd_fem_count_info.fem_point_offset;
    auto fem_count  = m_gipc.abd_fem_count_info.fem_point_num;
    return muda::CBufferView<int>(m_tetra_data.BoundaryType, fem_offset, fem_count);
}

muda::BufferView<double3> FEMLinearSubsystem::barrier_gradient() const
{
    auto offset    = m_gipc.abd_fem_count_info.fem_point_offset;
    auto fem_count = m_gipc.abd_fem_count_info.fem_point_num;
    return muda::BufferView<double3>{m_tetra_data.fb, m_gipc.vertexNum}.subview(offset, fem_count);
}

muda::BufferView<double3> FEMLinearSubsystem::shape_gradient() const
{
    auto offset    = m_gipc.abd_fem_count_info.fem_point_offset;
    auto fem_count = m_gipc.abd_fem_count_info.fem_point_num;
    return muda::BufferView<double3>{m_tetra_data.shape_grads, m_gipc.vertexNum}.subview(
        offset, fem_count);
}

muda::BufferView<double3> FEMLinearSubsystem::dx() const
{
    auto offset    = m_gipc.abd_fem_count_info.fem_point_offset;
    auto fem_count = m_gipc.abd_fem_count_info.fem_point_num;
    return muda::BufferView<double3>{m_gipc._moveDir, m_gipc.vertexNum}.subview(offset, fem_count);
}

muda::BufferView<double> FEMLinearSubsystem::mass() const
{
    auto fem_offset = m_gipc.abd_fem_count_info.fem_point_offset;
    auto fem_count  = m_gipc.abd_fem_count_info.fem_point_num;
    return muda::BufferView<double>{m_tetra_data.masses, m_gipc.vertexNum}.subview(
        fem_offset, fem_count);
}

void FEMLinearSubsystem::report_subsystem_info()
{
    this->right_hand_side_dof(dx().size() * 3);
}

namespace details
{
    template <int N>
    __device__ void fill_hessian_block(int I,
                                       muda::TripletMatrixViewer<gipc::Float, 3>& viewer,
                                       Eigen::Map<Eigen::Vector<uint32_t, N>> index,
                                       const gipc::Matrix<3 * N, 3 * N>& matrix)
    {
        constexpr int N2     = N * N;
        auto          offset = I * N2;
        for(int i = 0; i < N; ++i)
        {
            for(int j = 0; j < N; ++j)
            {
                viewer(offset++).write(index(i),
                                       index(j),
                                       matrix.template block<3, 3>(i * 3, j * 3));
            }
        }
    }

      template <int N>
    __device__ void fill_hessian_block(int                        I,
                                       const muda::CDense1D<int>& boundary,
                                       muda::TripletMatrixViewer<gipc::Float, 3>& viewer,
                                       Eigen::Map<Eigen::Vector<uint32_t, N>> index,
                                       const gipc::Matrix<3 * N, 3 * N>& matrix)
    {
        constexpr int N2     = N * N;
        auto          offset = I * N2;
        for(int i = 0; i < N; ++i)
        {
            for(int j = 0; j < N; ++j)
            {
                if(boundary(index(i)) != 0 || boundary(index(j)) != 0)
                {
                    viewer(offset++).write(index(i), index(j), gipc::Matrix3x3::Zero());
                }
                else
                {
                    viewer(offset++).write(index(i),
                                           index(j),
                                           matrix.template block<3, 3>(i * 3, j * 3));
                }
            }
        }
    }

}  // namespace details


void FEMLinearSubsystem::assemble(DenseVectorView gradient)
{
    using namespace muda;

    if(m_gipc.abd_fem_count_info.fem_point_num < 1)
        return;

    auto barrier_gradient = this->barrier_gradient();
    auto shape_gradient   = this->shape_gradient();

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(barrier_gradient.size(),
               [b     = barrier_gradient.viewer().name("barrier_gradient"),
                s     = shape_gradient.viewer().name("shape_gradient"),
                btype = boundary_type().cviewer().name("boundary_type"),
                gradient = gradient.viewer().name("gradient")] __device__(int i) mutable
               {
                   if(btype(i) != 0)
                   {
                       gradient.segment<3>(i * 3).as_eigen() = Vector3::Zero();
                   }
                   else
                   {
                       gradient.segment<3>(i * 3).as_eigen() =
                           eigen::as_eigen(b(i)) + eigen::as_eigen(s(i));
                   }
               });
}

void FEMLinearSubsystem::retrieve_solution(CDenseVectorView dx)
{
    using namespace muda;

    auto move_dir = this->dx();

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(move_dir.size(),
               [dx        = dx.viewer().name("dx"),
                move_dir  = move_dir.viewer().name("move_dir"),
                local_tol = m_local_tol] __device__(int i) mutable
               {
                   auto md = eigen::as_eigen(move_dir(i));
                   md      = dx.segment<3>(i * 3).as_eigen();
                   //if(md.squaredNorm() < local_tol * local_tol)
                   //{
                   //    md.setZero();
                   //}
               });
}


bool FEMLinearSubsystem::accuracy_statisfied(CDenseVectorView residual)
{
    using namespace muda;

    //m_local_squared_norm.resize(dx().size());

    //ParallelFor()
    //    .kernel_name(__FUNCTION__)
    //    .apply(dx().size(),
    //           [residual     = residual.viewer().name("residual"),
    //            squared_norm = m_local_squared_norm.viewer().name(
    //                "squared_norm")] __device__(int i) mutable {
    //               squared_norm(i) = residual.segment<3>(i * 3).as_eigen().squaredNorm();
    //           });

    //DeviceReduce().Max(m_local_squared_norm.data(),
    //                   m_max_squared_norm.data(),
    //                   m_local_squared_norm.size());

    //gipc::Float max_norm = m_max_squared_norm;
    //// m_local_tol          = 1e-11;
    //// GIPC_INFO("FEMLinearSubsystem max_local_residual_norm: {}", std::sqrt(max_norm));
    //return max_norm < m_local_tol * m_local_tol;
    return true;
}

void FEMLinearSubsystem::set_local_tolerance(gipc::Float tol)
{
    m_local_tol = tol;
}
}  // namespace gipc