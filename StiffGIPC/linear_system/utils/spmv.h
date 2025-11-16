#pragma once
#include <gipc/type_define.h>

#include <muda/ext/linear_system/dense_vector_view.h>

namespace gipc
{
class Spmv
{
  public:

    void warp_reduce_sym_spmv(Float                         a,
                              Eigen::Matrix3d*              triplet_values,
                              int*                          row_ids,
                              int*                          col_ids,
                              int                           triplet_count,
                              muda::CDenseVectorView<Float> x,
                              Float                         b,
                              muda::DenseVectorView<Float>  y);
};
}  // namespace gipc
