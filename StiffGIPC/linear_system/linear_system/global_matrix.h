#pragma once

//namespace muda::details
//{
//template <typename T, int N>
//class MatrixFormatConverter;
//}
#include"cuda_tools/cuda_device_buffer.h"
#include"Eigen/Eigen"

class GIPCTripletMatrix
{
  public:
    using BlockMatrix = Eigen::Matrix<double, 3, 3>;
    //using EntryValueType = T;
    //int Dimenstion              = M;
  public:
    cudatool::CudaDeviceBuffer<BlockMatrix> m_block_values;
    cudatool::CudaDeviceBuffer<int>         m_block_row_indices;
    cudatool::CudaDeviceBuffer<int>         m_block_col_indices;
    cudatool::CudaDeviceBuffer<uint64_t>    m_block_hash_value;
    cudatool::CudaDeviceBuffer<uint64_t>    m_block_sort_hash_value;
    cudatool::CudaDeviceBuffer<uint32_t>    m_block_index;
    cudatool::CudaDeviceBuffer<uint32_t>    m_block_sort_index;
    cudatool::CudaDeviceBuffer<uint32_t>    m_block_temp_buffer;
    int                                     m_block_rows = 0;
    int                                     m_block_cols = 0;

  public:
    GIPCTripletMatrix()                                    = default;
    ~GIPCTripletMatrix()                                   = default;
    GIPCTripletMatrix(const GIPCTripletMatrix&)            = default;
    GIPCTripletMatrix(GIPCTripletMatrix&&)                 = default;
    GIPCTripletMatrix& operator=(const GIPCTripletMatrix&) = default;
    GIPCTripletMatrix& operator=(GIPCTripletMatrix&&)      = default;

    void reshape(int row, int col)
    {
        m_block_rows = row;
        m_block_cols = col;
    }

    void resize_triplets(size_t nonzero_count)
    {
        m_block_values.resize(nonzero_count);
        m_block_row_indices.resize(nonzero_count);
        m_block_col_indices.resize(nonzero_count);
    }

    void reserve_triplets(size_t nonzero_count)
    {
        m_block_values.reserve(nonzero_count);
        m_block_row_indices.reserve(nonzero_count);
        m_block_col_indices.reserve(nonzero_count);
    }

    void resize(int row, int col, size_t nonzero_count)
    {
        reshape(row, col);
        resize_triplets(nonzero_count);
    }

    void resize_collision_hash_size(size_t nonzero_count)
    {
        m_block_hash_value.resize(nonzero_count);
        m_block_sort_hash_value.resize(nonzero_count);
        m_block_index.resize(nonzero_count);
        m_block_sort_index.resize(nonzero_count);
        m_block_temp_buffer.resize(nonzero_count);
    }

    void reset_zero()
    {
        m_block_values.reset_zero();
        m_block_row_indices.reset_zero();
        m_block_col_indices.reset_zero();
    }

    void update_hash_value(int fem_offset);

    auto block_values(int offset = 0) { return m_block_values.data() + offset; }
    auto block_values(int offset = 0) const
    {
        return m_block_values.data() + offset;
    }
    auto block_row_indices(int offset = 0)
    {
        return m_block_row_indices.data() + offset;
    }
    auto block_row_indices(int offset = 0) const
    {
        return m_block_row_indices.data() + offset;
    }
    auto block_col_indices(int offset = 0)
    {
        return m_block_col_indices.data() + offset;
    }
    auto block_col_indices(int offset = 0) const
    {
        return m_block_col_indices.data() + offset;
    }
    auto block_hash_value(int offset = 0)
    {
        return m_block_hash_value.data() + offset;
    }
    auto block_hash_value(int offset = 0) const
    {
        return m_block_hash_value.data() + offset;
    }

    auto block_sort_hash_value(int offset = 0)
    {
        return m_block_sort_hash_value.data() + offset;
    }
    auto block_sort_hash_value(int offset = 0) const
    {
        return m_block_sort_hash_value.data() + offset;
    }

    auto block_temp_buffer(int offset = 0)
    {
        return m_block_temp_buffer.data() + offset;
    }
    auto block_temp_buffer(int offset = 0) const
    {
        return m_block_temp_buffer.data() + offset;
    }

    auto block_index(int offset = 0) { return m_block_index.data() + offset; }
    auto block_index(int offset = 0) const
    {
        return m_block_index.data() + offset;
    }

    auto block_sort_index(int offset = 0)
    {
        return m_block_sort_index.data() + offset;
    }
    auto block_sort_index(int offset = 0) const
    {
        return m_block_sort_index.data() + offset;
    }

    auto block_rows() const { return m_block_rows; }
    auto block_cols() const { return m_block_cols; }
    auto triplet_count() const { return m_block_values.size(); }
    auto triplet_capacity() const { return m_block_values.capacity(); }

    void clear()
    {
        m_block_rows = 0;
        m_block_cols = 0;
        m_block_values.clear();
        m_block_row_indices.clear();
        m_block_col_indices.clear();
    }
    int global_triplet_offset           = 0;
    int global_collision_triplet_offset = 0;
    int global_external_max_capcity    = 0;
    int global_internal_capcity = 0;

    cudatool::CudaDeviceVar<int> d_abd_abd_contact_start_id = -1;
    cudatool::CudaDeviceVar<int> d_abd_fem_contact_start_id = -1;
    cudatool::CudaDeviceVar<int> d_fem_abd_contact_start_id = -1;
    cudatool::CudaDeviceVar<int> d_fem_fem_contact_start_id = -1;
    cudatool::CudaDeviceVar<int> d_unique_key_number        = 0;

    int h_abd_abd_contact_start_id = -1;
    int h_abd_fem_contact_start_id = -1;
    int h_fem_abd_contact_start_id = -1;
    int h_fem_fem_contact_start_id = -1;
    int h_unique_key_number        = 0;

    uint32_t abd_abd_contact_num = 0;
    uint32_t abd_fem_contact_num = 0;
    uint32_t fem_fem_contact_num = 0;
    uint32_t fem_abd_contact_num = 0;
};
