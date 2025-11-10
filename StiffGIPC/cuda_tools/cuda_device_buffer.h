#pragma once
#include "cuda_tools/cuda_tools.h"
namespace cudatool
{

template <typename T>
class CudaDeviceBuffer
{
  private:
    size_t m_size     = 0;
    size_t m_capacity = 0;
    T*     m_data     = nullptr;

  public:
    using value_type = T;

    CudaDeviceBuffer(size_t n);
    CudaDeviceBuffer();

    CudaDeviceBuffer(const CudaDeviceBuffer<T>& other);
    CudaDeviceBuffer(CudaDeviceBuffer&& other) noexcept;
    CudaDeviceBuffer& operator=(const CudaDeviceBuffer<T>& other);
    CudaDeviceBuffer& operator=(CudaDeviceBuffer<T>&& other);
    CudaDeviceBuffer& operator=(const std::vector<T>& other);

    CudaDeviceBuffer(const std::vector<T>& host);


    ~CudaDeviceBuffer();

    void copy_to_host(std::vector<T>& host) const;
    void copy_from_host(const std::vector<T>& host);

    void resize(size_t new_size);
    //void resize(size_t new_size, T value);
    void reserve(size_t new_capacity);
    void clear();
    void reset_zero();
    //void shrink_to_fit();
    //void fill(const T& v);

    //Dense1D<T>  viewer() MUDA_NOEXCEPT;
    //CDense1D<T> cviewer() const MUDA_NOEXCEPT;

    //BufferView<T>  view(size_t offset, size_t size = ~0) MUDA_NOEXCEPT;
    //BufferView<T>  view() MUDA_NOEXCEPT;
    //CBufferView<T> view(size_t offset, size_t size = ~0) const MUDA_NOEXCEPT;
    //CBufferView<T> view() const MUDA_NOEXCEPT;
    //operator BufferView<T>() MUDA_NOEXCEPT { return view(); }
    //operator CBufferView<T>() const MUDA_NOEXCEPT { return view(); }

    //

    size_t   size() const noexcept { return m_size; }
    size_t   capacity() const noexcept { return m_capacity; }
    T*       data() noexcept { return m_data; }
    const T* data() const noexcept { return m_data; }
};


//template <typename T>
//class CudaDeviceVar
//{
//  private:
//    T* m_data;
//
//  public:
//    using value_type = T;
//
//    CudaDeviceVar();
//    ~CudaDeviceVar();
//    CudaDeviceVar(const T& value);
//
//    CudaDeviceVar(const CudaDeviceVar& other);
//    CudaDeviceVar(CudaDeviceVar&& other) noexcept;
//    CudaDeviceVar& operator=(const CudaDeviceVar<T>& other);
//    CudaDeviceVar& operator=(CudaDeviceVar<T>&& other);
//
//    CudaDeviceVar& operator=(const T& val);  // copy from host
//    //operator T() const;                      // copy to host
//    T*       data() noexcept { return m_data; }
//    const T* data() const noexcept { return m_data; }
//    void clear() { m_data = nullptr; }
//    
//};

}  // namespace cudatool


#include "cuda_tools/cuda_device_buffer.inl"