//#include "cuda_tools/cuda_device_buffer.h"
namespace cudatool
{

template <typename T>
CudaDeviceBuffer<T>::CudaDeviceBuffer()
    : m_data(nullptr)
    , m_size(0)
    , m_capacity(0)
{
}

template <typename T>
CudaDeviceBuffer<T>::CudaDeviceBuffer(size_t n)
{
    CUDA_SAFE_CALL(cudaMalloc((void**)&m_data, n * sizeof(value_type)));
    m_size     = n;
    m_capacity = n;
}


template <typename T>
CudaDeviceBuffer<T>::CudaDeviceBuffer(const CudaDeviceBuffer<T>& other)
{
    m_size     = other.size();
    m_capacity = other.capacity();
    CUDA_SAFE_CALL(cudaMalloc((void**)&m_data, m_capacity * sizeof(T)));
    CUDA_SAFE_CALL(cudaMemcpy(m_data, other.data(), m_size * sizeof(T), cudaMemcpyDeviceToDevice));
}

template <typename T>
CudaDeviceBuffer<T>::CudaDeviceBuffer(CudaDeviceBuffer<T>&& other) noexcept
    : m_data(other.data())
    , m_size(other.m_size)
    , m_capacity(other.m_capacity)
{
    other.clear();
}

template <typename T>
CudaDeviceBuffer<T>& CudaDeviceBuffer<T>::operator=(const CudaDeviceBuffer<T>& other)
{
    if(this == &other)
        return *this;

    this->resize(other.size());
    CUDA_SAFE_CALL(cudaMemcpy(m_data, other.data(), m_size * sizeof(T), cudaMemcpyDeviceToDevice));
    return *this;
}

template <typename T>
CudaDeviceBuffer<T>& CudaDeviceBuffer<T>::operator=(CudaDeviceBuffer<T>&& other)
{
    if(this == &other)
        return *this;

    if(m_data)
    {
        CUDA_SAFE_CALL(cudaFree(m_data));
    }

    m_data     = other.data();
    m_size     = other.m_size;
    m_capacity = other.m_capacity;

    other.clear();

    return *this;
}

template <typename T>
CudaDeviceBuffer<T>& CudaDeviceBuffer<T>::operator=(const std::vector<T>& other)
{
    copy_from_host(other);
    return *this;
}

template <typename T>
CudaDeviceBuffer<T>::CudaDeviceBuffer(const std::vector<T>& host)
{
    m_size     = host.size();
    m_capacity = m_size;
    CUDA_SAFE_CALL(cudaMalloc((void**)&m_data, m_capacity * sizeof(T)));
    CUDA_SAFE_CALL(cudaMemcpy(m_data, host.data(), m_size * sizeof(T), cudaMemcpyHostToDevice));
}


template <typename T>
void CudaDeviceBuffer<T>::copy_to_host(std::vector<T>& host) const
{
    host.resize(m_size);
    CUDA_SAFE_CALL(cudaMemcpy(host.data(), m_data, m_size * sizeof(T), cudaMemcpyDeviceToHost));
}

template <typename T>
void CudaDeviceBuffer<T>::copy_from_host(const std::vector<T>& host)
{
    if(host.size() < 1)
        return;
    resize(host.size());
    CUDA_SAFE_CALL(cudaMemcpy(m_data, host.data(), host.size() * sizeof(value_type), cudaMemcpyHostToDevice));
}

template <typename T>
void CudaDeviceBuffer<T>::resize(size_t new_size)
{
    if(new_size <= m_capacity)
    {
        m_size = new_size;
    }
    else
    {
        m_capacity = new_size * 1.2;
        m_size     = new_size;
        CUDA_SAFE_CALL(cudaFree(m_data));
        CUDA_SAFE_CALL(cudaMalloc((void**)&m_data, m_capacity * sizeof(T)));
    }
}

template <typename T>
void CudaDeviceBuffer<T>::reset_zero()
{
    CUDA_SAFE_CALL(cudaMemset(m_data, 0, m_size * sizeof(T)));
}

//template <typename T>
//void CudaDeviceBuffer<T>::resize(size_t new_size, T value)
//{
//
//    if(new_size <= m_capacity)
//    {
//        m_size = new_size;
//    }
//    else
//    {
//        m_capacity = new_size;
//        m_size     = new_size;
//        CUDA_SAFE_CALL(cudaFree(m_data));
//        CUDA_SAFE_CALL(cudaMalloc((void**)&m_data, new_size * sizeof(value_type)));
//    }
//    int cudablocksize = 256;
//    int cudablocknum  = (new_size + cudablocksize - 1) / cudablocksize;
//    //kernel_fillm(new_size);
//}

template <typename T>
void CudaDeviceBuffer<T>::reserve(size_t new_capacity)
{
    if(new_capacity <= m_capacity)
    {
        return;
    }
    else
    {

        T* new_data;
        CUDA_SAFE_CALL(cudaMalloc((void**)&new_data, new_capacity * sizeof(T)));
        CUDA_SAFE_CALL(cudaMemcpy(new_data, m_data, m_size * sizeof(T), cudaMemcpyDeviceToDevice));
        CUDA_SAFE_CALL(cudaFree(m_data));
        m_data     = new_data;
        m_capacity = new_capacity;
    }
}

template <typename T>
void CudaDeviceBuffer<T>::clear()
{
    //CUDA_SAFE_CALL(cudaFree(m_data));
    m_data     = nullptr;
    m_size     = 0;
    m_capacity = 0;
}

//
//template <typename T>
//void CudaDeviceBuffer<T>::fill(const T& v)
//{
//    int cudablocksize = 256;
//    int cudablocknum  = (m_size + cudablocksize - 1) / cudablocksize;
//    //LaunchCudaKernal(cudablocksize, cudablocknum, 0, kernel_fill, m_data, value, new_size);
//}


template <typename T>
CudaDeviceBuffer<T>::~CudaDeviceBuffer()
{
    CUDA_SAFE_CALL(cudaFree(m_data));
}


/// <summary>
/// //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// </summary>
/// <typeparam name="T"></typeparam>

//template <typename T>
//CudaDeviceVar<T>::CudaDeviceVar()
//{
//    CUDA_SAFE_CALL(cudaMalloc((void**)&m_data, sizeof(T)));
//}
//
//template <typename T>
//CudaDeviceVar<T>::CudaDeviceVar(const T& value)
//{
//    CUDA_SAFE_CALL(cudaMalloc((void**)&m_data, sizeof(T)));
//    CUDA_SAFE_CALL(cudaMemcpy(m_data, &value, sizeof(T), cudaMemcpyHostToDevice));
//}
//
//template <typename T>
//CudaDeviceVar<T>::CudaDeviceVar(const CudaDeviceVar& other)
//{
//    CUDA_SAFE_CALL(cudaMalloc((void**)&m_data, sizeof(T)));
//    CUDA_SAFE_CALL(cudaMemcpy(m_data, other.data(), sizeof(T), cudaMemcpyDeviceToDevice));
//}
//
//template <typename T>
//CudaDeviceVar<T>::CudaDeviceVar(CudaDeviceVar&& other) noexcept
//    : m_data(other.data())
//{
//    other.clear();
//}
//
//template <typename T>
//CudaDeviceVar<T>& CudaDeviceVar<T>::operator=(const CudaDeviceVar<T>& other)
//{
//    if(this == &other)
//        return *this;
//    CUDA_SAFE_CALL(cudaMemcpy(m_data, other.data(), sizeof(T), cudaMemcpyDeviceToDevice));
//    return *this;
//}
//
//template <typename T>
//CudaDeviceVar<T>& CudaDeviceVar<T>::operator=(CudaDeviceVar<T>&& other)
//{
//    if(this == &other)
//        return *this;
//
//    if(m_data)
//        CUDA_SAFE_CALL(cudaFree(m_data));
//
//    m_data = other.data();
//
//    other.clear();
//
//    return *this;
//}
//
//template <typename T>
//CudaDeviceVar<T>& CudaDeviceVar<T>::operator=(const T& val)
//{
//    CUDA_SAFE_CALL(cudaMemcpy(m_data, &val, sizeof(T), cudaMemcpyHostToDevice));
//    return *this;
//}

//template <typename T>
//CudaDeviceVar<T>::operator T() const
//{
//    T value;
//    CUDA_SAFE_CALL(cudaMemcpy(&value, m_data, sizeof(T), cudaMemcpyDeviceToHost));
//    return value;
//}
//template <typename T>
//CudaDeviceVar<T>::~CudaDeviceVar()
//{
//    CUDA_SAFE_CALL(cudaFree(m_data));
//}
}  // namespace cudatool

