#include "dense_vector.hpp"

namespace CudaCPP
{

DenseVector::DenseVector(): m_data ( nullptr ), m_vector ( nullptr ), m_size( 0 )
{}


DenseVector::DenseVector( double* device_data, const size_t size,
                          const cusparseDnVecDescr_t device_vec_desc ): m_data( device_data ),
    m_vector( device_vec_desc ),
    m_size( size )
{}


DenseVector::~DenseVector()
{
    if ( m_vector != nullptr )
        cusparseDestroyDnVec( m_vector );

    if ( m_data != nullptr )
        cudaFree ( m_data );
}


std::unique_ptr <DenseVector> DenseVector::create( const double* host_data, const size_t size,
                                                   cudaDataType_t cuda_type )
{
    if ( size == 0 )
        return {};

    double* dev_data;

    if ( cudaMalloc( ( void** )&dev_data, size * sizeof( double ) ) != cudaSuccess )
        return {};

    if ( host_data != nullptr )
    {
        if ( cudaMemcpy ( dev_data, host_data, size * sizeof( double ), cudaMemcpyHostToDevice ) != cudaSuccess )
        {
            cudaFree ( dev_data );
            return {};
        }
    }

    cusparseDnVecDescr_t dev_vec_desc;

    if ( cusparseCreateDnVec( &dev_vec_desc, size, dev_data, cuda_type ) != CUSPARSE_STATUS_SUCCESS )
    {
        cudaFree( dev_data );
        return {};
    }

    return std::make_unique<DenseVector> ( dev_data, size, dev_vec_desc );
}

bool DenseVector::copyToHost( double* host_data_ptr, const size_t size )
{
    if ( host_data_ptr == nullptr )
        return 1;

    if ( size != m_size )
        return 1;

    if ( !size )
        return 1;

    if ( cudaMemcpy( host_data_ptr, m_data, m_size * sizeof( double ), cudaMemcpyDeviceToHost ) != cudaSuccess )
        return 1;

    return 0;
}


bool DenseVector::copy( const DenVecPtr& vector )
{
    if ( vector == nullptr )
        return 1;

    if ( vector->data() == nullptr )
        return 1;

    if ( vector->size() != m_size )
        return 1;

    if ( cudaMemcpy( m_data, vector->data(), m_size * sizeof( double ), cudaMemcpyDeviceToDevice ) != cudaSuccess )
        return 1;

    return 0;
}

bool DenseVector::norm( double& result,
                        cublasHandle_t cublas_handle ) const
{
    bool status = 0;
    cublasHandle_t cublas_handle_temp = cublas_handle;

    if ( m_data == nullptr )
        return !status;

    if ( cublas_handle == nullptr )
    {
        if ( cublasCreate ( &cublas_handle ) != CUBLAS_STATUS_SUCCESS )
            return !status;
    }

    if ( cublasDnrm2( cublas_handle, static_cast<int>( m_size ),  m_data, 1,
                      &result ) != CUBLAS_STATUS_SUCCESS )
        status = !status;

    if ( cublas_handle_temp == nullptr )
        cublasDestroy ( cublas_handle );

    return status;
}

bool DenseVector::dot( const DenVecPtr& vector, double& result,
                       cublasHandle_t cublas_handle ) const
{
    bool status = 0;
    cublasHandle_t cublas_handle_temp = cublas_handle;

    if ( ( vector->data() == nullptr ) )
        return !status;

    if   ( m_size != vector->size() )
        return !status;

    if ( cublas_handle == nullptr )
    {
        if ( cublasCreate ( &cublas_handle ) != CUBLAS_STATUS_SUCCESS )
            return !status;
    }

    if ( cublasDdot( cublas_handle,
                     static_cast<int>( m_size ),
                     m_data,
                     1,
                     vector->data(),
                     1,
                     &result
                   ) != CUBLAS_STATUS_SUCCESS )
        status = !status;

    if ( cublas_handle_temp == nullptr )
        cublasDestroy ( cublas_handle );

    return status;
}


bool DenseVector::axpy( DenVecPtr& vector, const double alpha,
                        cublasHandle_t cublas_handle ) const
{
    bool status = 0;
    cublasHandle_t cublas_handle_temp = cublas_handle;

    if ( vector->size() != m_size )
        return !status;

    if ( vector->data() == nullptr )
        return !status;

    if ( cublas_handle == nullptr )
    {
        if ( cublasCreate ( &cublas_handle ) != CUBLAS_STATUS_SUCCESS )
            return !status;
    }

    if ( cublasDaxpy( cublas_handle,
                      static_cast<int>( m_size ),
                      &alpha,
                      m_data,
                      1,
                      vector->data(),
                      1
                    ) != CUBLAS_STATUS_SUCCESS )
        status = !status;

    if ( cublas_handle_temp == nullptr )
        cublasDestroy ( cublas_handle );

    return status;
}


bool DenseVector::setZero( )
{
    if ( !m_size )
        return 1;

    if ( cudaMemset( m_data, 0x0, m_size * sizeof( double ) ) != cudaSuccess )
        return 1;

    return 0;
}

bool DenseVector::scale( const double alpha, cublasHandle_t cublas_handle )
{
    bool status = 0;

    cublasHandle_t cublas_handle_temp = cublas_handle;

    if ( cublas_handle == nullptr )
    {
        if ( cublasCreate ( &cublas_handle ) != CUBLAS_STATUS_SUCCESS )
            return !status;
    }

    if ( cublasDscal( cublas_handle,
                      static_cast<int>( m_size ),
                      &alpha,
                      m_data,
                      1 ) != CUBLAS_STATUS_SUCCESS )
        status = !status;

    if ( cublas_handle_temp == nullptr )
        cublasDestroy ( cublas_handle );

    return status;
}
}
