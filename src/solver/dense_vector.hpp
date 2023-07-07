#pragma once

#include "Stable.hpp"

namespace CudaCPP
{

class DenseVector
{
public:
    using DenVecPtr = std::unique_ptr<DenseVector>;

    DenseVector();
    DenseVector ( const DenseVector& ) = delete;
    DenseVector ( double* device_data, const size_t size, const cusparseDnVecDescr_t device_vec_desc );

    ~DenseVector();

    static DenVecPtr create ( const double* host_data, const size_t size, cudaDataType_t cuda_type );

    bool copyToHost ( double* host_data_ptr, const size_t size );
    bool copy( const DenVecPtr& vector );

    bool norm( double& result,
               cublasHandle_t cublas_handle = nullptr ) const;
    bool dot ( const DenVecPtr& vector, double& result, cublasHandle_t cublas_handle = nullptr ) const;
    bool axpy ( DenVecPtr& vector, const double alpha, cublasHandle_t cublas_handle = nullptr ) const;
    bool setZero( );
    bool scale ( const double alpha, cublasHandle_t cublas_handle = nullptr );

    inline double* data() const
    {
        return m_data;
    }

    inline size_t size() const
    {
        return m_size;
    }

    inline cusparseDnVecDescr_t vector() const
    {
        return m_vector;
    }

private:
    double* m_data;
    cusparseDnVecDescr_t m_vector;
    size_t m_size;
};

}
