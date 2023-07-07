#include "spcsr_matrix.hpp"

namespace CudaCPP
{
SpMatrixCSR::SpMatrixCSR(): m_values( nullptr ), m_col_ind( nullptr ), m_row_ptr( nullptr ), m_dim( 0 ),
    m_nnz( 0 ), m_mat_desc( nullptr )
{}

SpMatrixCSR::SpMatrixCSR( double* dev_values, int* dev_col_ind, int* dev_row_ptr, const size_t dim,
                          const size_t nnz, const cusparseSpMatDescr_t dev_mat_desc ):
    m_values( dev_values ), m_col_ind( dev_col_ind ), m_row_ptr( dev_row_ptr ),
    m_dim( dim ), m_nnz( nnz ), m_mat_desc( dev_mat_desc )
{}

SpMatrixCSR::~SpMatrixCSR()
{
    cudaFree( m_values );
    cudaFree( m_col_ind );
    cudaFree( m_row_ptr );
    cusparseDestroySpMat( m_mat_desc );
}


SpMatrixCSR::SpCSRPtr SpMatrixCSR::create( double* values, int* col_ind, int* row_ptr,
                                           const size_t dim, const size_t nnz, SpMatrixAttributes mat_attributes,
                                           const DataLocation data_location )
{
    if ( !dim || !nnz )
        return {};

    if ( row_ptr == nullptr && col_ind == nullptr && values == nullptr )
        return {};

    double* dev_values = nullptr;

    int* dev_col_ind = nullptr, *dev_row_ptr = nullptr;

    size_t offsets = dim + 1;

    switch ( data_location )
    {
        case DataLocation::HOST:
            if ( cudaMalloc( ( void** ) &dev_row_ptr, offsets * sizeof( int ) ) != cudaSuccess )
                return {};

            if ( cudaMalloc( ( void** ) &dev_col_ind, nnz * sizeof( int ) ) != cudaSuccess )
            {
                cudaFree( dev_row_ptr );
                return {};
            }

            if ( cudaMalloc( ( void** ) &dev_values,  nnz * sizeof( double ) ) != cudaSuccess )
            {
                cudaFree( dev_row_ptr );
                cudaFree( dev_col_ind );
                return {};
            }

            if ( cudaMemcpy( dev_row_ptr, row_ptr, offsets * sizeof( int ),
                             cudaMemcpyHostToDevice ) != cudaSuccess )
            {
                cudaFree( dev_row_ptr );
                cudaFree( dev_col_ind );
                cudaFree( dev_values );
                return {};
            }

            if ( cudaMemcpy( dev_col_ind, col_ind, nnz *  sizeof( int ),
                             cudaMemcpyHostToDevice ) != cudaSuccess )
            {
                cudaFree( dev_row_ptr );
                cudaFree( dev_col_ind );
                cudaFree( dev_values );
                return {};
            }

            if ( cudaMemcpy( dev_values, values, nnz * sizeof( double ),
                             cudaMemcpyHostToDevice ) != cudaSuccess )
            {
                cudaFree( dev_row_ptr );
                cudaFree( dev_col_ind );
                cudaFree( dev_values );
                return {};
            }

            break;

        case DataLocation::DEVICE:
            dev_values = values;
            dev_col_ind = col_ind;
            dev_row_ptr = row_ptr;
            break;
    }

    cusparseSpMatDescr_t dev_mat_desc;

    if ( cusparseCreateCsr( &dev_mat_desc, dim, dim, nnz, dev_row_ptr,
                            dev_col_ind, dev_values,
                            mat_attributes.row_offsets_type,
                            mat_attributes.col_ind_type,
                            mat_attributes.index_base,
                            mat_attributes.values_type ) != CUSPARSE_STATUS_SUCCESS )
    {
        cudaFree( dev_row_ptr );
        cudaFree( dev_col_ind );
        cudaFree( dev_values );
        return {};
    }

    if ( cusparseSpMatSetAttribute( dev_mat_desc,
                                    CUSPARSE_SPMAT_FILL_MODE,
                                    &mat_attributes.fill_mode,
                                    sizeof( mat_attributes.fill_mode ) ) != CUSPARSE_STATUS_SUCCESS )
    {
        cudaFree( dev_row_ptr );
        cudaFree( dev_col_ind );
        cudaFree( dev_values );
        return {};
    }

    if ( cusparseSpMatSetAttribute( dev_mat_desc,
                                    CUSPARSE_SPMAT_DIAG_TYPE,
                                    &mat_attributes.diag_type,
                                    sizeof( mat_attributes.diag_type ) ) != CUSPARSE_STATUS_SUCCESS )
    {
        cudaFree( dev_row_ptr );
        cudaFree( dev_col_ind );
        cudaFree( dev_values );
        return {};
    }

    return std::make_unique<SpMatrixCSR>( dev_values, dev_col_ind, dev_row_ptr, dim, nnz, dev_mat_desc );
}

bool SpMatrixCSR::copy( const SpCSRPtr& matrix )
{
    if ( m_dim != matrix->dim() )
        return 1;

    if ( m_nnz != matrix->nnz() )
        return 1;

    if ( cudaMemcpy( m_values, matrix->values(), m_nnz * sizeof( double ), cudaMemcpyDeviceToDevice ) != cudaSuccess )
        return 1;

    if ( cudaMemcpy( m_col_ind, matrix->colInd(), m_nnz * sizeof( int ), cudaMemcpyDeviceToDevice ) != cudaSuccess )
        return 1;

    if ( cudaMemcpy( m_row_ptr, matrix->rowPtr(), ( m_dim + 1 ) * sizeof( int ),
                     cudaMemcpyDeviceToDevice ) != cudaSuccess )
        return 1;

    return 0;
}

bool SpMatrixCSR::spMV( cusparseOperation_t op_A, const double alpha, const double beta,
                        std::unique_ptr<DenseVector>& vec_X,
                        std::unique_ptr<DenseVector>& vec_Y,
                        cusparseSpMVAlg_t alg_type,
                        cudaDataType compute_type,
                        cusparseHandle_t cusparse_handle )
{
    bool status = 0;
    cusparseHandle_t cusparse_handle_tmp = cusparse_handle;

    if ( vec_X == nullptr || vec_Y == nullptr )
        return !status;

    if ( cusparse_handle == nullptr )
    {
        if ( cusparseCreate( &cusparse_handle ) != CUSPARSE_STATUS_SUCCESS )
            return !status;
    }

    size_t buffer_size_MV;
    void*  d_buffer_MV;

    if ( cusparseSpMV_bufferSize(
                cusparse_handle,
                op_A,
                &alpha,
                m_mat_desc,
                vec_X->vector(),
                &beta,
                vec_Y->vector(),
                compute_type,
                alg_type,
                &buffer_size_MV
            ) != CUSPARSE_STATUS_SUCCESS )
    {
        cusparseDestroy( cusparse_handle );
        return !status;
    }

    if ( cudaMalloc( &d_buffer_MV, buffer_size_MV ) != cudaSuccess )
    {
        cusparseDestroy( cusparse_handle );
        return !status;
    }

    if ( cusparseSpMV( cusparse_handle,
                       op_A,
                       &alpha,
                       m_mat_desc,
                       vec_X->vector(),
                       &beta,
                       vec_Y->vector(),
                       compute_type,
                       alg_type,
                       d_buffer_MV
                     ) != CUSPARSE_STATUS_SUCCESS )
        status = !status;

    cudaFree( d_buffer_MV );

    if ( cusparse_handle_tmp == nullptr )
        cusparseDestroy( cusparse_handle );

    return status;
}

}
