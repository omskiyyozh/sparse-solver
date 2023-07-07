#include "incompleteLU.hpp"

namespace CudaCPP
{

IncompleteLU::IncompleteLU(): m_info( nullptr ), m_buffer( nullptr ), m_matrix_LU( nullptr )
{}

IncompleteLU::~IncompleteLU()
{
    if ( m_matrix_LU != nullptr )
        cusparseDestroyMatDescr( m_matrix_LU );

    if ( m_info != nullptr )
        cusparseDestroyCsrilu02Info( m_info );

    if ( m_buffer != nullptr )
        cudaFree( m_buffer );
}

bool IncompleteLU::init( SpCSRMatrixPtr& matrix, cusparseHandle_t cusparse_handle )
{
    if ( matrix == nullptr || cusparse_handle == nullptr )
        return 1;

    if ( m_matrix_LU == nullptr && m_buffer == nullptr && m_info == nullptr )
    {
        if ( cusparseCreateMatDescr( &m_matrix_LU ) != CUSPARSE_STATUS_SUCCESS )
            return 1;

        if ( cusparseSetMatType( m_matrix_LU, CUSPARSE_MATRIX_TYPE_GENERAL ) )
        {
            cusparseDestroyMatDescr( m_matrix_LU );
            return 1;
        }

        if ( cusparseSetMatIndexBase( m_matrix_LU, CUSPARSE_INDEX_BASE_ZERO ) )
        {
            cusparseDestroyMatDescr( m_matrix_LU );
            return 1;
        }

        if ( cusparseCreateCsrilu02Info( &m_info ) != CUSPARSE_STATUS_SUCCESS )
        {
            cusparseDestroyMatDescr( m_matrix_LU );
            return 1;
        }

        int buffer_size;

        if ( cusparseDcsrilu02_bufferSize(
                    cusparse_handle,
                    static_cast<int>( matrix->dim() ),
                    static_cast<int>( matrix->nnz() ),
                    m_matrix_LU,
                    matrix->values(),
                    matrix->rowPtr(),
                    matrix->colInd(),
                    m_info,
                    &buffer_size
                ) != CUSPARSE_STATUS_SUCCESS )
        {
            cusparseDestroyMatDescr( m_matrix_LU );
            return 1;
        }

        if ( cudaMalloc( &m_buffer, buffer_size ) != cudaSuccess )
        {
            cusparseDestroyMatDescr( m_matrix_LU );
            return 1;
        }
    }

    return 0;
}

bool IncompleteLU::analyze( SpCSRMatrixPtr& matrix, cusparseHandle_t cusparse_handle )
{
    if ( matrix == nullptr || cusparse_handle == nullptr )
        return 1;

    if ( cusparseDcsrilu02_analysis(
                cusparse_handle,
                static_cast<int>( matrix->dim() ),
                static_cast<int>( matrix->nnz() ),
                m_matrix_LU,
                matrix->values(),
                matrix->rowPtr(),
                matrix->colInd(),
                m_info,
                CUSPARSE_SOLVE_POLICY_USE_LEVEL,
                m_buffer
            ) != CUSPARSE_STATUS_SUCCESS )
        return 1;

    return 0;
}

bool IncompleteLU::factorize( SpCSRMatrixPtr& matrix, cusparseHandle_t cusparse_handle )
{
    int structural_zero;

    if ( cusparseXcsrilu02_zeroPivot( cusparse_handle,
                                      m_info,
                                      &structural_zero
                                    ) != CUSPARSE_STATUS_SUCCESS )
        return 1;

    if ( cusparseDcsrilu02(
                cusparse_handle,
                static_cast<int>( matrix->dim() ),
                static_cast<int>( matrix->nnz() ),
                m_matrix_LU,
                matrix->values(),
                matrix->rowPtr(),
                matrix->colInd(),
                m_info,
                CUSPARSE_SOLVE_POLICY_USE_LEVEL,
                m_buffer
            ) != CUSPARSE_STATUS_SUCCESS )
        return 1;

    return 0;
}

bool IncompleteLU::compute( SpCSRMatrixPtr& matrix, cusparseHandle_t cusparse_handle )
{

    if ( analyze( matrix, cusparse_handle ) )
        return 1;

    if ( factorize( matrix, cusparse_handle ) )
        return 1;

    return 0;
}

}
