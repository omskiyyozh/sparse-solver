#include "incompleteLU.hpp"

using Vectorxd = Eigen::VectorXd;
using SparseMatrixD = Eigen::SparseMatrix<double, Eigen::RowMajor>;
using SpMatrixCSR = CudaCPP::SpMatrixCSR;
using SpMatrixAttributes = CudaCPP::SpMatrixAttributes;
using DenseVectorD = CudaCPP::DenseVector;
using IncompleteLU = CudaCPP::IncompleteLU;

TEST ( incompleteLU, correctFactorization )
{
    SparseMatrixD matrix;
    ASSERT_TRUE( Eigen::loadMarket<SparseMatrixD>( matrix, "Trefethen_150.mtx" ) );
    Eigen::IncompleteLUT<double> preconditioner;
    preconditioner.compute( matrix );
    ASSERT_EQ( Eigen::Success, preconditioner.info() );

    const size_t nnz = matrix.nonZeros();
    const size_t dim = matrix.rows();

    SpMatrixAttributes attributes;
    attributes.col_ind_type = CUSPARSE_INDEX_32I;
    attributes.row_offsets_type = CUSPARSE_INDEX_32I;
    attributes.values_type = CUDA_R_64F;
    attributes.index_base = CUSPARSE_INDEX_BASE_ZERO;

    auto matrix_test = SpMatrixCSR::create( matrix.valuePtr(), matrix.innerIndexPtr(), matrix.outerIndexPtr(), dim, nnz,
                                            attributes );
    ASSERT_NE ( nullptr, matrix_test );

    cusparseHandle_t cusparse_handle;
    ASSERT_EQ( CUSPARSE_STATUS_SUCCESS, cusparseCreate( &cusparse_handle ) );
    IncompleteLU preconditioner_test;

    ASSERT_EQ( 0, preconditioner_test.init( matrix_test, cusparse_handle ) );
    ASSERT_EQ( 0, preconditioner_test.compute( matrix_test, cusparse_handle ) );

}
