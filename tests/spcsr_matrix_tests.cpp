#include "spcsr_matrix.hpp"

using Vectorxd = Eigen::VectorXd;
using SparseMatrixD = Eigen::SparseMatrix<double, Eigen::RowMajor>;
using SpMatrixCSR = CudaCPP::SpMatrixCSR;
using SpMatrixAttributes = CudaCPP::SpMatrixAttributes;
using DenseVectorD = CudaCPP::DenseVector;

TEST ( sparseMatrix, successfulCreation )
{
    SparseMatrixD matrix;
    ASSERT_TRUE( Eigen::loadMarket<SparseMatrixD>( matrix, "Trefethen_150.mtx" ) );

    const size_t nnz = matrix.nonZeros();
    const size_t dim = matrix.rows();
    std::vector <double> values ( matrix.valuePtr(), matrix.valuePtr() + nnz );
    std::vector <int> rows_ptr ( matrix.outerIndexPtr(), matrix.outerIndexPtr() + dim );
    std::vector <int> col_ind ( matrix.innerIndexPtr(), matrix.innerIndexPtr() + dim );

    SpMatrixAttributes attributes;
    attributes.col_ind_type = CUSPARSE_INDEX_32I;
    attributes.row_offsets_type = CUSPARSE_INDEX_32I;
    attributes.values_type = CUDA_R_64F;
    attributes.index_base = CUSPARSE_INDEX_BASE_ZERO;

    auto matrix_test = SpMatrixCSR::create( matrix.valuePtr(), matrix.innerIndexPtr(), matrix.outerIndexPtr(), dim, nnz,
                                            attributes );
    ASSERT_NE ( nullptr, matrix_test );

    std::vector <double> res_values ( nnz );
    std::vector <int> res_rows_ptr ( dim );
    std::vector <int> res_col_ind ( dim );

    ASSERT_EQ( cudaSuccess, cudaMemcpy( res_values.data(), matrix_test->values(), matrix_test->nnz() * sizeof( double ),
                                        cudaMemcpyDeviceToHost ) );
    ASSERT_EQ( cudaSuccess, cudaMemcpy( res_rows_ptr.data(), matrix_test->rowPtr(), matrix_test->dim() * sizeof( int ),
                                        cudaMemcpyDeviceToHost ) );
    ASSERT_EQ( cudaSuccess, cudaMemcpy( res_col_ind.data(), matrix_test->colInd(), matrix_test->dim() * sizeof( int ),
                                        cudaMemcpyDeviceToHost ) );

    ASSERT_EQ( values, res_values );
    ASSERT_EQ( rows_ptr, res_rows_ptr );
    ASSERT_EQ( col_ind, res_col_ind );
}

TEST ( sparseMatrix, creationWithZeroSizeOrNNZ )
{

    SpMatrixAttributes attributes;
    attributes.col_ind_type = CUSPARSE_INDEX_32I;
    attributes.row_offsets_type = CUSPARSE_INDEX_32I;
    attributes.values_type = CUDA_R_64F;
    attributes.index_base = CUSPARSE_INDEX_BASE_ZERO;

    double values [] = {1.0, 5.0, 4.5, 2.3, 1.2};
    int rows_ptr[] {1, 2, 3, 4, 5};
    int col_ptr[] {1, 2, 3, 4, 5};

    ASSERT_EQ( nullptr, SpMatrixCSR::create( values, rows_ptr, col_ptr, 0, sizeof( values ) / sizeof( double ),
                                             attributes ) );
    ASSERT_EQ( nullptr, SpMatrixCSR::create( values, rows_ptr, col_ptr, sizeof( values ) / sizeof( double ), 0,
                                             attributes ) );
}

TEST ( sparseMatrix, creationWithHostNullptr )
{
    SpMatrixAttributes attributes;
    attributes.col_ind_type = CUSPARSE_INDEX_32I;
    attributes.row_offsets_type = CUSPARSE_INDEX_32I;
    attributes.values_type = CUDA_R_64F;
    attributes.index_base = CUSPARSE_INDEX_BASE_ZERO;


    ASSERT_EQ( nullptr, SpMatrixCSR::create( nullptr, nullptr, nullptr, 15, 10,
                                             attributes ) );
}

TEST ( sparseMatrix, creationWithInvalidAttributes )
{
    SpMatrixAttributes attributes;
    attributes.col_ind_type = CUSPARSE_INDEX_32I;
    attributes.row_offsets_type = CUSPARSE_INDEX_32I;

    ASSERT_EQ( nullptr, SpMatrixCSR::create( nullptr, nullptr, nullptr, 15, 10,
                                             attributes ) );
}

TEST ( sparseMatrix, spMV )
{
    SparseMatrixD matrix;
    ASSERT_TRUE( Eigen::loadMarket<SparseMatrixD>( matrix, "Trefethen_150.mtx" ) );
    Vectorxd vec ( matrix.rows() );
    vec.setOnes();
    Vectorxd spmv_exp = matrix * vec;

    const size_t nnz = matrix.nonZeros();
    const size_t dim = matrix.rows();

    SpMatrixAttributes attributes;
    attributes.col_ind_type = CUSPARSE_INDEX_32I;
    attributes.row_offsets_type = CUSPARSE_INDEX_32I;
    attributes.values_type = CUDA_R_64F;
    attributes.index_base = CUSPARSE_INDEX_BASE_ZERO;

    auto matrix_test = SpMatrixCSR::create( matrix.valuePtr(), matrix.innerIndexPtr(), matrix.outerIndexPtr(), dim, nnz,
                                            attributes );
    auto vector_test = DenseVectorD::create( vec.data(), vec.size(), CUDA_R_64F );
    auto spmv_res = DenseVectorD::create( nullptr, vec.size(), CUDA_R_64F );

    ASSERT_NE ( nullptr, matrix_test );
    ASSERT_NE ( nullptr, vector_test );
    ASSERT_NE ( nullptr, spmv_res );

    ASSERT_EQ( 0, matrix_test->spMV( CUSPARSE_OPERATION_NON_TRANSPOSE, 1, 0, vector_test, spmv_res,
                                     CUSPARSE_SPMV_ALG_DEFAULT, CUDA_R_64F ) );

    const size_t spmv_res_size = spmv_res->size();
    double* host_spmv_res = new double[spmv_res_size];
    ASSERT_EQ( 0, spmv_res->copyToHost( host_spmv_res, spmv_res_size ) );

    for ( double* i = host_spmv_res, *j = spmv_exp.data(), *end = host_spmv_res + spmv_res_size; i < end; )
        EXPECT_DOUBLE_EQ( *i++, *j++ );

    delete [] host_spmv_res;
}

TEST( sparseMatrix, successfulCopy )
{
    SparseMatrixD matrix;
    ASSERT_TRUE( Eigen::loadMarket<SparseMatrixD>( matrix, "Trefethen_150.mtx" ) );

    const size_t nnz = matrix.nonZeros();
    const size_t dim = matrix.rows() + 1;
    std::vector <double> values ( matrix.valuePtr(), matrix.valuePtr() + nnz );
    std::vector <int> rows_ptr ( matrix.outerIndexPtr(), matrix.outerIndexPtr() + dim );
    std::vector <int> col_ind ( matrix.innerIndexPtr(), matrix.innerIndexPtr() + nnz );

    SpMatrixAttributes attributes;
    attributes.col_ind_type = CUSPARSE_INDEX_32I;
    attributes.row_offsets_type = CUSPARSE_INDEX_32I;
    attributes.values_type = CUDA_R_64F;
    attributes.index_base = CUSPARSE_INDEX_BASE_ZERO;

    auto matrix_test = SpMatrixCSR::create( matrix.valuePtr(), matrix.innerIndexPtr(), matrix.outerIndexPtr(), dim, nnz,
                                            attributes );
    ASSERT_NE ( nullptr, matrix_test );

    auto matrix_copy = SpMatrixCSR::create( matrix.valuePtr(), matrix.innerIndexPtr(), matrix.outerIndexPtr(), dim, nnz,
                                            attributes );

    ASSERT_NE ( nullptr, matrix_copy );

    ASSERT_EQ( 0, matrix_copy->copy( matrix_test ) );

    std::vector <double> res_values ( nnz );
    std::vector <int> res_rows_ptr ( dim );
    std::vector <int> res_col_ind ( nnz );

    ASSERT_EQ( cudaSuccess, cudaMemcpy( res_values.data(), matrix_copy->values(), matrix_test->nnz() * sizeof( double ),
                                        cudaMemcpyDeviceToHost ) );
    ASSERT_EQ( cudaSuccess, cudaMemcpy( res_rows_ptr.data(), matrix_copy->rowPtr(), matrix_test->dim() * sizeof( int ),
                                        cudaMemcpyDeviceToHost ) );
    ASSERT_EQ( cudaSuccess, cudaMemcpy( res_col_ind.data(), matrix_copy->colInd(), matrix_test->nnz() * sizeof( int ),
                                        cudaMemcpyDeviceToHost ) );

    ASSERT_EQ ( res_values, values );
    ASSERT_EQ( rows_ptr, res_rows_ptr );
    ASSERT_EQ( col_ind, res_col_ind );
}
