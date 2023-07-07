#include "bicgstab.hpp"

#define UNUSED(x) (void)(x)
using SparseMatrixd = Eigen::SparseMatrix <double, Eigen::RowMajor>;
using SparseVectorxd = Eigen::SparseVector<double, Eigen::RowMajor>;
using Vectorxd = Eigen::VectorXd;

TEST ( bicgstabCUDA, bicgstabGPUCorrect )
{
    SparseMatrixd A;
    ASSERT_TRUE( Eigen::loadMarket<SparseMatrixd>( A, "Trefethen_150.mtx" ) );
    Vectorxd b ( A.rows() );
    b.setOnes();
    Eigen::BiCGSTAB<SparseMatrixd, Eigen::IncompleteCholesky<double>> solver;
    solver.compute( A );
    auto x = solver.solve( b );
    UNUSED( x );
    Vectorxd x_gpu ( A.rows() );
    double* A_values = A.valuePtr();
    int* A_rows_ptr = A.outerIndexPtr();
    int* A_col_ptr = A.innerIndexPtr();
    double* b_ptr = b.data();
    double* x_ptr = x_gpu.data();
    double tol = std::numeric_limits<double>::epsilon();
    const size_t size = A.rows();
    const size_t nnz = A.nonZeros();
    const int result = bicgstabGPU( A_values, A_col_ptr, A_rows_ptr,
                                    b_ptr, x_ptr, tol,
                                    size, nnz, 20 );
    Eigen::saveMarketVector( x_gpu, "Trefethen_gpu.mtx" );
    EXPECT_EQ( 0, result );
}

int main( int argc, char** argv )
{
    testing::InitGoogleTest( &argc, argv );
    return RUN_ALL_TESTS();
}
