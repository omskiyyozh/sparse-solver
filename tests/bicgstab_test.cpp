#include "bicgstab_wrap.hpp"
#include "incompleteLU.hpp"

using SparseMatrixD = Eigen::SparseMatrix <double, Eigen::RowMajor>;
using SparseVectorxd = Eigen::SparseVector<double, Eigen::RowMajor>;
using Vectorxd = Eigen::VectorXd;
using SpMatrixCSR = CudaCPP::SpMatrixCSR;
using SpMatrixAttributes = CudaCPP::SpMatrixAttributes;
using DenseVectorD = CudaCPP::DenseVector;
using IncompleteLU = CudaCPP::IncompleteLU;
using BiCGSTAB = CudaCPP::BiCGSTAB<IncompleteLU>;

TEST( bicgstab, computesCorrectly )
{
    SparseMatrixD matrix;
    ASSERT_TRUE( Eigen::loadMarket<SparseMatrixD>( matrix, "Trefethen_150.mtx" ) );

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

    Vectorxd b ( dim );
    b.setOnes();

    auto vector = DenseVectorD::create( b.data(), dim, CUDA_R_64F );

    BiCGSTAB solver;
    auto x = solver.solve( matrix_test, vector );
    ASSERT_NE( nullptr, x );

    Eigen::BiCGSTAB<SparseMatrixD, Eigen::IncompleteLUT<double>> solver_eigen;
    solver_eigen.compute( matrix );
    Eigen::VectorXd x_eigen = solver_eigen.solve( b );

    ASSERT_EQ( solver_eigen.info(), Eigen::ComputationInfo::Success );

    std::vector<double> host_x ( dim );
    x->copyToHost( host_x.data(), dim );

    double* j = x_eigen.data();

    double x_eigen_norm  = x_eigen.norm();

    for ( auto i : host_x )
        *j++ -= i;

    double abs_error = x_eigen.norm();
    std::cerr << "Relative error: " << abs_error / x_eigen_norm << std::endl;

}
