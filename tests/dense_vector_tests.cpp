#include "dense_vector.hpp"

using Vectorxd = Eigen::VectorXd;
using DenseVectorD = CudaCPP::DenseVector;

TEST( denseVector, successfulCreation )
{
    std::vector<double> data {1.5, 2.1, 3.0, 4.8, 1.9};
    const size_t data_size = data.size();

    auto vector = DenseVectorD::create( data.data(), data_size, CUDA_R_64F );
    std::vector<double> host_data_from_vec ( data_size );

    ASSERT_EQ( data_size, vector->size() );
    ASSERT_EQ( cudaSuccess, cudaMemcpy( host_data_from_vec.data(), vector->data(), data_size * sizeof( double ),
                                        cudaMemcpyDeviceToHost ) );
    ASSERT_EQ ( data, host_data_from_vec );
}

TEST( denseVector, creationWithHostDataNullptr )
{
    const double* data_ptr = nullptr;
    auto vector = DenseVectorD::create( data_ptr, 12121, CUDA_R_64F );
    ASSERT_NE( nullptr, vector );
    ASSERT_NE( nullptr, vector->data() );
    ASSERT_NE( nullptr, vector->vector() );
}

TEST( denseVector, creationWithZeroSize )
{
    const double data[] = {1, 2, 3, 4, 5};
    auto vector = DenseVectorD::create( data, 0, CUDA_R_64F );
    ASSERT_EQ( nullptr, vector );
}

TEST ( denseVector, successfulCopyToHost )
{
    std::vector<double> data {1.5, 2.1, 3.0, 4.8, 1.9};
    const size_t data_size = data.size();

    auto vector = DenseVectorD::create( data.data(), data_size, CUDA_R_64F );
    std::vector<double> host_data_from_vec ( data_size );
    ASSERT_EQ( 0, vector->copyToHost( host_data_from_vec.data(), host_data_from_vec.size() ) );
    ASSERT_EQ( data, host_data_from_vec );
}
TEST ( denseVector, copyToHostWithInvalidHost )
{
    std::vector<double> data {1.5, 2.1, 3.0, 4.8, 1.9};
    auto vector = DenseVectorD::create( data.data(), data.size(), CUDA_R_64F );

    double smaller_vec[] {12.1, 12.1};
    double bigger_vec[] {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

    ASSERT_EQ( 1, vector->copyToHost( nullptr, 0 ) );
    ASSERT_EQ( 1, vector->copyToHost( smaller_vec, sizeof( smaller_vec ) / sizeof( double ) ) );
    ASSERT_EQ( 1, vector->copyToHost( bigger_vec, sizeof( smaller_vec ) / sizeof( double ) ) );

}

TEST( denseVector, successfulCopy )
{
    std::vector<double> data_1 {1.5, 2.1, 3.0, 4.8, 1.9};
    const size_t data_size_1 = data_1.size();

    auto vector_1 = DenseVectorD::create( data_1.data(), data_size_1, CUDA_R_64F );

    std::vector<double> data_2 {0.0, 175.3, 34 / 8, 12.5, 1.9};
    auto vector_2 = DenseVectorD::create( data_2.data(), data_2.size(), CUDA_R_64F );

    ASSERT_EQ( 0, vector_1->copy( vector_2 ) );

    std::vector<double> host_data_from_vec ( data_size_1 );

    ASSERT_EQ( 0, vector_1->copyToHost( host_data_from_vec.data(), data_size_1 ) );
    ASSERT_EQ ( data_2, host_data_from_vec );

}

TEST( denseVector, copyWithEmptyVector )
{
    std::vector<double> data_1 {1.5, 2.1, 3.0, 4.8, 1.9};

    auto vector_1 = DenseVectorD::create( data_1.data(), data_1.size(), CUDA_R_64F );

    auto vector_2 = std::make_unique<DenseVectorD>();
    std::unique_ptr<DenseVectorD> vector_3 ( new DenseVectorD() );
    std::unique_ptr<DenseVectorD> vector_4;

    ASSERT_EQ( 1, vector_1->copy( vector_2 ) );
    ASSERT_EQ( 1, vector_1->copy( vector_3 ) );
    ASSERT_EQ( 1, vector_1->copy( vector_4 ) );
}

TEST( denseVector, copyWithDifferentSizes )
{
    std::vector<double> data_1 {1.5, 2.1, 3.0, 4.8, 1.9};
    auto vector_1 = DenseVectorD::create( data_1.data(), data_1.size(), CUDA_R_64F );

    std::vector<double> data_2 {0.1, 0.0, 175.3, 34 / 8, 12.5, 1.9};
    auto vector_2 = DenseVectorD::create( data_2.data(), data_2.size(), CUDA_R_64F );

    ASSERT_EQ ( 1, vector_1->copy( vector_2 ) );
}

TEST( denseVector, normTests )
{
    std::vector<double> data {1.5, 2.1, 3.0, 4.8, 1.9};
    auto vector = DenseVectorD::create( data.data(), data.size(), CUDA_R_64F );

    double expected = std::sqrt( std::inner_product( data.begin(), data.end(), data.begin(), ( double )0 )  );

    double result = 0;

    ASSERT_EQ( 0, vector->norm( result ) );
    ASSERT_DOUBLE_EQ( result, expected );
    ASSERT_EQ( 1, DenseVectorD().norm( result ) );
}

TEST( denseVector, dotComputesCorrectly )
{
    std::vector<double> data {1.5, 2.1, 3.0, 4.8, 1.9};
    const size_t data_size = data.size();
    auto vector = DenseVectorD::create( data.data(), data_size, CUDA_R_64F );

    std::vector<double> data_2 {0.1, 0.0, 175.3, 12.5, 1.9};
    auto vector_2 = DenseVectorD::create( data_2.data(), data_size, CUDA_R_64F );

    double expected =  std::inner_product( data.begin(), data.end(), data_2.begin(), ( double )0 );
    double result = 0;

    ASSERT_EQ( 0, vector->dot( vector_2, result ) );
    ASSERT_EQ( expected, result );
}

TEST( denseVector, dotInvalidVector )
{
    std::vector<double> data_1 {1.5, 2.1, 3.0, 4.8, 1.9};
    auto vector_1 = DenseVectorD::create( data_1.data(), data_1.size(), CUDA_R_64F );

    std::vector<double> data_2 {0.1, 0.0, 175.3, 34 / 8, 12.5, 1.9};
    auto vector_2 = DenseVectorD::create( data_2.data(), data_2.size(), CUDA_R_64F );

    double result;
    ASSERT_EQ( 1, vector_1->dot( vector_2, result ) );
    ASSERT_EQ ( 1, vector_1->dot( std::make_unique<DenseVectorD>(), result ) );
}

TEST ( denseVector, axpyComputesCorrectly )
{
    std::vector<double> data {1.5, 2.1, 3.0, 4.8, 1.9};
    const size_t data_size = data.size();

    auto vector = DenseVectorD::create( data.data(), data_size, CUDA_R_64F );

    std::vector<double> data_2 {0.1, 0.0, 175.3, 12.5, 1.9};
    auto vector_2 = DenseVectorD::create( data_2.data(), data_size, CUDA_R_64F );

    const double alpha = 1;

    for ( auto i = data_2.begin(), j = data.begin(); i < data_2.end(); )
        *i++ += *j++;

    ASSERT_EQ( 0, vector->axpy( vector_2, alpha ) );

    std::vector<double> result ( data_size );

    ASSERT_EQ( 0, vector_2->copyToHost( result.data(), data_size ) );
    ASSERT_EQ( result, data_2 );
}

TEST( denseVector, axpyInvalidVector )
{
    std::vector<double> data_1 {1.5, 2.1, 3.0, 4.8, 1.9};
    auto vector_1 = DenseVectorD::create( data_1.data(), data_1.size(), CUDA_R_64F );

    std::vector<double> data_2 {0.1, 0.0, 175.3, 34 / 8, 12.5, 1.9};
    auto vector_2 = DenseVectorD::create( data_2.data(), data_2.size(), CUDA_R_64F );

    auto empty_vec = std::make_unique<DenseVectorD>();

    ASSERT_EQ( 1, vector_1->axpy( vector_2, ( double )1 ) );
    ASSERT_EQ ( 1, vector_1->axpy( empty_vec, ( double )1 ) );
}

TEST( denseVector, setZeroVector )
{
    auto vector = DenseVectorD::create( nullptr, 10, CUDA_R_64F );
    vector->setZero();
    std::vector<double> zero_vector ( 10, 0 );
    std::vector<double> zero_vector_res( 10 );
    ASSERT_EQ( 0, vector->copyToHost( zero_vector_res.data(), vector->size() ) );
    ASSERT_EQ( zero_vector, zero_vector_res );
}

TEST ( denseVector, correctScale )
{
    std::vector<double> data {1.5, 2.1, 3.0, 4.8, 1.9};
    auto vector = DenseVectorD::create( data.data(), data.size(), CUDA_R_64F );

    const double alpha = 2;
    ASSERT_EQ( 0, vector->scale( alpha ) );

    std::for_each( data.begin(), data.end(), [&]( auto & i )
    {
        return i *= alpha;
    } );

    std::vector<double> result ( vector->size() );
    ASSERT_EQ( 0, vector->copyToHost( result.data(), vector->size() ) );
    ASSERT_EQ( data, result );
}
