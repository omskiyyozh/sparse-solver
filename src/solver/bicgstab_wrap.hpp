#include "Stable.hpp"
#include "solver_base.hpp"

namespace CudaCPP
{

template <typename Preconditioner>
class BiCGSTAB : public SolverBase<Preconditioner>
{
public:
    using DenseVectorPtr = std::unique_ptr<DenseVector>;
    using SpCSRMatrixPtr = std::unique_ptr<SpMatrixCSR>;

    BiCGSTAB(): SolverBase<Preconditioner>() {};
    DenseVectorPtr solve ( const SpCSRMatrixPtr& matrix, const DenseVectorPtr& rhs ) override;
    ~BiCGSTAB() = default;
};

#define UNUSED(x) (void)x

#define BICGSTAB_NULLPTR_CHECK(status) \
    if ( status == nullptr ) \
    return {};

#define BICGSTAB_CHECK(status) \
    if ( status ) \
    return {};

#define BICGSTAB_CUDA_CHECK(status) \
    if ( status != cudaSuccess ) \
    return {};

#define BICGSTAB_CUSPARSE_CHECK(func)                                                   \
{                                                                              \
    cusparseStatus_t cusparse_status = (func);                                          \
    if (cusparse_status != CUSPARSE_STATUS_SUCCESS) {                                   \
    return {};                                                   \
    }                                                                          \
}

#define BICGSTAB_CUBLAS_CHECK(func)                                                   \
{                                                                              \
    cublasStatus_t cublas_status = (func);                                          \
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {                                   \
    return {};                                                   \
    }                                                                          \
}

template<typename Preconditioner>
std::unique_ptr<DenseVector> BiCGSTAB<Preconditioner>::solve( const SpCSRMatrixPtr& A, const DenseVectorPtr& rhs )
{
    cublasHandle_t cublas_handle;
    cusparseHandle_t cusparse_handle;

    BICGSTAB_CUBLAS_CHECK( cublasCreate( &cublas_handle ) );
    BICGSTAB_CUSPARSE_CHECK( cusparseCreate( &cusparse_handle ) );

    DenseVectorPtr     dev_X, dev_R, dev_R0, dev_P, dev_P_aux, dev_S,
                       dev_S_aux, dev_V, dev_T, dev_tmp;


    const size_t vector_size = rhs->size();

    dev_X = DenseVector::create( nullptr, vector_size, CUDA_R_64F );
    BICGSTAB_NULLPTR_CHECK( dev_X );

    dev_R = DenseVector::create( nullptr, vector_size, CUDA_R_64F );
    BICGSTAB_NULLPTR_CHECK( dev_R );

    dev_R0 = DenseVector::create( nullptr, vector_size, CUDA_R_64F );
    BICGSTAB_NULLPTR_CHECK( dev_R0 );

    dev_P = DenseVector::create( nullptr, vector_size, CUDA_R_64F );
    BICGSTAB_NULLPTR_CHECK( dev_P );

    dev_P_aux = DenseVector::create( nullptr, vector_size, CUDA_R_64F );
    BICGSTAB_NULLPTR_CHECK( dev_P_aux );

    dev_S = DenseVector::create( nullptr, vector_size, CUDA_R_64F );
    BICGSTAB_NULLPTR_CHECK( dev_S );

    dev_S_aux = DenseVector::create( nullptr, vector_size, CUDA_R_64F );
    BICGSTAB_NULLPTR_CHECK( dev_S_aux );

    dev_V = DenseVector::create( nullptr, vector_size, CUDA_R_64F );
    BICGSTAB_NULLPTR_CHECK( dev_V );

    dev_T = DenseVector::create( nullptr, vector_size, CUDA_R_64F );
    BICGSTAB_NULLPTR_CHECK( dev_T );

    dev_tmp = DenseVector::create( nullptr, vector_size, CUDA_R_64F );
    BICGSTAB_NULLPTR_CHECK( dev_tmp );

    SpCSRMatrixPtr M_lower, M_upper;

    SpMatrixAttributes attributes;
    attributes.index_base = CUSPARSE_INDEX_BASE_ZERO;
    attributes.values_type = CUDA_R_64F;
    attributes.row_offsets_type = CUSPARSE_INDEX_32I;
    attributes.col_ind_type = CUSPARSE_INDEX_32I;
    attributes.diag_type = CUSPARSE_DIAG_TYPE_UNIT;
    attributes.fill_mode = CUSPARSE_FILL_MODE_LOWER;

    const size_t nnz = A->nnz();

    double* d_M_values;
    BICGSTAB_CUDA_CHECK( cudaMalloc( ( void** ) &d_M_values,  nnz * sizeof( double ) ) );
    BICGSTAB_CUDA_CHECK( cudaMemcpy( d_M_values, A->values(), nnz * sizeof( double ),
                                     cudaMemcpyDeviceToDevice ) );

    M_lower = SpMatrixCSR::create( d_M_values, A->colInd(), A->rowPtr(), vector_size, nnz, attributes,
                                   DataLocation::DEVICE );

    BICGSTAB_NULLPTR_CHECK( M_lower );

    attributes.diag_type = CUSPARSE_DIAG_TYPE_NON_UNIT;
    attributes.fill_mode = CUSPARSE_FILL_MODE_UPPER;

    M_upper = SpMatrixCSR::create( d_M_values, A->colInd(), A->rowPtr(),
                                   vector_size, nnz, attributes, DataLocation::DEVICE );

    BICGSTAB_NULLPTR_CHECK( M_upper );

    BICGSTAB_CHECK( dev_X->setZero() );

    BICGSTAB_CHECK( compute( M_lower, cusparse_handle ) );

    cusparseSpMatDescr_t M_upper_desc = M_upper->matDesc();
    cusparseSpMatDescr_t M_lower_desc = M_lower->matDesc();

    cusparseDnVecDescr_t P_desc = dev_P->vector();
    cusparseDnVecDescr_t tmp_desc = dev_tmp->vector();
    cusparseDnVecDescr_t P_aux_desc = dev_P_aux->vector();
    cusparseDnVecDescr_t S_desc = dev_S->vector();
    cusparseDnVecDescr_t S_aux_desc = dev_S_aux->vector();

    double              coeff_tmp;
    size_t              buffer_size_L, buffer_size_U;
    void*               d_buffer_L, *d_buffer_U;
    cusparseSpSVDescr_t spsv_descr_L, spsv_descr_U;
    BICGSTAB_CUSPARSE_CHECK( cusparseSpSV_createDescr( &spsv_descr_L ) );
    BICGSTAB_CUSPARSE_CHECK( cusparseSpSV_bufferSize(
                                 cusparse_handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &coeff_tmp,
                                 M_lower_desc,
                                 P_desc,
                                 tmp_desc,
                                 CUDA_R_64F,
                                 CUSPARSE_SPSV_ALG_DEFAULT,
                                 spsv_descr_L,
                                 &buffer_size_L
                             ) );
    BICGSTAB_CUDA_CHECK( cudaMalloc( &d_buffer_L, buffer_size_L ) );

    BICGSTAB_CUSPARSE_CHECK( cusparseSpSV_analysis(
                                 cusparse_handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &coeff_tmp,
                                 M_lower_desc,
                                 P_desc,
                                 tmp_desc,
                                 CUDA_R_64F,
                                 CUSPARSE_SPSV_ALG_DEFAULT,
                                 spsv_descr_L,
                                 d_buffer_L
                             ) );

    BICGSTAB_CUSPARSE_CHECK( cusparseSpSV_createDescr( &spsv_descr_U ) )
    BICGSTAB_CUSPARSE_CHECK( cusparseSpSV_bufferSize(
                                 cusparse_handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &coeff_tmp,
                                 M_upper_desc,
                                 tmp_desc,
                                 P_aux_desc,
                                 CUDA_R_64F,
                                 CUSPARSE_SPSV_ALG_DEFAULT,
                                 spsv_descr_U,
                                 &buffer_size_U
                             ) );
    BICGSTAB_CUDA_CHECK( cudaMalloc( &d_buffer_U, buffer_size_U ) )
    BICGSTAB_CUSPARSE_CHECK( cusparseSpSV_analysis(
                                 cusparse_handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &coeff_tmp,
                                 M_upper_desc,
                                 tmp_desc,
                                 P_aux_desc,
                                 CUDA_R_64F,
                                 CUSPARSE_SPSV_ALG_DEFAULT,
                                 spsv_descr_U,
                                 d_buffer_U
                             ) );

    BICGSTAB_CHECK( dev_R0->copy( rhs ) );

    const double zero      = 0.0;
    const double one       = 1.0;
    const double minus_one = -1.0;

    BICGSTAB_CHECK( A->spMV( CUSPARSE_OPERATION_NON_TRANSPOSE,
                             minus_one,
                             one,
                             dev_X,
                             dev_R0,
                             CUSPARSE_SPMV_ALG_DEFAULT,
                             CUDA_R_64F,
                             cusparse_handle ) );

    double alpha = 1, delta, delta_prev, omega = 1;

    BICGSTAB_CHECK( dev_R0->dot( dev_R0, delta, cublas_handle ) );

    delta_prev = delta;
    BICGSTAB_CHECK( dev_R->copy( dev_R0 ) );

    double nrm_R;
    BICGSTAB_CHECK( dev_R0->norm( nrm_R, cublas_handle ) );
    double threshold = this->m_tolerance * nrm_R;

    for ( unsigned int i = 1; i <= this->m_max_iterations; i++ )
    {
        BICGSTAB_CHECK( dev_P->copy( dev_R ) );

        if ( i > 1 )
        {
            BICGSTAB_CHECK( dev_R0->dot( dev_R, delta, cublas_handle ) );
            double beta = ( delta / delta_prev ) * ( alpha / omega );
            delta_prev  = delta;
            double minus_omega = -omega;
            BICGSTAB_CHECK( dev_V->axpy( dev_P, minus_omega, cublas_handle ) );
            BICGSTAB_CHECK( dev_P->scale( beta, cublas_handle ) );
            BICGSTAB_CHECK( dev_R->axpy( dev_P, one, cublas_handle ) );
        }

        BICGSTAB_CHECK( dev_tmp->setZero() );
        BICGSTAB_CHECK( dev_P_aux->setZero() );

        BICGSTAB_CUSPARSE_CHECK( cusparseSpSV_solve( cusparse_handle,
                                                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                     &one, M_lower_desc,
                                                     P_desc,
                                                     tmp_desc,
                                                     CUDA_R_64F,
                                                     CUSPARSE_SPSV_ALG_DEFAULT,
                                                     spsv_descr_L
                                                   ) );

        BICGSTAB_CUSPARSE_CHECK( cusparseSpSV_solve( cusparse_handle,
                                                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                     &one,
                                                     M_upper_desc,
                                                     tmp_desc,
                                                     P_aux_desc,
                                                     CUDA_R_64F,
                                                     CUSPARSE_SPSV_ALG_DEFAULT,
                                                     spsv_descr_U
                                                   ) );

        BICGSTAB_CHECK( A->spMV( CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 one,
                                 zero,
                                 dev_P_aux,
                                 dev_V,
                                 CUSPARSE_SPMV_ALG_DEFAULT,
                                 CUDA_R_64F,
                                 cusparse_handle ) );
        double denominator;
        BICGSTAB_CHECK( dev_R0->dot( dev_V, denominator, cublas_handle ) );
        alpha = delta / denominator;
        BICGSTAB_CHECK( dev_P_aux->axpy( dev_X, alpha, cublas_handle ) );

        BICGSTAB_CHECK( dev_S->copy( dev_R ) );
        double minus_alpha = -alpha;
        BICGSTAB_CHECK( dev_V->axpy( dev_S, minus_alpha, cublas_handle ) );
        double nrm_S;
        BICGSTAB_CHECK( dev_S->norm( nrm_S, cublas_handle ) );

        if ( nrm_S < threshold )
            break;

        BICGSTAB_CHECK( dev_tmp->setZero() );
        BICGSTAB_CHECK( dev_S_aux->setZero() );

        BICGSTAB_CUSPARSE_CHECK( cusparseSpSV_solve(
                                     cusparse_handle,
                                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     &one,
                                     M_lower_desc,
                                     S_desc,
                                     tmp_desc,
                                     CUDA_R_64F,
                                     CUSPARSE_SPSV_ALG_DEFAULT,
                                     spsv_descr_L ) );

        BICGSTAB_CUSPARSE_CHECK( cusparseSpSV_solve(
                                     cusparse_handle,
                                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     &one,
                                     M_upper_desc,
                                     tmp_desc,
                                     S_aux_desc,
                                     CUDA_R_64F,
                                     CUSPARSE_SPSV_ALG_DEFAULT,
                                     spsv_descr_U ) );

        BICGSTAB_CHECK( A->spMV( CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 one,
                                 zero,
                                 dev_S_aux,
                                 dev_T,
                                 CUSPARSE_SPMV_ALG_DEFAULT,
                                 CUDA_R_64F,
                                 cusparse_handle ) );

        double omega_num, omega_den;
        BICGSTAB_CHECK( dev_T->dot( dev_S, omega_num, cublas_handle ) );
        BICGSTAB_CHECK( dev_T->dot( dev_T, omega_den, cublas_handle ) );
        omega = omega_num / omega_den;

        BICGSTAB_CHECK( dev_S_aux->axpy( dev_X, omega, cublas_handle ) );
        BICGSTAB_CHECK( dev_R->copy( dev_S ) );
        double minus_omega = -omega;
        BICGSTAB_CHECK( dev_T->axpy( dev_R, minus_omega, cublas_handle ) );
        BICGSTAB_CHECK( dev_R->norm( nrm_R, cublas_handle ) );

        if ( nrm_R < threshold )
            break;

    }

    cusparseSpSV_destroyDescr( spsv_descr_L );
    cusparseSpSV_destroyDescr( spsv_descr_U );
    cudaFree( d_buffer_L );
    cudaFree( d_buffer_U );
    cublasDestroy( cublas_handle );
    cusparseDestroy( cusparse_handle );

    return dev_X;
}

}
