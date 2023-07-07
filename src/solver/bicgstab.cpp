#include "bicgstab.hpp"

#define CUDA_CHECK(status) \
    if ( status != cudaSuccess ) \
        return EXIT_FAILURE;

#define CUSPARSE_CHECK(func)                                                   \
{                                                                              \
    cusparseStatus_t cusparse_status = (func);                                          \
    if (cusparse_status != CUSPARSE_STATUS_SUCCESS) {                                   \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CUBLAS_CHECK(func)                                                   \
{                                                                              \
    cublasStatus_t cublas_status = (func);                                          \
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {                                   \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define UNUSED(x) ( void )( x )

using data_type = double;

int bicgstabGPU ( const double* A, const int* cols_ptr, const int* rows_ptr,
                  const double* b, double* x, const double tol, const size_t size,
                  const size_t nnz, const int max_iterations )
{
    size_t     m           = size;
    size_t     num_offsets = m + 1;


    int*    d_A_rows, *d_A_columns;
    double* d_A_values, *d_M_values;
    double*     d_B, *d_X, *d_R, *d_R0, *d_P, *d_P_aux, *d_S, *d_S_aux, *d_V, *d_T, *d_tmp;

    CUDA_CHECK( cudaMalloc( ( void** ) &d_A_rows,    num_offsets * sizeof( int ) ) );
    CUDA_CHECK( cudaMalloc( ( void** ) &d_A_columns, nnz * sizeof( int ) ) );
    CUDA_CHECK( cudaMalloc( ( void** ) &d_A_values,  nnz * sizeof( double ) ) );
    CUDA_CHECK( cudaMalloc( ( void** ) &d_M_values,  nnz * sizeof( double ) ) );

    CUDA_CHECK( cudaMalloc( ( void** ) &d_B,     m * sizeof( double ) ) );
    CUDA_CHECK( cudaMalloc( ( void** ) &d_X,     m * sizeof( double ) ) );
    CUDA_CHECK( cudaMalloc( ( void** ) &d_R,     m * sizeof( double ) ) );
    CUDA_CHECK( cudaMalloc( ( void** ) &d_R0,    m * sizeof( double ) ) );
    CUDA_CHECK( cudaMalloc( ( void** ) &d_P,     m * sizeof( double ) ) );
    CUDA_CHECK( cudaMalloc( ( void** ) &d_P_aux, m * sizeof( double ) ) );
    CUDA_CHECK( cudaMalloc( ( void** ) &d_S,     m * sizeof( double ) ) );
    CUDA_CHECK( cudaMalloc( ( void** ) &d_S_aux, m * sizeof( double ) ) );
    CUDA_CHECK( cudaMalloc( ( void** ) &d_V,     m * sizeof( double ) ) );
    CUDA_CHECK( cudaMalloc( ( void** ) &d_T,     m * sizeof( double ) ) );
    CUDA_CHECK( cudaMalloc( ( void** ) &d_tmp,   m * sizeof( double ) ) );

    CUDA_CHECK( cudaMemcpy( d_A_rows, rows_ptr, num_offsets * sizeof( int ),
                            cudaMemcpyHostToDevice ) );
    CUDA_CHECK( cudaMemcpy( d_B, b, m * sizeof( double ),
                            cudaMemcpyHostToDevice ) );
    CUDA_CHECK( cudaMemcpy( d_A_columns, cols_ptr, nnz *  sizeof( int ),
                            cudaMemcpyHostToDevice ) );
    CUDA_CHECK( cudaMemcpy( d_A_values, A, nnz * sizeof( double ),
                            cudaMemcpyHostToDevice ) );
    CUDA_CHECK( cudaMemcpy( d_M_values, A, nnz * sizeof( double ),
                            cudaMemcpyHostToDevice ) );
    CUDA_CHECK( cudaMemcpy( d_X, x, m * sizeof( double ),
                            cudaMemcpyHostToDevice ) );

    cublasHandle_t   cublasHandle   = NULL;
    cusparseHandle_t cusparseHandle = NULL;
    CUBLAS_CHECK( cublasCreate( &cublasHandle ) )
    CUSPARSE_CHECK( cusparseCreate( &cusparseHandle ) )

    cusparseDnVecDescr_t d_B_d, d_X_d, d_R_d, d_R0_d, d_P_d, d_P_aux_d, d_S_d,
                         d_S_aux_d, d_V_d, d_T_d, d_tmp_d;

    CUSPARSE_CHECK( cusparseCreateDnVec( &d_B_d,     m, d_B, CUDA_R_64F ) );
    CUSPARSE_CHECK( cusparseCreateDnVec( &d_X_d,     m, d_X, CUDA_R_64F ) );
    CUSPARSE_CHECK( cusparseCreateDnVec( &d_R_d,     m, d_R, CUDA_R_64F ) );
    CUSPARSE_CHECK( cusparseCreateDnVec( &d_R0_d,    m, d_R0, CUDA_R_64F ) );
    CUSPARSE_CHECK( cusparseCreateDnVec( &d_P_d,     m, d_P, CUDA_R_64F ) );
    CUSPARSE_CHECK( cusparseCreateDnVec( &d_P_aux_d, m, d_P_aux, CUDA_R_64F ) );
    CUSPARSE_CHECK( cusparseCreateDnVec( &d_S_d,     m, d_S, CUDA_R_64F ) );
    CUSPARSE_CHECK( cusparseCreateDnVec( &d_S_aux_d, m, d_S_aux, CUDA_R_64F ) );
    CUSPARSE_CHECK( cusparseCreateDnVec( &d_V_d,   m, d_V,   CUDA_R_64F ) );
    CUSPARSE_CHECK( cusparseCreateDnVec( &d_T_d,   m, d_T,   CUDA_R_64F ) );
    CUSPARSE_CHECK( cusparseCreateDnVec( &d_tmp_d, m, d_tmp, CUDA_R_64F ) );

    cusparseIndexBase_t  baseIdx = CUSPARSE_INDEX_BASE_ZERO;

    cusparseSpMatDescr_t mat_A, mat_M_lower, mat_M_upper;
    cusparseMatDescr_t   mat_LU;
    int*                 d_M_rows      = d_A_rows;
    int*                 d_M_columns   = d_A_columns;
    cusparseFillMode_t   fill_lower    = CUSPARSE_FILL_MODE_LOWER;
    cusparseDiagType_t   diag_unit     = CUSPARSE_DIAG_TYPE_UNIT;
    cusparseFillMode_t   fill_upper    = CUSPARSE_FILL_MODE_UPPER;
    cusparseDiagType_t   diag_non_unit = CUSPARSE_DIAG_TYPE_NON_UNIT;

    CUSPARSE_CHECK( cusparseCreateCsr( &mat_A, m, m, nnz, d_A_rows,
                                       d_A_columns, d_A_values,
                                       CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                       baseIdx, CUDA_R_64F ) );
    // M_lower
    CUSPARSE_CHECK( cusparseCreateCsr( &mat_M_lower, m, m, nnz, d_M_rows,
                                       d_M_columns, d_M_values,
                                       CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                       baseIdx, CUDA_R_64F ) );
    CUSPARSE_CHECK( cusparseSpMatSetAttribute( mat_M_lower,
                                               CUSPARSE_SPMAT_FILL_MODE,
                                               &fill_lower, sizeof( fill_lower ) ) );
    CUSPARSE_CHECK( cusparseSpMatSetAttribute( mat_M_lower,
                                               CUSPARSE_SPMAT_DIAG_TYPE,
                                               &diag_unit, sizeof( diag_unit ) ) );
    // M_upper
    CUSPARSE_CHECK( cusparseCreateCsr( &mat_M_upper, m, m, nnz, d_M_rows,
                                       d_M_columns, d_M_values,
                                       CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                       baseIdx, CUDA_R_64F ) );
    CUSPARSE_CHECK( cusparseSpMatSetAttribute( mat_M_upper,
                                               CUSPARSE_SPMAT_FILL_MODE,
                                               &fill_upper, sizeof( fill_upper ) ) );
    CUSPARSE_CHECK( cusparseSpMatSetAttribute( mat_M_upper,
                                               CUSPARSE_SPMAT_DIAG_TYPE,
                                               &diag_non_unit,
                                               sizeof( diag_non_unit ) ) );

    // X0 = 0
    CUDA_CHECK( cudaMemset( d_X, 0x0, m * sizeof( double ) ) );

    // Incomplete-LU
    csrilu02Info_t infoM        = NULL;
    int            buffer_size_LU = 0;
    void*          d_buffer_LU;
    CUSPARSE_CHECK( cusparseCreateMatDescr( &mat_LU ) );
    CUSPARSE_CHECK( cusparseSetMatType( mat_LU, CUSPARSE_MATRIX_TYPE_GENERAL ) );
    CUSPARSE_CHECK( cusparseSetMatIndexBase( mat_LU, baseIdx ) );
    CUSPARSE_CHECK( cusparseCreateCsrilu02Info( &infoM ) );

    CUSPARSE_CHECK( cusparseDcsrilu02_bufferSize(
                        cusparseHandle,
                        static_cast<int>( m ),
                        static_cast<int>( nnz ),
                        mat_LU,
                        d_M_values,
                        d_M_rows,
                        d_M_columns,
                        infoM,
                        &buffer_size_LU
                    ) );
    CUDA_CHECK( cudaMalloc( &d_buffer_LU, buffer_size_LU ) );
    CUSPARSE_CHECK( cusparseDcsrilu02_analysis(
                        cusparseHandle,
                        static_cast<int>( m ),
                        static_cast<int>( nnz ),
                        mat_LU,
                        d_M_values,
                        d_M_rows,
                        d_M_columns,
                        infoM,
                        CUSPARSE_SOLVE_POLICY_USE_LEVEL,
                        d_buffer_LU
                    ) );
    int structural_zero;
    CUSPARSE_CHECK( cusparseXcsrilu02_zeroPivot( cusparseHandle,
                                                 infoM,
                                                 &structural_zero
                                               ) );
    // M = L * U
    CUSPARSE_CHECK( cusparseDcsrilu02(
                        cusparseHandle,
                        static_cast<int>( m ),
                        static_cast<int>( nnz ),
                        mat_LU,
                        d_M_values,
                        d_M_rows,
                        d_M_columns,
                        infoM,
                        CUSPARSE_SOLVE_POLICY_USE_LEVEL,
                        d_buffer_LU
                    ) );

    CUSPARSE_CHECK( cusparseDestroyCsrilu02Info( infoM ) );
    CUSPARSE_CHECK( cusparseDestroyMatDescr( mat_LU ) );
    CUDA_CHECK( cudaFree( d_buffer_LU ) );

    const double zero      = 0.0;
    const double one       = 1.0;
    const double minus_one = -1.0;

    double              coeff_tmp;
    size_t              buffer_size_L, buffer_size_U;
    void*               d_buffer_L, *d_buffer_U;
    cusparseSpSVDescr_t spsv_descr_L, spsv_descr_U;
    CUSPARSE_CHECK( cusparseSpSV_createDescr( &spsv_descr_L ) );
    CUSPARSE_CHECK( cusparseSpSV_bufferSize(
                        cusparseHandle,
                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &coeff_tmp,
                        mat_M_lower,
                        d_P_d,
                        d_tmp_d,
                        CUDA_R_64F,
                        CUSPARSE_SPSV_ALG_DEFAULT,
                        spsv_descr_L,
                        &buffer_size_L
                    ) );
    CUDA_CHECK( cudaMalloc( &d_buffer_L, buffer_size_L ) );
    CUSPARSE_CHECK( cusparseSpSV_analysis(
                        cusparseHandle,
                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &coeff_tmp,
                        mat_M_lower,
                        d_P_d,
                        d_tmp_d,
                        CUDA_R_64F,
                        CUSPARSE_SPSV_ALG_DEFAULT,
                        spsv_descr_L,
                        d_buffer_L
                    ) );

    // UPPER buffersize
    CUSPARSE_CHECK( cusparseSpSV_createDescr( &spsv_descr_U ) )
    CUSPARSE_CHECK( cusparseSpSV_bufferSize(
                        cusparseHandle,
                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &coeff_tmp,
                        mat_M_upper,
                        d_tmp_d,
                        d_P_aux_d,
                        CUDA_R_64F,
                        CUSPARSE_SPSV_ALG_DEFAULT,
                        spsv_descr_U,
                        &buffer_size_U
                    ) );
    CUDA_CHECK( cudaMalloc( &d_buffer_U, buffer_size_U ) )
    CUSPARSE_CHECK( cusparseSpSV_analysis(
                        cusparseHandle,
                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &coeff_tmp,
                        mat_M_upper,
                        d_tmp_d,
                        d_P_aux_d,
                        CUDA_R_64F,
                        CUSPARSE_SPSV_ALG_DEFAULT,
                        spsv_descr_U,
                        d_buffer_U
                    ) );

    // R0 = b - A * X0

    size_t buffer_size_MV;
    void*  d_buffer_MV;
    CUSPARSE_CHECK( cusparseSpMV_bufferSize(
                        cusparseHandle,
                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &minus_one,
                        mat_A,
                        d_X_d,
                        &one,
                        d_R0_d,
                        CUDA_R_64F,
                        CUSPARSE_SPMV_ALG_DEFAULT,
                        &buffer_size_MV
                    ) );
    CUDA_CHECK( cudaMalloc( &d_buffer_MV, buffer_size_MV ) );

    CUDA_CHECK( cudaMemcpy( d_R0, d_B, m * sizeof( double ), cudaMemcpyDeviceToDevice ) );
    // R = -A * X0 + R
    CUSPARSE_CHECK( cusparseSpMV( cusparseHandle,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  &minus_one,
                                  mat_A,
                                  d_X_d,
                                  &one,
                                  d_R0_d,
                                  CUDA_R_64F,
                                  CUSPARSE_SPMV_ALG_DEFAULT,
                                  d_buffer_MV
                                ) );

    double alpha = 1, delta, delta_prev, omega = 1;

    CUBLAS_CHECK( cublasDdot( cublasHandle,
                              static_cast<int>( m ),
                              d_R0,
                              1,
                              d_R0,
                              1,
                              &delta
                            ) );

    delta_prev = delta;

    CUDA_CHECK( cudaMemcpy( d_R,
                            d_R0,
                            m * sizeof( double ),
                            cudaMemcpyDeviceToDevice ) );
    double nrm_R;
    CUBLAS_CHECK( cublasDnrm2( cublasHandle, static_cast<int>( m ), d_R0, 1, &nrm_R ) );

    double threshold = tol * nrm_R;

    for ( int i = 1; i <= max_iterations; i++ )
    {

        CUDA_CHECK( cudaMemcpy( d_P, d_R, m * sizeof( double ),
                                cudaMemcpyDeviceToDevice ) );

        if ( i > 1 )
        {
            //    beta = (delta_i / delta_i-1) * (alpha / omega_i-1)
            //    delta_i = (R'_0, R_i-1)
            CUBLAS_CHECK( cublasDdot( cublasHandle,
                                      static_cast<int>( m ),
                                      d_R0,
                                      1,
                                      d_R,
                                      1,
                                      &delta
                                    ) );
            // beta = (delta_i / delta_i-1) * (alpha / omega_i-1);
            double beta = ( delta / delta_prev ) * ( alpha / omega );
            delta_prev  = delta;

            // P = R + beta * (P - omega * V)
            // P = -omega * V + P
            double minus_omega = -omega;
            CUBLAS_CHECK( cublasDaxpy( cublasHandle,
                                       static_cast<int>( m ),
                                       &minus_omega,
                                       d_V,
                                       1,
                                       d_P,
                                       1
                                     ) );
            // P = beta * P
            CUBLAS_CHECK( cublasDscal( cublasHandle,
                                       static_cast<int>( m ),
                                       &beta,
                                       d_P,
                                       1
                                     ) );
            // P = R + P
            CUBLAS_CHECK( cublasDaxpy( cublasHandle,
                                       static_cast<int>( m ),
                                       &one,
                                       d_R,
                                       1,
                                       d_P,
                                       1
                                     ) );
        }

        // P_aux = M_U^-1 M_L^-1 P_i
        // M_L^-1 P_i => tmp
        CUDA_CHECK( cudaMemset( d_tmp,   0x0, m * sizeof( double ) ) )
        CUDA_CHECK( cudaMemset( d_P_aux, 0x0, m * sizeof( double ) ) )

        CUSPARSE_CHECK( cusparseSpSV_solve( cusparseHandle,
                                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &one, mat_M_lower,
                                            d_P_d,
                                            d_tmp_d,
                                            CUDA_R_64F,
                                            CUSPARSE_SPSV_ALG_DEFAULT,
                                            spsv_descr_L
                                          ) );

        // M_U^-1 tmp => P_aux
        CUSPARSE_CHECK( cusparseSpSV_solve( cusparseHandle,
                                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &one,
                                            mat_M_upper,
                                            d_tmp_d,
                                            d_P_aux_d,
                                            CUDA_R_64F,
                                            CUSPARSE_SPSV_ALG_DEFAULT,
                                            spsv_descr_U
                                          ) );

        // alpha = (R'0, R_i-1) / (R'0, A * P_aux)
        // V = A * P_aux
        CUSPARSE_CHECK( cusparseSpMV( cusparseHandle,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                                      &one,
                                      mat_A,
                                      d_P_aux_d,
                                      &zero,
                                      d_V_d,
                                      CUDA_R_64F,
                                      CUSPARSE_SPMV_ALG_DEFAULT,
                                      d_buffer_MV
                                    ) );
        // denominator = R'0 * V
        double denominator;
        CUBLAS_CHECK( cublasDdot( cublasHandle, static_cast<int>( m ), d_R0, 1, d_V, 1,
                                  &denominator ) );
        alpha = delta / denominator;

        // X_i = X_i-1 + alpha * P_aux
        CUBLAS_CHECK( cublasDaxpy( cublasHandle, ( int )m, &alpha, d_P_aux, 1,
                                   d_X, 1 ) )

        // S = R_i-1 - alpha * (A * P_aux)
        // S = R_i-1
        CUDA_CHECK( cudaMemcpy( d_S, d_R, m * sizeof( double ),
                                cudaMemcpyDeviceToDevice ) )
        // S = -alpha * V + R_i-1
        double minus_alpha = -alpha;
        CUBLAS_CHECK( cublasDaxpy( cublasHandle, ( int )m, &minus_alpha, d_V, 1,
                                   d_S, 1 ) )
        // check ||S|| < threshold
        double nrm_S;
        CUBLAS_CHECK( cublasDnrm2( cublasHandle, ( int )m, d_S, 1, &nrm_S ) )

        if ( nrm_S < threshold )
            break;

        // S_aux = M_U^-1 M_L^-1 S
        // M_L^-1 S => tmp
        cudaMemset( d_tmp, 0x0, m * sizeof( double ) );
        cudaMemset( d_S_aux, 0x0, m * sizeof( double ) );
        CUSPARSE_CHECK( cusparseSpSV_solve( cusparseHandle,
                                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &one, mat_M_lower, d_S_d, d_tmp_d,
                                            CUDA_R_64F,
                                            CUSPARSE_SPSV_ALG_DEFAULT,
                                            spsv_descr_L ) )
        // M_U^-1 tmp => S_aux
        CUSPARSE_CHECK( cusparseSpSV_solve( cusparseHandle,
                                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &one, mat_M_upper, d_tmp_d,
                                            d_S_aux_d, CUDA_R_64F,
                                            CUSPARSE_SPSV_ALG_DEFAULT,
                                            spsv_descr_U ) )

        // omega = (A * S_aux, s) / (A * S_aux, A * S_aux)
        // T = A * S_aux
        CUSPARSE_CHECK( cusparseSpMV( cusparseHandle,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                                      &one,
                                      mat_A,
                                      d_S_aux_d,
                                      &zero,
                                      d_T_d,
                                      CUDA_R_64F,
                                      CUSPARSE_SPMV_ALG_DEFAULT,
                                      d_buffer_MV
                                    ) );
        // omega_num = (A * S_aux, s)
        double omega_num, omega_den;
        CUBLAS_CHECK( cublasDdot( cublasHandle, ( int )m, d_T, 1, d_S, 1,
                                  &omega_num ) )
        // omega_den = (A * S_aux, A * S_aux)
        CUBLAS_CHECK( cublasDdot( cublasHandle, ( int )m, d_T, 1, d_T, 1,
                                  &omega_den ) )
        // omega = omega_num / omega_den
        omega = omega_num / omega_den;

        // omega = X_i = X_i-1 + alpha * P_aux + omega * S_aux
        //        X_i = omega * S_aux + X_i
        CUBLAS_CHECK( cublasDaxpy( cublasHandle, ( int )m, &omega, d_S_aux, 1,
                                   d_X, 1 ) )
        // R_i+1 = S - omega * (A * S_aux)
        CUDA_CHECK( cudaMemcpy( d_R, d_S, m * sizeof( double ),
                                cudaMemcpyDeviceToDevice ) )
        // R_i+1 = -omega * T + R
        double minus_omega = -omega;
        CUBLAS_CHECK( cublasDaxpy( cublasHandle, ( int )m, &minus_omega, d_T, 1,
                                   d_R, 1 ) )

        CUBLAS_CHECK( cublasDnrm2( cublasHandle, ( int )m, d_R, 1, &nrm_R ) )

        if ( nrm_R < threshold )
            break;
    }

    CUDA_CHECK( cudaMemcpy( x, d_X, m * sizeof( double ), cudaMemcpyDeviceToHost ) );

    CUSPARSE_CHECK( cusparseSpSV_destroyDescr( spsv_descr_L ) );
    CUSPARSE_CHECK( cusparseSpSV_destroyDescr( spsv_descr_U ) );
    CUDA_CHECK( cudaFree( d_buffer_L ) );
    CUDA_CHECK( cudaFree( d_buffer_U ) );

    CUSPARSE_CHECK ( cusparseDestroySpMat( mat_A ) );
    CUSPARSE_CHECK ( cusparseDestroySpMat( mat_M_lower ) );
    CUSPARSE_CHECK ( cusparseDestroySpMat( mat_M_upper ) );
    CUSPARSE_CHECK( cusparseDestroyDnVec( d_B_d ) );
    CUSPARSE_CHECK( cusparseDestroyDnVec( d_X_d ) );
    CUSPARSE_CHECK( cusparseDestroyDnVec( d_R_d ) );
    CUSPARSE_CHECK( cusparseDestroyDnVec( d_R0_d ) );
    CUSPARSE_CHECK( cusparseDestroyDnVec( d_P_d ) );
    CUSPARSE_CHECK( cusparseDestroyDnVec( d_P_aux_d ) );
    CUSPARSE_CHECK( cusparseDestroyDnVec( d_S_d ) );
    CUSPARSE_CHECK( cusparseDestroyDnVec( d_S_aux_d ) );
    CUSPARSE_CHECK( cusparseDestroyDnVec( d_V_d ) );
    CUSPARSE_CHECK( cusparseDestroyDnVec( d_T_d ) );
    CUSPARSE_CHECK( cusparseDestroyDnVec( d_tmp_d ) );

    CUDA_CHECK( cudaFree( d_B ) );
    CUDA_CHECK( cudaFree( d_X ) );
    CUDA_CHECK( cudaFree( d_R ) );
    CUDA_CHECK( cudaFree( d_R0 ) );
    CUDA_CHECK( cudaFree( d_P ) );
    CUDA_CHECK( cudaFree( d_P_aux ) );
    CUDA_CHECK( cudaFree( d_S ) );
    CUDA_CHECK( cudaFree( d_S_aux ) );
    CUDA_CHECK( cudaFree( d_V ) );
    CUDA_CHECK( cudaFree( d_T ) );
    CUDA_CHECK( cudaFree( d_tmp ) );

    return 0;
}
