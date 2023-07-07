#pragma once

#include "Stable.hpp"
#include "dense_vector.hpp"

namespace CudaCPP
{

enum class DataLocation : char
{
    HOST,
    DEVICE
};

struct SpMatrixAttributes
{
    cusparseIndexBase_t index_base;
    cusparseFillMode_t fill_mode;
    cusparseDiagType_t diag_type;
    cudaDataType values_type;
    cusparseIndexType_t row_offsets_type;
    cusparseIndexType_t col_ind_type;
};

class SpMatrixCSR
{
public:
    using SpCSRPtr = std::unique_ptr<SpMatrixCSR>;

    SpMatrixCSR();
    SpMatrixCSR ( const SpMatrixCSR& ) = delete;
    SpMatrixCSR ( double* dev_values, int* dev_col_ind, int* dev_row_ptr, const size_t dim, const size_t nnz,
                  const cusparseSpMatDescr_t dev_mat_desc );
    ~SpMatrixCSR();

    static SpCSRPtr create ( double* values, int* col_ind, int* row_ptr,
                             const size_t dim, const size_t nnz, SpMatrixAttributes mat_attributes,
                             const DataLocation data_location = DataLocation::HOST );

    bool copy ( const SpCSRPtr& matrix );

    bool spMV ( cusparseOperation_t op_A, const double alpha, const double beta, std::unique_ptr<DenseVector>& vec_X,
                std::unique_ptr<DenseVector>& vec_Y, cusparseSpMVAlg_t alg_type, cudaDataType compute_type,
                cusparseHandle_t cusparse_handle = nullptr );

    inline double* values() const
    {
        return m_values;
    }

    inline int* colInd() const
    {
        return m_col_ind;
    }

    inline int* rowPtr() const
    {
        return m_row_ptr;
    }

    inline size_t dim() const
    {
        return m_dim;
    }

    inline size_t nnz() const
    {
        return m_nnz;
    }

    inline cusparseSpMatDescr_t matDesc() const
    {
        return m_mat_desc;
    }

private:
    double* m_values;
    int* m_col_ind;
    int* m_row_ptr;
    size_t m_dim;
    size_t m_nnz;
    cusparseSpMatDescr_t m_mat_desc;
};
}
