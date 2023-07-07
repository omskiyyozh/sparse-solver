#pragma once
#include "Stable.hpp"
#include "spcsr_matrix.hpp"

namespace CudaCPP
{
using SpCSRMatrixPtr = std::unique_ptr<SpMatrixCSR>;

class PreconditionerBase
{
public:
    PreconditionerBase() = default;
    virtual bool init ( SpCSRMatrixPtr& matrix, cusparseHandle_t cusparse_handle ) = 0;
    virtual bool compute( SpCSRMatrixPtr& matrix, cusparseHandle_t cusparse_handle ) = 0;

    virtual ~PreconditionerBase() = default;

protected:
    virtual bool analyze( SpCSRMatrixPtr& matrix, cusparseHandle_t cusparse_handle ) = 0;
    virtual bool factorize( SpCSRMatrixPtr& matrix, cusparseHandle_t cusparse_handle ) = 0;
};
}
