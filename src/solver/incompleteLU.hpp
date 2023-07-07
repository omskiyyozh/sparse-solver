#include "Stable.hpp"
#include "preconditioner_base.hpp"

namespace CudaCPP
{
class IncompleteLU: public PreconditionerBase
{
public:
    IncompleteLU();
    ~IncompleteLU();

    bool init ( SpCSRMatrixPtr& matrix, cusparseHandle_t cusparse_handle ) override;
    bool compute( SpCSRMatrixPtr& matrix, cusparseHandle_t cusparse_handle ) override;

protected:
    bool analyze( SpCSRMatrixPtr& matrix, cusparseHandle_t cusparse_handle ) override;
    bool factorize( SpCSRMatrixPtr& matrix, cusparseHandle_t cusparse_handle ) override;

private:
    csrilu02Info_t m_info;
    void* m_buffer;
    cusparseMatDescr_t m_matrix_LU;
};

}
