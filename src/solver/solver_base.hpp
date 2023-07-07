#pragma once
#include "Stable.hpp"
#include "preconditioner_base.hpp"
#include "dense_vector.hpp"

namespace CudaCPP
{
template <typename Preconditioner>
class SolverBase
{
    static_assert( std::is_base_of<PreconditionerBase, Preconditioner>::value,
                   "Preconditioner must be derived from PreconditionerBase" );

    using SpCSRMatrixPtr = std::unique_ptr<SpMatrixCSR>;
    using DenseVectorPtr = std::unique_ptr<DenseVector>;
public:
    SolverBase();

    bool compute( SpCSRMatrixPtr& matrix, cusparseHandle_t cusparse_handle );
    virtual DenseVectorPtr solve ( const SpCSRMatrixPtr& matrix, const DenseVectorPtr& rhs ) = 0;

    ~SolverBase() = default;

    inline void setTolerance( double tolerance )
    {
        m_tolerance = tolerance;
    }

    inline double tolerance() const
    {
        return m_tolerance;
    }

    inline void setMaxIterations( unsigned long max_iterations )
    {
        m_max_iterations = max_iterations;
    }

    inline unsigned long maxIterations() const
    {
        return m_max_iterations;
    }

protected:
    double m_tolerance;
    unsigned long m_max_iterations;
    Preconditioner m_preconditioner;
};

template<typename Preconditioner>
SolverBase<Preconditioner>::SolverBase(): m_max_iterations( 20 ),
    m_tolerance ( std::numeric_limits<double>::epsilon() )
{}

template<typename Preconditioner>
bool SolverBase<Preconditioner>::compute( SpCSRMatrixPtr& matrix, cusparseHandle_t cusparse_handle )
{
    if ( m_preconditioner.init( matrix, cusparse_handle ) )
        return 1;

    return m_preconditioner.compute( matrix, cusparse_handle );
}

}
