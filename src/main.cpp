#include "Stable.hpp"
#include "bicgstab.hpp"
#include "bicgstab_wrap.hpp"
#include "incompleteLU.hpp"

namespace po = boost::program_options;

using hr_clock = std::chrono::high_resolution_clock;

using SparseMatrixd = Eigen::SparseMatrix <double, Eigen::RowMajor>;
using SparseVectorxd = Eigen::SparseVector<double, Eigen::RowMajor>;
using Vectorxd = Eigen::VectorXd;

enum class ExecutionPolicy : char
{
    Serial = 0,
    OMP,
    CUDA,
    MPI
};

int main( int argc, char** argv )
{

    po::options_description description ( "Allowed options" );
    description.add_options()( "help,h", "produce help message" )
    ( "mtx,m", po::value<std::string>(), "set Matrix Market file to read " )
    ( "rhs,r", po::value<std::string>(), "set file with rhs vector" )
    ( "file,f", po::value<std::string>(), "set output file" )
    ( "parallelism,p", po::value<short int>()->default_value( 0 ), "set parallelism type\n"
      "0 (default): no parallelism\n"
      "1: CPU (OpenMP)\n"
      "2: GPU (CUDA)\n"
      "3: Distributed CPU (MPI)\n"
    );
    po::variables_map args;

    try
    {
        po::store ( po::command_line_parser( argc, argv ).options( description ).run(), args );
        po::notify ( args );

        if ( args.count( "help" ) || argc == 1 )
        {

            std::cout << description << std::endl;
            return 0;
        }

        std::ofstream output_file;
        output_file.exceptions( std::ofstream::failbit | std::ofstream::badbit );

        if ( args.count( "file" ) )
        {
            output_file.open( args["file"].as<std::string>() );
            std::cout.rdbuf( output_file.rdbuf() );
        }

        SparseMatrixd A;

        if ( !args.count( "mtx" ) )
        {
            std::cout << "Option 'mtx' is required but missing\n";
            return 2;
        }

        if ( !Eigen::loadMarket<SparseMatrixd>( A, args["mtx"].as<std::string>() ) )
        {
            std::cerr << "Unable to open file" << std::endl;
            return 2;
        }

        Vectorxd b ( A.rows() );

        if ( args.count( "rhs" ) )
        {
            if ( !Eigen::loadMarketVector<Vectorxd>( b, args["rhs"].as<std::string>() ) )
            {
                std::cerr << "Unable to open file" << std::endl;
                return 2;
            }
        }
        else
            b.setOnes();

        ExecutionPolicy policy = static_cast<ExecutionPolicy>( args["parallelism"].as<short int>() );

        Eigen::BiCGSTAB<SparseMatrixd, Eigen::IncompleteCholesky<double>> solver;
        solver.compute( A );

        switch ( policy )
        {
            case ExecutionPolicy::Serial:
            {
                auto start = hr_clock::now();
#define EIGEN_DONT_PARALLELIZE
                auto x = solver.solve( b );
#undef EIGEN_DONT_PARALLELIZE
                std::cout << x;
                std::cerr << std::chrono::duration_cast<std::chrono::milliseconds>( hr_clock::now() - start ).count()
                          << " ms\n";

                break;
            }

            case ExecutionPolicy::OMP:
            {
                const auto logical_cores = std::thread::hardware_concurrency();
                Eigen::setNbThreads( logical_cores / 2 );
                auto start = hr_clock::now();
                auto x = solver.solve( b );
                std::cout << x;
                std::cerr << std::chrono::duration_cast<std::chrono::milliseconds>( hr_clock::now() - start ).count()
                          << " ms\n";

                break;
            }

            case ExecutionPolicy::CUDA:
            {
                Vectorxd x_gpu ( A.rows() );
                double* A_values = A.valuePtr();
                int* A_rows_ptr = A.outerIndexPtr();
                int* A_col_ptr = A.innerIndexPtr();
                double* b_ptr = b.data();
                const size_t size = A.rows();
                const size_t nnz = A.nonZeros();
                CudaCPP::SpMatrixAttributes attributes;
                attributes.col_ind_type = CUSPARSE_INDEX_32I;
                attributes.row_offsets_type = CUSPARSE_INDEX_32I;
                attributes.values_type = CUDA_R_64F;
                attributes.index_base = CUSPARSE_INDEX_BASE_ZERO;

                auto dev_A = CudaCPP::SpMatrixCSR::create( A_values, A_col_ptr, A_rows_ptr, size, nnz, attributes );

                assert ( dev_A != nullptr );

                auto dev_B = CudaCPP::DenseVector::create( b_ptr, size, CUDA_R_64F );

                assert ( dev_B != nullptr );

                CudaCPP::BiCGSTAB<CudaCPP::IncompleteLU> solver_wrap;

                auto start = hr_clock::now();
                auto dev_x = solver_wrap.solve( dev_A, dev_B );
                std::cerr << std::chrono::duration_cast<std::chrono::milliseconds>( hr_clock::now() - start ).count()
                          << " ms\n";

                assert( dev_x != nullptr );

                std::vector<double> host_x ( size );

                if ( dev_x -> copyToHost( host_x.data(), size ) )
                    return 2;

                std::copy( host_x.begin(), host_x.end(), std::ostream_iterator<double>( std::cout, "\n" ) );
            }
            break;

            case ExecutionPolicy::MPI:
                std::cout << "MPI\n";
                break;

            default:
                std::cout << "Unknown parallelism type" << std::endl;
                return 2;

        };
    }
    catch ( std::exception& e )
    {
        std::cerr << e.what() << std::endl;
        return 2;
    }

    return 0;

}
