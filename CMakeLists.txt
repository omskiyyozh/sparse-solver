cmake_minimum_required(VERSION 3.21)

project(SparseSolver LANGUAGES CXX CUDA)

if (MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    add_compile_options(/W4 /WX /openmp)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_custom_target(AStyle SOURCES
    .astylerc
)


find_package (Eigen3 CONFIG REQUIRED)
find_package(CUDA)
include_directories("${CUDA_INCLUDE_DIRS}")
set(Boost_NO_WARN_NEW_VERSIONS 1)
find_package(Boost 1.78.0 REQUIRED COMPONENTS program_options)
include_directories(${Boost_INCLUDE_DIR})

find_package(GTest CONFIG REQUIRED)
include(GoogleTest)

add_custom_target(precomp SOURCES
    src/Stable.hpp
)

add_library (SolverCUDALib STATIC
    src/solver/bicgstab.cpp
    src/solver/bicgstab.hpp
    src/solver/dense_vector.hpp
    src/solver/dense_vector.cpp
    src/solver/preconditioner_base.hpp
    src/solver/incompleteLU.hpp
    src/solver/incompleteLU.cpp
    src/solver/spcsr_matrix.hpp
    src/solver/spcsr_matrix.cpp
    src/solver/solver_base.hpp
    src/solver/bicgstab_wrap.hpp
)

include_directories(src)
include_directories(src/solver)

target_precompile_headers(SolverCUDALib PUBLIC
    src/Stable.hpp
)
target_link_libraries(SolverCUDALib PUBLIC
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDA_cusolver_LIBRARY}
    ${CUDA_cusparse_LIBRARY}
)

add_executable(SparseSolver
    src/main.cpp
)
target_link_libraries(SparseSolver
    Eigen3::Eigen
    Boost::program_options
    SolverCUDALib
)
target_precompile_headers(SparseSolver
    REUSE_FROM SolverCUDALib
)
enable_testing(True)

set (TEST_MATRIX Trefethen_150.mtx)
if (NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/${TEST_MATRIX})
    file (COPY tests/${TEST_MATRIX} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endif()

add_executable(tests
    tests/main.cpp
    tests/dense_vector_tests.cpp
    tests/spcsr_matrix_tests.cpp
    tests/ilu_tests.cpp
    tests/bicgstab_test.cpp
)

target_link_libraries(tests PUBLIC
    GTest::gtest
    GTest::gtest_main
    SolverCUDALib
)
gtest_discover_tests(tests)
