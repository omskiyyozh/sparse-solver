#pragma once

#ifdef _MSC_VER
    #pragma warning (disable: 4127 4459 4505 4068)
#endif

#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include <math.h>
#include <algorithm>
#include <exception>
#include <chrono>
#include <thread>
#include <numeric>
#include <type_traits>
#include <utility>

#include <boost/program_options.hpp>

#include <omp.h>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <unsupported/Eigen/SparseExtra>
#include <Eigen/IterativeLinearSolvers>

#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>
#include <cusolver_common.h>
#include <cusolverSp.h>

#include "gtest/gtest.h"
