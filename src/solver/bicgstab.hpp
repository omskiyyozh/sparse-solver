#pragma once
#include "Stable.hpp"
int bicgstabGPU ( const double* A, const int* cols_ptr, const int* rows_ptr,
                  const double* b, double* x, const double tol,
                  const size_t size, const size_t nnz, const int max_iterations );
