from cython cimport boundscheck, wraparound, cdivision, nonecheck, nogil

import numpy as np
cimport numpy as np


ctypedef np.npy_float32 DTYPE_t
ctypedef np.npy_float64 DOUBLE_t


cdef void lapack_cholesky_decomp(
    double* A, 
    int N
)

cdef void lapack_cholesky_solve(
    double* A, 
    double* b,
    int N
)

cdef void mm_dot(
    double* A, 
    double* B,
    double* C,
    int M,
    int N,
    int K,
    int LDA,
    int LDB,
    int LDC,
    double ALPHA,
    double BETA,
    int transposeA,
    int transposeB
)


cdef void blas_axpy(
    double* X, 
    DOUBLE_t A,
    double* Y,
    int N
)


cdef void mv_dot(
    double* A, 
    double* X,
    double* Y,
    int N,
    int D,
    int LDA,
    int transposeA
)