from cython cimport boundscheck, wraparound, cdivision, nonecheck, nogil
# https://github.com/scipy/scipy/blob/v1.7.1/scipy/linalg/lapack.py
from scipy.linalg.cython_lapack cimport dpotrf, dpotrs  
# https://github.com/scipy/scipy/blob/v1.7.1/scipy/linalg/blas.py
from scipy.linalg.cython_blas cimport dgemv, daxpy, dgemm, dscal, dcopy 

import numpy as np
cimport numpy as np
np.import_array()


DTYPE = np.float64


@boundscheck(False)
@wraparound(False)
@cdivision(True)
cdef void lapack_cholesky_decomp(double* A, int N):
    """
    http://www.netlib.org/lapack/explore-html/d1/d7a/ \
    group__double_p_ocomputational_ga2f55f604a6003d03b5cd4a0adcfb74d6. \
    html
    """
    cdef int info = 0
    
    dpotrf('L', &N, A, &N, &info)

@boundscheck(False)
@wraparound(False)
@cdivision(True)
cdef void lapack_cholesky_solve(double* A, 
                           double* b,
                           int N):
    """
    http://www.netlib.org/lapack/explore-html/d1/d7a/ \
    group__double_p_ocomputational_ga167aa0166c4ce726385f65e4ab05e7c1. \
    html#ga167aa0166c4ce726385f65e4ab05e7c1
    """
    cdef int info = 0
    cdef int one = 1
    
    dpotrs('L', &N, &one, A, &N, b, &N, &info)

@boundscheck(False)
@wraparound(False)
@cdivision(True)
cdef void mm_dot(double* A, 
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
                 int transposeB):
    """
    http://www.netlib.org/lapack/explore-html/d1/d54/ \
    group__double__blas__level3_gaeda3cbd99c8fb834a60a6412878226e1. \
    html#gaeda3cbd99c8fb834a60a6412878226e1
    """
    if transposeA==1:
        dgemm('T', 'N', &M, &N, &K, &ALPHA, A, &LDA, B, &LDB, &BETA, C, &LDC)

    elif transposeB==1:
        dgemm('N', 'T', &M, &N, &K, &ALPHA, A, &LDA, B, &LDB, &BETA, C, &LDC)

    else:
        dgemm('N', 'N', &M, &N, &K, &ALPHA, A, &LDA, B, &LDB, &BETA, C, &LDC)

@boundscheck(False)
@wraparound(False)
@cdivision(True)
cdef void blas_axpy(double* X, 
               DOUBLE_t A,
               double* Y,
               int N):
    """
    http://www.netlib.org/lapack/explore-html/de/da4/ \
    group__double__blas__level1_ga8f99d6a644d3396aa32db472e0cfc91c. \
    html#ga8f99d6a644d3396aa32db472e0cfc91c
    """
    cdef int INCX = 1 
    cdef int INCY = 1 

    daxpy(&N, &A, X, &INCX, Y, &INCY)

@boundscheck(False)
@wraparound(False)
@cdivision(True)
cdef void mv_dot(double* A, 
                 double* X,
                 double* Y,
                 int N,
                 int D,
                 int LDA,
                 int transposeA):
    """
    http://www.netlib.org/lapack/explore-html/d7/d15/ \
    group__double__blas__level2_gadd421a107a488d524859b4a64c1901a9. \
    html#gadd421a107a488d524859b4a64c1901a9
    """
    cdef int INCX = 1 
    cdef int INCY = 1
    cdef double ALPHA = 1.0
    cdef double BETA = 0.0

    if transposeA==1:
        dgemv('T', &N, &D, &ALPHA, A, &LDA, X, &INCX, &BETA, Y, &INCY)
        
    else:
        dgemv('N', &N, &D, &ALPHA, A, &LDA, X, &INCX, &BETA, Y, &INCY)
