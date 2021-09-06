import numpy as np
cimport numpy as np

ctypedef np.npy_float32 DTYPE_t
ctypedef np.npy_float64 DOUBLE_t

cdef void mv_mul(
    double* A, 
    double* X,
    double* C,
    int N,
    int D,
    int transposeX
)

cdef double* negative_assign(
    double* a, 
    double* b, 
    int n
)

cdef void vs_mul(
    double* a, 
    double b, 
    double* c, 
    int n
)

cdef double* vv_sub(
    double* a, 
    double* b, 
    double* c, 
    int n
)

cdef double* vv_mul(
    double* a, 
    double* b, 
    double* c, 
    int n
)

cdef double* greater_than_0(
    double* a, 
    double* b, 
    int n
)

cdef double* equals_0(
    double* a, 
    double* b, 
    int n
)

cdef void row_add(
    double* X, 
    double* Y, 
    int p
)

