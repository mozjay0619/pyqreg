import numpy as np
cimport numpy as np

ctypedef np.npy_float32 DTYPE_t
ctypedef np.npy_float64 DOUBLE_t


cdef double dot_sub_dot_plus_dot(
    double* c, 
    double* x, 
    double* y, 
    double* b, 
    double* w, 
    double* u, 
    int n, 
    int p
)

cdef void div_div_plus_div(
    double* z, 
    double* x, 
    double* w, 
    double* s, 
    double* q, 
    int n
)

cdef void mul_sub(
    double* q, 
    double* tmp1_n, 
    double* r, 
    double* dx, 
    int n
)

cdef void neg_mul_plus_div(
    double* z, 
    double* dx, 
    double* x, 
    double* dz, 
    double* w, 
    double* ds, 
    double* s, 
    double* dw, 
    int n
)

cdef double dot_plus_dot(
    double* z, 
    double* x, 
    double* w, 
    double* s, 
    int n
)

cdef double dot_plus_mul_plus_mul_plus_dot_plus_mul_plus_mul(
    double* z, 
    double* x, 
    double* w, 
    double* s,
    double* dz, 
    double* dx, 
    double* dw, 
    double* ds,
    double alpha_p, 
    double alpha_d,
    int n
)

cdef void mul_div_sub_div(
    double* xi, 
    double* x, 
    double* s, 
    double mu, 
    int n
)

cdef void add_sub_sub(
    double* ri, 
    double* r, 
    double* dxdz, 
    double* dsdw, 
    double* xi, 
    int n
)

cdef void mul_add_sub_sub_add(
    double* dx, 
    double* Qinv, 
    double* Atdy, 
    double* xi, 
    double* r, 
    double* dxdz, 
    double* dsdw, 
    int n
)

cdef void div_sub_sub_mul_div_sub(
    double* z, 
    double* dx, 
    double* dz, 
    double* x, 
    double* dxdz, 
    double* w, 
    double* ds, 
    double* dw, 
    double* s, 
    double* dsdw, 
    double mu, 
    int n
)

cdef double min_step_length(
    double* x,
    double* dx,
    double* s,
    double* ds,
    int n
)

