"""
A Frisch-Newton algorithm as described in http://projecteuclid.org/euclid.ss/1030037960
qreg_ip_coef is a modified lp_fnm of Daniel Morillo & Roger Koenker
Translated from Ox to Matlab by Paul Eilers 1999
Modified by Roger Koenker 2000
More changes by Paul Eilers 2004
Translated to Julia by Patrick Kofod Mogensen 2015
"""

from cython cimport boundscheck, wraparound, cdivision, nonecheck, nogil
from libc.stdlib cimport malloc, free

import numpy as np
cimport numpy as np
np.import_array()

from .blas_lapack cimport lapack_cholesky_decomp
from .blas_lapack cimport lapack_cholesky_solve
from .blas_lapack cimport mm_dot
from .blas_lapack cimport blas_axpy
from .blas_lapack cimport mv_dot
from .mat_vec_ops cimport mv_mul
from .mat_vec_ops cimport negative_assign
from .mat_vec_ops cimport vs_mul
from .mat_vec_ops cimport vv_sub
from .mat_vec_ops cimport vv_mul
from .mat_vec_ops cimport greater_than_0
from .mat_vec_ops cimport equals_0


DTYPE = np.float64


@boundscheck(False)  
@wraparound(False)
@cdivision(True)
cdef double dot_sub_dot_plus_dot(double* c, double* x, double* y, double* b, double* w, double* u, int n, int p):
    """
    gap = dot(c, x) - dot(y, b) + dot(w, u)
    
    c in R[n, 1]
    y in R[p, 1]
    gap in R
    """
    cdef int i
    cdef double gap = 0
    
    for i in range(n):
        gap += (c[i] * x[i]) + (w[i] * u[i])
        
    for i in range(p):
        gap -= (y[i] * b[i])
        
    return gap


@boundscheck(False)  
@wraparound(False)
@cdivision(True)
cdef void div_div_plus_div(double* z, double* x, double* w, double* s, double* q, int n):
    """
    Qinv = 1 / (z / x + w / s)
    
    z in R[n, 1]
    """
    cdef int i
    
    for i in range(n):
        q[i] = 1 / (z[i]/x[i] + w[i]/s[i])
    
@boundscheck(False)  
@wraparound(False)
@cdivision(True)
cdef void mul_sub(double* q, double* tmp1_n, double* r, double* dx, int n):
    """
    dx = Qinv(Atdy - r)
    
    Qinv in R[n, 1]
    """
    cdef int i
    
    for i in range(n):
        dx[i] = q[i] * (tmp1_n[i] - r[i])
        
@boundscheck(False)  
@wraparound(False)
@cdivision(True)
cdef void neg_mul_plus_div(double* z, double* dx, double* x, double* dz, 
                           double* w, double* ds, double* s, double* dw, 
                           int n):
    """
    dz = -z * (1 + Xinvdx)
    dw = -w * (1 + Sinvds)
    
    z in R[n, 1]
    w in R[n, 1]
    """
    cdef int i
    
    for i in range(n):
        dz[i] = -1 * z[i] * (1 + dx[i] / x[i])
        dw[i] = -1 * w[i] * (1 + ds[i] / s[i])

@boundscheck(False)  
@wraparound(False)
@cdivision(True)
cdef double dot_plus_dot(double* z, double* x, double* w, double* s, int n):
    """
    mu = xz + sw
    
    x in R[n, 1]
    s in R[n, 1]
    mu in R
    """
    cdef int i
    cdef double mu = 0
    
    for i in range(n):
        mu += z[i] * x[i] + w[i] * s[i]
    
    return mu

@boundscheck(False)  
@wraparound(False)
@cdivision(True)
cdef double dot_plus_mul_plus_mul_plus_dot_plus_mul_plus_mul(
    double* z, double* x, double* w, double* s,
    double* dz, double* dx, double* dw, double* ds,
    double alpha_p, double alpha_d,
    int n):
    """
    mu_1 = (x + alpha_p*dx)(z + alpha_d*dz) + (s + alpha_p*ds)(w + alpha_d*dw)
    
    x in R[n, 1]
    mu_1 in R
    """
    cdef int i
    cdef double mu_1 = 0
    
    for i in range(n):
        mu_1 += (x[i] + alpha_p * dx[i]) * (z[i] + alpha_d * dz[i])
        mu_1 += (s[i] + alpha_p * ds[i]) * (w[i] + alpha_d * dw[i])
        
    return mu_1

@boundscheck(False)  
@wraparound(False)
@cdivision(True)
cdef void mul_div_sub_div(double* xi, double* x, double* s, double mu, int n):
    """
    xi = mu * (Xinv - Sinv)
    
    Xinv in R[n, 1]
    """
    cdef int i
    
    for i in range(n):
        xi[i] = mu * (1/x[i] - 1/s[i])

@boundscheck(False)  
@wraparound(False)
@cdivision(True)
cdef void add_sub_sub(double* ri, double* r, double* dxdz, double* dsdw, double* xi, int n):
    """
    ri = r + dxdz - dsdw - xi
    
    r in R[n, 1]
    """
    cdef int i
    
    for i in range(n):
        ri[i] = r[i] + dxdz[i] - dsdw[i] - xi[i]
        
@boundscheck(False)  
@wraparound(False)
@cdivision(True)
cdef void mul_add_sub_sub_add(
    double* dx, double* Qinv, double* Atdy, double* xi, double* r, double* dxdz, double* dsdw, int n):
    """
    dx = Qinv(Atdy + xi - r - dxdz + dsdw)
    
    Qinv in R[n, 1]
    """
    cdef int i
    
    for i in range(n):
        dx[i] = Qinv[i] * (Atdy[i] + xi[i] - r[i] - dxdz[i] + dsdw[i])

@boundscheck(False)  
@wraparound(False)
@cdivision(True)
cdef void div_sub_sub_mul_div_sub(double* z, double* dx, double* dz, double* x, double* dxdz, 
                                  double* w, double* ds, double* dw, double* s, double* dsdw, 
                                  double mu, int n):
    """
    dz = mu / x - z - z * dx / x - dxdz
    dw = mu / s - w - w * ds / s - dsdw
    
    x in R[n, 1]
    """
    cdef int i
    
    for i in range(n):
        dz[i] = mu / x[i] - z[i] - z[i] * dx[i] / x[i] - dxdz[i]
        dw[i] = mu / s[i] - w[i] - w[i] * ds[i] / s[i] - dsdw[i]

@boundscheck(False)  
@wraparound(False)
@cdivision(True)
cdef double min_step_length(double* x,
                            double* dx,
                            double* s,
                            double* ds,
                            int n):
    
    cdef int i
    cdef double min_d = 1e20
        
    for i in range(n):
        if dx[i] < 0.0:
            if -x[i]/dx[i] < min_d:
                min_d = -x[i]/dx[i]
        
    for i in range(n):
        if ds[i] < 0.0:
            if -s[i]/ds[i] < min_d:
                min_d = -s[i]/ds[i]
            
    return min_d


@boundscheck(False)  
@wraparound(False)
@cdivision(True)
def fit_coefs(np.ndarray[DOUBLE_t, ndim=2] X_input, 
              np.ndarray[DOUBLE_t, ndim=1] y_input,
              DOUBLE_t tau,
              DOUBLE_t eps):
    
    # Loop index
    cdef int i, j

    # Dimensions
    cdef int n = X_input.shape[0]
    cdef int p = X_input.shape[1]

    # Model data
    cdef double* X = <double*>(np.PyArray_DATA(X_input))
    cdef double* Y = <double*>(np.PyArray_DATA(y_input))
    
    # Model constants
    cdef int max_it = 500
    cdef double sigma = 0.99995
    
    # Unit vector
    cdef double* u = <double*>malloc(n * sizeof(double))
    for i in range(n):
        u[i] = 1.0

    # Primal constants
    cdef double* c = <double*>malloc(n * sizeof(double))
    cdef double* b = <double*>malloc(p * sizeof(double))

    # Primal variables
    cdef double* x = <double*>malloc(n * sizeof(double))
    cdef double* s = <double*>malloc(n * sizeof(double))

    # Dual variables
    cdef double* w = <double*>malloc(n * sizeof(double))
    cdef double* y = <double*>malloc(p * sizeof(double))
    cdef double* z = <double*>malloc(n * sizeof(double))
    
    # Set up caches
    cdef double* Qinv = <double*>malloc(n * sizeof(double))
    cdef double* r = <double*>malloc(n * sizeof(double))
    cdef np.ndarray[DOUBLE_t, ndim=2, mode="fortran"] AQinvT_np_ = np.empty([n, p], dtype=DTYPE, order="F")
    cdef double* AQinvT_np = <double*>(np.PyArray_DATA(AQinvT_np_))
    cdef np.ndarray[DOUBLE_t, ndim=2, mode="fortran"] AQinvAt_pp_ = np.empty([p, p], dtype=DTYPE, order="F")
    cdef double* AQinvAt_pp = <double*>(np.PyArray_DATA(AQinvAt_pp_))
    cdef double* Atdy = <double*>malloc(n * sizeof(double))
    cdef double* dy = <double*>malloc(p * sizeof(double))
    cdef double* dx = <double*>malloc(n * sizeof(double))
    cdef double* ds = <double*>malloc(n * sizeof(double))
    cdef double* dz = <double*>malloc(n * sizeof(double))
    cdef double* dw = <double*>malloc(n * sizeof(double))
    cdef double* dxdz = <double*>malloc(n * sizeof(double))
    cdef double* dsdw = <double*>malloc(n * sizeof(double))
    cdef double* xi = <double*>malloc(n * sizeof(double))
    cdef double* ri = <double*>malloc(n * sizeof(double))
    cdef double* d = <double*>malloc(2 * n * sizeof(double))
    cdef double alpha_d 
    cdef double alpha_p 
    cdef double gap
    cdef double mu
    cdef double mu_1
    cdef double* mus = <double*>malloc(2 * sizeof(double))
    
    # Set up cache for return coefficient array
    cdef np.ndarray[DOUBLE_t, ndim=1, mode="fortran"] _coefs = np.empty(p, dtype=DTYPE, order="F")      
    cdef double* coefs = <double*>(np.PyArray_DATA(_coefs))

    # Set up temporary caches
    cdef np.ndarray[DOUBLE_t, ndim=2, mode="fortran"] _TMP1_pp = np.empty([p, p], dtype=DTYPE, order="F")
    cdef double* TMP1_pp = <double*>(np.PyArray_DATA(_TMP1_pp))
    cdef np.ndarray[DOUBLE_t, ndim=2, mode="fortran"] _TMP1_np = np.empty([n, p], dtype=DTYPE, order="F")
    cdef double* TMP1_np = <double*>(np.PyArray_DATA(_TMP1_np))
    cdef double* tmp1_p = <double*>malloc(p * sizeof(double))
    cdef double* tmp1_n = <double*>malloc(n * sizeof(double))

    # Initialize primal x
    # x = (1 - tau) * u
    vs_mul(&u[0], (1-tau), &x[0], n)

    # Set primal constants
    # c = -Y
    negative_assign(&Y[0], &c[0], n)

    # b = X'x = Ax
    mv_dot(&X[0], &x[0], &b[0], n, p, n, 1)

    # Set initial feasible point
    # Initialize s = u - x
    vv_sub(&u[0], &x[0], &s[0], n)

    # Initialize y = -((X'X)\(X'y))
    mm_dot(&X[0], &X[0], &TMP1_pp[0],
           p, p, n, 
           n, n, p,
           1.0, 0.0, 1, 0)
    mv_dot(&X[0], &Y[0], &tmp1_p[0], n, p, n, 1)
    lapack_cholesky_decomp(&TMP1_pp[0], p)  # Since X'X post def
    lapack_cholesky_solve(&TMP1_pp[0], &tmp1_p[0], p)
    negative_assign(&tmp1_p[0], &y[0], p)

    # Initialize r = c - A'y = c - Xy
    mv_dot(&X[0], &y[0], &tmp1_n[0], n, p, n, 0)
    vv_sub(&c[0], &tmp1_n[0], &r[0], n)
    equals_0(&r[0], &tmp1_n[0], n)
    blas_axpy(&tmp1_n[0], 0.001, &r[0], n)
    
    # Initialize z 
    greater_than_0(&r[0], &tmp1_n[0], n)
    vv_mul(&r[0], &tmp1_n[0], &z[0], n)
    
    # Initialize w = z - r
    vv_sub(&z[0], &r[0], &w[0], n)
    
    # gap = dot(c, x) - dot(y, b) + dot(w, u)
    gap = dot_sub_dot_plus_dot(&c[0], &x[0], &y[0], &b[0], &w[0], &u[0], n, p)
    
    for i in range(max_it):
        
        # Qinv = 1 / (z / x + w / s)
        div_div_plus_div(&z[0], &x[0], &w[0], &s[0], &Qinv[0], n)
        
        # r = z - w
        vv_sub(&z[0], &w[0], &r[0], n)
        
        # AQinvT_np = (AQinv)t [n, p]
        mv_mul(&X[0], &Qinv[0], &AQinvT_np[0], n, p, 1)
        
        # AQinvAt_pp = AQinvAt [p, p]
        mm_dot(&AQinvT_np[0], &X[0], &AQinvAt_pp[0],
               p, p, n, 
               n, n, p,
               1.0, 0.0, 1, 0)
        
        # Cholesky factor AQinvAt
        lapack_cholesky_decomp(&AQinvAt_pp[0], p)
        
        # dy = (AQinvQt)inv(AQinv)r
        mv_dot(&AQinvT_np[0], &r[0], &dy[0], n, p, n, 1)
        lapack_cholesky_solve(&AQinvAt_pp[0], &dy[0], p)
        
        # Atdy 
        mv_dot(&X[0], &dy[0], &Atdy[0], n, p, n, 0)
        
        # Affine steps
        # dx = Qinv(Atdy - r)
        mul_sub(&Qinv[0], &Atdy[0], &r[0], &dx[0], n)
        
        # ds = -dx
        vs_mul(&dx[0], -1.0, &ds[0], n)
        
        # dz = -z * (1 + Xinvdx)
        # dw = -w * (1 + Sinvds)
        neg_mul_plus_div(&z[0], &dx[0], &x[0], &dz[0], 
                         &w[0], &ds[0], &s[0], &dw[0],
                         n)
        
        alpha_p = min_step_length(&x[0], &dx[0], &s[0], &ds[0], n)
        alpha_p = min(sigma * alpha_p, 1.0)
        alpha_d = min_step_length(&w[0], &dw[0], &z[0], &dz[0], n)
        alpha_d = min(sigma * alpha_d, 1.0)
        
        if min(alpha_p, alpha_d) < 1:
            
            # Current mu
            # mu = xz + sw
            mu = dot_plus_dot(&z[0], &x[0], &w[0], &s[0], n)
            
            # Potential mu
            # mu_1 = (x + alpha_p*dx)(z + alpha_d*dz) + (s + alpha_p*ds)(w + alpha_d*dw)
            mu_1 = dot_plus_mul_plus_mul_plus_dot_plus_mul_plus_mul(
                &z[0], &x[0], &w[0], &s[0],
                &dz[0], &dx[0], &dw[0], &ds[0],
                alpha_p, alpha_d, n)
            
            # Update rule
            mu = mu * (mu_1 / mu)**3 / (2 * n)

            # dxdz = dx * dz
            vv_mul(&dx[0], &dz[0], &dxdz[0], n)
            
            # dsdw = ds * dw
            vv_mul(&ds[0], &dw[0], &dsdw[0], n)

            # xi = mu * (Xinv - Sinv)
            mul_div_sub_div(&xi[0], &x[0], &s[0], mu, n)
            
            # ri = r + dxdz - dsdw - xi
            add_sub_sub(&ri[0], &r[0], &dxdz[0], &dsdw[0], &xi[0], n)
            
            # dy = (AQinvQt)inv(AQinv)r
            mv_dot(&AQinvT_np[0], &ri[0], &dy[0], n, p, n, 1)
            lapack_cholesky_solve(&AQinvAt_pp[0], &dy[0], p)
            
            # Atdy
            mv_dot(&X[0], &dy[0], &Atdy[0], n, p, n, 0)
            
            # Corrector steps
            # dx = Qinv(Atdy + xi - r - dxdz + dsdw)
            mul_add_sub_sub_add(&dx[0], &Qinv[0], &Atdy[0], &xi[0], &r[0], &dxdz[0], &dsdw[0], n)
            
            # ds = -dx
            vs_mul(&dx[0], -1.0, &ds[0], n)

            # dz = mu / x - z - z * dx / x - dxdz
            # dw = mu / s - w - w * ds / s - dsdw
            div_sub_sub_mul_div_sub(&z[0], &dx[0], &dz[0], &x[0], &dxdz[0], 
                                    &w[0], &ds[0], &dw[0], &s[0], &dsdw[0], 
                                    mu, n)
            
            alpha_p = min_step_length(&x[0], &dx[0], &s[0], &ds[0], n)
            alpha_p = min(sigma * alpha_p, 1.0)
            alpha_d = min_step_length(&w[0], &dw[0], &z[0], &dz[0], n)
            alpha_d = min(sigma * alpha_d, 1.0)
            
        blas_axpy(dx, alpha_p, x, n)
        blas_axpy(ds, alpha_p, s, n)
        blas_axpy(dy, alpha_d, y, p)
        blas_axpy(dw, alpha_d, w, n)
        blas_axpy(dz, alpha_d, z, n)
        
        # gap = dot(c, x) - dot(y, b) + dot(w, u)
        gap = dot_sub_dot_plus_dot(&c[0], &x[0], &y[0], &b[0], &w[0], &u[0], n, p)
        
        if gap < eps:
            break
    
    # Regression coefficients
    negative_assign(&y[0], &coefs[0], p)
    
    # Free memory
    free(u)
    free(c)
    free(b)
    free(x)
    free(s)
    free(w)
    free(y)
    free(z)
    free(Qinv)
    free(r)
    free(Atdy)
    free(dy)
    free(dx)
    free(ds)
    free(dz)
    free(dw)
    free(dxdz)
    free(dsdw)
    free(xi)
    free(ri)
    free(d)
    free(mus)
    free(tmp1_p)
    free(tmp1_n)

    return _coefs
