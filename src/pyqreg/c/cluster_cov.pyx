from cython cimport boundscheck, cdivision, nogil, nonecheck, wraparound
from libc.stdlib cimport free, malloc

import numpy as np

cimport numpy as np

np.import_array()

from .blas_lapack cimport _lapack_cholesky_inv, mm_dot
from .cluster_cov cimport _psi_function
from .mat_vec_ops cimport mv_mul
from .matrix_opaccum cimport _get_num_groups, _matrix_opaccum
from .stats cimport _invnormal, _normalden

DTYPE = np.float64


@boundscheck(False)
@wraparound(False)
@cdivision(True)
cdef void _psi_function(double* a, 
                        double* b,
                        double theta, 
                        int n):
    
    cdef int i
    
    for i in range(n):
        
        if a[i] <= 0:
            
            b[i] = theta - 1
        
        else:
            
            b[i] = theta 

@boundscheck(False)
@wraparound(False)
@cdivision(True)
def psi_function(np.ndarray[DOUBLE_t, ndim=1] _a,
                 double theta):
    
    cdef int n = len(_a)
    
    cdef double* a = <double*>(np.PyArray_DATA(_a))
    
    cdef np.ndarray[DOUBLE_t, ndim=1, mode="fortran"] _b = np.empty(n, dtype=DTYPE, order="F")
    cdef double* b = <double*>(np.PyArray_DATA(_b))
    
    _psi_function(a, b, theta, n)
    
    return _b
