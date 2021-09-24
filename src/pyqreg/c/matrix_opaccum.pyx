from cython cimport boundscheck, wraparound, cdivision, nonecheck, nogil
from libc.stdlib cimport malloc, free

import numpy as np
cimport numpy as np
np.import_array()

from .blas_lapack cimport mm_dot
from .mat_vec_ops cimport row_add, mv_mul


DTYPE = np.float64


@boundscheck(False)  
@wraparound(False)
@cdivision(True)
cdef int _get_num_groups(int* group, int n):
    
    cdef int i, G = 0
    cdef int prev = group[0]
    
    for i in range(n):
        
        if group[i] != prev:
            G += 1
            prev = group[i]
    
    return G + 1  # the last index didn't have chance to increment

@boundscheck(False)
@wraparound(False)
@cdivision(True)
def get_num_groups(np.ndarray[int, ndim=1] _group, int n):
    
    cdef int* group = <int*>(np.PyArray_DATA(_group))
    cdef int G = _get_num_groups(group, n)
    
    return G

@boundscheck(False)  
@wraparound(False)
@cdivision(True)
cdef void get_group_counts(int* group_counts, int* group, int n, int G):
    
    cdef int i
    cdef int prev = group[0]
    cdef int index_counter = 0
    cdef int group_counter = 0
    
    for i in range(n):
    
        if prev != group[i]:
            group_counts[index_counter] = group_counter
            index_counter += 1

        group_counter += 1
        prev = group[i]

    group_counts[G-1] = group_counter

@boundscheck(False)  
@wraparound(False)
@cdivision(True)
cdef void _matrix_opaccum(double* varlist,
                          double* opvar,
                          int* group,
                          double* Xe, 
                          double* XeeX,
                          int n,
                          int p,
                          int G):
    
    cdef int i, j
    
    cdef int* group_counts = <int*>malloc(G * sizeof(int))
    get_group_counts(group_counts, group, n, G)
    
    cdef int g = 0
    cdef int n_g = group_counts[g]
    
    for j in range(p):
    
        for i in range(n):

            Xe[j*G + g] += varlist[j*n + i] * opvar[i]

            if i == n - 1:

                g = 0
                n_g = group_counts[g]

            elif i == n_g - 1:

                g += 1
                n_g = group_counts[g]
                
    mm_dot(&Xe[0], &Xe[0], &XeeX[0],
           p, p, G, 
           G, G, p,
           1.0, 0.0, 1, 0)
    
    free(group_counts)
    

@boundscheck(False)  
@wraparound(False)
@cdivision(True)
def matrix_opaccum(np.ndarray[DOUBLE_t, ndim=2] _varlist,
                   np.ndarray[int, ndim=1] _group,
                   np.ndarray[DOUBLE_t, ndim=1] _opvar,
                   int G):
    """Computes $$\sum_{g=1}^{G} X_g^T e_g e_g^T X_g$$,
    using the following identity:
    $$
    \sum_{g=1}^{G} X_g^T e_g e_g^T X_g =
    \left[
      \begin{array}{cccc}
          e_1^T X_1\\
          e_2^T X_2\\
          \vdots\\
          e_G^T X_G\\
      \end{array}
    \right]^T
    \left[
      \begin{array}{cccc}
          e_1^T X_1\\
          e_2^T X_2\\
          \vdots\\
          e_G^T X_G\\
      \end{array}
    \right]
    $$

    We calculate e^T X instead of X^T e since X is Fortran contiguous.
    We use inner for loop to accumulate the dot products while traversing 
    down each column.
    """
    cdef int n = _varlist.shape[0]
    cdef int p = _varlist.shape[1]

    cdef int* group = <int*>(np.PyArray_DATA(_group))
    
    cdef double* varlist = <double*>(np.PyArray_DATA(_varlist))
    cdef double* opvar = <double*>(np.PyArray_DATA(_opvar))

    cdef np.ndarray[DOUBLE_t, ndim=2, mode="fortran"] _Xe = np.zeros([G, p], dtype=DTYPE, order="F")
    cdef double* Xe = <double*>(np.PyArray_DATA(_Xe))
    
    cdef np.ndarray[DOUBLE_t, ndim=2, mode="fortran"] _XeeX = np.zeros([p, p], dtype=DTYPE, order="F")
    cdef double* XeeX = <double*>(np.PyArray_DATA(_XeeX))
    
    _matrix_opaccum(varlist, opvar, group, Xe, XeeX, n, p, G)
    
    return _XeeX






