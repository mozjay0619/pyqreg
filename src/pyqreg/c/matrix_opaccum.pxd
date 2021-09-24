import numpy as np
cimport numpy as np

ctypedef np.npy_float32 DTYPE_t
ctypedef np.npy_float64 DOUBLE_t


cdef int _get_num_groups(
    int* group, 
    int n
)

cdef void get_group_counts(
    int* group_counts, 
    int* group, 
    int n, 
    int G
)

cdef void _matrix_opaccum(
    double* varlist,
    double* opvar,
    int* group,
    double* Xe, 
    double* XeeX,
    int n,
    int p,
    int G
)