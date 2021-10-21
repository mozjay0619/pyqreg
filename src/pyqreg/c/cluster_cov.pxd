from cython cimport boundscheck, cdivision, nogil, nonecheck, wraparound

import numpy as np

cimport numpy as np

ctypedef np.npy_float32 DTYPE_t
ctypedef np.npy_float64 DOUBLE_t


cdef void _psi_function(
	double* a, 
	double* b,
	double theta, 
	int n
)
