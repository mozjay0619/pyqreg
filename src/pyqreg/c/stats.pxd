import numpy as np
cimport numpy as np

ctypedef np.npy_float32 DTYPE_t
ctypedef np.npy_float64 DOUBLE_t


cdef double _invnormal(
	double p
)

cdef double _normalden(
	double z
)