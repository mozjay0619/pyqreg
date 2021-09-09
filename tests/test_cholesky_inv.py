import pytest

from src.pyqreg.utils import rng_generator
from src.pyqreg.c.blas_lapack import lapack_cholesky_inv

import numpy as np
from numpy.linalg import inv

def test_cholesky_inv():

	rng = rng_generator(1)
	n = 100

	A = rng.normal(size=[n, n])
	B = np.dot(A, A.transpose())
	B = np.array(B, np.double, copy=False, order='F', ndmin=1)

	assert np.all(np.close(lapack_cholesky_inv(B), inv(B)))
