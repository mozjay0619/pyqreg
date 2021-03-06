import numpy as np
import pytest
from numpy.linalg import inv

from src.pyqreg.c.blas_lapack import lapack_cholesky_inv
from src.pyqreg.utils import rng_generator


def test_cholesky_inv():

    rng = rng_generator(1)
    n = 10

    A = rng.normal(size=[n, n])
    B = np.dot(A, A.transpose())
    B = np.array(B, np.double, copy=False, order="F", ndmin=1)

    # We need to compute inv(B) since lapack inversion is inplace operation.
    assert np.all(np.isclose(inv(B), lapack_cholesky_inv(B)))

    rng = rng_generator(11)
    n = 100

    A = rng.normal(size=[n, n])
    B = np.dot(A, A.transpose())
    B = np.array(B, np.double, copy=False, order="F", ndmin=1)

    assert np.all(np.isclose(inv(B), lapack_cholesky_inv(B)))

    rng = rng_generator(111)
    n = 1000

    A = rng.normal(size=[n, n])
    B = np.dot(A, A.transpose())
    B = np.array(B, np.double, copy=False, order="F", ndmin=1)

    assert np.all(np.isclose(inv(B), lapack_cholesky_inv(B)))
