import numpy as np
import pytest

from src.pyqreg.c.fit_coefs import fit_coefs


def test_fit_coefs():

    X = np.hstack(
        [np.ones(1000).reshape([1000, 1]), np.arange(1000).reshape([1000, 1])]
    ).astype(np.double)
    y = np.arange(3, 1003).astype(np.double)

    X = np.array(X, np.double, copy=False, order="F", ndmin=1)
    y = np.array(y, np.double, copy=False, order="F", ndmin=1)

    coefs = fit_coefs(X, y, 0.5, 1e-6)

    assert np.all(np.isclose(coefs, np.asarray([3.0, 1.0])))
