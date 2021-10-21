import numpy as np
import pytest

from src.pyqreg.c.matrix_opaccum import matrix_opaccum


def test_group_matrix_opaccum_using_identity():

    X = np.arange(9 * 2).reshape(9, 2)
    e = np.arange(9).reshape(9, 1)

    X = np.array(X, np.double, copy=False, order="F", ndmin=1)
    e = np.array(e, np.double, copy=False, order="F", ndmin=1)

    group_array = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2]).astype(np.int32)

    X1 = X[0:3]
    X2 = X[3:6]
    X3 = X[6:9]

    e1 = e[0:3]
    e2 = e[3:6]
    e3 = e[6:9]

    tmp = np.hstack([X1.T @ e1, X2.T @ e2, X3.T @ e3])

    assert np.all(np.isclose(tmp @ tmp.T, matrix_opaccum(X, group_array, e.ravel(), 9)))

    group_array = np.arange(9).astype(np.int32)

    tmp = np.hstack([X[[i]].T @ e[[i]] for i in range(9)])

    assert np.all(np.isclose(tmp @ tmp.T, matrix_opaccum(X, group_array, e.ravel(), 9)))


def test_group_matrix_opaccum_using_formula():

    X = np.arange(9 * 2).reshape(9, 2)
    e = np.arange(9).reshape(9, 1)

    X = np.array(X, np.double, copy=False, order="F", ndmin=1)
    e = np.array(e, np.double, copy=False, order="F", ndmin=1)

    group_array = np.arange(9).astype(np.int32)

    output_array = np.zeros([2, 2])

    for i in range(9):
        output_array += X[[i]].T @ e[[i]] @ e[[i]].T @ X[[i]]

    output_array

    assert np.all(
        np.isclose(output_array, matrix_opaccum(X, group_array, e.ravel(), 9))
    )
