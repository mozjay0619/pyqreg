from .c.fit_coefs import fit_coefs

import numpy as np

class QuantReg():

	def __init__(self, y, X):

		self.X = np.array(X, np.double, copy=False, order='F', ndmin=1)
		self.y = np.array(y, np.double, copy=False, order='F', ndmin=1)

	def fit_ipm(self, q, eps=1e-6):
		"""Estimate coefficients using the interior point method.

		Paramters
		---------
		q : double
			Quantile value strictly between 0 and 1

		eps : double
			Duality gap stopping criterion
		"""
		coefs = fit_coefs(self.X, self.y, q, eps)

		return coefs

		