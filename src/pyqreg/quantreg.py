from .c.fit_coefs import fit_coefs
# from .c.stats import invnormal
# from .c.stats import normalden

from .utils import rng_generator

import numpy as np
from numpy.linalg import pinv


class QuantReg():

	def __init__(self, y, X):

		self.X = np.array(X, np.double, copy=False, order='F', ndmin=1)
		self.y = np.array(y, np.double, copy=False, order='F', ndmin=1)

	def fit(self, q, fit_method=None, seed=None, eps=1e-6, Mm_factor=0.8, max_bad_fixup=3, kappa_eps=1e-6):
		"""Solve by interior point method (Mehrotra's predictor corrector
		algorithm). If n >= 100,000, it will use preprocessing step following
		Portnoy and Koenker (1997).

		Parameters
		----------
		q : double
			Quantile value strictly between 0 and 1

		fit_method : str or None
			Coefficient estimation method. Default is None.

			- None : uses ipm if n < 100000, else, preproc-ipm.
			- ipm : interior point method.
			- preproc-ipm : interior point method with preprocessing.

		seed : int or None
			Random seed to use if preproc-ipm is used for subsampling.

		"""
		if fit_method is None:

			n = len(self.X)

			if n >= 100000:

				rng = rng_generator(seed)

				return self.fit_preproc_ipm(q, rng, Mm_factor, max_bad_fixup, kappa_eps)

			else:

				return self.fit_ipm(q, eps)

		elif fit_method=='ipm':

			return self.fit_ipm(q, eps)

		elif fit_method=='preproc-ipm':

			rng = rng_generator(seed)

			return self.fit_preproc_ipm(q, rng, Mm_factor, max_bad_fixup, kappa_eps)

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

	def fit_preproc_ipm(self, q, rng, eps=1e-6, Mm_factor=0.8, max_bad_fixup=3, kappa_eps=1e-6):
		"""Preprocessing phase as described in Portnoy and Koenker, 
		Statistical Science, (1997) 279-300.

		Python implementation of the R code "rq.fit.pfn".

		As was cautioned, use only when the problem size is very large. The
		recommended size of n according to the original author is > 100,000. 
		
		Parameters
		----------
		"""
		X = self.X 
		y = self.y

		n, p = X.shape

		m = int(((p + 1) * n)**(2/3))
		not_optimal = True

		while not_optimal:

			if m < n:	
				s = rng.choice(n, m, replace=False)
			else:
				return fit_coefs(X, y, q, eps)

			xx = X[s]
			yy = y[s]

			xx = np.array(xx, np.double, copy=False, order='F', ndmin=1)
			yy = np.array(yy, np.double, copy=False, order='F', ndmin=1)

			first_coefs = fit_coefs(xx, yy, q, eps)

			xxinv = pinv(xx.T @ xx)
			band = np.sqrt(((X @ xxinv)**2) @ np.ones(p))

			r = y - X @ first_coefs

			M = Mm_factor * m

			lo_q = max(1/n, q - M/(2 * n))
			hi_q = min(q + M/(2 * n), (n - 1)/n)

			kappa = np.quantile(r/np.maximum(kappa_eps, band), [lo_q, hi_q])

			sl = r < band * kappa[0]
			su = r > band * kappa[1]

			bad_fixup = 0

			while(not_optimal & (bad_fixup < max_bad_fixup)):

				xx = X[~su & ~sl]
				yy = y[~su & ~sl]

				if(any(sl)):
					glob_x = X[sl].T @ np.ones(np.sum(sl))
					glob_y = np.sum(y[sl])
					xx = np.vstack([xx, glob_x])
					yy = np.r_[yy, glob_y]
				if(any(su)):
					ghib_x = X[su].T @ np.ones(np.sum(su))
					ghib_y = np.sum(y[su])
					xx = np.vstack([xx, ghib_x])
					yy = np.r_[yy, ghib_y]

				xx = np.array(xx, np.double, copy=False, order='F', ndmin=1)
				yy = np.array(yy, np.double, copy=False, order='F', ndmin=1)

				coefs = fit_coefs(xx, yy, q, eps)

				if np.isnan(coefs[0]):
					coefs = first_coefs

				r = y - X @ coefs
				su_bad = (r < 0) & su
				sl_bad = (r > 0) & sl

				if(any(np.r_[su_bad, sl_bad])):
					if(np.sum(sl_bad) + np.sum(su_bad) > 0.1 * M):
						
						m = 2 * m
						break

					su = su & ~su_bad
					sl = sl & ~sl_bad
					bad_fixup = bad_fixup + 1

				else:
					
					not_optimal = False

		return coefs

