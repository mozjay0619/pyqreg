from .c.fit_coefs import fit_coefs
from .c.blas_lapack import lapack_cholesky_inv
from .c.stats import normalden, invnormal
from .c.cluster_cov import psi_function
from .c.matrix_opaccum import matrix_opaccum

from .utils import rng_generator

import numpy as np
from numpy.linalg import pinv


class QuantReg():

	def __init__(self, y, X):

		self.X = np.array(X, np.double, copy=False, order='F', ndmin=1)
		self.y = np.array(y, np.double, copy=False, order='F', ndmin=1)

	def fit(self, q, cov_type='robust', fit_method=None, seed=None, eps=1e-6, 
			Mm_factor=0.8, max_bad_fixup=3, kappa_eps=1e-6, cov_kwds=dict()):
		"""Solve by interior point method (Mehrotra's predictor corrector
		algorithm). If n >= 100,000, it will use preprocessing step following
		Portnoy and Koenker (1997).

		Parameters
		----------
		q : double
			Quantile value strictly between 0 and 1

		fit_method : str or None. Default None.
			Coefficient estimation method. 

			- None : uses ipm if n < 100000, else, preproc-ipm.
			- ipm : interior point method.
			- preproc-ipm : interior point method with preprocessing.

		cov_type : str. Default 'robust'.
			Type of covariance estimator to use. Available types are
			``iid`` for iid errors, ``robust`` for heteroskedastic errors,
			and ``cluster`` for clustered errors.

		seed : int or None
			Random seed to use if preproc-ipm is used for subsampling.

		cov_kwds : dict 
			Additional keywords used in the covariance specification.

			- groups : ndarray int type
				Integer-valued index of clusters or groups. Required if using
				the ``cluster`` cov_type.
			- kappa_type : str. Default 'silverman'.
				The scaling factor for the bandwidth. Available rule of thumbs
				type are ``silverman`` and ``median``.

		"""

		n = len(self.X)

		if fit_method is None:

			if n >= 100000:
				rng = rng_generator(seed)
				self.params = self.fit_preproc_ipm(q, rng, Mm_factor, max_bad_fixup, kappa_eps)
			else:
				self.params = self.fit_ipm(q, eps)

		elif fit_method=='ipm':
			self.params = self.fit_ipm(q, eps)

		elif fit_method=='preproc-ipm':
			rng = rng_generator(seed)
			self.params = self.fit_preproc_ipm(q, rng, Mm_factor, max_bad_fixup, kappa_eps)

		# Estimate covariance matrix

		if cov_type=='cluster':

			if 'groups' not in cov_kwds:
				raise ValueError('You must provide "groups" keyword value in cov_kwds if data is clustered')
			else:
				groups = cov_kwds['groups']

				if not np.issubdtype(groups.dtype, np.integer):
					raise TypeError('groups array must be integer type. Instead it is {}.'.format(groups.dtype))

				groups = groups.astype(np.int32)

			if 'kappa_type' not in cov_kwds:
				kappa_type = 'silverman'
			else:
				kappa_type = cov_kwds['kappa_type']

			self.vcov = self.cluster_cov(groups, self.params, q, kappa_type)
			self.bse = np.sqrt(np.diag(self.vcov))

		elif cov_type=='robust':

			if 'kappa_type' not in cov_kwds:
				kappa_type = 'silverman'
			else:
				kappa_type = cov_kwds['kappa_type']

			groups = np.arange(n).astype(np.int32)

			self.vcov = self.cluster_cov(groups, self.params, q, kappa_type)
			self.bse = np.sqrt(np.diag(self.vcov))

		elif cov_type=='iid':

			raise NotImplementedError


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

	def cluster_cov(self, groups, beta, q, kappa_type='silverman'):
		"""Covariance matrix estimator as proposed by Parente and Silva (2013).

		Translated from Stata code of qreg2. 

		Parameters
		----------
		groups : ndarray
			The group index array.

		beta : ndarray
			The estimated parameter values.

		q : double
			The quantile strictly between 0 and 1.

		kappa_type : str. Default 'silverman'.
			The scaling factor for the bandwidth. Available rule of thumbs
			type are ``silverman`` and ``median``.
		"""
		theta = q
		n = len(self.X)

		# print(groups)
		# print(np.mean(self.X))
		# print(np.mean(self.y))


		sort_args = groups.argsort(kind='mergesort')
		self.X = self.X[sort_args]
		self.y = self.y[sort_args]
		groups = groups[sort_args]

		self.X = np.array(self.X, np.double, copy=False, order='F', ndmin=1)
		self.y = np.array(self.y, np.double, copy=False, order='F', ndmin=1)
		groups = np.array(groups, np.int32, copy=False, order='F', ndmin=1)

		G = len(np.unique(groups))

		# Compute residuals
		resid = self.y - self.X@beta

		# Compute A

		# psi
		psi_resid = psi_function(resid, theta)

		A = matrix_opaccum(self.X, groups, psi_resid, G)

		# Compute B

		# h_nG
		h_nG = (invnormal(0.975)**(2/3)) * \
		((1.5 * ((normalden(invnormal(theta)))**2) / (2 * ((invnormal(theta))**2) + 1))**(1/3)) * \
		(n)**(-1/3)

		# kappa
		if kappa_type=='median':
			k = np.median(np.abs(resid))
		elif kappa_type=='silverman':
			k = min(np.std(resid),(np.percentile(resid, 75)-np.percentile(resid, 25))/1.34)

		# c^_G
		chat_G = k * (invnormal(theta + h_nG) - invnormal(theta - h_nG))

		# B weights
		dens = np.sqrt(
			(np.abs(resid) < chat_G).astype(np.float64) / (2 * chat_G)
		)
		_groups = np.arange(len(groups)).astype(np.int32)

		B = matrix_opaccum(self.X, _groups, dens, n)

		# Compute Binv A Binv
		B = np.array(B, np.double, copy=False, order='F', ndmin=1)
		lapack_cholesky_inv(B)

		return B@A@B

		
