import numpy as np
import scipy.stats as stats
from numpy.linalg import pinv
from scipy.stats import norm, t

from .c.blas_lapack import lapack_cholesky_inv
from .c.cluster_cov import psi_function
from .c.fit_coefs import fit_coefs
from .c.matrix_opaccum import matrix_opaccum
from .c.stats import invnormal, normalden
from .utils import rng_generator


class QuantReg:
    def __init__(self, y, X):

        if not X.flags["F_CONTIGUOUS"]:
            X = np.array(X, np.double, copy=False, order="F", ndmin=1)

        if not y.flags["F_CONTIGUOUS"]:
            y = np.array(y, np.double, copy=False, order="F", ndmin=1)

        self.y = y
        self.X = X

    def fit(
        self,
        q,
        cov_type="robust",
        fit_method=None,
        seed=None,
        eps=1e-6,
        Mm_factor=0.8,
        max_bad_fixup=3,
        kappa_eps=1e-6,
        kernel="epa",
        bandwidth="hsheather",
        cov_kwds=dict(),
    ):
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

        kernel : str, kernel to use in the kernel density estimation for the
            asymptotic covariance matrix:

            - epa: Epanechnikov
            - cos: Cosine
            - gau: Gaussian
            - par: Parzene

        bandwidth : str, Bandwidth selection method in kernel density
            estimation for asymptotic covariance estimate (full
            references in QuantReg docstring):

            - hsheather: Hall-Sheather (1988)
            - bofinger: Bofinger (1975)
            - chamberlain: Chamberlain (1994)

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
                self.params = self.fit_preproc_ipm(
                    q, rng, Mm_factor, max_bad_fixup, kappa_eps
                )
            else:
                self.params = self.fit_ipm(q, eps)

        elif fit_method == "ipm":
            self.params = self.fit_ipm(q, eps)

        elif fit_method == "preproc-ipm":
            rng = rng_generator(seed)
            self.params = self.fit_preproc_ipm(
                q, rng, Mm_factor, max_bad_fixup, kappa_eps
            )

        # Estimate covariance matrix

        if cov_type == "cluster":

            if "groups" not in cov_kwds:
                raise ValueError(
                    'You must provide "groups" keyword value in cov_kwds if data is clustered'
                )
            else:
                groups = cov_kwds["groups"]

                if not np.issubdtype(groups.dtype, np.integer):
                    raise TypeError(
                        "groups array must be integer type. Instead it is {}.".format(
                            groups.dtype
                        )
                    )

                groups = groups.astype(np.int32)

            if "kappa_type" not in cov_kwds:
                kappa_type = "silverman"
            else:
                kappa_type = cov_kwds["kappa_type"]

            self.vcov = self.cluster_cov(groups, self.params, q, kappa_type)
            self.bse = np.sqrt(np.diag(self.vcov))

        elif cov_type == "robust":

            self.vcov = self.iid_robust_cov(
                self.params, q, kernel, bandwidth, vcov="robust"
            )
            self.bse = np.sqrt(np.diag(self.vcov))

        elif cov_type == "iid":

            self.vcov = self.iid_robust_cov(
                self.params, q, kernel, bandwidth, vcov="iid"
            )
            self.bse = np.sqrt(np.diag(self.vcov))

        else:

            cov_type_names = ["iid", "robust", "cluster"]
            raise Exception("cov_type must be one of " + ", ".join(cov_type_names))

        # Compute two-sided p-values.
        self.tvalues = self.params / self.bse
        self.pvalues = np.empty(len(self.tvalues))
        for i, z in enumerate(np.abs(self.tvalues)):
            self.pvalues[i] = (
                1 - t.cdf(x=z, loc=0, scale=1, df=n - self.X.shape[1])
            ) * 2

        self.nobs = n

        return self

    def conf_int(self, alpha=0.05):
        """Compute the confidence intervals.

        Parameters
        ----------
        alpha : float
        """
        self.upb = (
            self.params
            + t.ppf(q=1 - alpha / 2.0, df=self.nobs - self.X.shape[1]) * self.bse
        )
        self.lob = (
            self.params
            - t.ppf(q=1 - alpha / 2.0, df=self.nobs - self.X.shape[1]) * self.bse
        )

        return np.squeeze(np.dstack([self.lob, self.upb]))

    def fit_ipm(self, q, eps=1e-6):
        """Estimate coefficients using the interior point method.

        Paramters
        ---------
        q : double
            Quantile value strictly between 0 and 1

        eps : double
            Duality gap stopping criterion
        """
        coefs = _fit_coefs(self.X, self.y, q, eps)

        return coefs

    def fit_preproc_ipm(
        self, q, rng, eps=1e-6, Mm_factor=0.8, max_bad_fixup=3, kappa_eps=1e-6
    ):
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

        m = int(((p + 1) * n) ** (2 / 3))
        not_optimal = True

        while not_optimal:

            if m < n:
                s = rng.choice(n, m, replace=False)
            else:
                return _fit_coefs(X, y, q, eps)

            xx = X[s]
            yy = y[s]

            xx = np.array(xx, np.double, copy=False, order="F", ndmin=1)
            yy = np.array(yy, np.double, copy=False, order="F", ndmin=1)

            first_coefs = _fit_coefs(xx, yy, q, eps)

            xxinv = pinv(xx.T @ xx)
            band = np.sqrt(((X @ xxinv) ** 2) @ np.ones(p))

            r = y - X @ first_coefs

            M = Mm_factor * m

            lo_q = max(1 / n, q - M / (2 * n))
            hi_q = min(q + M / (2 * n), (n - 1) / n)

            kappa = np.quantile(r / np.maximum(kappa_eps, band), [lo_q, hi_q])

            sl = r < band * kappa[0]
            su = r > band * kappa[1]

            bad_fixup = 0

            while not_optimal & (bad_fixup < max_bad_fixup):

                xx = X[~su & ~sl]
                yy = y[~su & ~sl]

                if any(sl):
                    glob_x = X[sl].T @ np.ones(np.sum(sl))
                    # Notes:
                    # 1. The resulting matrix is transposed one more time because np.ones is 1 dimensional.
                    # 2. Summing data with same residual signs will not change the residual sign of the summed.
                    glob_y = np.sum(y[sl])
                    xx = np.vstack([xx, glob_x])
                    yy = np.r_[yy, glob_y]
                if any(su):
                    ghib_x = X[su].T @ np.ones(np.sum(su))
                    ghib_y = np.sum(y[su])
                    xx = np.vstack([xx, ghib_x])
                    yy = np.r_[yy, ghib_y]

                xx = np.array(xx, np.double, copy=False, order="F", ndmin=1)
                yy = np.array(yy, np.double, copy=False, order="F", ndmin=1)

                coefs = _fit_coefs(xx, yy, q, eps)

                r = y - X @ coefs
                su_bad = (r < 0) & su
                sl_bad = (r > 0) & sl

                if any(np.r_[su_bad, sl_bad]):
                    if np.sum(sl_bad) + np.sum(su_bad) > 0.1 * M:

                        m = 2 * m
                        break

                    su = su & ~su_bad
                    sl = sl & ~sl_bad
                    bad_fixup = bad_fixup + 1

                else:

                    not_optimal = False

        return coefs

    def cluster_cov(self, groups, beta, q, kappa_type="silverman"):
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

        sort_args = groups.argsort(kind="mergesort")
        self.X = self.X[sort_args]
        self.y = self.y[sort_args]
        groups = groups[sort_args]

        self.X = np.array(self.X, np.double, copy=False, order="F", ndmin=1)
        self.y = np.array(self.y, np.double, copy=False, order="F", ndmin=1)
        groups = np.array(groups, np.int32, copy=False, order="F", ndmin=1)

        G = len(np.unique(groups))

        # Compute residuals
        resid = self.y - self.X @ beta

        # Compute A

        # psi
        psi_resid = psi_function(resid, theta)

        A = matrix_opaccum(self.X, groups, psi_resid, G)

        # Compute B

        # fmt: off
        # h_nG
        h_nG = (invnormal(0.975)**(2/3)) * \
        ((1.5 * ((normalden(invnormal(theta)))**2) / (2 * ((invnormal(theta))**2) + 1))**(1/3)) * \
        (n)**(-1/3)
        # fmt: on

        # kappa
        if kappa_type == "median":
            k = np.median(np.abs(resid))
        elif kappa_type == "silverman":
            k = min(
                np.std(resid),
                (np.percentile(resid, 75) - np.percentile(resid, 25)) / 1.34,
            )
        else:
            raise ValueError(
                "Incorrect kappa_type {}. Please choose between median and silverman".format(
                    kappa_type
                )
            )

        # c^_G
        chat_G = k * (invnormal(theta + h_nG) - invnormal(theta - h_nG))

        # B weights
        dens = np.sqrt((np.abs(resid) < chat_G).astype(np.float64) / (2 * chat_G))
        _groups = np.arange(len(groups)).astype(np.int32)

        B = matrix_opaccum(self.X, _groups, dens, n)

        # Compute Binv A Binv
        B = np.array(B, np.double, copy=False, order="F", ndmin=1)
        lapack_cholesky_inv(B)

        return B @ A @ B

    def iid_robust_cov(self, beta, q, kernel, bandwidth, vcov="robust"):
        """Covariance matrix estimation for iid data as written in the statsmodels:

        https://www.statsmodels.org/stable/_modules/statsmodels/regression/quantile_regression.html#QuantReg

        Parameters
        ----------
        kernel : str, kernel to use in the kernel density estimation for the
            asymptotic covariance matrix:

            - epa: Epanechnikov
            - cos: Cosine
            - gau: Gaussian
            - par: Parzene

        bandwidth : str, Bandwidth selection method in kernel density
            estimation for asymptotic covariance estimate (full
            references in QuantReg docstring):

            - hsheather: Hall-Sheather (1988)
            - bofinger: Bofinger (1975)
            - chamberlain: Chamberlain (1994)
        """
        kern_names = ["biw", "cos", "epa", "gau", "par"]
        if kernel not in kern_names:
            raise Exception("kernel must be one of " + ", ".join(kern_names))
        else:
            kernel = kernels[kernel]

        if bandwidth == "hsheather":
            bandwidth = hall_sheather
        elif bandwidth == "bofinger":
            bandwidth = bofinger
        elif bandwidth == "chamberlain":
            bandwidth = chamberlain
        else:
            raise Exception(
                "bandwidth must be in 'hsheather', 'bofinger', 'chamberlain'"
            )

        # Compute residuals
        resid = self.y - self.X @ beta

        nobs = len(self.X)

        iqre = stats.scoreatpercentile(resid, 75) - stats.scoreatpercentile(resid, 25)
        h = bandwidth(nobs, q)
        h = min(np.std(self.y), iqre / 1.34) * (norm.ppf(q + h) - norm.ppf(q - h))

        fhat0 = 1.0 / (nobs * h) * np.sum(kernel(resid / h))

        if vcov == "robust":
            d = np.where(resid > 0, (q / fhat0) ** 2, ((1 - q) / fhat0) ** 2)
            xtxi = pinv(np.dot(self.X.T, self.X))
            xtdx = np.dot(self.X.T * d[np.newaxis, :], self.X)
            vcov = xtxi @ xtdx @ xtxi
        elif vcov == "iid":
            vcov = (1.0 / fhat0) ** 2 * q * (1 - q) * pinv(np.dot(self.X.T, self.X))

        return vcov


# fmt: off
# From https://www.statsmodels.org/stable/_modules/statsmodels/regression/quantile_regression.html#QuantReg.
def _parzen(u):
    z = np.where(np.abs(u) <= .5, 4./3 - 8. * u**2 + 8. * np.abs(u)**3,
                 8. * (1 - np.abs(u))**3 / 3.)
    z[np.abs(u) > 1] = 0
    return z


kernels = {}
kernels['biw'] = lambda u: 15. / 16 * (1 - u**2)**2 * np.where(np.abs(u) <= 1, 1, 0)
kernels['cos'] = lambda u: np.where(np.abs(u) <= .5, 1 + np.cos(2 * np.pi * u), 0)
kernels['epa'] = lambda u: 3. / 4 * (1-u**2) * np.where(np.abs(u) <= 1, 1, 0)
kernels['par'] = _parzen


def hall_sheather(n, q, alpha=.05):
    z = norm.ppf(q)
    num = 1.5 * norm.pdf(z)**2.
    den = 2. * z**2. + 1.
    h = n**(-1. / 3) * norm.ppf(1. - alpha / 2.)**(2./3) * (num / den)**(1./3)
    return h


def bofinger(n, q):
    num = 9. / 2 * norm.pdf(2 * norm.ppf(q))**4
    den = (2 * norm.ppf(q)**2 + 1)**2
    h = n**(-1. / 5) * (num / den)**(1. / 5)
    return h


def chamberlain(n, q, alpha=.05):
    return norm.ppf(1 - alpha / 2) * np.sqrt(q*(1 - q) / n)
# fmt: on


def _fit_coefs(X, y, q, eps):
    """In cases of convergence issues, we increase the duality gap
    tolerance.
    """
    coefs = fit_coefs(X, y, q, eps)

    while any(np.isnan(coefs)):

        eps *= 5.0
        coefs = fit_coefs(X, y, q, eps)

    return coefs
