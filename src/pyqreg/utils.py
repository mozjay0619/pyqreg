import numpy as np
import pandas as pd
from scipy.stats import gamma


def rng_generator(seed=None, n=1):
    """Numpy random Generator generator.

    Parameters
    ----------
    seed : int or None
        Random seed.

    n : int
        The number of Generators to return.

    Reference
    ---------
    https://numpy.org/doc/stable/reference/random/generator.html#numpy.random.Generator
    """
    if n > 1:

        ss = np.random.SeedSequence(seed)
        child_seeds = ss.spawn(n)
        rng_streams = [np.random.default_rng(s) for s in child_seeds]
        return rng_streams

    else:

        return np.random.default_rng(seed)


def get_icc(pdf):
    """
    Calculate the intraclass correlation coefficient
    using one way ANOVA random effects model.

    Parameters
    ----------
    pdf : pandas dataframe
    """
    import statsmodels.formula.api as smf

    # fit one way ANOVA random effects model
    md = smf.mixedlm("y ~ 1", pdf, groups=pdf["G"])
    mdf = md.fit()

    # get the random effects variance
    sig_alpha = mdf.cov_re.values[0][0]

    # get the model variance
    sig_error = mdf.scale

    # compute the intraclass correlation coefficient
    ICC = sig_alpha / (sig_alpha + sig_error)

    return ICC


def generate_clustered_data(G, Gn, cross_cluster_var, rng):

    group_means = rng.normal(0, cross_cluster_var, size=G)

    errors = []
    groups = []
    indicators = []

    for i in range(G):
        errors.append(rng.normal(group_means[i], 10, size=Gn))
        groups.append([i] * Gn)
        indicators.append([rng.choice(2)] * Gn)

    errors = np.hstack(errors)
    groups = np.hstack(groups)
    indicators = np.hstack(indicators)

    n = G * Gn

    samples = rng.gamma(shape=2, scale=10, size=n)

    X = np.zeros([n, 2])
    X[:, 0] = 1
    X[:, 1] = indicators
    y = X[:, 0] * 10 + samples + errors

    return y, X, groups
