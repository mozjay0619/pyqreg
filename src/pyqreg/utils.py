import numpy as np


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