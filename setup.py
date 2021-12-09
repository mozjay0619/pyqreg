#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import pprint
import sys

from setuptools import find_packages, setup
from setuptools.command.install import install
from setuptools.extension import Extension

# Set up the logging environment
logging.basicConfig()
log = logging.getLogger()

# Use Cython if available
try:
    from Cython.Distutils import build_ext

    # Use Cython’s build_ext module which runs cythonize as part of the build process
except ImportError:
    USE_CYTHON = False
else:
    USE_CYTHON = True

cmdclass = {}
ext_modules = []

# Extension options (numpy header file)
include_dirs = []
try:
    import numpy as np

    include_dirs.append(np.get_include())
except ImportError:
    log.critical("Numpy and its headers are required to run setup(). Exiting")
    sys.exit(1)

opts = dict(
    include_dirs=include_dirs,
)
log.debug("opts:\n%s", pprint.pformat(opts))

# Build extension modules
if USE_CYTHON:
    ext_modules += [
        Extension("pyqreg.c.blas_lapack", ["src/pyqreg/c/blas_lapack.pyx"], **opts),
        Extension("pyqreg.c.cluster_cov", ["src/pyqreg/c/cluster_cov.pyx"], **opts),
        Extension("pyqreg.c.fit_coefs", ["src/pyqreg/c/fit_coefs.pyx"], **opts),
        Extension("pyqreg.c.mat_vec_ops", ["src/pyqreg/c/mat_vec_ops.pyx"], **opts),
        Extension(
            "pyqreg.c.matrix_opaccum", ["src/pyqreg/c/matrix_opaccum.pyx"], **opts
        ),
        Extension("pyqreg.c.stats", ["src/pyqreg/c/stats.pyx"], **opts),
    ]
    # First argument is the compilation target location.
    cmdclass.update({"build_ext": build_ext})

else:
    ext_modules += [
        Extension("pyqreg.c.blas_lapack", ["src/pyqreg/c/blas_lapack.c"], **opts),
        Extension("pyqreg.c.cluster_cov", ["src/pyqreg/c/cluster_cov.c"], **opts),
        Extension("pyqreg.c.fit_coefs", ["src/pyqreg/c/fit_coefs.c"], **opts),
        Extension("pyqreg.c.mat_vec_ops", ["src/pyqreg/c/mat_vec_ops.c"], **opts),
        Extension("pyqreg.c.matrix_opaccum", ["src/pyqreg/c/matrix_opaccum.c"], **opts),
        Extension("pyqreg.c.stats", ["src/pyqreg/c/stats.c"], **opts),
    ]

# circleci.py version
VERSION = "v0.3.9"

# circleci version verfication
class VerifyVersionCommand(install):
    """Custom command to verify that the git tag matches our version"""

    description = "verify that the git tag matches our version"

    def run(self):
        tag = os.getenv("CIRCLE_TAG")

        if tag != VERSION:
            info = "Git tag: {0} does not match the version of this app: {1}".format(
                tag, VERSION
            )
            sys.exit(info)


cmdclass.update({"verify": VerifyVersionCommand})


def readme():
    """print long description"""
    with open("README.rst") as f:
        return f.read()


setup(
    name="pyqreg",
    version=VERSION,
    description="Fast implementation of the quantile regression with support for iid, robust, and cluster standard errors.",
    long_description=readme(),
    url="https://github.com/mozjay0619/pyqreg",
    author="Jay Kim",
    author_email="mozjay0619@gmail.com",
    license="DSB 3-clause",
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={"": ["*.pxd", "*.pyx"]},
    python_requires=">=3",
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)

# python3 setup.py build_ext --inplace
