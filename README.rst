.. -*- mode: rst -*-

|CICD| |VERSION| |LICENCE| 

.. |CICD| image:: https://img.shields.io/circleci/build/github/mozjay0619/pyqreg?label=circleci&token=93f5878e444e751d779f2954eb5fce9bc9ab5b3e   
	:alt: CircleCI
.. |LICENCE| image:: https://img.shields.io/pypi/l/pyqreg?label=liscence   
	:alt: PyPI - License
.. |VERSION| image:: https://img.shields.io/pypi/v/pyqreg?color=green&label=pypi%20version   
	:alt: PyPI

Pyqreg
======

Pyqreg implements the quantile regression algorithm with fast estimation method using the interior point method following the preprocessing procedure in Portnoy and Koenker (1997). It provides methods for estimating the asymptotic covariance matrix for i.i.d and heteroskedastic errors, as well as clustered errors following Parente and Silva (2013).

Reference
---------
* https://github.com/pkofod/QuantileRegressions.jl/blob/master/src/InteriorPoint.jl
* https://people.eecs.berkeley.edu/~jordan/sail/readings/portnoy-koenker.pdf
* http://people.exeter.ac.uk/RePEc/dpapers/DP1305.pdf

Install
-------

pyqreg requires

* Python >= 3.6


You can install the latest release with:

.. code:: python

	pip3 install pyqreg
