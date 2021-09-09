.. -*- mode: rst -*-

|CICD| |VERSION| |LICENCE| |PythonVersion|

.. |CICD| image:: https://img.shields.io/circleci/build/github/mozjay0619/pyqreg?label=circleci&token=93f5878e444e751d779f2954eb5fce9bc9ab5b3e   
	:alt: CircleCI
.. |LICENCE| image:: https://img.shields.io/pypi/l/pyqreg?label=liscence   
	:alt: PyPI - License
.. |VERSION| image:: https://img.shields.io/pypi/v/pyqreg?color=success&label=pypi%20version
	:alt: PyPI
.. |PythonVersion| image:: https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue
.. _PythonVersion: https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue

Pyqreg
======

Pyqreg implements the quantile regression algorithm with fast estimation method using the interior point method following the preprocessing procedure in Portnoy and Koenker (1997). It provides methods for estimating the asymptotic covariance matrix for i.i.d and heteroskedastic errors, as well as clustered errors following Parente and Silva (2013).

References
----------
* Stephen Portnoy. Roger Koenker. "The Gaussian hare and the Laplacian tortoise: computability of squared-error versus absolute-error estimators." Statist. Sci. 12 (4) 279 - 300 (1997). 
* Koenker, R., Ng, P. A Frisch-Newton Algorithm for Sparse Quantile Regression. Acta Mathematicae Applicatae Sinica, English Series 21, 225–236 (2005). 
* Parente, Paulo and Santos Silva, João, (2013), Quantile regression with clustered data, No 1305, Discussion Papers, University of Exeter, Department of Economics. 

Install
-------

pyqreg requires

* Python >= 3.6
* Numpy

You can install the latest release with:

.. code:: python

	pip3 install pyqreg

