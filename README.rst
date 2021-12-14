.. -*- mode: rst -*-

|CICD| |VERSION| |LICENCE| |PythonVersion| |Black|

.. |CICD| image:: https://img.shields.io/circleci/build/github/mozjay0619/pyqreg?label=circleci&token=93f5878e444e751d779f2954eb5fce9bc9ab5b3e   
	:alt: CircleCI
.. |LICENCE| image:: https://img.shields.io/pypi/l/pyqreg?label=liscence   
	:alt: PyPI - License
.. |VERSION| image:: https://img.shields.io/pypi/v/pyqreg?color=success&label=pypi%20version
	:alt: PyPI
.. |PythonVersion| image:: https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue
.. _PythonVersion: https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue
.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
.. _Black: https://github.com/psf/black

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

pyqreg pre-install requirements are:

* Python >= 3.6
* Numpy

You can install the latest release with:

.. code:: python

	pip3 install pyqreg

Example
-------

We replicate the study in `statsmodels quantile regression <https://www.statsmodels.org/dev/examples/notebooks/generated/quantile_regression.html>`_ that looks at the relationship between income and expenditures on food for a sample of working class Belgian households in 1857 (the Engel data) using pyqreg.

.. code:: python

	import statsmodels.api as sm

	data = sm.datasets.engel.load_pandas().data
	data.head()

.. figure:: https://github.com/mozjay0619/pyqreg/blob/master/media/img1.png

Fit the quantile regression model using the statsmodels.

.. code:: python
	
	import statsmodels.formula.api as smf
	mod = smf.quantreg("foodexp ~ income", data)
	res = mod.fit(q=0.5)
	res.summary()

.. figure:: https://github.com/mozjay0619/pyqreg/blob/master/media/img6.png

Instead of using statsmodels quantile regression, we use that of pyqreg. Observe that the results are identical to that of statsmodels.

.. code:: python

	from pyqreg import quantreg
	mod = quantreg("foodexp ~ income", data)
	res = mod.fit(q=0.5)
	res.summary()

.. figure:: https://github.com/mozjay0619/pyqreg/blob/master/media/img2.png

We will also replicate the visualizations of many quantiles that are plotted against the OLS fit. We will use the exact codes used in the statsmodels example that places the results in a Pandas DataFrame.

.. figure:: https://github.com/mozjay0619/pyqreg/blob/master/media/img3.png

.. figure:: https://github.com/mozjay0619/pyqreg/blob/master/media/img4.png



Speed comparisons
-----------------

.. figure:: https://github.com/mozjay0619/pyqreg/blob/master/media/img5.png