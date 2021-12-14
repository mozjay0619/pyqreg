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

Pyqreg implements the quantile regression algorithm with fast estimation method using the linear programming interior point method following the preprocessing procedure in Portnoy and Koenker (1997). It provides methods for estimating the asymptotic covariance matrix for i.i.d and heteroskedastic errors, as well as clustered errors following Parente and Silva (2013).

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

Let's first fit the quantile regression model using the statsmodels.

.. code:: python
	
	import statsmodels.formula.api as smf
	mod = smf.quantreg("foodexp ~ income", data)
	res = mod.fit(q=0.5)
	res.summary()

.. figure:: https://github.com/mozjay0619/pyqreg/blob/master/media/img6.png

Now instead of using statsmodels quantile regression, we use that of pyqreg. Observe that the results are identical to that of statsmodels.

.. code:: python

	from pyqreg import quantreg
	mod = quantreg("foodexp ~ income", data)
	res = mod.fit(q=0.5)
	res.summary()

.. figure:: https://github.com/mozjay0619/pyqreg/blob/master/media/img2.png

The remaining parts of the study can be found in this notebook.


Speed comparisons
-----------------

Despite the identical regression results and similar APIs, pyqreg uses completely different optimization algorithm under the hood, making it anywhere between 10 to 30 times faster than the statsmodels quantile regression, depending on the data size, error distribution and quantile.

.. figure:: https://github.com/mozjay0619/pyqreg/blob/master/media/img5.png

The above plots the median time to convergence for each data size, which shows a large and growing difference in absolute speed with data size. 

Cluster robust standard error
-----------------------------

Unlike the statsmodels quantile regression, which only supports iid and heteroskedasticity robust standard errors, pyqreg also supports the cluster robust standard error estimation.

.. code:: python

	from pyqreg.utils import generate_clustered_data, rng_generator

	pyqreg_params = []
	pyqreg_ses = []

	statsmodels_params = []
	statsmodels_ses = []

	for i in range(500):
	    
	    rng = rng_generator(i)
	    
	    # Generate fake clustered data, with 150 groups,
	    # 500 data points in each group, using 15
	    # as cross cluster variance (normal distribution).
	    y, X, groups = generate_clustered_data(150, 500, 15, rng)
	    
	    from pyqreg import QuantReg
	    mod = QuantReg(y, X)
	    res = mod.fit(0.5, cov_type='cluster', cov_kwds={'groups': groups})
	    
	    pyqreg_params.append(res.params)
	    pyqreg_ses.append(res.bse)
	    
	    from statsmodels.regression.quantile_regression import QuantReg
	    mod = QuantReg(y, X)
	    res = mod.fit(0.5)
	    
	    statsmodels_params.append(res.params)
	    statsmodels_ses.append(res.bse)

The above code runs a simulation study, using fake generated clustered data. We will take a look at the simulated standard deviation of betas, and the two models' estimated standard errors.

.. code:: python

	print(np.asarray(statsmodels_params).std(axis=0))
	print(np.asarray(pyqreg_params).std(axis=0))

.. code:: 
	
	[1.81944934 2.52755859]
	[1.81947597 2.52758232]

As expected, the standard deviation of the estimated betas of the two models are very similar to each other. However, we see a huge divergence in the estimations in standard errors. The heteroskedasticity robust standard error completely underestimates the standard deviation, where as pyqreg produces an estimate that is asymptotically much more accurate:

.. code:: python

	print(np.asarray(statsmodels_ses).mean(axis=0))
	print(np.asarray(pyqreg_ses).mean(axis=0))

.. code:: 
	
	[0.14290666 0.20251073]
	[1.75910926 2.49862904]

But of course, if we run the same simulation with 0 cross cluster variance, both models' standard errors are consistent, which makes sense since all the off-diagonal terms in the covariance matrix will be close to 0, making the block diagonal matrix look more like the heteroskedasticity robust (or even iid) covariance diagonal matrix:

.. code:: python

	print(np.asarray(statsmodels_params).std(axis=0))
	print(np.asarray(pyqreg_params).std(axis=0))

.. code:: 
	
	[0.09985114 0.14226425]
	[0.09984286 0.14225007]

This time, both models produce the accurate standard errors:

.. code:: python

	print(np.asarray(statsmodels_ses).mean(axis=0))
	print(np.asarray(pyqreg_ses).mean(axis=0))

.. code:: 
	
	[0.103299   0.14637724]
	[0.10282833 0.14554498]
