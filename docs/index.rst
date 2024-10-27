.. Computational Physics Final Project documentation master file, created by
   sphinx-quickstart on Tue Dec  5 20:39:47 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Truncated Guassian Mixtures
===============================================================
Fit data to mixtures of *truncated* multivariate gaussians

.. math::
   p(x) = \sum_k w_k\  \phi_{[{\bf a}, {\bf b}]}({\bf x} | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)

where :math:`{\bf a}` and :math:`{\bf b}` are the bounding corners of the hypercube we would like to truncate the gaussian in.
This package supports

#. Full, Diagonal and Block diagonal covariances for each component
#. Inteface to allow one to perform the fit in some latent space given by a user-defined transformation, and carry along any target labels
#. Sampling and pdf evaluation from the resultant fit

.. include:: advantages.md
   :parser: myst_parser.sphinx_

We implement the Expectation Maximization algorithm as outlined in :cite:`asad` and use the better convergence properties of the Expectation-Conjugate Gradient method (outlined in :cite:`conjugate`) in the case that the covariances of the gaussians are diagonal.


.. bibliography:: refs.bib

Quick Start
===============================================================
You can install this library using

.. code-block:: bash

   pip install truncatedgaussianmixtures

You can then import truncatedgaussianmixtures. At the first import it may take a while, since this will create a local
julia installation. This will only happen once at the first ever import. 

The main function to use is :code:`fit_gmm`. The following is a simple example use case. 

.. code-block:: python

   import pandas as pd
   from truncatedgaussianmixtures import fit_gmm

   # Generate some data
   df = pd.DataFrame(np.random.randn(80_000, 2), columns=["x", "y"])
   
   # Truncated it to the unit square
   cond = (df['x'] < 1) & (df['x'] > 0)  
   cond &= (df['y'] < 1) & (df['y'] > 0)
   df = df[cond]

   # Fit it a truncated gaussian mixture model to it
   gmm = fit_gmm(data = df,      # data to fit to
                 N    = 1,       # Number of components of the mixture model
                 a    = [0,0],   # lower corner of the truncation
                 b    = [1,1],   # upper corner of the truncation
                 cov  = "diag"   # covariance structure: any of ("diag", "full")
          )

   # Sample from the gmm
   df_fit = gmm.sample(len(df));

   # Evaluate it at different points
   gmm.pdf(np.array([0,0]))

.. toctree::
   :maxdepth: 0
   :caption: Contents:
      FullTruncatedGaussianTests.ipynb

.. modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`