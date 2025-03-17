Welcome to GeoFPTax's documentation!
===================================

While this package is focused on the JAX implementation of the `Geo-FPT <https://arxiv.org/pdf/2303.15510v1>`_ code, it also includes 
the original ``C`` `implementation <https://github.com/serginovell/Geo-FPT/tree/main>`_ for testing purposes. The caveat being that the
JAX implementation resorts to 2 1D trapezoidal rule integrations instead of the quadrature used in the C code, this makes the result
slightly dependent on the number of points used to evaluate the kernel before the integrations. From the tests, 50 points seem to 
yield a good tradeoff between speed and acuracy, but if only the monopole is used, 10 points seem to be enough. 


Building
========

To install, do 

``python -m pip install git+https://github.com/dforero0896/geofptax.git --U``


By default, the package is installed without the ``C`` extension, to do so (if you want to test or use the C module instead) do

``GEOFPTAX_CEXT=1 python -m pip install git+https://github.com/dforero0896/geofptax.git --U``



.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api
   plots
   tests

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`