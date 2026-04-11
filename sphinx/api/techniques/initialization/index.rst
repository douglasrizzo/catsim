Initialization Techniques
#########################

Initialization techniques choose the initial ability estimate
:math:`\hat{\theta}_0` before an adaptive test has administered any items. The
choice of initializer affects the first item selection and can be useful for
controlled simulations, reproducible baselines, or stochastic starting points.

Implemented
***********

.. toctree::
   :maxdepth: 1

   fixed-point-initializer
   random-initializer

Class Hierarchy
***************

.. inheritance-diagram:: catsim.initialization.BaseInitializer catsim.initialization.FixedPointInitializer catsim.initialization.RandomInitializer
   :parts: 1
   :top-classes: catsim.initialization.BaseInitializer
