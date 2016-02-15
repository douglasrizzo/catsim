`catsim` -- Computerized Adaptive Testing Simulator
===================================================

.. image:: https://travis-ci.org/douglasrizzo/catsim.svg?branch=master
    :target: https://travis-ci.org/douglasrizzo/catsim:
    :alt: Build Status

.. image:: https://coveralls.io/repos/github/douglasrizzo/catsim/badge.svg?branch=master
    :target: https://coveralls.io/github/douglasrizzo/catsim?branch=master
    :alt: Test Coverage

.. image:: https://badge.fury.io/py/catsim.svg
    :target: https://badge.fury.io/py/catsim
    :alt: Latest Version

.. image:: https://landscape.io/github/douglasrizzo/catsim/master/landscape.svg?style=flat
   :target: https://landscape.io/github/douglasrizzo/catsim/master
   :alt: Code Health

**catsim** is a computerized adaptive testing simulator written in Python 3.4. It allow for the simulation of computerized adaptive tests, selecting different test initialization rules, item selection rules, proficiency reestimation methods and stopping criteria.

Computerized adaptive tests are educational evaluations, usually taken by examinees in a computer or some other digital means, in which the examinee's proficiency is evaluated after the response of each item. The new proficiency is then used to select a new item, closer to the examinee's real proficiency. This method of test application has several advantages compared to the traditional paper-and-pencil method, since high-proficiency examinees are not required to answer all the easy items in a test, answering only the items that actually give some information regarding his or hers true knowledge of the subject at matter. A similar, but inverse effect happens for those examinees of low proficiency level.

*catsim* allows users to simulate the application of a computerized adaptive test, given a sample of examinees, represented by their proficiency levels, and an item bank, represented by their parameters according to some Item Response Theory model.

Basic Usage
-----------

1. Have an `item matrix <item_matrix.rst>`_;
2. Have a sample, or a number of examinees;
3. Create a `initializer <initialization.rst>`_, an item `selector <selection.rst>`_, a proficiency `estimator <estimation.rst>`_ and a `stopping criterion <stopping.rst>`_;
4. Pass them to a `simulator <simulation.rst>`_ and start the simulation.

Optional:

5. Access the simulator's properties to get specifics of the results;
6. `Plot <plot.rst>`_ your results.

 .. code-block:: python
     :linenos:

     from catsim.initialization import RandomInitializer
     from catsim.selection import MaxInfoSelector
     from catsim.reestimation import HillClimbingEstimator
     from catsim.stopping import MaxItemStopper
     from catsim.cat import generate_item_bank
     initializer = RandomInitializer()
     selector = MaxInfoSelector()
     estimator = HillClimbingEstimator()
     stopper = MaxItemStopper(20)
     Simulator(generate_item_bank(100), 10).simulate(initializer, selector, estimator, stopper)
