`catsim` -- Computerized Adaptive Testing Simulator
###################################################

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

.. image:: https://requires.io/github/douglasrizzo/catsim/requirements.svg?branch=master
    :target: https://requires.io/github/douglasrizzo/catsim/requirements/?branch=master
    :alt: Requirements Status


.. image:: https://zenodo.org/badge/doi/10.5281/zenodo.46420.svg
    :target: http://dx.doi.org/10.5281/zenodo.46420
    :alt: Digital Object Identifier

Quick start
***********

**catsim** is a computerized adaptive testing simulator written in Python 3.4. It allow for the simulation of computerized adaptive tests, selecting different test initialization rules, item selection rules, proficiency reestimation methods and stopping criteria.

Computerized adaptive tests are educational evaluations, usually taken by examinees in a computer or some other digital means, in which the examinee's proficiency is evaluated after the response of each item. The new proficiency is then used to select a new item, closer to the examinee's real proficiency. This method of test application has several advantages compared to the traditional paper-and-pencil method, since high-proficiency examinees are not required to answer all the easy items in a test, answering only the items that actually give some information regarding his or hers true knowledge of the subject at matter. A similar, but inverse effect happens for those examinees of low proficiency level.

*catsim* allows users to simulate the application of a computerized adaptive test, given a sample of examinees, represented by their proficiency levels, and an item bank, represented by their parameters according to some Item Response Theory model.

Installation
============

Install it using ``pip install catsim``.

Important links
===============

- Official source code repo: https://github.com/douglasrizzo/catsim
- HTML documentation (stable release): http://douglasrizzo.github.io/catsim
- Issue tracker: https://github.com/douglasrizzo/catsim/issues

Dependencies
============

`catsim` depends on the latest versions of NumPy, SciPy, Matplotlib and scikit-learn,
which are automatically installed from `pip`.

To run the tests, you'll need `nose`.

To generate the documentation, Sphinx and its dependencies are needed.
