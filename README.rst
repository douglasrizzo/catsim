`catsim` - Computerized Adaptive Testing Simulator
##################################################

.. image:: https://travis-ci.org/douglasrizzo/catsim.svg?branch=master
    :target: https://travis-ci.org/douglasrizzo/catsim
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

**catsim** is a Python package for computerized adaptive testing (CAT) simulations. It provides multiple methods for

- `test initialization <https://douglasrizzo.github.io/catsim/initialization.html>`_ (selecting the initial proficiency of the examinees)
- `item selection <https://douglasrizzo.github.io/catsim/selection.html>`_
- `proficiency estimation <https://douglasrizzo.github.io/catsim/estimation.html>`_
- `test stopping <https://douglasrizzo.github.io/catsim/stopping.html>`_

These methods can either be used in a standalone fashion `[example] <https://douglasrizzo.github.io/catsim/introduction.html#autonomous-usage>`_ to power other software or be used with *catsim* to simulate the application of computerized adaptive tests `[example] <https://douglasrizzo.github.io/catsim/introduction.html#running-simulations>`_, given a sample of examinees, represented by their proficiency levels, and an item bank, represented by their parameters according to some `logistic Item Response Theory model <https://douglasrizzo.github.io/catsim/introduction.html#item-response-theory-models>`_.

What's a CAT
============

Computerized adaptive tests are educational evaluations, usually taken by examinees in a computer or some other digital means, in which the examinee's proficiency is evaluated after the response of each item. The new proficiency is then used to select a new item, closer to the examinee's real proficiency. This method of test application has several advantages compared to the traditional paper-and-pencil method or even linear tests applied electronically, since high-proficiency examinees are not required to answer all the easy items in a test, answering only the items that actually give some information regarding his or hers true knowledge of the subject at matter. A similar, but inverse effect happens for those examinees of low proficiency level.

More information is available `in the docs <https://douglasrizzo.github.io/catsim/introduction.html>`_ and over at `Wikipedia <https://en.wikipedia.org/wiki/Computerized_adaptive_testing>`_.

Installation
============

Install it using ``pip install catsim``.

Dependencies
============

All dependencies are listed on setup.py and should be installed automatically.

To run the tests, you'll need to install the testing requirements ``pip install catsim[testing]``.

To generate the documentation, Sphinx and its dependencies are needed.

Compatibility
=============

Since the beginning, *catsim* has only been compatible with Python 3.4 upwards.

Important links
===============

- Official source code repo: https://github.com/douglasrizzo/catsim
- HTML documentation (stable release): https://douglasrizzo.github.io/catsim
- Issue tracker: https://github.com/douglasrizzo/catsim/issues