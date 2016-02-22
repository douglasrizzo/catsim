.. catsim documentation master file, created by
   sphinx-quickstart on Sun Jun 14 23:12:17 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. include:: readme_head.rst

Contents
--------

.. toctree::
    :maxdepth: 2

    concepts.rst
    initialization.rst
    selection.rst
    estimation.rst
    stopping.rst
    simulation.rst
    irt.rst
    cat.rst
    plot.rst
    stats.rst
    contributing.rst
    references.rst

.. include:: readme_body.rst

Basic Usage
-----------

1. Have an `item matrix <https://douglasrizzo.github.io/catsim/item_matrix.html>`_;
2. Have a sample, or a number of examinees;
3. Create a `initializer <https://douglasrizzo.github.io/catsim/initialization.html>`_, an item `selector <https://douglasrizzo.github.io/catsim/selection.html>`_, a proficiency `estimator <https://douglasrizzo.github.io/catsim/estimation.html>`_ and a `stopping criterion <https://douglasrizzo.github.io/catsim/stopping.html>`_;
4. Pass them to a `simulator <https://douglasrizzo.github.io/catsim/simulation.html>`_ and start the simulation.
5. Access the simulator's properties to get specifics of the results;
6. `Plot <https://douglasrizzo.github.io/catsim/plot.html>`_ your results.

.. code-block:: python

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

Citing `catsim`
---------------

You can cite the package using the following bibtex entry:

.. code-block:: bibtex

    @misc{catsim,
        author = {Meneghetti, Douglas De Rizzo},
        title = {catsim: Computerized Adaptive Testing Simulator},
        url = {http://douglasrizzo.github.io/catsim/},
        year = {2016},
        doi = {10.5281/zenodo.46259}
    }
