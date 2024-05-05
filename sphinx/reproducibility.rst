Reproducibility
***************

**New** in :py:mod:`catsim` 0.18.0!

:py:class:`catsim.simulation.Simulable` objects that use random number generation (mainly initializers and item selectors) can have reproducible outputs by receiving a :py:class:`numpy.random.Generator` instance in the rng keyword argument of their main method.

In the snippet below, all selectors that have random behavior produce the same outputs, when given the same input arguments.

.. code-block:: python
    :caption: Generating reproducible outputs from :py:class:`catsim.simulation.Simulable` objects.

    from catsim.cat import generate_item_bank
    from catsim.selection import RandomesqueSelector, RandomSelector, The54321Selector
    from numpy.random import default_rng

    for _ in range(5):
        print(
            RandomSelector().select(items=generate_item_bank(5000, seed=42), administered_items=[], rng=default_rng(42)),
            The54321Selector(test_size=10).select(
            items=generate_item_bank(5000, seed=42), administered_items=[], rng=default_rng(42), est_theta=0
            ),
            RandomesqueSelector(bin_size=10).select(
            items=generate_item_bank(5000, seed=42), administered_items=[], rng=default_rng(42), est_theta=0
            ),
        )

Simulations can also be entirely reproduced by passing a seed to a :py:class:`catsim.simulation.Simulator` object, which will instantiate a :py:class:`numpy.random.Generator` and carry it over to the :py:class:`catsim.simulation.Simulable` components that use random number generation.

.. plot::
    :include-source: true
    :caption: Generating a reproducible CAT simulation using seeds.

    import matplotlib.pyplot as plt
    from catsim.cat import generate_item_bank
    from catsim.estimation import NumericalSearchEstimator
    from catsim.initialization import RandomInitializer
    from catsim.plot import test_progress
    from catsim.selection import MaxInfoSelector
    from catsim.simulation import Simulator
    from catsim.stopping import MinErrorStopper

    figure, axes = plt.subplots(2, 1, figsize=(10, 12))

    for ax in axes:
        item_bank = generate_item_bank(5000, seed=42)
        s = Simulator(item_bank, examinees=1, seed=42)
        s.simulate(RandomInitializer(), MaxInfoSelector(), NumericalSearchEstimator(), MinErrorStopper(0.2))
        test_progress(ax=ax, simulator=s, index=0, see=True, marker="|")
