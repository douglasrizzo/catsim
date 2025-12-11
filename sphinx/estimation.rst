Estimation Methods -- :mod:`catsim.estimation`
**********************************************

Estimators are the objects responsible for estimating examinees'
ability values, given a dichotomous (binary) response vector and an array of
the items answered by the examinee. In the domain of IRT, there are two main
types of methods for estimating :math:`\hat{\theta}`: Bayesian methods and
maximum-likelihood methods.

Maximum-likelihood methods choose the :math:`\hat{\theta}` value that maximizes
the likelihood (see :py:func:`catsim.irt.log_likelihood`) of an examinee having
a certain response vector, given the corresponding item parameters.

Bayesian methods use *a priori* information (usually assuming ability and
parameter distributions) to make new estimations. New estimations are then
used to refine assumptions about the parameter distributions, improving
future estimations.

All implemented classes in this module inherit from a base abstract class
:py:class:`BaseEstimator`. :py:class:`Simulator` allows that a custom estimator be
used during the simulation, as long as it also inherits from
:py:class:`BaseEstimator`.

.. inheritance-diagram:: catsim.estimation
   :parts: 1
   :top-classes: catsim._base.Simulable

:mod:`catsim` implements a few types of maximum-likelihood estimators.

.. automodule:: catsim.estimation
    :members:
    :show-inheritance:

Comparison between estimators
-----------------------------

The chart below displays the execution times of the same simulation (100 examinees, an item bank of 300 questions and 20 items per test) using different :math:`\hat{\theta}` estimation methods.

.. plot::
    :caption: Execution times of the same simulation (100 examinees, an item bank of 300 questions and 20 items per test) using different :math:`\hat{\theta}` estimation methods.

    import numpy as np
    import matplotlib.pyplot as plt

    from catsim.simulation import Simulator
    from catsim.initialization import FixedPointInitializer
    from catsim.selection import MaxInfoSelector
    from catsim.estimation import NumericalSearchEstimator
    from catsim.stopping import MinErrorStopper
    from catsim import ItemBank

    items = ItemBank.generate_item_bank(300)
    examinees = 100
    test_size = 20
    rng = np.random.default_rng()
    thetas = rng.normal(0, 1, examinees)
    sim_times = {}
    for m in NumericalSearchEstimator.available_methods():
        simulator = Simulator(items, thetas)
        simulator.simulate(
            FixedPointInitializer(0),
            MaxInfoSelector(),
            NumericalSearchEstimator(method=m),
            MinErrorStopper(0.4, max_items=test_size),
            verbose=True
        )
        sim_times[m] = simulator.duration

    plt.figure(figsize=(10,5))
    plt.bar(range(len(sim_times)), list(sim_times.values()), align='center')
    plt.xticks(range(len(sim_times)), list(sim_times.keys()))
    plt.show()


The charts below show the :math:`\hat{\theta}` found by the different estimation methods, given three dichotomous response vectors with different numbers of correct answers. All estimators reach comparable results, maximizing the log-likelihood function. The main difference among them is in the number of calls to :py:func:`catsim.irt.log_likelihood`. Methods that require less calls to maximize the function are usually more time efficient.


.. plot::
    :caption: :math:`\hat{\theta}` found by the different estimation methods, given three dichotomous response vectors with different numbers of correct answers. All estimators reach comparable results, maximizing the log-likelihood function.

    import numpy as np
    import matplotlib.pyplot as plt
    from catsim import ItemBank, irt
    from catsim.estimation import NumericalSearchEstimator

    test_size = 20
    item_bank = ItemBank.generate_item_bank(20)
    # Sort by difficulty ascending
    sorted_indices = item_bank.difficulty.argsort()
    items = item_bank.items[sorted_indices]
    sorted_bank = ItemBank(items)

    r0 = [True] * 7 + [False] * 13
    r1 = [True] * 10 + [False] * 10
    r2 = [True] * 15 + [False] * 5
    response_vectors = [r0, r1, r2]
    thetas = np.arange(-6.,6.,.1)

    fig, axes = plt.subplots(len(NumericalSearchEstimator.available_methods()), 1, figsize=(8,35))

    for idx, estimator in enumerate([
            NumericalSearchEstimator(method=m) for m in NumericalSearchEstimator.available_methods()
        ]):
        ax = axes[idx]
        for response_vector in response_vectors:
            ll_line = [irt.log_likelihood(theta, response_vector, sorted_bank.items) for theta in thetas]
            max_ll = estimator.estimate(item_bank=sorted_bank, administered_items=list(range(20)),
                                        response_vector=response_vector, est_theta=0)
            best_theta = irt.log_likelihood(max_ll, response_vector, sorted_bank.items)
            ax.plot(thetas, ll_line)
            label_text = f"{sum(response_vector)} correct, " + r"$\hat{\theta} \approx $" + f"{round(max_ll, 5)}"
            ax.plot(max_ll, best_theta, 'o', label=label_text)
            ax.set_xlabel(r'$\theta$', size=16)
            ax.set_ylabel(r'$\log L(\theta)$', size=16)
            ax.set_title(f"{estimator.method} ({round(estimator.avg_evaluations)} avg. evals)")
            ax.legend(loc='best')

    plt.tight_layout()
    plt.show()
