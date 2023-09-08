Estimation Methods -- :mod:`catsim.estimation`
**********************************************

Estimators are the objects responsible for estimating of examinees
ability values, given a dichotomous (binary) response vector and an array of
the items answered by the examinee. In the domain of IRT, there are two main
types of ways of estimating :math:`\hat{\theta}`: and these are the Bayesian
methods and maximum-likelihood ones.

Maximum-likelihood methods choose the :math:`\hat{\theta}` value that maximizes
the likelihood (see :py:func:`catsim.irt.log_likelihood`) of an examinee having
a certain response vector, given the corresponding item parameters.

Bayesian methods used *a priori* information (usually assuming ability and
parameter distributions) to make new estimations. The knowledge of new
estimations is then used to make new assumptions about the parameter
distributions, refining future estimations.

All implemented classes in this module inherit from a base abstract class
:py:class:`Estimator`. :py:class:`Simulator` allows that a custom estimator be
used during the simulation, as long as it also inherits from
:py:class:`Estimator`.

.. inheritance-diagram:: catsim.estimation

:mod:`catsim` implements a few types of maximum-likelihood estimators.

.. automodule:: catsim.estimation
    :members:
    :show-inheritance:

Comparison between estimators
-----------------------------

The chart below displays the execution times of the same simulation (100 examinees, an item bank of 300 questions and 20 items per test) using different :math:`\hat{\theta}` estimation methods.

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt

    from catsim.simulation import Simulator
    from catsim.initialization import FixedPointInitializer
    from catsim.selection import MaxInfoSelector
    from catsim.estimation import NumericalSearchEstimator
    from catsim.stopping import MaxItemStopper
    from catsim.cat import generate_item_bank

    items = generate_item_bank(300)
    examinees = 100
    test_size = 20
    thetas = np.random.normal(0,1,examinees)
    sim_times = {}
    for m in NumericalSearchEstimator.methods:
        simulator = Simulator(items, thetas,FixedPointInitializer(0),MaxInfoSelector(), NumericalSearchEstimator(method=m),MaxItemStopper(test_size))
        simulator.simulate(verbose=True)
        sim_times[m] = simulator.duration

    plt.figure(figsize=(10,5))
    plt.bar(range(len(sim_times)), list(sim_times.values()), align='center')
    plt.xticks(range(len(sim_times)), list(sim_times.keys()))
    plt.show()


The charts below show the :math:`\hat{\theta}` found by the different estimation methods, given three dichotomous response vectors with different numbers of correct answers. All estimators reach comparable results, maximizing the log-likelihood function . The main difference among them is in the number of calls to :py:func:`catsim.irt.log_likelihood`. Methods that require less calls to maximize the function are usually more time efficient.


.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from catsim.estimation import *
    from catsim.cat import generate_item_bank

    test_size = 20
    items = generate_item_bank(20)
    items = items[items[:,1].argsort()] # order by difficulty ascending
    r0 = [True] * 7 + [False] * 13
    r1 = [True] * 10 + [False] * 10
    r2 = [True] * 15 + [False] * 5
    response_vectors = [r0, r1, r2]
    thetas = np.arange(-6.,6.,.1)

    for estimator in [
            NumericalSearchEstimator(method=m) for m in NumericalSearchEstimator.methods
        ]:
        plt.figure()

        for response_vector in response_vectors:
            ll_line = [irt.log_likelihood(theta, response_vector, items) for theta in thetas]
            max_LL = estimator.estimate(items=items, administered_items=range(20),
                                        response_vector=response_vector, est_theta=0)
            best_theta = irt.log_likelihood(max_LL, response_vector, items)
            plt.plot(thetas, ll_line)
            plt.plot(max_LL, best_theta, 'o', label = str(sum(response_vector)) + ' correct, '+r'$\hat{\theta} \approx $' + format(round(max_LL, 5)))
            plt.xlabel(r'$\theta$', size=16)
            plt.ylabel(r'$\log L(\theta)$', size=16)
            plt.title(f"{estimator.method} ({round(estimator.avg_evaluations)} avg. evals)")
            plt.legend(loc='best')

        plt.show()
