Estimation Methods -- :mod:`catsim.estimation`
**********************************************

Estimators are the objects responsible for estimating of examinees
proficiency values, given a dichotomous (binary) response vector and an array of
the items answered by the examinee. In the domain of IRT, there are two main
types of ways of estimating :math:`\hat\theta`: and these are the Bayesian
methods and maximum-likelihood ones.

Maximum-likelihood methods choose the :math:`\hat\theta` value that maximizes
the likelihood (see :py:func:logLik) of an examinee having a certain response
vector, given the corresponding item parameters.

Bayesian methods used *a priori* information (usually assuming proficiency and
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

The plots below show a comparison of the different estimator available. Given
three dichotomous (binary) response vectors with different numbers of correct
answers, all the estimators find values for :math:`\hat\theta` that maximize the
log-likelihood function. Some estimators evaluate the log-likelihood less times
than others, while reaching similar results, which may make them (although not
necessarily) more efficient estimators.

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from catsim.estimation import *
    from catsim.cat import generate_item_bank

    test_size = 20
    randBinList = lambda n: [np.random.randint(0,2) for b in range(1,n+1)]
    items = generate_item_bank(20)
    items = items[items[:,1].argsort()] # order by difficulty ascending
    r0 = [1] * 7 + [0] * 13
    r1 = [1] * 10 + [0] * 10
    r2 = [1] * 15 + [0] * 5
    response_vectors = [r0, r1, r2]
    thetas = np.arange(-6.,6.,.1)

    for estimator in [FMinEstimator(), DifferentialEvolutionEstimator((-8, 8)), HillClimbingEstimator()]:
        plt.figure()

        for response_vector in response_vectors:
            ll_line = [irt.logLik(theta, response_vector, items) for theta in thetas]
            max_LL = estimator.estimate(response_vector, items, 0)
            best_theta = irt.logLik(max_LL, response_vector, items)
            plt.plot(thetas, ll_line)
            plt.plot(max_LL, best_theta, 'o', label = str(sum(response_vector)) + ' correct, '+r'$\hat{\theta} \approx $' + format(round(max_LL, 5)))
            plt.xlabel(r'$\theta$', size=16)
            plt.ylabel(r'$\log L(\theta)$', size=16)
            plt.title('MLE -- {0} ({1} avg. evals)'.format(type(estimator).__name__, round(estimator.avg_evaluations)))
            plt.legend(loc='best')

        plt.show()
