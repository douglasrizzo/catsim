import unittest
import numpy
from catsim import irt
from catsim import plot
from catsim.simulation import Simulator
from catsim.cat import generate_item_bank
from catsim.stopping import MaxItemStopper, MinErrorStopper
from catsim.selection import MaxInfoSelector, ClusterSelector
from catsim.initialization import RandomInitializer, FixedPointInitializer
from catsim.reestimation import HillClimbingEstimator, BinarySearchEstimator, DifferentialEvolutionEstimator, FMinEstimator


def test_item_bank_generation():
    for items in [
        generate_item_bank(5, '1PL'), generate_item_bank(5, '2PL'), generate_item_bank(5, '3PL'),
        generate_item_bank(
            5,
            '3PL',
            corr=0
        )
    ]:
        irt.validate_item_bank(items, raise_err=True)

    items = numpy.zeros((100))
    irt.validate_item_bank(items)
    items = irt.normalize_item_bank(items)
    irt.validate_item_bank(items, raise_err=True)


def test_plots():
    initializer = RandomInitializer()
    selector = MaxInfoSelector()
    estimator = HillClimbingEstimator()
    stopper = MaxItemStopper(20)
    s = Simulator(generate_item_bank(100), 10)
    s.simulate(initializer, selector, estimator, stopper)

    for item in s.items[0:10]:
        yield plot.item_curve, item[0], item[1], item[2], 'Test plot', 'icc', None, False
        yield plot.item_curve, item[0], item[1], item[2], 'Test plot', 'iic', None, False
        yield plot.item_curve, item[0], item[1], item[2], 'Test plot', 'both', None, False

    plot.gen3D_dataset_scatter(items=s.items, show=False)
    plot.test_progress(
        title='Test progress',
        simulator=s,
        index=0,
        info=True,
        see=True,
        reliability=True,
        show=False
    )


def test_simulations():
    examinees = 10
    initializers = [
        RandomInitializer('uniform',
                          (-5, 5)
                          # ), RandomInitializer(
                          #     'normal', (0, 1)
                          ),
        FixedPointInitializer(0)
    ]
    selectors = [MaxInfoSelector()]
    estimators = [
        HillClimbingEstimator(), BinarySearchEstimator(), DifferentialEvolutionEstimator((-8, 8)),
        FMinEstimator()
    ]
    stoppers = [MaxItemStopper(20), MinErrorStopper(.4)]

    for initializer in initializers:
        for selector in selectors:
            for estimator in estimators:
                for stopper in stoppers:
                    items = generate_item_bank(5000)
                    yield one_simulation, items, examinees, initializer, selector, estimator, stopper


def test_cism():
    from sklearn.cluster import KMeans

    examinees = 10
    initializers = [
        RandomInitializer('uniform',
                          (-5, 5)
                          # ), RandomInitializer(
                          #     'normal', (0, 1)
                          ),
        FixedPointInitializer(0)
    ]
    estimators = [
        HillClimbingEstimator(), BinarySearchEstimator(), DifferentialEvolutionEstimator((-8, 8)),
        FMinEstimator()
    ]
    stoppers = [MaxItemStopper(20), MinErrorStopper(.4)]

    for initializer in initializers:
        for estimator in estimators:
            for stopper in stoppers:
                items = generate_item_bank(5000)
                clusters = KMeans(n_clusters=8).fit_predict(items)
                ClusterSelector.weighted_cluster_infos(0, items, clusters)
                ClusterSelector.avg_cluster_params(items, clusters)
                selector = ClusterSelector(clusters=clusters, r_max=.2)
                yield one_simulation, items, examinees, initializer, selector, estimator, stopper


def one_simulation(items, examinees, initializer, selector, estimator, stopper):
    Simulator(items, examinees).simulate(initializer, selector, estimator, stopper)


if __name__ == '__main__':
    unittest.main()
