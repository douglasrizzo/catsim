import unittest
from catsim.cat import generate_item_bank
from catsim import irt
from catsim.initialization import RandomInitializer, FixedPointInitializer
from catsim.selection import MaxInfoSelector, ClusterSelector
from catsim.reestimation import HillClimbingEstimator, BinarySearchEstimator, DifferentialEvolutionEstimator, FMinEstimator
from catsim.stopping import MaxItemStopper, MinErrorStopper
from catsim.simulation import Simulator


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


def test_simulations():
    from sklearn.cluster import KMeans
    items = generate_item_bank(5000)
    examinees = 10
    clusters = KMeans(n_clusters=8).fit_predict(items)

    for initializer in [
        RandomInitializer('uniform',
                          (-5, 5)
                          # ), RandomInitializer(
                          #     'normal', (0, 1)
                          ),
        FixedPointInitializer(0)
    ]:
        for estimator in [
            HillClimbingEstimator(), BinarySearchEstimator(),
            DifferentialEvolutionEstimator((-8, 8)), FMinEstimator()
        ]:
            for selector in [MaxInfoSelector()]:
                for stopper in [MaxItemStopper(20), MinErrorStopper(.4)]:
                    yield one_simulation, items, examinees, initializer, selector, estimator, stopper

            for selector in [MaxInfoSelector(), ClusterSelector(clusters=clusters, r_max=.2)]:
                for stopper in [MaxItemStopper(20)]:
                    yield one_simulation, items, examinees, initializer, selector, estimator, stopper


def one_simulation(items, examinees, initializer, selector, estimator, stopper):
    Simulator(items, examinees).simulate(initializer, selector, estimator, stopper)
    pass


if __name__ == '__main__':
    unittest.main()
