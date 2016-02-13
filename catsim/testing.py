import unittest
from catsim.cat import generate_item_bank
from catsim import irt
from catsim.initialization import RandomInitializer
from catsim.selection import MaxInfoSelector, ClusterSelector
from catsim.reestimation import HillClimbingEstimator, BinarySearchEstimator, DifferentialEvolutionEstimator, FMinEstimator
from catsim.stopping import MaxItemStopper, MinErrorStopper
from catsim.simulation import Simulator


class TestStuff(unittest.TestCase):

    def test_item_bank_generation(self):
        for items in [
            generate_item_bank(5, '1PL'), generate_item_bank(5, '2PL'),
            generate_item_bank(5, '3PL'), generate_item_bank(
                5,
                '3PL',
                corr=0
            )
        ]:
            irt.validate_item_bank(items, raise_err=True)

            assert True

    def test_simulations(self):
        items = generate_item_bank(5000)

        initializer = RandomInitializer()
        selector = MaxInfoSelector()

        for estimator in [
            HillClimbingEstimator(), BinarySearchEstimator(),
            DifferentialEvolutionEstimator((-8, 8)), FMinEstimator()
        ]:
            for stopper in [MaxItemStopper(20), MinErrorStopper(.4)]:
                s = Simulator(items, 10)
                s.simulate(initializer, selector, estimator, stopper)
                # plot.test_progress(simulator=s, index=0, see=True)

        assert True

    def test_cism(self):
        from sklearn.cluster import KMeans

        items = generate_item_bank(100)
        clusters = KMeans(n_clusters=8).fit_predict(items)

        initializer = RandomInitializer()
        selector = ClusterSelector(clusters=clusters, r_max=.2)
        estimator = HillClimbingEstimator()
        stopper = MaxItemStopper(20)
        simulator = Simulator(items, 10)
        simulator.simulate(initializer, selector, estimator, stopper)

        assert True


if __name__ == '__main__':
    unittest.main()
