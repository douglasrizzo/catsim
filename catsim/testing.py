import unittest
from catsim.cat import generate_item_bank
from catsim import irt
from catsim import plot
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

    # def test_plots(self):
    #     initializer = RandomInitializer()
    #     selector = MaxInfoSelector()
    #     estimator = HillClimbingEstimator()
    #     stopper = MaxItemStopper(20)
    #     s = Simulator(generate_item_bank(100), 10)
    #     s.simulate(initializer, selector, estimator, stopper)
    #
    #     a, b, c = s.items[0, 0:3]
    #     plot.item_curve(a, b, c, title='Test plot', ptype='both', filepath='/tmp/irt.pdf')
    #     plot.gen3D_dataset_scatter(items=s.items, filepath='/tmp/3d.pdf')
    #     plot.test_progress(
    #         title='Test progress',
    #         simulator=s,
    #         index=0,
    #         filepath='/tmp/progress.pdf'
    #     )
    #
    #     assert True

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
