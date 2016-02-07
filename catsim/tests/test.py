import unittest
from catsim.cat import generate_item_bank
from catsim import irt
from catsim.initialization import RandomInitializer
from catsim.selection import MaxInfoSelector
from catsim.reestimation import HillClimbingEstimator, BinarySearchEstimator, DifferentialEvolutionEstimator, FMinEstimator
from catsim.stopping import MaxItemStopper
from catsim.simulation import Simulator


class TestStuff(unittest.TestCase):

    def item_bank_generation(self):
        for items in [
            generate_item_bank(5, '1PL'), generate_item_bank(5, '2PL'),
            generate_item_bank(5, '3PL'), generate_item_bank(
                5,
                '3PL',
                corr=0
            )
        ]:
            irt.validate_item_bank(items, raise_err=True)

        self.assertTrue(True)

    def simple_test_simulation(self):
        initializer = RandomInitializer()
        selector = MaxInfoSelector()
        estimator = HillClimbingEstimator()
        stopper = MaxItemStopper(20)
        Simulator(generate_item_bank(100), 10).simulate(initializer, selector, estimator, stopper)

    def different_estimators(self):
        initializer = RandomInitializer()
        selector = MaxInfoSelector()
        stopper = MaxItemStopper(20)

        for estimator in [
            HillClimbingEstimator(), BinarySearchEstimator(),
            DifferentialEvolutionEstimator((-8, 8)), FMinEstimator()
        ]:
            Simulator(
                generate_item_bank(100), 10
            ).simulate(initializer, selector, estimator, stopper)


if __name__ == '__main__':
    unittest.main()
