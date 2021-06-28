from catsim import plot
from catsim.cat import generate_item_bank
from catsim.estimation import NumericalSearchEstimator
from catsim.initialization import RandomInitializer
from catsim.selection import MaxInfoSelector
from catsim.simulation import Simulator
from catsim.stopping import MaxItemStopper

s = Simulator(generate_item_bank(100), 10)
s.simulate(
    RandomInitializer(),
    MaxInfoSelector(),
    NumericalSearchEstimator(),
    MaxItemStopper(20),
)
plot.test_progress(simulator=s, index=0)
plot.test_progress(simulator=s, index=0, info=True, var=True, see=True)
