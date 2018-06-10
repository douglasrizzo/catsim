from catsim.cat import generate_item_bank
from catsim import plot
from catsim.initialization import RandomInitializer
from catsim.selection import MaxInfoSelector
from catsim.estimation import HillClimbingEstimator
from catsim.stopping import MaxItemStopper
from catsim.simulation import Simulator

s = Simulator(generate_item_bank(100), 10)
s.simulate(RandomInitializer(), MaxInfoSelector(), HillClimbingEstimator(), MaxItemStopper(20))
plot.test_progress(simulator=s, index=0)
plot.test_progress(simulator=s, index=0, info=True, var=True, see=True)