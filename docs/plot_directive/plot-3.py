from catsim.cat import generate_item_bank
from catsim import plot
from catsim.initialization import RandomInitializer
from catsim.selection import MaxInfoSelector
from catsim.estimation import NumericalSearchEstimator
from catsim.stopping import MaxItemStopper
from catsim.simulation import Simulator

s = Simulator(generate_item_bank(100), 10)
s.simulate(RandomInitializer(), MaxInfoSelector(), NumericalSearchEstimator(), MaxItemStopper(20))
plot.item_exposure(title='Exposures', simulator=s, hist=True)
plot.item_exposure(title='Exposures', simulator=s, par='b')