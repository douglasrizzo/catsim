import matplotlib.pyplot as plt

from catsim import plot
from catsim.estimation import NumericalSearchEstimator
from catsim.initialization import RandomInitializer
from catsim.item_bank import ItemBank
from catsim.selection import MaxInfoSelector
from catsim.simulation import Simulator
from catsim.stopping import MaxItemStopper

fig, axes = plt.subplots(2, 1, figsize=(7, 12))
s = Simulator(ItemBank.generate_item_bank(100), 10)
s.simulate(RandomInitializer(), MaxInfoSelector(), NumericalSearchEstimator(), MaxItemStopper(20))
plot.test_progress(simulator=s, index=0, ax=axes[0])
plot.test_progress(simulator=s, index=0, info=True, var=True, see=True, ax=axes[1])
plt.tight_layout()
plt.show()
