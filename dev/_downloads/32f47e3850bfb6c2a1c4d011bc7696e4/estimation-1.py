import numpy as np
import matplotlib.pyplot as plt

from catsim.simulation import Simulator
from catsim.initialization import FixedPointInitializer
from catsim.selection import MaxInfoSelector
from catsim.estimation import NumericalSearchEstimator
from catsim.stopping import MaxItemStopper
from catsim import ItemBank

items = ItemBank.generate_item_bank(300)
examinees = 100
test_size = 20
rng = np.random.default_rng()
thetas = rng.normal(0, 1, examinees)
sim_times = {}
for m in NumericalSearchEstimator.available_methods():
    simulator = Simulator(items, thetas)
    simulator.simulate(
        FixedPointInitializer(0),
        MaxInfoSelector(),
        NumericalSearchEstimator(method=m),
        MaxItemStopper(test_size),
        verbose=True
    )
    sim_times[m] = simulator.duration

plt.figure(figsize=(10,5))
plt.bar(range(len(sim_times)), list(sim_times.values()), align='center')
plt.xticks(range(len(sim_times)), list(sim_times.keys()))
plt.show()