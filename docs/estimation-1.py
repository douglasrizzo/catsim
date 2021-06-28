import matplotlib.pyplot as plt
import numpy as np

from catsim.cat import generate_item_bank
from catsim.estimation import NumericalSearchEstimator
from catsim.initialization import FixedPointInitializer
from catsim.selection import MaxInfoSelector
from catsim.simulation import Simulator
from catsim.stopping import MaxItemStopper

items = generate_item_bank(300)
examinees = 100
test_size = 20
thetas = np.random.normal(0, 1, examinees)
sim_times = {}
for m in NumericalSearchEstimator.methods:
    simulator = Simulator(
        items,
        thetas,
        FixedPointInitializer(0),
        MaxInfoSelector(),
        NumericalSearchEstimator(method=m),
        MaxItemStopper(test_size),
    )
    simulator.simulate(verbose=True)
    sim_times[m] = simulator.duration

plt.figure(figsize=(10, 5))
plt.bar(range(len(sim_times)), list(sim_times.values()), align="center")
plt.xticks(range(len(sim_times)), list(sim_times.keys()))
plt.show()
