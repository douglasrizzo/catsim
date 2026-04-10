import numpy as np
import matplotlib.pyplot as plt

from catsim.simulation import SimulationRunner
from catsim.initialization import FixedPointInitializer
from catsim.selection import MaxInfoSelector
from catsim.estimation import NumericalSearchEstimator
from catsim.stopping import MinErrorStopper
from catsim import ItemBank

items = ItemBank.generate_item_bank(300)
examinees = 100
test_size = 20
rng = np.random.default_rng()
thetas = rng.normal(0, 1, examinees)
sim_times = {}
for m in NumericalSearchEstimator.available_methods():
    runner = SimulationRunner(
        items,
        FixedPointInitializer(0),
        MaxInfoSelector(),
        NumericalSearchEstimator(method=m),
        MinErrorStopper(0.4, max_items=test_size),
    )
    result = runner.run(thetas, verbose=True)
    sim_times[m] = result.duration

plt.figure(figsize=(10,5))
plt.bar(range(len(sim_times)), list(sim_times.values()), align='center')
plt.xticks(range(len(sim_times)), list(sim_times.keys()))
plt.show()