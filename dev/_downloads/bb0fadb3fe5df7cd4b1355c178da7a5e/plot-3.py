import matplotlib.pyplot as plt
from catsim.item_bank import ItemBank
from catsim import plot
from catsim.initialization import RandomInitializer
from catsim.selection import MaxInfoSelector
from catsim.estimation import NumericalSearchEstimator
from catsim.stopping import MinErrorStopper
from catsim.simulation import SimulationRunner

fig, axes = plt.subplots(2, 1, figsize=(7, 12))

runner = SimulationRunner(
    ItemBank.generate_item_bank(100),
    RandomInitializer(),
    MaxInfoSelector(),
    NumericalSearchEstimator(),
    MinErrorStopper(0.4, max_items=20),
)
result = runner.run(10)
plot.item_exposure(title='Exposures', simulation=result, hist=True, ax=axes[0])
plot.item_exposure(title='Exposures', simulation=result, par='b', ax=axes[1])
plt.tight_layout()
plt.show()