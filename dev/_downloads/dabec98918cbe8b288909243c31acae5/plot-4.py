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
plot.test_progress(simulation=result, index=0, ax=axes[0])
plot.test_progress(simulation=result, index=0, info=True, var=True, see=True, ax=axes[1])
plt.tight_layout()
plt.show()