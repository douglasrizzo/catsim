import matplotlib.pyplot as plt

from catsim import ItemBank
from catsim.estimation import NumericalSearchEstimator
from catsim.initialization import RandomInitializer
from catsim.plot import test_progress
from catsim.selection import MaxInfoSelector
from catsim.simulation import Simulator
from catsim.stopping import MinErrorStopper

figure, axes = plt.subplots(2, 1, figsize=(10, 12))

for ax in axes:
  item_bank = ItemBank.generate_item_bank(5000, seed=42)
  s = Simulator(item_bank, examinees=1, seed=42)
  s.simulate(RandomInitializer(), MaxInfoSelector(), NumericalSearchEstimator(), MinErrorStopper(0.2))
  test_progress(ax=ax, simulator=s, index=0, see=True, marker="|")
