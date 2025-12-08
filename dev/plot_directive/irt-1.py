import matplotlib.pyplot as plt
from catsim.item_bank import ItemBank
from catsim.plot import item_curve, PlotType
n_items = 2
item_bank = ItemBank.generate_item_bank(n_items)
fig, axes = plt.subplots(n_items, 1)
for idx, item in enumerate(item_bank.items):
  item_curve(item[0], item[1], item[2], item[3], ptype=PlotType.IIC, max_info=True, ax=axes[idx])
plt.show()