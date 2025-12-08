import matplotlib.pyplot as plt
from catsim.item_bank import ItemBank
from catsim.plot import item_curve, PlotType
item_bank = ItemBank.generate_item_bank(1)
item = item_bank.items[0]
fig, axes = plt.subplots(3, 1, figsize=(7, 15))
item_curve(item[0], item[1], item[2], item[3], ptype=PlotType.ICC, ax=axes[0])
item_curve(item[0], item[1], item[2], item[3], ptype=PlotType.IIC, ax=axes[1])
item_curve(item[0], item[1], item[2], item[3], ptype=PlotType.BOTH, ax=axes[2])
plt.tight_layout()
plt.show()