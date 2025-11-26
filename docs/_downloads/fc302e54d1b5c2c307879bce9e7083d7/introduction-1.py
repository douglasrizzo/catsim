import matplotlib.pyplot as plt

from catsim import ItemBank
from catsim.plot import PlotType, item_curve

item_bank = ItemBank.generate_item_bank(1)
item = item_bank[0]
item_curve(a=item[0], b=item[1], c=item[2], d=item[3], ptype=PlotType.BOTH)
plt.show()
