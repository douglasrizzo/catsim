from catsim.cat import generate_item_bank
from catsim import plot
item = generate_item_bank(1)[0]
plot.item_curve(item[0], item[1], item[2], item[3], ptype='icc')
plot.item_curve(item[0], item[1], item[2], item[3], ptype='iic')
plot.item_curve(item[0], item[1], item[2], item[3], ptype='both')