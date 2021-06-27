from catsim.cat import generate_item_bank
from catsim import plot
items = generate_item_bank(2)
for item in items:
    plot.item_curve(item[0], item[1], item[2], item[3], ptype='iic', max_info=True)