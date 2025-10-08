import matplotlib.pyplot as plt
from catsim import plot
from catsim.item_bank import ItemBank
item_bank = ItemBank.generate_item_bank(100)
plot.gen3d_dataset_scatter(item_bank); plt.show()