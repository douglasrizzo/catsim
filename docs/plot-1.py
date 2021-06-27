from catsim.cat import generate_item_bank
from catsim import plot
items = generate_item_bank(100)
plot.gen3d_dataset_scatter(items)