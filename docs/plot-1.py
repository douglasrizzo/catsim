from catsim import plot
from catsim.cat import generate_item_bank

items = generate_item_bank(100)
plot.gen3d_dataset_scatter(items)
