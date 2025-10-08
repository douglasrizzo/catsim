import numpy as np
import matplotlib.pyplot as plt
from catsim.estimation import *
from catsim import ItemBank

test_size = 20
item_bank = ItemBank.generate_item_bank(20)
# Sort by difficulty ascending
sorted_indices = item_bank.difficulty.argsort()
items = item_bank.items[sorted_indices]
sorted_bank = ItemBank(items)

r0 = [True] * 7 + [False] * 13
r1 = [True] * 10 + [False] * 10
r2 = [True] * 15 + [False] * 5
response_vectors = [r0, r1, r2]
thetas = np.arange(-6.,6.,.1)

fig, axes = plt.subplots(len(NumericalSearchEstimator.available_methods()), 1, figsize=(8,35))

for idx, estimator in enumerate([
        NumericalSearchEstimator(method=m) for m in NumericalSearchEstimator.available_methods()
    ]):
    ax = axes[idx]
    for response_vector in response_vectors:
        ll_line = [irt.log_likelihood(theta, response_vector, sorted_bank.items) for theta in thetas]
        max_LL = estimator.estimate(item_bank=sorted_bank, administered_items=list(range(20)),
                                    response_vector=response_vector, est_theta=0)
        best_theta = irt.log_likelihood(max_LL, response_vector, sorted_bank.items)
        ax.plot(thetas, ll_line)
        ax.plot(max_LL, best_theta, 'o', label = str(sum(response_vector)) + ' correct, '+r'$\hat{\theta} \approx $' + format(round(max_LL, 5)))
        ax.set_xlabel(r'$\theta$', size=16)
        ax.set_ylabel(r'$\log L(\theta)$', size=16)
        ax.set_title(f"{estimator.method} ({round(estimator.avg_evaluations)} avg. evals)")
        ax.legend(loc='best')

plt.tight_layout()
plt.show()