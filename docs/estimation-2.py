import matplotlib.pyplot as plt
import numpy as np

from catsim.cat import generate_item_bank
from catsim.estimation import *

test_size = 20
items = generate_item_bank(20)
items = items[items[:, 1].argsort()]  # order by difficulty ascending
r0 = [True] * 7 + [False] * 13
r1 = [True] * 10 + [False] * 10
r2 = [True] * 15 + [False] * 5
response_vectors = [r0, r1, r2]
thetas = np.arange(-6.0, 6.0, 0.1)

for estimator in [NumericalSearchEstimator(method=m) for m in NumericalSearchEstimator.methods]:
    plt.figure()

    for response_vector in response_vectors:
        ll_line = [irt.log_likelihood(theta, response_vector, items) for theta in thetas]
        max_LL = estimator.estimate(
            items=items,
            administered_items=range(20),
            response_vector=response_vector,
            est_theta=0,
        )
        best_theta = irt.log_likelihood(max_LL, response_vector, items)
        plt.plot(thetas, ll_line)
        plt.plot(
            max_LL,
            best_theta,
            "o",
            label=str(sum(response_vector)) + " correct, " + r"$\hat{\theta} \approx $" +
            format(round(max_LL, 5)),
        )
        plt.xlabel(r"$\theta$", size=16)
        plt.ylabel(r"$\log L(\theta)$", size=16)
        plt.title("{0} ({1} avg. evals)".format(estimator.method, round(estimator.avg_evaluations)))
        plt.legend(loc="best")

    plt.show()
