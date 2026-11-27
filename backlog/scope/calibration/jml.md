---
title: Joint Maximum Likelihood (JML) estimator
kind: technique
status: parked
priority: 4
effort: S
blocked_by: [scope/calibration/icc-gradients]
raised_by: []
---

# Joint Maximum Likelihood (JML) estimator

> Background: [the literature review §2.1](literature-review.md)

## Reference

Birnbaum, A. (1968). Some latent trait models and their use in inferring an examinee's ability. In F. M. Lord & M. R. Novick (Eds.), *Statistical Theories of Mental Test Scores*. Addison-Wesley.

Lord, F. M. (1980). *Applications of Item Response Theory to Practical Testing Problems*. Erlbaum.

Baker, F. B., & Kim, S.-H. (2004). *Item Response Theory: Parameter Estimation Techniques* (2nd ed.). Chapters 3–6.

## Theory

JML treats both item parameters and examinee abilities as fixed unknowns and alternates between two optimization steps:

### Step 1: Estimate abilities (fix items)

For each examinee $j$, maximize the individual log-likelihood:

$$\hat\theta_j = \arg\max_\theta \sum_{i \in \mathcal{A}_j} \left[u_{ij} \log P_i(\theta) + (1 - u_{ij}) \log(1 - P_i(\theta))\right]$$

This is exactly what catsim's `NumericalSearchEstimator` already does. Reuse it directly.

### Step 2: Estimate item parameters (fix abilities)

For each item $i$, maximize the per-item log-likelihood:

$$(\hat{a}_i, \hat{b}_i) = \arg\max_{a_i, b_i} \sum_{j} \left[u_{ij} \log P_i(\hat\theta_j) + (1 - u_{ij}) \log(1 - P_i(\hat\theta_j))\right]$$

Use `scipy.optimize.minimize(method='L-BFGS-B')` with analytic gradients from icc-gradients.

### Iteration

Alternate Steps 1 and 2 until convergence (max parameter change < tolerance or max iterations).

### The Neyman-Scott problem

JML estimates are **inconsistent** for fixed test length as the number of examinees grows. The number of nuisance parameters (one $\theta_j$ per examinee) grows with sample size, violating the regularity conditions for MLE consistency. Consequences:
- Discrimination $a$ is **overestimated**.
- Difficulty variance is **underestimated** (difficulty parameters are shrunk toward the mean).
- The bias does not vanish as $N \to \infty$ (it depends on test length, not sample size).

### Bias correction (optional)

The Wright-Douglas correction divides each $b_i$ by a factor $(I-1)/I$ where $I$ is the test length. More sophisticated corrections exist but are beyond scope for a baseline implementation.

### Perfect/zero scores

Examinees with all-correct or all-incorrect responses have undefined MLE $\theta$ (the likelihood is monotonic). Options:
- Exclude them from the item step (standard approach).
- Fence $\theta$ at $\pm K$ (e.g., $K = 4$).

## Mock implementation

```python
# src/catsim/calibration/jml.py
"""Joint Maximum Likelihood estimation (Birnbaum, 1968)."""

import numpy as np
import numpy.typing as npt
from scipy import optimize

from .. import irt
from ..estimation.numerical import NumericalSearchEstimator
from ..item_bank import ItemBank


class JMLCalibrator:
    """Estimate item parameters via Joint Maximum Likelihood.

    Parameters
    ----------
    model : {"1PL", "2PL"}
        Which IRT model to fit. 3PL is not recommended with JML.
    max_iter : int
        Maximum number of alternation cycles. Default 100.
    tol : float
        Convergence tolerance on max absolute parameter change. Default 0.001.
    theta_bounds : tuple[float, float]
        Bounds for ability estimation. Default (-6, 6).
    """

    def __init__(
        self,
        model: str = "2PL",
        max_iter: int = 100,
        tol: float = 1e-3,
        theta_bounds: tuple[float, float] = (-6.0, 6.0),
    ) -> None:
        self.model = model
        self.max_iter = max_iter
        self.tol = tol
        self.theta_bounds = theta_bounds

    def calibrate(
        self,
        responses: npt.NDArray[np.integer],
    ) -> dict:
        """Estimate item parameters from a binary response matrix.

        Parameters
        ----------
        responses : numpy.ndarray, shape (n_examinees, n_items)
            Binary response matrix.

        Returns
        -------
        dict with keys:
            'items': numpy.ndarray, shape (n_items, 4)
            'theta': numpy.ndarray, shape (n_examinees,)
            'converged': bool
            'n_iter': int
        """
        n_examinees, n_items = responses.shape
        mask = np.isfinite(responses) & (responses >= 0)

        # Initialize
        a = np.ones(n_items)
        b = np.zeros(n_items)
        p_obs = np.nanmean(np.where(mask, responses, np.nan), axis=0)
        p_obs = np.clip(p_obs, 0.01, 0.99)
        b = -np.log(p_obs / (1.0 - p_obs))

        theta = np.zeros(n_examinees)

        for iteration in range(self.max_iter):
            items = np.column_stack([a, b, np.zeros(n_items), np.ones(n_items)])

            # Step 1: estimate abilities (reuse NumericalSearchEstimator logic)
            theta_new = self._estimate_abilities(responses, mask, items, theta)

            # Fence perfect/zero scores
            scores = np.nansum(np.where(mask, responses, 0), axis=1)
            n_admin = np.sum(mask, axis=1)
            perfect = scores >= n_admin
            zero = scores <= 0
            theta_new[perfect] = self.theta_bounds[1]
            theta_new[zero] = self.theta_bounds[0]

            # Step 2: estimate item parameters
            a_new, b_new = self._estimate_items(
                responses, mask, items, theta_new
            )

            # Convergence check
            if self.model == "1PL":
                max_change = np.max(np.abs(b_new - b))
            else:
                max_change = max(
                    np.max(np.abs(a_new - a)), np.max(np.abs(b_new - b))
                )

            a, b, theta = a_new, b_new, theta_new
            if max_change < self.tol:
                break

        items_final = np.column_stack([a, b, np.zeros(n_items), np.ones(n_items)])
        return {
            "items": items_final,
            "theta": theta,
            "converged": max_change < self.tol,
            "n_iter": iteration + 1,
        }

    def _estimate_abilities(self, responses, mask, items, theta_init):
        """Estimate theta for each examinee via MLE."""
        n_examinees = responses.shape[0]
        theta_new = theta_init.copy()

        for j in range(n_examinees):
            admin = np.where(mask[j])[0]
            if len(admin) == 0:
                continue
            u_j = responses[j, admin]
            items_j = items[admin]

            def neg_ll(th):
                p = irt.icc_hpc(th[0], items_j)
                p = np.clip(p, 1e-10, 1 - 1e-10)
                return -np.sum(u_j * np.log(p) + (1 - u_j) * np.log(1 - p))

            result = optimize.minimize_scalar(
                lambda th: neg_ll([th]),
                bounds=self.theta_bounds,
                method="bounded",
            )
            theta_new[j] = result.x

        return theta_new

    def _estimate_items(self, responses, mask, items, theta):
        """Estimate item parameters given fixed abilities."""
        n_items = items.shape[0]
        a_new = items[:, 0].copy()
        b_new = items[:, 1].copy()

        for i in range(n_items):
            admin = np.where(mask[:, i])[0]
            if len(admin) < 2:
                continue
            u_i = responses[admin, i]
            th_i = theta[admin]

            if self.model == "1PL":
                x0 = [b_new[i]]
                bounds = [(-6.0, 6.0)]
            else:
                x0 = [a_new[i], b_new[i]]
                bounds = [(0.1, 5.0), (-6.0, 6.0)]

            def neg_ll(params, _u=u_i, _th=th_i):
                if self.model == "1PL":
                    _a, _b = 1.0, params[0]
                else:
                    _a, _b = params
                p = irt.icc(_th, _a, _b)  # scalar calls in loop — fine for JML
                p = np.clip(p, 1e-10, 1 - 1e-10)
                return -np.sum(_u * np.log(p) + (1 - _u) * np.log(1 - p))

            result = optimize.minimize(neg_ll, x0, method="L-BFGS-B", bounds=bounds)

            if self.model == "1PL":
                b_new[i] = result.x[0]
            else:
                a_new[i], b_new[i] = result.x

        return a_new, b_new
```

## Proposed location

`src/catsim/calibration/jml.py`. Re-export from `src/catsim/calibration/__init__.py`.

## Tests

1. **Recovery test (2PL)**: Generate 20 items, 1000 examinees. Assert recovered parameters correlate $r > 0.90$ with true values (lower bar than MML-EM due to JML bias).
2. **JML bias**: Compare JML-estimated discrimination against true values. Mean estimated $a$ should be higher than mean true $a$ (demonstrating the known upward bias).
3. **JML vs MML-EM**: On the same simulated data, MML-EM (mml-em) should produce lower RMSE than JML.
4. **Perfect/zero scores**: Examinees with all-correct/all-incorrect should get fenced theta, not crash.
5. **Convergence**: Verify `converged=True` within 100 iterations for well-behaved data.
6. **1PL mode**: With `model="1PL"`, all estimated `a` values should be 1.0.

## Notes

- JML is **not recommended** for production calibration. Its value in catsim is as a didactic baseline and for comparison against MML-EM.
- The `_estimate_abilities` step could reuse `NumericalSearchEstimator` directly, but the interface expects `ItemBank` and `administered_items` as lists, so it's simpler to use `scipy.optimize.minimize_scalar` directly on the log-likelihood.
- 3PL is not supported because JML's c-parameter estimates are even more biased than MML-EM's. Use MML-EM for 3PL.
- Computational cost: $O(K \cdot (J + I))$ per iteration where $K$ is the number of alternations, $J$ is examinees, $I$ is items. Much slower than MML-EM for large datasets because the ability step loops over all examinees.
