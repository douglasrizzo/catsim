---
title: Conditional Maximum Likelihood for Rasch/1PL
kind: technique
status: parked
priority: 4
effort: S
blocked_by: []
raised_by: []
---

# Conditional Maximum Likelihood for Rasch/1PL

> Background: [the literature review §2.3](literature-review.md)

## Reference

Andersen, E. B. (1972). The numerical solution of a set of conditional estimation equations. *Journal of the Royal Statistical Society, Series B*, 34, 42–54.

Baker, F. B., & Kim, S.-H. (2004). *Item Response Theory: Parameter Estimation Techniques* (2nd ed.). Chapter 6.

## Theory

For the Rasch model ($a = 1$, $c = 0$, $d = 1$), the total score $r_j = \sum_i u_{ij}$ is a **sufficient statistic** for $\theta_j$. Conditioning the likelihood on the observed total scores eliminates all ability parameters from the likelihood entirely, leaving a function of difficulty parameters only.

### Conditional log-likelihood

$$\ell_C(\mathbf{b}) = \sum_j \left[-\sum_{i \in \mathcal{A}_j} b_i \cdot u_{ij}\right] - \sum_j \log \gamma_{r_j}(\mathbf{b})$$

where $\gamma_r(\mathbf{b})$ is the **elementary symmetric function (ESF)** of order $r$ — the sum of all products of $\exp(-b_i)$ taken $r$ at a time over the administered items.

### Elementary symmetric functions

For a set of items with parameters $\{\exp(-b_1), \ldots, \exp(-b_I)\}$, the ESF of order $r$ is:

$$\gamma_r = \sum_{\mathcal{S} \subseteq \{1,\ldots,I\}, |\mathcal{S}|=r} \prod_{i \in \mathcal{S}} \exp(-b_i)$$

Computed recursively (Gustafsson, 1980):
- $\gamma_0^{(k)} = 1$ for all $k$
- $\gamma_r^{(k)} = \gamma_r^{(k-1)} + \exp(-b_k) \cdot \gamma_{r-1}^{(k-1)}$

where $\gamma_r^{(k)}$ uses items $1, \ldots, k$. Cost: $O(I^2)$ per examinee score group.

### Gradient

$$\frac{\partial \ell_C}{\partial b_i} = -n_i + \sum_r N_r \frac{\gamma_r^{(-i)}}{\gamma_r}$$

where $n_i$ is the number of correct responses to item $i$, $N_r$ is the number of examinees with total score $r$, and $\gamma_r^{(-i)}$ is the ESF of order $r$ computed without item $i$ (obtained via "backward elimination" from the full ESF table).

### Identifiability

The model is only identified up to a location shift. One constraint is needed:
- Fix $\sum_i b_i = 0$ (zero-mean), OR
- Fix one item as the anchor.

### Exclusions

Examinees with perfect scores ($r_j = I$) or zero scores ($r_j = 0$) are excluded because $\gamma_I$ and $\gamma_0$ don't contribute information about difficulty differences.

## Mock implementation

```python
# src/catsim/calibration/cml.py
"""Conditional Maximum Likelihood for Rasch model (Andersen, 1972)."""

import numpy as np
import numpy.typing as npt
from scipy import optimize


class CMLRaschCalibrator:
    """Estimate Rasch difficulty parameters via conditional ML.

    Parameters
    ----------
    max_iter : int
        Maximum Newton-Raphson iterations. Default 100.
    tol : float
        Convergence tolerance. Default 1e-4.
    """

    def __init__(self, max_iter: int = 100, tol: float = 1e-4) -> None:
        self.max_iter = max_iter
        self.tol = tol

    def calibrate(
        self,
        responses: npt.NDArray[np.integer],
    ) -> dict:
        """Estimate Rasch difficulty parameters.

        Parameters
        ----------
        responses : numpy.ndarray, shape (n_examinees, n_items)
            Binary response matrix (complete design, no missing data).

        Returns
        -------
        dict with keys:
            'items': numpy.ndarray, shape (n_items, 4)
                [1.0, b_i, 0.0, 1.0] for each item.
            'converged': bool
            'n_iter': int
        """
        n_examinees, n_items = responses.shape

        # Total scores
        scores = responses.sum(axis=1)
        # Exclude perfect and zero scores
        valid = (scores > 0) & (scores < n_items)
        responses_valid = responses[valid]
        scores_valid = scores[valid]

        # Score group frequencies
        score_freq = np.bincount(scores_valid, minlength=n_items + 1)

        # Item correct counts (from valid examinees only)
        n_correct = responses_valid.sum(axis=0).astype(float)

        # Initialize b
        p_obs = n_correct / len(responses_valid)
        p_obs = np.clip(p_obs, 0.01, 0.99)
        b = -np.log(p_obs / (1 - p_obs))
        b = b - b.mean()  # center

        result = optimize.minimize(
            self._neg_cond_ll,
            b[:-1],  # n_items - 1 free params (last is determined by sum=0)
            args=(n_items, n_correct, score_freq),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": self.max_iter, "ftol": self.tol},
        )

        b_free = result.x
        b_last = -b_free.sum()  # sum-to-zero constraint
        b_final = np.append(b_free, b_last)

        items = np.column_stack([
            np.ones(n_items),
            b_final,
            np.zeros(n_items),
            np.ones(n_items),
        ])

        return {
            "items": items,
            "converged": result.success,
            "n_iter": result.nit,
        }

    @staticmethod
    def _esf(exp_neg_b):
        """Compute elementary symmetric functions recursively.

        Parameters
        ----------
        exp_neg_b : numpy.ndarray, shape (n_items,)
            exp(-b) for each item.

        Returns
        -------
        numpy.ndarray, shape (n_items + 1,)
            gamma[r] = ESF of order r.
        """
        n = len(exp_neg_b)
        gamma = np.zeros(n + 1)
        gamma[0] = 1.0
        for k in range(n):
            # Process in reverse to avoid using updated values
            for r in range(min(k + 1, n), 0, -1):
                gamma[r] += exp_neg_b[k] * gamma[r - 1]
        return gamma

    @staticmethod
    def _neg_cond_ll(b_free, n_items, n_correct, score_freq):
        """Negative conditional log-likelihood and gradient."""
        b_last = -b_free.sum()
        b = np.append(b_free, b_last)
        exp_neg_b = np.exp(-b)

        # ESF
        n = len(b)
        gamma = np.zeros(n + 1)
        gamma[0] = 1.0
        for k in range(n):
            for r in range(min(k + 1, n), 0, -1):
                gamma[r] += exp_neg_b[k] * gamma[r - 1]

        # Log-likelihood
        ll = -np.sum(n_correct * b)
        for r in range(1, n):
            if score_freq[r] > 0:
                ll -= score_freq[r] * np.log(gamma[r] + 1e-30)

        # Gradient: d(ll)/d(b_i) = -n_i + sum_r N_r * gamma_r^{-i} / gamma_r
        # gamma_r^{-i} via backward elimination:
        #   gamma_r^{-i} = gamma_r - exp(-b_i) * gamma_{r-1}^{-i}
        grad = np.zeros(n)
        for i in range(n):
            gamma_without_i = np.zeros(n + 1)
            gamma_without_i[0] = 1.0
            for r in range(1, n + 1):
                gamma_without_i[r] = gamma[r] - exp_neg_b[i] * gamma_without_i[r - 1]

            grad[i] = -n_correct[i]
            for r in range(1, n):
                if score_freq[r] > 0 and gamma[r] > 0:
                    grad[i] += score_freq[r] * gamma_without_i[r] / gamma[r]

        # Transform gradient to free parameters (first n-1)
        # b_last = -sum(b_free), so d/d(b_free_k) = grad[k] - grad[n-1]
        grad_free = grad[:-1] - grad[-1]

        return -ll, -grad_free
```

## Proposed location

`src/catsim/calibration/cml.py`. Re-export from `src/catsim/calibration/__init__.py`.

## Tests

1. **Recovery test**: Generate 30 items, 500 examinees from a Rasch model. Assert RMSE(b) < 0.15 and correlation $r > 0.98$.
2. **Sum-to-zero**: Verify `np.sum(b) ≈ 0` (within 1e-6).
3. **Perfect/zero exclusion**: Examinees with all-correct/all-incorrect should be silently excluded. Adding more of them should not change the estimates.
4. **CML vs MML-EM (1PL)**: On the same Rasch data, CML and MML-EM with `model="1PL"` should produce highly correlated difficulty estimates ($r > 0.99$).
5. **ESF correctness**: For a small example (3 items), verify the ESF values match hand computation. E.g., with $\exp(-b) = [2, 3, 5]$: $\gamma_0 = 1$, $\gamma_1 = 10$, $\gamma_2 = 31$, $\gamma_3 = 30$.
6. **Gradient check**: Numerical gradient of the conditional log-likelihood matches the analytic gradient.

## Notes

- **1PL only**: CML is mathematically inapplicable to 2PL/3PL/4PL. The sufficient statistic property only holds for the Rasch model.
- **Complete designs only**: The current implementation assumes all examinees respond to all items. Extending to incomplete designs requires score-group-specific ESF computation, which is significantly more complex.
- **Niche value**: In a package that supports 4PL, CML's restriction to 1PL limits its practical use. Its main value is as a distribution-free alternative (no assumption on $g(\theta)$) for Rasch practitioners.
- **Numerical stability**: For large test lengths ($I > 50$), the ESFs can overflow. Use log-space computation or normalized ratios if this becomes an issue.
