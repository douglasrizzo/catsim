---
title: MML-EM calibrator for 1PL and 2PL
kind: technique
status: parked
priority: 4
effort: M
blocked_by: [scope/calibration/icc-gradients]
raised_by: []
---

# MML-EM calibrator for 1PL and 2PL

> Background: [the literature review §2.2](literature-review.md)

## Reference

Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of item parameters: Application of an EM algorithm. *Psychometrika*, 46(4), 443–459.

Baker, F. B., & Kim, S.-H. (2004). *Item Response Theory: Parameter Estimation Techniques* (2nd ed.). Marcel Dekker. Chapters 7–9.

## Theory

Marginal Maximum Likelihood (MML) integrates out examinee ability using a prior distribution $g(\theta)$, typically $N(0,1)$. This avoids the Neyman-Scott inconsistency problem of JML. The EM algorithm iterates:

### E-step

At each quadrature node $X_q$ ($q = 1, \ldots, Q$) with weight $A_q$ from Gauss-Hermite quadrature:

1. Compute the likelihood of examinee $j$'s response pattern at node $q$:

$$L_j(X_q) = \prod_{i \in \mathcal{A}_j} P_i(X_q)^{u_{ij}} (1 - P_i(X_q))^{1 - u_{ij}}$$

2. Compute posterior weight:

$$w_{jq} = \frac{L_j(X_q) \cdot A_q \cdot g(X_q)}{\sum_{q'} L_j(X_{q'}) \cdot A_{q'} \cdot g(X_{q'})}$$

3. Accumulate expected counts:

$$\bar{r}_{iq} = \sum_j w_{jq} \cdot u_{ij}$$
$$\bar{f}_q = \sum_j w_{jq}$$

where $\bar{r}_{iq}$ is the expected number of examinees at node $q$ who answer item $i$ correctly, and $\bar{f}_q$ is the expected total frequency at node $q$.

### M-step

For each item $i$, maximize the expected complete-data log-likelihood:

$$\ell_i = \sum_q \left[\bar{r}_{iq} \log P_i(X_q) + (\bar{f}_q - \bar{r}_{iq}) \log(1 - P_i(X_q))\right]$$

with respect to item parameters $(a_i, b_i)$ for 2PL, or $(b_i)$ only for 1PL (with $a$ fixed to a common value, typically 1.0).

Use `scipy.optimize.minimize(method='L-BFGS-B')` with bounds:
- $a \in [0.1, 5.0]$ (for 2PL)
- $b \in [-6.0, 6.0]$

Supply analytic gradients from icc-gradients's `icc_log_likelihood_gradient`.

### Convergence

Stop when:
- Maximum absolute parameter change $< 0.001$, OR
- Marginal log-likelihood change $< 10^{-4}$, OR
- Maximum iterations reached (default: 500).

### Quadrature

Use Gauss-Hermite quadrature with 40 nodes (default). The nodes and weights come from `numpy.polynomial.hermite.hermgauss(n)`. Note: Gauss-Hermite quadrature integrates w.r.t. $\exp(-x^2)$, so to integrate w.r.t. $N(0,1)$ the nodes must be scaled by $\sqrt{2}$ and weights divided by $\sqrt{\pi}$.

### Marginal log-likelihood

For monitoring convergence, the marginal log-likelihood is:

$$\log L = \sum_j \log \left(\sum_q L_j(X_q) \cdot A_q \cdot g(X_q)\right)$$

### 1PL as a constrained 2PL

The 1PL is a special case where all $a_i = a$ (a common discrimination, often fixed at 1.0). Implementation: in the M-step, fix `a` and only optimize `b`. Expose a `model` parameter: `"1PL"` or `"2PL"`.

## Mock implementation

```python
# src/catsim/calibration/__init__.py
"""Item parameter calibration for IRT models."""

from .mml_em import MMLEMCalibrator

__all__ = ["MMLEMCalibrator"]


# src/catsim/calibration/mml_em.py
"""Marginal Maximum Likelihood via EM (Bock & Aitkin, 1981)."""

from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy import optimize

from .. import irt


class MMLEMCalibrator:
    """Estimate item parameters via MML-EM for 1PL or 2PL models.

    Parameters
    ----------
    model : {"1PL", "2PL"}
        Which IRT model to fit.
    n_quad : int
        Number of Gauss-Hermite quadrature nodes. Default 40.
    max_iter : int
        Maximum EM iterations. Default 500.
    tol : float
        Convergence tolerance on max absolute parameter change. Default 0.001.
    """

    def __init__(
        self,
        model: Literal["1PL", "2PL"] = "2PL",
        n_quad: int = 40,
        max_iter: int = 500,
        tol: float = 1e-3,
    ) -> None:
        self.model = model
        self.n_quad = n_quad
        self.max_iter = max_iter
        self.tol = tol

    def calibrate(
        self,
        responses: npt.NDArray[np.integer],
    ) -> dict:
        """Estimate item parameters from a binary response matrix.

        Parameters
        ----------
        responses : numpy.ndarray, shape (n_examinees, n_items)
            Binary response matrix. 1 = correct, 0 = incorrect.
            NaN or -1 for not-administered items (sparse designs).

        Returns
        -------
        dict with keys:
            'items': numpy.ndarray, shape (n_items, 4)
                Estimated [a, b, c, d] for each item. c=0, d=1 for 1PL/2PL.
            'log_likelihood': list[float]
                Marginal log-likelihood at each iteration.
            'converged': bool
            'n_iter': int
        """
        n_examinees, n_items = responses.shape
        mask = np.isfinite(responses) & (responses >= 0)  # administered mask

        # Quadrature setup
        nodes, weights = np.polynomial.hermite.hermgauss(self.n_quad)
        nodes = nodes * np.sqrt(2)          # scale to N(0,1)
        weights = weights / np.sqrt(np.pi)  # normalize for N(0,1)

        # Initialize parameters
        a = np.ones(n_items)
        b = np.zeros(n_items)
        # Simple initial b: inverse-logit of proportion correct
        p_obs = np.nanmean(np.where(mask, responses, np.nan), axis=0)
        p_obs = np.clip(p_obs, 0.01, 0.99)
        b = -np.log(p_obs / (1 - p_obs))  # logit transform

        log_likelihoods = []

        for iteration in range(self.max_iter):
            items = np.column_stack([a, b, np.zeros(n_items), np.ones(n_items)])

            # === E-step ===
            r_bar, f_bar, marginal_ll = self._e_step(
                responses, mask, items, nodes, weights
            )
            log_likelihoods.append(marginal_ll)

            # === M-step ===
            a_new, b_new = self._m_step(items, nodes, r_bar, f_bar)

            # Check convergence
            if self.model == "1PL":
                max_change = np.max(np.abs(b_new - b))
            else:
                max_change = max(np.max(np.abs(a_new - a)), np.max(np.abs(b_new - b)))

            a, b = a_new, b_new

            if max_change < self.tol:
                break

        items_final = np.column_stack([a, b, np.zeros(n_items), np.ones(n_items)])
        return {
            "items": items_final,
            "log_likelihood": log_likelihoods,
            "converged": max_change < self.tol,
            "n_iter": iteration + 1,
        }

    def _e_step(self, responses, mask, items, nodes, weights):
        """Compute expected counts and marginal log-likelihood."""
        n_examinees, n_items = responses.shape
        n_quad = len(nodes)

        # P(X_q) for all items at all nodes: (n_quad, n_items)
        P = irt.icc_hpc(nodes[:, np.newaxis], items)  # broadcast

        r_bar = np.zeros((n_quad, n_items))
        f_bar = np.zeros(n_quad)
        marginal_ll = 0.0

        for j in range(n_examinees):
            admin = mask[j]
            u_j = responses[j, admin]

            # log-likelihood at each node: sum over administered items
            log_lik = np.sum(
                u_j * np.log(P[:, admin] + 1e-30)
                + (1 - u_j) * np.log(1 - P[:, admin] + 1e-30),
                axis=1,
            )  # (n_quad,)

            log_post = log_lik + np.log(weights + 1e-30)
            log_norm = np.logaddexp.reduce(log_post)
            w_j = np.exp(log_post - log_norm)  # (n_quad,) posterior weights

            marginal_ll += log_norm

            r_bar[:, admin] += w_j[:, np.newaxis] * u_j
            f_bar += w_j

        return r_bar, f_bar, marginal_ll

    def _m_step(self, items, nodes, r_bar, f_bar):
        """Optimize item parameters given expected counts."""
        n_items = items.shape[0]
        a_new = items[:, 0].copy()
        b_new = items[:, 1].copy()

        for i in range(n_items):
            if self.model == "1PL":
                # Only optimize b, keep a = 1
                x0 = [b_new[i]]
                bounds = [(-6.0, 6.0)]
            else:
                x0 = [a_new[i], b_new[i]]
                bounds = [(0.1, 5.0), (-6.0, 6.0)]

            result = optimize.minimize(
                self._neg_expected_ll_item,
                x0,
                args=(i, nodes, r_bar, f_bar, self.model),
                method="L-BFGS-B",
                bounds=bounds,
                jac=True,  # function returns (value, gradient)
            )

            if self.model == "1PL":
                b_new[i] = result.x[0]
            else:
                a_new[i], b_new[i] = result.x

        return a_new, b_new

    @staticmethod
    def _neg_expected_ll_item(params, item_idx, nodes, r_bar, f_bar, model):
        """Negative expected log-likelihood and gradient for a single item."""
        if model == "1PL":
            a, b = 1.0, params[0]
        else:
            a, b = params

        z = np.exp(-a * (nodes - b))
        psi = 1.0 / (1.0 + z)
        P = psi  # c=0, d=1
        Q = 1.0 - P

        r_i = r_bar[:, item_idx]
        f_q = f_bar

        # log-likelihood
        ll = np.sum(r_i * np.log(P + 1e-30) + (f_q - r_i) * np.log(Q + 1e-30))

        # gradients
        residual = (r_i - f_q * P) / (P * Q + 1e-30)
        psi_deriv = psi * (1.0 - psi)

        if model == "1PL":
            grad_b = np.sum(residual * (-a * psi_deriv))
            return -ll, np.array([-grad_b])
        else:
            grad_a = np.sum(residual * (nodes - b) * psi_deriv)
            grad_b = np.sum(residual * (-a * psi_deriv))
            return -ll, np.array([-grad_a, -grad_b])
```

## Proposed location

New subpackage `src/catsim/calibration/`:
- `__init__.py` — re-export `MMLEMCalibrator`
- `mml_em.py` — the calibrator class

## Tests

1. **Recovery test (2PL)**: Generate a response matrix from known 2PL parameters (20 items, 500 examinees). Run `MMLEMCalibrator(model="2PL")`. Assert recovered `a` and `b` correlate $r > 0.95$ with true values, and RMSE(b) < 0.3.
2. **Recovery test (1PL)**: Generate from Rasch parameters (a=1 for all). Run with `model="1PL"`. Assert RMSE(b) < 0.2.
3. **Convergence**: Verify `converged=True` and that `log_likelihood` is monotonically non-decreasing (EM guarantee).
4. **Sparse data**: Pass a response matrix with ~30% NaN (not-administered). Verify calibration still converges and parameters are reasonable.
5. **Known stationary point**: At the true parameters, the gradient should be near zero. Verify numerically.
6. **1PL vs 2PL**: When data is generated from a 1PL model, fitting 2PL should recover `a` values close to 1.0 for all items.
7. **Quadrature sensitivity**: Run with `n_quad=20` and `n_quad=40`. Results should agree within 0.05 on parameter estimates.

## Notes

- The E-step loops over examinees, which is $O(J \cdot Q \cdot I)$. For large datasets this is the bottleneck. A vectorized version (building a full $J \times Q$ posterior matrix) uses $O(J \cdot Q)$ memory but is much faster. Start with the loop; optimize in a follow-up if needed.
- The quadrature nodes from `hermgauss` integrate w.r.t. $\exp(-x^2)$, not $N(0,1)$. The scaling $x \to x\sqrt{2}$ and $w \to w/\sqrt{\pi}$ is essential — getting this wrong shifts all difficulty estimates.
- `c=0, d=1` are fixed for 1PL/2PL. The 3PL extension (mml-em-3pl) will relax the c constraint.
