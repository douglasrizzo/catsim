---
title: ICC analytic gradients w.r.t. item parameters
kind: technique
status: parked
priority: 4
effort: S
blocked_by: []
raised_by: []
---

# ICC analytic gradients w.r.t. item parameters

> Background: [the literature review §3](literature-review.md)

## Reference

Baker, F. B., & Kim, S.-H. (2004). *Item Response Theory: Parameter Estimation Techniques* (2nd ed.). Marcel Dekker. Chapter 5 (MLE derivatives) and Appendix.

## Theory

The 4PL ICC is:

$$P(\theta) = c + \frac{d - c}{1 + \exp(-a(\theta - b))}$$

Let $z = \exp(-a(\theta - b))$ and $Q = 1 - P$. The logistic kernel is $\psi = \frac{1}{1 + z}$, so $P = c + (d - c)\psi$.

### First derivatives w.r.t. item parameters

$$\frac{\partial P}{\partial a} = (d - c) \cdot (\theta - b) \cdot \psi(1 - \psi)$$

$$\frac{\partial P}{\partial b} = -(d - c) \cdot a \cdot \psi(1 - \psi)$$

$$\frac{\partial P}{\partial c} = 1 - \psi$$

$$\frac{\partial P}{\partial d} = \psi$$

### Log-likelihood derivatives

For a single observation $u \in \{0, 1\}$:

$$\frac{\partial \log L}{\partial \xi} = \frac{u - P}{P \cdot Q} \cdot \frac{\partial P}{\partial \xi}$$

where $\xi \in \{a, b, c, d\}$.

### Second derivatives (Hessian diagonal)

For Newton-Raphson in the M-step, we need the observed or expected second derivatives. The expected information for parameter $\xi$ is:

$$\mathbb{E}\left[-\frac{\partial^2 \log L}{\partial \xi^2}\right] = \frac{1}{P \cdot Q} \left(\frac{\partial P}{\partial \xi}\right)^2$$

This is the Fisher information for a single item w.r.t. its own parameter — not to be confused with the Fisher information for $\theta$.

## Mock implementation

```python
# src/catsim/irt.py  (new functions alongside existing icc/inf functions)

import numpy as np
import numpy.typing as npt
from typing import Any


def icc_derivatives(
    theta: npt.NDArray[np.floating[Any]],
    items: npt.NDArray[np.floating[Any]],
) -> dict[str, npt.NDArray[np.floating[Any]]]:
    """Analytic first derivatives of the 4PL ICC w.r.t. item parameters.

    Parameters
    ----------
    theta : numpy.ndarray, shape (n_quad,) or (n_examinees,)
        Ability values (quadrature nodes or examinee estimates).
    items : numpy.ndarray, shape (n_items, 4+)
        Item parameter matrix [a, b, c, d, ...].

    Returns
    -------
    dict with keys 'P', 'dP_da', 'dP_db', 'dP_dc', 'dP_dd'
        Each value has shape (n_theta, n_items) via broadcasting.
    """
    a, b, c, d = items[:, 0], items[:, 1], items[:, 2], items[:, 3]
    # theta is (n_theta,), params are (n_items,) → broadcast to (n_theta, n_items)
    theta_2d = theta[:, np.newaxis]  # (n_theta, 1)

    z = np.exp(-a * (theta_2d - b))               # (n_theta, n_items)
    psi = 1.0 / (1.0 + z)                         # logistic kernel
    psi_deriv = psi * (1.0 - psi)                  # ψ(1 - ψ)

    P = c + (d - c) * psi
    dP_da = (d - c) * (theta_2d - b) * psi_deriv
    dP_db = -(d - c) * a * psi_deriv
    dP_dc = 1.0 - psi
    dP_dd = psi

    return {"P": P, "dP_da": dP_da, "dP_db": dP_db, "dP_dc": dP_dc, "dP_dd": dP_dd}


def icc_log_likelihood_gradient(
    theta: npt.NDArray[np.floating[Any]],
    items: npt.NDArray[np.floating[Any]],
    r_bar: npt.NDArray[np.floating[Any]],
    f_bar: npt.NDArray[np.floating[Any]],
) -> dict[str, npt.NDArray[np.floating[Any]]]:
    """Gradient of the expected log-likelihood for MML-EM M-step.

    After the E-step, the expected counts r̄_iq and f̄_q summarize the data.
    The gradient for item i at quadrature node q is:

        (r̄_iq - f̄_q * P_iq) / (P_iq * Q_iq) * dP/dξ

    summed over quadrature nodes.

    Parameters
    ----------
    theta : numpy.ndarray, shape (n_quad,)
        Quadrature nodes.
    items : numpy.ndarray, shape (n_items, 4+)
        Current item parameter estimates.
    r_bar : numpy.ndarray, shape (n_quad, n_items)
        Expected number of correct responses at each node for each item.
    f_bar : numpy.ndarray, shape (n_quad,)
        Expected frequency (total posterior weight) at each node.

    Returns
    -------
    dict with keys 'grad_a', 'grad_b', 'grad_c', 'grad_d'
        Each has shape (n_items,) — gradient summed over quadrature nodes.
    """
    derivs = icc_derivatives(theta, items)
    P = derivs["P"]                               # (n_quad, n_items)
    Q = 1.0 - P

    # residual weight: (r̄_iq - f̄_q * P_iq) / (P_iq * Q_iq)
    f_bar_2d = f_bar[:, np.newaxis]                # (n_quad, 1)
    residual = (r_bar - f_bar_2d * P) / (P * Q + 1e-30)

    grad = {}
    for param in ("a", "b", "c", "d"):
        grad[f"grad_{param}"] = np.sum(residual * derivs[f"dP_d{param}"], axis=0)

    return grad
```

## Proposed location

`src/catsim/irt.py`, alongside existing `icc`, `icc_hpc`, `inf`, `inf_hpc`. These are pure IRT functions, not tied to any calibration method.

## Tests

1. **Numerical gradient check**: For each parameter ξ ∈ {a, b, c, d}, compute `(P(ξ+ε) - P(ξ-ε)) / (2ε)` with `ε = 1e-5` and compare against the analytic derivative. Assert `|analytic - numerical| < 1e-6` across a grid of θ values and diverse item parameters.
2. **Shape consistency**: Verify output shapes are `(n_theta, n_items)` for various input sizes.
3. **Boundary behavior**: Items with `c=0, d=1` (2PL) should have `dP_dc = 1 - ψ` and `dP_dd = ψ`. Items with `a=1` (1PL) should have `dP_da = (θ - b) * ψ(1-ψ)`.
4. **Log-likelihood gradient**: Construct a small synthetic dataset where the true parameters are known. Verify the gradient is near zero at the true parameters (stationary point).
5. **Hessian diagonal**: Verify `1/(P*Q) * (dP/dξ)^2` against numerical second derivative of the log-likelihood.

## Notes

- Use pure numpy (not numexpr) for gradients — the overhead of `numexpr.evaluate` is not worth it for the M-step, which runs once per EM iteration, not per examinee.
- The `1e-30` floor in the residual denominator prevents division by zero at extreme θ values where `P ≈ 0` or `P ≈ 1`.
- These functions will be consumed by both MML-EM (mml-em) and JML (jml).
