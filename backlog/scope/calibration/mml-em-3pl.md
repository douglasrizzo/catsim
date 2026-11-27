---
title: MML-EM extension for 3PL
kind: technique
status: parked
priority: 4
effort: S
blocked_by: [scope/calibration/mml-em]
raised_by: []
---

# MML-EM extension for 3PL

> Background: [the literature review §2.2](literature-review.md)

## Reference

Bock, R. D., & Aitkin, M. (1981). *Psychometrika*, 46(4), 443–459.

Mislevy, R. J. (1986). Bayes modal estimation in item response models. *Psychometrika*, 51(2), 177–195.

Baker, F. B., & Kim, S.-H. (2004). *Item Response Theory: Parameter Estimation Techniques* (2nd ed.). Chapters 9–10.

## Theory

The 3PL adds the pseudo-guessing parameter $c_i$:

$$P_i(\theta) = c_i + \frac{1 - c_i}{1 + \exp(-a_i(\theta - b_i))}$$

The c-parameter is notoriously difficult to estimate via MML-EM:

1. **Boundary problem**: $c$ tends to drift toward 0 or its upper bound. The likelihood surface is very flat in the c-direction, especially with small samples.
2. **Identifiability**: With few low-ability examinees, the lower asymptote is poorly determined.

### Solutions

**Box constraints**: Bound $c \in [0.0, 0.35]$ in the M-step optimizer. This is the simplest approach and sufficient for most applications.

**Bayesian regularization (Mislevy, 1986)**: Add a $\text{Beta}(\alpha, \beta)$ prior penalty on $c$ to the expected log-likelihood:

$$\ell_i^{*} = \ell_i + (\alpha - 1) \log c_i + (\beta - 1) \log(1 - c_i)$$

Common choice: $\text{Beta}(5, 17)$ which has mean $\approx 0.23$ and concentrates $c$ in the plausible $[0.05, 0.35]$ range. The gradient contribution from the prior is:

$$\frac{\partial}{\partial c_i}\left[(\alpha-1)\log c_i + (\beta-1)\log(1-c_i)\right] = \frac{\alpha - 1}{c_i} - \frac{\beta - 1}{1 - c_i}$$

### M-step changes

The M-step for item $i$ now optimizes over $(a_i, b_i, c_i)$ with:
- $a \in [0.1, 5.0]$, $b \in [-6.0, 6.0]$, $c \in [0.0, 0.35]$
- The ICC probability is $P = c + (1-c) \cdot \psi$ instead of $P = \psi$
- Gradients from icc-gradients's `icc_derivatives` include `dP_dc = 1 - ψ`

The E-step is unchanged from mml-em.

## Mock implementation

```python
# src/catsim/calibration/mml_em.py  (extend existing MMLEMCalibrator)

# Add "3PL" to the model Literal type:
#   model: Literal["1PL", "2PL", "3PL"] = "2PL"

# In __init__, add prior parameters:
#   c_prior_alpha: float = 5.0
#   c_prior_beta: float = 17.0

# Modify calibrate() to initialize c:
#   c = np.full(n_items, 0.2) if self.model == "3PL" else np.zeros(n_items)

# Modify _m_step to include c in optimization for 3PL:
def _m_step_3pl(self, items, nodes, r_bar, f_bar):
    """M-step for 3PL: optimize (a, b, c) per item."""
    n_items = items.shape[0]
    a_new = items[:, 0].copy()
    b_new = items[:, 1].copy()
    c_new = items[:, 2].copy()

    for i in range(n_items):
        x0 = [a_new[i], b_new[i], c_new[i]]
        bounds = [(0.1, 5.0), (-6.0, 6.0), (0.0, 0.35)]

        result = optimize.minimize(
            self._neg_expected_ll_item_3pl,
            x0,
            args=(i, nodes, r_bar, f_bar, self.c_prior_alpha, self.c_prior_beta),
            method="L-BFGS-B",
            bounds=bounds,
            jac=True,
        )
        a_new[i], b_new[i], c_new[i] = result.x

    return a_new, b_new, c_new

@staticmethod
def _neg_expected_ll_item_3pl(params, item_idx, nodes, r_bar, f_bar, alpha, beta):
    """Negative expected log-likelihood + Beta prior on c, with gradient."""
    a, b, c = params

    z = np.exp(-a * (nodes - b))
    psi = 1.0 / (1.0 + z)
    P = c + (1.0 - c) * psi
    Q = 1.0 - P

    r_i = r_bar[:, item_idx]
    f_q = f_bar

    ll = np.sum(r_i * np.log(P + 1e-30) + (f_q - r_i) * np.log(Q + 1e-30))

    # Beta prior on c
    if c > 0 and c < 1:
        ll += (alpha - 1) * np.log(c) + (beta - 1) * np.log(1 - c)

    # Gradients
    residual = (r_i - f_q * P) / (P * Q + 1e-30)
    psi_deriv = psi * (1.0 - psi)

    grad_a = np.sum(residual * (1.0 - c) * (nodes - b) * psi_deriv)
    grad_b = np.sum(residual * (-(1.0 - c) * a * psi_deriv))
    grad_c = np.sum(residual * (1.0 - psi))

    # Prior gradient on c
    if c > 0 and c < 1:
        grad_c += (alpha - 1) / c - (beta - 1) / (1 - c)

    return -ll, np.array([-grad_a, -grad_b, -grad_c])
```

## Proposed location

Same file as mml-em: `src/catsim/calibration/mml_em.py`. Extend `MMLEMCalibrator` to accept `model="3PL"`.

## Tests

1. **Recovery test (3PL)**: Generate 20 items with `c ~ Uniform(0.05, 0.25)`, 1000 examinees. Assert recovered c values correlate $r > 0.7$ with true values (lower bar than a/b because c is harder).
2. **c stays in bounds**: With `c_bounds=(0.0, 0.35)`, no estimated c should exceed 0.35 or go below 0.0.
3. **Prior effect**: Compare estimates with and without the Beta prior. With prior, c values should be more concentrated around the prior mean.
4. **3PL reduces to 2PL**: When true c=0 for all items, 3PL estimates should recover c ≈ 0 (within 0.05).
5. **Log-likelihood monotonicity**: EM guarantee still holds — marginal LL should be non-decreasing across iterations.
6. **Gradient correctness**: Numerical gradient check for the 3PL M-step objective at random parameter values.

## Notes

- The Beta(5, 17) prior is from Mislevy (1986) and is the standard choice in `mirt` and `girth`. Make the prior parameters configurable.
- With small samples ($N < 500$), c estimates are unreliable even with the prior. Document this limitation.
- The 4PL extension (adding d-parameter) is deferred to a discussion card because d is even harder to estimate and rarely needed in practice.
