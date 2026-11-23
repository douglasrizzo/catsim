**Status:** Planned
**Module:** `catsim.selection.kl.KLPSelector` (planned)
**Reference:** Chang and Ying (1996)
**Last validated:** 2026-11-23, against Chang & Ying (1996); Magis & Barrada (2017); catR 3.17 (`KL.R`, `nextItem.R`).
**Implementation guidance:** not yet updated to the 1.0 strategy contract (M22).

# Posterior Kullback-Leibler Selector

## Motivation

The KL selector integrates a KL divergence uniformly over a window around the
current estimate. KLP replaces that uniform window with the current posterior,
which removes the need for a hand-tuned half-width schedule and lets the prior
and the responses determine where the weight lies. Chang and Ying (1996) allow
any probability measure on the ability scale in place of the window (p. 221,
Equation 21) and give the posterior as the Bayesian choice (p. 222).

This makes KLP a Bayesian refinement of KL that behaves broadly early in the
test, while the posterior is wide, and concentrates automatically as it narrows.

## Definition

For candidate item $i$, KLP is Chang and Ying's Bayesian information index
(1996, p. 222, Equation 22)

$$
\mathrm{KLP}_i
=
\int K_i(\theta \,\|\, \hat{\theta})\, p(\theta \mid \mathbf{u})\, d\theta,
$$

where $\hat{\theta}$ is the current ability estimate,
$p(\theta \mid \mathbf{u})$ is the posterior given the responses so far, and
$K_i$ is the item's Kullback-Leibler divergence (p. 217, Definition 2.1 and
Equation 12)

$$
K_i(\theta \,\|\, \hat{\theta})
=
P_i(\hat{\theta}) \log \frac{P_i(\hat{\theta})}{P_i(\theta)}
+
[1-P_i(\hat{\theta})]
\log \frac{1-P_i(\hat{\theta})}{1-P_i(\theta)}.
$$

Chang and Ying print the integrand of Equations 21 and 22 as $K^{(n)}$, the
divergence of the whole test (p. 217, Equation 13). Equation 21 contains the
single-item index of Equation 17 as a special case (p. 221), so the item
divergence $K_i$ is meant; either reading selects the same item, because the
administered items' divergence is the same for every candidate.

Chang and Ying judged the posterior too expensive to update during a test and
approximated it by a normal density centered at $\hat{\theta}$ with variance
$1/I^{(n)}(\hat{\theta})$ (p. 222, Equations 23 and 24). catsim computes the
posterior on a quadrature grid $\{\theta_q\}_{q=1}^{Q}$ instead, with
normalized posterior probabilities $p_q = p(\theta_q \mid \mathbf{u})$, and
approximates the integral by

$$
\mathrm{KLP}_i
\approx
\sum_{q=1}^{Q} K_i(\theta_q \,\|\, \hat{\theta})\, p_q.
$$

The rule selects the non-administered item with the largest KLP score.

`catR` implements this criterion as `KLP`
([`KL.R`](https://github.com/cran/catR/blob/master/R/KL.R), called from
[`nextItem.R`](https://github.com/cran/catR/blob/master/R/nextItem.R); Magis
& Barrada, 2017, p. 4, Equation 7, and p. 5, Equation 8). It uses the same
divergence, with the current estimate as the reference point, and the same
posterior weighting. It departs in the numerics: it integrates the divergence
times the likelihood times the prior density, without normalizing the
posterior, by the trapezoidal rule on 33 equally spaced nodes over $[-4, 4]$
(`parInt = c(-4, 4, 33)`). Its default prior is standard normal, as is
catsim's; catsim's default grid has 121 equally weighted nodes over $[-6, 6]$
and normalizes the posterior. The normalizing constant does not change the
ranking; the grid and its range change the approximation slightly. Magis and
Barrada's Equations 7 and 9 print $\arg\min$; `nextItem.R` selects the largest
value. `catR`'s `KL` criterion is the same integral without the prior (p. 5,
Equation 9), not the window of catsim's KL selector.

## Parameters

```{eval-rst}
.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``grid``
     - ``QuadratureGrid.uniform()``
     - Quadrature grid used for posterior weighting.
   * - ``log_prior``
     - ``normal_log_prior(0, 1)``
     - Prior used when the selector computes the posterior itself.
   * - ``r_max``
     - ``1.0``
     - Soft exposure cap.
```

## Behavioral Contracts

Any correct implementation must satisfy the following properties:

1. KLP uses the posterior over theta, not a uniform window, as its integration weight.
2. The KL kernel still depends on `est_theta` as the reference point inside the divergence.
3. The full KL table cannot be precomputed across steps because it changes with `est_theta`.
4. With a diffuse prior and little response information, the selector behaves more globally; with a concentrated posterior, it becomes increasingly local.
5. Exposure filtering applies before the final argmax, with fallback to the unrestricted remaining pool.

## References

Chang, H.-H., & Ying, Z. (1996). A global information approach to computerized adaptive testing. *Applied Psychological Measurement*, 20(3), 213-229. <https://doi.org/10.1177/014662169602000303>. Open copy: [University of Minnesota Digital Conservancy](https://hdl.handle.net/11299/119092).

Magis, D., & Barrada, J. R. (2017). Computerized adaptive testing with R: Recent updates of the package catR. *Journal of Statistical Software*, 76(Code Snippet 1), 1-19. <https://doi.org/10.18637/jss.v076.c01> (open access).

## Implementation Guidance

:::{note}
This section is for contributors and agents implementing this technique.
It is removed when the spec is promoted to "Implemented" status.
:::

**Proposed location:** `src/catsim/selection/kl.py`

**Mock implementation:**

```python
# src/catsim/selection/kl.py  (alongside KLSelector from card 03)
"""Posterior Kullback-Leibler selector."""

import numpy

from .._typing import FloatArray
from ..estimation.bayesian import (
  LogPrior,
  QuadratureGrid,
  normal_log_prior,
  posterior,
)
from ..exceptions import NoItemsAvailableError
from ..item_bank import ItemBank
from .base import BaseSelector
from .kl import _kl_integrand  # reuse the kernel from card 03


class KLPSelector(BaseSelector):
  """Posterior-weighted Kullback-Leibler selector.

  Parameters
  ----------
  grid : QuadratureGrid or None
      Quadrature grid. Defaults to ``QuadratureGrid.uniform()``.
  log_prior : callable or None
      Vectorized log-density of the prior. Defaults to N(0, 1).
  r_max : float
      Exposure cap.
  """

  def __init__(
    self,
    grid: QuadratureGrid | None = None,
    log_prior: LogPrior | None = None,
    r_max: float = 1.0,
  ) -> None:
    if not 0 <= r_max <= 1:
      raise ValueError(f"r_max must be between 0 and 1, got {r_max}")
    super().__init__()
    self._grid = grid if grid is not None else QuadratureGrid.uniform()
    self._log_prior = log_prior if log_prior is not None else normal_log_prior()
    self._r_max = float(r_max)

  def __str__(self) -> str:
    return "Posterior Kullback-Leibler Selector"

  # ---- core kernel ----
  def _kl_table(self, item_bank: ItemBank, theta_hat: float) -> FloatArray:
    """(n_items, Q) matrix of KL(theta || theta_hat) at every grid node.

    Unlike MPWI's info table, this matrix is theta-hat dependent, so it cannot
    be precomputed once per item bank -- rebuild each step.
    """
    nodes = self._grid.nodes
    out = numpy.empty((item_bank.n_items, nodes.size), dtype=float)
    for i in range(item_bank.n_items):
      a, b, c, d = item_bank.items[i, :4]
      for q, t in enumerate(nodes):
        out[i, q] = _kl_integrand(float(t), theta_hat, a, b, c, d)
    return out

  def _get_posterior(
    self,
    item_bank: ItemBank,
    administered_items: list[int],
    response_vector: list[bool],
  ) -> FloatArray:
    if not administered_items:
      pp = self._log_prior(self._grid.nodes) + numpy.log(self._grid.weights)
      pp -= pp.max()
      pp = numpy.exp(pp)
      return pp / pp.sum()
    items = item_bank.items[administered_items, :4]
    return posterior(response_vector, items, self._grid, self._log_prior)

  def select(
    self,
    item_bank: ItemBank,
    administered_items: list[int],
    est_theta: float,
    rng: numpy.random.Generator | None = None,  # noqa: ARG002
    exposure_rates: FloatArray | None = None,
    response_vector: list[bool] | None = None,
  ) -> int | None:
    if response_vector is None:
      raise ValueError("KLPSelector requires the response_vector to compute the posterior.")

    pp = self._get_posterior(item_bank, administered_items, response_vector)
    kl_table = self._kl_table(item_bank, est_theta)
    # KLP_i = sum_q KL(theta_q || theta_hat)_i * p(theta_q | u)
    scores = kl_table @ pp

    candidate_mask = numpy.ones(item_bank.n_items, dtype=bool)
    candidate_mask[administered_items] = False
    if exposure_rates is None:
      exposure_rates = numpy.zeros(item_bank.n_items, dtype=float)
    pool = candidate_mask & (exposure_rates < self._r_max)
    if not pool.any():
      pool = candidate_mask
    if not pool.any():
      raise NoItemsAvailableError("There are no more items to apply.")

    candidates = numpy.flatnonzero(pool)
    return int(candidates[int(scores[candidates].argmax())])
```

**Implementation notes:**

- Unlike MPWI, the per-step KL table depends on `est_theta` and cannot be cached once per bank.
- This selector belongs with the KL selector because they share the KL kernel.
- Compute the posterior from the session's responses with `posterior()`. Estimators keep no per-session posterior ({doc}`ADR-0003 </adr/0003-runtime-state-ownership>`), and one vectorized call over the grid is cheap.

**Dependencies:**

- Planned card 01 (`Bayesian Estimation Infrastructure`)
- Planned card 03 (`KLSelector`) for kernel reuse

**Testing notes:**

- Compare against broad-window KL selection under diffuse priors.
- Verify exposure-cap fallback behavior.
- Confirm that the selector still uses `est_theta` as the KL reference point.
