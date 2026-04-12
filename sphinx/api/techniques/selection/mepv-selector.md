**Status:** Planned
**Module:** `catsim.selection.weighted_info.MEPVSelector` (planned)
**Reference:** van der Linden (1998); Choi and Swartz (2009)

# Minimum Expected Posterior Variance Selector

## Motivation

MEPV is the decision-theoretic sibling of MEI. Rather than maximizing expected
information after the next response, it minimizes the expected posterior
variance, which directly targets squared-error loss under the posterior.

This makes it conceptually attractive for pure ability estimation, even though
its empirical gains over MEI and MPWI are often modest.

## Definition

Let $p(\theta \mid \mathbf{u})$ be the current posterior and let

$$
\tilde{P}_i
=
\int P_i(\theta)\, p(\theta \mid \mathbf{u})\, d\theta
$$

be the posterior-predictive probability of a correct response to candidate item
$i$. The criterion is

$$
\mathrm{MEPV}_i
=
\tilde{P}_i\, \mathrm{Var}[\theta \mid \mathbf{u}, U_i=1]
+
(1-\tilde{P}_i)\, \mathrm{Var}[\theta \mid \mathbf{u}, U_i=0].
$$

The selector chooses the item with the smallest expected posterior variance.

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
     - Quadrature grid used for posterior approximation.
   * - ``log_prior``
     - ``normal_log_prior(0, 1)``
     - Prior used when the selector computes the posterior itself.
   * - ``estimator``
     - ``None``
     - Optional EAP estimator whose cached posterior is reused.
   * - ``r_max``
     - ``1.0``
     - Soft exposure cap.
```

## Behavioral Contracts

Any correct implementation must satisfy the following properties:

1. Each candidate's score is an expectation over two hypothetical response-conditioned posterior variances.
2. The rule is an argmin criterion, not an argmax criterion.
3. The selector can cache the item-response probability table on the quadrature grid because it is bank-static for a fixed grid.
4. The selector requires access to the response vector unless a reusable posterior is already available.
5. Exposure filtering applies before choosing the minimum-scoring candidate, with fallback to the unrestricted remaining pool.

## References

van der Linden, W. J. (1998). Bayesian item selection criteria for adaptive testing. *Psychometrika*, 63(2), 201-216. <https://doi.org/10.1007/BF02294772>

Choi, S. W., & Swartz, R. J. (2009). Comparison of CAT criteria for polytomous items. *Applied Psychological Measurement*, 33(6), 419-440. <https://doi.org/10.1177/0146621608327800>

## Implementation Guidance

:::{note}
This section is for contributors and agents implementing this technique.
It is removed when the spec is promoted to "Implemented" status.
:::

**Proposed location:** `src/catsim/selection/weighted_info.py`

**Mock implementation:**

```python
# src/catsim/selection/weighted_info.py  (alongside MLWISelector and MPWISelector)
"""Minimum Expected Posterior Variance selector (van der Linden, 1998)."""

import numpy
import numpy.typing as npt

from .. import irt
from ..estimation.bayesian import (
  LogPrior,
  QuadratureGrid,
  normal_log_prior,
  posterior,
  posterior_variance,
)
from ..exceptions import NoItemsAvailableError
from ..item_bank import ItemBank
from .base import BaseSelector

FloatArray = npt.NDArray[numpy.floating]


class MEPVSelector(BaseSelector):
  """Minimum Expected Posterior Variance selector.

  Parameters
  ----------
  grid : QuadratureGrid or None
      Quadrature grid. Defaults to ``QuadratureGrid.uniform()``.
  log_prior : callable or None
      Prior log-density. Defaults to N(0, 1).
  estimator : EAPEstimator or None
      If provided, the selector reuses the estimator's ``last_posterior`` instead
      of recomputing the quadrature.
  r_max : float
      Exposure cap.
  """

  def __init__(
    self,
    grid: QuadratureGrid | None = None,
    log_prior: LogPrior | None = None,
    estimator=None,  # type: EAPEstimator | None
    r_max: float = 1.0,
  ) -> None:
    if not 0 <= r_max <= 1:
      raise ValueError(f"r_max must be between 0 and 1, got {r_max}")
    super().__init__()
    self._grid = grid if grid is not None else QuadratureGrid.uniform()
    self._log_prior = log_prior if log_prior is not None else normal_log_prior()
    self._estimator = estimator
    self._r_max = float(r_max)

  def __str__(self) -> str:
    return "Minimum Expected Posterior Variance Selector"

  # ---- core kernel ----
  def _p_table(self, item_bank: ItemBank) -> FloatArray:
    """(n_items, Q) matrix of item response probabilities at each grid node."""
    cache_key = f"mepv:p_table:{id(self._grid)}"
    cached = item_bank._selector_cache.get(cache_key)
    if cached is not None:
      return cached
    nodes = self._grid.nodes
    table = numpy.array([
      [irt.icc(float(t), *item_bank.items[i, :4]) for t in nodes]
      for i in range(item_bank.n_items)
    ])
    item_bank._selector_cache[cache_key] = table
    return table

  def _get_posterior(
    self,
    item_bank: ItemBank,
    administered_items: list[int],
    response_vector: list[bool],
  ) -> FloatArray:
    if self._estimator is not None and self._estimator.last_posterior is not None:
      return self._estimator.last_posterior
    if not administered_items:
      lp = self._log_prior(self._grid.nodes) + numpy.log(self._grid.weights)
      lp -= lp.max()
      pp = numpy.exp(lp)
      return pp / pp.sum()
    items = item_bank.items[administered_items, :4]
    return posterior(response_vector, items, self._grid, self._log_prior)

  @staticmethod
  def _reweighted_variance(
    current_post: FloatArray,
    nodes: FloatArray,
    p_grid: FloatArray,  # shape (Q,) -- P_i(theta_q)
    response: bool,
  ) -> float:
    """Posterior variance after reweighting by the hypothetical response."""
    likelihood = p_grid if response else (1.0 - p_grid)
    unnorm = current_post * likelihood
    total = unnorm.sum()
    if total <= 0:
      return float("inf")
    new_post = unnorm / total
    return posterior_variance(new_post, nodes)

  def select(
    self,
    item_bank: ItemBank,
    administered_items: list[int],
    est_theta: float,  # noqa: ARG002 -- posterior carries the ability info
    rng: numpy.random.Generator | None = None,  # noqa: ARG002
    exposure_rates: npt.NDArray[numpy.floating] | None = None,
    response_vector: list[bool] | None = None,
  ) -> int | None:
    if response_vector is None:
      raise ValueError("MEPVSelector requires the response_vector to compute the posterior.")

    current_post = self._get_posterior(item_bank, administered_items, response_vector)
    p_table = self._p_table(item_bank)
    nodes = self._grid.nodes

    # Predicted response probability for each item under the current posterior
    p_tilde = p_table @ current_post  # shape (n_items,)

    # Expected posterior variance under each candidate's two possible responses
    epv = numpy.empty(item_bank.n_items, dtype=float)
    for i in range(item_bank.n_items):
      var_correct = self._reweighted_variance(current_post, nodes, p_table[i], response=True)
      var_wrong = self._reweighted_variance(current_post, nodes, p_table[i], response=False)
      epv[i] = p_tilde[i] * var_correct + (1 - p_tilde[i]) * var_wrong

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
    # MEPV = *minimum* expected posterior variance
    return int(candidates[int(epv[candidates].argmin())])
```

**Implementation notes:**

- Cache the response-probability table on the quadrature grid because it is bank-static.
- This selector is structurally close to MPWI and belongs in the same module.
- Runtime is still only `O(n_items * Q)` with a somewhat larger constant than MPWI.

**Dependencies:**

- Planned card 01 (`Bayesian Estimation Infrastructure`)
- Recommended reuse of planned card 06 (`EAPEstimator`)

**Testing notes:**

- Identical items should yield equal expected posterior variances.
- Banks with a clear posterior-precision winner should produce a stable argmin.
- Validate posterior reuse and exposure-cap fallback behavior.
