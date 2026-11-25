**Status:** Planned
**Module:** `catsim.selection.kl.KLPSelector` (planned)
**Reference:** Chang and Ying (1996)
**Last validated:** 2026-11-23, against Chang & Ying (1996); Magis & Barrada (2017); catR 3.17 (`KL.R`, `nextItem.R`).
**Implementation guidance:** updated to the 1.0 strategy contract (`StepContext` in) on 2026-11-24.

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
2. The KL kernel still depends on the current estimate, the context's `theta`, as the reference point inside the divergence.
3. The full KL table cannot be precomputed across steps because it changes with the current estimate.
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

**Proposed location:** `src/catsim/selection/kl.py`, next to `KLSelector`.

**Mock implementation:**

```python
# src/catsim/selection/kl.py (added to the existing module)

import numpy

from .. import irt
from ..context import StepContext
from ..estimation import LogPrior, QuadratureGrid, normal_log_prior, posterior
from ..typing import FloatArray
from .base import BaseSelector, below_cap, check_r_max, pick_best, remaining


def _kl_divergences(nodes: FloatArray, theta_hat: float, items: FloatArray) -> FloatArray:
  """Return the divergence K_i(theta_q || theta_hat), one row per node and one column per item.

  Returns
  -------
  numpy.ndarray
      The divergences, of shape ``(len(nodes), len(items))``.
  """
  log_p_hat, log_q_hat = irt.log_response_probabilities(theta_hat, items)
  log_p, log_q = irt.log_response_probabilities(nodes, items)
  return numpy.exp(log_p_hat) * (log_p_hat - log_p) + numpy.exp(log_q_hat) * (log_q_hat - log_q)


class KLPSelector(BaseSelector):
  """Posterior Kullback-Leibler item selector.

  Each candidate's KL divergence from the current estimate is summed over a quadrature grid,
  weighted by the posterior of the responses so far, and the item with the largest sum is selected.

  Parameters
  ----------
  grid : QuadratureGrid or None, optional
      Grid on which the posterior is computed. Default is ``QuadratureGrid.uniform()``.
  log_prior : LogPrior or None, optional
      Vectorized log-density of the prior. Default is ``normal_log_prior()``, the standard normal.
  r_max : float, optional
      Maximum exposure rate for items. Items with exposure rates >= r_max are only selected when no
      other item remains. Default is 1 (no restriction).

  Raises
  ------
  ValueError
      If r_max is not in (0, 1].
  """

  def __init__(self, grid: QuadratureGrid | None = None, log_prior: LogPrior | None = None, r_max: float = 1.0) -> None:
    # Kept as given for the repr, as EAPEstimator does: the default grid's repr would list every
    # node, and a prior function has no source form.
    self._grid_argument = grid
    self._log_prior_argument = log_prior
    self._grid = grid if grid is not None else QuadratureGrid.uniform()
    self._log_prior = log_prior if log_prior is not None else normal_log_prior()
    self._r_max = check_r_max(r_max)

  def __repr__(self) -> str:
    """Return the selector as a constructor call that names every argument, with None for a default."""
    return (
      f"{type(self).__name__}(grid={self._grid_argument!r}, log_prior={self._log_prior_argument!r}, "
      f"r_max={self._r_max!r})"
    )

  def select(self, context: StepContext) -> int:
    """Return the available item with the largest posterior-weighted KL divergence from the current estimate.

    Parameters
    ----------
    context : StepContext
        The session to select for.

    Returns
    -------
    int
        Index of the selected item.
    """
    candidates = below_cap(remaining(context), context.exposure_rates, self._r_max)
    item_bank = context.item_bank
    post = posterior(context.responses, item_bank.get_items(context.administered_items), self._grid, self._log_prior)
    divergences = _kl_divergences(self._grid.nodes, context.theta, item_bank.get_items(candidates))
    return pick_best(candidates, (post[:, None] * divergences).sum(axis=0), context.rng)
```

**Implementation notes:**

- The posterior comes from the context's responses at every call: estimators keep no per-session
  posterior ({doc}`ADR-0007 </adr/0007-strategy-contracts>`), and `posterior()` is one
  vectorized pass over the grid. With no responses it is the prior on the grid.
- The reference point of the divergence is the context's `theta`, the engine's current estimate
  from whichever estimator it runs, not the posterior mean; the definition's $\hat\theta$ is the
  current estimate.
- The divergence table depends on the current estimate, so it is computed for the candidates at
  every call and not cached (Contract 3).
- The log-probabilities come from `irt.log_response_probabilities`, which `KLSelector` also uses.
  `KLSelector._kl_global` computes the same divergence at its Gauss-Legendre nodes: share
  `_kl_divergences` with it rather than keeping two copies.
- The score sums over the nodes rather than using `post @ divergences`, as `KLSelector` and
  `MLWISelector` do: a matrix product can round identical columns differently, which breaks the
  exact ties of {doc}`ADR-0017 </adr/0017-random-tie-breaking>`.
- The selection and estimation layers do not import each other (`[tool.importlinter]` in
  `pyproject.toml`). Add `"catsim.selection.kl -> catsim.estimation"` to that contract's
  `ignore_imports`, as `MLWISelector` does for the same helpers, or move the grid and posterior
  helpers to a lower layer.
- The default grid is catsim's (121 nodes on $[-6, 6]$, equal weights). catR's 33 nodes on
  $[-4, 4]$ with the trapezoidal rule are a `QuadratureGrid` with trapezoid weights, for
  comparisons with catR.

**Testing notes:**

- Compare one score with a brute-force reference: integrate $K_i(\theta \,\|\, \hat\theta)$ times the
  normalized posterior density with `scipy.integrate.quad` for a short response pattern, and check
  that the selected item is the one with the largest integral.
- Contract 2: two contexts that differ only in `theta` select different items on a bank built for
  it.
- Contract 4: with a narrow posterior centered at the context's `theta` (a long response pattern),
  the selection approaches `MaxInfoSelector`'s, because the divergence near $\hat\theta$ is about
  $I_i(\hat\theta)(\theta - \hat\theta)^2/2$; with no responses it differs from it on a bank built
  for it.
- Contract 5: a cap that leaves some candidates, and one that leaves none, falling back to all
  remaining items.
