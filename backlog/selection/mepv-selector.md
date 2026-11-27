---
title: Minimum Expected Posterior Variance Selector
kind: technique
status: ready
priority: 3
effort: M
blocked_by: []
raised_by: []
---

**Module:** `catsim.selection.weighted_info.MEPVSelector` (planned)
**Reference:** van der Linden (1998); Choi and Swartz (2009)
**Last validated:** 2026-11-23, against van der Linden (1996, RR 96-01); Choi & Swartz (2009); Magis & Raîche (2012); catR 3.17 (`EPV.R`, `eapEst.R`, `eapSem.R`, `nextItem.R`).
**Implementation guidance:** written against the strategy contract of ADR-0007 (`StepContext` in) on 2026-11-26.

# Minimum Expected Posterior Variance Selector

## Motivation

MEPV is the decision-theoretic sibling of MEI. Rather than maximizing expected
information after the next response, it minimizes the expected posterior
variance, which directly targets squared-error loss under the posterior. van
der Linden (1996, p. 8) traces it to Owen's empirical Bayes procedure, which
mentioned minimizing the preposterior risk under quadratic loss as an
alternative item selection rule.

This makes it conceptually attractive for pure ability estimation, even though
its empirical gains over MEI and MPWI are modest. In van der Linden's
simulation with two-parameter logistic items, its bias and mean squared error
did not differ in practice from those of MPWI and MEI at 5 and 10 items; at 20
and 30 items MEI fell behind, while MEPV and MPWI did not (pp. 18–19).

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

Choi and Swartz (2009, Equation 4) weight the two branches by the posterior
predictive probabilities, as here. van der Linden's 1996 report (p. 12,
Equation 14), which calls the criterion minimum predicted expected variance, and
`catR` (Magis & Raîche, 2012, p. 9, Table 1) use the response probabilities at
the current ability estimate instead; the report mentions replacing them by
expectations over posterior distributions as a possible variant (p. 13). Both
Equation 14 of the report and Equation 4 of Choi and Swartz print a maximum;
the criterion's name and purpose make it a minimum. From a Bayesian point of
view, van der Linden prefers this criterion to maximum expected information,
because the reciprocal of the information only approximates the posterior
variance in large samples (p. 12).

`catR`'s [`EPV.R`](https://github.com/cran/catR/blob/master/R/EPV.R), called
from [`nextItem.R`](https://github.com/cran/catR/blob/master/R/nextItem.R),
follows van der Linden: it weights the branches by $P_i(\hat{\theta})$ and
$1 - P_i(\hat{\theta})$ at the current estimate, and takes each branch's
posterior variance as the square of the posterior standard deviation from
[`eapSem.R`](https://github.com/cran/catR/blob/master/R/eapSem.R), around the
EAP estimate from [`eapEst.R`](https://github.com/cran/catR/blob/master/R/eapEst.R)
after the hypothetical response is appended. Those integrals use the
trapezoidal rule on 33 equally spaced nodes over $[-4, 4]$
(`parInt = c(-4, 4, 33)`), and the default prior is standard normal, as is
catsim's. catsim agrees on the branch variances, which it computes on its
default grid of 121 equally weighted nodes over $[-6, 6]$, and departs in the
branch weights, which can rank items differently.

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
   * - ``r_max``
     - ``1.0``
     - Soft exposure cap.
```

## Behavioral Contracts

Any correct implementation must satisfy the following properties:

1. Each candidate's score is an expectation over two hypothetical response-conditioned posterior variances.
2. The rule is an argmin criterion, not an argmax criterion.
3. The selector can cache the item-response probability table on the quadrature grid because it is bank-static for a fixed grid.
4. The scores depend on the context's responses, through the posterior, and not on its `theta`: the branches are weighted by posterior-predictive probabilities, not by the probabilities at the current estimate.
5. Exposure filtering applies before choosing the minimum-scoring candidate, with fallback to the unrestricted remaining pool.

## References

van der Linden, W. J. (1998). Bayesian item selection criteria for adaptive testing. *Psychometrika*, 63(2), 201-216. <https://doi.org/10.1007/BF02294775>. Not openly available; the 1996 report below preceded it.

van der Linden, W. J. (1996). *Bayesian item selection criteria for adaptive testing* (Research Report 96-01). Enschede: University of Twente. Open copy: [University of Twente](https://research.utwente.nl/files/247173933/Research_Report_96_01.pdf).

Choi, S. W., & Swartz, R. J. (2009). Comparison of CAT item selection criteria for polytomous items. *Applied Psychological Measurement*, 33(6), 419-440. <https://doi.org/10.1177/0146621608327801>. Open copy: [PubMed Central](https://pmc.ncbi.nlm.nih.gov/articles/PMC2791416/).

Magis, D., & Raîche, G. (2012). Random generation of response patterns under computerized adaptive testing with the R package catR. *Journal of Statistical Software*, 48(8), 1-31. <https://doi.org/10.18637/jss.v048.i08> (open access).

## Implementation Guidance

:::{note}
This section is for contributors and agents implementing this technique.
It is removed when the spec is promoted to a technique page.
:::

**Proposed location:** `src/catsim/selection/weighted_info.py`, next to `MPWISelector`.

**Mock implementation:**

```python
# src/catsim/selection/weighted_info.py (added to the existing module, after MPWISelector and the
# _table_at_nodes helper its spec defines)

import numpy

from .. import irt
from ..compat import override
from ..context import StepContext
from ..estimation import LogPrior, QuadratureGrid, normal_log_prior, posterior
from ..item_bank import ItemBank
from ..typing import FloatArray
from .base import BaseSelector, below_cap, check_r_max, pick_best, remaining


def _expected_posterior_variance(post: FloatArray, nodes: FloatArray, probabilities: FloatArray) -> FloatArray:
  """Return each item's posterior variance after its response, expected under the posterior predictive.

  A branch's unnormalized posterior, the posterior times the probability of that response at each
  node, sums to the branch's posterior-predictive probability, so its squared deviations from the
  branch mean, summed with those weights, are that probability times the branch's variance.

  Returns
  -------
  numpy.ndarray
      One expected posterior variance per column of ``probabilities``.
  """
  expected = numpy.zeros(probabilities.shape[1])
  for likelihood in (probabilities, 1.0 - probabilities):
    branch = post[:, numpy.newaxis] * likelihood
    mass = branch.sum(axis=0)
    # A response with no posterior-predictive mass, where P rounds to 0 or 1 at every node the
    # posterior weighs, adds nothing; dividing by its zero mass would give NaN, which pick_best rejects.
    # sums over the node axis, as in MPWI, so identical items tie exactly (ADR-0017)
    moment = (nodes[:, numpy.newaxis] * branch).sum(axis=0)
    mean = numpy.divide(moment, mass, out=numpy.zeros_like(mass), where=mass > 0)
    expected += ((nodes[:, numpy.newaxis] - mean) ** 2 * branch).sum(axis=0)
  return expected


class MEPVSelector(BaseSelector):
  """Minimum Expected Posterior Variance selector.

  For each candidate item, the posterior of the responses so far is updated with a correct and
  with an incorrect response, and the two posterior variances are averaged with the posterior-
  predictive probabilities of the two responses. The item with the smallest average is selected.

  Parameters
  ----------
  grid : QuadratureGrid or None, optional
      Grid on which the posteriors are computed. Default is ``QuadratureGrid.uniform()``.
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
    # The last bank selected for and its response-probability table, as MPWISelector caches its own
    self._table: tuple[ItemBank, FloatArray] | None = None

  @override
  def __repr__(self) -> str:
    """Return the selector as a constructor call that names every argument, with None for a default."""
    return (
      f"{type(self).__name__}(grid={self._grid_argument!r}, log_prior={self._log_prior_argument!r}, "
      f"r_max={self._r_max!r})"
    )

  @override
  def select(self, context: StepContext) -> int:
    """Return the available item with the smallest expected posterior variance after its response.

    Parameters
    ----------
    context : StepContext
        The session to select for. Its ``theta`` is not used: the branches are weighted by
        posterior-predictive probabilities.

    Returns
    -------
    int
        Index of the selected item.
    """
    candidates = below_cap(remaining(context), context.exposure_rates, self._r_max)
    item_bank = context.item_bank
    post = posterior(context.responses, item_bank.get_items(context.administered_items), self._grid, self._log_prior)
    table = self._table
    if table is None or table[0] is not item_bank:
      table = self._table = (item_bank, _table_at_nodes(item_bank, self._grid.nodes, irt.icc))
    probabilities = table[1][:, candidates]
    expected = _expected_posterior_variance(post, self._grid.nodes, probabilities)
    # pick_best takes the highest score, and the smallest variance wins
    return pick_best(candidates, -expected, context.rng)
```

**Implementation notes:**

- The response-probability table depends only on the item bank and the grid's nodes, so the
  selector computes it once per bank through MPWI's `_table_at_nodes`, with `irt.icc`, and caches
  it with that bank, as MPWI caches its information table.
- The branch weights are the posterior-predictive probabilities, as the definition requires, so
  the selector does not read the context's `theta`. catR's weights, $P_i(\hat\theta)$ and
  $1 - P_i(\hat\theta)$ at the current estimate, would be
  `irt.icc(context.theta, item_bank.get_items(candidates))`; a comparison with catR needs that
  variant, and its rankings can differ.
- The posterior comes from the context's responses at every call, as in MPWI; estimators keep no
  per-session posterior ({doc}`ADR-0007 </adr/0007-strategy-contracts>`). Each call costs two
  passes over a $Q \times$ candidates table, about twice MPWI's.
- The import-linter exception that MPWI's spec relies on (`catsim.selection.weighted_info ->
  catsim.estimation`) covers this selector too.
- MPWI and MEPV share their arguments, constructor and repr; if Pylint's duplicate-code check
  flags them, share that code once rather than copying it.

**Testing notes:**

- A hand-worked case: a two-node grid at $\pm 1$ with equal weights, a prior that is flat on
  it, no responses, and an item with $P(-1) = 0.2$ and $P(1) = 0.8$. Both branches have
  variance $0.64$, so the expected posterior variance is $0.64$, against the prior's $1$.
- An independent route: `EAPEstimator` with the same grid and prior, on the context with each
  hypothetical response appended (`dataclasses.replace`), gives each branch's variance as `se**2`;
  weighted by the posterior-predictive probabilities, they reproduce the score that decides the
  selection.
- Contract 2: on a bank built for it, the selected item has the smallest expected variance, not
  the largest.
- Identical items have equal expected variances, and the tie is broken with the context's `rng`.
- Contract 5: a cap that leaves some candidates, and one that leaves none, falling back to all
  remaining items.
- Parity with catR (ADR-0020): add `criterion = "MEPV"` cases to `tests/parity/catr/selection.R`.
  catR weights the branches at the current estimate, so its rankings are a departure, a strict
  expected failure in `_DEPARTURES` with this spec's reason, unless the selector offers catR's
  weights as a setting; the branch variances, from catR's `eapSem` on a grid with trapezoid
  weights, can be compared directly.
