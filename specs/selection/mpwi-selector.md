**Status:** Planned
**Module:** `catsim.selection.weighted_info.MPWISelector` (planned)
**Reference:** van der Linden (1998)
**Last validated:** 2026-11-23, against van der Linden (1996, RR 96-01); Choi & Swartz (2009); Magis & Raîche (2012); catR 3.17 (`MWI.R`, `nextItem.R`).
**Implementation guidance:** updated to the 1.0 strategy contract (`StepContext` in) on 2026-11-24.

# Maximum Posterior Weighted Information Selector

## Motivation

MPWI is the Bayesian counterpart to MLWI. Instead of weighting each candidate's
information curve by the likelihood of ability values, it weights by the current
posterior, which naturally combines the prior and the observed response history.
van der Linden (1996, p. 10) presents it as a generalization of
likelihood-weighted information whose advantage is that the prior can carry
what is known about the examinee in advance, such as background variables
related to ability.

Once Bayesian infrastructure exists, this criterion is attractive because it
builds on the same `posterior()` helper as EAP and the other posterior-based
selectors. Choi and Swartz (2009, Equation 5 and Appendix) also prove that the
maximum expected posterior weighted information criterion, which averages
posterior-weighted information over the possible responses to the candidate,
always selects the same item as MPWI, so the simpler criterion gives the same
tests.

## Definition

For candidate item $i$, MPWI is

$$
\mathrm{MPWI}_i
=
\int I_i(\theta)\, p(\theta \mid \mathbf{u})\, d\theta.
$$

On a quadrature grid $\{\theta_q\}_{q=1}^{Q}$ with posterior weights
$p_q = p(\theta_q \mid \mathbf{u})$, the implementation guidance
approximates this as

$$
\mathrm{MPWI}_i
\approx
\sum_{q=1}^{Q} I_i(\theta_q)\, p_q.
$$

The rule selects the non-administered item with the largest MPWI score.

This is the maximum posterior expected information criterion of van der
Linden's 1996 report (p. 10, Equation 12), which preceded the 1998 article. The
sentence introducing it says "minimize"; the equation and the criterion's name
maximize. The report weights the observed information $J$ (p. 9, Equation 10),
which equals Fisher information under the two-parameter logistic model it
assumes (p. 5, Equation 1; p. 10, Equation 11). catsim uses Fisher information,
as Choi and Swartz (2009, Equation 2) and `catR` do; the two differ for items
with guessing or slipping. Magis and Raîche (2012, p. 9, Table 1) give the same
criterion with the posterior unnormalized, which ranks items the same way.

`catR` implements MPWI in
[`MWI.R`](https://github.com/cran/catR/blob/master/R/MWI.R), called from
[`nextItem.R`](https://github.com/cran/catR/blob/master/R/nextItem.R). It
agrees on the criterion: Fisher information weighted by the prior density times
the likelihood. It departs in the numerics: the posterior is not normalized,
and the integral is computed by the trapezoidal rule on 33 equally spaced nodes
over $[-4, 4]$ (`parInt = c(-4, 4, 33)`). Its default prior is standard
normal, as is catsim's; catsim's default grid has 121 equally weighted nodes
over $[-6, 6]$ and normalizes the posterior. The normalizing constant does not
change the ranking; the grid and its range change the approximation slightly.

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

1. Scores are posterior-weighted expected item information values, not likelihood-weighted values.
2. With an empty response vector, the selector reduces to prior-weighted expected information.
3. The scores depend on the context's responses, through the posterior, and not on its `theta`.
4. Exposure filtering applies before the final argmax, with fallback to the unrestricted remaining pool if every constrained candidate is excluded.

## References

van der Linden, W. J. (1998). Bayesian item selection criteria for adaptive testing. *Psychometrika*, 63(2), 201-216. <https://doi.org/10.1007/BF02294775>. Not openly available; the 1996 report below preceded it.

van der Linden, W. J. (1996). *Bayesian item selection criteria for adaptive testing* (Research Report 96-01). Enschede: University of Twente. Open copy: [University of Twente](https://research.utwente.nl/files/247173933/Research_Report_96_01.pdf).

Choi, S. W., & Swartz, R. J. (2009). Comparison of CAT item selection criteria for polytomous items. *Applied Psychological Measurement*, 33(6), 419-440. <https://doi.org/10.1177/0146621608327801>. Open copy: [PubMed Central](https://pmc.ncbi.nlm.nih.gov/articles/PMC2791416/).

Magis, D., & Raîche, G. (2012). Random generation of response patterns under computerized adaptive testing with the R package catR. *Journal of Statistical Software*, 48(8), 1-31. <https://doi.org/10.18637/jss.v048.i08> (open access).

## Implementation Guidance

:::{note}
This section is for contributors and agents implementing this technique.
It is removed when the spec is promoted to "Implemented" status.
:::

**Proposed location:** `src/catsim/selection/weighted_info.py`, next to `MLWISelector`.

**Mock implementation:**

```python
# src/catsim/selection/weighted_info.py (added to the existing module)

from collections.abc import Callable

import numpy

from .. import irt
from ..context import StepContext
from ..estimation.bayesian import LogPrior, QuadratureGrid, normal_log_prior, posterior
from ..item_bank import ItemBank
from ..typing import FloatArray
from .base import BaseSelector, below_cap, check_r_max, pick_best, remaining


def _table_at_nodes(
  item_bank: ItemBank, nodes: FloatArray, function: Callable[[FloatArray, FloatArray], FloatArray]
) -> FloatArray:
  """Return ``function`` of every item at every node, one row per node, cached in the item bank.

  The table depends only on the bank's parameters, which are read-only, and on the nodes, so one
  table serves every session and every selector that uses the same nodes.

  Returns
  -------
  numpy.ndarray
      The read-only table, of shape ``(len(nodes), item_bank.n_items)``.
  """
  cache = item_bank._selector_cache  # noqa: SLF001 - the cache ItemBank keeps for selectors
  key = f"{function.__name__}:{nodes.tobytes().hex()}"
  if key not in cache:
    table = function(nodes, item_bank.items)
    table.flags.writeable = False
    cache[key] = table
  return cache[key]


class MPWISelector(BaseSelector):
  """Maximum Posterior Weighted Information selector.

  Scores each candidate item by its information summed over a quadrature grid, weighted by the
  posterior of the responses so far, and selects the highest score.

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
    """Return the available item with the largest information weighted by the posterior of the responses.

    Parameters
    ----------
    context : StepContext
        The session to select for. Its ``theta`` is not used: the posterior carries what the
        responses say about the ability.

    Returns
    -------
    int
        Index of the selected item.
    """
    candidates = below_cap(remaining(context), context.exposure_rates, self._r_max)
    item_bank = context.item_bank
    post = posterior(context.responses, item_bank.get_items(context.administered_items), self._grid, self._log_prior)
    information = _table_at_nodes(item_bank, self._grid.nodes, irt.information)
    return pick_best(candidates, post @ information[:, candidates], context.rng)
```

**Implementation notes:**

- The information table depends only on the item bank and the grid's nodes, so it is computed
  once per bank and node set and cached in `ItemBank._selector_cache`, which exists for this
  selector and MEPV; the bank's parameters are read-only, so the table cannot go stale. The key
  holds the node values themselves, not `id(grid)`, which Python reuses once a grid is garbage
  collected. MEPV caches its probability table through the same helper, with `irt.icc`.
- `_selector_cache` is private to `ItemBank`, so Ruff's SLF001 flags the access from a selector;
  the sketch exempts that one line.
- The posterior comes from the context's responses at every call: estimators keep no per-session
  posterior ({doc}`ADR-0005 </adr/0005-explicit-strategy-contracts>`), and `posterior()` is one
  vectorized pass over the grid. With no responses it is the prior on the grid (Contract 2).
- The selection and estimation layers do not import each other (`[tool.importlinter]` in
  `pyproject.toml`). Add `"catsim.selection.weighted_info -> catsim.estimation.bayesian"` to that
  contract's `ignore_imports`, as `MEISelector` does for its estimator, or move the grid and
  posterior helpers to a lower layer.
- catR's 33 nodes on $[-4, 4]$ with the trapezoidal rule are a `QuadratureGrid` with trapezoid
  weights, for comparisons with catR; the normalizing constant does not change the ranking.

**Testing notes:**

- Contract 2: with no responses and a standard normal prior, the selected item is the one with
  the largest information integrated against the standard normal density, computed independently
  with `scipy.integrate.quad`.
- Contract 1 and 3: on a bank built for it, MPWI and MLWI select different items for the same
  responses under an informative prior, and the same items with different responses select
  different items.
- The cache: a second selector with an equal grid reuses the table, and a grid with other nodes
  gets its own.
- Contract 4: a cap that leaves some candidates, and one that leaves none, falling back to all
  remaining items.
