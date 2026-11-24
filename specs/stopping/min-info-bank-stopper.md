**Status:** Planned
**Module:** `catsim.stopping.bank.MinInfoInBankStopper` (planned)
**Reference:** Dodd, Koch, and De Ayala (1989); Choi, Grady, and Dodd (2011)
**Last validated:** 2026-11-23, against Dodd, Koch & De Ayala (1989), Choi, Grady & Dodd (2011, PubMed Central author manuscript) and catR 3.17.
**Implementation guidance:** updated to the 1.0 strategy contract (`StepContext` in, a single criterion for `StopRule`) on 2026-11-24.

# Minimum Information in Bank Stopper

## Motivation

This rule is the simplest forward-looking sibling of predicted-SEM stopping.
Instead of asking whether the target SE is reachable, it asks whether any
remaining item is still informative enough to be worth administering.

That makes it a lightweight safety-net stopper for thin banks or poorly targeted
regions of the ability scale: it gives much shorter tests to examinees at
extreme ability levels, for whom few informative items are available (Dodd et
al., 1989, p. 135). Used alone, it keeps administering items to examinees for
whom informative items are abundant, and it typically gives less precise
estimates than the minimum-error rule, so Choi et al. (2011, Introduction and
Method) combined the two.

## Definition

Let $\mathcal{R}$ denote the remaining items and
$I_i(\hat{\theta})$ the information of candidate item $i$ at the
current estimate. The rule stops when

$$
\max_{i \in \mathcal{R}} I_i(\hat{\theta}) \le \epsilon,
$$

where $\epsilon$ is the configured minimum-information threshold.

Dodd et al. (1989, pp. 135-136) stopped the test when no item left in the pool
had an information value of at least $\epsilon = .50$ at the current estimate,
that is, when the maximum was below $\epsilon$; Choi et al. (2011, Method, "CAT
Simulations") used the same threshold. The planned rule also stops when the
maximum equals $\epsilon$, as catR does. Both studies used polytomous items
under the graded response model.

catR implements this rule as `rule = "minInfo"` (`checkStopRule.R`, catR 3.17):
it computes the Fisher information of every item not yet administered at the
provisional estimate (`randomCAT.R` passes the bank without the administered
items) and stops when the largest is at most the threshold. As here, the rule
applies only after the minimum test length, and the test also ends when the
bank is exhausted.

## Parameters

```{eval-rst}
.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``min_info``
     - required
     - Information threshold :math:`\epsilon`.
```

## Behavioral Contracts

Any correct implementation must satisfy the following properties:

1. The stopper returns `True` when the most informative remaining item has information less than or equal to `min_info`.
2. The rule depends on the remaining item bank, so it must inspect non-administered items directly.
3. If no items remain, the stopper terminates.
4. Nonnegative thresholds are valid; negative ones are rejected.
5. The stopper is a single criterion: the minimum test length and the length cap come from `StopRule`'s `min_items` and a `MaxLengthStopper` it is combined with.

## References

Choi, S. W., Grady, M. W., & Dodd, B. G. (2011). A new stopping rule for computerized adaptive testing. *Educational and Psychological Measurement*, 71(1), 37-53. <https://doi.org/10.1177/0013164410387338>. Open copy: [PubMed Central](https://pmc.ncbi.nlm.nih.gov/articles/PMC3028267/) (author manuscript without the journal's page numbers; cited here by section).

Dodd, B. G., Koch, W. R., & De Ayala, R. J. (1989). Operational characteristics of adaptive testing procedures using the graded response model. *Applied Psychological Measurement*, 13(2), 129-143. <https://doi.org/10.1177/014662168901300202>. Open copy: [University of Minnesota Digital Conservancy](https://hdl.handle.net/11299/107113).

## Implementation Guidance

:::{note}
This section is for contributors and agents implementing this technique.
It is removed when the spec is promoted to "Implemented" status.
:::

**Proposed location:** `src/catsim/stopping/bank.py`, a new module for the stoppers that judge
what the remaining items can still add (this one, the KL stopper and the predicted-SEM stopper).

**Mock implementation:**

```python
# src/catsim/stopping/bank.py
"""Stop a test once the remaining items have too little left to add."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy

from .. import irt
from .base import BaseStopper

if TYPE_CHECKING:
  from ..context import StepContext
  from ..typing import IntArray


def _remaining(context: StepContext) -> IntArray:
  """Return the indices of the items neither administered nor excluded, in ascending order.

  ``catsim.selection.base.remaining`` does the same for selectors, but the stopping layer may not
  import the selection layer, and a stopper needs an empty result where a selector raises.

  Returns
  -------
  numpy.ndarray
      The remaining item indices, possibly none.
  """
  mask = numpy.ones(context.item_bank.n_items, dtype=bool)
  mask[numpy.asarray(context.administered_items, dtype=numpy.intp)] = False
  mask[numpy.asarray(context.excluded_items, dtype=numpy.intp)] = False
  return numpy.flatnonzero(mask)


@dataclass(frozen=True)
class MinInfoInBankStopper(BaseStopper):
  """Stop once no remaining item has more than a minimum information at the current estimate.

  Parameters
  ----------
  min_info : float
      Information threshold. The test stops once the most informative remaining item has at most
      this much information. Must not be negative.

  Raises
  ------
  ValueError
      If `min_info` is negative or NaN.
  """

  min_info: float

  def __post_init__(self) -> None:
    """Validate the threshold."""
    # written so that NaN fails the check too
    if not self.min_info >= 0:
      msg = f"min_info must not be negative, got {self.min_info}"
      raise ValueError(msg)

  def stop(self, context: StepContext) -> bool:
    """Return whether no remaining item has more than `min_info` information at the current estimate.

    Parameters
    ----------
    context : StepContext
        The session to check.

    Returns
    -------
    bool
        True if every remaining item has at most `min_info` information at ``context.theta``, or no
        item remains; False otherwise.
    """
    candidates = _remaining(context)
    if candidates.size == 0:
      return True
    information = irt.inf_hpc(context.theta, context.item_bank.get_items(candidates))
    return float(information.max()) <= self.min_info
```

**Implementation notes:**

- The stopper is one criterion. Choi et al. (2011) combined it with the minimum-error rule, which
  in catsim is a rule of several stoppers:
  `StopRule(MinInfoInBankStopper(0.5), StandardErrorStopper(0.3), MaxLengthStopper(30), min_items=5)`.
  `StopRule` applies `min_items` to every stopper, the length cap included.
- The information is Fisher information at the context's `theta`, the current estimate, as in
  the definition and catR; it does not read the estimator's standard error.
- A stopper is pure (`BaseStopper`): the same context gives the same answer, and no state is kept
  between calls.
- The stopping layer may not import the selection layer (`[tool.importlinter]` in
  `pyproject.toml`), hence `_remaining` here; the KL and predicted-SEM stoppers in the same module
  share it.

**Testing notes:**

- Hand-worked thresholds: a two-parameter logistic item with $b = \hat\theta$ has information
  $a^2/4$ at $\hat\theta$. With `min_info = 0.25`, remaining items with $a = 1$ (information
  exactly $0.25$, the equality case of Contract 1) and $a = 0.5$ stop the test, and one more
  remaining item with $a = 2$ (information $1$) keeps it going. Pick values whose information is
  exact in floating point: $a = \sqrt{2}$ gives $0.5000000000000001$, not $0.5$.
- Contract 2: administering the one informative item makes the stopper stop at the next check.
- Contract 3: with every item administered, it stops.
- Contract 4: negative and NaN thresholds are rejected; 0 is accepted.
- Contract 5: inside `StopRule(..., min_items=k)` it does not stop before $k$ items.
