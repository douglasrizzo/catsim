**Status:** Planned
**Module:** `catsim.stopping.stopping.MinInfoInBankStopper` (planned)
**Reference:** Dodd, Koch, and De Ayala (1989); Choi, Grady, and Dodd (2011)
**Last validated:** 2026-11-23, against Dodd, Koch & De Ayala (1989), Choi, Grady & Dodd (2011, PubMed Central author manuscript) and catR 3.17.
**Implementation guidance:** not yet updated to the 1.0 strategy contract (M22).

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
   * - ``min_items``
     - ``None``
     - Minimum number of items before stopping is allowed.
   * - ``max_items``
     - ``None``
     - Hard maximum test length.
```

## Behavioral Contracts

Any correct implementation must satisfy the following properties:

1. The stopper returns `True` when the most informative remaining item has information less than or equal to `min_info`.
2. The rule depends on the remaining item bank, so it must inspect non-administered items directly.
3. If no items remain, the stopper terminates.
4. Nonnegative thresholds are valid; negative ones are rejected.
5. The inherited minimum-length and maximum-length checks still take precedence.

## References

Choi, S. W., Grady, M. W., & Dodd, B. G. (2011). A new stopping rule for computerized adaptive testing. *Educational and Psychological Measurement*, 71(1), 37-53. <https://doi.org/10.1177/0013164410387338>. Open copy: [PubMed Central](https://pmc.ncbi.nlm.nih.gov/articles/PMC3028267/) (author manuscript without the journal's page numbers; cited here by section).

Dodd, B. G., Koch, W. R., & De Ayala, R. J. (1989). Operational characteristics of adaptive testing procedures using the graded response model. *Applied Psychological Measurement*, 13(2), 129-143. <https://doi.org/10.1177/014662168901300202>. Open copy: [University of Minnesota Digital Conservancy](https://hdl.handle.net/11299/107113).

## Implementation Guidance

:::{note}
This section is for contributors and agents implementing this technique.
It is removed when the spec is promoted to "Implemented" status.
:::

**Proposed location:** `src/catsim/stopping/stopping.py`

**Mock implementation:**

```python
# src/catsim/stopping/stopping.py  (alongside MinErrorStopper and PredictedSEMStopper)
"""Minimum-information-in-bank stopper (Choi, Grady & Dodd, 2011)."""

import numpy

from .._typing import FloatArray
from ..item_bank import ItemBank
from .stopping import TestLengthStopper


class MinInfoInBankStopper(TestLengthStopper):
  """Stop when no remaining item provides at least ``min_info`` information at theta.

  Parameters
  ----------
  min_info : float
      Information threshold. The test stops when ``max_remaining_info <= min_info``.
  min_items : int or None
      Minimum number of items before the test can stop.
  max_items : int or None
      Maximum number of items before the test must stop.
  """

  def __init__(
    self,
    min_info: float,
    min_items: int | None = None,
    max_items: int | None = None,
  ) -> None:
    if min_info < 0:
      raise ValueError(f"min_info must be non-negative, got {min_info}")
    super().__init__(min_items=min_items, max_items=max_items)
    self._min_info = float(min_info)

  def __str__(self) -> str:
    return f"MinInfoInBankStopper(min_info={self._min_info})"

  def stop(
    self,
    item_bank: ItemBank,
    administered_items: list[int],
    theta: float,
  ) -> bool:
    base_decision = super().stop(item_bank, administered_items, theta)
    if base_decision is True:
      return True

    remaining_mask = numpy.ones(item_bank.n_items, dtype=bool)
    remaining_mask[administered_items] = False
    if not remaining_mask.any():
      return True

    remaining_info = item_bank.information(theta)[remaining_mask]
    return float(remaining_info.max()) <= self._min_info

  def _check_stopping_criterion(  # noqa: PLR6301
    self,
    administered_items: FloatArray,  # noqa: ARG002
    theta: float,  # noqa: ARG002
  ) -> bool:
    return False
```

**Implementation notes:**

- This rule is most useful as a complement to `MinErrorStopper` or `PredictedSEMStopper`.
- A future `CompositeStopper` helper may be useful if chaining multiple stopping criteria becomes common.
- No extra dependencies are needed.

**Dependencies:** none

**Testing notes:**

- A bank with uniformly weak remaining items should trigger the stopper immediately once `min_items` is satisfied.
- A well-targeted bank should not trigger this rule early.
- Exhausted-bank handling and parameter validation should be covered.
