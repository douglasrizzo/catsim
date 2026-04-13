**Status:** Planned
**Module:** `catsim.stopping.stopping.MinInfoInBankStopper` (planned)
**Reference:** Choi, Grady, and Dodd (2011)

# Minimum Information in Bank Stopper

## Motivation

This rule is the simplest forward-looking sibling of predicted-SEM stopping.
Instead of asking whether the target SE is reachable, it asks whether any
remaining item is still informative enough to be worth administering.

That makes it a lightweight safety-net stopper for thin banks or poorly targeted
regions of the ability scale.

## Definition

Let $\mathcal{R}$ denote the remaining items and
$I_i(\hat{\theta})$ the information of candidate item $i$ at the
current estimate. The rule stops when

$$
\max_{i \in \mathcal{R}} I_i(\hat{\theta}) \le \epsilon,
$$

where $\epsilon$ is the configured minimum-information threshold.

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

Choi, S. W., Grady, M. W., & Dodd, B. G. (2011). A new stopping rule for computerized adaptive testing. *Educational and Psychological Measurement*, 71(1), 37-53. <https://doi.org/10.1177/0013164410387338>

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

from typing import Any

import numpy
import numpy.typing as npt

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
    administered_items: npt.NDArray[numpy.floating[Any]],  # noqa: ARG002
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
