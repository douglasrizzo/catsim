**Status:** Planned
**Module:** `catsim.stopping.stopping.PredictedSEMStopper` (planned)
**Reference:** Choi, Grady, and Dodd (2011)

# Predicted-SEM Stopper

## Motivation

Plain SE-threshold stopping can chase an unreachable target. If the remaining
item bank is too weak at the current ability estimate, a CAT may continue
administering items even though no available next item can push the standard
error below the desired threshold.

The predicted-SEM rule fixes that by asking a forward-looking question: if the
best remaining item were administered next, would the target SE become
attainable? If not, stop now.

## Definition

Let $\hat{\theta}$ be the current estimate, $\mathcal{A}$ the set
of administered items, and $\mathcal{R}$ the remaining items. Let

$$
i^* = \arg\max_{i \in \mathcal{R}} I_i(\hat{\theta})
$$

be the most informative remaining item at the current estimate. The current
standard error is

$$
SE(\hat{\theta} \mid \mathcal{A})
=
\sqrt{\frac{1}{I(\hat{\theta} \mid \mathcal{A})}}.
$$

The predicted standard error after hypothetically adding the best remaining item is

$$
SE(\hat{\theta} \mid \mathcal{A} \cup \{i^*\})
=
\sqrt{\frac{1}{I(\hat{\theta} \mid \mathcal{A}) + I_{i^*}(\hat{\theta})}}.
$$

The recommended composite rule is: stop if the current SE is already below the
target $\tau$, or if the predicted SE after adding the best remaining item
is still greater than $\tau$.

## Parameters

```{eval-rst}
.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``min_error``
     - required
     - Target standard error threshold :math:`\tau`.
   * - ``min_items``
     - ``None``
     - Minimum number of items before the rule is allowed to stop.
   * - ``max_items``
     - ``None``
     - Hard maximum test length inherited from ``TestLengthStopper``.
```

## Behavioral Contracts

Any correct implementation must satisfy the following properties:

1. If the current SE is already below the target, the stopper returns `True`.
2. If even the single most informative remaining item would leave SE above the target, the stopper returns `True`.
3. The stopper needs access to the full item bank because it depends on the remaining items, not only the administered set.
4. With no administered items, the stopper does not terminate solely from the predicted-SEM criterion.
5. The inherited minimum-length, maximum-length, and bank-exhaustion guards still apply.

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
# src/catsim/stopping/stopping.py  (alongside MinErrorStopper)
"""Predicted-SEM stopping rule (Choi, Grady & Dodd, 2011)."""

from typing import Any

import numpy
import numpy.typing as npt

from .. import irt
from ..item_bank import ItemBank
from .stopping import TestLengthStopper


class PredictedSEMStopper(TestLengthStopper):
  """Stop when adding the best remaining item cannot drop SE below the target.

  Parameters
  ----------
  min_error : float
      Target standard error of estimation.
  min_items : int or None
      Minimum number of items before the test can stop.
  max_items : int or None
      Maximum number of items before the test must stop.
  """

  def __init__(
    self,
    min_error: float,
    min_items: int | None = None,
    max_items: int | None = None,
  ) -> None:
    if min_error <= 0:
      raise ValueError(f"min_error must be positive, got {min_error}")
    super().__init__(min_items=min_items, max_items=max_items)
    self._min_error = float(min_error)

  def __str__(self) -> str:
    return f"PredictedSEMStopper(min_error={self._min_error})"

  # Override the full stop() because we need access to the item bank to look up
  # the most informative *remaining* item, not just the parameters of administered ones.
  def stop(
    self,
    item_bank: ItemBank,
    administered_items: list[int],
    theta: float,
  ) -> bool:
    # Hard stops + min_items handled by the base class
    base_decision = super().stop(item_bank, administered_items, theta)
    if base_decision is True:
      return True

    if not administered_items:
      return False

    administered_arr = item_bank.get_items(administered_items)
    current_info = irt.test_info(theta, administered_arr)
    current_sem = numpy.inf if current_info <= 0 else 1.0 / numpy.sqrt(current_info)
    if current_sem <= self._min_error:
      return True

    # Predicted SEM: pretend the most informative remaining item is added next.
    remaining_mask = numpy.ones(item_bank.n_items, dtype=bool)
    remaining_mask[administered_items] = False
    if not remaining_mask.any():
      return True
    remaining_info = item_bank.information(theta)[remaining_mask]
    best_remaining_info = float(remaining_info.max())

    predicted_info = current_info + best_remaining_info
    predicted_sem = numpy.inf if predicted_info <= 0 else 1.0 / numpy.sqrt(predicted_info)
    return predicted_sem > self._min_error

  def _check_stopping_criterion(  # noqa: PLR6301
    self,
    administered_items: npt.NDArray[numpy.floating[Any]],  # noqa: ARG002
    theta: float,  # noqa: ARG002
  ) -> bool:
    # Logic lives in stop() because we need the item bank, not just admin params.
    return False
```

**Implementation notes:**

- Override `stop()` instead of `_check_stopping_criterion()` because the rule needs the remaining items, not only the administered-item matrix.
- This stopper subsumes `MinErrorStopper` by including the immediate `current_sem <= target` check.
- No new dependencies are required.

**Dependencies:** none

**Testing notes:**

- A deliberately weak bank with an unreachable target SE should stop after a small constant number of items rather than running to `max_items`.
- A well-targeted bank should produce stopping lengths similar to `MinErrorStopper` at the same threshold.
- Exhausted-bank and `min_items` behavior should still match `TestLengthStopper`.
