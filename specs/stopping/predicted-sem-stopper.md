**Status:** Planned
**Module:** `catsim.stopping.stopping.PredictedSEMStopper` (planned)
**Reference:** Choi, Grady, and Dodd (2011)
**Last validated:** 2026-11-23, against Choi, Grady & Dodd (2011, PubMed Central author manuscript).
**Implementation guidance:** not yet updated to the 1.0 strategy contract (M22).

# Predicted-SEM Stopper

## Motivation

Plain SE-threshold stopping can chase an unreachable target. If the remaining
item bank is too weak at the current ability estimate, a CAT may continue
administering items even though no available next item can push the standard
error below the desired threshold. It also stops as soon as the target is
reached, even when one or two more items would still improve precision
substantially (Choi et al., 2011, Introduction).

The predicted standard error reduction (PSER) rule of Choi, Grady, and Dodd
(2011) asks whether another item is worth administering. It predicts how much
the best remaining item would reduce the standard error, stops before the
target is reached when that reduction is negligible, and continues past the
target when it is still substantial (Method, "Stopping Criteria"; Discussion).

## Definition

Let $\hat{\theta}$ be the current estimate, $\mathcal{A}$ the set
of administered items, $\mathcal{R}$ the remaining items and $\tau$ the target
standard error. The current standard error is

$$
SE(\hat{\theta} \mid \mathcal{A})
=
\sqrt{\frac{1}{I(\hat{\theta} \mid \mathcal{A})}},
$$

as in catsim's minimum-error stopper. For a remaining item $j$, let $SE_j^{+}$
be the standard error predicted after administering it, and let

$$
\Delta
=
\max_{j \in \mathcal{R}}
\left[ SE(\hat{\theta} \mid \mathcal{A}) - SE_j^{+} \right]
$$

be the largest predicted reduction. With two thresholds $H^{+}$ and $H^{-}$:

- if $SE(\hat{\theta} \mid \mathcal{A}) < \tau$, the test stops unless
  $\Delta \ge H^{+}$;
- if $SE(\hat{\theta} \mid \mathcal{A}) \ge \tau$, the test continues unless
  $\Delta < H^{-}$.

Choi et al. (2011, Method, "Stopping Criteria") call $H^{+}$ and $H^{-}$ the
hyper and hypo parameters. They used $\tau = 0.3$, $H^{+} = 0.03$ and
$H^{-} = 0.01$, with tests of 3 to 15 items (Method, "CAT Simulations"), and set
$H^{+}$ three times as large as $H^{-}$ because they considered stopping after
the target is met the smaller risk (Discussion).

Choi et al. predict the standard error from the predictive posterior variance,
the objective of minimum expected posterior variance (MEPV) item selection
(Method, "Stopping Criteria"):

$$
SE_j^{+}
=
\left[
  \sum_{r} p_j(r \mid \mathbf{u})\,
  \mathrm{Var}(\theta \mid \mathbf{u}, U_j = r)
\right]^{1/2},
$$

where $p_j(r \mid \mathbf{u})$ is the posterior predictive probability of
response $r$ to item $j$; they estimated ability by EAP with a standard normal
prior (Method, "CAT Simulations"). The planned stopper departs from them: it
predicts the standard error from Fisher information, consistently with its
current standard error,

$$
SE_j^{+}
=
\sqrt{\frac{1}{I(\hat{\theta} \mid \mathcal{A}) + I_j(\hat{\theta})}},
$$

so the largest reduction comes from the remaining item with the largest
information at $\hat{\theta}$, and the rule works with any estimator.

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
   * - ``hyper``
     - ``0.03``
     - Threshold :math:`H^{+}`: once the target is met, the test continues if some remaining item
       is predicted to reduce the standard error by at least this much.
   * - ``hypo``
     - ``0.01``
     - Threshold :math:`H^{-}`: before the target is met, the test stops if no remaining item is
       predicted to reduce the standard error by at least this much.
   * - ``min_items``
     - ``None``
     - Minimum number of items before the rule is allowed to stop.
   * - ``max_items``
     - ``None``
     - Hard maximum test length inherited from ``TestLengthStopper``.
```

## Behavioral Contracts

Any correct implementation must satisfy the following properties:

1. If the current SE is below the target and no remaining item is predicted to reduce it by at least `hyper`, the stopper returns `True`.
2. If the current SE is below the target but some remaining item is predicted to reduce it by at least `hyper`, the criterion does not stop the test.
3. If the current SE is at or above the target, the criterion stops the test if and only if no remaining item is predicted to reduce it by at least `hypo`.
4. The stopper needs access to the full item bank because it depends on the remaining items, not only the administered set.
5. With no administered items, the stopper does not terminate solely from the predicted-SEM criterion.
6. The inherited minimum-length, maximum-length, and bank-exhaustion guards still apply.

## References

Choi, S. W., Grady, M. W., & Dodd, B. G. (2011). A new stopping rule for computerized adaptive testing. *Educational and Psychological Measurement*, 71(1), 37-53. <https://doi.org/10.1177/0013164410387338>. Open copy: [PubMed Central](https://pmc.ncbi.nlm.nih.gov/articles/PMC3028267/) (author manuscript without the journal's page numbers; cited here by section).

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

import numpy

from .. import irt
from .._typing import FloatArray
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
    administered_items: FloatArray,  # noqa: ARG002
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
