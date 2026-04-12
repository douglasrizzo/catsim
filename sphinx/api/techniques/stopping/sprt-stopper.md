**Status:** Planned
**Module:** `catsim.stopping.classification.SPRTStopper` (planned)
**Reference:** Reckase (1983); Wald (1947); Weissman (2007)

# Sequential Probability Ratio Test Stopper

## Motivation

Current catsim stoppers are precision-oriented. They are appropriate when the
goal is to estimate ability, but they are inefficient when the real goal is a
binary decision such as pass/fail or master/non-master.

The sequential probability ratio test (SPRT) is the classical optimal stopping
rule for that setting. It stops as soon as the evidence for one side of the cut
score is strong enough, which minimizes expected test length at fixed error
rates.

## Definition

For cut score $\theta_c$ and indifference half-width $\delta$, define
two point hypotheses

$$
H_0: \theta = \theta_c - \delta,
\qquad
H_1: \theta = \theta_c + \delta.
$$

After $n$ responses, compute the log-likelihood ratio

$$
\Lambda_n
=
\log \frac{L(\mathbf{u} \mid \theta_c + \delta)}
             {L(\mathbf{u} \mid \theta_c - \delta)}.
$$

Wald's decision bounds are

$$
A = \log\frac{1-\beta}{\alpha},
\qquad
B = \log\frac{\beta}{1-\alpha},
$$

where $\alpha$ and $\beta$ are the nominal Type I and Type II error
rates. The rule is:

$$
\Lambda_n \ge A \Rightarrow \text{classify above cut},
\qquad
\Lambda_n \le B \Rightarrow \text{classify below cut},
$$

and continue testing otherwise.

## Parameters

```{eval-rst}
.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``cut_score``
     - required
     - Classification cut point :math:`\theta_c`.
   * - ``indifference``
     - ``0.2``
     - Half-width :math:`\delta` of the indifference region.
   * - ``alpha``
     - ``0.05``
     - Nominal Type I error rate.
   * - ``beta``
     - ``0.05``
     - Nominal Type II error rate.
   * - ``min_items``
     - ``None``
     - Minimum test length before classification may be declared.
   * - ``max_items``
     - ``None``
     - Hard maximum test length; if reached, the implementation guidance uses the sign of the current statistic for fallback classification.
```

## Behavioral Contracts

Any correct implementation must satisfy the following properties:

1. The stopper uses the response vector directly through a log-likelihood ratio; `theta` itself is not the primary decision statistic.
2. Crossing the upper bound classifies above the cut and crossing the lower bound classifies below the cut.
3. If neither bound is crossed, the test continues unless a shared hard-stop condition fires.
4. The stopper exposes the last classification decision and the last log-likelihood-ratio value for downstream reporting.
5. The rule requires a response-vector-aware stopping API extension; without it, it cannot be implemented correctly.

## References

Reckase, M. D. (1983). A procedure for decision making using tailored testing. In D. J. Weiss (Ed.), *New Horizons in Testing: Latent Trait Test Theory and Computerized Adaptive Testing* (pp. 237-255). New York: Academic Press.

Wald, A. (1947). *Sequential Analysis*. New York: Wiley.

Weissman, A. (2007). Mutual information item selection in adaptive classification testing. *Educational and Psychological Measurement*, 67(1), 41-58. <https://doi.org/10.1177/0013164406288164>

## Implementation Guidance

:::{note}
This section is for contributors and agents implementing this technique.
It is removed when the spec is promoted to "Implemented" status.
:::

**Proposed location:** `src/catsim/stopping/classification.py`

**Mock implementation:**

```python
# src/catsim/stopping/classification.py
"""Sequential Probability Ratio Test stopping rule for classification CATs."""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy
import numpy.typing as npt

from .. import irt
from ..item_bank import ItemBank
from .stopping import TestLengthStopper


class Classification(Enum):
  UNDECIDED = 0
  BELOW_CUT = 1
  ABOVE_CUT = 2


class SPRTStopper(TestLengthStopper):
  """Wald-style SPRT stopping rule for classification CATs.

  Parameters
  ----------
  cut_score : float
      Target ability threshold :math:`\\theta_c`.
  indifference : float
      Half-width :math:`\\delta` of the indifference region around ``cut_score``.
      Typical values are 0.1-0.3 on the logit scale.
  alpha : float
      Nominal Type I error rate (probability of classifying above cut when below).
  beta : float
      Nominal Type II error rate (probability of classifying below cut when above).
  min_items : int or None
      Minimum items before a classification can be declared.
  max_items : int or None
      Maximum items; if reached, classification is made by the sign of :math:`\\Lambda_n`.
  """

  def __init__(
    self,
    cut_score: float,
    indifference: float = 0.2,
    alpha: float = 0.05,
    beta: float = 0.05,
    min_items: int | None = None,
    max_items: int | None = None,
  ) -> None:
    if indifference <= 0:
      raise ValueError(f"indifference must be positive, got {indifference}")
    if not 0 < alpha < 1:
      raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if not 0 < beta < 1:
      raise ValueError(f"beta must be in (0, 1), got {beta}")
    super().__init__(min_items=min_items, max_items=max_items)
    self._theta_low = cut_score - indifference
    self._theta_high = cut_score + indifference
    self._log_A = float(numpy.log((1 - beta) / alpha))
    self._log_B = float(numpy.log(beta / (1 - alpha)))
    self._last_classification = Classification.UNDECIDED
    self._last_llr = 0.0

  def __str__(self) -> str:
    return f"SPRTStopper(cut={self._theta_low + (self._theta_high - self._theta_low) / 2})"

  @property
  def last_classification(self) -> Classification:
    return self._last_classification

  @property
  def last_log_likelihood_ratio(self) -> float:
    return self._last_llr

  # ---- core kernel ----
  def _log_likelihood_ratio(
    self,
    response_vector: list[bool],
    administered_items: npt.NDArray[numpy.floating],
  ) -> float:
    """Compute :math:`\\log L(u | theta_high) - \\log L(u | theta_low)`."""
    ll_high = irt.log_likelihood(self._theta_high, response_vector, administered_items)
    ll_low = irt.log_likelihood(self._theta_low, response_vector, administered_items)
    return ll_high - ll_low

  def stop(
    self,
    item_bank: ItemBank,
    administered_items: list[int],
    theta: float,  # noqa: ARG002 -- SPRT does not use the ability estimate
  ) -> bool:
    base_decision = super().stop(item_bank, administered_items, theta)
    if not administered_items:
      return base_decision

    # SPRTStopper needs the response vector, which the current BaseStopper signature
    # does not expose. This stopper requires the same `response_vector` signature
    # extension as the MEI/MLWI selectors -- see "Notes".
    raise NotImplementedError(
      "Wire response_vector through the stopping API before using SPRTStopper."
    )

  # NOTE: once `BaseStopper.stop` accepts `response_vector`, the body of `stop`
  # becomes:
  #
  #   administered_arr = item_bank.get_items(administered_items)
  #   llr = self._log_likelihood_ratio(response_vector, administered_arr)
  #   self._last_llr = llr
  #   if llr >= self._log_A:
  #     self._last_classification = Classification.ABOVE_CUT
  #     return True
  #   if llr <= self._log_B:
  #     self._last_classification = Classification.BELOW_CUT
  #     return True
  #   if base_decision and self._last_classification is Classification.UNDECIDED:
  #     self._last_classification = (
  #       Classification.ABOVE_CUT if llr > 0 else Classification.BELOW_CUT
  #     )
  #   return base_decision
```

**Implementation notes:**

- The critical dependency is a response-vector-aware `BaseStopper.stop` contract.
- The classification result should be exposed separately from the boolean stop signal.
- This family belongs in a dedicated `classification.py` module rather than the precision-stopper module.

**Dependencies:**

- Requires a stopper API extension to pass `response_vector` through the engine path.

**Testing notes:**

- Simulate examinees above and below the cut score and check empirical error rates against nominal `alpha` and `beta`.
- Verify correct upper-bound/lower-bound classification.
- Verify fallback classification behavior when `max_items` is reached before a bound crossing.
