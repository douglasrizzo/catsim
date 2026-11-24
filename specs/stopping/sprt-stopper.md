**Status:** Planned
**Module:** `catsim.stopping.classification.SPRTStopper` (planned)
**Reference:** Reckase (1983); Wald (1947)
**Last validated:** 2026-11-23, against Reckase (1981), Bartroff, Finkelman & Lai (2008, arXiv copy), Magis & Raîche (2012) and catR 3.17.
**Implementation guidance:** not yet updated to the 1.0 strategy contract (M22).

# Sequential Probability Ratio Test Stopper

## Motivation

Most catsim stoppers are precision-oriented. They are appropriate when the
goal is to estimate ability. When the real goal is a binary decision such as
pass/fail or master/non-master, catsim offers only the
{doc}`confidence-interval stopper </api/techniques/stopping/confidence-interval-stopper>`,
which decides from the ability estimate and its standard error.

The sequential probability ratio test (SPRT) decides from the likelihood of the
responses instead. Wald (1947) designed it for quality control, and Reckase
(1981, pp. 1-4) paired it with tailored testing: after each item, the test stops
as soon as the evidence for one side of the cut score is strong enough. For
independent, identically distributed observations the SPRT is optimal only when
the true ability is exactly one of its two hypothesized points; in an adaptive
test the responses are not identically distributed, and the test is truncated
at a maximum length, so that optimality theory does not apply (Bartroff et al.,
2008, Sections 1 and 2.1).

## Definition

For cut score $\theta_c$ and indifference half-width $\delta$, define
two point hypotheses

$$
H_0: \theta = \theta_c - \delta,
\qquad
H_1: \theta = \theta_c + \delta.
$$

Reckase (1981, p. 3) uses the limits of an indifference region around the cut
score as the two points: within it, either classification is acceptable.

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

where $\alpha$ is the probability of classifying above the cut an examinee at
$\theta_c - \delta$ and $\beta$ the probability of classifying below the cut an
examinee at $\theta_c + \delta$. The ratio, its bounds and the meaning of
$\alpha$ and $\beta$ are Reckase's (1981, p. 3, Equations 3-5), on the log scale.
Bartroff et al. (2008, Equation 5) write the bounds with $\alpha$ and $\beta$ in
each other's place; the two forms agree when $\alpha = \beta$. The rule is:

$$
\Lambda_n \ge A \Rightarrow \text{classify above cut},
\qquad
\Lambda_n \le B \Rightarrow \text{classify below cut},
$$

and continue testing otherwise.

With a maximum test length, Reckase (1981, p. 4) classified the examinees still
undecided after 20 items by whether the likelihood ratio exceeded 1, that is,
by the sign of $\Lambda_n$; the planned fallback at `max_items` follows him.
Later studies of this truncated SPRT classify by the midpoint $(A + B)/2$ of the
bounds instead (Bartroff et al., 2008, Equations 6 and 7), which is 0 when
$\alpha = \beta$. Truncation inflates the error rates: with a maximum of 50 items
and $\alpha = \beta \le .1$, Bartroff et al. (2008, Section 3.1, Table 1) found an
actual Type I error rate of about .16.

Reckase (1981, p. 5) assigned four grades by running one SPRT at each of three
cut scores, with an indifference region of one standard error of measurement on
either side of each; Weissman (2007) also pairs the SPRT with a four-category
test. The planned stopper takes a single cut score.

catR has no SPRT. Its only classification rule (`checkStopRule.R`,
`rule = "classification"`, catR 3.17) stops once the confidence interval
$\hat{\theta} \pm z\,SE(\hat{\theta})$, with $z$ set by catR's `alpha`, no
longer contains the cut score (Magis & Raîche, 2012, p. 11), the rule catsim
implements as the confidence-interval stopper. The SPRT departs from it: it
compares the likelihood at two points around the cut instead of using the
estimate and its standard error, its $\alpha$ and $\beta$ are classification
error rates at those points rather than the significance level of an interval,
and it needs an indifference region, which catR's rule does not have.

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

Bartroff, J., Finkelman, M. D., & Lai, T. L. (2008). Modern sequential analysis and its applications to computerized adaptive testing. *Psychometrika*, 73(3), 473-486. <https://doi.org/10.1007/s11336-007-9053-9>. Open copy: [arXiv](https://arxiv.org/abs/1106.2559) (author manuscript; cited here by section, equation and table).

Magis, D., & Raîche, G. (2012). Random generation of response patterns under computerized adaptive testing with the R package catR. *Journal of Statistical Software*, 48(8), 1-31. <https://doi.org/10.18637/jss.v048.i08> (open access).

Reckase, M. D. (1981). *The use of the sequential probability ratio test in making grade classifications in conjunction with tailored testing* (Research Report 81-4). Columbia: University of Missouri, Tailored Testing Research Laboratory. Open copy: [ERIC](https://files.eric.ed.gov/fulltext/ED208006.pdf).

Reckase, M. D. (1983). A procedure for decision making using tailored testing. In D. J. Weiss (Ed.), *New Horizons in Testing: Latent Trait Test Theory and Computerized Adaptive Testing* (pp. 237-255). New York: Academic Press. <https://doi.org/10.1016/B978-0-12-742780-5.50023-8>. Not openly available; Reckase (1981) describes the method.

Wald, A. (1947). *Sequential Analysis*. New York: Wiley. Not openly available; Reckase (1981, p. 3) gives its decision bounds.

Weissman, A. (2007). Mutual information item selection in adaptive classification testing. *Educational and Psychological Measurement*, 67(1), 41-58. <https://doi.org/10.1177/0013164406288164>. Not openly available; Reckase (1981) describes the SPRT.

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

import numpy

from .. import irt
from .._typing import FloatArray
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
    administered_items: FloatArray,
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
