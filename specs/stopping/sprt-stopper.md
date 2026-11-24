**Status:** Planned
**Module:** `catsim.stopping.classification.SPRTStopper` (planned)
**Reference:** Reckase (1983); Wald (1947)
**Last validated:** 2026-11-23, against Reckase (1981), Bartroff, Finkelman & Lai (2008, arXiv copy), Magis & Raîche (2012) and catR 3.17.
**Implementation guidance:** updated to the 1.0 strategy contract (`StepContext` in, a pure `classify(context)`, a single criterion for `StopRule`) on 2026-11-24.

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
```

## Behavioral Contracts

Any correct implementation must satisfy the following properties:

1. The stopper uses the response vector directly through a log-likelihood ratio; `theta` itself is not the primary decision statistic.
2. Crossing the upper bound classifies above the cut and crossing the lower bound classifies below the cut.
3. If neither bound is crossed, the stopper does not stop the test; another stopper of its `StopRule`, such as a `MaxLengthStopper`, can, and the classification then falls back to the sign of the statistic.
4. The stopper reports the classification and the log-likelihood ratio of any session from that session's context, through pure methods; it keeps neither between calls.
5. The decision depends only on the context's administered items and responses; the context's `theta` and `se` are not read.

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

**Proposed location:** `src/catsim/stopping/classification.py`, a new module for the
classification stoppers (this one, GLR and stochastic curtailment).

**Mock implementation:**

```python
# src/catsim/stopping/classification.py
"""Stop a classification test once the responses place the examinee on one side of a cut score."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy

from .. import irt
from .base import BaseStopper

if TYPE_CHECKING:
  from ..context import StepContext


class Classification(StrEnum):
  """The side of the cut score on which a classification stopper places the examinee."""

  BELOW_CUT = "below_cut"
  ABOVE_CUT = "above_cut"


@dataclass(frozen=True)
class SPRTStopper(BaseStopper):
  r"""Stop once the sequential probability ratio test decides on which side of a cut score the examinee lies.

  The statistic is the log-likelihood ratio of the responses at :math:`\theta_c + \delta` and
  :math:`\theta_c - \delta`, and Wald's bounds come from the two error rates.

  Parameters
  ----------
  cut_score : float
      Classification cut point :math:`\theta_c`.
  indifference : float, optional
      Half-width :math:`\delta` of the indifference region. Must be positive. Default is 0.2.
  alpha : float, optional
      Nominal probability of classifying above the cut an examinee at :math:`\theta_c - \delta`.
      Default is 0.05.
  beta : float, optional
      Nominal probability of classifying below the cut an examinee at :math:`\theta_c + \delta`.
      Default is 0.05.

  Raises
  ------
  ValueError
      If `indifference` is not positive, or `alpha` and `beta` are not positive with a sum below 1,
      which puts the bounds on either side of 0.
  """

  cut_score: float
  indifference: float = 0.2
  alpha: float = 0.05
  beta: float = 0.05

  def __post_init__(self) -> None:
    """Validate the indifference region and the error rates."""
    # written so that NaN fails the checks too
    if not self.indifference > 0:
      msg = f"indifference must be positive, got {self.indifference}"
      raise ValueError(msg)
    if not (self.alpha > 0 and self.beta > 0 and self.alpha + self.beta < 1):
      msg = f"alpha and beta must be positive with a sum below 1, got {self.alpha} and {self.beta}"
      raise ValueError(msg)

  @property
  def upper(self) -> float:
    """Return Wald's upper bound, log((1 - beta) / alpha), at which the test classifies above the cut."""
    return math.log((1 - self.beta) / self.alpha)

  @property
  def lower(self) -> float:
    """Return Wald's lower bound, log(beta / (1 - alpha)), at which the test classifies below the cut."""
    return math.log(self.beta / (1 - self.alpha))

  def log_likelihood_ratio(self, context: StepContext) -> float:
    r"""Return the log-likelihood ratio of the responses at the upper and the lower point, 0 before any response.

    Returns
    -------
    float
        The statistic :math:`\Lambda_n`.
    """
    points = numpy.array([self.cut_score - self.indifference, self.cut_score + self.indifference])
    items = context.item_bank.get_items(context.administered_items)
    at_lower, at_upper = irt.log_likelihood_hpc(points, context.responses, items)
    return float(at_upper - at_lower)

  def stop(self, context: StepContext) -> bool:
    """Return whether the statistic has reached either bound.

    Parameters
    ----------
    context : StepContext
        The session to check.

    Returns
    -------
    bool
        True if the log-likelihood ratio is at least the upper bound or at most the lower one.
    """
    statistic = self.log_likelihood_ratio(context)
    return statistic >= self.upper or statistic <= self.lower

  def classify(self, context: StepContext) -> Classification:
    """Return the side of the cut score that the responses favor.

    Once a bound is reached, it is the side that bound decides. For a test that ends with neither
    reached, as at a length cap, it is Reckase's (1981, p. 4) fallback: above the cut if the
    likelihood ratio exceeds 1. The bounds lie on either side of 0, so both come down to the sign
    of the statistic.

    Parameters
    ----------
    context : StepContext
        The session to classify.

    Returns
    -------
    Classification
        ``ABOVE_CUT`` if the log-likelihood ratio is positive, ``BELOW_CUT`` otherwise.
    """
    return Classification.ABOVE_CUT if self.log_likelihood_ratio(context) > 0 else Classification.BELOW_CUT
```

**Implementation notes:**

- No per-session state: one stopper serves every session of a run, so `stop` and `classify`
  compute the statistic from the context each time instead of storing the last classification
  or statistic. An application classifies a finished session by building its context, the kind
  `StopRule.satisfied` takes: `StepContext(item_bank=bank, rng=rng,
  administered_items=tuple(session.administered_item_ids), responses=tuple(session.responses),
  theta=session.current_theta, se=session.current_se)`. The SPRT reads only the items and
  responses, but the other stoppers of a rule, and the GLR and curtailment stoppers, read the
  estimate and its error, which default to 0 and infinity. `classify` is the decision whether the
  SPRT or the cap ended the test; `StopRule.satisfied` on the same context tells which.
- The stopper is one criterion. A minimum length and the cap at which the fallback applies come
  from the rule: `StopRule(SPRTStopper(cut_score=0.0), MaxLengthStopper(20), min_items=5)`.
- The statistic needs no ability estimate, so the stopper works with any estimator.
- `irt.log_likelihood_hpc` evaluates both points in one call and works from the logits.
- The stochastic-curtailment stopper extends this class and forces its decision at the midpoint
  of the bounds, as its spec says; with `alpha == beta` the midpoint is 0 and the two agree except
  for a statistic of exactly 0.

**Testing notes:**

- Hand-worked statistic: for a two-parameter logistic item with $b = \theta_c$, a correct
  response adds $\log[\sigma(a\delta)/\sigma(-a\delta)] = a\delta$ to the statistic and an
  incorrect one subtracts it. With $a = 1$, $\delta = 0.2$ and the default error rates
  ($A = \log 19 \approx 2.944$), 15 more correct than incorrect responses stop the test above the
  cut, and 14 do not (Contract 2); 15 more incorrect ones stop it below.
- Contract 3: undecided at a length cap, `classify` follows the sign of the statistic.
- Contract 4 and 5: classifying two sessions alternately on one stopper gives what classifying
  each alone gives, and the same responses with another `theta` or `se` give the same decision.
- Simulated examinees at $\theta_c - \delta$ and $\theta_c + \delta$, with a cap far beyond the
  usual length, are misclassified at about the nominal rates or less.
