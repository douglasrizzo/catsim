**Status:** Planned
**Module:** `catsim.stopping.classification.GLRStopper` (planned)
**Reference:** Bartroff, Finkelman, and Lai (2008)
**Last validated:** 2026-11-23, against Bartroff, Finkelman & Lai (2008, arXiv copy), Magis & Raîche (2012) and catR 3.17.
**Implementation guidance:** not yet updated to the 1.0 strategy contract (M22).

# Generalized Likelihood Ratio Stopper

## Motivation

SPRT compares two fixed boundary points around the cut score. That is simple, but
it can be conservative because it does not use the full information already
contained in the current ability estimate.

The generalized likelihood ratio (GLR) rule plugs in the current MLE and compares
it with the nearest boundary point of the competing hypothesis, so the statistic
"self-tunes" to the information accumulating in the estimate (Bartroff et al.,
2008, Section 2.1). In their simulations with a maximum of 50 items, the GLR
test had nearly the same power function as a truncated SPRT whose final
threshold was adjusted to the nominal Type I error, and was considerably
shorter: about half as long for examinees above the indifference region
(Section 3.1, Table 2).

## Definition

Bartroff et al. (2008, Section 2.1) test the hypothesis of mastery,
$H_0: \theta \ge \theta_+$, against $H_1: \theta \le \theta_-$, where
$(\theta_-, \theta_+)$ is the indifference region around the cut score; catsim
takes $\theta_\pm = \theta_c \pm \delta$. Let $\hat{\theta}_n$ be the MLE after
$n$ items and $\ell_n(\theta) = \log L(\mathbf{u} \mid \theta)$ the
log-likelihood. The two GLR statistics are

$$
\Lambda_n^{+} = \ell_n(\hat{\theta}_n) - \ell_n(\theta_c - \delta),
\quad \text{defined when } \hat{\theta}_n > \theta_c - \delta,
$$

$$
\Lambda_n^{-} = \ell_n(\hat{\theta}_n) - \ell_n(\theta_c + \delta),
\quad \text{defined when } \hat{\theta}_n < \theta_c + \delta.
$$

$\Lambda_n^{+}$ weighs the evidence against the region below the cut and
$\Lambda_n^{-}$ the evidence against the region above it. Before the maximum
length $N$, the test stops and classifies below the cut if
$\Lambda_n^{-} \ge A$, and above the cut if $\Lambda_n^{+} \ge B$
(Equations 8 and 9); if both hold, which happens only when the thresholds are
artificially small, it classifies above the cut (Section 2.1). At $n = N$ it
classifies below the cut if and only if $\hat{\theta}_N < \theta_c + \delta$ and
$\Lambda_N^{-} \ge C$ (Equation 10). Unlike the SPRT statistic, both statistics
can be positive while $\hat{\theta}_n$ lies inside the indifference region, so
the test can stop there, where either classification is acceptable.

The planned stopper departs from Bartroff et al. in four ways:

- **Alternative.** Bartroff et al. compare with the implied alternative
  $\theta_-^{(N)}$, the ability at which a fixed-length likelihood ratio test of
  $N$ items has Type II error $\beta$, rather than with $\theta_-$
  (Section 2.1); catsim uses $\theta_c - \delta$, as the SPRT does.
- **Thresholds.** Wald's approximations do not apply to GLR statistics, so
  Bartroff et al. calibrate $A$, $B$ and $C$ to the target error rates by Monte
  Carlo simulation or by normal approximations (Section 2.1, Equations 11-19); in
  their example, $A = 3.7$, $B = 3.3$ and $C = 1.4$ for
  $\alpha = \beta = .05$ and $N = 50$ (Section 3.1). The planned default uses
  one uncalibrated threshold, $\log(1/\alpha)$ (about 3.0 at
  $\alpha = .05$), for both decisions; the `threshold` parameter overrides it.
- **Minimum length.** Bartroff et al. allow stopping only after $\rho N$ items
  ($\rho N = 5$ in Section 3.1); catsim's `min_items` plays that role.
- **Maximum length.** At `max_items`, the implementation guidance classifies by
  the side of $\theta_c$ on which $\hat{\theta}_n$ lies, instead of Equation 10.

catR has no GLR rule; its only classification rule is the confidence-interval
rule (`checkStopRule.R`, `rule = "classification"`, catR 3.17; Magis & Raîche,
2012, p. 11), which the SPRT spec compares with the likelihood-ratio rules.

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
     - Classification cut point.
   * - ``indifference``
     - ``0.2``
     - Half-width of the indifference region.
   * - ``alpha``
     - ``0.05``
     - Nominal Type I error rate used to derive a default threshold.
   * - ``threshold``
     - ``None``
     - Explicit stopping threshold; overrides the alpha-based default if provided.
   * - ``min_items``
     - ``None``
     - Minimum length before classification can be declared.
   * - ``max_items``
     - ``None``
     - Hard maximum length before fallback classification.
```

## Behavioral Contracts

Any correct implementation must satisfy the following properties:

1. The statistic for classifying above the cut is defined only when the current estimate exceeds $\theta_c - \delta$, and the one for classifying below only when it is under $\theta_c + \delta$; inside the indifference region both are defined, and either can stop the test.
2. A statistic reaching its threshold classifies on its side of the cut; when both reach their thresholds, the rule classifies above the cut.
3. The statistic is computed from the current estimate plus boundary-point likelihoods, not from two fixed-point likelihoods as in SPRT.
4. The stopper exposes the last statistic value and the last classification outcome.
5. Like SPRT, the rule requires response-vector access through the stopping API.

## References

Bartroff, J., Finkelman, M. D., & Lai, T. L. (2008). Modern sequential analysis and its applications to computerized adaptive testing. *Psychometrika*, 73(3), 473-486. <https://doi.org/10.1007/s11336-007-9053-9>. Open copy: [arXiv](https://arxiv.org/abs/1106.2559) (author manuscript; cited here by section, equation and table).

Magis, D., & Raîche, G. (2012). Random generation of response patterns under computerized adaptive testing with the R package catR. *Journal of Statistical Software*, 48(8), 1-31. <https://doi.org/10.18637/jss.v048.i08> (open access).

## Implementation Guidance

:::{note}
This section is for contributors and agents implementing this technique.
It is removed when the spec is promoted to "Implemented" status.
:::

**Proposed location:** `src/catsim/stopping/classification.py`

**Mock implementation:**

```python
# src/catsim/stopping/classification.py  (alongside SPRTStopper)
"""Generalized Likelihood Ratio classification stopper (Bartroff, Finkelman & Lai, 2008)."""

import numpy

from .. import irt
from .._typing import FloatArray
from ..item_bank import ItemBank
from .classification import Classification
from .stopping import TestLengthStopper


class GLRStopper(TestLengthStopper):
  """Generalized Likelihood Ratio classification stopper.

  Parameters
  ----------
  cut_score : float
      Ability threshold :math:`\\theta_c`.
  indifference : float
      Half-width :math:`\\delta` of the indifference region.
  alpha : float
      Nominal Type I error rate. The stopping threshold is ``log(1/alpha)``
      (Bartroff et al. 2008 recommend slight inflation; exposed as ``threshold``).
  threshold : float or None
      Explicit stopping threshold on the GLR statistic. Overrides ``alpha`` if given.
  min_items : int or None
      Minimum items before a classification can be declared.
  max_items : int or None
      Maximum items before forced classification.
  """

  def __init__(
    self,
    cut_score: float,
    indifference: float = 0.2,
    alpha: float = 0.05,
    threshold: float | None = None,
    min_items: int | None = None,
    max_items: int | None = None,
  ) -> None:
    if indifference <= 0:
      raise ValueError(f"indifference must be positive, got {indifference}")
    if not 0 < alpha < 1:
      raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    super().__init__(min_items=min_items, max_items=max_items)
    self._theta_low = cut_score - indifference
    self._theta_high = cut_score + indifference
    self._threshold = float(threshold) if threshold is not None else float(numpy.log(1.0 / alpha))
    self._last_classification = Classification.UNDECIDED
    self._last_statistic = 0.0

  def __str__(self) -> str:
    return f"GLRStopper(threshold={self._threshold:.3f})"

  @property
  def last_classification(self) -> Classification:
    return self._last_classification

  # ---- core kernel ----
  def _glr_statistic(
    self,
    theta_hat: float,
    response_vector: list[bool],
    administered_items: FloatArray,
  ) -> tuple[float, Classification]:
    """Return (statistic, proposed_decision)."""
    if self._theta_low < theta_hat < self._theta_high:
      return 0.0, Classification.UNDECIDED

    ll_hat = irt.log_likelihood(theta_hat, response_vector, administered_items)
    if theta_hat >= self._theta_high:
      ll_boundary = irt.log_likelihood(self._theta_low, response_vector, administered_items)
      return ll_hat - ll_boundary, Classification.ABOVE_CUT

    ll_boundary = irt.log_likelihood(self._theta_high, response_vector, administered_items)
    return ll_hat - ll_boundary, Classification.BELOW_CUT

  def stop(
    self,
    item_bank: ItemBank,
    administered_items: list[int],
    theta: float,
  ) -> bool:
    base_decision = super().stop(item_bank, administered_items, theta)
    if not administered_items:
      return base_decision

    # Requires response_vector through the stopper API (see card 16 for the API change).
    raise NotImplementedError(
      "Wire response_vector through the stopping API before using GLRStopper."
    )

  # Final body once the API change from card 16 is in place:
  #
  #   administered_arr = item_bank.get_items(administered_items)
  #   stat, decision = self._glr_statistic(theta, response_vector, administered_arr)
  #   self._last_statistic = stat
  #   if stat >= self._threshold:
  #     self._last_classification = decision
  #     return True
  #   if base_decision and self._last_classification is Classification.UNDECIDED:
  #     self._last_classification = (
  #       Classification.ABOVE_CUT if theta > (self._theta_low + self._theta_high) / 2
  #       else Classification.BELOW_CUT
  #     )
  #   return base_decision
```

**Implementation notes:**

- The main approximation choice is the threshold calibration; the naive `log(1/alpha)` threshold is acceptable for a first pass.
- This family should be implemented alongside SPRT so the classification enums and API extensions are shared.
- The statistic reuses the current estimate, so it naturally depends on an estimator that remains stable under the classification regime.

**Dependencies:**

- Requires the same response-vector stopper API extension as SPRT.
- Pairs naturally with the existing numerical MLE estimator.

**Testing notes:**

- Compare average test length against SPRT at matched nominal error settings.
- Verify behavior inside the indifference region and at both tails.
- Validate explicit-threshold override behavior.
