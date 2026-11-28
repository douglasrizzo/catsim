---
title: Generalized Likelihood Ratio Stopper
kind: technique
status: ready
priority: 3
effort: M
blocked_by: []
raised_by: []
---

**Module:** `catsim.stopping.classification.GLRStopper` (planned)
**Reference:** Bartroff, Finkelman, and Lai (2008)
**Last validated:** 2026-11-23, against Bartroff, Finkelman & Lai (2008, arXiv copy), Magis & Raîche (2012) and catR 3.17.
**Implementation guidance:** written against the strategy contract of ADR-0007 (`StepContext` in, a pure `classify(context)`, a single criterion for `StopRule`) on 2026-11-26.

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

The SPRT spec compares the likelihood-ratio rules with catsim's
confidence-interval stopper.

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
```

## Behavioral Contracts

Any correct implementation must satisfy the following properties:

1. The statistic for classifying above the cut is defined only when the current estimate exceeds $\theta_c - \delta$, and the one for classifying below only when it is under $\theta_c + \delta$; inside the indifference region both are defined, and either can stop the test.
2. A statistic reaching its threshold classifies on its side of the cut; when both reach their thresholds, the rule classifies above the cut.
3. The statistic is computed from the current estimate plus boundary-point likelihoods, not from two fixed-point likelihoods as in SPRT.
4. The stopper reports the statistics and the classification of any session from that session's context, through pure methods; it keeps neither between calls.
5. Like SPRT, the rule reads the responses from the context; unlike it, it also reads the current estimate, the context's `theta`.

## References

Bartroff, J., Finkelman, M. D., & Lai, T. L. (2008). Modern sequential analysis and its applications to computerized adaptive testing. *Psychometrika*, 73(3), 473-486. <https://doi.org/10.1007/s11336-007-9053-9>. Open copy: [arXiv](https://arxiv.org/abs/1106.2559) (author manuscript; cited here by section, equation and table).

Magis, D., & Raîche, G. (2012). Random generation of response patterns under computerized adaptive testing with the R package catR. *Journal of Statistical Software*, 48(8), 1-31. <https://doi.org/10.18637/jss.v048.i08> (open access).

## Implementation Guidance

:::{note}
This section is for contributors and agents implementing this technique.
It is removed when the spec is promoted to a technique page.
:::

**Proposed location:** `src/catsim/stopping/classification.py`, next to `SPRTStopper` and the
`Classification` enum its spec defines.

**Mock implementation:**

```python
# src/catsim/stopping/classification.py (added to the module)


@dataclass(frozen=True)
class GLRStopper(BaseStopper):
  r"""Stop once a generalized likelihood ratio places the current estimate on one side of a cut score.

  The statistic for classifying above the cut compares the likelihood at the current estimate with
  the likelihood at :math:`\theta_c - \delta`, and the one for classifying below compares it with
  the likelihood at :math:`\theta_c + \delta` (Bartroff et al., 2008, Section 2.1).

  Parameters
  ----------
  cut_score : float
      Classification cut point :math:`\theta_c`.
  indifference : float, optional
      Half-width :math:`\delta` of the indifference region. Must be positive. Default is 0.2.
  alpha : float, optional
      Nominal Type I error rate, from which the default threshold ``log(1 / alpha)`` comes. Default
      is 0.05.
  threshold : float or None, optional
      Threshold of both statistics, overriding the one from `alpha`. Must be positive. Default is
      None.

  Raises
  ------
  ValueError
      If `indifference` or `threshold` is not positive, or `alpha` is not in (0, 1).
  """

  cut_score: float
  indifference: float = 0.2
  alpha: float = 0.05
  threshold: float | None = None

  def __post_init__(self) -> None:
    """Validate the indifference region, the error rate and the threshold."""
    # written so that NaN fails the checks too
    if not self.indifference > 0:
      msg = f"indifference must be positive, got {self.indifference}"
      raise ValueError(msg)
    if not 0 < self.alpha < 1:
      msg = f"alpha must be in (0, 1), got {self.alpha}"
      raise ValueError(msg)
    if self.threshold is not None and not self.threshold > 0:
      msg = f"threshold must be positive, got {self.threshold}"
      raise ValueError(msg)

  @property
  def stopping_threshold(self) -> float:
    """Return the threshold both statistics are compared with: `threshold`, or log(1 / alpha)."""
    return self.threshold if self.threshold is not None else math.log(1 / self.alpha)

  def statistics(self, context: StepContext) -> tuple[float, float]:
    """Return the statistics for classifying above and below the cut, -inf where one is not defined.

    Both are defined, and can be positive, while the estimate lies inside the indifference region.

    Returns
    -------
    tuple[float, float]
        The log-likelihood at ``context.theta`` minus that at the lower point, if the estimate
        is above the lower point, and minus that at the upper point, if it is below the upper one.
    """
    theta_hat = context.theta
    lower, upper = self.cut_score - self.indifference, self.cut_score + self.indifference
    items = context.item_bank.get_items(context.administered_items)
    at_hat, at_lower, at_upper = irt.log_likelihood(
      numpy.array([theta_hat, lower, upper]), [context.responses], items
    )[0]
    above = float(at_hat - at_lower) if theta_hat > lower else -math.inf
    below = float(at_hat - at_upper) if theta_hat < upper else -math.inf
    return above, below

  def _decision(self, context: StepContext) -> Classification | None:
    """Return the side whose statistic reaches the threshold, above the cut if both do, or None."""
    above, below = self.statistics(context)
    if above >= self.stopping_threshold:
      return Classification.ABOVE_CUT
    if below >= self.stopping_threshold:
      return Classification.BELOW_CUT
    return None

  @override
  def stop(self, context: StepContext) -> bool:
    """Return whether either statistic has reached the threshold.

    Parameters
    ----------
    context : StepContext
        The session to check.

    Returns
    -------
    bool
        True if a statistic is at least the threshold, False otherwise.
    """
    return self._decision(context) is not None

  def classify(self, context: StepContext) -> Classification:
    """Return the side of the cut score on which the session is classified.

    Once a statistic reaches the threshold, it is that statistic's side, above the cut if both do.
    For a test that ends with neither, as at a length cap, it is the side of the cut on which the
    current estimate lies, the spec's fallback in place of Bartroff et al.'s Equation 10.

    Parameters
    ----------
    context : StepContext
        The session to classify.

    Returns
    -------
    Classification
        The classification.
    """
    decision = self._decision(context)
    if decision is not None:
      return decision
    return Classification.ABOVE_CUT if context.theta > self.cut_score else Classification.BELOW_CUT
```

**Implementation notes:**

- No per-session state: `stop` and `classify` compute the statistics from the context each time
  instead of storing the last classification or statistic. An application classifies a finished
  session from its context, `stopper.classify(engine.context(session))`, which carries the
  session's current estimate; the statistics and the fallback read it, so a context built by hand
  must set `theta`, whose default of 0 would classify wrongly without an error.
  `StopRule.satisfied` on the same context tells whether this stopper or the length cap ended the
  test.
- The statistics are not set to 0 inside the indifference region: the definition keeps both
  defined there, and the test can stop there (Contract 1).
- $\hat\theta$ is the context's `theta`, the engine's current estimate. The definition's
  $\hat\theta_n$ is the maximum-likelihood estimate, so pair the stopper with
  `MLEstimator`; with EAP or MAP the estimate is drawn toward the prior, its
  log-likelihood is below the maximum, and both statistics are smaller than the GLR's.
- The default threshold, $\log(1/\alpha)$ for both decisions, is the spec's uncalibrated choice;
  Bartroff et al. calibrate separate thresholds by simulation or normal approximation (Section
  2.1). Pass `threshold` to use a calibrated value.
- The stopper is one criterion: `StopRule(GLRStopper(cut_score=0.0), MaxLengthStopper(50),
  min_items=5)` gives Bartroff et al.'s minimum of 5 items and maximum of 50 (Section 3.1).
- `irt.log_likelihood` evaluates the three points in one call and works from the logits.

**Testing notes:**

- Hand-worked statistic: 20 two-parameter logistic items with $a = 1$ and $b = \theta_c = 0$, 15
  answered correctly, have their maximum-likelihood estimate at $\log 3 \approx 1.099$, where the
  log-likelihood is $15 \log 0.75 + 5 \log 0.25 \approx -11.247$; at $-0.2$ it is about
  $-14.963$. The statistic for classifying above is about $3.716$, above the default threshold
  $\log 20 \approx 2.996$, so a context with `theta = log(3)` stops the test above the cut.
- Contract 1: on the same items, 210 correct responses of 400 put the estimate at
  $\operatorname{logit}(0.525) \approx 0.100$, inside the region, where the statistic for
  classifying above still exceeds the threshold; a stopper that zeroed the statistics inside the
  region would never stop there.
- Contract 2: thresholds small enough for both statistics classify above the cut.
- Contract 4 and 5: classifying two sessions alternately on one stopper gives what classifying
  each alone gives, and the same responses with another `theta` give other statistics.
- `threshold` overrides `alpha`, and an invalid `threshold` is rejected.
