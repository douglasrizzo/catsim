---
title: Stochastic Curtailment Stopper
kind: technique
status: ready
priority: 3
effort: M
blocked_by: []
raised_by: []
---

**Module:** `catsim.stopping.classification.StochasticCurtailmentStopper` (planned)
**Reference:** Finkelman (2008, 2010)
**Last validated:** 2026-11-23, against Huebner & Finkelman (2016) and Sie, Finkelman, Bartroff & Thompson (2015) (PubMed Central full texts), the abstracts of Finkelman (2008, 2010) and catR 3.17.
**Implementation guidance:** written against the strategy contract of ADR-0007 (`StepContext` in, a pure `classify(context)`, a single criterion for `StopRule`) on 2026-11-26.

# Stochastic Curtailment Stopper

## Motivation

SPRT and GLR stop only when their decision statistic already crosses a stopping
boundary. With a maximum test length, the truncated SPRT can keep administering
items whose responses will not change the final classification (Huebner &
Finkelman, 2016, Introduction). Stochastic curtailment, introduced for clinical
trials (Sie et al., 2015, Introduction), stops as soon as the probability that
the classification would change before the end of the test is small.

Finkelman (2008) applied it to the truncated SPRT. The curtailed test stops
whenever the SPRT does, so it never administers more items; in his simulation
with an operational item bank, it shortened the test by about 7 to 10 items for
examinees near the cut score, with virtually no loss of accuracy (Huebner &
Finkelman, 2016, Introduction and "SCSPRT").

## Definition

Let $J$ be the maximum test length, $\Lambda_k$ the SPRT log-likelihood ratio
after $k$ items, $A$ and $B$ the SPRT bounds (see the SPRT spec), and
$C = (A + B)/2$ their midpoint, at which the truncated SPRT forces a decision
after item $J$: above the cut if $\Lambda_J \ge C$, below otherwise. After $k$
items, the interim decision $D_k^*$ is the one the same rule gives now: above
the cut if $\Lambda_k \ge C$, below otherwise. The key probability is

$$
p_k = \Pr_{\theta^*}\left[ D_J = D_k^* \mid u_1, \dots, u_k \right],
$$

the probability that the decision forced after item $J$, computed from the
responses so far and the responses to the $J - k$ items still to come, agrees
with the interim decision. It is evaluated at the boundary point least
favorable to the interim decision: $\theta^* = \theta_c - \delta$ when $D_k^*$ is
above the cut and $\theta^* = \theta_c + \delta$ when it is below. This is the
conservative "conditional power" approach Finkelman (2008) took from clinical
trials (Huebner & Finkelman, 2016, "Computation of $P_{Exact}$ and $P_{CLT}$");
Sie et al. (2015, "Standard Formulation of Stochastic Curtailment") state it as
requiring the probability to reach the threshold under both $\theta_c - \delta$
and $\theta_c + \delta$.

After $k < J$ items, the stopper (Huebner & Finkelman, 2016, "SCSPRT"):

1. stops with the SPRT's decision if $\Lambda_k \ge A$ or $\Lambda_k \le B$;
2. otherwise stops with $D_k^*$ if $p_k \ge \gamma$;
3. otherwise administers another item.

After item $J$ the decision is forced by $C$. Finkelman (2008, 2010) used
$\gamma = .95$ (Huebner & Finkelman, 2016, "SCSPRT").

$p_k$ depends on the $J - k$ items still to come. Finkelman (2008) computes it
exactly, by enumerating the $2^{J-k}$ possible sets of responses to those items, or by a
central-limit approximation from $\Lambda_k$, $\theta^*$ and the items'
parameters (Huebner & Finkelman, 2016, "Computation of $P_{Exact}$ and
$P_{CLT}$"). When items are selected adaptively, the items to come are not known
in advance; Finkelman (2008) used as a representative set the $J - k$
non-administered items with the largest Fisher information at the current
estimate (Sie et al., 2015, "Curtailed CI Stopping Rule").

The planned stopper uses that representative set and a normal approximation of
its own. Given $\theta^*$, each item $i$ of the set adds to $\Lambda$ an
independent increment with mean and variance

$$
\mu_i = P_i(\theta^*)\,r_i^{(1)} + Q_i(\theta^*)\,r_i^{(0)},
\qquad
\sigma_i^2 = P_i(\theta^*)\,\big[r_i^{(1)}\big]^2 + Q_i(\theta^*)\,\big[r_i^{(0)}\big]^2 - \mu_i^2,
$$

where $Q_i = 1 - P_i$,
$r_i^{(1)} = \log[P_i(\theta_c + \delta)/P_i(\theta_c - \delta)]$ and
$r_i^{(0)} = \log[Q_i(\theta_c + \delta)/Q_i(\theta_c - \delta)]$. $\Lambda_J$ is
then approximately normal with mean $m = \Lambda_k + \sum_i \mu_i$ and standard
deviation $s = (\sum_i \sigma_i^2)^{1/2}$, so $p_k \approx 1 - \Phi[(C - m)/s]$
when $D_k^*$ is above the cut and $p_k \approx \Phi[(C - m)/s]$ when it is below.
Finkelman's (2008) own central-limit formulas are not openly available, so this
form may differ from his in detail.

Finkelman (2010, abstract) estimates the key probability "more aggressively"
than the standard formulation, which shortens tests further; Sie et al. (2015,
"Probability Calculations of the Stochastically Curtailed CI Stopping Rule") use
variations from it that evaluate the probability at the current estimate or at
an end point of its confidence interval. The planned stopper keeps the standard
formulation.

Sie et al. (2015) apply stochastic curtailment to the confidence-interval rule,
catsim's confidence-interval stopper, as well.

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
     - Classification cut point inherited from the SPRT base.
   * - ``indifference``
     - ``0.2``
     - Indifference half-width.
   * - ``alpha``
     - ``0.05``
     - Type I error rate for the underlying SPRT rule.
   * - ``beta``
     - ``0.05``
     - Type II error rate for the underlying SPRT rule.
   * - ``gamma``
     - ``0.95``
     - Curtailment threshold: the test stops when the probability that the decision forced at
       ``max_items`` equals the interim decision is at least ``gamma``. Higher values are more
       conservative; Finkelman (2008, 2010) used 0.95.
   * - ``max_items``
     - required
     - Maximum test length :math:`J`; the decision is forced there, and the key probability looks
       ahead to it.
```

## Behavioral Contracts

Any correct implementation must satisfy the following properties:

1. If the underlying SPRT rule already stops, stochastic curtailment must respect that decision immediately.
2. Curtailment evaluates the probability that the decision forced at `max_items` agrees with the interim decision, at the boundary point least favorable to that decision; it is not a second independent classification statistic.
3. If that probability is at least `gamma`, the rule stops and emits the interim decision: above the cut if the log-likelihood ratio is at least the midpoint of the SPRT bounds, below otherwise.
4. The rule never administers more items than the SPRT with the same bounds and maximum length.
5. Invalid `gamma` values and a missing `max_items` are rejected at construction time.
6. Like the SPRT, the rule reads the responses from the context; it reads the context's `theta` only to choose the representative items still to come.

## References

Finkelman, M. (2008). On using stochastic curtailment to shorten the SPRT in sequential mastery testing. *Journal of Educational and Behavioral Statistics*, 33(4), 442-463. <https://doi.org/10.3102/1076998607308573>. Not openly available; Huebner and Finkelman (2016) describe the method.

Finkelman, M. D. (2010). Variations on stochastic curtailment in sequential mastery testing. *Applied Psychological Measurement*, 34(1), 27-45. <https://doi.org/10.1177/0146621609336113>. Not openly available; Sie et al. (2015) describe some of its variations.

Huebner, A. R., & Finkelman, M. D. (2016). On computing the key probability in the stochastically curtailed sequential probability ratio test. *Applied Psychological Measurement*, 40(2), 142-156. <https://doi.org/10.1177/0146621615611633>. Open copy: [PubMed Central](https://pmc.ncbi.nlm.nih.gov/articles/PMC5982174/) (full text without page numbers; cited here by section).

Sie, H., Finkelman, M. D., Bartroff, J., & Thompson, N. A. (2015). Stochastic curtailment in adaptive mastery testing: Improving the efficiency of confidence interval-based stopping rules. *Applied Psychological Measurement*, 39(4), 278-292. <https://doi.org/10.1177/0146621614561314>. Open copy: [PubMed Central](https://pmc.ncbi.nlm.nih.gov/articles/PMC5978543/) (full text without page numbers; cited here by section).

## Open Questions

To be answered before the stopper is implemented, and removed from the spec when it is promoted.

1. **Where to evaluate the agreement probability.** This spec uses the standard form above: the
   probability is evaluated at the boundary point least favorable to the interim decision. The
   alternative is a plug-in variant that evaluates it at the current estimate $\hat{\theta}_k$,
   which is the rule this spec described before its revalidation. The plug-in variant does not
   take the worse of the two boundary points, so it lacks the standard form's conservatism; the
   standard form is the one the sources above state. Decide whether catsim offers the standard form
   only, the plug-in variant only, or both behind a parameter. Finkelman (2010) studies variations
   of stochastic curtailment; it is not openly available, so read it before choosing a variant from
   it.

## Implementation Guidance

:::{note}
This section is for contributors and agents implementing this technique.
It is removed when the spec is promoted to a technique page.
:::

**Proposed location:** `src/catsim/stopping/classification.py`, next to `SPRTStopper`, which it
extends.

**Mock implementation:**

```python
# src/catsim/stopping/classification.py (added to the module; the imports are those it does not have yet)

from scipy import stats

from .bank import _remaining

if TYPE_CHECKING:
  from irtlib.typing import FloatArray


@dataclass(frozen=True, kw_only=True)
class StochasticCurtailmentStopper(SPRTStopper):
  r"""Stop the SPRT early once its decision at the maximum length is unlikely to differ from the interim one.

  The interim decision is the one the truncated SPRT would force now: above the cut if the
  log-likelihood ratio is at least the midpoint of the bounds. The stopper stops when the SPRT
  does, or when the probability that the decision forced after `max_items` items agrees with the
  interim decision is at least `gamma`, evaluated at the boundary point least favorable to it.

  Parameters
  ----------
  cut_score, indifference, alpha, beta
      As for :class:`SPRTStopper`.
  max_items : int
      Maximum test length :math:`J`, at which the decision is forced. Keyword-only and required.
  gamma : float, optional
      Agreement probability at which the test is curtailed, in (0, 1). Default is 0.95.

  Raises
  ------
  ValueError
      If `max_items` is below 1 or `gamma` is not in (0, 1), or as for :class:`SPRTStopper`.
  """

  max_items: int
  gamma: float = 0.95

  def __post_init__(self) -> None:
    """Validate the SPRT's parameters, the maximum length and the curtailment threshold."""
    super().__post_init__()
    if self.max_items < 1:
      msg = f"max_items must be at least 1, got {self.max_items}"
      raise ValueError(msg)
    # written so that NaN fails the check too
    if not 0 < self.gamma < 1:
      msg = f"gamma must be in (0, 1), got {self.gamma}"
      raise ValueError(msg)

  @property
  def midpoint(self) -> float:
    """Return the midpoint of the SPRT's bounds, which decides the interim and the forced classification."""
    return (self.upper + self.lower) / 2

  @override
  def classify(self, context: StepContext) -> Classification:
    """Return the interim decision, which is also the decision forced after `max_items` items.

    Parameters
    ----------
    context : StepContext
        The session to classify.

    Returns
    -------
    Classification
        ``ABOVE_CUT`` if the log-likelihood ratio is at least the midpoint of the bounds,
        ``BELOW_CUT`` otherwise.
    """
    above = self.log_likelihood_ratio(context) >= self.midpoint
    return Classification.ABOVE_CUT if above else Classification.BELOW_CUT

  def agreement_probability(self, context: StepContext) -> float:
    """Return the normal approximation of the probability that the forced decision agrees with the interim one.

    Returns
    -------
    float
        The probability :math:`p_k`, 1 once no item is to come.
    """
    candidates = _remaining(context)
    to_come = self.max_items - len(context.administered_items)
    if to_come <= 0 or candidates.size == 0:
      return 1.0
    statistic = self.log_likelihood_ratio(context)
    above = statistic >= self.midpoint
    item_bank = context.item_bank
    # The representative items to come: the most informative remaining ones at the current
    # estimate, as Finkelman (2008) chose them.
    information = irt.information(context.theta, item_bank.get_items(candidates))
    items = item_bank.get_items(candidates[numpy.argsort(-information, kind="stable")[:to_come]])
    # The boundary point least favorable to the interim decision. Open Question 1: the plug-in
    # variant would evaluate at context.theta instead.
    theta_star = self.cut_score - self.indifference if above else self.cut_score + self.indifference
    increase, variance = self._increase(items, theta_star)
    mean, sd = statistic + increase, math.sqrt(variance)
    if sd == 0:
      return 1.0 if (mean >= self.midpoint) == above else 0.0
    z = (self.midpoint - mean) / sd
    return float(stats.norm.sf(z) if above else stats.norm.cdf(z))

  def _increase(self, items: FloatArray, theta: float) -> tuple[float, float]:
    """Return the mean and the variance of the statistic's increase over ``items``, answered at ``theta``."""
    lower, upper = self.cut_score - self.indifference, self.cut_score + self.indifference
    # from the logits, so an item whose probability rounds to 0 or 1 at both points gives finite ratios
    log_p_upper, log_q_upper = irt.log_response_probabilities(upper, items)
    log_p_lower, log_q_lower = irt.log_response_probabilities(lower, items)
    correct, incorrect = log_p_upper - log_p_lower, log_q_upper - log_q_lower
    probability = irt.icc(theta, items)
    means = probability * correct + (1 - probability) * incorrect
    variances = probability * correct**2 + (1 - probability) * incorrect**2 - means**2
    return float(means.sum()), max(float(variances.sum()), 0.0)

  @override
  def stop(self, context: StepContext) -> bool:
    """Return whether the SPRT stops, or the forced decision is likely enough to agree with the interim one.

    Parameters
    ----------
    context : StepContext
        The session to check.

    Returns
    -------
    bool
        True if the log-likelihood ratio has reached a bound or the agreement probability is at
        least `gamma`, False otherwise.
    """
    return super().stop(context) or self.agreement_probability(context) >= self.gamma
```

**Implementation notes:**

- The sketch implements the spec's definition: the agreement probability at the boundary point
  least favorable to the interim decision, the interim and forced decisions by the midpoint of
  the bounds, the representative set of the $J - k$ most informative remaining items at the
  current estimate, and Finkelman's $\gamma = 0.95$. It shows the standard form, the spec's
  current answer to Open Question 1; the plug-in variant changes only the `theta_star` line.
- It extends `SPRTStopper`: it stops whenever the SPRT does (Contract 1, hence Contract 4), and
  overrides `classify` with the midpoint decision. With `alpha == beta` the midpoint is 0 and the
  two classifications differ only for a statistic of exactly 0. `kw_only=True` lets the new
  required `max_items` follow the SPRT's fields that have defaults.
- No per-session state: the probability and the classification are computed from the context.
  The stopper reads `theta` only to rank the items to come.
- `max_items` belongs to the stopper because the look-ahead needs it. The stopper stops at that
  length by itself, since no item is left to come, so a `MaxLengthStopper` is optional; if a rule
  has one, give it the same length:
  `StopRule(StochasticCurtailmentStopper(0.0, max_items=30), MaxLengthStopper(30), min_items=5)`.
- `_remaining` is the bank stoppers' helper (the minimum-information spec). Once two modules use
  it, move it to `stopping/base.py`.
- The log-likelihood ratios of the items to come use the log-probabilities that
  `irt.log_response_probabilities` computes from the logits. Taking `numpy.log1p(-p)` of the probabilities instead gives
  $-\infty$ for an item whose probability rounds to 1 at both boundary points, and the difference
  of two such values is NaN, which the comparison with `gamma` would silently read as False.
- The look-ahead assumes the items to come are about as informative as the representative set,
  which holds for an information-based selector and less for an exposure-controlled one.

**Testing notes:**

- Hand-worked approximation: on a bank of two-parameter logistic items with $a = 1$ and
  $b = \theta_c = 0$, $\delta = 0.2$, each item to come adds $\pm 0.2$; at $\theta^* = -0.2$ its
  mean is $0.2\,(2\sigma(-0.2) - 1) \approx -0.019934$ and its variance about $0.039603$. With
  $J - k = 10$ and $\Lambda_k = 1.0$, the agreement probability is
  $\Phi(0.80066 / 0.62931) \approx 0.8984$, so the test continues; with $\Lambda_k = 2.0$ it is
  about $0.9979$ and the test stops above the cut (Contract 3).
- Contract 1: a statistic at a bound stops the test whatever the agreement probability.
- Contract 4: a seeded session with the curtailed stopper is a prefix of the session with the
  plain SPRT and the same seed, since `stop` draws no random numbers.
- Contract 5: `gamma` outside (0, 1), NaN, `max_items` below 1 and a missing `max_items` are
  rejected at construction.
- At $k = J$ the agreement probability is 1, so the stopper stops with the forced decision.
