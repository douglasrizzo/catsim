**Status:** Planned
**Module:** `catsim.stopping.classification.StochasticCurtailmentStopper` (planned)
**Reference:** Finkelman (2008, 2010)
**Last validated:** 2026-11-23, against Huebner & Finkelman (2016) and Sie, Finkelman, Bartroff & Thompson (2015) (PubMed Central full texts), the abstracts of Finkelman (2008, 2010) and catR 3.17.
**Implementation guidance:** not yet updated to the 1.0 strategy contract (M22).

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
exactly, by enumerating the $2^{J-k}$ response patterns to those items, or by a
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

catR implements no curtailment. Its only classification rule is the
confidence-interval rule (`checkStopRule.R`, `rule = "classification"`,
catR 3.17), catsim's confidence-interval stopper; Sie et al. (2015) apply
stochastic curtailment to that rule as well.

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
   * - ``min_items``
     - ``None``
     - Minimum length before curtailment may occur.
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
6. The rule inherits the response-vector dependency of the SPRT family.

## References

Finkelman, M. (2008). On using stochastic curtailment to shorten the SPRT in sequential mastery testing. *Journal of Educational and Behavioral Statistics*, 33(4), 442-463. <https://doi.org/10.3102/1076998607308573>. Not openly available; Huebner and Finkelman (2016) describe the method.

Finkelman, M. D. (2010). Variations on stochastic curtailment in sequential mastery testing. *Applied Psychological Measurement*, 34(1), 27-45. <https://doi.org/10.1177/0146621609336113>. Not openly available; Sie et al. (2015) describe some of its variations.

Huebner, A. R., & Finkelman, M. D. (2016). On computing the key probability in the stochastically curtailed sequential probability ratio test. *Applied Psychological Measurement*, 40(2), 142-156. <https://doi.org/10.1177/0146621615611633>. Open copy: [PubMed Central](https://pmc.ncbi.nlm.nih.gov/articles/PMC5982174/) (full text without page numbers; cited here by section).

Sie, H., Finkelman, M. D., Bartroff, J., & Thompson, N. A. (2015). Stochastic curtailment in adaptive mastery testing: Improving the efficiency of confidence interval-based stopping rules. *Applied Psychological Measurement*, 39(4), 278-292. <https://doi.org/10.1177/0146621614561314>. Open copy: [PubMed Central](https://pmc.ncbi.nlm.nih.gov/articles/PMC5978543/) (full text without page numbers; cited here by section).

## Implementation Guidance

:::{note}
This section is for contributors and agents implementing this technique.
It is removed when the spec is promoted to "Implemented" status.
:::

**Proposed location:** `src/catsim/stopping/classification.py`

**Mock implementation:**

```python
# src/catsim/stopping/classification.py  (alongside SPRTStopper/GLRStopper)
"""Stochastic curtailment for classification CATs (Finkelman 2008)."""

import numpy

from .. import irt
from .._typing import FloatArray
from ..item_bank import ItemBank
from .classification import Classification, SPRTStopper


class StochasticCurtailmentStopper(SPRTStopper):
  """SPRT with stochastic curtailment. Inherits Wald logic from SPRTStopper.

  At each step, if the base SPRT stops, use its decision. Otherwise, estimate
  the probability that the SPRT will still be undecided by ``horizon`` items
  under the plug-in MLE :math:`\\hat\\theta_n`. If below ``gamma``, curtail.

  Parameters
  ----------
  cut_score : float
      Ability threshold.
  indifference : float
      Indifference half-width.
  alpha, beta : float
      SPRT error rates.
  gamma : float
      Curtailment threshold. Lower = more conservative.
  horizon : int
      Number of additional items to consider in the forward projection.
  min_items, max_items : int or None
      Usual bounds.
  """

  def __init__(
    self,
    cut_score: float,
    indifference: float = 0.2,
    alpha: float = 0.05,
    beta: float = 0.05,
    gamma: float = 0.05,
    horizon: int = 20,
    min_items: int | None = None,
    max_items: int | None = None,
  ) -> None:
    if not 0 < gamma < 1:
      raise ValueError(f"gamma must be in (0, 1), got {gamma}")
    if horizon < 1:
      raise ValueError(f"horizon must be positive, got {horizon}")
    super().__init__(
      cut_score=cut_score,
      indifference=indifference,
      alpha=alpha,
      beta=beta,
      min_items=min_items,
      max_items=max_items,
    )
    self._gamma = float(gamma)
    self._horizon = int(horizon)

  def __str__(self) -> str:
    return f"StochasticCurtailmentStopper(gamma={self._gamma}, horizon={self._horizon})"

  # ---- core kernel ----
  def _curtailment_probability(
    self,
    theta_hat: float,
    remaining_items: FloatArray,
    current_llr: float,
  ) -> tuple[float, Classification]:
    """Normal-approximation forward projection of the LLR over the horizon.

    Returns (undecided_probability, tentative_decision).
    """
    if remaining_items.shape[0] == 0:
      decision = Classification.ABOVE_CUT if current_llr > 0 else Classification.BELOW_CUT
      return 0.0, decision

    # Per-item LLR contribution under the plug-in theta_hat:
    #   mean = P(theta_hat) * log(P_high/P_low) + (1 - P(theta_hat)) * log(Q_high/Q_low)
    a = remaining_items[:, 0]
    b = remaining_items[:, 1]
    c = remaining_items[:, 2]
    d = remaining_items[:, 3]

    p_hat = numpy.array([irt.icc(theta_hat, ai, bi, ci, di) for ai, bi, ci, di in zip(a, b, c, d)])
    p_high = numpy.array([irt.icc(self._theta_high, ai, bi, ci, di) for ai, bi, ci, di in zip(a, b, c, d)])
    p_low = numpy.array([irt.icc(self._theta_low, ai, bi, ci, di) for ai, bi, ci, di in zip(a, b, c, d)])

    eps = 1e-12
    log_ratio_correct = numpy.log(numpy.clip(p_high, eps, 1) / numpy.clip(p_low, eps, 1))
    log_ratio_wrong = numpy.log(numpy.clip(1 - p_high, eps, 1) / numpy.clip(1 - p_low, eps, 1))

    per_item_mean = p_hat * log_ratio_correct + (1 - p_hat) * log_ratio_wrong
    per_item_var = p_hat * log_ratio_correct**2 + (1 - p_hat) * log_ratio_wrong**2 - per_item_mean**2

    # Take the first `horizon` *best* items by information at theta_hat.
    k = min(self._horizon, remaining_items.shape[0])
    selected = numpy.argsort(-per_item_mean * numpy.sign(per_item_mean))[:k]
    mu = float(per_item_mean[selected].sum())
    sigma = float(numpy.sqrt(numpy.maximum(per_item_var[selected].sum(), 0.0)))

    if sigma == 0:
      final = current_llr + mu
      if final >= self._log_A:
        return 0.0, Classification.ABOVE_CUT
      if final <= self._log_B:
        return 0.0, Classification.BELOW_CUT
      return 1.0, Classification.UNDECIDED

    # P[B < Lambda_final < A] under N(current + mu, sigma^2)
    from scipy.stats import norm

    upper = (self._log_A - (current_llr + mu)) / sigma
    lower = (self._log_B - (current_llr + mu)) / sigma
    undecided = float(norm.cdf(upper) - norm.cdf(lower))

    # Tentative decision: lean in the direction of the expected final LLR.
    expected_final = current_llr + mu
    if expected_final >= 0:
      decision = Classification.ABOVE_CUT
    else:
      decision = Classification.BELOW_CUT
    return undecided, decision

  # The full stop() is built on top of SPRTStopper.stop() -- see card 16 for the
  # API change and the base implementation. Pseudocode:
  #
  #   sprt_stop = super().stop(item_bank, administered_items, theta)
  #   if sprt_stop:
  #     return True  # last_classification already set by SPRT
  #
  #   remaining = mask out administered items
  #   undecided_p, tentative = self._curtailment_probability(
  #     theta,
  #     item_bank.items[remaining, :4],
  #     self.last_log_likelihood_ratio,
  #   )
  #   if undecided_p <= self._gamma:
  #     self._last_classification = tentative
  #     return True
  #   return False
```

**Implementation notes:**

- This rule should inherit from the implemented SPRT stopper rather than duplicate its stopping logic.
- The normal approximation is the first practical implementation; a Monte Carlo forward projection can be a later refinement.
- Curtailment assumptions depend on the future selector being reasonably informative.

**Dependencies:**

- Depends on planned card 16 (`SPRTStopper`).
- Requires the same response-vector stopper API extension.

**Testing notes:**

- Compare mean test length with and without curtailment on examinees far from the cut.
- Verify that very small `gamma` reproduces plain SPRT behavior.
- Validate that invalid parameter settings are rejected.
