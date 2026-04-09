Stochastic Curtailment Stopper
==============================

:Status: Planned
:Module: ``catsim.stopping.classification.StochasticCurtailmentStopper`` (planned)
:Reference: Finkelman (2008, 2010)

Motivation
----------

SPRT and GLR stop only when their decision statistic already crosses a stopping
boundary. Stochastic curtailment goes one step further: it stops when future
responses are very unlikely to change the eventual classification decision.

This provides another layer of test shortening on top of sequential
classification, especially for examinees who are already clearly inside one
decision region.

Definition
----------

Let :math:`\Lambda_n` be the current sequential classification statistic and let
:math:`\hat{\theta}_n` be the current plug-in estimate. Stochastic curtailment
computes the probability that the eventual classification outcome is still
undecided or flips away from the currently implied decision if testing
continues.

In the fixed-horizon approximation described in the card, the key quantity is

.. math::

   \gamma_n
   =
   \Pr_{\hat{\theta}_n}
   \left[
     B < \Lambda_N < A \mid \mathcal{F}_n
   \right],

the probability that the future trajectory remains undecided through the chosen
horizon. If :math:`\gamma_n \le \gamma_{\text{target}}`, the test is curtailed
and the current trajectory's implied classification is declared.

Parameters
----------

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
     - ``0.05``
     - Curtailment threshold; lower values are more conservative.
   * - ``horizon``
     - ``20``
     - Number of future items considered in the forward-projection approximation.
   * - ``min_items``
     - ``None``
     - Minimum length before curtailment may occur.
   * - ``max_items``
     - ``None``
     - Hard maximum length.

Behavioral Contracts
--------------------

Any correct implementation must satisfy the following properties:

1. If the underlying SPRT rule already stops, stochastic curtailment must respect that decision immediately.
2. Curtailment depends on a forward projection under the current estimate; it is not a second independent classification statistic.
3. If the probability of remaining undecided is less than or equal to ``gamma``, the rule stops and emits the tentative classification implied by the projected trajectory.
4. Invalid ``gamma`` or ``horizon`` values are rejected at construction time.
5. The rule inherits the response-vector dependency of the SPRT family.

References
----------

Finkelman, M. D. (2008). On using stochastic curtailment to shorten the SPRT in sequential mastery testing. *Journal of Educational and Behavioral Statistics*, 33(4), 442-463. https://doi.org/10.3102/1076998607302623

Finkelman, M. D. (2010). Variations on stochastic curtailment in sequential mastery testing. *Applied Psychological Measurement*, 34(1), 27-45. https://doi.org/10.1177/0146621609349807

Implementation Guidance
-----------------------

.. note::

   This section is for contributors and agents implementing this technique.
   It is removed when the spec is promoted to "Implemented" status.

**Proposed location:** ``src/catsim/stopping/classification.py``

**Mock implementation:**

.. code-block:: python

   # src/catsim/stopping/classification.py  (alongside SPRTStopper/GLRStopper)
   """Stochastic curtailment for classification CATs (Finkelman 2008)."""

   import numpy
   import numpy.typing as npt

   from .. import irt
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
       remaining_items: npt.NDArray[numpy.floating],
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

**Implementation notes:**

- This rule should inherit from the implemented SPRT stopper rather than duplicate its stopping logic.
- The normal approximation is the first practical implementation; a Monte Carlo forward projection can be a later refinement.
- Curtailment assumptions depend on the future selector being reasonably informative.

**Dependencies:**

- Depends on planned card 16 (`SPRTStopper`).
- Requires the same response-vector stopper API extension.

**Testing notes:**

- Compare mean test length with and without curtailment on examinees far from the cut.
- Verify that very small ``gamma`` reproduces plain SPRT behavior.
- Validate that invalid parameter settings are rejected.
