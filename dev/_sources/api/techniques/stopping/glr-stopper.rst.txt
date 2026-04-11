Generalized Likelihood Ratio Stopper
====================================

:Status: Planned
:Module: ``catsim.stopping.classification.GLRStopper`` (planned)
:Reference: Bartroff, Finkelman, and Lai (2008); Finkelman (2010)

Motivation
----------

SPRT compares two fixed boundary points around the cut score. That is simple, but
it can be conservative because it does not use the full information already
contained in the current ability estimate.

The generalized likelihood ratio (GLR) rule plugs in the current MLE and compares
it with the nearest boundary point in the competing region. In practice this
often yields shorter classification tests than SPRT at comparable error control.

Definition
----------

Let :math:`\hat{\theta}_n` be the current MLE after :math:`n` items, with cut
score :math:`\theta_c` and indifference half-width :math:`\delta`. Define

.. math::

   \Lambda_n^{\mathrm{GLR}}
   =
   \begin{cases}
     \ell_n(\hat{\theta}_n) - \ell_n(\theta_c - \delta),
       & \hat{\theta}_n \ge \theta_c + \delta \\
     \ell_n(\hat{\theta}_n) - \ell_n(\theta_c + \delta),
       & \hat{\theta}_n \le \theta_c - \delta \\
     0, & \text{otherwise}
   \end{cases}

where :math:`\ell_n(\theta) = \log L(\mathbf{u} \mid \theta)`. A practical first
threshold is

.. math::

   \Lambda_n^{\mathrm{GLR}} \ge \log(1/\alpha),

though the original paper gives refined calibrations.

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

Behavioral Contracts
--------------------

Any correct implementation must satisfy the following properties:

1. If the current estimate lies inside the indifference region, the GLR statistic is zero and the rule does not classify on the statistic alone.
2. When the estimate is above the upper indifference boundary, the decision favored by the GLR statistic is "above cut"; when below the lower boundary, it is "below cut".
3. The statistic is computed from the current estimate plus boundary-point likelihoods, not from two fixed-point likelihoods as in SPRT.
4. The stopper exposes the last statistic value and the last classification outcome.
5. Like SPRT, the rule requires response-vector access through the stopping API.

References
----------

Bartroff, J., Finkelman, M. D., & Lai, T. L. (2008). Modern sequential analysis and its applications to computerized adaptive testing. *Psychometrika*, 73(3), 473-486. https://doi.org/10.1007/s11336-007-9053-9

Finkelman, M. D. (2010). Variations on stochastic curtailment in sequential mastery testing. *Applied Psychological Measurement*, 34(1), 27-45. https://doi.org/10.1177/0146621609349807

Implementation Guidance
-----------------------

.. note::

   This section is for contributors and agents implementing this technique.
   It is removed when the spec is promoted to "Implemented" status.

**Proposed location:** ``src/catsim/stopping/classification.py``

**Mock implementation:**

.. code-block:: python

   # src/catsim/stopping/classification.py  (alongside SPRTStopper)
   """Generalized Likelihood Ratio classification stopper (Bartroff, Finkelman & Lai, 2008)."""

   from typing import Any

   import numpy
   import numpy.typing as npt

   from .. import irt
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
       administered_items: npt.NDArray[numpy.floating],
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

**Implementation notes:**

- The main approximation choice is the threshold calibration; the naive ``log(1/alpha)`` threshold is acceptable for a first pass.
- This family should be implemented alongside SPRT so the classification enums and API extensions are shared.
- The statistic reuses the current estimate, so it naturally depends on an estimator that remains stable under the classification regime.

**Dependencies:**

- Requires the same response-vector stopper API extension as SPRT.
- Pairs naturally with the existing numerical MLE estimator.

**Testing notes:**

- Compare average test length against SPRT at matched nominal error settings.
- Verify behavior inside the indifference region and at both tails.
- Validate explicit-threshold override behavior.
