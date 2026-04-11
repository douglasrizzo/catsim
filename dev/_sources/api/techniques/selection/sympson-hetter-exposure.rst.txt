Sympson-Hetter Exposure Control
===============================

:Status: Planned
:Module: ``catsim.exposure.sympson_hetter.SympsonHetterControl`` (planned)
:Reference: Sympson and Hetter (1985); Stocking and Lewis (1998)

Motivation
----------

Sympson-Hetter (SH) is the classic operational exposure-control method for CAT.
It is conceptually orthogonal to item selection: a selector proposes an item,
then SH decides whether that proposed item is actually administered.

That separation is useful because the same control table can wrap different
selectors, and calibration happens offline rather than during live testing.

Definition
----------

For each item :math:`i`, SH stores an acceptance probability :math:`K_i`. Live
administration proceeds as follows:

1. Select a candidate item with the wrapped selector.
2. Draw a Bernoulli random variable with success probability :math:`K_i`.
3. If the draw succeeds, administer the item.
4. If the draw fails, temporarily mark the item ineligible for that examinee and ask the selector for another candidate.

The :math:`K_i` values are calibrated offline so the empirical exposure rate of
each item stays below a target :math:`r_{\max}`. Starting from
:math:`K_i^{(0)} = 1`, the basic update is

.. math::

   K_i^{(k+1)}
   =
   \min\left(1, K_i^{(k)} \frac{r_{\max}}{r_i^{(k)}}\right),

where :math:`r_i^{(k)}` is the empirical administration rate in calibration
iteration :math:`k`.

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``selector``
     - required
     - Wrapped selector that proposes candidates before SH acceptance filtering.
   * - ``k_values``
     - ``None``
     - Optional pre-calibrated SH acceptance table.

Behavioral Contracts
--------------------

Any correct implementation must satisfy the following properties:

1. Live administration separates candidate selection from candidate acceptance.
2. If ``k_values`` are missing, live selection is invalid until calibration has been performed.
3. A rejected item is not administered for the current examinee and the selector is queried again for another candidate.
4. Offline calibration updates item-specific acceptance probabilities toward the target marginal exposure rate.
5. The wrapper should preserve the selector contract so it can be slotted into the engine as a selector-like component.

References
----------

Sympson, J. B., & Hetter, R. D. (1985). Controlling item-exposure rates in computerized adaptive testing. *Proceedings of the 27th Annual Meeting of the Military Testing Association* (pp. 973-977). San Diego, CA: Navy Personnel Research and Development Center.

Stocking, M. L., & Lewis, C. (1998). Controlling item exposure conditional on ability in computerized adaptive testing. *Journal of Educational and Behavioral Statistics*, 23(1), 57-75. https://doi.org/10.3102/10769986023001057

Implementation Guidance
-----------------------

.. note::

   This section is for contributors and agents implementing this technique.
   It is removed when the spec is promoted to "Implemented" status.

**Proposed location:** ``src/catsim/exposure/sympson_hetter.py``

**Mock implementation:**

.. code-block:: python

   # src/catsim/exposure/sympson_hetter.py
   """Sympson-Hetter exposure control (1985)."""

   import numpy
   import numpy.typing as npt

   from ..exceptions import NoItemsAvailableError
   from ..item_bank import ItemBank
   from ..selection.base import BaseSelector

   FloatArray = npt.NDArray[numpy.floating]


   class SympsonHetterControl:
     """Wraps a selector with the Sympson-Hetter administration filter.

     Stores a calibrated per-item table of exposure-control parameters K_i.
     Calling ``select`` repeatedly invokes the wrapped selector and accepts the
     proposed item with probability K_i, falling back to the next best until one
     is accepted.

     Calibration is performed offline by ``calibrate`` against a population of
     simulated examinees.
     """

     def __init__(
       self,
       selector: BaseSelector,
       k_values: FloatArray | None = None,
     ) -> None:
       self._selector = selector
       self._k = k_values  # may be None until calibrated

     @property
     def k_values(self) -> FloatArray | None:
       return self._k

     # ---- live administration ----
     def select(
       self,
       item_bank: ItemBank,
       administered_items: list[int],
       est_theta: float,
       rng: numpy.random.Generator,
       exposure_rates: FloatArray | None = None,
     ) -> int | None:
       if self._k is None:
         raise RuntimeError("SympsonHetterControl was not calibrated; call .calibrate() first.")

       declined: list[int] = []
       for _ in range(item_bank.n_items):
         candidate = self._selector.select(
           item_bank,
           administered_items + declined,
           est_theta,
           rng=rng,
           exposure_rates=exposure_rates,
         )
         if candidate is None:
           raise NoItemsAvailableError("No item passed Sympson-Hetter filter.")
         if rng.uniform() <= self._k[candidate]:
           return candidate
         declined.append(candidate)
       raise NoItemsAvailableError("No item passed Sympson-Hetter filter.")

     # ---- offline calibration ----
     def calibrate(
       self,
       item_bank: ItemBank,
       simulate_session: callable,  # signature: (item_bank, k_values, rng) -> list[int] (administered indices)
       n_examinees: int = 1000,
       r_max: float = 0.20,
       max_iters: int = 30,
       tolerance: float = 0.005,
       rng: numpy.random.Generator | None = None,
     ) -> FloatArray:
       """Iterate the Sympson-Hetter update rule until per-item exposures are below r_max.

       ``simulate_session`` is a callable that runs one examinee through the full CAT
       using the *current* ``k_values`` and returns the list of administered item indices.
       The catsim engine should provide a thin adapter around ``CatEngine`` for this.
       """
       if rng is None:
         rng = numpy.random.default_rng()

       k = numpy.ones(item_bank.n_items, dtype=float)
       for _iteration in range(max_iters):
         counts = numpy.zeros(item_bank.n_items, dtype=int)
         for _ in range(n_examinees):
           administered = simulate_session(item_bank, k, rng)
           counts[administered] += 1
         rates = counts / n_examinees

         # SH update: K' = K * (r_max / r_i), clipped to [0, 1]
         with numpy.errstate(divide="ignore", invalid="ignore"):
           k_new = numpy.where(rates > 0, k * (r_max / rates), 1.0)
         k = numpy.minimum(k_new, 1.0)

         max_excess = float(numpy.max(rates - r_max))
         if max_excess < tolerance:
           break

       self._k = k
       return k

**Implementation notes:**

- A dedicated ``exposure/`` subpackage is justified because exposure control wraps selectors but is conceptually distinct from selection.
- The key design seam is the calibration callback; provide a helper that can drive ``CatEngine`` sessions without forcing users to write boilerplate.
- Conditional SH and content-balanced SH belong in follow-up work, not this first implementation.

**Dependencies:** none

**Testing notes:**

- Verify that uncalibrated live selection fails clearly.
- Compare exposure rates before and after calibration on a bank with a few dominant items.
- Confirm that ``K = 1`` reproduces the wrapped selector's behavior.
