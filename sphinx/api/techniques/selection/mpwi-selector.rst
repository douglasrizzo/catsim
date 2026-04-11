Maximum Posterior Weighted Information Selector
===============================================

:Status: Planned
:Module: ``catsim.selection.weighted_info.MPWISelector`` (planned)
:Reference: van der Linden (1998); Barrada, Veldkamp, and Olea (2009)

Motivation
----------

MPWI is the Bayesian counterpart to MLWI. Instead of weighting each candidate's
information curve by the likelihood of ability values, it weights by the current
posterior, which naturally combines the prior and the observed response history.

Once Bayesian infrastructure exists, this criterion is attractive because it
reuses the same posterior already needed by EAP and other posterior-based
selectors.

Definition
----------

For candidate item :math:`i`, MPWI is

.. math::

   \mathrm{MPWI}_i
   =
   \int I_i(\theta)\, p(\theta \mid \mathbf{u})\, d\theta.

On a quadrature grid :math:`\{\theta_q\}_{q=1}^{Q}` with posterior weights
:math:`p_q = p(\theta_q \mid \mathbf{u})`, the implementation guidance
approximates this as

.. math::

   \mathrm{MPWI}_i
   \approx
   \sum_{q=1}^{Q} I_i(\theta_q)\, p_q.

The rule selects the non-administered item with the largest MPWI score.

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``grid``
     - ``QuadratureGrid.uniform()``
     - Quadrature grid used for posterior weighting.
   * - ``log_prior``
     - ``normal_log_prior(0, 1)``
     - Prior used when the selector computes the posterior itself.
   * - ``estimator``
     - ``None``
     - Optional EAP estimator whose cached posterior is reused to avoid recomputation.
   * - ``r_max``
     - ``1.0``
     - Soft exposure cap.

Behavioral Contracts
--------------------

Any correct implementation must satisfy the following properties:

1. Scores are posterior-weighted expected item information values, not likelihood-weighted values.
2. If an estimator with a cached posterior is supplied, the selector may reuse it instead of recomputing the posterior.
3. With an empty response vector, the selector reduces to prior-weighted expected information.
4. The selector requires the response vector unless a reusable posterior is already available.
5. Exposure filtering applies before the final argmax, with fallback to the unrestricted remaining pool if every constrained candidate is excluded.

References
----------

van der Linden, W. J. (1998). Bayesian item selection criteria for adaptive testing. *Psychometrika*, 63(2), 201-216. https://doi.org/10.1007/BF02294772

Barrada, J. R., Veldkamp, B. P., & Olea, J. (2009). Multiple maximum exposure rates in computerized adaptive testing. *Applied Psychological Measurement*, 33(1), 58-73. https://doi.org/10.1177/0146621608315329

Implementation Guidance
-----------------------

.. note::

   This section is for contributors and agents implementing this technique.
   It is removed when the spec is promoted to "Implemented" status.

**Proposed location:** ``src/catsim/selection/weighted_info.py``

**Mock implementation:**

.. code-block:: python

   # src/catsim/selection/weighted_info.py  (alongside MLWISelector from card 09)
   """Posterior-weighted information selector (van der Linden, 1998)."""

   import numpy
   import numpy.typing as npt

   from .. import irt
   from ..estimation.bayesian import (
     LogPrior,
     QuadratureGrid,
     normal_log_prior,
     posterior,
   )
   from ..exceptions import NoItemsAvailableError
   from ..item_bank import ItemBank
   from .base import BaseSelector

   FloatArray = npt.NDArray[numpy.floating]


   class MPWISelector(BaseSelector):
     """Posterior-weighted information selector.

     Parameters
     ----------
     grid : QuadratureGrid or None
         Quadrature grid. Defaults to ``QuadratureGrid.uniform()`` (matches EAPEstimator).
     log_prior : callable or None
         Vectorized log-density of the prior. Defaults to N(0, 1).
     estimator : EAPEstimator or None
         If provided, the selector will reuse the estimator's ``last_posterior`` instead
         of recomputing the quadrature on every call. Strongly recommended in production.
     r_max : float
         Exposure cap.
     """

     def __init__(
       self,
       grid: QuadratureGrid | None = None,
       log_prior: LogPrior | None = None,
       estimator=None,  # type: EAPEstimator | None
       r_max: float = 1.0,
     ) -> None:
       if not 0 <= r_max <= 1:
         raise ValueError(f"r_max must be between 0 and 1, got {r_max}")
       super().__init__()
       self._grid = grid if grid is not None else QuadratureGrid.uniform()
       self._log_prior = log_prior if log_prior is not None else normal_log_prior()
       self._estimator = estimator
       self._r_max = float(r_max)
       # Pre-compute item information at grid nodes lazily, cached in item_bank._selector_cache.

     def __str__(self) -> str:
       return "Maximum Posterior Weighted Information Selector"

     # ---- core kernel ----
     def _info_table(self, item_bank: ItemBank) -> FloatArray:
       """(n_items, Q) matrix of item information evaluated at every quadrature node."""
       cache_key = f"mpwi:info_table:{id(self._grid)}"
       cached = item_bank._selector_cache.get(cache_key)
       if cached is not None:
         return cached
       nodes = self._grid.nodes
       table = numpy.array([
         [irt.inf(float(t), *item_bank.items[i, :4]) for t in nodes]
         for i in range(item_bank.n_items)
       ])
       item_bank._selector_cache[cache_key] = table
       return table

     def _get_posterior(
       self,
       item_bank: ItemBank,
       administered_items: list[int],
       response_vector: list[bool],
     ) -> FloatArray:
       """Reuse the estimator's posterior if available; otherwise compute one."""
       if self._estimator is not None and self._estimator.last_posterior is not None:
         return self._estimator.last_posterior
       if not administered_items:
         pp = self._log_prior(self._grid.nodes) + numpy.log(self._grid.weights)
         pp -= pp.max()
         pp = numpy.exp(pp)
         return pp / pp.sum()
       items = item_bank.items[administered_items, :4]
       return posterior(response_vector, items, self._grid, self._log_prior)

     def select(
       self,
       item_bank: ItemBank,
       administered_items: list[int],
       est_theta: float,  # noqa: ARG002 -- unused; the posterior owns the theta info
       rng: numpy.random.Generator | None = None,  # noqa: ARG002
       exposure_rates: npt.NDArray[numpy.floating] | None = None,
       response_vector: list[bool] | None = None,
     ) -> int | None:
       if response_vector is None:
         raise ValueError("MPWISelector requires the response_vector to compute the posterior.")

       pp = self._get_posterior(item_bank, administered_items, response_vector)
       info_table = self._info_table(item_bank)
       # MPWI_i = sum_q I_i(theta_q) * p(theta_q | u)
       scores = info_table @ pp

       candidate_mask = numpy.ones(item_bank.n_items, dtype=bool)
       candidate_mask[administered_items] = False
       if exposure_rates is None:
         exposure_rates = numpy.zeros(item_bank.n_items, dtype=float)
       pool = candidate_mask & (exposure_rates < self._r_max)
       if not pool.any():
         pool = candidate_mask
       if not pool.any():
         raise NoItemsAvailableError("There are no more items to apply.")

       candidates = numpy.flatnonzero(pool)
       return int(candidates[int(scores[candidates].argmax())])

**Implementation notes:**

- Cache the information table per item bank and grid because it is bank-static.
- Reusing ``EAPEstimator.last_posterior`` is the intended efficient path.
- This selector shares most of its structure with MLWI and should be implemented in the same module.

**Dependencies:**

- Planned card 01 (`Bayesian Estimation Infrastructure`)
- Recommended reuse of planned card 06 (`EAPEstimator`)

**Testing notes:**

- With no responses and a standard normal prior, scores should match prior-weighted information.
- Compare behavior with and without posterior reuse from ``EAPEstimator``.
- Validate exposure-cap fallback behavior.
