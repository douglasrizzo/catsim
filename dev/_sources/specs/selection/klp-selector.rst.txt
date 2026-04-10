Posterior Kullback-Leibler Selector
===================================

:Status: Planned
:Module: ``catsim.selection.kl.KLPSelector`` (planned)
:Reference: Chang and Ying (1996); catR documentation

Motivation
----------

The KL selector integrates a KL divergence uniformly over a window around the
current estimate. KLP replaces that uniform window with the current posterior,
which removes the need for a hand-tuned half-width schedule and lets the data
itself determine where the weighting mass should be.

This makes KLP a Bayesian refinement of KL that behaves broadly early in the
test and concentrates automatically later.

Definition
----------

For candidate item :math:`i`, KLP is

.. math::

   \mathrm{KLP}_i
   =
   \int K_i(\theta \,\|\, \hat{\theta})\, p(\theta \mid \mathbf{u})\, d\theta,

where

.. math::

   K_i(\theta \,\|\, \hat{\theta})
   =
   P_i(\hat{\theta}) \log \frac{P_i(\hat{\theta})}{P_i(\theta)}
   +
   [1-P_i(\hat{\theta})]
   \log \frac{1-P_i(\hat{\theta})}{1-P_i(\theta)}.

On a quadrature grid, the selector approximates this as a posterior-weighted dot
product between the KL kernel values and the posterior probabilities.

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
     - Optional EAP estimator whose cached posterior is reused.
   * - ``r_max``
     - ``1.0``
     - Soft exposure cap.

Behavioral Contracts
--------------------

Any correct implementation must satisfy the following properties:

1. KLP uses the posterior over theta, not a uniform window, as its integration weight.
2. The KL kernel still depends on ``est_theta`` as the reference point inside the divergence.
3. The implementation can reuse a cached posterior but cannot precompute the full KL table across steps because it changes with ``est_theta``.
4. With a diffuse prior and little response information, the selector behaves more globally; with a concentrated posterior, it becomes increasingly local.
5. Exposure filtering applies before the final argmax, with fallback to the unrestricted remaining pool.

References
----------

Chang, H.-H., & Ying, Z. (1996). A global information approach to computerized adaptive testing. *Applied Psychological Measurement*, 20(3), 213-229. https://doi.org/10.1177/014662169602000303

Magis, D., & Raiche, G. (2012). *catR* package documentation and ``nextItem`` routines.

Implementation Guidance
-----------------------

.. note::

   This section is for contributors and agents implementing this technique.
   It is removed when the spec is promoted to "Implemented" status.

**Proposed location:** ``src/catsim/selection/kl.py``

**Mock implementation:**

.. code-block:: python

   # src/catsim/selection/kl.py  (alongside KLSelector from card 03)
   """Posterior Kullback-Leibler selector."""

   import numpy
   import numpy.typing as npt

   from ..estimation.bayesian import (
     LogPrior,
     QuadratureGrid,
     normal_log_prior,
     posterior,
   )
   from ..exceptions import NoItemsAvailableError
   from ..item_bank import ItemBank
   from .base import BaseSelector
   from .kl import _kl_integrand  # reuse the kernel from card 03

   FloatArray = npt.NDArray[numpy.floating]


   class KLPSelector(BaseSelector):
     """Posterior-weighted Kullback-Leibler selector.

     Parameters
     ----------
     grid : QuadratureGrid or None
         Quadrature grid. Defaults to ``QuadratureGrid.uniform()``.
     log_prior : callable or None
         Vectorized log-density of the prior. Defaults to N(0, 1).
     estimator : EAPEstimator or None
         If provided, reuses ``estimator.last_posterior`` to avoid recomputing.
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

     def __str__(self) -> str:
       return "Posterior Kullback-Leibler Selector"

     # ---- core kernel ----
     def _kl_table(self, item_bank: ItemBank, theta_hat: float) -> FloatArray:
       """(n_items, Q) matrix of KL(theta || theta_hat) at every grid node.

       Unlike MPWI's info table, this matrix is theta-hat dependent, so it cannot
       be precomputed once per item bank -- rebuild each step.
       """
       nodes = self._grid.nodes
       out = numpy.empty((item_bank.n_items, nodes.size), dtype=float)
       for i in range(item_bank.n_items):
         a, b, c, d = item_bank.items[i, :4]
         for q, t in enumerate(nodes):
           out[i, q] = _kl_integrand(float(t), theta_hat, a, b, c, d)
       return out

     def _get_posterior(
       self,
       item_bank: ItemBank,
       administered_items: list[int],
       response_vector: list[bool],
     ) -> FloatArray:
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
       est_theta: float,
       rng: numpy.random.Generator | None = None,  # noqa: ARG002
       exposure_rates: npt.NDArray[numpy.floating] | None = None,
       response_vector: list[bool] | None = None,
     ) -> int | None:
       if response_vector is None:
         raise ValueError("KLPSelector requires the response_vector to compute the posterior.")

       pp = self._get_posterior(item_bank, administered_items, response_vector)
       kl_table = self._kl_table(item_bank, est_theta)
       # KLP_i = sum_q KL(theta_q || theta_hat)_i * p(theta_q | u)
       scores = kl_table @ pp

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

- Unlike MPWI, the per-step KL table depends on ``est_theta`` and cannot be cached once per bank.
- This selector belongs with the KL selector because they share the KL kernel.
- Posterior reuse from EAP is the intended efficient path.

**Dependencies:**

- Planned card 01 (`Bayesian Estimation Infrastructure`)
- Planned card 03 (`KLSelector`) for kernel reuse
- Recommended reuse of planned card 06 (`EAPEstimator`)

**Testing notes:**

- Compare against broad-window KL selection under diffuse priors.
- Verify posterior reuse and exposure-cap fallback behavior.
- Confirm that the selector still uses ``est_theta`` as the KL reference point.
