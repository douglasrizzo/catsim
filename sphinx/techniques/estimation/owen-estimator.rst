Owen Sequential Bayesian Estimator
==================================

:Status: Planned
:Module: ``catsim.estimation.bayesian.OwenEstimator`` (planned)
:Reference: Owen (1975); Vale and Weiss (1975)

Motivation
----------

Full-grid EAP and MAP revisit the complete response history each time the
estimate is updated. Owen's sequential Bayesian approach is designed for a
different operating point: constant-time streaming updates that carry only a
posterior mean and variance forward from one step to the next.

That makes it attractive when tests are long or latency matters more than exact
posterior fidelity.

Definition
----------

Let the current approximate posterior be represented as a Gaussian

.. math::

   \theta \mid \mathbf{u}_{1:n}
   \approx
   \mathcal{N}(\mu_n, \sigma_n^2).

After observing the next item response :math:`u_{n+1}` on item :math:`i_{n+1}`,
update by treating that Gaussian as the prior and applying one-step Bayesian
reweighting:

.. math::

   p(\theta \mid u_{n+1})
   \propto
   \phi\left(\frac{\theta - \mu_n}{\sigma_n}\right)
   P_{i_{n+1}}(\theta)^{u_{n+1}}
   [1-P_{i_{n+1}}(\theta)]^{1-u_{n+1}}.

Then match moments to obtain

.. math::

   \mu_{n+1} = \mathbb{E}[\theta \mid u_{n+1}],
   \qquad
   \sigma_{n+1}^2 = \mathrm{Var}[\theta \mid u_{n+1}].

The implementation guidance uses a local grid over
:math:`[\mu_n - h\sigma_n, \mu_n + h\sigma_n]` to compute these moments.

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``prior_mean``
     - ``0.0``
     - Initial Gaussian prior mean.
   * - ``prior_sd``
     - ``1.0``
     - Initial Gaussian prior standard deviation.
   * - ``n_nodes``
     - ``21``
     - Number of local quadrature nodes used in each sequential update.
   * - ``half_width``
     - ``5.0``
     - Local grid half-width in units of the current posterior standard deviation.
   * - ``reset_each_call``
     - ``False``
     - Whether to replay the whole history on every call instead of using incremental state.
   * - ``verbose``
     - ``False``
     - Optional diagnostics flag.

Behavioral Contracts
--------------------

Any correct implementation must satisfy the following properties:

1. With no administered items, the estimate equals the configured prior mean.
2. In incremental mode, the estimator updates only on newly observed responses and carries forward the current mean and standard deviation.
3. In replay mode, the estimate reconstructed from scratch must match the streamed estimate for the same response history.
4. Calling ``reset()`` clears the sequential state so the next estimate re-seeds from the configured prior.
5. On ordinary short tests, the sequential estimate should remain close to full-grid EAP, though not necessarily identical.

References
----------

Owen, R. J. (1975). A Bayesian sequential procedure for quantal response in the context of adaptive mental testing. *Journal of the American Statistical Association*, 70(350), 351-356. https://doi.org/10.1080/01621459.1975.10479896

Vale, C. D., & Weiss, D. J. (1975). *A study of computer-administered stradaptive ability testing* (Research Report 75-4). University of Minnesota.

van der Linden, W. J. (2008). Some new developments in adaptive testing. In *Handbook of Modern Item Response Theory*.

Implementation Guidance
-----------------------

.. note::

   This section is for contributors and agents implementing this technique.
   It is removed when the spec is promoted to "Implemented" status.

**Proposed location:** ``src/catsim/estimation/bayesian.py``

**Mock implementation:**

.. code-block:: python

   # src/catsim/estimation/bayesian.py  (appended to the shared module; see card 01)

   from collections.abc import Callable

   import numpy
   import numpy.typing as npt

   from .. import irt
   from ..item_bank import ItemBank
   from .base import BaseEstimator

   FloatArray = npt.NDArray[numpy.floating]


   class OwenEstimator(BaseEstimator):
     """Sequential Bayesian estimator (Owen, 1975) with Gaussian moment matching.

     Maintains ``(mu, sigma)`` as running state across calls within one CAT
     session. On each call, evaluates the newest response on a local grid
     centered at the current ``mu``, normalizes, and extracts fresh moments.

     Parameters
     ----------
     prior_mean : float
         Initial prior mean. Default 0.
     prior_sd : float
         Initial prior standard deviation. Default 1.
     n_nodes : int
         Size of the local quadrature grid. Default 21.
     half_width : float
         Half-width of the local grid in standard deviations. Default 5.0.
     reset_each_call : bool
         If True, replay the entire response history from scratch on every call
         (useful when a selector needs a one-shot estimate without maintaining
         session state in the estimator). If False, the estimator caches
         ``(mu, sigma)`` keyed by the length of ``administered_items`` and only
         performs an incremental update when called with one more response.
     """

     def __init__(
       self,
       prior_mean: float = 0.0,
       prior_sd: float = 1.0,
       n_nodes: int = 21,
       half_width: float = 5.0,
       reset_each_call: bool = False,
       verbose: bool = False,
     ) -> None:
       if prior_sd <= 0:
         raise ValueError(f"prior_sd must be positive, got {prior_sd}")
       if n_nodes < 5:
         raise ValueError(f"n_nodes must be >= 5, got {n_nodes}")
       if half_width <= 0:
         raise ValueError(f"half_width must be positive, got {half_width}")
       super().__init__(verbose=verbose)
       self._prior_mean = float(prior_mean)
       self._prior_sd = float(prior_sd)
       self._n_nodes = int(n_nodes)
       self._half_width = float(half_width)
       self._reset_each_call = bool(reset_each_call)
       self._state: tuple[float, float] | None = None
       self._state_length = 0

     def __str__(self) -> str:
       return "Owen Sequential Bayesian Estimator"

     @property
     def last_mu(self) -> float | None:
       return None if self._state is None else self._state[0]

     @property
     def last_sigma(self) -> float | None:
       return None if self._state is None else self._state[1]

     # ---- core kernel ----
     @staticmethod
     def _moment_update(
       mu: float,
       sigma: float,
       item_params: FloatArray,  # shape (4,) -- a, b, c, d
       response: bool,
       n_nodes: int,
       half_width: float,
     ) -> tuple[float, float]:
       """One Owen step: N(mu, sigma^2) -> posterior moments after one response.

       Uses a local uniform grid over ``[mu - half_width*sigma, mu + half_width*sigma]``.
       """
       lo = mu - half_width * sigma
       hi = mu + half_width * sigma
       nodes = numpy.linspace(lo, hi, n_nodes)

       a, b, c, d = item_params
       p = numpy.array([irt.icc(float(t), a, b, c, d) for t in nodes])

       # log prior at each node
       log_prior = -0.5 * ((nodes - mu) / sigma) ** 2
       # log likelihood of the single new response
       eps = 1e-12
       if response:
         log_lik = numpy.log(numpy.clip(p, eps, 1.0))
       else:
         log_lik = numpy.log(numpy.clip(1.0 - p, eps, 1.0))

       log_post = log_prior + log_lik
       log_post -= log_post.max()
       post = numpy.exp(log_post)
       post /= post.sum()

       new_mu = float(numpy.sum(nodes * post))
       new_var = float(numpy.sum((nodes - new_mu) ** 2 * post))
       new_sigma = float(numpy.sqrt(max(new_var, 1e-10)))
       return new_mu, new_sigma

     def estimate(
       self,
       item_bank: ItemBank,
       administered_items: list[int],
       response_vector: list[bool],
       est_theta: float,  # noqa: ARG002 -- Owen carries its own state
     ) -> float:
       n = len(administered_items)
       if n == 0:
         self._state = (self._prior_mean, self._prior_sd)
         self._state_length = 0
         return self._prior_mean

       # Fresh-replay path -- rebuild the state from scratch.
       if self._reset_each_call or self._state is None or n < self._state_length:
         mu, sigma = self._prior_mean, self._prior_sd
         for k in range(n):
           item_params = item_bank.items[administered_items[k], :4]
           mu, sigma = self._moment_update(
             mu, sigma, item_params, response_vector[k], self._n_nodes, self._half_width
           )
         self._state = (mu, sigma)
         self._state_length = n
         return mu

       # Incremental path -- update only on items not yet absorbed.
       mu, sigma = self._state
       for k in range(self._state_length, n):
         item_params = item_bank.items[administered_items[k], :4]
         mu, sigma = self._moment_update(
           mu, sigma, item_params, response_vector[k], self._n_nodes, self._half_width
         )
       self._state = (mu, sigma)
       self._state_length = n
       return mu

     def reset(self) -> None:
       """Drop session state so the next call re-seeds from the prior."""
       self._state = None
       self._state_length = 0

**Implementation notes:**

- This estimator belongs next to the Bayesian helpers and EAP because it shares the same conceptual family.
- The critical engineering concern is estimator state: reused estimator instances must be reset across sessions.
- A later 2PL-only closed-form update path can optimize further if needed.

**Dependencies:**

- Planned card 01 (`Bayesian Estimation Infrastructure`) for shared module location and conventions.

**Testing notes:**

- Compare streamed updates with replayed updates.
- Compare Owen estimates against EAP on ordinary short tests.
- Verify ``reset()`` semantics across sessions.
