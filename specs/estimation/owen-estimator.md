**Status:** Planned
**Module:** `catsim.estimation.bayesian.OwenEstimator` (planned)
**Reference:** Owen (1975)
**Last validated:** 2026-11-23, against Jensema (1977), McBride (1977) and Weiss & McBride (1984).
**Implementation guidance:** not yet updated to the 1.0 strategy contract (M22).

# Owen Sequential Bayesian Estimator

## Motivation

Full-grid EAP and MAP revisit the complete response history each time the
estimate is updated. Owen's sequential Bayesian approach is designed for a
different operating point: constant-time streaming updates that carry only a
posterior mean and variance forward from one step to the next. After each
response the posterior is replaced by a normal distribution with that mean and
variance, which serves as the prior for the next item (McBride, 1977, p. 122;
Jensema, 1977, p. 113).

That makes it attractive when tests are long or latency matters more than exact
posterior fidelity. The estimates depend on the prior: in simulations of Owen's
adaptive testing strategy, which pairs this estimator with its own item
selection, they were biased and unequally precise across ability levels unless
an accurate prior was used for each examinee (Weiss & McBride, 1984, p. 273;
McBride, 1977, p. 121).

## Definition

Let the current approximate posterior be represented as a Gaussian

$$
\theta \mid \mathbf{u}_{1:n}
\approx
\mathcal{N}(\mu_n, \sigma_n^2),
$$

starting from the prior, $\mathcal{N}(\mu_0, \sigma_0^2)$; Jensema (1977,
p. 113) starts from $\mathcal{N}(0, 1)$, the assumed ability distribution of
the population.

After observing the next item response $u_{n+1}$ on item $i_{n+1}$,
update by treating that Gaussian as the prior and applying one-step Bayesian
reweighting:

$$
p(\theta \mid u_{n+1})
\propto
\phi\left(\frac{\theta - \mu_n}{\sigma_n}\right)
P_{i_{n+1}}(\theta)^{u_{n+1}}
[1-P_{i_{n+1}}(\theta)]^{1-u_{n+1}}.
$$

Then match moments to obtain

$$
\mu_{n+1} = \mathbb{E}[\theta \mid u_{n+1}],
\qquad
\sigma_{n+1}^2 = \mathrm{Var}[\theta \mid u_{n+1}].
$$

The estimate after $n$ responses is $\mu_n$, and $\sigma_n^2$ is its variance
(Jensema, 1977, p. 113).

Owen's procedure assumes normal-ogive item characteristic curves with a guessing
parameter, $P_i(\theta) = c_i + (1 - c_i)\,\Phi(a_i(\theta - b_i))$
(Jensema, 1977, p. 111, Equations 1 and 2; McBride, 1977, p. 122), where
$\Phi$ and $\phi$ are the standard normal distribution and density functions.
For this model both moments have closed forms. Without guessing ($c_i = 0$), let
$s_n^2 = a_i^{-2} + \sigma_n^2$ and $z = (\mu_n - b_i)/s_n$, so that $\Phi(z)$
is the probability of a correct response under the current prior. With
$r = \phi(z)/\Phi(z)$ after a correct response and
$r = -\phi(z)/[1 - \Phi(z)]$ after an incorrect one,

$$
\mu_{n+1} = \mu_n + \frac{\sigma_n^2}{s_n}\,r,
\qquad
\sigma_{n+1}^2
=
\sigma_n^2\left[1 - \frac{\sigma_n^2}{s_n^2}\,r\,(r + z)\right].
$$

These are Jensema's Equations 7-9 and 11-12 (1977, p. 112), written with $\Phi$
in place of the error function: his $D = (b_i - \mu_n)/\sqrt{2 s_n^2}$ equals
$-z/\sqrt{2}$, so his $\tfrac{1}{2}(1 - \operatorname{erf} D)$ is $\Phi(z)$
(Equation 5, p. 111). With guessing, a correct response has the closed form of
his Equations 13-18, and an incorrect response keeps Equations 10-12, because the
likelihood $(1 - c_i)[1 - \Phi(a_i(\theta - b_i))]$ is proportional to the one
without guessing (p. 112).

catsim's planned estimator keeps Owen's update, one response at a time with the
two moments carried forward, but departs from his model: its item characteristic
curve is catsim's four-parameter logistic, not the normal ogive. The logistic
moments have no closed form, so catsim computes them numerically; the
implementation guidance uses a local grid over
$[\mu_n - h\sigma_n, \mu_n + h\sigma_n]$.

Because each step replaces the posterior by a normal distribution, the estimate
depends on the order in which the responses are absorbed, so it must follow the
administration order, and it differs from the EAP estimate of the same responses
under the same prior (Contract 5). Owen's procedure also selects the next item,
by the smallest expected posterior variance (Jensema, 1977, pp. 112-113,
Equations 19-22); this spec covers only the estimator, which can be paired with
any selector.

## Parameters

```{eval-rst}
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
```

## Behavioral Contracts

Any correct implementation must satisfy the following properties:

1. With no administered items, the estimate equals the configured prior mean.
2. In incremental mode, the estimator updates only on newly observed responses and carries forward the current mean and standard deviation.
3. In replay mode, the estimate reconstructed from scratch must match the streamed estimate for the same response history.
4. Calling `reset()` clears the sequential state so the next estimate re-seeds from the configured prior.
5. On ordinary short tests, the sequential estimate should remain close to full-grid EAP, though not necessarily identical.

## References

Jensema, C. J. (1977). Bayesian tailored testing and the influence of item bank characteristics. *Applied Psychological Measurement*, 1(1), 111-120. <https://doi.org/10.1177/014662167700100117>. Open copy: [University of Minnesota Digital Conservancy](https://hdl.handle.net/11299/98433).

McBride, J. R. (1977). Some properties of a Bayesian adaptive ability testing strategy. *Applied Psychological Measurement*, 1(1), 121-140. <https://doi.org/10.1177/014662167700100119>. Open copy: [University of Minnesota Digital Conservancy](https://hdl.handle.net/11299/98434).

Owen, R. J. (1975). A Bayesian sequential procedure for quantal response in the context of adaptive mental testing. *Journal of the American Statistical Association*, 70(350), 351-356. <https://doi.org/10.1080/01621459.1975.10479871>. Not openly available; Jensema (1977) describes the method.

Weiss, D. J., & McBride, J. R. (1984). Bias and information of Bayesian adaptive testing. *Applied Psychological Measurement*, 8(3), 273-285. <https://doi.org/10.1177/014662168400800303>. Open copy: [University of Minnesota Digital Conservancy](https://hdl.handle.net/11299/107561).

## Implementation Guidance

:::{note}
This section is for contributors and agents implementing this technique.
It is removed when the spec is promoted to "Implemented" status.
:::

**Proposed location:** `src/catsim/estimation/bayesian.py`

**Mock implementation:**

```python
# src/catsim/estimation/bayesian.py  (appended to the shared module; see card 01)

from collections.abc import Callable

import numpy

from .. import irt
from .._typing import FloatArray
from ..item_bank import ItemBank
from .base import BaseEstimator


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
  ) -> None:
    if prior_sd <= 0:
      raise ValueError(f"prior_sd must be positive, got {prior_sd}")
    if n_nodes < 5:
      raise ValueError(f"n_nodes must be >= 5, got {n_nodes}")
    if half_width <= 0:
      raise ValueError(f"half_width must be positive, got {half_width}")
    super().__init__()
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
```

**Implementation notes:**

- This estimator belongs next to the Bayesian helpers and EAP because it shares the same conceptual family.
- The critical engineering concern is estimator state: reused estimator instances must be reset across sessions.
- A later 2PL-only closed-form update path can optimize further if needed.

**Dependencies:**

- Planned card 01 (`Bayesian Estimation Infrastructure`) for shared module location and conventions.

**Testing notes:**

- Compare streamed updates with replayed updates.
- Compare Owen estimates against EAP on ordinary short tests.
- Verify `reset()` semantics across sessions.
