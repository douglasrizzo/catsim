**Status:** Planned
**Module:** `catsim.estimation.bayesian.OwenEstimator` (planned)
**Reference:** Owen (1975)
**Last validated:** 2026-11-23, against Jensema (1977), McBride (1977) and Weiss & McBride (1984).
**Implementation guidance:** updated to the 1.0 strategy contract (`StepContext` in, `Estimate` out) on 2026-11-24.

# Owen Sequential Bayesian Estimator

## Motivation

Full-grid EAP and MAP evaluate the likelihood of the complete response history
each time the estimate is updated. Owen's sequential Bayesian approach carries
only a posterior mean and variance from one response to the next: after each
response the posterior is replaced by a normal distribution with that mean and
variance, which serves as the prior for the next item (McBride, 1977, p. 122;
Jensema, 1977, p. 113). Each update uses only the newest response and the two
moments before it.

catsim's strategies keep no per-session state, so its planned estimator does
not carry the two moments between calls: every call replays the session's
responses from the prior, one update per response, in administration order. Its
cost therefore grows with the test length, as EAP's does, and speed is not the
reason to choose it. It reproduces Owen's procedure, and it shows how the
normal approximation and its dependence on the response order move the estimate
away from the full posterior (Contract 5).

The estimates depend on the prior: in simulations of Owen's adaptive testing
strategy, which pairs this estimator with its own item selection, they were
biased and unequally precise across ability levels unless an accurate prior
was used for each examinee (Weiss & McBride, 1984, p. 273; McBride, 1977,
p. 121).

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
```

## Behavioral Contracts

Any correct implementation must satisfy the following properties:

1. With no administered items, the estimate equals the configured prior mean.
2. The estimate depends only on the administered items and responses, in administration order, and on the configuration: the context's `theta` and `se` are not read, and the same responses give the same estimate whatever the estimator computed before, for this session or another.
3. The estimate after $n$ responses is one update of $\mathcal{N}(\mu_{n-1}, \sigma_{n-1}^2)$ with the $n$-th response, where $\mu_{n-1}$ and $\sigma_{n-1}$ are the estimate and standard error after the first $n - 1$ responses.
4. The reported standard error is $\sigma_n$; with no administered items, it is the configured prior standard deviation.
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

**Proposed location:** `src/catsim/estimation/bayesian.py`, next to `EAPEstimator`.

**Mock implementation:**

```python
# src/catsim/estimation/bayesian.py (added to the existing module)

import math

import numpy

from .. import irt
from ..context import Estimate, StepContext
from ..irt import THETA_MAX_EXTENDED, THETA_MIN_EXTENDED
from ..typing import FloatArray
from .base import BaseEstimator

_MIN_LOCAL_NODES = 5
# Keeps the next update's local grid from collapsing onto a single point.
_MIN_VARIANCE = 1e-10


def _owen_update(
  mu: float, sigma: float, item: FloatArray, response: bool, n_nodes: int, half_width: float
) -> tuple[float, float]:
  """Return the mean and standard deviation of N(mu, sigma^2) reweighted by the likelihood of one response.

  Both moments are sums over equally spaced nodes on ``mu ± half_width * sigma``.

  Returns
  -------
  tuple[float, float]
      The mean and the standard deviation after the response.
  """
  nodes = numpy.linspace(mu - half_width * sigma, mu + half_width * sigma, n_nodes)
  # log_likelihood_hpc works from the logit, so log(1 - P) stays exact where P rounds to 1
  log_post = -0.5 * ((nodes - mu) / sigma) ** 2 + irt.log_likelihood_hpc(nodes, (response,), item[numpy.newaxis])
  post = numpy.exp(log_post - log_post.max())
  post /= post.sum()
  new_mu = float(post @ nodes)
  new_var = float(post @ (nodes - new_mu) ** 2)
  return new_mu, math.sqrt(max(new_var, _MIN_VARIANCE))


class OwenEstimator(BaseEstimator):
  """Owen's sequential Bayesian estimator: a normal posterior matched to the moments after each response.

  The estimator keeps no moments between calls. Each call replays the context's responses in
  administration order, starting from the prior, so one instance serves every session of a run and
  look-ahead selectors can call it with hypothetical responses. The reported estimate is clipped to
  ``(THETA_MIN_EXTENDED, THETA_MAX_EXTENDED)``, the package's ability scale (ADR-0009).

  Parameters
  ----------
  prior_mean : float, optional
      Mean of the normal prior. Default is 0.
  prior_sd : float, optional
      Standard deviation of the normal prior. Default is 1.
  n_nodes : int, optional
      Number of nodes of the local grid of each update. Default is 21.
  half_width : float, optional
      Half-width of the local grid, in standard deviations of the current approximation. Default is 5.

  Raises
  ------
  ValueError
      If ``prior_sd`` or ``half_width`` is not positive, or ``n_nodes`` is below 5.
  """

  def __init__(
    self, prior_mean: float = 0.0, prior_sd: float = 1.0, n_nodes: int = 21, half_width: float = 5.0
  ) -> None:
    if not prior_sd > 0:
      msg = f"prior_sd must be positive, got {prior_sd}"
      raise ValueError(msg)
    if n_nodes < _MIN_LOCAL_NODES:
      msg = f"n_nodes must be at least {_MIN_LOCAL_NODES}, got {n_nodes}"
      raise ValueError(msg)
    if not half_width > 0:
      msg = f"half_width must be positive, got {half_width}"
      raise ValueError(msg)
    self._prior_mean = float(prior_mean)
    self._prior_sd = float(prior_sd)
    self._n_nodes = int(n_nodes)
    self._half_width = float(half_width)

  def __repr__(self) -> str:
    """Return the estimator as a constructor call that names every argument."""
    return (
      f"{type(self).__name__}(prior_mean={self._prior_mean!r}, prior_sd={self._prior_sd!r}, "
      f"n_nodes={self._n_nodes!r}, half_width={self._half_width!r})"
    )

  def estimate(self, context: StepContext) -> Estimate:
    """Return the mean and standard deviation of the normal approximation after every response, in order.

    Parameters
    ----------
    context : StepContext
        The session to estimate for. Its ``theta`` and ``se`` are not used.

    Returns
    -------
    Estimate
        The last mean, clipped to the ability scale, the last standard deviation as the standard
        error, and the number of local grid nodes evaluated over all the updates.
    """
    mu, sigma = self._prior_mean, self._prior_sd
    items = context.item_bank.get_items(context.administered_items)
    for item, response in zip(items, context.responses, strict=True):
      mu, sigma = _owen_update(mu, sigma, item, response, self._n_nodes, self._half_width)
    theta = min(max(mu, THETA_MIN_EXTENDED), THETA_MAX_EXTENDED)
    return Estimate(theta=theta, se=sigma, evaluations=len(context.responses) * self._n_nodes)
```

**Implementation notes:**

- Replay is the price of keeping no per-session state on the estimator (ADR-0004): a session of
  $n$ items costs $n(n+1)/2$ updates in all, $21 \cdot 465 \approx 10^4$ node evaluations of a
  single item for a 30-item test, each far cheaper than an EAP node, which sums the log-likelihood
  of every administered item.
- Do not seed the update from the context's `theta` and `se` to save the replay. They are the
  previous estimate only when this estimator produced it for this session: before the first
  response they are the initializer's value with an infinite error, a hand-built context has the
  defaults, and another estimator may have produced them. Seeding from them would break Contract 2.
- The update follows the spec's departure from Owen's normal-ogive model: catsim's four-parameter
  logistic curve, with the moments summed on the local grid. `irt.log_likelihood_hpc` evaluates
  the log-likelihood from the logit, so no clipping of $P$ is needed.
- Every estimator returns an estimate inside its bounds (ADR-0009), and Owen's procedure has none:
  a correct response to a hard, discriminating item can move the mean by several standard
  deviations, and a run of them can carry it past 6. The sketch clips only the reported estimate
  to the package's scale, so the moments it replays follow the definition; truncating the local
  grid to the scale instead would change every update near the ends.
- `evaluations` counts one per local node and response, the closest analogue of EAP's count of
  one per grid node.

**Testing notes:**

- Contract 1 and 4: no responses give `Estimate(prior_mean, prior_sd, 0)`.
- Contract 3: for a three-item pattern, compute each update's two moments independently by
  integrating $\phi((\theta - \mu)/\sigma) P(\theta)^u [1 - P(\theta)]^{1-u}$ with
  `scipy.integrate.quad`, chain them by hand, and compare with the estimate and standard error
  after one, two and three responses.
- Contract 2: the same context with different `theta` and `se` gives the same estimate, and
  estimating two sessions alternately on one instance gives what estimating each alone gives.
- Order dependence: reversing the order of two responses changes the Owen estimate but not EAP's
  under the same normal prior.
- Contract 5: on short tests, compare with `EAPEstimator(log_prior=normal_log_prior(prior_mean, prior_sd))`.
