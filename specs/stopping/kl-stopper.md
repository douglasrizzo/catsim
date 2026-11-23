**Status:** Planned
**Module:** `catsim.stopping.stopping.KLInfoStopper` (planned)
**Reference:** Chang and Ying (1996); Choi, Grady, and Dodd (2011)
**Last validated:** 2026-11-23, against Chang & Ying (1996) and Choi, Grady & Dodd (2011, PubMed Central author manuscript).
**Implementation guidance:** not yet updated to the 1.0 strategy contract (M22).

# Kullback-Leibler Information Stopper

## Motivation

The KL stopper is the global-information analogue of the minimum-information
bank stopper. Instead of looking only at Fisher information at a single point, it
uses integrated KL information, which remains meaningful even when the current
ability estimate is still uncertain.

It pairs naturally with KL-based item selection because selection and stopping
then operate on the same information scale.

No source proposes this rule. It applies the minimum-information rule (Choi et
al., 2011, Introduction; see the minimum-information stopper spec) to the KL
index that Chang and Ying (1996) defined for item selection.

## Definition

For each remaining item $i$, define the integrated KL information

$$
K_i(\hat{\theta})
=
\int_{\hat{\theta} - \delta_n}^{\hat{\theta} + \delta_n}
\left[
  P_i(\hat{\theta}) \log \frac{P_i(\hat{\theta})}{P_i(\theta)}
  +
  (1-P_i(\hat{\theta}))
  \log \frac{1-P_i(\hat{\theta})}{1-P_i(\theta)}
\right] d\theta,
$$

with half-width schedule

$$
\delta_n = \frac{c}{\sqrt{n}}.
$$

The test stops when

$$
\max_{i \in \mathcal{R}} K_i(\hat{\theta}) \le \kappa,
$$

where $\kappa$ is the configured KL threshold.

The integrand is Chang and Ying's KL item information
$K_i(\theta \,\|\, \hat{\theta})$ (1996, p. 217, Definition 2.1, Equation 12),
with their true ability $\theta_0$ replaced by the estimate, and the integral is
their KL index (p. 220, Equation 17). They derive $\delta_n = c/\sqrt{n}$ from
the confidence interval
$\hat{\theta}_n \pm c\,[I^{(n)}(\hat{\theta}_n)]^{-1/2}$, with $c$ chosen for a
coverage probability (p. 221, Equations 18 and 19), and used $c = 3$ (p. 223).
As in catsim's KL selector, $n$ is taken as 1 before the first item.

Because the window shrinks as items are administered, so does every item's
index, whatever the bank. Near $\hat{\theta}$ the divergence grows as
$\tfrac{1}{2} I_i(\hat{\theta})(\theta - \hat{\theta})^2$, since its curvature
there is the Fisher information (p. 219, Equation 16), so for a narrow window

$$
K_i(\hat{\theta})
\approx
\frac{I_i(\hat{\theta})\,\delta_n^3}{3}
=
\frac{c^3\, I_i(\hat{\theta})}{3\, n^{3/2}}.
$$

A fixed $\kappa$ is therefore reached partly through test length alone, which
is not the case for the Fisher threshold of the minimum-information rule.

## Parameters

```{eval-rst}
.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``min_kl``
     - required
     - Threshold :math:`\kappa` on the best remaining KL information.
   * - ``c``
     - ``3.0``
     - Positive schedule constant for the integration half-width; Chang and Ying (1996, p. 223)
       used :math:`c = 3`.
   * - ``min_items``
     - ``1``
     - Minimum number of items before stopping is allowed.
   * - ``max_items``
     - ``None``
     - Hard maximum test length.
```

## Behavioral Contracts

Any correct implementation must satisfy the following properties:

1. The rule stops when the largest KL global-information score among remaining items is less than or equal to `min_kl`.
2. The integration half-width follows `c / sqrt(max(1, n_administered))`.
3. The implementation reuses the same KL kernel as the KL selector or an equivalent mathematically identical kernel.
4. If no items remain, the stopper terminates.
5. The inherited minimum-length and maximum-length checks still apply.

## References

Chang, H.-H., & Ying, Z. (1996). A global information approach to computerized adaptive testing. *Applied Psychological Measurement*, 20(3), 213-229. <https://doi.org/10.1177/014662169602000303>. Open copy: [University of Minnesota Digital Conservancy](https://hdl.handle.net/11299/119092).

Choi, S. W., Grady, M. W., & Dodd, B. G. (2011). A new stopping rule for computerized adaptive testing. *Educational and Psychological Measurement*, 71(1), 37-53. <https://doi.org/10.1177/0013164410387338>. Open copy: [PubMed Central](https://pmc.ncbi.nlm.nih.gov/articles/PMC3028267/) (author manuscript without the journal's page numbers; cited here by section).

## Open Questions

To be answered before the stopper is implemented, and removed from the spec when it is promoted.

1. **Whether the threshold should follow the window.** As shown above, every item's index shrinks
   roughly as $c^3 I_i(\hat{\theta}) / (3 n^{3/2})$, so a fixed $\kappa$ (this spec) is reached
   partly because the test got longer, on any bank. The options are:
   - a fixed $\kappa$;
   - a threshold that shrinks with the window, $\kappa_0 \delta_n^3$, which compares roughly the
     same quantity as the minimum-information rule's Fisher threshold and so differs from that
     rule mainly for wide windows, early in the test;
   - not implementing this stopper, since no source proposes it.

## Implementation Guidance

:::{note}
This section is for contributors and agents implementing this technique.
It is removed when the spec is promoted to "Implemented" status.
:::

**Proposed location:** `src/catsim/stopping/stopping.py`

**Mock implementation:**

```python
# src/catsim/stopping/stopping.py  (alongside the other Choi 2011 stoppers)
"""Kullback-Leibler information stopper."""

import numpy
from scipy.integrate import quad

from .. import irt
from .._typing import FloatArray
from ..item_bank import ItemBank
from .stopping import TestLengthStopper


def _kl_integrand(theta: float, theta_hat: float, a: float, b: float, c: float, d: float) -> float:
  """KL divergence between response distributions at theta_hat and theta. Mirrors KLSelector."""
  eps = 1e-12
  p0 = float(numpy.clip(irt.icc(theta_hat, a, b, c, d), eps, 1 - eps))
  p = float(numpy.clip(irt.icc(theta, a, b, c, d), eps, 1 - eps))
  return p0 * numpy.log(p0 / p) + (1 - p0) * numpy.log((1 - p0) / (1 - p))


class KLInfoStopper(TestLengthStopper):
  """Stop when no remaining item has KL global information above ``min_kl``.

  Parameters
  ----------
  min_kl : float
      Threshold on integrated KL information for the best remaining item.
  c : float
      Schedule constant for the integration half-width ``delta_n = c / sqrt(n)``.
  min_items : int or None
      Minimum items before stopping. ``c`` requires at least 1 administered item.
  max_items : int or None
      Maximum items before forced stop.
  """

  def __init__(
    self,
    min_kl: float,
    c: float = 3.0,
    min_items: int | None = 1,
    max_items: int | None = None,
  ) -> None:
    if min_kl < 0:
      raise ValueError(f"min_kl must be non-negative, got {min_kl}")
    if c <= 0:
      raise ValueError(f"c must be positive, got {c}")
    super().__init__(min_items=min_items, max_items=max_items)
    self._min_kl = float(min_kl)
    self._c = float(c)

  def __str__(self) -> str:
    return f"KLInfoStopper(min_kl={self._min_kl}, c={self._c})"

  # ---- core kernel ----
  def _kl_global(self, theta_hat: float, item: FloatArray, half_width: float) -> float:
    a, b, c_param, d = item[0], item[1], item[2], item[3]
    val, _ = quad(
      _kl_integrand,
      theta_hat - half_width,
      theta_hat + half_width,
      args=(theta_hat, a, b, c_param, d),
    )
    return float(val)

  def stop(
    self,
    item_bank: ItemBank,
    administered_items: list[int],
    theta: float,
  ) -> bool:
    base_decision = super().stop(item_bank, administered_items, theta)
    if base_decision is True:
      return True

    n = max(1, len(administered_items))
    half_width = self._c / numpy.sqrt(n)

    remaining_mask = numpy.ones(item_bank.n_items, dtype=bool)
    remaining_mask[administered_items] = False
    if not remaining_mask.any():
      return True

    remaining_indices = numpy.flatnonzero(remaining_mask)
    kl_values = numpy.array([
      self._kl_global(theta, item_bank.items[i], half_width) for i in remaining_indices
    ])
    return float(kl_values.max()) <= self._min_kl

  def _check_stopping_criterion(  # noqa: PLR6301
    self,
    administered_items: FloatArray,  # noqa: ARG002
    theta: float,  # noqa: ARG002
  ) -> bool:
    return False
```

**Implementation notes:**

- Reuse the same KL integrand as the KL selector once both features exist; do not maintain divergent formulas.
- Runtime cost mirrors the KL selector: one quadrature per remaining item per step.
- If profiling shows this is hot, a grid-plus-trapezoid approximation is the first simplification to try.

**Dependencies:**

- Reuses the same KL kernel as planned card 03 (`KLSelector`), though it can be implemented independently first if needed.

**Testing notes:**

- Far-off item banks should trigger the stopper early.
- Well-targeted item banks should keep the KL score above threshold longer.
- Validate half-width scheduling and parameter validation.
