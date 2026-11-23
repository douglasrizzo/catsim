**Status:** Planned
**Module:** `catsim.exposure.sympson_hetter.SympsonHetterControl` (planned)
**Reference:** Sympson and Hetter (1985); Stocking and Lewis (1998)
**Last validated:** 2026-11-23, against Barrada et al. (2008); Georgiadou et al. (2007); Stocking & Lewis (1995, ETS RR-95-24).
**Implementation guidance:** not yet updated to the 1.0 strategy contract (M22).

# Sympson-Hetter Exposure Control

## Motivation

Sympson-Hetter (SH) is the classic operational exposure-control method for CAT,
the most widely used way of capping item exposure rates (Barrada et al., 2008,
p. 495). It is conceptually orthogonal to item selection: a selector proposes
an item, then SH decides whether that proposed item is actually administered.

That separation lets the same mechanism wrap any selector. The control table,
however, belongs to one selector, test design and examinee population: it is
calibrated offline, before live testing, by simulating that design with
examinees drawn from a typical ability distribution, and is recalibrated when
test conditions change (Stocking & Lewis, 1995, p. 8; Georgiadou et al., 2007,
p. 11). It caps exposure in that population only: an item given to no more
than 20% of all examinees can still be given to every high-ability examinee,
which led Stocking and Lewis to condition the control on ability (1995,
pp. 4–5).

## Definition

For each item $i$, SH stores an acceptance probability $K_i$, the probability
that the item is administered given that the selector proposed it (Stocking &
Lewis, 1995, p. 8). Live administration proceeds as follows (Georgiadou et al.,
2007, p. 12; Barrada et al., 2008, p. 495):

1. Select a candidate item with the wrapped selector.
2. Draw a Bernoulli random variable with success probability $K_i$.
3. If the draw succeeds, administer the item.
4. If the draw fails, do not administer the item, exclude it for the rest of that examinee's test, and ask the selector for another candidate.

Because a rejected item is lost to that examinee, an $n$-item test is
guaranteed only if at least $n$ items have $K_i = 1$ (Stocking & Lewis, 1995,
pp. 21–22).

The $K_i$ values are calibrated offline so that no item's exposure rate exceeds
a target $r_{\max}$. An item is administered only if it is proposed and then
accepted, so its administration rate is $r_i = K_i s_i$, where $s_i$ is the
rate at which the selector proposes it (Barrada et al., 2008, p. 495,
Equation 4). Starting from $K_i^{(0)} = 1$, as Sympson and Hetter recommended
(Stocking & Lewis, 1995, pp. 21–22), each calibration iteration simulates the
test with the current values and sets (Barrada et al., 2008, p. 495,
Equation 5; Stocking & Lewis, 1995, p. 8)

$$
K_i^{(k+1)}
=
\begin{cases}
1 & \text{if } s_i^{(k)} \le r_{\max},\\
r_{\max} / s_i^{(k)} & \text{if } s_i^{(k)} > r_{\max},
\end{cases}
$$

where $s_i^{(k)}$ is the proportion of simulated examinees for whom the
selector proposed item $i$ in iteration $k$. Since $r_i = K_i s_i$, this is the
update

$$
K_i^{(k+1)}
=
\min\left(1, K_i^{(k)} \frac{r_{\max}}{r_i^{(k)}}\right)
$$

written with the administration rate $r_i^{(k)}$, the form the implementation
guidance below uses. With simulated counts the two forms differ by sampling
noise, and only the first is defined for an item that was proposed but never
administered. The iterations stop when the $K_i$ have stabilized and the
largest administration rate is about $r_{\max}$, or after a fixed number of
iterations (Stocking & Lewis, 1995, p. 8; Barrada et al., 2008, p. 495).

## Parameters

```{eval-rst}
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
```

## Behavioral Contracts

Any correct implementation must satisfy the following properties:

1. Live administration separates candidate selection from candidate acceptance.
2. If `k_values` are missing, live selection is invalid until calibration has been performed.
3. A rejected item is not administered to the current examinee for the rest of the test, and the selector is queried again for another candidate.
4. Offline calibration updates item-specific acceptance probabilities toward the target marginal exposure rate.
5. The wrapper should preserve the selector contract so it can be slotted into the engine as a selector-like component.

## References

Sympson, J. B., & Hetter, R. D. (1985). Controlling item-exposure rates in computerized adaptive testing. In *Proceedings of the 27th Annual Meeting of the Military Testing Association* (pp. 973-977). San Diego, CA: Navy Personnel Research and Development Center. Not openly available; Barrada et al. (2008), Georgiadou et al. (2007) and Stocking and Lewis (1995) describe the method.

Stocking, M. L., & Lewis, C. (1998). Controlling item exposure conditional on ability in computerized adaptive testing. *Journal of Educational and Behavioral Statistics*, 23(1), 57-75. <https://doi.org/10.3102/10769986023001057>. Not openly available; the 1995 report below preceded it.

Stocking, M. L., & Lewis, C. (1995). *Controlling item exposure conditional on ability in computerized adaptive testing* (Research Report 95-24). Princeton, NJ: Educational Testing Service. <https://doi.org/10.1002/j.2333-8504.1995.tb01659.x>. Open copy: [ERIC](https://files.eric.ed.gov/fulltext/ED389740.pdf).

Barrada, J. R., Olea, J., Ponsoda, V., & Abad, F. J. (2008). Incorporating randomness in the Fisher information for improving item-exposure control in CATs. *British Journal of Mathematical and Statistical Psychology*, 61(2), 493-513. <https://doi.org/10.1348/000711007X230937>. Open copy: [first author's page](https://personal.unizar.es/barrada/papers/08_BJMSP_Barrada.pdf).

Georgiadou, E. G., Triantafillou, E., & Economides, A. A. (2007). A review of item exposure control strategies for computerized adaptive testing developed from 1983 to 2005. *The Journal of Technology, Learning, and Assessment*, 5(8). <https://ejournals.bc.edu/index.php/jtla/article/view/1647> (open access).

## Implementation Guidance

:::{note}
This section is for contributors and agents implementing this technique.
It is removed when the spec is promoted to "Implemented" status.
:::

**Proposed location:** `src/catsim/exposure/sympson_hetter.py`

**Mock implementation:**

```python
# src/catsim/exposure/sympson_hetter.py
"""Sympson-Hetter exposure control (1985)."""

import numpy

from .._typing import FloatArray
from ..exceptions import NoItemsAvailableError
from ..item_bank import ItemBank
from ..selection.base import BaseSelector


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
    response_vector: list[bool] | None = None,
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
        response_vector=response_vector,
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
```

**Implementation notes:**

- A dedicated `exposure/` subpackage is justified because exposure control wraps selectors but is conceptually distinct from selection.
- The key design seam is the calibration callback; provide a helper that can drive `CatEngine` sessions without forcing users to write boilerplate.
- Conditional SH and content-balanced SH belong in follow-up work, not this first implementation.

**Dependencies:** none

**Testing notes:**

- Verify that uncalibrated live selection fails clearly.
- Compare exposure rates before and after calibration on a bank with a few dominant items.
- Confirm that `K = 1` reproduces the wrapped selector's behavior.
