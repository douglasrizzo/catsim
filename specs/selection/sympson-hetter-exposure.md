**Status:** Planned
**Module:** `catsim.exposure.sympson_hetter.SympsonHetterControl` (planned)
**Reference:** Sympson and Hetter (1985); Stocking and Lewis (1998)
**Last validated:** 2026-11-23, against Barrada et al. (2008); Georgiadou et al. (2007); Stocking & Lewis (1995, ETS RR-95-24).
**Implementation guidance:** updated to the 1.0 strategy contract (`StepContext` in, declined items in `excluded_items`) on 2026-11-24; it cannot yet exclude a declined item for the rest of the test (see the implementation notes).

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

written with the administration rate $r_i^{(k)}$. With simulated counts the two
forms differ by sampling noise, and only the first is defined for an item that
was proposed but never administered. The iterations stop when the $K_i$ have
stabilized and the largest administration rate is about $r_{\max}$, or after a
fixed number of iterations (Stocking & Lewis, 1995, p. 8; Barrada et al., 2008,
p. 495).

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

**Proposed location:** `src/catsim/exposure/sympson_hetter.py`, a new top-level package.

**Mock implementation:**

```python
# src/catsim/exposure/sympson_hetter.py
"""Sympson-Hetter exposure control."""

import dataclasses
from collections.abc import Callable

import numpy

from ..context import StepContext
from ..selection.base import BaseSelector
from ..typing import FloatArray


class SympsonHetterControl(BaseSelector):
  """Administer a wrapped selector's proposal with its item's acceptance probability.

  A declined item goes into the ``excluded_items`` of the context passed back to the wrapped
  selector, which proposes again, until a proposal is accepted.

  Parameters
  ----------
  selector : BaseSelector
      The selector whose proposals are accepted or declined. It must not propose an item in the
      context's ``excluded_items``; the built-in selectors draw their candidates with ``remaining``.
  k_values : numpy.ndarray or None, optional
      Calibrated acceptance probability of every item in the bank, from :func:`calibrate`. Default
      is None, which leaves the control uncalibrated: it raises on its first selection.

  Raises
  ------
  ValueError
      If a value of ``k_values`` is not in [0, 1].
  """

  def __init__(self, selector: BaseSelector, k_values: FloatArray | None = None) -> None:
    if k_values is not None:
      k_values = numpy.array(k_values, dtype=float)
      if not ((k_values >= 0) & (k_values <= 1)).all():
        msg = "k_values must be probabilities in [0, 1]"
        raise ValueError(msg)
      k_values.flags.writeable = False
    self._selector = selector
    self._k_values = k_values

  def __repr__(self) -> str:
    """Return the control as a constructor call that names every argument."""
    return f"{type(self).__name__}(selector={self._selector!r}, k_values={self._k_values!r})"

  @property
  def k_values(self) -> FloatArray | None:
    """Return the acceptance probabilities, or None if the control is uncalibrated."""
    return self._k_values

  def select(self, context: StepContext) -> int:
    """Return the first proposal of the wrapped selector that its acceptance draw admits.

    Parameters
    ----------
    context : StepContext
        The session to select for.

    Returns
    -------
    int
        Index of the selected item.

    Raises
    ------
    RuntimeError
        If the control is uncalibrated, or its table does not have one value per item of the bank.
    NoItemsAvailableError
        From the wrapped selector, once every item is administered or excluded.
    """
    k_values = self._k_values
    if k_values is None or k_values.size != context.item_bank.n_items:
      msg = "SympsonHetterControl needs one calibrated acceptance probability per item; see calibrate()"
      raise RuntimeError(msg)
    # Each decline shrinks the wrapped selector's pool, so it raises NoItemsAvailableError before
    # this loop could run forever.
    while True:
      candidate = self._selector.select(context)
      # An item with K = 1 draws no random number, so K = 1 everywhere reproduces the wrapped
      # selector's sessions exactly, as pick_best does for untied scores (ADR-0008).
      if k_values[candidate] >= 1 or context.rng.random() < k_values[candidate]:
        return candidate
      context = dataclasses.replace(context, excluded_items=(*context.excluded_items, candidate))


def sympson_hetter_update(proposal_rates: FloatArray, r_max: float) -> FloatArray:
  """Return the acceptance probabilities of the next calibration iteration (Barrada et al., 2008, Equation 5).

  Returns
  -------
  numpy.ndarray
      1 for every item proposed to at most ``r_max`` of the examinees, ``r_max`` over its proposal
      rate for the others.
  """
  k_values = numpy.ones_like(proposal_rates, dtype=float)
  over = proposal_rates > r_max
  k_values[over] = r_max / proposal_rates[over]
  return k_values


def calibrate(
  proposal_rates: Callable[[FloatArray], FloatArray],
  n_items: int,
  r_max: float,
  *,
  max_iterations: int = 30,
  tolerance: float = 0.005,
) -> FloatArray:
  """Return acceptance probabilities calibrated so that no item's exposure rate exceeds ``r_max``.

  Starts from K = 1 (Stocking & Lewis, 1995, pp. 21-22) and applies :func:`sympson_hetter_update`
  until no value moves by more than ``tolerance`` and the largest exposure rate is within
  ``tolerance`` of ``r_max``, or for ``max_iterations`` iterations.

  Parameters
  ----------
  proposal_rates : Callable[[numpy.ndarray], numpy.ndarray]
      Simulates the test design with the control at the given acceptance probabilities, and returns
      each item's proposal rate: the proportion of simulated examinees to whom the wrapped selector
      proposed it.
  n_items : int
      Number of items in the bank.
  r_max : float
      Target maximum exposure rate.
  max_iterations : int, optional
      Largest number of simulations. Default is 30.
  tolerance : float, optional
      Convergence tolerance on the acceptance probabilities and on the exposure rates. Default is
      0.005.

  Returns
  -------
  numpy.ndarray
      The acceptance probability of every item.
  """
  k_values = numpy.ones(n_items, dtype=float)
  for _ in range(max_iterations):
    rates = proposal_rates(k_values)
    updated = sympson_hetter_update(rates, r_max)
    # Barrada et al. (2008, p. 495, Equation 4): an item's exposure rate is K times its proposal rate
    exposure = k_values * rates
    if numpy.max(numpy.abs(updated - k_values)) <= tolerance and numpy.max(exposure) <= r_max + tolerance:
      return updated
    k_values = updated
  return k_values
```

**Implementation notes:**

- **What the 1.0 engine does not provide.** The definition excludes a declined item for the rest
  of the examinee's test (step 4, Contract 3). The sketch excludes it only for the rest of the
  call: the engine builds every context with no excluded items, `select` returns only the item,
  and the control keeps no per-session state, so a declined item can be proposed again at a later
  step. Calibration has the same gap: a proposal rate counts the administered and the declined
  items of each simulated examinee, and the declines are recorded nowhere. Both need the session
  to keep each examinee's declined items and the engine to pass them in `excluded_items`; since
  `select` returns only the item, the declines would have to reach the engine some other way, for
  example as an acceptance step the engine runs after `select`. That is a change to the engine
  and to ADR-0004, which this spec does not decide; implement the control after it.
- The control honours an `excluded_items` it is given and adds to it with `dataclasses.replace`,
  so the test position, `len(context.administered_items)`, stays what the wrapped selector expects.
  Once the engine passes a session's declines, the control works unchanged.
- The calibration follows the first form of the update, from proposal rates, because the form
  with administration rates is undefined for an item that was proposed but never administered.
  The convergence check uses the expected exposure rate $K_i s_i$ of the same simulation.
- `catsim.exposure` is a new top-level package, so it needs a layer in the import-linter contract
  (`pyproject.toml`). The control imports `catsim.selection`, and a `proposal_rates` built on
  `SimulationRunner` imports the runtime layers, which the selection layer may not, so the package
  goes in a layer above `simulation` (beside `plot`, or its own).
- The acceptance draw uses the context's `rng`, the run's generator (ADR-0008), and an item with
  $K_i = 1$ draws nothing.
- Conditional SH (Stocking & Lewis, 1995) and content-balanced variants are later work.

**Testing notes:**

- Contract 2: an uncalibrated control raises `RuntimeError` on its first selection, and so does a
  table whose length differs from the bank's.
- With $K_i = 1$ for every item, a seeded run gives the same sessions as the wrapped selector.
- With $K_i = 0$ for the wrapped selector's best item and 1 elsewhere, the control returns the
  wrapped selector's second choice, and the context the wrapped selector saw last has the best
  item in `excluded_items` and the same administered items.
- `sympson_hetter_update` on hand-picked rates: with `r_max = 0.2`, proposal rates
  `[0.1, 0.2, 0.4, 0.8]` give `[1, 1, 0.5, 0.25]`.
- `calibrate` with a `proposal_rates` callable built for the test, returning fixed rates whatever
  the table, returns the update of those rates after two simulations.
