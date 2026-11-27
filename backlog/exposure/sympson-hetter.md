---
title: Sympson-Hetter Exposure Control
kind: technique
status: ready
priority: 2
effort: M
blocked_by: []
raised_by: [Chang & Ying (1999) walkthrough]
---

**Module:** `catsim.exposure.sympson_hetter.SympsonHetterControl` (planned)
**Reference:** Sympson and Hetter (1985); Stocking and Lewis (1998)
**Last validated:** 2026-11-23, against Barrada et al. (2008); Georgiadou et al. (2007); Stocking & Lewis (1995, ETS RR-95-24).
**Implementation guidance:** written against the engine's exposure control (`BaseExposureControl.accept(context, item)`, declined items kept in the session's `excluded_items`) and `simulate` on 2026-11-26.

# Sympson-Hetter Exposure Control

## Motivation

Sympson-Hetter (SH) is the classic operational exposure-control method for CAT,
the most widely used way of capping item exposure rates (Barrada et al., 2008,
p. 495). It is conceptually orthogonal to item selection: a selector proposes
an item, then SH decides whether that proposed item is actually administered.

That separation lets the same mechanism work with any selector. The control table,
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

1. Select a candidate item with the selector.
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
   * - ``k_values``
     - ``None``
     - Optional pre-calibrated SH acceptance table.
```

## Behavior and Comparison

catR's `simulateRespondents` caps exposure with `rmax` in two ways (`Mrmax`): before each
examinee, `"restricted"` makes every item whose exposure rate is at or above the cap unavailable,
and `"IE"`, the item-eligibility method, makes each item available with an eligibility probability
that it updates from the exposure rates of the run so far. Neither needs a calibration before the
run, and neither acts on the item a selector proposes, as this control does, so no parity case
compares this control with catR (ADR-0020).

## Behavioral Contracts

Any correct implementation must satisfy the following properties:

1. Live administration separates candidate selection from candidate acceptance.
2. If `k_values` are missing, live selection is invalid until calibration has been performed.
3. A rejected item is not administered to the current examinee for the rest of the test, and the selector is queried again for another candidate.
4. Offline calibration updates item-specific acceptance probabilities toward the target marginal exposure rate.
5. It works with any selector, as the engine's `exposure_control`.

## References

Sympson, J. B., & Hetter, R. D. (1985). Controlling item-exposure rates in computerized adaptive testing. In *Proceedings of the 27th Annual Meeting of the Military Testing Association* (pp. 973-977). San Diego, CA: Navy Personnel Research and Development Center. Not openly available; Barrada et al. (2008), Georgiadou et al. (2007) and Stocking and Lewis (1995) describe the method.

Stocking, M. L., & Lewis, C. (1998). Controlling item exposure conditional on ability in computerized adaptive testing. *Journal of Educational and Behavioral Statistics*, 23(1), 57-75. <https://doi.org/10.3102/10769986023001057>. Not openly available; the 1995 report below preceded it.

Stocking, M. L., & Lewis, C. (1995). *Controlling item exposure conditional on ability in computerized adaptive testing* (Research Report 95-24). Princeton, NJ: Educational Testing Service. <https://doi.org/10.1002/j.2333-8504.1995.tb01659.x>. Open copy: [ERIC](https://files.eric.ed.gov/fulltext/ED389740.pdf).

Barrada, J. R., Olea, J., Ponsoda, V., & Abad, F. J. (2008). Incorporating randomness in the Fisher information for improving item-exposure control in CATs. *British Journal of Mathematical and Statistical Psychology*, 61(2), 493-513. <https://doi.org/10.1348/000711007X230937>. Open copy: [first author's page](https://personal.unizar.es/barrada/papers/08_BJMSP_Barrada.pdf).

Georgiadou, E. G., Triantafillou, E., & Economides, A. A. (2007). A review of item exposure control strategies for computerized adaptive testing developed from 1983 to 2005. *The Journal of Technology, Learning, and Assessment*, 5(8). <https://ejournals.bc.edu/index.php/jtla/article/view/1647> (open access).

## Implementation Guidance

:::{note}
This section is for contributors and agents implementing this technique.
It is removed when the spec is promoted to a technique page.
:::

**Proposed location:** `src/catsim/exposure/sympson_hetter.py`, in the `catsim.exposure` package
beside `BaseExposureControl`.

**Mock implementation:**

```python
# src/catsim/exposure/sympson_hetter.py
"""Sympson-Hetter exposure control."""

from collections.abc import Callable

import numpy
import numpy.typing as npt

from ..compat import override
from ..context import StepContext
from ..typing import FloatArray
from .base import BaseExposureControl


class SympsonHetterControl(BaseExposureControl):
  """Administer the selector's proposal with its item's acceptance probability.

  The engine excludes a declined item from the session for the rest of its test and asks the
  selector again, until the control accepts a proposal.

  Parameters
  ----------
  k_values : array-like of float or None, optional
      Calibrated acceptance probability of every item in the bank, from :func:`calibrate`. Default
      is None, which leaves the control uncalibrated: it raises on its first proposal.

  Raises
  ------
  ValueError
      If a value of ``k_values`` is not in [0, 1].
  """

  def __init__(self, k_values: npt.ArrayLike | None = None) -> None:
    table = None
    if k_values is not None:
      table = numpy.array(k_values, dtype=float)
      if not ((table >= 0) & (table <= 1)).all():
        msg = "k_values must be probabilities in [0, 1]"
        raise ValueError(msg)
      table.flags.writeable = False
    self._k_values: FloatArray | None = table

  @override
  def __repr__(self) -> str:
    """Return the control as a constructor call that names every argument."""
    return f"{type(self).__name__}(k_values={self._k_values!r})"

  @property
  def k_values(self) -> FloatArray | None:
    """Return the acceptance probabilities, or None if the control is uncalibrated."""
    return self._k_values

  @override
  def accept(self, context: StepContext, item: int) -> bool:
    """Return whether the proposed item passes its acceptance draw.

    Parameters
    ----------
    context : StepContext
        The context the selector proposed the item from.
    item : int
        Index of the proposed item.

    Returns
    -------
    bool
        True with the item's acceptance probability.

    Raises
    ------
    RuntimeError
        If the control is uncalibrated, or its table does not have one value per item of the bank.
    """
    k_values = self._k_values
    if k_values is None or k_values.size != context.item_bank.n_items:
      msg = "SympsonHetterControl needs one calibrated acceptance probability per item; see calibrate()"
      raise RuntimeError(msg)
    # An item with K = 1 draws no random number, so K = 1 everywhere reproduces the run without a
    # control exactly, as pick_best does for untied scores (ADR-0017).
    return bool(k_values[item] >= 1 or context.rng.random() < k_values[item])


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
      each item's proposal rate: the proportion of simulated examinees to whom the selector proposed
      it, the value of a run's ``SimulationResult.proposal_rates()``.
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

- The engine does the rest of step 4 and Contract 3 (ADR-0011): it adds a declined item to the
  session's `excluded_items`, which lasts for the rest of the test, and asks the selector again. A
  session whose remaining items are all declined ends with `StopReason.NO_ITEMS`. The control only
  decides, and keeps no per-session state.
- Each item is proposed at most once per examinee, as it is administered or declined, and
  `simulate` counts both, so the value of `SimulationResult.proposal_rates()` is the $s_i$ of
  Equation 5. A `proposal_rates` callable for `calibrate` is one seeded run with the control:

  ```python
  def proposal_rates(k_values):
    engine = CatEngine(bank, initializer, MaxInfoSelector(), estimator, stopper,
                       exposure_control=SympsonHetterControl(k_values))
    return simulate(engine, true_thetas=true_thetas, rng=seed).proposal_rates().value
  ```

- The calibration follows the first form of the update, from proposal rates, because the form
  with administration rates is undefined for an item that was proposed but never administered.
  The convergence check uses the expected exposure rate $K_i s_i$ of the same simulation.
- The module imports only the layers below `exposure` (`catsim.context`, `catsim.typing` and
  `catsim.compat`), as the import-linter layers contract requires; `calibrate` takes the
  simulation as a callable, since the strategy layers may not import `catsim.simulation`.
- The acceptance draw uses the context's `rng`, the generator of the `select` call, which
  `simulate` gives each examinee's session (ADR-0013), and an item with $K_i = 1$ draws nothing.
- One acceptance draw per proposal assumes that applications do not select again on a page reload:
  a second `CatEngine.select` on the same session before the answer draws again, so the item on
  screen would pass with a probability of about $K_i^2$. Keeping the item shown is the
  application's (ADR-0006).
- Conditional SH (Stocking & Lewis, 1995) and content-balanced variants are later work.
- A trial implementation of this sketch (1,000 examinees, 20-item tests, `MaxInfoSelector`,
  `r_max = 0.2`) never met the 0.005 tolerance on $K_i$: after 30 iterations the largest change of a
  value stayed between 0.05 and 0.12 while the largest expected exposure rate settled at 0.21-0.22,
  so `calibrate` returned the last table. Compare a tolerance on the exposure rate alone, a larger
  calibration population and the average of the last tables before choosing the defaults.
- On a new population of the same size, the trial's table gave a largest exposure rate of 0.240
  for `r_max = 0.2`, with about 25 declines per 20-item test and an RMSE of 0.412 (0.408 for
  `MaxInfoSelector(r_max=0.2)`). The table controls the rates of the population it was calibrated
  on; the technique page should say that another population can exceed $r_{\max}$.
- A table belongs to the selector and test design it was calibrated with: the trial's
  `MaxInfoSelector` table, used with `MEISelector`, let one item reach a rate of 0.45. A selector
  that proposes a fixed sequence runs out of items as declines accumulate: with `LinearSelector`
  and a 60-item list, 298 of 300 sessions ended with `StopReason.NO_ITEMS`.
- The control must pickle, so that `simulate(..., workers=n)` can send it to worker processes
  (ADR-0015): it keeps its table in a plain attribute and holds no closures. An application stores
  a calibrated table with its engine configuration.

**Testing notes:**

- Contract 2: an uncalibrated control raises `RuntimeError` on its first proposal, and so does a
  table whose length differs from the bank's; the session is left unchanged.
- With $K_i = 1$ for every item, a seeded run gives the same sessions as the run without a control.
- With $K_i = 0$ for the selector's best item and 1 elsewhere, the engine administers the selector's
  second choice, and the session's `excluded_items` holds the best item for the rest of the test.
- `sympson_hetter_update` on hand-picked rates: with `r_max = 0.2`, proposal rates
  `[0.1, 0.2, 0.4, 0.8]` give `[1, 1, 0.5, 0.25]`.
- `calibrate` with a `proposal_rates` callable built for the test, returning fixed rates whatever
  the table, returns the update of those rates after two simulations.
