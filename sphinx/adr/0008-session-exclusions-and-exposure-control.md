# ADR-0008: The Session Keeps Its Excluded Items, and an Exposure Control May Decline a Proposal

## Status

Accepted 2026-11-24

## Context

Sympson-Hetter exposure control administers the item a selector proposes only with a calibrated
probability. A declined item is set aside for the rest of the examinee's test, and the calibration
counts how often each item is proposed. Under the strategy contract of ADR-0005 the engine built
every `StepContext` with no excluded items, `select` returned only an `int`, and strategies keep no
per-session state. A wrapper selector could exclude a declined item only for one `select` call, and
nothing recorded declines or proposals from one step to the next.

## Decision

- A session owns its exclusions. `CatSessionState.excluded_items` holds the items the session must
  not be given although it has not had them, in the order they were excluded, with no repeat and no
  administered item. It is saved with the session, and snapshots copy it. The engine passes it
  into every `StepContext`; a wrapper that passes a changed context on adds to it, never replaces it.
- An exposure control is an optional engine component, a strategy under ADR-0005:

  ```python
  class BaseExposureControl(ABC):  # catsim.exposure
    def accept(self, context: StepContext, item_id: int) -> bool: ...

  CatEngine(item_bank, initializer, selector, estimator, stopper, *, exposure_control=None)
  ```

- `select_next` asks the control about every proposal. A declined item joins the context's excluded
  items and the selector is asked again, until the control accepts an item or the selector raises
  `NoItemsAvailableError`, which ends the session with `StopReason.NO_ITEMS` (ADR-0007). Each
  proposal is checked first, so a selector that ignores the exclusions raises and the loop ends.
  The declines are committed once the selection ends; a selection that raises commits none.
- A response to an excluded item, or a proposal of one, raises `ExcludedItemError`, a `ValueError`.
- `ExposureTracker` counts proposals when an item is administered or declined: `record` adds an
  administration and a proposal, `record_decline` a proposal only, so declines never reach the
  rates an `r_max` cap is checked against (ADR-0004). `SimulationResult` reports
  `proposal_counts` and `proposal_rates`.

## Consequences

- Sympson-Hetter is a control of a few lines. An item is proposed at most once per examinee, so
  `proposal_rates` is the share its calibration needs.
- Without a control nothing draws, so the mechanism leaves seeded runs as they are; a control
  draws from the run's generator (ADR-0010).
- `select_next` can change the session, so an application saves sessions after `select_next` too.
  Each call selects anew: a second call before the answer, as on a page reload, may decline the
  item already shown, so an application keeps the item's id and shows it again.
- Items may be added to `excluded_items` before the first item, never removed during the test:
  each decline is already counted as a proposal.
- A session whose remaining items are all declined ends early, with `NO_ITEMS`.
- A tracker rebuilt without proposals takes one per administration.

## Alternatives Considered

- `select` returning the item with its declines: it breaks every selector and caller.
- Exclusions that last one selection: they depart from Sympson and Hetter, and an item declined and
  proposed again would be counted twice.
- A decline callback in `StepContext`: a side effect in a frozen context, which look-ahead selectors
  such as MEI and direct calls would trigger wrongly or lose.
- A wrapper keeping declines by `session_id`: per-session state on a strategy, lost on restore.
- The control in `RunContext`: it is part of the test design, like the selector.
- Counting a proposal each time `select_next` returns an item: a reload would count it twice.
