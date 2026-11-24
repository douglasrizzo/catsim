# ADR-0016: The Session Keeps Its Excluded Items, and an Exposure Control May Decline a Proposal

## Status

Accepted 2026-11-24

## Context

Sympson-Hetter exposure control administers the item a selector proposes only with a calibrated
probability. A declined item is set aside for the rest of the examinee's test, and the calibration
counts how often each item is proposed. Under the strategy contract of ADR-0004 the engine built
every `StepContext` with no excluded items, `select` returned only an `int`, and strategies keep no
per-session state. A wrapper selector could exclude a declined item only for one `select` call, and
nothing recorded declines or proposals from one step to the next.

## Decision

- A session owns its exclusions. `CatSessionState.excluded_items` holds the items the session must
  not be given although it has not had them, in the order they were excluded, with no repeat and no
  administered item. It is persisted with the session like its responses, and snapshots copy it.
  The engine passes it into every `StepContext`; a wrapper that passes a changed context on adds to
  it, never replaces it.
- An exposure control is an optional engine component, a strategy under ADR-0004:

  ```python
  class BaseExposureControl(ABC):  # catsim.exposure
    def accept(self, context: StepContext, item_id: int) -> bool: ...

  CatEngine(item_bank, initializer, selector, estimator, stopper, *, exposure_control=None)
  ```

- `select_next` asks the control about every proposal. A declined item joins the context's excluded
  items and the selector is asked again, until the control accepts an item or the selector raises
  `NoItemsAvailableError`, which ends the session with `StopReason.NO_ITEMS` (ADR-0011). Each
  proposal is checked first, so a selector that ignores the exclusions raises instead of looping.
  The declines are committed once the selection ends; a selection that raises commits none.
- A response to an excluded item, or a proposal of one, raises `ExcludedItemError`, a `ValueError`.
  To give such an item, the application removes it from `excluded_items` first.
- `ExposureTracker` counts proposals when an item is administered or declined: `record` adds an
  administration and a proposal, `record_decline` a proposal only, so declines never reach the
  rates an `r_max` cap is checked against (ADR-0003). `SimulationResult` reports
  `proposal_counts` and `proposal_rates`.

## Consequences

- Sympson-Hetter is a control of a few lines. An item is proposed at most once per examinee, so
  `proposal_rates` is the proposal share its calibration works from.
- Without a control the selector is called once and nothing draws, so seeded runs give the same
  tests with or without the mechanism. A control draws from the run's generator (ADR-0013).
- `select_next` can change the session, so an application saves sessions after `select_next` too.
  A second `select_next` before the answer, as on a page reload, asks again but counts nothing
  twice.
- A test reaches its planned length only if the control accepts enough items; a session whose
  remaining items are all declined ends with `NO_ITEMS`.
- A tracker saved without proposals is rebuilt with one proposal per administration, as in a run
  without a control.

## Alternatives Considered

- `select` returning the item with its declines: it breaks the `int` return for every selector and
  caller.
- Exclusions that last one selection: they depart from Sympson and Hetter, and an item declined and
  proposed again would be counted twice.
- A decline callback in `StepContext`: a side effect in a frozen context, which look-ahead selectors
  such as MEI and direct calls would trigger wrongly or lose.
- A wrapper selector keeping declines by `session_id`: per-session state on a strategy, lost when a
  session is restored.
- The control in `RunContext`: it is part of the test design, like the selector.
- Counting a proposal each time `select_next` returns an item: a reload would count it twice.
