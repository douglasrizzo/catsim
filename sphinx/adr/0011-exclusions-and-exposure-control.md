# ADR-0011: The Session Keeps Its Excluded Items, and an Exposure Control May Decline a Proposal

## Status

Proposed 2026-11-25

## Context

Sympson-Hetter exposure control administers the item a selector proposes only with a calibrated
probability. A declined item is set aside for the rest of the examinee's test, and the calibration
counts how often each item is proposed. A wrapper selector can exclude a declined item only for one
call, since strategies keep nothing about a session (ADR-0007). Recorded examinees with missing
responses (ADR-0014) and applications that withhold items need the same thing: items a session
must not be given although it has not had them.

## Decision

- A session owns its exclusions. `Session.excluded_items` holds the items it must not be given,
  in the order they were excluded, with no repeat and no administered item. `start` takes the
  initial ones, and every step context carries them.
- An exposure control is an optional strategy of the engine:

  ```python
  CatEngine(item_bank, initializer, selector, estimator, stopper, *, exposure_control=None)
  ```

- `select` asks the control about every proposal. A declined item joins the excluded items, and the
  selector is asked again, until the control accepts an item or the selector raises
  `NoItemsAvailableError`, which ends the session with `NO_ITEMS` (ADR-0010). `select` returns the
  accepted item, the session with its new exclusions and the items declined:

  ```python
  Selection(item: int | None, session: Session, declined: tuple[int, ...])
  ```

- A proposal of an excluded item raises `ExcludedItemError`, a `ValueError`, so a selector that
  ignores the exclusions fails at once.
- A proposal is an administration or a decline. The simulation runner counts both per item, and
  `SimulationResult` reports `proposal_counts` and `proposal_rates` beside the exposure.

## Consequences

- Sympson-Hetter is a control of a few lines. An item is proposed at most once per examinee, so
  `proposal_rates` is the share its calibration needs.
- Without a control nothing draws, so the mechanism changes no seeded run that does not use it.
- Exclusions are only added during a test, never removed: each decline is already counted as a
  proposal.
- A session whose remaining items are all excluded ends early, with `NO_ITEMS`.
- An application that keeps its own proposal counts adds `declined` and the administered item.

## Alternatives Considered

- `select` returning the item with its declines from the selector: it breaks every selector.
- Exclusions that last one selection: they depart from Sympson and Hetter, and an item declined and
  proposed again would be counted twice.
- A decline callback in `StepContext`: a side effect in a frozen context, which look-ahead
  selectors such as MEI and direct calls would trigger wrongly.
- Excluding a declined item by adding it to the administered items: it moves the test position,
  which several selectors read.
