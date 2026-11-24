# ADR-0011: Decide in One Place How a Session Ends

## Status

Accepted

## Context

`CatEngine` ended sessions in three places, with three free-form strings. `step()` asked the
stopper before selecting (`"stop_rule"`), turned `NoItemsAvailableError` into
`"item_bank_exhausted"` and a `None` from the selector into `"selector_stopped"`, and
`apply_response` asked the stopper again after each estimate. The manual workflow, `select_next`
then `apply_response`, shared none of that: `select_next` let the exception escape and left the
session in progress, so every application had to copy `step()`'s handling. A stopped session
accepted more responses, stepping it again relabelled it, and the stopper was asked twice about
every state.

The labels were also wrong. `NoItemsAvailableError` does not mean that the bank is exhausted: a
stratified selector raises it when a stratum runs out, and `LinearSelector` when its list does. The
built-in stoppers checked bank exhaustion themselves, so a bank that really ran out reported
`"stop_rule"`. No selector returned `None`.

## Decision

- A session ends for one of two reasons, the members of the `StopReason` enum:
  - `STOPPER` (`"stopper"`): the engine's stopper is satisfied by the session's current state;
  - `NO_ITEMS` (`"no_items"`): the selector has no item to offer.

  `stop_reason` is `None` while the session runs, and it is the only record of whether the
  session has ended: `is_stopped` reads it, and there is no separate status field. Lifecycle
  states such as paused or abandoned belong to the application.
- A selector says that it has no item by raising `NoItemsAvailableError`. That is the single "no
  item" signal: `select` returns an `int` (ADR-0004).
- `select_next` is the only place that decides whether a session continues. It returns `None` and
  records the reason when the stopper is satisfied or the selector has no item, and it returns
  `None` without changing anything for a session that has already stopped. A manual session is
  therefore one loop with no other stop check:

  ```python
  while (item_id := engine.select_next(state, context)) is not None:
    engine.apply_response(state, item_id, response, context)
  ```

  `step()` is the same three moves: select, get the response, apply it.
- The stopper is evaluated once per session state: by `select_next` for the initial state, before
  any response, and by `apply_response` after each new estimate. What `apply_response` records is
  final; `select_next` does not ask again.
- Stoppers are pure: the same inputs give the same answer, and asking has no side effects. Asking
  once per state therefore loses nothing.
- Running out of items is the engine's to detect, through the selector, and it reports `NO_ITEMS`.
  A stopper that checks bank exhaustion itself and says stop reports `STOPPER`, because all the
  engine knows is that its stopper said stop.
- `apply_response` applies a response atomically, and only to a running session. It raises
  `ValueError` for a stopped session and `TypeError` for an item index that is not an integer or a
  response that is not a `bool` or `numpy.bool_`. It estimates and asks the stopper on step contexts
  built from the session with the new item and response (ADR-0004), and commits the item, the
  response, the estimate, the exposure count and the stop decision only once both have succeeded.

## Consequences

- A manual session and `step()` or `run_session()` end the same way, with the same reason.
- Analysis code compares a stop reason with one of two values; as a `StrEnum`, each member also
  equals its string.
- `NO_ITEMS` does not tell a bank that ran out from a selector whose own pool ran out; comparing the
  number of administered items with the bank size does.
- A session of n items makes n + 1 stopper calls instead of 2n, which matters for stoppers that are
  costly to evaluate.
- A state whose items were appended by hand, outside `apply_response`, is not re-evaluated by the
  stopper. An application that persists sessions restores their stop reason with them.
- Custom selectors that return `None`, and code comparing the old strings (`"stop_rule"`,
  `"item_bank_exhausted"`, `"selector_stopped"`), must change.
