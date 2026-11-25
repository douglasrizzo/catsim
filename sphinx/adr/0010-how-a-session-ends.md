# ADR-0010: A Session Ends for One of Two Reasons, Decided in One Place

## Status

Accepted 2026-11-25

## Context

The engine once ended sessions in three places with three free-form strings, and the manual
workflow shared none of that logic, so every application copied it and a stopped session accepted
more responses. The labels were also wrong: `"item_bank_exhausted"` was reported when a stratified
selector's stratum ran out, while a bank that really ran out reported `"stop_rule"`, because the
built-in stoppers checked exhaustion themselves.

## Decision

- A session ends for one of the two reasons of the `StopReason` enum:
  - `STOPPER` (`"stopper"`): the engine's stopper is satisfied by the session's current state;
  - `NO_ITEMS` (`"no_items"`): the selector has no item to offer, or none that the exposure control
    accepts (ADR-0011).

  `stop_reason` is `None` while the session runs, and it is the only record of whether the session
  has ended.
- A selector says that it has no item by raising `NoItemsAvailableError`, the single "no item"
  signal: `select` returns an `int` (ADR-0007).
- The stopper is asked once per state: by `start` for the initial state and by `answer` for the
  state after each response. A satisfied stopper sets `STOPPER` on the session they return.
- `select` returns no item for a stopped session, and returns the session unchanged. When the
  selector has no item, it returns no item and the session with `NO_ITEMS`.
- `answer` raises for a stopped session, and for an item that `select` could not have proposed:
  administered, excluded or out of range.

```python
session = engine.start(rng=rng)
while True:
  selection = engine.select(session, rng=rng)
  session = selection.session
  if selection.item is None:
    break
  session = engine.answer(session, selection.item, respond(selection.item))
```

## Consequences

- Every test, simulated or live, ends the same way, with the same reason, through one loop.
- Analysis code compares a stop reason with one of two values; as a `StrEnum`, each member also
  equals its string.
- `NO_ITEMS` covers a bank that ran out, a selector whose own pool ran out, and a recorded examinee
  whose answered items ran out (ADR-0014); comparing the number of administered items with the bank
  size tells them apart.
- A session of n items makes n + 1 stopper calls, which matters for stoppers that are costly to
  evaluate.

## Alternatives Considered

- A status field beside `stop_reason`: two records of one fact, which can disagree.
- Asking the stopper in `select` as well: every state is judged twice, and stoppers are pure, so
  the answers can only agree.
- A selector returning `None` as a second "no item" signal: every caller would check for both.
- A third reason for a recorded examinee who ran out of answers: the selector cannot tell its
  exclusions apart, and the session's excluded items already say why.
