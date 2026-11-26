# ADR-0006: Each Step Is Computed From Explicit Inputs, and the Session Is a Value

## Status

Accepted 2026-11-25

## Context

`CatEngine` changed a mutable `CatSessionState` in place: `select_next` and `apply_response` appended
items, responses and estimates to it, and a `RunContext` carried the run's generator and an
`ExposureTracker` whose counts the engine updated. Because the session changed under them, step
results had to hold immutable snapshots of it. An application that ran live tests had to keep the
session object, the tracker and the context alive across requests, restore them after a restart and
share the tracker between sessions running side by side. Each of those was a way for state to
outlive a run and silently change a result.

The strategies already computed each step from an immutable `StepContext` that the engine built
from the session.

## Decision

- A session is an immutable value, `Session`: the items administered and their responses, in order;
  the estimate of every state, from the initial one to the one after the last response; the excluded
  items (ADR-0011); and the stop reason (ADR-0010). It holds plain data only.
- `CatEngine` holds the configuration of a test and never changes: the item bank, the initializer,
  the selector, the estimator, the stopper and an optional exposure control. Its step methods take a
  session and return a new one:

  ```python
  engine.start(*, rng, excluded_items=()) -> Session
  engine.select(session, *, rng, exposure_rates=None) -> Selection  # item | None, session, declined
  engine.answer(session, item, response, *, rng) -> Session
  ```

- A step's result depends only on its arguments and the engine's configuration. catsim keeps nothing
  between calls, so nothing has to be preserved for the next step to be right.
- The caller owns the session and what surrounds it. In a live test that is the application: it
  stores sessions, shows the item, runs sessions concurrently, keeps exposure counts and audits. In a
  simulation it is `simulate` (ADR-0014), which runs the same three methods.

## Consequences

- No session can change another or be changed after the fact; engines, banks and sessions are
  immutable, so threads and processes share them safely.
- A stored session is the record of the test, and an earlier state is an earlier value, so there are
  no step-result or snapshot types.
- Persisting a test is storing plain data, in whatever form the application chooses; catsim defines
  no file format for sessions.
- Simulated and live tests run the same code.
- Calling `select` twice on the same session can propose different items, since a selector may
  draw at random: keeping the item shown is the application's.
- Each `answer` copies the session's tuples, a cost proportional to the test length and small next
  to an estimate.

## Alternatives Considered

- A mutable session that the engine changes: its state outlives each call, and every result built
  from it needs a copy to stay right.
- Free functions taking the bank and every strategy at each call: every call repeats the
  configuration that the engine holds once.
- A mutable convenience wrapper beside the immutable session: a second way to run a test, which
  brings back the state this decision removes.
