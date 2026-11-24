# ADR-0004: Give Every Per-Step Strategy One Step Context

## Status

Accepted

## Context

The strategy interfaces first took engine-managed data through generic `**kwargs`, and this ADR
replaced them with explicit typed parameters. That left four contracts that named and ordered the
same session inputs differently:

| Strategy | Signature |
|---|---|
| initializer | `initialize(item_bank, rng)` |
| selector | `select(item_bank, administered_items, est_theta, rng=None, exposure_rates=None, response_vector=None)` |
| estimator | `estimate(item_bank, administered_items, response_vector, est_theta)` |
| stopper | `stop(item_bank, administered_items, theta)` |

The estimate was `est_theta` in two of them and `theta` in the third, and the responses came before
it in one and after it in another. `rng`, `exposure_rates` and `response_vector` were optional in
`select` although the engine always passed them, so the selectors that need them checked for `None`,
and a deterministic selector called directly without `rng` raised when items tied (ADR-0008).

Every input added so far changed every strategy signature, so a custom strategy written against an
older one constructed fine and failed at its first call. The planned techniques need more inputs:
the responses in `stop` for SPRT, GLR and curtailment stoppers, a generator for curtailment's Monte
Carlo refinement, and a set of items to pass over for Sympson-Hetter exposure control. Passing over
an item by adding it to the administered items would also move the test position, which several
selectors read from the number of administered items. Some planned techniques also kept
per-session state on the strategy object, such as a stopper's last classification, which the next
session on the same engine overwrites.

## Decision

The three strategies the engine calls at every step take one argument, a frozen
`catsim.StepContext` that the engine builds for each call:

```python
class BaseSelector(ABC):
  def select(self, context: StepContext) -> int: ...  # raises NoItemsAvailableError

class BaseEstimator(ABC):
  def estimate(self, context: StepContext) -> float: ...

class BaseStopper(ABC):
  def stop(self, context: StepContext) -> bool: ...

class BaseInitializer(ABC):
  def initialize(self, *, item_bank: ItemBank, rng: Generator) -> float: ...
```

- The context holds `item_bank`, `rng`, `administered_items`, `responses`, `excluded_items`, `theta`
  and `exposure_rates`. Only the item bank and the generator are required; the other fields default
  to a session with no responses and no exposure tracking (`exposure_rates` is `None` when the run
  has no tracker, and selectors then apply no cap). It checks that the responses hold one Boolean per
  administered item, so no strategy has to.
- The selector gets the session as it stands. The estimator gets the session with the newest item
  and response added, and `theta` still the previous estimate, its starting point. The stopper gets
  that same state with the new estimate. The engine builds the estimator's and the stopper's
  contexts before it changes the session (ADR-0011).
- The examinee's true ability and the session's bookkeeping are not in the context: strategies
  cannot see the simulated truth.
- `excluded_items` holds items a selector must not choose although they have not been administered.
  The engine passes none; a wrapper such as Sympson-Hetter passes them on with
  `dataclasses.replace`. Selectors draw their candidates through `catsim.selection.base.remaining`,
  which leaves out both, and the test position stays `len(context.administered_items)`.
- `select` returns an `int`. A selector with no item to offer raises `NoItemsAvailableError`; `None`
  is not a return value.
- The initializer keeps its two arguments, keyword-only: it runs once per session, before any step
  state exists, so there is nothing else to pass.
- Strategies hold configuration and caches derived from the item bank, never per-session state.
  One strategy object serves every session of a run, an application may interleave the sessions of
  several examinees on one engine, and look-ahead selectors such as MEI call their estimator with
  hypothetical responses. A strategy that reports something about a session, such as a
  classification, computes it from the context with a pure method instead of storing it.
- Adding a field to the context is not a breaking change: a new field gets a default, and a
  strategy reads only the fields it needs. Removing or renaming a field is.

## Consequences

- Custom strategies keep working when the context gains a field. The inputs have one name each,
  the context's field names, in every contract.
- `rng` and `responses` are always present, so the `None` checks are gone, and a deterministic
  selector called directly breaks ties as it does under the engine.
- Direct calls, in tests, notebooks and custom code, build a context:
  `selector.select(StepContext(item_bank=bank, rng=numpy.random.default_rng(0)))`.
- ty, which CI runs on catsim's own code, rejects a built-in strategy whose method does not match its
  base class's signature, and an engine call that does not match it. A custom strategy is checked
  only by its author's type checker; one with another signature fails at its first call.
- The context copies the session's items and responses into tuples for every call, which is
  proportional to the test length; MEI makes two copies per candidate item. Both are small next to
  an estimate.
- Contexts compare equal only to themselves: a generated equality would compare the exposure-rate
  arrays elementwise and raise.
