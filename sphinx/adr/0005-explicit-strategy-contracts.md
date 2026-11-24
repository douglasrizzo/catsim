# ADR-0005: Strategies Take One Step Context and Estimators Return an Estimate

## Status

Accepted 2026-11-24

## Context

The strategy interfaces took the session's inputs as explicit parameters, which the four
contracts named and ordered differently, and `rng` was optional in `select` although the engine
always passed it, so a deterministic selector called directly raised when items tied (ADR-0011).
Every added input changed every signature, so a custom strategy written against an older one failed
at its first call, and the planned techniques need more: the responses in `stop` for SPRT, GLR and
curtailment stoppers, and items to pass over for Sympson-Hetter exposure control. Some also kept
per-session state on the strategy object, which the next session on the same engine overwrites.

The estimator returned a bare `float`, so the precision stoppers computed the Fisher standard error
themselves, whatever the estimator. For EAP that is not the estimator's own error, the posterior
standard deviation, and the difference changes stopping decisions in short tests; catR uses each
estimator's own standard error. The estimator also counted its objective evaluations on the
object, over every session it served and MEI's look-ahead calls, so no count belonged to a session.

## Decision

The three strategies the engine calls at every step take one argument, a frozen
`catsim.StepContext` that the engine builds for each call:

```python
class BaseSelector(ABC):
  def select(self, context: StepContext) -> int: ...  # raises NoItemsAvailableError

class BaseEstimator(ABC):
  def estimate(self, context: StepContext) -> Estimate: ...

class BaseStopper(ABC):
  def stop(self, context: StepContext) -> bool: ...

class BaseInitializer(ABC):
  def initialize(self, *, item_bank: ItemBank, rng: Generator) -> float: ...
```

- The context holds the item bank, the run's generator and the session as each strategy needs it
  (`StepContext`'s docstring and {doc}`/extending` list its fields). The examinee's true ability is
  not in it, so strategies cannot see the simulated truth.
- `excluded_items` holds the items a selector must not choose although they have not been
  administered (ADR-0008), so the test position stays the number of administered items.
- The exposure control, an optional engine component (ADR-0008), follows the same rules:
  `accept(context, item_id) -> bool`.
- The initializer runs before any step state exists, so it keeps its two keyword-only arguments.
- Strategies hold configuration and caches derived from the item bank, never per-session state:
  one strategy object serves every session of a run, an application may interleave several
  examinees' sessions on one engine, and MEI calls its estimator with hypothetical responses. A
  strategy that reports something about a session computes it from the context.
- Adding a field to the context is not a breaking change: a new field gets a default, and a
  strategy reads only the fields it needs. Removing or renaming a field is.
- `estimate` returns a frozen `catsim.Estimate(theta, se, evaluations)`: the estimate, its standard
  error by the estimator's own formula (finite or `math.inf`, never NaN) and its objective
  evaluations, or `None`. The WLE reports the Fisher standard error, as catR's default does
  ({doc}`/api/techniques/estimation/warm-wle`). The session records every step's `Estimate`, and the
  precision stoppers judge `context.se`.

## Consequences

- Custom strategies keep working when the context gains a field, and each input has one name in
  every contract.
- `rng` and `responses` are always present, so a deterministic selector called directly, with a
  context it builds ({doc}`/extending`), breaks ties as under the engine.
- A precision stopper's decision depends on the estimator: the Bayesian standard errors are usually
  below the Fisher one ({doc}`/api/techniques/estimation/index` compares them).
- An estimator's cost is the sum of a session's recorded `evaluations`. Look-ahead estimates are
  never recorded, so MEI may share the engine's estimator.

## Alternatives Considered

- Explicit per-strategy parameters: every added input changes every signature.
- Passing over an item by adding it to the administered items: it moves the test position, which
  several selectors read.
- An optional `BaseEstimator.standard_error(context)`, defaulting to the Fisher one: EAP would
  compute the posterior twice per step, and caching it between the calls would put per-session
  state on an estimator that serves every session.
