# ADR-0007: Strategies Are Classes That Take One Step Context

## Status

Proposed 2026-11-25

## Context

catsim's users include method authors who write their own selectors, estimators and stoppers, and
who need to do it without changing catsim. catR has fixed rule names (`criterion = "MFI"`,
`method = "BM"`), so a new criterion needs a change to catR itself.

The strategy interfaces once took the session's inputs as explicit parameters, named and ordered
differently in each contract, so every added input broke every custom strategy. The estimator
returned a bare `float`, so the precision stoppers computed the Fisher standard error whatever the
estimator; for EAP that is not its own error, the posterior standard deviation, and the difference
changes stopping decisions in short tests. catR uses each estimator's own standard error. Some
strategies kept per-session state on the object, which the next session on the same engine
overwrote.

## Decision

Every strategy type is an abstract base class with a documented contract, and the built-in
strategies implement the same contracts with no private access. The three strategies called at
every step, and the exposure control (ADR-0011), take one argument, a frozen `catsim.StepContext`
built for each call:

```python
class BaseSelector(ABC):
  def select(self, context: StepContext) -> int: ...  # raises NoItemsAvailableError

class BaseEstimator(ABC):
  def estimate(self, context: StepContext) -> Estimate: ...

class BaseStopper(ABC):
  def stop(self, context: StepContext) -> bool: ...

class BaseExposureControl(ABC):
  def accept(self, context: StepContext, item: int) -> bool: ...

class BaseInitializer(ABC):
  def initialize(self, *, item_bank: ItemBank, rng: Generator) -> float: ...
```

- The context holds the item bank, the generator of the call, the session's items, responses,
  excluded items, current estimate and standard error, and the exposure rates. The examinee's true
  ability is not in it, so strategies cannot see the simulated truth.
- The initializer runs before any step exists, so it keeps its two keyword-only arguments.
- Strategies hold configuration and data derived from it and the bank (ADR-0008), never anything
  about a session: one strategy object serves every session, and look-ahead selectors such as MEI
  call an estimator with hypothetical responses.
- Adding a field to the context is not a breaking change: a new field gets a default, and a strategy
  reads only the fields it needs. Removing or renaming a field is.
- `estimate` returns a frozen `catsim.Estimate(theta, se, evaluations)`: the estimate, its standard
  error by the estimator's own formula (finite or `math.inf`, never NaN) and its objective
  evaluations, or `None`. The precision stoppers judge `context.se`.

## Consequences

- A method author writes a strategy in their own code, subclasses one base class and passes an
  instance to the engine; nothing registers it.
- Custom strategies keep working when the context gains a field, and each input has one name in
  every contract.
- A strategy can be called and tested on a context its author builds, and breaks ties as it does
  under the engine, since the context always carries a generator.
- A precision stopper's decision depends on the estimator, whose standard error it reads.

## Alternatives Considered

- Rule names resolved inside catsim, as catR's: a new technique needs a change to catsim.
- Explicit per-strategy parameters: every added input changes every signature.
- A registry of strategies by name: configuration by strings, with a second way to name a class.
- An optional `standard_error(context)` method: EAP would compute its posterior twice per step.
