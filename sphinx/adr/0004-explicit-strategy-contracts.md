# ADR-0004: Strategies Take One Step Context and Estimators Return an Estimate

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

The estimator returned a bare `float`, so the precision stoppers computed a standard error
themselves, always the Fisher one at the estimate (`irt.see`), whatever the estimator. For EAP that
is not the estimator's own error, the posterior standard deviation: on 200 simulated EAP sessions
in the 1.0 review, the Fisher standard error was a median 26% above the posterior standard
deviation after 3 items and 4% above after 20, and with a 0.4 threshold the stopping decision after
20 items differed in 47 sessions. catR uses each estimator's own standard error (Magis and Raîche
2012). The estimator also counted its calls and objective evaluations on the object (`calls`,
`evaluations`, `total_evaluations`, `avg_evaluations`). Those counters summed over every session the
estimator served and over MEI's look-ahead calls, so no count could be attributed to a session or a
step.

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

- The context holds `item_bank`, `rng`, `administered_items`, `responses`, `excluded_items`, `theta`,
  `se` and `exposure_rates`. Only the item bank and the generator are required; the other fields
  default to a session with no responses and no exposure tracking (`se` is `math.inf`;
  `exposure_rates` is `None` when the run has no tracker, and selectors then apply no cap). It
  checks that the responses hold one Boolean per administered item, so no strategy has to.
- The selector gets the session as it stands. The estimator gets the session with the newest item
  and response added, and `theta` and `se` still the previous estimate's, its starting point. The
  stopper gets that same state with the new estimate and its standard error. The engine builds the
  estimator's and the stopper's contexts before it changes the session (ADR-0011).
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

`estimate` returns a frozen `catsim.Estimate(theta, se, evaluations)`: the estimate, its standard
error by the estimator's own formula, and how many times the estimator evaluated its objective for
this estimate, or `None` where that count has no meaning. The standard error is finite, or
`math.inf` when the responses carry no information about the ability; it is never NaN.

| Estimator | `se` | `evaluations` |
|---|---|---|
| `NumericalSearchEstimator` | $1/\sqrt{I(\hat\theta)}$ (Magis and Raîche 2012, Eq. 3, p. 4) | SciPy's function evaluations; 0 for Dodd's step |
| `EAPEstimator` | the posterior standard deviation (Eq. 11, p. 6) | the grid's node count |
| `MAPEstimator` | $1/\sqrt{I(\hat\theta) - (\log p)''(\hat\theta)}$, which is $1/\sqrt{I(\hat\theta) + 1/\sigma^2}$ for a normal prior (Eq. 6, p. 5) | 201 search points plus the refinement's evaluations |
| `WarmLikelihoodEstimator` | $1/\sqrt{I(\hat\theta)}$, catR's default | the root scan's, Brent's and the ranking integrals' evaluations |

- For the WLE catsim departs from Magis and Raîche (2012, Eq. 14, p. 6), who add
  $(I'J - IJ')/(2I^2)$ to the information. catR dropped that term in version 3.0, whose NEWS
  says it sometimes led to mathematical problems, and its `semTheta` still returns $1/\sqrt{I}$
  for the WLE under the default `sem.type = "classic"`. catsim follows catR (the maintainer's
  decision, 2026-11-24), after comparing the formulas with the WLE's sampling spread at a known
  ability (every response pattern for 5 and 10 items, 3,000 simulated ones for 20 and 40):
  - The classic formula overestimates the error in short tests: by 10% (2PL) and 28% (3PL) at
    5 items. That is conservative: a standard-error stopper runs a few items longer than needed.
  - Eq. 14 roughly halves that gap. The sandwich formula of Magis (2016), catR's
    `sem.type = "new"`, is closest under the 3PL but underestimates the error under the 2PL
    (by 8% at 5 items), which would stop tests too early.
  - From 20 items on, the three agree within about 3%.
  - Under the 1PL and 2PL, Eq. 14's sum is the curvature of the weighted likelihood at its
    maximum, so it is positive. With guessing or slipping parameters it can be negative, which
    leaves no standard error: 13 of 3,920 response patterns (0.4% of their probability) on random
    3PL tests of 2 to 8 items, none of 4,560 on 2PL tests.
- The engine records every step's `Estimate` in the session. `CatSessionState.estimates` starts with
  `Estimate(theta=initial, se=math.inf, evaluations=0)` and gains the estimator's result for each
  response; `theta_history`, `current_theta` and `current_se` are read from it, and snapshots copy
  it.
- `StandardErrorStopper` and `ConfidenceIntervalStopper` judge the standard error the estimator
  reported, `context.se`, instead of computing one.
- `BaseEstimator` has no counters and no `__init__`. The per-step counts in the session's estimates
  replace them, and MEI's look-ahead estimates are never recorded, so MEI may share the engine's
  estimator.
- Rejected: an optional `BaseEstimator.standard_error(context)` method, defaulting to the Fisher
  standard error. EAP would compute the posterior twice per step, once for the mean and once for
  its standard deviation, and caching the posterior between the two calls would put per-session
  state on an estimator that serves every session.

## Consequences

- Custom strategies keep working when the context gains a field. The inputs have one name each,
  the context's field names, in every contract.
- `rng` and `responses` are always present, so the `None` checks are gone, and a deterministic
  selector called directly breaks ties as it does under the engine.
- Direct calls, in tests, notebooks and custom code, build a context:
  `selector.select(StepContext(item_bank=bank, rng=numpy.random.default_rng(0)))`. A direct
  `estimate` call gets an `Estimate` and reads `.theta`, `.se` or `.evaluations` from it.
- ty, which CI runs on catsim's own code, rejects a built-in strategy whose method does not match its
  base class's signature, and an engine call that does not match it. A custom strategy is checked
  only by its author's type checker; one with another signature fails at its first call.
- The context copies the session's items and responses into tuples for every call, which is
  proportional to the test length; MEI makes two copies per candidate item. Both are small next to
  an estimate.
- Contexts compare equal only to themselves: a generated equality would compare the exposure-rate
  arrays elementwise and raise.
- With `NumericalSearchEstimator` and `WarmLikelihoodEstimator` the precision stoppers decide as
  before: the standard error they report is the Fisher one the stoppers computed. With
  `EAPEstimator` and `MAPEstimator` the decisions change. Their standard error is usually below the
  Fisher one at the same estimate, and with a normal prior MAP's always is, so seeded runs that
  pair them with a precision stopper end sooner.
- A custom estimator returns an `Estimate`. The engine raises a `TypeError` naming the estimator when
  it returns anything else, instead of failing later on a missing `.theta`. An estimator with no
  standard error formula of its own can report `irt.see(theta, items)`, the Fisher one.
- An application that stores sessions stores their estimates and rebuilds them as `Estimate`s:
  `theta_history` is derived and read-only, and `CatSessionState` rejects entries that are not
  `Estimate`s.
- An estimator's cost over a session is the sum of its estimates' `evaluations`; summed over the
  sessions, it compares estimators across simulations. No count is kept outside a session.
