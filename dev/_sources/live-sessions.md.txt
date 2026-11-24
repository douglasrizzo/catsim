# Live CAT Sessions

In an application, catsim decides which item to show next and when the test ends; the application
shows the items, collects the answers and stores the sessions. This guide runs a session item by
item, reads why it ended, saves and resumes it, and shares item exposure across sessions. The
{doc}`introduction` explains the CAT loop, and {doc}`terms-and-notation` lists the terms used here.

## Set Up the Engine

```python
import dataclasses
import json

import numpy

from catsim import CatEngine, CatSessionSnapshot, CatSessionState, Estimate, ExposureTracker, ItemBank, RunContext
from catsim.estimation import EAPEstimator
from catsim.initialization import FixedPointInitializer
from catsim.selection import MaxInfoSelector
from catsim.stopping import MaxLengthStopper, StandardErrorStopper, StopRule

item_bank = ItemBank.generate(200, "3PL", rng=7)
stopper = StopRule(StandardErrorStopper(0.4), MaxLengthStopper(25), min_items=5)
engine = CatEngine(item_bank, FixedPointInitializer(0.0), MaxInfoSelector(r_max=0.3), EAPEstimator(), stopper)
tracker = ExposureTracker(item_bank.n_items)
context = RunContext(rng=numpy.random.default_rng(), exposure_tracker=tracker)
```

A {py:class}`~catsim.CatEngine` holds the item bank and the four strategies that make up the test:
where it starts, how it chooses items, how it estimates ability and when it stops. Here the test
starts at an ability of 0, gives the most informative item at the current estimate, and stops once
the standard error of the estimate is at most 0.4, after at least 5 and at most 25 items.
{doc}`running-simulations` shows how to compare such choices before going live. The generated bank
stands in for yours: wrap your calibrated parameters in {py:class}`~catsim.ItemBank`, one row per
item with the columns $a$, $b$, $c$ and $d$. An item is identified by its row, counted from 0.

Build the engine once, when the application starts, and share it: it keeps nothing about any
session, so one engine serves every examinee. A {py:class}`~catsim.RunContext` holds what the
sessions share: the random generator the strategies draw from, and the
{py:class}`~catsim.ExposureTracker` that counts how often each item is given (see "Share Exposure
across Sessions" below). An unseeded generator draws fresh randomness each time the application
starts.

## Run a Session Item by Item

```python
session = engine.start_session("attempt-1", context)
item_id = engine.select_next(session, context)
step = engine.apply_response(session, item_id, response=True, context=context)
print(step.selected_item_id, round(step.current_theta, 2), step.stopped)
```

`start_session` gives the session its initial estimate. Its first argument identifies the session,
any hashable value, such as the id of the attempt in your database. `select_next` returns the index
of the item to show. `apply_response` records the examinee's answer, `True` for correct, updates the
ability estimate, asks the stopper whether the test is over, and returns a
{py:class}`~catsim.CatStepResult` with the estimate and whether the session stopped. A response must
be a `bool`: `1` or `"yes"` raise `TypeError`.

A whole session is a loop that ends when `select_next` returns `None`, which it does once the test
has ended, so the loop needs no other check:

```python
def ask(item_id: int) -> bool:
    """Stand in for the examinee: answer correctly the items whose difficulty is below 0.5."""
    return bool(item_bank.difficulty[item_id] < 0.5)


while (item_id := engine.select_next(session, context)) is not None:
    engine.apply_response(session, item_id, response=ask(item_id), context=context)
print(session.administered_count, session.stop_reason, round(session.current_theta, 2))
```

In a web application, `select_next` and `apply_response` usually run in different requests: one
shows the item, the next receives the answer. Save the session after each of them (see "Save and
Resume a Session").

When the application can get the answer within the call, {py:meth}`~catsim.CatEngine.step`
selects an item, asks a {py:class}`~catsim.ResponseProvider` for the response and applies it, and
{py:meth}`~catsim.CatEngine.run_session` repeats `step` until the session ends. A provider is any
object with an `answer` method. This one replays answers recorded earlier for every item of the
bank:

```python
class RecordedAnswers:
    def __init__(self, answers: numpy.ndarray) -> None:
        self.answers = answers

    def answer(self, item_bank: ItemBank, item_id: int, session: CatSessionSnapshot, rng: numpy.random.Generator) -> bool:
        return bool(self.answers[item_id])


replayed = engine.run_session(engine.start_session("attempt-2", context), RecordedAnswers(item_bank.difficulty < 0.5), context)
```

## Read Why a Session Ended

`session.stop_reason` is `None` while the session runs, and `is_stopped` is false. Once it ends,
the reason is a {py:class}`~catsim.StopReason`:

| `stop_reason` | The session ended because |
|---|---|
| `StopReason.STOPPER` (`"stopper"`) | the stopper is satisfied: here, the standard error reached 0.4 or the test reached 25 items |
| `StopReason.NO_ITEMS` (`"no_items"`) | the selector had no item left to offer, or none that the exposure control accepted |

Each reason equals its string, so `session.stop_reason == "stopper"` holds, and a saved string
restores it. `NO_ITEMS` does not always mean that the bank ran out: a selector that follows a fixed
list or works through strata can run out first. A stopped session takes no more responses:
`apply_response` raises `ValueError`, and `select_next` keeps returning `None`.

To tell which criteria of a {py:class}`~catsim.stopping.StopRule` hold when a test ends (those that
ended it, when its `stop_reason` is `STOPPER`), compare the
session's `current_se` and `administered_count` with its targets, or ask the rule
({ref}`combining-stopping-rules`) about the session as {py:meth}`~catsim.CatEngine.context` gives
it:

```python
print(stopper.satisfied(engine.context(session)))
```

The rule returns the stoppers that hold, here `(StandardErrorStopper(max_se=0.4),)`: the standard
error reached its target before the test reached 25 items.

## Save and Resume a Session

A {py:class}`~catsim.CatSessionState` is a dataclass of plain values, so it converts to a
dictionary and back:

```python
saved = json.dumps(dataclasses.asdict(session))

fields = json.loads(saved)
restored = CatSessionState(**{**fields, "estimates": [Estimate(**estimate) for estimate in fields["estimates"]]})
print(restored == session)
```

The dictionary holds the session id, the estimates (each one's `theta`, its standard error `se` and
the estimator's `evaluations`), the administered items, the responses, the stop reason, the
excluded items and a true ability, which is `None` in a live session. Rebuild the estimates as
{py:class}`~catsim.Estimate` objects when restoring; the stop reason may stay a string. Data that
ties the session to a user or an attempt belongs to the application, stored next to these fields.

Two values need care in storage. The session id must be one your storage can hold: save a UUID as
a string. The initial estimate has an infinite standard error, which `json.dumps` writes as
`Infinity`; Python reads it back, but strict JSON parsers and many databases reject it, so encode it
in a form your storage accepts and decode it to `math.inf`.

Save the session after `start_session`, after every `select_next` and after every
`apply_response`. `select_next` can change the session: with an exposure control, the items it
declines join the session's excluded items. Also keep the id of the item on screen. On a page
reload, show that item again and pass it to `apply_response`; do not call `select_next` again: each
call selects anew, and the exposure control decides anew, so a second call may decline the item
already shown, choose another or end the test.

## Share Exposure across Sessions

An item given to many examinees becomes known to future ones. A selector with an exposure cap,
such as `MaxInfoSelector(r_max=0.3)`, passes over the items whose exposure rate has reached 0.3,
unless no other item remains. The rates come from the tracker in the context, so share one
tracker, and one context, among every session on the item bank; without a tracker the cap does
nothing.

A session sees each item's rate over the other sessions: the number of sessions that were given
the item, divided by the number of sessions started so far other than its own. Sessions still in
progress count too, so sessions that run side by side see each other's items as they are given,
and the first session sees every rate at 0. When sessions run one after another, as in a
simulation, this is the share of the examinees tested before the current one
({doc}`/adr/0004-runtime-state-ownership`). Above, `attempt-2` starts with a different item from
`attempt-1`: with two sessions started, each item `attempt-1` was given has a rate of 1 for
`attempt-2`. The {py:class}`~catsim.ExposureTracker` reference gives the exact count.

The tracker lives in memory. A tracker created at a restart starts from zero and applies no cap
until new sessions have run, so save its counts and rebuild it from them:

```python
saved_tracker = json.dumps(
    {"counts": tracker.counts.tolist(), "proposals": tracker.proposals.tolist(), "n_sessions": tracker.n_sessions}
)

tracker = ExposureTracker(item_bank.n_items, **json.loads(saved_tracker))
context = RunContext(rng=numpy.random.default_rng(), exposure_tracker=tracker)
```

`counts` holds each item's administrations, `proposals` its administrations plus the times an
exposure control declined it, and `n_sessions` the sessions started. Save them whenever you save a
session, in the same transaction if you can: a tracker saved only at shutdown is lost in a crash,
and a tracker older than the sessions saved after it under-counts them, which no check detects. A session that starts and is abandoned still
counts as a session, as does a second session started for the same examinee; each lowers every rate a little. A tracker counts only the sessions that run
on its context: processes with trackers of their own do not see each other's administrations.

## Decline Items with an Exposure Control

An engine can also take an *exposure control*, a {py:class}`~catsim.exposure.BaseExposureControl`,
which may decline the item the selector proposes. The control below administers each item with a
probability of its own, 0.5 for every item here; {doc}`extending` gives the rules every control
keeps.

```python
from typing_extensions import override

from catsim import ExcludedItemError, StepContext
from catsim.exposure import BaseExposureControl


class AdministrationProbabilities(BaseExposureControl):
    def __init__(self, probabilities: numpy.ndarray) -> None:
        self.probabilities = probabilities

    @override
    def accept(self, context: StepContext, item_id: int) -> bool:
        return bool(context.rng.random() < self.probabilities[item_id])


controlled = CatEngine(
    item_bank, FixedPointInitializer(0.0), MaxInfoSelector(), EAPEstimator(), stopper,
    exposure_control=AdministrationProbabilities(numpy.full(item_bank.n_items, 0.5)),
)
retake = controlled.start_session("attempt-3", context)
retake.excluded_items.extend(session.administered_items)
item_id = controlled.select_next(retake, context)
print(item_id, retake.excluded_items[session.administered_count:])
```

`retake` is a second attempt by the examinee of `attempt-1`. Its excluded items, filled before the
first item, keep it from repeating the first attempt's items. `select_next` then asks the control
about each proposal: the items it declines join the excluded items for the rest of the test, and the
tracker counts them in `proposals`; the print shows the item accepted and the ones declined before
it. A session whose remaining items are all declined ends with `StopReason.NO_ITEMS`.

During a test the excluded items only grow. Removing one to give a declined item after all is not
supported, because the decline is already counted as a proposal. A response to an excluded item
raises {py:class}`~catsim.ExcludedItemError`, a `ValueError`, and changes nothing:

```python
try:
    controlled.apply_response(retake, session.administered_items[0], response=True, context=context)
except ExcludedItemError:
    print("excluded")
```
