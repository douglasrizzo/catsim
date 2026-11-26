# Embedding catsim

An application that gives adaptive tests can let catsim compute each step: the next item, the
ability estimate after each response and whether the test ends. It computes them from inputs the
application passes and keeps nothing between calls. It does not run, store or serve
tests: the sessions, the item on screen, the exposure counts, concurrency, persistence and audits
belong to the application ({doc}`/adr/0001-compute-and-evaluate-not-serve`). This guide runs a test
step by step, then lists what catsim checks in its inputs and what stays with the application. The
{doc}`introduction` explains the CAT loop, and {doc}`terms-and-notation` lists the terms used here.

## Configure the Test Once

```python
import numpy

from catsim import CatEngine, ItemBank
from catsim.estimation import EAPEstimator
from catsim.initialization import FixedPointInitializer
from catsim.selection import MaxInfoSelector
from catsim.stopping import MaxLengthStopper, StandardErrorStopper, StopRule

item_bank = ItemBank.generate(200, "3PL", rng=7)
stopper = StopRule(StandardErrorStopper(0.4), MaxLengthStopper(25), min_items=5)
engine = CatEngine(item_bank, FixedPointInitializer(0.0), MaxInfoSelector(r_max=0.3), EAPEstimator(), stopper)
```

A {py:class}`~catsim.CatEngine` holds the item bank and the strategies that make up the test:
where it starts, how it chooses items, how it estimates ability and when it stops. Here the test
starts at an ability of 0, gives the most informative item at the current estimate, and stops once
the standard error of the estimate is at most 0.4, after at least 5 and at most 25 items.
{doc}`running-simulations` shows how to compare such choices before going live.

The generated bank stands in for yours: read a calibrated bank with
{py:meth}`ItemBank.from_csv <catsim.ItemBank.from_csv>`, or pass its parameters to
{py:class}`~catsim.ItemBank`. catsim refers to an item by its row in the bank, its *index*, counted
from 0, and each item also has a name, `item_bank.names[item]`. Mapping indices or names to the
application's own item records is the application's.

An engine never changes, and neither does its bank: build them once, when the application starts,
and share them between requests, threads and processes.

## Run a Test Step by Step

A test takes three methods: {py:meth}`~catsim.CatEngine.start` begins a
{py:class}`~catsim.Session`, {py:meth}`~catsim.CatEngine.select` proposes its next item and
{py:meth}`~catsim.CatEngine.answer` records the response to it. A session never changes: each
method returns a new one, `select` within a {py:class}`~catsim.Selection`.

```python
rng = numpy.random.default_rng()
session = engine.start(rng=rng)
selection = engine.select(session, rng=rng)
print(selection.item, item_bank.names[selection.item])
session = engine.answer(selection.session, selection.item, True, rng=rng)
print(round(session.current_theta, 2), round(session.current_se, 2), session.stop_reason)
```

```text
51 item_051
0.35 0.95 None
```

`select` returns a {py:class}`~catsim.Selection`: the item to show, the session to answer it in
and the items an exposure control declined, if the engine has one. Answer the item in
`selection.session`, which holds those exclusions. The response is `True` or 1 for a correct answer,
`False` or 0 for an incorrect one. The new session holds the updated estimate and its standard error,
and its `stop_reason` stays `None` while the test runs.

Every step takes `rng`, the random source of the strategies it calls: the initializer, the selector's
tie-breaks and random choices, and the exposure control. It may be a
{py:class}`numpy.random.Generator`, a seed, or `None` for fresh entropy at each call. How the
sessions draw is the application's choice: one generator for all of them, one per session, or
fresh entropy at every step.

A whole test is a loop that ends when `select` has no item to offer. Here a function stands in for
the examinee:

```python
def respond(item: int) -> bool:
    """Answer correctly the items whose difficulty is below 0.5."""
    return bool(item_bank.difficulty[item] < 0.5)


session = engine.start(rng=rng)
while True:
    selection = engine.select(session, rng=rng)
    session = selection.session
    if selection.item is None:
        break
    session = engine.answer(session, selection.item, respond(selection.item), rng=rng)
print(len(session.administered_items), session.stop_reason, round(session.current_theta, 2))
```

```text
17 stopper 0.57
```

In an application the loop is spread over requests: one calls `select` and shows the item, a later
one receives the answer and calls `answer`. Between the two, the application keeps the session and
the item on screen.

## Read Why a Test Ended

Once a test ends, `session.stop_reason` is a {py:class}`~catsim.StopReason`:

| `stop_reason` | The test ended because |
|---|---|
| `StopReason.STOPPER` (`"stopper"`) | the stopper is satisfied: here, the standard error reached 0.4 or the test reached 25 items |
| `StopReason.NO_ITEMS` (`"no_items"`) | the selector had no item left to offer, or none that the exposure control accepted |

Each reason equals its string, so `session.stop_reason == "stopper"` holds. `NO_ITEMS` does not
always mean that the bank ran out: a selector that follows a fixed list or works through strata can
run out first, and so can a session whose remaining items are all excluded. A session that has
ended takes no more responses: `select` returns no item and the same session, and `answer` raises
`ValueError`.

To tell which criteria of a {py:class}`~catsim.stopping.StopRule` hold as a test ends, ask the rule
about the session as {py:meth}`~catsim.CatEngine.context` gives it
({ref}`combining-stopping-rules`):

```python
print(stopper.satisfied(engine.context(session)))
```

```text
(StandardErrorStopper(max_se=0.4),)
```

The standard error reached its target before the test reached 25 items.

## What catsim Checks

The application passes catsim its own data at every call: a session it stored, an item, a
response, exposure rates. catsim checks each input where it enters, when a `Session` is built and
when `start`, `select` or `answer` is called, for everything it can check without recomputing
anything ({doc}`/adr/0009-checking-explicit-inputs`):

- item indices are integers, in the bank, with no repeats, and no item is both administered and
  excluded;
- there is one response per administered item, each a Boolean or the integer 0 or 1;
- the estimates are finite, and their standard errors positive or infinite;
- exposure rates hold one value per item, each between 0 and 1;
- `answer` takes no response for a session that has ended, or for an item the session has had or
  excluded.

A failed check raises `TypeError` or `ValueError` and changes nothing. The message names the input,
the position, the value and the rule, and an item by its index and name:

```python
running = engine.answer(engine.start(rng=rng), 51, True, rng=rng)
for item, response in [(3.7, True), (51, False), (60, "yes")]:
    try:
        engine.answer(running, item, response, rng=rng)
    except (TypeError, ValueError) as error:
        print(error)
```

```text
item is 3.7; item indices must be integers
item is 51 ('item_051'); the session has already been given that item
response is 'yes'; responses must be bools or the integers 0 and 1
```

catsim cannot check what only recomputation would reveal: whether a stored estimate is the one the
responses give, or whether the exposure rates are the application's real ones. Those are the
application's.

## What the Application Owns

**Sessions and their storage.** A {py:class}`~catsim.Session` is plain data: its items, its
responses, the estimate of every state from the initial one, its excluded items and its stop reason.
It holds no identifier, no examinee and no engine, which the application keeps beside it. How and
where to store sessions is the application's; catsim defines no format for them. A session built
from stored fields with `Session(...)` is checked like any other.

**The item on screen.** Each call to `select` selects anew. A selector or an exposure control that
draws at random, or exposure rates that changed, can make a second call propose another item. Keep
the item shown and `selection.session` until the answer arrives, and pass both to `answer`.

**Exposure counts and rates.** catsim keeps no exposure counts. A selector with an exposure cap,
such as `MaxInfoSelector(r_max=0.3)`, passes over the items whose rate has reached the cap, and the
rates are an input of `select`: one value per item, between 0 and 1, or `None`, under which no cap
applies. The application counts how often it gives each item and computes the rates by whatever rule
suits sessions that run side by side; {py:func}`~catsim.simulate` documents the rule a simulation
uses. Here the item proposed first is passed over when its rate, 0.5, is above the cap of 0.3:

```python
session = engine.start(rng=rng)
first = engine.select(session, rng=rng).item
rates = numpy.zeros(item_bank.n_items)
rates[first] = 0.5
print(first, engine.select(session, rng=rng, exposure_rates=rates).item)
```

```text
51 173
```

An application that keeps proposal counts, for example to calibrate an exposure control, adds each
selection's `declined` items and its item. An application that keeps items from an examinee, such
as those of an earlier attempt, passes them as `start(rng=rng, excluded_items=...)`.

**Concurrency.** Engines, banks and sessions never change, so threads and processes share them
safely. Two requests that answer the same session both succeed, each with a session of its own:
which one counts is the application's to decide.

**Persistence and audits.** A stored session is the record of the test, from the initial estimate
on. Keeping it, with whatever the application ties to it, such as the examinee, the attempt, the
time of each answer or the item on screen, and auditing it are the application's.
{doc}`/adr/0006-steps-from-explicit-inputs` explains why each step takes everything as an input.
