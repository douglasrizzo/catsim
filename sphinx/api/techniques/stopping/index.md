# Stopping Criteria

```{py:module} catsim.stopping
```

Stopping criteria determine when an adaptive test should terminate. The engine
asks its stopper about the session before the first item and after each
response, and the stopper says whether the test ends there. Each stopper checks
one criterion, and a {class}`~catsim.stopping.StopRule` combines several.

## Implemented

```{toctree}
:maxdepth: 1

max-length-stopper
standard-error-stopper
confidence-interval-stopper
```

(combining-stopping-rules)=

## Combining stopping rules

A variable-length test usually needs more than one criterion: a precision or
classification target that ends the test once it is met, a cap on the test length
for the examinees who never meet it, and often a minimum length, so that no test
ends after a handful of responses. {class}`~catsim.stopping.StopRule` holds such a
combination. With $m$ administered items, a minimum length $m_{\min}$ and stoppers
$S_1, \dots, S_K$, it stops when

$$
m \ge m_{\min} \quad\text{and}\quad S_k \text{ holds for some } k.
$$

```python
from catsim.stopping import MaxLengthStopper, StandardErrorStopper, StopRule

# a standard error of at most 0.3 or 40 items, whichever comes first, and never fewer than 5 items
rule = StopRule(StandardErrorStopper(0.3), MaxLengthStopper(40), min_items=5)
```

`min_items` holds back every stopper in the rule, the length cap included. A rule
is itself a stopper, so it goes wherever a stopper does, another rule included.

catR caps its stopping rule the same way: its `maxItems` meta-argument "imposes a
maximal test length and forces the test to stop if this length is reached, even
though the stopping rule is not yet satisfied" {cite:p}`Magis12{p. 18}`. catR
also stops as soon as any rule of its `stop` list holds, as a rule of the
matching stoppers does; the parity tests
({doc}`ADR-0020 </adr/0020-catr-as-the-reference>`) find catR's decisions for a
length and a precision rule together and for two classification thresholds, and
each stopper's page says which catR rule it is.

Stoppers are pure, so a rule can be asked again after a run.
{meth}`~catsim.stopping.StopRule.satisfied` returns the stoppers that hold for a
session's final items, estimate and standard error, which tells, for example,
whether a session met its precision target or ran to the cap.

No stopper checks whether the item bank is exhausted: when the selector has no
item left to give, the engine ends the session itself.

```{eval-rst}
.. autoclass:: catsim.stopping.StopRule
   :members:
   :show-inheritance:
```

## Class Hierarchy

```{autoclasstree} catsim.stopping
:full:
:strict:
:namespace: catsim
```

## References

```{bibliography}
:filter: docname in docnames
```

## Base Class

```{eval-rst}
.. autoclass:: catsim.stopping.BaseStopper
   :members:
```
