# Stopping Criteria

Stopping criteria determine when an adaptive test should terminate. A stopper
evaluates the current session state after each item and signals whether testing
should continue or stop.

## Implemented

```{toctree}
:maxdepth: 1

test-length-stopper
min-error-stopper
confidence-interval-stopper
```

## Class Hierarchy

```{autoclasstree} catsim.stopping
:full:
:strict:
:namespace: catsim
```

## Planned

:::{warning}
The techniques in this section are not yet available in the released package.
They describe planned additions to {mod}`catsim.stopping`.
:::

```{toctree}
:maxdepth: 1

predicted-sem-stopper
min-info-bank-stopper
kl-stopper
sprt-stopper
glr-stopper
stochastic-curtailment-stopper
```
