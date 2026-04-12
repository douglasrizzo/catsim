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

```{eval-rst}
.. inheritance-diagram:: catsim.stopping.BaseStopper catsim.stopping.TestLengthStopper catsim.stopping.MinErrorStopper catsim.stopping.ConfidenceIntervalStopper
   :parts: 1
   :top-classes: catsim.stopping.BaseStopper
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
