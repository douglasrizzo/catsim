# API Reference

```{eval-rst}
.. automodule:: catsim
```

Each object is documented at its public import path. The core types come from {mod}`catsim`; the
strategies, their base classes and the helpers for custom strategies from
{mod}`catsim.selection`, {mod}`catsim.estimation`, {mod}`catsim.initialization`,
{mod}`catsim.stopping` and {mod}`catsim.exposure`; functions and type aliases from
{mod}`catsim.irt`, {mod}`catsim.metrics`, {mod}`catsim.plot` and {mod}`catsim.typing`. The
technique pages document each strategy with its method.

```{toctree}
:maxdepth: 2

techniques/index
engine
state
context
simulation
item_bank
exposure
exceptions
metrics
irt
plot
typing
```
