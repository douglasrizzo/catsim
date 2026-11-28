# API Reference

```{eval-rst}
.. automodule:: catsim
```

Each object is documented at its public import path. The core types come from {mod}`catsim`; the
strategies, their base classes and the helpers for custom strategies from
{mod}`catsim.selection`, {mod}`catsim.estimation`, {mod}`catsim.initialization`,
{mod}`catsim.stopping` and {mod}`catsim.exposure`; functions, with the types they take or return,
from {mod}`catsim.metrics` and {mod}`catsim.plot`. The technique pages document each strategy with
its method.

The item bank (`irtlib.ItemBank`, with `irtlib.IrtModel`), the item response functions
(`irtlib.irt`) and the array types (`irtlib.typing`) come from
[irtlib](https://github.com/douglasrizzo/irtlib), the item response theory package catsim builds
on; their docstrings are their reference.

```{toctree}
:maxdepth: 2

engine
session
context
simulation
exposure
metrics
plot
exceptions
```
