# Selection Techniques

Item selection techniques determine which item from the bank to administer at each
step of an adaptive test. The choice of selector affects both measurement precision
and practical properties such as item exposure and content coverage.

## Implemented

```{toctree}
:maxdepth: 1

linear-selector
random-selector
randomesque-selector
the-54321-selector
max-info-selector
urry-selector
interval-info-selector
kl-selector
cluster-selector
stratified-selector
a-strat-selector
a-strat-b-block-selector
max-info-strat-selector
max-info-b-block-selector
```

## Class Hierarchy

```{eval-rst}
.. inheritance-diagram:: catsim.selection.BaseSelector catsim.selection.FiniteSelector catsim.selection.MaxInfoSelector catsim.selection.UrrySelector catsim.selection.IntervalInfoSelector catsim.selection.LinearSelector catsim.selection.RandomSelector catsim.selection.RandomesqueSelector catsim.selection.The54321Selector catsim.selection.ClusterSelector catsim.selection.StratifiedSelector catsim.selection.AStratSelector catsim.selection.AStratBBlockSelector catsim.selection.MaxInfoStratSelector catsim.selection.MaxInfoBBlockSelector
   :parts: 1
   :top-classes: catsim.selection.BaseSelector
```

## Planned

:::{warning}
The techniques in this section are not yet available in the released package.
They describe planned additions to {mod}`catsim.selection`.
:::

```{toctree}
:maxdepth: 1

progressive-selector
proportional-selector
mei-selector
mlwi-selector
sympson-hetter-exposure
mpwi-selector
klp-selector
mepv-selector
```
