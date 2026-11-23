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
cluster-selector
stratified-selector
a-strat-selector
a-strat-b-block-selector
max-info-strat-selector
max-info-b-block-selector
mei-selector
mlwi-selector
```

## Class Hierarchy

```{autoclasstree} catsim.selection
:direction: LR
:full:
:strict:
:namespace: catsim
```

## Planned

:::{warning}
The techniques in this section are not yet available in the released package.
They describe planned additions to {mod}`catsim.selection`.
:::

```{toctree}
:maxdepth: 1

kl-selector
progressive-selector
proportional-selector
sympson-hetter-exposure
mpwi-selector
klp-selector
mepv-selector
```
