# Urry Selector

:Class: {py:class}`catsim.selection.UrrySelector`
:Sources: {cite:t}`Urry1977`; {cite:t}`Lord1980`

## Motivation

The Urry selector, also called a `b`-matching rule, chooses the item whose
difficulty is closest to the current ability estimate. This criterion is easy to
compute, avoids the stronger preference for highly discriminating items shown by
maximum-information rules, and is historically tied to early tailored testing
systems.

## Definition

Let $b_i$ be the difficulty parameter of item $i$. The selector
chooses

$$
i^*
=
\arg\min_{i \in R_k} \left| b_i - \hat{\theta} \right|,
$$

where $R_k$ is the set of items neither administered nor excluded. Items at
exactly the same distance tie, and one of them is drawn at random
({doc}`/adr/0011-random-tie-breaking`).

## Parameters

This selector has no constructor parameters.

## Behavior and Comparison

Under the 1PL model, matching difficulty to ability is the same as maximizing
information, because every item's information curve has the same shape and peaks
at $b_i$. Once discriminations vary (2PL) or guessing enters (3PL, 4PL), a more
informative item can lie farther from $\hat{\theta}$, and the two rules differ;
b-matching remains a simple and interpretable heuristic.
{cite:t}`Magis12{p. 10}` describe Urry's criterion the same way: completely equivalent to
maximum information under the 1PL model, slightly different under other models.
They attribute it to Urry (1970), an earlier work than the article cited below.

## Guarantees

1. It returns the remaining item whose difficulty is closest to the current
   estimate.
2. It never returns an item that was already administered or excluded.
3. It applies no exposure cap: the run's exposure rates do not change its choice.
4. Under the 1PL, the chosen item coincides with a maximum-information choice,
   because information is maximized near $b_i = \hat{\theta}$.
5. When no item remains, it raises `NoItemsAvailableError`.

## References

```{bibliography}
:filter: docname in docnames
```

## API Reference

```{eval-rst}
.. autoclass:: catsim.selection.UrrySelector
   :members:
   :show-inheritance:
```
