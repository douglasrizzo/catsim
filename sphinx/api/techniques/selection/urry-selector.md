**Status:** Implemented
**Module:** {py:class}`catsim.selection.UrrySelector`
**Reference:** Urry (1977); Lord (1980)

# Urry Selector

## Motivation

The Urry selector, also called a `b`-matching rule, chooses the item whose
difficulty is closest to the current ability estimate. This criterion is easy to
compute, avoids the stronger preference for highly discriminating items shown by
maximum-information rules, and is historically tied to early tailored testing
systems.

Under the 1PL model, matching difficulty to ability is the same as maximizing
information, because every item's information curve has the same shape and peaks
at $b_i$. Once discriminations vary (2PL) or guessing enters (3PL, 4PL), a more
informative item can lie farther from $\hat{\theta}$, and the two rules differ;
b-matching remains a simple and interpretable heuristic. Magis and Raîche
(2012, p. 10) describe Urry's criterion the same way: completely equivalent to
maximum information under the 1PL model, slightly different under other models.
They attribute it to Urry (1970), an earlier work than the article cited below.

## Definition

Let $b_i$ be the difficulty parameter of item $i$. The selector
chooses

$$
i^*
=
\arg\min_{i \in R_k} \left| b_i - \hat{\theta} \right|,
$$

where $R_k$ is the set of non-administered items. The current package
implementation sorts all items by absolute distance from $\hat{\theta}$ to
their $b$ values and returns the first remaining item.

## Parameters

This selector has no constructor parameters.

## Behavioral Contracts

Any correct implementation must satisfy the following properties:

1. It ranks items by increasing absolute distance between item difficulty and `est_theta`.
2. It never returns an already administered item.
3. It ignores exposure-rate inputs; only difficulty proximity controls the choice.
4. In Rasch/1PL settings, the chosen item coincides with a maximum-information choice because information is maximized near $b_i = \hat{\theta}$.
5. If no non-administered items remain, it raises `NoItemsAvailableError`.

## References

Urry, V. W. (1977). Tailored testing: A successful application of latent trait theory. *Journal of Educational Measurement*, 14(2), 181-196. <https://doi.org/10.1111/j.1745-3984.1977.tb00035.x>. Not openly available; Magis and Raîche (2012) describe the criterion.

Lord, F. M. (1980). *Applications of Item Response Theory to Practical Testing Problems*. Hillsdale, NJ: Lawrence Erlbaum Associates.

Magis, D., & Raîche, G. (2012). Random generation of response patterns under computerized adaptive testing with the R package catR. *Journal of Statistical Software*, 48(8), 1-31. <https://doi.org/10.18637/jss.v048.i08> (open access).

## API Reference

```{eval-rst}
.. autoclass:: catsim.selection.UrrySelector
   :members:
   :show-inheritance:
```
