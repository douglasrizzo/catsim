**Status:** Implemented
**Module:** {py:class}`catsim.selection.AStratBBlockSelector`
**Reference:** Chang et al. (2001)

# A-Stratified with B-Blocking Selector

## Motivation

Plain a-stratification can be weakened when item difficulty and discrimination
are correlated, because strata formed only by $a$ may also inherit skewed
difficulty distributions. The b-blocking refinement addresses this by spreading
difficulty more evenly before discrimination-based ordering is applied.

In {mod}`catsim`, the selector operationalizes this through a two-step presort:
global ordering by difficulty followed by within-stratum ordering by
discrimination.

## Definition

Let $b_i$ be item difficulty and $a_i$ item discrimination. The
package implementation performs the following presort:

1. Sort all items by increasing $b_i$.
2. Partition that ordering into `test_size` contiguous strata.
3. Within each stratum, sort items by increasing $a_i$.
4. Concatenate the reordered strata and use the generic stratified family rule.

At position $k+1$, the selector returns the first non-administered item in
the $k$ th stratum of this final ordering.

## Parameters

```{eval-rst}
.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``test_size``
     - required
     - Number of strata and intended test length.
```

## Behavioral Contracts

Any correct implementation must satisfy the following properties:

1. The global presort starts from increasing item difficulty, not increasing discrimination.
2. Each difficulty block is internally reordered by increasing discrimination before selection begins.
3. The selector only draws from the active stage stratum determined by the current test position.
4. It never returns an already administered item.
5. If the active stratum has no non-administered items left, it raises `NoItemsAvailableError`.

## References

Chang, H.-H., Qian, J., & Ying, Z. (2001). A-stratified multistage computerized adaptive testing with b blocking. *Applied Psychological Measurement*, 25(4), 333-341. <https://doi.org/10.1177/01466210122032181>

## API Reference

```{eval-rst}
.. autoclass:: catsim.selection.AStratBBlockSelector
   :members:
   :show-inheritance:
```
