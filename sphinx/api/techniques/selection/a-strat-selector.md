**Status:** Implemented
**Module:** {py:class}`catsim.selection.AStratSelector`
**Reference:** Chang & Ying (1999)

# A-Stratified Selector

## Motivation

The a-stratified method reduces overexposure of highly discriminating items by
reserving them for later stages of the test. Early stages use lower-
discrimination items, when the ability estimate is still unstable; later stages
unlock higher-discrimination items once $\hat{\theta}$ is more reliable.
Within each stage the selector stays adaptive by matching item difficulty to the
current estimate.

## Definition

Following Chang and Ying (1999), as described by Barrada et al. (2006,
p. 157), items are sorted by increasing discrimination and split into
`n_strata` contiguous strata,

$$
a_{(1)} \le a_{(2)} \le \dots \le a_{(J)}.
$$

The test is split into as many stages, and the item at test position $t$ comes
from stratum $s(t)$ (see {doc}`stratified-selector`). Within that stratum the
selector returns the non-administered item whose difficulty is closest to the
current estimate:

$$
i^* = \arg\min_{i \in \mathcal{S}_{s(t)} \cap R_t} \left| b_i - \hat{\theta} \right|.
$$

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
     - Intended test length.
   * - ``n_strata``
     - ``None``
     - Number of strata and stages; ``None`` uses ``test_size``.
```

## Behavioral Contracts

Any correct implementation must satisfy the following properties:

1. Strata are contiguous groups of increasing discrimination.
2. Early stages draw only from lower-discrimination strata and later stages from higher-discrimination strata.
3. The returned item minimizes $|b_i - \hat{\theta}|$ among the non-administered items of the active stratum.
4. The selector never returns an already administered item.
5. If the active stratum has been exhausted, the selector raises `NoItemsAvailableError`.

## References

Chang, H.-H., & Ying, Z. (1999). A-stratified multistage computerized adaptive testing. *Applied Psychological Measurement*, 23(3), 211-222. <https://doi.org/10.1177/01466219922031338>. Not openly available; Barrada et al. (2006) describe the method.

Barrada, J. R., Mazuela, P., & Olea, J. (2006). Maximum information stratification method for controlling item exposure in computerized adaptive testing. *Psicothema*, 18(1), 156-159. <https://www.psicothema.com/pdf/3191.pdf> (open access).

## API Reference

```{eval-rst}
.. autoclass:: catsim.selection.AStratSelector
   :members:
   :show-inheritance:
```
