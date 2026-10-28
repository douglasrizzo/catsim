**Status:** Implemented
**Module:** {py:class}`catsim.selection.MaxInfoBBlockSelector`
**Reference:** Barrada et al. (2006); Chang et al. (2001)

# Maximum-Information with B-Blocking Selector

## Motivation

MIS-B applies the blocking idea of Chang, Qian and Ying (2001) to
maximum-information stratification: it blocks on the ability at which each item
is most informative, so that every stratum covers the whole ability range.
Barrada et al. (2006, pp. 158–159) found that blocking reduced the skewness of
the exposure rates both with and without correlation between $a$ and $b$, and
recommended MIS-B whenever a stratified method is used.

## Definition

With $\theta_i^{\max}$ and $I_i^{\max}$ as in {doc}`max-info-strat-selector`:

1. Sort all items by increasing $\theta_i^{\max}$ and group them into
   consecutive blocks of `n_strata` items.
2. Within each block, sort the items by increasing $I_i^{\max}$.
3. The $j$-th item of every block joins stratum $j$.

At test position $t$ the selector returns the non-administered item of stratum
$s(t)$ whose point of maximum information is closest to the current estimate,

$$
i^* = \arg\min_{i \in \mathcal{S}_{s(t)} \cap R_t} \left| \theta_i^{\max} - \hat{\theta} \right|.
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

1. In every block of `n_strata` items ordered by $\theta_i^{\max}$, the `j`-th item by maximum information belongs to stratum `j`.
2. Every stratum spans the range of $\theta_i^{\max}$ in the bank.
3. The returned item minimizes $|\theta_i^{\max} - \hat{\theta}|$ among the non-administered items of the active stratum.
4. The selector never returns an already administered item.
5. If the active stratum has been exhausted, it raises `NoItemsAvailableError`.

## References

Barrada, J. R., Mazuela, P., & Olea, J. (2006). Maximum information stratification method for controlling item exposure in computerized adaptive testing. *Psicothema*, 18(1), 156-159. <https://www.psicothema.com/pdf/3191.pdf> (open access).

Chang, H.-H., Qian, J., & Ying, Z. (2001). A-stratified multistage computerized adaptive testing with b blocking. *Applied Psychological Measurement*, 25(4), 333-341. <https://doi.org/10.1177/01466210122032181>. Not openly available; Barrada et al. (2006) describe the method.

## API Reference

```{eval-rst}
.. autoclass:: catsim.selection.MaxInfoBBlockSelector
   :members:
   :show-inheritance:
```
