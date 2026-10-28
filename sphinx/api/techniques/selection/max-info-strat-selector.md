**Status:** Implemented
**Module:** {py:class}`catsim.selection.MaxInfoStratSelector`
**Reference:** Barrada et al. (2006)

# Maximum-Information Stratified Selector

## Motivation

The a-stratified method ignores the pseudo-guessing parameter, both when
stratifying and when selecting. Under the 3PL and 4PL models an item's
discrimination no longer ranks its maximum information, and its information no
longer peaks at $b$. Maximum-information stratification (MIS) replaces $a$ by
the item's maximum information and $b$ by the ability at which that maximum is
reached, which improved precision over the a-stratified method in Barrada et
al. (2006, pp. 158–159).

## Definition

For each item $i$, let

$$
\theta_i^{\max} = \arg\max_{\theta} I_i(\theta),
\qquad
I_i^{\max} = I_i(\theta_i^{\max}).
$$

Items are sorted by increasing $I_i^{\max}$ and split into `n_strata`
contiguous strata. At test position $t$ the selector returns the
non-administered item of stratum $s(t)$ (see {doc}`stratified-selector`) whose
point of maximum information is closest to the current estimate:

$$
i^* = \arg\min_{i \in \mathcal{S}_{s(t)} \cap R_t} \left| \theta_i^{\max} - \hat{\theta} \right|.
$$

Under the 1PL and 2PL models $\theta_i^{\max} = b_i$ and $I_i^{\max}$ ranks
items like $a_i$, so MIS coincides with the a-stratified method.

Barrada et al. (2006, p. 157, Equations 1 and 2) give $I_i^{\max}$ and
$\theta_i^{\max}$ in closed form for the 3PL model, with the scaling constant
$D = 1.7$; catsim's IRT functions use no scaling constant.

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

1. Strata are contiguous groups of increasing maximum information.
2. The returned item minimizes $|\theta_i^{\max} - \hat{\theta}|$ among the non-administered items of the active stratum.
3. It never returns an already administered item.
4. If the active stratum has been exhausted, it raises `NoItemsAvailableError`.

## References

Barrada, J. R., Mazuela, P., & Olea, J. (2006). Maximum information stratification method for controlling item exposure in computerized adaptive testing. *Psicothema*, 18(1), 156-159. <https://www.psicothema.com/pdf/3191.pdf> (open access).

## API Reference

```{eval-rst}
.. autoclass:: catsim.selection.MaxInfoStratSelector
   :members:
   :show-inheritance:
```
