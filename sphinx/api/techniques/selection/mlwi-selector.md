**Status:** Implemented
**Module:** {py:class}`catsim.selection.MLWISelector`
**Reference:** Veerkamp & Berger (1997)

# Maximum Likelihood Weighted Information Selector

## Motivation

MLWI softens local maximum-information selection by weighting each item's
information curve by the likelihood of ability values given the observed
responses. This makes it a frequentist analogue of posterior-weighted selectors:
it spreads attention across plausible ability values without introducing a prior.

That positioning is useful strategically. It gives catsim a weighted-information
selector that is conceptually close to later Bayesian variants while staying
independent of the Bayesian infrastructure.

## Definition

For candidate item $i$, MLWI is defined as

$$
\mathrm{MLWI}_i
=
\int I_i(\theta)\,L(\mathbf{u}\mid\theta)\,d\theta.
$$

The implementation approximates the integral on a fixed grid
$\{\theta_q\}_{q=1}^{Q}$ with spacing $\Delta_q$:

$$
\mathrm{MLWI}_i
\approx
\sum_{q=1}^{Q} I_i(\theta_q)\,L(\mathbf{u}\mid\theta_q)\,\Delta_q.
$$

Likelihood values are normalized numerically with a log-sum-exp style shift
before exponentiation; the constant factor cancels under $\arg\max$.

## Parameters

```{eval-rst}
.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Parameter
     - Default
     - Description
   * - ``n_nodes``
     - ``41``
     - Number of integration nodes used in the fixed-grid approximation; at least 5.
   * - ``bounds``
     - ``(THETA_MIN_EXTENDED, THETA_MAX_EXTENDED)``
     - Theta range covered by the fixed integration grid.
   * - ``r_max``
     - ``1.0``
     - Exposure cap applied before the final maximization.
```

## Behavioral Contracts

Any correct implementation must satisfy the following properties:

1. With an empty response vector, the likelihood weights reduce to a constant and the selector therefore reduces to choosing the largest integral of item information over the configured bounds.
2. On long tests with sharply peaked likelihoods, MLWI approaches maximum-information selection evaluated near the MLE.
3. The likelihood weighting must be computed in a numerically stable way that avoids underflow on long response vectors.
4. The selector requires the response vector; the current point estimate alone is insufficient to define the likelihood weights.
5. Exposure filtering is applied before the final argmax, with fallback to the unconstrained non-administered pool if every constrained candidate is excluded.

## References

Veerkamp, W. J. J., & Berger, M. P. F. (1997). Some new item selection criteria for adaptive testing. *Journal of Educational and Behavioral Statistics*, 22(2), 203-226. <https://doi.org/10.3102/10769986022002203>

:::{seealso}
Sister spec: {doc}`/api/techniques/selection/mei-selector`
:::

## Reading the Source

**Checked against** the 1994 research report that preceded the journal
article, openly available from the
[University of Twente](https://ris.utwente.nl/ws/files/5155670/RR_94-6.pdf)
(Veerkamp & Berger, *Some new item selection criteria for adaptive testing*,
Research Report 94-6). The 1997 journal version was not checked. Page numbers
are the report's own.

- **p. 7, Equations 7 and 8:** the general weighted information criterion,
  $\max_i \int W_n(\theta)\, I_i(\theta)\, d\theta$, of which the other criteria
  are special cases.
- **pp. 11–12, Equations 17 and 18:** the likelihood weighted information
  criterion uses the likelihood as the weight, $W_n = L_n(\theta; \mathbf{x}_n)$,
  and selects the item maximizing $\int L_n(\theta; \mathbf{x}_n)\, I_i(\theta)\, d\theta$.
  The authors note that it needs no provisional ability estimate.
- **p. 15, Equation 20:** in their simulation the integral is approximated by a
  sum over a grid, as catsim does.
- **pp. 17–18:** in their simulation, the criterion did better than maximum
  information at the ML estimate.

The criterion also matches Table 1 of Magis and Raîche (2012, p. 9).

**Checked by** `TestMLWISelector` in `tests/test_selection.py`, including
`test_select_matches_manual_weighted_information` and
`test_select_with_empty_response_vector_reduces_to_information_integral`.

## API Reference

```{eval-rst}
.. autoclass:: catsim.selection.MLWISelector
   :members:
   :show-inheritance:
```
