# Numerical Search Estimator (MLE)

:Class: {py:class}`catsim.estimation.NumericalSearchEstimator`
:Sources: {cite:t}`Birnbaum1968`; {cite:t}`Lord1980`; {cite:t}`Dod90`

## Motivation

The numerical-search estimator is the package's frequentist maximum-likelihood
ability estimator. Given administered items and observed binary responses, it
finds the ability value that maximizes the item-response-theory log-likelihood.

It searches the ability scale with a single bounded one-dimensional optimizer
and includes Dodd's heuristic to handle degenerate all-correct and all-incorrect
response patterns, so every estimate is finite and inside the search bounds.

## Definition

The estimator maximizes the log-likelihood $\log L(\theta)$ of the responses
$u_i \in \{0,1\}$ to the administered items, which
{py:func}`catsim.irt.log_likelihood` computes. The returned estimate is

$$
\hat{\theta}_{\mathrm{MLE}}
=
\arg\max_{\theta \in [\theta_{\min}, \theta_{\max}]} \log L(\theta),
$$

with search bounds $[\theta_{\min}, \theta_{\max}]$ given by `bounds`, by default
the package's extended theta range $[-6, 6]$ (`THETA_MIN_EXTENDED`,
`THETA_MAX_EXTENDED`).

When every response is identical, the unconstrained MLE is not finite. In that
case the implementation takes Dodd's heuristic step from the current estimate
$\hat{\theta}_t$: half the distance to the largest item difficulty in the bank
after correct answers, or to the smallest after incorrect ones,

$$
\hat{\theta}_{t+1} =
\begin{cases}
\max\left(\hat{\theta}_t,\ \hat{\theta}_t + \dfrac{b_{\max} - \hat{\theta}_t}{2}\right)
  & \text{if every response is correct,} \\[1ex]
\min\left(\hat{\theta}_t,\ \hat{\theta}_t - \dfrac{\hat{\theta}_t - b_{\min}}{2}\right)
  & \text{if every response is incorrect,}
\end{cases}
$$

clipped to $[\theta_{\min}, \theta_{\max}]$. {cite:t}`Dod90{p. 359}` introduced this
"variable stepsize" for the rating scale model: after a response in the upper
categories the estimate is incremented by half the distance to the highest scale
value in the item pool, and after one in the lower categories it is decreased by
half the distance to the lowest. The dichotomous form, with the bank's largest
and smallest $b$, is the one {cite:t}`Barrada2008{p. 499}` use. Dodd's estimates
start near the middle of the pool (p. 360), so the step always points the way
the responses do. catsim's estimates can start anywhere: beyond the bank's
hardest item, half the distance to it would move the estimate down after a
correct answer, so the $\max$ and $\min$ keep it where it is instead. Dodd also
studied fixed steps of 0.4 and 0.7, which catsim does not implement.

The estimator returns a {py:class}`~catsim.Estimate`. Its standard error
is the asymptotic standard error of the maximum likelihood estimate
{cite:p}`Magis12{p. 4, Equation 3}`,

$$
SE(\hat{\theta}) = \frac{1}{\sqrt{I(\hat{\theta})}},
$$

where $I$ is the test information of the administered items, and it is infinite
where $I(\hat{\theta}) = 0$. After Dodd's step it is the same formula at the
stepped estimate.

### Computation

For mixed response patterns, the implementation minimizes $-\log L(\theta)$ with
SciPy's bounded Brent method, {py:func}`scipy.optimize.minimize_scalar`
{cite:p}`Brent2002`, which never evaluates outside the bounds and takes `tol` as its absolute
tolerance on $\theta$ (SciPy's `xatol`).

## Parameters

```{eval-rst}
.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``tol``
     - ``1e-6``
     - Absolute tolerance of the search, in the ability scale. Must be positive.
   * - ``bounds``
     - ``(THETA_MIN_EXTENDED, THETA_MAX_EXTENDED)``
     - Range of the estimate: the search interval, and the interval Dodd's step is clipped to.
       Must be finite, with the lower bound below the upper one.
```

## Behavior and Comparison

With guessing, the likelihood can have several local maxima {cite:p}`Chang1996{p. 223}`;
{cite:p}`Veerkamp1994{p. 4}`, and the search returns one of them. With
guessing or slipping parameters, the likelihood of a mixed pattern can also
still be rising at a bound, and the estimate is then within `tol` of that bound.

## Guarantees

1. For mixed response patterns, the estimate is a local maximum of the log-likelihood inside `bounds`, to within `tol`.
2. For all-correct and all-incorrect patterns, the estimate is Dodd's step: finite, inside `bounds`, and never moved against the responses.
3. The standard error is the asymptotic one above, at the returned estimate, and `math.inf` where the test information is 0.

## References

```{bibliography}
:filter: docname in docnames
```

## API Reference

```{eval-rst}
.. autoclass:: catsim.estimation.NumericalSearchEstimator
   :members:
   :show-inheritance:
```
