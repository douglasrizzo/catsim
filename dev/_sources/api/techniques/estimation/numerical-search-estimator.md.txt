**Status:** Implemented
**Module:** {py:class}`catsim.estimation.NumericalSearchEstimator`
**Reference:** Birnbaum (1968); Lord (1980); Dodd (1990)

# Numerical Search Estimator (MLE)

## Motivation

The numerical-search estimator is the package's frequentist maximum-likelihood
ability estimator. Given administered items and observed binary responses, it
finds the ability value that maximizes the item-response-theory log-likelihood.

It searches the ability scale with a single bounded one-dimensional optimizer
and includes Dodd's heuristic to handle degenerate all-correct and all-incorrect
response patterns, so every estimate is finite and inside the search bounds.

## Definition

For administered items $i = 1, \dots, m$ and responses
$u_i \in \{0,1\}$, the estimator maximizes

$$
\ell(\theta)
=
\sum_{i=1}^{m}
\left[
  u_i \log P_i(\theta)
  +
  (1-u_i)\log(1-P_i(\theta))
\right].
$$

The returned estimate is

$$
\hat{\theta}_{\mathrm{MLE}}
=
\arg\max_{\theta \in [\theta_{\min}, \theta_{\max}]} \ell(\theta),
$$

with search bounds $[\theta_{\min}, \theta_{\max}]$ given by `bounds`, by default
the package's extended theta range $[-6, 6]$ (`THETA_MIN_EXTENDED`,
`THETA_MAX_EXTENDED`). For mixed
response patterns, the implementation minimizes $-\ell(\theta)$ with SciPy's
bounded Brent method (Brent, 2002), which never evaluates outside the bounds and
takes `tol` as its absolute tolerance on $\theta$ (SciPy's `xatol`). With
guessing, the likelihood can have several local maxima (Chang & Ying, 1996,
p. 223; Veerkamp & Berger, 1994, p. 4), and the search returns one of them. With
guessing or slipping parameters, the likelihood of a mixed pattern can also
still be rising at a bound, and the estimate is then within `tol` of that bound.

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

clipped to $[\theta_{\min}, \theta_{\max}]$. Dodd (1990, p. 359) introduced this
"variable stepsize" for the rating scale model: after a response in the upper
categories the estimate is incremented by half the distance to the highest scale
value in the item pool, and after one in the lower categories it is decreased by
half the distance to the lowest. The dichotomous form, with the bank's largest
and smallest $b$, is the one Barrada et al. (2008, p. 499) use. Dodd's estimates
start near the middle of the pool (p. 360), so the step always points the way
the responses do. catsim's estimates can start anywhere: beyond the bank's
hardest item, half the distance to it would move the estimate down after a
correct answer, so the $\max$ and $\min$ keep it where it is instead. Dodd also
studied fixed steps of 0.4 and 0.7, which catsim does not implement.

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

## Behavioral Contracts

Any correct implementation must satisfy the following properties:

1. For mixed response patterns, it returns a local maximum of the log-likelihood inside the search bounds, to within `tol`.
2. For all-correct and all-incorrect patterns, it returns Dodd's step, which never moves against the responses and lies inside the search bounds.
3. The estimator increments its call counter on every estimation request.
4. Evaluation counts are the optimizer's reported function evaluations.
5. A `tol` that is not positive is rejected at construction.
6. With no administered items, it returns `est_theta` unchanged.

## References

Birnbaum, A. (1968). Some latent trait models and their use in inferring an examinee's ability. In F. M. Lord & M. R. Novick (Eds.), *Statistical Theories of Mental Test Scores* (pp. 397-479). Reading, MA: Addison-Wesley.

Lord, F. M. (1980). *Applications of Item Response Theory to Practical Testing Problems*. Hillsdale, NJ: Lawrence Erlbaum Associates.

Dodd, B. G. (1990). The effect of item selection procedure and stepsize on computerized adaptive attitude measurement using the rating scale model. *Applied Psychological Measurement*, 14(4), 355-366. <https://doi.org/10.1177/014662169001400403>. Open copy: [University of Minnesota Digital Conservancy](https://hdl.handle.net/11299/113936).

Barrada, J. R., Olea, J., Ponsoda, V., & Abad, F. J. (2008). Incorporating randomness in the Fisher information for improving item-exposure control in CATs. *British Journal of Mathematical and Statistical Psychology*, 61(2), 493-513. <https://doi.org/10.1348/000711007X230937>. Open copy: [first author's page](https://personal.unizar.es/barrada/papers/08_BJMSP_Barrada.pdf).

Chang, H.-H., & Ying, Z. (1996). A global information approach to computerized adaptive testing. *Applied Psychological Measurement*, 20(3), 213-229. <https://doi.org/10.1177/014662169602000303>. Open copy: [University of Minnesota Digital Conservancy](https://hdl.handle.net/11299/119092).

Veerkamp, W. J. J., & Berger, M. P. F. (1994). *Some new item selection criteria for adaptive testing* (Research Report 94-6). Enschede: University of Twente. Open copy: [University of Twente](https://ris.utwente.nl/ws/files/5155670/RR_94-6.pdf).

Brent, R. P. (2002). *Algorithms for Minimization Without Derivatives*. Mineola, NY: Dover Publications. A reprint of the 1973 Prentice-Hall edition, a scan of which is on the [author's page](https://maths-people.anu.edu.au/~brent/pub/pub011.html).

## API Reference

```{eval-rst}
.. autoclass:: catsim.estimation.NumericalSearchEstimator
   :members:
   :show-inheritance:
```
