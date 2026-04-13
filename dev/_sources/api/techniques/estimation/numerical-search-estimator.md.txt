**Status:** Implemented
**Module:** {py:class}`catsim.estimation.NumericalSearchEstimator`
**Reference:** Birnbaum (1968); Lord (1980); Dodd (1990)

# Numerical Search Estimator (MLE)

## Motivation

The numerical-search estimator is the package's frequentist maximum-likelihood
ability estimator. Given administered items and observed binary responses, it
finds the ability value that maximizes the item-response-theory log-likelihood.

The class exposes several one-dimensional optimization methods under a single
contract, which makes it useful both as a practical estimator and as a reference
implementation for comparing search strategies. It also includes Dodd's
heuristic to handle degenerate all-correct and all-incorrect response patterns.

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

with search bounds given by the package's extended theta range. For mixed
response patterns, the implementation solves this optimization numerically using
either internal search procedures or SciPy's scalar minimization routines.

When every response is identical, the unconstrained MLE is not finite. In that
case the implementation returns Dodd's heuristic estimate if `dodd=True` and
otherwise returns $+\infty$ for all-correct or $-\infty$ for
all-incorrect patterns.

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
     - Numerical tolerance used by the search routine.
   * - ``dodd``
     - ``True``
     - Whether to use Dodd's heuristic when the response vector contains only one response type.
   * - ``verbose``
     - ``False``
     - Whether to print evaluation counts during optimization.
   * - ``method``
     - ``"bounded"``
     - Search method. Supported values are ``ternary``, ``dichotomous``,
       ``fibonacci``, ``golden``, ``golden2``, ``brent``, and ``bounded``.
```

## Behavioral Contracts

Any correct implementation must satisfy the following properties:

1. For mixed response patterns, it returns the theta value that numerically maximizes the log-likelihood under the configured search routine.
2. For all-correct and all-incorrect patterns, it uses Dodd's heuristic when `dodd=True` and signed infinity otherwise.
3. The estimator increments its call counter on every estimation request.
4. For SciPy-backed methods, evaluation counts reflect the optimizer's reported function evaluations.
5. The public set of available methods matches the estimator methods that can be selected at construction time.

## References

Birnbaum, A. (1968). Some latent trait models and their use in inferring an examinee's ability. In F. M. Lord & M. R. Novick (Eds.), *Statistical Theories of Mental Test Scores* (pp. 397-479). Reading, MA: Addison-Wesley.

Lord, F. M. (1980). *Applications of Item Response Theory to Practical Testing Problems*. Hillsdale, NJ: Lawrence Erlbaum Associates.

Dodd, B. G. (1990). The effect of item selection procedure and stepsize on computerized adaptive attitude measurement using the rating scale model. *Applied Psychological Measurement*, 14(4), 355-366. <https://doi.org/10.1177/014662169001400403>

Brent, R. P. (2002). *Algorithms for Minimization Without Derivatives*. Mineola, NY: Dover Publications.

## API Reference

```{eval-rst}
.. autoclass:: catsim.estimation.NumericalSearchEstimator
   :members:
   :show-inheritance:
```
