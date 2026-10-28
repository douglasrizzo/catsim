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
either internal search procedures or SciPy's scalar minimization routines. The
internal procedures and `bounded` stay within the bounds; `brent` and `golden`
only start from them as a bracket and can return a value outside. With guessing,
the likelihood can have several local maxima (Chang & Ying, 1996, p. 223;
Veerkamp & Berger, 1994, p. 4), and every method returns one of them.

When every response is identical, the unconstrained MLE is not finite. In that
case the implementation returns Dodd's heuristic estimate if `dodd=True` and
otherwise returns $+\infty$ for all-correct or $-\infty$ for
all-incorrect patterns. Dodd's heuristic moves the current estimate half the
distance to the largest item difficulty in the bank after a correct answer, or
to the smallest after an incorrect one. Dodd (1990, p. 359) introduced this
"variable stepsize" for the rating scale model, moving toward the highest or
lowest scale value in the item pool; the dichotomous form, with the bank's
largest and smallest $b$, is the one Barrada et al. (2008, p. 499) use. Dodd also
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
