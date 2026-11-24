**Status:** Implemented
**Module:** {py:class}`catsim.estimation.MAPEstimator`
**Reference:** Samejima (1969); Bock & Aitkin (1981)

# Maximum A Posteriori Estimator

## Motivation

MAP, also called the Bayes modal estimator, is the posterior mode: the
maximum-likelihood estimate with a prior added to the objective. Samejima
(1969, p. 7, Equations 2-5 and 2-6) defines the Bayes modal estimator as the
value that maximizes the prior density times the likelihood; Bock and Mislevy
(1982, p. 432) credit Bock and Aitkin (1981) with the name MAP, and Magis and
Raîche (2012, p. 4) attribute the estimator to Birnbaum (1969). The prior keeps
the estimate finite for all-correct and all-incorrect patterns, where the MLE
does not exist, while the computation stays a one-dimensional optimization like
the MLE's, with no quadrature grid.

It is not necessarily cheaper than EAP: Bock and Mislevy (1982, p. 432) argue
the opposite for adaptive testing, since EAP's quadrature sums accumulate item by
item while MAP needs a new optimization after every response.

## Definition

The MAP estimate is the posterior mode:

$$
\hat{\theta}_{\mathrm{MAP}}
=
\arg\max_{\theta}\,
p(\theta)\,L(\mathbf{u}\mid\theta)
=
\arg\max_{\theta}\,
\left[\log p(\theta)+\log L(\mathbf{u}\mid\theta)\right].
$$

The implementation minimizes the negative log-posterior

$$
-\log p(\theta) - \log L(\mathbf{u}\mid\theta)
$$

over the configured theta bounds. The estimator first evaluates the
log-posterior at 201 equally spaced points across the bounds (0.06 apart on the
default $[-6, 6]$), in one vectorized call, and takes the best point. A bounded
scalar optimizer (SciPy's bounded Brent method) then refines the mode between
that point's two neighbors, to within `tol`.

The grid search matters for two reasons. A prior that is zero on part of the
bounds, such as `uniform_log_prior(-1, 1)` searched over $[-6, 6]$, makes the
objective $+\infty$ there, and a bounded scalar search cannot cross that flat
region; it would drift to a bound outside the prior's support. And a multimodal
posterior, from a multimodal prior or a 3PL likelihood, has local modes a local
search can stop at; the grid finds the highest one, provided the posterior does
not change sharply between grid points. If three or more grid points tie for
the maximum, as with a uniform prior and no responses, the posterior has no
unique mode and the middle of the tied points is returned. The prior's support
must contain at least one grid point, which any interval wider than the grid
spacing does.

Under the 1PL and 2PL models the log-likelihood is concave, so with a normal
prior the derivative of the log-posterior points toward the prior mean at the
MLE and toward the MLE at the prior mean, and its single root lies between them
(contract 2). With a uniform prior over a wide enough interval, MAP coincides
with the MLE (Magis & Raîche, 2012, p. 5).

The estimator returns a {py:class}`~catsim.context.Estimate`. Its
standard error is

$$
SE(\hat{\theta}_{\mathrm{MAP}})
=
\frac{1}{\sqrt{I(\hat{\theta}) - (\log p)''(\hat{\theta})}},
$$

the test information plus the prior's negative curvature at the mode. The
curvature is a central difference, so any prior works. For a normal prior with
variance $\sigma^2$ the curvature is $-1/\sigma^2$, and this is the standard
error catR gives the Bayes modal estimator,
$1/\sqrt{I(\hat{\theta}) + 1/\sigma^2}$ (Magis & Raîche, 2012, p. 5, Equation
6). Inside a uniform prior's support the curvature is 0; at the support's edge
the difference is not finite and the prior adds nothing. Both give the Fisher
standard error, $1/\sqrt{I(\hat{\theta})}$. Where the sum is not positive, as
with a uniform prior and no responses, the standard error is infinite. The
evaluation count is the 201 grid points plus the refinement's posterior
evaluations, or 201 alone when tied grid points leave no refinement to run.

## Parameters

```{eval-rst}
.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Parameter
     - Default
     - Description
   * - ``log_prior``
     - ``normal_log_prior(0, 1)``
     - Prior used to regularize the likelihood objective.
   * - ``tol``
     - ``1e-6``
     - Absolute tolerance of the refinement step, passed to the bounded scalar optimizer.
   * - ``bounds``
     - ``(THETA_MIN_EXTENDED, THETA_MAX_EXTENDED)``
     - Search range for the penalized objective.
```

## Behavioral Contracts

Any correct implementation must satisfy the following properties:

1. With no administered items, the estimate is the mode of the configured prior over the search range, to within `tol`.
2. Under the 1PL and 2PL models with a normal prior, the MAP estimate lies between the prior mean and the MLE, so it is closer to the prior mean than the MLE.
3. Shifting the prior mean while holding the response data fixed shifts the MAP estimate in the same direction.
4. On long, information-rich tests, MAP and EAP should converge toward similar values because the posterior becomes sharply concentrated.
5. The estimator returns finite values on extreme response patterns because the prior keeps the objective bounded.
6. The estimate lies inside the prior's support. A prior whose support contains none of the grid points, such as one outside the search bounds, raises `ValueError`.
7. The standard error is $1/\sqrt{I(\hat{\theta}) + 1/\sigma^2}$ with a normal prior of variance $\sigma^2$, the Fisher standard error with a uniform prior, and `math.inf` where the information plus the prior's negative curvature is not positive.
8. The evaluation count is the number of grid points plus the refinement's posterior evaluations.

## References

Samejima, F. (1969). Estimation of latent ability using a response pattern of graded scores. *Psychometrika Monograph Supplement*, No. 17. <https://doi.org/10.1007/BF03372160>. Open copy: [Psychometric Society](https://psychometricsociety.org/sites/main/files/file-attachments/mn17.pdf).

Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of item parameters: Application of an EM algorithm. *Psychometrika*, 46(4), 443-459. <https://doi.org/10.1007/BF02293801>. Not openly available.

Bock, R. D., & Mislevy, R. J. (1982). Adaptive EAP estimation of ability in a microcomputer environment. *Applied Psychological Measurement*, 6(4), 431-444. <https://doi.org/10.1177/014662168200600405>. Open copy: [University of Minnesota Digital Conservancy](https://hdl.handle.net/11299/101546).

Birnbaum, A. (1969). Statistical theory for logistic mental test models with a prior distribution of ability. *Journal of Mathematical Psychology*, 6(2), 258-276. <https://doi.org/10.1016/0022-2496(69)90005-4>. Not openly available.

Magis, D., & Raîche, G. (2012). Random generation of response patterns under computerized adaptive testing with the R package catR. *Journal of Statistical Software*, 48(8), 1-31. <https://doi.org/10.18637/jss.v048.i08> (open access).

:::{seealso}
{doc}`/api/techniques/estimation/bayesian-infrastructure`,
{doc}`/api/techniques/estimation/eap-estimator`
:::

## API Reference

```{eval-rst}
.. autoclass:: catsim.estimation.MAPEstimator
   :members:
   :show-inheritance:
```
