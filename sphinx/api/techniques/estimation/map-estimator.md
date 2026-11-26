# Maximum A Posteriori Estimator

:Class: {py:class}`catsim.estimation.MAPEstimator`
:Sources: {cite:t}`Samejima1969`; {cite:t}`Bock1981`

## Motivation

MAP, also called the Bayes modal estimator, is the posterior mode: the
maximum-likelihood estimate with a prior added to the objective.
{cite:t}`Samejima1969{p. 7, Equations 2-5 and 2-6}` defines the Bayes modal
estimator as the value that maximizes the prior density times the likelihood;
{cite:t}`Bock1982{p. 432}` credit {cite:t}`Bock1981` with the name MAP, and
{cite:t}`Magis12{p. 4}` attribute the estimator to {cite:t}`Birnbaum1969`. The prior keeps
the estimate finite for all-correct and all-incorrect patterns, where the MLE
does not exist, while the computation stays a one-dimensional optimization like
the MLE's, with no quadrature grid.

It is not necessarily cheaper than EAP: {cite:t}`Bock1982{p. 432}` argue
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

The estimator returns a {py:class}`~catsim.Estimate`. Its
standard error is

$$
SE(\hat{\theta}_{\mathrm{MAP}})
=
\frac{1}{\sqrt{I(\hat{\theta}) - (\log p)''(\hat{\theta})}},
$$

the test information plus the prior's negative curvature at the mode. The
curvature is a central difference, so any prior works. For a normal prior with
variance $\sigma^2$ the curvature is $-1/\sigma^2$, which gives the Bayes modal
standard error $1/\sqrt{I(\hat{\theta}) + 1/\sigma^2}$ {cite:p}`Magis12{p. 5, Equation 6}`.
Inside a uniform prior's support the curvature is 0; at the support's edge
the difference is not finite and the prior adds nothing. Both give the Fisher
standard error, $1/\sqrt{I(\hat{\theta})}$. Where the sum is not positive, as
with a uniform prior and no responses, the standard error is infinite.

### Computation

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

## Behavior and Comparison

Under the 1PL and 2PL models the log-likelihood is concave, so with a normal
prior the derivative of the log-posterior points toward the prior mean at the
MLE and toward the MLE at the prior mean, and its single root lies between them
(the second guarantee below). With a uniform prior over a wide enough interval,
MAP coincides with the MLE {cite:p}`Magis12{p. 5}`.

On long, information-rich tests, MAP and EAP converge toward similar values
because the posterior becomes sharply concentrated.

:::{dropdown} Comparison with catR

The estimator is catR's Bayes modal estimator, `thetaEst` with `method = "BM"`,
and its standard error is that of catR's `semTheta`. With `bounds=(-4, 4)`,
catR's default range, and the matching prior, the parity tests
({doc}`ADR-0020 </adr/0020-catr-as-the-reference>`) find the same estimates and
standard errors on fixed responses, all-correct and all-incorrect ones
included, under a standard normal prior, a normal prior of mean 0.5 and standard
deviation 1.5 and a uniform prior on $[-3, 3]$, and in whole tests on recorded
responses under MFI, bOpt and MLWI selection. They find no departure. catR's
Jeffreys prior has no `log_prior` counterpart, since it depends on the
administered items.
:::

## Guarantees

1. The estimate is finite for all-correct and all-incorrect responses, because the prior keeps the objective bounded.
2. Under the 1PL and 2PL models with a normal prior, the MAP estimate lies between the prior mean and the MLE, so it is closer to the prior mean than the MLE.
3. Shifting the prior mean while holding the response data fixed shifts the MAP estimate in the same direction.
4. The estimate lies inside the prior's support.
5. The standard error is $1/\sqrt{I(\hat{\theta}) + 1/\sigma^2}$ with a normal prior of variance $\sigma^2$, the Fisher standard error with a uniform prior, and `math.inf` where the information plus the prior's negative curvature is not positive.
6. With no administered items, the estimate is the mode of the prior over the search range, to within `tol`.

## References

```{bibliography}
:filter: docname in docnames
```

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
