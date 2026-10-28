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

with SciPy's bounded Brent method over the configured bounds. With guessing, the
posterior can have several modes; the search returns one of them.

Under the 1PL and 2PL models the log-likelihood is concave, so with a normal
prior the derivative of the log-posterior points toward the prior mean at the
MLE and toward the MLE at the prior mean, and its single root lies between them
(contract 2). With a uniform prior over a wide enough interval, MAP coincides
with the MLE (Magis & Raîche, 2012, p. 5).

With no administered items, the estimate is the prior's mode, found by evaluating
the prior on 201 equally spaced points across the bounds. On the default bounds
$[-6, 6]$ that is within 0.03 of the exact mode.

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
     - Absolute tolerance of the bounded scalar optimizer.
   * - ``bounds``
     - ``(THETA_MIN_EXTENDED, THETA_MAX_EXTENDED)``
     - Search range for the penalized objective.
   * - ``verbose``
     - ``False``
     - Diagnostics flag inherited from the estimator base class.
```

## Behavioral Contracts

Any correct implementation must satisfy the following properties:

1. With no administered items, the estimate defaults to the mode of the configured prior over the search range.
2. Under the 1PL and 2PL models with a normal prior, the MAP estimate lies between the prior mean and the MLE, so it is closer to the prior mean than the MLE.
3. Shifting the prior mean while holding the response data fixed shifts the MAP estimate in the same direction.
4. On long, information-rich tests, MAP and EAP should converge toward similar values because the posterior becomes sharply concentrated.
5. The estimator returns finite values on extreme response patterns because the prior keeps the objective bounded.

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
