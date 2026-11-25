# Estimation Techniques

```{py:module} catsim.estimation
```

Ability estimation techniques compute the examinee's latent trait estimate
$\hat{\theta}$ after each item response. The choice of estimator affects
accuracy, bias, and the availability of uncertainty quantification.

Every estimator returns a {py:class}`~catsim.Estimate`: the estimate,
its standard error by the estimator's own formula, which the precision stoppers
judge, and the number of objective evaluations it took. The session records one
per step.

| Estimator | Standard error |
|---|---|
| {py:class}`~catsim.estimation.NumericalSearchEstimator` | $1/\sqrt{I(\hat\theta)}$, the Fisher standard error {cite:p}`Magis12{Eq. 3, p. 4}` |
| {py:class}`~catsim.estimation.EAPEstimator` | the posterior standard deviation (Eq. 11, p. 6) |
| {py:class}`~catsim.estimation.MAPEstimator` | $1/\sqrt{I(\hat\theta) - (\log p)''(\hat\theta)}$, where $\log p$ is the prior's log-density; $1/\sqrt{I(\hat\theta) + 1/\sigma^2}$ for a normal prior of variance $\sigma^2$ (Eq. 6, p. 5) |
| {py:class}`~catsim.estimation.WarmLikelihoodEstimator` | $1/\sqrt{I(\hat\theta)}$, catR's default |

The two Bayesian standard errors are usually below the Fisher one at the same
estimate, and with a normal prior MAP's always is, so a precision stopper ends
their tests sooner than the Fisher standard error would. On 200 simulated EAP
sessions, the Fisher standard error at the estimate was a median 26% above the
posterior standard deviation after 3 items and 4% above after 20, and with a 0.4
threshold the stopping decision after 20 items differed in 47 sessions.

## Implemented

```{toctree}
:maxdepth: 1

numerical-search-estimator
bayesian-infrastructure
eap-estimator
map-estimator
warm-wle
```

## Class Hierarchy

```{autoclasstree} catsim.estimation
:full:
:strict:
:namespace: catsim
```

## References

```{bibliography}
:filter: docname in docnames
```

## Base Class

```{eval-rst}
.. autoclass:: catsim.estimation.BaseEstimator
   :members:
```
