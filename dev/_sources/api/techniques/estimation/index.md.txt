# Estimation Techniques

Ability estimation techniques compute the examinee's latent trait estimate
$\hat{\theta}$ after each item response. The choice of estimator affects
accuracy, bias, and the availability of uncertainty quantification.

Every estimator returns a {py:class}`~catsim.context.Estimate`: the estimate,
its standard error by the estimator's own formula, which the precision stoppers
judge, and the number of objective evaluations it took. The session records one
per step.

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
