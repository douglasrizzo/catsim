# Estimation Techniques

Ability estimation techniques compute the examinee's latent trait estimate
$\hat{\theta}$ after each item response. The choice of estimator affects
accuracy, bias, and the availability of uncertainty quantification.

## Implemented

```{toctree}
:maxdepth: 1

numerical-search-estimator
bayesian-infrastructure
```

## Class Hierarchy

```{eval-rst}
.. inheritance-diagram:: catsim.estimation.BaseEstimator catsim.estimation.NumericalSearchEstimator
   :parts: 1
   :top-classes: catsim.estimation.BaseEstimator
```

## Planned

:::{warning}
The techniques in this section are not yet available in the released package.
They describe planned additions to {mod}`catsim.estimation`.
:::

```{toctree}
:maxdepth: 1

warm-wle
eap-estimator
map-estimator
owen-estimator
```
