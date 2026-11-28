# Estimation Techniques

```{py:module} catsim.estimation
```

Ability estimation techniques compute the examinee's ability estimate
$\hat{\theta}$ after each response. The choice of estimator affects
accuracy, bias, and the availability of uncertainty quantification.

Every estimator returns a {py:class}`~catsim.Estimate`: the estimate,
its standard error by the estimator's own formula, which the precision stoppers
judge, and the number of objective evaluations it took. The session records one
per step.

| Estimator | Standard error |
|---|---|
| {py:class}`~catsim.estimation.NumericalSearchEstimator` | $1/\sqrt{I(\hat\theta)}$, the Fisher standard error {cite:p}`Magis12{Eq. 3, p. 4}` |
| {py:class}`~catsim.estimation.EAPEstimator` | the posterior standard deviation {cite:p}`Magis12{Eq. 11, p. 6}` |
| {py:class}`~catsim.estimation.MAPEstimator` | $1/\sqrt{I(\hat\theta) - (\log p)''(\hat\theta)}$, where $\log p$ is the prior's log-density; $1/\sqrt{I(\hat\theta) + 1/\sigma^2}$ for a normal prior of variance $\sigma^2$ {cite:p}`Magis12{Eq. 6, p. 5}` |
| {py:class}`~catsim.estimation.WarmLikelihoodEstimator` | $1/\sqrt{I(\hat\theta)}$, the Fisher standard error at the estimate |

The two Bayesian standard errors are usually below the Fisher one at the same
estimate, and with a normal prior MAP's always is, so a precision stopper ends
their tests sooner than the Fisher standard error would. On 200 simulated EAP
sessions, the Fisher standard error at the estimate was a median 26% above the
posterior standard deviation after 3 items and 4% above after 20, and with a 0.4
threshold the stopping decision after 20 items differed in 47 sessions.

(estimation-scale)=

## Ability scale and constant responses

Once an item has been administered, every estimator returns a finite estimate
inside its range, whatever the responses
({doc}`ADR-0018 </adr/0018-ability-scale-and-extreme-patterns>`). The range is
`bounds` for the estimators that search the scale, and the quadrature grid for
EAP; both span $[-6, 6]$ by default, six standard deviations either side of the
mean of a standard normal population. Narrower bounds restrict mixed responses
too: where the likelihood's maximum, or the posterior's mode, lies beyond a
bound, the estimate is that bound.

After all-correct or all-incorrect responses the maximum likelihood estimate is
infinite, and each estimator handles them its own way:

- {py:class}`~catsim.estimation.NumericalSearchEstimator` takes Dodd's step from
  the current estimate toward the bank's hardest or easiest item;
- {py:class}`~catsim.estimation.EAPEstimator` and
  {py:class}`~catsim.estimation.MAPEstimator` return the posterior mean and mode,
  which the prior draws toward its center; with
  irtlib's `uniform_log_prior`, the MAP estimate is the bound
  the responses point to, the maximum likelihood estimate within the bounds;
- {py:class}`~catsim.estimation.WarmLikelihoodEstimator` returns the root of
  Warm's modified score, which is finite for these responses, or the bound it
  points to.

The figure draws the likelihood of five responses to the same five items. Mixed
responses peak inside the scale, and their EAP estimate lies near the peak. Five
correct responses rise to the edge of the scale, with no peak, yet every
estimator puts that examinee at a finite ability.

```{eval-rst}
.. plot::
    :filename-prefix: estimation-likelihood
    :caption: The likelihood of two patterns of five responses, relative to its maximum. The dashed line in each curve's colour marks its EAP estimate; the black lines mark the other estimators' estimates after the five correct responses.

    import matplotlib.pyplot as plt
    from catsim import CatEngine, simulate
    from catsim.estimation import EAPEstimator, MAPEstimator, NumericalSearchEstimator, WarmLikelihoodEstimator
    from catsim.initialization import FixedPointInitializer
    from catsim.plot import likelihood
    from catsim.selection import LinearSelector
    from catsim.stopping import MaxLengthStopper
    from irtlib import ItemBank

    item_bank = ItemBank.generate(5, "2PL", rng=4)

    def session(responses, estimator):
        engine = CatEngine(item_bank, FixedPointInitializer(0.0), LinearSelector(range(5)), estimator, MaxLengthStopper(5))
        return simulate(engine, responses=[responses], rng=0).sessions[0]

    fig, ax = plt.subplots()
    likelihood(session([1, 0, 1, 1, 0], EAPEstimator()), item_bank, ax=ax, label="Mixed responses")
    likelihood(session([1] * 5, EAPEstimator()), item_bank, ax=ax, label="All correct")
    others = [
        ("MAP", MAPEstimator(), "-."),
        ("WLE", WarmLikelihoodEstimator(), ":"),
        ("Numerical search", NumericalSearchEstimator(), (0, (5, 2, 1, 2, 1, 2))),
    ]
    for name, estimator, style in others:
        ax.axvline(session([1] * 5, estimator).current_theta, color="black", linestyle=style, linewidth=1, label=name)
    ax.legend(loc="upper left")
    plt.show()
```

## Implemented

```{toctree}
:maxdepth: 1

numerical-search-estimator
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
