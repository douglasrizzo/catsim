# Introduction

{py:mod}`catsim` is a Python package for computerized adaptive testing (CAT). It computes the
steps of an adaptive test, from the choice of each item to the decision to stop, and simulates
tests on examinees whose abilities are known, to compare testing methods. An application that
gives adaptive tests can use those computations as its core. This page introduces CAT and the item
response theory behind it, and ends with a map of catsim's parts. {doc}`terms-and-notation` lists
the terms and symbols these pages use.

## What a Test Measures

A test measures an examinee's *ability*: a characteristic that cannot be observed directly, such as
knowledge of a subject or a personality trait. Educational and psychological tests are common
examples. What can be observed are the examinee's answers to the test's items, and the ability is
inferred from them as a number on a continuous scale.

## The Problem with Linear Tests

Traditional *linear tests* use a paper-and-pencil approach where all examinees answer the same items in the same order. This creates several problems:

- High-ability examinees waste time on easy questions
- Low-ability examinees become frustrated with overly difficult questions
- Test length cannot be optimized for individual examinees

A computerized adaptive test avoids them by choosing each examinee's items during the test. It rests
on item response theory, the subject of the next section.

## Item Response Theory Models

{py:mod}`catsim` uses item response theory {cite:p}`Lord68,Rasch66`, a family of models developed
in the second half of the 20th century to measure abilities and to compare examinees on a common
ability scale, which also justifies its use in adaptive testing. {py:mod}`catsim` implements its
one-, two-, three- and four-parameter logistic models, in which examinees and items are represented
by numbers, the models' parameters.

The logistic models are unidimensional: a test measures a single ability. The test is made of
*items*, the questions an examinee answers, each scored correct or incorrect.

An examinee's ability is written $\theta$. It can take any real value, and its scale is set by
whoever calibrates the test. It is common to set it so that the abilities of the population follow
the standard normal distribution $N(0, 1)$, under which about 99.99% of examinees lie between $-4$
and $4$. The ability cannot be observed, so it is estimated from the responses: the *ability
estimate* $\hat{\theta}$, which tends to get closer to the true $\theta$ as the test gets longer.
By default, {py:mod}`catsim`'s estimators keep their estimates within $[-6, 6]$, wider than any plausible
examinee or item bank ({doc}`/adr/0018-ability-scale-and-extreme-patterns`).

Under the logistic models, item $i$ has up to four parameters:

- its *discrimination* $a_i$: how sharply the item separates examinees of lower and higher ability.
  Examinees whose $\theta$ is above the item's difficulty tend to answer a highly discriminating item
  correctly, and the others incorrectly; as $a_i$ gets lower, this threshold blurs and the item
  gives less information. It is common for $a_i > 0$.
- its *difficulty* $b_i$: the point of the ability scale at which the item's characteristic curve is
  steepest, where it separates examinees best. The item is most informative at or near it (exactly at
  it when $c_i = 0$ and $d_i = 1$). It is on the same scale as $\theta$, so difficulties and abilities
  can be compared directly. A CAT needs items at every difficulty, so that it can give each
  examinee the items that suit their ability.
- its *guessing parameter* $c_i$, the lower asymptote of its characteristic curve: the probability
  that an examinee of very low ability still answers the item correctly. It lies between 0 and the
  upper asymptote, $0 \leq c_i < d_i$, and the lower it is, the better the item.
- its *upper asymptote* $d_i$: the probability that an examinee of very high ability answers the
  item correctly. $1 - d_i$, the *slipping parameter*, is the probability that they still answer it
  incorrectly. It lies between the guessing parameter and 1, $c_i < d_i \leq 1$, and the higher it
  is, the better the item.

When every item has $d_i = 1$, the four-parameter logistic model reduces to the three-parameter
one. When, in addition, every $c_i = 0$, it reduces to the two-parameter model, and when every
$a_i$ is also equal, to the one-parameter model. The one-parameter model with every $a_i = 1$ is
the Rasch model {cite:p}`Rasch66`. The formulas below hold for all of them, given the item
parameters, so {py:mod}`catsim` treats the four models alike.

Under IRT, the probability that an examinee of ability $\theta$ answers item $i$ correctly is the
item's *characteristic curve* {cite:p}`Ayala2009,Magis13`

$$
P_i(\theta) = c_i + \frac{d_i - c_i}{1 + e^{-a_i(\theta - b_i)}}.
$$

The *information* the item gives at ability $\theta$, how much a response to it sharpens an
estimate near $\theta$, is {cite:p}`Ayala2009,Magis13`

$$
I_i(\theta) = \frac{a_i^2 [P_i(\theta) - c_i]^2 [d_i - P_i(\theta)]^2}{(d_i - c_i)^2 P_i(\theta) Q_i(\theta)},
$$

where $Q_i(\theta) = 1 - P_i(\theta)$ is the probability of an incorrect response. The following
figure draws both curves for one item: the item is most informative near its difficulty.

```{eval-rst}
.. plot::
    :filename-prefix: introduction-item-curves
    :caption: Item characteristic and information curves for an item. The dot represents the point at which the item is most informative.

    import matplotlib.pyplot as plt
    from catsim.plot import PlotType, item_curve
    from irtlib import ItemBank

    item_bank = ItemBank.generate(1, rng=0)
    item_curve(0, item_bank, kind=PlotType.BOTH)
    plt.show()
```

The sum of the information of the items administered to an examinee is the *test information*
{cite:p}`Ayala2009`:

$$
I(\theta) = \sum_i I_i(\theta).
$$

The uncertainty of an ability estimate is its *standard error* $SE(\hat{\theta})$. For a maximum
likelihood estimate it is {cite:p}`Ayala2009`

$$
SE(\hat{\theta}) = \frac{1}{\sqrt{I(\hat{\theta})}}.
$$

Every item adds information, so the more items an examinee answers, the smaller the standard
error. These quantities come from [irtlib](https://github.com/douglasrizzo/irtlib), the item
response theory package catsim builds on, in its `irtlib.irt` module.

### The Item Matrix

In {py:mod}`catsim` and irtlib, the items are a {py:class}`numpy.ndarray` with one row per item and
one column per parameter, the *item matrix*. Its four columns hold $a$, $b$, $c$ and $d$, in this
order. A matrix with one, two or three columns is read as $[b]$, $[a, b]$ or $[a, b, c]$, and the missing
parameters take the values $a = 1$, $c = 0$ and $d = 1$. An item bank, irtlib's `ItemBank`, holds
an item matrix, and `ItemBank.generate` draws one at random:

```pycon
>>> from irtlib import ItemBank
>>> ItemBank.generate(5, "1PL")
ItemBank(n_items=5, model=1PL)
>>> ItemBank.generate(5, "2PL")
ItemBank(n_items=5, model=2PL)
>>> ItemBank.generate(5, "3PL")
ItemBank(n_items=5, model=3PL)
>>> ItemBank.generate(5, "4PL")
ItemBank(n_items=5, model=4PL)
>>> ItemBank.generate(5, "3PL", corr=0.5).items.shape
(5, 4)
```

These examples draw five items under each logistic model. Whatever the model, the item matrix has
four columns. In the last example, the discrimination and the difficulty have a correlation of
$0.5$, which simulations may need {cite:p}`Chang2001`. Each item also has a name, so that results
can say which item is which; a calibrated bank and its names can be read from a CSV file
(`ItemBank.from_csv`).

## Computerized Adaptive Tests

Computerized adaptive testing emerged in the 1970s as *tailored testing* {cite:p}`Lord77`. A linear
test estimates an examinee's ability once, at the end. A computerized adaptive test (CAT) updates the
estimate after each response. Knowing more about the ability at each step lets the test
choose more informative items *during* the test, so the standard error of the estimate falls
faster. The figure compares the two on one simulated population: as items accumulate, the adaptive
test's estimates come closer to the true abilities than the linear test's.

```{eval-rst}
.. plot::
    :filename-prefix: introduction-adaptive-and-linear
    :caption: The RMSE of 300 simulated examinees' estimates after each item, from one 200-item bank. The adaptive test chooses each examinee's items by maximum information; the linear test gives everyone the 30 items most informative at the population's mean.

    import matplotlib.pyplot as plt
    import numpy
    from catsim import CatEngine, simulate
    from catsim.estimation import EAPEstimator
    from catsim.initialization import FixedPointInitializer
    from catsim.plot import metric_by_length
    from catsim.selection import LinearSelector, MaxInfoSelector
    from catsim.stopping import MaxLengthStopper
    from irtlib import ItemBank

    item_bank = ItemBank.generate(200, rng=0)
    true_thetas = numpy.random.default_rng(1).normal(0, 1, 300)
    linear_items = numpy.argsort(-item_bank.information(0.0))[:30].tolist()
    fig, ax = plt.subplots()
    for label, selector in [("Adaptive", MaxInfoSelector()), ("Linear", LinearSelector(linear_items))]:
        engine = CatEngine(item_bank, FixedPointInitializer(0.0), selector, EAPEstimator(), MaxLengthStopper(30))
        metric_by_length(simulate(engine, true_thetas=true_thetas, rng=3), "rmse", ax=ax, label=label)
    ax.legend()
    plt.show()
```

### The CAT Lifecycle

A computerized adaptive test follows a well-defined lifecycle:

```{mermaid}
flowchart TD
    start([Start]) --> init[Initial ability estimate]
    init --> select[Item selection and administration]
    select --> answer[Response]
    answer --> estimate[Ability estimation]
    estimate --> stop{Stopping criterion<br>reached?}
    stop -- No --> select
    stop -- Yes --> finish([End])
```

1. The examinee's initial ability estimate is set;
2. an item is selected at the current ability estimate and administered;
3. the ability is estimated again from the responses to all items so far;
4. **if** a stopping criterion is met, the test ends; **else** it goes back to step 2.

The literature proposes many methods for each of these four phases. In {py:mod}`catsim`, each
phase is a strategy with a module of its own, an *initializer*, a *selector*, an *estimator* and a
*stopper*, so a test can combine any method for each phase. The sections below introduce each
phase.

#### Initialization

The initializer runs once per examinee and sets the initial ability estimate $\hat\theta_0$. It may
give every examinee the same value ({py:class}`catsim.initialization.FixedPointInitializer`), or draw
one from a uniform or a normal distribution ({py:class}`catsim.initialization.UniformInitializer`,
{py:class}`catsim.initialization.NormalInitializer`). The literature also starts at other points,
such as the ability at which the item bank is most informative.

In {py:mod}`catsim`, initializers are in the {py:mod}`catsim.initialization` module.

#### Item Selection

At the current estimate $\hat{\theta}$, the selector chooses an item from the item bank and the
examinee answers it. The best-known method gives the item that is most informative at
$\hat{\theta}$ ({py:class}`catsim.selection.MaxInfoSelector`). It tends to give the same few items
to many examinees and to leave the rest of the bank unused, so other selection methods were
proposed.

How often an item is given is measured by its *exposure rate*, the share of examinees given it:

$$
r_i = \frac{q_i}{N},
$$

where $q_i$ is the number of examinees given item $i$ and $N$ the number of examinees. An item given
to many examinees becomes known to future ones. Exposure-control methods keep the rates below a cap
$r_{\max}$ or choose among the best items at random
({doc}`selection techniques </api/techniques/selection/index>`). A simulation reports the rates with
{py:meth}`~catsim.SimulationResult.exposure_rates`.

In {py:mod}`catsim`, selectors are in the {py:mod}`catsim.selection` module.

#### Ability Estimation

After each response, the estimator updates the ability estimate $\hat{\theta}$ from the responses
so far, each correct or incorrect, and the parameters of the items answered. In {py:mod}`catsim`,
the responses are a sequence with one entry for each administered item, in the order they were
given: `True` or 1 for a correct response, `False` or 0 for an incorrect one.

```pycon
>>> responses = [True, True, False, True, False]
```

Estimators fall into two groups. Maximum likelihood estimators return the ability at which the
responses are most likely, the maximum of the *log-likelihood* (`irtlib.irt.log_likelihood`).
Bayesian estimators combine the likelihood with a prior distribution of the abilities in the
population. Each estimator also gives the standard error of its estimate, by its own formula
({doc}`estimation techniques </api/techniques/estimation/index>`).

In {py:mod}`catsim`, estimators are in the {py:mod}`catsim.estimation` module.

#### Stopping Criterion

Since a CAT selects its items during the test, a stopping criterion decides when the test ends and
no further items are given. A stopper may end the test at a fixed number of items
({py:class}`catsim.stopping.MaxLengthStopper`), once the standard error of the estimate falls to a
threshold ({py:class}`catsim.stopping.StandardErrorStopper`), or once a confidence interval for the
ability estimate falls within one ability band ({py:class}`catsim.stopping.ConfidenceIntervalStopper`).
A {py:class}`catsim.stopping.StopRule` combines several stoppers, ending the test as soon as any one
holds, after an optional minimum number of items.

In {py:mod}`catsim`, stoppers are in the {py:mod}`catsim.stopping` module.

## catsim's Parts

```{mermaid}
flowchart LR
    bank[(ItemBank)] --> engine[CatEngine]
    strategies["Strategies:<br>initializer, selector,<br>estimator, stopper"] --> engine
    app([Your application]) -- "start, select, answer:<br>one session, step by step" --> engine
    simulate[simulate] -- "one session per<br>examinee" --> engine
    simulate -- returns --> result[SimulationResult]
```

- **The item bank.** An `ItemBank`, from irtlib, holds the calibrated items, their parameters and
  their names, and never changes.
- **The engine and the session.** A {py:class}`~catsim.CatEngine` holds an item bank and the
  strategies of a test. Its three steps follow the lifecycle above: `start` sets the initial
  estimate, `select` chooses the next item and `answer` records a response, updates the estimate
  and asks the stopper whether the test ends. `start` returns a {py:class}`~catsim.Session`, the
  record of one examinee's test so far, and `select` and `answer` take one and return the next.
  catsim keeps nothing between steps:
  an application that gives tests keeps its sessions and calls the steps as its examinees answer
  ({doc}`embedding-catsim`).
- **The strategies.** One for each phase, and an optional exposure control that may decline the
  item the selector proposes. The technique pages describe the built-in ones
  ({doc}`/api/techniques/index`), and a method author can write others ({doc}`extending`).
- **Simulations and their results.** {py:func}`~catsim.simulate` runs one session for each simulated
  examinee, given by a true ability or by recorded responses, and returns a
  {py:class}`~catsim.SimulationResult`, whose tables and metrics compare the final estimates with
  the true abilities ({doc}`running-simulations`). {doc}`reproducibility` shows how to seed a study
  so that it repeats.

{doc}`/adr/0006-steps-from-explicit-inputs` records why each step takes everything it needs as an
input.
