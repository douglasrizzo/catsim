# Introduction

{py:mod}`catsim` is a Python package for computerized adaptive testing (CAT). It runs adaptive
tests one item at a time, in an application, and simulates them on examinees whose abilities are
known, to compare testing methods. This page introduces CAT and the item response theory behind it,
and ends with a map of catsim's parts. {doc}`terms-and-notation` lists the terms and symbols these
pages use.

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
examinee or item bank ({doc}`/adr/0012-ability-scale-and-extreme-patterns`).

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
    :caption: Item characteristic and information curves for an item. The dot represents the point at which the item is most informative.

    import matplotlib.pyplot as plt
    from catsim import ItemBank
    from catsim.plot import PlotType, item_curve

    item_bank = ItemBank.generate(1, rng=0)
    item_curve(item_bank[0], kind=PlotType.BOTH)
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
error. {py:mod}`catsim` computes these quantities with the functions of the {py:mod}`catsim.irt`
module.

### The Item Matrix

In {py:mod}`catsim`, the items are a {py:class}`numpy.ndarray` with one row per item and one column
per parameter, the *item matrix*. Its four columns hold $a$, $b$, $c$ and $d$, in this order. A
matrix with one, two or three columns is read as $[b]$, $[a, b]$ or $[a, b, c]$, and the missing
parameters take the values $a = 1$, $c = 0$ and $d = 1$. An {py:class}`~catsim.ItemBank` holds an
item matrix, and {py:meth}`ItemBank.generate <catsim.ItemBank.generate>` draws one at random:

```pycon
>>> from catsim import ItemBank
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
$0.5$, which simulations may need {cite:p}`Chang2001`.

## Computerized Adaptive Tests

Computerized adaptive testing emerged in the 1970s as *tailored testing* {cite:p}`Lord77`. A linear
test estimates an examinee's ability once, at the end. A computerized adaptive test (CAT) updates the
estimate after each response. Knowing more about the ability at each step lets the test
choose more informative items *during* the test, so the standard error of the estimate falls
faster.

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
give every examinee the same value ({py:class}`catsim.initialization.FixedPointInitializer`), draw
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
({doc}`selection techniques </api/techniques/selection/index>`). A simulation reports the rates as
{py:attr}`~catsim.SimulationResult.exposure_rates`.

In {py:mod}`catsim`, selectors are in the {py:mod}`catsim.selection` module.

#### Ability Estimation

After each response, the estimator updates the ability estimate $\hat{\theta}$ from the responses
so far, each correct or incorrect, and the parameters of the items answered. In {py:mod}`catsim`,
the responses are a list of booleans, `True` for a correct response, one for each administered
item, in the order they were given:

```pycon
>>> responses = [True, True, False, True, False]
```

Estimators fall into two groups. Maximum likelihood estimators return the ability at which the
responses are most likely, the maximum of the *log-likelihood* ({py:func}`catsim.irt.log_likelihood`).
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

## Architecture

```{mermaid}
flowchart LR
    bank[(Item bank)] --> engine[CatEngine]
    strategies["Strategies:<br>initializer, selector,<br>estimator, stopper"] --> engine
    app([Your application]) -- "one session,<br>item by item" --> engine
    runner[SimulationRunner] -- "one session per<br>simulated examinee" --> engine
    runner -- returns --> result[SimulationResult]
```

A {py:class}`~catsim.CatEngine` holds an item bank and four strategies, one for each phase of the
lifecycle above, and runs each examinee's test one item at a time. An application asks it for each
item and gives it each response ({doc}`live-sessions`). A {py:class}`~catsim.SimulationRunner` gives
the same test to simulated examinees whose true abilities are known, and returns a
{py:class}`~catsim.SimulationResult` that compares the final estimates with those abilities
({doc}`running-simulations`). {doc}`reproducibility` shows how to seed a simulation study so that
it repeats, and {doc}`extending` how to write a strategy of your own.
{doc}`/adr/0003-explicit-cat-engine` records why catsim is built around the engine.
