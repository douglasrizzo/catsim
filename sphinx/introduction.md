# Introduction

{py:mod}`catsim` is a Python package for computerized adaptive testing (CAT), providing both stepwise CAT execution and CAT simulation tools for a modern assessment workflow that adapts to each examinee's ability level in real time.

## What are Latent Traits?

Assessment instruments measure *latent traits* - internal characteristics that cannot be directly observed, such as knowledge or personality traits. Educational and psychological tests are common examples. When a latent trait is expressed numerically, it is called an *ability* or *proficiency*.

## The Problem with Linear Tests

Traditional *linear tests* use a paper-and-pencil approach where all examinees answer the same items in the same order. This creates several problems:

- High-ability examinees waste time on easy questions
- Low-ability examinees become frustrated with overly difficult questions
- Test length cannot be optimized for individual examinees

## What is Computerized Adaptive Testing?

Computerized adaptive testing (CAT) emerged in the 1970s as *tailored testing*
{cite:p}`Lord77`. Unlike linear tests, CAT adapts in real time:

1. After each item, the examinee's ability is re-estimated
2. The next item is selected based on the updated ability estimate
3. Each examinee receives items matched to their ability level

This approach is more efficient and engaging, reducing test length while maintaining accuracy.

## About catsim

CAT systems rely on Item Response Theory (IRT), which provides the mathematical
framework for modeling tests, items, and examinees. While CAT packages exist for
R {cite:p}`Magis12`, {py:mod}`catsim` brings these capabilities to Python. Built
on {py:mod}`numpy` and [SciPy](https://scipy.org/), it provides:

- Multiple initialization, selection, estimation, and stopping methods
- A reusable CAT engine for manual and embedded workflows
- Simulation tools for comparing different CAT methodologies

## Item Response Theory Models

{py:mod}`catsim` borrows many concepts from Item Response Theory
{cite:p}`Lord68,Rasch66`, a series of models created in the second part of the 20th century
with the goal of *measuring latent traits*. {py:mod}`catsim` makes use of Item
Response Theory one-, two-, three- and four-parameter logistic models, a series of models
in which examinees and items are represented by a set of numerical values (the
models' parameters). Item Response Theory itself was created with the goal of
measuring latent traits as well as assessing and comparing individuals' abilities
by allocating them in ability scales, inspiring as well as justifying its use in
adaptive testing.

The logistic models of Item Response Theory are unidimensional, which means that a given assessment instrument only measures a single ability (or dimension of knowledge). The instrument, in turn, is composed of *items* in which examinees manifest their latent traits when answering them.

In unidimensional IRT models, an examinee's ability is represented as $\theta$. Usually $-\infty < \theta < \infty$, but since the scale of $\theta$ is up to the individuals creating the instrument, it is common for the values to follow the normal distribution $N(0, 1)$, such that $-4 < \theta < 4$. Additionally, $\hat{\theta}$ is the estimate of $\theta$. Since a latent trait cannot be measured directly, estimates need to be made, which tend to get closer to the theoretical true $\theta$ as the test progresses in length. {doc}`terms-and-notation` lists the terms and symbols these pages use.

Under the logistic models of IRT, an item is represented by the following parameters:

> - $a$ represents an item's *discrimination* parameter, that is, how well it discriminates individuals who answer the item correctly (or, in an alternative interpretation, individuals who agree with the idea of the item) and those who don't. An item with a high $a$ value tends to be answered correctly by all individuals whose $\theta$ is above the items difficulty level and wrongly by all the others; as this value gets lower, this threshold gets blurry and the item starts not to be as informative. It is common for $a > 0$.
> - $b$ represents an item's *difficulty* parameter. This parameter, which is measured in the same scale as $\theta$, shows at which point of the ability scale an item is more informative, that is, where it discriminates the individuals who agree and those who disagree with the item. Since $b$ and $\theta$ are measured in the same scale, $b$ follows the same distributions as $\theta$. For a CAT, it is good for an item bank to have as many items as possible in all difficulty levels, so that the CAT may select the best item for each individual in all ability levels.
> - $c$ represents an item's *pseudo-guessing* parameter, the lower asymptote of its characteristic curve. This parameter denotes what is the probability of individuals with low ability values to still answer the item correctly. {py:mod}`catsim` requires $0 \leq c < d$, and the lower the value of this parameter, the better the item is considered.
> - $d$ represents an item's *upper asymptote*. This parameter denotes what is the probability of individuals with high ability values to answer the item correctly, so $1 - d$ is the probability that they still answer it incorrectly. {py:mod}`catsim` requires $c < d \leq 1$, and the higher the value of this parameter, the better the item is considered.

For a set of items $I$, when $\forall i \in I, d_i = 1$, the four-parameter
logistic model is reduced to the three-parameter logistic model. When, in
addition, $\forall i \in I, c_i = 0$, the three-parameter logistic model is
reduced to the two-parameter logistic model.
Additionally, if all values of $a$ are equal, the two-parameter logistic
model is reduced to the one-parameter logistic model. Finally, when
$\forall i \in I, a_i = 1$, we have the Rasch model {cite:p}`Rasch66`. Thus,
{py:mod}`catsim` can treat all of the logistic models presented above,
since the underlying functions of all logistic models related to test simulations
are the same, given the correct item parameters.

Under IRT, the probability of an examinee with a given $\hat{\theta}$ value
to answer item $i$ correctly, given the item parameters, is given by
{cite:p}`Ayala2009,Magis13`

$$
P(X_i = 1| \theta) = c_i + \frac{d_i-c_i}{1+ e^{-a_i(\theta-b_i)}}.
$$

The information this item gives is computed as {cite:p}`Ayala2009,Magis13`

$$
I_i(\theta) = \frac{a^2[(P(\theta)-c)]^2[d - P(\theta)]^2}{(d-c)^2(1-P(\theta))P(\theta)}.
$$

Both of these functions are graphically represented in the following figure. It is possible to see that an item is most informative when its difficulty parameter is close the examinee's ability.

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

The sum of the information of all items in a test is called *test information*
{cite:p}`Ayala2009`:

$$
I(\theta) = \sum_{j \in J} I_j(\theta).
$$

The amount of error in the estimate of an examinee's ability after a test is
called the *standard error of estimation* {cite:p}`Ayala2009` and it is given by

$$
SEE = \sqrt{\frac{1}{I(\theta)}}
$$

Since the denominator in the calculation of the $SEE$ is $I(\theta)$, it is clear to see that the more items an examinee answers, the smaller SEE gets.

{py:mod}`catsim` provides these functions in the {py:mod}`catsim.irt` module.

### The Item Matrix

In {py:mod}`catsim`, a collection of items is represented as a {py:class}`numpy.ndarray` whose rows and columns represent items and their parameters, respectively. Thus, it is referred to as the *item matrix*. The item matrix has four columns, which hold the parameters $a$, $b$, $c$ and $d$, in this order. A matrix with one, two or three columns is read as $[b]$, $[a, b]$ or $[a, b, c]$, and the missing parameters are filled in with $a = 1$, $c = 0$ and $d = 1$. An {py:class}`catsim.ItemBank` holds an item matrix, and its {py:meth}`~catsim.ItemBank.generate` method generates one as follows:

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

These examples depict the generation of five items according to the
different logistic models. Whatever the model, the item matrix has four columns.
In the last example, parameters $a$ and $b$ have a
correlation of $0.5$, an adjustment that may be useful in case simulations
require it {cite:p}`Chang2001`.

The item matrix holds only item parameters. Item exposure is run-level output, derived from simulation results ({py:attr}`catsim.SimulationResult.exposure_rates`). The exposure rate is commonly denoted as $r$ and computed as follows:

$$
r_i = \frac{q_i}{N}
$$

Where $q_i$ represents the number of tests item $i$ has been used on and $N$ is the total number of tests applied.

## Computerized adaptive tests

Unlike linear tests, in which items are sequentially presented to examinees and their ability estimated at the end of the test, in a computerized adaptive test (CAT), an examinee's ability is updated after the response of each item. The updated knowledge of an examinee's ability at each step of the test allows for the selection of more informative items *during* the test itself, which in turn reduces the standard error of estimation of their ability at a faster rate.

### The CAT Lifecycle

In general, a computerized adaptive test has a very well-defined lifecycle:

```{mermaid}
flowchart TD
    start([Start]) --> init[Initial ability estimation]
    init --> select[Item selection and administration]
    select --> answer[Capture answer]
    answer --> estimate[Proficiency estimation]
    estimate --> stop{Stopping criterion<br>reached?}
    stop -- No --> select
    stop -- Yes --> finish([End])
```

1. The examinee's initial ability is estimated;
2. An item is selected based on the current ability estimation;
3. The ability is reestimated based on the answers to all items up until now;
4. **If** a stopping criterion is met, stop the test. **Else** go back to step 2.

There is a considerable amount of literature covering these four phases proposed by many authors. In {py:mod}`catsim`, each phase is separated in its own module, which makes it easy to create simulations combining different methods for each phase. Each module will be explained separately, along with its API.

#### Initialization

The initialization procedure is done only once during each examinee's test. In it, the initial value of an examinee's ability $\hat\theta_0$ is selected. This procedure may be done in a variety of ways: a standard value can be chosen to initialize all examinees ({py:class}`catsim.initialization.FixedPointInitializer`); it can be drawn from a uniform or a normal distribution ({py:class}`catsim.initialization.UniformInitializer`, {py:class}`catsim.initialization.NormalInitializer`); the place in the item bank with items of more information can be chosen to initialize $\hat\theta_0$ etc.

In {py:mod}`catsim`, initialization procedures can be found in the {py:mod}`catsim.initialization` module.

#### Item Selection

With a set value for $\hat{\theta}$, an item is chosen from the item bank and presented to the examinee, which the examinee answers and its answer, along with the answers to all previous items, is used to estimate $\hat{\theta}$.

Item selection methods are diverse. The most famous method is to choose the item that maximizes the *gain of information*, represented by {py:class}`catsim.selection.MaxInfoSelector`. This method, however, has been shown to have some drawbacks, like overusing few items from the item bank while ignoring items with inferior parameters. In order to correct that, other item selection methods were proposed.

In {py:mod}`catsim`, item selection procedures can be found in the {py:mod}`catsim.selection` module.

#### Proficiency Estimation

Proficiency estimation occurs whenever an examinee answers a new item. Given a dichotomous (binary) response vector and the parameters of the corresponding items that were answered, it is the job of an estimator to return a new value for the examinee's $\hat{\theta}$. This value reflects the examinee's ability, given his or hers answers up until that point of the test.

In Python, an example of a list that may be used as a valid dichotomous response vector is as follows:

```pycon
>>> responses = [1,1,1,0,1,1,0,1,0,0,1,0,0,0,1,0]
```

Estimation techniques are generally separated between maximum-likelihood estimation procedures (whose job is to return the $\hat{\theta}$ value that maximizes the *log-likelihood* function, presented in {py:func}`catsim.irt.log_likelihood`); and Bayesian estimation procedures, which tend to use a priori information of the distributions of examinee's abilities to estimate new values for them.

In {py:mod}`catsim`, ability estimation procedures can be found in the {py:mod}`catsim.estimation` module.

#### Stopping Criterion

Since items in a CAT are selected on-the-fly, a stopping criterion must be chosen such that, when achieved, no new items are presented to the examinee and the test is deemed finished. These stopping criteria might be achieved when the test reaches a fixed number of items ({py:class}`catsim.stopping.MaxLengthStopper`), when the standard error the estimator reports with its estimate (the $SEE$ above for maximum likelihood, the posterior standard deviation for EAP) reaches a lower threshold ({py:class}`catsim.stopping.StandardErrorStopper`), or when a confidence interval for the ability estimate falls within one ability band ({py:class}`catsim.stopping.ConfidenceIntervalStopper`). A {py:class}`catsim.stopping.StopRule` combines several of them, ending the test as soon as any one holds, after an optional minimum number of items.

In {py:mod}`catsim`, test stopping criteria can be found in the {py:mod}`catsim.stopping` module.

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
{doc}`/adr/0001-explicit-cat-engine` records why catsim is built around the engine.
