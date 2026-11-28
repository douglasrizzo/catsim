# Terms and Notation

The catsim documentation uses one term for each concept. This page lists each term with the
synonyms you may know from the literature and the name to look for in the API, then the symbols
that the formulas use.

## Examinees and items

examinee
: The person who takes a test. Also called *respondent* or *test taker*.

ability $\theta$
: The latent trait a test measures, on a continuous scale. Also called *proficiency*, *latent
  trait* or *person parameter*. A simulation knows each examinee's true ability when it is run
  with `true_thetas`.

ability estimate $\hat{\theta}$
: The estimate of an examinee's ability from the responses so far: *provisional* while the test
  runs, *final* when it ends. In the API: `theta` in an `Estimate` and in a strategy's context,
  `current_theta` on a session, the `theta` column of a simulation result's tables.

item
: One question of a test, scored correct or incorrect. In the API an item is its row in the item
  bank, its *index*, counted from 0 (`item`). Each item also has a *name*, unique in its bank
  (`name`), which results, plots and error messages show.

item bank
: The calibrated items a test draws from. Also called *item pool*. In the API: `ItemBank`, which
  holds $M$ items, `n_items`, and their `names`.

logistic models
: The item response models catsim supports: the one-, two-, three- and four-parameter logistic
  models (1PL to 4PL). The 1PL with every discrimination equal to 1 is the *Rasch model*.
  `ItemBank.model` tells which of them a bank's parameters need.

discrimination $a_i$
: How sharply item $i$ separates examinees of lower and higher ability: the slope of its
  characteristic curve. Also called the *slope parameter*. `ItemBank.discrimination`.

difficulty $b_i$
: The ability at which item $i$'s characteristic curve is steepest, on the same scale as $\theta$.
  Also called the *location* or *threshold*. `ItemBank.difficulty`.

guessing parameter $c_i$
: The lower asymptote of item $i$'s characteristic curve: the probability that an examinee of very
  low ability answers correctly. Also called the *pseudo-guessing parameter*.
  `ItemBank.pseudo_guessing`; 0 under the 1PL and 2PL.

upper asymptote $d_i$
: The upper asymptote of item $i$'s characteristic curve: the probability that an examinee of very
  high ability answers correctly. $1 - d_i$ is called the *slipping* or *carelessness* parameter.
  `ItemBank.upper_asymptote`; 1 under every model but the 4PL.

item characteristic curve $P_i(\theta)$
: The probability of a correct response to item $i$ at ability $\theta$. Also called the *item
  response function* (IRF). `ItemBank.icc`.

item information $I_i(\theta)$
: The Fisher information of item $i$ at ability $\theta$: how much a response to the item sharpens
  an estimate near $\theta$. `ItemBank.information`.

test information $I(\theta)$
: The sum of the information of the administered items. The more information, the smaller the
  standard error. The functions for these quantities are in irtlib's `irtlib.irt` module.

## The adaptive test

response $u_i$
: The examinee's answer to item $i$, scored correct ($u_i = 1$, `True` or 1 in the API) or
  incorrect ($u_i = 0$, `False` or 0).

responses $\mathbf{u}$
: The responses to the administered items, in the order they were given. Also called a *response
  pattern* or *response vector*. In the API: `responses`.

administered items
: The items given so far, in order. In the API: `administered_items`.

excluded items
: Items a session must not be given although it has not had them, such as the items its exposure
  control declined. They do not count toward the test length. In the API: `excluded_items`.

candidate items
: The items a selector may still choose: those neither administered nor excluded. A selector may
  narrow them further, for example to the items below its exposure cap. In these pages a
  *candidate* is always an item, never an examinee. In the API: the `candidates` of the selection
  helpers.

test length $n$
: The number of administered items. In the API: the `test_length` column and metric of a simulation
  result. A `MaxLengthStopper` ends the test at `max_items`; a selector that plans its schedule for a
  test of known length takes it as `test_size`.

session
: One examinee's test, from the initial estimate to the stop. In the API: `Session`, the record of
  the test so far, which each step of `CatEngine` returns anew.

initializer
: The strategy that sets the ability estimate before the first response, the *starting point*
  or *initial ability estimate* of the literature.

selector
: The strategy that chooses the next item. Also called the *item selection rule* or *item selection
  criterion*.

estimator
: The strategy that estimates ability from the responses, such as maximum likelihood (ML),
  expected a posteriori (EAP), maximum a posteriori (MAP) or Warm's weighted likelihood (WLE). Also
  called the *ability estimation method*.

stopper
: The strategy that decides whether the test ends: a stopping criterion. Also called a *stopping
  rule* or *termination criterion*. Every stopper derives from `BaseStopper`, for example
  `MaxLengthStopper` or `StandardErrorStopper`. `StopRule` is the stopper that combines several
  others; these pages say "rule" only for it.

exposure control
: Keeping items from being given so often that they become known to future examinees. Some
  selectors do it themselves, with an exposure cap `r_max` or by choosing at random. An engine can
  also take an *exposure control*, a `BaseExposureControl`, which may decline the item the selector
  proposes; the session then excludes that item for the rest of its test.

stop reason
: Why a session ended: `StopReason.STOPPER` when its stopper was satisfied, `StopReason.NO_ITEMS`
  when its selector had no item to offer, or none that the exposure control accepted.

standard error $SE(\hat{\theta})$
: The uncertainty of an ability estimate, on the scale of $\theta$. Also called the *standard
  error of estimation* (SEE) or *standard error of measurement* (SEM). Each estimator computes it
  by its own formula; for maximum likelihood it is $1/\sqrt{I(\hat{\theta})}$. In the API: `se` in
  an `Estimate` and in a strategy's context, `current_se` on a session, the `se` column of a
  simulation result's tables.

## Simulations

simulation
: A run of one session per examinee, by `simulate`, which returns a `SimulationResult`. A run is
  one *condition*; a study compares several.

generated examinee
: A simulated examinee given by a true ability, who answers each item correctly with probability
  $P_i(\theta)$. In the API: an entry of `true_thetas`.

recorded examinee
: A simulated examinee whose answers come from data, as in a *post-hoc* study of a test given
  linearly. In the API: a row of the *response matrix* `responses`, one column per item of the bank,
  with NaN where an answer is missing.

replication
: One of several runs of the same condition with other seeds, which together show how much its
  metrics vary.

common random numbers
: Running the conditions a study compares on the same random draws, so that their difference
  carries less Monte Carlo error. Run on the same seed, catsim's simulated examinees give every
  engine the same answer to each item.

part, worker
: A *part* is a contiguous share of a simulation's examinees, whose sessions run in order with
  exposure counts of their own (`parts`). A *worker* is a process that runs parts (`workers`).

exposure rate $r_i$
: The share of examinees who were given item $i$. In the API: `exposure_rates()` of a simulation
  result, and the `exposure_rates` a selector is given at each step. Exposure-control selectors
  limit it with a cap, `r_max`. `simulate` gives the rule for the rates each session sees.

exposure feedback
: The dependence of a session on the sessions before it through the exposure rates it sees, under a
  selector's `r_max` cap.

proposal rate
: The share of examinees for whom the selector proposed item $i$: the item was given or the
  exposure control declined it. In the API: `proposal_rates()` of a simulation result. Without an
  exposure control it equals the exposure rate.

overlap rate
: The expected share of items that two examinees have in common. In the API: `overlap_rate()` of a
  simulation result, `None` unless every session has the same test length.

bias, MSE, RMSE
: How far the final estimates are from the true abilities, over the examinees: the mean of
  $\hat{\theta} - \theta$, the mean of $(\hat{\theta} - \theta)^2$, and its square root. A positive
  bias means abilities are overestimated. In the API: `bias()`, `mse()` and `rmse()` of a
  simulation result.

Monte Carlo standard error
: How much a metric would vary over replications: the uncertainty that the finite number of
  simulated examinees leaves in it. Each metric of a simulation result comes with it, as `mc_se`.

decile
: One tenth of a simulation's examinees by true ability, from decile 0, the lowest abilities, to 9.
  In the API: `deciles()` of a simulation result, and the `decile` argument of its metrics.

## Notation

| Symbol | Meaning | In the API |
|---|---|---|
| $\theta$ | ability; in a simulation, the true ability | `true_thetas` |
| $\hat{\theta}$ | ability estimate | `theta`, `current_theta` |
| $i$ | item index | `item`, an item's row in the `ItemBank` |
| $a_i$, $b_i$, $c_i$, $d_i$ | discrimination, difficulty, guessing parameter and upper asymptote of item $i$ | `discrimination`, `difficulty`, `pseudo_guessing`, `upper_asymptote` |
| $P_i(\theta)$ | probability of a correct response to item $i$ | `ItemBank.icc` |
| $Q_i(\theta)$ | $1 - P_i(\theta)$, probability of an incorrect response | |
| $I_i(\theta)$ | information of item $i$ | `ItemBank.information` |
| $I(\theta)$ | test information, $\sum_i I_i(\theta)$ over the administered items | |
| $SE(\hat{\theta})$ | standard error of the estimate | `se`, `current_se` |
| $u_i$ | response to item $i$: 1 if correct, 0 if not | an entry of `responses` |
| $\mathbf{u}$ | responses $(u_1, \dots, u_n)$ | `responses` |
| $L(\theta)$ | likelihood of the responses at ability $\theta$ | |
| $n$ | test length, the number of administered items | `test_length` |
| $N$ | number of examinees | `len(result.sessions)` |
| $j$ | examinee index | the `session` column of a simulation result's tables |
| $M$ | bank size, the number of items in the item bank | `ItemBank.n_items` |
| $r_i$ | exposure rate of item $i$ | `exposure_rates` |
| $r_{\max}$ | exposure cap | `r_max` |

A sum over $i$ runs over the administered items unless the text says otherwise. A set of items gets
a letter of its own, such as $R$ for the items not yet administered: $I$ always denotes information.
