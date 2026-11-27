---
title: Bayesian MCMC calibration
kind: technique
status: parked
priority: 4
effort: L
blocked_by: []
raised_by: []
---

# Bayesian MCMC calibration

> Background: [the literature review §2.4](literature-review.md)

## Why it's deferred

Bayesian MCMC calibration (Patz & Junker, 1999) is the most principled approach for 3PL/4PL estimation — priors naturally regularize c and d, and full posteriors on item parameters are produced rather than point estimates. However, it requires significant infrastructure:

1. **Sampler**: MH-within-Gibbs (Patz-Junker) or Hamiltonian Monte Carlo. Neither is trivial to implement correctly.
2. **Convergence diagnostics**: Rhat, effective sample size (ESS), trace plots. These are their own module.
3. **Prior specification**: Configurable priors for each parameter ($a \sim \text{LogNormal}$, $b \sim N(0, \sigma^2)$, $c \sim \text{Beta}(\alpha, \beta)$, $d \sim \text{Beta}(\alpha, \beta)$).
4. **Computational cost**: Thousands of iterations with full likelihood evaluations per iteration. Orders of magnitude slower than MML-EM.

## Design questions

1. **Build or wrap?** Should catsim implement its own Gibbs sampler, or provide a bridge to PyMC / NumPyro / emcee? Building from scratch is educational but hard to get right (tuning, adaptation). Wrapping adds an optional dependency but gives production-quality sampling.

2. **Scope**: Full Bayesian calibration, or just Bayesian regularization of MML-EM (Mislevy, 1986)? The latter (adding priors to the M-step) is already partially addressed in mml-em-3pl and is much simpler.

3. **Output format**: MML-EM returns point estimates. MCMC returns posterior samples. How should `ItemBank` represent uncertainty on item parameters? Options:
   - Store posterior means as point estimates (loses information).
   - Add an optional `ItemBankPosterior` that stores full chains.
   - Return raw chains and let the user summarize.

## When to revisit

- If MML-EM for 3PL (mml-em-3pl) proves insufficient for c-parameter estimation in practice.
- If users request posterior uncertainty on item parameters.
- If catsim adds a dependency on a probabilistic programming library for another feature.

## References

- Patz, R. J., & Junker, B. W. (1999). A straightforward approach to Markov chain Monte Carlo methods for item response models. *JEBS*, 24, 146–178.
- Albert, J. H. (1992). Bayesian estimation of normal ogive item response curves using Gibbs sampling. *JES*, 17, 251–269.
- Mislevy, R. J. (1986). Bayes modal estimation in item response models. *Psychometrika*, 51, 177–195.
