---
title: MML-EM for 4PL
kind: technique
status: parked
priority: 4
effort: L
blocked_by: [scope/calibration/mml-em-3pl]
raised_by: []
---

# MML-EM for 4PL

> Background: [the literature review §3](literature-review.md)

## Why it's deferred

The 4PL model adds the upper asymptote $d_i < 1$, representing the probability that even very high-ability examinees may respond incorrectly (slipping). Estimation of $d$ via MML-EM faces the same problems as $c$ but worse:

1. **Weak identifiability**: $d$ is only informed by the response behavior of very high-ability examinees on the item. Most datasets have insufficient information in the upper tail.
2. **Boundary attraction**: Without regularization, $d$ estimates drift to exactly 1.0 (reducing to 3PL) or to implausible values.
3. **Requires strong priors**: A Beta prior on $d$ (e.g., $\text{Beta}(20, 2)$ centered near 0.95) is essentially mandatory. At that point, the estimate is heavily prior-driven.
4. **Rare in practice**: The 4PL is rarely used in operational testing. Most applied work uses 1PL–3PL.

## Design questions

1. **Is MML-EM the right approach?** Bayesian MCMC handles the d-parameter more naturally via informative priors with proper posterior uncertainty. MML-EM with a penalty is an approximation.

2. **Implementation strategy**: If we proceed, it would extend mml-em-3pl (3PL) by adding $d$ to the M-step optimization:
   - Bounds: $d \in [0.8, 1.0]$
   - Beta prior: $\text{Beta}(\alpha_d, \beta_d)$ with configurable hyperparameters
   - Gradients: $\partial P / \partial d = \psi$ (already in icc-gradients)

3. **Validation**: How do we validate 4PL calibration? Recovery tests require very large samples ($N > 5000$) and long tests ($I > 50$) to have any hope of recovering $d$ values.

## When to revisit

- If users request 4PL calibration specifically.
- If Bayesian MCMC (discussion card) is implemented, making 4PL estimation more natural.
- If a concrete use case emerges where $d < 1$ is scientifically justified and the data supports estimation.

## References

- Barton, M. A., & Lord, F. M. (1981). An upper asymptote for the three-parameter logistic item-response model. *Research Bulletin 81-20*. Educational Testing Service.
- Loken, E., & Rulison, K. L. (2010). Estimation of a four-parameter item response theory model. *British Journal of Mathematical and Statistical Psychology*, 63, 509–525.
