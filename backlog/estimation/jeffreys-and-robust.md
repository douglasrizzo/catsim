---
title: Jeffreys prior and robust ability estimation
kind: technique
status: needs-research
priority: 3
effort: M
blocked_by: []
raised_by: [catR comparison]
---

# Jeffreys prior and robust ability estimation

**Confidence the gaps are real:** low to medium. These were listed from memory of catR, and each
needs checking.

## What is known

- **Jeffreys prior.** catR's Bayesian estimators take `priorDist` `"norm"`, `"unif"` or
  `"Jeffreys"`. catsim's `MAPEstimator` and `EAPEstimator` accept any log prior. The Jeffreys prior,
  though, depends on the administered items (the square root of the test information), which a
  fixed log-prior callable cannot express. catsim's WLE equals the Jeffreys-prior mode for the 1PL
  and 2PL (the WLE technique page).
- **Robust estimation.** catR's `thetaEst` may have a `"ROB"` method, possibly Schuster & Yuan
  (2011). Verify that it exists.

## Open questions

- Is a Jeffreys-prior MAP or EAP worth adding, given the WLE covers the 1PL and 2PL? Under the 3PL
  the two differ.
- Does the robust estimator exist in catR, and does it fit the estimator contract (ADR-0007)?
- Which of these would a parity case need (ADR-0020)? A parity case cannot use a catR option catsim
  lacks.
