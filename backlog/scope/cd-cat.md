---
title: Cognitive Diagnosis CAT (CD-CAT) family
kind: technique
status: parked
priority: 4
effort: XL
blocked_by: []
raised_by: []
---

# CD-CAT family (PWKL, GDI, JSD, SWDGDI, …)

> Background: [the literature review §2.1.D and §5 Q5](../literature-review.md)

## Why this is a discussion card

Cognitive Diagnosis CAT (CD-CAT) replaces the unidimensional IRT ability $\theta$ with an **attribute profile** $\boldsymbol{\alpha} \in \{0,1\}^K$ under a diagnostic classification model (DCM) like DINA, DINO, or G-DINA. The goal is to identify which latent attributes (skills) an examinee has mastered, rather than to rank them on a scale. catsim has **no DCM/DINA layer** and no attribute-profile state, so adding CD-CAT is not a single feature — it is a parallel subsystem. This card documents the family and what would be required; it is not ready to implement.

## References

- Cheng, Y. (2009). When cognitive diagnosis meets CAT: CD-CAT. *Psychometrika*, 74(4), 619–632.
- Kaplan, M., de la Torre, J., & Barrada, J. R. (2015). New item selection methods for CD-CAT. *Applied Psychological Measurement*, 39(3), 167–188.
- de la Torre, J., & Chiu, C.-Y. (2016). A general method of empirical Q-matrix validation. *Psychometrika*, 81, 253–273.
- He, Y., & Tan, X. (2020). Jensen–Shannon divergence item selection for MC-DINA CAT. *Applied Psychological Measurement*.
- Wang, C. (2019). Content-balanced item selection for CD-CAT (SWDGDI, CP_SWDGDI).
- Ma, C., et al. (2023). Heuristic search framework for CD-CAT item selection. *Behavior Research Methods*.

## Theory

CD-CAT needs three parallel components that catsim does not currently have:

1. **Diagnostic model**: DINA, DINO, HO-DINA, G-DINA, MC-DINA, or one of the reduced models. Each is parameterized by a Q-matrix (item × attribute) plus slip/guess parameters per item.
2. **Attribute-profile estimator**: maximum likelihood, EAP over $2^K$ profiles, or a Bayesian network update. Replaces the current unidimensional MLE/EAP estimator entirely.
3. **Attribute-aware selector**: information-theoretic criteria computed on the posterior over $2^K$ profiles, not on a continuous $\theta$ grid.

### The selector family

| Name | Criterion | Reference |
|---|---|---|
| **PWKL** | Posterior-weighted Kullback–Leibler over attribute profiles | Cheng 2009 |
| **MPWKL** | Modified PWKL (softer weighting) | Kaplan, de la Torre, Barrada 2015 |
| **GDI** | G-DINA discrimination index — maximizes expected drop in posterior entropy | de la Torre & Chiu 2016 |
| **GDIP** | Posterior-weighted GDI | Kaplan et al. 2015 |
| **MGDI** | Modified GDI with attribute-level normalization | Kaplan et al. 2015 |
| **JSD** | Jensen–Shannon divergence (MC-DINA variant) | He & Tan 2020 |
| **SWDGDI** | Stratified weighted GDI for attribute balance | Wang 2019 |
| **Heuristic-search** | Branch-and-bound over short horizons | Ma et al. 2023 |

They all share the same basic loop: "compute a divergence / entropy functional of the posterior over $\boldsymbol{\alpha}$ for each candidate item, pick the argmax."

## What would need to change in catsim

1. **New module `catsim.diagnostic/`** containing:
   - Q-matrix representation (binary matrix, possibly sparse).
   - A `DCMItemBank` (or a `mode` flag on `ItemBank`) that stores slip/guess per item instead of a/b/c/d.
   - DINA/G-DINA likelihood and posterior helpers.
   - Attribute-profile estimators.
   - The PWKL / GDI / JSD selector family.
2. **Engine extension**: `CatEngine` and `CatSessionState` currently carry a single `theta` scalar. They'd need a parallel mode where the running state is an attribute posterior (a `(2**K,)` float array) or an estimated profile (a `(K,)` bool array).
3. **`CatStepResult` extension**: stepwise outputs would need a profile column or a posterior snapshot.
4. **Simulation / evaluation**: classification accuracy (attribute-wise and profile-wise) replaces RMSE/bias as the core metric.

This is substantially more than one card's worth of work. It deserves a **dedicated subproject** with its own plan file and its own ADR before implementation begins.

## Open questions

1. **Scope**: is CD-CAT on the long-term roadmap, or is catsim strictly unidimensional IRT? The maintainer's answer here is decisive.
2. **DCM subset**: if CD-CAT is in scope, which DCMs to support? Minimal useful set is DINA + G-DINA. DINO, HO-DINA, and MC-DINA can follow.
3. **Data layout**: parallel subpackage (`catsim.diagnostic.*`) vs. unified `ItemBank` with a `mode="dcm"` switch. The subpackage is cleaner; the unified layout shares more engine code.
4. **Estimation**: MAP over profiles is trivially $O(2^K)$ but unusable beyond $K \approx 15$. For larger $K$, approximate Bayesian inference (variational, Gibbs) is needed — a whole research subfield on its own.
5. **Priority**: CD-CAT is high-impact in the education literature but has no user base in catsim today. Adding it pre-emptively risks scope creep; waiting for user demand risks missing a growing niche.

## Notes

- **External dependencies**: potentially `scipy.sparse` for large Q-matrices; nothing else for DINA/G-DINA baselines.
- **Not blocked by any existing catsim work** — but it is **a parallel subsystem**, not an incremental addition. Treat it as its own roadmap item.
- **ADR required before implementation**: decisions about data layout, DCM subset, and engine extension are exactly the durable architectural commitments that belong in `sphinx/adr/`.
- **First step if greenlit**: write a scoping ADR + a DINA-only prototype as a proof of concept, *not* the full selector family. Start with data + estimator, then one selector (PWKL).
- **Reference implementations**: `CDM` R package and the `pyCDM` / `cdcatR` work from de la Torre's group are the authoritative code bases.
