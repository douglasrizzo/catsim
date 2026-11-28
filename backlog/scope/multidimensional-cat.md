---
title: Multidimensional CAT (MCAT) family
kind: technique
status: parked
priority: 4
effort: XL
blocked_by: []
raised_by: []
---

# Multidimensional CAT (MCAT) family

> **Needs detailing before it leaves `parked`.** An agent wrote this card in April 2026 from an
> unverified literature survey, against the API from before the rescoping. Check its sources and
> redo its design against ADR-0002 and ADR-0007 before any work.

## Why this is a discussion card

MCAT replaces unidimensional ability $\theta \in \mathbb{R}$ with a multidimensional ability vector $\boldsymbol{\theta} \in \mathbb{R}^D$, backed by a multidimensional IRT (MIRT) model. irtlib's `irtlib.irt`, which catsim builds on (ADR-0024), is hardcoded to the 4PL unidimensional logistic. There is **no matrix theta state, no multidimensional ICC, no Fisher-information matrix** anywhere in the package. Adding MCAT is not a strategy card — it is a fundamental extension of irtlib's response functions and `ItemBank`, and of catsim's sessions and every estimator/selector/stopper. This card documents the scope and what would need to change.

## References

- Reckase, M. D. (2009). *Multidimensional Item Response Theory*. Springer.
- Segall, D. O. (1996). Multidimensional adaptive testing. *Psychometrika*, 61, 331–354.
- Mulder, J., & van der Linden, W. J. (2009). Multidimensional adaptive testing with optimal design criteria for item selection. *Psychometrika*, 74, 273–296.
- Wang, C., & Chang, H.-H. (2011). Item selection in multidimensional computerized adaptive testing — gaining information from different angles. *Psychometrika*, 76, 363–384.
- Yao, L. (2012). Multidimensional CAT selection methods for domain scores. *Applied Psychological Measurement*.
- Diao, Q., & van der Linden, W. J. (2011). Constrained multidimensional adaptive testing with a shadow-test approach. *Applied Psychological Measurement*.
- Finkelman, M. D., Nering, M. L., & Roussos, L. A. (2009). A conditional exposure control method for MCAT.

## Theory

A MIRT 2PL item has a vector discrimination $\mathbf{a}_i \in \mathbb{R}^D$ and a scalar difficulty $b_i$:

$$
P_i(\boldsymbol{\theta}) \;=\; \mathrm{logit}^{-1}\!\big(\mathbf{a}_i^\top \boldsymbol{\theta} - b_i\big)
$$

The Fisher information at $\boldsymbol{\theta}$ is a **matrix**:

$$
\mathbf{I}_i(\boldsymbol{\theta}) \;=\; P_i(\boldsymbol{\theta})\,\big[1 - P_i(\boldsymbol{\theta})\big]\,\mathbf{a}_i \mathbf{a}_i^\top
$$

Selection criteria are scalar functionals of this matrix:

| Name | Criterion | Reference |
|---|---|---|
| **D-optimality** | Maximize $\det \mathbf{I}_{\text{test}}(\boldsymbol{\theta})$ | Segall 1996 |
| **A-optimality** | Minimize $\mathrm{tr}\, \mathbf{I}_{\text{test}}^{-1}(\boldsymbol{\theta})$ | Mulder & van der Linden 2009 |
| **E-optimality** | Minimize the largest eigenvalue of $\mathbf{I}_{\text{test}}^{-1}$ | Mulder & van der Linden 2009 |
| **KL-based** | KL divergence between response distributions at different $\boldsymbol{\theta}$ | Wang & Chang 2011 |
| **Domain-score optimization** | Maximize info on a linear combination of dimensions | Yao 2012 |

Estimation also generalizes: multivariate EAP (posterior over $\mathbb{R}^D$, quadrature grows as $Q^D$), multivariate MLE, multivariate MAP. Quadrature becomes a problem quickly — $Q=20, D=3$ is already $8{,}000$ grid points.

## What would need to change in catsim

1. **`irtlib.irt` refactor, in irtlib** (its parked `scope/multidimensional-models` card): vectorize or templatize `icc`, `inf`, `test_info`, `log_likelihood` over dimensionality. The unidimensional case must remain a hot path — it is by far the most common use case.
2. **irtlib's `ItemBank` refactor**: item parameter storage is currently `(n_items, 5)`. MIRT items need `(n_items, D + 3)` or similar. Must decide: parallel `MIRTItemBank` vs. dimensionality-aware `ItemBank`.
3. **`CatSessionState`**: `theta` becomes a vector. `CatStepResult` records a vector path.
4. **Estimators**: every current estimator is scalar. Need multivariate MLE (gradient/Hessian), multivariate EAP (tensor-product quadrature or sparse grid), multivariate MAP.
5. **Selectors**: the whole selector family is scalar-theta. Need matrix-info variants (D/A/E-optimality, KL, posterior-KL).
6. **Stoppers**: current SEE-based stoppers become "trace of inverse info" or "determinant of info" stoppers.
7. **Exposure control**: mostly unchanged, since exposure is a per-item property.
8. **Notebooks / documentation**: entire new tutorial surface.

## Open questions

1. **Is MCAT in scope at all?** This is the decisive question. If the answer is "no", close this card. If "yes, eventually", file a scoping ADR. If "yes now", open a parallel roadmap.
2. **Dimensionality ceiling**: most MCAT research uses $D \le 5$. Is that enough? Higher dimensions require sparse-grid or MCMC estimation and blow up the scope considerably.
3. **Data layout**: parallel subpackage (`catsim.multidim.*`) vs. a refactored core. Parallel is safer (no regression risk for existing users) but fragments the library.
4. **Backwards compatibility**: if `irtlib.irt` is refactored, does the unidimensional API stay identical? It must, or every existing catsim user breaks.
5. **Constraints on items**: MIRT item calibration is harder than unidimensional IRT. catsim is a CAT simulator, not a calibrator — but users will expect catsim to *consume* calibrated MIRT banks, and the file format question is open.

## Notes

- **External dependencies**: none strictly required; multivariate quadrature and linear algebra are already in `scipy`/`numpy`. MIRT calibration itself is typically done in `mirt` (R) or `girth` (Python); catsim would only consume the calibrated parameters.
- **Not blocked by any existing catsim work** — but it is **a fundamental refactor**, not an additive feature. Treat as its own roadmap item.
- **ADR required**: the dimensionality-aware refactor (or lack thereof) is exactly the sort of architectural commitment that belongs in `sphinx/adr/`. Decide before any code moves.
- **Cross-reference**: CD-CAT ([cd-cat](cd-cat.md)) and MCAT are orthogonal extensions. They can be pursued independently but share the "parallel subsystem" pattern.
- **Pragmatic first step if greenlit**: add `D=2` MIRT as a second hot path, share as much `CatEngine` / `CatSessionState` code as possible via generics, and ship D-optimality + multivariate EAP. Leave $D \ge 3$ for a follow-up.
