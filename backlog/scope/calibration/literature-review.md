# Literature Review: Item Parameter Estimation (Calibration) for catsim

**Status:** Draft v1
**Started:** 2026-04-09
**Goal:** Survey methods for estimating IRT item parameters (a, b, c, d) from response data. Identify what catsim should implement, in what order, and with what infrastructure.

> **Feature cards** are the other files in this folder.

---

## 1. Context

catsim is a CAT simulator. It currently assumes **pre-calibrated items** — the `ItemBank` takes an `(n_items, 5)` array of `[a, b, c, d, exposure_rate]` and every strategy (selection, estimation, stopping) treats these as known constants. Users either generate synthetic banks via `ItemBank.generate_item_bank()` or supply real calibrations from external tools (`mirt`, `ltm`, `girth`).

This means catsim cannot:
- Calibrate items from field-test data.
- Re-calibrate items online as new responses accumulate.
- Evaluate item fit to decide whether to retire or flag items.
- Be used as a self-contained IRT toolkit for researchers who want parameter estimation alongside simulation.

Adding calibration closes the loop: generate items → calibrate → simulate CATs → evaluate. It also makes catsim competitive with `girth` (Python) and `mirt` (R) for the calibration step, rather than requiring users to leave the ecosystem.

---

## 2. Methods

### 2.1 Joint Maximum Likelihood (JML)

**Reference:** Birnbaum (1968) in Lord & Novick; Lord (1980) *Applications of IRT*.

Treats both item parameters and examinee abilities as fixed unknowns. Alternates between:
1. Fix item params, estimate each θ_j by maximizing the individual log-likelihood.
2. Fix abilities, estimate each item's params by maximizing the per-item log-likelihood.

**Pros:** Simple to implement; directly parallels the way catsim already estimates θ (via `NumericalSearchEstimator`). Works for 1PL/2PL/3PL.

**Cons:** The **Neyman-Scott problem** — the number of nuisance parameters (one θ per examinee) grows with sample size, making item parameter estimates **inconsistent** for fixed test length. JML overestimates discrimination and underestimates difficulty variance. Bias corrections (Warm-type adjustments, fencing) mitigate but don't fully resolve this. Not recommended for production calibration; useful as a didactic baseline.

**Feasibility:** **H** — reuses existing `NumericalSearchEstimator` for the θ step and `scipy.optimize.minimize` for the item step.

### 2.2 Marginal Maximum Likelihood via EM (MML-EM)

**Reference:** Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of item parameters: Application of an EM algorithm. *Psychometrika*, 46(4), 443–459.

The **standard method**. Integrates out ability using a prior distribution (typically N(0,1)), avoiding the Neyman-Scott problem entirely.

**E-step:** For each quadrature node X_q (e.g., 40 Gauss-Hermite points):
- Compute the posterior weight of each examinee at X_q: $w_{jq} \propto L(\mathbf{u}_j | X_q) \cdot \phi(X_q)$.
- Accumulate expected counts: $\bar{r}_{iq} = \sum_j w_{jq} u_{ij}$ (expected correct responses to item $i$ at node $q$) and $\bar{f}_q = \sum_j w_{jq}$ (expected frequency at node $q$).

**M-step:** For each item, maximize
$$\sum_q \left[\bar{r}_{iq} \log P_i(X_q) + (\bar{f}_q - \bar{r}_{iq}) \log(1 - P_i(X_q))\right]$$
with respect to item parameters. Use Newton-Raphson (with analytic gradients/Hessian) or `scipy.optimize.minimize(method='L-BFGS-B')` with box constraints.

Iterate until log-likelihood change < ε (typically 1e-4) or max parameter change < 0.001.

**Quadrature:** 20–40 Gauss-Hermite nodes via `numpy.polynomial.hermite.hermgauss(n)`. Standard in all reference implementations.

**Pros:** Statistically consistent; the gold standard for 1PL/2PL. Well-understood convergence.

**Cons:** For 3PL, the c-parameter is notoriously hard to estimate — tends to drift to boundary. Needs constrained optimization or a Bayesian penalty on c. For 4PL, the d-parameter has the same issue.

**Feasibility:** **M** — needs analytic gradients of the ICC w.r.t. item parameters (new infrastructure), plus the EM loop. The quadrature is identical in structure to the Bayesian `QuadratureGrid` from card 01, but operates over examinees rather than items.

### 2.3 Conditional Maximum Likelihood (CML)

**Reference:** Andersen, E. B. (1972). The numerical solution of a set of conditional estimation equations. *Journal of the Royal Statistical Society, Series B*, 34, 42–54.

Only works for the **Rasch/1PL model**. The total score $r_j = \sum_i u_{ij}$ is a sufficient statistic for $\theta_j$, so conditioning on $\{r_j\}$ eliminates all ability parameters from the likelihood. What remains is a function of difficulty parameters only. Maximized directly via Newton-Raphson.

**Pros:** No distributional assumption on ability; theoretically clean; consistent and efficient for 1PL.

**Cons:** Inapplicable to 2PL/3PL/4PL. Examinees with perfect or zero scores are discarded. Computationally intensive for large examinee counts (requires elementary symmetric functions).

**Feasibility:** **S** — the conditional log-likelihood and its gradient have closed-form expressions. The elementary symmetric functions can be computed recursively. However, it's limited to 1PL, which reduces its practical value in a package that supports 4PL.

### 2.4 Bayesian MCMC

**References:** Albert, J. H. (1992). Bayesian estimation of normal ogive item response curves using Gibbs sampling. *Journal of Educational Statistics*, 17, 251–269. Patz, R. J., & Junker, B. W. (1999). A straightforward approach to Markov chain Monte Carlo methods for item response models. *Journal of Educational and Behavioral Statistics*, 24, 146–178.

Places priors on item parameters (a ~ LogNormal, b ~ Normal, c ~ Beta, d ~ Beta) and uses MCMC (Gibbs, Metropolis-Hastings) to sample from the joint posterior. The Patz-Junker MH-within-Gibbs approach is the standard.

**Pros:** Handles 3PL/4PL gracefully via priors that regularize c and d. Produces full posterior distributions on item parameters, not just point estimates. Useful for small samples.

**Cons:** Computationally expensive. Requires convergence diagnostics (Rhat, ESS). Much harder to implement correctly than EM.

**Feasibility:** **L** — significant implementation effort. Defer to a discussion card.

### 2.5 Item Fit Statistics

**References:** Wright, B. D., & Masters, G. N. (1982). *Rating Scale Analysis*. MESA Press. Yen, W. M. (1981). Using simulation results to choose a latent trait model. *Applied Psychological Measurement*, 5, 245–262.

Once items are calibrated, **fit statistics** evaluate whether the model fits each item's observed response pattern:

- **Infit mean-square (information-weighted):** Weighted average of squared standardized residuals, weighted by item information. Sensitive to unexpected responses near the item's difficulty. Expected value = 1.0.
- **Outfit mean-square (unweighted):** Unweighted average of squared standardized residuals. Sensitive to outlier responses far from the item's difficulty.
- **Point-biserial correlation:** Pearson correlation between the binary item response and the total score. Flags items with negative or near-zero discrimination.
- **Likelihood ratio tests:** Compare nested models (1PL vs 2PL, 2PL vs 3PL) via -2ΔLL with chi-square df = parameter difference.

**Feasibility:** **S** — mostly linear algebra on the response matrix and the fitted model. No optimization. Can be implemented as standalone functions.

---

## 3. Ranking

| Method | Impact | Feasibility | Priority | Rationale |
|---|:-:|:-:|:-:|---|
| **ICC gradients (infrastructure)** | 5 | H | **1** | Unblocks MML-EM and JML; analytic gradients are reusable everywhere |
| **MML-EM for 2PL** | 5 | M | **1** | The gold standard; first calibration method to ship |
| **MML-EM for 1PL** | 4 | S | **1** | Trivial extension of 2PL (fix a=1) |
| **MML-EM for 3PL** | 4 | M | **2** | c-parameter needs box constraints or penalty; extends the 2PL M-step |
| **Item fit statistics** | 4 | S | **2** | Essential diagnostic complement to calibration |
| **JML estimator** | 3 | H | **2** | Useful baseline and didactic tool; not recommended for production |
| **CML for Rasch** | 2 | S | **3** | Clean but 1PL-only; niche value |
| **Bayesian MCMC** | 3 | L | **4** | Heavy; defer unless 3PL/4PL calibration via EM proves insufficient |
| **MML-EM for 4PL** | 2 | L | **4** | d-parameter needs priors; rare in practice |

### Suggested milestones

**Milestone 1 (Priority 1):**
1. ICC analytic gradients w.r.t. item parameters — infrastructure card.
2. MML-EM for 1PL and 2PL — the core calibrator.

**Milestone 2 (Priority 2):**
3. MML-EM for 3PL — c-parameter with constraints/penalty.
4. Item fit statistics — infit, outfit, point-biserial.
5. JML estimator — baseline alternative.

**Milestone 3 (Priority 3+):**
6. CML for Rasch.
7. Bayesian MCMC (discussion).
8. 4PL extension (discussion).

---

## 4. Sources

- Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of item parameters: Application of an EM algorithm. *Psychometrika*, 46, 443–459.
- Birnbaum, A. (1968). Some latent trait models and their use in inferring an examinee's ability. In F. M. Lord & M. R. Novick (Eds.), *Statistical Theories of Mental Test Scores*. Addison-Wesley.
- Lord, F. M. (1980). *Applications of Item Response Theory to Practical Testing Problems*. Erlbaum.
- Andersen, E. B. (1972). The numerical solution of a set of conditional estimation equations. *JRSS-B*, 34, 42–54.
- Albert, J. H. (1992). Bayesian estimation of normal ogive item response curves using Gibbs sampling. *JES*, 17, 251–269.
- Patz, R. J., & Junker, B. W. (1999). A straightforward approach to MCMC for IRT. *JEBS*, 24, 146–178.
- Wright, B. D., & Masters, G. N. (1982). *Rating Scale Analysis*. MESA Press.
- Baker, F. B., & Kim, S.-H. (2004). *Item Response Theory: Parameter Estimation Techniques* (2nd ed.). Marcel Dekker.
- `girth` Python package: https://github.com/eribean/girth — MML-EM reference implementation.
- `mirt` R package: Chalmers, R. P. (2012). mirt: A multidimensional IRT package for the R environment. *Journal of Statistical Software*, 48(6), 1–29.

---

## 5. Card index

The cards are the other files in this folder; [the backlog index](../../README.md) lists them with
their dependencies. Calibration is outside catsim's scope (ADR-0001), so every card is parked for a
possible separate package.
