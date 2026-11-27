# Literature Review: New CAT Strategies for catsim

**Status:** Draft v3 (agent-written in April 2026; trimmed 2026-10-11)
**Goal:** Candidate CAT methods that catsim does not implement yet.

> **Read with care.** A coding agent wrote this survey, and nobody has checked it against
> its sources: the descriptions, citation-count proxies and rankings are unverified. Methods
> implemented since then were removed from it. They are documented, and checked against their
> sources, on the Sphinx technique pages; start from the Study Guide (`sphinx/study-guide.md`).

---

## 1. Where things stand

**Implemented** on feature branches (not merged yet), with checked Sphinx pages:

| Card | Technique | Branch |
|--:|---|---|
| 01 | Bayesian infrastructure | `feat/bayesian-infrastructure` |
| 06 | EAP estimator | `feat/eap-estimator` |
| 07 | MAP estimator | `feat/map-estimator` |
| 02 | Warm's WLE | `feat/warm-wle` |
| 08, 09 | MEI and MLWI selectors | `feat/expected-info-selectors` |
| 03 | KL selector | `feat/kl-selector` |
| 04, 05 | Progressive and proportional selectors | `feat/progressive-selectors` |

**Planned:** the other 11 cards became "Planned" Sphinx pages under
`sphinx/api/techniques/` (commit `9d801e5`, 2026-04-09); read and refine them there. Their
formulas and references have not been checked. They are the predicted-SEM, minimum
bank information, KL, SPRT, GLR and stochastic-curtailment stoppers, Sympson–Hetter exposure
control, the MPWI, KLP and MEPV selectors, and Owen's estimator.

**Deferred:** the design questions listed in §7, each now a card in this folder.

---

## 2. Findings

Where possible, each item is tagged with:
- **Year** of the original / current citation.
- **Citation proxy** (N ≈ Google Scholar citation magnitude, coarse).
- **Catsim feasibility** — H (add in a day), M (needs new infrastructure like EAP quadrature), L (heavy new dependency or architectural change).

### 2.1 Item-selection methods catsim is missing

Grouped into families that share infrastructure.

#### A. Information-variant selectors (no new infrastructure)

| Method | Author / Year | What it does | Feasibility |
|---|---|---|---|
| **Posterior KL (KLP)** | Chang & Ying 1996 variant; catR implementation | Weights KL integrand by the current posterior over theta | M — needs a posterior (EAP) |
| **Maximum Posterior Weighted Information (MPWI)** | van der Linden 1998; Barrada et al. 2009 | Fisher info weighted by the posterior | M — needs a posterior |
| **Minimum Expected Posterior Variance (MEPV)** | van der Linden 1998 | Chooses item minimizing expected posterior variance after the next response | M — needs a posterior |
| **Shannon entropy selection** | Tatsuoka 2002 | Minimize posterior entropy; prominent in CD-CAT but generalizes to IRT | M |

Note: Magis and Raîche (2012, p. 10) report that Choi and Swartz (2009) showed the maximum expected
posterior weighted information (MEPWI) criterion to be equivalent to MPWI.

#### B. Exposure-control item selectors (randomization families)

| Method | Author / Year | What it does | Feasibility |
|---|---|---|---|
| **Sympson–Hetter exposure control** | Sympson & Hetter 1985 | Two-phase: simulate exposure probabilities offline, then use them online to decide whether to administer a selected item | M — needs an offline calibration run/API |
| **Sympson–Hetter with Content (SHC) / SHT** | Kingsbury & Zara 1989; Stocking & Lewis 1998 | Adds content balancing to SH | M |

#### C. Constrained / multi-objective CAT

| Method | Author / Year | What it does | Feasibility |
|---|---|---|---|
| **Weighted Deviations Method (WDM)** | Stocking & Swanson 1993 | Penalizes constraint violations inside the objective; the classical content-balancing baseline | M — needs item metadata (content codes, limits) in the item bank |
| **Shadow-Test Approach (STA)** | van der Linden & Reese 1998; STA review Veldkamp 2021 | Each step, assemble a full-length shadow test satisfying all constraints via MIP, administer its best item | **L** — requires an ILP/MIP solver (e.g., `pulp`, `cvxpy`). Real package `RSCAT` uses CPLEX |

#### D. CD-CAT item selectors (require a diagnostic classification model)

Flagged for completeness; all are **L** feasibility because catsim has no DCM/DINA layer.

- **PWKL / MPWKL** — Cheng 2009; Kaplan, de la Torre, Barrada 2015.
- **GDI / GDIP / MGDI** — de la Torre & Chiu 2016.
- **Jensen–Shannon Divergence (JSD)** — He & Tan 2020 (MC-DINA).
- **SWDGDI / CP_SWDGDI** — attribute-balanced indices (Wang 2019).
- **Heuristic-search CD-CAT** — *Behavior Research Methods* 2023.

#### E. ML / RL selectors (post-2018)

Flagged for completeness; all are **L** — they require training data, neural net runtime, and a state interface larger than catsim currently exposes.

- **POMDP / DQN CAT** — Chen et al. 2018; Nurakhmetov 2019; Li et al. 2023.
- **NCAT** — Zhuang et al. 2022.
- **BOBCAT** (meta-learning) — Ghosh & Lan 2021.
- **BECAT** (subset selection) — Zhuang et al. 2023.
- **GMOCAT** (multi-objective) — Wang et al. 2023.
- **Expected Model Change (EMC)** — Bi et al. 2020.
- **Uncertainty-based** — Veldkamp & Verschoor 2019.

Reference: Zhuang et al., *Survey of Computerized Adaptive Testing: A Machine Learning Perspective*, arXiv:2404.00712, 2024 — comprehensive taxonomy used to cross-check this section.

### 2.2 Ability-estimation methods catsim is missing

EAP, MAP and WLE are implemented (see §1). Remaining candidates:

| Method | Author / Year | Notes | Feasibility |
|---|---|---|---|
| **Owen's sequential Bayesian** | Owen 1975 | Analytic Gaussian update; historically important for CAT starting rules and classical Bayesian CAT. Outperformed by EAP in modern comparisons but cheap and historically canonical. | H |
| **Multiple Imputation (MI) / Plausible Values** | Mislevy 1991; used in PISA/NAEP | Stabilizes early-test estimates by drawing plausible values from the posterior. | M |

### 2.3 Stopping criteria catsim is missing

| Method | Author / Year | Notes | Feasibility |
|---|---|---|---|
| **Predicted SEM / Global-information stop** | Choi, Grady, Dodd 2011 *EPM* | Stop when no remaining item in the bank can reduce the SEM below threshold. Fixes the "chase-the-target" issue of SE-threshold stopping. | **H** |
| **Minimum information in bank** | Choi et al. 2011 | Sibling criterion: stop when max item info ≤ small constant | H |
| **KL-based stopping** | Choi et al. 2011 | Stop when KL info of best remaining item is small | H |
| **SPRT classification stopping** | Reckase 1983; Weissman 2007 | Sequential Probability Ratio Test for mastery/non-mastery categorical decisions — whole class of stoppers catsim has none of | H |
| **Generalized Likelihood Ratio (GLR)** | Bartroff, Finkelman, Lai 2008 | SPRT refinement, used in licensure tests | H |
| **Stochastic curtailment** | Finkelman 2008 | Early stop when continuing cannot change the classification decision | M |

### 2.4 Initialization catsim is missing

| Method | Notes | Feasibility |
|---|---|---|
| **BestPrior / grid search on prior** | Uses EAP on a 0-response "prior" to start in a sensible spot | H (depends on EAP) |
| **Robbins–Monro start** | Classical stochastic approximation for adaptive level setup | H |

---

## 3. Ranking

Scoring (1–5, 5 = highest):
- **Impact** — prevalence in modern CAT pipelines and review papers.
- **Authors** — centrality of the authors in CAT methodology (Chang, van der Linden, Barrada, Magis, Warm, Bock, Cheng, Zhuang).
- **Feasibility** — ease of landing in catsim without new heavy dependencies or architectural change.
- **Priority** — composite; 1 = ship next, 5 = defer.

| Candidate | Impact | Authors | Feasibility | Priority | Rationale |
|---|:-:|:-:|:-:|:-:|---|
| **Predicted-SEM / global-info stopper** | 4 | 4 | 5 | **2** | Principled fix for the SE-threshold "chase" problem. |
| **MPWI (posterior-weighted info)** | 4 | 5 | 4 | **2** | Needs EAP posterior; standard catR baseline. |
| **Posterior KL (KLP)** | 4 | 5 | 4 | **2** | Same story as MPWI. |
| **Sympson–Hetter exposure control** | 5 | 5 | 3 | **2** | Operational gold standard for exposure; needs an offline calibration API. |
| **SPRT classification stopper** | 4 | 4 | 5 | **3** | Opens catsim to mastery-test use cases; very little code. |
| **MEPV (min expected posterior variance)** | 3 | 5 | 3 | **3** | Redundant with MEI/MPWI gains; include for completeness later. |
| **Owen sequential Bayesian** | 2 | 4 | 5 | **3** | Historical; marginal over EAP but cheap to add. |
| **Weighted Deviations Method (content balancing)** | 4 | 4 | 3 | **3** | Requires item-bank metadata scheme; useful but a small refactor. |
| **Stochastic curtailment stopping** | 3 | 3 | 4 | **3** | Nice companion to SPRT. |
| **Shadow-Test Approach (STA)** | 5 | 5 | 2 | **4** | Gold standard for constrained CAT but drags in an ILP dependency. |
| **CD-CAT family (PWKL, GDI, JSD, etc.)** | 4 | 4 | 1 | **4** | Requires an entire DCM/DINA estimation layer catsim doesn't have. |
| **Deep RL / BOBCAT / NCAT / DQN-CAT** | 3 | 3 | 1 | **5** | Research-grade; training cost, neural deps, scope creep. |
| **MCAT selectors** | 4 | 5 | 1 | **5** | Out of scope without multidimensional IRT in `irt.py`. |

The original first milestone (WLE, EAP, MAP, KL, progressive) and part of the second (MEI, MLWI)
are implemented; the remaining Priority 1–3 items are the planned Sphinx pages listed in §1.

---

## 4. Cross-references

- **`catR`** (Magis & Raîche; Magis & Barrada 2017) — de facto reference implementation. Has: MFI, bOpt, KL, KLP, MLWI, MPWI, MEI, MEPV, Progressive, Proportional, Random, thOpt, GDI/GDIP (CD-CAT), Sympson–Hetter exposure control, MLE/BM/EAP/WLE estimators, 6 stopping rules including predicted SE, minimum information, classification SPRT.
- **`mirtCAT`** (Chalmers) — MIRT + unidimensional, ships EAP/MAP/MLE/WLE/plausible values, shadow testing via `lpSolveAPI`, content constraints.
- **`RSCAT`** — R shadow-test implementation, demonstrates ILP-based constrained CAT.
- **Survey: Zhuang et al., arXiv:2404.00712, 2024** — "Survey of Computerized Adaptive Testing: A Machine Learning Perspective". Used for the ML-side taxonomy in §2.1.E.
- **Review: Veldkamp 2021 / 2022 special issue of *Behaviormetrika*** — shadow-test approach review.

---

## 5. Open questions for the maintainer

1. **Scope boundary**: is catsim staying strictly unidimensional, or is MCAT in the longer-term roadmap? This determines whether the item-bank schema needs generalizing now.
2. **Content metadata**: would you accept a small optional schema (e.g. content code, enemy sets, word count) on `ItemBank` to support content balancing and Sympson–Hetter-with-content? Otherwise, several high-impact methods are blocked.
3. **Optional heavy deps**: are you open to an `extras_require` install (`catsim[shadow]`) that pulls in an ILP solver for the shadow-test approach? Without it, constrained CAT cannot be done properly.
4. **CD-CAT**: high-impact research area, but requires an entirely new estimation layer (DINA / G-DINA). Is a future `catsim.diagnostic` subpackage on the table, or out of scope?

---

## 6. Sources

Unverified, like the rest of this survey. Sources for implemented methods are on their Sphinx pages.

- Zhuang, Y. et al. (2024). *Survey of Computerized Adaptive Testing: A Machine Learning Perspective*. arXiv:2404.00712. https://arxiv.org/html/2404.00712v2
- Chang, H.-H., & Ying, Z. (1996). A global information approach to computerized adaptive testing. *Applied Psychological Measurement*, 20(3), 213–229.
- Warm, T. (1989). Weighted likelihood estimation of ability in item response theory. *Psychometrika*, 54, 427–450. https://link.springer.com/article/10.1007/BF02294627
- Bock, R. D., & Mislevy, R. J. (1982). Adaptive EAP estimation of ability in a microcomputer environment. https://conservancy.umn.edu/bitstreams/26802601-a1af-4f90-8452-0ec288043608/download
- van der Linden, W. J. (1998). Bayesian item selection criteria for adaptive testing. *Psychometrika*, 63, 201–216.
- van der Linden, W. J., & Reese, L. M. (1998). A model for optimal constrained adaptive testing. *Applied Psychological Measurement*, 22, 259–270.
- Veerkamp, W. J. J., & Berger, M. P. F. (1997). Some new item selection criteria for adaptive testing. *Journal of Educational and Behavioral Statistics*, 22, 203–226.
- Revuelta, J., & Ponsoda, V. (1998). A comparison of item exposure control methods in computerized adaptive testing. *Journal of Educational Measurement*, 35, 311–327.
- Sympson, J. B., & Hetter, R. D. (1985). Controlling item-exposure rates in computerized adaptive testing.
- Stocking, M. L., & Swanson, L. (1993). A method for severely constrained item selection in adaptive testing. *Applied Psychological Measurement*.
- Choi, S. W., Grady, M. W., & Dodd, B. G. (2011). A new stopping rule for CAT. *Educational and Psychological Measurement*, 71, 37–53.
- Reckase, M. D. (1983). A procedure for decision making using tailored testing. (SPRT classification)
- Weissman, A. (2007). Mutual information item selection in adaptive classification testing. *EPM*. https://journals.sagepub.com/doi/10.1177/0013164406288164
- Cheng, Y. (2009). When cognitive diagnosis meets CAT: CD-CAT. *Psychometrika*, 74, 619–632.
- Kaplan, M., de la Torre, J., & Barrada, J. R. (2015). New item selection methods for CD-CAT. *Applied Psychological Measurement*. https://pmc.ncbi.nlm.nih.gov/articles/PMC5978538/
- de la Torre, J., & Chiu, C.-Y. (2016). A general method of empirical Q-matrix validation.
- Veldkamp, B. P. (2022). Review of the shadow-test approach to adaptive testing. *Behaviormetrika*. https://link.springer.com/article/10.1007/s41237-021-00150-y
- Choi, S. W., Moellering, K. T., Li, J., & van der Linden, W. J. (2016). Optimal reassembly of shadow tests in CAT. https://pmc.ncbi.nlm.nih.gov/articles/PMC5978635/
- Chen, Y. et al. / Li, X. et al. (2023). An adaptive testing item selection strategy via a deep reinforcement learning approach. *Behavior Research Methods*. https://link.springer.com/article/10.3758/s13428-024-02498-x
- Zhuang, Y. et al. (2022). NCAT: Neural CAT. (Zhuang et al. 2022b.)
- Ghosh, A., & Lan, A. (2021). BOBCAT: Bilevel Optimization-Based CAT.
- Ma, C. et al. (2023). Novel item selection strategies for CD-CAT: a heuristic search framework. *Behavior Research Methods*. https://link.springer.com/article/10.3758/s13428-023-02228-9
- catR package manual, CRAN. https://cran.r-project.org/web/packages/catR/catR.pdf
- catR `nextItem` source. https://github.com/cran/catR/blob/master/R/nextItem.R
- RSCAT (R shadow-test approach). https://cran.r-project.org/web/packages/RSCAT/index.html
- Components of the item selection algorithm in CAT (review). https://pmc.ncbi.nlm.nih.gov/articles/PMC5968224/
- Prior distribution and entropy in CAT ability estimation through MAP or EAP (2023). https://www.mdpi.com/1099-4300/25/1/50
- ALICAT: customized item selection in CAT. https://journal-bcs.springeropen.com/articles/10.1186/s13173-020-00098-z
- Warm's WLE discussion: https://www.rasch.org/rmt/rmt231d.htm
- Comparison of CAT item selection criteria for polytomous items (MFI, MLWI, MPWI, MEI, MEPV). https://pmc.ncbi.nlm.nih.gov/articles/PMC2791416/
- catR `MEI` documentation. https://rdrr.io/cran/catR/man/MEI.html
- catR `MWI` documentation. https://rdrr.io/cran/catR/man/MWI.html

_(Append new URLs here as the review continues.)_

---

## 7. Discussion cards

Cards that need a scope or design decision before implementation.

| Card | Why it's deferred |
|---|---|
| [item-bank-metadata-schema](content/item-bank-metadata-schema.md) | Cross-cutting design: content codes, enemy sets, word counts on `ItemBank`. Blocks WDM and SH-with-content. |
| [weighted-deviations](content/weighted-deviations.md) | Needs item-bank metadata schema. |
| [sympson-hetter-with-content](exposure/sympson-hetter-with-content.md) | Needs item-bank metadata schema. |
| [shadow-test](content/shadow-test.md) | Requires an ILP/MIP solver as an optional extras install. |
| [cd-cat](scope/cd-cat.md) | Requires a whole DCM/DINA layer not currently in catsim. |
| [multidimensional-cat](scope/multidimensional-cat.md) | Requires multidimensional IRT in `irt.py`. |
| [ml-rl-selectors](scope/ml-rl-selectors.md) | Research-grade; training/data/neural deps; scope creep risk. |
