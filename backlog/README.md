# Backlog

Every piece of planned work on catsim is a card in this folder:

- new techniques, each with its full design;
- features and fixes;
- decisions waiting for the maintainer;
- research walkthroughs.

Pick a card and do its work in a PR, which deletes the card. To add work, copy
[`_template.md`](_template.md) into the folder of its area. The index below is generated from the
cards' frontmatter by [`_index.py`](_index.py), which a pre-commit hook runs, so commit the
regenerated README with the card.

The folders follow catsim's modules:

- `analysis/`: results, metrics and plots;
- `estimation/`, `selection/`, `stopping/`, `exposure/` and `initialization/`;
- `content/`: content-constrained CAT;
- `item-bank/`;
- `discovery/`: walkthroughs that find what catsim lacks;
- `maintenance/`;
- `scope/`: work outside the scope an ADR sets.

## The frontmatter

| Field | Values |
|---|---|
| `title` | What the card does, as a short imperative, or the question a decision card asks |
| `kind` | `technique` (a method, with its design), `feature`, `fix` (wrong behaviour or wrong docs), `docs`, `decision` (the maintainer chooses), `research` (a walkthrough or an audit), `maintenance` |
| `status` | `ready` (enough to start), `needs-research` (claims to verify, or a design to write, before work), `needs-decision` (the maintainer chooses first), `blocked` (by the cards in `blocked_by`, or by what the card says), `parked` (outside the scope an ADR sets; moving it into scope needs that ADR changed) |
| `priority` | `1` gives wrong results or blocks common research · `2` research use needed a substantial workaround · `3` useful, though nothing is blocked · `4` a later phase or a scope change |
| `effort` | `XS`, `S`, `M`, `L`, `XL` |
| `blocked_by` | Paths of other cards, relative to this folder and without `.md`, such as `[analysis/derived-results]` |
| `raised_by` | What surfaced the card: a walkthrough, a review or a PR, as short text without commas |

Priority follows the value to research, not compatibility: breaking an interface is fine when it
gives the better design (`AGENTS.md`).

<!-- backlog-index:start -->

49 cards: 27 ready, 6 needs-research, 9 needs-decision, 3 blocked, 4 parked.

## Waiting for the maintainer

| Card | Area | Priority |
|---|---|---|
| [Decide whether to reconsider the study layer that ADR-0016 ruled out](analysis/study-layer.md) | analysis | 2 |
| [Build a live-testing package on catsim's public API](discovery/live-package.md) | discovery | 2 |
| [Generate banks of a chosen shape, and document the scaling constant D where users look](item-bank/bank-generation.md) | item-bank | 2 |
| [Save and load simulation results, and the cost of unpickling them](analysis/saving-results.md) | analysis | 3 |
| [Give the worker tests a timeout, so a hang fails fast?](maintenance/worker-test-timeout.md) | maintenance | 3 |
| [One annotation for responses across the package?](maintenance/responses-annotation.md) | maintenance | 3 |
| [Simplify SimulationResult's constructor checks and other small duplication?](maintenance/result-constructor-checks.md) | maintenance | 3 |
| [When to delete the rescoping's reference baseline?](maintenance/reference-baseline-removal.md) | maintenance | 3 |
| [A stability and versioning policy?](maintenance/stability-policy.md) | maintenance | 4 |

## Priority 1

| Card | Area | Kind | Status | Effort | Blocked by | Raised by |
|---|---|---|---|---|---|---|
| [Empirical standard error, robust reported-SE summaries and interval coverage](analysis/precision-metrics.md) | analysis | feature | ready | M |  | Wang & Vispoel (1998) walkthrough; Babcock & Weiss (2012) walkthrough |
| [Say that examinee order changes a run with an exposure cap](analysis/exposure-cap-examinee-order.md) | analysis | docs | ready | XS |  | Chang & Ying (1999) walkthrough |
| [Subset, truncate and pool results, and a by-length table](analysis/derived-results.md) | analysis | feature | ready | M |  | Wang & Vispoel (1998) walkthrough; Chang & Ying (1999) walkthrough |
| [Constant-pattern rules for ML, and a hybrid estimator](estimation/constant-patterns-and-hybrid.md) | estimation | technique | needs-research | M |  | Wang & Vispoel (1998) walkthrough; Babcock & Weiss (2012) walkthrough |
| [Estimate-change stopper, the previous estimate in StepContext, and notebook 04's recipe](stopping/estimate-change-stopper.md) | stopping | fix | ready | M |  | Babcock & Weiss (2012) walkthrough |

## Priority 2

| Card | Area | Kind | Status | Effort | Blocked by | Raised by |
|---|---|---|---|---|---|---|
| [Constant-pattern, bound and stop-criterion columns in the result tables](analysis/table-columns.md) | analysis | feature | ready | S |  | Wang & Vispoel (1998) walkthrough; Babcock & Weiss (2012) walkthrough |
| [Correlation, test-length spread and a one-row run summary](analysis/run-summary.md) | analysis | feature | ready | S |  | Babcock & Weiss (2012) walkthrough |
| [Decide whether to reconsider the study layer that ADR-0016 ruled out](analysis/study-layer.md) | analysis | decision | needs-decision | S |  | Wang & Vispoel (1998) walkthrough; Chang & Ying (1999) walkthrough |
| [Exposure-evenness summaries (χ² index, unused and over-exposed shares)](analysis/exposure-evenness.md) | analysis | feature | ready | S |  | Chang & Ying (1999) walkthrough |
| [Audit catsim against catR and mirtCAT](discovery/catr-mirtcat-audit.md) | discovery | research | ready | M |  |  |
| [Build a live-testing package on catsim's public API](discovery/live-package.md) | discovery | research | needs-decision | XL |  |  |
| [Method-author walkthrough - implement a planned technique from its card](discovery/method-author-walkthrough.md) | discovery | research | ready | M |  |  |
| [Sympson-Hetter Exposure Control](exposure/sympson-hetter.md) | exposure | technique | ready | M |  | Chang & Ying (1999) walkthrough |
| [Generate banks of a chosen shape, and document the scaling constant D where users look](item-bank/bank-generation.md) | item-bank | decision | needs-decision | S |  | Wang & Vispoel (1998) walkthrough; Babcock & Weiss (2012) walkthrough; Chang & Ying (1999) walkthrough |
| [Minimum Information in Bank Stopper](stopping/min-info-bank-stopper.md) | stopping | technique | ready | S |  | Babcock & Weiss (2012) walkthrough |

## Priority 3

| Card | Area | Kind | Status | Effort | Blocked by | Raised by |
|---|---|---|---|---|---|---|
| [A public function that draws a generated examinee's answers](analysis/generated-answers.md) | analysis | feature | needs-research | S |  | notebook rewrite (#226) |
| [Comparable exposure histograms and more metrics in the plots](analysis/plot-options.md) | analysis | feature | ready | S |  | Chang & Ying (1999) walkthrough; Babcock & Weiss (2012) walkthrough |
| [Evaluate stopping rules post hoc on a maximum-length run](analysis/post-hoc-stopping.md) | analysis | feature | blocked | M | [derived-results](analysis/derived-results.md) | Babcock & Weiss (2012) walkthrough |
| [Save and load simulation results, and the cost of unpickling them](analysis/saving-results.md) | analysis | decision | needs-decision | M |  | Wang & Vispoel (1998) walkthrough; Babcock & Weiss (2012) walkthrough; Chang & Ying (1999) walkthrough |
| [Content-constrained selection (content balancing, WDM, Sympson-Hetter with content, shadow tests)](content/content-constrained-selection.md) | content | technique | needs-research | L |  | catR comparison; strategies literature review (April 2026) |
| [Choose the WLE's standard error (Warm's corrected SE, the sandwich SE)](estimation/wle-standard-errors.md) | estimation | technique | needs-research | S |  | Wang & Vispoel (1998) walkthrough |
| [Owen Sequential Bayesian Estimator](estimation/owen-estimator.md) | estimation | technique | ready | M |  |  |
| [Starting rules (catR `randomCAT`'s `start` list)](initialization/starting-rules.md) | initialization | technique | needs-research | M |  | catR comparison |
| [Build catsim on irtlib's item response theory and remove catsim's copies](maintenance/build-on-irtlib.md) | maintenance | maintenance | ready | L |  | irtlib ADR-0004 and the maintainer on 2026-11-27 |
| [Compute normal quantiles with scipy.special instead of importing scipy.stats](maintenance/normal-quantile-import.md) | maintenance | maintenance | ready | XS |  | import-time measurement of 2026-11-26 |
| [Editorial review of every document against the charter, and drift checks](maintenance/editorial-review.md) | maintenance | maintenance | ready | M |  | documentation charter |
| [Give the worker tests a timeout, so a hang fails fast?](maintenance/worker-test-timeout.md) | maintenance | decision | needs-decision | XS |  | parallel parts (#222) |
| [Link irtlib's names in the documentation once irtlib publishes its own](maintenance/irtlib-docs-links.md) | maintenance | docs | blocked | S | [build-on-irtlib](maintenance/build-on-irtlib.md) | maintainer on 2026-11-27 |
| [Make the stratified selectors' strata cache one attribute](maintenance/strata-cache-thread-race.md) | maintenance | fix | ready | XS |  | ADR audit (#225) |
| [One annotation for responses across the package?](maintenance/responses-annotation.md) | maintenance | decision | needs-decision | S |  | ADR audit (#225) |
| [Release the next version](maintenance/release-checklist.md) | maintenance | maintenance | blocked | M |  | rescoping plan |
| [Simplify SimulationResult's constructor checks and other small duplication?](maintenance/result-constructor-checks.md) | maintenance | decision | needs-decision | S |  | ADR audit (#225) |
| [Tighter and wider catR parity tests](maintenance/parity-extensions.md) | maintenance | maintenance | ready | M |  | catR parity whole tests (#223); technique pages (#226) |
| [When to delete the rescoping's reference baseline?](maintenance/reference-baseline-removal.md) | maintenance | decision | needs-decision | XS |  | reference baseline (#218) |
| [Check a-stratified selection's test size against the stopper](selection/astrat-test-length.md) | selection | feature | ready | S |  | Chang & Ying (1999) walkthrough |
| [Maximum Posterior Weighted Information Selector](selection/mpwi-selector.md) | selection | technique | ready | M |  |  |
| [Minimum Expected Posterior Variance Selector](selection/mepv-selector.md) | selection | technique | ready | M |  |  |
| [Posterior Kullback-Leibler Selector](selection/klp-selector.md) | selection | technique | ready | M |  |  |
| [thOpt selection (catR `criterion = "thOpt"`)](selection/thopt-selector.md) | selection | technique | needs-research | S |  | catR comparison |
| [Generalized Likelihood Ratio Stopper](stopping/glr-stopper.md) | stopping | technique | ready | M |  |  |
| [Kullback-Leibler Information Stopper](stopping/kl-stopper.md) | stopping | technique | ready | M |  |  |
| [Predicted-SEM Stopper](stopping/predicted-sem-stopper.md) | stopping | technique | ready | M |  |  |
| [Sequential Probability Ratio Test Stopper](stopping/sprt-stopper.md) | stopping | technique | ready | M |  |  |
| [Stochastic Curtailment Stopper](stopping/stochastic-curtailment-stopper.md) | stopping | technique | ready | M |  |  |

## Priority 4

| Card | Area | Kind | Status | Effort | Blocked by | Raised by |
|---|---|---|---|---|---|---|
| [A stability and versioning policy?](maintenance/stability-policy.md) | maintenance | decision | needs-decision | S |  | rescoping requirements |

## Parked: outside the scope an ADR sets

| Card | Area | Effort |
|---|---|---|
| [Cognitive Diagnosis CAT (CD-CAT) family](scope/cd-cat.md) | scope | XL |
| [Machine-learning / reinforcement-learning CAT selectors](scope/ml-rl-selectors.md) | scope | XL |
| [Multidimensional CAT (MCAT) family](scope/multidimensional-cat.md) | scope | XL |
| [Polytomous item response models](scope/polytomous-models.md) | scope | XL |

## Background

- [Discovery: walkthroughs that find what catsim lacks](discovery/README.md)

<!-- backlog-index:end -->

## Technique cards

A technique card carries the full design of a method catsim does not implement yet, in the sections
of [`_template.md`](_template.md). The sections from Motivation to References are those of an
implemented technique page; Tests, Open Questions and Implementation Guidance are for contributors.

The cards are MyST Markdown, like the documentation. GitHub does not render MyST roles and
directives (`{doc}`, `{eval-rst}`, `:::{note}`); they render once the technique has a page.
mdformat is kept off this folder (`.pre-commit-config.yaml`), because it rewrites that syntax and
breaks the `$$` math.

The technique cards from the former `specs/` folder had their citations, cited locations,
definitions and catR comparisons revalidated against their sources in November 2026. Their status
lines say when, and against which sources. Their implementation sketches follow the strategy
contract and were last run on 2026-11-26:

- each sketch was added at its proposed location to a copy of `src/catsim`;
- it was imported and run on a small bank, on hand-built contexts, and under `CatEngine` and
  `simulate`.

No test runs the sketches.

### Workflow

1. **Research** (`needs-research`): verify the card's claims against the primary sources (and
   catR's source, when the card cites catR), check what catsim already has, and write the technique
   sections from the template, following the citation convention below.
2. **Revalidate** the card with the checklist below before it is implemented or its references are
   published.
3. **Implement** the technique once its open questions are answered, with tests for its guarantees
   and for its "Tests" section, and a parity case against catR where catR implements the method
   (ADR-0020).
4. **Promote** the card into the documentation, in the same PR:
   - split a "Behavioral Contracts" section, which the cards from `specs/` still have, into
     Guarantees (what a user observes) and Tests (what only an implementation sees);
   - move its sections from Motivation to References to `sphinx/api/techniques/<family>/`, and add
     the page to the toctree of that family's `index.md`;
   - replace the status lines with the header of the technique-page template,
     `sphinx/api/techniques/_template.md`, pointing at the class with `{py:class}`;
   - add an "API Reference" section with the class's `autoclass`, as the template has;
   - move its references into `sphinx/references.bib`: add an entry for each source the bib lacks,
     turn each citation into a role with its location, and replace the References list with the
     template's filtered `{bibliography}`;
   - link the page from the class docstring ("For full algorithmic details, see :doc:`...`");
   - point the technique's redirect in `sphinx/conf.py`, if it has one, at the new page;
   - delete the card.

### Citation convention

- Cite each formula and design choice where it appears, with its location in the source (page,
  equation, table or figure).
- Cite only locations you have read in the source.
- Say where the card departs from its sources.
- Give each source its DOI. When the published version is not open, give an open copy ("Open
  copy: ...") or, if there is none, say so and name an open source that describes the method ("Not
  openly available; X (Year) describes the method.").

Cards write their citations by hand, e.g. "(Barrada et al., 2008, p. 498, Equation 12)", and give
each source in full in "References", because GitHub does not render the citation roles. Technique
pages generate theirs from `sphinx/references.bib`:

- a citation is a role with its location, `` {cite:p}`Barrada2008{p. 498, Equation 12}` ``;
- each source is a bib entry with its DOI and its note (`Open copy: \url{...}`, `Open access` or
  `Not openly available; ...`);
- the References section is a `{bibliography}` filtered to the page's citations, as
  `sphinx/api/techniques/_template.md` shows.

### Revalidation checklist

1. **Citations.** Authors, year, title, venue and DOI match the source. Every DOI resolves at
   doi.org: a publisher that refuses automated requests is fine, a 404 is not. Every entry has an
   open copy or the "Not openly available" note.
2. **Cited locations.** Every cited page, equation and table has been read in the source; a
   location not read is not cited. Pages are the printed ones, which can differ from the PDF's.
3. **Definitions and formulas** match the cited equations. Every departure from the source, or from
   catR, is stated explicitly.
4. **catR** is the reference implementation where it implements the method (KLP, MPWI, MEPV and
   the minimum-information stopper, among others). The card says where it agrees and where it
   departs, which the technique page puts in its closed "Comparison with catR" dropdown, and its
   tests name the parity cases the implementation adds (ADR-0020). A card never says what catR
   lacks.
5. **Implementation guidance** follows the strategy contract of ADR-0007:
   - per-step strategies take a `StepContext`, and estimators return an `Estimate`, whose
     standard error is positive or `math.inf`;
   - an exposure control implements `BaseExposureControl.accept(context, item)`, and the engine
     keeps its declines in the session's `excluded_items`;
   - selectors break exact ties with `pick_best` or `best_pool` (ADR-0017), and sum scores over a
     grid's node axis rather than with a matrix product, which can round identical items' columns
     differently;
   - an override carries `@override`, from `typing`;
   - strategy objects hold configuration and caches derived from the item bank, never per-session
     state, because one engine serves many sessions (ADR-0007, ADR-0008);
   - each stopper is a single criterion, combined with others by `StopRule`;
   - no `[:, :4]` slices of the item matrix.
6. **Status.** The top of the card says when it was last validated and against which sources.
