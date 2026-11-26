# Documentation charter

These rules govern every document in catsim: the Sphinx pages under `sphinx/`, the docstrings they
render, `README.md`, the notebooks, `specs/`, `CONTRIBUTING.md` and the GitHub release notes. They
are instructions for coding agents and contributors, not website content: `sphinx/conf.py` excludes
this file from the build. Read it before writing or reviewing documentation of any kind.

## Audiences

In order of priority:

1. **Researchers** running simulation studies to compare selection, estimation and stopping methods.
2. **Method authors** writing custom strategies.
3. **Integrators** embedding catsim's computations in a live testing application. catsim does not
   serve tests (ADR-0001): integrators get one guide on the boundary, "Embedding catsim", and no
   recipes for storing, resuming or serving sessions.
4. **Contributors** changing catsim itself.

Each kind of document below names its audience. Write for that reader only. Researchers know IRT
and may not know software design; integrators know Python and may not know IRT.

## Rules for every document

### Voice

- Write in the present tense. Describe what is, never what changed.
- Do not write "now", "no longer", "replaces", "instead of" or "previously" about an earlier
  version, "new" for something just added, card IDs, or PR and issue numbers. The ban covers every
  page, docstring and notebook. The exceptions are the Context section of an ADR and the 1.0 notice.
- Put plain language first. Use a symbol only after the text has introduced it in words.
- Describe what users observe, not how catsim computes it, unless the computation is the point, as
  on a technique page.
- Use the terms and symbols of `terms-and-notation.md` (see "Terms and notation" below).
- In code, import each object from its shortest public path: `from catsim import CatEngine`,
  `from catsim.selection import MaxInfoSelector`, not from the module file that defines it
  (`catsim.engine`, `catsim.selection.information`).

### catR and other packages

catsim's methods rest on their published sources, not on another package. catR, the reference
implementation the parity tests compare catsim with, appears in three places only:

- each technique page's closed "Comparison with catR" dropdown, the page's only mention of catR;
- the catR ADR, which says why catR is the reference;
- the parity section of `CONTRIBUTING.md`, on running and recording the parity tests.

An ADR may name another package in a sentence where its practice was one of the forces behind the
decision. No document says what another package lacks, or presents a catsim choice as an
improvement on another package's. Citing a paper by catR's authors as the source of a formula is a
citation, not a mention: cite it like any other source.

### History

- The GitHub release notes are the only place for "what changed". Each release gets a hand-written
  summary (highlights, breaking changes, results that change for the same seed), followed by
  GitHub's generated list of pull requests. There is no CHANGELOG file and no migration page.
- The 1.0 notice, in the README and on the landing page, is a warning: 0.x code needs changes and
  seeded results differ. It points to `catsim<1` and the 1.0.0 release notes, and lists no changes:
  those are in the release notes.

### Where a formula lives

- A **method's** formulas (the definition of an estimator, a selector or a stopper) live on its
  technique page, in the Definition section, with the source's equation and page.
- The method's **docstring** says in one sentence what the method computes, in words, then gives
  the call contract and links the page: "The root of Warm's modified score equation, which removes
  the MLE's first-order bias; see the WLE page."
- A **quantity's** formula (the item characteristic curve, information, the log-likelihood, test
  information, the standard error) lives in the docstring of its `catsim.irt` function. Those
  functions have no technique page, and their return value is the formula. Technique pages refer to
  the function and do not re-derive the quantity.
- The introduction shows $P(\theta)$ and $I(\theta)$ to teach the concepts. It is not a second
  reference.
- No formula appears twice on one rendered page. A technique page renders its class's docstring
  under "API Reference", so a class docstring that repeats the Definition shows the math twice.

### Figures

Figures in the Sphinx pages and docstrings come from the plot directive, which draws them as SVG in
the documentation's style, `_static/figures.mplstyle`, applied in `conf.py`. Examples set no style of
their own. The notebooks draw in matplotlib's default style.

### Implementation detail in user pages

Mention an implementation detail only where it changes what users observe: estimates kept within
$[-6, 6]$, random tie-breaking, the sessions of a simulation run one after another. Describe it in
the user's terms and link the ADR for the reason. Leave out the rest: internal types and their
fields, helpers, validation rules, invariants, how the random generator reaches each strategy.

### One home per topic

Each topic has one home. Other documents say what they need in a sentence and link the home; they
do not re-explain it. The map names ADRs by topic.

| Topic | Home | Related documents |
|---|---|---|
| What CAT and IRT are; the CAT loop | Introduction | README, landing page |
| Terms and symbols | Terms and notation | every page |
| Install, quickstart, citation | README | landing page |
| Running a simulation study; reading `SimulationResult` | "Running simulations" guide | the getting-started notebook |
| Comparing strategies over replications | "Running simulations" guide | the comparing-strategies notebook |
| A test step by step: the `start`/`select`/`answer` loop, stop reasons | "Embedding catsim" guide | the step-by-step notebook |
| What catsim checks in its inputs, and what the application owns (sessions, the item shown, exposure counts, concurrency) | "Embedding catsim" guide | the explicit-inputs and input-checking ADRs for why |
| Seeds, each examinee's streams, parts and workers | "Reproducibility" guide | the random-numbers and parallel-runs ADRs for why |
| Item names; reading and writing a bank as CSV | `ItemBank` docstring | "Running simulations" |
| Custom strategies: `StepContext`, `Estimate`, the base classes and their helpers | "Extending catsim" guide | the custom-strategies notebook |
| How a simulation counts exposure rates (their denominator) | `simulate` docstring | the exposure-rates ADR for why; "Embedding catsim" for the rates an application passes |
| A method's definition, parameters and behaviour | its technique page | its docstring (one link) |
| Each estimator's standard error | the estimator's technique page | the estimation index compares them |
| The WLE's standard error against Warm's corrected one | WLE technique page | |
| Estimates within $[-6, 6]$; all-correct and all-incorrect patterns | estimation index page | the ability-scale ADR for why |
| Combining stoppers with `StopRule` | stopping index page | each stopper's page |
| Tie-breaking and exposure control in selection | selection index page | the tie-breaking ADR for why |
| A quantity's formula | its `catsim.irt` function's docstring | technique pages |
| A metric's formula (bias, MSE, RMSE, overlap rate) | the metric function's docstring | "Running simulations" |
| Planned methods | Planned techniques page | their specs in `specs/` |
| How to contribute; the gate; the release process | `CONTRIBUTING.md` | |
| How catsim is checked against catR (the parity tests) | `CONTRIBUTING.md` | the catR ADR for why; each technique page's catR dropdown states its departures |
| Why the architecture is shaped this way | the ADR | guides, technique pages |
| What changed in a release | the GitHub release notes | the 1.0 notice |

Some homes in this map are still to be written (the guides, the Planned techniques page). Until a
home exists, leave the material where it is and do not start a second copy.

## Kinds of documents

Each kind has one job, audience and level. "Not here" names where the excluded material goes.

### README

- **Job:** tell a visitor on GitHub or PyPI what catsim is and get them running: what it is (two
  sentences); install (`pip install catsim`, and `catsim[plot]` for the plots); a quickstart of
  about 15 lines; where to go next (guides, techniques, tutorials, API reference, ADRs); how to cite;
  related packages for calibrating items.
- **Audience and level:** every newcomer; introductory.
- **Not here:** developer setup (`CONTRIBUTING.md`), a CAT primer (the introduction), design notes
  (ADRs).
- **Length:** about 350-450 words.
- **Review:**
  - Does the quickstart run as written against the current release?
  - Does every link and image use an absolute URL, so that it works on PyPI too?
  - Is anything there only for contributors?

### Landing page (`sphinx/index.md`)

- **Job:** the website's front door, with an entry path for each audience: running a simulation
  study ("Running simulations"), writing a strategy ("Extending catsim") and embedding catsim's
  computations in an application ("Embedding catsim"). It links the guides, the techniques, the
  tutorials and the API reference. It has its own source, separate from the README.
- **Audience and level:** researchers, method authors and integrators; introductory.
- **Not here:** install minutiae and badges (README).
- **Length:** one screen, about 150-300 words.
- **Review:**
  - Can each audience reach its first page in one click?
  - Does it restate the README or a guide rather than link it?

### Concept pages (introduction, terms and notation)

- **Job:** the introduction is the CAT and IRT primer: what a test measures, the logistic models,
  the CAT loop (start, select, respond, estimate, stop), and a short map of catsim's parts (item
  bank, engine and session, strategies, `simulate` and its result) that links the guides. "Terms and notation" gives one term per concept
  and the symbols.
- **Audience and level:** researchers and integrators new to CAT; introductory.
- **Belongs:** concepts, $P(\theta)$ and $I(\theta)$ as illustrations, figures, textbook citations.
- **Not here:** internal types and their fields (`StepContext`), helpers (`pick_best`), validation
  rules and invariants, how exposure rates are counted, custom strategies ("Extending catsim").
- **Length:** the introduction reads in about 15 minutes (at most about 2,000 words); "Terms and
  notation" is as long as its tables need.
- **Review:**
  - Could a reader who has never seen catsim's code follow every sentence?
  - Is every symbol introduced in words before it is used?
  - Do the terms and symbols match "Terms and notation"?

### Guides

"Running simulations", "Reproducibility", "Extending catsim" and "Embedding catsim".

- **Job:** how to do one task end to end, with code the reader can copy.
- **Audience and level:** "Running simulations" and "Reproducibility" serve researchers and are
  intermediate. "Extending catsim" serves method authors and is advanced. "Embedding catsim" serves
  integrators and is intermediate: the step loop, what catsim checks and what the application owns,
  with no recipes for storing or serving sessions.
- **Belongs:** the steps in order; runnable code; what the output means; the choices the reader
  makes, each linked to the page that explains it; the pitfalls readers meet, such as a precision
  target the bank cannot reach, or an overlap rate that is `None` for tests of different lengths.
- **Not here:** design rationale (link the ADR), history, a method's definition (link its technique
  page), every parameter of a class (link the API reference).
- **Length:** about 800-1,500 words each.
- **Review:**
  - Can the reader finish the task from this page alone, following links only for depth?
  - Does every code block run as written, in order?
  - Does it re-explain a topic whose home is elsewhere?

### Technique pages (`sphinx/api/techniques/<family>/`)

- **Job:** a method as a researcher needs it: what it does and when to choose it, its formula, its
  parameters, how it behaves and compares with the alternatives, what it guarantees, its sources.
- **Audience and level:** researchers; intermediate to advanced in psychometrics.
- **Template:** `sphinx/api/techniques/_template.md`: Motivation, Definition (with an optional
  Computation subsection on how catsim solves it numerically), Parameters, Behavior and Comparison,
  Guarantees, References, API Reference.
- **Belongs:**
  - the method's formulas, each cited where it appears, with the source's page and equation, as the
    citation convention in `specs/README.md` sets out;
  - every departure from the sources, stated;
  - in a closed "Comparison with catR" dropdown at the end of Behavior and Comparison, the matching
    catR function and settings, what the parity tests find and each departure from catR; no
    dropdown when catR has no counterpart;
  - the parameter ranges the literature recommends;
  - Guarantees: properties a user relies on and can observe ("never repeats an item", "falls back
    when every item is over the cap", "finite for all-correct patterns").
- **Not here:**
  - software internals: context fields, caches, private helpers;
  - test specifications ("any correct implementation must..."): properties only an implementation
    sees ("keeps no per-session state", "the evaluation count is the grid's node count") go to
    docstrings and tests;
  - a quantity's formula (refer to its `catsim.irt` function);
  - history.
- **References:** generated from `sphinx/references.bib`, never typed on the page.
  - Cite with `` {cite:t}`key{p. N}` `` in a sentence and `` {cite:p}`key{p. N}` `` in parentheses,
    one role per source when it carries a location (a role with several keys sorts them); inside an
    eval-rst block use the reST role, `` :cite:t:`key` ``. The `:Sources:` field cites the keys.
  - Add each source to `references.bib` with its DOI and a note: `Open copy: \url{...}`,
    `Open access`, or `Not openly available; X (Year) describes the method`. Brace the title words
    whose case matters (`{R}`, `{catR}`, `{IRT}`, `{Bayesian}`, proper names): the style lowercases
    titles.
    Enter a chapter of an edited book as `@inproceedings`: the style prints no note for `@incollection`.
  - End the References section with a `{bibliography}` directive filtered to the page,
    `:filter: docname in docnames`; a source the page lists without citing it in the text goes in
    the directive's content.
- **Length:** about 400-1,000 words before the references.
- **Review:**
  - Is every formula cited to a page or equation that someone has read?
  - Does every source the page relies on have a bib entry with its DOI and its note?
  - Does the page say where catsim departs from its sources?
  - Is catR mentioned only inside the page's dropdown?
  - Is every guarantee something a user can observe?
  - Does the class docstring rendered under "API Reference" repeat a formula from the page?

### API reference (docstrings)

- **Job:** how to call an object correctly: the call contract.
- **Audience and level:** anyone calling the object; precise and brief.
- **Belongs:** in numpy style:
  - a one-line summary;
  - the behaviour a caller relies on, implementation-level guarantees included;
  - Parameters (in the class docstring only, not again in `__init__`), Returns and Raises;
  - a short example where it helps, as a doctest (pytest runs them, so they must pass);
  - one link: the technique page for a strategy, otherwise the guide that covers the object, if
    any.
- **Not here:** a method's math (its technique page), design rationale (the linked page cites the
  ADR), history, comparisons with other packages such as catR (the technique page's catR dropdown).
- **Length:** about 80 words of prose, counting the text before the first numpydoc section. Only the
  core types and functions (`CatEngine`, `Session`, `simulate`, `SimulationResult`, `ItemBank`,
  `StepContext`) may run longer, and only with what a caller needs.
- **Review:**
  - Could a caller use the object correctly from the docstring alone?
  - Is any sentence rationale, history or method math?
  - Is the prose within budget?

### ADRs (`sphinx/adr/`)

- **Job:** why the architecture is shaped this way.
- **Audience and level:** contributors and method authors; advanced.
- **Template:** `sphinx/adr/_template.md`: Status, Context, Decision, Consequences, Alternatives
  Considered. Add each new ADR to the toctree in `sphinx/architecture_decisions.md`.
- **Belongs:** the forces behind the decision (Context), the rule itself (Decision), what follows from
  it, good and bad (Consequences), the options rejected and why (Alternatives Considered).
- **Not here:** reference tables such as a type's fields or each estimator's standard error
  (docstrings and technique pages), study data (the technique page, summarized), release-note
  material (the release notes), project process (`CONTRIBUTING.md`).
- **Rules:**
  - Write in a timeless voice. Only Context may describe the code before the decision.
  - The Status line is dated: "Accepted 2026-11-24".
  - Show code only to state an interface, in a few lines.
  - From 1.0.0 on, an accepted ADR is not rewritten: a changed decision gets a new ADR that
    supersedes it, and the old one's Status becomes "Superseded YYYY-MM-DD by ADR-NNNN".
- **Length:** 300-600 words.
- **Review:**
  - Does it record a decision, or does it describe code?
  - Would a reader who disagrees find the rejected options and the reasons?
  - Is anything in it a table, a measurement or a note for the release?

### Notebooks (`notebooks/*.ipynb`)

- **Job:** learning by doing, as a ladder by level and audience: getting started (beginner
  researcher), comparing strategies (intermediate researcher), a test step by step (researcher or
  integrator), custom strategies (method author, advanced).
- **Audience and level:** set per notebook, stated in its first cell.
- **Belongs:**
  - one example of each technique family, not every class (the technique pages list them);
  - prose that says what each output and figure means, not what the code does;
  - named constants for every number the prose relies on, each explained (when a selector's
    `test_size` must equal the stopper's `max_items`, say why);
  - few seeds, explained once.
- **Not here:** internals (context plumbing, validation rules, how exposure rates are counted),
  exhaustive class coverage. A notebook that needs an internal to work has found an API gap: fix the
  API or record the gap, do not teach the internal.
- **Length:** about 25 cells or fewer; each notebook runs in under a minute on Colab.
- **Rules:** edit the `.ipynb` file itself and commit it without outputs (nbstripout); pre-commit
  runs the notebooks a commit changes and CI runs them all (nbmake). A notebook is the hands-on
  version of a guide and does not repeat it paragraph by paragraph.
- **Review:**
  - Does each figure come with a sentence on what to read from it?
  - Does every number in the prose follow from the code shown?
  - Does it reach into anything the package does not export?

### Planned techniques page

- **Job:** which methods catsim plans to add, why each matters, its key source and a link to its spec
  on GitHub. One table, under a sentence saying these methods are not yet available. No dates or
  order.
- **Audience and level:** researchers; intermediate.
- **Not here:** implementation sketches (`specs/`).
- **Length:** the sentence and one row per planned method.
- **Review:** Does every spec have a row, and every row a spec? Is every reference one the spec's
  revalidation checked?

### Specs (`specs/`)

- **Job:** contributor specifications for planned methods. Each keeps its implementation sketch,
  written against the current API.
- **Audience and level:** contributors; advanced.
- **Template:** `specs/_template.md`. Its user-facing sections match the technique-page template, so
  promoting a spec drops the contributor-only sections (Tests, Open Questions, Implementation
  Guidance) and adds the API Reference. `specs/README.md` holds the workflow, the citation convention
  and the revalidation checklist.
- **Rules:** a PR that renames something a sketch calls renames it in the sketch too. The sketch is
  the first `python` block under `## Implementation Guidance`. A spec writes its citations and its
  References list by hand, since GitHub does not render the citation roles; promoting it moves
  each source into `sphinx/references.bib`.
- **Review:** Does the sketch still run against the current API? Does the spec say when it was last
  validated, and against which sources?

### CONTRIBUTING (`CONTRIBUTING.md`)

- **Job:** how to contribute: development setup, the quality gate, building the docs, adding a
  technique, the release process.
- **Audience and level:** contributors.
- **Not here:** architecture rationale (ADRs), instructions only agents need (`AGENTS.md`).
- **Length:** as short as the process allows.
- **Review:** Does every command run as written? Does the gate match what pre-commit and CI run?

### GitHub release notes

- **Job:** the only record of what changed: per release, a hand-written summary (highlights,
  breaking changes with what users must do, results that change for the same seed), then GitHub's
  generated list of pull requests.
- **Audience and level:** users upgrading.
- **Not here:** rationale beyond a sentence (link the ADR).
- **Review:** Is every breaking change listed, with what to change? Is every change to seeded results
  named?

### The 1.0 notice (README and landing page)

- **Job:** warn 0.x users that their code needs changes and their seeded results differ, and point
  them to `catsim<1` and the 1.0.0 release notes. A warning box on the landing page, a blockquote in
  the README.
- **Audience and level:** 0.x users; introductory.
- **Not here:** any list of changes, including the API's new shape (the release notes).
- **Length:** two or three sentences.

## Terms and notation

`sphinx/terms-and-notation.md` is the source of truth for terms and symbols. Use its preferred term
for each concept everywhere; its synonyms appear only there and in quotations. When a document needs
a term or symbol the page lacks, add it to the page first. The terms most often mixed up:

- **ability** $\theta$, not proficiency or latent trait;
- **guessing parameter** $c$, the lower asymptote (the API's `pseudo_guessing`);
- **upper asymptote** $d$; $1 - d$ is the slipping parameter;
- **responses** $u$, not response vector or response pattern;
- **standard error** $SE(\hat{\theta})$, not SEE or SEM;
- **candidate item** for an item a selector may still choose, written in full at its first use on a
  page; "candidate" never means an examinee or a trial value of $\theta$;
- **stopper** for a stopping-criterion object; `StopRule` only for the class that combines stoppers,
  so do not write "stopping rule";
- $n$ for the test length, $N$ for the number of examinees, $M$ for the bank size, $i$ for the item
  index and $j$ for the examinee; $I$ is always information, never a set of items.

## Review checklist for every document

Beyond the checklist of its kind, a docs review (and the review of any PR that touches documents)
asks:

1. **Voice:** does anything tell history ("now", "no longer", "replaces", "instead of",
   "previously", a card ID, a PR number) outside an ADR's Context and the 1.0 notice?
2. **Audience:** is there anything the page's reader does not need, such as an internal name or
   design rationale?
3. **One home:** does it re-explain a topic whose home is elsewhere, rather than link it?
4. **Terms:** does every term and symbol match "Terms and notation"?
5. **Code:** does every example run as written against the current API, importing from the package's
   exports? No test runs the code in Sphinx pages and README; only doctests and notebooks are run.
6. **Build:** does the docs build pass with no new warnings?
7. **Length:** is the document within its budget?
