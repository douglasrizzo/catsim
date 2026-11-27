# Technique specs

This folder holds the specifications of CAT techniques that catsim does not implement yet. They are
not part of the published documentation, and none of them is available in the package.

`specs/<family>/` mirrors the technique families of the documentation,
`sphinx/api/techniques/<family>/`:

- `estimation/`: ability estimators;
- `selection/`: item selectors and exposure control;
- `stopping/`: stopping criteria.

A spec has the sections of an implemented technique page (motivation, definition, parameters,
behavior and comparison, guarantees, references) plus sections for contributors: "Tests", the
properties only an implementation sees, each checked by a test; "Implementation Guidance", which
sketches the code for whoever implements it; and, when design choices are still open, "Open
Questions", which lists them with their options and the option the spec assumes until they are
answered. New specs start from [`_template.md`](_template.md).

The specs are MyST Markdown, like the documentation. GitHub does not render MyST roles and
directives (`{doc}`, `{eval-rst}`, `:::{note}`); they render once a spec is promoted into the
documentation. mdformat is kept off this folder (`.pre-commit-config.yaml`) because it rewrites
that syntax and breaks the `$$` math.

## Status

The specs' citations, cited locations, definitions and catR comparisons were last revalidated
against their sources in November 2026 (checklist steps 1-4 and 6 below); each spec's status lines
say when and against which sources. Their implementation sketches follow the strategy
contract and the revalidated definitions (step 5), and the Sympson-Hetter sketch is an exposure
control of the engine. The sketches are illustrative and no test runs them; the specs with open
questions show the option each spec assumes. Every sketch was last run on 2026-11-26: each
was added at its proposed location to a copy of `src/catsim`, imported, and run on a small bank, on
hand-built contexts and under `CatEngine` and `simulate`.

## Workflow

1. **Write** the spec from [`_template.md`](_template.md) in its family's folder, with
   `**Status:** Planned`, following the citation convention below.
2. **Revalidate** it with the checklist below, before it is implemented or its references are
   published.
3. **Implement** the technique, with tests for its guarantees and its "Tests" section, once its
   open questions are answered and the spec updated to the answers.
4. **Promote** the spec into the documentation:
   - move it to `sphinx/api/techniques/<family>/` and add it to the toctree of that family's
     `index.md`;
   - replace the status lines with the header of the technique-page template,
     `sphinx/api/techniques/_template.md`, pointing at the class with `{py:class}`;
   - remove the "Tests", "Open Questions" and "Implementation Guidance" sections and add an "API
     Reference" section with the class's `autoclass`, as the template has;
   - move its references into `sphinx/references.bib` (see the citation convention below): add an
     entry for each source the bib lacks, turn each citation into a role with its location, and
     replace the References list with the template's filtered `{bibliography}`;
   - link the page from the class docstring ("For full algorithmic details, see :doc:`...`");
   - remove the technique's row from the Planned techniques page of the documentation.

## Citation convention

The placeholders in [`_template.md`](_template.md) set the convention for specs and technique
pages:

- Cite each formula and design choice where it appears, with its location in the source (page,
  equation, table or figure).
- Cite only locations you have read in the source.
- Say where the spec departs from its sources.
- Give each source its DOI. When the published version is not open, give an open copy ("Open
  copy: ...") or, if there is none, say so and name an open source that describes the method
  ("Not openly available; X (Year) describes the method.").

Specs write their citations by hand, e.g. "(Barrada et al., 2008, p. 498, Equation 12)", and give
each source in full in "References", because GitHub does not render the citation roles. Technique
pages generate theirs from `sphinx/references.bib`: a citation is a role with its location,
`` {cite:p}`Barrada2008{p. 498, Equation 12}` ``; each source is a bib entry with its DOI and its
note (`Open copy: \url{...}`, `Open access` or `Not openly available; ...`); and the References
section is a `{bibliography}` filtered to the page's citations, as the technique-page template,
`sphinx/api/techniques/_template.md`, shows. Promoting a spec moves its references into the bib.

## Revalidation checklist

Check each spec against its sources before it is implemented or its references are published:

1. **Citations.** Authors, year, title, venue and DOI match the source. Every DOI resolves at
   doi.org: a publisher that refuses automated requests is fine, a 404 is not. Every entry has an
   open copy or the "Not openly available" note.
2. **Cited locations.** Every cited page, equation and table has been read in the source; a
   location not read is not cited. Pages are the printed ones, which can differ from the PDF's.
3. **Definitions and formulas** match the cited equations. Every departure from the source, or from
   catR, is stated explicitly.
4. **catR** is the reference implementation where it implements the method (KLP, MPWI, MEPV and
   the minimum-information stopper): the spec says where it agrees and where it departs, which the
   promoted page puts in its closed "Comparison with catR" dropdown, and its testing notes name the
   parity cases the implementation adds (ADR-0020). A spec never says what catR lacks.
5. **Implementation guidance** follows the strategy contract of ADR-0007:
   - per-step strategies take a `StepContext`, and estimators return an `Estimate`, whose
     standard error is positive or `math.inf`;
   - Sympson-Hetter is an exposure control (`BaseExposureControl.accept(context, item)`), whose
     declines the engine keeps in the session's `excluded_items`;
   - selectors break exact ties with `pick_best` or `best_pool` (ADR-0017), and sum scores over a
     grid's node axis rather than with a matrix product, which can round identical items'
     columns differently;
   - code inside the package imports `override` from `catsim.compat`, never from
     `typing_extensions`, which catsim installs only below Python 3.12 (ADR-0021);
   - strategy objects hold configuration and caches derived from the item bank, never per-session
     state, because one engine serves many sessions: the SPRT and GLR stoppers classify with a pure
     `classify(context)`, and Owen's estimator replays the responses at every call;
   - each stopper is a single criterion, combined with others by `StopRule`;
   - no `[:, :4]` slices of the item matrix;
   - the MPWI and MEPV per-node tables are cached by the selector, keyed by the item bank, as the
     stratified selectors cache their strata (ADR-0008); the item bank caches nothing for strategies.
6. **Status.** The top of the spec says when it was last validated and against which sources.
