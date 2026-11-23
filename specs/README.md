# Technique specs

This folder holds the specifications of CAT techniques that catsim does not implement yet. They are
not part of the published documentation, and none of them is available in the package.

`specs/<family>/` mirrors the technique families of the documentation,
`sphinx/api/techniques/<family>/`:

- `estimation/`: ability estimators;
- `selection/`: item selectors and exposure control;
- `stopping/`: stopping criteria.

A spec has the sections of an implemented technique page (motivation, definition, parameters,
behavioral contracts, references) plus an "Implementation Guidance" section that sketches the code
for whoever implements it. New specs start from [`_template.md`](_template.md).

The specs are MyST Markdown, like the documentation. GitHub does not render MyST roles and
directives (`{doc}`, `{eval-rst}`, `:::{note}`); they render once a spec is promoted into the
documentation. mdformat is kept off this folder (`.pre-commit-config.yaml`) because it rewrites
that syntax and breaks the `$$` math.

## Status

The specs here were written together in April 2026 and have not been through the revalidation
checklist below yet. Their implementation sketches predate the catsim 1.0 strategy contract and do
not follow it; they will be updated once that contract is on `dev`. Do not implement a technique
from a sketch that does not follow the contract.

## Workflow

1. **Write** the spec from [`_template.md`](_template.md) in its family's folder, with
   `**Status:** Planned`, following the citation convention below.
2. **Revalidate** it with the checklist below, before it is implemented or its references are
   published.
3. **Implement** the technique, with tests for its behavioral contracts.
4. **Promote** the spec into the documentation:
   - move it to `sphinx/api/techniques/<family>/` and add it to the toctree of that family's
     `index.md`;
   - set `**Status:** Implemented` and point `**Module:**` at the class with `{py:class}`;
   - remove the "Implementation Guidance" section and add an "API Reference" section with the
     class's `autoclass`, as the implemented pages have;
   - link the page from the class docstring ("For full algorithmic details, see :doc:`...`");
   - remove the technique's row from the Planned techniques page of the documentation.

## Citation convention

The placeholders in [`_template.md`](_template.md) set the convention for specs and technique
pages:

- Cite each formula and design choice where it appears, with its location in the source (page,
  equation, table or figure), e.g. "(Barrada et al., 2008, p. 498, Equation 12)".
- Cite only locations you have read in the source.
- Say where the spec departs from its sources.
- In "References", give each source in full with its DOI. When the published version is not open,
  link an open copy ("Open copy: ...") or, if there is none, say so and name an open source that
  describes the method ("Not openly available; X (Year) describes the method.").

## Revalidation checklist

Check each spec against its sources before it is implemented or its references are published:

1. **Citations.** Authors, year, title, venue and DOI match the source. Every DOI resolves at
   doi.org: a publisher that refuses automated requests is fine, a 404 is not. Every entry has an
   open copy or the "Not openly available" note.
2. **Cited locations.** Every cited page, equation and table has been read in the source; a
   location not read is not cited. Pages are the printed ones, which can differ from the PDF's.
3. **Definitions and formulas** match the cited equations. Every departure from the source, or from
   catR, is stated explicitly.
4. **catR** is the reference implementation where it implements the method (KLP, MPWI, MEPV,
   Sympson-Hetter-style exposure control, classification stopping): the spec says where it agrees
   and where it departs.
5. **Implementation guidance** follows the 1.0 strategy contract:
   - per-step strategies take a `StepContext`, and estimators return an `Estimate`;
   - Sympson-Hetter declines items through the context's `excluded_items`;
   - strategy objects hold configuration and caches derived from the item bank, never per-session
     state, because one engine serves many sessions: the SPRT and GLR stoppers classify with a pure
     `classify(context)` instead of keeping `_last_classification`, and Owen's estimator replays
     the responses instead of keeping `self._state`;
   - each stopper is a single criterion, combined with others by `StopRule`;
   - no `[:, :4]` slices of the item matrix;
   - the MPWI and MEPV per-node tables are cached in `ItemBank._selector_cache`.
6. **Status.** The top of the spec says when it was last validated and against which sources.
