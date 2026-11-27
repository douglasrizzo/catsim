---
title: <What the card does, as a short imperative, or the question a decision card asks>
kind: feature
status: ready
priority: 3
effort: M
blocked_by: []
raised_by: []
---

% CARD TEMPLATE. Copy this file into the folder of the card's area, name it with a slug
% (lowercase words joined by hyphens, no number prefix), fill in the frontmatter and the sections, and
% remove these comments. The README defines every frontmatter value; backlog/_index.py checks them and
% rewrites the README's index (a pre-commit hook), so commit the regenerated README with the card.
%
% Every card has the sections from Problem to Open Questions; leave out a section with nothing to say.
% A technique card (kind: technique) also has the sections from Motivation to Implementation Guidance
% below, which carry the method's full design: the old spec format.

# \<Title>

## Problem

\<What is wrong or missing, for whom, and what it costs them.>

## Evidence

\<Measurements, the walkthrough or PR that found it, a failing case. Numbers, not impressions.>

## What catsim has now

\<Checked against the code, with file and function names.>

## Proposal

\<The shape of the change, with the options when there is more than one, and a recommendation. A
decision card states its options and recommendation here, or under its own headings.>

## Tests

\<What would fail if the change were wrong. Expected values come from a hand-worked case, a closed
form, a published value or a brute-force reference, never from the code under test.>

## Open Questions

1. \<Question: the options, their trade-off, and which one the card assumes until it is answered.>

% -----------------------------------------------------------------------------------------------------
% TECHNIQUE CARDS ONLY, below. The sections from Motivation to References have the names and order of
% the technique-page template, sphinx/api/techniques/_template.md, and follow its rules
% (sphinx/AGENTS.md, "Technique pages"): implementing the technique moves them onto a technique page.
% Tests, Open Questions and Implementation Guidance are for contributors and stay out of the page.
%
% Technique cards cite by hand, "Author (Year, p. N)", with a hand-written References list, because
% GitHub does not render the {cite} roles of technique pages. The README's "Technique cards" section
% has the workflow, the citation convention and the revalidation checklist.
% -----------------------------------------------------------------------------------------------------

**Module:** `catsim.<family>.<module>.<ClassName>` (planned)
**Reference:** Author (Year)
**Last validated:** YYYY-MM-DD, against \<the sources and catR version checked>.
**Implementation guidance:** written against the strategy contract of ADR-0007 on YYYY-MM-DD.

## Motivation

\<1-3 paragraphs for a researcher: what the method does, the problem it solves, when to choose it
over the alternatives and what makes it distinctive.>

## Definition

\<Mathematical formulation. Use `$$` blocks for display equations.>

$$
<equation>
$$

\<Prose explaining the components of the equation. Cite the source of each formula or design choice
where it appears, with the page, equation, table or figure, e.g. "(Barrada et al., 2008, p. 498,
Equation 12)". Cite only locations you have read. Say where the implementation departs from its
sources.>

% Optional: how the implementation will solve the definition numerically, when it changes what users
% observe. Delete the subsection otherwise.

### Computation

\<Search range, grid, tolerance; where they depart from the sources.>

## Parameters

```{eval-rst}
.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``param_name``
     - ``default``
     - <description, including recommended ranges if the literature provides them>
```

## Behavior and Comparison

\<How the method behaves in practice (short tests, all-correct or all-incorrect responses, exposure)
and how it compares with its alternatives, with sources.>

\<Where catR implements the method, a last paragraph on which catR function and settings match and
where catsim departs; the technique page folds it into its "Comparison with catR" dropdown. Leave it
out when catR has no counterpart.>

## Guarantees

% Properties a user relies on and can observe, in the user's terms. They are promoted to the technique
% page, and each one also gets a test.

1. \<A property a user can observe, e.g. "It never returns an item that was already administered.">
2. \<Add as many as are meaningful.>

## References

\<Full bibliographic citations, one per line: every source the implementation follows or was checked
against, with its DOI. When the published version is not open, add an open copy ("Open copy: ...")
or, if there is none, say so and name the open source that describes the method ("Not openly
available; X (Year) describes the method."). When the technique is implemented, these entries move
into sphinx/references.bib and this list becomes the technique page's generated one.>

% For a technique card, Tests (above) holds the properties only an implementation sees, e.g. "It keeps
% no per-session state.", and test scenarios with expected values from a hand-worked case, a published
% value or catR's output.

## Implementation Guidance

**Proposed location:** `src/catsim/<family>/<module>.py`

**Mock implementation:**

```python
<mock code>
```

**Implementation notes:**

\<Performance, caching, edge cases, design decisions.>
