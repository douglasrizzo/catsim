**Status:** Planned
**Module:** `catsim.<family>.<module>.<ClassName>` (planned)
**Reference:** Author (Year)
**Last validated:** YYYY-MM-DD, against \<the sources and catR version checked>.
**Implementation guidance:** written against the current strategy contract on YYYY-MM-DD.

% SPEC TEMPLATE for a planned technique. Copy this file into specs/<family>/, fill in each section
% and remove these comments before committing.
%
% The sections from Motivation to References have the names and order of the technique-page
% template, sphinx/api/techniques/_template.md, and follow its rules (sphinx/AGENTS.md, "Technique
% pages"): promoting the spec moves them onto a technique page. The sections after References are
% for contributors and are removed on promotion. specs/README.md describes the workflow.
%
% Specs cite by hand, "Author (Year, p. N)", with a hand-written References list: specs are not
% built by Sphinx, and GitHub does not render the {cite} roles of technique pages. Promotion moves
% each source into sphinx/references.bib and turns the citations into roles.

# \<Technique Name>

## Motivation

\<1-3 paragraphs for a researcher: what the method does, the problem it solves, when to choose it
over the alternatives and what makes it distinctive.>

## Definition

\<Mathematical formulation. Use `$$` blocks for display equations.>

$$
<equation>
$$

\<Prose explaining the components of the equation. Cite the source of each
formula or design choice where it appears, with the page, equation, table or
figure, e.g. "(Barrada et al., 2008, p. 498, Equation 12)". Cite only locations
you have read. Say where the implementation departs from its sources.>

% Optional: how the implementation will solve the definition numerically, when it changes what
% users observe. Delete the subsection otherwise.

### Computation

\<Search range, grid, tolerance; where they depart from the sources or from catR.>

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

\<How the method behaves in practice (short tests, extreme response patterns, exposure) and how it
compares with its alternatives and with catR's implementation, with sources.>

## Guarantees

% Properties a user relies on and can observe, in the user's terms. They are promoted to the
% technique page, and each one also gets a test.

1. \<A property a user can observe, e.g. "It never returns an item that was already administered.">
2. \<Add as many as are meaningful.>

## References

\<Full bibliographic citations, one per line, in a consistent format: every source
the implementation follows or was checked against, with its DOI. When the
published version is not open, add a link to an open copy ("Open copy: ...") or,
if there is none, say so and name the open source that describes the method
("Not openly available; X (Year) describes the method."). When the spec is promoted,
these entries move into sphinx/references.bib and this list becomes the technique
page's generated one.>

% -----------------------------------------------------------------------
% CONTRIBUTOR-ONLY SECTIONS: Tests, Open Questions and Implementation Guidance are removed when the
% spec is promoted to a technique page.
% -----------------------------------------------------------------------

## Tests

Besides the guarantees above, any correct implementation must satisfy these properties, each
checked by a test:

1. \<A property only an implementation sees, e.g. "It keeps no per-session state." or "The
   evaluation count is the number of grid nodes.">
2. \<A test scenario with an expected value that does not come from the implementation: a
   hand-worked case, a published value or catR's output, with its source.>

% OPEN QUESTIONS: optional. List the design choices that are still open, each with its options, its
% trade-off and the option the spec assumes for now. Answer them before implementing.

## Open Questions

1. \<Question: the options, their trade-off, and which one the spec assumes.>

% IMPLEMENTATION GUIDANCE: the sketch is the first python block under this heading. Keep it runnable
% against the current API.

## Implementation Guidance

:::{note}
This section is for contributors and agents implementing this technique.
It is removed when the spec is promoted to a technique page.
:::

**Proposed location:** `src/catsim/<family>/<module>.py`

**Mock implementation:**

```python
<mock code>
```

**Implementation notes:**

\<Performance, caching, edge cases, design decisions.>
