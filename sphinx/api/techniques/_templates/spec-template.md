**Status:** Implemented
**Module:** {py:class}`catsim.<module>.<ClassName>`
**Reference:** Author (Year). *Title*. Journal, volume(issue), pages. <https://doi.org/>...

% SPEC TEMPLATE — copy this file and fill in each section.
% Remove this comment block before committing.
%
% Status values: Implemented | Planned
% For Planned specs, include the "Implementation Guidance" section at the bottom.
% Remove "Implementation Guidance" when promoting to Implemented.

# \<Technique Name>

## Motivation

\<1–3 paragraphs: what problem does this technique solve, when to prefer it over
alternatives, and what makes it distinctive.>

## Definition

\<Mathematical formulation. Use `.. math::` for display equations.>

$$
<equation>
$$

\<Prose explaining the components of the equation.>

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

## Behavioral Contracts

Any correct implementation must satisfy the following properties:

1. \<Property 1 — a testable statement about input/output behaviour.>
2. \<Property 2.>
3. \<Add as many as are meaningful.>

## References

\<Full bibliographic citations, one per line, in a consistent format.>

## Reading the Source

\<For each source: whether it was checked against its full text, a link to an
open copy if there is one, and the page, equation, table or figure where each
formula or design choice appears. Cite only locations you have read. If a source
could not be checked, say so and name what was checked instead. Note where the
implementation departs from the source. See the Study Guide for the conventions.>

**Checked by** \<test classes or test names that verify the contracts.>

% -----------------------------------------------------------------------

% IMPLEMENTATION GUIDANCE — remove this entire section when promoting to

% "Implemented" status. It is for contributors and agents only.

% -----------------------------------------------------------------------

## Implementation Guidance

:::{note}
This section is for contributors and agents implementing this technique.
It is removed when the spec is promoted to "Implemented" status.
:::

**Proposed location:** `src/catsim/<module>/<filename>.py`

**Mock implementation:**

```python
<mock code>
```

**Implementation notes:**

\<Performance, caching, edge cases, design decisions.>

**Testing notes:**

\<Suggested test scenarios and expected outcomes.>
