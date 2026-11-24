# \<Technique Name>

:Class: {py:class}`catsim.<family>.<ClassName>`
:Sources: \<Author (Year)>; \<Author (Year)>

% TECHNIQUE PAGE TEMPLATE. Copy this file into sphinx/api/techniques/<family>/, fill in each
% section, delete these comments and add the page to the toctree of the family's index.md.
% The rules for technique pages are in sphinx/AGENTS.md ("Technique pages"); the citation
% convention is in specs/README.md. The header is a MyST field list, which needs "fieldlist" in
% myst_enable_extensions (sphinx/conf.py). Give one :Class: line per class the page documents.
% Budget: about 400-1,000 words before the references.

## Motivation

\<1-3 paragraphs for a researcher: what the method does, the problem it solves, when to choose it
over the alternatives and what makes it distinctive. Plain language; introduce every symbol in
words before using it.>

## Definition

\<The method's definition, with `$$` blocks for display equations.>

$$
<equation>
$$

\<What each component of the equation is. Cite the source of each formula where it appears, with
its page and equation, e.g. "(Barrada et al., 2008, p. 498, Equation 12)", citing only locations
you have read. Refer to the `catsim.irt` function for a quantity such as information or the
log-likelihood rather than re-derive it. Use the symbols of terms-and-notation.md.>

% Optional: how catsim solves the definition numerically (search range, grid, tolerance), when it
% changes what users observe. Delete the subsection otherwise.

### Computation

\<How catsim computes the definition, and where it departs from its sources or from catR.>

## Parameters

```{eval-rst}
.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``<param_name>``
     - ``<default>``
     - <what it controls, its valid values and the range the literature recommends>
```

## Behavior and Comparison

\<How the method behaves in practice: on short tests, with extreme response patterns, on exposure,
with the other strategies it is usually paired with. How it compares with its alternatives and with
catR's implementation, with sources. Link the ADR for the reason behind a catsim choice.>

## Guarantees

% Properties a user relies on and can observe, in the user's terms. Properties only an
% implementation sees (no per-session state, evaluation counts) go to the docstring and the tests.

1. \<A property a user can observe, e.g. "It never returns an item that was already administered.">
2. \<E.g. "When every remaining item is over the exposure cap, it falls back to the most
   informative one.">
3. \<E.g. "The estimate is finite for all-correct and all-incorrect response patterns.">

## References

\<One full citation per source, with its DOI. When the published version is not open, add an open
copy ("Open copy: ...") or, if there is none, say so and name an open source that describes the
method ("Not openly available; X (Year) describes the method.").>

## API Reference

```{eval-rst}
.. autoclass:: catsim.<family>.<ClassName>
   :members:
   :show-inheritance:
```
