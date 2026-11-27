# \<Technique Name>

:Class: {py:class}`catsim.<family>.<ClassName>`
:Sources: {cite:t}`<key>`; {cite:t}`<key>`

% TECHNIQUE PAGE TEMPLATE. Copy this file into sphinx/api/techniques/<family>/, fill in each
% section, delete these comments and add the page to the toctree of the family's index.md.
% The rules for technique pages are in sphinx/AGENTS.md ("Technique pages"); the citation
% convention is in backlog/README.md. The header is a MyST field list, which needs "fieldlist" in
% myst_enable_extensions (sphinx/conf.py). Give one :Class: line per class the page documents.
% Every source is an entry of sphinx/references.bib, cited by its key: {cite:t} in a sentence,
% {cite:p} in parentheses, one role per source when it carries a location, and :cite:t: (reST)
% inside an eval-rst block.
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
its page and equation, e.g. {cite:t}`Barrada2008{p. 498, Equation 12}` or
{cite:p}`Barrada2008{p. 498, Equation 12}`, citing only locations you have read. Refer to the
`catsim.irt` function for a quantity such as information or the log-likelihood rather than
re-derive it. Use the symbols of terms-and-notation.md.>

% Optional: how catsim solves the definition numerically (search range, grid, tolerance), when it
% changes what users observe. Delete the subsection otherwise.

### Computation

\<How catsim computes the definition, and where it departs from its sources.>

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

\<How the method behaves in practice: on short tests, with all-correct or all-incorrect responses, on exposure,
with the other strategies it is usually paired with. How it compares with its alternatives, with
sources. Link the ADR for the reason behind a catsim choice.>

% Optional: a closed dropdown, the page's only mention of catR. Which catR function and settings
% the method matches, what the parity tests check, and each departure they keep as an expected
% failure. Delete it when catR has no counterpart: do not say that catR lacks the method.

:::{dropdown} Comparison with catR

\<The catR function and settings that match, what the parity tests (ADR-0020) find, and each
departure.>
:::

## Guarantees

% Properties a user relies on and can observe, in the user's terms. Properties only an
% implementation sees (no per-session state, evaluation counts) go to the docstring and the tests.

1. \<A property a user can observe, e.g. "It never returns an item that was already administered.">
2. \<E.g. "When every remaining item is over the exposure cap, it falls back to the most
   informative one.">
3. \<E.g. "The estimate is finite for all-correct and all-incorrect responses.">

## References

% The list is generated from the keys the page cites. Add each source to sphinx/references.bib with
% its DOI and a note: "Open copy: \url{...}" when the published version is not open and a copy is,
% "Open access" when it is, or "Not openly available; X (Year) describes the method." Brace the
% title words whose case matters ({R}, {catR}, {IRT}, {Bayesian}, proper names): the style
% lowercases titles. Enter a chapter of an edited book as @inproceedings: the style prints no
% note for @incollection. A source the page lists without citing it in the text goes in the directive's
% content, one key per line after a blank line.

```{bibliography}
:filter: docname in docnames
```

## API Reference

```{eval-rst}
.. autoclass:: catsim.<family>.<ClassName>
   :members:
   :show-inheritance:
```
