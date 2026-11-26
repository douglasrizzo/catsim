# ADR-0004: Items Have Names, and Computations Use Their Indices

## Status

Accepted 2026-11-25

## Context

An item was known only by its row in the bank. Results, plots and error messages showed indices,
so a researcher had to keep a separate list to tell which item of a published bank was which, and
a report could not name an item. A researcher reproducing a study in catsim found items had no
identification.

## Decision

- Every item has a name: a non-empty string, unique within its bank. `ItemBank(items, names=None)`
  takes one name per row, or generates them.
- A generated name is `item_` followed by the row index, zero-based like the indices, and
  zero-padded to the width of the last index: `item_000` to `item_499` in a 500-item bank.
- Computations use row indices. The strategy contract, the step context, the session and the step
  methods take and return indices, `0` to `n_items - 1`.
- Outputs read by people carry names: the tables of a simulation result have a name column
  (ADR-0016), plots label items by name, and an error about an item gives its name and its index.
- `bank.names` holds the names in row order, and the bank translates between names and indices.
- Every other per-item input, such as a recorded response matrix or a technique's cluster labels or
  content groups, matches items by position: column or element `i` belongs to item `i`.

## Consequences

- An item keeps its identity in every output, whatever order a study's tables are sorted in.
  Padding keeps the generated names in bank order when sorted as text.
- Computations index NumPy arrays directly, with no lookup per step.
- Researchers align their own data with the bank through the names, for example with pandas,
  `frame[list(bank.names)]`, before passing it by position.
- Mapping names to an application's own item records is the application's.

## Alternatives Considered

- Indices only: outputs cannot say which item is which.
- Names as the identity in computations: every step would translate names to rows, and the strategy
  contract would grow a second way to refer to an item.
- Generated names equal to the index, `"0"`, `"1"`: they read like indices, and stop matching them
  as soon as a study renumbers or subsets a bank.
- One-based generated names, as R users might expect: they would be off by one from every index
  catsim reports.
- Per-item inputs keyed by name: two ways to pass the same input.
