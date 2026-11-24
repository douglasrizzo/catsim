# ADR-0017: Items Are an n×4 Matrix of 4PL Parameters

## Status

Accepted 2026-11-23

## Context

The logistic models differ in which item parameters they vary: the 1PL only the difficulty $b$
(with one discrimination for every item), the 2PL also the discrimination $a$, the 3PL also the
guessing parameter $c$ and the 4PL also the upper asymptote $d$. Each is the 4PL with the parameters
it leaves out fixed at $a = 1$, $c = 0$ and $d = 1$. Calibration software writes the parameters its
model estimates, so item banks arrive with one to four columns.

The item bank kept a fifth column of exposure rates beside the parameters (ADR-0003), its
validation could be turned off with `validate=False`, the expansion and validation of a matrix were
public functions of `catsim.irt`, and the bank reported its model as the number of parameters, an
`int`. A bank built without validation could carry a NaN or out-of-range parameter into every value
derived from it: the probabilities, the information and each selector's scores come out NaN or
wrong, with no error.

## Decision

- Every item bank holds an n×4 matrix of 4PL parameters, one row per item, with the columns
  $a$, $b$, $c$ and $d$. Every function of `catsim.irt` and every strategy reads that shape.
- `ItemBank(items)` takes a matrix of one to four columns, read as $[b]$, $[a, b]$, $[a, b, c]$ or
  $[a, b, c, d]$, and fills the missing parameters with $a = 1$, $c = 0$ and $d = 1$; a 1-D array
  is one item. A matrix with more than four columns is rejected, never truncated.
- `ItemBank(items)` always validates: every parameter is finite, with $a > 0$ and
  $0 \leq c < d \leq 1$. The expansion and the checks are private to the item bank, which is their
  only entry point.
- The bank reports its model as an `IrtModel`, a `StrEnum` whose members equal `"1PL"` to `"4PL"`:
  the simplest model that fits every item.

```python
bank = ItemBank(items)  # items: n×1 [b], n×2 [a, b], n×3 [a, b, c] or n×4 [a, b, c, d]
bank.items              # n×4, read-only
bank.model              # IrtModel.PL1 to IrtModel.PL4
```

## Consequences

- One shape serves every model, so no function or strategy branches on the model. With $c = 0$ and
  $d = 1$ the 4PL's characteristic curve is exactly the 2PL's.
- A 1PL bank stores four columns and runs the 4PL formulas, a cost small next to selection and
  estimation.
- Every bank's parameters are valid, so the values derived and cached from them are finite and in
  range, and no strategy checks them.
- A caller with a 1-3 column matrix gets the n×4 one from `bank.items`, the form the `catsim.irt`
  functions take.
- `IrtModel` members compare equal to their names, so code and configuration can pass `"3PL"`
  wherever catsim takes a model, as in `ItemBank.generate(n_items, model="3PL")`.

## Alternatives Considered

- A matrix as wide as its model, with each function dispatching on the width: every function and
  strategy would branch on the shape.
- Further columns for metadata such as exposure or content area: exposure is runtime state
  (ADR-0003), and other metadata belongs to the application, keyed by the item's index.
- Optional validation, for speed: a bank built without it carries invalid parameters into every
  derived value, while validation runs once per bank.
- The model as its number of parameters: an `int` reads like a column count, which is always 4,
  and cannot be passed where catsim takes a model name.
