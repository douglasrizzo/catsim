# ADR-0003: Items Are an n×4 Matrix of 4PL Parameters

## Status

Accepted 2026-11-25

## Context

The logistic models differ in which item parameters they vary: the 1PL only the difficulty $b$
(with one discrimination for every item), the 2PL also the discrimination $a$, the 3PL also the
guessing parameter $c$ and the 4PL also the upper asymptote $d$. Each is the 4PL with the parameters
it leaves out fixed at $a = 1$, $c = 0$ and $d = 1$. Calibration software writes the parameters its
model estimates, so item banks arrive with one to four columns.

A bank built without validation carries a NaN or out-of-range parameter into every value derived
from it: the probabilities, the information and each selector's scores come out NaN or wrong, with
no error.

## Decision

The item bank is irtlib's `ItemBank` (ADR-0024), whose ADR-0009 records the rules below; catsim's
strategies rely on them.

- Every item bank holds an n×4 matrix of 4PL parameters, one row per item, with the columns $a$,
  $b$, $c$ and $d$. Every item response function and every strategy reads that shape.
- `ItemBank(items)` takes a matrix of one to four columns, read as $[b]$, $[a, b]$, $[a, b, c]$ or
  $[a, b, c, d]$, and fills the missing parameters with $a = 1$, $c = 0$ and $d = 1$; a 1-D array
  is one item. A matrix with more than four columns is rejected, never truncated.
- `ItemBank(items)` always validates: every parameter is finite, with $a > 0$ and
  $0 \leq c < d \leq 1$. The expansion and the checks are private to the item bank, which is their
  only entry point.
- The bank is immutable: its arrays are read-only, and it holds nothing about any session or run.
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
- Every bank's parameters are valid, so the values derived and cached from them (ADR-0008) are
  finite and in range, and no strategy checks them.
- One bank serves any number of sessions, runs and processes, since nothing in it changes.
- `IrtModel` members compare equal to their names, so code and configuration can pass `"3PL"`
  wherever catsim takes a model.

## Alternatives Considered

- A matrix as wide as its model, with each function dispatching on the width: every function and
  strategy would branch on the shape.
- Further columns for exposure or metadata: exposure belongs to a run or an application
  (ADR-0012), and the per-item inputs of techniques are arrays aligned with the rows (ADR-0004).
- Optional validation, for speed: a bank built without it carries invalid parameters into every
  derived value, while validation runs once per bank.
- The model as its number of parameters: an `int` reads like a column count, which is always 4,
  and cannot be passed where catsim takes a model name.
