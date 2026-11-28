---
title: One annotation for responses across the package?
kind: decision
status: needs-decision
priority: 3
effort: S
blocked_by: []
raised_by: [ADR audit (#225)]
---

# One annotation for responses across the package?

## The question

The functions that take responses accept Booleans and the integers 0 and 1 (ADR-0009). Most of
them are annotated `Sequence[bool]`, which a type checker
reads as rejecting 0 and 1. Session fields are `tuple[bool, ...]`, and response matrices are
`npt.ArrayLike`.

## Options

1. **`Sequence[bool]` everywhere.** The integers 0 and 1 stay accepted at run time and go
   undocumented in the types. A caller passing `[1, 0, 1]` gets a type error and a working call.
2. **A `Response` alias in `irtlib.typing`,** such as `bool | numpy.bool_ | Literal[0, 1]`, used for
   every response argument. Sessions keep storing `bool`.
3. **Accept Booleans only.** Reject 0 and 1 at run time, and keep `Sequence[bool]`.

Recommendation: option 2. The annotations then say what the checks accept. The alias goes in
`irtlib.typing` beside `FloatArray` and `IntArray` (AGENTS.md), through a pull request in irtlib.
