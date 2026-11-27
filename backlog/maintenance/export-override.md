---
title: Should catsim export override for custom strategies?
kind: decision
status: needs-decision
priority: 4
effort: XS
blocked_by: []
raised_by: [notebook rewrite (#226)]
---

# Should catsim export `override` for custom strategies?

## The question

A custom strategy overrides a base-class method and marks it with `@override`. That decorator comes
from `typing` on Python 3.12+, and from `typing_extensions` on 3.11, which catsim installs only
below 3.12 (ADR-0021). Users therefore write a version check, as notebook 04 does. "Extending
catsim" imports from `typing` and adds a sentence for 3.11. Inside the package, code imports it from
`catsim.compat`, which is not public API.

## Options

1. **Keep it as it is.** Users import from `typing`, and the docs keep their sentence for 3.11.
   Once catsim requires Python 3.12, the question goes away.
2. **Make `catsim.compat.override` public:** document it, and list it in ADR-0023's public paths.
   The cost is one more public name, and it isn't a CAT concept.
3. **Re-export it from the strategy packages,** such as `catsim.selection.override`. Same cost, more
   places.

Recommendation: option 1. The friction is one import line, and it ends when Python 3.11 support
does.
