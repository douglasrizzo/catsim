---
title: Should catsim's IRT functions take a scaling constant D?
kind: decision
status: needs-decision
priority: 3
effort: S
blocked_by: []
raised_by: [catR comparison, Babcock & Weiss (2012) walkthrough]
---

# Should catsim's IRT functions take a scaling constant D?

## The question

catR's functions take `D`: 1 by default, 1.702 for the normal-ogive metric. catsim's logistic
functions have no scaling constant, so a bank calibrated with D = 1.7 must have its discriminations
multiplied by 1.7 before use. All three research walkthroughs did this by hand. The Babcock & Weiss
agent folded D into a.

## Options

1. **No argument, better documentation.** The [bank-generation](../item-bank/bank-generation.md)
   card puts the rescaling where users look: the `ItemBank` and `catsim.irt` docstrings. The item
   matrix stays the one thing that defines an item (ADR-0003).
2. **A `D` on `ItemBank`.** The bank stores it and rescales a once. Every computation stays as it
   is, but the bank's parameters then mean something different from its matrix.
3. **A `D` on every IRT function and strategy, as catR has.** This is the widest change, and every
   caller must pass the same D.

Recommendation: option 1. It needs no decision beyond accepting it, and the bank-generation card
covers it.
