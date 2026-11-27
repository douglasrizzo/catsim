---
title: Polytomous item response models
kind: technique
status: parked
priority: 4
effort: XL
blocked_by: []
raised_by: []
---

# Polytomous item response models

catsim covers unidimensional, dichotomous items (ADR-0002): a response is correct or incorrect, and
an item is a row of the four-parameter matrix (ADR-0003). Polytomous models, such as the graded
response model, the partial credit model and the generalized partial credit model, score responses
in ordered categories. They are common in questionnaires and in mirtCAT.

This card is parked because ADR-0002 excludes these models, and the design does not anticipate them.
Moving it into scope needs a decision recorded in ADR-0002, which would touch:

- the item matrix (ADR-0003), which has one row of four parameters per item;
- `Session.responses`, which are booleans;
- every estimator's likelihood, every information-based selector and the result tables.

Related: [multidimensional-cat](multidimensional-cat.md), the other model family ADR-0002 leaves out.
