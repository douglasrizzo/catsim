---
title: Weighted Deviations Method (WDM) content-balancing selector
kind: technique
status: needs-research
priority: 4
effort: M
blocked_by: [content/item-bank-metadata-schema]
raised_by: []
---

# Weighted Deviations Method (WDM)

> Background: [the literature review §2.1.C](../literature-review.md)

## Why this is a discussion card

WDM is the classical content-balancing baseline. Every working reference implementation (catR, mirtCAT, RSCAT) ships some form of it. catsim cannot add it until [item-bank-metadata-schema.md](./item-bank-metadata-schema.md) lands — there is nothing to balance without content codes and target proportions on the bank.

## Reference

- Stocking, M. L., & Swanson, L. (1993). A method for severely constrained item selection in adaptive testing. *Applied Psychological Measurement*, 17(3), 277–292.
- Swanson, L., & Stocking, M. L. (1993). A model and heuristic for solving very large item selection problems. *Applied Psychological Measurement*, 17, 151–166.

## Theory

A constrained CAT imposes requirements of the form "the final test must contain $\ge L_c$ and $\le U_c$ items from content area $c$", plus similar bounds on word count, item type, etc. At each step, WDM chooses the item that maximizes Fisher information **minus** a weighted sum of penalty terms for violating any constraint:

$$
\text{score}_i \;=\; I_i(\hat\theta) \;-\; \sum_{c \in C} w_c \cdot d_c(i)
$$

where $d_c(i)$ is the signed deviation introduced if item $i$ is administered next (e.g. "this would push the algebra count one above its upper bound"), and $w_c$ is a per-constraint weight (higher = stricter).

In practice the penalty is evaluated for each candidate on every constraint and the candidate with the best trade-off is picked. Unlike shadow testing, WDM is **greedy** — it does not look ahead to whether the constraint set is still feasible later.

## Sketch

```python
# src/catsim/selection/content.py

class WeightedDeviationsSelector(BaseSelector):
  """Greedy content-balanced selector (Stocking & Swanson 1993)."""

  def __init__(
    self,
    target_proportions: dict[str, float],   # e.g. {"algebra": 0.3, ...}
    weights: dict[str, float] | None = None,
    test_length: int = 40,
    r_max: float = 1.0,
  ) -> None:
    ...

  def select(self, item_bank, administered_items, est_theta, ...):
    # 1. metadata = item_bank.metadata  # requires the metadata schema card
    # 2. For each candidate i:
    #      info_i = irt.inf(est_theta, *item_bank.items[i, :4])
    #      deviation_i = sum over content codes of max(0, projected_count - target)
    #      score_i = info_i - sum(self._weights[c] * deviation_i[c] for c in codes)
    # 3. Return argmax score.
    ...
```

The weight calibration is intentionally simple in the Stocking–Swanson paper — start with $w_c = 1$ and let users tune. More sophisticated variants use a dual-space update.

## Proposed location

`src/catsim/selection/content.py` — a new module for content-balancing selectors. Do not add to `selection/info.py`; keep constrained-CAT code in its own file for clarity.

## Open questions

1. **Target specification**: target proportions (soft) vs. hard lower/upper bounds. The Stocking–Swanson paper uses both; catsim's first cut can do soft targets only.
2. **Per-constraint weight defaults**: expose `weights` as optional with all-ones default, or force the user to specify them?
3. **Interaction with exposure caps**: should WDM compose with `r_max` the same way as `MaxInfoSelector`, or does the exposure cap fight the content balancing? Probably compose — use the same `candidate_mask` logic.
4. **Test-length coupling**: WDM needs to know the remaining test length to compute deviation, which implies the selector needs access to either the test length or the `CatSessionState`. Passing `remaining_items_count` through the `select()` signature is one option; reading it from a configured `test_length` is another.
5. **Metadata access**: the selector assumes the item bank carries `metadata.content_codes`. The metadata schema discussion card must decide the exact attribute name before this card can be implemented.

## Notes

- **External dependencies**: none.
- **Blocked by**: [item-bank-metadata-schema.md](./item-bank-metadata-schema.md). Cannot implement without it.
- **Pairs with**: shadow-test approach (the optimal alternative — WDM is the heuristic baseline).
- **Tests**: on a bank with known content distribution, a test simulated with WDM should match the target proportions within a tolerance that shrinks as `test_length` grows. Also check that WDM does not worsen `SEE(theta)` by more than ~10% vs. unconstrained MaxInfo on the same bank.
