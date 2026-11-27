---
title: Sympson–Hetter with Content (SHC) exposure control
kind: technique
status: needs-research
priority: 4
effort: M
blocked_by: [exposure/sympson-hetter, content/item-bank-metadata-schema]
raised_by: []
---

# Sympson–Hetter with Content (SHC)

> Background: [the literature review §2.1.B](../literature-review.md)

## Why this is a discussion card

SHC extends the Sympson–Hetter exposure-control wrapper ([sympson-hetter](sympson-hetter.md)) with **content-level** exposure caps — not just "item $i$ is seen at most $r_{\max}$ of the time" but "content domain $c$ is exposed at most $r_{\max,c}$ of the time, and items within $c$ balance the load". This requires content metadata on the `ItemBank`, which is currently missing.

## Reference

- Kingsbury, G. G., & Zara, A. R. (1989). Procedures for selecting items for computerized adaptive tests. *Applied Measurement in Education*, 2(4), 359–375.
- Stocking, M. L., & Lewis, C. (1998). Controlling item exposure conditional on ability in computerized adaptive testing. *Journal of Educational and Behavioral Statistics*, 23(1), 57–75.
- Parshall, C. G., Davey, T., & Nering, M. L. (1998). *Test development exposure control for adaptive testing*. (NCME monograph)

## Theory

Plain Sympson–Hetter computes one exposure parameter $K_i$ per item such that the Bernoulli decision "administer item $i$ given it was the top-scoring candidate" results in a marginal exposure rate $\le r_{\max}$. SHC adds a **second layer** of control at the content-domain level:

$$
\text{administer}(i) \;=\; \text{Bernoulli}\big(K_i \cdot K_{c(i)}\big)
$$

where $c(i)$ is the content code of item $i$ and $K_c$ is a domain-level dampening factor calibrated via the same offline simulation loop as the per-item $K_i$. In practice SHC is calibrated in an **alternating** fashion: one pass updates item $K_i$'s, one pass updates domain $K_c$'s, repeat until both exposure constraints are satisfied.

The Stocking & Lewis (1998) variant is **ability-conditional** — they calibrate $K_i$ as a function of $\hat\theta$ bins, which keeps the exposure cap tight for high-ability examinees (who hit the same narrow slice of the bank). Catsim's MVP should do the simpler unconditional version first and flag conditional-on-theta as a follow-up.

## Sketch

```python
# src/catsim/exposure/sympson_hetter_content.py

class SympsonHetterWithContentWrapper(SympsonHetterWrapper):
  """SHC: adds per-content exposure control on top of SH (Stocking & Lewis 1998)."""

  def __init__(
    self,
    base_selector: BaseSelector,
    exposure_table: FloatArray,           # per-item K_i, from offline calibration
    content_exposure_table: dict[str, float],  # per-content K_c
    r_max: float = 0.2,
    r_max_content: dict[str, float] | None = None,
    rng: numpy.random.Generator | None = None,
  ) -> None:
    ...

  def select(self, item_bank, administered_items, est_theta, ...):
    # 1. Get the base selector's preferred item.
    # 2. Look up item_bank.metadata.content_codes[i] for that item.
    # 3. Draw Bernoulli(K_i * K_c). If rejected, blacklist and retry.
    ...
```

The offline calibration loop lives next to the `SympsonHetter` calibrator from card 13 but needs to track **two** marginal rate sets (item and content) and converge both.

## Proposed location

`src/catsim/exposure/sympson_hetter_content.py` — sibling of the plain SH wrapper. Share calibration helpers via a base module.

## Open questions

1. **Metadata schema**: blocked by the same content-code discussion as WDM. See [item-bank-metadata-schema](../content/item-bank-metadata-schema.md).
2. **Calibration convergence**: the alternating calibration is not guaranteed to converge when the content constraints and item-level exposure caps are incompatible. Need a clear failure mode: raise, warn, or clip to best effort?
3. **Ability-conditional variant**: Stocking & Lewis (1998) condition $K_i$ on $\hat\theta$ bins. Worth doing in v1 or defer?
4. **Interaction with content-balancing selectors**: if the user pairs SHC with a WDM selector (content-balanced inputs + content-balanced exposure), the two mechanisms can over-constrain. Document the caveat; do not try to reconcile here.

## Notes

- **External dependencies**: none.
- **Blocked by**:
  - [item-bank-metadata-schema](../content/item-bank-metadata-schema.md) — needs `content_codes`.
  - [sympson-hetter](sympson-hetter.md) — reuses the base SH calibration loop.
- **Tests**: run a simulation with a known content distribution and target per-content exposure, verify marginal exposure rates stay within tolerance, and verify that item-level SH constraints are also satisfied.
- **Pairs with**: Weighted Deviations Method ([weighted-deviations](../content/weighted-deviations.md)) — SHC controls *exposure*, WDM controls *composition*. They solve different problems and can coexist.
