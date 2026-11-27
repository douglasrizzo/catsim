---
title: ItemBank metadata schema for content, enemies, word count
kind: decision
status: needs-decision
priority: 3
effort: L
blocked_by: []
raised_by: []
---

# ItemBank metadata schema

> Background: [the literature review §5 Q2](../literature-review.md)

## Why this is a discussion card

Several high-impact CAT strategies — content balancing (WDM), shadow-test approach, Sympson–Hetter with content — require each item to carry **non-IRT metadata**:

- **Content code(s)**: which content domain or sub-domain the item tests. Can be multi-label.
- **Enemy sets**: sets of items that must not co-appear (clones, overlapping stems).
- **Word count / reading time**: a numeric budget constraint.
- **Item exposure group** (optional): for clusters that must share an exposure cap.

The current `ItemBank` stores a dense `(n_items, 5)` float array: `[a, b, c, d, exposure_rate]`. Any metadata added is either a sidecar or a columnar extension. That is a **design decision with ripple effects** on serialization, caching, validation, and the public constructor signature. Hence it is filed here as a design discussion, not a ready-to-implement card.

## Options

### Option A — sidecar columns on `ItemBank`

Add optional `ItemBank` attributes:

```python
item_bank = ItemBank(
    items=array_5_cols,
    content_codes=["algebra", "geometry", ...],      # list[str] or list[set[str]]
    enemies=[{3, 17}, {42, 99, 100}],                # list[set[int]]
    word_counts=numpy.array([...], dtype=int),       # optional
    exposure_groups=numpy.array([...], dtype=int),   # optional
)
```

Pros: minimally invasive; each field is a plain attribute; selectors that don't use metadata are unaffected.

Cons: drift risk — every new constraint becomes a new attribute; no single validation point; no clear place for domain-specific metadata (e.g. "item format", "stimulus ID").

### Option B — a `constraints` dict on `ItemBank`

```python
item_bank = ItemBank(
    items=array_5_cols,
    constraints={
        "content": [...],
        "enemies": [...],
        "word_count": numpy.array([...]),
    },
)
```

Pros: single extension point; easy to serialize as JSON alongside the item matrix; selectors that need constraints pull by key and raise a clear error if missing.

Cons: dict access loses IDE autocompletion and type-checking precision; `ty` won't know the shapes.

### Option C — a separate `ItemBankMetadata` dataclass

```python
@dataclass
class ItemBankMetadata:
    content_codes: list[str] | None = None
    enemies: list[frozenset[int]] | None = None
    word_counts: FloatArray | None = None
    exposure_groups: IntArray | None = None

item_bank = ItemBank(items=..., metadata=ItemBankMetadata(...))
```

Pros: typed, validated at construction, easy to extend with new fields, plays well with `ty`. Clear separation between **calibrated data** (`items`) and **operational constraints** (`metadata`).

Cons: one more concept for users to learn; slightly more verbose.

## Recommendation

**Option C**. It aligns with the AGENTS.md principle that `ItemBank` should remain focused on "calibrated item data and related derived data". Operational constraints are exactly the sort of thing that belongs in a sibling type, not welded onto `items`.

## Constraint-satisfaction helpers

Once metadata is on the bank, a selector needs to ask questions like:

- "Which candidates share a content code with item $i$?"
- "Which candidates are in an enemy set that already has an administered member?"
- "What is the current count of administered items per content code?"

These helpers live next to the metadata, not in each selector:

```python
# src/catsim/item_bank.py (or a new item_bank_constraints.py)

def administered_content_counts(
    metadata: ItemBankMetadata,
    administered_items: list[int],
) -> dict[str, int]:
    ...

def enemy_blocked_candidates(
    metadata: ItemBankMetadata,
    administered_items: list[int],
) -> npt.NDArray[numpy.bool_]:
    ...
```

## Open questions

1. **Single vs. multi-label content**: operational CATs frequently have one primary + several secondary codes. Does catsim need to support the multi-label case on day one?
2. **Constraint specification format**: content balancing needs a target distribution (e.g. "30% algebra, 20% geometry, ..."). Where does that live — on `ItemBank` alongside the codes, or on the selector that enforces it? The selector is probably the right place; the bank only knows labels.
3. **Serialization**: if `ItemBank` is pickled for simulation runs, the metadata must round-trip. Add a format-versioned header.
4. **Validation**: how strict at construction time? E.g. should `enemies` referencing an out-of-range index raise immediately, or only when the selector hits it?
5. **Migration path**: does the constructor accept `metadata=None` (backward compatible) or should the existing `items=` constructor start warning on missing metadata?

## Downstream cards

Landing this schema unblocks:

- [weighted-deviations](weighted-deviations.md) — needs `content_codes` and a target distribution.
- [sympson-hetter-with-content](../exposure/sympson-hetter-with-content.md) — needs `content_codes`.
- (future) Shadow-test approach — needs the full constraints vocabulary.

## Notes

- **No implementation until a decision is made.** The concrete `ItemBankMetadata` signature depends on answering Q1–Q5 above with the maintainer.
- **ADR candidate**: once the design is chosen, record it as an ADR under `sphinx/adr/` — this is exactly the sort of durable decision that belongs there rather than in a plan file.
- **Tests**: construction validation, round-trip through pickle, and the helper functions. No estimator or selector tests until a downstream card lands.
