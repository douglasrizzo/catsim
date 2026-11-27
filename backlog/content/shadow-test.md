---
title: Shadow-Test Approach (STA) constrained CAT
kind: technique
status: needs-research
priority: 4
effort: L
blocked_by: [content/item-bank-metadata-schema]
raised_by: []
---

# Shadow-Test Approach (STA)

> Background: [the literature review §2.1.C and §5 Q3](../literature-review.md)

## Why this is a discussion card

STA is the gold standard for constrained CAT in the operational literature. Unlike greedy heuristics (WDM), it **guarantees** that every administered item belongs to at least one full-length test that satisfies *all* the test-specification constraints. The price is an **integer linear program (ILP) solved at every step**. catsim does not ship an ILP solver, and pulling one in is a significant dependency decision that needs maintainer sign-off before work begins.

## Reference

- van der Linden, W. J., & Reese, L. M. (1998). A model for optimal constrained adaptive testing. *Applied Psychological Measurement*, 22(3), 259–270.
- van der Linden, W. J. (2005). *Linear models for optimal test design*. Springer.
- Veldkamp, B. P. (2022). Review of the shadow-test approach to adaptive testing. *Behaviormetrika*, 49, 169–190.
- Choi, S. W., Moellering, K. T., Li, J., & van der Linden, W. J. (2016). Optimal reassembly of shadow tests. RSCAT R package.

## Theory

At step $n$, the CAT has administered $n-1$ items and selected a running ability estimate $\hat\theta_{n-1}$. STA then does the following **at every step**:

1. **Assemble** a full-length test of size $N$ — called the *shadow test* — by solving an ILP that:
   - includes every already-administered item;
   - maximizes total Fisher information at $\hat\theta_{n-1}$;
   - subject to **all** test-specification constraints (content, enemies, word count, item type, …).
2. **Pick** the best unadministered item *inside the shadow test* — usually the one with the highest Fisher information at $\hat\theta$.
3. Administer it, observe the response, re-estimate $\theta$, and re-solve the ILP at the next step with updated constraints.

Because the shadow test is a **feasible full-length test**, any item picked from it is guaranteed to be part of some final test that satisfies all constraints. The ILP re-solves every step, so shifting ability estimates and newly administered items feed into the next shadow. STA converges in practice to near-optimal (max-info, constraint-feasible) tests, outperforming WDM consistently.

### ILP formulation (sketch)

Binary decision variables $x_i \in \{0,1\}$ indicating whether item $i$ is in the shadow test. Maximize

$$
\sum_i x_i \cdot I_i(\hat\theta)
$$

subject to

$$
\sum_i x_i = N, \qquad x_i = 1 \;\forall\, i \in \text{administered}, \qquad L_c \le \sum_{i \in c} x_i \le U_c \;\forall\, c, \qquad \sum_{i \in E} x_i \le 1 \;\forall\, E \in \text{enemies}, \quad \dots
$$

This is a mid-sized ILP — a few hundred to a few thousand binary variables, tens to low hundreds of constraints. Solvable in $< 100$ ms by `pulp`/`CBC` for typical banks; in $< 20$ ms by commercial solvers (CPLEX, Gurobi).

## Sketch

```python
# src/catsim/selection/shadow_test.py  (only imported if the extra is installed)

try:
    import pulp
except ImportError as e:
    raise ImportError(
        "ShadowTestSelector requires the 'shadow' extra. "
        "Install with: pip install catsim[shadow]"
    ) from e

class ShadowTestSelector(BaseSelector):
  """Shadow-test approach (van der Linden & Reese 1998)."""

  def __init__(
    self,
    test_length: int,
    constraints: ShadowTestConstraints,  # see below
    solver: str = "cbc",
  ) -> None:
    ...

  def select(self, item_bank, administered_items, est_theta, ...):
    # 1. Build the ILP with current administered items as fixed-to-1.
    # 2. Objective: max sum_i x_i * Fisher_info(est_theta, i)
    # 3. Constraints: total = test_length, content bounds, enemies, ...
    # 4. Solve.
    # 5. From the solution, pick the unadministered item with the highest info.
    ...
```

`ShadowTestConstraints` is a typed container holding content bounds, enemy sets, word-count budget, and any other domain-specific constraints. It overlaps heavily with the [item-bank-metadata-schema.md](./item-bank-metadata-schema.md) discussion — in practice, STA wants both *per-item metadata* (which item is which) and *test-level constraints* (total algebra in [10, 14]).

## Proposed location

`src/catsim/selection/shadow_test.py` — guarded by a `try: import pulp` so that the rest of catsim still installs cleanly when the `shadow` extra is absent.

Ship as an optional extra:

```toml
# pyproject.toml
[project.optional-dependencies]
shadow = ["pulp>=2.7"]
```

## Open questions

1. **Solver choice**: `pulp` bundles CBC and is pure-Python-friendly; `cvxpy` is more general but slower on ILPs; commercial solvers (Gurobi, CPLEX) are fastest but licensed. Default to `pulp`+CBC with a pluggable `solver=` parameter?
2. **Re-solve vs. incremental**: the simplest implementation re-solves the full ILP at every step. Choi et al. (2016) describe incremental "shadow test reassembly" that updates the previous solution. Is the performance win worth the complexity on day one? Probably no — profile first.
3. **Constraint specification**: the ILP needs a compact data structure for bounds, counts, and item groupings. Share with the metadata schema discussion, or define a separate `ShadowTestConstraints` type?
4. **Failure handling**: if the ILP is infeasible (e.g. too many administered items violate a bound), should STA fall back to a relaxed objective (minimize total deviation + max info) or raise?
5. **Interaction with exposure control**: STA does not natively handle exposure. Can we compose STA with `SympsonHetterWrapper` by feeding the SH-adjusted info values into the ILP objective? Worth trying, but file as a follow-up.

## Notes

- **External dependencies**: `pulp` (BSD, pure Python) as an optional extra. Do NOT put it in the core install.
- **Release policy**: per AGENTS.md, docs publication is coupled to package release, so the `shadow` extra needs a docs page (motivation, install instructions, usage example, a worked constraint spec).
- **Tests**:
  - Build a tiny 50-item bank with 3 content domains and synthetic bounds; verify STA always selects items from a feasible shadow.
  - Verify the selected item sequence keeps the test trajectory within constraints until the final item.
  - Benchmark on a 500-item bank to confirm CBC solves in the target latency budget.
- **Pairs with**: [weighted-deviations](weighted-deviations.md) — STA is the optimal alternative to WDM's greedy heuristic.
- **ADR**: adding an optional heavy dependency is exactly the sort of thing that belongs in an ADR under `sphinx/adr/`. Record the decision there once the maintainer signs off.
