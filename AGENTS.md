# AGENTS.md

Repository guidance for coding agents working in `catsim`.

## Scope

These instructions apply to the entire repository unless a deeper `AGENTS.md` overrides them.

## Project Summary

`catsim` is a computerized adaptive testing toolkit. The current architecture is centered on:

- `CatEngine` for stepwise CAT execution
- `CatSessionState` for per-session runtime state
- `SimulationRunner` for batch execution
- `SimulationResult` for aggregate outputs

This repository no longer uses the legacy simulator-centric execution model as its architectural
center. Agents should preserve the current engine/session/result design and avoid reintroducing
hidden runtime coupling.

## Repository Layout

- `src/catsim/`: package source code
- `tests/`: automated test suite
- `notebooks/`: user-facing tutorial notebooks, version controlled as `.ipynb`
- `sphinx/`: documentation sources
- `sphinx/adr/`: Architecture Decision Records

## Core Architecture Rules

- Keep execution logic aligned with the explicit engine model:
  - `CatEngine`
  - `CatSessionState`
  - `SimulationRunner`
  - `SimulationResult`
- Do not reintroduce legacy simulator-coupled interfaces or hidden simulator backreferences.
- Keep runtime state outside domain objects when possible.
  - `ItemBank` should remain focused on calibrated item data and related derived data.
  - Run-level exposure and aggregate metrics belong in runtime/result objects.
- Preserve explicit strategy contracts.
  - Initializers, selectors, estimators, and stoppers should use explicit parameters rather than
    engine-facing `**kwargs` when the engine owns the call contract.
- Preserve stable step semantics.
  - `CatStepResult` should represent a durable step outcome, not a mutable live-session alias.

## Source Code Guidelines

- Prefer small, explicit changes over broad implicit behavior.
- Preserve public API clarity over convenience wrappers that obscure state ownership.
- Keep type annotations accurate; this repository uses `ty` as part of its quality gates.
- When changing architecture-level behavior, update or add an ADR in `sphinx/adr/`.
- Follow existing naming and module boundaries unless there is a strong reason to change them.

## Documentation Rules

- `README.md` is generated from `sphinx/readme_head.md` and `sphinx/readme_body.md`.
  - If shared README text changes, update the Sphinx source files rather than editing `README.md`
    in isolation.
- Durable technical rationale belongs in ADRs, not in temporary plan files.
- Tutorials are maintained as notebooks under `notebooks/`.
- When public APIs change, update:
  - relevant Sphinx pages
  - tutorial notebooks
  - README/Sphinx readme content when appropriate

## Tutorial Notebook Rules

- Keep tutorial notebooks as `.ipynb` files in `notebooks/`.
- Do not reintroduce generated-notebook workflows unless explicitly requested.
- Prefer examples that teach the current architecture:
  - `CatEngine`
  - `SimulationRunner`
  - `SimulationResult`
- Notebook outputs are normalized through pre-commit hooks.
- Notebook execution is part of validation, not optional polish.

## Testing And Validation

- Use `uv` for project commands.
- Preferred validation commands:
  - `uv run pytest -q`
  - `uv run ruff check src/catsim tests`
  - `uv run ty check`
- Notebook validation command:
  - `uv run pytest --nbmake --nbmake-timeout=300 notebooks/*.ipynb`

### When To Run What

- For targeted code changes, run the relevant subset of tests first.
- Before finalizing meaningful code changes, run at least:
  - relevant pytest coverage for touched modules
  - `ruff check`
  - `ty check`
- When notebook content or notebook-related tooling changes, validate the affected notebooks.

### Environment Caveat

- Notebook execution via `pytest --nbmake` may require an environment that allows Jupyter kernel
  startup and local socket binding.
- If notebook execution fails due to sandbox/kernel restrictions rather than notebook defects,
  call that out explicitly instead of misclassifying it as an application failure.

## Pre-commit And CI Expectations

- Pre-commit is part of the normal workflow and should remain aligned with the repository’s real
  quality requirements.
- Do not add redundant checks to CI when they are already covered by pre-commit unless there is a
  clear defense-in-depth reason.
- Preserve the current notebook hook strategy:
  - `nbstripout`
  - notebook Ruff hooks
  - notebook execution via `pytest --nbmake`
- Do not route generic JSON formatting hooks through notebooks if they conflict with notebook-
  specific tooling.

## Release And CI Policy

- Documentation availability is part of release readiness.
- Publishing workflows are intentionally coupled to the documentation deployment path.
- Do not decouple package publication from docs deployment unless explicitly requested.

## Commit And Git Hygiene

- Prefer atomic commits that group one logical change each.
- Do not mix architecture refactors, docs rewrites, workflow changes, and unrelated cleanup in one
  commit unless the changes are inseparable.
- Never commit local-only directories such as:
  - `.claude/`
  - `.pre-commit-cache/`
  - other tool caches or scratch directories
- `.plans/` is local working space and is gitignored.
  - Do not treat `.plans/` as durable project documentation.
- Preserve user changes in a dirty worktree. Do not revert unrelated modifications.

## Plans vs ADRs

- Use `.plans/` only for temporary execution planning.
- Use ADRs in `sphinx/adr/` for durable architectural decisions.
- If a plan has been fully implemented and its rationale matters long term, summarize the lasting
  decision in an ADR instead of keeping the plan as the canonical record.

## Practical Commands

- Run full tests:
  - `uv run pytest -q`
- Run targeted tests:
  - `uv run pytest tests/<module>.py -q`
- Run lint:
  - `uv run ruff check src/catsim tests`
- Run type checks:
  - `uv run ty check`
- Run notebook validation:
  - `uv run pytest --nbmake --nbmake-timeout=300 notebooks/*.ipynb`
- Run focused pre-commit hooks:
  - `uv run pre-commit run --files <paths...>`

## Decision Records

The main architectural decisions already recorded for this repository live under `sphinx/adr/`.
Agents making durable architecture changes should extend that record set instead of recreating
temporary rationale elsewhere.
