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
  - Initializers, selectors, estimators, stoppers and exposure controls should use explicit
    parameters rather than engine-facing `**kwargs` when the engine owns the call contract.
- Preserve stable step semantics.
  - `CatStepResult` should represent a durable step outcome, not a mutable live-session alias.

## Source Code Guidelines

- Prefer small, explicit changes over broad implicit behavior.
- Preserve public API clarity over convenience wrappers that obscure state ownership.
- Keep type annotations accurate; this repository uses `ty` as part of its quality gates.
- Annotate NumPy arrays with the public aliases in `catsim.typing` (`FloatArray`, `IntArray`).
  Ruff bans `numpy.typing.NDArray` elsewhere; for another dtype, add an alias there.
- A method that overrides a base-class method carries `@override` (from `typing_extensions`); ty's
  `missing-override-decorator` enforces it in `src/` and `tests/`.
- Closed option sets are `StrEnum`s, like `IrtModel`; code that branches on one uses `match`, with
  a final `case _`.
- No backward-compatibility aliases or deprecation shims before 1.0: a renamed or removed name is
  gone.
- Randomness comes only from an `rng: int | numpy.random.Generator | None` argument, passed through
  `numpy.random.default_rng`, where None draws fresh entropy (strategies draw from the generator
  they are given). No global NumPy random state, no stdlib `random`.
- When changing architecture-level behavior, update or add an ADR in `sphinx/adr/`.
- Follow existing naming and module boundaries unless there is a strong reason to change them.
- Outside `src/catsim`, import each name from its public package path
  (`from catsim.selection import MaxInfoSelector`), not from the module file that defines it. Name
  things by the one-name-per-concept table (`theta`, `current_theta`, `administered_items`, `items`,
  `responses`, `n_items`, `model`, `rng`). Both rules are in
  `sphinx/adr/0006-public-import-paths-and-names.md`.
- A new top-level module in `src/catsim/` must be placed in the import-linter `layers` contract in
  `pyproject.toml`. The contract is exhaustive, so `lint-imports` fails until the module is placed.

## Documentation

Before writing or reviewing documentation of any kind, read `sphinx/AGENTS.md`, the documentation
charter. Each kind of document has one job and audience:

- `README.md`: what catsim is, install, a quickstart; newcomers on GitHub and PyPI. A plain file,
  separate from the website's landing page (`sphinx/index.md`); every link in it is absolute.
- `sphinx/introduction.md` and `sphinx/terms-and-notation.md`: the CAT and IRT primer and the
  glossary; readers new to CAT.
- Guides: one task end to end; researchers and application developers.
- Technique pages (`sphinx/api/techniques/`): a method's definition, behaviour and sources;
  researchers.
- Docstrings: the call contract; anyone calling the object.
- ADRs (`sphinx/adr/`): why the architecture is shaped this way; contributors.
- Notebooks: learning by doing, one level per notebook. They are `.ipynb` files in `notebooks/`,
  edited directly, with no generated-notebook workflow.
- `specs/`: planned methods with implementation sketches; contributors.

Hard rules:

- A docstring is the call contract: no method math, design rationale or history.
- History ("now", "no longer", "replaces", card IDs, PR numbers) goes only in the GitHub release
  notes and the 1.0 notice. There is no CHANGELOG file.
- Use the terms and symbols of `sphinx/terms-and-notation.md`.
- No internal names or design rationale in user pages: link the ADR.
- One home per topic; other documents link to it.
- When a public API changes, update its docstrings, the Sphinx pages, the notebooks and the README
  in the same PR. No test runs the code in Sphinx pages or the README: run it by hand.

## Testing And Validation

- Use `uv` for project commands.
- Local commands run on Python 3.13 (`.python-version`), the version of CI's pre-commit job, because
  `uv.lock` resolves some dependencies per Python version (numpy 2.4 on 3.11, 2.5 from 3.12) and ty
  checks against the installed stubs. CI's test matrix still runs 3.11 to 3.15.
- Pre-commit runs ruff, ty, pylint, import-linter, vulture, typos and the notebooks; run them all
  with `uv run pre-commit run --all-files`.
- Preferred validation commands:
  - `uv run pytest -q` (also runs the docstring examples in `src/catsim`; warnings are errors)
  - `uv run ruff check .`
  - `uv run pylint src/catsim`
  - `uv run ty check`
  - `uv run lint-imports` (the layer contracts in `pyproject.toml`)
- Notebook validation command:
  - `uv run pytest --nbmake --nbmake-timeout=300 notebooks/*.ipynb`
- Every test has a docstring naming what it tests for: the behaviour, or the bug that would make it
  fail. Test docstrings have no argument sections.
- ty checks `tests/` too. A test double subclasses the base class it stands in for; a line that
  passes a deliberately wrong type carries `# ty: ignore[<rule>]` (ty does not read mypy's
  `# type: ignore[<code>]`).

### When To Run What

- For targeted code changes, run the relevant subset of tests first.
- Before finalizing meaningful code changes, run at least:
  - relevant pytest coverage for touched modules
  - `uv run pre-commit run --all-files`
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
  - the Ruff hooks, which also cover notebooks
  - notebook execution via `pytest --nbmake`
- Do not route generic JSON formatting hooks through notebooks if they conflict with notebook-
  specific tooling.

## Release And CI Policy

- Documentation availability is part of release readiness.
- Publishing workflows are intentionally coupled to the documentation deployment path.
- Do not decouple package publication from docs deployment unless explicitly requested.
- Bump the version with `uv version` only; `pyproject.toml` is its single source.

## Commit And Git Hygiene

- Prefer atomic commits that group one logical change each.
- Do not mix architecture refactors, docs rewrites, workflow changes, and unrelated cleanup in one
  commit unless the changes are inseparable.
- Never commit local-only directories such as:
  - `.claude/`
  - `.pre-commit-cache/`
  - `papers/`
  - other tool caches or scratch directories
- A pre-commit hook rejects staged paths under `.claude/`, `.plans/`, `.pre-commit-cache/` and
  `papers/`, even when added with `git add -f`.
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
  - `uv run ruff check .`
- Run every pre-commit hook (ruff, ty, pylint, import-linter, vulture, typos, notebooks):
  - `uv run pre-commit run --all-files`
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
