# AGENTS.md

Repository guidance for coding agents working in `catsim`.

## Scope

These instructions apply to the entire repository unless a deeper `AGENTS.md` overrides them.

## Project Summary

`catsim` is a library for computing and evaluating computerized adaptive tests: the IRT quantities,
the CAT strategies, and simulation studies that compare them (ADR-0001). A live testing application
can use its computations as its core, but catsim does not run, store or serve live tests. The
architecture is centered on:

- `ItemBank`, the immutable calibrated items, with their names (ADR-0003, ADR-0004)
- `CatEngine`, the immutable configuration of a test, whose step methods (`start`, `select`,
  `answer`) take a `Session` and return a new one (ADR-0006)
- `Session`, an immutable value that the caller keeps
- `simulate`, which runs sessions over generated or recorded examinees, in parts on workers
  (ADR-0014, ADR-0015), and returns a `SimulationResult` (ADR-0016)

### Rescoping in progress (temporary; remove when `rescope` merges into dev)

The `rescope` branch moves the code to the ADRs above; `.plans/rescope/` holds the plan. Until it
merges into dev, the code may still show the previous design (`CatSessionState`, `RunContext`,
`ExposureTracker`, `SimulationRunner`): follow the ADRs, not the code, when they differ. On this
branch the notebooks, the narrative documentation, the README and the website may be out of date;
their checks are off (commit e504be7a) and they are brought up to date at the end. Tests, doctests,
ruff, ty, pylint, import-linter and vulture stay on. Feature branches start from `rescope`, and
their PRs target it.

## Repository Layout

- `src/catsim/`: package source code
- `tests/`: automated test suite
- `notebooks/`: user-facing tutorial notebooks, version controlled as `.ipynb`
- `sphinx/`: documentation sources
- `sphinx/adr/`: Architecture Decision Records

## Core Architecture Rules

- Compute each step from explicit inputs (ADR-0006). catsim keeps nothing between calls that a
  caller must preserve: the session, the exposure counts and the generator are the caller's.
  Serving concerns (storing sessions, the item on screen, concurrency, audits) belong to the
  application, never to catsim.
- Do not reintroduce mutable session objects, simulator-coupled interfaces or backreferences from a
  strategy to its caller.
- Cache only derived data, in the object it derives from; a cache that depends on a session's
  history is an optional, explicit input (ADR-0008).
- Check explicit inputs where they enter catsim, for everything that needs no recomputation
  (ADR-0009).
- Preserve the strategy contracts: every strategy takes one `StepContext` (ADR-0007), and
  strategies hold configuration and derived data only, never anything about a session.
- Every random draw comes from a generator the caller passes; simulations draw from one stream per
  examinee (ADR-0013).
- Keep every built-in strategy picklable, so runs can use workers (ADR-0015).
- Runtime dependencies are NumPy and SciPy only (ADR-0021).
- Performance is not a requirement. When a change makes catsim slower, say so to the maintainer
  when it happens, with measurements.

## Source Code Guidelines

- Prefer small, explicit changes over broad implicit behavior.
- Preserve public API clarity over convenience wrappers that obscure state ownership.
- Keep type annotations accurate; this repository uses `ty` as part of its quality gates.
- Annotate NumPy arrays with the public aliases in `catsim.typing` (`FloatArray`, `IntArray`).
  Ruff bans `numpy.typing.NDArray` elsewhere; for another dtype, add an alias there.
- A method that overrides a base-class method carries `@override`, imported from `catsim.compat` in
  `src/` and `tests/` (typing-extensions is installed only below Python 3.12); ty's
  `missing-override-decorator` enforces it in both.
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
  `responses`, `item`, `name`, `n_items`, `model`, `rng`). Both rules are in
  `sphinx/adr/0023-public-import-paths-and-names.md`.
- A new top-level module in `src/catsim/` must be placed in the import-linter `layers` contract in
  `pyproject.toml`. The contract is exhaustive, so `lint-imports` fails until the module is placed.

## Documentation

Before writing or reviewing documentation of any kind, read `sphinx/AGENTS.md`, the documentation
charter. Each kind of document has one job and audience:

- `README.md`: what catsim is, install, a quickstart; newcomers on GitHub and PyPI. A plain file,
  separate from the website's landing page (`sphinx/index.md`); every link in it is absolute.
- `sphinx/introduction.md` and `sphinx/terms-and-notation.md`: the CAT and IRT primer and the
  glossary; readers new to CAT.
- Guides: one task end to end; researchers, method authors and, in "Embedding catsim", integrators.
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
  in the same PR. No test runs the code in Sphinx pages or the README: run it by hand. On the
  `rescope` branch, update the docstrings in the same PR, and leave the Sphinx pages, notebooks and
  README to the end of the rescoping (see "Rescoping in progress").

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
