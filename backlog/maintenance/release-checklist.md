---
title: Release the next version
kind: maintenance
status: blocked
priority: 3
effort: M
blocked_by: []
raised_by: [rescoping plan]
---

# Release the next version

Blocked until the maintainer decides to release. Nothing here is a deadline: catsim releases when the
maintainer is satisfied with the package.

## Mechanics

- Wait for irtlib's first release on PyPI, with a license (ADR-0024). Then replace the direct
  reference `irtlib @ git+https://...@<sha>` in `pyproject.toml` with a version range of the
  released irtlib, run `uv lock`, and add irtlib's floor to the CI job that tests the lowest direct
  dependencies. PyPI rejects an upload whose metadata holds a URL dependency, and the publish job
  runs after the production docs deploy, so a release with the URL still in place deploys the docs
  and then fails to publish. TestPyPI rejects it too, so until then the TestPyPI job of the dev →
  main pull request fails after the dev docs deploy; it is not a required check.
- Bump the version with `uv version` only; `pyproject.toml` is its single source.
- The release tag equals the version exactly, with no `v`.
- Bump `UV_VERSION` by hand in `ci-cd.yml` and `linkcheck.yml`, which must stay equal (0.13.9 when
  this card was written).
- On the first dev → main PR, watch the "Protect main" ruleset's
  `require_extra_approval_for_unattributed_changes`: its effect with 0 required approvals is
  unverified.
- Write the release notes as `CONTRIBUTING.md`'s "Releasing" section says: a hand-written summary,
  then GitHub's generated list of PRs. The draft below is a start; check every renamed name against
  0.21.0's API before publishing, since some renames may predate the rescoping.

## After the release deploys

- The README's logo URLs point at `raw.githubusercontent.com/.../dev/sphinx/logo/...`, because the
  logo files were not on `main`; switch them to `main`.
- Drop the `_PUBLISHED_BY_THE_RELEASE` entries of `linkcheck_ignore` in `sphinx/conf.py` whose targets
  are now online (the release notes, the notebooks on `main`, the site's `api/` pages).
- Point the compatibility notice's "release notes" link at the release's own notes.

## Draft of the release notes' summary (against 0.21.0)

### Breaking changes, and what to do

- **Sessions are immutable values.** `CatSessionState`, `CatSessionSnapshot`, `CatStepResult`,
  `RunContext` and `ExposureTracker` are gone. `CatEngine` has three steps:
  `start(*, rng, excluded_items=())`, `select(session, *, rng, exposure_rates=None)`, which returns a
  `Selection(item, session, declined)`, and `answer(session, item, response, *, rng)`, each returning
  a new `Session`. `start_session`, `select_next`, `apply_response`, `step`, `run_session` and
  `should_stop` are gone. `Session` lost `session_id`, `true_theta`, `is_stopped`,
  `administered_count` and `snapshot`: keep identifiers beside the session, test
  `stop_reason is None`, use `len(session.administered_items)`. See the "Embedding catsim" guide.
- **Exposure rates are an input.** Pass `exposure_rates=` to `select` (one rate per item in
  [0, 1], or None for no cap); catsim keeps no counts.
- **`simulate` replaces `SimulationRunner`.** `simulate(engine, *, true_thetas=None,
  responses=None, rng=None, parts=1, workers=1, progress=None)`. `ResponseProvider` and
  `SimulatedResponseProvider` are gone: recorded examinees are rows of `responses` (bool, 0/1, NaN
  for missing). `progress` is a callable `progress(finished, total)`, not `True`.
- **Results.** `SimulationResult` is immutable. `final_thetas`, `final_ses`, `test_lengths`,
  `theta_histories`, `administered_items`, `responses`, `evaluations` and `mean_evaluations` are
  gone: use the tables `steps()`, `examinees()` and `deciles()`, and the metric methods `bias()`,
  `mse()`, `rmse()`, `test_length()`, `se()`, `exposure_rates()`, `proposal_rates()` and
  `overlap_rate()`, which are methods (not properties) and return `Metric(value, mc_se)`.
- **Renamed arguments (ADR-0023).** `item_id` and `selected_item_id` become `item`
  (`BaseExposureControl.accept(context, item)`); `LinearSelector(indices)`; the metric functions take
  `(true_thetas, thetas)` (formerly `actual`, `predicted`); `est_theta`/`updated_theta` become
  `theta`; `response_vector` becomes `responses`; `seed` becomes `rng`; `item_type`/`itemtype`
  become `model`; `administered_item_ids` becomes `administered_items`.
- **Items have names.** `ItemBank(items, names=None)` generates `item_000`, ...; `ItemBank.from_csv`
  and `to_csv` read and write banks. Result tables carry a `name` column; plots label items by name;
  error messages give `index ('name')`.
- **Plots** take their data first and a bank: `item_curve(item, item_bank, ...)`,
  `item_exposure(rates, item_bank, ...)`, `session_progress(session, item_bank, *, true_theta=None)`.
- **Custom strategies** must be picklable classes defined in a module to run on workers.
  `override` comes from `typing`.
- **Python 3.12 or later.** catsim builds on irtlib (ADR-0024), which requires 3.12.
- **The item response theory comes from irtlib.** `ItemBank` and `IrtModel` are imported from
  `irtlib`, the functions of `catsim.irt` from `irtlib.irt`, and `FloatArray` and `IntArray` from
  `irtlib.typing`; `catsim.irt`, `catsim.typing` and `catsim.ItemBank` are gone, with no alias.
  `irtlib.irt.log_likelihood` and `log_likelihood_function` take one row of responses per
  examinee, so one examinee's call is `log_likelihood(theta, [responses], items)[0]`.
  `validate_response_vector` is gone. `irtlib.irt.reliability` is $I/(I + 1)$, the reliability for
  abilities N(0, 1), where 0.21's `reliability` was $1 - 1/I$, negative when $I < 1$.
- **Dependencies.** tqdm is not a dependency; matplotlib is the `plot` extra; irtlib is.

### Results that change for the same seed

- Seeded simulations of generated examinees change: each examinee draws its whole response pattern
  and its session's draws from streams of its own (ADR-0013), so the same seed gives other
  numbers than 0.21.
- A generator passed as `rng` gives other results than the int it was built from: `simulate` draws
  one 128-bit seed from it.
- Runs on recorded responses without randomization give the same items and estimates as before
  (the reference baseline in `tests/reference/`).

### Highlights

- `parts` and `workers`: results never depend on `workers`; under an exposure cap, `parts` is part
  of the configuration (ADR-0015).
- Every metric with its Monte Carlo standard error, overall and by decile of the true ability
  (ADR-0016).
- Inputs checked where they enter, with messages that name the input, position, value and rule
  (ADR-0009).
- Parity tests against catR at three levels (ADR-0020).
