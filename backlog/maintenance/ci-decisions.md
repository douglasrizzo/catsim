---
title: Two CI choices - a timeout for the worker tests, and whether a broken link fails CI
kind: decision
status: needs-decision
priority: 3
effort: XS
blocked_by: []
raised_by: [parallel parts (#222), documentation sync (#226)]
---

# Two CI choices: a timeout for the worker tests, and whether a broken link fails CI

## 1. A timeout for the worker tests

Two mutations of `catsim.workers` make the test suite hang rather than fail:

- removing `_raise_failure` from `_relay`'s `queue.Empty` branch;
- dropping the end-of-task message in `_work`'s `finally`.

In CI, a hang like that runs into the job's 20-minute timeout before anything reports.

**Options:**

- (a) Add pytest-timeout to the dev group, with `@pytest.mark.timeout(60)` on the worker tests, so
  a hang fails within a minute and names the test.
- (b) Add a timeout to the tests' own waits.
- (c) Nothing.

**Recommendation:** (a).

## 2. Should a broken link fail CI?

The linkcheck step is `continue-on-error: true` (`.github/workflows/ci-cd.yml`), so a broken link
shows in the log but never fails a PR. `sphinx/conf.py` already ignores hosts that refuse
automated clients, and targets that only exist once a release deploys. With those ignored, the run
was clean in #226.

**Options:**

- (a) Fail the build on a broken link. A flaky host can then fail an unrelated PR.
- (b) Keep `continue-on-error`.
- (c) A scheduled weekly job that runs linkcheck and fails on its own, outside PRs.

**Recommendation:** (c).
