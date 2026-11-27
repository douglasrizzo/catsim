---
title: Give the worker tests a timeout, so a hang fails fast?
kind: decision
status: needs-decision
priority: 3
effort: XS
blocked_by: []
raised_by: [parallel parts (#222)]
---

# Give the worker tests a timeout, so a hang fails fast?

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
