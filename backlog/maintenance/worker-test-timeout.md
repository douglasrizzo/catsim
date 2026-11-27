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

## The question

Two mutations of `catsim.workers` make the test suite hang rather than fail:

- removing `_raise_failure` from `_relay`'s `queue.Empty` branch;
- dropping the end-of-task message in `_work`'s `finally`.

In CI, a hang like that runs into the job's 20-minute timeout before anything reports.

## Options

1. **pytest-timeout in the dev group,** with a per-test timeout on the worker tests, for example
   `@pytest.mark.timeout(60)`. A hang then fails within a minute, naming the test.
2. **A timeout argument** on the tests' own waits, if every hang passes through a wait the test
   controls.
3. **Nothing.** The job timeout catches it, slowly.

Recommendation: option 1. It is one dev dependency, and the hang fails as a test.
