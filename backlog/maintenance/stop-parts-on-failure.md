---
title: Stop the running parts when one part of a parallel run fails
kind: maintenance
status: ready
priority: 4
effort: S
blocked_by: []
raised_by: [parallel parts (#222)]
---

# Stop the running parts when one part of a parallel run fails

## Problem

When one part of `simulate(..., parts=k, workers=w)` fails, `simulate` raises the failure. The parts
already running keep going in the background until they finish, and their results are dropped. On a
long run that wastes the workers' time after the error is already known.

Python 3.14's `concurrent.futures.ProcessPoolExecutor.terminate_workers()` can stop them. catsim
supports Python 3.11+.

## Proposal

On a failure, call `terminate_workers()` where it exists, and keep today's behaviour below 3.14.
Check that a terminated pool leaves no orphan processes and that the original error still reaches
the caller unchanged.

## Tests

On Python 3.14+: a run whose second part fails returns the failure while the other parts are
stopped. The test checks that the pool's processes are gone. Skip it below 3.14.
