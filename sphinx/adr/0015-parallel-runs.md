# ADR-0015: A Run Is Split Into Parts, and the Parts Run on Workers

## Status

Accepted 2026-11-25

## Context

The runner ran its sessions one after another. A recipe in the documentation split a study into
parts with seeds of their own, ran each in a process and built one result from the parts by hand:
four processes ran 4000 examinees in 3.2 s against 9.1 s in one, with identical part results. The
recipe failed from a notebook on macOS, whose worker processes cannot import a function defined in
the notebook, and an engine using EAP or MAP with the default prior could not be sent to a process,
since that prior was a local function.

Exposure feedback makes a session depend on the sessions before it (ADR-0012), so a run with it
cannot give its sequential result in parallel. Parts with their own feedback were checked: with
`r_max = 0.2` and 8 parts, runs on 4 and on 2 processes were identical, and the cap held, with a
largest exposure rate of 0.200, as in the sequential run. catR runs sequentially.

## Decision

- `simulate(..., parts=1, workers=1)` splits the run's examinees, in order, into `parts` contiguous
  parts of near-equal size, and runs the parts on `workers` processes.
- The parts are part of the configuration: each part runs its sessions in order, with exposure
  counts of its own. The result records `parts`.
- The output depends on the configuration, the examinees, the seed and `parts`; never on `workers`,
  on which process runs which part, or on timing. `parts=1` is the sequential run.
- Without exposure feedback, `parts` does not change the output either, since every examinee draws
  from its own streams (ADR-0013).
- The workers are processes from Python's standard library. The engine is sent to them, so every
  built-in strategy is picklable. An engine whose strategies the workers cannot import, such as a
  class defined in a notebook or a script's main module, raises an error before any part runs, saying
  that the class must move to a module.
- `progress`, a callable or `None`, is called in the calling process with the number of sessions
  finished and the run's total (ADR-0021).

## Consequences

- A study runs from a notebook in parallel with the built-in strategies.
- A run's result is reproduced on any machine by its seed and `parts`, whatever the number of cores.
- With exposure feedback and `parts > 1`, each session sees the exposure of the earlier sessions of
  its part only, so the result differs from the sequential one, which `parts=1` gives.
- Parallelism across replications or conditions is the study's own loop (ADR-0016).
- Custom strategies must be picklable to run on workers, which a class in a module is.

## Alternatives Considered

- Workers only, refused with exposure feedback: most CAT studies use exposure control, and they
  would stay sequential.
- Splitting by `workers`: the result would change with the number of cores.
- Exposure counts shared between workers: the result would depend on timing.
- The recipe alone: it fails from notebooks, and every study writes the same stitching code.
