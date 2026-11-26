# ADR-0014: A Simulated Examinee Is a Response Pattern

## Status

Accepted 2026-11-25

## Context

Simulation studies use two kinds of examinee: generated ones, given by a true ability whose answers
are drawn from the model, and recorded ones, whose answers to every item come from real data, as in
a post-hoc study of a test given linearly. Real data pooled from several test forms has gaps.

`SimulationRunner.run` took true abilities only, and answered each item as it was given through a
`ResponseProvider`.

## Decision

- `simulate(engine, *, true_thetas=None, responses=None, rng=None, ...)` runs one session per
  examinee and returns a `SimulationResult` (ADR-0016). It takes true abilities, a response matrix,
  or both.
- A recorded examinee is a row of `responses`, one column per item of the bank (ADR-0004). An answer
  is a Boolean or 0 or 1, and NaN marks a missing response.
- A generated examinee is a recorded examinee whose row is drawn before the test, from its own
  stream (ADR-0013): one uniform draw $u_j$ per item, and the answer to item $j$ is correct when
  $u_j < P_j(\theta)$.
- Every session answers from its row. The items whose response is missing are excluded from the
  start (ADR-0011), so a recorded examinee is only given items they answered, and one who runs out
  of them ends with `NO_ITEMS` (ADR-0010).
- With both inputs, the rows are recorded examinees whose true abilities are known, as when a study
  draws its own response matrix.
- The metrics that need the true abilities, such as bias and RMSE, raise when the run has none.

## Consequences

- Generated and recorded examinees take one code path, and a study can move from one to the other.
- A generated examinee gives the same answer to an item whatever the engine and whenever the item
  comes, which is what common random numbers need (ADR-0013).
- Drawing a whole pattern costs one uniform per item per examinee, small next to a test.
- A post-hoc study on incomplete data runs without imputation.
- Response models other than the item bank's, such as an examinee who guesses or one whose ability
  drifts, are simulated by drawing the response matrix with them and passing it as recorded.

## Alternatives Considered

- A response callback, asked as each item is given: two kinds of examinee and two code paths, and a
  generated examinee's answers depend on the order of the items.
- Rejecting missing responses: post-hoc studies on pooled forms would need imputation first.
- Imputing missing responses inside catsim: it invents answers, and the choice of method belongs to
  the study.
