**Status:** Planned
**Module:** `catsim.stopping.bank.PredictedSEMStopper` (planned)
**Reference:** Choi, Grady, and Dodd (2011)
**Last validated:** 2026-11-23, against Choi, Grady & Dodd (2011, PubMed Central author manuscript).
**Implementation guidance:** updated to the 1.0 strategy contract (`StepContext` in, a single criterion for `StopRule`) on 2026-11-24.

# Predicted-SEM Stopper

## Motivation

Plain SE-threshold stopping can chase an unreachable target. If the remaining
item bank is too weak at the current ability estimate, a CAT may continue
administering items even though no available next item can push the standard
error below the desired threshold. It also stops as soon as the target is
reached, even when one or two more items would still improve precision
substantially (Choi et al., 2011, Introduction).

The predicted standard error reduction (PSER) rule of Choi, Grady, and Dodd
(2011) asks whether another item is worth administering. It predicts how much
the best remaining item would reduce the standard error, stops before the
target is reached when that reduction is negligible, and continues past the
target when it is still substantial (Method, "Stopping Criteria"; Discussion).

## Definition

Let $\hat{\theta}$ be the current estimate, $\mathcal{A}$ the set
of administered items, $\mathcal{R}$ the remaining items and $\tau$ the target
standard error. The current standard error is

$$
SE(\hat{\theta} \mid \mathcal{A})
=
\sqrt{\frac{1}{I(\hat{\theta} \mid \mathcal{A})}},
$$

as in catsim's minimum-error stopper. For a remaining item $j$, let $SE_j^{+}$
be the standard error predicted after administering it, and let

$$
\Delta
=
\max_{j \in \mathcal{R}}
\left[ SE(\hat{\theta} \mid \mathcal{A}) - SE_j^{+} \right]
$$

be the largest predicted reduction. With two thresholds $H^{+}$ and $H^{-}$:

- if $SE(\hat{\theta} \mid \mathcal{A}) < \tau$, the test stops unless
  $\Delta \ge H^{+}$;
- if $SE(\hat{\theta} \mid \mathcal{A}) \ge \tau$, the test continues unless
  $\Delta < H^{-}$.

Choi et al. (2011, Method, "Stopping Criteria") call $H^{+}$ and $H^{-}$ the
hyper and hypo parameters. They used $\tau = 0.3$, $H^{+} = 0.03$ and
$H^{-} = 0.01$, with tests of 3 to 15 items (Method, "CAT Simulations"), and set
$H^{+}$ three times as large as $H^{-}$ because they considered stopping after
the target is met the smaller risk (Discussion).

Choi et al. predict the standard error from the predictive posterior variance,
the objective of minimum expected posterior variance (MEPV) item selection
(Method, "Stopping Criteria"):

$$
SE_j^{+}
=
\left[
  \sum_{r} p_j(r \mid \mathbf{u})\,
  \mathrm{Var}(\theta \mid \mathbf{u}, U_j = r)
\right]^{1/2},
$$

where $p_j(r \mid \mathbf{u})$ is the posterior predictive probability of
response $r$ to item $j$; they estimated ability by EAP with a standard normal
prior (Method, "CAT Simulations"). The planned stopper departs from them: it
predicts the standard error from Fisher information, consistently with its
current standard error,

$$
SE_j^{+}
=
\sqrt{\frac{1}{I(\hat{\theta} \mid \mathcal{A}) + I_j(\hat{\theta})}},
$$

so the largest reduction comes from the remaining item with the largest
information at $\hat{\theta}$, and the rule works with any estimator.

## Parameters

```{eval-rst}
.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``min_error``
     - required
     - Target standard error threshold :math:`\tau`.
   * - ``hyper``
     - ``0.03``
     - Threshold :math:`H^{+}`: once the target is met, the test continues if some remaining item
       is predicted to reduce the standard error by at least this much.
   * - ``hypo``
     - ``0.01``
     - Threshold :math:`H^{-}`: before the target is met, the test stops if no remaining item is
       predicted to reduce the standard error by at least this much.
```

## Behavioral Contracts

Any correct implementation must satisfy the following properties:

1. If the current SE is below the target and no remaining item is predicted to reduce it by at least `hyper`, the stopper returns `True`.
2. If the current SE is below the target but some remaining item is predicted to reduce it by at least `hyper`, the criterion does not stop the test.
3. If the current SE is at or above the target, the criterion stops the test if and only if no remaining item is predicted to reduce it by at least `hypo`.
4. The stopper needs access to the full item bank because it depends on the remaining items, not only the administered set.
5. With no administered items, the stopper does not terminate solely from the predicted-SEM criterion.
6. The stopper is a single criterion: the minimum test length and the length cap come from `StopRule`'s `min_items` and a `MaxLengthStopper` it is combined with. With no remaining item, no reduction is possible, and it stops.

## References

Choi, S. W., Grady, M. W., & Dodd, B. G. (2011). A new stopping rule for computerized adaptive testing. *Educational and Psychological Measurement*, 71(1), 37-53. <https://doi.org/10.1177/0013164410387338>. Open copy: [PubMed Central](https://pmc.ncbi.nlm.nih.gov/articles/PMC3028267/) (author manuscript without the journal's page numbers; cited here by section).

## Open Questions

To be answered before the stopper is implemented, and removed from the spec when it is promoted.

1. **How to predict the next standard error.** This spec predicts it from Fisher information, as
   stated above; Choi et al. predict it from the predictive posterior variance, the MEPV kernel.
   The choice interacts with the 1.0 strategy contract, under which stoppers read the standard
   error the estimator reports (the step context's `se`). With EAP or MAP that is a posterior
   standard deviation, so the posterior-variance prediction would be the consistent one; with the
   likelihood estimators it would be the Fisher-information one. The options are:
   - Fisher information always (this spec);
   - the predictive posterior variance always, which needs a posterior on a grid, as MEPV does;
   - whichever matches the estimator's own standard error.

## Implementation Guidance

:::{note}
This section is for contributors and agents implementing this technique.
It is removed when the spec is promoted to "Implemented" status.
:::

**Proposed location:** `src/catsim/stopping/bank.py`, next to `MinInfoInBankStopper` and the
`_remaining` helper its spec defines.

**Mock implementation:**

```python
# src/catsim/stopping/bank.py (added to the module; the imports are those it does not have yet)

import math


@dataclass(frozen=True)
class PredictedSEMStopper(BaseStopper):
  """Stop once the best remaining item is not predicted to reduce the standard error enough.

  The predicted standard error reduction rule of Choi, Grady and Dodd (2011): once the standard
  error is below `min_error`, the test stops unless an item would still reduce it by `hyper`;
  before that, it continues unless no item would reduce it by `hypo`.

  Parameters
  ----------
  min_error : float
      Target standard error. Must be positive.
  hyper : float, optional
      Reduction that keeps the test going once the target is met. Must not be negative. Default
      is 0.03.
  hypo : float, optional
      Reduction below which the test stops before the target is met. Must not be negative.
      Default is 0.01.

  Raises
  ------
  ValueError
      If `min_error` is not positive, or `hyper` or `hypo` is negative (NaN included).
  """

  min_error: float
  hyper: float = 0.03
  hypo: float = 0.01

  def __post_init__(self) -> None:
    """Validate the target and the two thresholds."""
    # written so that NaN fails the checks too
    if not self.min_error > 0:
      msg = f"min_error must be positive, got {self.min_error}"
      raise ValueError(msg)
    if not (self.hyper >= 0 and self.hypo >= 0):
      msg = f"hyper and hypo must not be negative, got {self.hyper} and {self.hypo}"
      raise ValueError(msg)

  def stop(self, context: StepContext) -> bool:
    """Return whether the largest predicted reduction of the standard error is below its threshold.

    Parameters
    ----------
    context : StepContext
        The session to check.

    Returns
    -------
    bool
        True if the best remaining item is predicted to reduce the standard error by less than
        `hyper` when it is below `min_error`, or by less than `hypo` otherwise, or if no item
        remains; False otherwise, and always False before the first item.
    """
    if not context.administered_items:
      return False
    candidates = _remaining(context)
    # Contract 6: with no item left, nothing can reduce the error, whatever the thresholds.
    if candidates.size == 0:
      return True
    theta, item_bank = context.theta, context.item_bank
    # Open Question 1: the spec predicts from Fisher information at the estimate, for the current
    # error as for the predicted one. Judging the estimator's own error would start from
    # current = context.se and information = 1 / context.se**2; the posterior-variance prediction
    # would replace `predicted` with MEPV's expected posterior variance.
    information = irt.test_info(theta, item_bank.get_items(context.administered_items))
    current = 1 / math.sqrt(information) if information > 0 else math.inf
    best = float(irt.inf_hpc(theta, item_bank.get_items(candidates)).max())
    predicted = 1 / math.sqrt(information + best) if information + best > 0 else math.inf
    # Where no remaining item has information at the estimate, none can reduce the error.
    reduction = current - predicted if math.isfinite(predicted) else 0.0
    return reduction < (self.hyper if current < self.min_error else self.hypo)
```

**Implementation notes:**

- The sketch implements the spec's PSER rule (Definition), with the Fisher-information prediction
  the spec assumes for Open Question 1. The comment above the computation marks the two lines the
  other answers change. Under the Fisher choice the stopper does not read `context.se`, so with
  EAP or MAP its current error is not the one `StandardErrorStopper` judges; that difference is
  what the open question weighs.
- The rule replaces `StandardErrorStopper` rather than adding to it: both branches compare with
  the target. Combine it with a cap and a minimum length:
  `StopRule(PredictedSEMStopper(0.3), MaxLengthStopper(15), min_items=3)`, Choi et al.'s 3 to 15
  items.
- `min_error` is the spec's name for the target; `StandardErrorStopper` calls its target
  `max_se`. Choose one name for both when implementing.
- A stopper is pure (`BaseStopper`): the same context gives the same answer.

**Testing notes:**

- Hand-worked, on a bank of identical items with $a = 1$, $b = \hat\theta$ (information $1/4$
  each): after $n$ items the error is $2/\sqrt{n}$ and the reduction $2/\sqrt{n} - 2/\sqrt{n+1}$.
  With the defaults, it is $0.01003$ after 21 items (the test continues) and $0.00937$ after 22
  (it stops, Contract 3, with the error still $0.426$, above the $0.3$ target).
- Contract 1 and 2: after 45 such items the error is $0.298 < 0.3$; with only them remaining the
  reduction is $0.0033 < 0.03$ and the test stops, and with one remaining item of information 4 at
  $\hat\theta$ the reduction is $0.042$ and it continues.
- Contract 5: no administered items never stop.
- Contract 6: with every item administered, it stops.
- Parameter validation, NaN included.
