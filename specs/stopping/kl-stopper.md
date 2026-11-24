**Status:** Planned
**Module:** `catsim.stopping.bank.KLInfoStopper` (planned)
**Reference:** Chang and Ying (1996); Choi, Grady, and Dodd (2011)
**Last validated:** 2026-11-23, against Chang & Ying (1996) and Choi, Grady & Dodd (2011, PubMed Central author manuscript).
**Implementation guidance:** updated to the 1.0 strategy contract (`StepContext` in, a single criterion for `StopRule`) on 2026-11-24.

# Kullback-Leibler Information Stopper

## Motivation

The KL stopper is the global-information analogue of the minimum-information
bank stopper. Instead of looking only at Fisher information at a single point, it
uses integrated KL information, which remains meaningful even when the current
ability estimate is still uncertain.

It pairs naturally with KL-based item selection because selection and stopping
then operate on the same information scale.

No source proposes this rule. It applies the minimum-information rule (Choi et
al., 2011, Introduction; see the minimum-information stopper spec) to the KL
index that Chang and Ying (1996) defined for item selection.

## Definition

For each remaining item $i$, define the integrated KL information

$$
K_i(\hat{\theta})
=
\int_{\hat{\theta} - \delta_n}^{\hat{\theta} + \delta_n}
\left[
  P_i(\hat{\theta}) \log \frac{P_i(\hat{\theta})}{P_i(\theta)}
  +
  (1-P_i(\hat{\theta}))
  \log \frac{1-P_i(\hat{\theta})}{1-P_i(\theta)}
\right] d\theta,
$$

with half-width schedule

$$
\delta_n = \frac{c}{\sqrt{n}}.
$$

The test stops when

$$
\max_{i \in \mathcal{R}} K_i(\hat{\theta}) \le \kappa,
$$

where $\kappa$ is the configured KL threshold.

The integrand is Chang and Ying's KL item information
$K_i(\theta \,\|\, \hat{\theta})$ (1996, p. 217, Definition 2.1, Equation 12),
with their true ability $\theta_0$ replaced by the estimate, and the integral is
their KL index (p. 220, Equation 17). They derive $\delta_n = c/\sqrt{n}$ from
the confidence interval
$\hat{\theta}_n \pm c\,[I^{(n)}(\hat{\theta}_n)]^{-1/2}$, with $c$ chosen for a
coverage probability (p. 221, Equations 18 and 19), and used $c = 3$ (p. 223).
As in catsim's KL selector, $n$ is taken as 1 before the first item.

Because the window shrinks as items are administered, so does every item's
index, whatever the bank. Near $\hat{\theta}$ the divergence grows as
$\tfrac{1}{2} I_i(\hat{\theta})(\theta - \hat{\theta})^2$, since its curvature
there is the Fisher information (p. 219, Equation 16), so for a narrow window

$$
K_i(\hat{\theta})
\approx
\frac{I_i(\hat{\theta})\,\delta_n^3}{3}
=
\frac{c^3\, I_i(\hat{\theta})}{3\, n^{3/2}}.
$$

A fixed $\kappa$ is therefore reached partly through test length alone, which
is not the case for the Fisher threshold of the minimum-information rule.

## Parameters

```{eval-rst}
.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``min_kl``
     - required
     - Threshold :math:`\kappa` on the best remaining KL information.
   * - ``c``
     - ``3.0``
     - Positive schedule constant for the integration half-width; Chang and Ying (1996, p. 223)
       used :math:`c = 3`.
```

## Behavioral Contracts

Any correct implementation must satisfy the following properties:

1. The rule stops when the largest KL global-information score among remaining items is less than or equal to `min_kl`.
2. The integration half-width follows `c / sqrt(max(1, n_administered))`.
3. The implementation reuses the same KL kernel as the KL selector or an equivalent mathematically identical kernel.
4. If no items remain, the stopper terminates.
5. The stopper is a single criterion: the minimum test length and the length cap come from `StopRule`'s `min_items` and a `MaxLengthStopper` it is combined with.

## References

Chang, H.-H., & Ying, Z. (1996). A global information approach to computerized adaptive testing. *Applied Psychological Measurement*, 20(3), 213-229. <https://doi.org/10.1177/014662169602000303>. Open copy: [University of Minnesota Digital Conservancy](https://hdl.handle.net/11299/119092).

Choi, S. W., Grady, M. W., & Dodd, B. G. (2011). A new stopping rule for computerized adaptive testing. *Educational and Psychological Measurement*, 71(1), 37-53. <https://doi.org/10.1177/0013164410387338>. Open copy: [PubMed Central](https://pmc.ncbi.nlm.nih.gov/articles/PMC3028267/) (author manuscript without the journal's page numbers; cited here by section).

## Open Questions

To be answered before the stopper is implemented, and removed from the spec when it is promoted.

1. **Whether the threshold should follow the window.** As shown above, every item's index shrinks
   roughly as $c^3 I_i(\hat{\theta}) / (3 n^{3/2})$, so a fixed $\kappa$ (this spec) is reached
   partly because the test got longer, on any bank. The options are:
   - a fixed $\kappa$;
   - a threshold that shrinks with the window, $\kappa_0 \delta_n^3$, which compares roughly the
     same quantity as the minimum-information rule's Fisher threshold and so differs from that
     rule mainly for wide windows, early in the test;
   - not implementing this stopper, since no source proposes it.

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

from scipy.integrate import quad

from ..selection.kl import _kl_integrand


@dataclass(frozen=True)
class KLInfoStopper(BaseStopper):
  r"""Stop once no remaining item has more than a minimum KL information around the current estimate.

  Each remaining item's KL divergence from the current estimate is integrated over
  :math:`\hat\theta \pm c / \sqrt{\max(1, n)}`, as in :class:`~catsim.selection.KLSelector`, where
  :math:`n` is the number of administered items.

  Parameters
  ----------
  min_kl : float
      Threshold on the largest remaining KL information. Must not be negative.
  c : float, optional
      Half-width of the integration window before it narrows as items are administered. Must be
      positive. Default is 3.

  Raises
  ------
  ValueError
      If `min_kl` is negative or `c` is not positive (NaN included).
  """

  min_kl: float
  c: float = 3.0

  def __post_init__(self) -> None:
    """Validate the threshold and the window constant."""
    # written so that NaN fails the checks too
    if not self.min_kl >= 0:
      msg = f"min_kl must not be negative, got {self.min_kl}"
      raise ValueError(msg)
    if not self.c > 0:
      msg = f"c must be positive, got {self.c}"
      raise ValueError(msg)

  def stop(self, context: StepContext) -> bool:
    """Return whether no remaining item has more than `min_kl` KL information around the current estimate.

    Parameters
    ----------
    context : StepContext
        The session to check.

    Returns
    -------
    bool
        True if every remaining item's integrated KL information is at most `min_kl`, or no item
        remains; False otherwise.
    """
    candidates = _remaining(context)
    if candidates.size == 0:
      return True
    theta_hat = context.theta
    half_width = self.c / math.sqrt(max(1, len(context.administered_items)))
    items = context.item_bank.items
    largest = max(
      quad(_kl_integrand, theta_hat - half_width, theta_hat + half_width, args=(theta_hat, *items[i].tolist()))[0]
      for i in candidates
    )
    # Open Question 1: a threshold that follows the window would compare with
    # self.min_kl * half_width**3 instead.
    return largest <= self.min_kl
```

**Implementation notes:**

- The sketch shows the spec's current answer to Open Question 1, a fixed threshold; the last
  comparison is the only line the other answer changes.
- Contract 3 asks for KLSelector's kernel, but the stopping layer may not import the selection
  layer (`[tool.importlinter]` in `pyproject.toml`). Either move `_log_probabilities` and
  `_kl_integrand` from `catsim/selection/kl.py` to a module both layers import, such as
  `catsim.irt`, and import them from there in both, or add
  `"catsim.stopping.bank -> catsim.selection.kl"` to the contract's `ignore_imports`. Moving the
  kernel also removes the import of a private name.
- The window uses $n$ = 1 before the first item, as KLSelector does, so the stopper needs no
  minimum length of its own; a minimum and a cap come from `StopRule`, as in
  `StopRule(KLInfoStopper(0.01), MaxLengthStopper(30), min_items=5)`.
- One `quad` per remaining item per check, the KL selector's cost. If profiling shows it is hot,
  a fixed Gauss-Legendre rule over the window, vectorized over the items, is the first
  simplification to try.
- A stopper is pure (`BaseStopper`): the same context gives the same answer.

**Testing notes:**

- Contract 1, hand-worked through the definition's approximation: for a narrow window, an item's
  index is about $c^3 I_i(\hat\theta) / (3 n^{3/2})$. On a bank of identical items, check the
  computed index against that value after many items, and set `min_kl` just above and just below
  it to see the stopper stop and continue.
- Contract 2: the same bank and estimate with more administered items stop at a threshold that
  fewer items do not reach.
- Contract 3: the index matches `KLSelector`'s integral for the same item, estimate and window.
- Contract 4: with every item administered, it stops.
- Parameter validation: negative or NaN `min_kl`, and `c` not positive, are rejected.
