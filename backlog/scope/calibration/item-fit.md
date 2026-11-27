---
title: Item fit statistics
kind: technique
status: parked
priority: 4
effort: S
blocked_by: []
raised_by: []
---

# Item fit statistics

> Background: [the literature review §2.5](literature-review.md)

## Reference

Wright, B. D., & Masters, G. N. (1982). *Rating Scale Analysis*. MESA Press.

Yen, W. M. (1981). Using simulation results to choose a latent trait model. *Applied Psychological Measurement*, 5, 245–262.

## Theory

After calibration, fit statistics evaluate whether the estimated model describes each item's observed response pattern. These are essential diagnostics — calibration without fit checking is incomplete.

### Standardized residuals

For examinee $j$ on item $i$, the expected response is $E_{ij} = P_i(\hat\theta_j)$ and the variance is $W_{ij} = P_i(\hat\theta_j)(1 - P_i(\hat\theta_j))$. The standardized residual is:

$$z_{ij} = \frac{u_{ij} - E_{ij}}{\sqrt{W_{ij}}}$$

### Infit mean-square (information-weighted)

$$\text{infit}_i = \frac{\sum_j W_{ij} \cdot z_{ij}^2}{\sum_j W_{ij}} = \frac{\sum_j (u_{ij} - E_{ij})^2}{\sum_j W_{ij}}$$

Weighted by variance (= information), so it is most sensitive to unexpected responses from examinees whose ability is near the item's difficulty. Expected value = 1.0 under good fit.

### Outfit mean-square (unweighted)

$$\text{outfit}_i = \frac{1}{N} \sum_j z_{ij}^2 = \frac{1}{N} \sum_j \frac{(u_{ij} - E_{ij})^2}{W_{ij}}$$

Unweighted average of squared standardized residuals. Sensitive to outlier responses far from the item's difficulty. Expected value = 1.0.

### Interpretation

| Statistic | $< 0.7$ | $0.7 - 1.3$ | $> 1.3$ | $> 2.0$ |
|---|---|---|---|---|
| Infit/Outfit | Overfit (too predictable) | Good fit | Underfit (noisy) | Degraded measurement |

### Point-biserial correlation

$$r_{pb,i} = \frac{\bar\theta_1 - \bar\theta_0}{s_\theta} \cdot \sqrt{\frac{n_1 \cdot n_0}{N^2}}$$

where $\bar\theta_1$ is the mean ability of examinees who answered correctly, $\bar\theta_0$ for incorrect, $s_\theta$ is the overall ability SD, and $n_1$, $n_0$ are the counts. Alternatively, compute as the Pearson correlation between $u_{ij}$ and $\hat\theta_j$.

Items with $r_{pb} \leq 0$ have negative discrimination and should be flagged for removal. Items with $r_{pb} < 0.15$ have weak discrimination.

## Mock implementation

```python
# src/catsim/calibration/item_fit.py
"""Item fit statistics for calibrated IRT models."""

import numpy as np
import numpy.typing as npt
from typing import Any

from .. import irt


def item_fit(
    responses: npt.NDArray[np.integer],
    items: npt.NDArray[np.floating[Any]],
    theta: npt.NDArray[np.floating[Any]],
) -> dict[str, npt.NDArray[np.floating[Any]]]:
    """Compute infit, outfit, and point-biserial for all items.

    Parameters
    ----------
    responses : numpy.ndarray, shape (n_examinees, n_items)
        Binary response matrix. NaN for not-administered.
    items : numpy.ndarray, shape (n_items, 4+)
        Calibrated item parameters [a, b, c, d, ...].
    theta : numpy.ndarray, shape (n_examinees,)
        Ability estimates for each examinee.

    Returns
    -------
    dict with keys:
        'infit': numpy.ndarray, shape (n_items,)
        'outfit': numpy.ndarray, shape (n_items,)
        'point_biserial': numpy.ndarray, shape (n_items,)
    """
    n_examinees, n_items = responses.shape
    mask = np.isfinite(responses) & (responses >= 0)

    # Expected probabilities: (n_examinees, n_items)
    E = np.zeros_like(responses, dtype=float)
    for j in range(n_examinees):
        E[j] = irt.icc_hpc(theta[j], items)

    W = E * (1.0 - E)  # variance

    residual = np.where(mask, responses - E, 0.0)
    residual_sq = residual ** 2

    # Infit: information-weighted
    infit_num = np.sum(np.where(mask, residual_sq, 0.0), axis=0)
    infit_den = np.sum(np.where(mask, W, 0.0), axis=0)
    infit = infit_num / (infit_den + 1e-30)

    # Outfit: unweighted mean of z²
    z_sq = np.where(mask, residual_sq / (W + 1e-30), 0.0)
    n_admin = np.sum(mask, axis=0)
    outfit = np.sum(z_sq, axis=0) / (n_admin + 1e-30)

    # Point-biserial: Pearson correlation between u_ij and theta_j
    point_biserial = np.zeros(n_items)
    for i in range(n_items):
        admin_mask = mask[:, i]
        if admin_mask.sum() < 3:
            point_biserial[i] = np.nan
            continue
        u_i = responses[admin_mask, i].astype(float)
        th_i = theta[admin_mask]
        if np.std(u_i) < 1e-10 or np.std(th_i) < 1e-10:
            point_biserial[i] = np.nan
            continue
        point_biserial[i] = np.corrcoef(u_i, th_i)[0, 1]

    return {
        "infit": infit,
        "outfit": outfit,
        "point_biserial": point_biserial,
    }
```

## Proposed location

`src/catsim/calibration/item_fit.py`. Re-export from `src/catsim/calibration/__init__.py`.

## Tests

1. **Perfect fit**: Generate responses exactly from the model ($u_{ij} = 1$ if $P_i(\theta_j) > \text{uniform}$). With large N, infit and outfit should be close to 1.0 (within [0.8, 1.2]).
2. **Misfitting item**: Inject one item whose responses are random (ignoring ability). Its outfit should be $> 1.5$.
3. **Negative discrimination**: Create an item where high-ability examinees answer incorrectly more often. Its point-biserial should be $< 0$.
4. **All-correct / all-incorrect items**: Items with no variance in responses should produce `NaN` for point-biserial, not crash.
5. **Sparse data**: Responses with ~30% NaN. Verify that only administered responses contribute to each item's statistics.
6. **Symmetry**: Infit and outfit are invariant to examinee ordering.

## Notes

- These functions are standalone — they don't require MML-EM. They work with any calibrated item bank and ability estimates, including externally calibrated items.
- No external dependencies beyond numpy and the existing `irt.icc_hpc`.
- Future extensions: standardized infit/outfit (t-statistics via Wilson-Hilferty transformation), chi-square-based fit tests (Yen's Q1).
