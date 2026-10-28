**Status:** Implemented
**Module:** {py:class}`catsim.selection.MEISelector`
**Reference:** van der Linden (1998)

# Maximum Expected Information Selector

## Motivation

Maximum-information selectors look only at the present estimate. MEI instead
asks which candidate item is expected to produce the most informative next test
state after the next response is observed.

That one-step-ahead view makes MEI stronger than plain Fisher selection in cases
where the current estimate is likely to move substantially after the next item.
It is therefore a forward-looking selector rather than a purely local one.

## Definition

For candidate item $i$, let $P_i(\hat{\theta})$ be the probability
of a correct response at the current estimate. Let
$\hat{\theta}_i^{(1)}$ and $\hat{\theta}_i^{(0)}$ be the updated
ability estimates after a hypothetical correct or incorrect response to item
$i$. Then

$$
\mathrm{MEI}_i
=
P_i(\hat{\theta})\,I(\hat{\theta}_i^{(1)})
+
[1-P_i(\hat{\theta})]\,I(\hat{\theta}_i^{(0)}),
$$

where $I(\cdot)$ is the information of the administered test extended by
the candidate item. The selector chooses the non-administered item with the
largest expected information value.

This departs from the published criterion. Magis and Raîche (2012, p. 9,
Table 1, and p. 10) and Choi and Swartz (2009, Equation 3, extended to
polytomous items), who attribute MEI to van der Linden (1998), weight the
**candidate item's** information at each
updated estimate, not the whole test's. Both use the observed information by
default (Magis & Raîche, 2012, p. 17); Choi and Swartz also report a variant
with Fisher information. The two forms can rank items differently.

With the default estimator (maximum likelihood with Dodd's heuristic), while the
responses so far are all correct, the hypothetical correct branch is still an
all-correct pattern, so Dodd's heuristic returns the same updated estimate for
every candidate; the same happens on the incorrect branch while all responses
are incorrect. On that branch the look-ahead is a fixed jump rather than an
estimate. Bayesian estimators and WLE give a genuine updated estimate on both
branches from the first item.

## Parameters

```{eval-rst}
.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Parameter
     - Default
     - Description
   * - ``estimator``
     - ``NumericalSearchEstimator()``
     - Estimator used to compute the hypothetical one-step-ahead ability updates. Any estimator can be
       passed; EAP, MAP or WLE avoid the limitation described above.
   * - ``r_max``
     - ``1.0``
     - Exposure cap applied after candidate filtering.
```

## Behavioral Contracts

Any correct implementation must satisfy the following properties:

1. The score for each candidate equals $P\,I_{\text{correct}} + (1-P)\,I_{\text{incorrect}}$ under the configured estimator.
2. The selector requires the observed response vector so it can evaluate one-step-ahead updates from the actual response history.
3. On at least some banks, MEI must differ from plain maximum-information selection; otherwise it is not implementing the forward-looking criterion.
4. Both hypothetical response branches for every candidate item are evaluated before selection.
5. Exposure filtering applies to the candidate set before the final argmax, with the same no-items behavior as other selectors.

## References

van der Linden, W. J. (1998). Bayesian item selection criteria for adaptive testing. *Psychometrika*, 63(2), 201-216. <https://doi.org/10.1007/BF02294775>. Not openly available; the 1996 research report that preceded it is listed in the [University of Twente repository](https://research.utwente.nl/en/publications/bayesian-item-selection-criteria-for-adaptive-testing-2/), and Magis and Raîche (2012) and Choi and Swartz (2009) describe the criterion.

Choi, S. W., & Swartz, R. J. (2009). Comparison of CAT item selection criteria for polytomous items. *Applied Psychological Measurement*, 33(6), 419-440. <https://doi.org/10.1177/0146621608327801>. Open copy: [PubMed Central](https://pmc.ncbi.nlm.nih.gov/articles/PMC2791416/).

Magis, D., & Raîche, G. (2012). Random generation of response patterns under computerized adaptive testing with the R package catR. *Journal of Statistical Software*, 48(8), 1-31. <https://doi.org/10.18637/jss.v048.i08> (open access).

:::{seealso}
Sister spec: {doc}`/api/techniques/selection/mlwi-selector`
:::

## API Reference

```{eval-rst}
.. autoclass:: catsim.selection.MEISelector
   :members:
   :show-inheritance:
```
