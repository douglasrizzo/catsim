# Maximum Expected Information Selector

:Class: {py:class}`catsim.selection.MEISelector`
:Sources: {cite:t}`vanderLinden1998`

## Motivation

Maximum-information selectors look only at the present estimate. MEI asks
which candidate item is expected to produce the most informative next test
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
the candidate item. The selector chooses the item, neither administered nor
excluded, with the largest expected information value.

{cite:author}`vanderLinden1996`'s {cite:year}`vanderLinden1996` report, which preceded the
{cite:year}`vanderLinden1998` article, defines MEI
with the observed information of **every response**, to the administered
items and the candidate's predicted one, each evaluated at
its updated estimate, and weights the two branches by the response
probabilities at the current estimate (p. 11, Equation 13). He notes that the
criterion therefore re-evaluates the information of the items already
administered. {cite:t}`Choi2009{Equation 3, extended to polytomous items}`
use the same whole-pattern measure. catsim follows this definition with Fisher
rather than observed information; the two are equal under the 2PL model the
report assumes (p. 10) and differ with guessing or slipping.

### Computation

Each selection estimates ability twice for every candidate item, once per
hypothetical response, so MEI is the slowest selector in catsim and its cost
grows in proportion to the bank. With the default estimator, one selection
after 20 items takes about 0.11 s on a bank of 500 items and 1 s on a bank of
5,000, measured on one core of an Apple M4 Pro. A simulation of 1,000 examinees
taking 20 items each from a 500-item bank spends about half an hour in item
selection.

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
       passed; EAP, MAP or WLE avoid the limitation described below. It may be the engine's own
       estimator: the selector reads only ``.theta`` from its look-ahead estimates, and the session
       never records them.
   * - ``r_max``
     - ``1.0``
     - Maximum exposure rate, in :math:`(0, 1]`. Items at or above it are left
       out before the expected information is computed, unless every remaining
       item is.
```

## Behavior and Comparison

With the default estimator (maximum likelihood with Dodd's heuristic), while the
responses so far are all correct, the hypothetical correct branch is still an
all-correct pattern, so Dodd's heuristic returns the same updated estimate for
every candidate; the same happens on the incorrect branch while all responses
are incorrect. On that branch the look-ahead is a fixed jump rather than an
estimate. Bayesian estimators and WLE give a genuine updated estimate on both
branches from the first item.

:::{dropdown} Comparison with catR

`catR` departs from van der Linden's definition: its
[`MEI.R`](https://github.com/cran/catR/blob/master/R/MEI.R) uses only the
**candidate item's** information at each updated estimate
{cite:p}`Magis12{p. 9, Table 1}`, observed information by default (p. 17). The two forms
can rank items differently. catsim keeps the whole-pattern measure of the
definition it follows, and no `MEISelector` setting scores the candidate alone.

The parity tests ({doc}`ADR-0020 </adr/0020-catr-as-the-reference>`) run catR's
MEI with `infoType = "Fisher"` and its Bayes modal estimator on $[-4, 4]$, which
`MEISelector(estimator=MAPEstimator(bounds=(-4, 4)))` matches. Before the first
response, where the administered test adds no information, catR's values and
ranking are catsim's. After two responses the values differ, a departure the
tests keep as an expected failure, although in that context the two rank the
items alike. No whole test compares the two.
:::

## Guarantees

1. It never returns an item that was already administered or excluded.
2. When every remaining item is at or above the exposure cap, it falls back to the
   remaining item with the largest expected information.
3. When no item remains, it raises `NoItemsAvailableError`.
4. Its look-ahead estimates never enter the session, so it may share the engine's
   estimator.

## References

```{bibliography}
:filter: docname in docnames
```

:::{seealso}
{doc}`/api/techniques/selection/mlwi-selector`, the likelihood-weighted information criterion
:::

## API Reference

```{eval-rst}
.. autoclass:: catsim.selection.MEISelector
   :members:
   :show-inheritance:
```
