Maximum Expected Information Selector
=====================================

:Status: Planned
:Module: ``catsim.selection.expected_info.MEISelector`` (planned)
:Reference: van der Linden (1998)

Motivation
----------

Maximum-information selectors look only at the present estimate. MEI instead
asks which candidate item is expected to produce the most informative next test
state after the next response is observed.

That one-step-ahead view makes MEI stronger than plain Fisher selection in cases
where the current estimate is likely to move substantially after the next item.
It is therefore a forward-looking selector rather than a purely local one.

Definition
----------

For candidate item :math:`i`, let :math:`P_i(\hat{\theta})` be the probability
of a correct response at the current estimate. Let
:math:`\hat{\theta}_i^{(1)}` and :math:`\hat{\theta}_i^{(0)}` be the updated
ability estimates after a hypothetical correct or incorrect response to item
:math:`i`. Then

.. math::

   \mathrm{MEI}_i
   =
   P_i(\hat{\theta})\,I(\hat{\theta}_i^{(1)})
   +
   [1-P_i(\hat{\theta})]\,I(\hat{\theta}_i^{(0)}),

where :math:`I(\cdot)` is the information of the administered test extended by
the candidate item. The selector chooses the non-administered item with the
largest expected information value.

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Parameter
     - Default
     - Description
   * - ``estimator``
     - ``NumericalSearchEstimator()``
     - Estimator used to compute the hypothetical one-step-ahead ability updates. The card explicitly allows alternative estimators such as WLE, EAP, or MAP.
   * - ``r_max``
     - ``1.0``
     - Exposure cap applied after candidate filtering.

Behavioral Contracts
--------------------

Any correct implementation must satisfy the following properties:

1. The score for each candidate equals :math:`P\,I_{\text{correct}} + (1-P)\,I_{\text{incorrect}}` under the configured estimator.
2. The selector requires the observed response vector so it can evaluate one-step-ahead updates from the actual response history.
3. On at least some banks, MEI must differ from plain maximum-information selection; otherwise it is not implementing the forward-looking criterion.
4. Both hypothetical response branches for every candidate item are evaluated before selection.
5. Exposure filtering applies to the candidate set before the final argmax, with the same no-items behavior as other selectors.

References
----------

van der Linden, W. J. (1998). Bayesian item selection criteria for adaptive testing. *Psychometrika*, 63(2), 201-216. https://doi.org/10.1007/BF02294772

Choi, S. W., & Swartz, R. J. (2009). Comparison of CAT item selection criteria for polytomous items. *Applied Psychological Measurement*, 33(6), 419-440.

.. seealso::

   Sister planned spec: :doc:`/techniques/selection/mlwi-selector`
