Maximum-Information with B-Blocking Selector
============================================

:Status: Implemented
:Module: :py:class:`catsim.selection.MaxInfoBBlockSelector`
:Reference: Barrada et al. (2006); Chang et al. (2001)

Motivation
----------

MIS-B combines the MIS idea with a blocking step on the location of each item's
information peak. The goal is the same as in a-stratified b-blocking: reduce the
risk that stage strata inherit an unwanted ordering artifact from correlated item
parameters.

In the current package this is realized by ordering first on the theta value at
which each item reaches maximum information, then using maximum information to
form within-block ordering.

Definition
----------

For each item :math:`i`, let

.. math::

   \theta_i^{\max} = \arg\max_{\theta} I_i(\theta),
   \qquad
   I_i^{\max} = I_i(\theta_i^{\max}).

The package implementation presorts as follows:

1. Sort items by increasing :math:`\theta_i^{\max}`.
2. Partition that order into ``test_size`` contiguous strata.
3. Within each stratum, sort items by increasing :math:`I_i^{\max}`.
4. During selection, resort the active stratum by current information
   :math:`I_i(\hat{\theta})` in descending order.

The returned item is the first non-administered item in that final active-stratum
ordering.

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``test_size``
     - required
     - Number of strata and intended test length.

Behavioral Contracts
--------------------

Any correct implementation must satisfy the following properties:

1. The initial blocking variable is the theta value where each item reaches maximum information.
2. Within each block, items are presorted by increasing maximum information before runtime selection begins.
3. The active stratum is resorted dynamically by information at the current ``est_theta``.
4. The selector never returns an already administered item.
5. If the active stratum has been exhausted, it raises ``NoItemsAvailableError``.

References
----------

Barrada, J. R., Mazuela, P., & Olea, J. (2006). Maximum information stratification method for controlling item exposure in computerized adaptive testing. *Psicothema*, 18(1), 156-159. https://pubmed.ncbi.nlm.nih.gov/17296025/

Chang, H.-H., Qian, J., & Ying, Z. (2001). A-stratified multistage computerized adaptive testing with b blocking. *Applied Psychological Measurement*, 25(4), 333-341. https://doi.org/10.1177/01466210122032181

.. seealso::

   API reference: :py:class:`catsim.selection.MaxInfoBBlockSelector`
