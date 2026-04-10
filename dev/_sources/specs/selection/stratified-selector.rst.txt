Stratified Selector
===================

:Status: Implemented
:Module: :py:class:`catsim.selection.StratifiedSelector`
:Reference: Chang & Ying (1999); Chang et al. (2001); Barrada et al. (2006)

Motivation
----------

The stratified selector is the abstract family base for selectors that partition
the item bank into stage-aligned strata. The common idea is that different parts
of the test should draw from different portions of a presorted bank so that item
usage and measurement characteristics evolve across the test instead of being
driven by one globally greedy rule.

In :mod:`catsim`, this class documents the concrete family contract used by the
implemented stratified selectors. It is not instantiated directly but defines the
shared slicing and stage-selection semantics.

Definition
----------

Let :math:`N` be the configured test size and :math:`J` the number of items in
the bank. The bank is divided into :math:`N` contiguous strata by equally spaced
slice boundaries

.. math::

   s_k = \left\lfloor \frac{kJ}{N} \right\rfloor,
   \qquad k = 0, 1, \dots, N-1.

At test position :math:`k+1`, the selector operates only on the stratum

.. math::

   \{s_k, s_k+1, \dots, s_{k+1}-1\}

after applying the subclass's presort and optional per-step postsort rule. The
selected item is the first non-administered item in the active stratum after that
ordering step.

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
     - Number of stages/strata the item bank is partitioned into.
   * - ``sort_once``
     - subclass-defined
     - Whether the subclass can reuse one presort for the entire test or must
       postsort on each step.

Behavioral Contracts
--------------------

Any correct implementation must satisfy the following properties:

1. The active stratum is determined solely by the number of administered items and the configured ``test_size``.
2. Only items from the active stratum are eligible for selection at that step.
3. If the subclass declares ``sort_once=True``, the presorted order is reused for repeated selections on the same item bank.
4. If the active stratum has no non-administered items left, the selector raises ``NoItemsAvailableError``.
5. The class itself is abstract; concrete behavior depends on subclass-specific presort and postsort rules.

References
----------

Chang, H.-H., & Ying, Z. (1999). A-stratified multistage computerized adaptive testing. *Applied Psychological Measurement*, 23(3), 211-221. https://doi.org/10.1177/01466219922031338

Chang, H.-H., Qian, J., & Ying, Z. (2001). A-stratified multistage computerized adaptive testing with b blocking. *Applied Psychological Measurement*, 25(4), 333-341. https://doi.org/10.1177/01466210122032181

Barrada, J. R., Mazuela, P., & Olea, J. (2006). Maximum information stratification method for controlling item exposure in computerized adaptive testing. *Psicothema*, 18(1), 156-159. https://pubmed.ncbi.nlm.nih.gov/17296025/

.. seealso::

   API reference: :py:class:`catsim.selection.StratifiedSelector`
