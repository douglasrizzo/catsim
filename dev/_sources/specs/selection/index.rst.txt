Selection Techniques
####################

Item selection techniques determine which item from the bank to administer at each
step of an adaptive test. The choice of selector affects both measurement precision
and practical properties such as item exposure and content coverage.

Implemented
***********

.. toctree::
   :maxdepth: 1

   max-info-selector
   urry-selector
   interval-info-selector
   cluster-selector
   stratified-selector
   a-strat-selector
   a-strat-b-block-selector
   max-info-strat-selector
   max-info-b-block-selector

The following selectors have straightforward implementations that do not require
detailed algorithmic specification:

- :py:class:`~catsim.selection.RandomSelector` — selects an item uniformly at random
  from the unadministered pool.
- :py:class:`~catsim.selection.RandomesqueSelector` — selects uniformly at random
  from the *k* most informative unadministered items.
- :py:class:`~catsim.selection.LinearSelector` — selects items in fixed sequential order.
- :py:class:`~catsim.selection.The54321Selector` — selects items in reverse order of
  difficulty, stepping down across five difficulty levels.

Planned
*******

.. warning::

   The techniques in this section are not yet available in the released package.
   They describe planned additions to :mod:`catsim.selection`.

.. toctree::
   :maxdepth: 1

   kl-selector
   progressive-selector
   proportional-selector
   mei-selector
   mlwi-selector
   sympson-hetter-exposure
   mpwi-selector
   klp-selector
   mepv-selector
