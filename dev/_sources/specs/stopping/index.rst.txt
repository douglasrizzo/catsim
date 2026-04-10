Stopping Criteria
#################

Stopping criteria determine when an adaptive test should terminate. A stopper
evaluates the current session state after each item and signals whether testing
should continue or stop.

Implemented
***********

.. toctree::
   :maxdepth: 1

   predicted-sem-stopper
   min-info-bank-stopper
   kl-stopper
   sprt-stopper
   glr-stopper
   stochastic-curtailment-stopper

   test-length-stopper
   min-error-stopper
   confidence-interval-stopper

Planned
*******

.. warning::

   The techniques in this section are not yet available in the released package.
   They describe planned additions to :mod:`catsim.stopping`.

.. toctree::
   :maxdepth: 1
