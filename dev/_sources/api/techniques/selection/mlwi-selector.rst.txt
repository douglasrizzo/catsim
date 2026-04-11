Maximum Likelihood Weighted Information Selector
================================================

:Status: Planned
:Module: ``catsim.selection.weighted_info.MLWISelector`` (planned)
:Reference: Veerkamp & Berger (1997)

Motivation
----------

MLWI softens local maximum-information selection by weighting each item's
information curve by the likelihood of ability values given the observed
responses. This makes it a frequentist analogue of posterior-weighted selectors:
it spreads attention across plausible ability values without introducing a prior.

That positioning is useful strategically. It gives catsim a weighted-information
selector that is conceptually close to later Bayesian variants while staying
independent of the Bayesian infrastructure.

Definition
----------

For candidate item :math:`i`, MLWI is defined as

.. math::

   \mathrm{MLWI}_i
   =
   \int I_i(\theta)\,L(\mathbf{u}\mid\theta)\,d\theta.

The planned implementation approximates the integral on a fixed grid
:math:`\{\theta_q\}_{q=1}^{Q}` with spacing :math:`\Delta_q`:

.. math::

   \mathrm{MLWI}_i
   \approx
   \sum_{q=1}^{Q} I_i(\theta_q)\,L(\mathbf{u}\mid\theta_q)\,\Delta_q.

Likelihood values are normalized numerically with a log-sum-exp style shift
before exponentiation; the constant factor cancels under :math:`\arg\max`.

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Parameter
     - Default
     - Description
   * - ``n_nodes``
     - ``41``
     - Number of integration nodes used in the fixed-grid approximation. The feature card requires at least 5.
   * - ``bounds``
     - ``(THETA_MIN_EXTENDED, THETA_MAX_EXTENDED)``
     - Theta range covered by the fixed integration grid.
   * - ``r_max``
     - ``1.0``
     - Exposure cap applied before the final maximization.

Behavioral Contracts
--------------------

Any correct implementation must satisfy the following properties:

1. With an empty response vector, the likelihood weights reduce to a constant and the selector therefore reduces to choosing the largest integral of item information over the configured bounds.
2. On long tests with sharply peaked likelihoods, MLWI approaches maximum-information selection evaluated near the MLE.
3. The likelihood weighting must be computed in a numerically stable way that avoids underflow on long response vectors.
4. The selector requires the response vector; the current point estimate alone is insufficient to define the likelihood weights.
5. Exposure filtering is applied before the final argmax, with fallback to the unconstrained non-administered pool if every constrained candidate is excluded.

References
----------

Veerkamp, W. J. J., & Berger, M. P. F. (1997). Some new item selection criteria for adaptive testing. *Journal of Educational and Behavioral Statistics*, 22(2), 203-226. https://doi.org/10.3102/10769986022002203

.. seealso::

   Sister planned spec: :doc:`/api/techniques/selection/mei-selector`
