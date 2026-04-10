Bayesian Estimation Infrastructure
==================================

:Status: Planned
:Module: ``catsim.estimation.bayesian`` (planned)
:Reference: Bock & Mislevy (1982); Bock & Aitkin (1981)

Motivation
----------

Several planned estimators and selectors need the same Bayesian substrate: a
prior on :math:`\theta`, a fixed quadrature grid, and a numerically stable way
to turn responses into a discrete posterior. This page specifies that shared
subsystem so later features do not duplicate incompatible posterior code.

The infrastructure is intentionally not an estimator by itself. Its job is to
provide common posterior primitives for EAP, MAP-adjacent utilities, and
posterior-weighted selectors such as MPWI, KLP, and MEPV.

Definition
----------

For a response vector :math:`\mathbf{u}` on administered items with parameters
:math:`\boldsymbol{\xi}`, the posterior is

.. math::

   p(\theta \mid \mathbf{u})
   =
   \frac{p(\theta)\,L(\mathbf{u}\mid\theta)}
        {\int p(\theta)\,L(\mathbf{u}\mid\theta)\,d\theta}.

On a fixed grid :math:`\{\theta_q\}_{q=1}^{Q}` with quadrature weights
:math:`\{w_q\}_{q=1}^{Q}`, the discrete approximation is

.. math::

   \tilde{p}(\theta_q \mid \mathbf{u})
   \propto
   w_q\,p(\theta_q)\,L(\mathbf{u}\mid\theta_q),
   \qquad
   \sum_{q=1}^{Q}\tilde{p}(\theta_q \mid \mathbf{u}) = 1.

Downstream techniques then reuse the same posterior moments:

.. math::

   \hat{\theta}_{\mathrm{EAP}}
   =
   \sum_{q=1}^{Q} \theta_q \tilde{p}(\theta_q \mid \mathbf{u}),
   \qquad
   \operatorname{Var}(\theta \mid \mathbf{u})
   =
   \sum_{q=1}^{Q}(\theta_q-\hat{\theta}_{\mathrm{EAP}})^2
   \tilde{p}(\theta_q \mid \mathbf{u}).

The default planned configuration is a uniform grid over the extended theta
range with a standard normal prior, matching the conventions already used in the
feature card.

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
     - Number of quadrature nodes. Exposed on consuming estimators/selectors so
       integration resolution can be tuned without changing the shared kernel.
   * - ``low``
     - ``THETA_MIN_EXTENDED``
     - Lower bound of the planned uniform grid.
   * - ``high``
     - ``THETA_MAX_EXTENDED``
     - Upper bound of the planned uniform grid.
   * - ``log_prior``
     - ``normal_log_prior(0, 1)``
     - Vectorized log-density used in posterior evaluation. The infrastructure
       also plans to expose a uniform prior factory for flatter weighting.

Behavioral Contracts
--------------------

Any correct implementation must satisfy the following properties:

1. The posterior probabilities on the quadrature grid sum to 1 after normalization.
2. The posterior computation is numerically stable on long response vectors; it must not return NaNs or all-zero weights solely due to exponentiation underflow.
3. With an empty response vector, the posterior reduces to the normalized prior multiplied by the grid weights.
4. Posterior mean and posterior variance are computed from the same normalized discrete posterior and therefore remain internally consistent.
5. As informative items accumulate, posterior variance should generally contract rather than expand without bound.

References
----------

Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of item parameters. *Psychometrika*, 46, 443-459.

Bock, R. D., & Mislevy, R. J. (1982). Adaptive EAP estimation of ability in a microcomputer environment. *Applied Psychological Measurement*, 6(4), 431-444. https://doi.org/10.1177/014662168200600405

Standard fixed-grid and Gauss-Hermite quadrature references apply to the numerical integration scheme.

.. seealso::

   Related planned specs: :doc:`/techniques/estimation/eap-estimator`,
   :doc:`/techniques/estimation/map-estimator`
