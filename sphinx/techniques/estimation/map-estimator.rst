Maximum A Posteriori Estimator
==============================

:Status: Planned
:Module: ``catsim.estimation.bayesian.MAPEstimator`` (planned)
:Reference: Samejima (1969); Bock & Aitkin (1981)

Motivation
----------

MAP is the mode-based Bayesian counterpart to MLE. It adds prior information to
the likelihood objective but keeps the optimization workflow familiar, making it
cheaper than quadrature-based EAP while still regularizing extreme short-test
patterns.

This makes MAP the natural option for users who want Bayesian shrinkage without
the overhead of computing posterior moments on every call.

Definition
----------

The MAP estimate is the posterior mode:

.. math::

   \hat{\theta}_{\mathrm{MAP}}
   =
   \arg\max_{\theta}\,
   p(\theta)\,L(\mathbf{u}\mid\theta)
   =
   \arg\max_{\theta}\,
   \left[\log p(\theta)+\log L(\mathbf{u}\mid\theta)\right].

Equivalently, the planned estimator minimizes the penalized negative log-posterior

.. math::

   -\log p(\theta) - \log L(\mathbf{u}\mid\theta)

over the configured theta bounds. Unlike EAP, this requires no quadrature grid;
it only requires a prior function and a scalar optimizer.

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Parameter
     - Default
     - Description
   * - ``log_prior``
     - ``normal_log_prior(0, 1)``
     - Prior used to regularize the likelihood objective.
   * - ``tol``
     - ``1e-6``
     - Tolerance passed to the bounded scalar optimizer.
   * - ``bounds``
     - ``(THETA_MIN_EXTENDED, THETA_MAX_EXTENDED)``
     - Search range for the penalized objective.
   * - ``verbose``
     - ``False``
     - Planned diagnostics flag inherited from the estimator base class.

Behavioral Contracts
--------------------

Any correct implementation must satisfy the following properties:

1. With no administered items, the estimate defaults to the mode of the configured prior over the search range.
2. For short tests with a centered prior, the MAP estimate is closer to the prior mean than the corresponding plain MLE on the same data.
3. Shifting the prior mean while holding the response data fixed shifts the MAP estimate in the same direction.
4. On long, information-rich tests, MAP and EAP should converge toward similar values because the posterior becomes sharply concentrated.
5. The estimator returns finite values on extreme response patterns because the prior keeps the objective bounded.

References
----------

Samejima, F. (1969). Estimation of latent ability using a response pattern of graded scores. *Psychometrika Monograph Supplement*, No. 17.

Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of item parameters. *Psychometrika*, 46, 443-459.

.. seealso::

   Related planned specs: :doc:`/techniques/estimation/bayesian-infrastructure`,
   :doc:`/techniques/estimation/eap-estimator`
