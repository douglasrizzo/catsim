**Status:** Implemented
**Module:** {py:mod}`catsim.estimation.bayesian`; {py:class}`catsim.estimation.QuadratureGrid`
**Reference:** Bock & Mislevy (1982); Bock & Aitkin (1981)

# Bayesian Estimation Infrastructure

## Motivation

Bayesian estimators and selectors need the same substrate: a prior on
$\theta$, a fixed quadrature grid, and a numerically stable way to turn
responses into a discrete posterior. This page specifies that shared subsystem
so that later features do not duplicate incompatible posterior code.

The infrastructure is not an estimator by itself. The EAP estimator is built on
it, the MAP estimator reuses its prior factories, and the planned
posterior-weighted selectors (MPWI, KLP, MEPV) would reuse its posterior.

## Definition

For a response vector $\mathbf{u}$ on administered items with parameters
$\boldsymbol{\xi}$, the posterior is

$$
p(\theta \mid \mathbf{u})
=
\frac{p(\theta)\,L(\mathbf{u}\mid\theta)}
     {\int p(\theta)\,L(\mathbf{u}\mid\theta)\,d\theta}.
$$

On a fixed grid $\{\theta_q\}_{q=1}^{Q}$ with quadrature weights
$\{w_q\}_{q=1}^{Q}$, the discrete approximation is

$$
\tilde{p}(\theta_q \mid \mathbf{u})
\propto
w_q\,p(\theta_q)\,L(\mathbf{u}\mid\theta_q),
\qquad
\sum_{q=1}^{Q}\tilde{p}(\theta_q \mid \mathbf{u}) = 1.
$$

The log-likelihoods are summed at each node, as Bock and Mislevy (1982, p. 432)
recommend for adaptive testing, and the posterior is normalized after
subtracting the largest log-posterior value, so long response vectors do not
underflow. Downstream techniques reuse the same posterior moments:

$$
\hat{\theta}_{\mathrm{EAP}}
=
\sum_{q=1}^{Q} \theta_q \tilde{p}(\theta_q \mid \mathbf{u}),
\qquad
\operatorname{Var}(\theta \mid \mathbf{u})
=
\sum_{q=1}^{Q}(\theta_q-\hat{\theta}_{\mathrm{EAP}})^2
\tilde{p}(\theta_q \mid \mathbf{u}).
$$

These are Equations 4 and 5 of Bock and Mislevy (1982, p. 433), without the
square root on the variance; they credit the quadrature formulas to Bock and
Aitkin (1981) (p. 432). Like Bock and Mislevy, catsim uses evenly spaced nodes
rather than Gauss–Hermite points, which offer little advantage here because the
likelihood is not a polynomial (p. 433).

By default the grid has 41 equally spaced nodes on $[-6, 6]$ (spacing 0.3) and
the prior is $\mathcal{N}(0, 1)$.

### Grid resolution

The grid is accurate while it resolves the posterior. Compared with a
2,401-node reference, the default grid reproduces the posterior mean and
standard deviation to within 0.1% while the posterior standard deviation is at
least 0.2. Below that it degrades: errors reach about 7% at a posterior standard
deviation of 0.15 and about 44% at 0.11. For long tests or tight precision
targets, pass a finer grid, for example `QuadratureGrid.uniform(n_nodes=121)`
(spacing 0.1). This matches the advice of Bock and Mislevy (1982, p. 433), who
suggested about 80 points between $\pm 4$ standard deviations, also a spacing
of 0.1, to keep precision down to a posterior standard deviation of 0.2.

Bock and Mislevy also applied Sheppard's correction to the posterior variance
(p. 433).
catsim does not: the nodes are evaluated rather than grouped, so the
uncorrected moments already match the reference above, and the correction would
shrink the standard deviation by about 10% at a posterior standard deviation of
0.2.

## Parameters

```{eval-rst}
.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Parameter
     - Default
     - Description
   * - ``n_nodes``
     - ``41``
     - Number of quadrature nodes of ``QuadratureGrid.uniform``. Consuming
       estimators take a whole grid, so resolution can be tuned without
       changing the shared kernel.
   * - ``low``
     - ``THETA_MIN_EXTENDED``
     - Lower bound of the uniform grid.
   * - ``high``
     - ``THETA_MAX_EXTENDED``
     - Upper bound of the uniform grid.
   * - ``log_prior``
     - ``normal_log_prior(0, 1)``
     - Vectorized log-density used in posterior evaluation.
       ``uniform_log_prior(low, high)`` gives a flat prior instead.
```

## Behavioral Contracts

Any correct implementation must satisfy the following properties:

1. The posterior probabilities on the quadrature grid sum to 1 after normalization.
2. The posterior computation is numerically stable on long response vectors; it must not return NaNs or all-zero weights solely due to exponentiation underflow.
3. With an empty response vector, the posterior reduces to the normalized prior multiplied by the grid weights.
4. Posterior mean and posterior variance are computed from the same normalized discrete posterior and therefore remain internally consistent.
5. As informative items accumulate, posterior variance should generally contract rather than expand without bound.

## References

Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of item parameters: Application of an EM algorithm. *Psychometrika*, 46(4), 443-459. <https://doi.org/10.1007/BF02293801>. Not openly available; Bock and Mislevy (1982, p. 432) describe its quadrature formulas.

Bock, R. D., & Mislevy, R. J. (1982). Adaptive EAP estimation of ability in a microcomputer environment. *Applied Psychological Measurement*, 6(4), 431-444. <https://doi.org/10.1177/014662168200600405>. Open copy: [University of Minnesota Digital Conservancy](https://hdl.handle.net/11299/101546).

:::{seealso}
{doc}`/api/techniques/estimation/eap-estimator`,
{doc}`/api/techniques/estimation/map-estimator`
:::

## API Reference

```{eval-rst}
.. autoclass:: catsim.estimation.QuadratureGrid
   :members:
   :show-inheritance:

.. autofunction:: catsim.estimation.normal_log_prior

.. autofunction:: catsim.estimation.uniform_log_prior

.. autofunction:: catsim.estimation.posterior

.. autofunction:: catsim.estimation.posterior_mean

.. autofunction:: catsim.estimation.posterior_variance
```
