Cluster Selector
================

:Status: Implemented
:Module: :py:class:`catsim.selection.ClusterSelector`
:Reference: Meneghetti (2015)

Motivation
----------

Cluster-based selection is an exposure-management strategy that organizes items
into externally supplied groups and chooses from those groups instead of from the
entire bank at once. This makes it possible to distribute usage across groups of
similar items while still preserving an information-based rule inside the chosen
cluster.

In this package the selector is explicitly operational rather than purely
theoretical: users supply the cluster memberships, choose how cluster quality is
scored, and choose how to behave when all items in the chosen cluster exceed the
exposure cap.

Definition
----------

Let :math:`C_g` denote cluster :math:`g`, and let :math:`R_k` denote the
remaining items. The package supports three cluster-scoring rules:

.. math::

   S_g^{\mathrm{item}}(\hat{\theta})
   =
   \max_{i \in C_g \cap R_k} I_i(\hat{\theta}),

.. math::

   S_g^{\mathrm{cluster}}(\hat{\theta})
   =
   \sum_{i \in C_g} I_i(\hat{\theta}),

.. math::

   S_g^{\mathrm{weighted}}(\hat{\theta})
   =
   \frac{1}{|C_g|}\sum_{i \in C_g} I_i(\hat{\theta}).

The selector first picks the best cluster according to the configured scoring
rule among clusters that still contain non-administered items. It then chooses an
item inside that cluster:

.. math::

   i^*
   =
   \arg\max_{i \in C_{g^*} \cap R_k,\ r_i < r_{\max}} I_i(\hat{\theta})

whenever such items exist. If none do, ``r_control="passive"`` falls back to the
highest-information item in the cluster, whereas ``r_control="aggressive"``
chooses the lowest-exposure item in the cluster.

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``clusters``
     - required
     - Cluster membership label for each item in the bank.
   * - ``method``
     - ``"item_info"``
     - Cluster-scoring rule: maximum item information, sum of cluster
       information, or average cluster information.
   * - ``r_max``
     - ``1.0``
     - Soft maximum exposure rate for choosing an item inside the selected cluster.
   * - ``r_control``
     - ``"passive"``
     - Fallback rule when every remaining item in the chosen cluster violates
       the exposure cap.

Behavioral Contracts
--------------------

Any correct implementation must satisfy the following properties:

1. It selects a cluster before selecting an item, and the cluster rule depends only on the configured ``method``.
2. The returned item always belongs to the selected cluster and has not been administered previously.
3. If at least one eligible item in the selected cluster has exposure rate below ``r_max``, the returned item is the highest-information such item.
4. Under passive control, exposure-cap failure inside the chosen cluster falls back to the highest-information remaining item in that cluster.
5. Under aggressive control, exposure-cap failure inside the chosen cluster falls back to the lowest-exposure remaining item in that cluster.

References
----------

Meneghetti, D. R. (2015). *Metodologia de selecao de itens em testes adaptativos informatizados baseada em agrupamento por similaridade* (Master's thesis). Centro Universitario da FEI. https://www.researchgate.net/publication/283944553_Metodologia_de_selecao_de_itens_em_Testes_Adaptativos_Informatizados_baseada_em_Agrupamento_por_Similaridade

.. seealso::

   API reference: :py:class:`catsim.selection.ClusterSelector`
