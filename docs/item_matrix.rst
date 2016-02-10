The Item Matrix
===============

In catsim, an item bank, or matrix, is a :py:class:`numpy.ndarray` whose rows represent items and columns represent item characteristics. The most important characteristics are situated in the first three columns of the matrix:

* column 1 contains item discrimination, commonly known in the literature as the :math:`a` parameter. For all purposes of the simulation, :math:`a > 0`;
* column 2 contains item difficulty, or the :math:`b` parameter. Usually, :math:`b \in [-\inf; \inf]`, but in practice, the values of :math:`b` depend on the scale examinees' proficiencies are being measured. In literature, both proficiencies and item difficulties are standardized to :math:`N(0; 1)` and it is in that domain that catsim has been mostly tested;
* column 3 contains the item's pseudo-guessing parameter, which represents a fixed probability all examinees have of answering the item correctly. Being a probabiliy :math:`0 > c >1`.

After the simulation, catsim adds a fourth column to the item matrix, representing the items exposure rate, commonly denoted as :math:`r`. Its value denotes how many times an item has been used and it is calculated as follows:

.. math:: r_i = \frac{q_i}{N}

Where :math:`q_i` represents the number of tests item :math:`i` has been used and :math:`N` is the total number of tests applied.
