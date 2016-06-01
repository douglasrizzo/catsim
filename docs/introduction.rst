Introduction
************

Assessment instruments are widely used to measure individuals *latent traits*, that is, internal characteristics that cannot be directly measured. An example of such assessment instruments are educational and psychological tests. Each test is composed of a series of items and an examinee's answers to these items allow for the measurement of one or more of his or hers latent traits. When a latent trait is expressed in numerical form, it is called an *ability* or *proficiency*.

Ordinary tests, hereon called linear tests, are applied using the orthodox paper and pencil strategy, in which tests are printed and all examinees are presented with the same items in the same order. One of the drawbacks of this methodology is that both individuals with high and low proficiencies must answer all items in order to have their proficiency estimated. An individual with high proficiency might get bored of answering the whole test if it only contains items that he or she considers easy; on the other hand, an individual of low proficiency might get frustrated if he is confronted by items considered and hard and might give up on the test or answer the items without paying attention.

With these concerns in mind, a new paradigm in assessment emerged in the 70s. Initially named *tailored testing* in [Lord77]_, these were tests in which items were chosen to be presented to the examinee in real time, based on the examinee's responses to previous items. The name was changed to computerized adaptive testing (CAT) due to the advances in techonology that facilitated the application of such a testing methodology using electronic devices, like computers and tablets.

In a CAT, the examinee's proficiency is evaluated after the response of each item. The new proficiency is then used to select a new item, closer to the examinee's real proficiency. This method of test application has several advantages compared to the traditional paper-and-pencil method, since high-proficiency examinees are not required to answer all the easy items in a test, answering only the items that actually give some information regarding his or hers true knowledge of the subject at matter. A similar, but inverse effect happens for those examinees of low proficiency level.

Finally, the advent of CAT allowed for researchers to create their own variant ways of starting a test, choosing items, estimating proficiencies and stopping the test. Fortunately, the mathematical formalization provided by Item Response Theory (IRT) allows for tests to be computationally simulated and the different methodologies of applying a CAT to be compared under different constraints. Packages with these functionalities already exist in the R language ([Magis12]_) but not yet in Python. :py:mod:`catsim` was created to fill this gap, using the facilities of established scientific packages such as :py:mod:`numpy` and :py:mod:`scipy`, as well as the object-oriented programming paradigm supported by Python to create a simple, comprehensive and user-extendable CAT simulation package.

Item Response Theory Models
===========================

As a CAT simulator, :py:mod:`catsim` borrows many concepts from Item Response Theory ([Lord68]_ and [Rasch66]_), a series of models created in the second part of the 20th century with the goal of *measuring latent traits*. :py:mod:`catsim` makes use of Item Response Theory one-, two- and three-parameter logistic models, a series of models in which examinees and items are represented by a set of numerical values (the models' parameters). Item Response Theory itself was created with the goal of measuring latent traits as well as assessing and comparing individuals' proficiencies by allocating them in proficiency scales, inspiring as well as justifying its use in adaptive testing.

The logistic models of Item Response Theory are unidimensional, which means that a given assessment instrument only measures a single proficiency (or dimension of knowledge). The instrument, in turn, is composed of *items* in which examinees manifest their latent traits when answering them.

In unidimensional IRT models, an examinee's proficiency is represented as :math:`\theta`. Usually :math:`-\inf < \theta < \inf`, but since the scale of :math:`\theta` is up to the individuals creating the instrument, it is common for the values to be around the normal distribution :math:`N(0; 1)`, such that :math:`-4 < \theta < 4`.Additionally, :math:`\hat\theta` is the estimate of :math:`\theta`. Since a latent trait can't be measured directly, estimates need to be made, which tend to get closer to the theorically real :math:`\theta` as the test progresses in length.

Under the logistic models of IRT, an item is represented by the following parameters:

    * :math:`a` represents an item's *discrimination* parameter, that is, how well it discriminates individuals who answer the item correctly (or, in an alternative interpretation, individuals who agree with the idea of the item) and those who don't. An item with a high :math:`a` value tends to be answered correctly by all individuals whose :math:`\theta` is above the items difficulty level and wrongly by all the others; as this value gets lower, this threshold gets blurry and the item starts not to be as informative. It is common for :math:`a > 0`.
    * :math:`b` represents an item's *difficulty* parameter. This parameter, which is measured in the same scale as :math:`\theta`, shows at which point of the proficiency scale an item is more informative, that is, where it discriminates the individuals who agree and those who disagree with the item. Since :math:`b` and :math:`\theta` are measured in the same scale, :math:`b` follows the same distributions as :math:`\theta`. For a CAT, it is good for an item bank to have as many items as possible in all difficulty levels, so that the CAT may select the best item for each individual in all ability levels.
    * :math:`c` represents an item's *pseudo-guessing* parameter. This parameter denotes what is the probability of individuals with low proficiency values to still answer the item correctly. Since :math:`c` is a probability, :math:`0 < c \leq 1`, but the lower the value of this parameter, the better the item is considered.
    * :math:`d` represents an item's *upper asymptote*. This parameter denotes what is the probability of individuals with high proficiency values to still answer the item incorrectly. Since :math:`d` is a probability, :math:`0 < d \leq 1`, but the higher the value of this parameter, the better the item is considered.

For a set of items :math:`I`, when :math:`\forall i \in I, c_i = 0`, the three-parameter logistic model is reduced to the two-parameter logistic model. Additionally, if all values of :math:`a` are equal, the two-parameter logistic model is reduced to the one-parameter logistic model. Finally, when :math:`\forall i \in I, a_i = 1`, we have the Rasch model ([Rasch66]_). Thus, :py:mod:`catsim` is able of treating all of the logistic models presented above, since the underlying functions of all logistic models related to test simulations are the same, given the correct item paramaters.

Under IRT, the probability of an examinee with a given :math:`\hat\theta` value to answer item :math:`i` correctly, given the item parameters, is given by ([Ayala2009]_, [Magis13]_)

.. math:: P(X_i = 1| \theta) = c_i + \frac{d_i-c_i}{1+ e^{a_i(\theta-b_i)}}.

The information this item gives is calculated as ([Ayala2009]_, [Magis13]_)

.. math:: I_i(\theta) = \frac{a^2[(P(\theta)-c)]^2[d - P(\theta)]^2}{(d-c)^2(1-P(\theta))P(\theta)}.

Both of these functions are graphically represented in the following figure. It is possible to see that an item is most informative when its difficulty parameter is close the examinee's proficiency.

.. plot::

    from catsim.cat import generate_item_bank
    from catsim import plot
    item = generate_item_bank(1)[0]
    plot.item_curve(item[0], item[1], item[2], item[3], ptype='both')

The sum of the information of all items in a test is called *test information* [Ayala2009]_:

.. math:: I(\theta) = \sum_{j \in J} I_j(\theta).

The amount of error in the estimate of an examinee's proficiency after a test is called the *standard error of estimation* [Ayala2009]_ and it is given by

.. math:: SEE = \sqrt{\frac{1}{I(\theta)}}

Since the denominator in the calculation of the :math:`SEE` is :math:`I(\theta)`, it is clear to see that the more items an examinee answers, the smaller SEE gets.

:py:mod:`catsim` provides these functions in the :py:func:`catsim.irt` module.

The Item Matrix
---------------

In :py:mod:`catsim`, a collection of items is represented as a :py:class:`numpy.ndarray` whose rows and columns represent items and their parameters, respectively. Thus, it is referred to as the *item matrix*. The most important features of the items are situated in the first three columns of the matrix, which represent the parameters :math:`a`, :math:`b` and :math:`c`, respectively. Item matrices can be generated via the :py:func:`catsim.cat.generate_item_bank` function as follows:

>>> generate_item_bank(5, '1PL')
>>> generate_item_bank(5, '2PL')
>>> generate_item_bank(5, '3PL')
>>> generate_item_bank(5, '3PL', corr=0.5)

These examples depict the generation of an array of five items according to the different logistic models. In the last example, parameters :math:`a` and :math:`b` have a correlation of :math:`0.5`, an adjustment that may be useful in case simulations require it [Chang2001]_.

After the simulation, catsim adds a fourth column to the item matrix, representing the items exposure rate, commonly denoted as :math:`r`. Its value denotes how many times an item has been used and it is calculated as follows:

.. math:: r_i = \frac{q_i}{N}

Where :math:`q_i` represents the number of tests item :math:`i` has been used on and :math:`N` is the total number of tests applied.

Computerized adaptive tests
===========================

Unlike linear tests, in which items are sequentially presented to examinees and their proficiency estimated at the end of the test, in a computerized adaptive test (CAT), an examinees' proficiency is calculated after the response of each item. The updated knowledge of an examinee's proficiency at each step of the test allows for the selection of more informative items *during* the test itself, which in turn reduce the standard error of estimation of their proficiency at a faster rate. This behavior

The CAT Lifecycle
-----------------

In general, a computerized adaptive test has a very well-defined lifecycle:

.. graphviz::

    digraph cat_simple {
    	bgcolor="transparent";
    	rankdir=TB;
    	a[label=<START>, shape=box];
    	b[label=<Initial proficiency<br/>estimation>];
    	c[label=<Item selection and <br/>administration>];
    	d[label=<Capture answer>];
    	e[label=<Proficiency estimation>];
    	rank=same;
    	f[label=<Stopping criterion<br/>reached?>, shape=diamond];
    	g[label=<END>, shape=box];
    	a -> b -> c -> d -> e -> f;
    	f -> g[label=<YES>];
    	f -> c[label=<NO>];
    }

1. The examinee's initial proficiency is estimated;
2. An item is selected based on the current proficiency estimation;
3. The proficiency is reestimated based on the answers to all items up until now;
4. **If** a stopping criterion is met, stop the test. **Else** go back to step 2.

There is a considerable amount of literature covering these four phases proposed by many authors. In :py:mod:`catsim`, each phase is separated in its own module, which makes it easy to create simulations combining different methods for each phase. Each module will be explained separately, along with its API.

Initialization
^^^^^^^^^^^^^^

The initialization procedure is done only once during each examinee's test. In it, the initial value of an examinee's proficiency :math:`\hat\theta_0` is selected. This procedure may be done in a variety of ways: a standard value can be chosen to initialize all examinees (:py:class:`catsim.initialization.FixedInitializer`); it can be chosen randomly from a probability distribution (:py:class:`catsim.initialization.RandomInitializer`); the place in the item bank with items of more information can be chosen to initialize :math:`\hat\theta_0` etc.

In :py:mod:`catsim`, initialization procedures can be found in the :py:mod:`catsim.initialization` module.

Item Selection
^^^^^^^^^^^^^^

With a set value for :math:`\hat\theta`, an item is chosen from the item bank and presented to the examinee, which the examinee answers and its answer, along with the answers to all previous items, is used to estimate :math:`\hat\theta`.

Item selection methods are diverse. The most famous method is to choose the item that maximizes the *gain of information*, represented by :py:class:`catsim.selection.MaxInfoSelector`. This method, however, has been shown to have some drawbacks, like overusing few items from the item bank while ignoring items with inferior parameters. In order to correct that, other item selection methods were proposed.

In :py:mod:`catsim`, an examinee's response to a given item is simulated by sampling a binary value from the Bernoulli distribution, in which the value of :math:`p` is given by the IRT logistic model characteristic function (:py:func:`catsim.irt.icc`), given by:

.. math:: P(X_i = 1| \theta) = c_i + \frac{1-c_i}{1+ e^{a_i(\theta-b_i)}}

In :py:mod:`catsim`, item selection procedures can be found in the :py:mod:`catsim.selection` module.

Proficiency Estimation
^^^^^^^^^^^^^^^^^^^^^^

Proficiency estimation occurs whenever an examinee answers a new item. Given a dichotomous (binary) response vector and the parameters of the corresponding items that were answered, it is the job of an estimator to return a new value for the examinee's :math:`\hat\theta`. This value reflects the examinee's proficiency, given his or hers answers up until that point of the test.

In Python, an example of a list that may be used as a valid dichotomous response vector is as follows:

>>> response_vector = [1,1,1,0,1,1,0,1,0,0,1,0,0,0,1,0]

Estimation techniques are generally separated between maximum-likelihood estimation procedures (whose job is to return the :math:`\hat\theta` value that maximizes the *log-likelihood* function, presented in :py:func:`catsim.irt.logLik`); and Bayesian estimation procedures, which tend to use a priori information of the distributions of examinee's proficiencies to estimate new values for them.

In :py:mod:`catsim`, proficiency estimation procedures can be found in the :py:mod:`catsim.estimation` module.

Stopping Criterion
^^^^^^^^^^^^^^^^^^

Since items in a CAT are selected on-the-fly, a stopping criterion must be chosen such that, when achieved, no new items are presented to the examinee and the test is deemed finished. These stopping criteria might be achieved when the test reaches a fixed number of items or when the standard error of estimation (:py:func:`catsim.irt.see`) reaches a lower threshold etc. Both of these stopping criteria are implemented as :py:class:`catsim.stopping.MaxItemStopper` and :py:class:`catsim.stopping.MaxItemStopper`, respectively.

In :py:mod:`catsim`, test stopping criteria can be found in the :py:mod:`catsim.stopping` module.

Package architecture
********************

:py:mod:`catsim` was built using an object-oriented architecture, an uncommon feat for scientific packages in Python, but which introduces many benefits for its maintenance and expansion. As explained in previous sessions, each phase in the CAT lifecycle is represented by a different module in the package. Additionaly, each module involved in the CAT lifecycle has a base abstract class, which must be implemented if a new methodology is to be presented to that module's respective phase. This way, new users can implement their own methods for each phase of the CAT lifecycle, or even an entire new CAT lifecycle while still using :py:mod:`catsim` and its features to simulate tests, plot results etc. Modules and their corresponding abstract classes are presented on :numref:`modules_classes`.

.. table:: Modules and their corresponding abstract classes
    :name: modules_classes

    ===============================  ==============
    Module                           Abstract class
    ===============================  ==============
    :py:mod:`catsim.initialization`  :py:class:`catsim.initialization.Initializer`
    :py:mod:`catsim.selection`       :py:class:`catsim.selection.Selector`
    :py:mod:`catsim.estimation`      :py:class:`catsim.estimation.Estimator`
    :py:mod:`catsim.stopping`        :py:class:`catsim.stopping.Stopper`
    ===============================  ==============

Examples
********
