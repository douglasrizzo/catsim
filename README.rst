.. image:: https://travis-ci.org/douglasrizzo/catsim.svg?branch=master

.. image:: https://coveralls.io/repos/github/douglasrizzo/catsim/badge.svg?branch=master :target: https://coveralls.io/github/douglasrizzo/catsim?branch=master 

Introduction
------------

**catsim** is a computerized adaptive testing simulator written in Python 3.4. It allow for the simulation of computerized adaptive tests, selecting different test initialization rules, item selection rules, proficiency reestimation methods and stopping criteria.

Computerized adaptive tests are educational evaluations, usually taken by examinees in a computer or some other digital means, in which the examinee's proficiency is evaluated after the response of each item. The new proficiency is then used to select a new item, closer to the examinee's real proficiency. This method of test application has several advantages compared to the traditional paper-and-pencil method, since high-proficiency examinees are not required to answer all the easy items in a test, answering only the items that actually give some information regarding his or hers true knowledge of the subject at matter. A similar, but inverse effect happens for those examinees of low proficiency level.

*catsim* allows users to simulate the application of a computerized adaptive test, given a sample of examinees, represented by their proficiency levels, and an item bank, represented by their parameters according to some Item Response Theory model.
