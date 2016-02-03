.. 
  image:: https://travis-ci.org/douglasrizzo/catsim.svg?branch=master

Introduction
------------

Hello, my name is Douglas and this is my project *catsim*, which stands for *Computerized Adaptive Testing SIMulator*. I created it as part of my master's program, in which I proposed a `Cluster-based Item Selection Method (CISM) <https://www.researchgate.net/publication/283944553_Metodologia_de_seleo_de_itens_em_Testes_Adaptativos_Informatizados_baseada_em_Agrupamento_por_Similaridade>`_ for computerized adaptive tests.

Computerized adaptive tests are educational evaluations, usually taken by examinees in a computer or some other digital means, in which the examinee's proficiency is evaluated after the response of each item. The new proficiency is then used to select a new item, closer to the examinee's real proficiency. This method of test application has several advantages compared to the traditional paper-and-pencil method, since high-proficiency examinees are not required to answer all the easy items in a test, answering only the items that actually give some information regarding his or hers true knowledge of the subject at matter. A similar, but inverse effect happens for those examinees of low proficiency level.

*catsim* allows users to simulate the application of a computerized adaptive test, given a sample of examinees, represented by their proficiency levels, and an item bank, represented by their parameters according to some Item Response Theory model.