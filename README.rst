.. image:: https://travis-ci.org/douglasrizzo/catsim.svg?branch=master

Introduction
------------

Hello, my name is Douglas and this is my project *catsim*, which stands for *Computerized Adaptive Testing SIMulator*. I created it as part of my master's program, in which I proposed a `Cluster-based Item Selection Method (CISM) <https://www.researchgate.net/publication/283944553_Metodologia_de_seleo_de_itens_em_Testes_Adaptativos_Informatizados_baseada_em_Agrupamento_por_Similaridade>`_ for computerized adaptive tests.

Computerized adaptive tests are educational evaluations, usually taken by examinees in a computer or some other digital means, in which the examinee's proficiency is evaluated after the response of each item. The new proficiency is then used to select a new item, closer to the examinee's real proficiency. This method of test application has several advantages compared to the traditional paper-and-pencil method, since high-proficiency examinees are not required to answer all the esay items of a test, answering only the items that actually give some information regarding his or hers true knowledge of the subject at matter. A similar, but inverse effect happens for those examinees of low proficiency level.

Several techniques and methodologies have been proposed to select items in real time in such a way that the examinees' true proficiency is calculated with high precision, while making sure that the item bank is used in a homogeneous way. *catsim* is a methodology that tries to use traditional clustering algorithms, like K-means and agglomerative clustering, in order to cluster items so that items in the cluster can be used in place of each other during the application of a computerized adaptive test.