# Study Guide

This guide is for readers who want to understand the methods in {mod}`catsim`
and check them against the literature. Every implemented technique page ends
with a **Reading the Source** section that says where each formula and design
choice comes from, whether the source was checked against its full text, and
which tests verify the implementation.

## How the pages were checked

Each source on a technique page is in one of three states:

- **Checked against the full text.** The page, equation, table or figure cited
  was read in the source itself.
- **Checked against a secondary source.** The original is not openly available,
  so the method was checked against an open description of it, which the page
  names.
- **Not checked.** No text was read; the page says so.

Locations are given only when they were read. Where {mod}`catsim` departs from
a source, the page says how.

Pages marked **Planned** describe techniques that are not implemented yet.
Their formulas and references have not been checked; check them against the
sources before implementing.

## Reading order

1. **IRT foundations.** The four-parameter logistic model, item and test
   information, the likelihood and the standard error, in {mod}`catsim.irt`.
   Then maximum-likelihood estimation and Dodd's heuristic, in
   {doc}`/api/techniques/estimation/numerical-search-estimator`.
2. **Selection by information.**
   {doc}`/api/techniques/selection/max-info-selector` and
   {doc}`/api/techniques/selection/urry-selector`.
3. **Exposure control by randomization.**
   {doc}`/api/techniques/selection/randomesque-selector`,
   {doc}`/api/techniques/selection/the-54321-selector`,
   {doc}`/api/techniques/selection/progressive-selector` and
   {doc}`/api/techniques/selection/proportional-selector`.
4. **Exposure control by stratification.** Start with the family page,
   {doc}`/api/techniques/selection/stratified-selector`, then
   {doc}`/api/techniques/selection/a-strat-selector`,
   {doc}`/api/techniques/selection/a-strat-b-block-selector`,
   {doc}`/api/techniques/selection/max-info-strat-selector` and
   {doc}`/api/techniques/selection/max-info-b-block-selector`.
5. **Global and weighted information.**
   {doc}`/api/techniques/selection/interval-info-selector`,
   {doc}`/api/techniques/selection/kl-selector`,
   {doc}`/api/techniques/selection/mlwi-selector` and
   {doc}`/api/techniques/selection/mei-selector`.
6. **Bayesian and bias-corrected estimation.**
   {doc}`/api/techniques/estimation/bayesian-infrastructure`,
   {doc}`/api/techniques/estimation/eap-estimator`,
   {doc}`/api/techniques/estimation/map-estimator` and
   {doc}`/api/techniques/estimation/warm-wle`.
7. **Initialization and stopping.**
   {doc}`/api/techniques/initialization/random-initializer`,
   {doc}`/api/techniques/stopping/test-length-stopper`,
   {doc}`/api/techniques/stopping/min-error-stopper` and
   {doc}`/api/techniques/stopping/confidence-interval-stopper`.

The {doc}`/api/techniques/selection/cluster-selector` is catsim's own method
(Meneghetti, 2015) and can be read at any point after step 2.

## Sources

Sources with an open copy were read in full. For the others, the last column
names what was checked instead.

| Source | Used by | Open copy | Checked |
| --- | --- | --- | --- |
| Bock & Mislevy (1982) | Bayesian infrastructure, EAP | [UMN Digital Conservancy](https://hdl.handle.net/11299/101546) | Full text |
| Chang & Ying (1996) | KL | [UMN Digital Conservancy](https://hdl.handle.net/11299/119092) | Full text |
| Lord (1977) | Maximum information | [UMN Digital Conservancy](https://hdl.handle.net/11299/98423) | Full text |
| Samejima (1977) | Minimum error stopper | [UMN Digital Conservancy](https://hdl.handle.net/11299/98492) | Full text |
| Dodd (1990) | Dodd's heuristic | [UMN Digital Conservancy](https://hdl.handle.net/11299/113936) | Full text |
| Veerkamp & Berger (1994 report; 1997 article) | Interval information, MLWI | [Report 94-6](https://ris.utwente.nl/ws/files/5155670/RR_94-6.pdf) | Full text of the report |
| Barrada, Mazuela & Olea (2006) | Stratified family, random initializer | [Psicothema](https://www.psicothema.com/pdf/3191.pdf) | Full text |
| Barrada, Olea, Ponsoda & Abad (2008) | Progressive, proportional, random initializer | [First author's page](https://personal.unizar.es/barrada/papers/08_BJMSP_Barrada.pdf) | Full text |
| Barrada, Olea, Ponsoda & Abad (2010) | {py:func}`~catsim.cat.overlap_rate`, item bank generator | [First author's page](https://personal.unizar.es/barrada/papers/10_APM_Barrada.pdf) | Full text |
| Magis & Raîche (2012) | WLE, MAP, MEI, MLWI, Urry, randomesque, confidence interval stopper | [Journal of Statistical Software](https://doi.org/10.18637/jss.v048.i08) | Full text |
| Meneghetti & Aquino Junior (2017) | Cluster | [arXiv](https://arxiv.org/abs/1707.03012) | Full text |
| Warm (1989) | WLE | — | Equation via Magis & Raîche (2012) and catR |
| van der Linden (1998) | MEI | — | Criterion via Magis & Raîche (2012) |
| Revuelta & Ponsoda (1998) | Progressive | — | Via Barrada et al. (2008) |
| Segall (2004) | Proportional | — | Via Barrada et al. (2008) |
| Chang & Ying (1999); Chang, Qian & Ying (2001) | a-stratified, b-blocking | — | Via Barrada et al. (2006) |
| Samejima (1969); Bock & Aitkin (1981); Birnbaum (1969) | MAP | — | Attribution via Bock & Mislevy (1982) and Magis & Raîche (2012) |
| Kingsbury & Zara (1989) | Randomesque | — | Via Magis & Raîche (2012) |
| Urry (1977) | Urry | — | Via Magis & Raîche (2012) |
| McBride & Martin (1983) | 5-4-3-2-1 | — | Not checked |
| Lord (1983) | WLE | — | Not checked |
| Meneghetti (2015) | Cluster | — | Not checked |
| Birnbaum (1968); Lord (1980); Brent (2002); Thissen (2000); Wainer (2000) | Several pages | — | Not checked (books) |

Full citations are on the technique pages and in the {doc}`/references`.

## Reference implementation

Several methods were also compared with
[catR](https://github.com/cran/catR), the R package by Magis and colleagues,
whose source is open. The technique pages say where catsim and catR differ,
for example in the progressive and proportional schedules and in the
information MEI evaluates.
