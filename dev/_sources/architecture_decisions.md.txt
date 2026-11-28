# Architecture Decision Records

Each Architecture Decision Record (ADR) states one decision that shapes catsim's architecture: the
forces behind it, the rule, what follows from it and the options rejected. They are written for
contributors and method authors who want to know why the code is shaped as it is before they
change or extend it; the guides and technique pages describe what users observe and link the ADR
for the reason. From 1.0 on, an accepted ADR is not rewritten: a changed decision gets a new ADR
that supersedes it, and the old one's Status names its successor. The records read in order, from
the scope of the package to the specifics of its strategies.

```{toctree}
:maxdepth: 1

adr/0001-compute-and-evaluate-not-serve
adr/0002-unidimensional-dichotomous-scope
adr/0003-four-parameter-item-matrix
adr/0004-item-names-and-indices
adr/0005-item-bank-csv-format
adr/0006-steps-from-explicit-inputs
adr/0007-strategy-contracts
adr/0008-what-may-be-cached
adr/0009-checking-explicit-inputs
adr/0010-how-a-session-ends
adr/0011-exclusions-and-exposure-control
adr/0012-exposure-rates
adr/0013-random-numbers-and-reproducibility
adr/0014-simulated-examinees-are-response-patterns
adr/0015-parallel-runs
adr/0016-simulation-results-and-metrics
adr/0017-random-tie-breaking
adr/0018-ability-scale-and-extreme-patterns
adr/0019-published-progressive-schedule
adr/0020-catr-as-the-reference
adr/0021-dependencies
adr/0022-plotting-as-an-optional-extra
adr/0023-public-import-paths-and-names
adr/0024-catsim-builds-on-irtlib
```
