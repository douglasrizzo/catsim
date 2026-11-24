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

adr/0001-unidimensional-dichotomous-scope
adr/0002-four-parameter-item-matrix
adr/0003-explicit-cat-engine
adr/0004-runtime-state-ownership
adr/0005-explicit-strategy-contracts
adr/0006-public-import-paths-and-names
adr/0007-how-a-session-ends
adr/0008-session-exclusions-and-exposure-control
adr/0009-immutable-step-results
adr/0010-random-numbers-and-reproducibility
adr/0011-random-tie-breaking
adr/0012-ability-scale-and-extreme-patterns
adr/0013-published-progressive-schedule
adr/0014-plotting-as-an-optional-extra
```
