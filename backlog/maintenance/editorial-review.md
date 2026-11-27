---
title: Editorial review of every document against the charter, and drift checks
kind: maintenance
status: ready
priority: 3
effort: M
blocked_by: []
raised_by: [documentation charter]
---

# Editorial review of every document against the charter, and drift checks

## The review

Read every document against `sphinx/AGENTS.md`, each by the checklist of its kind and the general
review questions:

- `README.md`;
- the Sphinx pages;
- the docstrings;
- the notebooks;
- the ADRs;
- `CONTRIBUTING.md`.

Fix what fails. Known items to settle along the way:

- **Length budgets:** the WLE page (1,120 words) and the stratified page (1,194) are over the
  1,000-word budget. Trim them, or record why their methods need the length.
- **ADR-0023:** it calls `catsim.metrics` a module of functions and aliases, but it also defines the
  `Metric` class. Say so, or state the rule so that `Metric` fits it.
- **$N$ means two things.** "Terms and notation" uses $N$ for the number of examinees, while the
  progressive and proportional pages and ADR-0019 use $N$ for the planned test length, Barrada et
  al.'s notation. Write the planned test length with the terms page's symbol, adding one if the page
  lacks it, and say once that Barrada et al. write it $N$.
- **Bibliography notes.** Rasch (1966), De Ayala (2009) and Meneghetti (2015) have no `note`. Find
  whether each has an open copy; Meneghetti (2015) is the maintainer's thesis, so ask for its URL.
  Magis (2016)'s note says "Not openly available" but names no open source that describes its formula,
  and Magis and Barrada (2017) do not; find one, or say there is none. Check against the sources, and
  run linkcheck.

## Drift checks

Some of the charter's rules can be checked mechanically. Decide which are worth a hook, keeping only
those whose false positives stay rare:

- history phrases ("now", "no longer", "replaces", "previously") outside ADR Context and the
  compatibility notice;
- discouraged synonyms from "Terms and notation" ("latent trait", "response pattern", "stopping rule"
  outside catR's names);
- ADR structure (Status, Context, Decision, Consequences, Alternatives);
- docstring prose budgets.

## When

Any time. A review is most useful after a batch of cards that changed many documents.
