---
title: Fill in the missing notes in the bibliography
kind: maintenance
status: needs-research
priority: 3
effort: XS
blocked_by: []
raised_by: [documentation sync (#226)]
---

# Fill in the missing notes in the bibliography

Every entry in `sphinx/references.bib` carries a note: "Open access", an open copy, or "Not openly
available; X describes the method" (the citation convention in [`_template.md`](../_template.md)).
Some entries don't meet that rule:

- **No note:** Rasch (1966), De Ayala (2009) and Meneghetti (2015). Find out whether each has an open
  copy. Meneghetti (2015) is the maintainer's own thesis; ask the maintainer for its public URL.
- **No open source that describes the formula:** Magis (2016)'s note says "Not openly available", but
  Magis and Barrada (2017) do not describe its formula. Find an open source that does, or say in the
  note that there is none.

Check each against the source, never from memory, and run the docs build with linkcheck.
