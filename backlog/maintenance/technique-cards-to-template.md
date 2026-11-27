---
title: Reshape the technique cards to the technique-page template
kind: maintenance
status: ready
priority: 3
effort: M
blocked_by: []
raised_by: [technique pages (#226)]
---

# Reshape the technique cards to the technique-page template

The eleven technique cards that came from `specs/` are listed below. They still have "Behavioral
Contracts" sections, where the technique-page template, `sphinx/api/techniques/_template.md`, and
the card template, [`_template.md`](../_template.md), have "Guarantees":

- [owen-estimator](../estimation/owen-estimator.md)
- [klp-selector](../selection/klp-selector.md)
- [mepv-selector](../selection/mepv-selector.md)
- [mpwi-selector](../selection/mpwi-selector.md)
- [sympson-hetter](../exposure/sympson-hetter.md)
- [glr-stopper](../stopping/glr-stopper.md)
- [kl-stopper](../stopping/kl-stopper.md)
- [min-info-bank-stopper](../stopping/min-info-bank-stopper.md)
- [predicted-sem-stopper](../stopping/predicted-sem-stopper.md)
- [sprt-stopper](../stopping/sprt-stopper.md)
- [stochastic-curtailment-stopper](../stopping/stochastic-curtailment-stopper.md)

For each card:

- Split "Behavioral Contracts" into "Guarantees" (what a user observes, promoted to the technique
  page) and the contributor-only "Tests" section (what only an implementation sees).
- Put the sections in the template's order.
- Keep the content. The citations were revalidated in November 2026, and the sketches run against
  the current API.

A card being implemented gets this reshaping in its own PR, so do it here only for cards nobody is
working on.
