---
title: Method-author walkthrough - implement a planned technique from its card
kind: research
status: ready
priority: 2
effort: M
blocked_by: []
raised_by: []
---

# Method-author walkthrough: implement a planned technique from its card

## Goal

An agent plays a method author: someone who knows the technique's literature and wants it in catsim.
Working from one technique card, it implements the technique in a worktree as a contributor would.
It reports where the extension points, the strategy contract (ADR-0007), the templates and the
"Extending catsim" guide got in the way.

## How

Follow [the discovery README](README.md), with these changes:

- **What the persona reads:** the public docs, plus the card, `CONTRIBUTING.md` and the ADRs.
  Reading `src/` is allowed only for the base classes and one sibling technique, as a contributor
  would.
- **Choosing the technique:** pick one that exercises an extension point no built-in technique
  uses yet:
  - the [SPRT stopper](../stopping/sprt-stopper.md): a pure `classify(context)`, combined in a
    `StopRule`;
  - or the [MPWI selector](../selection/mpwi-selector.md): a posterior grid cached per item bank
    (ADR-0008).

  Running both, with two agents, compares a stopper with a selector.
- **The output:** a PR-ready branch, or the point where the agent got stuck. Plus the findings
  report, which becomes cards.

## Done when

The findings are triaged into cards, and the walkthrough's branch is either opened as a PR or
deleted.
