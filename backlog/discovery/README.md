# Discovery: walkthroughs that find what catsim lacks

A walkthrough has an agent play one of catsim's users, do that user's real work with catsim, and
report every place where catsim got in the way. Its findings are checked against the code and become
cards. The cards in this folder are the walkthroughs not yet run.

## How to run a researcher walkthrough

Start one agent per published study, each with a prompt built from the template below. The rules
are what make the findings useful:

- **The persona learns catsim only as a user would:** `README.md`, the pages in `sphinx/`, the
  notebooks, and docstrings read through Python (`help`, `inspect.getdoc`, `dir`). It does not read
  `src/`, `tests/` or `backlog/`, so a missing doc shows up as friction.
- **It writes only to a scratch folder** and leaves the repository untouched.
- **It verifies before claiming absence:** before it reports a missing feature, it searches the docs
  and the public modules. It also says whether catsim lacks the feature or the agent could not find
  it.
- **It times its runs.** It scales a design down only when it must, and says what it cut and why.
- **It does not fabricate.** Anything that did not run is reported with its error.

Prompt template:

```text
You are role-playing a psychometrics researcher who has just installed catsim and wants to replicate
<study, full citation>. Your real purpose is to find catsim's shortcomings: missing features,
awkward API, outputs you must compute by hand, documentation gaps, bugs and slowness.

Reproduce its design as faithfully as you can: <design elements, outcome measures>. Where you don't
remember a detail, choose a sensible value and state the assumption. Produce the paper's main tables
and figures, or the closest you can, and say whether your results agree with its conclusions.

<the rules above, with the scratch folder's path>

Report, in under 1,500 words:
1. a summary of what you replicated and whether it agrees with the paper;
2. findings, most severe first, each with: category (missing feature, API friction, documentation
   gap, bug, performance, reporting gap), severity, what you tried, your workaround with a snippet,
   and what catsim could offer;
3. what worked well;
4. the paths of your scripts and outputs.
```

The coordinator then checks each finding against the code, drops what falls outside catsim's scope
(ADR-0001), folds findings into existing cards where they fit, and writes cards for the rest. Each
card's `raised_by` names the walkthrough.

## The first run (2026-11-26)

Three researcher walkthroughs, on dev at a35f67fa:

| Study | Design | Scale and time | Agreement with the paper |
|---|---|---|---|
| Babcock & Weiss (2012), termination criteria | 3PL; 4 banks: {100, 500} × {flat, peaked}; MLE and EAP; 16 rules (fixed, SE, minimum information, Δθ, combinations) | 128 runs, 147,200 sessions, 315 s on 8 workers | Yes for SE and Δθ rules. SE + minimum information agrees with EAP but not with MLE ([constant-patterns-and-hybrid](../estimation/constant-patterns-and-hybrid.md)) |
| Chang & Ying (1999), a-stratified selection | 360 items, L = 40, K = 4; MI, a-STR, a-STR with b-blocking, MI with Sympson–Hetter (own code), `r_max = .2` | 5 × 5,000 examinees plus a 13-point grid; 8.6 min | Exposure results reproduced clearly; the precision cost was larger than "comparable" |
| Wang & Vispoel (1998), ability estimators | 3PL; 2 pools; 30-item tests read at 5, 10, 20 and 30 items; ML (3 variants), WLE, MAP and EAP with two priors; 13 θ × 1,000 | 208,000 tests, 261 s on 12 workers | Yes, on all main conclusions |

None of the designs had to be scaled down. The studies' scripts are kept locally by the maintainer,
not in the repository; the cards hold their measurements.

What the agents asked to keep:

- `ability_levels()` with Monte Carlo errors;
- common random numbers from one seed, which made exact post-hoc checks and sharp paired
  differences possible;
- `StopRule(..., min_items=)`;
- `steps()` and `examinees()` as dicts of columns;
- the stratified selectors and `strata()`;
- `BaseExposureControl` with `proposal_rates()`, which fit Sympson–Hetter exactly;
- picklable priors and custom strategies on workers;
- the estimation pages' SE table and their notes on extreme patterns;
- clear error messages;
- plots that share one axes through `label`.
