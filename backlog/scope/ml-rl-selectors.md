---
title: Machine-learning / reinforcement-learning CAT selectors
kind: technique
status: parked
priority: 4
effort: XL
blocked_by: []
raised_by: []
---

# ML / RL CAT family (BOBCAT, NCAT, DQN-CAT, BECAT, GMOCAT, EMC, …)

> **Needs detailing before it leaves `parked`.** An agent wrote this card in April 2026 from an
> unverified literature survey, against the API from before the rescoping. Check its sources and
> redo its design against ADR-0002 and ADR-0007 before any work.

## Why this is a discussion card

Post-2018 CAT research has produced a wave of **learned selectors** — item-selection policies trained on historical response data (supervised) or on simulated CAT trajectories (reinforcement learning). They uniformly beat classical selectors on benchmark datasets, sometimes by large margins. However:

- They require **training data** (historical examinee responses) or a **simulator** used as the RL environment.
- They require a **neural network runtime** (PyTorch is the de facto standard; tensorflow and JAX also appear).
- They require a **state interface** larger than what `BaseSelector` currently exposes — typically the full posterior or the raw response history as a tensor.
- They require a **checkpoint/model-distribution mechanism** so users can load pretrained policies.

This is a research-grade scope expansion that catsim has not committed to. This card documents the landscape and the architectural implications; it is not ready to implement.

## References

- Chen, Y., et al. (2018). A reinforcement learning approach to adaptive testing. *EDM*.
- Nurakhmetov, D. (2019). Reinforcement learning applied to adaptive classification testing.
- Ghosh, A., & Lan, A. (2021). BOBCAT: Bilevel optimization-based meta-learning for CAT. *NeurIPS*. https://arxiv.org/abs/2108.07386
- Zhuang, Y., et al. (2022). Fully adaptive framework: neural CAT (NCAT). *AAAI*.
- Bi, H., Ma, H., Huang, Z., et al. (2020). Expected model change (EMC) for CAT. *ICDM*.
- Li, X., et al. (2023). An adaptive testing item selection strategy via a deep reinforcement learning approach. *Behavior Research Methods*.
- Zhuang, Y., et al. (2023). BECAT: Behavioral testing via subset selection for CAT.
- Wang, X., et al. (2023). GMOCAT: Gradient-based multi-objective CAT. *KDD*.
- Veldkamp, B. P., & Verschoor, A. J. (2019). Robust CAT with uncertainty in item parameter estimates.
- **Survey**: Zhuang, Y., et al. (2024). *Survey of Computerized Adaptive Testing: A Machine Learning Perspective*. arXiv:2404.00712. https://arxiv.org/html/2404.00712v2

## Taxonomy

| Family | Representative | Training signal | Runtime cost |
|---|---|---|---|
| **Supervised meta-learning** | BOBCAT (Ghosh & Lan 2021) | Historical test logs; bilevel objective (inner = ability estimate fit, outer = selection policy) | Forward pass of a small MLP per step |
| **Deep RL** | Chen 2018, NCAT (Zhuang 2022), DQN-CAT (Li 2023) | Simulated trajectories; reward = accuracy / test length | MLP or small transformer per step |
| **Subset selection** | BECAT (Zhuang 2023) | Historical logs; learn which items are globally informative | Precomputed score table per bank |
| **Multi-objective** | GMOCAT (Wang 2023) | Simulated trajectories with multi-dim reward (accuracy, exposure, content) | MLP + gradient-based scalarization |
| **Expected model change** | EMC (Bi 2020) | No training — uses gradient of model parameters | Gradient computation per candidate |
| **Uncertainty-aware** | Veldkamp & Verschoor 2019 | No training — propagates item-parameter uncertainty | Numerical integration per candidate |

The last two (**EMC** and **uncertainty-aware**) are *not* learned — they are classical selectors with more sophisticated numerical machinery. They could be implemented without a torch dependency and are better candidates for a near-term card than the learned policies.

## What would need to change in catsim

For **any** learned policy:

1. **Dependency**: torch (and possibly `transformers` for some architectures) as an optional extra. Core install stays lean.
2. **`BaseSelector` extension**: the learned selector needs to see not just `est_theta` but the full `response_vector` (already partially addressed by the API change in [sprt-stopper](../stopping/sprt-stopper.md)) and possibly a history tensor of `(item_id, response)` pairs.
3. **Checkpoint loader**: policies are trained once, distributed as weights. Need a `load_policy(path)` / `from_pretrained(url)` pattern.
4. **Training entry point**: separate from runtime. Either ship a `catsim.training` subpackage (big scope) or document that users must train their own policies with third-party tools and load the result.
5. **Benchmarks**: classical and learned selectors should be comparable on a shared task. This requires a small curated response-data fixture and a consistent evaluation loop.

For **EMC** and **uncertainty-aware** (the non-learned cousins):

1. No new dependency.
2. `BaseSelector.select` extension to receive item-parameter covariance (for uncertainty-aware).
3. A standard closed-form or numerical-integration kernel, of the same shape as the existing MEI/MEPV cards.

## Open questions

1. **Scope commitment**: is catsim willing to become a platform for learned selectors, or is it strictly a classical-CAT library? The maintainer's answer is decisive.
2. **Dependency policy**: if learned selectors are in scope, is `torch` an acceptable optional extra? It adds ~500 MB of wheels and is the single biggest dep catsim would ever carry.
3. **Training vs. inference**: does catsim ship only *inference* for pretrained policies, or also *training* code? Training is significantly more work and drags in data-loading, logging, and hyperparameter tooling.
4. **Dataset distribution**: learned policies need benchmark datasets (e.g. ASSISTments, EdNet). Do we redistribute them, or document how users should download them?
5. **Reproducibility**: every paper in this family uses a different train/test split, different scoring, and different baselines. Adopting their results wholesale risks overclaiming catsim's capabilities.
6. **EMC / uncertainty-aware as a separate card**: these could become cards of their own independently of the RL family. They are classical selectors that only happen to share a section in the Zhuang survey. Worth splitting off once the maintainer confirms they are in scope.

## Notes

- **External dependencies**: `torch` (optional) for the learned family; `scipy` only for EMC and uncertainty-aware. The latter two are the low-risk, high-value split-outs.
- **No implementation work until the scope decision is made.** This is the most architecturally invasive family in the literature review and should not be started without an ADR.
- **First pragmatic step if greenlit**:
  1. Split EMC and uncertainty-aware into cards of their own (no torch needed).
  2. For the learned family, ship a single reference implementation (BOBCAT) with pretrained weights on a small public dataset as a proof of concept, gated behind `catsim[learned]`.
  3. Defer the full family (NCAT, DQN-CAT, GMOCAT, BECAT) to community contributions.
- **ADR required before any learned-selector code lands.** The torch dependency and the training-code scope are exactly the durable decisions that belong in `sphinx/adr/`.
- **Pairs with**: no other card. This family stands alone.
