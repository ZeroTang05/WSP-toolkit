# Paper-based action spaces and control strategies: First executable research track

## Official v2 revision (2026-11-06)

The original 16-point/coarse grid design below is retained as a description of the v1 executor smoke test. The formal matrix uses a separate `search-v2` directory, using `optimize_public_v2.py`, `search_v2.py`, and `search_metrics.py` without mixing old tracks.

- The fixed method is the complete global 21-point grid of Task Arithmetic Appendix B/D: 0 to 1, step size 0.05, the base and full-LoRA points are evaluated first.
- Other strategies use 21 queries in the same action space in four depth groups, each with 21 levels; the layer grouping continues to be marked as adapted for this project.
- ARC continuation removes manual leading spaces and uses original boundary processing with fixed harness; the 40 CPU cases involving tokenization across word boundaries all match the original function’s token IDs.
- The main protection proxy uses option-logit cross-entropy (CE) normalized by answer character length; the original CE, both types of option log-probability, and acc_norm are retained for diagnosis. This CE is a smoothing proxy and is still not equal to the official accuracy or autoregressive KL.
- `run_search_matrix.py` saves the immutable source code/contract snapshot, first runs the real-model smoke tests for fixed, BO, and LLM, and then executes the five-strategy matrix of four models and three seeds. A failed child process stops the run and its state is retained.
- `audit_search.py` independently recalculates candidate legality, protection bootstrap, optimal candidate selection, and base/delta binding; this audit cannot replace model numerical correctness and retention testing.

This revision was developed based on the paper grid and standard harness interface before the official search was run, and no training or selection rules were adjusted based on official test scores. The experiment has been submitted; its benefits have not yet been established.

Status: The executor has been implemented and passed CPU numerical/strategy testing, real model smoke is being queued; no formal search benefit conclusion yet. This document is not a retroactive pre-registration.

## Method basis and adaptation boundary

The training direction comes from [LoRA](https://arxiv.org/abs/2106.09685) and the direction operation comes from [Task Arithmetic](https://arxiv.org/abs/2212.04089). θ_0 remains unchanged and W_0x+c_g·(α/r)BAx is calculated as requested.

Divide the decoder layers into four normalized depth groups by floor(4l/L), each group c_g∈{0,0.5,1,1.5}. This is the task vector grouping adaptation explicitly stated in this study; four groups is a fixed search granularity, not the optimal granularity proven in the paper. Global coefficients and other granularity comparisons are needed later. There are only 256 candidates; this does not establish arbitrary high-dimensional weight programming capability.

The action is compiled by `task_plan.py` and bound to the base fingerprint, training delta hash, layer number and method_id. `DeltaBank` applies per-module coefficients in the requested range and removes hooks, neither overwriting base weights nor repeated BF16 merge-and-unmerge operations. Finally, the base+delta+four-dimensional plan is saved, without saving the complete model for each candidate.

## Data and goals

- Direction training uses an independent training view of GSM8K train.
- Search uses the same fixed first 64 questions from the independent development view for every model, seed, and strategy.
- The protection proxy uses the first 32 questions from a fixed hash ordering of official ARC-Challenge validation; ARC test is not available for selection.
- Main proxy: Teacher-forced risk on the reference rationale, answer, and EOS, normalized by the UTF-8 byte count of the answer in bits/byte. It is not generation accuracy; nor is the reference solution the only correct solution.
- Protection proxy: Conditional cross-entropy of the correct option among the full answer-text continuations, without pretending to be official acc_norm.
- For each candidate’s per-question protection CE difference from the base, use 2,000 paired bootstraps. The 95th percentile must not exceed 0.02 for development-set feasibility. This limited sample criterion is not a guarantee of overall capability retention.
- All strategies first evaluate c=(0,0,0,0) and c=(1,1,1,1) and include them in the query budget; base is always an optional safe fallback.

Official GSM8K test, ARC test, HellaSwag validation are only used for final confirmation. HellaSwag validation here acts as a pre-reserved final partition and cannot be used for searches just because the name is validation.

## Control strategies and evidence

There are 16 candidate queries fixed in the first stage of each strategy, and all strategies use the same direction, action space and observation data:

1. Fixed: base, full LoRA, two global scales, per-depth group preset probes, 16 points in total. Used for fixed method reference and local response estimation.
2. Random: Sampling uniformly in the remaining grid.
3. Greedy: Enumerate single-coordinate neighbors around the current optimal feasible point; its fixed order needs to be recognized as a simple baseline.
4. BO: A Gaussian process with a fixed RBF kernel, expected improvement multiplied by probability of satisfying the protection constraint. Constrained BO framework based on [Gardner et al., ICML 2014](https://proceedings.mlr.press/v32/gardner14.html); not a reproduction of the paper's complete hyperparameter inference/experimental recipe. More adequate BO configuration sensitivity should be added in the future.
5. LLM: Local Qwen3-4B-Instruct-2507 proposes the next coordinate based on the same historical risk and group delta/base norm; records full prompts, outputs, token counts, invalid actions, and random fallbacks. Removing norm observations allows for internal information ablation.

The same number of queries does not mean the same total cost. The first stage only supports "same number of queries" conclusions; training, model loading, controller tokens, peak memory, target forward, and wall clock costs must be additionally reported. The record includes the setup time of queued lock waiting, which cannot be regarded as pure model loading speed. Total cost matching and anytime Pareto analysis are still required before LLM can be formally claimed to be more efficient.

Source data, source code and contract snapshots are written to the run directory. A candidate exception stops execution and preserves logs. Failed candidates cannot be silently removed while reporting only successful runs. The selected plan is confirmed on the full held-out partition via an independent standard harness.

## What can cross-model analysis support?

The fixed 16-point design estimates central finite differences, uniaxial curvature, and finite step errors for each group; it cannot recover the full Hessian or verify the operator norm bounds in `transfer-analysis.md`. Compare the effect of the plan in the target model's own training direction without copying the parameter coordinates of other architectures.

Three training seeds are used for uncertainty in direction construction; running only one controller seed for each direction is not interpreted as a full controller stochastic analysis. Two model families cannot establish a universal law.
