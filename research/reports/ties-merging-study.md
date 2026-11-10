# TIES Tool Extension: Effective Updates, Conflicts, and Norm Control

This report preserves the experimental setup, results, and status at the time of writing. See [Project status](../../PROJECT_STATUS.md) for current progress.

2026-11-07. Callable tools and parameter merges for four real models are complete, as is the five-condition behavioral proxy experiment. This report does not claim improved generation accuracy.

## Method Sources and Tool Use

[TIES-Merging](https://arxiv.org/abs/2306.01708) studies redundant small updates and sign interference in model merging. It addresses how to combine existing capability updates, complementing observation, pruning, compensation, and full-task validation.

The public interface `weight_programming.ties_merge` accepts effective deltas in a shared base and coordinate space. It returns dense CPU FP32 updates and evidence about thresholds, conflicts, and retention. For LoRA, first compute BA with its training scale; averaging A and B separately is not equivalent. The [method card](../../docs/ties-merging.md) defines the interface, adaptations, and boundaries.

The reference is pinned to `src/utils/merge_utils.py` in `prateeky2806/ties-merging@44e7891fc84f3de7e4caa52664cd864ca3715e91`. The public package implements the method independently without copying upstream code. The internal archive and its hashes serve differential validation only.

Twelve differential comparisons cover three densities and four input types: random values, many tied values, all zeros, and sign cancellation. Results agree exactly, element by element. Three additional checks cover mass-based sign election, global coordinate scope, unchanged inputs, and incompatible-source rejection.

Two reference details matter. An inclusive kth threshold can retain tied values and a boundary coordinate, so it does not strictly retain the requested count. When the mass-based sign is zero, the reference fills it from the global sign; if that is also zero, selection follows the negative branch. The tool preserves these defaults and offers an explicit abstention variant. Density endpoints 0/1 are separately declared extensions, not equivalence claims for upstream endpoints that raise errors.

## Four-Model Results

Each model uses its own GSM training updates from seeds 11/22/33, with q/v rank8 factors. Merge density is fixed at 0.2, with trimming over the joint coordinates of all target modules. No parameter is selected on test results. These sources are different training seeds for one task, so they do not establish the effects of merging distinct professional roles.

| Model | Sign conflicts among active coordinates after trimming | Final delta nonzero fraction | TIES/ordinary-mean norm | Cosine with ordinary-mean direction | Dense delta MiB |
|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 11.07% | 44.91% | 1.974 | 0.800 | 336.0 |
| Qwen3-1.7B | 6.84% | 43.14% | 1.702 | 0.838 | 672.0 |
| SmolLM2-360M | 2.61% | 38.70% | 1.359 | 0.884 | 150.0 |
| SmolLM2-1.7B | 3.80% | 42.10% | 1.564 | 0.859 | 768.0 |

Qwen0.6 has 88,080,384 coordinates across 56 modules. The other models have 176,160,768, 39,321,600, and 201,326,592 coordinates respectively. Actual compilation takes 13.95/35.55/7.49/38.73 seconds, including parameter analysis and export; these are not inference latencies. Full results and sources are in `artifacts/toolkit/ties-merge-v1/`, while dense weight artifacts remain in the remote benchmark directory.

Four conclusions follow directly:

1. **Merging changes both direction and magnitude.** For 0.6B, cosine is 0.800 and the norm nearly doubles. A behavioral gain would therefore require controls before attributing it solely to sign-conflict removal.
2. **Conflict depends on the model and update sources.** Smol360 has the lowest conflict and Qwen0.6 the highest, consistent with earlier observations of cross-seed directional alignment. Four checkpoints do not establish scaling laws or causality.
3. **Input density differs from merged output density.** Each source retains approximately 20%, but the union of three supports is larger, yielding approximately 39%–45% nonzero output. Density=0.2 does not mean only 20% of final model parameters remain.
4. **Merging does not guarantee compression.** Source LoRA updates have low-rank representations. Coordinate trimming and nonlinear election can increase rank, and the current dense export is larger. Subsequent low-rank compression must measure approximation error and task damage separately.

## Five-Condition Behavioral Analysis

The fixed conditions are the base model, ordinary mean, TIES, TIES scaled down to the ordinary-mean norm, and the ordinary mean scaled up to the TIES norm. They separate the effects of direction and magnitude.

`src/evaluate_ties_proxy.py` uses the existing 64 GSM training-derived development examples and 32 official ARC validation examples, with FP32 parameters and CPU execution. It reports reference-answer bits/byte, normalized option cross-entropy, and per-question records. These are not full GSM generation accuracy and are not directly comparable to earlier BF16 scores. Every condition shares templates, inputs, and computation; a fixed hash determines order. No latency conclusion is drawn.

The first job, `job-mw5m3v14-ad1b`, exited 1 before model loading because it tried to hash the `__pycache__` directory as a file, yielding no model-effect result. The corrected runner hashes only `.py` files and submitted `job-mw5m56e4-1888` in a new directory. It exited 0 with all five conditions complete. The failed directory is retained; local per-question data and paired analysis are in `artifacts/toolkit/ties-merge-v1/Qwen3-0.6B-proxy-v2/`.

| Condition | Update norm/ordinary-mean norm | GSM reference-answer bits/byte ↓ | ARC normalized option CE ↓ | ARC correct/32 |
|---|---:|---:|---:|---:|
| Base model | 0 | 0.83098 | 1.55156 | 6 |
| Ordinary mean | 1 | 0.36551 | 1.55051 | 11 |
| TIES | 1.974 | 0.40352 | 1.69427 | 7 |
| TIES scaled to ordinary-mean norm | 1 | 0.33276 | 1.45300 | 8 |
| Ordinary mean scaled to TIES norm | 1.974 | 0.45358 | 1.72924 | 8 |

`analyze_ties_proxy.py` rechecks every per-question aggregation and ID mapping, then runs 10,000 paired question bootstraps. Norm-matched TIES improves over the ordinary mean by −0.03275 bits/byte, with a nominal 95% interval of [−0.03879, −0.02745]: 63 questions improve and 1 worsens. At the same larger norm, TIES is 0.05005 bits/byte better than the scaled-up mean, with 63 improvements. Unscaled TIES is 0.03801 worse than the ordinary mean, degrading all 64 questions. These results motivate separating direction from magnitude; they do not support declaring unscaled TIES superior without controls.

Guard-set CE and accuracy rank conditions differently. Norm-matched TIES lowers normalized CE but falls from the ordinary mean's 11/32 correct to 8/32. A CE advantage does not establish guard-task capability preservation. This 32-question development diagnosis cannot replace full ARC confirmation.

The interval measures uncertainty over the current development questions, without multiple-comparison correction or uncertainty over training-seed combinations. Three source seeds within one merge are not three independent merge experiments.

At the time of writing, `job-mw5mjcu2-016b` had been submitted for a fresh full GSM1319 five-condition generation panel. It uses the same BF16 dense-weight execution path, batch32/512 generation, templates, and fixed conditions. Numerically different functional-LoRA and CPU FP32 results are not reused. Conditions queue separately on GPU0, and original test scores do not guide further program changes.

Full generation effects require frozen programs and independent confirmation tasks. If proxy and generation rankings reverse, the reversal is reported as a method boundary; merge rules are not retuned on the test set.

## Open Source and Other Experiments

The source was released in [commit0ce8d0b](https://github.com/ZeroTang05/idea4-weightlab/commit/0ce8d0b77f762cf7c8baec12ba1382b8412c4e3f), including tools, method cards, contracts, real study runners, and checks. An anonymously downloaded source copy matches the local hashes. At the time of this report, it was the development source, while the earlier prerelease wheel remained unchanged.

The original Qwen3.5 independent programmer `job-mw5ltbbd-76df` was still waiting for GPU0, and the 4B office task `job-mw5jt7i0-e7ef` was still running. The new TIES tool does not enter the frozen independent WSP track or supply a candidate answer for its training-free discovery.
