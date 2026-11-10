# Full public baselines: first results

This report preserves the experimental settings, results, and status at the time of writing. For the current project progress, see [Project Status](../../PROJECT_STATUS.md).

Date: 2026-11-06. This report includes only complete base-model baselines independently recomputed question by question. No conclusion about optimization gains is available yet.

| Model| GSM8K flexible EM, n=1319 | ARC-C acc_norm, n=1172 | HellaSwag acc_norm, n=10042 |
|---|---:|---:|---:|
| Qwen3-0.6B | 36.47% (481 questions)| 30.97% (363 questions)| 41.99% (4217 questions)|
| SmolLM2-360M-Instruct | 6.07% (80 questions)| 35.41% (415 questions)| 51.91% (5213 questions)|

Protocol: lm-evaluation-harness 0.4.9.2, zero-shot, chat template, greedy decoding, BF16, and batch 8. GSM8K uses non-thinking mode and at most 512 new tokens. ARC and HellaSwag use a fixed version of the official task scoring, retaining acc and acc_norm. GSM strict-format EM is also retained and is 0 for both models. This protocol differs from the few-shot, thinking, or generation-length settings that may be used on a model card.

The first supported observation is a **reversal in task ordering**: under this protocol, Qwen0.6 scores higher in math, while Smol360 scores higher on both retention tasks. A single task therefore cannot establish an overall ranking, and an absolute post-optimization score cannot replace the improvement relative to each model’s baseline.

The models differ in parameter count, training data, tokenizer, and post-training, so the differences cannot be attributed solely to model family or architecture. Evaluations of two 1.7B models, optimization across three seeds, relative-capacity controls, and method-interaction analysis remain incomplete.

Results are stored locally at `artifacts/public/<model>/baseline/`: original per-question predictions, harness results, protocols, source code snapshots, model file fingerprints, and independent audits have all been saved. Each audit re-aggregates the original records and matches the harness metrics; the Wilson interval only reflects the uncertainty of question sampling and does not account for variation across optimization seeds or pretraining contamination.
