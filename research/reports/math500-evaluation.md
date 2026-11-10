# Complete MATH-500: Host-side evaluation and base model baseline

This report preserves the experimental settings, results, and status at the time of writing. For the current project progress, see [Project Status](../../PROJECT_STATUS.md).

2026-11-07. The full base-model evaluation has been submitted. At the latest check it was running and waiting for GPU0 to start the first question. No conclusions about math performance or loop reduction are available yet.

## Data and Isolation

The complete test.jsonl, dataset card, and eval.yaml were obtained from the [official HuggingFaceH4/MATH-500 dataset](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) at revision `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`. The 500 official unique_id values, original questions, and answers are retained. Individual source-file SHA256 hashes are recorded in `benchmarks/math500-evaluation-v1/manifest.json`.

After whitespace normalization, exact question-text comparison with the 15,458 original exercises found 0 overlaps. This check does not cover semantic near-duplication, pre-training contamination, or the broader question spectrum. The evidence is `artifacts/toolkit/math500-evaluation-v1/practice-overlap.json`.

The test set is stored separately from the exercise tree, is not added to `PracticeSet`, and is not mounted in the programmer’s kernel. The new `weight_programming.evaluation.evaluate_math` only receives explicit test/evaluation labels on the host side; generation input is constructed from the problem alone, the answer is used by the host scorer, and the reference solution does not enter the generation prompt.

## Evaluation protocol

`src/run_math500_evaluation.py` fixes the evaluation at 500 questions × 4 samples, for 2,000 generations in total. It uses thinking, a 32,768-token limit, T=.6, top-p=.95, top-k=20, and presence=1.5. The base model is the verified Qwen3.5-0.8B revision used in the original work. It runs in a separate HF 5.3.0/BF16/SDPA environment without the optional DeltaNet fast path. Equivalence to vLLM or the authors’ undisclosed harness is not established.

Each question holds GPU0 exclusively while its four samples are generated. The model then moves back to the CPU and releases the lock. Queue and residency costs are recorded so a long evaluation does not monopolize the device. Each sample seed is derived from a fixed policy seed, the official question ID, and the sample index; the original work’s random stream is unknown.

For each sample, the output records text, token IDs, scores, truncation, finish reason, and the declared loop-tail indicators. The four-sample average accuracy and its standard error are aggregated by question; pass@4 is not substituted for accuracy. The completion mark is only written after all questions have been successfully completed. Partial directories reject silent restarts.

For a later frozen edit, `--replacements` imports native replacement tensors, copies the base model, and verifies the base, files, and edit permissions. Reference replay and independent discovery use different sources and output directories; evaluation results may not be returned to the independent programmer for further editing.

## Real operation and verification

- `job-mw5memb5-6af0`: The generation-path check exited 0. It used a real small Qwen3 and an explicit fixture grader to verify streaming storage, sample identity, and that answers do not enter prompts; it does not measure math ability.
- `job-mw5mf2op-d59e`: The full original Qwen3.5 baseline was running at the latest check and reported `waiting_gpu_0, position=1, items=500`; loading and backend verification of 473 tensors had completed.
- Output directory: remote `benchmarks/math500-full-v1/original`; `setup.json` has been saved locally, but there is no completion result yet.
- The independent programmer is still `job-mw5ltbbd-76df`; its source and context were frozen before the new test and evaluation tool was added, and it was not given access to this evaluation interface.

For public usage, see [host evaluation document](../../docs/final-evaluation.md). The full evaluation execution path is in place. Reproduction of the original effect, independent discovery, and direction comparison remain incomplete.
