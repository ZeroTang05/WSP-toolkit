# Independent Model Programmer: Practical Drivers and First Commit Experiments

This report preserves the experimental settings, results, and status at the time of writing. For the current project progress, see [Project Status](../../PROJECT_STATUS.md).

2026-11-07. An executable model driver has been added alongside the tool schemas. The first submitted experiment is still waiting for the GPU; parameter edits or math scores are not reported in this file.

## New content

`src/weight_programming/programmer.py` connects the independent model callback to the real `ProgrammingSession`:

- Create a new system/user context and add only this experiment’s tool responses.
- Accept three explicit output types: CPU code, a forward request, and a final report.
- Record actual input and output tokens, code execution count, the forward ledger, and complete responses.
- When the context reaches the upper limit, the same kernel, current edits, budget, and programmer’s own notes are retained, and the research session is not loaded.
- Save the current native-precision tensor hash and variant identity in the final record; do not treat completion of a program as evidence of a capability gain.
- Explicitly configure whether the forward budget must be exhausted; this original-style experiment requires 50 successful requests before final submission.

`src/run_weight_programmer.py` is the real local model callback. It freezes model file hashes, tool source code, environments, initial prompts, and exercise data manifests before queuing them to the GPU; it does not put author edit reports or current session conclusions into prompts.

The controller now holds observation exports and mounts them read-only in the kernel, preventing working-directory path redirection and modification of the original observations. For complete instructions, see [public interface document](../../docs/model-programmer.md).

## Actual verification scope

Remote job `job-mw5lq07b-bf8b` exited 0 and passed 5 checks. Using a real Linux isolation kernel, a small Qwen3, and the forward service, the checks covered coordinate-edit transfer, array reading, context continuation, rejection of premature submissions, service-failure logging, and read-only exports. Model callbacks are scripted inputs in these checks, not model performance experiments.

The original CPU numerical-tool environment was completed and its packages imported in the separate `.venv-wsp`: SciPy 1.15.3, scikit-learn 1.7.2, pandas 2.3.3, SymPy 1.13.1, NetworkX 3.4.2. The shared training environment has not been upgraded.

## First practical model experiment

- Job: `job-mw5ltbbd-76df`, running at the latest check, with output `waiting_gpu_0`.
- Target: Qwen3.5-0.8B at the revision used in the original work.
- Programmer: Qwen3-4B-Instruct-2507 loaded independently with greedy decoding and non-thinking mode; it is not the authors’ original programmer model.
- Exercises: the original 15,458 authorized exercises, checked by hash and count; reference edits and final evaluation inputs are excluded.
- Permissions: 300 CPU executions and 50 successful forward requests; target generation limit is 32,768 tokens, thinking and sampling follow the current freezing policy.
- Programmer limits: 500 replies, maximum 8,192 tokens per time, a 32,768-token context, continuation using its own notes is allowed.
- Execution: Standalone Transformers 5.3.0/Torch 2.5.1, BF16/SDPA, GPU 0 lock.
- Output: Remote `benchmarks/weight-programmer-v1/Qwen3.5-0.8B-qwen4b-seed401`; the frozen protocol has been saved locally and there is no completion record yet.

This parameter-programming experiment uses a new model context. Its HF backend, programmer model, and driver differ from the original work, so it cannot be called a complete reproduction. It is unknown whether the pre-training has seen the author's materials; this experiment establishes isolation of runtime inputs and does not establish that pretraining is free of contamination.

## Next step research and analysis

Upon completion, review the actual program trace, failure types, and permissions; freeze final edits before conducting an independent complete mathematical evaluation. If a gain occurs, report per-question improvements and degradations, looping and truncation, length, and retained abilities. Faster completion does not replace accuracy.

If the complete results meet the predefined matching criteria, compare the authors’ reference edits and this experiment’s edits by affected sites, coordinate-wise signs, subspaces, functional changes on common inputs, dose, and reverse/randomized controls. No edits are complete yet, so their direction cannot be characterized in advance as the same or different.

The 4B office task `job-mw5jt7i0-e7ef` is still executing on the CPU; this round of verification has 12 actual actions and 25 model calls, and neither the single-agent nor collaborative comparison has finished. Keep the job running.
