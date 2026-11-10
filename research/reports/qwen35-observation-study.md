# Original Qwen3.5 base: hybrid-architecture observations

This report preserves the experimental settings, results, and status at the time of writing. For the current project progress, see [Project Status](../../PROJECT_STATUS.md).

2026-11-07. Execution of the original checkpoint is complete. These results do not establish the effect of weight edits or reproduce the MIT performance result.

The base is fixed to Qwen/Qwen3.5-0.8B revision `2fc06364715b967f1860aea9cf38778875588b17`. The new runtime uses a separate Transformers 5.3.0 and Torch 2.5.1+cu121 environment; the original experiment environment was not changed. The official HF complete ConditionalGeneration class preserves the original parameter coordinates without renaming or remapping that could obscure correspondence between differences.

All 473 executed or loaded tensors were checked against the source checkpoint. The source also contains 15 MTP-branch tensors that the HF backend ignores and does not execute. Their names are recorded in the backend identity; edits to them cannot be treated as effective operations. Vision weights are loaded and checked, but the current interface accepts text only.

## Real execution results

The run used 32 training-derived calibration texts totaling 6,528 tokens, with 92–404 tokens per text. It ran on CPU in BF16 without a GPU. The first request returns token log-prob, top-3, residual 0/12/24, and MLP input 0/12/23. The residual array is 32×404×1024, and the MLP array is 32×404×3584. The estimated return payload was approximately 715.5 MB, and the request took 88.54 seconds.

The second request read attention at layers 3 and 23 for two texts totaling 454 tokens. Each layer array had shape 2×8×246×246, and the request took 5.80 seconds. The 18 linear-attention layers do not expose ordinary attention maps; the legal full attention layers are 3, 7, 11, 15, 19, and 23.

Both budgeted requests succeeded, using 34 actual model forward passes in total; the script recorded approximately 100.34 seconds overall. Source code snapshots, base and data hashes, backend identities, array semantics and budgets are saved. See `artifacts/toolkit/wsp-hybrid-observation-v1/Qwen3.5-0.8B/` for small results; large arrays stay on the server.

## What research can currently be supported?

Researchers can read residuals, MLP inputs, and supported attention from the base model in the same coordinate system, form hypotheses, submit weight proposals at original precision, and compare them through the budgeted service. The toolkit now supports real internal observations of the original hybrid architecture as well as Qwen3-0.6B.

These observations did not reveal the cause of the loops, nor did they prove that a particular layer or direction was suitable for modification. RMS and related statistics must be interpreted alongside context and interventions; large values alone do not establish importance. This tutorial uses non-thinking reference text. The final original-protocol track must separately establish a looping and truncation baseline with thinking enabled and a 32,768-token generation cap.

Generation on 8 training exercises using the original base has been submitted with the original temperature, top-p, top-k, presence penalty, and 32,768-token limit; it is waiting for the shared GPU0 lock. This run is intended to obtain failure trajectories, not public-test accuracy; comparability of HF with the original vLLM has not yet been established. Full MATH-500 validation, reference replay, isolated independent discovery, and editing direction/mechanism comparison are still pending.
