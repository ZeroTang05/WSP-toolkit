# Budgeted generation and dense observation: an executable tool interface

This report preserves the experimental settings, results, and status at the time of writing. For the current project progress, see [Project Status](../../PROJECT_STATUS.md).

2026-11-07. The toolkit was expanded using the archived first-round MIT Experiment 8 protocol. This HF backend supports fixed versions of Qwen3/Llama decoders; the separate Qwen3.5 adaptation has completed real observations, described in [qwen35-observation-study.md](qwen35-observation-study.md). Comparability with vLLM and a complete reproduction of the original work remain open.

## Interface and evidence chain

`WeightStore` saves the editing proposal on the CPU, `ForwardService` executes a request on a fixed weight snapshot, and `BudgetLedger` records success, failure, and calls that are still being executed. They are responsible for writing, observing, and budgeting respectively; the programmer cannot arbitrarily change the sampling protocol by changing the request fields.

The underlying method comes from the [author tool protocol](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/runs/p50c/turns/t001/prompt.txt). This implementation reconstructs the protocol independently. Its outputs record the generation-library version, math scorer, and observation-axis semantics; it does not claim to reuse an undisclosed forward server.

| Request | Implemented return | Key semantics |
|---|---|---|
| generate original／edited | Text, token IDs, termination reason, length, loop-tail diagnostics, optional scores| The temperature/top-p/top-k/presence penalty in the configuration is fixed; each question and each sample has a stable seed|
| generate both | The above results and paired differences between the base and edited models| Both sides have the same question, the same prompt, and the same sampling seed; repeated samples are aggregated by question to calculate the standard error.|
| teacher_forced | token log-prob, top-k, residual, MLP input, attention and length/mask| Arrays are right-padded; see meta for position and normalization definitions. Padding is not treated as a real token|
| generation activations | Average activation at the prompt’s final position and over the generated prefix at the specified layer| Teacher-forced replay after sampling includes the last sampled token; additional computation is recorded separately|

Generation forwards and dense observations share a budget of successful requests. Sample count, token count, attention size, and returned bytes after padding are all capped. Equal request counts do not imply equal computation. Metadata also records physical model forward passes, input/output tokens, replay, and elapsed time.

## Numerical definition and implementation boundaries

- `token_logprobs[t]` represents the base model conditional log-probability of the t-th input token given its prefix; the first token and padding are NaN.
- `topk[t]` represents the next token distribution at the t-th query position. It has different subscript semantics from the previous item; it stores ID and normalized log-prob.
- `hidden_0` is the token embedding output, `hidden_l` is the original residual output of block l−1, and the last layer is also before the final model norm.
- `mlp_l` is the input of layer `down_proj`, the gated intermediate activation, not the MLP output or gradient-based importance.
- Attention returns actual probabilities only for applicable layers. When requested, it temporarily switches to eager computation and restores the backend afterward. Linear attention is not represented as an ordinary attention matrix.
- The presence penalty is applied once to tokens already generated, not to tokens appearing only in the prompt, and does not accumulate with repeated occurrences. This does not establish token-by-token equivalence across engines.

The HF adapter for Transformers 4.57.6 supports Qwen3/Llama and checks loaded tensors against the source checkpoint; unsupported model types are rejected. Qwen3.5 uses a separate Transformers 5.3.0 adapter and has completed 32 original-base observations. The two runtimes are installed separately.

Proposals use content addressing across the kernel and service. The service reads the latest committed proposal and freezes its verified snapshot for each request; subsequent edits do not affect a running request. It restores model parameters on completion or failure. The actual tensor hash at native precision is used for variant identification, distinguishing FP32 proposals from execution weights.

This library is not a process sandbox. Executing arbitrary Python, disabling network/GPU, hiding reference editing and test data, and CPU/GPU scheduling remain the responsibility of the host. The current service is a Python request boundary, not an HTTP service or a fully isolated agent executor yet.

## Math scoring and loop diagnostics

`MathVerifyGrader` adopts the boxed extraction convention and fixed dependencies that have been verified by the author's source code: math-verify 0.9.0, latex2sympy2_extended 1.11.0, ANTLR 4.13.2. The relevant extraction code retains the source and MIT license. It performs time-bounded symbolic scoring on the main thread; it does not quietly switch to string equality in threads with missing or unsupported packages.

When the grader is not configured, the service explicitly returns `correct=null` and `accuracy=null`; text extraction and loop diagnosis are still available, and cannot be regarded as task scores. Periodic tails, repeated 4-gram ratio, blank degeneration, and reaching the length limit are recorded separately. None is used as a direct substitute for answer correctness.

## Necessary inspections completed

A small random checkpoint built with the real Qwen3 implementation was checked against native forward results: token-by-token probability and top-k alignment, the relationship between residuals and final normalization, MLP linear reconstruction, attention probabilities and the causal mask, and padding across different lengths. Further checks covered new proposal snapshots across processes, weight and budget restoration after failure, exact agreement in paired sampling with identical weights, and inclusion of the last token in activation replay. All four combined checks passed, as did the original four direct-weight checks.

The Qwen3-0.6B CPU observation tutorial completed on a real model: 128 training-derived texts, 27,776 tokens, 1 successful request, 128 actual model forwards, and approximately 94.44 seconds. The estimated payload for residuals, MLP inputs, and probabilities was 0.91 GB. Observations on FP32 reference text do not establish generation accuracy or capability gains. The code was frozen; small results are in `artifacts/toolkit/forward-observation-v1/Qwen3-0.6B/`. The real activations were further used in [Pruning and compensation combined experiments](reconstruction-study.md).

## Callable example

The following assumes that the host has loaded supported models, tokenizers, and authorized exercise sets; public test labels are not included:

```python
from weight_programming import WeightStore, ForwardPolicy, PracticeSet, ForwardService
from weight_programming.hf_backend import HuggingFaceBackend

store = WeightStore(model_dir, edit_dir)
backend = HuggingFaceBackend(model, tokenizer, store)
practice = PracticeSet(training_rows, role="train", name="my-practice")
service = ForwardService(backend, store, practice, work_dir,
                         policy=ForwardPolicy(), grader=math_grader)

result = service.request({
    "type": "teacher_forced", "model": "original",
    "texts": ["A reference continuation or a matched failure context"],
    "return": {"token_logprobs": True, "topk": 5,
               "hidden_states": [0, 4], "mlp_act": [3]}
})

paired = service.request({
    "type": "generate", "model": "both",
    "problems": [registered_training_id], "n": 2, "max_tokens": 4096
})
```

The tutorial run invokes neither training nor an optimizer. Future work includes isolated kernel/service communication, the original architecture and generation backend, complete public-task confirmation, and a record of tool choices by an independent programmer.
