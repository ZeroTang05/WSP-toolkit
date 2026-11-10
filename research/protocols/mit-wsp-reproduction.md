# Original tool provenance, independent reproduction, and edit-direction comparison

2026-11-06. The updated final research requirements also identify capabilities the first-stage toolkit must support. The base-model baseline, reference replay, and independent discovery have not yet been completed, and the existing GSM results of Qwen3-0.6B cannot be called a reproduction of this experiment.

## 1. Verified original source

The original work is the MIT CSAIL authors’ [Sparks of Weight-Space Programming research blog](https://rikiyat.github.io/blog/weight-space-programming/), a blog post; this review did not find a formal peer-reviewed paper. The interface specified here is based on the [Experiment 8 first round task and tool protocol](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/runs/p50c/turns/t001/prompt.txt) released by the author. For the results/file mapping, see [Experiment README](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/README.md).

Core facts: The target is Qwen3.5-0.8B, thinking is on; the generation budget is 32,768, sampling T=0.6, top-p=.95, top-k=20, presence penalty=1.5; no training or gradients. The authors report that the MATH-500 for 50-pass editing is 7.9% → 48.1%, with substantially less loop-related truncation. This is a different condition from the original Qwen3-0.6B non-thinking experiment of this project.

A replayable final delta is provided on the authors’ [Model Release Page](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming) and repository. Specific editing coordinates, coefficients and column indices are only saved in the reference area for comparison after reproduction; they are not included in the independent programmer’s prompts or toolkit defaults.

The first round protocol, experimental instructions, final report and reference list have been archived in `research/references/mit-wsp/`, each recording the Git blob SHA. The repository [license](https://github.com/RikiyaT/lm_understand-logs/blob/main/LICENSE) distinguishes between scripts and logs/data, and the original license and source are retained when archiving.

## 2. What tools does the original work actually give us, and what must we include?

The original work gives the editor direct access to analyze and write tensors, then uses a controlled forward service to obtain evidence. Its action space is broader than a list of LoRA and pruning algorithms.

| ability| The interface/meaning of the original protocol| Requirements and status of this tool library|
|---|---|---|
| Read weights and configuration| `NAMES`, `load`, `load_original`, persistent Python variables and tokenizer on CPU| `WeightStore` and `kernel_bindings` have been implemented; the persistent CPU isolation kernel is running in real 0.6B|
| Direct weight writing| `save_edit`, `drop_edit`, `edited`; shape/scope constraints| Added independent FP32 proposal storage, original precision export, change counting and protection strategy; source checkpoint does not change|
| Numerical/geometric analysis| NumPy/PyTorch, etc. on CPU, analyze tensors and observation results| Existing update spectra, norms, and comparisons with new directions; more matrix analyzes can be calculated within the kernel without the need to pretend to be independent algorithms|
| Generate query| original/edited/both, returns text, grade, length, truncation and pairing changes; explicit budget| The unified budget service and separate Qwen3.5 runtime are connected; the complete generation effect is still to be measured.|
| dense forward observation| Teacher-forced reading returns token log-prob/top-k, residual, MLP input and attention at supported layers| Real observations of Qwen3 and Qwen3.5 have been completed; the hybrid architecture only returns attention in the supported full-attention layers|
| Causal editing experiments| Change intensity, position isolation, comparison, and pair generation confirmation| Some operations have already been performed; they need to be combined into a clear experimental working method, rather than just observing correlations.|

The archived first-round protocol defines capacities and return values; a budget request cannot be assumed to mean one single-sample forward pass. The processing of failed requests is recorded separately in the budget and actual cost ledgers.

The complete original agent harness and forward server have not been fully disclosed; logs and protocols can support interface reconstruction, but we should not claim that we directly reuse the complete author executor. Its full attention and linear attention hybrid architecture also requires new model adaptation, and the assumption of "all layers have ordinary attention maps" cannot be used.

## 3. Direct weight primitives that have been added

The new package `src/weight_programming/` includes:

- `WeightStore`: Read shard safetensors, bind source fingerprints, maintain current proposals and original tensors, discard edits, and export replacement tensors with source precision.
- `EditPolicy`: Explicitly protect parameter names/editable prefixes, reject wrong shapes, non-finite values, and whole-tensor scaling prohibited by the policy. Thresholds are a declared choice of this implementation and are not equivalent to undisclosed author inspection code.
- `contrast_direction`: Matches the observed mean difference direction; it is a candidate explanation and still requires causal confirmation.
- `project_output_direction`: Directional update given output direction, column set, and intensity, no preset layers, channels, or vectors.
- `compare_edits`: Signed edit cosine, edit norms, and overlap in column support in the same coordinates; geometric similarity is not regarded as the same mechanism.

For matrix W, unit output direction u, and column selector m, the orientation primitive is

\[
W'=W-\alpha\,u\big(m\odot(u^TW)\big).
\]

This algebraic form can be checked in the authors' references; references to earlier studies of directional interventions include [Arditi et al. 2024](https://arxiv.org/abs/2406.11717). This library only provides general primitives and does not hard-code solutions for reference experiments.

Four CPU checks have been passed on the server, covering algebra/column support and direction sign invariance, proposal persistence and original precision materialization, source checkpoint invariance and range constraints, and matching direction estimation. They demonstrate local implementation properties and do not demonstrate improved mathematical abilities. The new package has been added to the installation package list, and the complete public release is still waiting for the interface, tutorial and license to be sorted out.

## 4. Three reproduction tracks, each answering a different question

### A. Reference replay: Calibrate the comparator

Pin the original base revision, tokenizer, template, execution dependencies, sampling settings, and scorer in a standalone reference workspace; apply the author's delta and check the exact hash of each modified tensor. The base model and reference edit are remeasured in the same execution environment to establish a comparable reference, rather than directly using blog decimals for statistical matching.

This step validates the model/scoring chain and the published artifacts, but does not prove that our programmers will discover the edit. There are rounding differences between the original author's BF16 finished product and the FP32 submission. When comparing, the "selected modified position" and "actually changed position under the original precision" are recorded respectively.

### B. Independent Discovery: Core Competency Test

Launch a programmer in a new isolated session with only the model, permitted practice data, general tools, and budget. **The current research session has already read the reference solution, and subsequent discoveries in this session cannot be directly called blind discoveries.** The independent session cannot access to the reference log, modification list, reference directions, or original-specific default parameters; all selections are generated solely from its own analysis and experimentation.

Freeze the programmer’s model version, initial prompt, tool availability, budget, and human intervention. Separate the no-training permissions of the original work and the hybrid optimization permissions of LoRA/Distillation allowed by this library; the main track of the final reproduction of the original work uses the former and cannot be replaced by the benefits of the latter.

First construct and select on the practice set, then confirm on practice questions not used for selection. The full official MATH-500 will be evaluated only after the final program and weights is frozen, and GSM8K and instruction-following transfer evaluation will be retained. Record multiple samples for each question. The question is the main pairing unit. Repeated sampling of the same question cannot be regarded as an independent question.

"Same effect" requires giving acceptable differences and interval rules in advance. The quality, looping, truncation, and generation costs of local reference replay on the same executor should be compared with independent solutions; reproducing exactly a reported decimal is not the only criterion, nor does merely ending inference early count as capability recovery.

### C. Comparison of direction and mechanism after the effect reaches the target

Let the reference and independent edits on the same base be Δ_ref and Δ_ours. Compare them in this order:

1. **Location/Support.** Which layers, modules, and channels are modified; precise position coincidence, actual change amount, and Frobenius energy of each layer. The same column index only means that the support is the same.
2. **Parameter direction.** Computes signed cosines, norm ratios, and per-layer cosines in the joint space of the same tensor coordinates. Same direction and opposite direction of different intensities must be distinguished.
3. **Subspace.** Multi-directional editing looks at singular spectra and principal angles, and cannot misjudge differences due to factor rotation or sign selection of u and −u.
4. **Functional direction.** Compare Δ_ref X and Δ_ours X on the same set of base activations X; measure changes in loop/normal context and output log-prob. You need to start from a common activation, and you cannot mix inputs from the two edited models that have drifted.
5. **Causal mechanism.** Positional isolation, intensity curves, norm-matched random directions, undo/reverse editing and controlled crossover replacement. The crossover operation must control the norm and support, otherwise the intensity difference will be regarded as a direction difference.

The results can be "same support and same direction", "same subspace but different intensities", "different locations but close functions" or "same scores but different repair mechanisms". Only when the evidence is sufficient can a conclusion be drawn; neither geometric proximity nor overall performance closeness is sufficient to determine the same mechanism alone.

## 5. Output list and next step

The first-stage deliverables must include an MIT interface coverage table and callable primitives. Final phase deliverables: base model/reference finished product replay, complete independent programmer trajectory, frozen weight variants, per-question public-task results, cycle and cost metrics, and layer-by-layer/subspace/function/causal direction comparison reports.

2026-11-07: Budgeted and dense observations, math scoring, explicit loop tail diagnostics, and persistent isolation kernels have been implemented, see [persistent-kernel.md](../reports/persistent-kernel.md) and [forward-service.md](../reports/forward-service.md). The independent TF5.3 environment and the original Qwen3.5 real observation have been completed; [New context programmer](../reports/fresh-programmer-study.md) and [Complete MATH500 base model evaluation](../reports/math500-evaluation.md) have been submitted, still waiting for GPU/not completed. Confirmation of the complete effect of the base model, comparability of the author's executors, reference replay, independent discovery of benefits and comparison of direction mechanisms have not yet been established. The old TF4.57.6 experimental environment is retained and will not be mixed with the hybrid architecture runtime.

Method development, 0.6B experiments and 4B professional workflows continue to provide verification for the tool library, and the final original track retains the target model and permissions alone. None of these easier experiments replaces the requested independent discovery and edit-direction comparison.
