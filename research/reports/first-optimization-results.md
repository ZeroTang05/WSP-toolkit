# First batch of optimization results and failure diagnosis

This report preserves the experimental settings, results, and status at the time of writing. For the current project progress, see [Project Status](../../PROJECT_STATUS.md).

Status: evidence available on 2026-11-06 from a single optimization seed. The full experimental program and paper conclusions are not yet complete.

## Standard LoRA reference: complete splits

Qwen3-0.6B used seed11, 100 steps, effective batch16, and q/v rank8. The base model and LoRA were evaluated with the same batch8 protocol:

| Task| base model| LoRA | change|
|---|---:|---:|---:|
| GSM8K flexible EM, 1,319 questions| 36.47% | 39.80% | +3.34pp |
| ARC-C acc_norm, 1,172 questions| 30.97% | 30.97% | 0.00pp |
| HellaSwag acc_norm, 10,042 questions| 41.99% | 42.86% | +0.87pp |

The paired per-question bootstrap 95% intervals are [+0.15, +6.60]pp, [-2.05, +1.96]pp, and [+0.24, +1.47]pp respectively. They only reflect question sampling, do not take into account optimization seed changes and multiple comparisons, and do not constitute a formal guarantee of non-inferiority. In GSM8K, 262 questions changed from wrong to right, and 218 questions changed from right to wrong; a net increase of 44 questions.

The complete retention tasks did not support inferring impaired standard accuracy from a large early increase in raw option cross-entropy (CE). Raw cross-entropy, character-length-normalized option metrics, and generation accuracy must be checked separately. The revised retention proxy was frozen using the standard harness boundaries and normalization before these retention test results were inspected.

The revised unified batch32 generation protocol yielded 485/1319 (36.77%) for the base model and 532/1319 (40.33%) for the same LoRA. These results are kept separate from the earlier batch8 results; gains must use a baseline from the same protocol. Generation with the same weights differs slightly across BF16 execution batch sizes, so the execution protocol must specify backend settings.

### Task Arithmetic fixed-grid reference: first seed across three complete tasks

The 21-point global development grid selected 0.8 times the training update, with all four depth groups set to 0.8. On full GSM8K it scored 590/1319 (44.73%). Under the same batch32 protocol, that is 105 more correct answers than the base model (+7.96pp; paired per-question bootstrap 95% interval [4.70, 11.14]pp) and 58 more than standard LoRA (+4.40pp; interval [2.20, 6.75]pp). Relative to the base, 294 answers improved and 189 degraded; relative to LoRA, 148 improved and 90 degraded. Program and document fingerprints were checked against the formal inference protocol; the original samples and both per-question analyses were downloaded.

The complete retention split is now verified:

| Same confirmation protocol, seed11| base model| StandardLoRA| Fixed grid 0.8|
|---|---:|---:|---:|
| GSM8K, 1319 questions, batch32| 36.77% | 40.33% | 44.73% |
| ARC-C acc_norm, 1172 questions, batch8| 30.97% | 30.97% | 29.95% |
| HellaSwag acc_norm, 10,042 questions, batch8| 41.99% | 42.86% | 44.24% |

The fixed 0.8 setting answered 351 ARC-C questions correctly, which is 12 questions less than the base model (−1.02pp). The paired question bootstrap 95% is [−2.90, +0.85]pp, with 58 questions improved and 70 questions degraded. It answered 4,443 HellaSwag questions correctly, with a net increase of 226 questions (+2.25pp), and an interval of [+1.59, +2.88]pp. 652 questions were improved and 426 questions were degraded. Both positive and negative results are reported; the math improvement alone is insufficient. Passing the CE retention proxy does not establish retained classification accuracy. Relevant counterexamples and boundaries are recorded in `tool-semantics.md`.

This is the fixed search reference for Task Arithmetic global scaling, not the LLM controller gain, nor the four-dimensional adaptive policy gain. The coefficients were selected using the development proxy, without tuning parameters on final answers. The selection objective and retention CE are declared local adaptations of the original paper, not a complete reproduction. Single seed and unadjusted question intervals do not establish cross-model, cross-seed, or general ability retention conclusions.

Added `analyze_model_trends.py` and `model-trend-protocol.md`. They calculate a two-level seed/question bootstrap from complete per-question records, sampling seeds independently across models and questions jointly, while pairing C1/C2 within each model. Model comparisons are output only when both sides have 3 seeds. At present there are only 6 single-seed task summaries and no conclusion about model differences. All 4 statistical tests passed.

## Pruning failure sample

Wanda’s row-wise 50% pruning, calibrated on prunable projection layers using 128 training-derived examples, scored 197/1319 (14.94%) on full GSM8K. This set of configurations does not maintain mathematical domain capabilities; unstructured masks do not automatically speed up current dense backends. The structural Taylor adaptation separately runs three calibration samples, two pruning widths, and real prefill/decode profiles. This Wanda result cannot be used to replace the structural compression conclusion.

## The first complete five-strategy development comparison: Qwen0.6, seed11

All five completed tracks were audited using the same training direction, 64 math development proxy questions, 32 retention proxy questions, and 21 queries per strategy. Lower risk is better; these are teacher-forced development proxies, not full generation accuracy.

| Strategy| Selected math risk bits/byte| Protection difference upper95 (threshold 0.02)| Search loop seconds|
|---|---:|---:|---:|
| Fixed global grid| 0.31848 | 0.00032 | 106.00 |
| Random | 0.31890 | −0.01531 | 102.63 |
| Single-axis lexicographic detection (code key greedy)| 0.59424 | 0.01880 | 94.86 |
| BO | 0.31513 | −0.02574 | 110.58 |
| LLM | 0.36277 | −0.01680 | 142.21 |

The LLM selected [0.75, 0.35, 0.45, 0.55]. All 19 actual proposals were valid, with 0 fallbacks, so the weak proxy result cannot be attributed to JSON failure or random fallback. The controller used 27,026 input tokens and 1,211 output tokens. Recorded controller time was 39.02 seconds and target evaluation took 102.72 seconds. Peak CUDA allocation was 9.82 GB, versus about 1.58 GB for the other strategies; a single run cannot establish a stable cost difference. Search-loop time excludes loading and queuing and cannot be combined with the new resident-budget track to calculate total cost.

`greedy` selects single-axis candidates lexicographically and may explore only one axis within a limited budget. It is not complete standard coordinate descent, so outperforming it does not establish superiority over greedy optimization in general. Fixed grid and BO are strong references at this stage. The above single seed development results do not support the LLM controller advantage and are not a substitute for its full GSM/retention task confirmation or cross-model conclusions. Evidence: `qwen06-seed11-five-policy-development.json` and the downloaded original LLM trajectory.

### Random-selected state: complete GSM8K confirmed; retention tasks pending

For the same Qwen0.6 seed11, batch32, and 1319 questions, the random strategy selected [.75,.85,.70,.95] and answered 589 questions correctly (44.66%). The selected plan was checked against the actual parameter state, complete doc_id/sample hashes, runner/helper, and evaluation settings. Compared with the base model, it has +104 questions and +7.88pp, and the paired question bootstrap 95% is [4.62, 11.07]pp; compared with the standard LoRA, it has +57 questions and +4.32pp, and the interval is [1.97, 6.75]pp.

This is one fewer correct answer than fixed 0.8, a difference of −0.08pp with an interval of [−1.82, +1.59]pp. Random corrected 69 answers that fixed missed, while 70 changed in the opposite direction; correctness differed on 139 questions. The similar mean scores do not establish method equivalence or model interchangeability, and do not establish that parameter states can be reused interchangeably. The retention tasks have not yet been completed; the results are single seed only and do not cover training variation and multiple comparisons.

### Resident Budget Track: First Clock/Ledger Smoke

The fixed-grid smoke test in the new track used only 4 development and 4 retention questions, completed 21 queries, and passed selection auditing and recomputation from the original ledger. The GPU lock wait was 2612.57 seconds, recorded separately from the 2617.10-second setup time including the wait; the total active time is 17.08 seconds, active setup is 4.54 seconds, target evaluation is 11.51 seconds, policy decision-making is 1.01 seconds, and the remaining records/statistics are about 0.016 seconds. Input/output token counts agree with the controller timing summary. Long queue waits must not be reported as model computation costs; this smoke test does not establish formal quality or efficiency gains. The random/BO/LLM budget smoke is still to be implemented, and the complete budget matrix has not yet been completed.

## Development diagnostics of real expert processes

### The first complete development partition: standard PEFT backend, seed11, V3 input

All 883 official FinQA development questions were completed, and their programs were independently re-executed. In the end, 349 questions (39.52%) were correct. The Wilson 95% interval for question sampling alone was [36.35%, 42.79%]; the program was valid for 872 questions (98.75%), and 350 questions (39.64%) were correct on the first plan. Gold support ID recall was 72.04%, and the rate of covering all support IDs was 66.93%. This is not the final test result and has not been paired with the full prompt/multitask baselines, so it cannot establish a benefit from role specialization. The initial 0/8 result does not summarize the complete split.

The verifier accepted 815 plans, of which 338 were initially correct and 477 were incorrect; it rejected 68 plans, of which 12 were initially correct and 56 were incorrect. After 68 repairs, no question changed from wrong to right and one changed from right to wrong. In the end, there was a net loss of 1 correct question. Verification/repair under this condition does not improve the end-to-end results and cannot be extrapolated to mean that all verifiers are invalid.

The new `audit_finqa_prompts.py` uses the frozen prompt constructor and hash-checked tokenizer of this run to reconstruct the prompt hash and input length of all 2,717 calls, all of which are consistent. The selected table content lacks a header for 705 of 883 questions. This describes the scope of the input representation issue; it does not establish that the missing headers caused those 705 answers to be wrong. After the first prompt hash was tampered with in a copy of the real 8-question run, the enhanced audit explicitly rejected the copy; the original record was not modified. Sources, per-question predictions, execution audits and original calls are located at `artifacts/public/finqa/peft-seed11-full/dev`.

26,981 tokens were actually generated, 1,639,362 tokens were input, 2,717 calls were made, the recorded generation time was 874.64 seconds, and the CUDA peak value was 4,126,879,744 bytes. The timing is an accounting check of the original run and has not been independently repeated. The peak for all 883 questions cannot be compared with a separate 8-question run to infer deployment benefits. See the next section for a complete test under the same conditions; other seeds, non-PEFT backends, and baselines continue to be executed.

### The first complete official test partition: the same PEFT seed11 V3 role workflow

All 1,147 questions were completed, with independent program re-execution, source checks, and per-question scoring. The final score was 425 correct (37.05%), with a question-sampling-only Wilson 95% interval of [34.31%, 39.89%]; 1,135 programs (98.95%) were valid. The first plan was correct on 426 questions (37.14%). After verification, across 99 retries, 1 question changed from wrong to right, and 2 questions changed from right to wrong. In the end, there was a net loss of 1 question correct. This result is not a multi-seed mean, and a gain over a complete baseline under matched conditions has not yet been established.

| Same frozen conditions| Official dev| Official test|
|---|---:|---:|
| Number of samples| 883 | 1147 |
| Initially correct plans| 350 | 426 |
| Final correct answers| 349 | 425 |
| Final execution accuracy| 39.52% | 37.05% |
| Valid programs| 872 | 1135 |
| Retries| 68 | 99 |
| Wrong to right/right to wrong| 0／1 | 1／2 |

The test verifier accepted 410 correct and 638 incorrect plans, and rejected 16 correct and 83 incorrect plans; gold support ID recall was 71.83%, and the rate of covering all support IDs was 66.96%. Both splits show low error detection and no net repair gain. These observations do not establish that all verifiers are ineffective or identify the causes of their errors.

All 3,540 prompt hashes and input token lengths match reconstruction using the frozen helper and tokenizer; the selected table content of 882 questions lacks a header. 35,245 tokens were generated, 2,151,759 tokens were input, the recorded generation was 1154.87 seconds, the warmup was reported separately for 1.28 seconds, and the CUDA peak value was 4,373,295,616 bytes. The complete single-run timing is not a repeated profile; its peak cannot be compared with that of an 8-question run to infer deployment benefits.

The original predictions, scored results, protocol, source code, and enhanced audit have been downloaded to `artifacts/public/finqa/peft-seed11-full/test`. Further seeds, original functional backends, baselines, and V4 header-repair matrices continue under their respective frozen conditions. Development and test results are reported separately; the diagnostic development split is not included in the test score.

### Synthetically Supervised Operator Shortcut Risk

In the fixed training supervision, programs starting with add have 2,543 rejects and 404 accepts, while those starting with divide have 1,166 accepts and only 2 rejects. `audit_verifier_data.py` splits the data by question ID into 5,015 fitting examples and 1,236 probe examples. A classifier using only the first operator’s majority label scores 1,049/1,236=84.87% on the probe; the global majority label baseline determined by the fitting partition is 47.98%. The probe is held out only from fitting this count-based classifier; it is not independent of LLM SFT. It does not measure the LLM’s generalization or real error detection.

V4 adds only table headers; its operator-label distribution matches V3 item by item and in the probe. This suggests a shortcut risk, but does not prove that the verifier learned those rules or attribute full development-set errors to them. For the paper basis and necessary controls for subsequent real candidate supervision, see `verifier-followup.md`; the new conditions have not yet been trained.

### Smoke records from the previous 8 questions

FinQA’s three independently trained role deltas can run in a shared-base workflow. Among the first 8 development samples, the program validity rate was 100%, the execution accuracy rate was 0%, the verifier accepted every plan, and the evidence coverage rate was 3/8. This is a smoke diagnostic, not a full development- or test-set score.

Even if all gold support row IDs are selected, denominator selection errors, constant/percentage processing errors, and subtraction direction errors still occur. A subsequent audit found that ID coverage did not mean complete information: V3 did not automatically retain the year header when selecting table rows. Therefore, we cannot first attribute all these errors to model capabilities or negative example training. We need to isolate the input representation problem first. Negative examples from actual training-split errors remain a separate follow-up control; official numerical scoring must not be relaxed to improve results.

A new independent audit of the original records compared 8 questions in the same batch: a single multitask adapter answered 3 correctly, independent role adapters answered 0 correctly, both had 100% program validity, and the verifier accepted every plan. Therefore current development diagnostics do not support role specialization over multitasking adaptation. `audit_finqa_run.py` re-executed the original program, checked the public input fingerprints, the original output of each stage of the model, the call budget and the billing summary; copies with deliberately tampered verdicts were rejected. Eight questions are insufficient to estimate the full-split effect, so the formal comparison continues.

### Header loss and pre-declared repair comparison

`PM/2017/page_38.pdf-1`'s question requires changes in operating income from 2016 to 2017. V3 selects `table_3` and leaves only `operating income | 11503 | 10815 | 688 | ...`, whose year column title comes from `table_0`. FinQA author's [Fixed version row text transform](https://github.com/czyssrs/FinQA/blob/0f16e2867befa6840783e58be38c9efb9229d742/code/utils/general_utils.py#L21) writes column headers to text; the original implementation did not preserve this information.

V4 adopts explicit public header retention rules, which is an adaptation of this project and does not claim to reproduce the author's entire data pipeline. Among the 6,251 training samples, 4,754 have headers added; planner/verifier each changes 4,754 prompts, and multitasking changes 9,508. Full comparison confirms that the answers, programs, IDs and order remain unchanged, and that the common inputs for retriever training and development/testing are consistent byte by byte. Both the old and new directories and frozen runs are retained. The longest planner and verifier sequences are 703 and 649, respectively, both below the same limit of 12,288; 9 FinQA-related CPU tests passed.

The first 8-question input control changed only the inference table header and reused the original role factors. It scored 1/8 correct versus 0/8 for the same original V3 batch; all 8/8 programs were valid, and the verifier accepted every plan. All retriever prompts and outputs are identical. Headers are added to the planner and verifier prompts; original prompts and token lengths were reconstructed, and the missing-header count fell from 8 to 0. The PM sample's `subtract(10815, 11503)` is changed to `subtract(11503, 10815)`, and the result -688 changes to 688, which is correct; the DVN sample's numerator 23 is changed to 60, but the final answer is still wrong. Only two programs have changed, so the improvement in one question cannot be extrapolated into an overall quality conclusion, nor can it be called a parameter optimization benefit.

An 8-question diagnostic that removes the retriever and supplies the full public context also scored 1/8 correct with 8/8 valid programs; it changes both context and call cost and is not an equal-cost replacement. The full V4 input and retraining controls are still running. The evidence is located at `artifacts/public/finqa/header-input-seed11-smoke`, `no-retriever-seed11-smoke` and `provenance/finqa-header-input-smoke-comparison.json`.

`run_finqa_header_control.py` declares a three-seed, full dev/test comparison of input repair and retraining: the original expert with headers, an equally budgeted retrained expert with headers, corresponding equally budgeted multitask conditions, and a prompt baseline. The control reuses the original seed11 training code and recipe; the retriever reuses the original delta with the same training-file fingerprint. Negative examples, update steps, and labels are not changed simultaneously. It is an exploratory fix after seeing the failure of V3 development, and cannot be called a new unseen test confirmation. Job `job-mw4sz1kg-d751` has been submitted, but there is no quality conclusion yet.

### Shared base vs. true independent copy

Using the same three experts, 8 questions, and call and generation budgets, an independent audit found identical stage-by-stage prompts and outputs, with 209 tokens generated. Peak CUDA allocation was 3,771,593,728 bytes for the shared base and 10,655,127,552 bytes for the three copies, a single-run descriptive reduction of approximately 64.6%. The generation times are 6.23 seconds and 6.15 seconds respectively. There is no evidence of speed advantage. The logical base parameters occupy 3,441,149,952 and 10,323,449,856 bytes respectively; the delta files on both sides are 19,301,976 bytes. Logical copy size is not treated as actual additional disk use, and equivalence across all inputs cannot be inferred from 8 questions; complete splits and repeated profiles in randomized order are needed. Evidence: `artifacts/public/provenance/shared-replicas-smoke-comparison.json`.

## Method Semantics and Backends

For the same real Qwen0.6 LoRA factors, the custom functional implementation and PEFT 0.21.2 produced identical FP32 logits on the tested prompt. In BF16, the maximum logit difference was 0.21875 and the position-wise prompt argmax matched. A prompt's argmax consistency does not guarantee complete generation consistency, and the standard PEFT workflow still needs to be independently evaluated.

The original functional/PEFT records of the 8 questions in the actual FinQA workflow have been independently audited and compared stage by stage. The prompts and generated outputs match, both execute 0/8 questions correctly, and both generate 209 tokens. The CUDA peak values ​​are 3,771,593,728 and 3,783,832,576 bytes respectively; a single generation takes 6.23 seconds and 6.77 seconds. This adds evidence of consistency for the tested generations and does not prove that full partitions are equivalent or that custom backends are faster. The full PEFT dev/test is still running; see `artifacts/public/provenance/functional-peft-smoke-comparison.json` for proof.

The precision check now converts only floating-point parameters, preserving native RoPE buffers and their aliases; full model bfloat16 conversion also changes non-persistent buffers. Model interventions should record execution state beyond the parameters; parameter norms alone cannot explain all behavioral changes.

The evidence is located in the Qwen3-0.6B, confirmation-v2, finqa, peft-reference directories under `artifacts/public/`. The formal five-strategy, multi-model, multi-seed, capacity-control, complete-workflow, and compression-efficiency matrices are still running.
