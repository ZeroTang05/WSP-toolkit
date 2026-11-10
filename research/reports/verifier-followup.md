# Verifier Supervision Distribution: Failure Diagnosis and Paper Basis

This report preserves the experimental settings, results, and status at the time of writing. For the current project progress, see [Project Status](../../PROJECT_STATUS.md).

Date: 2026-11-06. This plan was formulated after the full V3 development diagnostic; the header-repair control and main full baseline remain frozen. It does not claim that new training or multi-candidate workflows have been implemented.

## Supervision audits completed

`audit_verifier_data.py` reads the fixed training view and checks the manifest hash, using only the official FinQA train. Among the 6,251 supervision examples, those starting with add contain 2,543 rejects and 404 accepts; those starting with divide contain 1,166 accepts and only 2 rejects. Deterministic operator replacement substantially changes the label distribution conditional on the operator.

A question-ID hash divides the data into 5,015 fitting examples and 1,236 probe examples. A classifier using only the first operator and its majority label is correct on 1,049 probe examples (84.87%); the global majority label reject, determined on the fitting split, achieves 47.98% on the same probe. The probe is held out only from fitting this count-based classifier; some questions may have been used for the language model SFT update. Therefore, 84.87% does not measure LLM capability, real error detection, or generalization to unseen data.

Among the 371 divide-first plans in the full V3 development set, 213 incorrect plans were accepted, 9 incorrect plans were rejected, and 149 correct plans were accepted. Training supervision contains very few negative divide examples. This correlation suggests a mismatch in the supervision distributions and does not prove that errors in a specific model are entirely caused by first operator shortcuts or missing table headers. Across the 68 retries after rejection, 0 questions changed from wrong to right and 1 changed from right to wrong. This further illustrates that program validity cannot stand in for workflow gains.

The current audit and problem group probe can be rerun:

```sh
.venv/bin/python src/audit_verifier_data.py \
  --view benchmarks/views/finqa-roles-v3 \
  --output research/finqa-verifier-data-audit.json
```

## Sources and boundaries of papers in new training directions

[Cobbe et al. 2021, Section 4.2](https://arxiv.org/html/2110.14168v2#S4.SS2) trains the verifier using the generator’s actual candidates and final answer correctness, and selects candidates based on the verification score. Its approach offers two changes that can be studied independently: supervision on real candidates and inference-time sampling and ranking. The original paper is not "artificially replacing the first operator to generate negative examples", nor is it "rewriting it after binary rejection".

When applied to FinQA, candidates should be generated only from official train questions, with results labeled by the fixed official executor and exe_ans. The FinQA program, LoRA parameterization, generative accept/reject targets, and the reduced number of candidates are all adaptations of this project and do not justify carrying over the paper’s results or claiming a complete reproduction. A correct final number does not prove that the evidence was understood or the reasoning process was correct.

[Lightman et al. 2023](https://arxiv.org/abs/2305.20050) study the difference between feedback on reasoning steps and feedback on final outcomes. There is currently no step validity label, so process supervision is not within the scope of implementation; the fact that a numerical calculation executes does not label a step as semantically correct.

## What must be isolated for subsequent control?

1. Keep the V3/V4 header comparison fixed and first identify the effect of the representation change. Do not change both negative examples and headers simultaneously and then attribute an effect to each.
2. Fix the training-split questions, generator, sample sources, and random seeds. Record which questions were used to update the generator; exclusion from that update does not establish that they were unseen during pretraining. Split candidates by question, so candidates for the same question cannot cross the supervised training and validation sets.
3. Train separately on synthetic negatives and actual model errors, matching question coverage, positive/negative ratios, update steps, and rank/total delta bytes. Account separately for candidate generation and labeling costs. Simply increasing the number of training steps cannot answer the question of data distribution.
4. With the same verifier, compare the current rewrite-after-rejection approach with multi-candidate ranking, alongside random selection, executed-answer voting, and an oracle candidate upper-bound diagnostic. Report candidate counts, generation-token limits, and actual time/GPU memory separately; equal token limits do not establish fully equal costs.
5. Retain the original single-shot, prompt-only, standard multitask, original-role, and shared-base/copy references. Base all claims on full dev/test results, three seeds, and end-to-end execution accuracy. The oracle is diagnostic only and cannot count as a deployable gain.

These are source-backed requirements for follow-up experiments; there are no measured results for the new conditions yet. The first step remains collecting the submitted header, baseline, and role matrices.
