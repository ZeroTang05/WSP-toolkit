# Collaboration Benefits Verification Benchmark: HotpotQA Candidate Document Reading

2026-11-06. This implements the first stage: verify multi-agent collaboration before extracting role-specific models. The experiment has been implemented and submitted; no task-quality results are available, so a collaboration benefit has not been established.

## Implemented test subjects

Use the distractor setting of [HotpotQA official data](https://hotpotqa.github.io/). All methods for each question can read the same set of candidate documents, and no gold answers or supporting facts are provided. Here we study multi-hop reading of a given document, which is not called open domain/fullwiki retrieval.

The data source is `hotpotqa/hotpot_qa`, fixed revision `1908d6afbbead072334abe2965f91bd2709910ab`, downloaded through hf-mirror; save the SHA256 of each parquet. Official scoring code fixed `hotpotqa/hotpot` commit `3635853403a8735609ee997664e1528f4480762a`. The original source code is retained, its functions are loaded through AST without rewriting the scoring formula, and per-question answers, supporting facts and joint indicators are output.

The 90,447 training questions are split by ID hash: 128 questions for calibration, 256 questions for development, and the rest are reserved for future role training. Training questions whose normalized text exactly matches a public dev question are excluded during construction; this is not a near-duplicate/pre-training contamination exclusion guarantee. The complete public dev is 7,405 questions. It is only for confirmation after freezing and cannot be called an official test.

`prepare_hotpot_collaboration.py` exports separate inputs/labels. Inference inputs only include ID, question, candidate document title and sentence, and input validation rejects redundant fields. Target partition labels are loaded for scoring only after all predictions are completed; in addition, two disjoint training calibration examples and their labels are allowed to be loaded for few-shot hints. This constraint is an in-process experimental protocol, not operating system access isolation.

## Five comparison conditions

| condition| Calling process| The upper limit of allocated output tokens|
|---|---|---|
| direct | Strong prompts reason and answer directly across documents| B |
| self_refine | Initial answer→check→revise| B/4, B/4, the rest|
| independent_vote | Three independent answers, majority vote after normalization| B/3, B/3, the rest|
| collaborate | Decompose role→Evidence role→Answer role; pass earlier messages| B/4, B/4, the rest|
| no_communication | Same role, same budget, but the evidence/answer role cannot see the earlier messages| Same as collaborate|

For integer budgets, round each allocation down and give the remainder to the last call. Voting does not add an LLM judge; tied votes take the earliest valid candidate, and its supporting facts are also taken from that candidate, and gold facts are not spliced. If there is no valid candidate, it is recorded as a format failure. Each condition ultimately uses the same JSON format and parsing rules; all roles retain the complete original candidate document, so no additional information is given only to the collaborative role.

Role splitting refers to [Role Capacity Research](https://arxiv.org/abs/2607.07548), communication comparison refers to [Multiagent Debate](https://arxiv.org/abs/2305.14325)’s collaboration issues; strong single Agent comparison basis [ACL 2024 re-evaluation](https://aclanthology.org/2024.acl-long.331/). The current fixed three-stage reading process is an adaptation for this project, and we do not pretend to be a complete reproduction of the algorithms of these papers.

According to the findings of demonstrations in this ACL study, the same two training examples are added to each call of all conditions, including complete candidate documents and correct output formats; the first two questions of the calibration view are selected and are not filtered by effects. The example IDs and prompt hashes are recorded in the protocol, and neither example IDs nor normalized questions overlap with evaluated questions. It is an explicit two-shot strong prompt reference and still does not claim to cover all possible single-agent prompt optimizations.

## Fixed execution and selection rules

The first candidate is the downloaded Qwen3-4B-Instruct-2507. Its strength and any collaboration benefit remain to be measured. There are currently no additional training or API teacher fees.

- B takes 384, 768; random seeds 11, 22, 33. Each call uses the question ID/seed/call sequence number to derive a random seed, temperature0.7, top_p0.9, non-thinking mode.
- A single base model, BF16, SDPA, each role is prefetched separately, with no cross-role KV cache sharing. Explicit failure when the sequence exceeds the model's maximum context does not silently crop the document.
- A total of 256×5×2×3=7,680 complete problem processes for all development conditions. It is a training-derived development comparison, not an official quality result, nor an 8-question execution test.
- For each budget, use the development answer EM in direct/refine/vote to select the strongest reference; then select the budget with the greatest gain from "collaboration minus reference", and select the lower budget with the same gain.
- If there is no positive development gain in both budgets, a negative result is recorded and role specialization is not automatically entered; the next step should be to study task/process adaptation.
- If the development gain is positive, freeze the reference and budget, and run 3 seeds each for collaborate, selected reference, and no_communication on the full public dev, for a total of 66,645 complete processes.
- Confirm that the main comparison is the reference for the selected collaboration pair development; use a bootstrap paired by seed and question, and the lower bound of the interval is greater than 0 to pass the collaboration effect threshold of this protocol. Do not re-select the most advantageous budget or reference in public dev.

Three seeds only describe inference randomness, not three training seeds. Format compliance is also part of the results of structured tasks, but if the benefits mainly come from the format legality rate, it must be explained separately and cannot be said to be an improvement in general reasoning ability. The answer EM is to pre-specify the main indicator; F1, supporting fact F1 and joint indicator are reported together, and do not selectively report only favorable metrics.

## The connection between efficiency and role extraction

The shared total output cap controls only part of the resources. The actual input tokens, output tokens, each call time, process time and GPU memory are saved item by item; additional prefill for multiple roles to repeatedly read documents must be included. The two budget points constitute the initial quality-cost comparison, which cannot be calculated solely based on the upper limit of the same token.

The development track will save the complete input and output of the planner, evidence_reader, and answerer. After passing the collaboration threshold, role supervision will be generated from the retained training questions; the current development questions can no longer be incorporated into the training and then falsely claimed to have no effect. The special model stage still requires common role LoRA, unified multi-tasking student, and non-LLM program selection and control. There is currently no role delta training.

Code entry:

```sh
.venv-bench/bin/python src/prepare_hotpot_collaboration.py \
  --raw benchmarks/raw/hotpotqa-distractor \
  --output benchmarks/views/hotpot-collaboration-v1

.venv/bin/python -u src/run_hotpot_collaboration.py \
  --root /home/hdd_1/tangziye/idea4 \
  --model /home/hdd_1/tangziye/Model/Qwen3-4B-Instruct-2507 \
  --confirm-if-positive
```

The server still only uses GPU0 and unified GPU lock. The runner freezes four source code files; the preparation job `job-mw542h8m-9518` is currently being downloaded, and the latest coordination job is `job-mw54pcsb-e5d4`, waiting for the actual download process 989786. Among the earlier coordination attempts, one exited due to package PID mismatch, and the other was actively terminated during the waiting stage to complete the same training example comparison; neither GPU evaluation was started. The download was never restarted, and different prompt conditions were not mixed into the results.

It has passed three targeted numerical/semantic checks: only the message path has changed under the same document permissions; voting and budgeting are consistent with the definition; per-question and joint aggregation is consistent with the fixed official original evaluation function. These checks only support implementation semantics and do not count as collaboration benefits.
