# 4B Professional Workflow Execution and Collaboration Comparison

2026-11-07. The executor has been implemented and started real tasks; full comparison results are not yet available.

## Frozen comparison conditions

The five conditions share the same unedited Qwen3-4B-Instruct-2507, tool interface, initial file, memory rules and resource limit:

| condition| Model calls per decision| Allocation of 384-token decision-making upper limit|
|---|---|---|
| single | Briefly analyze and choose one action| 384 |
| self_refine | Draft an action, then check and revise| 192＋192 |
| independent_select | Generate two independent candidates, then select one with the base model| 128＋128＋128 |
| collaborate | The planning agent proposes; the execution agent checks and acts| 128＋256 |
| no_communication | The planning agent is also called, but the message is not handed over to the execution agent.| 128＋256 |

Each task allows up to 20 decisions and 8,192 generated tokens in total. What is matched here is the upper limit and does not claim that the actual tokens, input calculations or FLOPs are the same; the actual input/output tokens, model time, end-to-end time and number of calls are recorded one by one. Planning calls within message ablation are still billed.

All conditions retain the last 8 actions/observations, each observation can be up to 3000 characters and marked with truncation; the complete original output remains in the trace. The question, user time context, and action directory are always visible; the model does not receive evaluation fields, expected file paths, or scoring feedback. External LLM applications and networks are not available. Scoring is done by a standalone container running the official helper after execution terminates.

`src/officebench_workflow.py` implements the methods and budgets; `src/run_officebench_workflow.py` freezes the source code, verifies 675 source files, loads the model and saves the model fingerprint and call-by-call prompts. The structured role interface reserves space for subsequent delta replacements of the same base role; no roles have been trained or edited this round.

## Current execution

On the three-application task 3-1/0, single and collaborative conditions run in CPU FP32 at full budget to avoid crowding out the existing GPU0 queue. The task requires reading project forms, generating PDFs and forming simulated emails. Job `job-mw5jt7i0-e7ef` has actually loaded the 4B model and generated planning/action trajectories, and table reading and local mailing list actions have been executed. This is a clearly designated public diagnosis case, not a complete benchmark confirmation; the advantages of collaboration cannot be judged from one case.

Method order is rotated according to the task ID hash and is not arranged based on the result. The current process is still running and will not be restarted. Formal comparison of GPU BF16 requires a separate frozen runtime and cannot be strictly paired directly with CPU FP32 scores.

## New evidence for data partitioning

Connecting the 167 workspaces when any nonempty testbed file has an identical SHA256 yields 62 connected groups. The largest group contains 69 tasks; other groups contain 22, 21, 15, and 14 tasks. Duplicate files occur 169 times. `research/acquisition/analyze_officebench_lineage.py` can be rerun, and the results and hashes of each file are in `artifacts/toolkit/officebench-source-lineage.json`.

This is only the minimum constraint for partitioning: there are many more tasks that may be connected between similar templates and near-duplicate files. The random division of question numbers cannot be called an independent role generalization experiment. The current case is not used for role training; subsequent development/validation must be grouped by file lineage and explicitly named derived split or use independent development tasks. The accounting master experiment and legal migration are still progressing according to [professional workflow plan](professional-workflow-protocol.md).
