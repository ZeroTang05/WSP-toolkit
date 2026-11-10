# Professional Workflow: Collaboration Gains, Separation of Role Programming and Tool Function

2026-11-07. Research protocol, there are no professional assignment results under this protocol. The scope, sources and limitations of the data set can be found in [Main Route](../roadmap.md#41-professional-workflow-candidates-and-selection).

## Research object

The proposed chain is: feasible division of labor for professional tasks → collaboration gains in a fixed multi-agent system → further gains from role-specific parameter operations → lower actual cost through compression under explicit quality constraints. Each arrow requires independent experimentation and cannot be directly inferred from the "seeming complexity" of specialized tasks.

4B is the target base for each role in the third stage; the strong programmer is responsible for offline diagnosis, selecting operations and proposing programs. The programmer is absent from task execution during the final test. If the stronger model still plans, reviews, or takes over online, it belongs to another hybrid system condition, and its tokens, delays and fees are all included.

## Data and task gradient

OfficeBench is used for cross-application execution and intermediate file transfer; Finch is used for form/document-driven accounting work; legal migration starts by relying on a reproducible set in LegalAgentBench and Harvey LAB. Task volume, professional depth and evaluation reliability are on different axes and cannot be selected based solely on the number of questions.

| stage tasks| final result/intermediate state| Possible division of labor hypothesis| First difficulty| Intended tool family|
|---|---|---|---|---|
| Work across applications| Documents, forms, emails/application status| Planning, application execution, results verification| File status and action protocols, missing steps| Observation, role LoRA, layer intervention|
| Accounting work| Traceable numerical analysis, workbooks, reports| Source table alignment, computational modeling, review and delivery| Units, formulas, cross-table references, numerical and text consistency| Role adaptation, updated combination, causal positioning, compensation|
| legal work| Search/Analysis/Documentation with Sources| Material compilation, evidence and legal analysis, writing review| Citation support, fact constraints, long context, multi-material synthesis| Observation, role adaptation, intervention diagnosis; independent research by fact editor|

These roles are proposed system designs, not official divisions of labor established by each dataset. Test two-role and three-role configurations first; do not assume that more agents are better.

Group public tasks by source workspace/case/company/template lineage. Document lineages for development, role training, and final confirmation cannot overlap. If there is no official training set, create another development task or publish a clearly named derived split; the final report covers the number of tasks, elimination rules, and uncovered input types. The text-only 4B model uses a fixed parser to avoid using visual aids of different strengths under different conditions.

## Experiment A: Does collaboration really work?

| condition| Models and budgets| Solved attribution issues|
|---|---|---|
| Strong single agent| Same as 4B, same environment/tools, clear plan and task memory| Prevent weak-prompt baselines from exaggerating collaboration gains|
| Single Agent revision| Same as 4B, self-inspection and revision, total budget matching| Does division of labor outperform multiple rounds of reasoning?|
| Independent sampling + selection| Same as 4B, multiple candidates and fixed selector, total budget matching| Is division of labor better than parallel exploration?|
| Multiple Agents on the same base| Same as 4B, only the role prompts differ| Pure topology and role-prompt gains|

Jointly report quality under a fixed resource budget and the cost to achieve fixed quality. The length of the workflow is different, so you cannot just match the number of calls; at least provide the total generated token constraints and the real wall clock/device resident cost to explain the degree of parallelism. The selector or reviewer does not see answer keys or hidden rubric feedback.

Evaluation is based on official completion/result scores. For paired comparisons of the same task, the confidence intervals are clustered according to independent workspaces, and related subtasks or multiple samplings are not regarded as independent samples. Predefine the minimum meaningful improvement, combined with interval and cost frontier judgments. Only if development shows a collaboration gain and confirmation supports it will the system become the main target for role extraction; the other systems will be retained as negative results.

## Experiment B: Where does parameter programming gain come from?

Fixed selected topology, role prompts, input tools, and stopping rules. The comparisons are as follows: original 4B role; ordinary role SFT/LoRA; shared multi-task LoRA; same tool fixed formula/random or BO search; LLM uses tool library to propose programs. All conditions share available training materials and record additional teacher trajectories and query costs.

Role training material should contain the context actually visible to the role, its actions and outputs, and verifiable feedback. Qualifying the final product does not mean that each intermediate step is qualified; the entire success trajectory cannot be accepted as a positive example without review. Nor do model-generated think texts automatically constitute trustworthy internal reasoning oversight.

Do single role replacement first, then joint replacement. Define system quality Q, unedited system S and replacement S_i of role i:

    Single role effect Δ_i = Q(S_i) − Q(S)
    Two roles interact I_ij = Q(S_ij) − Q(S_i) − Q(S_j) + Q(S)

Interactions are derived from system experiments and are not inferred from weight cosine. Individual role scores improve but system degrades, possibly from file protocol changes, review distribution mismatch, or error propagation; test original upstream input and real new upstream input simultaneously. A sum of local scores does not establish success when joint replacement yields no system gain.

## What evidence is needed for tool use?

| Observation problem| Reasonable operation and starting point of thesis| alternative explanations that must be verified|
|---|---|---|
| Role action or output protocol is unstable| LoRA role adaptation; consider DPO if you have reliable preferences| Ordinary post-training on the same data is sufficient, and there is no gain in the LLM selection tool|
| The two types of updates harm each other after being combined| Task Arithmetic; TIES symbol conflict handling| Impairment comes from cue/training data differences, non-parametric conflicts|
| The specific failure is related to an internal component| Layer ablation, paired activation patching| Intervention destroys general language ability, or just outliers|
| Redundant occupied storage/computing| Wanda、LLM-Pruner、ShortGPT、SVD-LLM | Sparse zero values ​​or low rank representations do not translate into actual operational gains|
| The role's capabilities are damaged after the cut| SparseGPT local compensation; LLM-Pruner style LoRA recovery| Only the calibration reconstruction error is reduced, the complete work results are not improved|

Source: [LoRA](https://arxiv.org/abs/2106.09685), [DPO](https://arxiv.org/abs/2305.18290), [Task Arithmetic](https://arxiv.org/abs/2212.04089), [TIES](https://arxiv.org/abs/2306.01708), [activation patching](https://arxiv.org/abs/2309.16042),[Wanda](https://arxiv.org/abs/2306.11695),[LLM-Pruner](https://arxiv.org/abs/2305.11627),[ShortGPT](https://arxiv.org/abs/2403.03853),[SVD-LLM](https://arxiv.org/abs/2403.07378),[SparseGPT](https://arxiv.org/abs/2301.00774v3). These sources respectively support the algorithmic motivation of the tools and do not support the conclusion that they are already effective in the above-mentioned professional systems.

## Experiment C: Role Extraction and Efficiency

The role-specific model starts with the 4B base + role delta, and the compression ratio and performance changes are drawn separately. It is necessary to distinguish: only save multiple small deltas to save disk; share the base and switch roles to save resident weight; only by making the structure smaller or applying sparse kernels can the computing power be directly reduced. The three cannot be collectively called "the model has been accelerated".

Report by full workflow: official quality, key constraints all-pass, fact/number/citation errors, failed steps, generated tokens, tool time, model time, end-to-end latency, peak memory, disk and offline construction costs. Compression variants below a pre-specified lower quality limit do not count towards effective efficiency gains. Insufficient budget or performance degradation are also expected deliverables.

The first phase delivers tools and method cards; the second phase delivers 0.6B single tool/combination/programmer analysis; the third phase delivers workflow adapters, collaboration controls, role replacement matrices, quality cost frontiers and replayable role model programs. The professional workflow adopts reviewable numerical values/document inspection priority; when LLM refereeing is required, the rubric version, referee configuration, repeated scoring differences and expert spot checks are retained.
