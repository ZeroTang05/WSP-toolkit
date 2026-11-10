# Three-stage roadmap and reproduction of the original work

Update date: 2026-11-09. The research has three stages: release a general open-source toolkit, evaluate effects on a 0.6B model, and improve professional multi-agent workflows with a 4B model. For current progress, see [project status](../PROJECT_STATUS.md).

The first stage has integrated 34 tool entries and 47 methods; see [Implementation List](../docs/tool-inventory.md). The second stage has a complete GSM8K method comparison and continues to test gains from autonomous tool selection by a large model. The third phase first verifies the benefits of collaboration, and then modifies and replaces the dedicated models for each role. Reproduction of the original work and comparison of edit directions are also available under [Special protocol](protocols/mit-wsp-reproduction.md).

## 1. Common tools and research division of labor

WSP-toolkit provides independent observation, interpretation, intervention, editing and evaluation capabilities. The model adapter declares supported sites and permissions, and the task configuration declares the data and metrics. A common interface supports legal, accounting, office, and general tasks. The original reproduction continues to use persistent Python, direct weight reading and writing, and permission to run forward passes within a budget.

Each capability is recorded according to a unified tool ID. The common interface, basic interpretability methods, and feature and computational-path analysis share a tool registry. Activation patching, TIES, Wanda, and SparseGPT reuse existing implementations. Subsequent expansions will continue to maintain literature sources, method status, and implementation entries in the same directory. [Tool Design and Merger Route](../docs/tool-design.md#9-existing-implementations-and-integration-roadmap) · [Tool Catalog](../docs/tool-catalog.json)

## 2. Phase 1: Open source tool library that can be used independently

1. Expose an MCP model-calling interface and a directly usable Python interface with shared tool definitions and execution code. First verify capability queries, activation collection, and artifact slicing, then expand the public tools. Provide a support matrix, installation instructions, method foundations, and third-party licenses. [Interface design](../docs/tool-design.md#mcp-and-python-interfaces)
2. Two tutorials are provided: read-only explanation and weight editing. Read-only tutorials cover snapshots, heat maps, conceptual or hierarchical analysis, and ad hoc interventions. Editing tutorials cover existing weight editing or compression algorithms, candidate versions, and paired evaluations.
3. Verify the same interface with real Qwen3 and Llama checkpoints, and use Qwen3.5 to check supported sites in a hybrid architecture. In addition, a real classification encoder is used to verify the applicable scope of the observation interface.
4. Export replayable run records, model versions, and evidence files for reuse in subsequent 0.6B and 4B experiments. [Public release record](reports/open-source-release.md) · [Tool design](../docs/tool-design.md#8-three-delivery-and-experiment-stages)

## 3. Phase 2: Interpretable performance improvement on 0.6B model

The main goal is to test whether the tool library can produce repeatable capability gains; domain-fidelity compression is another goal and is reported separately.

Qwen3-0.6B is the main verification model; a 1.7B model from the same family and models from another family are retained for trend comparisons. Formal results use full public partitions with independent seeds. The base model/standard LoRA/fixed scaling result of 485→532→590/1319 under the existing unified protocol is the starting point, which does not mean that LLM programmer is better than ordinary methods.

It is recommended to separate three experimental questions:

| experiment| What to fix| What to compare|
|---|---|---|
| Tool role| Data, basic models, direction sources and budgets| Base model, standard LoRA, single tool, tool combination|
| Observation value| Optional operations, query/training budget| No internal observations, statistical observations, causal intervention observations|
| Programmer Value| Same tools, data, permissions, total cost| Fixed formula, random/BO, LLM selection|

GSM8K is used for reviewable reasoning effects, and ARC-C/HellaSwag is used for cross-task impairment; professional sub-capabilities can also be evaluated, but high scores on professional multiple-choice questions cannot be called workflow competence. Report accuracy, per-question improvement/degradation, output interface, actual training/search cost; compression also reports quality-storage-latency frontier.

Clearly distinguish between gradient-free updates, gradient observations, and experiments that allow training. When using a trained direction, the training source and cost of that direction must also be taken into account; "Searching without training" does not mean that the complete route has no training.

Phase output: reproducible experimental procedures, real effects/failure cases, single tool and combination ablation, cross-scale/series difference analysis, and downloadable adapters/selectors and complete program records.

## 4. The third stage: Parameter programming of 4B model in professional multi-Agent workflow

The target model being modified at this stage is 4B. The programmer `M_programmer` is recorded separately from the target `M_target=4B`. The 4B model is the target, rather than merely a teacher whose effect is measured through smaller students. The programmer can be a stronger model, but its construction cost must be accounted for separately.

First compare the strong single-Agent, multi-step revision/independent sampling and multi-Agent workflows of the same 4B; after establishing the collaboration benefits, fix the topology, tool permissions, file representation and budget. Then modify the 4B parameters for each role, represent each as the shared base plus a role delta, and test single-role and joint replacement. Role specialization is an optimization method at this stage and should not replace the verification of the entire 4B workflow.

Must compare: Unedited 4B workflow, common role LoRA, unified multi-tasking adapter, non-LLM same tool search and LLM parameter program. Teacher trajectories, additional training data, prompt changes, and tool backend changes must be controlled separately. If the RAG or file parsing is done better, it should be recorded as an environment/input improvement and cannot be counted as a parameter programming benefit.

### 4.1 Professional Workflow Candidates and Selection

| Data set| What it evaluates| Relationship to this project| Integration requirements|
|---|---|---|---|
| [Finch／FinWorkBench](https://arxiv.org/abs/2512.13168) | 172 accounting portfolio processes, 384 tasks, professional work across forms and documents| **Main candidate for professional complexity**; Suitable for data compilation, financial analysis, and results review roles| [Official pipeline](https://github.com/FinWorkBench/Finch) includes file preprocessing and judge; Excel/multimodal processing, evaluation dependency and human scoring alignment need to be verified|
| [OfficeBench](https://github.com/zlwang-cs/OfficeBench) | 300 cross-application office tasks, matching and execution evaluation| **First integration and method-development bridge**; facilitates test planning, execution, and cross-application collaboration| Pin application state and action interfaces, retaining official scoring; tool output cannot be secretly generated by stronger models|
| [LegalAgentBench](https://github.com/CSHaitao/LegalAgentBench) | 300 tasks in Chinese law, 17 types of corpora, 37 external knowledge tools, including process indicators| **Main candidate for Chinese legal transfer**; research retrieval, evidence concatenation, writing role| Verify one by one whether the corpus and tool services can be reproduced locally, as well as the intermediate supervision required for scoring|
| [Harvey LAB](https://github.com/harveyai/harvey-labs) | Produce legal work results based on case materials, including tasks and execution/scoring framework| **Extension to difficult legal work products**; more in line with professional work goals than legal knowledge questions and answers| Fixed release. The current [official score](https://github.com/harveyai/harvey-labs/blob/main/docs/eval-strategies.md) is based on rubric and LLM judgment and requires expert spot checks and referee consistency analysis|
| [TheAgentCompany](https://github.com/TheAgentCompany/TheAgentCompany) | 175 task snapshots of a simulated company, covering office and workplace collaboration| For cross-role/state interaction extensions| Environment containers, simulated colleagues, and the environment LLM have substantial costs; the environment model cannot be regarded as a free auxiliary capability.|
| [GDPval](https://openai.com/index/gdpval/) | There are 220 public gold-set tasks, covering work products from 44 occupations.| External results quality inspection| The first version is a one-shot delivery evaluation, not a native interactive workflow; if multiple agents are constructed, it is marked as the system design of this project|
| [APEX-Agents](https://www.mercor.com/blog/introducing-apex-agents/) | Long-term professional assignments and workspaces in investment banking, consulting and corporate law| Candidates for high-difficulty comprehensive migration| First check the fixed version and repeatable deployment conditions of the task, environment and rubric before deciding to invest.|

It is recommended to use the sequence of "OfficeBench to connect execution and attribution → Finch to do the main accounting experiment → LegalAgentBench or Harvey to do the legal migration". It is not required to run seven complete systems at the same time. The order here is based on the research judgment of task form and current implementation, not the actual measurement conclusion that these tasks are suitable for 4B or multiple agents.

HotpotQA is relegated to general mechanism comparison; FinQA is reserved for financial reasoning and existing failure analysis. They are not a substitute for the professional work product experiments described above.

### 4.2 Several necessary differences in professional evaluation

Workflow datasets do not automatically prove that multiple agents are better. Tool implementation failures, professional reasoning errors, formatting issues, missing results, and referee disagreements should be counted separately. The completion rate, key constraint pass rate, fact/number/reference correctness and actual time consumption need to be retained; the rubric average score and the entire task all-pass are reported separately.

The 4B text model processes tables, PDFs or images and requires fixed and public parsing/execution tools. If a visual model is used as an aid, it must be clear whether its role, cost and all conditions are the same. If only some tasks can be supported, the task scope must be declared in advance according to the input capability and the coverage rate shall be reported; failed tasks cannot be excluded after evaluation, and a subset score cannot be presented as the full benchmark.

When there is no training partition for a public task, priority is given to creating an independent development/training task, and then retaining the public set for confirmation. If you must divide it yourself, group it by workspace, source file lineage, case/company, and clearly name it as derived split; you cannot put dependent tasks under the same workflow into training and testing respectively, and you cannot let the expected result file enter the performer input.

If all 4B conditions in a complex process are close to zero, first determine whether the environment and output interfaces are available, as well as task capacity bottlenecks; retain the failure results and do not manufacture gains through hidden simplification or extra work by a stronger model.

## 5. Interpretation criteria for tool selection

When a candidate tool enters the core library, it should be able to form a complete chain:

> Observable failures/resource issues at work → Corresponding assumptions in the literature → Available statistics/supervision on this model → Valid updates → Falsifiable performance indicators.

For example: the role output protocol is unstable, look at role supervision and adaptation first; combined updates cancel each other out, study update arithmetic and merge interference; GPU memory/latency is too high, study rank and structure compression; ability after deletion decreases, study compensation and recovery; the attribution of a certain behavior is unclear, do intervention diagnosis first. They each have their own uses, and "many tools" cannot replace complementarity in methods.

The first stage proves that the tool library is well-founded and usable; the second stage proves that it produces interpretable real effects on small models; and the third stage proves that these operations can improve 4B professional workflow. The three phases each have deliverable results, and are ultimately connected into the same main line of research.

## 6. Final stage: reproduction of the original effect and comparison of direction

The target model is individually fixed to the original Qwen3.5-0.8B and cannot be replaced by the second-stage Qwen3-0.6B. First calibrate the reference replay using the author's delta, and then let the new isolated programmer analyze independently from only the base model, practice data and common tools. The current session has already read the reference solution, and its subsequent edits cannot be counted as blind discovery.

The training-free main track freezes templates, sampling, budgeting, and scoring; complete math tasks are tested only after final edit freeze. After the effect reaches the predefined standard, the layer/column support, parameter direction, subspace, functional changes on common inputs and causal intervention are compared between the two. Replay of the original work, independent discovery and comparison of directions are delivered separately. You cannot just complete the easier one and call the reproduction successful. For specific original sources, filled primitives and gaps, see [mit-wsp-reproduction.md](protocols/mit-wsp-reproduction.md).

2026-11-07 Progress: [Real observation interface](reports/forward-service.md) and [SparseGPT compensation experiment](reports/reconstruction-study.md) have been implemented; [Professional workflow protocol](protocols/professional-workflow-protocol.md) further defines collaboration baseline, role replacement and error propagation. No professional collaboration or parametric programming benefits have been established under this protocol; public release records are provided below.

2026-11-07 Public milestone: [Source repository](https://github.com/ZeroTang05/WSP-toolkit) and [0.4.0a4.dev0 pre-release](https://github.com/ZeroTang05/WSP-toolkit/releases/tag/v0.4.0a4.dev0). Contains direct parameter tools, budget observations, persistent CPU kernel, paired intervention and SparseGPT compensation; the research interface is different from the core API. Initial disclosure does not imply that all candidate methods, full post-training recovery, or all research effects have been completed.

For content expansion, see [From literature to tool](methods/literature-expansion.md): further explanation of the complementarity of seven compression routes, parameter reference for classical post-training, role distillation related work and clear experimental output.
