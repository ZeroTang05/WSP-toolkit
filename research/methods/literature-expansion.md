# From literature to tools: Extending professional workflows and role models

2026-11-07. This document supports the research sequence “open-source toolkit → 0.6B effectiveness → 4B professional multi-agent workflow.” It explains tool selection, post-training references, actionable research questions, and expected outputs. Candidate interfaces are identified as proposals. See the [three-stage roadmap](../roadmap.md) for the main sequence and the [professional workflow protocol](../protocols/professional-workflow-protocol.md) for system-level controls. This review draws on paper abstracts, official paper pages, and author repositories; candidate methods still require section-by-section alignment with the algorithms and reference code before implementation.

## 1. Organization of research propositions and tool library

The proposed research position is: **Use a language model to operate literature-supported tools for internal observation, parameter editing, and structural compression, constructing interpretable, reusable task-specific and role-specific models from smaller models.**

The professional domain determines which capabilities need to be retained; the method determines which operations are available. Legal, accounting, and document work share the same model-programming language while using their own task data, role interfaces, and quality standards. The toolkit is organized around complementary questions:

| Layer | Core question | Output | Downstream use |
|---|---|---|---|
| Observation | What are the properties of parameters, hidden states, activation distributions, and update spectra? | Statistics and tensors with data provenance | Form candidate modifications |
| Diagnosis | At which sites does intervention change a specified behavior? | Paired intervention effects and failure cases | Narrow the scope of editing or compression |
| Adaptation and combination | How can role capabilities be built and existing capabilities combined? | Weight deltas, adapters, or representation interventions | Improve target behavior |
| Compression | Which parameters, dimensions, layers, or precisions can be reduced? | Masks, structural variants, low-rank or quantized representations | Reduce storage or actual compute costs |
| Compensation and recovery | How can capability loss from modifications be controlled? | Reconstruction results or recovery updates | Preserve role and general capabilities |
| System validation | Do local improvements benefit the whole workflow? | Work-product scores, interaction effects, and quality–cost frontier | Accept or reject a programming hypothesis |

Weight I/O, budgeted forward passes, and variant export provide the shared foundation. The ability to write arbitrary tensors alone does not constitute a complete methodology; methods with different training permissions also require separate tracks.

Each tool should have a method card covering the paper and version, original problem, inputs and outputs, data and gradient requirements, implementation differences, conditions for combination, evaluation metrics, and known failure cases. Tool documentation should label each method as a reference reproduction, declared adaptation, engineering infrastructure, or implementation candidate.

## 2. Pruning and Compression: Why Keep Different Routes

| Literature | Tool operations derived from the paper | Appropriate use in this project | Distinctions to preserve |
|---|---|---|---|
| [Wanda](https://arxiv.org/abs/2306.11695) | Score connections using weight magnitude and input activation norm; select a mask per output row | Build low-cost sparse candidates using role-specific data | Scores are not causal contributions; unstructured zeros do not automatically speed up inference |
| [SparseGPT](https://arxiv.org/abs/2301.00774) | Second-order input statistics, column-wise pruning, and error compensation | Compare deletion alone with compensation after deletion to study quality recovery | Reconstructing layer outputs is a local objective; arbitrary least-squares fitting is not necessarily the original method |
| [LLM-Pruner](https://arxiv.org/abs/2305.11627) | Identify dependency groups, score gradient importance, remove structures, and recover with LoRA | Extract role submodels with genuinely smaller shapes | Requires gradients and dependency graphs; the current MLP adaptation is not the complete reference implementation |
| [ShortGPT](https://arxiv.org/abs/2403.03853) | Observe Block Influence and remove layers | Explore differences in depth redundancy across roles | BI ranks candidates; measure actual layer-removal damage and interactions among removed layers |
| [SVD-LLM](https://arxiv.org/abs/2403.07378) | Activation-statistics-driven low-rank decomposition and post-compression updates | Compare weight approximation with role-output fidelity | Compressing only a LoRA delta does not compress the full base model; a full reproduction must include layer-wise recovery |
| [SliceGPT](https://arxiv.org/abs/2401.15024) | Equivalent coordinate transformations, principal activation directions, and dense dimension slicing | Extend width compression to ordinary dense hardware | Rotation and residual compatibility depend on architecture; mixed architectures require individual validation |
| [AWQ](https://arxiv.org/abs/2306.00978) | Activation-aware protection of salient channels, equivalent scaling, and low-bit representation | Test whether specialized models can further reduce GPU memory and bandwidth | Quantization requires a corresponding inference kernel; simulated quantization error alone is insufficient |

These seven sources cover connection sparsity, compensation, structural dependencies, depth, low rank, width, and numerical precision. Their operations and deployment conditions differ, making them complementary.

Prioritize a complete “compression → compensation → task evaluation” chain using existing Wanda, SparseGPT, depth, and delta-rank operations. Add structural recovery next; place SliceGPT and AWQ in the deployment expansion layer. The first version can publish their method cards and backend requirements, listing them as callable tools only after integration.

## 3. How does classic post-training provide reference for model programming?

Post-training supplies both capabilities and essential effectiveness baselines. The following analogies are limited by their mathematical objects and supervision conditions.

| Reference method | Parameter or state change | Reusable tool ideas | Relationship and limitations |
|---|---|---|---|
| SFT and [LoRA](https://arxiv.org/abs/2106.09685) | Supervised loss produces updates; LoRA represents them as low-rank BA | Construct role updates, select modules, allocate rank, and export effective deltas | Baseline for role capability. Account for the data and training costs of learned directions |
| [Task Arithmetic](https://arxiv.org/abs/2212.04089) | Scale or combine task deltas on the same base | Delta addition and subtraction, interpolation, layer-wise coefficients, and direction comparison | Makes training outputs programmable; linear combination does not guarantee general capability |
| [TIES](https://arxiv.org/abs/2306.01708) | Trim small updates, elect dominant signs, and merge agreeing updates | Conflict statistics and merging | Consider when combining two existing role updates degrades performance; first compare with ordinary averaging |
| [DARE](https://arxiv.org/abs/2311.03099) | Randomly drop and rescale deltas | Update sparsification, random baselines, and preprocessing before merging | Sparse deltas differ from pruning the base and do not directly reduce dense inference compute |
| [DPO](https://arxiv.org/abs/2305.18290) | Adjust relative response probabilities using preference pairs and a reference model | Role preference data, relative-probability diagnosis, and preference updates | Applicable to review quality, citations, or action selection; a mean activation difference between successes and failures does not constitute DPO |
| [Knowledge Distillation](https://arxiv.org/abs/1503.02531) | Use a teacher distribution to guide a student; behavioral-trajectory distillation is a broader form of transfer | Collect teacher behavior, supervise roles, and recover behavior after compression | With text traces alone, call the method sequence or behavioral distillation and do not claim logit matching |
| [EWC](https://arxiv.org/abs/1612.00796) | Constrain parameter movement according to importance for old tasks | Importance estimates and protection penalties on retention sets | A reference for preserving domain capabilities; applying it to LLM role editing requires validation |
| [ReFT](https://arxiv.org/abs/2404.03592) | Freeze the base and learn interventions on latent representations | Persistent representation adaptation, compared with weight adaptation | Differs from temporary activation patching; this research extension should not be described generically as changing model weights |
| [ROME](https://arxiv.org/abs/2202.05262) | Locate and make a low-rank edit to a specific factual association | Targeted fact editing, locality, and paraphrase-generalization evaluation | A factual-editing reference; it does not establish that an entire legal or accounting capability can be injected |

For role updates D_i on the same base, study D = Σ_i α_iD_i against ordinary role LoRA. Combine the effective LoRA updates BA: separately averaging A and B introduces cross terms and depends on factor coordinates.

Capability retention can be a constrained research objective: maximize role or workflow quality while limiting degradation on protected tasks to a prespecified ε. An EWC-style local quadratic penalty Σ_j F_j(Δθ_j)^2 is a proxy, not proof of global retention. Define ε in advance by task and business error type, without adjusting it after seeing results.

Use three tracks: direct weight programming; compression permitting local fitting such as second-order reconstruction; and post-training permitting supervised or preference training. Complete routes may be compared or combined sequentially, with permissions and costs recorded separately.

## 4. Focus the expansion of the tool library on specific operations

The following research interface names are proposals, not implemented APIs. Define their input, output, and literature contracts before coding.

| Candidate operation | Input → output | Source and rationale | Experiment to validate value |
|---|---|---|---|
| `observe_role_distribution` | Role-visible context → activation energy, second-order statistics, spectrum | Wanda/SVD-LLM; determine whether calibration distributions vary by role | Compare general and role-specific calibration at the same compression rate |
| `paired_intervention` | Aligned input pairs, positions, metrics → intervention responses | Activation patching; narrow candidate sites | Compare random sites, incorrect input pairs, and no intervention |
| `build_dependency_groups` | Architecture, module relationships → dependency groups that can be removed safely | LLM-Pruner; prevent invalid structures from isolated slicing | Export and reload an actual structural model; measure role and system quality |
| `compensate_pruning` | Second-order statistics, masks, weights → compensated weights | SparseGPT; reduce local deletion damage | Compare direct zeroing and compensation with the same mask; separately measure reconstruction error and task results |
| `recover_role_adapter` | Pruned model, role data → recovery update | LLM-Pruner/LoRA; complete recovery after compression | Compare no recovery, general-data recovery, and role-data recovery under matched training budgets |
| `merge_role_deltas` | Same-base updates → ordinary/TIES/DARE merged variants | Update arithmetic and conflict handling; study shared capabilities and role conflicts | Compare ordinary averaging, an equal-norm single direction, and multitask training on the same data |
| `compress_role_update` | Effective delta, role statistics, rank → low-rank delta | Declared adaptation inspired by truncated SVD/SVD-LLM | Check whether weight error, activation error, and task quality move together |
| `estimate_retention_cost` | Protection sets, candidate updates → behavioral damage and optional local importance | EWC-inspired; make retention a testable constraint | Measure correlations and counterexamples between proxy rankings and actual damage on independent protection sets |
| `distill_role_policy` | Teacher role context and actions → student role model | Distillation/sub-goal distillation; provide a strong standard baseline | Compare ordinary SFT, behavioral distillation, and updates selected by a program on the same data |
| `evaluate_role_substitution` | Fixed workflow, role variants → single-role/joint substitution matrix | This project's system experiment infrastructure, **not a new published algorithm** | Compare local role scores and end-to-end quality, including protocol changes and error propagation |

Paired intervention follows [Zhang and Nanda's methodological study](https://arxiv.org/abs/2309.16042): perturbations and metrics can change localization conclusions, so explicitly retain the clean/corrupt construction, token alignment, target token, and contrast metrics.

A unified interface does not make all combinations valid: width slicing changes adapter shapes; update merging requires a shared base and coordinates; early-layer edits may invalidate later-layer statistics. Record variant provenance at each step and re-observe when needed.

## 5. Professional workflow: select according to task products and divisibility

| Task source | Features supported by literature or author resources | Role in this project | Proposed roles and key evaluation |
|---|---|---|---|
| [OfficeBench](https://arxiv.org/abs/2407.19056) | Cross-application office work, long-horizon planning, and action execution | Execution bridge; test work state and intermediate-file handoffs | Planning/execution/checking; final documents and application state, omitted steps, redundant actions |
| [Finch](https://arxiv.org/abs/2512.13168v5) | 172 composite workflows, 384 tasks, spreadsheets, and other business materials | Candidate for the main accounting experiment; closer to the objective than isolated financial Q&A | Data alignment/analytical modeling/financial review; formula and unit consistency, numerical consistency, and work-product completeness |
| [LegalAgentBench](https://aclanthology.org/2025.acl-long.116/) | Chinese law, 17 corpus categories, 37 knowledge tools, and intermediate-progress evaluation | Transfer to Chinese legal tool reasoning | Material retrieval/evidence analysis/synthesis; tool success, source support, and answer correctness |
| [Harvey LAB](https://github.com/harveyai/harvey-labs) | Task instructions, documents, rubrics, and an execution-scoring framework | Extension to legal work products and documents | Material review/drafting/review; factual and citation support, clause coverage, and delivered-document quality |

The roles are proposed designs to be tested here; the datasets do not establish that these divisions of labor are effective. LegalAgentBench's tool reasoning does not cover all legal-document production capabilities and should be complemented by work-product tasks.

The selection order is OfficeBench as a bridge → Finch as the main accounting experiment → legal transfer. These stages add cross-application actions, professional numerical and file dependencies, and professional text and evidence constraints, respectively. Existing FinQA/HotpotQA can remain as sub-capability and mechanism controls.

After splitting a workflow into nodes, record each node's input artifacts, visible evidence, allowed tools, output files or schema, upstream and downstream dependencies, and verifiable errors. Accounting source-table extraction, modeling, and review should not share hidden gold answers; legal reviewers should not receive answer clues from test rubrics.

Split data by workspace, company or case, source file, and template lineage; related subtasks must not cross training and validation sets. If no official training set exists, publish named, sourced derived splits or build training materials independently. Use official work-product evaluation as the primary measure, supplemented by role-behavior analysis; custom progress scores cannot replace original task completion rates.

## 6. The relationship between role specialization and existing Agent distillation

[Sub-goal Distillation](https://proceedings.mlr.press/v274/hashemzadeh25a.html) trains planning and execution separately, providing a direct reference for treating role boundaries as knowledge-transfer boundaries. Its task environment differs from the accounting and legal workflows here, so transfer effects still require measurement.

[AgentArk](https://arxiv.org/abs/2602.03955) studies dynamic distillation of multi-agent reasoning into a single model. It is direct related work on incorporating multi-agent benefits into weights, so that concept itself cannot be claimed as original. The testable distinction proposed here is to **retain the division of labor in professional workflows, use literature-based tools to select, localize, combine, and compress each role model, and optimize system-wide quality and cost**. This is a research position; whether it constitutes a novel contribution depends on a more complete related-work review and experimental evidence.

Two comparisons are necessary: ordinary role-behavior distillation, to test whether standard training suffices; and multi-agent-to-single-model distillation, to test the value of retaining multiple deployed roles. The single-model condition must retain the same tools and reasonable task memory, without artificial restrictions.

Domain specialization and role specialization also differ. Beyond legal or accounting knowledge, a role is defined by its input distribution, action permissions, output protocols, and handoff responsibilities. A model can improve at accounting Q&A while becoming less suitable for generating workbooks that downstream roles can reconcile.

## 7. Three-stage research questions, experiments and outputs

| Stage | Question | Core experiment | Deliverables |
|---|---|---|---|
| 1: Open source tool library | Are the tools sourced, complementary, and capable of forming a complete program? | Two tutorials: observation → localization → editing; compression → compensation → recovery → evaluation | Public package, method cards, paper mapping, architecture table, variant format, model-call schema, and real program trace |
| 2: 0.6B effect | Do observations and program choices outperform standard practice? | Base model/ordinary LoRA/fixed tool recipe/random or BO/LLM program; then ablate observations and individual tools | Complete task results, paired improvements and regressions, cross-seed results, method-selection analysis, and quality–efficiency frontier |
| 3A: 4B collaboration | Does division of labor on professional tasks outperform a strong single agent under the same budget? | Single agent, revision, independent candidate selection, multiple agents, with and without role communication | Fixed data and environment versions, collaboration quality and cost curves, and failure taxonomy |
| 3B: 4B parameter programming | Once the system is fixed, can role-parameter modifications provide additional gains? | Original role, ordinary role LoRA, shared multitask update, fixed/non-LLM search, and LLM tool programming | Single-role and joint substitution matrices, handoff compatibility, and replayable role deltas/structural models |
| 3C: Role extraction efficiency | Can role-specific models reduce actual system cost under quality constraints? | Original 4B, role adapter, role structural compression, ordinary distillation, and multi-agent-to-single-model distillation | System-wide quality–GPU memory–latency frontier, construction costs, and cost-recovery analysis |

Only after 3A confirms collaboration gains should the corresponding system become the primary subject of 3B and 3C. Stage 3B first modifies the 4B role's parameters. Further reduction to a smaller model is a separate role-extraction extension and does not change the main requirement for a 4B target model.

Focus the experimental analysis on:

1. **Whether observation helps:** Under matched operation and query budgets, compare programs and final effects with internal observation removed, statistical observation added, and causal intervention added.
2. **Whether proxies are reliable:** Check whether reconstruction error, BI, update spectra, and generation accuracy or work-product quality agree; disclose ranking reversals and failures.
3. **Whether roles have different redundancies:** Derive compression candidates from each role's calibration data, swap them across roles, and compare damage at equal sparsity or model size.
4. **Where merging gains come from:** Include an equal-norm single-direction merge control; do not add local role scores and present the sum as system improvement.
5. **Whether role updates cooperate:** Replace roles i and j separately and jointly, calculating I_ij = Q(S_ij) − Q(S_i) − Q(S_j) + Q(S).
6. **Whether gains reflect more computation:** Match total generated tokens and report elapsed time; list programmer, teacher-data, training, search, and final-execution costs separately.
7. **Which capabilities remain:** Report in-role, adjacent-role, and general protection sets separately; compression must meet prespecified quality floors.

These are hypotheses to test, with no prespecified value for an inevitable improvement. Clear success requires improved quality or lower actual cost at the same quality on independently confirmed data, supported by standard-method controls. Otherwise, report interpretable negative results and tool limits.

## 8. Boundaries between current evidence and next round of content work

[The first public prerelease](../reports/open-source-release.md) already exists. There are 0.6B local reconstruction, paired intervention, and several complete-task results; see the [compensation study](../reports/reconstruction-study.md), [activation intervention](../reports/activation-patching-study.md), and [parameter programming analysis](../reports/programming-expansion.md), respectively. These show measurable effects from the tools. They do not yet establish that the complete new tool chain outperforms ordinary methods, or that 4B professional multi-agent collaboration and parameter programming provide gains.

The next research sequence is to connect role statistics to real failure analysis; connect compression to complete tasks and recovery; add TIES/DARE method cards and alignment with references; establish an ordinary role-distillation baseline; then select SliceGPT/AWQ deployment extensions. Increasing the number of interfaces or repeatedly running small implementation checks cannot substitute for these research outputs.
