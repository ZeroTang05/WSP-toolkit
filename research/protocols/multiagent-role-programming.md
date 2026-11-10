# Verify collaboration first, then construct role-specific models using parametric programming

2026-11-06. This research direction changes the order of the application work: first establish a task benefit from multi-agent collaboration, then extract or construct small models that can replace individual roles, and finally test end-to-end quality and efficiency. Existing weight tools and cross-model analyses continue; existing role training does not establish a collaboration benefit.

The implemented first-stage protocol is in [collaboration-benchmark.md](collaboration-benchmark.md). It compares five HotpotQA conditions at two budgets and three inference seeds on training-derived development data, then confirms on the complete public dev set after freezing a positive development result. Data preparation and the subsequent coordination job have been submitted; no collaboration quality results are available yet.

## 1. Core assumptions and research objects

The research chain is:

```mermaid
flowchart LR
    A[Single Agent and equal-budget multiple inferences] --> B[Verification of multi-Agent collaboration benefits]
    B --> C[Freeze role interface and workflow]
    C --> D[Large model observes and programs small models]
    D --> E[Role-by-role and joint replacement]
    E --> F [End-to-end quality and cost frontier]
```

The technical framing is **parameter specialization and compression of role-conditioned policies**. It does not assume that there is a ready-made "retrieval neuron" in the base that can be cut out without damage. Role boundaries first come from the input, output, tool permissions and responsibilities of the workflow; then role behavior data and intervention measurements are used to determine parameter updates and deletable structures.

Three hypotheses need to be tested separately:

- H1: After fixing the model capabilities, available information and tool conditions, the collaborative structure is better than the budget-quality frontier of strong single agent/independent multi-sampling.
- H2: For roles that cannot be adequately replaced by ordinary small models, better replacements can be obtained through role supervision and parameter programming.
- H3: In addition to common role LoRA, unified multi-task adaptation and non-LLM search, LLM programmers bring reproducible additional quality or efficiency gains.

Establishing H1 does not imply H2 or H3. The small model ultimately maintains the quality of the teacher system and reduces the cost, which is also a valuable result; "further improving quality and improving efficiency" is a stronger goal that requires actual Pareto dominance evidence.

## 2. Directly related work and differential positioning

| paper| Existing methods and what this study should learn from| This study must additionally answer|
|---|---|---|
| [Multiagent Debate，ICML 2024](https://arxiv.org/abs/2305.14325) | Multiple instances propose answers, communicate, and revise to provide a starting point for collaborative assessment| Compared with independent sampling/single-agent revision with the same budget, is there any benefit from the exchange itself?|
| [Rethinking the Bounds，ACL 2024](https://aclanthology.org/2024.acl-long.331/) | In its comparison, a single agent with strong prompts can approach the best discussion effect| A single weak baseline is not enough to prove the advantages of collaboration, and a strong single Agent must be added|
| [MAGDi，ICML 2024](https://proceedings.mlr.press/v235/chen24ah.html) | Use multi-Agent interaction graphs to perform structural distillation on small models| What is the value of a replacement that preserves the role interface versus distilling it into a single model?|
| [MoRAgent，ICML 2025](https://proceedings.mlr.press/v267/han25j.html) | reasoner/executor/summarizer three roles LoRA and role data generation| Shared bases or role LoRA cannot be considered original contributions|
| [MapCoder-Lite，EACL 2026](https://aclanthology.org/2026.findings-eacl.346/) | Success trajectory, supervision and error correction, independent LoRA for each Agent; Section 5.3 Shared Freezing 7B Base| Parameter programming should go beyond ordinary role fine-tuning under the same role data, training costs and interfaces|
| [Think Big, Search Small, 2026 preprint](https://arxiv.org/abs/2607.07548)| Distinguish task delegation, execution, fixed answer generation, and study role capacity asymmetry| Role replacement sensitivity should be tested first before deciding to reduce the budget; it cannot be assumed that every role can be reduced equally|
| [MAS-OPD, 2026 preprint](https://arxiv.org/abs/2609.34234)| Teacher supervision, role strengths and collaborative attributions on student trajectories| The students’ state distribution changes after they take over, and pure teacher trajectory fine-tuning may not be enough.|
| [Chain-of-Agents, 2025 preprint](https://arxiv.org/abs/2508.13167)| Training multi-actor reasoning within a single model from multi-Agent trajectories| Compare with a single model trained to absorb the workflow, rather than assuming that multiple roles must be retained.|

The full text of MapCoder-Lite has been checked: shared base + agent-wise LoRA is a direct precedent. For recent papers such as MAS-OPD, the abstracts and method sections have been checked, and the source code, training permissions and budget need to be checked item by item before subsequent implementation; they will not be written as reproduced methods.

The differences that can be studied are: **Use each role’s end-to-end marginal contribution as a constraint, and let a large model select literature-driven parameter programs, structures, and resource allocations to construct replaceable specialized small models.** This is still a candidate contribution and must be further compared with the full methodological and measured controls of the above work.

## 3. The first stage: Verify that multi-Agent is indeed useful

### 3.1 Task selection

The priority candidate is offline-reproducible multi-hop question answering, with planning/decomposition, evidence execution/extraction, and answer synthesis roles, following the role-capacity literature. Use [HotpotQA official resource](https://hotpotqa.github.io/)'s public data, fixed documents, and official answer/supporting fact scores as a starting point. First make clearly named distractor reading conditions, and then independently expand the search conditions; search within candidate paragraphs cannot be called fullwiki search. When public dev serves as the final reporting partition, construction, workflow selection, and parameter tuning must use a separate development split derived from train. Hidden-test results cannot be claimed when local labels are unavailable.

This selection is based on clearer role boundaries and evidence indicators, rather than on the knowledge that local models will benefit. Code generation can be done as a secondary task, refer to MapCoder-Lite and use public training/hidden testing boundaries. FinQA retains existing experiments, but currently cannot be automatically considered to pass H1.

The existing specialized FinQA role V3 with seed 11 scores 425/1147, but changes training, the role workflow, and output format. The untrained single-agent baseline rarely produces valid DSL on dev, so its difference from the trained-role setup includes gains from format compliance and supervision. It does not by itself establish a multi-agent collaboration benefit. The missing V3 header is another confound. The new stage requires a valid baseline with the same input representation and the same output interface.

### 3.2 First stage comparison

| Condition | Held constant | What it isolates |
|---|---|---|
| Strong single agent with prompting| Same models, tools, available evidence| Direct problem-solving ability|
| Single-Agent multi-step reflection/revision| Same tools and total resource limit| Benefit from multiple reasoning rounds|
| Independent multi-sampling + fixed aggregation| Candidate count and resource limit| Sampling diversity and voting gains|
| Same model, role-based, no message exchange| Role prompts and number of calls| The contribution of the role prompt itself|
| Same model, role specialization and communication| Same evidence and tool permission boundaries| Collaborative message contributions|
| Scramble/remove communications| Communication budget; avoid leaking labels| Exploratory diagnosis of whether messages are used effectively|

Multiple contexts of a model can form logical multi-Agents, which may not require multiple physical copies; whether they are deployed in parallel is reported as an execution condition. Compare multiple budget points, reporting at least total input/output tokens, calls, tool operations, actual latency, and peak memory. The same upper limit does not mean the same actual computational cost, and it is not called equal cost.

First use a training-derived development set to select the workflow and budget, and then evaluate the complete public partition after freezing. Define the collaborative benefits of the same budget b

\[
\Delta_{\rm collab}(b)=Q(G_T;b)-\max_{s\in\mathcal S}Q(s;b),
\]

Here \(\mathcal S\) contains strong single-agent, revision, and independent-sampling baselines. The maximum denotes a strong baseline selected on development data in advance; the most favorable opponent cannot be selected after testing. Use question pairing and independent run-seed analysis for the fixed comparisons, and use budget curves to confirm that the benefits don't just come from more calculations.

**Stage Threshold:** For pre-frozen main budgets and comparisons, the collaboration benefits are positive and the lower bound of the pairing interval is greater than 0; absolute effect sizes and other budget results are also reported. If the specific minimum practical gain needs to be adopted, it must be registered before new confirmation. If it fails, study the task/workflow fit first or accept the negative result. Do not continue to package the scenario as a successful system and then conduct role extraction.

## 4. The second stage: Responsibilities of large model programming and small models

After the teacher system \(G_T\) passes the first stage, freeze its topology, role definitions, and interfaces. The large model can serve in two different roles here: as a teacher who provides behavioral supervision, and as a model programmer who operates the tool. The two contributions must be separated.

The student role can be written as

\[
\pi_r^{(s)}=f_{\theta_s+\Delta_r,\mathcal G_r},\quad
(\Delta_r,\mathcal G_r)=\mathrm{Execute}(p_r;\theta_s).
\]

\(p_r\) is a sourced parameter program, including observation, training/editing, compression, and selection; it is not an arbitrary floating point weight list output by the teacher. The parameter coordinates of teachers and students are usually different, and the delta of the large model cannot be directly added to the small model.

The tool chain is divided into three parts:

1. **Generate role capabilities.** Split real teacher-workflow trajectories by role. Ordinary LoRA SFT is a required reference; the data only comes from training problems. Successful trajectories alone may omit recovery behavior; retain labeled student failures and teacher corrections with their sources. One cannot claim to achieve complete KL distillation without teacher logits.
2. **Observation and programming.** Test activation, update spectra, gradient-related importance, and layer-by-layer deletion responses on students; choose update scaling/combination, rank compression, Wanda, or valid structural pruning. For the local targets and sources provided by the tool, see `programming-expansion.md` and `method-foundations.md`.
3. **Workflow feedback.** Integrate candidate roles into the complete workflow and measure final success and actual cost on training and development data. This role cannot simply minimize reference-text NLL: existing BO programs show that a proxy optimum does not guarantee optimal generation.

Large model program search and fixed recipe/random/BO use the same role data, tool permissions, training and evaluation budgets. If LLM also generates additional training examples, the non-LLM strategy should use the same data, or cross-ablate "data generation" and "parameter decision-making"; otherwise, the benefits cannot be located.

It is not assumed that every role should be equally small: test sensitivity first and allow for a larger planning role and a smaller execution role. Roles across model scales cannot share the same tensor storage on different bases; only roles on the same base apply the \(B+\sum D_r\) formula.

## 5. The third stage: role-by-role replacement, interaction and deployment

Define real replacement payoff for role r

\[
g_r=Q(G_T[r\leftarrow\pi_r^{(s)}])-Q(G_T),
\]

and the actual cost change of replacement. Here the entire workflow is re-enacted and students cannot be assessed solely on the perfect context given by the teacher.

Replace two roles jointly

\[
I_{rs}=Q(G_{r,s})-Q(G_r)-Q(G_s)+Q(G_T).
\]

It characterizes joint replacement interactions; while a single actor is competent, the jointly replaced system may still fail due to changes in state distribution. With three roles, all 8 replacement subsets can be tested, which is better at locating bottlenecks than just testing "all teachers/all students". Role-fixed context offline scoring is diagnostic only; the real context after the student takes over is the primary result.

Finally, at least compare: full teacher system, all unspecialized small models, unified multi-task students, ordinary role LoRA students, parametric programming role students, and feasible single model distillation of the entire workflow. Another comparison uses identical student weights to compare shared bases and independent replicas, only isolating the deployment method.

In the compression stage, the acceptable quality loss \(\epsilon\) is first registered, and the cost is minimized after constraining \(Q(G_s)\ge Q(G_T)-\epsilon\); maintaining quality and further improving quality are reported separately. Efficiency includes disk, peak memory, total tokens, first token/end-to-end delay, and throughput; single-GPU serial sharing and multi-GPU concurrency cannot be mixed into the same delay condition. If independent copies can be parallelized, GPU memory may be traded for latency.

Include offline teacher, programming, and training costs. If the new system saves \(c_T-c_s>0\) each time, the rough amortized number of calls is \(N^*=C_{offline}/(c_T-c_s)\), where the cost must be calculated separately in the same unit (time, energy consumption or expense). Just because the storage becomes smaller does not mean that the denominator is positive.

## 6. Expected output and current status

| Output | Contents | Current status |
|---|---|---|
| Collaboration Benchmark| Complete public tasks, strong single agent/multi-sampling, multiple budget points| New research protocol; not yet passed stage thresholds|
| Role interface and trajectory set| Role input/output/permissions, failure recovery, training sources| Need to be constructed on the workflow that passes the threshold|
| Parameter library| Paper → Observation → Operation → Model Effect, including training and compression| Existing implementations and new combination/rank/depth operations|
| Role replacement map| Single role, joint replacement, input distribution drift, failed delivery| To be carried out after the establishment of the first phase|
| Programmer contribution experiments| Comparison of LoRA/non-LLM/LLM with the same data and tool budget| No evidence of superiority of LLM yet|
| Deployment Quality—Cost Frontier| Sharing/copying, structure compression, real latency and amortization| There is only existing evidence of small-scale deployment, which does not mean that the new direction has been completed.|

This direction and the existing four-model general parameter experiments support each other: the former supplies role constraints and end-to-end value, while the latter supplies interpretable program effects and cross-model differences. The main success criteria for subsequent applications adopt the order of this article, and no longer train three deltas first instead of verifying the effectiveness of collaboration.

## 7. From parameter tools to workflow: conditional quality bridge

In sequence learning, the previous output changes the subsequent input, and the single-step supervision error cannot be directly interpreted as system quality; this issue is the starting point of [DAgger/Ross et al. 2011](https://proceedings.mlr.press/v15/ross11a.html) research. The following uses standard probabilistic coupling/KL chain rule to organize the replacement constraints of this project. We do not advocate new theorems, nor do we claim to have obtained usable numerical certificates.

Assume that the workflow has at most H role calls, the history is h_t, and the roles are determined by the same scheduling rule r(h_t). The teacher and the replacement system have the same problem distribution, tool/environment transfer and termination conventions; after early termination, pad with absorbing empty actions. The role output a_t is defined in the common message/action space, rather than the native token number of different tokenizers. Let the complete trajectory distribution of the two systems be P_T, P_S, and the task success function u(τ)∈[0,1].

Define the per-call differences on teacher histories

\[
e_t(h)=\mathrm{TV}(\pi^T_{r(h)}(\cdot\mid h),\pi^S_{r(h)}(\cdot\mid h)).
\]

Replace conditional distributions one call at a time. Total variation cannot increase under the same subsequent transitions, so

\[
|Q_T-Q_S|\le\mathrm{TV}(P_T,P_S)
\le\sum_{t=1}^{H}\mathbb E_{h_t\sim P_T}e_t(h_t).
\]

The proof constructs hybrid trajectory distributions with teacher actions for the first t steps and student actions thereafter. Adjacent distributions differ at one call; the triangle inequality gives the bound. This bound uses the sum of H calling errors on the expected final probability of success; it is not an H² bound on the cumulative step loss, nor is it the sum of each role's local accuracy.

If the teacher's conditional distribution is absolutely continuous with respect to the student's, the chain rule and Pinsker's inequality can also be used to obtain another bound:

\[
|Q_T-Q_S|\le
\sqrt{\frac12\sum_t\mathbb E_{P_T}
\mathrm{KL}(\pi^T_{r(h_t)}\|\pi^S_{r(h_t)})}.
\]

KL uses a complete conditional message distribution, including termination, which is not equal to the cross-entropy of a few reference texts; without a complete teacher distribution or reliable coverage, SFT loss cannot be substituted into this formula. Multiple calls to the same role need to be weighted by the number of visits, and the number of role occurrences and historical distribution may also depend on previous behavior.

In the first stage, if the true advantage of a relatively strong single agent is established, m=Q_T−Q_single>0, then the upper bound of the substitution error is smaller than m, which is a sufficient condition for retaining the advantage. This explains why there is an advantage in testing collaboration first: it provides a reference for allowable damage for subsequent specialization. With limited data, uncertainty in both m and the error must be assessed; point estimates are not certificates.

A corresponding resource-allocation problem selects a parameter program p_r for each role, minimizes C(G,{p_r}), constrains estimated overall workflow quality, and uses the bound above as a conservative diagnostic. It does not require that the roles have the same rank or the same compression rate; nor does it require that the more the imitation is like the teacher, the better, because the student may correct the teacher's mistakes. In the end, the task benefits and costs after actual replacement are still the main ones.

Current Wanda scores, BI, spectral energy, and local reconstruction errors are not direct estimates of these message distribution differences. It is necessary to actually measure the chain of "tool change → role output change → subsequent context change → workflow success or failure". Single-role fixed-teacher context comparison, real workflow replacement, and joint replacement are therefore three different experiments and cannot be substituted for each other.
