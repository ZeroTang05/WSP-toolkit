# WSP-toolkit Method Foundations

This document records the literature behind early method selection. Paired activation patching, SparseGPT local compensation, and TIES merging have since been implemented. Existing code, capabilities awaiting integration, and new interpretability methods are tracked together in the tool design and catalog. [Integration roadmap](../../docs/tool-design.md#9-existing-implementations-and-integration-roadmap) · [Tool catalog](../../docs/tool-catalog.json) · [Methods and permissions](../../docs/tool-permissions.md)

The [model-programming expansion](../reports/programming-expansion.md) implements Model Soups/Task Arithmetic update combinations, delta compression inspired by SVD-LLM objectives, and ShortGPT BI-based depth operations, with real weight-geometry measurements on four models. It describes each source method, adaptation, local optimality property, and limit on task-level conclusions. That report supersedes the corresponding pending statuses below. Combination and weight-only compression have been applied to 12 trained updates; at the time of this record, activation studies, causal layer deletion, and full public-task evaluations were still running.

Date: 2026-11-06. This is the literature basis for early method selection. See the [implementation inventory](../../docs/tool-inventory.md) for current implementations, real examples, and environments, and the [tool design](../../docs/tool-design.md) for interface contracts.

## 1. Design Principles and Corrections to Earlier Work

Derive tools through the chain **paper-defined problem and method → measurable quantities → parameter operations → experimental tests**. Reproduce a method before exposing it as a tool, then investigate agent selection strategies. Every new algorithm requires provenance and an explicit account of implementation differences. A working experiment does not establish reliable algorithmic provenance, and a cited algorithm does not establish a valid experiment.

Distinguish three categories:

- **Method reproduction:** Implement the target paper's objective, statistics, selection rules, and update steps faithfully, with a reference implementation and numerical comparisons.
- **Paper-inspired adaptation or new hypothesis:** State what changes, why adaptation is necessary, and which controls it requires. The original algorithm's name must not imply a complete reproduction.
- **Engineering infrastructure:** Tensor reads, hooks, checkpoints, export, rollback, and related functions support operability and reproducibility. They are not algorithmic contributions, and each engineering function does not require an artificial paper justification.

The earlier code audit found:

| Existing operation | Classification | Treatment |
|---|---|---|
| Scale the entire MLP `down_proj` by 0.8/0.9/1.1/1.2 | Manually defined one-dimensional weight-perturbation search | Retain as exploratory diagnosis or a baseline, rather than a core paper-derived optimization method |
| Channel score `mean(abs(activation)) * norm(down_column, 2)` | Custom activation-aware channel heuristic | Replace with a clearly defined method or label it as a new hypothesis; it is neither Wanda nor LLM-Pruner |
| Jointly remove gate/up rows and down columns | Valid MLP structural transformation | Retain the operation without claiming a complete LLM-Pruner implementation |
| Zero MLP outputs layer by layer | Ablation diagnosis | Measure intervention effects without claiming to derive an optimal edit direction |
| Export, rollback, file hashing, and independent loading | Experimental infrastructure | Establish traceability and recoverability, without claiming optimization effectiveness |

Earlier reductions in search-set NLL belong to these exploratory operations. They do not establish reproduction of a paper method, generation-accuracy gains, or improved general capability. Existing artifacts are retained unchanged; subsequent paper-based studies use versioned experiment directories.

## 2. Literature Behind the Pruning Tools

Write a linear layer as Y=WX: W rows are output dimensions, W columns are input dimensions, and X columns are calibration tokens. Papers may use different normalization conventions; fix these conventions before comparing implementations.

### 2.1 OBD/OBS: Loss Changes and Compensation When Deleting Parameters

OBD estimates deletion damage using a local second-order approximation. With approximately zero gradient and Hessian off-diagonal terms ignored, a single-weight score is 1/2 H_ii w_i². OBS retains parameter coupling. Its classical local quadratic problem gives the following compensation direction when deleting parameter i:

δw = − w_i / (H⁻¹)_ii · H⁻¹ e_i.

This result depends on a local second-order approximation and suitable invertibility or positive-definiteness conditions. It is not a global optimality guarantee for arbitrary LLM parameters.

Required tools estimate local curvature, calculate deletion damage, and generate compensation updates. Large models require layer-local approximations rather than a full-parameter Hessian. Sources: [OBD, 1989](https://proceedings.neurips.cc/paper/1989/hash/6c9882bbac1c7093bd25041881277658-Abstract.html), [OBS, 1992](https://proceedings.neurips.cc/paper/1992/hash/303ed4c69846ab36c2904d3ba8573050-Abstract.html).

### 2.2 Wanda: Per-Weight Importance from Input Activations

The paper's basic score is S_ij = |W_ij| · ||X_j,:||₂. Low-scoring weights are selected within each output row. Tools should expose activation second-moment energy collection, per-weight scoring, and rowwise mask generation separately. Verify statistics, masks, and target sparsity against the reference implementation. Arbitrarily aggregating scores across columns does not preserve the Wanda method.

Wanda provides a gradient-free, low-cost baseline. Unstructured zero weights do not automatically accelerate dense inference. Sources: [Paper, ICLR 2024, preprint 2023](https://arxiv.org/abs/2306.11695), [Author code](https://github.com/locuslab/wanda/blob/main/lib/prune.py).

### 2.3 SparseGPT: Pruning with Layer-Output Reconstruction

The objective can be written as min_W' ||WX − W'X||²_F with a target sparsity constraint. Input second-order statistics and approximate second-order updates propagate deletion error into retained weights. This differs from ordinary magnitude pruning followed by fine-tuning.

Proposed tools are `collect_input_gram` and `sparsegpt_prune_and_compensate`, returning sparse weights, masks, and layer reconstruction errors. Damping, blocking, column updates, and semi-structured modes must follow the author implementation. An ordinary least-squares solver alone is not SparseGPT. Sources: [Paper, ICML 2023](https://arxiv.org/abs/2301.00774), [Author code](https://github.com/IST-DASLab/sparsegpt/blob/master/sparsegpt.py).

### 2.4 LLM-Pruner: Group Pruning of Dependent Structures

LLM-Pruner identifies coupled structures, estimates gradient-related importance, and performs LoRA recovery. Check dependency rules, group scoring, and recovery settings against Sections 3.1, 3.2, and 3.3 of the paper respectively.

Proposed tools are `build_pruning_groups`, `estimate_group_importance`, and `prune_groups`. As an explanatory first-order approximation, deleting parameter group G changes loss by approximately −Σ_(i∈G) w_i ∂L/∂w_i. Absolute values, per-example aggregation, and second-order terms define different estimator choices and must be declared for each reproduction variant. This simplified expression does not replace the full algorithm.

Qwen3's gated MLP requires at least coupled gate/up rows and down columns. This is a partial adaptation; the complete model dependency graph and GQA structures require separate checks. Sources: [Paper, NeurIPS 2023, Methods Section 3](https://arxiv.org/html/2305.11627v3), [Author code](https://github.com/horseee/LLM-Pruner).

### 2.5 ShortGPT/SliceGPT: Reducing Depth or Dense Dimensions

ShortGPT estimates layer redundancy from input/output changes and provides a layer-removal route. SliceGPT uses computational invariance and rotations to remove rows and columns, producing smaller dense matrices. Both are useful deployment-efficiency comparisons. Arbitrary layer removal is not ShortGPT, and arbitrary matrix cropping is not SliceGPT.

Required tools cover layer-influence scoring and removal, plus the paper-defined rotation and slicing procedures. Sources: [ShortGPT](https://arxiv.org/abs/2403.03853), [SliceGPT author page](https://www.microsoft.com/en-us/research/publication/slicegpt-compress-large-language-models-by-deleting-rows-and-columns/).

Domain-preserving pruning includes [Cus-Prun](https://aclanthology.org/2025.findings-acl.1201/). Introduce domain data or constraints after reproducing a general pruning method, then compare the adaptation with that method and existing domain-pruning work. Domain-specific calibration data alone does not establish novelty.

## 3. Post-Training as a Reference for Parameter Changes

This section covers both post-training—SFT, preference optimization, and parameter-efficient fine-tuning—and post-training compression. Their data, gradient, and compute requirements differ and must be reported separately.

### 3.1 SFT: Defining Update Directions Through Task Objectives

SFT minimizes answer negative log-likelihood. With ordinary SGD, a single update is Δθ = −η∇L_SFT. Adam additionally uses momentum and second-moment preconditioning, so its update is not simply the negative gradient.

Proposed diagnostics include gradient norms, layerwise gradient distributions, angles between candidate updates and the negative gradient, and directional derivatives <∇L, Δθ>. These are local measurements and do not guarantee improvement after a finite step. Gradients come from training or calibration data, never the final test set.

Use standard SFT to establish what ordinary optimization achieves, then test whether agent-selected parameter operations yield comparable objective improvements at lower cost. Source: [InstructGPT, 2022, SFT and subsequent RLHF process](https://arxiv.org/abs/2203.02155).

### 3.2 LoRA: Restricting the Effective Degrees of Freedom

LoRA parameterizes ΔW=(α/r)BA, freezes base weights, and trains low-rank factors. It motivates representing updates with fewer degrees of freedom; it does not establish the effectiveness of arbitrary handwritten low-rank matrices.

Proposed tools construct, merge, constrain, and revert low-rank updates. Compare them with actual trained LoRA updates using effective rank, layer distribution, and subspace overlap. Applying SVD to gradients to construct a handwritten update is a new adaptation, not the original LoRA algorithm. A single-step gradient need not be low-rank.

LoRA reduces trainable parameters. Merging it usually leaves base-model parameter counts and inference FLOPs unchanged, so it does not by itself compress the model. Source: [LoRA, ICLR 2022, preprint 2021](https://arxiv.org/abs/2106.09685).

### 3.3 RLHF: Constraining Behavioral Drift with a Reference Policy

The relevant RLHF objective combines reward improvement with a KL penalty relative to a reference policy. Small parameter distance does not guarantee small behavioral drift; a bound on ||Δθ|| alone is insufficient.

Proposed tools measure token/sequence probability changes and estimate KL on guard data while retaining actual domain-task metrics. KL and sequence rewards depend on sampling distribution, direction, and estimation method. Small KL on finite data does not guarantee retention of every capability. Using KL to constrain candidate edits borrows the objective form; it does not implement PPO.

Source: [InstructGPT](https://arxiv.org/abs/2203.02155).

### 3.4 DPO: Optimizing with Explicit Preference Pairs

DPO compares preferred answer y+ and dispreferred answer y− through log-probability ratios to a reference model. Define z = β[log πθ(y+|x)/πref(y+|x) − log πθ(y−|x)/πref(y−|x)]; the objective is −E log σ(z).

Proposed tools measure preference margins and diagnose gradients of that objective. Enable this route only with credible preference pairs. Raising the probability of an answer category alone is not DPO. Derived directions still depend on real data rather than an LLM guessing them from descriptions.

Source: [DPO, NeurIPS 2023, Objective Formula and Its Analysis](https://arxiv.org/html/2305.18290v3).

### 3.5 Distillation: Recovery Objectives After Compression

Classical knowledge distillation trains students on teacher soft outputs, optionally combined with hard-label losses. For a pruned student sharing the tokenizer, teacher/student output KL can define a recovery objective. Declare temperature and the weights of hard and soft targets.

Proposed tools cache teacher outputs, measure student drift, and train recovery updates. Measuring KL alone is diagnostic; updating student parameters under that objective is recovery training. Compare both effectiveness and total cost between training-free compression and compression followed by distillation.

Source: [Hinton et al., Distilling the Knowledge in a Neural Network, 2015](https://arxiv.org/abs/1503.02531).

### 3.6 Task Arithmetic: Operating on Trained Directions

Task Arithmetic defines a task vector as fine-tuned parameters minus base parameters, then studies scaling, addition, subtraction, and combinations. It supplies directions with training provenance, without establishing that arbitrary deltas guessed from task descriptions will work.

An earlier pilot implements global coefficients. The public-task track adds searches over four normalized-depth groups, explicitly labeled a **layer-group adaptation and hypothesis to test**, rather than a complete reproduction of the original experiments. Standard LoRA SFT supplies the direction. Every control strategy shares the direction and candidate space, and its training cost is recorded separately. The full route is therefore not training-free optimization.

Source: [Editing Models with Task Arithmetic, ICLR 2023](https://arxiv.org/abs/2212.04089). Search and confirmation settings are documented in `research/search-protocol.md`.

### 3.7 Outcome Verifiers: Training on Actual Candidates Before Selection

[Cobbe et al., Training Verifiers to Solve Math Word Problems, 2021, Section 4.2](https://arxiv.org/html/2110.14168v2#S4.SS2) train a verifier on generator candidates sampled for training questions and labeled by final-answer correctness, then rank multiple candidates at inference. This motivates `generate_training_candidates`, `label_execution_outcomes`, `fit_verifier_delta`, and `rank_candidates`. The original model scales, sample counts, verifier heads, and auxiliary objectives are not equivalent to this project's LoRA binary-label training.

Current deterministic FinQA operator perturbations provide an engineering baseline rather than reproduction of training on real generator candidates. Rewriting once after rejection also differs from ranking multiple candidates. An audit of all training supervision found that a simple classifier using only the first operator scores 84.87% on 1,236 examples independent of its fitting data. This demonstrates a shortcut risk, not proof that the language model learned the same shortcut. Build real planner candidates on the training partition using the cited method, and separately control negative-example sources and inference selection. Renaming the old baseline is insufficient.

[Lightman et al., Let's Verify Step by Step, 2023](https://arxiv.org/abs/2305.20050) distinguish process from outcome supervision. This project lacks step-correctness labels; executable DSL steps or a correct final number do not establish process supervision. Real-candidate training and ranking remain future work in this record. See `research/verifier-followup.md` for data and budget controls.

## 4. Minimal Tool Families Derived from the Literature

| Tool family | Measurement or operation | Method source | Status in the early record |
|---|---|---|---|
| Weight and activation observation | Weight magnitude, input L2 energy, input Gram statistics | Wanda, SparseGPT | Input L2 energy checked against pinned Wanda code; Gram/curvature not yet implemented |
| Structural importance | Dependency grouping and group-gradient importance | LLM-Pruner | Explicit first-order gated-MLP adaptation implemented; full LLM-Pruner excluded |
| Sparse pruning and compensation | Per-weight masks, second-order compensation, layer reconstruction error | Wanda, SparseGPT, OBS | Wanda mask checked; SparseGPT/OBS compensation not yet implemented |
| Structural compression | Coupled group removal and export of valid configurations | LLM-Pruner; potentially ShortGPT | Coupled MLP slicing and independent reload checked; full dependency graph and ShortGPT not yet implemented |
| Update-direction diagnosis | Task gradients, directional derivatives, and standard trained updates | Explicit SFT/DPO objectives and update references | Planned; gradient diagnosis alone is not an optimization algorithm |
| Low-rank updates | Merging, constrained operations, and actual LoRA comparisons | LoRA | Frozen-base training, delta-only export, functional loading, and FP32 merge comparison implemented; public-task effects pending |
| Behavioral preservation and recovery | Reference-distribution drift, domain metrics, distillation recovery | RLHF KL objectives and knowledge distillation | Pending; the earlier English NLL guard is not KL |
| Experiment management | Commit, rollback, provenance/license/configuration records, independent evaluation | Engineering reproducibility | Executable method registry, source hashes, checkpoints, and separate scorers available; no full-system formal verification claim |

Each algorithm result should record method_id, paper and code versions, implementation type (reproduction/adaptation), objective, calibration-data fingerprint, gradient/curvature permissions, edit granularity, budget, and validation error. Formal reproductions additionally pin official code commits rather than a mutable main branch.

Agents should choose among validated operations: algorithms, pruning budgets, update sites, ranks, candidate step sizes, and rollback. Selection itself can be a research hypothesis, but must be compared with non-LLM search under the same permissions and total cost. Exposing arbitrary tensor writes alone does not establish a valid research method.

## 5. Experimental Route and Acceptance Criteria

1. **Reproduce two anchors first.** Use Wanda for gradient-free weight pruning and an explicit LLM-Pruner variant for structural pruning. Add SparseGPT as a second-order compensation control without requiring all candidates in the first round.
2. **Build a training reference.** Run budget-controlled SFT/LoRA on the same domain data and record actual parameter and behavioral changes. Add DPO only when preference data exists. Training is allowed as a reference and optional recovery path; its cost cannot be hidden under a training-free programming claim.
3. **Delegate choices to an agent.** Original-method and agent-selected configurations share implementation, data permissions, and total budget. Compare fixed strategies, ordinary search, and agent selection.
4. **Test transfer and deployment efficiency.** Use task data excluded from selection to report generation accuracy, key domain subtasks, behavioral drift, memory, and deployment time. A previously inspected final test cannot serve as the sole confirmation set for the new study.

Numerical acceptance requires Wanda statistics and masks to match the reference under explicit tie handling. SparseGPT comparisons must check reconstruction errors and updates for identical inputs and settings. LLM-Pruner checks cover groups, scores, retained structures, and recovery settings. Evaluate task metrics after layerwise numerical checks pass.

The research question is: **Can an LLM use interpretable measurements and constrained operations from established pruning and post-training methods to select, combine, and apply operations more effectively than fixed configurations and ordinary search at matched resources?** New parameter transformations require their own derivations and experiments; they must not be presented as paper reproductions.
