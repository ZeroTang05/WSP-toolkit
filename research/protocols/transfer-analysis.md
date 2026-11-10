# From tool semantics to cross-model trends: a testable bridge

Status: Candidate analytical framework, not validated conclusion or novelty claim. It complements `paper-spec.md` and does not replace public benchmarks, full baselines, and real workflow experiments.

## 1. What is migrated is the operation plan, not the parameter coordinates

Different models have different parameter dimensions and coordinate systems, and the delta of one model cannot be directly added to another model. Comparable or transferable objects are tool-level plans, such as "Apply a 0.5 coefficient to the mathematical task direction of this model" and "Allocate a pruning budget to the valid MLP group of normalized mid-levels." The target model generates the actual patches using its own data observations, task directions, and architecture adapters.

This migration may still require training/calibration calculations for the target model; "no additional queries to the target dev set" cannot be stated as zero cost or data-free transfer. There have been related studies on cross-architecture pruning/policy migration, and the formal novelty still requires further full-text comparison.

## 2. Conditional local response bounds

For a subset of continuous, parameter-shape-preserving tools, let the architecture-specific compilation for model m produce K intervention directions D_m with action coordinates a∈R^K. Define smooth proxy risk on a common task:

f_m(a)=L_m(θ_m+D_m a).

Different models must use comparable definitions of risk, such as normalized conditional probability losses on the same multiple-choice candidate. Native per-token NLL from different tokenizers are not directly comparable without normalization. The generation accuracy and workflow success rate are not smooth, and this section does not directly guarantee them.

Define the projected gradient g_m=∇_a f_m(0) and the projected curvature H_m=∇²_a f_m(0). If f_m and f_n are third-order differentiable in the same action ball and have corresponding third-order remainder bounds, then by second-order Taylor expansion and triangle inequality:

|[f_m(a)−f_m(0)]−[f_n(a)−f_n(0)]|
≤ ||a||₂ ||g_m−g_n||₂
+ 1/2 ||a||₂² ||H_m−H_n||_op
+ (β_m+β_n)||a||₂³/6.

If the parameter errors of the numerical implementation do not exceed ε_m(a), ε_n(a) respectively, and the Lipschitz constants of the local risk with respect to the parameters are ℓ_m, ℓ_n, the compiled implementation and baseline error terms need to be added:

ℓ_m[ε_m(a)+ε_m(0)] + ℓ_n[ε_n(a)+ε_n(0)].

This is a conditional corollary of standard Taylor analysis and does not claim to be a new theory per se. The key research question is whether projected response differences are a better predictor of operational gain consistency and migration failure than parameter quantities, series labels, or raw weight statistics.

This analysis does not apply directly to discrete layer deletion; structural operations still require legality verification and actual intervention experiments. Bounds may be useless when the step size is large, the distribution is unseen, or the proxy risk is disconnected from the task metrics. The estimated curvature and numerical error under limited data are also not proof of the full input domain.

## 3. How to empirically test

1. Fix the common tasks, data splits, training-direction sources, and action-coordinate semantics. The direction comes from the implementation of each model's own paper method and does not copy the weights of other models.
2. Measure local projected gradients/finite difference curvatures on the calibration set and record measurement costs; compare purely structural/scale features with functional response features.
3. Record the true gain, guard damage, and prediction residuals for tool operations on the development set; verify when local approximations fail.
4. Construct a source-model→target-model plan migration matrix, and compare source plan direct compilation, post-query adjustment with a small amount of target development, de novo random/Bayesian search, and LLM strategies.
5. Compare the selection quality and the number of additional target queries under fixed resources; finally use the official confirmation set to check, and do not adjust the plan based on the confirmation set.
6. Analyze the same series/cross-series, fixed rank/fixed relative parameter budget, different tasks and seeds. Gains of the same sign are only observed on a few models, and a universal rule cannot be claimed.

## 4. Connection with workflow applications

Individual improvements in specialty roles do not guarantee end-to-end improvement. Workflow risks also depend on interface distribution, error propagation, and validator behavior. Therefore, it is necessary to measure the response of a single role and the response of the entire workflow after replacing the role, and compare whether the rankings of the two are consistent.

For the new formalization, see [multiagent-role-programming.md, Section 7](multiagent-role-programming.md): Constructing the trajectory TV/KL upper bound based on role conditional message distribution, indicating that the collaborative advantage can provide a reference for allowable damage for subsequent replacements. This standard probability analysis assumes a shared environment, bounded calls, and other stated conditions, has not been numerically certified by the current activation or SVD statistics, and does not prove quality preservation.

To study the common compression backbone plus role delta, it can be written as:

θ_c=G_z(θ_0+Δ_shared), θ_k=θ_c+Δ_k.

However, "sparse shared backbone + external dedicated memory" is already the direct subject of [Knowledge Offloading](https://arxiv.org/abs/2605.29075), and there are also related works such as multi-task shared backbone pruning [LoGIC](https://ojs.aaai.org/index.php/AAAI/article/view/39195). This representation cannot be taken as an original construct; one needs to examine whether the workflow-level constraint construct is more efficient than the generic/task-by-task construct.

Likewise, [WeAct](https://arxiv.org/html/2610.26847v2) already includes capability separation, tooled loading/unloading, retention capabilities, and cache/time cost analysis. The potential differences in this project should lie in the role of tool method semantic validation, operation construction and migration, and workflow goals, rather than the registry, switching, or cache cleaning itself.
