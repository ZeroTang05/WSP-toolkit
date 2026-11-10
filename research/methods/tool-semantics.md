# Tool semantics: provable local properties and limits

This document formalizes local properties using standard algebra. These derivations do not introduce a new optimization theorem. Algorithm fidelity still requires comparison with the original papers and author implementations, and task benefits require complete experiments.

[Programming expansion, Section 2](../reports/programming-expansion.md) formalizes update combinations, weight-spectrum truncation, output reconstruction under a fixed activation distribution, and BI-based depth selection. These respectively provide exact update algebra, a Frobenius-optimal approximation, a locally optimal output approximation, and a heuristic structure-selection rule. The last has no task-optimality guarantee. New registry entries declare method provenance; the standalone CLI does not yet run uniformly through `ContractRegistry`, so registration does not establish runtime verification.

## 1. Wanda Mask

Source: [Wanda](https://arxiv.org/abs/2306.11695). For each output row, stably sort columns by |W_ij|·sqrt(E_j) and select k=floor(p·d_in) unique column indices. Sorting returns a permutation of columns, so exactly k positions are selected in each row. Setting those positions to zero and copying all other values preserves the tensor shape and leaves every unmasked position exactly unchanged.

This property counts selected positions, not the exact reduction in nonzero weights: some selected weights may already be zero. It does not establish minimum pruning loss, preservation of domain capabilities, or faster inference. Input statistics with the same shape can yield different masks; the name of a statistic in an interface does not establish its provenance. The tool must collect its own statistics and validate them against pinned source code and the author's implementation.

`prune_public.py` collects input energy, applies the validated row-wise rule, and saves the bitmask. The agent selects only declared allowed configurations and does not submit invented statistic tensors. Low-level functions in the general Python library are not an independent security boundary.

## 2. Valid structural slicing of gated MLP

For a bias-free gated MLP, define h(x)=σ(W_g x)⊙(W_u x) and output y=W_d h(x). Let S be a 0/1 row-selection matrix for the retained channels, and M=SᵀS. Since σ and the Hadamard product act element-wise:

S h(x)=σ(SW_g x)⊙(SW_u x).

After removing the corresponding gate/up rows and down columns:

y_slice=(W_d Sᵀ)[σ(SW_g x)⊙(SW_u x)]=W_d M h(x).

Thus, in real arithmetic, valid slicing is equivalent to the original MLP with the same channel mask. This does not establish equivalence to the unpruned model. Biases require separate handling; the current structural adapter rejects biased layers before changing any layer. It also rejects missing layer selectors, duplicate or out-of-bounds channels, and nonuniform widths.

Source: the dependency-group idea in [LLM-Pruner](https://arxiv.org/abs/2305.11627). This project implements a first-order MLP adaptation, not the full dependency graph, all importance variants, or the recovery procedure.

## 3. LoRA and functional delta

Source: [LoRA](https://arxiv.org/abs/2106.09685). Under real arithmetic, (W_0+c·α/r·BA)x=W_0x+c·α/r·B(Ax). Keeping W_0 fixed and adding the low-rank term to the module output therefore expresses the same weight update.

With finite precision, addition order, matrix multiplication shape, and merge quantization affect error. The algebraic identity alone does not establish bitwise equality in BF16. CPU output and exception-isolation tests and an FP32 materialization comparison on real Qwen0.6 are available; BF16 generation and final task effects must still be measured with the actual executor.

Layer-group coefficients refine the parameterization of the same valid update. See `search-protocol.md` for the source and adaptation limits. They do not confer the ability to handwrite any valid direction.

## 4. Combination and architectural adaptation

Even when T_1 and T_2 pass their individual checks, Post(T_1) must satisfy Pre(T_2) for their combination to be valid. For example, structural pruning may change the shape required by the original LoRA factors or invalidate the base fingerprint. The standard benchmark executor currently rejects a mask and delta passed together until a separate combination protocol is defined and verified.

Verifying output error on a finite input set at each step does not directly bound final accuracy error on the workflow distribution. Cross-model use also requires recompiling module locations and shapes, rather than copying a delta between architectures. Tool correctness, controller selection, domain-quality preservation, and deployment efficiency each require their own evidence.

## 5. Proxy guard constraints do not imply accuracy preservation

A decrease in mean classification cross-entropy (CE) does not establish that accuracy increases or stays constant. Consider two binary classification questions. If the base assigns probabilities 0.51 and 0.01 to the true classes, accuracy is 1/2 and mean CE is approximately 2.639. If an edit changes both probabilities to 0.49, accuracy is 0/2 and mean CE is approximately 0.713. CE decreases substantially while accuracy falls. This is a counterexample, not an empirical model result or new theorem. Applying option softmax after character-length normalization does not remove the general limitation.

The bootstrap on development CE for this project's 32 questions tests only the declared proxy constraint. It does not establish the paper's accuracy-preservation condition, an RLHF reference-policy KL constraint, or a general capability guarantee. The fixed 0.8 candidate satisfies the development proxy, yet its full ARC-C score changes from 363/1172 to 351/1172. The 95% paired-question interval includes zero, so overall degradation is not established; the accuracy point estimate and seed variation must nevertheless be reported separately. The final test result does not change the protocol's selection rule.

## 6. Input binding and information retention are different conditions

The FinQA authors' [pinned row-to-text function](https://github.com/czyssrs/FinQA/blob/0f16e2867befa6840783e58be38c9efb9229d742/code/utils/general_utils.py#L21) expresses column headers alongside values. V3 passes public-input provenance and hash checks, but selecting data rows still loses the year header. Correct provenance therefore does not guarantee a sufficient representation. Reconstructing prompts from the complete development run confirms this representation gap in 705 questions; it does not establish that all 705 outcomes were caused by the gap.

Assume the public input contains a header h. Let R be the non-header data rows, and define the closure of selected IDs A as C(A)=A∪{h} when A∩R is nonempty, and C(A)=A otherwise. Then A⊆C(A), C(C(A))=C(A), and A⊆B implies C(A)⊆C(B). V4 retains the shared header using this rule, leaving retriever predictions and their ID-recall statistics unchanged. These are input-integrity engineering conditions and elementary set properties, not a new pruning algorithm. They do not guarantee preservation of every annotation, unit, or full-text context. The repaired version must be compared with frozen V3 under the same actual budget.

An eight-question input control using the same original role factor is complete. The retriever prompt and output were unchanged. Retaining the header corrected the subtraction order in the PM example, and the overall result changed from 0/8 to 1/8; another program changed its operand but remained wrong. Input representation can therefore affect output, and this improvement cannot be attributed to parameter editing. Full-partition and retraining controls are incomplete, so they do not yet support a domain-fidelity conclusion.
