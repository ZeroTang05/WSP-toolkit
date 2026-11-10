# Tool logic, method assumptions, and permissions

Local MCP and Python share permission settings. Structural export requires edit permission; Taylor selection also requires gradient permission and host-registered training data. `train_update` requires separate training permission. Materializing a training update as a weight patch also checks edit permission. [Usage and installation](mcp-tools.md#training-and-structural-compression)

The host must explicitly enable `execute_python` on Linux. Each session binds one model version, and artifacts can be proposed as patches. `edit_facts` requires edit and gradient permissions and a host-registered separate EasyEdit environment. It returns a patch ID; the caller decides when to create a candidate version. [Tool usage](mcp-tools.md)

2026-11-07. Tools are organized around observation, diagnosis, adaptation and combination, compression, compensation, and confirmation. Each method has its own assumptions and experimental permissions.

| Tool or method | Problem addressed | Evidence or assumptions | Permissions and cost |
|---|---|---|---|
| Weight reading and writing, matrix analysis, direction projection | Observe and submit specific parameter operations | Same base coordinates, explicit location, behavioral hypothesis | Available for direct programming under the original protocol, subject to protected-parameter and write rules |
| Budgeted generation and teacher-forced execution | Inspect behavior, probabilities, and internal states | Practice set with fixed provenance; frozen precision and sampling | Original interface; request budget and actual forward/token cost are accounted separately |
| Paired activation patching | Test whether an activation site changes a specified output | Explicit clean/corrupt pairs, aligned positions, target and contrast metrics | Mechanistic diagnosis extension. Temporary interventions do not modify weights persistently and do not enter the original protocol’s permission track. |
| Wanda, BI, and similar observation scores | Quickly propose removal candidates | Calibration distribution, scoring assumptions, and actual removal results | Observation requires causal confirmation; adaptations and original published methods are described separately |
| SparseGPT compensation | Reduce local output damage caused by removal | Second-order input statistics, damping, sparsity pattern, and independent checks | Absence of SGD does not eliminate a fitting objective; column-wise reconstruction belongs to a separate compression/compensation track and is not automatically classified as independent discovery under the original protocol |
| LoRA/SFT and post-pruning recovery | Establish role capabilities or recover from damage | Training material, loss, training budget, and ordinary post-training controls | Explicit training-enabled track; direction training costs must be reported in no-training search claims |
| Update combination and low-rank approximation | Adjust existing directions or reduce representation cost | Same base and coordinates, update provenance, and local approximation objective | Training-derived directions require provenance; geometric optimality does not guarantee task optimality |
| DPO and other preference optimization | Improve behavior with a reliable preference definition | Reliable preference pairs, reference distribution, and objective | Requires appropriate data and training conditions; arbitrary probability gains do not establish DPO |

Sources: [method foundations](../research/methods/method-foundations.md), [SparseGPT implementation](../research/reports/reconstruction-study.md), [paired activation method](https://arxiv.org/abs/2309.16042v2), and [original MIT protocol](../research/references/mit-wsp/exp8_prompt.txt). Method sources support the algorithms' motivation; effects must still be established on the current model and task.

## Why paired activation tools have a separate track

Weight editing asks whether a persistent parameter change improved behavior. Activation patching asks what a state at a specified site can change in a counterfactual input pair. The latter can help locate candidate sites for the former but does not automatically specify a weight edit. Restoring an output through a site's activation does not establish that knowledge resides exclusively there or that an independently removable circuit exists.

`ActivationPatchingStudy` performs three real forward passes: clean, corrupt, and patched. It reports logit difference, target probability, target log-probability, and their changes. It uses equal-token-length input pairs and explicit positions, without guessing alignment; raw residuals and down_proj inputs are supported. Restoration ratios are not clipped to [0,1]; a near-zero denominator returns null, and a worse clean result is flagged as invalidating the restoration interpretation.

Permission separation expresses experimental semantics; Python cannot establish compliance of an arbitrary algorithm. The original-style `dispatch_tool` exposes persistent CPU analysis and budgeted forward passes without an activation-intervention interface. Training, iterative empirical-loss fitting, and simulated networks still require review of the program trace. Extension tracks may use reconstruction, gradient diagnosis, or post-training, with separate names, accounting, and comparisons; their gains cannot stand in for original no-training results.

## Interface for an LLM

`tool_schemas(session)` generates two callable schemas from the actual backend and budget: `execute_python` and `model_forward`. For example, Qwen3.5 lists only 3/7/11/15/19/23 as valid standard attention layers and allows residual coordinates 0 through 24. `dispatch_tool` calls real `ProgrammingSession` methods and rejects unknown tools and extra Python execution arguments. The executor checks coupled limits on counts, tokens, and return sizes.

This provides an interface description across model providers. The next step is to have a new model context actually use the tools, retaining the complete program and costs, then compare it with fixed recipes and random search. An independent original-protocol context must not receive the reference edit or analysis conclusions from the current research session.
