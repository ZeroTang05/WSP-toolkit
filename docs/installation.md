# Installation and supported capabilities

This page describes source version `0.4.0a5.dev0`, including the general tools, interpretability analysis, training, and compression methods in the design catalog. The prerelease package is `0.4.0a4.dev0`. Original project code is licensed under Apache-2.0. [Current source](https://github.com/ZeroTang05/WSP-toolkit/tree/main) · [Prerelease package](https://github.com/ZeroTang05/WSP-toolkit/releases/tag/v0.4.0a4.dev0) · [Third-party attribution](../THIRD_PARTY.md)

## Install by runtime

Core tensor I/O, direction operations, and second-order reconstruction do not require Transformers. Forward backends use mutually exclusive extras with exact versions in separate virtual environments.

```sh
python -m pip install .                  # Core tensor and reconstruction API
python -m pip install '.[qwen3,math]'     # Qwen3/Llama, Transformers 4.57.6
python -m pip install '.[qwen35,math]'    # Qwen3.5, Transformers 5.3.0, Torch>=2.4
```

The last two commands are options for separate environments: do not install qwen3 and qwen35 together. The original server environment uses Torch 2.5.1+cu121; choose the GPU build for the installed driver and prevent default dependency installation from replacing a shared experiment environment. The legacy research CLI and pinned lm-eval track use the qwen3/benchmark extra.

## Local MCP and Python read-only tools

Install the local MCP server from current source and supply a downloaded checkpoint and a separate workspace.

```sh
python -m pip install '.[qwen3,mcp]'
wsp-mcp --model /path/to/checkpoint --workspace /path/to/workspace
```

Configure `wsp-mcp` as a local process in the MCP client. `tools/list` discovers `describe_model`, `model_forward`, `read_artifact`, `list_tensors`, `read_tensor`, and `get_run`. Research scripts call the same tools with `open_inspection(model_dir, workspace).registry`. The host registers the checkpoint at model load. Use a full commit hash with `--revision` for a Hub repository.

The following example runs both entry points on a real model and compares generation, activations, and weight slices.

```sh
python examples/mcp_inspection.py --model /path/to/checkpoint
```

`describe_model` returns the base version and observable sites. `model_forward` saves collected values to the workspace and returns artifact IDs. `read_artifact` reads small slices by ID and coordinates, so large matrices are stored once and callers select the parts they need. The MCP and Python examples use separate run directories and budgets.

The example ran on real SmolLM2-135M-Instruct (Llama architecture). Both entry points discovered the same tools and agreed on slices of `1 × 5 × 576` activations, BF16 weights, and generation replay under the same configuration.

## Persistent Python sessions

On Linux, `--allow-python` enables `execute_python` and requires Bubblewrap and a virtual environment. In Python use `open_inspection(..., allow_python=True)`. A `session_id` retains variables and binds to one `variant_id`; use a new session ID for another candidate.

- `load(name, region)` reads a weight slice from that version; omit `region` for the whole tensor.
- `input_artifacts` binds registered array IDs to names for the call; code reads them with `read_array(name)`.
- `outputs` declares output names and axis labels to the host; code submits arrays with `save_array(name, value)`.
- `edits` can refer to output names to create patches, with `--allow-edits` also enabled. Apply patches with `apply_patch`.

`get_run` retrieves results and usage. The workspace's `python/runs` stores each cell's code and run record. Closing the tool instance releases the isolated process. A real Qwen3-0.6B example checked weight reads, retained variables, forward snapshots, and candidate reads after applying a patch. [Runnable example](../examples/mcp_python.py) · [Bubblewrap](https://github.com/containers/bubblewrap)

## Weight patches, activation interventions, and plots

Start the MCP service with `--allow-edits` to enable patches, projection, merging, and pruning; in Python pass `allow_edits=True` to `open_inspection`. `create_patch` saves replacement tensors at original precision and a manifest; `apply_patch` returns a candidate ID for forward passes and weight reads. `revert_patch` returns the parent ID for the next run. Small slices may use inline `values`; existing arrays may use `value_artifact_id`.

`export_patch` returns a patch package path relative to the workspace. Copy the package to `imports/<patch_id>` in a new workspace, validate it with `import_patch`, and apply it to the same parent. `export_model` produces a complete Transformers checkpoint, retaining the original configuration and tokenizer and rewriting only changed Safetensors shards. The example covers creation through reload, plus direction projection, task-vector averaging, TIES, DARE, Wanda, and two SparseGPT paths. DARE randomly drops part of the delta with `drop_probability`, rescales by the retention probability, and merges; a fixed seed replays the mask. [DARE](https://arxiv.org/abs/2311.03099)

```sh
python examples/mcp_patch.py --model /path/to/checkpoint
```

`--allow-interventions` separately authorizes activation replacement, zero ablation, direction injection, and site scans. For direction injection use `method="direction"`, `direction` or `direction_artifact_id`, and `strength`. The selected-channel direction is normalized and added to the activation at the specified strength. With the optional `analysis` dependencies installed, `render_evidence` reads registered arrays and outputs PNG or PDF. [Contrastive Activation Addition](https://arxiv.org/abs/2312.06681)

```sh
python -m pip install '.[qwen3,mcp,analysis]'
python examples/mcp_intervention.py --render --model /path/to/checkpoint
```

Both examples ran on real SmolLM2. A BF16 weight validated patching: reversion restored the base reading, and replay and complete checkpoint reload reproduced the candidate. The intervention example checked a shared baseline for two sites and produced activation, attention, and intervention-effect heatmaps.

## Deltas, development-set evaluation, and representation analysis

- `analyze_weight_delta` compares same-base candidates, WSP LoRA factors, or native PEFT LoRA imported by the host. It returns per-tensor and per-layer changes, principal directions, and original-precision rounding errors. `compress_update` saves smaller updates using ordinary or input-weighted low-rank truncation and optionally generates an applicable dense patch. The update is compressed; dense checkpoints retain their original shapes.
- The host registers development data for `evaluate_variants` using `--development-jsonl` and `--dataset-id`. It currently extracts boxed answers from model responses and text-matches them against reference answers. Versions share questions and sampling seeds; the tool reports paginated per-sample results and question-bootstrap intervals. `score_preferences` calculates the log-probability margin between two answers from real forward passes, then subtracts the reference version's margin.
- `compare_models` supports same-base weight deltas, internal representation similarity, and development-set behavior. The host can load a second local checkpoint with `--comparison-model`. Representation comparisons use the same raw text and each model's last valid token. Read the second workspace's results with `workspace="comparison"`.
- `scan_lens` reads each layer using the model's own final normalization and vocabulary head. With `analysis` installed, `probe_representation` compares true labels, control labels, and an input-embedding baseline on explicitly separated samples. With `gradients` installed and `--allow-gradients`, `attribute_output` computes token contributions using integrated gradients and returns the difference between their sum and the target score change.

```sh
python -m pip install '.[qwen3,mcp,analysis,gradients]'
python examples/mcp_delta.py --model /path/to/checkpoint
python examples/mcp_evaluation.py --cross-checkpoint --model /path/to/checkpoint --dataset benchmarks/wsp-practice-original-v1/gsm8k_train.jsonl
python examples/mcp_representation.py --attribution --model /path/to/checkpoint --dataset benchmarks/wsp-practice-original-v1/gsm8k_train.jsonl
```

These examples make basic calls using real SmolLM2 and GSM8K training-derived samples. The top three tokens from the final-layer readout matched the forward result. Two separately loaded checkpoints reproduced the similarity of a same-base candidate comparison. Layer readout supports direct readout and registered tuned lenses, both using the model's final output head.

## Cross-layer attention rollout and maximum flow

First query `trace_circuit_methods` with `describe_model`. On supported models, `model_forward` collects standard self-attention matrices from layer 0 through the target layer. Pass `run_id` and the method to `trace_circuit`. `attention_rollout` returns cross-layer propagation scores for all tokens, up to 256 tokens. `attention_flow` requires NetworkX from the `analysis` dependencies and a `query_position`; it computes maximum flow from that position to each input token, up to 32 tokens. Both results can be read with `read_artifact`, plotted using `attention_heatmap`, and are labeled descriptive evidence about attention weights. [Method paper](https://aclanthology.org/2020.acl-main.385/)

```sh
python -m pip install '.[qwen3,mcp,analysis]'
python examples/mcp_attention_rollout.py --model /path/to/checkpoint
```

The example reads 30 layers of attention on real SmolLM2-135M-Instruct, checks the first layer's mathematical relationship with the original matrix, and generates `5 × 5` rollout and `1 × 5` maximum-flow heatmaps. [Existing implementations and integration roadmap](tool-design.md#9-existing-implementations-and-integration-roadmap) · [Machine-readable catalog](tool-catalog.json)

## Structural compression and adapter training

- `compress_model` is enabled by `--allow-edits`. `block_influence` calculates changes in internal states across layers from calibration text and deletes a specified number of lowest-scoring layers. `taylor_mlp` also requires `--allow-gradients` and `--training-jsonl`; it ranks feed-forward channels by weights and gradients and shortens related matrix dimensions together. It follows the repository's grouping method, keeping width aligned to 128. It returns a complete checkpoint, parameter count, and manifest path. [ShortGPT](https://arxiv.org/abs/2403.03853) · [LLM-Pruner](https://arxiv.org/abs/2305.11627)
- `train_update` requires the `peft` dependency, `--allow-training`, and `--training-jsonl`. Samples are referenced by `id`, and `problem` is the prompt. LoRA and post-pruning recovery read `answer`; DPO reads `chosen` and `rejected`; sequence distillation reads `teacher_answer` and `teacher_model`. Post-pruning recovery opens a workspace from the checkpoint exported by `compress_model`. Standard PEFT files and WSP update factors are saved together; `materialize=true` requires edit permission. When training on a candidate, explicitly supply its parent model when reloading PEFT files. [LoRA](https://arxiv.org/abs/2106.09685) · [DPO](https://arxiv.org/abs/2305.18290) · [Sequence distillation](https://aclanthology.org/D16-1139/)

```sh
python -m pip install '.[qwen3,mcp,gradients,peft]'
python examples/mcp_training.py --model /path/to/checkpoint --dataset benchmarks/wsp-practice-original-v1/gsm8k_train.jsonl
```

On real SmolLM2-135M-Instruct, removing one layer reduced parameters from 134,515,008 to 130,974,912. Feed-forward channel pruning reduced parameters to 121,243,968 and width from 1,536 to 1,280. Both checkpoints reloaded. LoRA, DPO, distillation, and post-pruning recovery each ran one step; the reloaded LoRA adapter matched the toolkit's saved effective update. [Training and compression example](../examples/mcp_training.py)

## Sparse features and two-model comparison

The host places a dictionary in `features/<encoder_id>/`, containing `manifest.json` and `weights.safetensors`. `describe_model` lists dictionaries matching the current version. `model_forward` collects the sites specified by the dictionary; pass snapshot and dictionary IDs to `extract_features`. Results include the strongest triggering tokens, mean active feature count, per-source reconstruction errors, and readable feature-response arrays. Plot arrays with the `activation_heatmap` view of `render_evidence`, using `token_row` and `feature` axes.

| Dictionary file | Contents |
|---|---|
| `manifest.json` | `format_version=1`; `method` is `sae` or `crosscoder`; `activation` is `relu`, `topk`, or `jumprelu`. TopK requires `k`. `sources` registers `variant_id`, `site_id`, and optional `input_scale` and `center_on_decoder_bias` in input order. `weights_sha256` holds the weights-file digest |
| `weights.safetensors` | `b_enc` is the feature bias. For each source `i`, `W_enc.i` is `[channel, feature]`, `W_dec.i` is `[feature, channel]`, and `b_dec.i` is `[channel]`. JumpReLU also provides `threshold[feature]` |

The `crosscoder` view of `compare_models` accepts `encoder_id`, `left_artifact_id`, and `right_artifact_id`; snapshots must have the same input token sequence. This view calls the same feature extraction function. The public SAE example reads layer 3 feed-forward output from the SmolLM2 base using an EleutherAI dictionary. The joint dictionary example fits a small dictionary to SmolLM2-Instruct and a real weight candidate, then compares both on different inputs. [Base checkpoint](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) · [Public dictionary](https://huggingface.co/EleutherAI/sae-SmolLM2-135M-64x) · [Crosscoder paper](https://transformer-circuits.pub/2024/crosscoders/index.html) · [Both examples](../examples/mcp_features.py)

```sh
python examples/mcp_features.py --model /path/to/SmolLM2-135M --sae /path/to/layers.3.mlp
python examples/mcp_features.py --model /path/to/SmolLM2-135M-Instruct --dataset benchmarks/wsp-practice-original-v1/gsm8k_train.jsonl
```

## Hidden-dimension slicing and packed quantization

SliceGPT, GPTQ, and AWQ in `compress_model` share calibration text and export manifests. SliceGPT changes hidden dimensions in a separate upstream environment and restores sliced skip-connection matrices at reload. GPTQ and AWQ export 4-bit integer weights using a separate LLM Compressor environment; the manifest includes original bytes, actual exported bytes, and packed tensor count. All three methods were reloaded and run on SmolLM2-135M-Instruct. [Installation and usage](mcp-tools.md#packed-quantization) · [SliceGPT example](../examples/mcp_slicegpt.py) · [GPTQ and AWQ example](../examples/mcp_quantization.py)

## Feature attribution graphs

`trace_circuit(method="feature_attribution_graph")` requires a host-registered separate Circuit Tracer environment, per-layer feature dictionaries matching the model version, and gradient permission. It outputs the full edge matrix, node coordinates, plottable summaries, and output deviation after replacing feed-forward layers using only the dictionary. A graph was generated and checked on real SmolLM2-135M-Instruct. [Installation and usage](mcp-tools.md#feature-attribution-graphs) · [Example](../examples/mcp_feature_circuit.py)

## Factual editing

`edit_facts` uses ROME for one fact and MEMIT for batches. The host registers EasyEdit source and a separate environment and grants gradient and weight-edit permissions. The tool collects real activations from calibration text and saves algorithmic changes as a patch. After applying it, compare direct questions, paraphrases, and neighboring facts. This was validated in a SmolLM2-135M-Instruct CPU example. [Installation and usage](mcp-tools.md#factual-editing) · [Example](../examples/mcp_fact_editing.py)

## API and evidence

| Interface | Purpose | Existing evidence |
|---|---|---|
| WeightStore, EditPolicy | Original/current tensors, proposals, reversion, original-precision export | Checks for unchanged source, precision materialization, and edit constraints |
| contrast_direction, project_output_direction | Matched mean differences and selected-column output-direction operations | Explicit algebra, column support, and direction-sign invariance |
| input_hessian, sparsegpt_reconstruct | Second-order statistics and fixed-mask or adaptive pruning compensation | Numerical comparisons against author code and real 0.6B twelve-condition experiment |
| ForwardService, ForwardPolicy, PracticeSet | Generation/teacher-forced execution with fixed sampling, budgets, and data roles | Qwen3/Llama and Qwen3.5 hybrid-architecture checks; real base observations |
| HuggingFaceBackend | Qwen3/Llama dense observation and generation | Real 0.6B, 128 texts, 27,776 tokens |
| Qwen35Backend | Original Qwen3.5 text execution and hybrid-layer observation | 473 tensors checked individually; 32 texts, 6,528 tokens; 15 MTP tensors are not executed |
| CPUKernel, ProgrammingSession | Persistent CPU analysis, isolated files/network/devices, and forward-service connection | Real 0.6B kernel and edit–observation channel executed |

Import the two backends from their respective modules; the host loads models and schedules devices:

```python
from weight_programming import WeightStore, ProgrammingSession, sparsegpt_reconstruct
from weight_programming.hf_backend import HuggingFaceBackend       # qwen3 extra
# Or in a separate environment:
from weight_programming.qwen35_backend import Qwen35Backend        # qwen35 extra
```

The isolated CPU kernel requires Linux Bubblewrap and a virtual-environment Python; macOS can use the core tensor API. The library does not schedule GPU jobs. Research matrix scripts in the source have not all been organized as public APIs; the OfficeBench executor remains a separate research module.

Evidence for performance improvements, complete tasks, and efficiency is recorded separately. Real observations do not establish mathematical improvement; local reconstruction does not establish full-model fidelity; interface coverage does not establish an independent MIT reproduction.

## Classifiers and native adapters

For DistilBERT single-label classifiers, `model_forward(mode="predict")` returns label scores, probabilities, and the predicted label. It reuses snapshots, heatmaps, probes, representation comparison, and weight patches. Scores for 16 real SST-2 inputs matched direct Transformers inference elementwise. The classifier and SmolLM2 also passed cross-architecture CKA on the same texts, comparing structural similarity of representations across samples. [Classification example](../examples/mcp_classification.py) · [Data acquisition](../research/acquisition/prepare_sst2.py)

Register native PEFT LoRA with `--peft-adapter` at startup or `peft_adapters` in Python. The host can bind an adapter for an existing candidate with `tools.delta.import_peft_adapter(parent_variant_id, adapter_dir)`. It follows PEFT's actual scaling and transpose semantics and enters the existing delta, merging, and compression interfaces. `compress_update` can assign different ranks to different tensors; `project_weight` can directly consume two sets of same-coordinate activations. [Usage](mcp-tools.md#imported-adapters-and-directions) · [Native adapter example](../examples/mcp_native_peft.py)
