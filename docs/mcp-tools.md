# MCP and Python tools

The same tool definitions serve a local Model Context Protocol (MCP) client and Python. Start with `describe_model` to find the model's version ID and observation sites. A site names a specific internal location, such as the output of a feed-forward layer.

Both entry points share the same tools. Call `describe_model` first to obtain the model version and observable sites, then read weights or run the model as needed. Each run saves large arrays and details in the workspace. Use the returned artifact IDs to read, compare, or plot results.

## Setup

From the repository root, install the model adapter and MCP support. The example below was exercised with a local SmolLM2-135M-Instruct checkpoint on a CPU. [Model checkpoint](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)

```sh
python -m pip install '.[qwen3,mcp]'
python examples/mcp_inspection.py --model /path/to/checkpoint
```

Configure your MCP client to start `wsp-mcp` as a standard-input/output process. Pass `--model /path/to/checkpoint --workspace /path/to/workspace`. If your client runs outside the Python environment, give it the full path to the installed `wsp-mcp` command. Use `tools/list` to see the tools available in that session. A Hub repository can be selected with `--model owner/repo --revision <full-commit-sha>`.

In Python, the same request looks like this.

```python
from weight_programming import open_inspection

tools = open_inspection("/path/to/checkpoint", "/path/to/workspace")
model = tools.registry.call("describe_model", {})
run = tools.registry.call("model_forward", {
    "variant_id": model["variant_id"],
    "mode": "teacher_forced",
    "texts": ["The capital of France is"],
    "observe": ["block.0.residual_out"],
})
snapshot = run["artifacts"]["hidden_1"]["artifact_id"]
print(tools.registry.call("read_artifact", {
    "artifact_id": snapshot,
    "region": [[0, 1], [0, 1], [0, 4]],
})["values"])
```

This reads four values from the first token of the first sample. The three ranges select the sample, token, and channel. Each range includes its start and excludes its end.

## Tools and permissions

| Task| Tools | Setup |
|---|---|---|
| Discover, run, and read| `describe_model`, `model_forward`, `list_tensors`, `read_tensor`, `read_artifact`, `get_run` | Default. Model capabilities select generation, text scoring, or classification |
| Edit and replay| `create_patch`, `apply_patch`, `revert_patch`, `export_patch`, `import_patch`, `export_model` | `--allow-edits`. Edits are saved as workspace artifacts linked to their parent version |
| Observe, explain, and compare| `scan_lens`, `compare_models`, `trace_circuit`, `extract_features`, `probe_representation`, `render_evidence`, `attribute_output` | `analysis` adds probes, heatmaps, directed graphs, and attention maximum flow. `gradients` with `--allow-gradients` enables integrated gradients. Feature extraction needs a registered dictionary |
| Intervene| `run_intervention`, `localize_behavior` | `--allow-interventions`. Replace, zero, or shift selected activations. The `concepts` extra adds LEACE |
| Run Python | `execute_python` | `--allow-python` on Linux with Bubblewrap and a virtual environment. Use `--allow-edits` to create a patch from an output array |
| Analyze or combine updates| `analyze_weight_delta`, `project_weight`, `merge_updates`, `prune_and_compensate`, `compress_update` | Delta analysis is read-only. Other tools require `--allow-edits`. Methods include TIES, DARE, Wanda, SparseGPT, and low-rank update compression |
| Compare answers| `score_preferences`, `evaluate_variants` | Register development rows with `--development-jsonl` for `evaluate_variants`. Rows contain `id`, `problem`, and `answer` |
| Compress or train| `compress_model`, `train_update` | See the settings below |
| Edit factual associations| `edit_facts` | `--allow-edits --allow-gradients` and a separate EasyEdit environment |

For comparisons across separately loaded checkpoints, use `--comparison-model` and select `right_workspace="comparison"`. Crosscoder comparison also takes the two snapshot IDs and checks that their input token sequences align.

## Classification models

DistilBERT single-label classifiers use `model_forward(mode="predict", texts=[...])`. The returned artifacts contain `class_logits`, `class_probabilities`, and `predicted_label_ids`; `summary.label_names` gives their label order. Observations include embeddings, encoder block outputs, feed-forward inputs and outputs, and bidirectional attention. The same artifact reader, heatmaps, linear probes, CKA comparison, and weight patch tools consume these results. `describe_model` reports the available modes and sites.

A classification model assigns a label to text, such as positive or negative. `predict` saves label scores and internal values at selected sites. Classifiers retain their own special tokens, and attention can inspect both preceding and following text. Vocabulary readout, generation, and causal interventions are exposed according to backend capabilities.

```sh
python -m pip install '.[qwen3,mcp,analysis]' 'pyarrow==17.0.0'
python research/acquisition/prepare_sst2.py --output /path/to/sst2-train.jsonl
python examples/mcp_classification.py --model /path/to/DistilBERT-SST2 \
  --dataset /path/to/sst2-train.jsonl --comparison-model /path/to/SmolLM2-135M-Instruct
```

The example uses 16 real SST-2 training rows. Class scores matched direct Transformers inference exactly. A linear probe scored 7/8 on separate demonstration rows, with a 4/8 shuffled-label control. The example also checked all six attention layers, an additive classifier-bias patch, and cross-architecture CKA. [Checkpoint](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) · [SST-2](https://huggingface.co/datasets/stanfordnlp/sst2) · [Runnable example](../examples/mcp_classification.py)

## Imported adapters and directions

Install `peft` and pass `--peft-adapter /path/to/adapter` when starting MCP. Repeat the option to register several adapters. Python accepts `open_inspection(..., peft_adapters=[...])` for base-model adapters; the host can bind an adapter to an edited parent with `tools.delta.import_peft_adapter(parent_variant_id, adapter_dir)`. The returned `adapter_id` works with `analyze_weight_delta`, `compress_update`, and `merge_updates`. Startup registrations appear in `describe_model.registered_adapters`.

PEFT reads its native checkpoint format and provides each layer's actual scale and transpose rule. Import supports verified linear LoRA, including rsLoRA and per-module scale/rank patterns. Additional transformations such as DoRA require separate verification. Source files, configuration, parent version and effective factors are recorded together. [PEFT format](https://huggingface.co/docs/peft/en/developer_guides/checkpoint) · [LoRA](https://arxiv.org/abs/2106.09685) · [Runnable native-adapter example](../examples/mcp_native_peft.py)

```sh
python -m pip install '.[qwen3,mcp,peft]'
python examples/mcp_native_peft.py --model /path/to/SmolLM2-135M-Instruct
```

`merge_updates` accepts `patch_ids`, `adapter_ids`, or both, with at least two sources in total. Patch sources use their saved native-precision deltas; adapter sources use effective LoRA deltas. For task arithmetic, coefficients follow patch order then adapter order. `compress_update.rank` can be one integer or a map from every affected tensor name to its retained rank. This rank is the number of low-rank directions kept for that tensor.

`project_weight` accepts an explicit `direction` or two `direction_artifacts` from the selected parent. For artifact inputs, it computes the normalized mean of target minus source activations, checks the actual output coordinate site, and saves the direction as a reusable artifact. `sample_indices` and `token_position` select aligned rows. The existing projection formula then removes the requested direction from selected weight columns. [Direction removal](https://arxiv.org/abs/2406.11717) · [Runnable direction example](../examples/mcp_delta.py)

For larger direction analyses, `analyze_weight_delta` accepts `direction_solver="randomized"`, `direction_seed`, and `direction_iterations`. It uses PyTorch `svd_lowrank`, records the retained rank and residual error, and limits the selected matrix to 32 million values. The default exact SVD accepts up to four million values. [PyTorch low-rank SVD](https://docs.pytorch.org/docs/stable/generated/torch.svd_lowrank.html)

For activation direction injection, set `run_intervention.method` to `direction`. Pass `strength` and either `direction`, a list with one value per channel, or `direction_artifact_id`, the ID of a one-dimensional `channel` array. The tool normalizes the direction across the selected channels before scaling it and adding it to the specified token positions. [Contrastive Activation Addition](https://arxiv.org/abs/2312.06681) · [Runnable example](../examples/mcp_intervention.py)

## Interpretability methods

**Graph rendering.** Call `trace_circuit` with `method="attention_flow"`, then pass its `graph_artifact_id` to `render_evidence` with `view="graph"`. `read_artifact` returns the nodes, edges, and numeric source ID. Each arrow summarizes the maximum flow between a query position and an input token across the selected layers. The figure preserves the `descriptive_attention` evidence label. Set `max_edges` to limit visible edges by absolute score, or `format="pdf"` to save a vector figure. [Attention Flow](https://aclanthology.org/2020.acl-main.385/) · [Runnable example](../examples/mcp_attention_rollout.py)

**LEACE.** Install `analysis` and `concepts`. Call `probe_representation` with `fit_eraser=true`, labeled activations, and separate `train_indices` and `evaluation_indices`. The tool fits a transform on the training rows and reports held-out linear probe accuracy before and after applying it. Pass `concept_erasure.eraser_artifact_id` to `run_intervention` with `method="concept_erasure"` on the same model version and site. An optional `control_seed` applies a seeded random change of the same size per token. On SmolLM2-135M-Instruct, the probe went from 15 correct to 8 correct on 16 held-out noun and verb examples. [LEACE paper](https://arxiv.org/abs/2306.03819) · [Author library](https://github.com/EleutherAI/concept-erasure) · [English EWT dataset](https://universaldependencies.org/treebanks/en_ewt/index.html)

The LEACE example takes a directory with `en_ewt-ud-train.conllu` and `en_ewt-ud-dev.conllu`. [English EWT r2.17](https://github.com/UniversalDependencies/UD_English-EWT/tree/r2.17)

```sh
python -m pip install '.[qwen3,mcp,analysis,concepts]' 'conllu==6.0.0'
python examples/mcp_concept_erasure.py --model /path/to/checkpoint --dataset /path/to/UD_English-EWT
```

**Answer attribution.** With `gradients` and `--allow-gradients`, pass `method="contrastive_sequence"`, `text`, `baseline_text`, `chosen`, and `rejected` to `attribute_output`. Use equally long prompt token sequences and include leading spaces where the answers require them. The tool fixes both answers and integrates the input gradient of their log-probability margin, with the same `sum` or `mean` aggregation as `score_preferences`. The output contains one signed contribution per prompt token. `integration_error` compares their sum with the observed margin change. [Inseq](https://aclanthology.org/2023.acl-demo.40/) · [Runnable example with `--attribution`](../examples/mcp_representation.py)

**Tuned Lens.** Capture intermediate residuals and the final residual in a calibration run. The Python host calls `tools.representations.fit_tuned_lens(run_id, site_ids, sample_indices)` to fit `hidden + W @ hidden + bias` at each selected site while the model stays frozen. The loss compares the translated distribution with the model's final distribution. Pass the returned `decoder_ref` and `method="tuned_lens"` to `scan_lens` through Python or MCP. The artifact binds the translator to the model version and sites. On four held-out SmolLM2-135M-Instruct prompts, mean first-block KL divergence from the final distribution moved from 24.64 to 0.55. Lower values mean closer predictions. [Tuned Lens](https://arxiv.org/abs/2303.08112) · [Runnable example](../examples/tuned_lens.py)

**AtP*.** Enable `--allow-interventions --allow-gradients` on a Llama or Qwen3 model. Call `localize_behavior` with `method="atp_star"`, candidate `site_ids`, and paired texts. GradDrop sets one layer's residual-branch gradients to zero at a time and ranks residual and feed-forward input sites by the resulting effect estimates. The tool verifies up to `verify_top_k` sites with exact activation replacement. Use `review_pairs` for held-out checks and `control_site_ids` for exact controls. `prefilter` records the gradient estimates separately from `ranked_sites`, which contains measured effects. Both `forward_budget` and `backward_budget` are checked before computation. [AtP*](https://arxiv.org/abs/2403.00745) · [Runnable example with `--atp-star`](../examples/mcp_intervention.py)

## Causal circuits

On Llama and Qwen3, enable `--allow-interventions` and call `describe_model` to read `circuit_nodes`. The ordered nodes are the input embedding, each attention and feed-forward block, and the final output. An edge goes from a component that writes to the shared residual state to a later component that reads it. Pass aligned `clean_text` and `corrupt_text` with `target_token` and `alternative_token` in `pairs`. Each pair can select `positions`, or use all positions by default.

| Method | Inputs and evidence |
|---|---|
| `path_patching` | An ordered `path` ending at `output`. The tool sends a corrupted activation along that path and holds the other inputs to its receivers at their clean values. The graph labels the score as a whole-path effect |
| `acdc` | Candidate `edges` and `threshold`. The tool tests removals in reverse execution order and compares each change with the current circuit. The default score is KL divergence from the clean model output. With `review_pairs`, it scores retaining the circuit, removing it, and keeping a seeded random set of the same size. The graph draws pruned edges as dashed lines |
| `eap` | Candidate `edges` and optional `top_k`. With `--allow-gradients`, it estimates each edge effect from two forward passes and one backward pass per pair. It ranks edges by mean absolute effect and keeps the signed estimates in the report |

Use `read_artifact` to read the scores, or `render_evidence(view="graph")` with the returned `graph_artifact_id`. On the SmolLM2-135M-Instruct example, ACDC kept one of four candidate edges. On a separate pair, the output's KL divergence from the clean model was 0.00059 with the selected edge kept and 0.54039 with a same-size random selection. [Path Patching](https://arxiv.org/abs/2304.05969) · [ACDC](https://arxiv.org/abs/2304.14997) · [EAP](https://aclanthology.org/2024.blackboxnlp-1.25/) · [Runnable example](../examples/mcp_circuits.py)

### Feature attribution graphs

Circuit Tracer uses a per-layer transcoder set to approximate each feed-forward layer with sparse features. Call `trace_circuit` with `method="feature_attribution_graph"`, `variant_id`, `transcoder_id`, `text`, and `target_tokens`. The result links the signed edge matrix, its node coordinates, a drawable graph, and a separate fidelity report. The report compares the original model's next-token distribution with a forward pass using the transcoders alone. Graph nodes also show reconstruction errors, the portions of the original activations the dictionary did not capture. [Circuit Tracer](https://aclanthology.org/2025.blackboxnlp-1.14/) · [Attribution graphs methodology](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)

```sh
uv venv --python 3.11 .venv-tracer
uv pip install --python .venv-tracer/bin/python \
  'circuit-tracer==0.5.0' 'nnsight==0.6.0' 'transformers==4.57.3'
```

Enable `--allow-gradients --circuit-tracer-python /absolute/path/to/.venv-tracer/bin/python` in MCP. Python uses `allow_gradients=True` and `circuit_tracer_python`. Register a compatible Llama or Qwen3 set under `transcoders/<transcoder_id>/` in the workspace. Supply one `layer_<index>.safetensors` per layer. The `manifest.json` contains `format_version=1`, `model_kind="transcoder_set"`, the exact `variant_id`, `feature_input_hook` and `feature_output_hook` such as `mlp.hook_in` and `mlp.hook_out`, and `weights_sha256`, keyed by filename. The tool checks the binding and digests before tracing. [Example registration](../examples/fit_per_layer_transcoders.py)

```sh
python examples/mcp_feature_circuit.py --model /path/to/checkpoint \
  --dataset benchmarks/wsp-practice-original-v1/gsm8k_train.jsonl \
  --circuit-tracer-python .venv-tracer/bin/python
```

The example fits small dictionaries on SmolLM2-135M-Instruct using 32 calibration texts. On a separate prompt, Circuit Tracer produced a 219-node graph including 32 feature nodes. The raw replacement's KL divergence from the original was 4.47. This demonstrates how the report quantifies the gap between the two predictions. [Runnable example](../examples/mcp_feature_circuit.py)

## Training and structural compression

- With `--allow-edits`, `compress_model` can remove blocks with ShortGPT block influence or remove feed-forward channels with grouped Taylor scores. It exports a separate Transformers checkpoint. The grouped Taylor method also requires `--allow-gradients` and a registered training dataset. The supported structure export adapters are Llama and Qwen3. [ShortGPT](https://arxiv.org/abs/2403.03853) · [LLM-Pruner](https://arxiv.org/abs/2305.11627)
- Install the `peft` extra and pass `--allow-training --training-jsonl /path/to/cases.jsonl` to enable `train_update`. It supports supervised LoRA, DPO preference training, sequence distillation, and recovery on an exported pruned checkpoint. It returns standard PEFT files and update factors that `analyze_weight_delta` can read. Materializing a patch requires `--allow-edits`. [LoRA](https://arxiv.org/abs/2106.09685) · [DPO](https://arxiv.org/abs/2305.18290) · [Sequence distillation](https://aclanthology.org/D16-1139/)

Training rows need `id` and `problem`. Supervised training and recovery use `answer`, DPO uses `chosen` and `rejected`, and distillation uses `teacher_answer` and `teacher_model`. On an edited parent, load the matching parent model explicitly when reloading its PEFT adapter.

### Packed quantization

`compress_model` supports `gptq` and `awq` through LLM Compressor. Both calibrate on the supplied `calibration_texts` and write packed 4-bit weights in `compressed-tensors` format. GPTQ uses second-order input statistics to limit reconstruction error. AWQ scales activation-sensitive channels before quantization and saves the asymmetric zero points needed to reload the weights. `group_size` accepts 32, 64, or 128 and must divide each targeted linear layer's input width. The vocabulary head remains in its original precision. [GPTQ](https://arxiv.org/abs/2210.17323) · [AWQ](https://arxiv.org/abs/2306.00978) · [LLM Compressor](https://github.com/vllm-project/llm-compressor)

```sh
uv venv --python 3.11 .venv-quantization
uv pip install --python .venv-quantization/bin/python 'llmcompressor==0.15.0'
```

For MCP, start `wsp-mcp` with `--allow-edits --quantization-python /absolute/path/to/.venv-quantization/bin/python`. The equivalent Python option is `quantization_python` in `open_inspection`. The example below uses the local GSM8K training-source file.

```sh
python examples/mcp_quantization.py --model /path/to/checkpoint \
  --dataset benchmarks/wsp-practice-original-v1/gsm8k_train.jsonl \
  --quantization-python .venv-quantization/bin/python
```

On SmolLM2-135M-Instruct, the source weight file was 269.06 MB. GPTQ exported 113.18 MB and AWQ exported 114.03 MB, with 210 packed linear tensors each. Both reloaded through Transformers and produced finite next-token scores on CPU. [Runnable example](../examples/mcp_quantization.py)

### SliceGPT

Use `method="slicegpt"`, `calibration_texts`, and a `target_hidden_size` that is a multiple of eight below the original width. The current adapter is for Llama. SliceGPT rotates internal states toward directions measured on calibration text, then removes dimensions. The export contains SafeTensors, a slicing configuration, and the transformed shortcut matrices. Use `load_sliced_checkpoint` from `weight_programming.slicegpt_worker` in the SliceGPT environment to restore the model. [SliceGPT](https://arxiv.org/abs/2401.15024) · [Pinned upstream source](https://github.com/microsoft/TransformerCompression/tree/183cd92d8469bf3f1e979585a08dc382692bfbed)

```sh
git clone https://github.com/microsoft/TransformerCompression.git
git -C TransformerCompression checkout 183cd92d8469bf3f1e979585a08dc382692bfbed
uv venv --python 3.11 .venv-slicegpt
uv pip install --python .venv-slicegpt/bin/python ./TransformerCompression \
  'numpy<2' 'pyarrow==17.0.0' 'pandas<3'
```

Start MCP with `--allow-edits --slicegpt-python /absolute/path/to/.venv-slicegpt/bin/python`, or use `slicegpt_python` in `open_inspection`. On SmolLM2-135M-Instruct, the hidden width changed from 576 to 384 and the exported model reloaded with 127,180,800 parameters, down from 134,515,008. [Runnable example](../examples/mcp_slicegpt.py)

## Persistent Python sessions

Pass a `session_id`, `variant_id`, and `code` to `execute_python`. Calls with the same session ID retain variables and use the same model version. `load(name, region)` reads a weight slice, and `read_array(name)` reads a snapshot from the request's `input_artifacts` mapping. `save_array(name, value)` writes a declared `outputs` array. Each output name maps to its list of axis labels.

With edit permission, `edits` can turn an output array into a patch. Each edit specifies `tensor`, `op`, `value` (the output name), and an optional `region`. Use `apply_patch` to register the resulting candidate. The Python host API enables the tool with `open_inspection(..., allow_python=True)`. Use a `with` block to close the session processes at the end.

## Factual editing

`edit_facts` runs ROME for one fact at one layer or MEMIT for up to eight facts across selected layers. The tool measures feed-forward activations on `calibration_texts` and passes their second-moment matrices to EasyEdit. These matrices summarize how the input channels vary together. EasyEdit loads a separate float32 copy of the model and returns the edited weights as a patch. [ROME](https://arxiv.org/abs/2202.05262) · [MEMIT](https://arxiv.org/abs/2210.07229) · [EasyEdit](https://aclanthology.org/2024.acl-demos.9/)

Each fact has a `subject`, a `prompt` containing one `{}`, a `target_true` original answer, and a `target_new` target answer. Use `apply_patch` to create a candidate, then use `score_preferences` to compare direct questions, paraphrases, and neighboring facts. The current interface targets Llama and Qwen3; the example below was validated on SmolLM2-135M-Instruct.

Install EasyEdit separately from WSP-toolkit. The example used this pinned source with Python 3.11. [Pinned EasyEdit source](https://github.com/zjunlp/EasyEdit/tree/5c519afb14e39c7f70bc5536ebd777a62d8493bf)

```sh
git clone https://github.com/zjunlp/EasyEdit.git
git -C EasyEdit checkout 5c519afb14e39c7f70bc5536ebd777a62d8493bf
uv venv --python 3.11 .venv-easyedit
uv pip install --python .venv-easyedit/bin/python -r EasyEdit/requirements.txt safetensors
```

From the WSP-toolkit environment, run the example with a local checkpoint and the GSM8K training JSONL described below.

```sh
python examples/mcp_fact_editing.py --model /path/to/checkpoint \
  --dataset benchmarks/wsp-practice-original-v1/gsm8k_train.jsonl \
  --editor-python .venv-easyedit/bin/python --editor-source EasyEdit
```

For your own MCP session, pass `--fact-editor-python` and `--fact-editor-source` with absolute paths, along with `--allow-edits --allow-gradients`. The Python API takes `fact_editor_python`, `fact_editor_source`, `allow_edits=True`, and `allow_gradients=True` in `open_inspection()`.

On the small CPU example, the new answer's log-probability margin over the original increased by 9.49 for ROME and by 9.29 and 15.47 for the two MEMIT facts. The Eiffel Tower paraphrase margin also increased, by 3.96 and 3.62 respectively. The correct answer's margin on the unchanged France-capital question shifted by −0.07 and −0.03. [Scoring example](../examples/mcp_fact_editing.py)

## Feature dictionaries

Place `manifest.json` and `weights.safetensors` in `features/<encoder_id>/` under the workspace. The manifest binds the dictionary to a model version and a site. `extract_features` checks the binding and file digest, then reports the strongest responses and reconstruction error. An SAE analyzes one snapshot. A crosscoder encodes two aligned snapshots into shared features and reconstructs each side separately. [Released SmolLM2 dictionary](https://huggingface.co/EleutherAI/sae-SmolLM2-135M-64x) · [Crosscoder method](https://transformer-circuits.pub/2024/crosscoders/index.html)

| File | Format |
|---|---|
| `manifest.json` | `format_version=1`, `method` (`sae` or `crosscoder`), `activation` (`relu`, `topk`, or `jumprelu`), ordered `sources`, and `weights_sha256`. For TopK, supply `k`. Each source has `variant_id`, `site_id`, and optionally `input_scale` and `center_on_decoder_bias` |
| `weights.safetensors` | `b_enc[feature]`. For each source `i`, `W_enc.i[channel,feature]`, `W_dec.i[feature,channel]`, and `b_dec.i[channel]`. JumpReLU also uses `threshold[feature]` |

## Runnable examples

| Example | What it demonstrates |
|---|---|
| [Inspection](../examples/mcp_inspection.py) | The same weight slice, activation, and generation through MCP and Python |
| [Persistent Python](../examples/mcp_python.py) | Retain variables, read an activation snapshot, and publish a candidate patch in a Linux CPU session |
| [Patch](../examples/mcp_patch.py) | Edit, replay, export, reloading, merging, and pruning |
| [Intervention](../examples/mcp_intervention.py) | Activation replacement, ablation, site scanning, and optional heatmaps |
| [Update analysis](../examples/mcp_delta.py) | Measure a weight delta and compress it |
| [Evaluation](../examples/mcp_evaluation.py) | Paired development evaluation and preference scores |
| [Representation](../examples/mcp_representation.py) | Layer predictions, a linear probe, and optional gradient attribution |
| [Tuned Lens](../examples/tuned_lens.py) | Fit a residual translator and compare its held-out predictions with direct readout |
| [Attention paths](../examples/mcp_attention_rollout.py) | Attention rollout, maximum flow, and a directed graph from the same snapshot |
| [Causal circuits](../examples/mcp_circuits.py) | Measure an isolated path, prune candidate edges, and compare an EAP estimate with a small exact intervention |
| [Feature attribution graph](../examples/mcp_feature_circuit.py) | Fit a small transcoder set, trace features, and measure the replacement's output error |
| [Concept erasure](../examples/mcp_concept_erasure.py) | Fit LEACE on labeled activations and compare it with a same-size random intervention |
| [Training and compression](../examples/mcp_training.py) | Reload structurally smaller checkpoints and run the four training methods for one step each |
| [Packed quantization](../examples/mcp_quantization.py) | Export and reload GPTQ and AWQ checkpoints with packed integer weights |
| [SliceGPT](../examples/mcp_slicegpt.py) | Export a smaller hidden dimension and reload the model with its rotated shortcuts |
| [Features](../examples/mcp_features.py) | Read a published TopK SAE or fit and inspect a small crosscoder on real snapshots |
| [Factual editing](../examples/mcp_fact_editing.py) | Apply ROME and MEMIT patches and score edits, a paraphrase, and a nearby fact |

Run each script with `--help` for its inputs. The training, evaluation, representation, crosscoder, and factual editing examples use a local GSM8K training-source JSONL. The acquisition script downloads the fixed public practice set into a separate directory. [Acquisition script](../research/acquisition/download_wsp_practice.py)

```sh
python research/acquisition/download_wsp_practice.py --output ./benchmarks/wsp-practice-original-v1
```

Tool demonstrations check execution, data alignment, and saved artifacts on small inputs. The separate benchmark reports document task accuracy. [Benchmarks](benchmarks.md)

The original research interface is available through `ProgrammingSession`. [Isolated workspace guide](model-programmer.md)
