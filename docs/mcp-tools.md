# MCP and Python tools · MCP 与 Python 工具

The same tool definitions serve a local Model Context Protocol (MCP) client and Python. Start with `describe_model` to find the model's version ID and observation sites. A site names a specific internal location, such as the output of a feed-forward layer.

两种入口共用同一套工具。先调用 `describe_model` 获取模型版本和观测位置，再按需要读取权重或运行模型。每次运行会把大数组和明细保存到工作区。返回的产物 ID 可以用来读取、比较或绘图。

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

| Task · 任务 | Tools | Setup |
|---|---|---|
| Discover, run, and read · 发现、前向和读取 | `describe_model`, `model_forward`, `list_tensors`, `read_tensor`, `read_artifact`, `get_run` | Default. `model_forward` supports generation and scoring given text |
| Edit and replay · 补丁与重放 | `create_patch`, `apply_patch`, `revert_patch`, `export_patch`, `import_patch`, `export_model` | `--allow-edits`. Edits are saved as workspace artifacts linked to their parent version |
| Observe, explain, and compare · 表示、归因和比较 | `scan_lens`, `compare_models`, `trace_circuit`, `extract_features`, `probe_representation`, `render_evidence`, `attribute_output` | `analysis` adds probes, heatmaps, and attention maximum flow. `gradients` with `--allow-gradients` enables integrated gradients. Feature extraction needs a registered dictionary |
| Intervene · 临时干预 | `run_intervention`, `localize_behavior` | `--allow-interventions`. Replace, zero, or shift selected activations and record the effect |
| Run Python · 隔离 Python | `execute_python` | `--allow-python` on Linux with Bubblewrap and a virtual environment. Use `--allow-edits` to create a patch from an output array |
| Analyze or combine updates · 分析与组合更新 | `analyze_weight_delta`, `project_weight`, `merge_updates`, `prune_and_compensate`, `compress_update` | Delta analysis is read-only. Other tools require `--allow-edits`. Methods include TIES, DARE, Wanda, SparseGPT, and low-rank update compression |
| Compare answers · 比较回答 | `score_preferences`, `evaluate_variants` | Register development rows with `--development-jsonl` for `evaluate_variants`. Rows contain `id`, `problem`, and `answer` |
| Compress or train · 结构压缩与训练 | `compress_model`, `train_update` | See the settings below |
| Edit factual associations · 知识编辑 | `edit_facts` | `--allow-edits --allow-gradients` and a separate EasyEdit environment |

For comparisons across separately loaded checkpoints, use `--comparison-model` and select `right_workspace="comparison"`. Crosscoder comparison also takes the two snapshot IDs and checks that their input token sequences align.

For activation direction injection, set `run_intervention.method` to `direction`. Pass `strength` and either `direction`, a list with one value per channel, or `direction_artifact_id`, the ID of a one-dimensional `channel` array. The tool normalizes the direction across the selected channels before scaling it and adding it to the specified token positions. [Contrastive Activation Addition](https://arxiv.org/abs/2312.06681) · [Runnable example](../examples/mcp_intervention.py)

## Training and structural compression

- With `--allow-edits`, `compress_model` can remove blocks with ShortGPT block influence or remove feed-forward channels with grouped Taylor scores. It exports a separate Transformers checkpoint. The grouped Taylor method also requires `--allow-gradients` and a registered training dataset. The supported structure export adapters are Llama and Qwen3. [ShortGPT](https://arxiv.org/abs/2403.03853) · [LLM-Pruner](https://arxiv.org/abs/2305.11627)
- Install the `peft` extra and pass `--allow-training --training-jsonl /path/to/cases.jsonl` to enable `train_update`. It supports supervised LoRA, DPO preference training, sequence distillation, and recovery on an exported pruned checkpoint. It returns standard PEFT files and update factors that `analyze_weight_delta` can read. Materializing a patch requires `--allow-edits`. [LoRA](https://arxiv.org/abs/2106.09685) · [DPO](https://arxiv.org/abs/2305.18290) · [Sequence distillation](https://aclanthology.org/D16-1139/)

Training rows need `id` and `problem`. Supervised training and recovery use `answer`, DPO uses `chosen` and `rejected`, and distillation uses `teacher_answer` and `teacher_model`. On an edited parent, load the matching parent model explicitly when reloading its PEFT adapter.

## Persistent Python sessions

Pass a `session_id`, `variant_id`, and `code` to `execute_python`. Calls with the same session ID retain variables and use the same model version. `load(name, region)` reads a weight slice, and `read_array(name)` reads a snapshot from the request's `input_artifacts` mapping. `save_array(name, value)` writes a declared `outputs` array. Each output name maps to its list of axis labels.

With edit permission, `edits` can turn an output array into a patch. Each edit specifies `tensor`, `op`, `value` (the output name), and an optional `region`. Use `apply_patch` to register the resulting candidate. The Python host API enables the tool with `open_inspection(..., allow_python=True)`. Use a `with` block to close the session processes at the end.

## Factual editing · 知识编辑

`edit_facts` runs ROME for one fact at one layer or MEMIT for up to eight facts across selected layers. The tool measures feed-forward activations on `calibration_texts` and passes their second-moment matrices to EasyEdit. These matrices summarize how the input channels vary together. EasyEdit loads a separate float32 copy of the model and returns the edited weights as a patch. [ROME](https://arxiv.org/abs/2202.05262) · [MEMIT](https://arxiv.org/abs/2210.07229) · [EasyEdit](https://aclanthology.org/2024.acl-demos.9/)

每条事实用 `subject`（主体）、`prompt`（含一个 `{}` 的句式）、`target_true`（原答案）和 `target_new`（目标答案）表示。用 `apply_patch` 创建候选版本，再用 `score_preferences` 比较直接提问、改写后的提问和邻近事实。当前接入口面向 Llama 和 Qwen3，下面的示例在 SmolLM2-135M-Instruct 上验证。

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
| [Attention paths](../examples/mcp_attention_rollout.py) | Attention rollout and maximum flow from the same snapshot |
| [Training and compression](../examples/mcp_training.py) | Reload structurally smaller checkpoints and run the four training methods for one step each |
| [Features](../examples/mcp_features.py) | Read a published TopK SAE or fit and inspect a small crosscoder on real snapshots |
| [Factual editing](../examples/mcp_fact_editing.py) | Apply ROME and MEMIT patches and score edits, a paraphrase, and a nearby fact |

Run each script with `--help` for its inputs. The training, evaluation, representation, crosscoder, and factual editing examples use a local GSM8K training-source JSONL. The acquisition script downloads the fixed public practice set into a separate directory. [Acquisition script](../research/acquisition/download_wsp_practice.py)

```sh
python research/acquisition/download_wsp_practice.py --output ./benchmarks/wsp-practice-original-v1
```

Tool demonstrations check execution, data alignment, and saved artifacts on small inputs. The separate benchmark reports document task accuracy. [Benchmarks](benchmarks.md)

## Method roadmap · 方法路线

The existing tools are the entry points for the following planned methods.

| Tool family · 工具类别 | Planned methods · 计划接入 |
|---|---|
| Intervene and localize · 干预与定位 | Concept erasure, AtP* |
| Inspect and render · 解释与绘图 | Tuned Lens, contrastive sequence attribution, graph rendering |
| Trace circuits · 电路追踪 | Path patching, ACDC, EAP, feature attribution graphs |
| Compress · 压缩 | SliceGPT, GPTQ, AWQ |

The original research interface is available through `ProgrammingSession`. [Isolated workspace guide](model-programmer.md)
