# Getting started · 上手指南

中文读者可以先在本地准备一个 safetensors 格式的模型文件夹。安装工具包后，用第 2 节的命令读取一块权重。命令里的 `/path/to/Qwen3-0.6B` 换成你的模型目录，`./weight-edits` 用来保存修改记录。只读取权重时，普通电脑的 CPU（处理器）即可运行。

## 1. Install the toolkit

Use Python 3.10 or newer in a virtual environment. Start from the [repository](https://github.com/ZeroTang05/WSP-toolkit) and run the installation command in its root folder.

| Use case | Install from source |
|---|---|
| Read, edit, merge, or reconstruct weight matrices on CPU | `python -m pip install .` |
| Run Qwen3 or Llama with the model adapter | `python -m pip install '.[qwen3,math]'` |
| Connect a local MCP client to Qwen3 or Llama | `python -m pip install '.[qwen3,mcp]'` |
| Inspect a DistilBERT classifier with snapshots and probes | `python -m pip install '.[qwen3,mcp,analysis]'` |
| Run Qwen3.5 with the model adapter | `python -m pip install '.[qwen35,math]'` |
| Run the model programmer with its numerical analysis libraries | `python -m pip install '.[qwen35,math,analysis]'` |

The Qwen3 and Qwen3.5 adapters use different pinned Transformers versions. Install each in its own environment. Qwen3.5 requires PyTorch 2.4 or newer. The isolated analysis workspace requires Linux and Bubblewrap, which creates a restricted filesystem for Python. The provided full MATH-500 runner uses a CUDA GPU. [Model programmer setup](model-programmer.md) · [Full evaluation setup](final-evaluation.md)

The Python distribution is `idea4-weightlab` and the import path is `weight_programming`. The MCP and shared Python tools use the current source installation. [Earlier prerelease wheel](https://github.com/ZeroTang05/WSP-toolkit/releases/tag/v0.4.0a4.dev0)

## 2. Inspect a real checkpoint

Supply a local model folder containing its safetensors weight files. For a sharded checkpoint, include `model.safetensors.index.json` alongside the shards. The example defaults to a weight name from Qwen3-0.6B. Use `--name` for a different tensor.

```sh
python examples/inspect_weight.py --model /path/to/Qwen3-0.6B --edit-dir ./weight-edits
```

`WeightStore` locates the tensor in the checkpoint and links the edit directory to that checkpoint's file fingerprints. `load_original` reads a copy of the tensor as a 32-bit floating-point matrix. The example reports its shape and norm. It runs on the CPU and creates a manifest in the edit directory.

## 3. Save and evaluate an edit

After your analysis produces a replacement tensor, pass it to `store.save_edit(name, replacement)`. The store checks the tensor's shape, numeric values, and edit permissions. Use `store.drop_edit(name)` to discard a proposed edit. Use `store.materialize_edits(output_dir)` to export the selected replacement tensors in the checkpoint's original precision. The export contains the edited tensors and a manifest identifying their base checkpoint.

For a controlled comparison, freeze the edit and keep the questions, prompt, sampling settings, and scoring rules the same across models. Keep the evaluation answers with the evaluator. [Final math evaluation](final-evaluation.md)

## 4. Find the right entry point

- **MCP.** Run `wsp-mcp` as a local standard-input/output process from an MCP client. The same tool registry is available through `open_inspection()` in Python. [Setup, permissions, and examples](mcp-tools.md)
- **Python API.** The `weight_programming` package contains the reusable tools for weight operations, model observations, and evaluation.
- **Research scripts.** The `src/run_*.py` scripts run specific experiments and expect the models, data views, or trained updates described in their command help. Use `PYTHONPATH=src python src/run_ties_study.py --help` to see one example.
- **Method contracts.** The files under `contracts/` and the tool permissions guide describe how each operation relates to the research method. [Tool permissions](tool-permissions.md)
- **Tool design and implementation.** Each catalog entry records its sources, implementation, and runnable examples. [Detailed design](tool-design.md) · [Implementation and handoff](tool-inventory.md)

## 5. Check your installation

The core checks cover weight loading, edit validation, pruning reconstruction, and TIES merging. Run them from the repository root.

```sh
PYTHONPATH=src:tests python -m unittest test_weight_programming test_reconstruction test_merging
```

Use the Qwen3 environment for `test_forward_service`, `test_activation_patching`, and `test_math_evaluation`. Use the Qwen3.5 environment for `test_qwen35_backend`. `test_cpu_kernel`, `test_programming_session`, and `test_programmer` also require Linux and Bubblewrap. Install `.[test,math]` in the appropriate environment for schema checks and math scoring.
