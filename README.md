# WSP-toolkit

**Give an AI model tools to inspect, edit, and evaluate another model.**

WSP stands for **Weight-Space Programming**. The toolkit wraps PyTorch, Hugging Face Transformers, Safetensors, PEFT, and established research methods into tools for AI clients and researchers. Read model weights and internal activations, make a targeted change, and compare the resulting behavior with the base model. Edits are saved separately, with a fingerprint that ties each edit to its source checkpoint.

[Getting started](docs/getting-started.md) · [MCP and Python tools](docs/mcp-tools.md) · [Benchmarks](docs/benchmarks.md) · [Releases](https://github.com/ZeroTang05/WSP-toolkit/releases)

[Project status](PROJECT_STATUS.md) · [Tool inventory](docs/tool-inventory.md) · [Repository guide](docs/maintaining.md)

## What you can do

| Goal | Tool |
|---|---|
| Read weights and save reversible edits | Read tensor slices and create, replay, export, or revert patches |
| Observe a model processing text | Record activations and attention. Use heatmaps, layer predictions, attribution, or sparse feature dictionaries to inspect the results |
| Measure what an internal component changes | Temporarily replace, zero, or shift selected activations and compare the model's output |
| Trace a prediction through the model | Test residual paths with Path Patching and ACDC, estimate edges with EAP, or draw a Circuit Tracer feature graph |
| Edit a model's factual associations | Create a ROME or MEMIT patch and measure how it changes the model's answers |
| Compare models or compress updates | Compare weight changes, representations, and behavior. Merge updates with TIES or DARE, or compress them with a low-rank approximation |
| Inspect a text classifier or import trained weights | Analyze DistilBERT class scores and snapshots, or import native PEFT LoRA adapters into the same update tools |
| Make a smaller model or train an update | Export a checkpoint with fewer layers or channels, use SliceGPT or packed GPTQ and AWQ quantization, or train a LoRA adapter |

The local Model Context Protocol (MCP) server and Python registry use the same tool definitions. Model adapters report which internal locations are available for each checkpoint. [Tool reference and examples](docs/mcp-tools.md)

The design catalog covers 34 tool entries and 47 named methods. Capabilities are verified separately for Llama, Qwen3, Qwen3.5, and DistilBERT. [Design and method sources](docs/tool-design.md) · [Tool inventory](docs/tool-inventory.md)

On Linux, an isolated Python session can retain variables across tool calls, read model snapshots, and propose a weight patch. [Persistent Python sessions](docs/mcp-tools.md#persistent-python-sessions)

The original Weight-Space Programming experiment inspired the isolated workspace and budgeted model tools. Activation patching, pruning, and update merging have their own method references and experiment settings. [Original research](https://rikiyat.github.io/blog/weight-space-programming/) · [Tool permissions and methods](docs/tool-permissions.md)

## A measured result

On the full **1,319-question GSM8K** test set, Qwen3-0.6B went from **485 correct (36.77%)** to **600 correct (45.49%)** after three trained updates were combined with TIES and scaled to the same size as their ordinary average. That is **+8.72 percentage points**. The unscaled TIES update scored **417/1,319 (31.61%)**, so update size mattered in this experiment. [All five conditions and the evaluation setup](docs/benchmarks.md)

## Get started

Use Python 3.10 or later. Install from source to use the interfaces described here.

```sh
git clone https://github.com/ZeroTang05/WSP-toolkit.git
cd WSP-toolkit
python -m pip install '.[qwen3,mcp]'
```

Use a local Qwen3 or Llama-family safetensors checkpoint. The example runs on a CPU and checks tool discovery, weight slices, and activations through both interfaces.

```sh
python examples/mcp_inspection.py --model /path/to/checkpoint
```

The examples were exercised with SmolLM2-135M-Instruct. The installation guide lists model environments and a direct weight reader. [Getting started](docs/getting-started.md) · [SmolLM2 checkpoint](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)

| Guide | What it covers |
|---|---|
| [Model programmer](docs/model-programmer.md) | Give a model the inspection and editing tools and record its work |
| [TIES merging](docs/ties-merging.md) | Merge trained updates and control their size |
| [Final math evaluation](docs/final-evaluation.md) | Evaluate frozen weights on a separate test set |
| [Benchmark results](docs/benchmarks.md) | Compare all tested conditions and see which studies are still open |

## License and references

Project code is released under [Apache-2.0](LICENSE). [Third-party notices](THIRD_PARTY.md) document the upstream code and its licenses. Model checkpoints and datasets have their own terms.
