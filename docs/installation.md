# 工具库安装与可用范围

本页对应 `0.4.0a5.dev0` 源码，包含设计清单的通用工具、可解释性分析、训练与压缩方法。预发布安装包为 `0.4.0a4.dev0`。项目原创代码采用 Apache-2.0。 [当前源码](https://github.com/ZeroTang05/WSP-toolkit/tree/main) · [预发布安装包](https://github.com/ZeroTang05/WSP-toolkit/releases/tag/v0.4.0a4.dev0) · [第三方归属](../THIRD_PARTY.md)

## 按运行时安装

核心张量读写、方向操作、二阶重建不要求安装Transformers。前向后端采用互斥的精确版本extra，分别放入独立虚拟环境。

```sh
python -m pip install .                  # 核心张量与重建API
python -m pip install '.[qwen3,math]'     # Qwen3／Llama，Transformers4.57.6
python -m pip install '.[qwen35,math]'    # Qwen3.5，Transformers5.3.0，Torch>=2.4
```

后两条是两个环境的选项，不能在同一环境同时安装qwen3与qwen35。服务器原作环境使用Torch2.5.1+cu121；GPU构建版本按实际驱动选择，不能让默认依赖安装替换共享实验环境。旧研究CLI与固定lm-eval轨道使用qwen3／benchmark extra。

## 本地 MCP 与 Python 只读工具

从当前源码安装本地 MCP 服务，传入一个已经下载的检查点和独立工作目录。

```sh
python -m pip install '.[qwen3,mcp]'
wsp-mcp --model /path/to/checkpoint --workspace /path/to/workspace
```

MCP 客户端将 `wsp-mcp` 配置为本地进程。`tools/list` 会发现 `describe_model`、`model_forward`、`read_artifact`、`list_tensors`、`read_tensor` 和 `get_run`。研究脚本使用 `open_inspection(model_dir, workspace).registry` 调用同一工具。模型加载时由宿主登记检查点。Hub 仓库用 `--revision` 指定完整提交号。

下列示例在真实模型上用两种入口分别运行一次，比较生成、激活和权重切片。

```sh
python examples/mcp_inspection.py --model /path/to/checkpoint
```

`describe_model` 先返回基座版本和可观测站点。`model_forward` 将实际采集的数值保存到工作目录，返回产物 ID。`read_artifact` 根据 ID 和切片坐标读取少量数值。这样大矩阵只存一份，调用方可以选自己需要的部分。MCP 启动与 Python 示例使用各自的运行目录和预算。

已在真实 SmolLM2-135M-Instruct（Llama 架构）上运行该示例。两种入口发现的工具一致，读取同一份配置下的 `1 × 5 × 576` 激活、BF16 权重和生成回放，切片数值一致。

## 持久 Python 会话

`--allow-python` 在 Linux 上开放 `execute_python`，要求 Bubblewrap 和虚拟环境。Python 入口用 `open_inspection(..., allow_python=True)`。同一个 `session_id` 会保留变量并固定到一个 `variant_id`。查看另一个候选时使用新的会话 ID。

- `load(name, region)` 从该版本读取权重切片。省略 `region` 时读取整个张量。
- `input_artifacts` 把已登记的数组 ID 绑定到本次调用的名字。代码用 `read_array(name)` 读取。
- `outputs` 把输出名字和轴标签告诉宿主。代码用 `save_array(name, value)` 提交数组。
- `edits` 可以引用输出名字生成补丁，需同时开启 `--allow-edits`。应用补丁仍使用 `apply_patch`。

`get_run` 可以查结果和用量。工作区的 `python/runs` 保存每个单元的代码和运行记录。关闭工具实例时回收隔离进程。真实 Qwen3-0.6B 示例核对了权重读数、变量保留、前向快照与补丁应用后的候选读数。 [运行示例](../examples/mcp_python.py) · [Bubblewrap](https://github.com/containers/bubblewrap)

## 权重补丁、激活干预与图表

启动 MCP 服务时用 `--allow-edits` 开放补丁、投影、合并和剪枝。Python 调用 `open_inspection` 时传 `allow_edits=True`。`create_patch` 保存原精度替换张量与清单，`apply_patch` 返回可供前向和权重读取使用的候选 ID。`revert_patch` 返回父版本 ID，下一次运行传这个 ID 即可。小切片可以内联 `values`，已有数组可以传 `value_artifact_id`。

`export_patch` 返回相对于工作目录的补丁包路径。将包复制到新工作目录的 `imports/<patch_id>`，用 `import_patch` 校验，再按同一父版本应用。`export_model` 产出完整 Transformers 检查点，保留原配置和分词器，只重写改动的 Safetensors 分片。这个例子从创建一路执行到重载，也调用了方向投影、任务向量平均、TIES、DARE、Wanda 和两种 SparseGPT 路径。DARE 先用 `drop_probability` 随机丢弃部分增量，按保留概率重缩放后再合并。固定种子可重放掩码。 [DARE](https://arxiv.org/abs/2311.03099)

```sh
python examples/mcp_patch.py --model /path/to/checkpoint
```

`--allow-interventions` 单独授权激活替换、置零消融、方向注入和站点扫描。方向注入使用 `method="direction"`，传 `direction` 或 `direction_artifact_id` 及 `strength`。方向在所选通道归一化后按强度加到激活上。安装 `analysis` 可选依赖后，`render_evidence` 读取已登记数组并输出 PNG 或 PDF。 [对比激活加法](https://arxiv.org/abs/2312.06681)

```sh
python -m pip install '.[qwen3,mcp,analysis]'
python examples/mcp_intervention.py --render --model /path/to/checkpoint
```

两个示例都在真实 SmolLM2 上运行。补丁用一个 BF16 权重验证，撤销恢复了基座读数，重放和完整检查点重载复现了候选。干预示例验证了两个站点共用基线，并生成了激活、注意力和干预效果热力图。

## 差分、开发集评测与表示分析

- `analyze_weight_delta` 比较同基座候选、WSP LoRA 因子或由宿主导入的原生 PEFT LoRA。返回逐张量及逐层变化、主方向和原精度舍入误差。`compress_update` 用普通或输入加权的低秩截断保存更小的更新，按需生成可应用的密集补丁。压缩的是更新，密集检查点仍保留原形状。
- `evaluate_variants` 由宿主使用 `--development-jsonl` 和 `--dataset-id` 登记开发集。当前从模型回答中提取带框答案，再与标准答案做文本匹配。每个版本共用题目和采样种子，报告可分页的逐样本结果及按题重抽样的区间。`score_preferences` 从真实前向计算两段回答的对数概率差，再减去参考版本的分差。
- `compare_models` 支持同基座权重差分、内部表示相似度和开发集行为比较。宿主可用 `--comparison-model` 加载第二个本地检查点。表示比较使用相同原始文本及各自最后一个有效词元。第二个工作区的结果用 `workspace="comparison"` 读取。
- `scan_lens` 用模型自己的最终归一化和词表头读取各层。安装 `analysis` 后，`probe_representation` 在显式分开的样本上比较真实标签、控制标签和输入嵌入基线。安装 `gradients` 并加 `--allow-gradients` 后，`attribute_output` 用积分梯度计算词元贡献，返回贡献总和与目标分数变化的差。

```sh
python -m pip install '.[qwen3,mcp,analysis,gradients]'
python examples/mcp_delta.py --model /path/to/checkpoint
python examples/mcp_evaluation.py --cross-checkpoint --model /path/to/checkpoint --dataset benchmarks/wsp-practice-original-v1/gsm8k_train.jsonl
python examples/mcp_representation.py --attribution --model /path/to/checkpoint --dataset benchmarks/wsp-practice-original-v1/gsm8k_train.jsonl
```

这些示例在真实 SmolLM2 和 GSM8K 训练来源样本上简单调用工具。最后一层读出的前三个词元与前向结果一致。两个分别加载的检查点复现了同一基座候选比较的相似度。层级读出支持直接读出与已登记的调优透镜，两者共用模型最终输出头。

## 跨层注意力传播与最大流

先用 `describe_model` 查询 `trace_circuit_methods`。在支持的模型上，`model_forward` 采集从第 0 层到目标层的普通自注意力矩阵。调用 `trace_circuit` 时传 `run_id` 和方法。`attention_rollout` 返回所有词元的跨层传播分数，上限 256 个词元。`attention_flow` 需要 `analysis` 依赖中的 NetworkX，额外传 `query_position`，计算指定位置到各输入词元的最大流，上限 32 个词元。两个结果都能用 `read_artifact` 读取、用 `attention_heatmap` 绘图，并标记为注意力权重的描述性证据。 [方法论文](https://aclanthology.org/2020.acl-main.385/)

```sh
python -m pip install '.[qwen3,mcp,analysis]'
python examples/mcp_attention_rollout.py --model /path/to/checkpoint
```

示例在真实 SmolLM2-135M-Instruct 上读取 30 层注意力，核对首层与原始矩阵的计算关系，并生成 `5 × 5` 传播热力图和 `1 × 5` 最大流热力图。 [已有实现与合并路线](tool-design.md#9-已有实现与合并路线) · [机器可读工具目录](tool-catalog.json)

## 结构压缩与适配器训练

- `compress_model` 用 `--allow-edits` 开放。`block_influence` 从校准文本计算各层前后内部状态的变化，删除得分最低的指定数量层。`taylor_mlp` 还需要 `--allow-gradients` 和 `--training-jsonl`，按权重与梯度给前馈通道排序，再同时缩短相关矩阵的维度。它沿用仓库的分组方法，保留宽度按 128 对齐。工具返回完整检查点、参数量和清单路径。 [ShortGPT](https://arxiv.org/abs/2403.03853) · [LLM-Pruner](https://arxiv.org/abs/2305.11627)
- `train_update` 需要 `peft` 依赖、`--allow-training` 和 `--training-jsonl`。样本用 `id` 引用，`problem` 是提示。LoRA 和剪后恢复读取 `answer`，DPO 读取 `chosen` 与 `rejected`，序列蒸馏读取 `teacher_answer` 与 `teacher_model`。剪后恢复从 `compress_model` 的导出检查点打开工作区。标准 PEFT 文件与 WSP 更新因子同时保存，`materialize=true` 需要编辑授权。若在候选版本上训练，重载 PEFT 文件时需显式提供对应父模型。 [LoRA](https://arxiv.org/abs/2106.09685) · [DPO](https://arxiv.org/abs/2305.18290) · [序列蒸馏](https://aclanthology.org/D16-1139/)

```sh
python -m pip install '.[qwen3,mcp,gradients,peft]'
python examples/mcp_training.py --model /path/to/checkpoint --dataset benchmarks/wsp-practice-original-v1/gsm8k_train.jsonl
```

在真实 SmolLM2-135M-Instruct 上，删一层后参数量由 134,515,008 降至 130,974,912。前馈通道剪枝后为 121,243,968，前馈宽度从 1,536 降至 1,280。两份检查点均重新加载。LoRA、DPO、蒸馏和剪后恢复各执行一步，LoRA 适配器重载后与工具包保存的有效更新一致。 [训练与压缩示例](../examples/mcp_training.py)

## 稀疏特征与双模型比较

宿主将字典放入 `features/<encoder_id>/`，目录里是 `manifest.json` 和 `weights.safetensors`。`describe_model` 列出和当前版本匹配的字典。`model_forward` 采集字典指定的站点，再把快照 ID 和字典 ID 传给 `extract_features`。结果包含最强触发词元、平均活跃特征数、逐来源重建误差，以及可读的特征响应数组。数组可用 `render_evidence` 的 `activation_heatmap` 视图绘制，指定轴为 `token_row` 和 `feature`。

| 字典文件 | 内容 |
|---|---|
| `manifest.json` | `format_version=1`，`method` 为 `sae` 或 `crosscoder`，`activation` 为 `relu`、`topk` 或 `jumprelu`。TopK 需传 `k`。`sources` 按输入顺序登记 `variant_id`、`site_id`，以及可选的 `input_scale`、`center_on_decoder_bias`。`weights_sha256` 保存权重文件的校验摘要 |
| `weights.safetensors` | `b_enc` 是特征偏置。每个来源 `i` 的 `W_enc.i` 为 `[通道, 特征]`，`W_dec.i` 为 `[特征, 通道]`，`b_dec.i` 为 `[通道]`。JumpReLU 额外提供 `threshold[特征]` |

`compare_models` 的 `crosscoder` 视图接收 `encoder_id`、`left_artifact_id`、`right_artifact_id`，两份快照必须有相同的输入词元序列。这个视图调用同一个特征提取函数。公开 SAE 示例读取 SmolLM2 基座第 3 层前馈模块输出，使用 EleutherAI 发布的字典。联合字典示例用 SmolLM2-Instruct 和其真实权重候选拟合小字典，再在不同的输入上比较两边。 [基础检查点](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) · [公开字典](https://huggingface.co/EleutherAI/sae-SmolLM2-135M-64x) · [联合字典论文](https://transformer-circuits.pub/2024/crosscoders/index.html) · [两种用法](../examples/mcp_features.py)

```sh
python examples/mcp_features.py --model /path/to/SmolLM2-135M --sae /path/to/layers.3.mlp
python examples/mcp_features.py --model /path/to/SmolLM2-135M-Instruct --dataset benchmarks/wsp-practice-original-v1/gsm8k_train.jsonl
```

## 隐藏维度切片与打包量化

`compress_model` 的 SliceGPT、GPTQ 和 AWQ 共用校准文本与导出清单。SliceGPT 用单独的上游环境改变隐藏维度，重载时恢复切片后的跳连矩阵。GPTQ 和 AWQ 使用独立的 LLM Compressor 环境导出 4 位整数权重，清单包含原文件字节、实际导出字节和打包张量数。三种方法都已在 SmolLM2-135M-Instruct 上重载并执行前向。 [安装与运行](mcp-tools.md#packed-quantization--打包量化) · [SliceGPT 示例](../examples/mcp_slicegpt.py) · [GPTQ 与 AWQ 示例](../examples/mcp_quantization.py)

## 特征归因图

`trace_circuit(method="feature_attribution_graph")` 需要宿主登记独立的 Circuit Tracer 环境和匹配模型版本的逐层特征字典，并授权梯度。工具输出完整边矩阵、节点坐标、可绘图摘要，以及只用字典替代前馈层后的输出偏差。已在真实 SmolLM2-135M-Instruct 上生成并核对图。 [安装与运行](mcp-tools.md#feature-attribution-graphs--特征归因图) · [示例](../examples/mcp_feature_circuit.py)

## 知识编辑

`edit_facts` 用 ROME 编辑一条事实，用 MEMIT 批量编辑事实。宿主登记 EasyEdit 的源码和独立环境，并授权梯度与权重编辑。工具用校准文本采集真实激活，保存算法改动为补丁。应用补丁后，可以比较直接问题、同义问题和邻近事实。已在 SmolLM2-135M-Instruct 的 CPU 示例中验证。 [安装与运行](mcp-tools.md#factual-editing--知识编辑) · [示例](../examples/mcp_fact_editing.py)

## API与证据

| 接口 | 作用 | 已有证据 |
|---|---|---|
| WeightStore、EditPolicy | 原始／当前张量、提案、撤销、原精度导出 | 源不变、精度物化及编辑约束检查 |
| contrast_direction、project_output_direction | 匹配均值差及选列输出方向操作 | 明确代数、列支持与方向符号不变性 |
| input_hessian、sparsegpt_reconstruct | 二阶统计、固定mask或自适应剪枝补偿 | 作者代码数值对照与真实0.6B十二条件实验 |
| ForwardService、ForwardPolicy、PracticeSet | 固定采样、预算及数据角色的生成／teacher-forced | Qwen3／Llama与Qwen3.5混合架构检查；真实基座观测 |
| HuggingFaceBackend | Qwen3／Llama密集观测及生成 | 真实0.6B 128文本、27,776 tokens |
| Qwen35Backend | 原作Qwen3.5文本执行及混合层观测 | 473张量逐项核对；32文本、6,528 tokens；不执行15个MTP张量 |
| CPUKernel、ProgrammingSession | 持久CPU分析、隔离文件／网络／设备及与前向服务连接 | 实际0.6B内核与编辑—观测通道已运行 |

两个后端从各自模块导入，宿主负责加载模型与设备调度：

```python
from weight_programming import WeightStore, ProgrammingSession, sparsegpt_reconstruct
from weight_programming.hf_backend import HuggingFaceBackend       # qwen3 extra
# 或在另一环境：
from weight_programming.qwen35_backend import Qwen35Backend        # qwen35 extra
```

CPU隔离内核需要Linux bubblewrap和虚拟环境Python；macOS可使用核心张量API。库不自行安排GPU作业。源码中的研究矩阵脚本不是全部已整理为公共API；OfficeBench执行器仍处于单独研究模块。

所有真实性能改善、完整任务与效率证据分别记录。真实观测完成不表示数学能力改善，局部重建不表示整模型保真，接口覆盖也不表示已经完成独立MIT复现。

## 分类器与原生适配器

DistilBERT 单标签分类器用 `model_forward(mode="predict")` 返回标签分数、概率和预测标签。它复用快照、热力图、探针、表示比较和权重补丁。SST-2 示例的 16 条真实输入与 Transformers 直接推理分数逐项一致。分类器与 SmolLM2 还通过了同文本的跨架构 CKA（按样本比较表示相似结构）。 [分类示例](../examples/mcp_classification.py) · [数据获取](../research/acquisition/prepare_sst2.py)

原生 PEFT LoRA 用启动选项 `--peft-adapter` 或 Python 的 `peft_adapters` 登记。已有候选的适配器可由宿主调用 `tools.delta.import_peft_adapter(parent_variant_id, adapter_dir)` 绑定。它沿用 PEFT 的实际缩放与转置语义，并进入现有差分、合并和压缩接口。`compress_update` 可为不同张量设置不同秩；`project_weight` 可直接消费两组同坐标激活。 [运行说明](mcp-tools.md#imported-adapters-and-directions--外部适配器与方向) · [原生适配器示例](../examples/mcp_native_peft.py)
