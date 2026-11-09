# WSP-toolkit 实现清单与交接

2026-11-09。当前源码版本为 `0.4.0a5.dev0`。设计目录中的 **34 个工具条目、47 种具名方法**已接入。模型注册由宿主完成，工具发现由 Python 登记表和 MCP 原生接口提供；模型可调用的工具按实际后端、依赖和授权开放。

入口：[英文首页](../README.md) · [中文首页](../README.zh-CN.md) · [安装与运行](../docs/mcp-tools.md) · [详细设计](wsp-toolkit-tool-design.md) · [机器可读目录](toolkit-design-catalog.json)。

## 工具如何组合

```mermaid
flowchart LR
    A[大模型或研究脚本] --> B[MCP 与 Python 共用登记表]
    B --> C[权重读取与版本补丁]
    B --> D[前向与内部快照]
    C --> E[差分、方向、合并与压缩]
    D --> F[表示、探针、归因与回路]
    E --> G[候选模型与配对评测]
    F --> G
    G --> H[可重放补丁、模型文件与证据产物]
```

基础层封装 PyTorch、Transformers、Safetensors 和 PEFT。分析层复用已有算法和保存的快照。不同方法生成相同格式的补丁或产物，后续工具可以直接引用它们的 ID。

## 逐项实现与简单验证

每个示例使用真实模型、真实前向或真实训练样本。表中同一行共享示例，方法通过原工具的参数选择。

| 工具条目 | 已接入能力 | 实现与运行示例 |
|---|---|---|
| `register_model`、`describe_model` | 固定检查点、指纹、模型能力与观测位置 | [注册](../src/weight_programming/model_registry.py)、[服务](../src/weight_programming/inspection.py)；[双入口示例](../examples/mcp_inspection.py) |
| `list_tensors`、`read_tensor` | 分页列权重、读取原精度切片和来源版本 | [存储](../src/weight_programming/tensor_store.py)；[示例](../examples/mcp_inspection.py) |
| `create_patch`、`apply_patch`、`revert_patch` | 原精度补丁、父版本验证、候选与撤销 | [版本与补丁](../src/weight_programming/patches.py)；[完整示例](../examples/mcp_patch.py) |
| `export_patch`、`import_patch`、`export_model` | 跨工作区重放及完整检查点导出、重载 | [实现](../src/weight_programming/patches.py)；[示例](../examples/mcp_patch.py) |
| `model_forward` | 生成、给定文本评分、分类预测；残差、前馈和注意力快照 | [前向服务](../src/weight_programming/forward_service.py)、[编码器](../src/weight_programming/encoder_backend.py)；[生成示例](../examples/mcp_inspection.py)、[分类示例](../examples/mcp_classification.py) |
| `read_artifact`、`get_run`、`list_tools` | 数组切片、明细分页、运行状态、用量与实际可用工具 | [产物](../src/weight_programming/artifacts.py)、[登记表](../src/weight_programming/tool_registry.py)；[示例](../examples/mcp_inspection.py) |
| `execute_python` | Linux 隔离会话、变量保留、权重与数组交换、补丁提案 | [实现](../src/weight_programming/python_tools.py)；[真实 Qwen3 示例](../examples/mcp_python.py) |
| `analyze_weight_delta` | 候选或原生 PEFT LoRA 差分、逐层变化、精确及随机低秩方向、实际舍入误差 | [实现](../src/weight_programming/delta_tools.py)；[差分示例](../examples/mcp_delta.py)、[PEFT 示例](../examples/mcp_native_peft.py) |
| `project_weight` | 显式方向或两组激活均值差；验证输出坐标后投影所选列 | [服务](../src/weight_programming/inspection.py)、[已有方向算法](../src/weight_programming/directions.py)；[示例](../examples/mcp_delta.py) |
| `merge_updates` | 补丁与已登记适配器的任务向量加权、TIES、DARE | [实现](../src/weight_programming/editing_tools.py)；[补丁示例](../examples/mcp_patch.py)、[适配器示例](../examples/mcp_native_peft.py) |
| `prune_and_compensate` | Wanda、固定掩码 SparseGPT、自适应 SparseGPT | [封装](../src/weight_programming/editing_tools.py)、[重建算法](../src/weight_programming/reconstruction.py)；[示例](../examples/mcp_patch.py) |
| `compress_update` | 普通或输入加权低秩截断、逐张量秩预算、可选密集补丁 | [实现](../src/weight_programming/delta_tools.py)；[示例](../examples/mcp_delta.py)、[分配秩的示例](../examples/mcp_native_peft.py) |
| `evaluate_variants`、`score_preferences` | 开发集配对评测、逐题区间、回答概率及相对参考分差 | [实现](../src/weight_programming/evaluation_tools.py)；[评测示例](../examples/mcp_evaluation.py)、[概率与归因示例](../examples/mcp_representation.py) |
| `compare_models` | 同坐标权重、线性 CKA、开发集行为、联合字典 | [实现](../src/weight_programming/comparison_tools.py)；[行为](../examples/mcp_evaluation.py)、[联合字典](../examples/mcp_features.py)、[跨架构](../examples/mcp_classification.py) |
| `scan_lens` | 直接词表读出、调优透镜及与最终分布的差距 | [实现](../src/weight_programming/representation_tools.py)；[直接读出](../examples/mcp_representation.py)、[拟合与复核](../examples/tuned_lens.py) |
| `probe_representation` | 独立样本线性探针、控制标签、输入基线、LEACE 概念移除 | [实现](../src/weight_programming/representation_tools.py)；[分类](../examples/mcp_classification.py)、[LEACE](../examples/mcp_concept_erasure.py) |
| `attribute_output` | 词元积分梯度、两段完整答案的对比分差归因 | [实现](../src/weight_programming/attribution_tools.py)；[示例](../examples/mcp_representation.py) |
| `run_intervention`、`localize_behavior` | 替换、消融、方向注入、LEACE；精确扫描与 AtP* 梯度筛选后复核 | [干预](../src/weight_programming/intervention_tools.py)、[AtP*](../src/weight_programming/attribution_patching.py)；[干预与筛选](../examples/mcp_intervention.py)、[概念移除](../examples/mcp_concept_erasure.py) |
| `render_evidence` | 激活、干预、注意力、相似度和归因热力图，以及有向图 | [实现](../src/weight_programming/evidence_tools.py)；[热力图](../examples/mcp_intervention.py)、[有向图](../examples/mcp_attention_rollout.py) |
| `extract_features` | 稀疏自编码器、双来源联合字典、响应样本与重建误差 | [实现](../src/weight_programming/feature_tools.py)；[公开字典与联合字典示例](../examples/mcp_features.py) |
| `trace_circuit` | 注意力传播与最大流、路径替换、ACDC、EAP、Circuit Tracer 特征归因图 | [回路](../src/weight_programming/edge_tracing.py)、[特征图](../src/weight_programming/feature_circuits.py)；[注意力](../examples/mcp_attention_rollout.py)、[边干预](../examples/mcp_circuits.py)、[特征图](../examples/mcp_feature_circuit.py) |
| `compress_model` | 删层、Taylor 前馈通道剪枝、SliceGPT、GPTQ、AWQ | [结构工具](../src/weight_programming/structure_tools.py)、[独立环境封装](../src/weight_programming/compression_exports.py)；[训练与结构](../examples/mcp_training.py)、[切片](../examples/mcp_slicegpt.py)、[量化](../examples/mcp_quantization.py) |
| `train_update` | LoRA、DPO、序列蒸馏、剪后恢复，保存 PEFT 与通用因子 | [实现](../src/weight_programming/training_tools.py)；[真实一步训练示例](../examples/mcp_training.py) |
| `edit_facts` | ROME、MEMIT，补丁、改写和邻近事实对照 | [实现](../src/weight_programming/fact_tools.py)；[真实示例](../examples/mcp_fact_editing.py) |

## 真实运行结果

这些数值来自各工具示例，分别说明其实现与保存、读取、重载是否吻合。

| 示例 | 观测结果 |
|---|---|
| SmolLM2 只读双入口 | Python 与 MCP 的权重、激活和生成回放读数一致；激活形状为 `1 × 5 × 576` |
| 补丁、重放与重载 | BF16 目标权重改变一个元素，首个残差通道从 2.421107 变为 2.393364；撤销、跨工作区重放及完整重载均吻合 |
| 差分与方向 | 两个已知变化的奇异值为 0.5、0.125；秩 1 更新存储 2,112 个参数；成对激活方向投影与公式逐元素吻合 |
| 原生 PEFT | rsLoRA 的两个目标分别保留秩 2、3；实际缩放为 4.242641、5.196152；导入、分析、随机方向分解、合并及导出前向已核对 |
| DistilBERT 与 SST-2 | 16 条真实样本的分类分数与直接推理最大误差为 0；8 条独立示例的探针为 7/8，控制为 4/8；6 层注意力及分类偏置补丁已核对 |
| 跨架构 CKA | DistilBERT 与 SmolLM2 使用 8 条相同文本、各自最后有效词元的表示；示例相似度为 0.109967 |
| Qwen3 注意力 | Qwen3-0.6B 的 28 层快照、`5 × 5` 传播矩阵、`1 × 5` 最大流矩阵及有向图已核对；同一示例也运行于 SmolLM2 的 30 层 |
| 调优透镜 | 4 条独立 SmolLM2 提示中，首层与最终分布的平均 KL 从 24.64 降至 0.55。KL 越低表示两份预测分布越接近 |
| LEACE | 独立的 16 条名词、动词示例中，移除前后线性探针从 15/16 变为 8/16；变换来自作者库 |
| 边级回路 | ACDC 在 4 条候选边中保留 1 条；独立输入上保留所选边的输出 KL 为 0.00059，同规模随机选择为 0.54039 |
| 特征归因图 | 小型真实拟合字典生成 219 个节点、32 个特征节点；保存完整矩阵并核对图边；只用字典替代的输出 KL 为 4.465409 |
| SliceGPT | 隐藏宽度由 576 变为 384，参数由 134,515,008 降至 127,180,800；严格重载与前向通过 |
| GPTQ 与 AWQ | 原权重 269,060,552 字节；分别导出 113,175,416 与 114,029,456 字节，各 210 个打包张量；均在 CPU 重载并前向 |
| 四种训练 | LoRA、DPO、序列蒸馏和剪后恢复各完成真实一步训练；PEFT 重载与保存的有效因子相符 |

特征归因图报告保留原模型与替代模型的差距。当前演示字典的输出 KL 为 4.47，使用它解释具体行为时应一起查看重建误差节点和保真报告。

## 运行环境与支持范围

| 后端或插件 | 环境与边界 |
|---|---|
| Qwen3、Llama、DistilBERT | Transformers 4.57.6；CPU 或单张 CUDA 卡。分类器提供预测与观测；生成模型提供词表读出、生成和已验证的干预 |
| Qwen3.5 | 单独的 Transformers 5.3.0 环境；混合注意力按层声明能力。已有真实观测记录保留在研究文档 |
| 原生 PEFT | PEFT 0.21.2，线性 LoRA；实际比例与转置规则由库解析。额外适配器变换需要单独验证 |
| Circuit Tracer | 独立环境：Circuit Tracer 0.5.0、NNsight 0.6.0、Transformers 4.57.3；需逐层字典与梯度授权 |
| 量化 | 独立 LLM Compressor 0.15.0；输出 compressed-tensors（打包权重格式） |
| SliceGPT | 固定上游提交 `183cd92d8469bf3f1e979585a08dc382692bfbed` 的独立环境；Llama 检查点 |
| ROME、MEMIT | 固定 EasyEdit 源码和独立环境；需权重编辑及梯度授权 |
| 持久 Python | Linux 与 Bubblewrap；宿主分别授权 Python 和权重写入 |

安装命令、数据获取方式和完整示例参数见[工具说明](../docs/mcp-tools.md)。研究目录中的 EWC、ReFT、角色蒸馏等扩展候选由后续实验选择；设计目录保留它们各自的候选状态。

## 接手位置

修改工具时，先看 `src/weight_programming/inspection.py` 的登记定义，再进入对应模块。算法新增为现有工具的 `method`（方法参数），模型差异由后端解析。`examples/` 保留每组能力的实际用法。

公开仓库为 `ZeroTang05/WSP-toolkit`，分支为 `main`。`SOURCE_MANIFEST.json` 记录发布文件摘要。当前源码提交可通过 `git rev-parse HEAD` 查询。

本轮开发和验证使用本地 CPU。模型与数据下载保留在本机缓存，临时模型产物由示例自动回收。0.6B 完整效果实验和 4B 专业多智能体工作流仍按研究协议推进；工具开发的交付物是这里列出的接口、方法、示例和可复查产物格式。
