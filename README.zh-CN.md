# WSP-toolkit · 模型权重编程工具包

**让大模型调用工具，查看、修改和评测另一个模型。**

WSP 是 Weight-Space Programming（权重空间编程）的缩写。工具包把 PyTorch、Hugging Face Transformers、Safetensors、PEFT 等底层库和已有论文方法封装为大模型可以调用的工具。权重是模型运行时使用的数值。你可以读取权重，观察模型处理输入时的中间数值，再把修改保存成补丁，与原模型比较。

[English（英文首页）](README.md) · [上手指南](docs/getting-started.md) · [工具与示例](docs/mcp-tools.md) · [完整实验结果](docs/benchmarks.md) · [下载发布包](https://github.com/ZeroTang05/WSP-toolkit/releases)

[当前项目状态](PROJECT_STATUS.md) · [已实现工具清单](docs/tool-inventory.md) · [项目目录说明](docs/maintaining.md)

| 想做什么 | 可以用什么 |
|---|---|
| 查看权重并保存可撤销的修改 | 读取张量切片，创建、重放、撤销或导出补丁 |
| 看模型读到某段文字时内部发生了什么 | 记录激活和注意力，再查看热力图、逐层预测、词元归因或稀疏特征 |
| 检验某个部件会不会影响答案 | 临时替换、清零或沿指定方向改变激活，测量输出变化 |
| 追踪模型如何形成预测 | 用路径替换、ACDC 和 EAP 分析模块之间的连接，用 Circuit Tracer 画特征归因图 |
| 修改模型对事实的回答倾向 | 用 ROME 或 MEMIT 生成权重补丁，再比较修改前后的回答 |
| 比较模型并合并已有更新 | 比较权重、内部表示和回答。用 TIES 或 DARE 合并更新 |
| 分析分类模型或导入微调权重 | 读取 DistilBERT 的分类分数与快照，导入原生 PEFT LoRA 适配器，复用分析、合并和压缩工具 |
| 缩小模型或训练轻量更新 | 删层、删通道、用 SliceGPT 切片、用 GPTQ 或 AWQ 导出 4 位权重，或训练 LoRA 适配器 |

研究者可以用 Python 调用，大模型可以通过 MCP（模型连接工具的协议）调用。两种入口共用工具定义，由模型适配器说明当前模型可以观察哪些位置。 [工具与示例](docs/mcp-tools.md)

当前设计目录覆盖 **34 个工具条目、47 种具名方法**。Llama、Qwen3、Qwen3.5 和 DistilBERT 按实际能力分别接入。每种方法保留文献来源、实现位置和真实示例。 [详细设计](docs/tool-design.md) · [工具实现清单](docs/tool-inventory.md)

在 Linux 上，隔离的 Python 会话可以跨调用保留变量、读取模型快照，并提交权重补丁。 [隔离 Python 会话](docs/mcp-tools.md#persistent-python-sessions)

### 一组完整实测

Qwen3-0.6B 在 GSM8K 数学测试的 **1,319 道题**上，原模型答对 **485 题（36.77%）**。把三次训练得到的更新用 TIES 合并，再将更新的整体大小调到和普通平均一致后，答对 **600 题（45.49%）**，提高 **8.72 个百分点**。直接使用合并结果答对 417 题，说明这次实验中更新的幅度会明显影响效果。 [五组完整对照](docs/benchmarks.md)

### 从哪里开始

1. 安装工具并准备本地模型。用只读示例检查权重、激活和两个调用入口。 [工具与示例](docs/mcp-tools.md)
2. 需要隔离的 Python 分析工作区时，按程序设计者指南配置运行环境。 [程序设计者指南](docs/model-programmer.md)
3. 要评估修改效果时，先保存修改，再在独立测试集上评分。 [数学评测指南](docs/final-evaluation.md)

工具设计参考了 MIT 团队的权重空间编程实验。独立复现 Qwen3.5 数学提升和专业多智能体工作流评测仍在研究中。 [原作](https://rikiyat.github.io/blog/weight-space-programming/) · [研究进度](docs/benchmarks.md)

项目代码采用 Apache-2.0 许可证。 [许可证](LICENSE) · [第三方代码说明](THIRD_PARTY.md)
