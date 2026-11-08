# WSP-toolkit · 模型权重编程工具包

**查看模型内部状态，修改指定权重，用同一套题检验效果。**

WSP 是 Weight-Space Programming（权重空间编程）的缩写。权重是模型运行时使用的数值。用这个工具包，你可以读出某一层的权重，观察模型处理输入时产生的中间数值，再把修改保存到单独的目录，方便和原模型比较。

[English（英文首页）](README.md) · [上手指南](docs/getting-started.md) · [完整实验结果](docs/benchmarks.md) · [下载发布包](https://github.com/ZeroTang05/WSP-toolkit/releases)

| 想做什么 | 可以用什么 |
|---|---|
| 看一层权重有哪些数值 | `WeightStore` 读取模型文件，并把修改单独保存 |
| 看模型读到某段文字时内部发生了什么 | `ForwardService` 记录指定层的中间数值、注意力和输出分数 |
| 检验某个部件会不会影响答案 | `ActivationPatchingStudy` 替换指定的中间数值，测量输出变化 |
| 让大模型自己分析和试改权重 | `ProgrammingSession` 提供隔离的分析环境，并记录每次工具调用 |
| 合并不同训练得到的修改，或补偿删减损失 | TIES 合并训练更新，SparseGPT 调整删减后保留下来的权重 |

### 一组完整实测

Qwen3-0.6B 在 GSM8K 数学测试的 **1,319 道题**上，原模型答对 **485 题（36.77%）**。把三次训练得到的更新用 TIES 合并，再将更新的整体大小调到和普通平均一致后，答对 **600 题（45.49%）**，提高 **8.72 个百分点**。直接使用合并结果答对 417 题，说明这次实验中更新的幅度会明显影响效果。 [五组完整对照](docs/benchmarks.md)

### 从哪里开始

1. 按上手指南安装，再读一个已有模型的权重矩阵。 [上手指南](docs/getting-started.md)
2. 想让模型自己使用工具，按程序设计者指南配置运行环境。 [程序设计者指南](docs/model-programmer.md)
3. 想评估修改效果，先保存修改，再在独立测试集上评分。 [数学评测指南](docs/final-evaluation.md)

工具设计参考了 MIT 团队的权重空间编程实验。独立复现 Qwen3.5 数学提升和专业多智能体工作流评测仍在研究中。 [原作](https://rikiyat.github.io/blog/weight-space-programming/) · [研究进度](docs/benchmarks.md)

项目代码采用 Apache-2.0 许可证。 [许可证](LICENSE) · [第三方代码说明](THIRD_PARTY.md)
