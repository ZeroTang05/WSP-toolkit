# 公开完整基线：第一批结果

本文保留成文时的实验设置、结果与状态。当前项目进度见[项目状态](../../PROJECT_STATUS.md)。

日期：2026-11-06。只包含完整、已经逐题重算的原模型基线；尚无优化增益结论。

| 模型 | GSM8K flexible EM，n=1319 | ARC-C acc_norm，n=1172 | HellaSwag acc_norm，n=10042 |
|---|---:|---:|---:|
| Qwen3-0.6B | 36.47%（481题） | 30.97%（363题） | 41.99%（4217题） |
| SmolLM2-360M-Instruct | 6.07%（80题） | 35.41%（415题） | 51.91%（5213题） |

协议：lm-evaluation-harness 0.4.9.2，zero-shot，chat template，greedy，BF16，batch 8；GSM8K采用non-thinking、最多512新token。ARC和HellaSwag采用固定版本官方任务评分，保留acc和acc_norm；GSM严格格式EM也保留，两模型均为0。这里的协议不同于模型卡上可能采用的few-shot、thinking或生成长度设置。

第一条可确认的观察是**任务排序反转**：本协议下，Qwen0.6数学分数较高，Smol360的两个保护任务分数较高。因此不能用单一任务建立模型的总体强弱排序，也不能用优化后的绝对分数替代各自相对基线的提升量。

这两个模型的参数量、训练数据、tokenizer、后训练过程均不同，不能将差异解释为纯粹系列或架构因果效应。两个1.7B模型、三种子优化、相对容量控制与方法交互分析尚未完成。

结果文件位于本地`artifacts/public/<model>/baseline/`：原始逐题预测、harness结果、协议、源码快照、模型文件指纹和独立audit均已保存。每项audit重新聚合原始记录，匹配harness指标；Wilson区间只反映题目采样不确定性，未包含优化种子与预训练污染不确定性。
