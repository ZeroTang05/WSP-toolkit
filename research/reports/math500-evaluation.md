# 完整MATH-500：宿主端评测与原模型基线

本文保留成文时的实验设置、结果与状态。当前项目进度见[项目状态](../../PROJECT_STATUS.md)。

2026-11-07。完整原模型评测已提交，最新确认running、等待GPU0第一题。没有数学成绩或循环改善结论。

## 数据与隔离

从[官方HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)固定revision `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`取得完整test.jsonl、dataset card和eval.yaml，保留500个官方unique_id、原始问题与答案。源文件逐项SHA256保存在`benchmarks/math500-evaluation-v1/manifest.json`。

与15,458个原作练习做空白归一化后的精确问题文本比较，重合0项。该检查不覆盖语义近重复、预训练污染或更广泛的题目谱系。证据为`artifacts/toolkit/math500-evaluation-v1/practice-overlap.json`。

测试集与练习树分开保存，未加入`PracticeSet`，未挂载进程序设计者内核。新`weight_programming.evaluation.evaluate_math`仅在宿主端接收显式test／evaluation标记；生成输入由problem单独构造，answer由宿主评分器使用，reference solution不进入生成项。

## 执行协议

`src/run_math500_evaluation.py`固定完整500题×4样本，共2,000次生成；thinking、32,768上限、T=.6、top-p=.95、top-k=20、presence=1.5。原模型为已核验的原作Qwen3.5-0.8B revision，执行采用独立HF 5.3.0／BF16／SDPA，未安装可选DeltaNet fast path，不能声称vLLM或作者未公开harness等价。

每题四样本期间独占GPU0，结束后模型转回CPU再释放锁。记录排队和驻留成本，避免完整长评测一直占用设备。样本随机种子按固定policy seed、官方题目ID、sample索引生成；原作实际随机流未知。

结果逐样本写入文本、token IDs、分数、截断、finish reason与声明的循环尾部指标；按题聚合四样本平均正确率及标准误，不用pass@4代替accuracy。完成标记只在所有题目全部成功后写出。部分目录拒绝静默重启。

后续冻结编辑可通过`--replacements`导入native替换张量，复制并校验基座、文件及修改权限。参考重放与独立发现使用不同来源和输出目录；评测结果不得返回独立程序设计者继续选编辑。

## 真实运行与验证

- `job-mw5memb5-6af0`：生成路径检查退出0，使用真实小型Qwen3和明确的fixture grader验证流式存储、样本身份与答案不进prompt；不是数学能力分数。
- `job-mw5mf2op-d59e`：完整原Qwen3.5基线，最新running并输出`waiting_gpu_0, position=1, items=500`，已完成473张量加载与后端核验。
- 输出目录：远端`benchmarks/math500-full-v1/original`；本地已保存`setup.json`，尚无完成结果。
- 独立程序设计者仍是`job-mw5ltbbd-76df`；其源码和上下文在新增测试评测工具前已经冻结，没有把这个评测入口提供给它。

公开用法见[宿主端评测文档](../../docs/final-evaluation.md)。当前进展补齐了完整评测执行路径，并不代表原作效果重放、独立发现或方向比较已经完成。
