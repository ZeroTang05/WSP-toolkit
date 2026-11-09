# 带预算的生成与密集观测：可执行工具接口

2026-11-07。依据已归档的MIT实验8首轮协议扩充工具库。本次实现的HF后端支持固定版本的Qwen3／Llama decoder；新增Qwen3.5独立适配已完成真实观测，见[qwen35-observation-study.md](qwen35-observation-study.md)；vLLM可比性和原作完整复现未完成。

## 接口与证据链

`WeightStore`保存CPU上的编辑提案，`ForwardService`执行一次确定权重快照上的请求，`BudgetLedger`记录成功、失败和仍在执行的调用。它们分别负责写入、观测和预算；程序设计者不能通过更改请求字段随意改变采样协议。

底层方法来源是[作者工具协议](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/runs/p50c/turns/t001/prompt.txt)。本实现独立重建协议；生成库的版本、数学评分器与观测轴语义均记录在输出，不声称复用了未公开的前向服务器。

| 请求 | 已实现的返回 | 关键语义 |
|---|---|---|
| generate original／edited | 文本、token IDs、终止原因、长度、循环尾部诊断、可选成绩 | 配置中的temperature／top-p／top-k／presence penalty固定；每题每次样本有稳定种子 |
| generate both | 原模型和编辑模型的上述结果及配对差异 | 两边同题、同提示、同采样种子；重复样本按题目聚合后计算标准误 |
| teacher_forced | token log-prob、top-k、残差、MLP输入、注意力和长度／mask | 使用右填充数组；位置与归一化定义见meta，不以padding当作真实token |
| generation activations | 指定层的prompt末位置与生成前缀平均激活 | 采样后teacher-forced replay；包含最后一个被采样token，额外计算单独记账 |

生成前向与密集观测共享一个成功请求预算；样本数、token数、attention大小、填充后的返回字节均有上限。相同请求次数不等于相同实际计算，元数据另记录物理model forward次数、输入／输出tokens、重放及耗时。

## 数值定义与实现边界

- `token_logprobs[t]`表示输入第t个token在其前缀下的原始模型条件概率；第一个token和padding为NaN。
- `topk[t]`表示第t个query位置的下一token分布。它与前一项的下标语义不同；存储ID及归一化log-prob。
- `hidden_0`为token embedding输出，`hidden_l`为block l−1的原始残差输出，末层也在最终模型norm之前。
- `mlp_l`为该层`down_proj`的输入，即门控中间激活。它不是MLP输出，也不是梯度重要性。
- attention仅在适用层返回真实概率；请求时临时切换到eager计算，完成后恢复后端。线性注意力不能伪装成普通注意力矩阵。
- presence penalty只对已经生成过的token减一次分，不惩罚仅在prompt中出现的token，不按出现次数叠加；仍不据此声称不同引擎逐token相同。

HF适配器目前限定Transformers 4.57.6的Qwen3／Llama，核对加载后的张量与源检查点；不接受未实现的模型类型。Qwen3.5另用Transformers5.3.0适配器，已完成32条原始基座观测；两条运行时分别安装。

内核与服务使用内容寻址的提案文件。服务每次读取最新提交并冻结已验证快照，不在请求运行中跟随下一次编辑；在结束或失败时恢复模型参数。原生精度下的实际张量hash用于变体标识，区分FP32提案与执行权重。

本库不是进程沙箱。执行任意Python、禁用网络／GPU、隐藏参考编辑与测试数据、以及CPU／GPU调度仍由宿主负责。当前服务是Python请求边界，尚非HTTP服务或完整隔离agent执行器。

## 数学评分与循环诊断

`MathVerifyGrader`采用已核对作者源码的boxed提取约定及固定依赖：math-verify 0.9.0、latex2sympy2_extended 1.11.0、ANTLR 4.13.2。相关提取代码保留出处及MIT许可。它在主线程执行有时限的符号评分；不在缺包或不支持的线程里悄悄改用字符串相等。

未配置评分器时，服务明确返回`correct=null`及`accuracy=null`；文本提取和循环诊断仍可用，不能把它当成任务成绩。周期尾部、重复4-gram比例、空白退化和达到长度上限分别记录，不用其中一项直接替代“答案正确”。

## 已完成的必要检验

在真正的Qwen3模型实现上构造小型随机checkpoint，对照原生前向结果检查：逐token概率及top-k对齐、残差与末norm关系、MLP线性重建、注意力概率和因果mask、不同长度padding。另测跨进程式新提案快照、失败后的权重／预算恢复、同权重配对采样完全一致，以及激活重放确实包含末token。四项组合检查通过，原先四项直接权重检查也再次通过。

真实Qwen3-0.6B CPU观测教程已完成：128条训练派生文本，27,776个tokens，1次成功请求、128次实际模型前向，约94.44秒。残差、MLP输入及概率等预估载荷0.91GB。FP32参考文本观测不作为生成准确率或能力提升结论；代码已冻结，小型结果见`artifacts/toolkit/forward-observation-v1/Qwen3-0.6B/`。真实激活进一步用于[剪枝与补偿组合实验](reconstruction-study.md)。

## 可调用示例

以下假设宿主已加载支持的模型、tokenizer和经授权的练习集；不包含公开测试标签：

```python
from weight_programming import WeightStore, ForwardPolicy, PracticeSet, ForwardService
from weight_programming.hf_backend import HuggingFaceBackend

store = WeightStore(model_dir, edit_dir)
backend = HuggingFaceBackend(model, tokenizer, store)
practice = PracticeSet(training_rows, role="train", name="my-practice")
service = ForwardService(backend, store, practice, work_dir,
                         policy=ForwardPolicy(), grader=math_grader)

result = service.request({
    "type": "teacher_forced", "model": "original",
    "texts": ["A reference continuation or a matched failure context"],
    "return": {"token_logprobs": True, "topk": 5,
               "hidden_states": [0, 4], "mlp_act": [3]}
})

paired = service.request({
    "type": "generate", "model": "both",
    "problems": [registered_training_id], "n": 2, "max_tokens": 4096
})
```

教程的实际运行不调用训练或优化器。后续还需实现隔离内核／服务传输、原作架构与生成后端、完整公开任务确认，以及独立程序设计者的工具选择轨迹。
