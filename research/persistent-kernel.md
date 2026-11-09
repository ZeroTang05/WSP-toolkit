# 持久CPU分析内核与模型前向的连接

2026-11-07。新增`cpu_kernel.py`、`kernel_worker.py`与`programming_session.py`。依据[MIT实验8工具协议](references/mit-wsp/exp8_prompt.txt)独立重建操作能力，不是新剪枝算法。

## 已运行的能力

Linux服务器真实Qwen3-0.6B内核已启动成功，311个张量名可读取，变量跨代码块保留。写入原模型返回只读错误；外网连接返回network unreachable；看不到项目参考目录或GPU设备。

`CPUKernel`通过bubblewrap建立新的根文件系统、进程／网络命名空间及设备目录。挂载系统运行库、虚拟环境、冻结工具包、只读模型、只读授权练习及可写工作区。项目、参考编辑、最终评测集、凭据和预算账本不挂载。不支持Linux bubblewrap时拒绝启动。

最多300次执行，每次900秒、96GiB地址空间；返回前10,000字符并转ASCII，自动显示最后表达式。异常也消耗一次代码执行。普通异常后变量保留；硬超时杀掉内核并记录变量丢失，必须显式重建，不会静默重启或重置预算。宿主保存代码、返回、执行计数及源码哈希；关闭时记录closed。

默认关闭梯度不能证明任意Python不训练。禁止训练、迭代拟合和模拟网络仍需要审阅执行轨迹，不能声称操作系统隔离能证明算法语义。Python输出与异常被截断捕获；原生扩展直接写文件描述符的输出被丢弃。

## 分析—编辑—观测循环

`ProgrammingSession`连接内核、`WeightStore`和`ForwardService`：内核提交`/work/edited`提案；宿主读取并验证快照；在内核之外执行预算前向；将成功观测复制至`/work/observations/request_XXXX`。权威预算、观测和哈希留在未挂载的controller。可写副本不改变宿主证据。

通道集成检查已通过：内核编辑使后续请求绑定新变体ID；模型完成后恢复基座；内核可以读取返回的MLP数组。另外覆盖只读模型、外网、GPU设备、截断、预算和硬超时。小模型检查只证明工具通道，不是能力提升成绩。

```python
from weight_programming.programming_session import ProgrammingSession

with ProgrammingSession(factory, model_dir, practice_dir, practice_set,
                        work_dir, control_dir, python=venv_python) as session:
    session.execute("# Inspect a candidate matrix\nw=load(NAMES[1]); print(w.shape)")
    result = session.forward({
        "type": "teacher_forced", "model": "original", "texts": texts,
        "return": {"hidden_states": [0, 4], "mlp_act": [3]}
    })
    session.execute("import numpy as np\nx=np.load('/work/observations/request_0001/mlp_3.npy')\nprint(x.shape)")
```

factory接收宿主WeightStore，返回支持的模型后端。四个数据目录必须互不嵌套，controller须为新目录。独立发现还需要新的程序设计者上下文、冻结提示和可访问材料；本会话已经读过参考解，不能担任盲发现者。

## 原作模型运行时

Qwen3.5-0.8B revision `2fc06364715b967f1860aea9cf38778875588b17`已下载，12文件共1,769,978,895字节，逐文件SHA256已记录。新`qwen35_backend.py`针对Transformers5.3.0完整ConditionalGeneration检查点，保留`model.language_model.*`坐标并区分全／线性注意力。

适配器现已通过混合架构检查，并完成32条真实基座观测，见[qwen35-observation-study.md](qwen35-observation-study.md)。旧4.57.6环境不变，新环境独立为`.venv-wsp`。Torch2.5.1 CUDA12.1方案依据[官方安装表](https://pytorch.org/get-started/previous-versions/#v251)与[Transformers5.3依赖](https://github.com/huggingface/transformers/blob/v5.3.0/setup.py)。vLLM数值和吞吐可比性仍需单独验证。
