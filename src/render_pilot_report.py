"""Create a local research report from downloaded authoritative experiment JSON."""
import json
from pathlib import Path


def render(root):
    evidence=root/'artifacts/evidence'
    lines=['# idea4：论文依据工具与预实验结果','',
        '日期：2026-11-06。所有实验通过 SSH MCP 在 Luck-Medium 的 GPU 0 上执行；GPU 1 的其他项目任务未被修改。', '',
        '本次完成的是论文依据工具的实现、数值核对和实际参数修改尝试。数据为合成整数算术，不能代表完整数学、代码或专业领域能力；未验证 Agent 优于普通搜索。','',
        '## 方法来源与实际实现','',
        '- **Wanda**：逐权重幅度乘输入激活 L2 范数，按输出行剪枝；激活统计与掩码对照固定版本作者代码。Qwen3 与领域校准数据属于适配。[论文](https://arxiv.org/abs/2306.11695)。',
        '- **MLP 组 Taylor 剪枝**：来自 LLM-Pruner 的分组与梯度重要性思路，仅适配 gated MLP，不声称完整复现其依赖图和全部估计器。[论文](https://arxiv.org/abs/2305.11627)。',
        '- **LoRA SFT**：冻结基础参数，实际训练 rank=8、alpha=16 的 q/v 低秩因子，40 步 AdamW；V2 监督答案和 EOS。[论文](https://arxiv.org/abs/2106.09685)。',
        '- **Task Arithmetic**：使用真实保存的训练后模型与原模型之差，应用 θ_base + 0.5(θ_tuned−θ_base)，修改方向由实际后训练提供。[论文](https://arxiv.org/abs/2212.04089)。','',
        '## 数据与选择规则','',
        'V2 有 48 个校准、64 个搜索、128 个确认样本；排除了旧预实验与 V1 共 466 个操作数对。校准／搜索／确认之间也不重复操作数对，确认集使用不同问法。两种模型使用同一批 V2 数据，不能当作两份独立数据样本。', '',
        '在搜索集和 8 句英文保护数据上冻结选择，再评估确认集。性能候选按答案 NLL 最小且英文 NLL 不超过原模型 1.03 倍选择，包含原模型。压缩候选按相对于 LoRA 模型的答案 NLL≤1.05 倍且英文 NLL≤1.03 倍筛选。最终确认结果不反馈调参。', '',
        '严格正确率同时要求数值正确和只输出整数。原模型可能会解释或被生成上限截断，因此这一指标包含格式遵循，不能解释为纯算术推理能力。英文保护测试只有另外 8 句话，不足以代表通用能力。','']
    for variant,label in [('06','Qwen3-0.6B'),('4b','Qwen3-4B-Instruct-2507')]:
        folder=evidence/f'paper{variant}-v2'
        summary=json.loads((folder/'summary.json').read_text())
        selection=json.loads((folder/'selection.json').read_text())
        rows={r['name']:r for r in summary['results']}
        lines += [f'## {label}：V2 独立确认','',
            '| 版本 | 严格正确率（128 题） | 答案 NLL | 总参数减少 | 固定前向耗时 ms | 英文 NLL |',
            '|---|---:|---:|---:|---:|---:|']
        for name in ['original','lora40','taskhalf','wanda10','taylor08']:
            r=rows[name]
            lines.append(f"| {name} | {r['exact_match']*100:.2f}%（{round(r['exact_match']*128)}/128） | {r['answer_nll']:.3f} | {r['parameter_reduction']*100:.2f}% | {r['forward_median_ms']:.2f} | {r['guard_nll']:.3f} |")
        lines += ['',f"搜索阶段选出的性能候选：**{selection['performance_candidate']}**；通过搜索代理约束的压缩候选：{', '.join(selection['compression_candidates_passing_search_proxy']) or '无'}。这只是搜索选择，不是确认全部能力保留。",'']
        for name in ['taskhalf','wanda10','taylor08']:
            r=rows[name]
            if 'specialist_delta_ci95' in r:
                low,high=r['specialist_delta_ci95']
                lines.append(f"- {name} 相对 LoRA 模型严格正确率变化 {r['specialist_exact_match_delta']*100:+.2f} 个百分点，成对 bootstrap 95% 区间 [{low*100:+.2f}, {high*100:+.2f}]。")
            elif name=='taskhalf':
                low,high=r['paired_bootstrap_delta_ci95']
                lines.append(f"- taskhalf 相对原模型变化 {r['exact_match_delta']*100:+.2f} 个百分点，成对 bootstrap 95% 区间 [{low*100:+.2f}, {high*100:+.2f}]。")
        lines.append('')
    lines += ['## 如何解释结果','',
        'LoRA 与 task vector 属于参数优化尝试，不减少基础架构参数。Wanda 的非结构化零权重保留稠密存储与运算，不能仅凭稀疏率声称实际加速。Taylor 变体确实删除矩阵行列，但总参数减少比例远低于 MLP 通道删除比例。', '',
        '运行时是同一 GPU 上 batch=1、256 输入 token、含 logits、无 KV cache 的完整前向，保留 12 次样本。它不是实际服务端到端延迟；不同进程的温度、时钟和历史缓存会带来波动，几个百分点的差异不能当作可靠加速结论。显存是该进程在此测试中的峰值，也不能替代完整部署测量。', '',
        'V1 保留了失败证据：只优化答案数字 NLL 而不监督 EOS，可能生成重复数字；16-token 上限也会截断解释。V2 根据这一发现修订目标和评估，使用排除旧题目的新确认集。V1 与 V2 的正确率不宜直接比较为同条件算法改进。', '',
        '## 验证与交付','',
        '- 9 项工具测试通过，含固定作者代码的统计／掩码对照、冻结基础权重的实际 LoRA 训练、合并等价性、答案与 EOS 标签边界、回滚及结构保存／重新加载。',
        '- 独立读取实际导出 safetensors，核对每种方法的修改范围、矩阵切片、低秩更新或 task vector 公式，并重新核对原模型文件哈希。',
        '- tokenizer JSON 相同，另外逐条核对 240 个聊天模板及 256 条答案／保护文本编码；验证记录随证据保存。',
        '- V2 保存了运行代码快照、数据、协议、选择记录、原始生成、分数、完整候选和补丁元数据。',
        '- 服务器项目：`/home/hdd_1/tangziye/idea4`；完整模型在 `runs/paper06-v2/` 和 `runs/paper4b-v2/`，本地 `artifacts/evidence/` 保存轻量证据，不复制大模型。',
        '- 命令入口：服务器 `.venv/bin/paper-weightlab` 和 `.venv/bin/weightlab`，也可直接执行对应 Python 文件；用法见 `PAPERS_RUNBOOK.md`。','',
        '## 研究判断与后续','',
        '本次验证了“论文方法 → 可操作工具 → 后训练参照 → 实际权重编辑 → 独立评估”的工作链，不能据此认定训练自由的通用权重编程成立，也不能认定已获得领域保真且可靠加速的模型。', '',
        '下一阶段应优先加入真实领域任务与充分保护集，在匹配成本下比较固定算法、非 LLM 搜索和 Agent；结构剪枝需进一步验证通用能力损伤与真实部署收益。SparseGPT、完整 LLM-Pruner、DPO/RLHF 和蒸馏恢复仍属于后续扩展，本次没有宣称已实现。','']
    (root/'pilot-results.md').write_text('\n'.join(lines))


if __name__=='__main__':
    render(Path(__file__).resolve().parents[1])
