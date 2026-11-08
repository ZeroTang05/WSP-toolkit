"""为结构变更或训练加载独立的模型实例。"""

import torch
from torch.nn import functional as F


@torch.no_grad()
def load_variant(backend, versions, variant_id, dtype="auto"):
    """加载当前检查点并重放已验证候选，供可变更的独立作业使用。"""
    snapshot = versions.snapshot(variant_id)
    model = type(backend.model).from_pretrained(versions.store.model_dir, local_files_only=True,
                                                dtype=dtype, attn_implementation="eager").to(backend.device)
    parameters = dict(model.named_parameters(remove_duplicate=False))
    for name, value in snapshot["values"].items():
        parameter = parameters[name]
        if parameter.shape != value.shape or parameter.dtype not in (value.dtype, torch.float32):
            raise ValueError(f"Candidate cannot be represented by the isolated model: {name}")
        parameter.copy_(value.to(parameter.device, dtype=parameter.dtype))
    return model.eval().requires_grad_(False)


def supervised_ids(backend, problem, answer, limit):
    """沿用模型对话模板，明确划分提示与监督回答。"""
    prefix = backend.chat([{"role": "user", "content": problem}], False, generate=True)
    prompt = backend.encode(prefix)
    target = backend.encode(answer)
    if len(prompt) + len(target) > min(limit, backend.context_length):
        raise ValueError("Training sample exceeds the declared context budget")
    return prompt, target


def response_logprobs(model, prompt, target):
    """返回回答词元的逐位置对数概率，供训练和校准使用。"""
    device = next(model.parameters()).device
    ids = torch.tensor([prompt + target], device=device)
    # 给定回答的第一个词元由提示末尾位置预测。
    logits = model(input_ids=ids, use_cache=False, logits_to_keep=len(target) + 1).logits[0, :-1].float()
    wanted = torch.tensor(target, device=device).unsqueeze(1)
    return F.log_softmax(logits, dim=-1).gather(1, wanted).squeeze(1)
