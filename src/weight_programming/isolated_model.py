"""Load independent model instances for structural changes or training."""

import torch
from torch.nn import functional as F


@torch.no_grad()
def load_variant(backend, versions, variant_id, dtype="auto"):
    """Loads the current checkpoint and replays the verified candidates for use by mutable independent jobs."""
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
    """Follow the model dialogue template and clearly divide prompts and supervised responses."""
    prefix = backend.chat([{"role": "user", "content": problem}], False, generate=True)
    prompt = backend.encode(prefix)
    target = backend.encode(answer)
    if len(prompt) + len(target) > min(limit, backend.context_length):
        raise ValueError("Training sample exceeds the declared context budget")
    return prompt, target


def response_logprobs(model, prompt, target):
    """Returns the position-wise log probability of the answer token for use in training and calibration."""
    device = next(model.parameters()).device
    ids = torch.tensor([prompt + target], device=device)
    # The first token of a given response is predicted by the prompt-end position.
    logits = model(input_ids=ids, use_cache=False, logits_to_keep=len(target) + 1).logits[0, :-1].float()
    wanted = torch.tensor(target, device=device).unsqueeze(1)
    return F.log_softmax(logits, dim=-1).gather(1, wanted).squeeze(1)
