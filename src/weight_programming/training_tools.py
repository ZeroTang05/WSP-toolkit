"""Train PEFT LoRA on independent model instances and save reusable weight updates."""

import json
import math
from time import perf_counter

import torch
from torch.nn import functional as F

from .editing_tools import _save
from .isolated_model import load_variant, response_logprobs, supervised_ids


_LORA = "https://arxiv.org/abs/2106.09685"
_PAPERS = {
    "lora": [_LORA],
    "dpo": [_LORA, "https://arxiv.org/abs/2305.18290"],
    "distillation": [_LORA, "https://aclanthology.org/D16-1139/"],
    "post_pruning_recovery": [_LORA, "https://arxiv.org/abs/2305.11627"],
}


class TrainingTools:
    """Use host-registered training data and a separately authorized compute budget."""

    def __init__(self, tools):
        self.tools = tools

    def _examples(self, rows, method):
        """Parse host-provided answer and preference pairs by training goals."""
        backend = self.tools.forward.backend
        maximum = self.tools.forward.policy.max_text_tokens
        examples = []
        for row in rows:
            if method == "dpo":
                chosen, rejected = row.get("chosen"), row.get("rejected")
                if not all(isinstance(answer, str) and answer.strip() for answer in (chosen, rejected)) or chosen == rejected:
                    raise ValueError("DPO requires distinct registered chosen and rejected answers")
                selected = supervised_ids(backend, row["problem"], chosen, maximum)
                disfavored = supervised_ids(backend, row["problem"], rejected, maximum)
                if selected[1] == disfavored[1]:
                    raise ValueError("DPO answers must have different token sequences")
                examples.append((selected, disfavored))
            else:
                teacher = row.get("teacher_model")
                if method == "distillation" and not (isinstance(teacher, str) and teacher.strip()):
                    raise ValueError("Sequence distillation requires a registered teacher model ID")
                key = "teacher_answer" if method == "distillation" else "answer"
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(f"Training rows require a nonempty {key}")
                examples.append((supervised_ids(backend, row["problem"], row[key], maximum),))
        return examples

    def train_update(self, parent_variant_id, dataset_id, case_ids, targets, method="lora", rank=4, alpha=8,
                     steps=1, learning_rate=0.001, beta=0.1, seed=11, materialize=False):
        """Exports standard PEFT adapters, unified update IDs, and optional dense patches."""
        from peft import LoraConfig, TaskType, get_peft_model

        tools = self.tools
        if not tools.allow_training or (materialize and not tools.allow_edits):
            raise PermissionError("Adapter training requires training permission and patch creation requires edit permission")
        if method not in _PAPERS:
            raise ValueError(f"Unknown training method: {method}")
        data = tools.training_data
        if data is None or data.name != dataset_id or data.role != "train":
            raise ValueError("Use a host-registered training dataset")
        if not all(math.isfinite(value) and value > 0 for value in (alpha, learning_rate, beta)):
            raise ValueError("Training scale, learning rate and beta must be finite and positive")
        versions = tools.versions
        backend = tools.forward.backend
        rows = [data.rows[key] for key in case_ids]
        examples = self._examples(rows, method)
        per_step = sum(len(prompt) + len(target) for example in examples for prompt, target in example)
        input_tokens = per_step * (steps + (method == "dpo"))
        if input_tokens > tools.forward.policy.max_total_text_tokens:
            raise ValueError("Adapter training exceeds the token budget")
        model = load_variant(backend, versions, parent_variant_id, dtype=torch.float32)
        if method == "post_pruning_recovery":
            path = versions.store.model_dir / "structure_manifest.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["parameters"] != sum(p.numel() for p in model.parameters()):
                raise ValueError("Recovery checkpoint does not match its structural export")
        for name in targets:
            module = model.get_submodule(name)
            if (not isinstance(module, torch.nn.Linear) or name + ".weight" not in versions.groups
                    or rank > min(module.in_features, module.out_features)):
                raise ValueError(f"Invalid LoRA target or rank: {name}")
        started = perf_counter()
        references = None
        if method == "dpo":
            with torch.no_grad():
                references = [float(response_logprobs(model, *pair[0]).sum() - response_logprobs(model, *pair[1]).sum())
                              for pair in examples]
        devices = [backend.device.index] if backend.device.type == "cuda" else []
        with backend._execution_lock, torch.random.fork_rng(devices=devices), torch.enable_grad():
            torch.manual_seed(seed)
            adapter = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, target_modules=list(targets),
                                                      r=rank, lora_alpha=alpha, lora_dropout=0, bias="none"))
            adapter.train(method != "dpo")
            trainable = [parameter for parameter in adapter.parameters() if parameter.requires_grad]
            optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0)
            losses = []
            for _ in range(steps):
                optimizer.zero_grad(set_to_none=True)
                step_loss = 0.0
                for index, example in enumerate(examples):
                    positive = response_logprobs(adapter, *example[0])
                    if method == "dpo":
                        negative = response_logprobs(adapter, *example[1])
                        loss = -F.logsigmoid(beta * (positive.sum() - negative.sum() - references[index]))
                    else:
                        loss = -positive.mean()
                    loss = loss / len(examples)
                    loss.backward()
                    step_loss += float(loss.detach())
                torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
                optimizer.step()
                losses.append(step_loss)
        factors = {}
        for name in targets:
            layer = adapter.get_base_model().get_submodule(name)
            a = layer.lora_A["default"].weight.detach().float().cpu().contiguous()
            b = layer.lora_B["default"].weight.detach().float().cpu().contiguous() * layer.scaling["default"]
            factors[name + ".weight"] = (a, b)
        provenance = {"parent_variant_id": parent_variant_id, "dataset_id": dataset_id,
                      "dataset_sha256": data.identity["sha256"], "case_ids": case_ids, "steps": steps,
                      "rank": rank, "alpha": alpha, "learning_rate": learning_rate, "seed": seed,
                      "beta": beta if method == "dpo" else None, "losses": losses, "input_tokens": input_tokens}
        if method == "distillation":
            provenance["teacher_models"] = sorted({row["teacher_model"] for row in rows})
        if method == "post_pruning_recovery":
            provenance["structural_source"] = {key: record[key] for key in ("variant_id", "method", "parameters")}
        adapter.eval()
        # Candidate adapters require the caller to explicitly load a matching parent model.
        if parent_variant_id != tools.variant_id:
            adapter.peft_config["default"].base_model_name_or_path = ""
        method_provenance = {"papers": _PAPERS[method], "implementation_grade": "bounded PEFT adapter training"}
        adapter_id = tools.delta.publish_update(factors, method, provenance, native_adapter=adapter,
                                                method_provenance=method_provenance)
        patch_id = None
        if materialize:
            parent = versions.read_tensors(parent_variant_id, factors)
            replacements = {name: parent[name].float() + b @ a for name, (a, b) in factors.items()}
            patch_id = _save(versions, parent_variant_id, replacements, method)["patch_id"]
        return {"adapter_id": adapter_id, "adapter_path": f"updates/{adapter_id}/peft", "patch_id": patch_id,
                "method": method, "parent_variant_id": parent_variant_id, "losses": losses,
                "input_tokens": input_tokens, "trainable_parameters": sum(p.numel() for p in trainable),
                "elapsed_seconds": perf_counter() - started}
