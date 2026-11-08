"""用真实词元嵌入和显式基线计算目标输出的归因。"""

from time import perf_counter

import torch


class AttributionTools:
    """在冻结模型上计算梯度，并使用共用预算与产物格式。"""

    def __init__(self, forward, versions, artifacts):
        self.forward = forward
        self.versions = versions
        self.artifacts = artifacts

    def attribute_output(self, variant_id, text, baseline_text, target_token, alternative_token=None,
                         query_position=None, steps=16, method="integrated_gradients"):
        """在两组同长度词元嵌入之间积分，返回每个位置的有符号贡献。"""
        from captum.attr import IntegratedGradients

        backend, policy = self.forward.backend, self.forward.policy
        ids, baseline_ids = backend.encode(text), backend.encode(baseline_text)
        length = len(ids)
        if length != len(baseline_ids) or length > min(backend.context_length, policy.max_text_tokens):
            raise ValueError("Baseline and input need the same legal token length")
        if not 0 <= target_token < backend.vocab_size or (alternative_token is not None and
                (not 0 <= alternative_token < backend.vocab_size or target_token == alternative_token)):
            raise ValueError("Select legal distinct target and alternative token IDs")
        query = length - 1 if query_position is None else query_position
        if not 0 <= query < length:
            raise ValueError("Query position is outside the input")
        if length * (steps + 2) > policy.max_total_text_tokens:
            raise ValueError("Integrated gradients exceed the token budget")
        if 4 * length * backend.vocab_size * 4 > min(policy.max_return_bytes, 512_000_000):
            raise ValueError("The full score array exceeds the gradient memory budget")
        snapshot = self.versions.snapshot(variant_id)
        ledger = self.forward.ledger
        request = {"type": "attribute_output", "method": method, "text": text, "baseline_text": baseline_text,
                   "target_token": target_token, "alternative_token": alternative_token,
                   "query_position": query, "steps": steps}
        attempt = ledger.begin(request, variant_id)
        run_id = f"request_{attempt:04d}"
        started, forwards = perf_counter(), 0

        def score(embeddings):
            nonlocal forwards
            forwards += 1
            mask = torch.ones(embeddings.shape[:2], device=embeddings.device, dtype=torch.long)
            result = backend.model(inputs_embeds=embeddings, attention_mask=mask, use_cache=False, logits_to_keep=0)
            values = result.logits[:, query].float()
            return values[:, target_token] if alternative_token is None else values[:, target_token] - values[:, alternative_token]

        try:
            with backend.activate(snapshot):
                embedding = backend.model.get_input_embeddings()
                with torch.no_grad():
                    inputs = embedding(torch.tensor([ids], device=backend.device))
                    baseline = embedding(torch.tensor([baseline_ids], device=backend.device))
                    input_score, baseline_score = float(score(inputs).item()), float(score(baseline).item())
                with torch.enable_grad():
                    attribution = IntegratedGradients(score).attribute(inputs, baselines=baseline, n_steps=steps,
                                                                       internal_batch_size=4)
            values = attribution.detach().sum(dim=-1).float().cpu().numpy()
            artifact_id = self.artifacts.save_array("token_attribution", values, ["sample", "token"], run_id=run_id,
                                                    metric="signed_integrated_gradient", variant_id=variant_id)
            usage = {"executed_model_forwards": forwards, "input_tokens": (steps + 2) * length,
                     "elapsed_seconds": perf_counter() - started, "artifact_id": artifact_id}
            ledger.finish(attempt, True, usage)
        except Exception as error:
            ledger.finish(attempt, False, {"attempted_model_forwards": forwards,
                                           "error_type": type(error).__name__, "error": str(error)[:500]})
            raise
        contribution = float(values.sum(dtype="float64"))
        difference = input_score - baseline_score
        return {"run_id": run_id, "variant_id": variant_id, "method": method, "artifact_id": artifact_id,
                "token_ids": ids, "baseline_token_ids": baseline_ids, "query_position": query,
                "input_score": input_score, "baseline_score": baseline_score, "score_change": difference,
                "attribution_sum": contribution, "integration_error": contribution - difference, "usage": usage}
