"""用真实词元嵌入和显式基线计算目标输出的归因。"""

from time import perf_counter

import torch

from .evaluation_tools import aligned_completion


class AttributionTools:
    """在冻结模型上计算梯度，并使用共用预算与产物格式。"""

    def __init__(self, forward, versions, artifacts):
        self.forward = forward
        self.versions = versions
        self.artifacts = artifacts

    def attribute_output(self, variant_id, text, baseline_text, target_token=None, alternative_token=None,
                         query_position=None, steps=16, method="integrated_gradients", chosen=None,
                         rejected=None, aggregation="sum"):
        """在两组同长度词元嵌入之间积分，返回每个位置的有符号贡献。"""
        if method == "contrastive_sequence":
            return self._contrastive_sequence(variant_id, text, baseline_text, chosen, rejected, aggregation, steps)
        backend = self.forward.backend
        ids, baseline_ids = self._prompts(text, baseline_text)
        length = len(ids)
        if not 0 <= target_token < backend.vocab_size or (alternative_token is not None and
                (not 0 <= alternative_token < backend.vocab_size or target_token == alternative_token)):
            raise ValueError("Select legal distinct target and alternative token IDs")
        query = length - 1 if query_position is None else query_position
        if not 0 <= query < length:
            raise ValueError("Query position is outside the input")
        request = {"type": "attribute_output", "method": method, "text": text, "baseline_text": baseline_text,
                   "target_token": target_token, "alternative_token": alternative_token,
                   "query_position": query, "steps": steps}

        def build_score(embedding, read):
            def score(embeddings):
                values = read(embeddings)[:, query].float()
                return values[:, target_token] if alternative_token is None else values[:, target_token] - values[:, alternative_token]
            return score

        result = self._integrate(variant_id, ids, baseline_ids, request, build_score, length, length)
        return {**result, "query_position": query}

    def _contrastive_sequence(self, variant_id, text, baseline_text, chosen, rejected, aggregation, steps):
        """固定两段回答，归因输入词元如何改变回答的对数概率分差。"""
        backend, policy = self.forward.backend, self.forward.policy
        ids, baseline_ids = self._prompts(text, baseline_text)
        if chosen == rejected:
            raise ValueError("Chosen and rejected answers must differ")
        suffixes = [aligned_completion(backend, ids, text + answer) for answer in (chosen, rejected)]
        if suffixes != [aligned_completion(backend, baseline_ids, baseline_text + answer) for answer in (chosen, rejected)]:
            raise ValueError("Each answer must tokenize identically after the input and baseline prompts")
        if len(ids) + max(map(len, suffixes)) > min(backend.context_length, policy.max_text_tokens):
            raise ValueError("Prompt and answer exceed the context budget")
        request = {"type": "attribute_output", "method": "contrastive_sequence", "text": text, "baseline_text": baseline_text,
                   "chosen": chosen, "rejected": rejected, "aggregation": aggregation, "steps": steps}

        def build_score(embedding, read):
            targets = [torch.tensor(suffix, device=backend.device) for suffix in suffixes]
            continuations = [embedding(target[:-1]) for target in targets]

            def score(embeddings):
                scores = []
                for target, continuation in zip(targets, continuations):
                    supplied = torch.cat((embeddings, continuation.expand(len(embeddings), -1, -1)), dim=1)
                    # 最后 k 个输出从提示末词元开始，正好预测 k 个回答词元。
                    logprobs = torch.log_softmax(read(supplied, len(target)).float(), dim=-1)
                    selected = logprobs.gather(-1, target.expand(len(embeddings), -1).unsqueeze(-1)).squeeze(-1)
                    scores.append(selected.double().sum(-1) if aggregation == "sum" else selected.double().mean(-1))
                return scores[0] - scores[1]

            return score

        lengths = [len(suffix) for suffix in suffixes]
        result = self._integrate(variant_id, ids, baseline_ids, request, build_score,
                                 2 * (len(ids) - 1) + sum(lengths), sum(lengths))
        return {**result, "aggregation": aggregation, "chosen_token_ids": suffixes[0], "rejected_token_ids": suffixes[1]}

    def _prompts(self, text, baseline_text):
        backend, policy = self.forward.backend, self.forward.policy
        ids, baseline_ids = backend.encode(text), backend.encode(baseline_text)
        if len(ids) != len(baseline_ids) or len(ids) > min(backend.context_length, policy.max_text_tokens):
            raise ValueError("Baseline and input need the same legal token length")
        return ids, baseline_ids

    def _integrate(self, variant_id, ids, baseline_ids, request, build_score, tokens_per_point, logits_per_point):
        """共用嵌入插值、预算、真实前向记账和积分误差计算。"""
        from captum.attr import IntegratedGradients

        backend, policy = self.forward.backend, self.forward.policy
        steps = request["steps"]
        input_tokens = (steps + 2) * tokens_per_point
        if input_tokens > policy.max_total_text_tokens:
            raise ValueError("Integrated gradients exceed the token budget")
        if 4 * logits_per_point * backend.vocab_size * 4 > min(policy.max_return_bytes, 512_000_000):
            raise ValueError("The score array exceeds the gradient memory budget")
        snapshot = self.versions.snapshot(variant_id)
        ledger = self.forward.ledger
        attempt = ledger.begin(request, variant_id)
        run_id = f"request_{attempt:04d}"
        started, forwards = perf_counter(), 0

        def read(embeddings, logits_to_keep=0):
            nonlocal forwards
            forwards += 1
            mask = torch.ones(embeddings.shape[:2], device=embeddings.device, dtype=torch.long)
            return backend.model(inputs_embeds=embeddings, attention_mask=mask, use_cache=False,
                                 logits_to_keep=logits_to_keep).logits

        try:
            with backend.activate(snapshot):
                embedding = backend.model.get_input_embeddings()
                with torch.no_grad():
                    inputs = embedding(torch.tensor([ids], device=backend.device))
                    baseline = embedding(torch.tensor([baseline_ids], device=backend.device))
                    score = build_score(embedding, read)
                    input_score, baseline_score = float(score(inputs).item()), float(score(baseline).item())
                with torch.enable_grad():
                    attribution = IntegratedGradients(score).attribute(inputs, baselines=baseline, n_steps=steps,
                                                                       internal_batch_size=4)
            values = attribution.detach().sum(dim=-1).float().cpu().numpy()
            artifact_id = self.artifacts.save_array("token_attribution", values, ["sample", "token"], run_id=run_id,
                                                    metric="signed_integrated_gradient", variant_id=variant_id,
                                                    method=request["method"])
            usage = {"executed_model_forwards": forwards, "input_tokens": input_tokens,
                     "elapsed_seconds": perf_counter() - started, "artifact_id": artifact_id}
            ledger.finish(attempt, True, usage)
        except Exception as error:
            ledger.finish(attempt, False, {"attempted_model_forwards": forwards,
                                           "error_type": type(error).__name__, "error": str(error)[:500]})
            raise
        contribution = float(values.sum(dtype="float64"))
        difference = input_score - baseline_score
        return {"run_id": run_id, "variant_id": variant_id, "method": request["method"], "artifact_id": artifact_id,
                "token_ids": ids, "baseline_token_ids": baseline_ids,
                "input_score": input_score, "baseline_score": baseline_score, "score_change": difference,
                "attribution_sum": contribution, "integration_error": contribution - difference, "usage": usage}
