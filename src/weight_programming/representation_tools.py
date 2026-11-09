"""从已保存的激活读取层级预测和线性探针结果。"""

import numpy as np
import torch
from torch.nn import functional as F


def activation_rows(artifacts, artifact_id, sample_indices=None, token_position=None):
    """按各样本的有效词元定位激活，默认取最后一个词元。"""
    array, info = artifacts.array(artifact_id)
    if info["axes"] != ["sample", "token", "channel"]:
        raise ValueError("Expected an activation array with sample, token and channel axes")
    lengths, _ = artifacts.array(f"{info['run_id']}/lengths")
    indices = list(range(len(lengths))) if sample_indices is None else list(sample_indices)
    if len(set(indices)) != len(indices) or any(type(i) is not int or not 0 <= i < len(lengths) for i in indices):
        raise ValueError("Select distinct samples within the activation array")
    positions = lengths[indices] - 1 if token_position is None else np.full(len(indices), token_position)
    if np.any(positions < 0) or np.any(positions >= lengths[indices]):
        raise ValueError("The selected token must be valid in every selected sample")
    return np.asarray(array[indices, positions]), info, indices, positions.tolist()


class RepresentationTools:
    """在与原前向相同的版本和位置上分析已有激活。"""

    def __init__(self, tools):
        self.tools = tools

    def _lens_sites(self, site_ids):
        """将层级读出的逻辑站点映射到前向快照。"""
        sites = {}
        for site_id in site_ids:
            site = self.tools._sites.get(site_id)
            if site is None or site["kind"] != "hidden":
                raise ValueError("Layer readout requires a residual or embedding site")
            sites[site_id] = f"hidden_{site['index']}"
        return sites

    def _lens_logits(self, tensor):
        """直接读出和调优透镜使用同一个最终归一化与词表头。"""
        backend = self.tools.forward.backend
        head = backend.model.get_output_embeddings()
        return head(backend.decoder.norm(tensor.to(head.weight.dtype))).float()

    def fit_tuned_lens(self, run_id, site_ids, sample_indices, *, steps=24, learning_rate=0.001, token_position=None):
        """由 Python 宿主用真实校准快照拟合仿射变换，模型参数保持冻结。"""
        tools = self.tools
        run = tools.get_run(run_id)
        backend = tools.forward.backend
        width = backend.hidden_size
        sites = self._lens_sites(site_ids)
        final_key = f"hidden_{backend.num_layers}"
        if (not sites or len(sites) != len(site_ids) or final_key in sites.values() or not 1 <= len(sample_indices) <= 64
                or not 1 <= steps <= 128 or not np.isfinite(learning_rate) or learning_rate <= 0):
            raise ValueError("Fit distinct intermediate sites with 1–64 samples and 1–128 steps")
        if len(sites) * width * (width + 1) * 4 > 256 * 1024**2:
            raise ValueError("Requested translators exceed the 256 MiB fit limit")
        if len(sample_indices) * backend.vocab_size * 16 > min(tools.forward.policy.max_return_bytes, 512_000_000):
            raise ValueError("Vocabulary scores exceed the lens fit memory budget")

        def inputs(key):
            rows, _, _, _ = activation_rows(tools.artifacts, run["artifacts"][key], sample_indices, token_position)
            return torch.from_numpy(rows).to(device=backend.device, dtype=torch.float32)

        fitted, losses = [], {}
        with backend.activate(tools.versions.snapshot(run["variant_id"])), torch.enable_grad():
            with torch.no_grad():
                target = self._lens_logits(inputs(final_key)).log_softmax(dim=-1)
            for site_id, key in sites.items():
                hidden = inputs(key)
                weight = torch.zeros(width, width, device=backend.device, dtype=torch.float32, requires_grad=True)
                bias = torch.zeros(width, device=backend.device, dtype=torch.float32, requires_grad=True)
                optimizer = torch.optim.Adam((weight, bias), lr=learning_rate)

                def loss():
                    # 作者实现用零初始化的残差仿射变换，初始读出等于直接读出。
                    prediction = self._lens_logits(hidden + F.linear(hidden, weight, bias)).log_softmax(dim=-1)
                    return F.kl_div(prediction, target, log_target=True, reduction="batchmean")

                for step in range(steps):
                    optimizer.zero_grad(set_to_none=True)
                    value = loss()
                    if step == 0:
                        initial = float(value.detach())
                    value.backward()
                    optimizer.step()
                with torch.no_grad():
                    losses[site_id] = {"initial_kl": initial, "final_kl": float(loss())}
                    fitted.append(torch.cat((weight, bias[:, None]), dim=1).cpu().numpy())
        data = np.stack(fitted)
        if not np.isfinite(data).all() or any(not np.isfinite(row["final_kl"]) for row in losses.values()):
            raise ValueError("Tuned lens fit produced nonfinite values")
        ref = tools.artifacts.save_array("tuned_lens", data, ["site", "output_channel", "input_channel_or_bias"],
            method="tuned_lens", variant_id=run["variant_id"], site_ids=list(sites), source_run_id=run_id,
            source_artifact_ids=[run["artifacts"][key] for key in [*sites.values(), final_key]],
            fitting={"sample_indices": list(sample_indices), "token_position": token_position,
                     "steps": steps, "learning_rate": learning_rate, "losses": losses})
        return {"decoder_ref": ref, "variant_id": run["variant_id"], "losses": losses}

    @torch.inference_mode()
    def scan_lens(self, run_id, site_ids, method="logit_lens", sample_indices=None, token_position=None, topk=5,
                  target_token=None, decoder_ref=None):
        """用模型的最终归一化和词表头读取各层残差。"""
        tools = self.tools
        run = tools.get_run(run_id)
        backend = tools.forward.backend
        if topk > backend.vocab_size or target_token is not None and target_token >= backend.vocab_size:
            raise ValueError("Vocabulary selection outside the registered model")
        sites = self._lens_sites(site_ids)
        final_key = f"hidden_{backend.num_layers}"
        snapshot = tools.versions.snapshot(run["variant_id"])
        translators = {}
        if method == "tuned_lens":
            data, binding = tools.artifacts.array(decoder_ref)
            names = binding.get("site_ids", [])
            width = backend.hidden_size
            if (binding.get("method") != "tuned_lens" or binding.get("variant_id") != run["variant_id"]
                    or binding["axes"] != ["site", "output_channel", "input_channel_or_bias"]
                    or data.shape != (len(names), width, width + 1) or len(names) != len(set(names))
                    or not data.dtype.kind == "f"):
                raise ValueError("Tuned lens must match the model variant and translator coordinates")
            for site_id, key in sites.items():
                if key != final_key:
                    if site_id not in names:
                        raise ValueError(f"Tuned lens has no translator for {site_id}")
                    parameters = data[names.index(site_id)].copy()
                    if not np.isfinite(parameters).all():
                        raise ValueError(f"Nonfinite tuned lens translator for {site_id}")
                    translators[site_id] = torch.from_numpy(parameters).to(device=backend.device, dtype=torch.float32)

        def readout(key, translator=None):
            if key not in run["artifacts"]:
                raise ValueError(f"The forward run did not capture {key}")
            values, _, samples, positions = activation_rows(tools.artifacts, run["artifacts"][key],
                                                           sample_indices, token_position)
            if not 0 < len(samples) <= 32:
                raise ValueError("Select between one and 32 samples for a layer readout")
            tensor = torch.from_numpy(values).to(backend.device)
            if translator is not None:
                tensor = tensor.float() + F.linear(tensor.float(), translator[:, :-1], translator[:, -1])
            logprobs = self._lens_logits(tensor).log_softmax(dim=-1).cpu()
            return logprobs, samples, positions

        records = []
        with backend.activate(snapshot):
            final = readout(final_key)[0] if final_key in run["artifacts"] else None
            for site_id, key in sites.items():
                logprobs, samples, positions = readout(key, translators.get(site_id))
                values, ids = torch.topk(logprobs, topk, dim=-1)
                kl = torch.nn.functional.kl_div(logprobs, final, log_target=True, reduction="none").sum(-1) if final is not None else None
                for index, (sample, position) in enumerate(zip(samples, positions)):
                    tokens = [{"token_id": token, "text": backend.tokenizer.decode([token]), "log_probability": score}
                              for token, score in zip(ids[index].tolist(), values[index].tolist())]
                    records.append({"site_id": site_id, "sample": sample, "position": position, "top_tokens": tokens,
                                    "kl_from_final": max(0.0, float(kl[index])) if kl is not None else None,
                                    "target_log_probability": float(logprobs[index, target_token]) if target_token is not None else None})
        return {"method": method, "variant_id": run["variant_id"], "run_id": run_id, "decoder_ref": decoder_ref,
                "final_reference_available": final is not None,
                "report_artifact_id": tools.artifacts.save_records("lens", records), "preview": records[:5]}

    def probe_representation(self, artifact_id, labels, train_indices, evaluation_indices, seed=0,
                             control_labels=None, input_baseline_artifact_id=None, token_position=None, fit_eraser=False):
        """拟合线性分类器并在独立样本上比较真实标签、控制和输入基线。"""
        from sklearn.linear_model import LogisticRegression

        artifacts = self.tools.artifacts
        rows, info, indices, _ = activation_rows(artifacts, artifact_id, token_position=token_position)
        if len(labels) != len(indices) or not np.isfinite(rows).all():
            raise ValueError("Provide one label per finite sample row")
        train, evaluation = list(train_indices), list(evaluation_indices)
        if (not train or not evaluation or len(set(train + evaluation)) != len(train) + len(evaluation)
                or any(type(i) is not int or not 0 <= i < len(indices) for i in train + evaluation)):
            raise ValueError("Training and evaluation indices must be distinct valid samples")
        gold = np.asarray(labels)
        if np.unique(gold[train]).size < 2 or not set(gold[evaluation]).issubset(set(gold[train])):
            raise ValueError("Training needs at least two classes and must cover evaluation labels")
        control = np.asarray(control_labels) if control_labels is not None else gold.copy()
        if control_labels is None:
            rng = np.random.default_rng(seed)
            control[train] = rng.permutation(control[train])
            control[evaluation] = rng.permutation(control[evaluation])
        if (control.shape != gold.shape or np.unique(control[train]).size < 2
                or not set(control[evaluation]).issubset(set(control[train]))):
            raise ValueError("Control labels must cover the same rows and their evaluation classes")

        def fit(values, target):
            classifier = LogisticRegression(max_iter=500, random_state=seed)
            classifier.fit(values[train], target[train])
            predicted = classifier.predict(values[evaluation])
            return classifier, predicted, float(np.mean(predicted == target[evaluation]))

        model, predicted, accuracy = fit(rows, gold)
        _, control_predicted, control_accuracy = fit(rows, control)
        baseline_accuracy = None
        if input_baseline_artifact_id is not None:
            baseline, baseline_info, _, _ = activation_rows(artifacts, input_baseline_artifact_id,
                                                           token_position=token_position)
            if baseline_info["run_id"] != info["run_id"] or baseline_info.get("site_id") != "embedding.out":
                raise ValueError("The input baseline must be the embedding snapshot of the same run")
            _, _, baseline_accuracy = fit(baseline, gold)
        if fit_eraser:
            erased, factors, covariance = self._fit_eraser(rows, gold, train, evaluation)
            _, _, erased_accuracy = fit(erased, gold)
            site_id = info["site_id"]
            variant_id = self.tools.get_run(info["run_id"])["variant_id"]
            eraser_id = artifacts.save_array("leace_factors", factors, ["channel", "factor"], method="leace",
                                             site_id=site_id, variant_id=variant_id, source_artifact_ids=[artifact_id],
                                             train_indices=train, evaluation_indices=evaluation, token_position=token_position,
                                             cross_covariance_norms=covariance)
        class_names = model.classes_[-model.coef_.shape[0]:].tolist()
        weights = artifacts.save_array("probe_weights", model.coef_, ["class", "channel"], classes=class_names,
                                      site_id=info.get("site_id"))
        report = [{"sample": sample, "label": labels[sample], "prediction": predicted[index],
                   "control_label": control[sample], "control_prediction": control_predicted[index]}
                  for index, sample in enumerate(evaluation)]
        result = {"source_artifact_id": artifact_id, "site_id": info.get("site_id"), "method": "logistic_regression",
                  "accuracy": accuracy, "control_accuracy": control_accuracy, "selectivity": accuracy - control_accuracy,
                  "input_baseline_accuracy": baseline_accuracy, "train_samples": len(train),
                  "evaluation_samples": len(evaluation), "classes": class_names, "coefficients_artifact_id": weights,
                  "report_artifact_id": artifacts.save_records("probe_predictions", report)}
        if fit_eraser:
            result["concept_erasure"] = {"eraser_artifact_id": eraser_id, "variant_id": variant_id,
                                         "evaluation_accuracy": erased_accuracy, "cross_covariance_norms": covariance}
        return result

    @staticmethod
    def _fit_eraser(rows, labels, train, evaluation):
        """用作者实现只在训练行拟合 LEACE，比较训练和复核行的线性关联。"""
        from concept_erasure import LeaceEraser

        classes = np.unique(labels[train])
        if rows.shape[1] > 4096 or len(classes) > 32:
            raise ValueError("LEACE supports up to 4096 channels and 32 classes per fit")
        activations = torch.as_tensor(rows, dtype=torch.float64)
        targets = torch.as_tensor(labels[:, None] == classes[None, :]).to(activations)
        # 采用论文闭式投影，保留原库的协方差估计，关闭额外的方差约束。
        eraser = LeaceEraser.fit(activations[train], targets[train], constrain_cov_trace=False)
        erased = eraser(activations)
        packed = torch.cat((eraser.proj_left, eraser.proj_right.T, eraser.bias[:, None]), dim=1)

        def cross_covariance(values, indices):
            x, y = values[indices], targets[indices]
            return float(((x - x.mean(0)).T @ (y - y.mean(0)) / len(indices)).norm())

        evidence = {name: {"before": cross_covariance(activations, indices), "after": cross_covariance(erased, indices)}
                    for name, indices in (("train", train), ("evaluation", evaluation))}
        return erased.numpy(), packed.numpy(), evidence
