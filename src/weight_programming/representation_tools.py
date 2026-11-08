"""从已保存的激活读取层级预测和线性探针结果。"""

import numpy as np
import torch


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

    @torch.inference_mode()
    def scan_lens(self, run_id, site_ids, method="logit_lens", sample_indices=None, token_position=None, topk=5,
                  target_token=None):
        """用模型的最终归一化和词表头读取各层残差。"""
        tools = self.tools
        run = tools.get_run(run_id)
        backend = tools.forward.backend
        if topk > backend.vocab_size or target_token is not None and target_token >= backend.vocab_size:
            raise ValueError("Vocabulary selection outside the registered model")
        sites = []
        for site_id in site_ids:
            site = tools._sites.get(site_id)
            if site is None or site["kind"] != "hidden":
                raise ValueError("Layer readout requires a residual or embedding site")
            sites.append((site_id, f"hidden_{site['index']}"))
        final_key = f"hidden_{backend.num_layers}"
        snapshot = tools.versions.snapshot(run["variant_id"])

        def readout(key):
            if key not in run["artifacts"]:
                raise ValueError(f"The forward run did not capture {key}")
            values, _, samples, positions = activation_rows(tools.artifacts, run["artifacts"][key],
                                                           sample_indices, token_position)
            if not 0 < len(samples) <= 32:
                raise ValueError("Select between one and 32 samples for a layer readout")
            head = backend.model.get_output_embeddings()
            tensor = torch.from_numpy(values).to(device=backend.device, dtype=head.weight.dtype)
            logprobs = torch.log_softmax(head(backend.decoder.norm(tensor)).float(), dim=-1).cpu()
            return logprobs, samples, positions

        records = []
        with backend.activate(snapshot):
            final = readout(final_key)[0] if final_key in run["artifacts"] else None
            for site_id, key in sites:
                logprobs, samples, positions = readout(key)
                values, ids = torch.topk(logprobs, topk, dim=-1)
                kl = torch.nn.functional.kl_div(logprobs, final, log_target=True, reduction="none").sum(-1) if final is not None else None
                for index, (sample, position) in enumerate(zip(samples, positions)):
                    tokens = [{"token_id": token, "text": backend.tokenizer.decode([token]), "log_probability": score}
                              for token, score in zip(ids[index].tolist(), values[index].tolist())]
                    records.append({"site_id": site_id, "sample": sample, "position": position, "top_tokens": tokens,
                                    "kl_from_final": max(0.0, float(kl[index])) if kl is not None else None,
                                    "target_log_probability": float(logprobs[index, target_token]) if target_token is not None else None})
        return {"method": method, "variant_id": run["variant_id"], "run_id": run_id,
                "final_reference_available": final is not None,
                "report_artifact_id": tools.artifacts.save_records("lens", records), "preview": records[:5]}

    def probe_representation(self, artifact_id, labels, train_indices, evaluation_indices, seed=0,
                             control_labels=None, input_baseline_artifact_id=None, token_position=None):
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
        class_names = model.classes_[-model.coef_.shape[0]:].tolist()
        weights = artifacts.save_array("probe_weights", model.coef_, ["class", "channel"], classes=class_names,
                                      site_id=info.get("site_id"))
        report = [{"sample": sample, "label": labels[sample], "prediction": predicted[index],
                   "control_label": control[sample], "control_prediction": control_predicted[index]}
                  for index, sample in enumerate(evaluation)]
        return {"source_artifact_id": artifact_id, "site_id": info.get("site_id"), "method": "logistic_regression",
                "accuracy": accuracy, "control_accuracy": control_accuracy, "selectivity": accuracy - control_accuracy,
                "input_baseline_accuracy": baseline_accuracy, "train_samples": len(train),
                "evaluation_samples": len(evaluation), "classes": class_names, "coefficients_artifact_id": weights,
                "report_artifact_id": artifacts.save_records("probe_predictions", report)}
