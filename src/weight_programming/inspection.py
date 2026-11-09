"""把现有前向服务接到通用的模型观测工具。"""

from copy import deepcopy
from functools import partial
from importlib.util import find_spec
from pathlib import Path
import re

from .artifacts import ArtifactStore
from .attribution_tools import AttributionTools
from .circuit_tools import attention_flow, attention_flow_graph, attention_layers, attention_rollout
from .comparison_tools import ComparisonTools
from .delta_tools import DeltaTools
from .edge_tracing import EdgeTracing
from .editing_tools import merge_updates, project_weight, prune_and_compensate
from .evidence_tools import render_evidence
from .evaluation_tools import BoxedExactGrader, EvaluationTools
from .fact_tools import FactTools
from .feature_circuits import FeatureCircuits
from .feature_tools import FeatureTools
from .forward_service import ForwardService, PracticeSet
from .intervention_tools import InterventionTools, _PAIR as _INTERVENTION_PAIR
from .model_registry import register_model
from .patches import VersionStore
from .python_tools import ARRAY_PATTERN, RUN_PATTERN, SESSION_PATTERN, PythonTools
from .representation_tools import RepresentationTools, activation_rows
from .structure_tools import StructureTools
from .tool_registry import ToolRegistry, ToolSpec, object_schema
from .training_tools import TrainingTools


_ID = {"type": "string", "minLength": 1}
_TEXTS = {"type": "array", "items": _ID, "minItems": 1}
_INDEX = {"type": "integer", "minimum": 0}
_POSITIVE = {"type": "integer", "minimum": 1}
_SITES = {"type": "array", "items": _ID, "uniqueItems": True}


def result_schema(*fields):
    """两种调用方式都能依赖的最小结果字段。"""
    return {"type": "object", "required": list(fields)}


class InspectionTools:
    """在一个由宿主注册的检查点上发现、运行和读取观测。"""

    def __init__(self, forward, *, allow_edits=False, allow_interventions=False, allow_gradients=False,
                 comparison=None, training_data=None, allow_training=False, allow_python=False, kernel_python=None,
                 fact_editor_python=None, fact_editor_source=None, quantization_python=None, slicegpt_python=None,
                 circuit_tracer_python=None):
        if training_data is not None and (training_data.role != "train" or not training_data.rows):
            raise ValueError("Register a nonempty training dataset with role=train")
        if allow_training and training_data is None:
            raise ValueError("Training permission requires a host-registered training dataset")
        if allow_training and find_spec("peft") is None:
            raise ValueError("Install the peft extra to enable adapter training")
        if (fact_editor_python is None) != (fact_editor_source is None):
            raise ValueError("Provide both the EasyEdit Python and source paths")
        if fact_editor_python is not None and not (allow_edits and allow_gradients):
            raise ValueError("Registering the fact editor requires edit and gradient permissions")
        self.forward = forward
        self.is_language_model = "generate" in forward.backend.modes
        if not self.is_language_model and (allow_interventions or allow_gradients or allow_training):
            raise ValueError("This classifier backend supports prediction, snapshots and weight editing")
        self.artifacts = ArtifactStore(forward.work_dir)
        self.versions = VersionStore(forward.work_dir.parent, forward.store, forward.backend.parameters)
        base = self.versions.base
        self.base_id = base["base_id"]
        self.variant_id = base["variant_id"]
        self._sites = forward.backend.observation_sites()
        full_attention = set(forward.backend.full_attention_layers)
        self._attention_prefix = next((i for i in range(forward.backend.num_layers) if i not in full_attention),
                                      forward.backend.num_layers)
        self._attention_limits = {"attention_rollout": 256}
        if find_spec("networkx") is not None:
            self._attention_limits["attention_flow"] = 32
        self._module_paths = {id(module): name for name, module in forward.backend.model.named_modules()}
        self._aliases = {}
        for name, parameter in forward.backend.parameters.items():
            self._aliases.setdefault(id(parameter), []).append(name)
        self.allow_edits = allow_edits
        self.allow_interventions = allow_interventions
        self.allow_gradients = allow_gradients
        self.allow_training = allow_training
        self.training_data = training_data
        self.interventions = InterventionTools(forward, self.versions, self.artifacts, self._sites,
                                               allow_gradients=allow_gradients)
        self.edge_tracing = EdgeTracing(self)
        self.feature_circuits = FeatureCircuits(self, circuit_tracer_python)
        self.delta = DeltaTools(self.versions, self.artifacts, self._input_rows, forward.backend)
        self.registered_adapters = []
        self.structures = StructureTools(self, quantization_python=quantization_python, slicegpt_python=slicegpt_python)
        self.training = TrainingTools(self)
        self.evaluation = EvaluationTools(forward, self.versions, self.artifacts, self.model_forward)
        self.comparison = comparison
        self.comparisons = ComparisonTools(self)
        self.representations = RepresentationTools(self)
        self.features = FeatureTools(self)
        self.attribution = AttributionTools(forward, self.versions, self.artifacts)
        self.python = PythonTools(self, kernel_python) if allow_python else None
        self.fact_editor = FactTools(self, fact_editor_python, fact_editor_source) if fact_editor_python else None
        permissions = {"observe"}
        if forward.practice.role == "development" and forward.grader is not None:
            permissions.add("evaluate")
        if allow_edits:
            permissions.add("edit")
        if allow_interventions:
            permissions.add("intervene")
        if allow_training:
            permissions.add("train")
        if allow_python:
            permissions.add("python")
        if allow_gradients:
            if find_spec("captum") is None:
                raise ValueError("Install the gradients extra to enable attribution")
            permissions.add("gradient")
        self.registry = ToolRegistry(self._specs(), permissions=permissions)

    def _specs(self):
        """在同一处声明 Python 调用参数与 MCP 工具定义。"""
        teacher = object_schema({"mode": {"const": "teacher_forced"}, "variant_id": _ID,
                                 "texts": _TEXTS, "observe": _SITES,
                                 "topk": _INDEX, "token_logprobs": {"type": "boolean"}},
                                "mode", "variant_id", "texts")
        generation = object_schema({"mode": {"const": "generate"}, "variant_id": _ID,
                                    "prompts": _TEXTS, "observe": _SITES,
                                    "n": _POSITIVE, "max_new_tokens": _POSITIVE},
                                   "mode", "variant_id", "prompts", "max_new_tokens")
        prediction = object_schema({"mode": {"const": "predict"}, "variant_id": _ID,
                                    "texts": _TEXTS, "observe": _SITES}, "mode", "variant_id", "texts")
        forward_schemas = [teacher, generation] if self.is_language_model else [prediction]
        array_region = {"type": "array", "items": {"type": "array", "items": _INDEX,
                                                     "minItems": 2, "maxItems": 2}}
        edit = object_schema({"tensor": _ID, "op": {"enum": ["add", "replace"]},
                              "region": array_region, "value_region": array_region,
                              "values": {"type": ["array", "number"]}, "value_artifact_id": _ID},
                             "tensor", "op")
        edit["oneOf"] = [{"required": ["values"],
                          "not": {"anyOf": [{"required": ["value_artifact_id"]}, {"required": ["value_region"]}]}},
                         {"required": ["value_artifact_id"], "not": {"required": ["values"]}}]
        input_artifacts = {"type": "object", "additionalProperties": _ID}
        python_specs = ()
        if self.python is not None:
            array_name = {"type": "string", "pattern": ARRAY_PATTERN}
            named_inputs = {**input_artifacts, "propertyNames": array_name, "maxProperties": 8}
            named_outputs = {"type": "object", "propertyNames": array_name, "maxProperties": 8,
                             "additionalProperties": {**_SITES, "maxItems": 4}}
            python_edit = object_schema({"tensor": _ID, "op": {"enum": ["add", "replace"]},
                                         "region": array_region, "value": array_name}, "tensor", "op", "value")
            python_specs = (ToolSpec("execute_python", "Run Python in a persistent isolated CPU session on one variant.",
                                      object_schema({"session_id": {"type": "string", "pattern": SESSION_PATTERN},
                                                     "variant_id": _ID, "code": {**_ID, "maxLength": 32_768},
                                                     "input_artifacts": named_inputs, "outputs": named_outputs,
                                                     "edits": {"type": "array", "items": python_edit, "maxItems": 16}},
                                                    "session_id", "variant_id", "code"),
                                      self.python.execute_python, permission="python",
                                      output_schema=result_schema("run_id", "session_id", "status", "artifacts", "patch_id")),)
        preference_pair = object_schema({"id": _ID, "prompt": _ID, "chosen": _ID, "rejected": _ID},
                                        "id", "prompt", "chosen", "rejected")
        workspace = {"enum": ["primary", "comparison"]}
        evaluation_fields = {"dataset_id": _ID, "case_ids": {**_SITES, "minItems": 1, "maxItems": 128},
                             "max_new_tokens": _POSITIVE, "samples": _POSITIVE, "metric": {"const": "boxed_exact"},
                             "bootstrap_draws": {"type": "integer", "minimum": 100, "maximum": 10000}}
        training_ids = {**_SITES, "minItems": 1, "maxItems": 32}
        structural_fields = [object_schema({"variant_id": _ID, "method": {"const": "block_influence"},
                                            "calibration_texts": {**_TEXTS, "maxItems": 32}, "drop_layers": _POSITIVE},
                                           "variant_id", "method", "calibration_texts", "drop_layers")]
        if self.allow_gradients and self.training_data is not None:
            structural_fields.append(object_schema({"variant_id": _ID, "method": {"const": "taylor_mlp"},
                                                    "calibration_case_ids": training_ids,
                                                    "fraction": {"type": "number", "exclusiveMinimum": 0,
                                                                 "exclusiveMaximum": 1}},
                                                   "variant_id", "method", "calibration_case_ids", "fraction"))
        if self.structures.external.quantization_python is not None:
            structural_fields.append(object_schema({"variant_id": _ID, "method": {"enum": ["gptq", "awq"]},
                "calibration_texts": {**_TEXTS, "maxItems": 128}, "group_size": {"enum": [32, 64, 128]}},
                "variant_id", "method", "calibration_texts"))
        if self.structures.external.slicegpt_python is not None:
            structural_fields.append(object_schema({"variant_id": _ID, "method": {"const": "slicegpt"},
                "calibration_texts": {**_TEXTS, "maxItems": 128}, "target_hidden_size": {**_POSITIVE, "multipleOf": 8}},
                "variant_id", "method", "calibration_texts", "target_hidden_size"))
        structures = ()
        if self.structures.supported:
            structures = (ToolSpec("compress_model", "Export a smaller checkpoint using structural pruning or packed quantization.",
                                   {"type": "object", "oneOf": structural_fields}, self.structures.compress_model,
                                   permission="edit", output_schema=result_schema("checkpoint_path", "parameters", "layer_count")),)
        fact_editing = ()
        if self.fact_editor is not None:
            fact_text = {**_ID, "maxLength": 2048}
            fact = object_schema({"subject": fact_text, "prompt": fact_text,
                                  "target_true": fact_text, "target_new": fact_text},
                                 "subject", "prompt", "target_true", "target_new")
            positive_number = {"type": "number", "exclusiveMinimum": 0}
            fact_editing = (ToolSpec("edit_facts", "Propose a ROME or MEMIT patch using a host-configured EasyEdit runtime.",
                                     object_schema({"parent_variant_id": _ID, "method": {"enum": ["rome", "memit"]},
                                                    "facts": {"type": "array", "items": fact, "minItems": 1, "maxItems": 8},
                                                    "layers": {"type": "array", "items": _INDEX, "minItems": 1, "maxItems": 6, "uniqueItems": True},
                                                    "calibration_texts": {**_TEXTS, "maxItems": 32},
                                                    "steps": {**_POSITIVE, "maximum": 64}, "learning_rate": positive_number,
                                                    "relative_damping": positive_number, "mom2_weight": positive_number},
                                                   "parent_variant_id", "method", "facts", "layers", "calibration_texts"),
                                     self.fact_editor.edit_facts, permission="edit",
                                     output_schema=result_schema("patch_id", "report_artifact_id", "changed_elements")),)
        sample_indices = {"type": "array", "items": _INDEX, "minItems": 1, "uniqueItems": True}
        feature_options = {"sample_indices": {**sample_indices, "maxItems": 128}, "token_position": _INDEX,
                           "feature_ids": {**sample_indices, "maxItems": 32},
                           "top_features": {**_POSITIVE, "maximum": 32}, "top_examples": {**_POSITIVE, "maximum": 8}}
        crosscoder_options = object_schema({"encoder_id": _ID, "left_artifact_id": _ID, "right_artifact_id": _ID,
                                             **feature_options}, "encoder_id", "left_artifact_id", "right_artifact_id")
        feature_common = {"encoder_id": _ID, "artifact_id": _ID, **feature_options}
        sae_schema = object_schema({**feature_common, "method": {"const": "sae"}}, "encoder_id", "artifact_id")
        crosscoder_schema = object_schema({**feature_common, "method": {"const": "crosscoder"},
                                           "right_artifact_id": _ID, "right_workspace": workspace},
                                          "encoder_id", "artifact_id", "method", "right_artifact_id")
        probe = ()
        if find_spec("sklearn") is not None:
            probe_fields = {"artifact_id": _ID, "labels": _TEXTS, "train_indices": sample_indices,
                            "evaluation_indices": sample_indices, "seed": _INDEX, "control_labels": _TEXTS,
                            "input_baseline_artifact_id": _ID, "token_position": _INDEX}
            if self.interventions.has_leace:
                probe_fields["fit_eraser"] = {"type": "boolean"}
            probe = (ToolSpec("probe_representation", "Fit a linear probe and optionally fit a LEACE eraser.",
                              object_schema(probe_fields, "artifact_id", "labels", "train_indices", "evaluation_indices"),
                              self.representations.probe_representation,
                              output_schema=result_schema("coefficients_artifact_id", "accuracy", "control_accuracy")),)
        plots = ()
        if find_spec("matplotlib") is not None:
            plot_fields = {"artifact_id": _ID, "format": {"enum": ["png", "pdf"]}}
            plot_schemas = [object_schema({**plot_fields,
                "view": {"enum": ["activation_heatmap", "attention_heatmap", "patching_heatmap",
                                  "similarity_heatmap", "attribution_heatmap"]}, "region": array_region,
                "axes": {"type": "array", "items": _ID, "minItems": 2, "maxItems": 2}}, "artifact_id", "view")]
            if find_spec("networkx") is not None:
                plot_schemas.append(object_schema({**plot_fields, "view": {"const": "graph"},
                    "max_edges": {**_POSITIVE, "maximum": 32}}, "artifact_id", "view"))
            plots = (ToolSpec("render_evidence", "Render a heatmap or directed graph from a registered artifact.",
                              {"type": "object", "oneOf": plot_schemas}, partial(render_evidence, self.artifacts),
                              output_schema={**result_schema("artifact_id", "plot_path"),
                                             "anyOf": [{"required": ["top_cells"]}, {"required": ["top_edges"]}]}),)
        traces = ()
        trace_schemas = []
        if self._attention_prefix:
            trace_fields = {"run_id": _ID, "sample_index": _INDEX, "through_layer": _INDEX}
            trace_schemas.append(object_schema({**trace_fields, "method": {"const": "attention_rollout"}}, "run_id", "method"))
            if "attention_flow" in self._attention_limits:
                trace_schemas.append(object_schema({**trace_fields, "method": {"const": "attention_flow"},
                                                    "query_position": _INDEX}, "run_id", "method", "query_position"))
        if self.edge_tracing.methods:
            pair = {**_INTERVENTION_PAIR, "required": ["clean_text", "corrupt_text", "target_token", "alternative_token"]}
            cases = {"type": "array", "items": pair, "minItems": 1, "maxItems": 8}
            links = {"type": "array", "items": object_schema({"source": _ID, "target": _ID}, "source", "target"),
                     "minItems": 1, "maxItems": 128, "uniqueItems": True}
            metrics = ["logit_difference", "target_probability", "target_log_probability"]
            common = {"variant_id": _ID, "pairs": cases, "metric": {"enum": metrics},
                      "forward_budget": {**_POSITIVE, "maximum": 256}}
            required = ("variant_id", "pairs", "method")
            trace_schemas.extend((
                object_schema({**common, "method": {"const": "path_patching"}, "review_pairs": cases,
                               "path": {**_SITES, "minItems": 2, "maxItems": 32}, "strength": {"type": "number"}},
                              *required, "path"),
                object_schema({**common, "method": {"const": "acdc"}, "review_pairs": cases, "edges": links,
                               "metric": {"enum": ["kl_divergence", *metrics]}, "control_seed": _INDEX,
                               "threshold": {"type": "number", "exclusiveMinimum": 0}}, *required, "edges", "threshold"),
            ))
            if "eap" in self.edge_tracing.methods:
                trace_schemas.append(object_schema({**common, "method": {"const": "eap"}, "edges": links,
                                                    "top_k": {**_POSITIVE, "maximum": 128}}, *required, "edges"))
        if self.feature_circuits.python is not None:
            influence = {"type": "number", "exclusiveMinimum": 0, "maximum": 1}
            trace_schemas.append(object_schema({"method": {"const": "feature_attribution_graph"}, "variant_id": _ID,
                "transcoder_id": _ID, "text": _ID, "target_tokens": {"type": "array", "items": _INDEX,
                    "minItems": 1, "maxItems": 4, "uniqueItems": True},
                "max_feature_nodes": {**_POSITIVE, "maximum": 128}, "node_threshold": influence, "edge_threshold": influence},
                "method", "variant_id", "transcoder_id", "text", "target_tokens"))
        if trace_schemas:
            traces = (ToolSpec("trace_circuit", "Trace attention, residual edges, or a replacement-model feature graph.",
                               {"type": "object", "oneOf": trace_schemas},
                               self.trace_circuit,
                               output_schema=result_schema("artifact_id", "evidence_type", "source_run_id", "source_artifact_ids")),)
        attribution_fields = {"variant_id": _ID, "text": _ID, "baseline_text": _ID,
                              "steps": {"type": "integer", "minimum": 2, "maximum": 64}}
        attribution_required = ("variant_id", "text", "baseline_text")
        attribution_schemas = [
            object_schema({**attribution_fields, "target_token": _INDEX, "alternative_token": _INDEX,
                           "query_position": _INDEX, "method": {"const": "integrated_gradients"}},
                          *attribution_required, "target_token"),
            object_schema({**attribution_fields, "chosen": _ID, "rejected": _ID,
                           "aggregation": {"enum": ["sum", "mean"]}, "method": {"const": "contrastive_sequence"}},
                          *attribution_required, "method", "chosen", "rejected"),
        ]
        lens_fields = {"run_id": _ID, "site_ids": {**_SITES, "minItems": 1, "maxItems": 32},
                       "sample_indices": {**sample_indices, "maxItems": 32}, "token_position": _INDEX,
                       "topk": {"type": "integer", "minimum": 1, "maximum": 50}, "target_token": _INDEX}
        lens_schemas = [object_schema({**lens_fields, "method": {"const": "logit_lens"}}, "run_id", "site_ids"),
                        object_schema({**lens_fields, "method": {"const": "tuned_lens"}, "decoder_ref": _ID},
                                      "run_id", "site_ids", "method", "decoder_ref")]
        specs = (
            *python_specs,
            ToolSpec("describe_model", "List the registered checkpoint and its available observation sites.",
                     object_schema({"variant_id": _ID, "workspace": workspace}), self.describe_model,
                     output_schema=result_schema("variant_id", "model_type", "layers", "sites")),
            ToolSpec("model_forward", "Run a budgeted forward request with optional observations.",
                     {"type": "object", "oneOf": forward_schemas}, self.model_forward,
                     output_schema=result_schema("run_id", "variant_id", "status", "artifacts", "usage")),
            ToolSpec("read_artifact", "Read an array slice or a small page of generated records.",
                     object_schema({"artifact_id": _ID, "region": array_region, "offset": _INDEX, "workspace": workspace,
                                    "limit": {"type": "integer", "minimum": 1, "maximum": 32}},
                                   "artifact_id"), self.read_artifact,
                     output_schema=result_schema("artifact_id", "run_id", "kind")),
            ToolSpec("list_tensors", "List checkpoint tensor names, shapes and original precision.",
                     object_schema({"variant_id": _ID, "prefix": {"type": "string"}, "layer": _INDEX,
                                    "cursor": {"type": "object"},
                                    "limit": {"type": "integer", "minimum": 1, "maximum": 100}},
                                   "variant_id"), self.list_tensors,
                     output_schema=result_schema("variant_id", "tensors", "next_cursor")),
            ToolSpec("read_tensor", "Read a bounded slice directly from the checkpoint at native precision.",
                     object_schema({"variant_id": _ID, "name": _ID, "region": array_region},
                                   "variant_id", "name", "region"), self.read_tensor,
                     output_schema=result_schema("variant_id", "name", "shape", "dtype", "values")),
            ToolSpec("get_run", "Inspect a recorded forward run and its result identifiers.",
                     object_schema({"run_id": _ID, "workspace": workspace}, "run_id"), self.get_run,
                     output_schema=result_schema("run_id", "variant_id", "status", "usage")),
            ToolSpec("create_patch", "Save an atomic patch with native-precision replacement tensors.",
                     object_schema({"parent_variant_id": _ID, "method": _ID,
                                    "edits": {"type": "array", "items": edit, "minItems": 1}},
                                   "parent_variant_id", "edits", "method"), self.create_patch,
                     permission="edit", output_schema=result_schema("patch_id", "tensors")),
            ToolSpec("apply_patch", "Validate the parent and register a runnable candidate variant.",
                     object_schema({"parent_variant_id": _ID, "patch_id": _ID}, "parent_variant_id", "patch_id"),
                     self.versions.apply_patch, permission="edit",
                     output_schema=result_schema("variant_id", "parent_variant_id", "tensors")),
            ToolSpec("revert_patch", "Get the parent variant for subsequent calls.",
                     object_schema({"variant_id": _ID}, "variant_id"), self.versions.revert_patch,
                     permission="edit", output_schema=result_schema("variant_id", "reverted_variant_id")),
            ToolSpec("export_patch", "Copy a registered patch to a portable workspace bundle.",
                     object_schema({"patch_id": _ID}, "patch_id"), self.versions.export_patch,
                     permission="edit", output_schema=result_schema("patch_id", "bundle_path")),
            ToolSpec("import_patch", "Validate a patch placed in the workspace imports directory by the host.",
                     object_schema({"bundle_id": _ID}, "bundle_id"), self.versions.import_patch,
                     permission="edit", output_schema=result_schema("patch_id", "parent_variant_id")),
            ToolSpec("export_model", "Export a complete Transformers checkpoint for a registered variant.",
                     object_schema({"variant_id": _ID, "format": {"const": "transformers_safetensors"}},
                                   "variant_id"), self.versions.export_model, permission="edit",
                     output_schema=result_schema("variant_id", "format", "checkpoint_path", "base_id")),
            ToolSpec("project_weight", "Remove part of a declared output direction from selected columns.",
                     object_schema({"parent_variant_id": _ID, "tensor": _ID,
                                    "direction": {"type": "array", "items": {"type": "number"}, "minItems": 1},
                                    "direction_artifacts": {"type": "array", "items": _ID, "minItems": 2, "maxItems": 2},
                                    "sample_indices": sample_indices, "token_position": _INDEX,
                                    "columns": {"type": "array", "items": _INDEX, "minItems": 1, "uniqueItems": True},
                                    "strength": {"type": "number"}}, "parent_variant_id", "tensor", "columns"),
                     self.project_weight, permission="edit",
                     output_schema=result_schema("patch_id", "columns", "realized_norm")),
            ToolSpec("merge_updates", "Merge same-parent patches using task arithmetic, TIES or DARE.",
                     object_schema({"parent_variant_id": _ID, "patch_ids": {"type": "array", "items": _ID,
                                                                          "minItems": 1, "uniqueItems": True},
                                    "adapter_ids": {"type": "array", "items": _ID, "minItems": 1, "uniqueItems": True},
                                    "method": {"enum": ["task_arithmetic", "ties", "dare"]},
                                    "coefficients": {"type": "array", "items": {"type": "number"}},
                                    "density": {"type": "number", "minimum": 0, "maximum": 1},
                                    "drop_probability": {"type": "number", "minimum": 0, "exclusiveMaximum": 1},
                                    "seed": {**_INDEX, "maximum": 2**32 - 1},
                                    "dare_merge": {"enum": ["task_arithmetic", "ties"]},
                                    "target_norm": {"type": "number", "exclusiveMinimum": 0}},
                                   "parent_variant_id", "method"),
                     self.merge_updates, permission="edit",
                     output_schema=result_schema("patch_id", "input_patch_ids", "merge_evidence", "realized_norm")),
            ToolSpec("prune_and_compensate", "Prune a linear layer using observed inputs and save its mask and patch.",
                     object_schema({"parent_variant_id": _ID, "tensor": _ID, "input_artifact_id": _ID,
                                    "method": {"enum": ["wanda", "sparsegpt_fixed_mask", "sparsegpt_adaptive"]},
                                    "sparsity": {"type": "number", "minimum": 0, "maximum": 1},
                                    "mask_artifact_id": _ID,
                                    "relative_damping": {"type": "number", "minimum": 0}},
                                   "parent_variant_id", "tensor", "input_artifact_id", "method"),
                     self.prune_and_compensate, permission="edit",
                     output_schema=result_schema("patch_id", "mask_artifact_id", "realized_zero_fraction")),
            *self.interventions.specs(),
            ToolSpec("analyze_weight_delta", "Measure changes in registered variants or an imported LoRA update.",
                     object_schema({"base_variant_id": _ID, "variant_id": _ID, "adapter_id": _ID,
                                    "tensors": _SITES, "other_variant_id": _ID,
                                    "direction_rank": {"type": "integer", "minimum": 0, "maximum": 32},
                                    "direction_solver": {"enum": ["exact", "randomized"]},
                                    "direction_seed": {**_INDEX, "maximum": 2**32 - 1},
                                    "direction_iterations": {"type": "integer", "minimum": 1, "maximum": 8},
                                    "input_artifacts": input_artifacts}, "base_variant_id"),
                     self.delta.analyze_weight_delta,
                     output_schema=result_schema("report_artifact_id", "delta_norm", "changed_tensors")),
            ToolSpec("compress_update", "Compress an existing patch or WSP LoRA update and optionally materialize it.",
                     object_schema({"parent_variant_id": _ID,
                                    "rank": {"oneOf": [_POSITIVE, {"type": "object", "minProperties": 1,
                                                                    "additionalProperties": _POSITIVE}]},
                                    "patch_id": _ID, "adapter_id": _ID,
                                    "method": {"enum": ["truncated_svd", "activation_weighted_svd"]},
                                    "input_artifacts": input_artifacts, "materialize": {"type": "boolean"}},
                                   "parent_variant_id", "rank"), self.delta.compress_update, permission="edit",
                     output_schema=result_schema("adapter_id", "stored_parameters", "report_artifact_id")),
            *structures,
            *fact_editing,
            ToolSpec("train_update", "Train an adapter on authorized data using LoRA, DPO, distillation or pruning recovery.",
                     object_schema({"parent_variant_id": _ID, "dataset_id": _ID, "case_ids": training_ids,
                                    "targets": training_ids,
                                    "method": {"enum": ["lora", "dpo", "distillation", "post_pruning_recovery"]},
                                    "rank": {**_POSITIVE, "maximum": 128}, "alpha": _POSITIVE,
                                    "steps": {**_POSITIVE, "maximum": 32},
                                    "learning_rate": {"type": "number", "exclusiveMinimum": 0},
                                    "beta": {"type": "number", "exclusiveMinimum": 0},
                                    "seed": {**_INDEX, "maximum": 2**32 - 1}, "materialize": {"type": "boolean"}},
                                   "parent_variant_id", "dataset_id", "case_ids", "targets"),
                     self.training.train_update, permission="train",
                     output_schema=result_schema("adapter_id", "adapter_path", "patch_id", "method", "losses")),
            ToolSpec("evaluate_variants", "Run paired generation on a host-registered development dataset.",
                     object_schema({"baseline_variant_id": _ID,
                                    "candidate_variant_ids": {**_SITES, "minItems": 1, "maxItems": 8},
                                    **evaluation_fields},
                                   "baseline_variant_id", "candidate_variant_ids", "dataset_id", "case_ids", "max_new_tokens"),
                     self.evaluation.evaluate_variants, permission="evaluate",
                     output_schema=result_schema("summaries", "comparisons", "dataset_sha256")),
            ToolSpec("score_preferences", "Compare chosen and rejected response likelihoods relative to a reference.",
                     object_schema({"variant_id": _ID, "reference_variant_id": _ID,
                                    "pairs": {"type": "array", "items": preference_pair, "minItems": 1},
                                    "aggregation": {"enum": ["sum", "mean"]}},
                                   "variant_id", "reference_variant_id", "pairs"), self.evaluation.score_preferences,
                     output_schema=result_schema("report_artifact_id", "mean_relative_margin")),
            ToolSpec("compare_models", "Compare shared weight coordinates, representations or development behavior.",
                     object_schema({"left_variant_id": _ID, "right_variant_id": _ID, "right_workspace": workspace,
                                    "views": {"type": "array", "items": {"enum": ["weights", "representations", "crosscoder"] +
                                                                                  (["behavior"] if self.is_language_model else [])},
                                              "minItems": 1, "uniqueItems": True},
                                    "texts": {**_TEXTS, "minItems": 2, "maxItems": 128}, "tensors": _SITES,
                                    "left_sites": {**_SITES, "minItems": 1, "maxItems": 32},
                                    "right_sites": {**_SITES, "minItems": 1, "maxItems": 32},
                                    "crosscoder": crosscoder_options,
                                    "evaluation": object_schema(evaluation_fields, "dataset_id", "case_ids", "max_new_tokens")},
                                   "left_variant_id", "right_variant_id", "views"), self.comparisons.compare_models,
                     output_schema=result_schema("left_variant_id", "right_variant_id", "views")),
            ToolSpec("extract_features", "Read sparse SAE or shared crosscoder features from saved activations.",
                     {"type": "object", "oneOf": [sae_schema, crosscoder_schema]}, self.features.extract_features,
                     output_schema=result_schema("encoder_id", "method", "artifact_id", "reconstruction", "report_artifact_id")),
            ToolSpec("scan_lens", "Read intermediate residuals directly or through a fitted translator.",
                     {"type": "object", "oneOf": lens_schemas}, self.representations.scan_lens,
                     output_schema=result_schema("report_artifact_id", "final_reference_available")),
            *traces,
            *probe,
            ToolSpec("attribute_output", "Attribute a token score or an answer margin using integrated gradients.",
                     {"type": "object", "oneOf": attribution_schemas},
                     self.attribution.attribute_output, permission="gradient",
                     output_schema=result_schema("artifact_id", "score_change", "integration_error")),
            *plots,
        )
        language_tools = {"scan_lens", "score_preferences", "attribute_output", "run_intervention", "localize_behavior",
                          "train_update", "evaluate_variants"}
        return tuple(spec for spec in specs if self.is_language_model or spec.name not in language_tools)

    def workspace_tools(self, workspace):
        """解析宿主明确注册的主模型或比较模型。"""
        if workspace == "primary":
            return self
        if workspace == "comparison" and self.comparison is not None:
            return self.comparison
        raise ValueError("No model is registered in the requested workspace")

    def describe_model(self, variant_id=None, workspace="primary"):
        """返回模型的实际能力，供调用者选择可用站点。"""
        if workspace != "primary":
            return self.workspace_tools(workspace).describe_model(variant_id)
        variant = self.versions.variant(self.variant_id if variant_id is None else variant_id)
        backend = self.forward.backend
        full = set(backend.full_attention_layers)
        transcoders = self.feature_circuits.available(variant["variant_id"])
        return {**variant,
                "model_type": backend.identity["model_type"], "backend": backend.identity["backend"],
                "context_tokens": backend.context_length, "vocab_size": backend.vocab_size,
                "modes": list(backend.modes), "layer_count": backend.num_layers,
                "output_kind": "next_token_logits" if self.is_language_model else "sequence_classification",
                "label_names": [backend.model.config.id2label[i] for i in range(backend.model.config.num_labels)]
                if not self.is_language_model else None,
                "layers": [{"index": layer, "attention": "full" if layer in full else "linear"}
                           for layer in range(backend.num_layers)],
                "sites": deepcopy(list(self._sites.values())), "weight_editing": self.allow_edits,
                "activation_interventions": self.allow_interventions,
                "intervention_methods": (["replace", "ablation", "direction"] +
                                         (["concept_erasure"] if self.interventions.has_leace else []))
                if self.allow_interventions else [],
                "localization_methods": (["exact_scan"] + (["atp_star"] if self.interventions.has_atp_star else []))
                if self.allow_interventions else [],
                "python_execution": self.python is not None,
                "fact_editing_methods": ["rome", "memit"] if self.fact_editor is not None else [],
                "compression_methods": ((["block_influence"] if self.structures.supported else []) +
                                        (["taylor_mlp"] if self.structures.supported and self.allow_gradients
                                          and self.training_data is not None else []) +
                                        (["gptq", "awq"] if self.structures.external.quantization_python else []) +
                                        (["slicegpt"] if self.structures.external.slicegpt_python else []))
                if self.allow_edits else [],
                "lens_methods": ["logit_lens", "tuned_lens"] if self.is_language_model else [],
                "feature_dictionaries": self.features.available(variant["variant_id"]),
                "feature_transcoders": transcoders,
                "registered_adapters": self.registered_adapters,
                "gradients": self.allow_gradients,
                "training_dataset_id": self.training_data.name
                if self.training_data is not None and (self.allow_training or (self.allow_edits and self.allow_gradients))
                else None,
                "trace_circuit_methods": {**{method: {"through_layers": list(range(self._attention_prefix)), "max_tokens": limit}
                                              for method, limit in self._attention_limits.items() if self._attention_prefix},
                                          **self.edge_tracing.capabilities(),
                                          **({"feature_attribution_graph": {"max_text_tokens": 31, "transcoder_ids": transcoders}}
                                             if self.feature_circuits.python else {})},
                "circuit_nodes": list(self.edge_tracing.circuit.order) if self.edge_tracing.circuit else [],
                "comparison_variant_id": self.comparison.variant_id if self.comparison else None}

    def trace_circuit(self, run_id=None, method=None, sample_index=0, through_layer=None, query_position=None, **arguments):
        """分析已采集的注意力，保留有效词元、来源和证据类型。"""
        if method in ("path_patching", "acdc", "eap"):
            return self.edge_tracing.run(method, **arguments)
        if method == "feature_attribution_graph":
            return self.feature_circuits.trace(**arguments)
        if method not in self._attention_limits:
            raise ValueError(f"Unsupported circuit method: {method}")
        run = self.get_run(run_id)
        if run["status"] != "succeeded" or "valid_tokens" not in run["artifacts"]:
            raise ValueError("Attention analysis requires a completed teacher-forced observation")
        last = self.forward.backend.num_layers - 1 if through_layer is None else through_layer
        if not 0 <= last < self._attention_prefix:
            raise ValueError("Attention analysis requires a full self-attention prefix through the selected layer")
        site_ids = [f"block.{layer}.attention.probs" for layer in range(last + 1)]
        matrices, source_ids, positions, token_ids = attention_layers(self.artifacts, run_id, site_ids, sample_index,
                                                                      self._attention_limits[method])
        if method == "attention_flow":
            if query_position not in positions:
                raise ValueError("Attention flow requires a valid query token position")
            matrix, raw_flow = attention_flow(matrices, positions.index(query_position))
            query_positions = [query_position]
        else:
            matrix = attention_rollout(matrices)
            raw_flow = None
            query_positions = positions
        evidence = {"method": method, "evidence_type": "descriptive_attention", "source_run_id": run_id,
                    "variant_id": run["variant_id"], "source_artifact_ids": source_ids,
                    "through_layer": last, "sample_index": sample_index, "token_positions": positions,
                    "token_ids": token_ids, "query_positions": query_positions}
        if raw_flow is not None:
            evidence["raw_flow"] = raw_flow
        artifact_id = self.artifacts.save_array(method, matrix, ["query_token", "key_token"],
                                                metric=method.replace("_", " ") + " mass", **evidence)
        if raw_flow is not None:
            graph = attention_flow_graph(self.forward.backend.tokenizer, positions, token_ids, query_position,
                                         matrix[0], evidence, artifact_id)
            evidence["graph_artifact_id"] = self.artifacts.save_records("circuit_graph", [graph])
        return {"artifact_id": artifact_id, **evidence}

    def read_artifact(self, artifact_id, region=None, offset=None, limit=None, workspace="primary"):
        """从明确选定的工作区读取产物。"""
        return self.workspace_tools(workspace).artifacts.read(artifact_id, region, offset, limit)

    def _tensor_info(self, name):
        """补充执行模型中已知的轴和共享参数名称。"""
        from torch.nn import Embedding, Linear

        info = self.forward.store.tensor_info(name)
        module_path, _, field = name.rpartition(".")
        parameter = self.forward.backend.parameters.get(name)
        module = self.forward.backend.model.get_submodule(module_path) if parameter is not None else None
        axes = None
        if isinstance(module, Linear):
            axes = ["output", "input"] if field == "weight" else ["output"]
        elif isinstance(module, Embedding) and field == "weight":
            axes = ["token", "channel"]
        return {**info, "axes": axes, "aliases": list(self._aliases.get(id(parameter), []))}

    def list_tensors(self, variant_id, prefix="", layer=None, cursor=None, limit=20):
        """按稳定顺序枚举模型权重，游标绑定模型和筛选条件。"""
        origins = self.versions.tensor_versions(variant_id)
        layer_prefix = ""
        if layer is not None:
            if layer >= self.forward.backend.num_layers:
                raise ValueError("Layer outside the model")
            layer_prefix = self._module_paths[id(self.forward.backend.layers[layer])] + "."
        selected = [name for name in self.forward.store.names
                    if name.startswith(prefix) and name.startswith(layer_prefix)]
        scope = {"variant_id": variant_id, "prefix": prefix, "layer": layer}
        start = 0
        if cursor is not None:
            if {key: cursor.get(key) for key in scope} != scope or cursor.get("after") not in selected:
                raise ValueError("Cursor does not match this tensor query")
            start = selected.index(cursor["after"]) + 1
        page = selected[start:start + limit]
        next_cursor = dict(scope, after=page[-1]) if start + limit < len(selected) else None
        tensors = [dict(self._tensor_info(name), source_variant_id=origins.get(name, self.variant_id))
                   for name in page]
        return {"variant_id": variant_id, "tensors": tensors,
                "next_cursor": next_cursor}

    def read_tensor(self, variant_id, name, region):
        """读取检查点切片并保留来源、坐标和原始精度。"""
        values, source = self.versions.read_slice(variant_id, name, region)
        return {"variant_id": variant_id, **self._tensor_info(name),
                "source_variant_id": source, "region": region, "values": values.tolist()}

    def create_patch(self, parent_variant_id, edits, method):
        """从小型内联数组或已有观测产物读取修改值。"""
        def read_value(edit):
            if "values" in edit:
                return edit["values"]
            artifact = self.artifacts.read(edit["value_artifact_id"], region=edit.get("value_region"))
            if artifact["kind"] != "array":
                raise ValueError("An edit value must reference an array artifact")
            return artifact["values"]

        source_ids = [edit["value_artifact_id"] for edit in edits if "value_artifact_id" in edit]
        return self.versions.create_patch(parent_variant_id, edits, method, read_value, source_ids)

    def project_weight(self, parent_variant_id, tensor, columns, direction=None, strength=1.0,
                       direction_artifacts=None, sample_indices=None, token_position=None):
        """从显式方向或两组同坐标激活构造方向，复用已有投影算法。"""
        import torch

        from .directions import contrast_direction

        if (direction is None) == (direction_artifacts is None):
            raise ValueError("Choose an explicit direction or exactly two activation artifacts")
        sources = []
        direction_id = None
        if direction_artifacts is not None:
            if len(direction_artifacts) != 2:
                raise ValueError("Provide source and target activation artifacts")
            data = [activation_rows(self.artifacts, ref, sample_indices, token_position) for ref in direction_artifacts]
            allowed = self.forward.backend.projection_output_sites(tensor)
            site_ids = [info["site_id"] for _, info, _, _ in data]
            if len(set(site_ids)) != 1 or site_ids[0] not in allowed:
                raise ValueError("Direction snapshots must share a verified output coordinate site for this weight")
            if any(self.get_run(info["run_id"])["variant_id"] != parent_variant_id for _, info, _, _ in data):
                raise ValueError("Direction snapshots must come from the selected parent variant")
            direction = contrast_direction(*(torch.from_numpy(rows.copy()) for rows, _, _, _ in data))
            sources = list(direction_artifacts)
            direction_id = self.artifacts.save_array("contrast_direction", direction.numpy(), ["channel"],
                site_id=site_ids[0], variant_id=parent_variant_id, source_artifact_ids=sources,
                method="paired_mean_difference", sample_indices=data[0][2],
                source_token_positions=data[0][3], target_token_positions=data[1][3])
            sources.append(direction_id)
        elif sample_indices is not None or token_position is not None:
            raise ValueError("Sample and token selection apply to activation-derived directions")
        result = project_weight(self.versions, parent_variant_id, tensor, direction, columns, strength, sources)
        return {**result, "direction_artifact_id": direction_id}

    def merge_updates(self, parent_variant_id, method, patch_ids=(), adapter_ids=(), **options):
        """让原生 PEFT 导入和训练更新复用同一合并实现。"""
        factors = [self.delta._adapter(adapter_id, parent_variant_id) for adapter_id in adapter_ids]
        deltas = [{name: b @ a for name, (a, b) in update.items()} for update in factors]
        return merge_updates(self.versions, parent_variant_id, patch_ids, method,
                             adapter_deltas=deltas, adapter_ids=adapter_ids, **options)

    def prune_and_compensate(self, parent_variant_id, tensor, input_artifact_id, method,
                             sparsity=None, mask_artifact_id=None, relative_damping=0.01):
        """用同版本、同线性层的有效输入词元做校准。"""
        inputs = self._input_rows(parent_variant_id, tensor, input_artifact_id)
        return prune_and_compensate(self.versions, self.artifacts, parent_variant_id, tensor, inputs,
                                    input_artifact_id, method, sparsity, mask_artifact_id, relative_damping)

    def _input_rows(self, parent_variant_id, tensor, input_artifact_id):
        """分析与编辑共用对齐检查，只返回有效词元。"""
        import torch

        values, info = self.artifacts.array(input_artifact_id)
        site = self._sites.get(info.get("site_id"))
        if site is None or site["kind"] != "mlp" or values.ndim != 3:
            raise ValueError("Use a teacher-forced snapshot of the layer's MLP input")
        module = self.forward.backend.linear_input_module(site["index"])
        if tensor != self._module_paths[id(module)] + ".weight":
            raise ValueError("Calibration input belongs to a different weight")
        run_id = info["run_id"]
        if self.get_run(run_id)["variant_id"] != parent_variant_id:
            raise ValueError("Calibration input belongs to a different parent variant")
        valid, _ = self.artifacts.array(f"{run_id}/valid_tokens")
        return torch.from_numpy(values[valid])

    def get_run(self, run_id, workspace="primary"):
        """查询本工作区账本上的运行状态。"""
        if workspace != "primary":
            return self.workspace_tools(workspace).get_run(run_id)
        if re.fullmatch(RUN_PATTERN, run_id):
            return PythonTools.get_run(self.versions.root, run_id)
        if not re.fullmatch(r"request_[0-9]+", run_id):
            raise ValueError("Invalid run ID")
        number = int(run_id.removeprefix("request_"))
        if run_id != f"request_{number:04d}":
            raise ValueError("Invalid run ID")
        row = self.forward.ledger.lookup(number)
        return {"run_id": run_id, "variant_id": row["variant_id"], "status": row["status"],
                "usage": row.get("details", {}), "artifacts": self.artifacts.references(run_id)}

    def close(self):
        """回收由当前工具实例启动的隔离进程。"""
        if self.python is not None:
            self.python.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def model_forward(self, variant_id, mode, texts=None, prompts=None, observe=(),
                      topk=0, token_logprobs=False, n=1, max_new_tokens=None):
        """把通用站点翻译成旧前向协议，复用预算、推理和落盘。"""
        self.versions.variant(variant_id)
        if mode not in self.forward.backend.modes:
            raise ValueError(f"Mode unavailable for this model: {mode}")
        sites = {}
        indices = {"hidden": [], "mlp": [], "mlp_out": [], "attention": []}
        for site_id in observe:
            if site_id not in self._sites or mode not in self._sites[site_id]["modes"]:
                raise ValueError(f"Unsupported observation site for {mode}: {site_id}")
            site = self._sites[site_id]
            kind, index = site["kind"], site["index"]
            sites[f"{kind}_{index}"] = site
            indices[kind].append(index)
        original = variant_id == self.variant_id
        request = {"type": mode, "model": "original" if original else "edited"}
        if mode in ("teacher_forced", "predict"):
            request["texts"] = texts
            request["return"] = {"hidden_states": indices["hidden"], "mlp_act": indices["mlp"],
                                 "attn_layers": indices["attention"], "topk": topk,
                                 "token_logprobs": token_logprobs}
            if indices["mlp_out"]:
                request["return"]["mlp_out"] = indices["mlp_out"]
        elif mode == "generate":
            request.update(prompts=prompts, n=n, max_tokens=max_new_tokens,
                           activations={"layers": indices["hidden"]})
        else:
            raise ValueError(f"Unknown mode: {mode}")
        snapshot = None if original else self.versions.snapshot(variant_id)
        result = self.forward.request(request, variant_snapshot=snapshot)
        artifacts = self.artifacts.publish_forward(result, sites)
        summary = {key: result[key] for key in ("input_tokens", "executed_model_forwards",
                                                "alignment", "summaries", "label_names") if key in result}
        return {"run_id": f"request_{result['attempt']:04d}", "variant_id": variant_id,
                "status": "succeeded", "artifacts": artifacts, "summary": summary,
                "usage": {"elapsed_seconds": result["elapsed_seconds"], "budget": result["budget"]}}


def open_inspection(model_dir, workspace, *, device="cpu", policy=None, revision=None, allow_edits=False,
                    allow_interventions=False, allow_gradients=False, practice=None, comparison=None,
                    training_data=None, allow_training=False, allow_python=False, kernel_python=None,
                    fact_editor_python=None, fact_editor_source=None, quantization_python=None, slicegpt_python=None,
                    circuit_tracer_python=None, peft_adapters=()):
    """从本地检查点打开研究工具，MCP 启动也调用此入口。"""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

    from .hf_backend import HuggingFaceBackend

    workspace = Path(workspace).resolve()
    store, _ = register_model(model_dir, workspace, revision=revision)
    model_dir = store.model_dir
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    load_options = {"local_files_only": True, "dtype": torch.float32,
                    "attn_implementation": "eager"}
    if config.model_type == "qwen3_5":
        from transformers import Qwen3_5ForConditionalGeneration

        from .qwen35_backend import Qwen35Backend

        model = Qwen3_5ForConditionalGeneration.from_pretrained(model_dir, **load_options)
        backend_type = Qwen35Backend
    elif config.model_type in ("qwen3", "llama"):
        model = AutoModelForCausalLM.from_pretrained(model_dir, **load_options)
        backend_type = HuggingFaceBackend
    elif config.model_type == "distilbert":
        from .encoder_backend import DistilBertBackend

        model = AutoModelForSequenceClassification.from_pretrained(model_dir, **load_options)
        backend_type = DistilBertBackend
    else:
        raise ValueError(f"Unsupported model type: {config.model_type}")
    backend = backend_type(model.to(device), tokenizer, store)
    grader = BoxedExactGrader() if "generate" in backend.modes and practice is not None and practice.role == "development" else None
    if practice is None:
        practice = PracticeSet([], role="development", name="direct_inputs")
    forward = ForwardService(backend, store, practice, workspace / "forward", policy, grader)
    tools = InspectionTools(forward, allow_edits=allow_edits, allow_interventions=allow_interventions,
                           allow_gradients=allow_gradients, comparison=comparison,
                           training_data=training_data, allow_training=allow_training,
                           allow_python=allow_python, kernel_python=kernel_python,
                           fact_editor_python=fact_editor_python, fact_editor_source=fact_editor_source,
                           quantization_python=quantization_python, slicegpt_python=slicegpt_python,
                           circuit_tracer_python=circuit_tracer_python)
    for adapter_dir in peft_adapters:
        tools.registered_adapters.append(tools.delta.import_peft_adapter(tools.variant_id, adapter_dir))
    return tools
