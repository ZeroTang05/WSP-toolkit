"""Text-only execution of Qwen3.5 conditional checkpoints, Transformers 5.3.0.

Uses the official HF hybrid decoder, including its unfused PyTorch DeltaNet
implementation. Does not claim numerical or throughput equivalence to vLLM.
Load Qwen3_5ForConditionalGeneration, not a remapped text-only state dict:
this preserves the published model.language_model.* weight coordinates.
"""
import importlib.metadata
from pathlib import Path
import torch
import threading
from .hf_backend import HuggingFaceBackend


class Qwen35Backend(HuggingFaceBackend):
    def __init__(self, model, tokenizer, store, verify_tensors=True):
        if importlib.metadata.version('transformers') != '5.3.0':
            raise ValueError('Qwen3.5 adapter requires its separate Transformers 5.3.0 runtime')
        if model.config.model_type != 'qwen3_5' or model.__class__.__name__ != 'Qwen3_5ForConditionalGeneration':
            raise ValueError('Full Qwen3_5ForConditionalGeneration checkpoint required')
        self.model=model.eval();self.model.requires_grad_(False)
        self._execution_lock=threading.RLock()
        self.tokenizer=tokenizer;self.store=store
        self.decoder=model.model.language_model;self.layers=self.decoder.layers
        config=model.config.text_config
        self.num_layers=len(self.layers);self.hidden_size=config.hidden_size
        self.mlp_width=config.intermediate_size;self.vocab_size=config.vocab_size
        self.num_heads=config.num_attention_heads;self.context_length=config.max_position_embeddings
        types=tuple(config.layer_types)
        if len(types)!=self.num_layers or any(t not in ['full_attention','linear_attention'] for t in types):
            raise ValueError('Unrecognized hybrid layer layout')
        self.full_attention_layers=tuple(i for i,t in enumerate(types) if t=='full_attention')
        self.device=next(model.parameters()).device
        if self.device.type not in ['cpu','cuda'] or any(p.device!=self.device for p in model.parameters()):
            raise ValueError('One CPU/CUDA device required')
        self.parameters=dict(model.named_parameters(remove_duplicate=False))
        if Path(model.config._name_or_path).resolve()!=store.model_dir:
            raise ValueError('Loaded model path differs from source')
        # Official HF 5.3 ignores the checkpoint's multi-token-prediction branch.
        # Its weights are not executable/editable through this backend. Vision
        # weights remain loaded and checked, though only text inputs are exposed.
        inactive=[];verified=0
        if verify_tensors:
            state=model.state_dict()
            for name in store.names:
                if name not in state:
                    if name.startswith('mtp.'):
                        inactive.append(name);continue
                    raise ValueError('Source tensor missing from model: '+name)
                original=store._native(name);actual=state[name].detach().cpu()
                if original.shape!=actual.shape or not torch.equal(original.to(actual.dtype),actual):
                    raise ValueError('Loaded tensor differs from base: '+name)
                verified+=1
        self.identity=dict(backend='hf-qwen35-text-v1',transformers=importlib.metadata.version('transformers'),
            torch=torch.__version__,device=str(self.device),dtype=str(next(model.parameters()).dtype),
            model_type=model.config.model_type,source_verified_tensorwise=verify_tensors,
            verified_source_tensors=verified,inactive_source_tensors=inactive,
            num_layers=self.num_layers,layer_types=list(types),full_attention_layers=list(self.full_attention_layers),
            attention_implementation=config._attn_implementation,
            note='Official HF hybrid decoder; text inputs only; MTP is not executed. No vLLM equivalence claim.')
