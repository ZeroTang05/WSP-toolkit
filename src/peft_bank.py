"""Standard PEFT multi-LoRA reference using exactly the same trained factors.

FP32 is compared numerically. PEFT's BF16 addition/cast order can differ from
our functional hook backend, so BF16 generation quality is measured separately.
"""
import contextlib
import threading
import torch
from delta_bank import DeltaBank


class PeftBank:
    def __init__(self,model,base_id,role_paths):
        from peft import LoraConfig,TaskType,get_peft_model
        verified=DeltaBank(model,base_id)
        for role,path in role_paths.items():verified.register(role,path)
        self.roles=verified.roles;self._lock=threading.RLock();self._active=None
        self.model=None
        for role,(meta,data) in self.roles.items():
            ranks={spec['rank'] for spec in meta['targets'].values()}
            alphas={spec['alpha'] for spec in meta['targets'].values()}
            if len(ranks)!=1 or len(alphas)!=1:raise ValueError('PEFT reference requires uniform rank and alpha')
            targets=set(meta['targets'])
            expected={name for name,module in model.named_modules() if isinstance(module,torch.nn.Linear)
                and name.endswith(('.self_attn.q_proj','.self_attn.v_proj'))}
            if self.model is None and targets!=expected:raise ValueError('Unexpected q/v target coverage')
            config=LoraConfig(task_type=TaskType.CAUSAL_LM,r=next(iter(ranks)),lora_alpha=next(iter(alphas)),
                target_modules=['q_proj','v_proj'],lora_dropout=0.,bias='none',inference_mode=True,init_lora_weights=False)
            if self.model is None:self.model=get_peft_model(model,config,adapter_name=role)
            else:self.model.add_adapter(role,config)
            with torch.no_grad():
                for name,spec in meta['targets'].items():
                    module=self.model.get_base_model().get_submodule(name)
                    module.lora_A[role].to(dtype=torch.float32);module.lora_B[role].to(dtype=torch.float32)
                    module.lora_A[role].weight.copy_(data[name+'.A'].to(module.lora_A[role].weight.device))
                    module.lora_B[role].weight.copy_(data[name+'.B'].to(module.lora_B[role].weight.device))
        self.model.eval().requires_grad_(False)
        self.model.base_model.disable_adapter_layers()

    @contextlib.contextmanager
    def activate(self,role,module_coefficients=None):
        with self._lock:
            if self._active is not None:raise RuntimeError('Nested PEFT activation forbidden')
            meta,_=self.roles[role]
            coefficients={} if module_coefficients is None else module_coefficients
            if set(coefficients)-set(meta['targets']):raise ValueError('Unknown target coefficient')
            values={float(coefficients.get(name,1.)) for name in meta['targets']}
            if len(values)!=1 or not all(torch.isfinite(torch.tensor(x)) for x in values):
                raise ValueError('PEFT workflow reference supports one finite scale per role')
            scale=next(iter(values))*meta['coefficient'];self._active=role
            try:
                self.model.set_adapter(role);self.model.requires_grad_(False)
                self.model.base_model.enable_adapter_layers()
                self.model.requires_grad_(False)
                for name,spec in meta['targets'].items():
                    self.model.get_base_model().get_submodule(name).scaling[role]=scale*spec['alpha']/spec['rank']
                yield self.model
            finally:
                self.model.base_model.disable_adapter_layers();self.model.requires_grad_(False);self._active=None
