import copy
import tempfile
from pathlib import Path
import unittest
import torch
from transformers import LlamaConfig,LlamaForCausalLM
from delta_bank import DeltaBank,write_delta,fingerprint
from peft_bank import PeftBank


class PeftReferenceTests(unittest.TestCase):
    def test_fp32_same_factors_and_exception_isolation(self):
        torch.manual_seed(17)
        config=LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=2,
            num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=64,attention_dropout=0.)
        model=LlamaForCausalLM(config).eval().requires_grad_(False)
        ids=torch.tensor([[1,4,7,12]])
        with torch.inference_mode():base_logits=model(ids,use_cache=False).logits
        targets={};factors={};base=dict(source='fixture',files={'config.json':'test-base'})
        for i in range(2):
            for name in ['q_proj','v_proj']:
                key=f'model.layers.{i}.self_attn.{name}';layer=model.get_submodule(key)
                targets[key]=dict(in_features=layer.in_features,out_features=layer.out_features,rank=2,alpha=4.)
                factors[key+'.A']=torch.randn(2,layer.in_features)*.1
                factors[key+'.B']=torch.randn(layer.out_features,2)*.1
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder)/'delta';write_delta(factors,targets,base,directory,1.,{})
            custom=DeltaBank(model,fingerprint(base['files']));custom.register('role',directory)
            peft=PeftBank(copy.deepcopy(model),fingerprint(base['files']),{'role':directory,'other':directory})
            with torch.inference_mode(),custom.activate('role'):expected=model(ids,use_cache=False).logits
            with torch.inference_mode(),peft.activate('role'):
                self.assertTrue(all(not parameter.requires_grad for parameter in peft.model.parameters()))
                actual=peft.model(ids,use_cache=False).logits
            torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
            coefficients={name:.5 for name in targets}
            with torch.inference_mode(),custom.activate('role',module_coefficients=coefficients):expected_half=model(ids,use_cache=False).logits
            with torch.inference_mode(),peft.activate('other',module_coefficients=coefficients):actual_half=peft.model(ids,use_cache=False).logits
            torch.testing.assert_close(actual_half,expected_half,rtol=1e-5,atol=1e-6)
            with self.assertRaises(RuntimeError):
                with peft.activate('role'):raise RuntimeError('aborted stage')
            with torch.inference_mode():restored=peft.model(ids,use_cache=False).logits
            torch.testing.assert_close(restored,base_logits,rtol=0,atol=0)
