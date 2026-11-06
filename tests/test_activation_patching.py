from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from weight_programming import WeightStore
from weight_programming.hf_backend import HuggingFaceBackend
from weight_programming.patching import ActivationPatchingStudy


class Tokenizer:
    def encode(self,text,add_special_tokens=False):return [ord(x)%32 for x in text]


class PatchingTests(unittest.TestCase):
    def setup_study(self,folder):
        root=folder/'model';torch.manual_seed(22)
        model=Qwen3ForCausalLM(Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=24,num_hidden_layers=2,
            num_attention_heads=2,num_key_value_heads=1,head_dim=8)).eval()
        model.save_pretrained(root);model.config._name_or_path=str(root)
        store=WeightStore(root,folder/'edits');backend=HuggingFaceBackend(model,Tokenizer(),store)
        study=ActivationPatchingStudy(backend,store,folder/'study',data_role='calibration',corruption_description='Matched synthetic token replacement')
        return study,backend

    def test_analytic_full_embedding_and_last_residual_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            study,backend=self.setup_study(Path(tmp))
            kwargs=dict(clean_text='ab',corrupt_text='cb',target_token=1,alternative_token=2,kind='residual')
            for layer,positions in [(0,[0,1]),(2,[1])]:
                result=study.run(**kwargs,layer=layer,positions=positions)
                for key in result['clean']:self.assertAlmostEqual(result['clean'][key],result['patched'][key],places=6)
            # Last input token is identical: patching only its embedding is null.
            result=study.run(**kwargs,layer=0,positions=[1])
            self.assertEqual(result['corrupt'],result['patched'])
            self.assertEqual(result['executed_model_forwards'],3)
            self.assertEqual(study.ledger.summary()['successful'],3)

    def test_failed_patched_forward_removes_hook_and_keeps_cost(self):
        with tempfile.TemporaryDirectory() as tmp:
            study,backend=self.setup_study(Path(tmp));original=backend._read;count=[0]
            def fail(*args,**kwargs):
                count[0]+=1
                if count[0]==3:raise RuntimeError('injected forward failure')
                return original(*args,**kwargs)
            module=backend.layers[0].mlp.down_proj;hooks=len(module._forward_pre_hooks)
            with patch.object(backend,'_read',side_effect=fail):
                with self.assertRaises(RuntimeError):
                    study.run(clean_text='ab',corrupt_text='cb',kind='mlp_input',layer=0,positions=[0],channels=[1,3],target_token=1,alternative_token=2)
            self.assertEqual(len(module._forward_pre_hooks),hooks)
            self.assertEqual(study.ledger.summary()['failed'],1)
            self.assertFalse(list((Path(tmp)/'study').glob('request_*.json')))
            with self.assertRaises(ValueError):
                study.run(clean_text='a',corrupt_text='cb',kind='residual',layer=0,positions=[0],target_token=1,alternative_token=2)
            self.assertEqual(study.ledger.summary()['failed'],1)


if __name__=='__main__':unittest.main()
