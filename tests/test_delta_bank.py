import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from delta_bank import DeltaBank,export_legacy_adapter,fingerprint


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Module()
        self.model.layers=nn.ModuleList([nn.Module()])
        self.model.layers[0].self_attn=nn.Module()
        self.model.layers[0].self_attn.q_proj=nn.Linear(5,3,bias=False)
    def forward(self,x):return self.model.layers[0].self_attn.q_proj(x)


class DeltaBankTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.tmp=tempfile.TemporaryDirectory()
        self.path=Path(self.tmp.name)
        self.model=Toy().eval()
        self.original=self.model.model.layers[0].self_attn.q_proj.weight.detach().clone()
        self.A=torch.randn(2,5)
        self.B=torch.randn(3,2)
        torch.save({'0.q_proj.A':self.A,'0.q_proj.B':self.B},self.path/'old.pt')
        self.files={'config.json':hashlib.sha256(b'dummy').hexdigest()}
        (self.path/'base.json').write_text(json.dumps(dict(source=str(self.path),files=self.files)))
        export_legacy_adapter(self.path/'old.pt',self.path/'base.json',self.path/'half',alpha=2,coefficient=.5)
        export_legacy_adapter(self.path/'old.pt',self.path/'base.json',self.path/'full',alpha=2,coefficient=1.)
        self.bank=DeltaBank(self.model,fingerprint(self.files))
        self.bank.register('half',self.path/'half')
        self.bank.register('full',self.path/'full')
    def tearDown(self):self.tmp.cleanup()

    def test_functional_output_matches_materialized_delta(self):
        x=torch.randn(4,5)
        for role,coefficient in [('half',.5),('full',1.)]:
            with self.bank.activate(role):
                output=self.model(x)
            expected=nn.functional.linear(x,self.original+coefficient*(self.B@self.A))
            torch.testing.assert_close(output,expected,rtol=1e-5,atol=1e-6)

    def test_switches_and_exceptions_do_not_mutate_base(self):
        x=torch.randn(4,5)
        before=self.model(x)
        for i in range(50):
            with self.bank.activate('half' if i%2 else 'full'):self.model(x)
        with self.assertRaises(RuntimeError):
            with self.bank.activate('half'):
                raise RuntimeError('failed workflow stage')
        self.assertTrue(torch.equal(before,self.model(x)))
        self.assertTrue(torch.equal(self.original,self.model.model.layers[0].self_attn.q_proj.weight))
        self.assertFalse(self.model.model.layers[0].self_attn.q_proj._forward_hooks)

    def test_nested_activation_and_external_kv_rejected(self):
        with self.bank.activate('half'):
            with self.assertRaises(RuntimeError):
                with self.bank.activate('full'):pass
        with self.assertRaises(ValueError):self.bank.generate('half',past_key_values=object())

    def test_wrong_base_and_shape_fail_before_activation(self):
        with self.assertRaises(ValueError):DeltaBank(self.model,'different-base').register('x',self.path/'half')
        meta=json.loads((self.path/'half/manifest.json').read_text())
        meta['targets']['model.layers.0.self_attn.q_proj']['in_features']=6
        (self.path/'half/manifest.json').write_text(json.dumps(meta))
        with self.assertRaises(ValueError):DeltaBank(self.model,fingerprint(self.files)).register('x',self.path/'half')
        self.assertTrue(torch.equal(self.original,self.model.model.layers[0].self_attn.q_proj.weight))

    def test_coefficient_override_preserves_manifest_and_base(self):
        before=(self.path/'half/manifest.json').read_bytes()
        self.bank.register('zero',self.path/'half',coefficient_override=0.)
        x=torch.randn(4,5)
        with self.bank.activate('zero'):
            self.assertFalse(self.model.model.layers[0].self_attn.q_proj._forward_hooks)
            output=self.model(x)
        self.assertTrue(torch.equal(output,nn.functional.linear(x,self.original)))
        self.assertEqual(before,(self.path/'half/manifest.json').read_bytes())

    def test_request_scoped_module_coefficient(self):
        x=torch.randn(4,5);name='model.layers.0.self_attn.q_proj'
        with self.bank.activate('full',module_coefficients={name:.25}):output=self.model(x)
        expected=nn.functional.linear(x,self.original+.25*(self.B@self.A))
        torch.testing.assert_close(output,expected,rtol=1e-5,atol=1e-6)
        with self.assertRaises(ValueError):
            with self.bank.activate('full',module_coefficients={'wrong':1.}):pass
        self.assertFalse(self.model.model.layers[0].self_attn.q_proj._forward_hooks)


if __name__=='__main__':unittest.main()
