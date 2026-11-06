import unittest
from types import SimpleNamespace
import torch
from torch import nn
from weightlab import Lab


class StructurePreconditionTests(unittest.TestCase):
    def toy(self):
        layers=[]
        for _ in range(2):
            mlp=nn.Module()
            mlp.gate_proj=nn.Linear(4,8,bias=False);mlp.up_proj=nn.Linear(4,8,bias=False)
            mlp.down_proj=nn.Linear(8,4,bias=False)
            layers.append(SimpleNamespace(mlp=mlp))
        return SimpleNamespace(layers=layers,model=SimpleNamespace(config=SimpleNamespace(intermediate_size=8)))

    def test_late_invalid_group_cannot_partially_slice_earlier_layer(self):
        lab=self.toy();before=lab.layers[0].mlp.gate_proj.weight.detach().clone()
        with self.assertRaises(ValueError):Lab.structural_prune(lab,[torch.tensor([0,2,4,6]),torch.tensor([0,0,1,2])])
        self.assertTrue(torch.equal(before,lab.layers[0].mlp.gate_proj.weight))
        self.assertEqual(lab.layers[0].mlp.down_proj.in_features,8)

    def test_real_algebra_matches_masked_mlp(self):
        lab=self.toy();indices=torch.tensor([0,2,4,6]);x=torch.randn(3,4);expected=[]
        for layer in lab.layers:
            m=layer.mlp;h=torch.nn.functional.silu(m.gate_proj(x))*m.up_proj(x)
            mask=torch.zeros(8);mask[indices]=1
            expected.append(m.down_proj(h*mask))
        Lab.structural_prune(lab,[indices,indices])
        for layer,reference in zip(lab.layers,expected):
            m=layer.mlp;actual=m.down_proj(torch.nn.functional.silu(m.gate_proj(x))*m.up_proj(x))
            torch.testing.assert_close(actual,reference,rtol=1e-5,atol=1e-6)

    def test_bias_or_missing_selector_rejected(self):
        lab=self.toy();indices=torch.tensor([0,2,4,6])
        with self.assertRaises(ValueError):Lab.structural_prune(lab,[indices])
        lab.layers[1].mlp.up_proj=nn.Linear(4,8,bias=True)
        with self.assertRaises(ValueError):Lab.structural_prune(lab,[indices,indices])
