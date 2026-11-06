import ast
import math
from pathlib import Path
import time
import unittest
from unittest.mock import patch
import torch
import transformers
from weight_programming import input_hessian, sparsegpt_reconstruct


class ReconstructionTests(unittest.TestCase):
    def test_matches_author_solver_when_mask_counts_match(self):
        # Execute only the archived reference class; omit unrelated quant imports.
        source = Path(__file__).resolve().parents[1] / 'research/references/sparsegpt/sparsegpt.py'
        parsed = ast.parse(source.read_text())
        cls = next(node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name == 'SparseGPT')
        namespace = dict(torch=torch, nn=torch.nn, transformers=transformers, math=math, time=time, DEBUG=False)
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), 'exec'), namespace)
        torch.manual_seed(12); w=torch.randn(4,16); x=torch.randn(90,16)
        h=input_hessian(x); reference=namespace['SparseGPT'](torch.nn.Linear(16,4,bias=False))
        reference.layer.weight.data.copy_(w); reference.H=h.float()
        # Each block has 32 entries. Author <= sorted[floor(32*s)] selects k+1.
        with patch('torch.cuda.synchronize', return_value=None):
            reference.fasterprune(7.5/32, blocksize=8)
        actual, mask, _=sparsegpt_reconstruct(w,h,sparsity=.25,blocksize=8)
        torch.testing.assert_close(actual,reference.layer.weight,atol=2e-6,rtol=2e-5)
        self.assertEqual(int(mask.sum()),16)

    def test_known_correlated_input_and_fixed_mask_block_invariance(self):
        x=torch.tensor([[1.,1.],[2.,2.],[-1.,-1.],[3.,3.]])
        w=torch.tensor([[2.,3.]])
        mask=torch.tensor([[True,False]])
        h=input_hessian(x)
        out,_,_=sparsegpt_reconstruct(w,h,prune_mask=mask,relative_damping=.01)
        # Closed form optimum after deleting first weight with relative ridge.
        expected=3 + 2*float(h[0,1])/(float(h[1,1])*1.01)
        self.assertAlmostEqual(float(out[0,1]),expected,places=5)
        self.assertLess(float(((out-w)@x.T).square().sum()),float(((w.masked_fill(mask,0)-w)@x.T).square().sum()))
        torch.manual_seed(8); w=torch.randn(3,9); h=input_hessian(torch.randn(70,9)); mask=torch.rand(3,9)<.4
        a,_,_=sparsegpt_reconstruct(w,h,prune_mask=mask,blocksize=2)
        b,_,_=sparsegpt_reconstruct(w,h,prune_mask=mask,blocksize=9)
        torch.testing.assert_close(a,b); self.assertTrue(a[mask].eq(0).all())
        torch.testing.assert_close(h,h.T)

    def test_boundaries_dead_inputs_and_bad_hessian(self):
        w=torch.ones(2,3); h=torch.diag(torch.tensor([2.,0.,1.]))
        a,_,_=sparsegpt_reconstruct(w,h,sparsity=0)
        self.assertTrue(a[:,1].eq(0).all()); torch.testing.assert_close(a[:,[0,2]],w[:,[0,2]])
        a,_,_=sparsegpt_reconstruct(w,h,sparsity=1)
        self.assertTrue(a.eq(0).all())
        with self.assertRaises(ValueError):sparsegpt_reconstruct(w,torch.ones(3,3),sparsity=.5,relative_damping=0)
        with self.assertRaises(ValueError):sparsegpt_reconstruct(w,h,sparsity=.5,prune_mask=w.bool())


if __name__=='__main__': unittest.main()
