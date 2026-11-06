import importlib.util
import ast
from types import SimpleNamespace
import sys
import unittest
from pathlib import Path
import torch
from torch import nn
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from paper_methods import InputEnergy, wanda_mask, LoRALinear, training_loss


class PaperMethodTests(unittest.TestCase):
    def test_sft_supervises_answer_and_eos_but_not_prompt(self):
        class Tokenizer:
            eos_token_id = 4
            def encode(self, text, **kwargs):
                return [1,2] if text=='prefix' else [3]
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = nn.Parameter(torch.zeros(1,4,5,device='cuda'))
            def forward(self, ids, **kwargs):
                self.seen = ids
                return SimpleNamespace(logits=self.logits[:,:ids.shape[1]])
        model = Model()
        lab = SimpleNamespace(tokenizer=Tokenizer(), prefix=lambda row:'prefix', model=model)
        training_loss(lab, {'answer':'3'}, include_eos=True).backward()
        self.assertEqual(model.seen.tolist(), [[1,2,3,4]])
        self.assertTrue(torch.equal(model.logits.grad[0,0], torch.zeros(5,device='cuda')))
        self.assertLess(model.logits.grad[0,1,3].item(), 0)
        self.assertLess(model.logits.grad[0,2,4].item(), 0)

    def test_mask_against_executed_author_ast(self):
        # Execute the actual author score/mask branch, without importing its
        # dataset and old model dependencies. No rewritten reference formula.
        tree = ast.parse((ROOT/'vendor/wanda/prune_reference.py').read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='prune_wanda')
        nodes = []
        for n in ast.walk(function):
            if isinstance(n, ast.For) and any(isinstance(s, ast.Assign) and
                any(isinstance(t, ast.Name) and t.id=='W_metric' for t in s.targets) for s in n.body):
                nodes = [s for s in n.body if (isinstance(s, ast.Assign) and
                    any(isinstance(t, ast.Name) and t.id in ('W_metric','W_mask') for t in s.targets))
                    or (isinstance(s, ast.If) and 'prune_n' in ast.unparse(s.test))]
                break
        self.assertEqual(len(nodes), 3)
        torch.manual_seed(9)
        layer = nn.Linear(13, 7, bias=False)
        energy = torch.rand(13)
        scope = dict(torch=torch, subset={'x':layer}, name='x', prune_n=0, prune_m=0,
            wrapped_layers={'x':SimpleNamespace(scaler_row=energy)},
            args=SimpleNamespace(use_variant=False, sparsity_ratio=.35))
        exec(compile(ast.Module(body=nodes, type_ignores=[]), '<pinned-wanda>', 'exec'), scope)
        self.assertTrue(torch.equal(wanda_mask(layer.weight, energy, .35), scope['W_mask']))

    def test_energy_against_pinned_author_implementation(self):
        path = ROOT/'vendor/wanda/layerwrapper.py'
        spec = importlib.util.spec_from_file_location('author_wanda', path)
        author = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(author)
        ref = author.WrappedGPT(nn.Linear(5, 3, bias=False))
        impl = InputEnergy(5, 'cpu')
        torch.manual_seed(1)
        for x in [torch.randn(2, 7, 5), torch.randn(1, 3, 5), torch.randn(2, 8, 5)]:
            ref.add_batch(x, None)
            impl.add(x)
        torch.testing.assert_close(impl.energy(), ref.scaler_row, rtol=1e-6, atol=1e-6)

    def test_wanda_known_mask_and_stable_ties(self):
        weight = torch.tensor([[1., 2., 3., 4.], [4., 3., 2., 1.]])
        energy = torch.tensor([100., 1., 1., 1.])
        expected = torch.tensor([[False, True, True, False], [False, False, True, True]])
        self.assertTrue(torch.equal(wanda_mask(weight, energy, .5), expected))
        ties = wanda_mask(torch.ones(2,4), torch.ones(4), .5)
        self.assertTrue(torch.equal(ties, torch.tensor([[True,True,False,False]]*2)))
        with self.assertRaises(ValueError):
            wanda_mask(weight, energy, 1.)

    def test_actual_lora_training_freezes_base_and_merges(self):
        torch.manual_seed(4)
        base = nn.Linear(5, 3, bias=False)
        original = base.weight.detach().clone()
        adapted = LoRALinear(base, rank=2, alpha=2)
        x = torch.randn(20,5)
        target = x @ (original + torch.randn(3,2) @ torch.randn(2,5)*.1).T
        before = ((adapted(x)-target)**2).mean().item()
        optimizer = torch.optim.AdamW([adapted.A, adapted.B], lr=.03)
        for _ in range(80):
            optimizer.zero_grad()
            loss = ((adapted(x)-target)**2).mean()
            loss.backward()
            optimizer.step()
        after = ((adapted(x)-target)**2).mean().item()
        self.assertLess(after, before*.1)
        self.assertTrue(torch.equal(original, base.weight))
        merged = nn.functional.linear(x, original+adapted.delta())
        torch.testing.assert_close(adapted(x), merged, rtol=1e-5, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
