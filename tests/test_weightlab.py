import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from weightlab import Lab, prepare


class ToolTests(unittest.TestCase):
    def test_dataset_reproducible_and_disjoint(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            prepare(a)
            prepare(b)
            self.assertEqual(Path(a, 'data.json').read_bytes(), Path(b, 'data.json').read_bytes())
            data = json.loads(Path(a, 'data.json').read_text())
            self.assertEqual([len(data[k]) for k in ('calibration', 'search', 'test')], [48, 64, 128])
            with self.assertRaises(FileExistsError):
                prepare(a)

    def setUp(self):
        torch.manual_seed(3)
        self.lab = Lab.__new__(Lab)
        config = Qwen3Config(vocab_size=256, hidden_size=64, intermediate_size=256,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            attn_implementation='sdpa')
        self.lab.model = Qwen3ForCausalLM(config).to(device='cuda', dtype=torch.bfloat16).eval()
        self.lab.layers = self.lab.model.model.layers

    def test_exact_rollback_even_on_exception(self):
        original = self.lab.layers[0].mlp.down_proj.weight.detach().clone()
        with self.assertRaises(RuntimeError):
            with self.lab.scale(0, 0.8):
                self.assertFalse(torch.equal(original, self.lab.layers[0].mlp.down_proj.weight))
                raise RuntimeError('simulate rejected experiment')
        self.assertTrue(torch.equal(original, self.lab.layers[0].mlp.down_proj.weight))

    @torch.no_grad()
    def test_structural_pruning_matches_mask_and_reloads(self):
        ids = torch.tensor([[1, 25, 8, 42]], device='cuda')
        keep = self.lab.masks([torch.rand(256), torch.rand(256)], 0.5)
        self.lab.model.float()
        with self.lab.masked(keep):
            expected = self.lab.model(ids, use_cache=False).logits.float()
        original_n = sum(p.numel() for p in self.lab.model.parameters())
        self.lab.structural_prune(keep)
        actual = self.lab.model(ids, use_cache=False).logits.float()
        self.assertLess((expected-actual).abs().max().item(), 0.0001)
        self.assertLess(sum(p.numel() for p in self.lab.model.parameters()), original_n)
        with tempfile.TemporaryDirectory() as path:
            self.lab.model.save_pretrained(path)
            loaded = Qwen3ForCausalLM.from_pretrained(path, torch_dtype=torch.float32,
                attn_implementation='sdpa').to('cuda').eval()
            for name, value in self.lab.model.state_dict().items():
                self.assertTrue(torch.equal(value, loaded.state_dict()[name]), name)
            reloaded = loaded(ids, use_cache=False).logits.float()
            self.assertLess((actual-reloaded).abs().max().item(), 0.0001,
                            f'Reload max error: {(actual-reloaded).abs().max().item()}')
        # Architecture edits must create normal parameters that survive dtype conversion.
        self.lab.model.bfloat16()
        with torch.inference_mode():
            self.assertTrue(torch.isfinite(self.lab.model(ids, use_cache=False).logits).all())

    def test_export_refuses_source(self):
        with tempfile.TemporaryDirectory() as path:
            self.lab.source = Path(path).resolve()
            with self.assertRaises(ValueError):
                self.lab.export(path, {})
            with self.assertRaises(ValueError):
                self.lab.export(Path(path)/'nested', {})


if __name__ == '__main__':
    unittest.main()
