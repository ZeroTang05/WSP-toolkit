"""Integration check on Linux with bubblewrap; no GPU used."""
from pathlib import Path
import shutil
import tempfile
import unittest
import torch
from safetensors.torch import save_file
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from weight_programming.cpu_kernel import CPUKernel


@unittest.skipUnless(Path('/proc').exists() and shutil.which('bwrap'),'Linux bubblewrap required')
class KernelTests(unittest.TestCase):
    def test_persistence_scope_edits_limits_and_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);model=folder/'model';practice=folder/'practice';model.mkdir();practice.mkdir()
            name='model.layers.0.mlp.down_proj.weight'
            save_file({name:torch.ones(2,3)},str(model/'model.safetensors'))
            tok=PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({'[UNK]':0,'x':1},unk_token='[UNK]')))
            tok.save_pretrained(model)
            with CPUKernel(model,practice,folder/'work',folder/'control',executions=7,seconds=10) as kernel:
                self.assertFalse(kernel.ready['cuda_available']);self.assertFalse(kernel.ready['grad_enabled'])
                self.assertEqual(kernel.execute('v=17; v')['output'].strip(),'17')
                self.assertEqual(kernel.execute('v+5')['output'].strip(),'22')
                self.assertEqual(kernel.execute("open('/model/no','w')")['status'],'error')
                self.assertEqual(kernel.execute("import socket; socket.socket().connect(('1.1.1.1',443))")['status'],'error')
                self.assertEqual(kernel.execute(f"w=load({name!r}); w[0,0]=2; save_edit({name!r},w); len(edited())")['output'].strip(),'1')
                self.assertTrue(kernel.execute("print('x'*11000)")['truncated'])
                self.assertEqual(kernel.execute("from pathlib import Path; print(Path('/home').exists()); print(list(Path('/dev').glob('nvidia*')))")['output'],'False\n[]\n')
                with self.assertRaises(RuntimeError):kernel.execute('1')
            import json
            self.assertEqual(json.loads((folder/'control/kernel.json').read_text())['status'],'closed')
            with CPUKernel(model,practice,folder/'work2',folder/'control2',seconds=.1) as kernel:
                with self.assertRaises(TimeoutError):kernel.execute('while True: pass')
                with self.assertRaises(RuntimeError):kernel.execute('1')


if __name__=='__main__':unittest.main()
