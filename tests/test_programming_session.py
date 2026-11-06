from pathlib import Path
import shutil
import tempfile
import unittest
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM,PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from weight_programming import PracticeSet
from weight_programming.hf_backend import HuggingFaceBackend
from weight_programming.programming_session import ProgrammingSession


@unittest.skipUnless(Path('/proc').exists() and shutil.which('bwrap'),'Linux bubblewrap required')
class SessionTests(unittest.TestCase):
    def test_kernel_edits_reach_frozen_forward_and_observations_reach_kernel(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);model_dir=folder/'model';practice=folder/'practice';practice.mkdir()
            tok=PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({'[UNK]':0,'a':1,'b':2},unk_token='[UNK]')))
            torch.manual_seed(3)
            model=Qwen3ForCausalLM(Qwen3Config(vocab_size=3,hidden_size=16,intermediate_size=24,
                num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=8)).eval()
            model.save_pretrained(model_dir);tok.save_pretrained(model_dir);model.config._name_or_path=str(model_dir)
            factory=lambda store:HuggingFaceBackend(model,tok,store)
            with ProgrammingSession(factory,model_dir,practice,PracticeSet([],role='train',name='tiny'),
                    folder/'work',folder/'control') as session:
                name='model.layers.0.mlp.down_proj.weight';before=model.state_dict()[name].clone()
                request={'type':'teacher_forced','model':'original','texts':['a b'],
                    'return':{'token_logprobs':True,'mlp_act':[0]}}
                original=session.forward(request)
                result=session.execute(f"w=load({name!r}); w[0,0]+=1; save_edit({name!r},w)")
                self.assertEqual(result['status'],'ok')
                request['model']='edited';edited=session.forward(request)
                self.assertNotEqual(original['variant_id'],edited['variant_id'])
                torch.testing.assert_close(model.state_dict()[name],before)
                result=session.execute("import numpy as np; print(np.load('/work/observations/request_0002/mlp_0.npy').shape)")
                self.assertEqual(result['output'].strip(),'(1, 1, 24)')
                self.assertEqual(session.service.ledger.summary()['successful'],2)


if __name__=='__main__':unittest.main()
