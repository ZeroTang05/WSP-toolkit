"""Run only in the separate Transformers 5.3.0 runtime."""
from pathlib import Path
import tempfile
import unittest
import torch
from transformers import Qwen3_5Config,Qwen3_5TextConfig,Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
from weight_programming import WeightStore,ForwardPolicy,PracticeSet,ForwardService
from weight_programming.qwen35_backend import Qwen35Backend


class Tokenizer:
    pad_token_id=0;eos_token_id=2
    def encode(self,text,add_special_tokens=False):return [3,4,5,6][:len(text)]
    def apply_chat_template(self,messages,**kwargs):return 'abc'
    def decode(self,tokens,**kwargs):return ' '.join(str(x) for x in tokens)


class Qwen35Tests(unittest.TestCase):
    def setup_model(self,root):
        text=Qwen3_5TextConfig(vocab_size=64,hidden_size=32,intermediate_size=48,num_hidden_layers=4,
            num_attention_heads=2,num_key_value_heads=1,head_dim=16,linear_num_key_heads=2,
            linear_num_value_heads=2,linear_key_head_dim=16,linear_value_head_dim=16,
            layer_types=['linear_attention','linear_attention','linear_attention','full_attention'],
            max_position_embeddings=1024,rope_parameters={'rope_type':'default','rope_theta':10000.,'partial_rotary_factor':.5,'mrope_section':[2,1,1]},pad_token_id=0,eos_token_id=2)
        vision=Qwen3_5VisionConfig(depth=1,hidden_size=32,intermediate_size=48,num_heads=2,
            out_hidden_size=32,patch_size=2,spatial_merge_size=2,temporal_patch_size=2,num_position_embeddings=16,deepstack_visual_indexes=[])
        torch.manual_seed(13)
        model=Qwen3_5ForConditionalGeneration(Qwen3_5Config(text_config=text.to_dict(),vision_config=vision.to_dict())).eval()
        model.save_pretrained(root);model.config._name_or_path=str(root)
        store=WeightStore(root,root.parent/'edits')
        return model,store,Qwen35Backend(model,Tokenizer(),store)

    def test_hybrid_observation_probabilities_and_raw_residual(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'model';model,store,backend=self.setup_model(root)
            self.assertEqual(backend.full_attention_layers,(3,))
            service=ForwardService(backend,store,PracticeSet([],role='train',name='tiny'),Path(tmp)/'service')
            result=service.request({'type':'teacher_forced','model':'original','texts':['abcd','ab'],
                'return':{'token_logprobs':True,'topk':3,'hidden_states':[0,4],'mlp_act':[0,3],'attn_layers':[3]}})
            folder=Path(result['artifact_dir']);ids=torch.tensor([[3,4,5,6]])
            with torch.no_grad():reference=model(input_ids=ids,use_cache=False).logits[0].log_softmax(-1)
            actual=np.load(folder/'token_logprobs.npy')
            np.testing.assert_allclose(actual[0,1:4],reference[torch.arange(3),ids[0,1:]].numpy(),rtol=2e-5,atol=2e-5)
            attn=np.load(folder/'attention_3.npy');self.assertEqual(attn.shape,(2,2,4,4))
            np.testing.assert_allclose(attn[0].sum(-1),1,atol=1e-6);self.assertTrue((attn[0,:,:, :].trace(axis1=-2,axis2=-1)>0).all())
            self.assertTrue((np.triu(attn[0],1)==0).all());self.assertTrue((attn[1,:,2:,:]==0).all())
            _,captured=backend._read([3,4,5,6],[4],[0],[],False)
            np.testing.assert_allclose(np.load(folder/'hidden_4.npy')[0],captured['hidden_4'][0].numpy(),rtol=2e-5,atol=2e-5)
            with self.assertRaises(ValueError):
                service.request({'type':'teacher_forced','model':'original','texts':['ab'],'return':{'attn_layers':[0]}})

    def test_direct_edits_restore_and_paired_generation_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            model,store,backend=self.setup_model(Path(tmp)/'model')
            name='model.language_model.layers.0.mlp.down_proj.weight';original=store.load_original(name)
            proposal=original.clone();proposal[0,0]+=1
            store.save_edit(name,proposal)
            with backend.activate(store.snapshot()):
                torch.testing.assert_close(backend.parameters[name],proposal)
                backend._read([3,4],[],[],[],True)
            torch.testing.assert_close(backend.parameters[name],original)
            store.drop_edit(name)
            service=ForwardService(backend,store,PracticeSet([],role='train',name='tiny'),Path(tmp)/'service',ForwardPolicy(thinking=False))
            result=service.request({'type':'generate','model':'both','prompts':['abc'],'n':1,'max_tokens':4})
            import json
            folder=Path(result['artifact_dir'])
            original=json.loads((folder/'original.jsonl').read_text().splitlines()[0])
            edited=json.loads((folder/'edited.jsonl').read_text().splitlines()[0])
            self.assertEqual(original['generated_token_ids'],edited['generated_token_ids'])


if __name__=='__main__':unittest.main()
