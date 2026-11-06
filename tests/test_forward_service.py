import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from weight_programming import WeightStore,ForwardPolicy,ForwardService,PracticeSet,project_output_direction
from weight_programming.hf_backend import HuggingFaceBackend,PresencePenalty
from weight_programming.protocol import validate_request
from weight_programming.grading import extract,loop_diagnostics


class TinyTokenizer:
    pad_token_id=2;eos_token_id=0
    def encode(self,text,add_special_tokens=False):return [ord(c)%32 for c in text]
    def decode(self,ids,skip_special_tokens=False):return ''.join(chr(i+64) for i in ids if not skip_special_tokens or i not in [0,1,2])
    def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=False,enable_thinking=False):
        return ''.join(m['role']+':'+m['content']+'\n' for m in messages)+('assistant:' if add_generation_prompt else '')


class ForwardServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);base=self.root/'base'
        torch.manual_seed(51)
        config=Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=2,
            num_attention_heads=2,num_key_value_heads=1,head_dim=8,max_position_embeddings=2048,
            bos_token_id=1,eos_token_id=0,pad_token_id=2,tie_word_embeddings=True)
        Qwen3ForCausalLM(config).save_pretrained(base)
        self.model=Qwen3ForCausalLM.from_pretrained(base,torch_dtype=torch.float32,attn_implementation='sdpa').eval()
        self.store=WeightStore(base,self.root/'edits');self.backend=HuggingFaceBackend(self.model,TinyTokenizer(),self.store)
        self.practice=PracticeSet([dict(id='a',problem='abc',answer='1',source='toy')],role='development',name='semantic-test-only')

    def tearDown(self):self.temp.cleanup()

    def service(self,passes=8):return ForwardService(self.backend,self.store,self.practice,self.root/'work',ForwardPolicy(passes=passes,thinking=False))

    def test_teacher_probabilities_hidden_mlp_attention_and_padding(self):
        ids=self.backend.encode('abc');captured=[]
        handle=self.model.model.layers[0].mlp.register_forward_hook(lambda module,args,out:captured.append(out.detach().clone()))
        with torch.no_grad():reference=self.model(torch.tensor([ids]),use_cache=False,output_hidden_states=True)
        handle.remove();native_mlp_output=captured[0][0]
        result=self.service().request(dict(type='teacher_forced',model='original',texts=['abc','de'],
            **{'return':dict(token_logprobs=True,topk=4,hidden_states=[0,1,2],mlp_act=[0],attn_layers=[0])}))
        output=Path(result['artifact_dir']);lp=np.load(output/'token_logprobs.npy')
        expected=torch.log_softmax(reference.logits[0,:-1].float(),dim=-1).gather(1,torch.tensor(ids[1:])[:,None]).squeeze(1)
        np.testing.assert_allclose(lp[0,1:3],expected.numpy(),atol=1e-5);self.assertTrue(np.isnan(lp[0,0]));self.assertTrue(np.isnan(lp[1,2]))
        top_ids=np.load(output/'topk_ids.npy');top_lp=np.load(output/'topk_logprobs.npy')
        values,indices=torch.topk(torch.log_softmax(reference.logits[0].float(),dim=-1),4,dim=-1)
        np.testing.assert_array_equal(top_ids[0],indices.numpy());np.testing.assert_allclose(top_lp[0],values.numpy(),atol=1e-5)
        hidden1=np.load(output/'hidden_1.npy');np.testing.assert_allclose(hidden1[0],reference.hidden_states[1][0].numpy(),atol=1e-5)
        raw_last=torch.from_numpy(np.load(output/'hidden_2.npy')[0])
        with torch.no_grad():normalized=self.model.model.norm(raw_last)
        torch.testing.assert_close(normalized,reference.hidden_states[-1][0],atol=1e-5,rtol=1e-5)
        mlp=torch.from_numpy(np.load(output/'mlp_0.npy')[0])
        with torch.no_grad():reconstructed=self.model.model.layers[0].mlp.down_proj(mlp)
        torch.testing.assert_close(reconstructed,native_mlp_output,atol=1e-5,rtol=1e-5)
        attention=np.load(output/'attention_0.npy');np.testing.assert_allclose(attention[0].sum(-1),1.,atol=1e-5)
        self.assertTrue(np.all(np.triu(attention[0],1)==0));self.assertTrue(np.all(attention[1,:,2:,:]==0));self.assertTrue(np.all(attention[1,:,:,2:]==0))
        self.assertEqual(self.model.config._attn_implementation,'sdpa')

    def test_fresh_edit_snapshot_and_failed_request_restore_model_and_budget(self):
        service=self.service(passes=1);name='model.layers.0.mlp.down_proj.weight';original=self.store.load_original(name)
        writer=WeightStore(self.root/'base',self.root/'edits');edit=project_output_direction(original,torch.ones(16),[2,5],1.)
        writer.save_edit(name,edit);request=dict(type='teacher_forced',model='edited',texts=['abc'],**{'return':dict(hidden_states=[1])})
        with patch.object(self.backend,'teacher_forced',side_effect=RuntimeError('deliberate execution failure')):
            with self.assertRaises(RuntimeError):service.request(request)
        torch.testing.assert_close(self.model.get_parameter(name),original);self.assertEqual(service.ledger.summary()['failed'],1)
        self.assertFalse(list((self.root/'work').glob('*.partial')))
        result=service.request(request);self.assertNotEqual(result['variant_id'],result['original_variant_id']);self.assertEqual(result['budget']['successful'],1)
        torch.testing.assert_close(self.model.get_parameter(name),original)
        with self.assertRaises(RuntimeError):service.request(request)

    def test_paired_generation_same_stream_and_activation_replay_include_final_token(self):
        result=self.service().request(dict(type='generate',model='both',problems=['a'],n=2,max_tokens=4,activations=dict(layers=[0,1],mlp=True)))
        output=Path(result['artifact_dir']);a=[json.loads(v) for v in (output/'original.jsonl').read_text().splitlines()];b=[json.loads(v) for v in (output/'edited.jsonl').read_text().splitlines()]
        self.assertEqual([r['generated_token_ids'] for r in a],[r['generated_token_ids'] for r in b]);self.assertIsNone(result['paired']['gain'])
        item=self.backend.prepare_generation(self.practice.generation_items(result['request']),result['request'],self.service().policy)[0]
        generated=a[0]['generated_token_ids'];_,captured=self.backend._read(item['token_ids']+generated,[0,1],[0],[],False)
        feature=np.load(output/a[0]['activations']['hidden_1']['file'])
        expected=captured['hidden_1'][0,len(item['token_ids']):].mean(0).numpy()
        np.testing.assert_allclose(feature['answer_mean'],expected,atol=1e-6)
        self.assertEqual(a[0]['activations']['hidden_1']['answer_tokens'],len(generated))

    def test_request_limits_and_generated_only_presence_penalty(self):
        with self.assertRaises(ValueError):validate_request(dict(type='generate',model='both',prompts=['a']*257),ForwardPolicy(),2,[0,1])
        with self.assertRaises(ValueError):validate_request(dict(type='teacher_forced',model='original',texts=['a'],**{'return':dict(attn_layers=[0])}),ForwardPolicy(),2,[])
        with self.assertRaises(ValueError):validate_request(dict(type='teacher_forced',model='original',texts=['a'],**{'return':dict(mlp_act=[2])}),ForwardPolicy(),2,[0,1])
        processor=PresencePenalty(2,1.5);scores=torch.zeros(1,8)
        torch.testing.assert_close(processor(torch.tensor([[1,2]]),scores),scores)
        penalized=processor(torch.tensor([[1,2,3,3]]),scores);self.assertEqual(float(penalized[0,3]),-1.5);self.assertEqual(float(penalized[0,1]),0.)
        penalized=processor(torch.tensor([[1,2,3,3,4]]),scores);self.assertEqual(float(penalized[0,3]),-1.5);self.assertEqual(float(penalized[0,4]),-1.5)
        self.assertEqual(extract('thinking \\boxed{1}</think> final \\boxed{\\frac{1}{2}}')[0],r'\frac{1}{2}')
        self.assertIsNone(extract(r'first \boxed{1}, last \boxed{unfinished')[0])
        self.assertTrue(loop_diagnostics(('one two three four five '*20)+'partial')['verbatim_periodic_tail'])


if __name__=='__main__':unittest.main()
