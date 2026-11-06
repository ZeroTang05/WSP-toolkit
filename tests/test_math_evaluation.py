import json
from pathlib import Path
import tempfile
import unittest
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM,PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from weight_programming import WeightStore,ForwardPolicy
from weight_programming.hf_backend import HuggingFaceBackend
from weight_programming.evaluation import evaluate_math


class EvaluationTests(unittest.TestCase):
    def test_real_generation_streaming_and_no_gold_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);model_dir=folder/'model'
            tok=PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({'[UNK]':0,'a':1,'b':2},unk_token='[UNK]')))
            tok.chat_template="{% for message in messages %}{{ message['content'] }}{% endfor %}"
            torch.manual_seed(3)
            model=Qwen3ForCausalLM(Qwen3Config(vocab_size=3,hidden_size=16,intermediate_size=24,
                num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=8,eos_token_id=2)).eval()
            model.save_pretrained(model_dir);tok.save_pretrained(model_dir);model.config._name_or_path=str(model_dir)
            store=WeightStore(model_dir,folder/'edits');backend=HuggingFaceBackend(model,tok,store)
            original_prepare=backend.prepare_generation;seen=[]
            def prepare(items,request,policy):
                seen.extend(items);return original_prepare(items,request,policy)
            backend.prepare_generation=prepare
            class Grader:
                identity={'test':'nonsemantic fixture grader'}
                def __call__(self,item,text,finish):return dict(correct=True,prediction='fixture')
            rows=[dict(id='test/a',problem='a b',answer='PRIVATE_GOLD',data_role='evaluation',source_split='test'),
                dict(id='test/b',problem='b a',answer='ANOTHER_GOLD',data_role='evaluation',source_split='test')]
            result=evaluate_math(backend,store,rows,folder/'out',ForwardPolicy(max_new_tokens=4),Grader(),samples=2)
            self.assertEqual(result['problems'],2);self.assertEqual(result['samples'],4)
            self.assertEqual(result['accuracy'],1.);self.assertEqual(result['problem_cluster_standard_error'],0.)
            self.assertTrue(all('answer' not in item and 'GOLD' not in item['prompt'] for item in seen))
            records=[json.loads(x) for x in (folder/'out/samples.jsonl').read_text().splitlines()]
            self.assertEqual(len({(r['id'],r['sample']) for r in records}),4)
            self.assertTrue(all('generated_token_ids' in r for r in records))
            with self.assertRaises(FileExistsError):evaluate_math(backend,store,rows,folder/'out',ForwardPolicy(),Grader())
            with self.assertRaises(ValueError):evaluate_math(backend,store,[dict(rows[0],data_role='train')],folder/'bad',ForwardPolicy(),Grader())


if __name__=='__main__':unittest.main()
