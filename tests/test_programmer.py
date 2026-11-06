"""Driver lifecycle through the actual isolated kernel and frozen HF bridge."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM,PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from weight_programming import PracticeSet,ForwardPolicy
from weight_programming.hf_backend import HuggingFaceBackend
from weight_programming.programming_session import ProgrammingSession
from weight_programming.programmer import run_programmer,parse_reply,notes_text


@unittest.skipUnless(Path('/proc').exists() and shutil.which('bwrap'),'Linux bubblewrap required')
class ProgrammerTests(unittest.TestCase):
    def session(self,folder):
        model_dir=folder/'model';practice=folder/'practice';practice.mkdir()
        tok=PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({'[UNK]':0,'a':1,'b':2},unk_token='[UNK]')))
        torch.manual_seed(3)
        model=Qwen3ForCausalLM(Qwen3Config(vocab_size=3,hidden_size=16,intermediate_size=24,
            num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=8)).eval()
        model.save_pretrained(model_dir);tok.save_pretrained(model_dir);model.config._name_or_path=str(model_dir)
        return ProgrammingSession(lambda store:HuggingFaceBackend(model,tok,store),model_dir,practice,
            PracticeSet([],role='train',name='tiny'),folder/'work',folder/'control',policy=ForwardPolicy(passes=1))

    def test_actual_edit_forward_rollover_and_final_choice(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            with self.session(folder) as session:
                replies=["```python\n# Record my own finding and edit one coordinate.\nfrom pathlib import Path\nPath('/work/notes.md').write_text('one coordinate intervention')\nw=load('model.layers.0.mlp.down_proj.weight'); w[0,0]+=1; save_edit('model.layers.0.mlp.down_proj.weight',w)\n```",
                    '```final\nToo early\n```',
                    '```forward\n'+json.dumps({'type':'teacher_forced','model':'edited','texts':['a b'],'return':{'mlp_act':[0]}})+'\n```',
                    "```python\nimport numpy as np\nprint(np.load('/work/observations/request_0001/mlp_0.npy').shape)\n```",
                    '```final\nSubmitted measured coordinate edit.\n```']
                seen=[]
                def generate(messages,tools,cap,seed):
                    seen.append(list(messages))
                    return dict(text=replies[len(seen)-1],input_tokens=100,output_tokens=40)
                def count(messages,tools):return 10000 if len(messages)>2 and not messages[-1]['content'].startswith('CONTEXT ROLLOVER') else 100
                result=run_programmer(session,generate,'Analyze this tiny model.',programmer_identity={'test':'scripted lifecycle, not model performance'},
                    count_tokens=count,max_context_tokens=1000,max_reply_tokens=100,require_forward_budget=True)
                self.assertEqual(result['status'],'finished');self.assertEqual(result['turns'],5)
                self.assertEqual(result['budget']['forward']['successful'],1)
                self.assertEqual(result['edited_names'],['model.layers.0.mlp.down_proj.weight'])
                self.assertGreater(result['rollovers'],0)
                records=[json.loads(x) for x in (folder/'control/programmer/turns.jsonl').read_text().splitlines()]
                self.assertTrue(any('tool_error' in r and 'forward-pass budget' in r['tool_error']['message'] for r in records))
                self.assertTrue(any('one coordinate intervention' in m[-1]['content'] for m in seen[1:]))

    def test_provider_failure_preserves_terminal_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            with self.session(folder) as session:
                def failing(*args):raise RuntimeError('provider unavailable')
                with self.assertRaisesRegex(RuntimeError,'provider unavailable'):
                    run_programmer(session,failing,'Analyze.',programmer_identity={'test':'provider failure'})
                result=json.loads((folder/'control/programmer/complete.json').read_text())
                self.assertEqual(result['status'],'programmer_error');self.assertEqual(result['turns'],0)
                self.assertEqual(result['budget']['python_executions'],0)

    def test_bad_accounting_does_not_execute_tool(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            with self.session(folder) as session:
                result=run_programmer(session,lambda *args:dict(text='```python\n1\n```',input_tokens=2,output_tokens=-1),
                    'Analyze.',programmer_identity={'test':'invalid accounting'})
                self.assertEqual(result['status'],'programmer_error');self.assertEqual(session.kernel.count,0)


class ReplyTests(unittest.TestCase):
    def test_no_prose_and_local_notes_only(self):
        with self.assertRaises(ValueError):parse_reply('I will act.\n```python\n1\n```')
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);outside=folder/'outside';outside.write_text('private')
            work=folder/'work';work.mkdir();(work/'notes.md').symlink_to(outside)
            self.assertIn('Rejected',notes_text(work))


if __name__=='__main__':unittest.main()
