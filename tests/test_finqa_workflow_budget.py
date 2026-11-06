"""Control-flow tests only; real model quality and CUDA costs need GPU runs."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from torch import nn
import finqa_workflow_v2 as workflow
from finqa_views import public_input


class Inputs(dict):
    def __init__(self):super().__init__(input_ids=torch.tensor([[1,2]]));self.input_ids=self['input_ids']
    def to(self,device):return self


class Tokenizer:
    pad_token_id=0
    def __call__(self,*args,**kwargs):return Inputs()
    def decode(self,*args,**kwargs):return 'add(1, 1)'


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__();self.parameter=nn.Parameter(torch.zeros(1));self.config=SimpleNamespace(max_position_embeddings=10000)
        self.budgets=[]
    def generate(self,**kwargs):
        self.budgets.append(kwargs['max_new_tokens']);return torch.tensor([[1,2,3]])


class WorkflowBudgetTests(unittest.TestCase):
    def test_refinement_budget_and_labels_read_only_after_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);model=FakeModel()
            row=dict(id='sample',pre_text=['One plus one is two.'],post_text=[],table=[],
                qa=dict(question='What is one plus one?',program='add(1, 1)',exe_ans=2.,gold_inds={'text_0':'support'}))
            labels=root/'dev.json';labels.write_text(json.dumps([row]))
            inputs=root/'inputs.jsonl';inputs.write_text(json.dumps(dict(id=row['id'],**public_input(row),table=[]))+'\n')
            lab=SimpleNamespace(model=model,tokenizer=Tokenizer(),prefix=lambda r:r['prompt'],
                manifest=lambda:dict(source=str(root),files={'config.json':'fixture'}))
            original_read=Path.read_text
            def guarded_read(path,*args,**kwargs):
                if path==labels:self.assertEqual(len(model.budgets),7,'Gold answers accessed before all calls finished')
                return original_read(path,*args,**kwargs)
            args=['workflow','--model',str(root),'--inputs',str(inputs),'--labels',str(labels),
                '--output',str(root/'out'),'--partition','dev','--mode','base_refine','--smoke','1']
            with patch.object(sys,'argv',args),patch.object(workflow,'Lab',return_value=lab),\
                patch.object(torch.cuda,'synchronize'),patch.object(torch.cuda,'reset_peak_memory_stats'),\
                patch.object(torch.cuda,'max_memory_allocated',return_value=0),patch.object(Path,'read_text',guarded_read):
                workflow.main()
            self.assertEqual(model.budgets,[8,8,8,194,194,194,194])
            result=json.loads((root/'out/complete.json').read_text())
            self.assertEqual(result['role_calls'],4)
            self.assertEqual(result['execution_accuracy'],1.)
            self.assertTrue(result['base_parameters_immutable'])
