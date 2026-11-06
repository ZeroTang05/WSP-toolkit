import sys
import unittest
from dataclasses import replace
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from contract_registry import ContractRegistry,ToolRequest


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.registry=ContractRegistry(ROOT/'contracts/methods.json')
        spec=self.registry.methods['wanda.row_unstructured.v1']
        self.request=ToolRequest(spec['id'],spec['observation'],spec['selection'],spec['mutation'],spec['gradient_permission'],'calibration')

    def test_statistic_and_selection_substitutions_rejected(self):
        for field,value in [('observation','mean_absolute_input'),('selection','global_sort'),('gradient_permission','train')]:
            with self.assertRaises(ValueError):self.registry.validate(replace(self.request,**{field:value}))

    def test_test_labels_cannot_be_used_by_mutation_tool(self):
        with self.assertRaises(ValueError):self.registry.validate(replace(self.request,data_role='test'))

    def test_failed_contract_is_atomic(self):
        layer=torch.nn.Linear(5,3,bias=False)
        before=layer.weight.detach().clone()
        with self.assertRaises(ValueError):self.registry.apply_wanda_linear(self.request,layer,torch.ones(4),.4)
        self.assertTrue(torch.equal(before,layer.weight))

    def test_real_wanda_postconditions(self):
        layer=torch.nn.Linear(5,3,bias=False)
        before=layer.weight.detach().clone()
        result=self.registry.apply_wanda_linear(self.request,layer,torch.ones(5),.4)
        self.assertEqual(result['masked'],6)
        self.assertTrue(torch.all((layer.weight==0).sum(1)==2))
        self.assertTrue(torch.all((layer.weight==before)|(layer.weight==0)))

    def test_integrated_paper_wanda_uses_contract(self):
        from paper_methods import run_wanda
        class MiniLab:
            def __init__(self):self.layers=torch.nn.ModuleList([torch.nn.Linear(5,5,bias=False) for _ in range(2)])
            def score(self,rows):
                for row in rows:
                    x=row
                    for layer in self.layers:x=layer(x)
                return {}
        lab=MiniLab()
        result=run_wanda(lab,[torch.randn(1,3,5) for _ in range(3)],.4)
        self.assertEqual(result['contract_id'],'wanda.row_unstructured.v1')
        self.assertEqual(len(result['modules']),2)
        self.assertTrue(all(r['masked']==10 for r in result['modules']))


if __name__=='__main__':unittest.main()
