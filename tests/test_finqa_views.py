from copy import deepcopy
import sys
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from finqa_views import public_input,training_targets


class FinQAViewTests(unittest.TestCase):
    def setUp(self):
        self.row=dict(id='document-1',pre_text=['Revenue was 5.'],post_text=['Cost was 2.'],
            table=[['item','amount'],['revenue','5']],qa=dict(question='What was the profit?',
                answer='3',program='subtract(5, 2)',gold_inds={'text_0':'Revenue was 5.'},
                steps=[dict(res='3')],model_input='GOLD ANSWER: 3'),text_retrieved_all=['leaky field'])
    def test_inference_ignores_gold_and_retrieval_fields(self):
        other=deepcopy(self.row)
        for field in ['answer','program','gold_inds','steps','model_input']:
            other['qa'][field]='arbitrary injected gold'
        other['text_retrieved_all']=['different gold context']
        self.assertEqual(public_input(self.row),public_input(other))
    def test_canonical_evidence_ids_and_train_targets(self):
        view=public_input(self.row)
        self.assertEqual([r['id'] for r in view['evidence']],['text_0','text_1','table_0','table_1'])
        targets=training_targets(self.row,'train')
        self.assertEqual(targets['planner']['target'],'subtract(5, 2)')
        with self.assertRaises(ValueError):training_targets(self.row,'test')
        with self.assertRaises(ValueError):training_targets(self.row,'dev')
    def test_unresolved_evidence_not_silently_used(self):
        self.row['qa']['gold_inds']={'text_99':'not in public context'}
        with self.assertRaises(ValueError):training_targets(self.row,'train')


if __name__=='__main__':unittest.main()
