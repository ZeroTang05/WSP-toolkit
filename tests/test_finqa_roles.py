import unittest
from copy import deepcopy
from finqa_roles import negative_program,parse_retrieval,select_evidence
from finqa_executor import execute


class FinQARoleTests(unittest.TestCase):
    def test_negative_really_executes_to_different_answer(self):
        row=dict(table=[],qa=dict(program='subtract(5, 2)',exe_ans=3.))
        program,kind=negative_program(row)
        self.assertEqual(kind,'valid_execution_wrong_answer')
        result=execute(program,[])
        self.assertTrue(result['valid']);self.assertNotEqual(result['result'],3.)

    def test_selection_only_uses_supplied_public_evidence(self):
        view=dict(question='Which?',evidence=[dict(id='text_0',text='x'),dict(id='table_0',text='y')])
        before=deepcopy(view)
        self.assertEqual(parse_retrieval('["text_0", "text_0"]',view),(['text_0'],True))
        self.assertEqual(parse_retrieval('["gold_0"]',view),([],False))
        self.assertEqual(select_evidence(view,['text_0'])['evidence'],view['evidence'][:1])
        self.assertEqual(view,before)

    def test_table_selection_retains_public_column_labels(self):
        view=dict(question='2016 to 2017?',evidence=[dict(id='table_0',text='year | 2017 | 2016'),
            dict(id='table_1',text='income | 11503 | 10815'),dict(id='text_0',text='other')])
        before=deepcopy(view)
        selected=select_evidence(view,['table_1'])
        self.assertEqual(selected['evidence'],view['evidence'][:2])
        self.assertEqual(select_evidence(view,['missing','table_99'])['evidence'],[])
        self.assertEqual(select_evidence(view,['text_0'])['evidence'],view['evidence'][2:])
        self.assertEqual(view,before)
