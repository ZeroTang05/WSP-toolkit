"""Run with the declared math optional dependencies installed."""
import ast
import re
from pathlib import Path
import unittest
from weight_programming.grading import MathVerifyGrader,extract


class GradingTests(unittest.TestCase):
    def test_pinned_symbolic_equivalence_and_extraction_failures(self):
        grade=MathVerifyGrader()
        cases=[(r'\boxed{\frac{1}{2}}','0.5',True),(r'\boxed{\sqrt{4}}','2',True),
            (r'\boxed{3}','4',False),(r'\boxed{3','3',False),
            (r'\boxed{1}</think> final \boxed{2}','2',True)]
        for prediction,gold,expected in cases:
            self.assertEqual(grade(dict(answer=gold),prediction,'eos')['correct'],expected,(prediction,gold))

    def test_answer_extraction_matches_archived_author_functions(self):
        source=Path(__file__).resolve().parents[1]/'research/references/mit-wsp/mathqa.py'
        tree=ast.parse(source.read_text());names={'answer_text','_box_at','last_boxed','extract'}
        module=ast.Module(body=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names],type_ignores=[])
        reference=dict(_BOX_FORMS=re.compile(r'\\boxed|\\fbox'))
        exec(compile(module,str(source),'exec'),reference)
        examples=[r'\boxed{1}',r'\boxed {\frac{1}{2}}',r'\boxed 5$',r'\fbox{3}',r'\boxed{}',
            r'\boxed{1} then \boxed{2',r'\boxed{\{1,2\}}',r'\boxed{1}</think> answer \boxed{2}',
            r'\boxed{1} then \fbox{2}','no answer']
        for example in examples:self.assertEqual(extract(example),reference['extract'](example),example)


if __name__=='__main__':unittest.main()
