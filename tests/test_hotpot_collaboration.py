import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from hotpot_collaboration import run_case,parse,public_context,official_functions,score

SCORER=Path(__file__).resolve().parents[1]/'research/acquisition/hotpot_evaluate_v1.py'


class CollaborationTests(unittest.TestCase):
    def setUp(self):
        self.row=dict(id='question',question='Which city?',documents=[dict(title='Source',sentences=['A city.'])])
        self.official=official_functions(SCORER)

    def test_communication_is_the_changed_information_path(self):
        def generate(prompt,budget,seed):
            if 'You are the decomposition agent.' in prompt:text='PRIVATE_PLAN_71'
            elif 'You are the evidence agent.' in prompt:text='PRIVATE_EVIDENCE_83'
            else:text='{"answer":"city","supporting_facts":[["Source",0]]}'
            return dict(text=text,output_tokens=1,input_tokens=1)
        communicated=run_case(self.row,'collaborate',384,11,generate,str.lower,'TRAINING_SHARED_REFERENCE_92')
        isolated=run_case(self.row,'no_communication',384,11,generate,str.lower,'TRAINING_SHARED_REFERENCE_92')
        self.assertIn('PRIVATE_PLAN_71',communicated['calls'][1]['prompt'])
        self.assertIn('PRIVATE_EVIDENCE_83',communicated['calls'][2]['prompt'])
        self.assertNotIn('PRIVATE_PLAN_71',isolated['calls'][1]['prompt'])
        self.assertNotIn('PRIVATE_EVIDENCE_83',isolated['calls'][2]['prompt'])
        self.assertEqual([c['allocation'] for c in communicated['calls']],[c['allocation'] for c in isolated['calls']])
        for c in communicated['calls']+isolated['calls']:
            self.assertIn('A city.',c['prompt']);self.assertIn('TRAINING_SHARED_REFERENCE_92',c['prompt'])
        with self.assertRaises(ValueError):public_context(dict(self.row,answer='hidden'))

    def test_vote_uses_generated_answers_and_conserves_budget(self):
        answers=iter(['city','town','town'])
        def generate(prompt,budget,seed):return dict(text=json.dumps(dict(answer=next(answers),supporting_facts=[['Source',0]])))
        result=run_case(self.row,'independent_vote',385,11,generate,str.lower)
        self.assertEqual(result['prediction']['answer'],'town');self.assertEqual(result['planned_output_tokens'],385)
        self.assertIsNone(parse('{"answer":"town","supporting_facts":'))
        self.assertIsNone(parse('{"answer":"town","supporting_facts":[["Source",true]]}'))

    def test_per_item_and_joint_scores_match_pinned_official_evaluation(self):
        labels=[dict(_id='a',answer='The blue city',supporting_facts=[['Source',0]]),
                dict(_id='b',answer='yes',supporting_facts=[['Source',0],['Other',1]]),
                dict(_id='c',answer='no',supporting_facts=[['Source',0]])]
        predictions=[dict(id='a',prediction=dict(answer='blue city',supporting_facts=[['Source',0],['Wrong',9]])),
                     dict(id='b',prediction=dict(answer='yes',supporting_facts=[['Source',0]])),
                     dict(id='c',prediction=dict(answer='yes',supporting_facts=[]))]
        _,ours=score(predictions,labels,self.official)
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);(folder/'gold.json').write_text(json.dumps(labels))
            (folder/'pred.json').write_text(json.dumps(dict(answer={r['id']:r['prediction']['answer'] for r in predictions},
                sp={r['id']:r['prediction']['supporting_facts'] for r in predictions})))
            output=io.StringIO()
            with contextlib.redirect_stdout(output):self.official['eval'](folder/'pred.json',folder/'gold.json')
            import ast
            reference=ast.literal_eval(output.getvalue())
            for key,value in reference.items():self.assertAlmostEqual(ours[key],value)


if __name__=='__main__':unittest.main()
