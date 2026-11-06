import unittest
from officebench_workflow import run_case,MODES,parse_action,public_context


class Environment:
    public=dict(username='Alice',task='Create the requested document')
    task_id='3-1/0'
    description=dict(apps={'shell':{'actions':{'command':'command string'}},'llm':{'hidden':'excluded'}})
    state=dict(workdir='/',current_app=None)
    def __init__(self,finish=True):self.finish=finish
    def step(self,action):return dict(success=True,done=self.finish,observation='Action completed')


class WorkflowTests(unittest.TestCase):
    def generator(self,records):
        def generate(role,system,prompt,allocation,seed):
            records.append(dict(role=role,system=system,prompt=prompt,allocation=allocation))
            text='PLAN_MARKER' if role=='planner' else '{"choice":0}' if role=='selector' else '{"analysis":"check","action":{"app":"system","action":"finish_task","answer":"done"}}'
            return dict(text=text,output_tokens=min(3,allocation),input_tokens=1,seconds=0)
        return generate

    def test_communication_intervention_and_role_formats(self):
        for mode in ['collaborate','no_communication']:
            calls=[];result=run_case(Environment(),mode,self.generator(calls),token_budget=40,decision_tokens=30)
            actor=next(c for c in calls if c['role']=='actor')
            self.assertEqual('PLAN_MARKER' in actor['prompt'],mode=='collaborate')
            self.assertEqual([c['allocation'] for c in calls],[10,20])
            self.assertEqual(result['termination'],'model_finished')
        calls=[];result=run_case(Environment(),'independent_select',self.generator(calls),decision_tokens=30)
        self.assertIn('JSON choice',calls[-1]['system']);self.assertNotIn('must contain',calls[-1]['system'])
        self.assertIsNotNone(result['decisions'][0]['action'])

    def test_all_modes_obey_actual_token_budget_and_reject_private_fields(self):
        for mode in MODES:
            calls=[];result=run_case(Environment(finish=False),mode,self.generator(calls),token_budget=7,decision_tokens=24,max_steps=10)
            self.assertLessEqual(result['output_tokens'],7)
            self.assertLessEqual(len(result['decisions']),10)
            self.assertTrue(all('"hidden"' not in c['prompt'] for c in calls))
        with self.assertRaises(ValueError):public_context({'task':'x','evaluation':[1]}, {},{},[])
        self.assertEqual(parse_action('{"analysis":"x","action":{"app":"shell","action":"command","command":"ls"}}')['command'],'ls')
        self.assertIsNone(parse_action('not an action'))


if __name__=='__main__':unittest.main()
