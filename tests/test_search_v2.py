import unittest
import numpy as np
from search_v2 import GRID,FIXED,LEVELS,propose
from search_metrics import choice_metrics,encode_pair


class SearchV2Tests(unittest.TestCase):
    def test_original_global_grid_is_full_fixed_control(self):
        self.assertEqual(len(FIXED),21);self.assertEqual(len(set(FIXED)),21)
        self.assertEqual({p[0] for p in FIXED},set(LEVELS))
        self.assertTrue(all(len(set(point))==1 for point in FIXED))
        self.assertTrue(all(point in GRID for point in FIXED))

    def test_character_normalization_changes_decision_as_expected(self):
        result=choice_metrics([-2.,-3.],['a','longer'],1)
        self.assertEqual(result['acc_norm'],1)
        self.assertGreater(result['raw_ce'],result['norm_ce'])
        self.assertAlmostEqual(sum(np.exp(result['norm_logp'])),1.)

    def test_boundary_moves_whitespace_into_continuation(self):
        class Tokenizer:
            def encode(self,s,**kwargs):return list(map(ord,s))
        prefix,target=encode_pair(Tokenizer(),'Q\n','answer')
        self.assertEqual(''.join(map(chr,prefix)),'Q')
        self.assertEqual(''.join(map(chr,target)),'\nanswer')

    def test_fixed_queries_include_both_anchors_and_never_repeat(self):
        records=[];rng=np.random.default_rng(11)
        for i in range(21):
            point=propose('fixed',records,rng)
            self.assertNotIn(list(point),[r['coordinates'] for r in records])
            records.append(dict(coordinates=list(point)))
        self.assertEqual(records[0]['coordinates'],[0.]*4)
        self.assertEqual(records[1]['coordinates'],[1.]*4)
