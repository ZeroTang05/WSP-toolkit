import unittest
import numpy as np
from search_policies import propose,incumbent,FIXED,gp_predict


class SearchPolicyTests(unittest.TestCase):
    def test_fixed_design_covers_four_local_probes_without_duplicates(self):
        self.assertEqual(len(FIXED),16);self.assertEqual(len(set(FIXED)),16)
        for axis in range(4):
            for value in [.5,1.5]:
                self.assertIn(tuple(value if i==axis else 1. for i in range(4)),FIXED)

    def test_all_controls_stay_legal_and_incumbent_respects_constraint(self):
        for policy in ['fixed','random','greedy','bo']:
            rows=[];rng=np.random.default_rng(7)
            for i in range(16):
                point=propose(policy,rows,rng)
                self.assertNotIn(list(point),[row['coordinates'] for row in rows])
                risk=sum((x-.5)**2 for x in point)
                rows.append(dict(index=i,coordinates=list(point),domain_risk=risk,
                    feasible=point[0]<=.5,guard_upper95=point[0],guard_margin=.5))
            self.assertTrue(incumbent(rows)['feasible'])

    def test_gp_interpolates_observed_risk(self):
        x=[[0.]*4,[1.]*4];mean,std=gp_predict(x,[1.,0.],x)
        np.testing.assert_allclose(mean,[1.,0.],atol=1e-5)
        self.assertTrue(np.all(std>0));self.assertTrue(np.all(np.isfinite(std)))
