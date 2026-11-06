import unittest
import torch
from weight_programming.merging import ties_merge


class MergeTests(unittest.TestCase):
    def test_mass_election_is_not_majority_of_models_or_factor_mean(self):
        # Two weak positive votes lose to one strong negative mass.
        values=[torch.tensor([1.,0.,2.,4.]),torch.tensor([1.,0.,2.,4.]),torch.tensor([-5.,0.,2.,4.])]
        merged,info=ties_merge([{'w':v} for v in values],['base']*3,density=1)
        torch.testing.assert_close(merged['w'],torch.tensor([-5.,0.,2.,4.]))
        self.assertEqual(info['conflicting_coordinates'],1)
        self.assertEqual(info['merged_nonzero'],3)
        for a,b in zip(values,[torch.tensor([1.,0.,2.,4.]),torch.tensor([1.,0.,2.,4.]),torch.tensor([-5.,0.,2.,4.])]):
            torch.testing.assert_close(a,b)

    def test_global_threshold_scope_and_explicit_ties(self):
        values=[{'first':torch.tensor([1.,2.]),'second':torch.tensor([100.,200.])}]
        merged,info=ties_merge(values,['base'],density=.25)
        # Upstream inclusive threshold keeps two of four, not one per module.
        torch.testing.assert_close(merged['first'],torch.zeros(2))
        torch.testing.assert_close(merged['second'],torch.tensor([100.,200.]))
        self.assertEqual(info['retained_mask_counts'],[2])
        sources=[{'w':torch.tensor([1.,1.])},{'w':torch.tensor([-1.,-1.])}]
        reference,_=ties_merge(sources,['base']*2,density=1)
        abstain,_=ties_merge(sources,['base']*2,density=1,zero_sign='abstain')
        torch.testing.assert_close(reference['w'],-torch.ones(2))
        torch.testing.assert_close(abstain['w'],torch.zeros(2))

    def test_incompatible_sources_rejected(self):
        with self.assertRaises(ValueError):ties_merge([{'w':torch.ones(3)}]*2,['a','b'])
        with self.assertRaises(ValueError):ties_merge([{'w':torch.ones(3)},{'w':torch.ones(4)}],['a','a'])
        with self.assertRaises(ValueError):ties_merge([{'w':torch.tensor([float('nan')])}],['a'])


if __name__=='__main__':unittest.main()
