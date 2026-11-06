import unittest
import torch
from analyze_delta import norm_squared


class DeltaAnalysisTests(unittest.TestCase):
    def test_small_gram_norm_and_cross_inner_product_match_dense_updates(self):
        torch.manual_seed(11)
        A1=torch.randn(3,7);B1=torch.randn(5,3);A2=torch.randn(2,7);B2=torch.randn(5,2)
        dense1=.5*(B1@A1);dense2=1.3*(B2@A2)
        self.assertAlmostEqual(norm_squared(A1,B1,.5),float(dense1.square().sum()),places=4)
        compact=float(((A1@A2.T)*(B1.T@B2)).sum())*.5*1.3
        self.assertAlmostEqual(compact,float((dense1*dense2).sum()),places=4)
