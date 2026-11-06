import unittest
import torch
from delta_programming import concatenate,truncate,weighted_truncate,residual


class DeltaProgrammingTests(unittest.TestCase):
    def test_composition_matches_updates_not_factor_averages(self):
        torch.manual_seed(3);items=[(torch.randn(3,7).double(),torch.randn(5,3).double()) for _ in range(2)]
        A,B=concatenate(items,[.3,.7]);target=sum(w*b@a for (a,b),w in zip(items,[.3,.7]))
        torch.testing.assert_close(B@A,target)
        self.assertGreater(float(torch.linalg.norm((.3*items[0][1]+.7*items[1][1])@(.3*items[0][0]+.7*items[1][0])-target)),.1)

    def test_truncation_attains_dense_svd_error(self):
        torch.manual_seed(4);A=torch.randn(5,11).double();B=torch.randn(9,5).double()
        a,b,record=truncate(A,B,2);dense=B@A;s=torch.linalg.svdvals(dense)
        self.assertAlmostEqual(float((dense-b@a).square().sum()),float(s[2:].square().sum()),places=8)
        self.assertAlmostEqual(residual(A,B,a,b)**2,record['discarded_frobenius_squared'],places=8)

    def test_activation_weighted_objective_and_singular_covariance(self):
        torch.manual_seed(5);A=torch.randn(4,8).double();B=torch.randn(7,4).double();X=torch.randn(8,30).double()
        X[0]*=100;Z=A@X;C=Z@Z.T/X.shape[1]
        a,b,record=weighted_truncate(A,B,C,2);u,v,_=truncate(A,B,2)
        error=float(((B@A-b@a)@X).square().sum()/X.shape[1])
        self.assertLessEqual(error,float(((B@A-v@u)@X).square().sum()/X.shape[1])+1e-8)
        self.assertAlmostEqual(error,record['calibration_discarded_output_mse'],places=6)
        weighted_truncate(A,B,torch.zeros(4,4),2)
