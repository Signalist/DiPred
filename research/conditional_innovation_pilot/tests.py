"""Numerical reparameterization, equal context, and release-input boundaries."""
import unittest
from core import *


class PilotTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.local,self.diagonal=operators("cpu")
        self.std=torch.tensor([.11,.017]).reshape(2,1,1)

    def test_source_roundtrip_and_unrolled_direct_equivalence(self):
        q=torch.randn(2,20,2,5,6,dtype=torch.double)*.03
        source=encode_source(q,self.local)
        torch.testing.assert_close(reconstruct(source,self.local),q,rtol=0,atol=1e-11)
        direct=[]
        for lead in range(20):
            value=torch.zeros_like(source[:,0])
            for initial in range(lead+1):
                contribution=source[:,initial]
                for _ in range(lead-initial):
                    contribution=simple_predict(contribution,None,self.local)
                value=value+contribution
            direct.append(value)
        torch.testing.assert_close(torch.stack(direct,1),q,rtol=0,atol=1e-11)

    def test_ridge_equivalence_when_spatial_design_is_constant(self):
        design=torch.randn(14,4,dtype=torch.double)
        target=torch.randn(14,20,2,dtype=torch.double)[:,:,:,None,None].expand(-1,-1,-1,3,4)
        source=encode_source(target,self.local)
        projection=design@torch.linalg.solve(design.T@design+.01*torch.eye(4),design.T)
        direct=(projection@target.reshape(14,-1)).reshape_as(target)
        fitted_source=(projection@source.reshape(14,-1)).reshape_as(source)
        torch.testing.assert_close(reconstruct(fitted_source,self.local),direct,rtol=0,atol=1e-11)

    def test_all_structures_zero_start_and_full_twenty_lead_context(self):
        x=torch.randn(2,88,5,6)
        changed=x.clone()
        changed[:,45]+=1  # lead20 of v prediction mean in full context
        for kind in ("D","T_local","T_diag"):
            model=SmallModel(kind,self.local,self.diagonal,self.std)
            self.assertEqual(sum(p.numel() for p in model.parameters()),4424)
            self.assertEqual(int(torch.count_nonzero(model(x))),0)
            with torch.no_grad():
                for p in model.parameters():p.fill_(.01)
            self.assertGreater(float((model(x)[:,0]-model(changed)[:,0]).abs().max()),0)

    def test_release_features_have_no_truth_argument(self):
        history=torch.randn(2,2,20,5,6)
        paths=torch.randn(2,32,2,20,5,6)
        norm=dict(mean=torch.zeros(1,2,1,1,1),std=self.std.reshape(1,2,1,1,1))
        future=torch.randn_like(history)
        before=conditions(history,paths,norm)
        future.add_(100)
        after=conditions(history,paths,norm)
        self.assertTrue(torch.equal(before,after))
        self.assertEqual(before.shape,(2,88,5,6))
        import inspect
        self.assertEqual(list(inspect.signature(conditions).parameters),["history","paths","norm"])

    def test_common_shift_crps_value_gradient_and_member_structure(self):
        p=torch.randn(32,2,5,6,dtype=torch.double)
        truth=torch.randn_like(p[0])
        q=torch.randn_like(truth,requires_grad=True)
        full=empirical_crps((p-q)/self.std,truth/self.std).mean()
        reduced=((p-truth-q)/self.std).abs().mean()
        pair=.5*((p[:,None]-p[None,:])/self.std).abs().mean()
        torch.testing.assert_close(full,reduced-pair,rtol=0,atol=1e-12)
        torch.testing.assert_close(torch.autograd.grad(full,q)[0],
                                   torch.autograd.grad(reduced,q)[0],rtol=0,atol=1e-12)
        shifted=p-q
        torch.testing.assert_close(shifted-shifted.mean(0),p-p.mean(0),rtol=0,atol=1e-12)

    def test_incomplete_release_block_is_rejected(self):
        norm=dict(mean=torch.zeros(1,2,1,1,1),std=self.std.reshape(1,2,1,1,1))
        with self.assertRaises(ValueError):
            conditions(torch.zeros(1,2,20,5,6),torch.zeros(1,32,2,19,5,6),norm)

    def test_no_substantive_mse_tradeoff_in_selection(self):
        native=dict(crps=.1,mse=.01,smse=.2)
        self.assertTrue(admissible(native,native))
        self.assertFalse(admissible(dict(crps=.09,mse=.01001,smse=.2),native))
        self.assertFalse(admissible(dict(crps=.09,mse=.01,smse=.201),native))


if __name__=="__main__":
    configure()
    unittest.main(verbosity=2)
