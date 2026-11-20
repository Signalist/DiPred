"""Seeded unit/information-boundary tests. Synthetic tensors test code, not claims."""
import unittest

import torch
import torch.nn.functional as F

from common import *
from mechanisms import SymmetricMixer,model_new,kernel_variant,simple_predict
from causal import BiasHead,BlockPredictor,inputs
from metrics import empirical_crps,energy_score,consecutive_event


class Boundaries(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.h=torch.randn(2,2,20,32,64)
        self.p=torch.randn(2,4,2,20,32,64)
        self.norm={'mean':torch.zeros(1,2,1,1,1),'std':torch.ones(1,2,1,1,1)}

    def test_decomposition_and_permutation(self):
        y=torch.randn(2,2,20,32,64)
        bias=self.p.mean(1)-y
        delta=self.p-self.p.mean(1,keepdim=True)
        self.assertTrue(torch.allclose(self.p-y[:,None],bias[:,None]+delta,atol=1e-6))
        perm=torch.randperm(4)
        self.assertTrue(torch.equal(self.p[:,perm].sort(1).values,self.p.sort(1).values))
        self.assertTrue(torch.allclose(self.p[:,perm].mean(1)-y,bias,atol=1e-6))
        self.assertTrue(torch.equal(delta[:,perm].sort(1).values,delta.sort(1).values))

    def test_frame_pair_sign_and_count(self):
        p=torch.arange(20.).reshape(1,20,1,1,1).expand(32,20,2,2,2)
        truth=torch.full_like(p,10.)
        e=p-truth
        self.assertEqual(e[:,:-1].shape[1],19)
        self.assertTrue(torch.equal(e[:,1:],e[:,:-1]+1))
        self.assertEqual(float(e[0,0,0,0,0]),-10.)

    def test_crps_explicit_pairs(self):
        p=torch.randn(7,2,3,4,5,dtype=torch.float64)
        y=torch.randn(2,3,4,5,dtype=torch.float64)
        expected=(p-y).abs().mean(0)-.5*(p[:,None]-p[None,:]).abs().mean((0,1))
        self.assertTrue(torch.allclose(empirical_crps(p,y),expected,atol=1e-12))

    def test_energy_explicit_pairs(self):
        p=torch.randn(5,2,3,2,2,dtype=torch.float64);y=torch.randn(2,3,2,2,dtype=torch.float64)
        expected=(p-y).flatten(1).square().mean(1).sqrt().mean()
        expected-=.5*(p[:,None]-p[None,:]).flatten(2).square().mean(2).sqrt().mean()
        self.assertAlmostEqual(float(energy_score(p,y)),float(expected),places=12)

    def test_symmetric_kernel_all_D4(self):
        mixer=SymmetricMixer()
        x=torch.randn(2,2,9,9)
        for k in range(4):
            self.assertTrue(torch.allclose(mixer(x).rot90(k,(-2,-1)),mixer(x.rot90(k,(-2,-1))),atol=1e-6))
        self.assertTrue(torch.allclose(mixer(x).flip(-1),mixer(x.flip(-1)),atol=1e-6))

    def test_state_machine_complete_block(self):
        m=BlockPredictor('native',self.norm)
        out=m.predict(self.h,self.p)
        self.assertTrue(torch.equal(out,self.p))
        with self.assertRaises(RuntimeError):m.predict(self.h,self.p)
        with self.assertRaises(ValueError):m.reveal(self.h[:,:,:19])
        y=torch.randn_like(self.h)
        m.reveal(y)
        self.assertTrue(torch.equal(m.previous,self.p.mean(1)-y))
        with self.assertRaises(RuntimeError):m.reveal(y)
        with self.assertRaises(ValueError):m.predict(self.h,self.p)
        m.predict(y,self.p)

    def test_current_truth_poison_cannot_change_forecast(self):
        # Truth is absent from the callable signature. Mutating an unrelated
        # current target affects only the later reveal, never issued outputs.
        head=BiasHead().eval()
        with torch.no_grad():head.net[-1].weight.normal_(std=.1)
        a=BlockPredictor('with_memory',self.norm,head)
        b=BlockPredictor('with_memory',self.norm,head)
        y=torch.randn_like(self.h)
        first=a.predict(self.h,self.p)
        y.fill_(float('nan'))
        second=b.predict(self.h,self.p)
        self.assertTrue(torch.equal(first,second))
        with self.assertRaises(ValueError):a.reveal(y)

    def test_input_ablation_boundaries(self):
        shifted=self.p+100
        a=inputs(self.h,self.p,None,'history_only',self.norm)
        b=inputs(self.h,shifted,None,'history_only',self.norm)
        self.assertTrue(torch.equal(a,b))
        prev=torch.randn_like(self.h)
        a=inputs(self.h,self.p,None,'current_fields',self.norm)
        b=inputs(self.h,self.p,prev,'current_fields',self.norm)
        self.assertTrue(torch.equal(a,b))
        c=inputs(self.h,self.p,prev,'with_memory',self.norm)
        self.assertFalse(torch.equal(b,c))

    def test_common_shift_keeps_delta(self):
        b=torch.randn_like(self.h)
        corrected=self.p-b[:,None]
        self.assertTrue(torch.allclose(self.p-self.p.mean(1,keepdim=True),
                       corrected-corrected.mean(1,keepdim=True),atol=1e-6))

    def test_consecutive_event(self):
        p=torch.zeros(2,2,20,2,2)
        p[0,1,2:5]=2;p[1,1,[2,4,6]]=2
        self.assertEqual(consecutive_event(p,1.).tolist(),[True,False])


if __name__=='__main__':
    torch.set_num_threads(4)
    unittest.main(verbosity=2)
