"""Fixed-seed boundary and numerical checks, independent of model selection."""
import unittest
from run import torch,tests,propagate,inputs,records,out,read,digest,EXTRA,ref,protocol,dump,repo


class Boundaries(unittest.TestCase):
    def test_loss(self):
        tests()

    def test_oracle_and_free_inputs(self):
        torch.manual_seed(17)
        b=torch.randn(4,20,2,5,6)
        later=b.clone()
        later[:,1:]+=5
        copy={'kind':'A1_persistence'}
        self.assertTrue(torch.equal(propagate(b,copy,'first'),propagate(later,copy,'first')))
        self.assertTrue(torch.equal(propagate(b,copy,'teacher')[:,1:],b[:,:-1]))
        self.assertEqual(propagate(b,copy,'zero').count_nonzero(),0)
        self.assertTrue(torch.equal(propagate(b,copy,'first'),b[:,:1].expand_as(b)))

    def test_no_feedback(self):
        torch.manual_seed(23)
        history=torch.randn(1,2,20,32,64)
        p=torch.randn(1,4,2,20,32,64)
        norm=dict(mean=torch.zeros(1,2,1,1,1),std=torch.ones(1,2,1,1,1))
        self.assertTrue(torch.equal(inputs(history,p,None,'current_fields',norm),
                                    inputs(history,p,torch.randn_like(history),'current_fields',norm)))

    def test_splits(self):
        for a,b in [('train','development'),('train','test'),('test','development')]:
            for key in ('group','file'):
                self.assertTrue({r[key] for r in records[a]}.isdisjoint({r[key] for r in records[b]}))


if __name__=='__main__':
    unittest.main(verbosity=2)
