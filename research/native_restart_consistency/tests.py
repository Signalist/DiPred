"""Synthetic implementation checks; never evidence about DiPred effect sizes."""
import unittest,inspect
from core import *


def gaussian_paths(initial,steps,generator):
    a=torch.tensor([.8,.6],dtype=torch.double);sigma=torch.tensor([.2,.3],dtype=torch.double)
    state=initial.clone();values=[]
    for _ in range(steps):
        state=a*state+sigma*torch.randn(state.shape,generator=generator,dtype=torch.double)
        values.append(state)
    return torch.stack(values,2)[:,:,:,None,None]

class Tests(unittest.TestCase):
    def setUp(self):torch.manual_seed(73)

    def test_physical_indices_history_and_member_identity(self):
        history=torch.arange(400,420).reshape(1,20,1,1).expand(2,-1,-1,-1)
        parent=torch.arange(420,440).reshape(1,1,20,1,1).expand(32,2,-1,-1,-1).clone()
        parent[:,1]+=torch.arange(32)[:,None,None,None]*1000
        for k in [0,2,5,10]:
            hp=restart_history(history,parent,k);self.assertEqual(hp.shape,(32,2,20,1,1))
            self.assertEqual(hp[0,0,:,0,0].tolist(),list(range(400+k,420+k)))
            for i in range(32):torch.testing.assert_close(hp[i,1,20-k:],parent[i,1,:k])
            ix=physical_indices(400,k);self.assertEqual(ix['compared'],list(range(420+k,440)))
            b,s=tails(parent,parent,k);self.assertEqual(b.shape[2],20-k);self.assertEqual(s.shape[2],20-k)

    def test_future_truth_not_an_input_and_k0_same_condition(self):
        h=torch.randn(2,20,3,4);a=torch.randn(32,2,20,3,4);future=torch.randn_like(h)
        before=restart_history(h,a,5);future+=1000
        torch.testing.assert_close(before,restart_history(h,a,5),atol=0,rtol=0)
        torch.testing.assert_close(restart_history(h,a,0),h[None].expand_as(a),atol=0,rtol=0)
        self.assertEqual(list(inspect.signature(restart_history).parameters),['history','parent','k'])
        self.assertEqual(list(inspect.signature(generate).parameters),['model','norm','history','seeds'])

    def test_all_seed_families_unique_and_noise_not_reset(self):
        seeds=[]
        for split,records in RECORDS.items():
            for r in records:
                for start in CFG['starts']:
                    for repeat in range(2):
                        for family,k in [('A',0),('B',0),('C',0)]+[('S',k) for k in CFG['k']]:seeds+=seeds_for(split,r['file'],start,repeat,family,k)
        self.assertEqual(len(seeds),53760);self.assertEqual(len(set(seeds)),len(seeds))
        h=torch.zeros(32,2,20,2,3);x=seeded_noise(h,seeds[:32]);self.assertEqual(len({tensor_hash(v) for v in x}),32)
        torch.testing.assert_close(x,seeded_noise(h,seeds[:32]),rtol=0,atol=0)

    def test_explicit_energy_formula_and_identical_v_not_u(self):
        x=torch.randn(4,2,3,dtype=torch.double);y=torch.randn(5,2,3,dtype=torch.double)
        d=lambda a,b:float((a-b).square().mean().sqrt())
        expected=2*np.mean([d(a,b) for a in x for b in y])-np.mean([d(x[i],x[j]) for i in range(4) for j in range(4) if i!=j])-np.mean([d(y[i],y[j]) for i in range(5) for j in range(5) if i!=j])
        self.assertAlmostEqual(energy_distance(x,y)['u'],expected,places=12)
        same=energy_distance(x,x);self.assertAlmostEqual(same['v'],0,places=12);self.assertLess(same['u'],0)

    def test_quality_axes_and_time_mean_scored_after_averaging(self):
        p=torch.randn(32,2,10,3,4,dtype=torch.double);y=torch.randn(2,10,3,4,dtype=torch.double);std=torch.tensor([.7,1.3]).reshape(2,1,1,1)
        rows,cells=quality(p,y,std);r=rows[0]
        self.assertAlmostEqual(r['mse'],float((p.mean(0)-y).square().mean()),places=12)
        correct=float(empirical_crps((p/std).mean(2),(y/std).mean(1)).mean())
        self.assertAlmostEqual(r['time_mean_crps'],correct,places=12)
        self.assertGreater(abs(correct-r['crps']),.01)
        self.assertAlmostEqual(r['crps'],np.mean([r['crps'] for r in cells]),places=12)
        torch.testing.assert_close(projections(p).shape and projections(p),projections(p),rtol=0,atol=0)
        self.assertEqual(projections(p).shape,(32,16))

    def test_linear_gaussian_markov_composition_distribution(self):
        gen=torch.Generator().manual_seed(571)
        initial=torch.tensor([.5,-.3],dtype=torch.double).expand(2048,2)
        a=torch.tensor([.8,.6],dtype=torch.double);sigma=torch.tensor([.2,.3],dtype=torch.double)
        direct=gaussian_paths(initial,20,gen);parent=gaussian_paths(initial,20,gen)
        audits=[]
        for k in [0,2,5,10]:
            mid=initial if k==0 else parent[:,:,k-1,0,0]
            child=gaussian_paths(mid,20,gen)[:,:,:20-k]
            steps=torch.arange(k+1,21,dtype=torch.double)
            mu=initial[0,:,None]*a[:,None]**steps
            var=sigma[:,None]**2*(1-a[:,None]**(2*steps))/(1-a[:,None]**2)
            self.assertLess(float((child.mean(0)[:,:,0,0]-mu).abs().max()),.04)
            self.assertLess(float((child.var(0,unbiased=False)[:,:,0,0]-var).abs().max()),.025)
            values=[]
            for j in range(16):
                b=direct[j*128:(j+1)*128,:,k:];s=child[j*128:(j+1)*128];c=parent[j*128:(j+1)*128,:,k:]
                values.append(energy_distance(b,s)['u']-energy_distance(b,c)['u'])
            self.assertLess(abs(np.mean(values)),.012)
            audits.append(dict(k=k,mean_excess=float(np.mean(values)),distribution_mean_max_error=float((child.mean(0)[:,:,0,0]-mu).abs().max())))
        dump(OUT/'synthetic_markov.json',dict(passed=True,n=2048,dimension=2,system='x_next=diag(.8,.6)x+diag(.2,.3)normal',checks=audits))

if __name__=='__main__':
    configure();OUT.mkdir(parents=True,exist_ok=True)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if result.wasSuccessful():dump(OUT/'UNIT_TESTS.json',dict(passed=True,tests=result.testsRun,config_sha256=CONFIG_HASH))
    sys.exit(0 if result.wasSuccessful() else 1)
