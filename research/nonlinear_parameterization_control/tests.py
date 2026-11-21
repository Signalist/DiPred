"""Must pass before any formal training. Synthetic checks are not effect evidence."""
import inspect,unittest
from experiment import *

class Checks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.a,self.d=operators('cpu');self.std=torch.tensor([.11,.017]).reshape(2,1,1)

    def test_zero_identity_same_initialization_complete_context(self):
        initial=[]
        for kind in ARMS:
            torch.manual_seed(42);m=SmallModel(kind,self.a,self.d,self.std)
            initial.append(state_hash(m.state_dict()))
            x=torch.randn(2,88,5,6)
            self.assertEqual(sum(v.numel() for v in m.parameters()),4424)
            self.assertEqual(int(torch.count_nonzero(m(x))),0)
            with torch.no_grad():
                for p in m.parameters():p.fill_(.01)
            changed=x.clone();changed[:,45]+=1
            self.assertGreater(float((m(changed)[:,0]-m(x)[:,0]).abs().max()),0)
        self.assertEqual(len(set(initial)),1)

    def test_full_reduced_crps_value_and_gradient(self):
        p=torch.randn(2,32,20,2,3,4,dtype=torch.double)
        y=torch.randn(2,20,2,3,4,dtype=torch.double)
        q=torch.randn_like(y,requires_grad=True)
        full=torch.stack([empirical_crps((p[i]-q[i])/self.std,y[i]/self.std).mean() for i in range(2)]).mean()
        reduced=loss(p,y,q,self.std)
        pair=.5*((p[:,:,None]-p[:,None,:])/self.std).abs().mean()
        torch.testing.assert_close(full,reduced-pair,atol=1e-12,rtol=0)
        torch.testing.assert_close(torch.autograd.grad(full,q)[0],torch.autograd.grad(reduced,q)[0],atol=1e-12,rtol=0)

    def test_every_recursive_source_and_encoder_has_gradient(self):
        for op in (self.a,self.d):
            s=torch.randn(1,20,2,5,6,dtype=torch.double,requires_grad=True)
            q=reconstruct(s,op)
            g=torch.autograd.grad(q[:,-1].square().sum(),s)[0]
            self.assertTrue(bool((g.abs().sum((0,2,3,4))>0).all()))
            torch.testing.assert_close(expanded(s,op),q,atol=1e-11,rtol=0)
            torch.testing.assert_close(encode_source(q,op),s,atol=1e-11,rtol=0)
        for kind in ARMS:
            m=SmallModel(kind,self.a,self.d,self.std)
            with torch.no_grad():m.encoder[-1].weight.fill_(.01)
            m(torch.randn(2,88,5,6)).square().mean().backward()
            self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) and float(p.grad.abs().sum())>0 for p in m.parameters()))

    def test_operator_dtype_device_and_saved_coefficients(self):
        spec=json.loads((ROOT/'results/residual_dynamics_phase3/coefficients.json').read_text())
        for key,op in [('linear',self.a),('scale',self.d)]:
            torch.testing.assert_close(op['coef'],torch.tensor(spec[key]['coef'],dtype=torch.double),rtol=0,atol=0)
            x=torch.randn(1,2,5,6,dtype=torch.double)
            ref=simple_predict(x,None,op)
            for device in ('cpu','cuda'):
                y=simple_predict(x.to(device=device,dtype=torch.float32),None,op)
                self.assertEqual(y.dtype,torch.float32)
                self.assertEqual(y.device.type,device)
                torch.testing.assert_close(y.cpu().double(),ref,rtol=2e-5,atol=2e-6)

    def test_no_truth_input_or_partial_block(self):
        self.assertEqual(list(inspect.signature(conditions).parameters),['history','paths','norm'])
        self.assertEqual(list(inspect.signature(SmallModel.forward).parameters),['self','condition'])
        norm=dict(mean=torch.zeros(1,2,1,1,1),std=self.std.reshape(1,2,1,1,1))
        h=torch.randn(1,2,20,5,6);p=torch.randn(1,32,2,20,5,6)
        x=conditions(h,p,norm);future=torch.randn_like(h);future+=100
        torch.testing.assert_close(x,conditions(h,p,norm),atol=0,rtol=0)
        with self.assertRaises(ValueError):conditions(h,p[:,:,:,:19],norm)

    def test_scientific_and_protected_selection_are_distinct(self):
        rows=[dict(step=0,crps=.1,mse=.01,smse=.2),dict(step=250,crps=.09,mse=.011,smse=.21),dict(step=500,crps=.095,mse=.009,smse=.19)]
        self.assertEqual(choose(rows,False)['step'],250);self.assertEqual(choose(rows,True)['step'],500)
        rows[1]['crps']=.1;rows[2]['crps']=.1
        self.assertEqual(choose(rows,False)['step'],0)

    def test_real_train_cache_sign_normalization_and_crps(self):
        check_dependencies()
        d=load_data('train') # verifies identity, ordered32 members, history/truth hashes and normalization
        torch.testing.assert_close(d.b,d.p.mean(1)-d.y,atol=0,rtol=0)
        x=d.x[:1];p=d.p[:1,:,:,:,:2,:3].double();y=d.y[:1,:,:,:2,:3].double()
        q=torch.zeros_like(y,requires_grad=True)
        full=empirical_crps((p[0]-q[0])/d.std,y[0]/d.std).mean()
        reduced=loss(p,y,q,d.std);pair=.5*((p[:,:,None]-p[:,None,:])/d.std).abs().mean()
        torch.testing.assert_close(full,reduced-pair,atol=1e-12,rtol=0)
        torch.testing.assert_close(torch.autograd.grad(full,q)[0],torch.autograd.grad(reduced,q)[0],atol=1e-12,rtol=0)
        rows=[]
        for kind in ARMS:
            m=make_model(kind,42,d.std)
            self.assertEqual(int(torch.count_nonzero(m(x))),0)
            with torch.no_grad():
                for param in m.parameters():param.fill_(.01)
            rr,maps=sensitivity(m,x,'preflight_synthetic_nonzero_weights_real_train_input');rows+=rr
        csvwrite(OUT/'preflight_sensitivity.csv',rows)
        hashes={str(EXTRA/name):digest(EXTRA/name) for name in ['train_400_native.pt','train_420_native.pt','train_fields.pt','development_400_native.pt','development_420_native.pt','development_fields.pt']}
        dump(OUT/'preflight_data.json',dict(cache_hashes=hashes,train_blocks=len(d.x),member_seeds=list(range(500000,500032)),normalization={k:v.detach().cpu().tolist() for k,v in d.norm.items()},bias_sign='mean(P)-Y'))

if __name__=='__main__':
    configure();OUT.mkdir(parents=True,exist_ok=True)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    if result.wasSuccessful():dump(OUT/'PREFLIGHT.json',dict(passed=True,tests=result.testsRun,config_sha256=CONFIG_HASH,code={str(p.relative_to(ROOT)):digest(p) for p in Path(__file__).resolve().parent.glob('*.py')},utc=time.time()))
    sys.exit(0 if result.wasSuccessful() else 1)
