"""Independent patch coupling and numerical score checks."""
from study import *

def numpy_necc(values,raw,k):
    n,c,t,h,w=values.shape;r=k//2;out=np.zeros_like(values,dtype='float64');count=np.zeros((h,w))
    for iy in range(-r,h+r):
        for ix in range(-r,w+r):
            ya,yb=max(0,iy-r),min(h,iy+r+1);xa,xb=max(0,ix-r),min(w,ix+r+1)
            if ya>=yb or xa>=xb:continue
            for cc in range(c):
                for tt in range(t):
                    a=raw[:,cc,tt,ya:yb,xa:xb].reshape(-1);v=values[:,cc,tt,ya:yb,xa:xb].reshape(-1)
                    rank=np.argsort(np.argsort(a,kind='stable'),kind='stable')
                    out[:,cc,tt,ya:yb,xa:xb]+=np.sort(v)[rank].reshape(n,yb-ya,xb-xa)
            count[ya:yb,xa:xb]+=1
    assert np.all(count==k*k)
    return out/count

def mechanics():
    setup();torch.manual_seed(42);v=torch.randn(4,2,2,5,7,device='cuda').sort(0).values;raw=torch.randn_like(v);errors=[]
    for k in (3,9):
        z=necc(v,raw,k);ref=numpy_necc(v.cpu().numpy(),raw.cpu().numpy(),k)
        err=float(np.max(np.abs(z.cpu().numpy()-ref)));assert err<2e-6
        meanerr=float((z.mean((-1,-2,0))-v.mean((-1,-2,0))).abs().max());assert meanerr<2e-6
        errors.append(dict(width=k,numpy_max_error=err,global_mean_max_error=meanerr))
    gen=Generator();d=load(SOURCE/'calibration.pt','cuda');desc={k:d['features'][k][:1] for k in KINDS};variance=[]
    for name in methods():
        if name.endswith('schaake'):continue
        z,var=gen.sample(name,desc,42,return_variance=True);error=float((var-1).abs().max());assert error<3e-5
        assert torch.isfinite(z).all();variance.append(dict(method=name,diagonal_variance_error=error))
    # Independent normal CDF marginal sanity check for the PCA Gaussian field.
    rng=np.random.default_rng(42);ix=np.linspace(0,2*20*32*64-1,16,dtype=int);b=gen.lib['pca'].flatten(1)[:,ix].cpu().numpy()
    from scipy.special import ndtr
    u=ndtr(rng.normal(size=(16384,32))@b)
    order=np.sort(u,axis=0);target=(np.arange(len(u))+.5)/len(u);ks=float(np.max(np.abs(order-target[:,None])));assert ks<.025
    # Nearest-neighbor templates never use held-out identities.
    train=set(m['file'] for m in gen.lib['meta']);cal=set(m['file'] for m in d['meta']);assert train.isdisjoint(cal)
    for kind in KINDS:
        ids=gen.ids(kind,desc[kind]).tolist();assert len({gen.lib['meta'][i]['file'] for i in ids})==32
    archeck=None
    if gen.ar is not None:
        rho=gen.ar['mean'].cpu().numpy().astype('float64');g=torch.Generator(device='cuda').manual_seed(42)
        ref=torch.randn(32,2,20,32,64,device='cuda',generator=g)[:2].cpu().numpy().astype('float64')
        for axis,r in zip((4,3,2),rho[:3]):
            ref=np.moveaxis(ref,axis,-1)
            for j in range(1,ref.shape[-1]):ref[...,j]=r*ref[...,j-1]+np.sqrt(1-r*r)*ref[...,j]
            ref=np.moveaxis(ref,-1,axis)
        ref[:,1]=rho[3]*ref[:,0]+np.sqrt(1-rho[3]**2)*ref[:,1]
        actual=gen.sample('ar__plain',desc,42)[:2].cpu().numpy();arerr=float(np.max(np.abs(actual-ref)));assert arerr<2e-4
        realized=[]
        for seed in range(64):realized.append(gen.sample('ar_F_last__mixed',desc,420000+seed).flatten(1)[:,ix].cpu().numpy())
        u=ndtr(np.concatenate(realized));ksar=float(np.max(np.abs(np.sort(u,axis=0)-(np.arange(len(u))+.5)[:,None]/len(u))));assert ksar<.07
        archeck=dict(numpy_recurrence_max_error=arerr,conditional_mixed_marginal_ks=ksar,realizations=len(u))
    dump(OUT/'mechanism_audit.json',dict(passed=True,necc=errors,gaussian_variance=variance,normal_cdf_ks_16_coordinates=ks,ar=archeck,conditional_templates_unique_train_trajectories=32,continuous_uniform_claim='Analytic: linear zero-mean Gaussian fields have normalized unit diagonal; normal CDF gives uniform marginals. Finite rank-coupled fields use discrete fixed nodes.',source_sha256=digest(Path(__file__).parent/'study.py')))
    print('MECHANICS',errors,'KS',ks,flush=True)

def crps(z,y):
    return np.abs(z-y).mean(0)-sum(np.abs(z[i]-z).sum(0) for i in range(len(z)))/(2*len(z)**2)

def funcs(z,p,n):
    a=z*n['std']+n['mean'];center=a.mean((-1,-2),keepdims=True)
    tke=(.5*((a-center)**2).sum(1).mean((-1,-2))).max(-1)
    energy=.5*(a*a).sum(1).mean((1,2,3))
    vort=(a[:,1,:,1:-1,2:]-a[:,1,:,1:-1,:-2])/(2*DX)+(a[:,0,:,2:,1:-1]-a[:,0,:,:-2,1:-1])/(2*DX)
    ens=(vort*vort).mean((-1,-2)).max(-1);speed=np.sqrt((a*a).sum(1)).reshape(len(a),-1).max(1)
    mvpe=np.sqrt((((z-p)*n['std'])**2).sum(1)).reshape(len(z),-1).max(1)
    return np.stack([tke,energy,ens,speed,mvpe],1)

def numerical(split):
    d=load(OUT/split/'audit_0000.pt');saved=load(OUT/split/'parts/0000.pt')['scores'];errors=[]
    n={k:v.numpy().astype('float64') for k,v in d['norm'].items()};y=d['y'].numpy().astype('float64');p=d['p'].numpy().astype('float64');stats={k:v.numpy().astype('float64') for k,v in d['stats'].items()}
    selected=json.loads((OUT/'selection.json').read_text())['selected'] if (OUT/'selection.json').exists() else 'F_last__local'
    for name in dict.fromkeys(['ecc32','smooth','necc9','necc_rank9','pca__global',selected]):
        if name not in d['samples']:continue
        z=d['samples'][name].numpy().astype('float64');flat=z.reshape(32,-1);target=y.reshape(-1)
        es=np.sqrt(((flat-target)**2).mean(1)).mean()-sum(np.sqrt(((flat[i]-flat)**2).mean(1)).sum() for i in range(32))/(2*32**2)
        vs=np.mean([((np.sqrt(np.abs(np.diff(z,axis=dim+1))).mean(0)-np.sqrt(np.abs(np.diff(y,axis=dim))))**2).mean() for dim in (3,2,1,0)])
        f=funcs(z,p,n);fy=funcs(y[None],p,n)[0];fc=(crps(f,fy)/stats['scale']).mean()
        for metric,v in [('es',es),('vs',vs),('functional_crps',fc)]:errors.append(dict(split=split,method=name,metric=metric,error=abs(v-saved[name][metric])))
        if not (name.startswith('necc') and not name.startswith('necc_rank')):assert np.array_equal(np.sort(z,axis=0),d['values'].numpy())
    df=pd.DataFrame(errors);df.to_csv(OUT/f'{split}_numerical.csv',index=False);assert df.error.max()<3e-5
    print('NUMERICAL',split,float(df.error.max()),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['mechanics','calibration','development','late','future']);a.add_argument('--conditional-scores',action='store_true');a.add_argument('--ar',action='store_true');a.add_argument('--scorefit',action='store_true');args=a.parse_args()
    if args.conditional_scores or args.ar or args.scorefit:
        import study
        OUT=OUT/'conditional_scores';study.OUT=OUT
        if args.ar:OUT=OUT/'ar';study.OUT=OUT;study.AR_STAGE=True
        if args.scorefit:OUT=OUT/'scorefit';study.OUT=OUT;study.SCOREFIT_STAGE=True
    mechanics() if args.action=='mechanics' else numerical(args.action)
