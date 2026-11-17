"""Scientific checks: known perturbations, split isolation, independent scores."""
from structure import *

def main():
    from local_oracle import local_transform
    setup();torch.manual_seed(42)
    p=F.interpolate(torch.randn(8,2,6,8,16,device='cuda'),size=(20,32,64),mode='trilinear',align_corners=True)
    c=p.new_tensor([[.3,-.4,.2,.07,-.09,.03,-.02]]).repeat(8,1)
    y=transform(p,c);recovered,err=optimize_oracle(p,y,steps=180)
    recovery=dict(max_coefficient_error=float((recovered-c).abs().max()),median_coefficient_error=float((recovered-c).abs().median()),remaining_mse_fraction=float((err/mse(p,y)).mean()))
    assert recovery['remaining_mse_fraction']<1e-4
    assert recovery['max_coefficient_error']<.01
    assert torch.equal(transform(p,torch.zeros_like(c)),p)
    local=local_transform(p,c[:,:,None,None].expand(-1,-1,2,4))
    local_error=float((local-y).abs().max());assert local_error<2e-6
    # Independent SciPy trilinear interpolation, including border handling.
    from scipy.ndimage import map_coordinates,zoom
    cc=torch.randn(1,7,2,4,device='cuda')*.04
    actual=local_transform(p[:1],cc).cpu().numpy()[0]
    # scipy zoom maps endpoint centers, matching align_corners=True.
    coeff=zoom(cc.cpu().numpy()[0],(1,16,16),order=1)
    tt,yy,xx=np.meshgrid(np.arange(20),np.arange(32),np.arange(64),indexing='ij')
    coords=np.stack([tt+coeff[2][None],yy+coeff[1][None],xx+coeff[0][None]])
    reference=np.stack([map_coordinates(p[0,k].cpu().numpy(),coords,order=1,mode='nearest')*(1+coeff[k+3][None])+coeff[k+5][None] for k in range(2)])
    scipy_error=float(np.max(np.abs(reference-actual)));assert scipy_error<2e-5
    # NumPy physical MSE and coefficient-independent zero correction.
    pn=p.cpu().numpy();yn=y.cpu().numpy();sd=norm('cpu')['std'].numpy()
    independent=np.mean(((pn-yn)*sd).astype('float64')**2,axis=(1,2,3,4))
    error=float(np.max(np.abs(independent-mse(p,y).cpu().numpy())))
    assert error<1e-9
    files={s:{m['file'] for m in load(OUT/f'{s}.pt')['meta']} for s in ('train','calibration')}
    assert not files['train']&files['calibration']
    manifest=load_manifest=json.loads(MANIFEST.read_text())
    all_splits={s:{r['file'] for r in v} for s,v in manifest['files'].items()}
    assert all(not all_splits[s]&all_splits[t] for s in all_splits for t in all_splits if s!=t)
    dump(OUT/'mechanism_audit.json',dict(passed=True,synthetic_recovery=recovery,independent_numpy_mse_max_error=error,local_vs_global_max_error=local_error,independent_scipy_warp_max_error=scipy_error,exact_zero_identity=True,manifest_splits_disjoint=True,source_sha256=digest(Path(__file__).parent/'structure.py'),interpretation='Known synthetic low-dimensional perturbation is identifiable here; this does not prove unique real-flow phase/amplitude decomposition.'))
    print(recovery,flush=True)

def segmented(split):
    setup();d=load(OUT/f'{split}.pt');rows=[]
    for i in range(0,len(d['p']),8):
        p=d['p'][i:i+8].cuda();y=d['y'][i:i+8].cuda();q=[]
        for t in range(0,20,5):
            c,_=optimize_oracle(p[:,:,t:t+5],y[:,:,t:t+5]);q.append(transform(p[:,:,t:t+5],c))
        err=mse(torch.cat(q,2),y)
        for j,m in enumerate(d['meta'][i:i+8]):rows.append(dict(**m,method='four_segment_oracle',mse=float(err[j])))
    df=pd.DataFrame(rows);df.to_csv(OUT/f'{split}_oracle4.csv',index=False)
    print('ORACLE4',split,float(df.groupby('file').mse.mean().mean()),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--segments',choices=['train','calibration']);args=a.parse_args()
    segmented(args.segments) if args.segments else main()
