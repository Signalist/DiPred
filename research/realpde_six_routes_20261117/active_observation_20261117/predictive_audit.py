"""Independent sensor-space conditioning of the predictive-mode variant."""
from predictive import *
import importlib.util
_spec=importlib.util.spec_from_file_location('active_mask_audit',Path(__file__).with_name('audit.py'));common_audit=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(common_audit)

def numerical():
    model=load(ROOT/'model.pt');n={k:v.numpy().astype(float) for k,v in norm('cpu').items()};rows=[]
    for split in ('calibration','development','late','future'):
        for path in sorted((ROOT/split).glob('audit_*.pt')):
            s=load(path);cfg=s['config'];family=model['families'][str(cfg['shrink'])];rank=cfg['rank'];scale=s['scale'].numpy().astype(float);at=(family['basis'][:rank].numpy()*scale[:,5:]).reshape(rank,-1);sd=scale[:,4,SITES[:,0],SITES[:,1]].T.reshape(-1);h=family['cross'][:rank].numpy()*sd;cov=family['obs_cov'].numpy()*sd[:,None]*sd[None,:];idx=s['indices'];rowsix=(idx[:,None]*2+np.arange(2)[None]).reshape(-1);p=s['p'].numpy().astype(float);physical=p*n['std'][0]+n['mean'][0];error=(s['observed_physical'].numpy().astype(float)-physical[:,4,SITES[idx,0],SITES[idx,1]].T).reshape(-1);z=cfg['gain']*h[:,rowsix]@np.linalg.solve(cov[np.ix_(rowsix,rowsix)],error);q=physical[:,5:]+(z@at).reshape(2,15,32,64);y=s['truth'].numpy().astype(float)*n['std'][0]+n['mean'][0];mse=((q-y[:,5:])**2).mean();ze=float(np.max(abs(z-s['z'].numpy())));me=abs(mse-s['analytic_mse']);assert ze<1e-5 and me<1e-9,(path,ze,me)
            # Recompute every greedy step using a fresh full-subset Schur solve.
            if path.stem.endswith('voi'):
                g=at@at.T/at.shape[1];selected=[]
                for k in range(8):
                    gains=[];selected_scalar=(np.asarray(selected,dtype=int)[:,None]*2+np.arange(2)[None]).flatten()
                    for j in range(128):
                        if j in selected:gains.append(-np.inf);continue
                        ii=np.arange(2*j,2*j+2);c=h[:,ii];v=cov[np.ix_(ii,ii)]
                        if selected:
                            old=cov[np.ix_(selected_scalar,selected_scalar)];connection=cov[np.ix_(selected_scalar,ii)];sol=np.linalg.solve(old,connection);c=c-h[:,selected_scalar]@sol;v=v-connection.T@sol
                        gains.append(float(np.trace(g@c@np.linalg.solve(v,c.T))))
                    selected.append(int(np.argmax(gains)))
                assert np.array_equal(selected,idx)
            rows.append(dict(split=split,snapshot=path.name,latent_error=ze,mse_error=me))
    df=pd.DataFrame(rows);df.to_csv(ROOT/'numerical_audit.csv',index=False);assert len(df)==32;return dict(snapshots=len(df),latent_max_error=float(df.latent_error.max()),mse_max_error=float(df.mse_error.max()),greedy_full_subset_recompute=True)

if __name__=='__main__':
    setup();n=numerical();print('NUMERICAL',n,flush=True);common_audit.OUT=ROOT;common_audit.SparseUpdatePredictor=make_predictor;r=common_audit.request_checks();d=common_audit.direct();dump(ROOT/'audit.json',dict(passed=True,numerical=n,requests=r,direct=d,extension_protocol_matches=json.loads((ROOT/'selection.json').read_text())['extension_protocol_sha256']==digest(Path(__file__).with_name('PREDICTIVE_PROTOCOL.md')),source_sha256=digest(Path(__file__))));print('PREDICTIVE_AUDIT_PASSED',flush=True)
