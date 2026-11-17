"""Reuse the existing full-field memory and frozen local warp on identical data."""
from state import *
sys.path.insert(0,str(W.parent/'Shared'))
from online_residual_subspace import ResidualSubspaceStream
import local_learn as local

@torch.no_grad()
def run(split):
    raw=load(SOURCE/f'{split}.pt');small=load(OUT/f'{split}.pt');basis=load(OUT/'basis.pt');n=norm();rows=[];cfg=load(SOURCE/'local/warp_F_last.pt','cuda');head=local.Head('warp').cuda().eval();head.load_state_dict(cfg['model']);feat=local.apply_projection(local.raw_features(raw,'F_last'),cfg['projection']);selection={}
    if split!='calibration':selection=json.loads((OUT/'control_selection.json').read_text())
    for scenario in ('natural','step','ramp'):
        d=shifted(small,scenario,basis);offset=torch.tensor(d['drift_coefficient'][:,:2]*np.sqrt(2),device='cuda',dtype=torch.float32)[:,:,None,None,None];point=raw['p'].cuda()+offset/n['std'];warp=head(feat,point,cfg['summary']['shrink']);truth=raw['y'].cuda()
        errors=st.mse(warp,truth).cpu().numpy()
        for i,m in enumerate(d['meta']):rows.append(dict(**m,scenario=scenario,method='local_warp',shrink=cfg['summary']['shrink'],age=int(d['age'][i]),mse=float(errors[i])))
        for name,gated in [('memory_unit',False),('memory_gated',True)]:
            shrinks=(0.,.25,.5,1.) if split=='calibration' else (selection[name]['shrink'],)
            for ix in group_ids(d['meta']):
                stream=ResidualSubspaceStream(gated=gated);previous_truth=None
                for count,i in enumerate(ix):
                    p=(point[i:i+1]*n['std']+n['mean']).cpu()
                    # Passing the last actually revealed target to this legacy
                    # API is deliberate. A gapped current history is NOT used.
                    history=(raw['x'][i:i+1]*n['std'].cpu()+n['mean'].cpu()) if previous_truth is None else previous_truth
                    corrected,_=stream.issue(history,p);delta=corrected-p
                    current_truth=(raw['y'][i:i+1]*n['std'].cpu()+n['mean'].cpu());previous_truth=current_truth
                    for shrink in shrinks:
                        mse=float((p+shrink*delta-current_truth).double().square().mean());rows.append(dict(**d['meta'][i],scenario=scenario,method=name,shrink=shrink,age=int(d['age'][i]),mse=mse))
        print('CONTROLS',split,scenario,flush=True)
    df=pd.DataFrame(rows);df.to_csv(OUT/f'{split}_controls_grid.csv',index=False)
    if split=='calibration':
        for method in ('memory_unit','memory_gated'):
            g=df[(df.scenario=='natural')&(df.method==method)].groupby(['shrink','file']).mse.mean().groupby('shrink').mean();selection[method]=dict(shrink=float(g.idxmin()),calibration_mse=float(g.min()))
        dump(OUT/'control_selection.json',selection)
    keep=(df.method=='local_warp')
    for name,cfg in selection.items():keep|=(df.method==name)&(df.shrink==cfg['shrink'])
    df=df[keep].copy();df.to_csv(OUT/f'{split}_controls.csv',index=False);print(selection,flush=True)

if __name__=='__main__':
    setup();p=argparse.ArgumentParser();p.add_argument('split',choices=['calibration','development','late','future']);a=p.parse_args();run(a.split)
