from controls import *

@torch.no_grad()
def run():
    setup();n=st.norm();recs=records('development');sel=json.loads((OUT/'selection.json').read_text());names=REFERENCES+[k for k in KINDS if sel['models'][k]['gain']>0];rows=[];archives={};raw={}
    for rec in recs:raw[rec['file']]=torch.cat([read(rec,s)[:,:20] for s in range(0,201,20)],1).cuda()[None]
    for name in names:
        api=Predictor(name);saved=[]
        for rec in recs:
            a=(raw[rec['file']]-n['mean'])/n['std'];h=a[:,:,:20];chain=[]
            for block in range(10):
                free=api.predict(h,normalized=True)['normalized'];observed=api.predict(a[:,:,block*20:(block+1)*20],normalized=True)['normalized']
                # No target reaches the predictor. Target is selected only after both issues.
                y=a[:,:,(block+1)*20:(block+2)*20]
                for mode,q in [('free',free),('observed',observed)]:
                    assert torch.isfinite(q).all()
                    rows.append(dict(method=name,file=rec['file'],group=rec['group'],block=block+1,issue_time=20*(block+1),score_time=20*(block+2),mode=mode,mse=float(mse(q,y)[0]),energy=float((q*n['std']+n['mean']).square().mean())))
                chain.append(free[0].cpu());h=free
            saved.append(dict(file=rec['file'],free=torch.stack(chain)))
        archives[name]=saved;print('ROLLOUT',name,flush=True);del api
    df=pd.DataFrame(rows);df.to_csv(OUT/'rollout_windows.csv',index=False);tr=df.groupby(['method','mode','file','group'])[['mse','energy']].mean().reset_index();tr.to_csv(OUT/'rollout_trajectories.csv',index=False);tr.groupby(['method','mode'])[['mse','energy']].mean().to_csv(OUT/'rollout_summary.csv');save(archives,OUT/'rollout_issued.pt');print(tr.groupby(['method','mode']).mse.mean().to_string(),flush=True)

if __name__=='__main__':run()
