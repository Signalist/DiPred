from task_error import *
import shutil

@torch.no_grad()
def run():
    setup();d=load(OUT/'train.pt','cuda');cal=load(OUT/'calibration.pt','cuda');w=weights(d['meta']);wc=weights(cal['meta']);ck=load(OUT/'models.pt','cuda');sel=json.loads((OUT/'selection.json').read_text());target=d['jy']-d['jp'];tasks=Tasks();rows=[]
    for name in ['models.pt','selection.json','selection_grid.csv']:
        dest=OUT/('initial_'+name)
        if not dest.exists():shutil.copy2(OUT/name,dest)
    ck['models']['task_persistence']=dict(style='task_persistence')
    rr=raw(d,'task_stats');mean=(w[:,None]*rr).sum(0);sd=((rr-mean).square()*w[:,None]).sum(0).sqrt().clamp_min(1e-8);z=(rr-mean)/sd
    pr=projection(rr,w,8);zp=project(rr,pr)
    for style in ['task_affine','direct']:
        name='task_affine' if style=='task_affine' else 'direct/task_stats/8';best=None
        for alpha in [.01,.1,1.,10.]:
            if style=='task_affine':coef=torch.cat([ridge(z[:,[i,i+4]],target[:,i:i+1],w,alpha) for i in range(4)],1)
            else:coef=ridge(zp,target,w,alpha)
            for shrink in [0.,.25,.5,1.]:
                model=dict(style=style,kind='task_stats',coef=coef,shrink=shrink,alpha=alpha,mean=mean,sd=sd,projection=pr,rank=8)
                pred=prediction(cal,model,tasks);loss=float((((pred-cal['jy']).square()/ck['unit']).mean(1)*wc).sum());rows.append(dict(model=name,alpha=alpha,shrink=shrink,score=loss))
                if best is None or loss<best['score']-1e-14:best=dict(model,score=loss)
        ck['models'][name]=best;print(name,best['alpha'],best['shrink'],best['score'],flush=True)
    for name in ['task_persistence','task_affine','direct/task_stats/8']:
        model=ck['models'][name];sel['models'][name]={k:v for k,v in model.items() if k in ['style','kind','rank','shrink','alpha','score']};sel['readout_parameters'][name]=int(model['coef'].numel()) if 'coef' in model else 0
    save(ck,OUT/'models.pt');sel.update(model_sha256=digest(OUT/'models.pt'),simple_controls_addendum_sha256=digest(Path(__file__).with_name('SIMPLE_CONTROL_ADDENDUM.md')),original_selection_sha256=digest(OUT/'initial_selection.json'));dump(OUT/'selection.json',sel);pd.DataFrame(rows).to_csv(OUT/'simple_control_grid.csv',index=False)

if __name__=='__main__':run()
