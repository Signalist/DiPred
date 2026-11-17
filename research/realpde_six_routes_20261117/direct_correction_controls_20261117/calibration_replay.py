"""Freeze actual batch1 calibration choices before any development scoring."""
from controls import *
import shutil

@torch.no_grad()
def run():
    setup();d=data('calibration');sel=json.loads((OUT/'selection.json').read_text());rows=[];changes=[];maximum=0.
    assert not (OUT/'development/windows.csv').exists(), 'This freeze must precede development evaluation'
    for name in KINDS:
        api=Predictor(name);ck=load(OUT/f'{name}.pt','cuda');options={};old=ck['summary']['gain']
        for gain in [0.,.25,.5,1.]:
            losses=[]
            for i,meta in enumerate(d['meta']):
                q=mix(api.head,d['x'][i:i+1],d['p'][i:i+1],d['z'][i:i+1],gain);loss=mse(q,d['y'][i:i+1])[0];losses.append(loss);rows.append(dict(**meta,method=name,gain=gain,mse=float(loss)))
            options[gain]=aggregate(torch.stack(losses),d['meta'])
        choice=min(options,key=options.get)
        for i in [0,4]:
            out=api.predict(d['x'][i:i+1],normalized=True);expected=mix(api.head,d['x'][i:i+1],d['p'][i:i+1],d['z'][i:i+1],old);maximum=max(maximum,float((out['normalized']-expected).abs().max()))
        assert maximum<1e-7,maximum
        if choice!=old:changes.append(dict(method=name,old=old,new=choice))
        shutil.copy2(OUT/f'{name}.pt',OUT/f'{name}_batched_calibration.pt')
        ck['summary'].update(gain=choice,options=options,calibration_mse=options[choice],calibration_batch_size=1)
        save(ck,OUT/f'{name}.pt');sel['models'][name]=ck['summary'];sel['model_hashes'][name]=digest(OUT/f'{name}.pt')
    sel['winner']=min(KINDS,key=lambda k:sel['models'][k]['calibration_mse']);dump(OUT/'selection.json',sel);pd.DataFrame(rows).to_csv(OUT/'actual_calibration_grid.csv',index=False)
    dump(OUT/'calibration_replay.json',dict(passed=True,gain_changes=changes,actual_public_max_error=maximum,winner=sel['winner'],scope='Before development scoring; batch1 calibration refreeze only. Original batched choices retained.'))
    print('ACTUAL_CALIBRATION',changes,maximum,sel['winner'],flush=True)

if __name__=='__main__':run()
