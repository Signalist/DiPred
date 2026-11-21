"""Expose all nonzero development candidates; no new candidate, fit or selection."""
from core import *
from research.provenance_uncertainty_conditioning.analyze import aggregate,intervals


def main():
    assert not (OUT/'DEVELOPMENT_ANALYSIS.json').exists()
    rows=read_csv(OUT/'development.csv')
    keys=['trajectory','Re','start','repeat','k','view','channel']
    ix={tuple(r[k] for k in keys)+(r['arm'],r['seed'],r['step']):r for r in rows}
    differences=[]
    for r in rows:
        if r['step']=='0':continue
        targets=[(r['arm']+' minus native',r['arm'],'0')]
        if r['arm']=='C_mask':targets.append(('C_mask minus C_aug','C_aug',r['step']))
        if r['arm']=='C_unc':targets.append(('C_unc minus C_mask','C_mask',r['step']))
        for contrast,arm,step in targets:
            other=ix[tuple(r[k] for k in keys)+(arm,r['seed'],step)]
            differences.append(dict({k:r[k] for k in keys},split='development',seed=r['seed'],step=r['step'],contrast=contrast,**{m:float(r[m])-float(other[m]) for m in QUAL}))
    tr=aggregate(differences,['split','trajectory','Re','k','view','channel','seed','step','contrast'],QUAL)
    mean=aggregate(tr,['split','trajectory','Re','k','view','channel','step','contrast'],QUAL)
    for r in mean:r['seed']='mean3'
    allrows=tr+mean
    csvwrite(OUT/'development_nonzero_paired_trajectory.csv',allrows)
    csvwrite(OUT/'development_nonzero_paired.csv',intervals(allrows,['split','k','view','channel','seed','step','contrast'],QUAL))
    csvwrite(OUT/'development_nonzero_by_Re.csv',aggregate(allrows,['split','Re','k','view','channel','seed','step','contrast'],QUAL))
    pertr=aggregate(rows,['trajectory','Re','k','view','channel','seed','step','arm'],QUAL)
    csvwrite(OUT/'development_summary.csv',aggregate(pertr,['k','view','channel','seed','step','arm'],QUAL))
    dump(OUT/'DEVELOPMENT_ANALYSIS.json',dict(used_only_existing_development_scores=True,selection_unchanged_sha256=digest(OUT/'selection.json'),steps=[250,500,750,1000],statement='All nonzero predeclared candidates disclosed; these diagnostics do not promote a new test candidate or constitute independent confirmation.',code_sha256=digest(Path(__file__)),utc=time.time()))
    print('DISCLOSED all36 nonzero candidates, matched-step incremental comparisons and point/diversity costs',flush=True)

if __name__=='__main__':main()
