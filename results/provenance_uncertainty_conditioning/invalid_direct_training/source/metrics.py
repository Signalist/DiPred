"""Complete probability scores, seam diagnostics and member time dependence."""
from core import *
EXTRA=['acf_rmse','temporal_covariance_rms','temporal_cov_difference_B','temporal_corr_difference_B','temporal_cov_difference_S','temporal_corr_difference_S','independent_BC_cov_difference','independent_BC_corr_difference','seam_rms','seam_increment_mse','parent_seam_rms','restart_ed_u','independent_BC_ed_u','restart_excess_ed_u']


def structure(p):
    # Per-channel population covariance of member-centered fields, averaged over space.
    p=p.double();length=p.shape[2]
    centered=p-p.mean(0,keepdim=True)
    x=centered.permute(1,2,0,3,4).reshape(2,length,-1)
    cov=x@x.transpose(1,2)/x.shape[-1]
    scale=cov.diagonal(dim1=1,dim2=2).clamp_min(0).sqrt()
    corr=cov/(scale[:,:,None]*scale[:,None,:]+1e-12)
    ac=[]
    for lag in range(1,length):
        a=p[:,:,:-lag].permute(1,0,2,3,4).reshape(2,-1);b=p[:,:,lag:].permute(1,0,2,3,4).reshape(2,-1)
        a=a-a.mean(1,keepdim=True);b=b-b.mean(1,keepdim=True)
        ac.append((a*b).sum(1)/(a.square().sum(1)*b.square().sum(1)).sqrt().clamp_min(1e-30))
    return dict(cov=cov,corr=corr,acf=torch.stack(ac,1))


def matrix_difference(a,b,ch):return float((a[ch]-b[ch]).square().sum((-1,-2)).sqrt().mean()/a.shape[-1])


@torch.no_grad()
def scores(paths,truth,std,view,k,parent,reference,structure_refs):
    rows,cells=native.quality(paths,truth,std);p=paths.double()/std;y=truth.double()/std
    shape=structure(p);actual=structure(y[None]);bref,sref,cref=structure_refs
    b,s,c=reference
    for row in rows:
        ch=slice(None) if row['channel']=='overall' else slice('uv'.index(row['channel']),'uv'.index(row['channel'])+1)
        row.update({m:None for m in EXTRA})
        row.update(acf_rmse=float((shape['acf'][ch]-actual['acf'][ch]).square().mean().sqrt()),temporal_covariance_rms=float(shape['cov'][ch].square().mean().sqrt()),
            temporal_cov_difference_B=matrix_difference(shape['cov'],bref['cov'],ch),temporal_corr_difference_B=matrix_difference(shape['corr'],bref['corr'],ch),
            temporal_cov_difference_S=matrix_difference(shape['cov'],sref['cov'],ch),temporal_corr_difference_S=matrix_difference(shape['corr'],sref['corr'],ch),
            independent_BC_cov_difference=matrix_difference(bref['cov'],cref['cov'],ch),independent_BC_corr_difference=matrix_difference(bref['corr'],cref['corr'],ch))
        if view=='joint':
            delta=p[:,:,k]-p[:,:,k-1];dy=y[:,k]-y[:,k-1]
            pa=parent.double()/std;pd=pa[:,:,k]-pa[:,:,k-1]
            row.update(seam_rms=float(delta[:,ch].square().mean().sqrt()),seam_increment_mse=float((delta[:,ch]-dy[ch]).square().mean()),parent_seam_rms=float(pd[:,ch].square().mean().sqrt()))
        if view=='tail':
            baseline=native.energy_distance(b[:,ch].double()/std[ch],c[:,ch].double()/std[ch])['u']
            # B is independent of the parent/child family. No U between coupled methods.
            # Native B compared with itself is not an independent pair: leave its restart U blank.
            if not torch.equal(paths,b):
                d=native.energy_distance(b[:,ch].double()/std[ch],p[:,ch])['u']
                row.update(restart_ed_u=d,independent_BC_ed_u=baseline,restart_excess_ed_u=d-baseline)
            else:row['independent_BC_ed_u']=baseline
    for row in cells:
        ch='uv'.index(row['channel']);t=row['tail_lead']-1;scale=float(std[ch,0,0,0])
        row.update(energy=float(native.energy_score(p[:,ch:ch+1,t:t+1],y[ch:ch+1,t:t+1])),standardized_variance=row['variance']/scale**2,standardized_width90=row['width90']/scale,time_mean_crps=row['crps'])
    return rows,cells,{key:value.cpu() for key,value in shape.items()}
