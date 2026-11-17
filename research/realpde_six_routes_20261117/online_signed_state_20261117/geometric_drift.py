"""Actual issue/reveal experiment for a known forecast-side geometric fault."""
from geometric import *
from common import records,read
ROOT=OUT/'geometric_drift';VECTOR=np.array([.5,-.4,.5,.06,-.06,.03,-.03])

def run():
    net=st.model();n=norm();configs={};rows=[];coefficients=[];recover=[];events=[]
    for kind,folder in [('nonlinear','geometric'),('linear','linear_geometry')]:
        sel=json.loads((OUT/folder/'selection.json').read_text());configs[kind]=(load(OUT/folder/'model.pt'),{k:v['config'] for k,v in sel['models'].items() if k!='zero'})
    for ri,rec in enumerate(records('development')):
        states={scenario:{kind:{k:GeometricState(cfg,model) for k,cfg in cfgs.items()} for kind,(model,cfgs) in configs.items()} for scenario in ('step','ramp')}
        for j,start in enumerate(range(0,481,20)):
            meta=dict(file=rec['file'],group=rec['group'],start=start,index=j,age=j-12);raw=(read(rec,start)[None].cuda()-n['mean'])/n['std'];outputs={};ps={};injected={}
            with torch.no_grad():
                clean=net.rollout(raw[:,:,:20],500000,sign=0)
                for scenario in states:
                    amount=0. if j<12 else (1. if scenario=='step' else min(1.,(j-11)/4));c=VECTOR*amount*(1 if ri%2==0 else -1);p=clean if amount==0 else st.transform(clean,torch.tensor(c,dtype=torch.float32,device='cuda')[None]);ps[scenario]=p;injected[scenario]=c;outputs[scenario]={'faulted_base':p,'clean_reference':clean}
                    for kind,family in states[scenario].items():
                        for k,state in family.items():
                            cc=state.issue(start);outputs[scenario][kind+'_'+k]=st.transform(p,torch.tensor(cc,dtype=torch.float32,device='cuda')[None]);coefficients.append(dict(**meta,scenario=scenario,method=kind+'_'+k,issued=cc,injected=c))
            events.append(dict(**meta,event='issue_all',previous_reveals=j));y=raw[:,:,20:]
            with torch.no_grad():
                for scenario,methods in outputs.items():
                    for method,q in methods.items():rows.append(dict(**meta,scenario=scenario,method=method,mse=float(st.mse(q,y)[0])))
            for scenario,p in ps.items():
                for kind,identifier in [('nonlinear',st.optimize_oracle),('linear',st.linear_oracle)]:
                    c,_=identifier(p,y)
                    for state in states[scenario][kind].values():state.reveal(c[0].cpu().numpy().astype(float),start+40)
                    if j==12:
                        diag,_=identifier(p,clean)
                        with torch.no_grad():
                            before=float(st.mse(p,clean)[0]);after=float(st.mse(st.transform(p,diag),clean)[0]);recover.append(dict(file=rec['file'],scenario=scenario,identifier=kind,fault_mse=before,remaining_mse=after,oracle_removed_percent=100*(1-after/before),injected=injected[scenario].tolist(),identified=diag[0].cpu().tolist()))
            events.append(dict(**meta,event='full20_reveal',previous_reveals=j+1))
        print('GEOMETRIC_DRIFT',rec['file'],flush=True)
    pd.DataFrame(rows).to_csv(ROOT/'windows.csv',index=False);save(coefficients,ROOT/'coefficients.pt');dump(ROOT/'recoverability.json',recover);(ROOT/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events));dump(ROOT/'provenance.json',dict(protocol_sha256=digest(Path(__file__).with_name('GEOMETRIC_DRIFT_PROTOCOL.md')),source_sha256=digest(Path(__file__)),windows=300,scenarios=['step','ramp'],scope='Controlled forecast warp only. No PDE/parameter/physical-state claim.'))

def report():
    from probability_report import paired
    d=pd.read_csv(ROOT/'windows.csv');tr=d[d.age>=0].groupby(['scenario','method','file','group']).mse.mean().reset_index();tr.to_csv(ROOT/'postchange_trajectories.csv',index=False);rows=[]
    for scenario,g in tr.groupby('scenario'):
        a=g[g.method=='faulted_base'].set_index('file').mse
        for method,h in g.groupby('method'):
            if method=='faulted_base':continue
            b=h.set_index('file').mse.reindex(a.index);rows.append(dict(scenario=scenario,method=method,**paired(a.values,b.values)))
    gains=pd.DataFrame(rows);gains.to_csv(ROOT/'postchange_gains.csv',index=False);d.groupby(['scenario','method','age']).mse.mean().to_csv(ROOT/'recovery_curve.csv');lines=['# 受控几何预测故障恢复','',
    '这只是预测输出端注入已知warp的诊断：真实X/Y及PDE参数不变，不是物理参数漂移。原自然校准的模型、记忆长度、滤波噪声和shrink全部冻结；300窗口/12条已用开发轨迹，前12块无注入、后13块观察恢复。所有状态只在完整20反馈后更新。',
    '', '| 场景 | 方法 | 后变更MSE改善% | 95%轨迹CI | 轨迹胜数 |','|---|---|---:|---|---:|']
    for r in rows:lines.append(f"| {r['scenario']} | {r['method']} | {r['gain_percent']:.3f} | [{r['ci_low']:.3f}, {r['ci_high']:.3f}] | {r['trajectory_wins']}/12 |")
    lines+=['','`clean_reference`不是可部署的修正器；它用于量化注入故障的额外损失。`recoverability.json`用当窗clean P反求系数，只是目标辅助诊断；不能把其收益当作发报性能。','', '全部逐窗误差、发报系数和按反馈年龄的曲线保留。收益是对同场景faulted_base的后变更平均，而不是证明恢复到原始误差水平；自然预测和自由滚动的先前限制仍然成立。']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n');print(gains.to_string(index=False),flush=True)

if __name__=='__main__':
    setup();ROOT.mkdir(parents=True,exist_ok=True);p=argparse.ArgumentParser();p.add_argument('action',choices=['run','report']);args=p.parse_args();run() if args.action=='run' else report()
