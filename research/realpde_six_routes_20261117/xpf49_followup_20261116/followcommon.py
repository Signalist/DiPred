from pathlib import Path
import os,sys,json,hashlib,time
import numpy as np
import torch
W=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(W/'f_risk_query_20261116'))
from common import setup as old_setup,cache,norm,model,feature,load,save,dump,digest,records,MANIFEST,SHARED
from query import queries,target_queries,pool,QueryHead,mask,cdf,contain_pairs
sys.path.insert(0,str(W));sys.path.insert(0,str(W/'xpf16_fullpercent_study'));sys.path.insert(0,str(W/'legacy_xpf16_feature_ablation'))
from reliability.features import base_features
from selected16_quantile_head import BASE_INDICES
from evaluate import models_for,all_bounds
from scores import distribution,interval_metrics
from calibration import TwoTailCalibrator
from train49 import configs
OUT=W/'runs/xpf49_followup_20261116'
OLD=W/'runs/xpf16_fullpercent_20261111'
ABL=W/'runs/legacy_xpf16_feature_ablation_20261116'
PRIOR=W/'runs/f_risk_query_20261116'
def setup():old_setup();OUT.mkdir(parents=True,exist_ok=True)
@torch.no_grad()
def features(x,p,f):
 n=norm();stats=load(SHARED,'cuda')['statistics'];geo=torch.ones(len(x),3,device='cuda')
 raw=base_features(x*n['std']+n['mean'],p*n['std']+n['mean'],n['mean'],n['std'],geo)
 base=((raw-stats['base_mean'])/stats['base_std']).clamp(-10,10)
 last=((f.half().float()-stats['last_mean'])/stats['last_std']).clamp(-10,10)
 return base,last
