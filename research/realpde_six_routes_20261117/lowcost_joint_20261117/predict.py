"""Selected low-cost joint predictor; no observation/target caches are opened."""
from pathlib import Path
import torch
import study

class LowCostJointPredictor:
    def __init__(self):
        study.setup();study.OUT=study.W/'runs/lowcost_joint_20261117/conditional_scores/scorefit'
        self.core=study.Predictor(method='ar__mixed')
    @torch.no_grad()
    def predict(self,observed,normalized=False,sampling_seed=42):
        if observed.shape!=(1,2,20,32,64):raise ValueError('Expected full observed history [1,2,20,32,64]')
        if not torch.isfinite(observed).all():raise ValueError('Observed history is not finite')
        n=self.core.n;observed=observed.to(n['mean'].device);x=observed if normalized else (observed-n['mean'])/n['std']
        samples=self.core.predict(x,'ar__mixed',sampling_seed)
        return samples if normalized else samples*n['std']+n['mean']

def export():
    folder=study.W/'runs/lowcost_joint_20261117/conditional_scores/scorefit';lib=study.load(folder/'library.pt')
    compact={k:lib[k] for k in ('pca','stats')};study.save(compact,folder/'compact_generator.pt')
    study.dump(folder/'compact_manifest.json',dict(source_library_sha256=study.digest(folder/'library.pt'),compact_sha256=study.digest(folder/'compact_generator.pt'),bytes=(folder/'compact_generator.pt').stat().st_size,fields=list(compact),all_selected_inference_dependences_preserved=True))

if __name__=='__main__':export()
