"""Disclose valid, invalidated and reused-cache cost separately."""
from core import *


def main():
    assert not (OUT/'RESOURCE_ACCOUNTING.json').exists()
    valid=json.loads((OUT/'TRAINING_COMPLETE.json').read_text());cost=json.loads((OUT/'EVALUATION_COST.json').read_text())
    bad=OUT/'invalid_direct_training';invalid=json.loads((bad/'TRAINING_COMPLETE.json').read_text())
    oldmanifest=[json.loads(s) for s in (bad/'extension_manifest.jsonl').read_text().splitlines()]
    oldgroups=[json.loads(s) for s in (bad/'evaluation_manifest.jsonl').read_text().splitlines()]
    evaluation_start=json.loads((bad/'EVALUATION_STARTED.json').read_text())['utc']
    invalidation=json.loads((bad/'INVALIDATION.json').read_text())['utc']
    update_times={}
    for label,dir in [('valid',OUT),('invalid',bad)]:
        data=read_csv(dir/'training.csv');update_times[label]=sum(float(r['seconds']) for r in data)
    train_process=json.loads((OUT/'training_process_start.json').read_text())[0]['process_start_utc_estimate']
    invalid_process=json.loads((bad/'training_process_start.json').read_text())[0]['process_start_utc_estimate']
    paths={
      'valid_training_full32_four_step_forward_paths':9000*32,
      'invalid_training_full32_four_step_forward_paths':9000*32,
      'valid_nonzero_development_four_step_paths':4*9*48*4*32,
      'invalid_nonzero_development_four_step_paths':4*9*48*4*32,
      'invalid_frozen_native_extension_paths':len(oldmanifest)*192,
      'valid_new_frozen_native_extension_paths':cost['native_extension_full20_paths'],
      'valid_reused_frozen_native_extension_paths':cost['reused_native_extension_full20_paths'],
      'valid_formal_new_adapter_paths':cost['adapter_full20_paths']}
    oldcache=native.WORK/'runs/provenance_uncertainty_conditioning'
    oldbytes=sum(p.stat().st_size for p in oldcache.rglob('*') if p.is_file())
    validbytes=sum(p.stat().st_size for p in CACHE.rglob('*') if p.is_file())
    dump(OUT/'RESOURCE_ACCOUNTING.json',dict(valid_training_seconds=valid['seconds'],valid_training_including_preload_seconds=valid['utc']-train_process,valid_formal_evaluation_seconds=cost['wall_seconds'],invalid_training_seconds=invalid['seconds'],invalid_training_including_preload_seconds=invalid['utc']-invalid_process,invalid_evaluation_wall_upper_bound_seconds=invalidation-evaluation_start,invalid_complete_evaluation_groups=len(oldgroups),invalid_native_extension_groups=len(oldmanifest),measured_optimizer_update_seconds=update_times,paths=paths,new_cache_bytes_valid=validbytes,retained_invalid_cache_bytes=oldbytes,total_retained_new_cache_bytes=validbytes+oldbytes,full32_inference_forward_calls_per_path=4,restart_forward_calls_per_path_including_parent=8,backward_cost_note='Each training forward retains all four denoising steps and one CRPS backward; this is not inference timing. No extra candidates/updates were added to the valid protocol.',all_prior_attempts_excluded_from_scientific_efficacy=True,utc=time.time()))
    print('ACCOUNTED valid + invalid work and retained caches; no hidden correction cost')

if __name__=='__main__':main()
