"""Check actual predictions, full residual memory and calibration state updates."""

import torch

from clean_reliability_common import configure, dump, manifest
from experiment_fullblock_memory_evaluation import OUT, NAMES, MODES, Provider, controller
from experiment_reliability_transport_e12 import normalization, geometry, history, target
from conditional_online_risk import risk_features
from infer_fullblock_memory_risk import FullBlockMemoryReliabilityPredictor


@torch.no_grad()
def main():
    configure()
    provider=Provider(normalization())
    initial=torch.load(OUT/"calibration.pt",map_location="cuda",weights_only=False)
    geo=geometry(manifest()["files"]["test"])
    checks=[]
    for name in ("full","mean"):
        for mode in MODES:
            actual=FullBlockMemoryReliabilityPredictor(geo,name,mode)
            original=controller(mode,initial[name]["separable_integral"],12)
            for start in (0,20,40,60):
                observed=history("test",start)
                result=actual.predict(observed)
                point,bounds=provider.bounds("test",start)
                lower,upper=bounds[name]
                h=(observed-provider.norm["mean"])/provider.norm["std"]
                lo,hi=original.issue(lower,upper,risk_features(h,point,.5*(upper-lower)))
                physical=point*provider.norm["std"]+provider.norm["mean"]
                expected=dict(prediction=physical,lower=physical+lo*provider.norm["std"],upper=physical+hi*provider.norm["std"],
                    risk_half_width=.5*(hi-lo)*provider.norm["std"])
                assert all(torch.equal(result[key],value) for key,value in expected.items())
                assert actual.last_call_labels==[99,66,33,0]
                assert result["memory_available"]==(start>0)
                assert result["previous_feedback_windows"]==start//20
                try:
                    actual.predict(observed)
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("Duplicate issue accepted")
                truth=target("test",start)
                try:
                    actual.reveal(truth[:,:,:10])
                except ValueError:
                    pass
                else:
                    raise AssertionError("Partial feedback accepted")
                actual.reveal(truth)
                residual=(truth-provider.norm["mean"])/provider.norm["std"]-point
                original.reveal(residual)
                assert torch.equal(actual.previous_error,residual)
                for key in ("beta","shape","amplitude","bias"):
                    assert torch.equal(getattr(actual.calibrator,key),getattr(original,key))
                if hasattr(original,"memory"):
                    assert len(actual.calibrator.memory)==len(original.memory)
                    for a,b in zip(actual.calibrator.memory,original.memory):
                        assert all(torch.equal(x,y) for x,y in zip(a,b))
                if hasattr(original,"gram"):
                    assert torch.equal(actual.calibrator.gram,original.gram)
                    assert torch.equal(actual.calibrator.cross,original.cross)
                # The new stream guard rejects noncontiguous observations.
                try:
                    actual.predict(torch.zeros_like(truth))
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("Changed stream accepted without reset")
                checks.append(dict(memory=name,strategy=mode,start=start,batch=12,
                    point_intervals_and_calibration_exact=True,previous_error_exact=True,native_four_calls=True))
            actual.reset()
            assert actual.pending is None and actual.previous_error is None and actual.calibrator is None
    dump(OUT/"direct_inference_audit.json",dict(passed=True,cases=checks,actual_forecasts=192,
        native_labels=[99,66,33,0],duplicate_and_partial_guards=True,contiguous_stream_guard=True,reset_checked=True,
        conditional_coefficients_and_full_deque_or_moment_states_exact=True,
        no_forecast_cache_or_sampling_ensemble_required=True,
        limit="Exact implementation reproduction on historical data, not independent statistical confirmation."))
    print("FULLBLOCK_MEMORY_DIRECT_INFERENCE_AUDIT_PASS",flush=True)


if __name__=="__main__":
    main()
