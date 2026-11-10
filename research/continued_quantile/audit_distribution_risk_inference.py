"""Replay actual native four-call quantile-control inference and full states."""

import torch

from clean_reliability_common import configure,dump,manifest
from experiment_random_window_continuation_followup import OUT,Provider,MODES,controller
from experiment_reliability_transport_e12 import normalization,geometry,history,target
from experiment_fullblock_memory_risk import load
from conditional_online_risk import risk_features
from infer_distribution_risk import DistributionRiskPredictor


@torch.no_grad()
def main():
    configure()
    assert (OUT/"complete.json").exists()
    provider = Provider(normalization())
    initial = load(OUT/"calibration.pt")
    geo = geometry(manifest()["files"]["test"])
    cases = []
    for mode in MODES:
        actual = DistributionRiskPredictor(geo,"density_quantile",mode)
        original = controller(mode,initial["density_quantile"]["separable_integral"],len(geo))
        for start in (0,20,40,60):
            observed = history("test",start)
            result = actual.predict(observed)
            point,bounds = provider.bounds("test",start)
            lower,upper = bounds["density_quantile"]
            h = (observed-provider.norm["mean"])/provider.norm["std"]
            lo,hi = original.issue(lower,upper,risk_features(h,point,.5*(upper-lower)))
            physical = point*provider.norm["std"]+provider.norm["mean"]
            expected = dict(prediction=physical,lower=physical+lo*provider.norm["std"],upper=physical+hi*provider.norm["std"],
                            risk_half_width=.5*(hi-lo)*provider.norm["std"])
            assert all(torch.equal(result[key],value) for key,value in expected.items())
            assert actual.last_call_labels==[99,66,33,0]
            assert result["memory_available"]==(start>0) and result["previous_feedback_windows"]==start//20
            try:
                actual.predict(observed)
            except RuntimeError:
                pass
            else:
                raise AssertionError("Duplicate issue accepted")
            truth = target("test",start)
            try:
                actual.reveal(truth[:,:,:10])
            except ValueError:
                pass
            else:
                raise AssertionError("Partial feedback accepted")
            actual.reveal(truth)
            residual = (truth-provider.norm["mean"])/provider.norm["std"]-point
            original.reveal(residual)
            assert torch.equal(actual.previous_error,residual)
            for key in ("beta","shape","amplitude","bias"):
                assert torch.equal(getattr(actual.calibrator,key),getattr(original,key))
            if hasattr(original,"memory"):
                assert len(actual.calibrator.memory)==len(original.memory)
                for left,right in zip(actual.calibrator.memory,original.memory):
                    assert all(torch.equal(a,b) for a,b in zip(left,right))
            if hasattr(original,"gram"):
                assert torch.equal(actual.calibrator.gram,original.gram)
                assert torch.equal(actual.calibrator.cross,original.cross)
            try:
                actual.predict(torch.zeros_like(truth))
            except RuntimeError:
                pass
            else:
                raise AssertionError("Noncontiguous stream accepted")
            cases.append(dict(objective="density_quantile",strategy=mode,start=start,batch=len(geo),
                              point_interval_previous_error_and_all_states_exact=True,native_four_calls=True))
            print("DISTRIBUTION_RISK_ACTUAL_INFERENCE",mode,start,flush=True)
        actual.reset()
        assert actual.pending is None and actual.previous_error is None and actual.calibrator is None
    dump(OUT/"direct_inference_audit.json",dict(passed=True,cases=cases,actual_forecasts=96,
        native_labels=[99,66,33,0],full_deque_and_moment_states_exact=True,
        duplicate_partial_noncontiguous_and_reset_guards=True,no_sampling_or_issued_cache_required=True,
        limitation="Historicaltestinput implementation reproduction, not new statisticalconfirmation."))
    print("DISTRIBUTION_RISK_INFERENCE_AUDIT_PASS",flush=True)


if __name__=="__main__":
    main()
