"""Independent marginal-density values, location/scale gradients and pilot checks."""

import json

import numpy as np
from scipy.stats import norm,t
from scipy.optimize import brentq
import torch

from clean_reliability_common import ROOT,configure,dump,digest
from experiment_distribution_error_risk import OUT,OBJECTIVES,NAMES,new_head,load,MEMORY
from experiment_reliability_transport_e12 import normalization
from temporal_mean_memory_control import training_data
from distribution_error_risk import density_loss,Q95


def main():
    configure()
    assert (OUT/"complete.json").exists()
    protocol = json.loads((OUT/"protocol.json").read_text())
    assert digest(ROOT/"experiment_distribution_error_risk.py")==protocol["source_sha256"]
    assert all(digest(ROOT/name)==sha for name,sha in protocol["helper_sha256"].items())
    assert all(digest(ROOT/name)==sha for name,sha in protocol["frozen_artifacts"].items())
    frozen = json.loads((OUT/"heads_frozen.json").read_text())
    assert len(set(frozen["minibatches_and_availability_masks_sha256"].values()))==1
    assert all(digest(OUT/(name+".pt"))==sha for name,sha in frozen["artifacts"].items())
    base,last,h,memory,residual = training_data(normalization())
    probe = torch.linspace(0,818,32,device="cuda").round().long()
    head = new_head(load(MEMORY/"statistics.pt")["memory_scale"])
    density_checks = []
    for kind in ("density_gaussian","density_student4"):
        lower,upper = head.bounds(base[probe],last[probe],h[probe],memory[probe],True)
        nll = density_loss(lower,upper,residual[probe],kind)
        gl,gu = torch.autograd.grad(nll,(lower,upper))
        lo,hi = [value.detach().cpu().numpy().astype(np.float64) for value in (lower,upper)]
        y = residual[probe].cpu().numpy().astype(np.float64)
        mid,radius = .5*(hi+lo),np.maximum(.5*(hi-lo),1e-4)
        scale = radius/Q95[kind]
        error = y-mid
        if kind=="density_gaussian":
            expected_nll = float(-norm.logpdf(y,loc=mid,scale=scale).mean())
            location_gradient = -error/(scale**2)/y.size
            radius_gradient = (1-(error/scale)**2)/radius/y.size
            probability = norm.cdf(Q95[kind])
        else:
            expected_nll = float(-t.logpdf(y,4,loc=mid,scale=scale).mean())
            location_gradient = -5*error/(4*scale**2+error**2)/y.size
            radius_gradient = (1-5*error**2/(4*scale**2+error**2))/radius/y.size
            probability = t.cdf(Q95[kind],4)
        assert abs(float(nll)-expected_nll)<2e-5
        g_location = (gl+gu).detach().cpu().numpy()
        g_radius = (gu-gl).detach().cpu().numpy()
        assert np.allclose(g_location,location_gradient,atol=1e-8,rtol=3e-5)
        assert np.allclose(g_radius,radius_gradient,atol=1e-8,rtol=3e-5)
        assert float(np.linalg.norm(g_location))>0, "Location must receive likelihood gradients."
        qlo = np.where(y>lo,-.05,np.where(y<lo,.95,.45))
        qhi = np.where(y>hi,-.95,np.where(y<hi,.05,-.45))
        q_norm = np.linalg.norm((qhi-qlo)/y.size)
        expected_coefficient = q_norm/np.linalg.norm(radius_gradient)
        actual = frozen["coefficients"][kind]["coefficient"]
        assert abs(actual/expected_coefficient-1)<3e-5
        # SciPy1.6.3 Student-t ppf has a2.8e-10 CDF round-trip error here.
        # The fitted network runs in float32, so verify its quantile against
        # adjacent representable values and an independently refined root.
        # Keep every training constant/checkpoint and the primary1e-6 criterion.
        cdf = norm.cdf if kind=="density_gaussian" else lambda value:t.cdf(value,4)
        refined_q = brentq(lambda value:cdf(value)-.95,1,4,xtol=1e-14)
        assert abs(cdf(refined_q)-.95)<1e-12
        model_q = np.float32(Q95[kind])
        low_q = float(np.nextafter(model_q,np.float32(-np.inf)))
        high_q = float(np.nextafter(model_q,np.float32(np.inf)))
        assert cdf(low_q)<=.95<=cdf(high_q)
        density_checks.append(dict(kind=kind,initial_nll=float(nll),scipy_nll=expected_nll,
            max_location_gradient_error=float(abs(g_location-location_gradient).max()),
            max_radius_gradient_error=float(abs(g_radius-radius_gradient).max()),
            coefficient=actual,numpy_analytic_coefficient=float(expected_coefficient),
            initial_radius_gradient_ratio=float(actual*np.linalg.norm(radius_gradient)/q_norm),
            location_and_scale_both_receive_gradients=True,decoded_upper_quantile_probability=float(probability),
            scipy_ppf_cdf_round_trip_error=float(probability-.95),independently_refined_q95=float(refined_q),
            nominal_probability_bracketed_by_adjacent_float32_quantiles=True))

    rows = json.loads((OUT/"per_case.json").read_text())
    previous = json.loads((ROOT/"runs/point_error_scale_alignment_20261109/per_case.json").read_text())
    keys = ("file","start","channel","variant")
    lookup = {tuple(row[key] for key in keys):row for row in previous if row["map"]=="quantile"}
    controls = 0
    for row in rows:
        if row["map"]=="density_reference":
            old = lookup[tuple(row[key] for key in keys)]
            assert all(row[key]==value for key,value in old.items() if isinstance(value,float))
            controls+=1
    assert controls==1760 and len(rows)==7040
    events = [json.loads(line) for line in (OUT/"pilot_events.jsonl").read_text().splitlines()]
    assert len(events)==20
    for issue,reveal in zip(events[::2],events[1::2]):
        assert issue["event"]=="issued" and reveal["event"]=="revealed" and issue["start"]==reveal["start"]
        assert issue["time"]<=reveal["time"]
    from statistics import mean
    directions = {"standardized_interval_score":-1,"standardized_width":-1,"relative_aurc":-1,
                  "top10_energy":1,"per_frame_top10_energy":1,"worst_frame_coverage":1,
                  "worst_patch_coverage":1,"coverage_error":-1}
    gate = json.loads((OUT/"gate.json").read_text())
    for mode,result in gate["results"].items():
        for name,expected in result["values"].items():
            groups = {}
            for row in rows:
                if row["map"]==name and row["variant"]==mode and row["start"]>=600:
                    groups.setdefault((row["group"],row["channel"]),[]).append(row)
            for metric,value in expected.items():
                actual = mean(abs(mean(r["coverage"] for r in group)-.9) for group in groups.values()) if metric=="coverage_error" else mean(mean(r[metric] for r in group) for group in groups.values())
                assert actual==value
        old = result["values"]["density_quantile"]
        passed = []
        for name in ("density_gaussian","density_student4"):
            new = result["values"][name]
            failures = [key for key,direction in directions.items() if direction*(new[key]-old[key])< -1e-6]
            assert failures==result["comparisons"][name]["regressions"]
            strict = new["standardized_interval_score"]<old["standardized_interval_score"]-1e-6
            assert strict==result["comparisons"][name]["strict_score_gain"]
            if not failures and strict:
                passed.append(name)
        assert passed==result["passed"]
    dump(OUT/"independent_training_pilot_audit.json",dict(passed=True,density_and_gradient_checks=density_checks,
        same_initial90percentintervals=True,matched_minibatches_and_availability=True,
        existing_frozen_reference_rows_exact=controls,all_pilot_rows=len(rows),complete_issue_reveal_pairs=10,
        pilot_group_metrics_and_gates_exact=True,independent_calibration_not_used_for_fitting=True,
        limitation="Per-location marginal density assumptions, not jointfield uncertainty or prooftruePDEerrorfamily. Reusedtrainpilot, not independentconfirmation."))
    print("ERROR_DENSITY_RISK_AUDIT_PASS",flush=True)


if __name__=="__main__":
    main()
