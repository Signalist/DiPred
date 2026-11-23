"""Locked inputs and metadata-only confirmation eligibility."""
import csv

from research.corrector.run import sha
from research.corrector_window_study.data import RAW, ROOT, STARTS, read_json, verify_weights


def sources(old, local, manifest, selected):
    out = {}
    for name in ("data", "model", "metrics", "run", "verify"):
        path = ROOT / f"research/corrector_window_study/{name}.py"
        out[str(path.relative_to(ROOT))] = sha(path)
    for path in sorted(old.iterdir()):
        if path.suffix in (".json", ".csv", ".md"):
            out[str(path.relative_to(ROOT))] = sha(path)
    for row in selected:
        path = local / f'{row["arm"]}_{row["seed"]}_{row["step"]}.pt'
        assert sha(path) == row["checkpoint_sha256"], path
        out[str(path.relative_to(ROOT))] = row["checkpoint_sha256"]
    verify_weights(manifest["weights"])
    out.update(manifest["weights"])
    return out


def confirmation(manifest):
    meta = ROOT / "metadata/manifest.json"
    npath = ROOT / "metadata/normalization.json"
    protocol = ROOT / "results/residual_dynamics/protocol.json"
    dynamics = ROOT.parent / "Dynamics/Results/residual_dynamics_stream/protocol.json"
    workspace = ROOT.parent / "Forecast/Runtime"
    archive = workspace / "runs/goal_task_error_20261117/calibration/trajectories.csv"
    crossfit = workspace / "runs/crossfit6_corrector_pilot/manifest.json"
    report = workspace / "results/selected16_noise32_report_20261110/data/provenance.json"
    with archive.open() as stream:
        rows = csv.DictReader(stream)
        column = next(k for k in rows.fieldnames if k in ("file","trajectory","trajectory_id"))
        calibrated = {r[column].split("/")[-1] for r in rows}
    crossfit_roles = {r["file"]:tag for tag, records in read_json(crossfit)["files"].items() for r in records}
    reported = {key.split("/")[-1] for key in read_json(report)["case_files"]}
    metadata = read_json(meta)
    fit = read_json(npath)["fit_files"]
    split = {r["file"]: (tag, r) for tag, records in metadata["files"].items() for r in records}
    study = {f: tag for tag, row in manifest["splits"].items() for f in row["files"]}
    duplicates = {r["excluded"]: r for r in metadata["duplicates"]}
    filesystem = sorted(p.name for p in RAW.glob("*.h5"))
    inventory = []
    for file in filesystem:
        role, item = split.get(file, ("unknown", {}))
        dup = duplicates.get(file)
        reason = None
        if file in study:
            reason = "previous window study: " + study[file]
        elif dup:
            reason = "metadata trajectory duplicate of " + dup["retained"]
        elif item.get("frames", 500) < max(STARTS)+40:
            reason = "insufficient frames for the frozen confirmation schedule"
        elif role == "calibration":
            assert file in calibrated
            reason = "documented historical calibration; no fresh validation provenance"
        else:
            raise RuntimeError(f"unclassified real trajectory: {file}")
        inventory.append(dict(file=file, known_role=role, corrector_window_role=study.get(file),
                              backbone_normalization_fit=file in fit, frames=item.get("frames"),
                              historical_calibration_used=file in calibrated,
                              other_corrector_crossfit_role=crossfit_roles.get(file),
                              reported_in_prior_selected16=file in reported,
                              first200_identity=item.get("sha256_first200", (dup or {}).get("sha256_first200")),
                              duplicate_of=(dup or {}).get("retained"), exclusion=reason))
    return dict(confirmation=[], full_local_real_inventory=inventory,
                inventory_only=True, no_target_or_score_read_for_selection=True,
                sources={str(p.relative_to(ROOT.parent)): sha(p) for p in (meta,npath,protocol,dynamics,archive,crossfit,report) if p.exists()},
                evidence_scope=metadata["scope"],
                supplemental_scope="train_sim and other PDE collections are not real trajectories of the frozen task")
