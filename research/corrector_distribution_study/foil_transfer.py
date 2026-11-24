"""Prepare a metadata-selected, frozen RealPDEBench foil PIV transfer cohort."""

import argparse
import hashlib
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa

from research.corrector.run import dump, sha
from research.corrector_window_study.data import RAW, ROOT, read_json

from . import confirm

FOIL = ROOT.parents[1] / "OLD/DATASET-5PDE/foil"
LEGACY_REFERENCE = "3750_0.h5"


def prior_parameters():
    metadata = read_json(ROOT / "metadata/manifest.json")
    old_rows = [r for split in metadata["files"].values() for r in split]
    known = {(int(r[key] if key == "re" else r["file"].split("_")[0]), r["aoa"])
             for r in old_rows for key in ("re", "file")}
    return known, {r["sha256_first200"] for r in old_rows}


def reference_spacing():
    with h5py.File(RAW / LEGACY_REFERENCE, "r") as handle:
        return (float(np.median(np.diff(handle["t"][:]))),
                float(np.median(np.diff(handle["y"][::2, 0]))),
                float(np.median(np.diff(handle["x"][0, ::2]))))


def scalar(batch, field):
    return batch.column(batch.schema.get_field_index(field))[0]


def array(batch, field, dtype, shape):
    return np.frombuffer(scalar(batch, field).as_buffer(), dtype=dtype).reshape(shape)


def adapt(batch, root, name, case_folder, reference):
    shape = tuple(scalar(batch, f"shape_{axis}").as_py() for axis in ("t", "h", "w"))
    assert shape == (3990, 128, 256)
    time = array(batch, "t", "<f8", (shape[0],))
    assert np.allclose(np.diff(time), .005, atol=1e-9, rtol=0)
    y = array(batch, "y", "<f8", shape[1:])
    x = array(batch, "x", "<f8", shape[1:])
    assert (np.diff(y[:, 0]) > 0).all() and (np.diff(x[0]) > 0).all()
    # Foil rows ascend in physical y. The earlier DiPred inputs descend.
    y_indices = slice(None, None, -2)
    x_indices = slice(None, None, 2)
    model_y = y[y_indices, x_indices][::2, 0]
    model_x = x[y_indices, x_indices][0, ::2]
    spacing = (float(np.median(np.diff(time[::4]))),
               float(np.median(np.diff(model_y))), float(np.median(np.diff(model_x))))
    np.testing.assert_allclose(spacing, reference, rtol=.04, atol=1e-9)
    dest = case_folder / name
    with h5py.File(dest, "x") as handle:
        for c in ("u", "v"):
            f = array(batch, c, "<f4", shape)[0:2000:4, y_indices, x_indices]
            assert f.shape == (500, 64, 128) and np.isfinite(f).all()
            handle.create_dataset(c, data=f)
        handle.create_dataset("t", data=time[0:2000:4])
        handle.create_dataset("y", data=y[y_indices, x_indices])
        handle.create_dataset("x", data=x[y_indices, x_indices])
    with h5py.File(dest, "r") as handle:
        first = confirm.fields(handle, 0, 200)
    return dict(path=str(dest), sha256=sha(dest),
                sha256_first200=hashlib.sha256(first.tobytes()).hexdigest(),
                first_model_time=float(time[0]), model_spacing=list(spacing), frames=500,
                arrow_shape=list(shape), source=root)


def prepare(out):
    frozen, _ = confirm.frozen_inputs()
    known, prior_hashes = prior_parameters()
    official = FOIL / "out_dist_test_params_real.json"
    names = read_json(official)
    selected = {name: (re, angle) for name, (re, angle) in names.items()
                if (int(re), angle) not in known}
    assert len(selected) == 8 and set(names) - set(selected) == {"17812_15.0.h5", "3750_0.0.h5"}
    ref = reference_spacing()
    out.mkdir(parents=True, exist_ok=False)
    cases = out / "inputs"
    cases.mkdir()
    rows = {}
    for arrow in sorted((FOIL / "hf_dataset/real").glob("*.arrow")):
        with pa.memory_map(str(arrow)) as source:
            batch = pa.ipc.open_stream(source).read_next_batch()
            assert batch.num_rows == 1
            name = scalar(batch, "sim_id").as_py()
            if name not in selected:
                continue
            assert name not in rows
            re, angle = selected[name]
            row = adapt(batch, str(arrow.relative_to(FOIL)), name, cases, ref)
            assert row["sha256_first200"] not in prior_hashes
            prior_hashes.add(row["sha256_first200"])
            rows[name] = dict(row, trajectory_id=Path(name).stem, benchmark_re=re, angle_deg=angle,
                              source_shard_bytes=arrow.stat().st_size)
            print("ADAPTED", name, row["source"], row["model_spacing"], flush=True)
            if set(rows) == set(selected):
                break
    assert set(rows) == set(selected)
    version = FOIL.parent / "version.json"
    dump(out / "LOCKED.json", dict(
        phase="external transfer on RealPDEBench v2.0.1 real foil; independent evaluation settings, exposure not certified",
        frozen_sha256=sha(confirm.STUDY / "FROZEN.json"), script_sha256=sha(Path(confirm.__file__)),
        adapter_sha256=sha(Path(__file__)), source_version_sha256=sha(version),
        official_ood_params_sha256=sha(official), excluded_metadata_overlap=sorted(set(names)-set(rows)),
        cases=[rows[n] for n in sorted(rows)], windows=frozen["windows"], member_seeds=frozen["members"],
        reference_spacing=list(ref), normalized_or_refit=False,
        source_stride=[4,-2,2], subsequent_model_stride=[1,2,2],
        primary="recent minus S2 standardized CRPS", interval="95% paired trajectory bootstrap, descriptive"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("prepare", "score"))
    p.add_argument("out", type=Path)
    a = p.parse_args()
    out = a.out.resolve()
    if a.phase == "prepare":
        prepare(out)
    else:
        locked = read_json(out / "LOCKED.json")
        assert locked["adapter_sha256"] == sha(Path(__file__))
        confirm.score(out)


if __name__ == "__main__":
    main()
