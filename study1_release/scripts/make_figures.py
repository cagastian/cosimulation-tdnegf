#!/usr/bin/env python3
"""
make_figures.py -- AFM-vs-FM dashboard + spin-current/spectrogram figures
for Study 1 (updated timing protocol), reusing the same figure code the
main project uses (run_coupled.py / compare_figures.py) rather than
duplicating it. Safe to re-run before both runs finish -- falls back from
the finished npz to the live checkpoint per model, same as the project's
other scan scripts.

    python make_figures.py

writes into ./img/: study1_spin_current.{pdf,png} as soon as EITHER model has
data (1 or 2 rows, whichever is available), and study1_dashboard.{pdf,png}
once BOTH models have data (fig_compare_dashboard isn't generalized to a
single model the way fig_compare_spin_current is). w_max=20, same
convention as the project's compare_spin_current.png.
"""

import json
import sys
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from run_coupled import setup_style, log
from compare_figures import fig_compare_dashboard, fig_compare_spin_current

DATA_DIR = HERE / "data"
IMG_DIR = HERE / "img"
TAGS = {"AFM": "AFM_jK0.05_jI0.014_w0.001_study1",
        "FM": "FM_jK0.05_jI0.014_w0.001_study1"}

# run_coupled.py only writes {tag}_meta.json on completion (save_run), so a
# still-running model has a checkpoint but no meta.json yet -- these are
# run_study1.sh's protocol constants, needed for _drive_marks/_tidy to draw
# the t_on/t_off/kondo-connect markers while plotting from a live
# checkpoint. Keep in sync with run_study1.sh if that protocol ever changes.
DEFAULT_META = {
    "omega0": 0.001, "envelope": True, "t_on_periods": 5.0,
    "t_off_periods": 15.0, "kondo_ramp": True, "t_on_kondo_periods": 1.0,
}


def load_any(tag):
    """Like run_coupled.load_run, but falls back to the live checkpoint if
    the run hasn't finished yet -- same convention as jk_scan_heatmap.py.
    Returns None (not a crash) if the checkpoint is mid-write when read --
    the watcher loop just tries again next cycle."""
    final = DATA_DIR / f"{tag}_coupled.npz"
    ckpt = DATA_DIR / f"{tag}_ckpt.npz"
    path = final if final.exists() else (ckpt if ckpt.exists() else None)
    if path is None:
        return None
    try:
        with np.load(path) as z:
            out = {k: z[k] for k in z.files}
    except (zipfile.BadZipFile, EOFError, OSError):
        return None
    if "t" not in out or len(out["t"]) == 0:
        return None
    meta_p = DATA_DIR / f"{tag}_meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else dict(DEFAULT_META)
    meta.setdefault("omega0", float(out.get("omega0", 0.01)))
    meta.setdefault("title", tag.replace("_", " "))
    return out, meta, path is final


def main():
    setup_style()
    afm = load_any(TAGS["AFM"])
    fm = load_any(TAGS["FM"])

    if afm is None and fm is None:
        log("  [skip] no data yet for either model")
        return

    IMG_DIR.mkdir(parents=True, exist_ok=True)

    models = []
    for label, r in (("AFM", afm), ("FM", fm)):
        if r is None:
            log(f"  [note] {label}: no data yet")
            continue
        out, meta, done = r
        log(f"  [note] {label}: plotting from {'final' if done else 'checkpoint'}")
        models.append((label, out, meta))

    fig_compare_spin_current(models, IMG_DIR / "study1_spin_current", w_max=20.0)
    log(f"  wrote {IMG_DIR / 'study1_spin_current'}.{{pdf,png}} ({len(models)}/2 models)")

    if afm is not None and fm is not None:
        fig_compare_dashboard(afm[0], afm[1], fm[0], fm[1],
                              IMG_DIR / "study1_dashboard")
        log(f"  wrote {IMG_DIR / 'study1_dashboard'}.{{pdf,png}}")
    else:
        log("  [skip] study1_dashboard needs both models")


if __name__ == "__main__":
    main()
