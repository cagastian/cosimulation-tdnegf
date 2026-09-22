#!/usr/bin/env python3
"""
run_study2.py -- (J_I, omega0) grid scan for the paper, AFM+FM.

Protocol (in drive periods T = 2*pi/omega0, same "fast" style as the
project's T10 scans, just retimed): Kondo connects at 0.5T, drive starts
precessing at 1T, stops precessing at 12T, run ends at 20T. J_K=0.05,
J_alpha=0.1 fixed.

Grid: J_I in {0.005, 0.010, ..., 0.050} (10 values), omega0 in
{0.001, 0.002, ..., 0.050} (50 values) -- see the conversation this was set
up in for why the FULL grid as specified is an ~5-month sequential compute
commitment (the small-omega0 end dominates: t_final = 20*2*pi/omega0 scales
as 1/omega0, so omega0=0.001 alone costs about as much as this project's
existing 50T reference run). EDIT J_I_VALUES / OMEGA0_VALUES below before
launching if the grid has been scoped down.

Run strictly sequentially -- never two --backend julia jobs at once (see
CONTEXT.md's "Operational lessons"). Safe to interrupt and re-run: already
-finished (tag_coupled.npz present) points are skipped, so this can be
killed and restarted at any time without losing completed points or
duplicating work.

    python run_study2.py [--dry-run]

--dry-run prints the point count and total estimated steps without running
anything.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA_DIR = HERE / "data"
IMG_DIR = HERE / "img"

J_K = 0.05
J_ALPHA = 0.1
DT = 0.5
N_PERIODS = 20.0
T_ON_PERIODS = 1.0
T_OFF_PERIODS = 12.0
T_ON_KONDO_PERIODS = 0.5
K_RAMP = 0.1
K_RAMP_KONDO = 0.02

# Keep as explicit strings (not np.arange floats) so tags/CLI args are
# reproducible and match exactly on re-run -- see jk_scan_heatmap.py's
# JK_LABELS comment for why this matters (float repr drift).
#
# omega0 grid confirmed as {0.01, 0.02, 0.03, 0.04, 0.05} (not the originally
# -typed 0.001-0.05 step 0.001 -- that was a typo and would have been an
# ~5-month sequential run since cost scales as 1/omega0; this 5-point grid
# is an estimated ~8 days for the full 100-point grid).
J_I_VALUES = [f"{v:.3f}" for v in np.arange(0.005, 0.0501, 0.005)]
OMEGA0_VALUES = [f"{v:.3f}" for v in np.arange(0.01, 0.0501, 0.01)]
MODELS = ["AFM", "FM"]

LD_PRELOAD = ("/home/sebasgom/.julia/juliaup/julia-1.12.7+0.x64.linux.gnu/"
             "lib/julia/libstdc++.so.6")


def tag_for(model, ji, w0):
    return f"{model}_jK{J_K:g}_jI{ji}_w{w0}_scan2d"


def steps_for(w0):
    T = 2 * np.pi / float(w0)
    return (N_PERIODS * T) / DT


def all_points():
    for model in MODELS:
        for ji in J_I_VALUES:
            for w0 in OMEGA0_VALUES:
                yield model, ji, w0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    points = list(all_points())
    total_steps = sum(steps_for(w0) for _, _, w0 in points)
    print(f"{len(points)} total points "
         f"({len(J_I_VALUES)} J_I x {len(OMEGA0_VALUES)} omega0 x {len(MODELS)} models), "
         f"~{total_steps:,.0f} total integration steps")
    if args.dry_run:
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    env = {"LD_PRELOAD": LD_PRELOAD}
    import os
    full_env = {**os.environ, **env}

    done = skipped = 0
    for model, ji, w0 in points:
        tag = tag_for(model, ji, w0)
        if (DATA_DIR / f"{tag}_coupled.npz").exists():
            skipped += 1
            continue
        cmd = [
            "python", "run_coupled.py",
            "--backend", "julia", "--jl-dir", ".",
            "--project", "/home/sebasgom/TDNEGF",
            "--model", model, "--J-alpha", str(J_ALPHA), "--J-K", str(J_K),
            "--J-I", ji, "--omega0", w0, "--dt", str(DT),
            "--n-periods", str(N_PERIODS), "--nm-relax", "20",
            "--envelope", "--t-on-periods", str(T_ON_PERIODS),
            "--t-off-periods", str(T_OFF_PERIODS), "--k-ramp", str(K_RAMP),
            "--kondo-ramp", "--t-on-kondo-periods", str(T_ON_KONDO_PERIODS),
            "--k-ramp-kondo", str(K_RAMP_KONDO),
            "--out-dir", str(DATA_DIR), "--img-dir", str(IMG_DIR),
            "--tag", tag,
        ]
        print(f"### {tag} start {time.strftime('%c')}", flush=True)
        rc = subprocess.run(cmd, cwd=str(REPO), env=full_env).returncode
        print(f"### {tag} done rc={rc} {time.strftime('%c')}", flush=True)
        if rc != 0:
            print(f"!!! {tag} failed (rc={rc}) -- stopping so it can be "
                 f"investigated rather than burning through the rest of "
                 f"the grid on a possibly-broken config", flush=True)
            sys.exit(rc)
        done += 1

    print(f"finished: {done} run, {skipped} already done, "
         f"{len(points) - done - skipped} remaining")


if __name__ == "__main__":
    main()
