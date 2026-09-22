#!/usr/bin/env python3
"""
jk_scan_heatmap.py -- (J_K, t) heatmap of <W_p> across the AFM J_K scan
(0.02, 0.04, 0.06, 0.08, 0.09, 0.10, 0.11, 0.12; theta-envelope precessing
0->5.2T, Kondo connecting at 0.2T, T=10 total). Safe to re-run at any point
while the scan is still going -- picks up whichever runs have finished
(final npz) or are mid-flight (checkpoint), and just skips any that
haven't started yet.

    python jk_scan_heatmap.py

writes img_coupled/summary/jk_scan_wp_heatmap.{pdf,png}
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from run_coupled import setup_style, TIME_LABEL, _savefig, log

DATA_DIR = Path("./data_coupled")
IMG_DIR = Path("./img_coupled/summary")
# string labels, not floats: must match the literal --tag substitution the
# launch scripts used (bash's "0.10" stays "0.10", but f"{0.10:g}" would
# collapse to "0.1" and silently miss the file -- keep these as strings).
JK_LABELS = ["0.02", "0.04", "0.06", "0.08", "0.09", "0.10", "0.11", "0.12"]
TAG_FMT = "AFM_jK{jk}_theta_env0-5.2_kondo0.2_T10_scan"

# same markers every other figure in this project uses for this protocol
T_ON, T_OFF, T_ON_KONDO = 0.0, 5.2, 0.2   # in drive periods
OMEGA0 = 0.01
T = 2 * np.pi / OMEGA0


def load_wp(jk_label):
    tag = TAG_FMT.format(jk=jk_label)
    final = DATA_DIR / f"{tag}_coupled.npz"
    ckpt = DATA_DIR / f"{tag}_ckpt.npz"
    path = final if final.exists() else (ckpt if ckpt.exists() else None)
    if path is None:
        return None
    d = np.load(path, allow_pickle=True)
    if "t" not in d.files or len(d["t"]) == 0:
        return None
    return d["t"], d["Wp"].mean(axis=1), final.exists()


def main():
    setup_style()
    rows = []
    for jk_label in JK_LABELS:
        r = load_wp(jk_label)
        if r is None:
            log(f"  [skip] J_K={jk_label}: no data yet")
            continue
        rows.append((float(jk_label),) + r)

    if not rows:
        log("no data yet for any J_K -- nothing to plot")
        return

    n = min(len(t) for _, t, _, _ in rows)
    t_common = rows[0][1][:n]
    jk_vals = [r[0] for r in rows]
    grid = np.array([Wp[:n] for _, _, Wp, _ in rows])

    fig, ax = plt.subplots(figsize=(6.5, 3.8), layout="constrained")
    mesh = ax.pcolormesh(t_common, jk_vals, grid, cmap="viridis",
                         vmin=0, vmax=1, shading="nearest")
    ax.set_xlabel(TIME_LABEL)
    ax.set_ylabel(r"$J_K$")
    n_done = sum(done for *_, done in rows)
    ax.set_title(rf"$\langle\hat{{W}}_p\rangle$(t, $J_K$), AFM  "
                f"({n_done}/{len(JK_LABELS)} finished" +
                (", latest still running)" if n_done < len(rows) else ")"))
    fig.colorbar(mesh, ax=ax, label=r"$\langle\hat{W}_p\rangle$")

    for x in (T_ON * T, T_OFF * T):
        ax.axvline(x, color="w", ls="--", lw=0.8, alpha=0.85)
    ax.axvline(T_ON_KONDO * T, color="w", ls=":", lw=0.9, alpha=0.85)

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    stem = IMG_DIR / "jk_scan_wp_heatmap"
    _savefig(fig, stem)
    log(f"  wrote {stem}.{{pdf,png}}  ({len(rows)}/{len(JK_LABELS)} J_K rows, "
        f"{n_done} finished)")


if __name__ == "__main__":
    main()
