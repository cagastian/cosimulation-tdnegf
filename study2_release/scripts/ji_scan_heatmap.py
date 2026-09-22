#!/usr/bin/env python3
"""
ji_scan_heatmap.py -- (J_I, t) heatmap of <W_p> across the AFM J_I scan
(0.001, 0.0025, 0.005, 0.0075, 0.01, 0.02; theta-envelope precessing
0->5.2T, Kondo connecting at 0.2T, T=10 total, J_K=0.02 fixed). Structural
twin of jk_scan_heatmap.py -- same load-partial-or-finished behavior, same
markers, safe to re-run at any point while the scan is still going.

J_I=0.0025 is the shared anchor with the (now complete) J_K scan: it's
literally the same run as AFM_jK0.02_theta_env0-5.2_kondo0.2_T10_scan (no
"_jI" suffix -- that run predates this scan and was never re-launched under
a new tag), so its tag is special-cased below instead of being generated
from JI_LABELS.

    python ji_scan_heatmap.py

writes img_coupled/summary/ji_scan_wp_heatmap.{pdf,png}
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from run_coupled import setup_style, TIME_LABEL, _savefig, log

DATA_DIR = Path("./data_coupled")
IMG_DIR = Path("./img_coupled/summary")
JK_FIXED = "0.02"
# string labels, not floats: must match the literal --tag substitution the
# launch script uses (see run_ji_scan.sh) -- f"{0.010:g}" could collapse
# trailing zeros and silently miss the file.
JI_LABELS = ["0.001", "0.0025", "0.005", "0.0075", "0.01", "0.02"]
TAG_FMT = "AFM_jK{jk}_jI{ji}_theta_env0-5.2_kondo0.2_T10_scan"
ANCHOR_JI = "0.0025"
ANCHOR_TAG = "AFM_jK0.02_theta_env0-5.2_kondo0.2_T10_scan"   # the J_K-scan's own tag

# same markers every other figure in this project uses for this protocol
T_ON, T_OFF, T_ON_KONDO = 0.0, 5.2, 0.2   # in drive periods
OMEGA0 = 0.01
T = 2 * np.pi / OMEGA0


def tag_for(ji_label):
    return ANCHOR_TAG if ji_label == ANCHOR_JI else TAG_FMT.format(jk=JK_FIXED, ji=ji_label)


def load_wp(ji_label):
    tag = tag_for(ji_label)
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
    for ji_label in JI_LABELS:
        r = load_wp(ji_label)
        if r is None:
            log(f"  [skip] J_I={ji_label}: no data yet")
            continue
        rows.append((float(ji_label),) + r)

    if not rows:
        log("no data yet for any J_I -- nothing to plot")
        return

    n = min(len(t) for _, t, _, _ in rows)
    t_common = rows[0][1][:n]
    ji_vals = [r[0] for r in rows]
    grid = np.array([Wp[:n] for _, _, Wp, _ in rows])

    fig, ax = plt.subplots(figsize=(6.5, 3.8), layout="constrained")
    mesh = ax.pcolormesh(t_common, ji_vals, grid, cmap="viridis",
                         vmin=0, vmax=1, shading="nearest")
    ax.set_xlabel(TIME_LABEL)
    ax.set_ylabel(r"$J_I$")
    n_done = sum(done for *_, done in rows)
    ax.set_title(rf"$\langle\hat{{W}}_p\rangle$(t, $J_I$), AFM, $J_K={JK_FIXED}$  "
                f"({n_done}/{len(JI_LABELS)} finished" +
                (", latest still running)" if n_done < len(rows) else ")"))
    fig.colorbar(mesh, ax=ax, label=r"$\langle\hat{W}_p\rangle$")

    for x in (T_ON * T, T_OFF * T):
        ax.axvline(x, color="w", ls="--", lw=0.8, alpha=0.85)
    ax.axvline(T_ON_KONDO * T, color="w", ls=":", lw=0.9, alpha=0.85)

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    stem = IMG_DIR / "ji_scan_wp_heatmap"
    _savefig(fig, stem)
    log(f"  wrote {stem}.{{pdf,png}}  ({len(rows)}/{len(JI_LABELS)} J_I rows, "
        f"{n_done} finished)")


if __name__ == "__main__":
    main()
