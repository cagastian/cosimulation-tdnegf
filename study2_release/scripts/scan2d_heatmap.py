#!/usr/bin/env python3
"""
scan2d_heatmap.py -- (J_I, omega0) heatmaps of <W_p> and spin-current FFT
power, AFM and FM side by side. Safe to re-run at any point while the scan
is still going -- picks up whichever (model, J_I, omega0) points have
finished (final npz) or are mid-flight (checkpoint), and just leaves
missing cells blank (NaN, shown as white).

    python scan2d_heatmap.py

writes into ./img/: scan2d_wp_heatmap.{pdf,png},
scan2d_current_fft_heatmap.{pdf,png}
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from run_coupled import setup_style, log, _savefig, _after_transient
from run_study2 import J_I_VALUES, OMEGA0_VALUES, J_K, tag_for

DATA_DIR = HERE / "data"
IMG_DIR = HERE / "img"
MODELS = ["AFM", "FM"]


def load_point(model, ji, w0):
    tag = tag_for(model, ji, w0)
    final = DATA_DIR / f"{tag}_coupled.npz"
    ckpt = DATA_DIR / f"{tag}_ckpt.npz"
    path = final if final.exists() else (ckpt if ckpt.exists() else None)
    if path is None:
        return None
    with np.load(path) as z:
        if "t" not in z.files or len(z["t"]) == 0:
            return None
        t = z["t"]
        Wp = z["Wp"].mean(axis=1)
        Is = z["Is"]
    # <W_p> in the final static window (post-precession, last 25% of the run)
    # -- the number that answers "did the flux survive the drive".
    i0 = int(0.75 * len(t))
    wp_final = float(Wp[i0:].mean())
    # spin-current FFT power at the drive frequency omega0, over the
    # precessing window only (same _after_transient cut fig_currents uses).
    j0 = _after_transient(t)
    dt = t[1] - t[0]
    sig = Is[j0:, 0, 2] - Is[j0:, 0, 2].mean()   # alpha=z, lead 0
    if len(sig) < 8:
        return wp_final, np.nan, final.exists()
    freqs = np.fft.rfftfreq(len(sig), d=dt)
    amp = np.abs(np.fft.rfft(sig))
    w0_target = 2 * np.pi * freqs / float(w0)
    k = np.argmin(np.abs(w0_target - 1.0))
    fft_w0 = float(amp[k])
    return wp_final, fft_w0, final.exists()


def build_grids():
    ji_vals = np.array([float(v) for v in J_I_VALUES])
    w0_vals = np.array([float(v) for v in OMEGA0_VALUES])
    grids = {m: {"wp": np.full((len(J_I_VALUES), len(OMEGA0_VALUES)), np.nan),
                "fft": np.full((len(J_I_VALUES), len(OMEGA0_VALUES)), np.nan)}
            for m in MODELS}
    n_done = {m: 0 for m in MODELS}
    n_total = len(J_I_VALUES) * len(OMEGA0_VALUES)
    for m in MODELS:
        for i, ji in enumerate(J_I_VALUES):
            for j, w0 in enumerate(OMEGA0_VALUES):
                r = load_point(m, ji, w0)
                if r is None:
                    continue
                wp, fft, done = r
                grids[m]["wp"][i, j] = wp
                grids[m]["fft"][i, j] = fft
                n_done[m] += int(done)
    return ji_vals, w0_vals, grids, n_done, n_total


def plot_heatmap_pair(ji_vals, w0_vals, grids, key, label, cmap, path, n_done, n_total,
                      log_scale=False):
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6), layout="constrained")
    from matplotlib.colors import LogNorm
    vals = np.concatenate([grids[m][key][np.isfinite(grids[m][key])] for m in MODELS])
    if len(vals) == 0:
        log(f"  [skip] no finished/checkpointed points yet for {key}")
        plt.close(fig)
        return
    if log_scale:
        vmin = max(vals[vals > 0].min() if np.any(vals > 0) else 1e-12, 1e-12)
        norm = LogNorm(vmin=vmin, vmax=vals.max())
    else:
        norm = None
        vmin, vmax = (0, 1) if key == "wp" else (vals.min(), vals.max())
    for a, m in zip(ax, MODELS):
        mesh = a.pcolormesh(w0_vals, ji_vals, grids[m][key], cmap=cmap,
                            norm=norm, vmin=None if norm else vmin,
                            vmax=None if norm else vmax, shading="nearest")
        a.set_title(f"{m} ({n_done[m]}/{n_total} finished)")
        a.set_xlabel(r"$\omega_0$")
        fig.colorbar(mesh, ax=a, label=label)
    ax[0].set_ylabel(r"$J_I$")
    fig.suptitle(rf"$J_K={J_K:g}$", y=1.05)
    _savefig(fig, path)
    log(f"  wrote {path}.{{pdf,png}}")


def main():
    setup_style()
    ji_vals, w0_vals, grids, n_done, n_total = build_grids()
    if sum(n_done.values()) == 0 and not any(
            np.isfinite(grids[m]["wp"]).any() for m in MODELS):
        log("no data yet for any grid point -- nothing to plot")
        return
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    plot_heatmap_pair(ji_vals, w0_vals, grids, "wp", r"$\langle\hat{W}_p\rangle$ (final)",
                      "viridis", IMG_DIR / "scan2d_wp_heatmap", n_done, n_total)
    plot_heatmap_pair(ji_vals, w0_vals, grids, "fft", r"$|FFT(I^{S_z}_L)|(\omega_0)$",
                      "inferno", IMG_DIR / "scan2d_current_fft_heatmap", n_done, n_total,
                      log_scale=True)


if __name__ == "__main__":
    main()
