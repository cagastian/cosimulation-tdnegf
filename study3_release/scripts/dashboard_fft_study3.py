#!/usr/bin/env python3
"""
dashboard_fft_study3.py -- AFM-vs-FM spin-current spectrogram comparison for
Study 3's (fixed) T=40 runs, one figure PER omega0 (3 separate figures, not
one combined grid -- each omega0 is its own physical scenario and gets its
own AFM/FM comparison). 2 rows (AFM, FM) x 4 columns per figure: col 0 is
<S^alpha_tot>/N (x/y/z together, same context panel as the single-run
dashboard's panel (a)), cols 1-3 are (omega/omega0, t) spectrogram heatmaps
of the pumped spin current, one per alpha=x,y,z -- port of
compare_figures.fig_compare_spin_current with col 0 swapped from the full
current I^Salpha(t) to S^alpha_tot/N (per user request). Model identity is
the col-0 title (same convention as fig_compare_spin_current), not a row
bracket -- there's no room for both a bracket and a labeled colorbar at
only 2 rows.

    python dashboard_fft_study3.py [--w-max 8]

writes img/dashboard_fft_study3_w<omega0>.{pdf,png}, one per omega0.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from run_coupled import (
    setup_style, load_run, N, C_AXIS, _panel_letter, _savefig, log,
    _tidy, _after_transient, TIME_LABEL, _legend_top,
    _spectrogram_specs, _plot_spectrogram_panel,
)
from compare_figures import ALPHA_CMAP, _cap_drive_marks

DATA_DIR = HERE / "data"
IMG_DIR = HERE / "img"

OMEGA0_VALUES = ["0.01", "0.02", "0.04"]
MODEL_JI = {"AFM": "0.015", "FM": "0.0015"}
MODELS = ["AFM", "FM"]


def tag_for(model, w0):
    return f"{model}_jK0.05_jI{MODEL_JI[model]}_w{w0}_study3"


def make_figure(w0, w_max):
    models = []
    for model in MODELS:
        tag = tag_for(model, w0)
        p = DATA_DIR / f"{tag}_coupled.npz"
        if not p.exists():
            log(f"  [skip] {tag}: no data yet")
            continue
        out, meta = load_run(tag, DATA_DIR)
        models.append((model, out, meta))

    if not models:
        log(f"  omega0={w0}: no data yet -- skipping")
        return

    n = len(models)
    with plt.rc_context({"font.size": 13, "axes.labelsize": 13,
                        "axes.titlesize": 13, "xtick.labelsize": 11,
                        "ytick.labelsize": 11}):
        fig, ax = plt.subplots(n, 4, figsize=(10.2, 1.9 * n + 0.5),
                               squeeze=False, layout="constrained")
        letters = iter("abcdefghijklmnop")

        for c in range(4):
            for r in range(1, n):
                ax[r, c].sharex(ax[0, c])

        for r, (model, out, meta) in enumerate(models):
            t, T = out["t"], 2 * np.pi / meta["omega0"]
            i0 = _after_transient(t)
            for al in "xyz":
                comp = out["S"].sum(axis=1)[:, "xyz".index(al)] / N
                ax[r, 0].plot(t[i0:], comp[i0:], color=C_AXIS[al], lw=0.8,
                             label=rf"$\alpha={al}$")
            ax[r, 0].set_title(model)
            ax[r, 0].set_ylabel(r"$\langle\hat{S}^\alpha_{\mathrm{tot}}\rangle/N$")
            _tidy(ax[r, 0], t, T, meta, letter=None, nbins=4)
            _panel_letter(ax[r, 0], next(letters), xy=(0.01, 1.05), fontsize=11)

        y_cap = ax[0, 0].get_ylim()[1]
        _legend_top(ax[0, 0], ncol=3, fontsize=11, handlelength=0.6)
        _cap_drive_marks(ax[0, 0], y_cap=y_cap)

        for r, (model, out, meta) in enumerate(models):
            specs, norm, T = _spectrogram_specs(out, meta, w_max=w_max)
            im = None
            for ci, (al, (title, wr, tt, Sxx)) in enumerate(zip("xyz", specs), start=1):
                im = _plot_spectrogram_panel(ax[r, ci], None, wr, tt, Sxx, norm,
                                             meta, T, w_max, letter=next(letters),
                                             cmap=ALPHA_CMAP[al])
                if ci > 1:
                    ax[r, ci].set_ylabel("")
                    ax[r, ci].tick_params(labelleft=False)
            fig.colorbar(im, ax=ax[r, 1:], shrink=0.8, pad=0.02,
                        label=f"{model} FFT A.U.")

        for c in range(4):
            for r in range(n):
                if r == n - 1:
                    ax[r, c].set_xlabel(TIME_LABEL)
                else:
                    ax[r, c].tick_params(labelbottom=False)

        IMG_DIR.mkdir(parents=True, exist_ok=True)
        path = IMG_DIR / f"dashboard_fft_study3_w{w0}"
        _savefig(fig, path)
        log(f"  wrote {path}.{{pdf,png}} ({n}/{len(MODELS)} models, omega0={w0})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w-max", type=float, default=8.0)
    args = ap.parse_args()
    setup_style()
    for w0 in OMEGA0_VALUES:
        make_figure(w0, args.w_max)


if __name__ == "__main__":
    main()
