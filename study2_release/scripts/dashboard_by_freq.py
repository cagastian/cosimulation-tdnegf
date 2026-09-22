#!/usr/bin/env python3
"""
dashboard_by_freq.py -- 2x4 dashboard variant showing how the dynamics
change with drive frequency omega0, at fixed J_I per model (Study 2's
grid; AFM and FM can use different J_I, since each model's interesting
regime sits at a different J_I). Same 4 columns as
compare_figures.fig_compare_dashboard, but:
  - <W_p> is replaced by its mean over the 3 plaquettes (1 curve instead
    of 3) so it overlays cleanly across frequencies.
  - S_tot is shown as just S_z (the drive/precession axis) for the same
    declutter reason -- 3 omega0 x 3 alpha components would be 9 lines.
    Colored red (C_AXIS["z"]) to match this project's established
    x/y/z color convention rather than inventing a new color mapping.
  - all 4 columns overlay one curve per omega0, distinguished by
    linestyle (solid/dashed/dotted for omega0/2*omega0/4*omega0) --
    color is reserved for the x/y/z convention (S_z's red here), so
    frequency identity rides on linestyle everywhere, not color.
  - x-axis is t/T (drive periods), not raw time: different omega0 runs
    have different absolute durations (T = 2*pi/omega0), but the same
    period-based protocol (t_on=1T, t_off=12T, kondo-on=0.5T) -- plotting
    in periods is what makes the 3 curves directly comparable on one axis.
  - legend is inset in panel (a) rather than a separate figure-level
    legend, labeled omega_0/omega_1=2*omega_0/omega_2=4*omega_0.
GMN column stays blank (no GMN.jl solve exists for these scan runs,
matching the project's usual missing-GMN-data convention -- see
compare_figures._gmn_column).

    python dashboard_by_freq.py [--J-I-afm 0.015] [--J-I-fm 0.010]

writes img/dashboard_by_freq_AFM-jI<>_FM-jI<>.{pdf,png}. Safe to re-run
before all omega0 points exist for either model -- skips a (model, omega0)
curve with a log message if its data isn't there yet.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from run_coupled import setup_style, N, C_AXIS, _panel_letter, _savefig, log
from compare_figures import _row_bracket
from run_study2 import tag_for

DATA_DIR = HERE / "data"
IMG_DIR = HERE / "img"

OMEGA0_VALUES = ["0.010", "0.020", "0.040"]
FREQ_LS = {"0.010": "-", "0.020": "--", "0.040": ":"}
FREQ_LABELS = {"0.010": r"$\omega_0$", "0.020": r"$\omega_1=2\omega_0$",
              "0.040": r"$\omega_2=4\omega_0$"}
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}   # matches the "xyz" loop order S was recorded in
MODELS = ["AFM", "FM"]
T_ON, T_OFF, T_ON_KONDO = 1.0, 12.0, 0.5   # periods -- same for every omega0


def load_point(model, ji, w0):
    tag = tag_for(model, ji, w0)
    p = DATA_DIR / f"{tag}_coupled.npz"
    if not p.exists():
        return None
    with np.load(p) as z:
        return {k: z[k] for k in ("t", "S", "Wp", "E_N")}, float(z["omega0"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--J-I-afm", default="0.015")
    ap.add_argument("--J-I-fm", default="0.010")
    ap.add_argument("--component", choices=("x", "y", "z"), default="z")
    args = ap.parse_args()
    model_ji = {"AFM": f"{float(args.J_I_afm):.3f}", "FM": f"{float(args.J_I_fm):.3f}"}
    comp = args.component
    comp_idx = AXIS_INDEX[comp]
    comp_color = C_AXIS[comp]

    setup_style()
    fig, ax = plt.subplots(2, 4, figsize=(9.8, 4.4), layout="constrained")
    letters = iter("abcdefgh")
    titles = [rf"$\langle\hat{{S}}^{comp}_{{\mathrm{{tot}}}}\rangle/N$",
             r"$\langle\hat{\bar{W}}_p\rangle$",
             r"$\mathcal{N}$", r"$\mathcal{N}_{\text{GM}}^{6,6}$"]

    n_curves = 0
    for r, model in enumerate(MODELS):
        ji = model_ji[model]
        for w0 in OMEGA0_VALUES:
            r_data = load_point(model, ji, w0)
            if r_data is None:
                log(f"  [skip] {model} J_I={ji} omega0={w0}: no data yet")
                continue
            out, omega0 = r_data
            T = 2 * np.pi / omega0
            tp = out["t"] / T
            ls = FREQ_LS[w0]
            label = FREQ_LABELS[w0]
            n_curves += 1

            Scomp = out["S"].sum(axis=1)[:, comp_idx] / N
            ax[r, 0].plot(tp, Scomp, color=comp_color, ls=ls, lw=0.9, label=label)

            wp_mean = out["Wp"].mean(axis=1)
            ax[r, 1].plot(tp, wp_mean, color="black", ls=ls, lw=0.9, label=label)

            ax[r, 2].plot(tp, out["E_N"], color="black", ls=ls, lw=0.9, label=label)

        for c in range(4):
            a = ax[r, c]
            a.set_title(titles[c] if r == 0 else "")
            # Panel (a) only: shorten the first dashed (T_ON) and the dotted
            # (T_ON_KONDO) markers to 88% height so they don't run into the
            # inset J_K text / legend up top. T_OFF, and every marker in
            # every other panel, stay full height (bottom to top).
            first_ymax = 0.88 if (r == 0 and c == 0) else 1.0
            a.axvline(T_ON, color="0.4", ls="--", lw=0.8, zorder=0, ymax=first_ymax)
            a.axvline(T_OFF, color="0.4", ls="--", lw=0.8, zorder=0)
            a.axvline(T_ON_KONDO, color="0.4", ls=":", lw=0.9, zorder=0, ymax=first_ymax)
            a.margins(x=0.02, y=0.15)
            _panel_letter(a, next(letters))
            if r == 1:
                a.set_xlabel(r"Time/$T_0$ $(\gamma/\hbar)$")
        ax[r, 1].set_ylim(-0.1, 1.15)
        ax[r, 3].set_xlim(ax[r, 0].get_xlim())   # GMN panel has no data to set its own scale
        _row_bracket(ax[r, -1], rf"{model}, $J_I={float(ji):g}$")

    if n_curves == 0:
        log("no data for any (model, omega0) point yet -- nothing to plot")
        return

    ax[0, 0].text(0.04, 0.96, r"$J_K=0.05$", transform=ax[0, 0].transAxes,
                 ha="left", va="top", fontsize=8)
    ax[0, 0].legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.9)

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"dashboard_by_freq_S{comp}_AFM-jI{model_ji['AFM']}_FM-jI{model_ji['FM']}"
    path = IMG_DIR / stem
    _savefig(fig, path)
    log(f"  wrote {path}.{{pdf,png}} ({n_curves}/{len(MODELS) * len(OMEGA0_VALUES)} curves)")


if __name__ == "__main__":
    main()
