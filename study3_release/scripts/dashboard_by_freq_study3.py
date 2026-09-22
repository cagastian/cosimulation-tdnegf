#!/usr/bin/env python3
"""
dashboard_by_freq_study3.py -- AFM-vs-FM frequency-comparison dashboard for
Study 3's (fixed) T=40 runs. Same 2x4 layout and column set as Study 2's
dashboard_by_freq.py (S^alpha_tot, mean <W_p>, log-negativity N, GMN blank),
one row per model, one curve per omega0 (solid/dashed/dotted =
omega0/2*omega0/4*omega0), but:

  - x-axis is RAW absolute time (hbar/gamma), not t/T. Study 3's whole
    point (after the 2026-09-17 timing fix, see ../../CONTEXT.md "Study 3
    timing bug and fix") is that all three omega0 now share the identical
    absolute window (t_on=3141.6, t_on_kondo=5026.5, t_off=18849.6,
    t_final=25132.7, all in hbar/gamma) -- so unlike Study 2 (different
    absolute duration per omega0, hence periods-space is what makes that
    comparison fair), plotting Study 3 in raw time is what makes ITS
    comparison fair: the three omega0 curves now literally overlay the
    same physical stretch of time, just with more/fewer drive cycles
    inside it.
  - drive/Kondo markers are drawn ONCE per panel (not per omega0 curve) --
    they're the same absolute time for all three now, so one set of
    axvlines suffices (computed from the omega0=0.01 reference periods:
    t_on=5T_ref, t_on_kondo=8T_ref, t_off=30T_ref, T_ref=2*pi/0.01).
  - tags/J_I are Study 3's own (AFM J_I=0.015, FM J_I=0.0015, fixed --
    Study 3 doesn't scan J_I like Study 2 does), not run_study2.tag_for.

    python dashboard_by_freq_study3.py [--component {x,y,z}]

writes img/dashboard_by_freq_study3_S<component>.{pdf,png}.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from run_coupled import setup_style, N, _panel_letter, _savefig, log
from compare_figures import _row_bracket

DATA_DIR = HERE / "data"
IMG_DIR = HERE / "img"

OMEGA0_VALUES = ["0.01", "0.02", "0.04"]
FREQ_LS = {"0.01": "-", "0.02": "--", "0.04": ":"}
FREQ_LABELS = {"0.01": r"$\omega_0=0.01$", "0.02": r"$\omega_1=2\omega_0$",
              "0.04": r"$\omega_2=4\omega_0$"}
# Column 0 (S^alpha_tot) only -- 3 shades of red, darkest for the reference
# omega0 down to lightest for the fastest harmonic, so frequency identity
# reads in color there instead of relying on linestyle alone.
FREQ_COLOR_COL0 = {"0.01": "#7a0000", "0.02": "#c1272d", "0.04": "#f4777f"}
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
MODEL_JI = {"AFM": "0.015", "FM": "0.0015"}
MODELS = ["AFM", "FM"]

# Reference window (omega0=0.01's own periods -- all three omega0 now share
# these absolute times, see the fix writeup in ../../CONTEXT.md).
T_REF = 2 * np.pi / 0.01
T_ON, T_ON_KONDO, T_OFF = 5 * T_REF, 8 * T_REF, 30 * T_REF


def tag_for(model, w0):
    return f"{model}_jK0.05_jI{MODEL_JI[model]}_w{w0}_study3"


def load_point(model, w0):
    tag = tag_for(model, w0)
    p = DATA_DIR / f"{tag}_coupled.npz"
    if not p.exists():
        return None
    with np.load(p) as z:
        return {k: z[k] for k in ("t", "S", "Wp", "E_N")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--component", choices=("x", "y", "z"), default="z")
    args = ap.parse_args()
    comp = args.component
    comp_idx = AXIS_INDEX[comp]

    setup_style()
    fig, ax = plt.subplots(2, 4, figsize=(9.8, 4.4), layout="constrained")
    letters = iter("abcdefgh")
    titles = [rf"$\langle\hat{{S}}^{comp}_{{\mathrm{{tot}}}}\rangle/N$",
             r"$\langle\hat{\bar{W}}_p\rangle$",
             r"$\mathcal{N}$", r"$\mathcal{N}_{\text{GM}}^{6,6}$"]

    n_curves = 0
    for r, model in enumerate(MODELS):
        for w0 in OMEGA0_VALUES:
            out = load_point(model, w0)
            if out is None:
                log(f"  [skip] {model} omega0={w0}: no data yet")
                continue
            t = out["t"]
            ls = FREQ_LS[w0]
            label = FREQ_LABELS[w0]
            n_curves += 1

            Scomp = out["S"].sum(axis=1)[:, comp_idx] / N
            ax[r, 0].plot(t, Scomp, color=FREQ_COLOR_COL0[w0], ls=ls, lw=0.9, label=label)

            wp_mean = out["Wp"].mean(axis=1)
            ax[r, 1].plot(t, wp_mean, color="black", ls=ls, lw=0.9, label=label)

            ax[r, 2].plot(t, out["E_N"], color="black", ls=ls, lw=0.9, label=label)

        for c in range(4):
            a = ax[r, c]
            a.set_title(titles[c] if r == 0 else "")
            first_ymax = 0.88 if (r == 0 and c == 0) else 1.0
            a.axvline(T_ON, color="0.4", ls="--", lw=0.8, zorder=0, ymax=first_ymax)
            a.axvline(T_OFF, color="0.4", ls="--", lw=0.8, zorder=0)
            a.axvline(T_ON_KONDO, color="0.4", ls=":", lw=0.9, zorder=0, ymax=first_ymax)
            a.margins(x=0.02, y=0.15)
            _panel_letter(a, next(letters))
            if r == 1:
                a.set_xlabel(r"Time $(\hbar/\gamma)$")
        ax[r, 1].set_ylim(-0.1, 1.15)
        ax[r, 3].set_xlim(ax[r, 0].get_xlim())
        _row_bracket(ax[r, -1], rf"{model}, $J_I={float(MODEL_JI[model]):g}$")

    if n_curves == 0:
        log("no data for any (model, omega0) point yet -- nothing to plot")
        return

    ax[0, 0].text(0.04, 0.96, r"$J_K=0.05$", transform=ax[0, 0].transAxes,
                 ha="left", va="top", fontsize=8)
    ax[0, 0].legend(loc="lower center", ncol=3, fontsize=8, frameon=True,
                    framealpha=0.9)

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"dashboard_by_freq_study3_S{comp}"
    path = IMG_DIR / stem
    _savefig(fig, path)
    log(f"  wrote {path}.{{pdf,png}} ({n_curves}/{len(MODELS) * len(OMEGA0_VALUES)} curves)")


if __name__ == "__main__":
    main()
