#!/usr/bin/env python3
"""
compare_figures.py -- four curated AFM-vs-FM comparison figures.

Builds side-by-side/overlaid comparisons of the matched FM_jK0.05 /
AFM_jK0.05 kondo-ramp production runs (identical drive/Kondo-ramp protocol,
differing only in model), styled to match dashboard.py's look. Reuses
run_coupled.py's style constants and plotting helpers rather than
reinventing them.

    python compare_figures.py [--img-dir DIR]

writes (into --img-dir, default img_coupled/summary/)
compare_{dashboard,charge_and_diff,moments}.
{pdf,png} plus compare_spin_current_{precessing,static}.{pdf,png} -- the
spin-current FFT split into the driven window (Kondo-connected AND the
drive precessing) and the post-drive one (Kondo-connected AND the drive has
already gone static), since averaging both into one FFT mixes two different
regimes (see _kondo_window in run_coupled.py) -- plus
compare_spectrogram_{AFM,FM}.{pdf,png}, the continuous-time complement: a
(omega/omega0, t) heatmap per model instead of picking one window or the
other. Separate files, not one shared figure: AFM's and FM's current
magnitudes differ too much to put on one color scale (see
run_coupled.fig_spectrogram).
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator
from run_coupled import (
    setup_style, load_run, N, C_AXIS, C_LEAD, C_WP, LS_WP, LW_WP,
    TIME_LABEL, _panel_letter, _tidy, _after_transient, _connected_window,
    _kondo_window, _legend_top, fig_spectrogram,
    _savefig, log,
)

FM_TAG = "FM_jK0.05_env5-40_kondo15_julia"
AFM_TAG = "AFM_jK0.05_env5-40_kondo15_julia"
DATA_DIR = "./data_coupled"
IMG_DIR = Path("./img_coupled/summary")

GMN_DIR = Path("./GMN_calculation_res")
GMN_FILES = {"FM": GMN_DIR / "FM_GMN.txt", "AFM": GMN_DIR / "AFM_GMN.txt"}

def _row_bracket(ax_last, label, x=1.12, tick=0.03, y0=0.05, y1=0.95):
    """Right-edge bracket + rotated row label, port of dashboard.py's
    plot_dashboard_2x4_stacks row-label annotation."""
    ax_last.plot([x - tick, x, x, x - tick], [y1, y1, y0, y0],
                color="black", lw=1.0, transform=ax_last.transAxes, clip_on=False)
    ax_last.annotate(label, xy=(x + 0.04, 0.5), xycoords="axes fraction",
                     fontsize=8, fontweight="bold", rotation=-90,
                     ha="left", va="center", annotation_clip=False)


# ---------------------------------------------------------------------------
# Figure 1 -- dashboard comparison, 2x4, rows = AFM/FM, cols = S_tot/Wp/E_N/GMN
# ---------------------------------------------------------------------------

def _gmn_column(ax, out, model):
    ax.set_title(r"GMN")
    path = GMN_FILES[model]
    if not path.exists():
        return   # blank, titled panel -- matches dashboard.py's panel_GMN
                  # precedent for missing GMN data
    vals = np.loadtxt(path)
    t_gmn = out["t_gmn"]
    if len(vals) != len(t_gmn):
        log(f"  [gmn] {path.name}: {len(vals)} vals vs {len(t_gmn)} snapshot "
            f"times, skipping")
        return
    ax.plot(t_gmn, vals, ls="--", marker="o", markersize=2.2, lw=0.6,
            color="#f86503", alpha=0.9)


def fig_compare_dashboard(out_afm, meta_afm, out_fm, meta_fm, path):
    fig, ax = plt.subplots(2, 4, figsize=(7.8, 3.4), sharex=True,
                           layout="constrained")
    rows = [("AFM", out_afm, meta_afm), ("FM", out_fm, meta_fm)]
    letters = iter("abcdefgh")

    for r, (model, out, meta) in enumerate(rows):
        t, T = out["t"], 2 * np.pi / meta["omega0"]

        Stot = out["S"].sum(axis=1) / N
        for a, lbl in enumerate("xyz"):
            ax[r, 0].plot(t, Stot[:, a], color=C_AXIS[lbl], lw=0.75,
                         label=rf"$\alpha={lbl}$")
        ax[r, 0].set_title(r"$\langle\hat{S}^{\alpha}_{\mathrm{tot}}\rangle/N$")

        for p in range(out["Wp"].shape[1]):
            ax[r, 1].plot(t, out["Wp"][:, p], color=C_WP[p % len(C_WP)],
                         ls=LS_WP[p % len(LS_WP)], lw=LW_WP[p % len(LW_WP)],
                         label=rf"$p={p}$")
        ax[r, 1].set_ylim(-0.1, 1.15)
        ax[r, 1].set_title(r"$\langle\hat{W}_p\rangle$")

        ax[r, 2].plot(t, out["E_N"], color="black", lw=1.0)
        ax[r, 2].set_title(r"$\mathcal{N}$")

        _gmn_column(ax[r, 3], out, model)

        for c in range(4):
            _tidy(ax[r, c], t, T, meta, letter=next(letters))
            if r == 1:
                ax[r, c].set_xlabel(TIME_LABEL)
                ax[r, c].xaxis.set_major_locator(MaxNLocator(nbins=4))
                ax[r, c].set_title("")

        _row_bracket(ax[r, -1], model)

    _legend_top(ax[0, 0], ncol=3)
    _legend_top(ax[0, 1], ncol=3)
    _savefig(fig, path)


# ---------------------------------------------------------------------------
# Figure 2 -- spin current + spectrum, 2x2, cols = AFM/FM
# ---------------------------------------------------------------------------

LETTER_XY_INSET = (0.04, 0.88)   # top-left, inside the axes (not above the frame)

# z routinely swamps x/y when all three overlap on one axis (it's plotted
# last in the "xyz" loop, so it paints over them); sending it behind gives
# x/y drawing priority without changing legend order.
Z_XYZ = {"x": 2, "y": 2, "z": 1}


def _force_sci_notation(ax):
    """Force a ×10^n offset on the y-axis regardless of magnitude, same
    mechanic fig_wp_current uses for its accent axis."""
    fmt = ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(fmt)
    off = ax.yaxis.get_offset_text()
    off.set_ha("right")
    off.set_position((1.0, 1.0))
    off.set_size(8)


def fig_compare_spin_current(out_afm, meta_afm, out_fm, meta_fm, path,
                             phase="precessing"):
    """
    Row 0 is always the full time series (context, with the usual drive/kondo
    markers). Row 1 -- the FFT -- is NOT: Kondo-connected time splits into a
    driven regime (drive precessing) and a relaxing one (drive gone static,
    field held wherever it froze), and averaging both into one FFT window
    mixes them. `phase` picks which one via _kondo_window; call this twice
    (see __main__) to get both as separate figures.
    """
    # sharex="row" only -- NOT sharey: AFM's and FM's I^S live on very
    # different scales, so each column keeps its own independent y-axis
    # (and its own forced scientific-notation offset).
    fig, ax = plt.subplots(2, 2, figsize=(7.0, 3.6), sharex="row",
                           layout="constrained")
    cols = [("AFM", out_afm, meta_afm), ("FM", out_fm, meta_fm)]
    letters = iter("abcd")

    for c, (model, out, meta) in enumerate(cols):
        t, T = out["t"], 2 * np.pi / meta["omega0"]
        i0 = _after_transient(t)
        for a, al in enumerate("xyz"):
            ax[0, c].plot(t[i0:], out["Is"][i0:, 0, a], color=C_AXIS[al],
                         lw=0.8, label=rf"$\alpha={al}$", zorder=Z_XYZ[al])
        ax[0, c].set_title(model)
        _force_sci_notation(ax[0, c])
        _tidy(ax[0, c], t, T, meta, letter=None, nbins=4)
        _panel_letter(ax[0, c], next(letters), xy=LETTER_XY_INSET)
        ax[0, c].set_xlabel(TIME_LABEL)

    ax[0, 0].set_ylabel(r"$I^{S_\alpha}_L$")
    _legend_top(ax[0, 0], ncol=3)

    for c, (model, out, meta) in enumerate(cols):
        w0 = meta["omega0"]
        win = _kondo_window(out, meta, phase=phase)
        if win is None:
            ax[1, c].text(0.5, 0.5, f"no '{phase}' window\n(Kondo-connected)",
                          ha="center", va="center", transform=ax[1, c].transAxes,
                          fontsize=9)
            ax[1, c].set_xticks([]); ax[1, c].set_yticks([])
            _panel_letter(ax[1, c], next(letters), xy=LETTER_XY_INSET)
            continue
        i0, i1 = win
        tt, dt = out["t"][i0:i1 + 1], out["t"][1] - out["t"][0]
        freq = np.fft.rfftfreq(len(tt), d=dt) * 2 * np.pi / w0
        peak = 0.0
        As = []
        for a, al in enumerate("xyz"):
            y = out["Is"][i0:i1 + 1, 0, a] - out["Is"][i0:i1 + 1, 0, a].mean()
            A = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
            peak = max(peak, A.max())
            As.append(A)
        for a, al in enumerate("xyz"):
            ax[1, c].semilogy(freq, As[a], color=C_AXIS[al], lw=0.8,
                              label=rf"$\alpha={al}$", zorder=Z_XYZ[al])
        ax[1, c].set_xlim(0, 8)
        ax[1, c].set_ylim(peak * 1e-6, peak * 3)   # independent per column
        for h in range(1, 9):
            ax[1, c].axvline(h, color="0.85", ls=":", lw=0.6, zorder=0)
        ax[1, c].set_xlabel(r"$\omega/\omega_0$")
        ax[1, c].yaxis.set_minor_locator(ticker.NullLocator())
        _panel_letter(ax[1, c], next(letters), xy=LETTER_XY_INSET)

    ax[1, 0].set_ylabel(r"$|I^{S_\alpha}_L(\omega)|^2$")
    ax[1, 0].legend(loc="upper right", ncol=1, fontsize=8)

    fig.suptitle("Kondo-connected, drive precessing" if phase == "precessing"
                else "Kondo-connected, drive static (post-precession)",
                fontsize=9, y=1.04)
    _savefig(fig, path)


# ---------------------------------------------------------------------------
# Figure 3 -- charge current + spin-current lead difference, 2x2, rows =
# I_q/diff, cols = AFM/FM. Model is now spatial (column), so q and alpha go
# back to their plain C_LEAD/C_AXIS encodings -- no dual per-panel legend
# needed. Same skeleton as fig_compare_moments: row's quantity is the
# column-0 ylabel, row 0's title is the model name, only the last row (time
# is shared across both) carries xticks/xlabel.
# ---------------------------------------------------------------------------

def fig_compare_charge_and_diff(out_afm, meta_afm, out_fm, meta_fm, path):
    fig, ax = plt.subplots(2, 2, figsize=(7.0, 4.4), sharex=True,
                           layout="constrained")
    cols = [("AFM", out_afm, meta_afm), ("FM", out_fm, meta_fm)]

    for c, (model, out, meta) in enumerate(cols):
        i0 = _after_transient(out["t"])
        for q, (lbl, ls) in enumerate((("L", "-"), ("R", (0, (4, 2))))):
            ax[0, c].plot(out["t"][i0:], out["Ic"][i0:, q], color=C_LEAD[lbl],
                         lw=0.8, ls=ls, label=rf"$q={lbl}$")

        dIs = out["Is"][:, 0, :] - out["Is"][:, 1, :]
        for a, al in enumerate("xyz"):
            ax[1, c].plot(out["t"][i0:], dIs[i0:, a], color=C_AXIS[al], lw=0.8,
                         label=rf"$\alpha={al}$", zorder=Z_XYZ[al])

    T = 2 * np.pi / meta_afm["omega0"]
    letters = iter("abcd")
    for r in range(2):
        for c, (model, out, meta) in enumerate(cols):
            _tidy(ax[r, c], out["t"], T, meta, letter=next(letters))
            ax[r, c].set_title(model if r == 0 else "")

    ax[0, 0].set_ylabel(r"$I_q$")
    ax[1, 0].set_ylabel(r"$\Delta I^{S_\alpha} = I_L^{S_\alpha}-I_R^{S_\alpha}$")

    for c in range(2):
        ax[0, c].tick_params(labelbottom=False)
        ax[1, c].set_xlabel(TIME_LABEL)

    _legend_top(ax[0, 0], ncol=2)
    _legend_top(ax[1, 0], ncol=3)

    _savefig(fig, path)


# ---------------------------------------------------------------------------
# Figure 4 -- QSL vs NM moments, 2x2, rows = S_tot/sigma, cols = AFM/FM.
# AFM's and FM's moments live on very different scales (overlaying them on
# one axis flattens AFM to near-invisible), so each column keeps its own
# independent y-axis; the row's quantity name moves to the (column-0-only)
# ylabel, and the column header (AFM/FM) becomes the (row-0-only) title.
# ---------------------------------------------------------------------------

def fig_compare_moments(out_afm, meta_afm, out_fm, meta_fm, path):
    fig, ax = plt.subplots(2, 2, figsize=(7.0, 3.6), sharex=True,
                           layout="constrained")
    cols = [("AFM", out_afm, meta_afm), ("FM", out_fm, meta_fm)]

    for c, (model, out, meta) in enumerate(cols):
        Stot = out["S"].sum(axis=1) / N
        for a, al in enumerate("xyz"):
            ax[0, c].plot(out["t"], Stot[:, a], color=C_AXIS[al], lw=0.75,
                         label=rf"$\alpha={al}$")

        sig_tot = out["sigma"].sum(axis=1) / N
        for a, al in enumerate("xyz"):
            ax[1, c].plot(out["t"], sig_tot[:, a], color=C_AXIS[al], lw=0.75)

    T = 2 * np.pi / meta_afm["omega0"]
    letters = iter("abcd")
    for r in range(2):
        for c, (model, out, meta) in enumerate(cols):
            _tidy(ax[r, c], out["t"], T, meta, letter=next(letters))
            ax[r, c].set_title(model if r == 0 else "")

    ax[0, 0].set_ylabel(r"$\langle\hat{S}^{\alpha}_{\mathrm{tot}}\rangle/N$")
    ax[1, 0].set_ylabel(r"$\langle\hat{\sigma}^{\alpha}\rangle_{\mathrm{NM}}/N$")

    for c in range(2):
        ax[0, c].tick_params(labelbottom=False)
        ax[1, c].set_xlabel(TIME_LABEL)

    _legend_top(ax[0, 0], ncol=3)

    _savefig(fig, path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--img-dir", default=str(IMG_DIR),
                   help=f"output directory (default {IMG_DIR})")
    args = p.parse_args()
    IMG_DIR = Path(args.img_dir)

    setup_style()
    out_afm, meta_afm = load_run(AFM_TAG, DATA_DIR)
    out_fm, meta_fm = load_run(FM_TAG, DATA_DIR)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    fig_compare_dashboard(out_afm, meta_afm, out_fm, meta_fm,
                          IMG_DIR / "compare_dashboard")
    log(f"  wrote {IMG_DIR / 'compare_dashboard'}.{{pdf,png}}")
    for phase in ("precessing", "static"):
        stem = f"compare_spin_current_{phase}"
        fig_compare_spin_current(out_afm, meta_afm, out_fm, meta_fm,
                                 IMG_DIR / stem, phase=phase)
        log(f"  wrote {IMG_DIR / stem}.{{pdf,png}}")
    for model, out, meta in (("AFM", out_afm, meta_afm), ("FM", out_fm, meta_fm)):
        stem = f"compare_spectrogram_{model}"
        fig_spectrogram(out, meta, IMG_DIR / stem)
        log(f"  wrote {IMG_DIR / stem}.{{pdf,png}}")
    fig_compare_charge_and_diff(out_afm, meta_afm, out_fm, meta_fm,
                                IMG_DIR / "compare_charge_and_diff")
    log(f"  wrote {IMG_DIR / 'compare_charge_and_diff'}.{{pdf,png}}")
    fig_compare_moments(out_afm, meta_afm, out_fm, meta_fm,
                        IMG_DIR / "compare_moments")
    log(f"  wrote {IMG_DIR / 'compare_moments'}.{{pdf,png}}")
