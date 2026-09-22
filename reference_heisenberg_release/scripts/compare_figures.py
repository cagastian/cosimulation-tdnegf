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
compare_{dashboard,charge_and_diff,moments}.{pdf,png} plus
compare_spin_current.{pdf,png} -- N rows (one per model) x 4 columns: col 0
the full I^Salpha time series, cols 1-3 (omega/omega0, t) spectrogram
heatmaps per alpha=x,y,z, replacing what used to be a static FFT snapshot
(see run_coupled._spectrogram_specs and fig_compare_spin_current) -- plus
compare_spectrogram_{AFM,FM}.{pdf,png}, the same heatmap machinery as a
standalone, full-size, w_max-adjustable figure per model. Each model gets
its own color normalization/colorbar throughout: current magnitudes differ
by orders of magnitude across models, so one shared scale would flatten the
smaller ones. If the Heisenberg run's data is present, also writes
spin_current_heisenberg.{pdf,png} (Heisenberg alone) and
compare_spin_current_3model.{pdf,png} (AFM+FM+Heisenberg together, w_max=8).
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap
from run_coupled import (
    setup_style, load_run, N, C_AXIS, C_LEAD, C_WP, LS_WP, LW_WP,
    TIME_LABEL, _panel_letter, _tidy, _after_transient, _legend_top,
    fig_spectrogram, _spectrogram_specs, _plot_spectrogram_panel,
    _savefig, log,
)

# Per-alpha monochrome heatmap colormaps (black -> C_AXIS[alpha] -> white),
# used by fig_compare_spin_current so each spectrogram row's own color
# matches its trace's color in row 0's legend -- lets the per-panel alpha
# titles be dropped without losing that information.
ALPHA_CMAP = {al: LinearSegmentedColormap.from_list(f"cmap_{al}",
                                                     ["black", C_AXIS[al], "white"])
             for al in "xyz"}

FM_TAG = "FM_jK0.05_env5-40_kondo15_julia"
AFM_TAG = "AFM_jK0.05_env5-40_kondo15_julia"
HEISENBERG_TAG = "FM_heisenberg_jK0.05_theta_env0-5.2_kondo0.2_T10"
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


def _cap_drive_marks(ax, y_cap):
    """
    _tidy draws the drive/Kondo axvlines (color '0.4') full-height in axes
    fraction (0-1), which auto-stretches as _legend_top later reserves space
    above the data -- the lines end up running straight through the legend
    text. Redraw them capped at y_cap (data coords, from the axis bottom)
    instead. Call after _legend_top has finalized the y-limits.
    """
    marks = [l for l in ax.lines if l.get_color() == "0.4"]
    info = [(l.get_xdata()[0], l.get_linestyle(), l.get_linewidth()) for l in marks]
    for l in marks:
        l.remove()
    y0 = ax.get_ylim()[0]
    for x, ls, lw in info:
        ax.vlines(x, y0, y_cap, color="0.4", ls=ls, lw=lw, alpha=0.85, zorder=0)


def fig_compare_spin_current(models, path, w_max=20.0):
    """
    Paper figure, N rows (one per model) x 4 columns: col 0 is the full
    I^Salpha time series (context, with the usual drive/Kondo markers),
    cols 1-3 are (omega/omega0, t) spectrogram heatmaps, one per
    alpha=x,y,z -- the continuous-time replacement for a static FFT:
    instead of picking a precessing-only or static-only window (the old
    two-figure split), the heatmap shows the whole precessing -> static
    transition at once. Reuses run_coupled._spectrogram_specs/
    _plot_spectrogram_panel, the same machinery fig_spectrogram uses.
    `models` is a list of (label, out, meta) tuples, in row order -- 1 row
    for a single model (e.g. Heisenberg alone), 2 for an AFM/FM pair, or
    more (e.g. AFM+FM+Heisenberg together) for a combined comparison. Model
    as row (not column) so the omega/omega0 axis can be shared across a
    row's 3 adjacent spectrogram panels -- only the leftmost spectrogram
    column (col 1) keeps omega/omega0 tick labels/ylabel, cols 2-3 just
    show the heatmap (the range is always the fixed [0, w_max] regardless
    of model, so nothing is lost). Each ROW still gets its own color
    normalization and colorbar: current magnitudes can differ by orders of
    magnitude across models, so one shared scale would flatten the smaller
    ones (same reasoning as col 0's independent y-axes). x-axis (time) is
    shared -- and non-bottom tick labels hidden -- only WITHIN groups of
    rows whose total run duration matches (see the t_max grouping below);
    a model run with a different (e.g. much shorter) protocol keeps its own
    independent x-axis and its own tick labels/xlabel instead of being
    squashed into a sliver by a longer run's range. When every model shares
    one duration (the common case -- an AFM/FM pair), this reduces to a
    single shared x-axis with only the bottom row carrying tick
    labels/xlabel, same as before.
    """
    n = len(models)
    with plt.rc_context({"font.size": 13, "axes.labelsize": 13,
                        "axes.titlesize": 13, "xtick.labelsize": 11,
                        "ytick.labelsize": 11}):
        fig, ax = plt.subplots(n, 4, figsize=(10.2, 1.9 * n + 0.5),
                               squeeze=False, layout="constrained")
        letters = iter("abcdefghijklmnop")

        # Group rows by matching total run duration (t[-1], 1% relative
        # tolerance) -- only rows within the same group get their x-axes
        # linked and their non-bottom tick labels hidden.
        t_max = [out["t"][-1] for _, out, _ in models]
        groups = []
        for r, tm in enumerate(t_max):
            for g in groups:
                if abs(tm - t_max[g[0]]) / max(tm, t_max[g[0]]) < 0.01:
                    g.append(r)
                    break
            else:
                groups.append([r])
        for c in range(4):
            for g in groups:
                for r in g[1:]:
                    ax[r, c].sharex(ax[g[0], c])
        bottom_of_group = {r: (r == max(g)) for g in groups for r in g}

        for r, (model, out, meta) in enumerate(models):
            t, T = out["t"], 2 * np.pi / meta["omega0"]
            i0 = _after_transient(t)
            for a, al in enumerate("xyz"):
                ax[r, 0].plot(t[i0:], out["Is"][i0:, 0, a], color=C_AXIS[al],
                             lw=0.8, label=rf"$\alpha={al}$", zorder=Z_XYZ[al])
            ax[r, 0].set_title(model)
            ax[r, 0].set_ylabel(r"$I^{S_\alpha}_L$")
            _force_sci_notation(ax[r, 0])
            _tidy(ax[r, 0], t, T, meta, letter=None, nbins=4)
            _panel_letter(ax[r, 0], next(letters), xy=(0.01, 1.05), fontsize=11)

        y_cap = ax[0, 0].get_ylim()[1]   # natural data top, before _legend_top's pad
        _legend_top(ax[0, 0], ncol=3, fontsize=11, handlelength=0.6)
        # Row 0 only: it's the one with the legend to dodge. The other rows
        # never had a legend, and their axis autoscale is still live at this
        # point (_legend_top -- which fixes ylim, disabling autoscale -- was
        # only called on row 0) -- capping them too would send their vlines'
        # y_cap into that row's autoscale as real data and blow its y-range
        # out, squashing its actual signal flat. See _cap_drive_marks's
        # docstring.
        _cap_drive_marks(ax[0, 0], y_cap=y_cap)

        for r, (model, out, meta) in enumerate(models):
            specs, norm, T = _spectrogram_specs(out, meta, w_max=w_max)
            im = None
            # specs is always [x, y, z] (panels order in _spectrogram_specs) --
            # titles are dropped in favor of the per-alpha cmap below, which
            # already encodes the alpha via col 0's legend colors.
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
                if bottom_of_group[r]:
                    ax[r, c].set_xlabel(TIME_LABEL)
                else:
                    ax[r, c].tick_params(labelbottom=False)

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
    fig_compare_spin_current([("AFM", out_afm, meta_afm), ("FM", out_fm, meta_fm)],
                             IMG_DIR / "compare_spin_current")
    log(f"  wrote {IMG_DIR / 'compare_spin_current'}.{{pdf,png}}")
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

    if (Path(DATA_DIR) / f"{HEISENBERG_TAG}_coupled.npz").exists():
        out_hb, meta_hb = load_run(HEISENBERG_TAG, DATA_DIR)
        fig_compare_spin_current([("Heisenberg (FM)", out_hb, meta_hb)],
                                 IMG_DIR / "spin_current_heisenberg")
        log(f"  wrote {IMG_DIR / 'spin_current_heisenberg'}.{{pdf,png}}")

        fig_compare_spin_current([("AFM", out_afm, meta_afm), ("FM", out_fm, meta_fm),
                                  ("Heisenberg (FM)", out_hb, meta_hb)],
                                 IMG_DIR / "compare_spin_current_3model", w_max=8.0)
        log(f"  wrote {IMG_DIR / 'compare_spin_current_3model'}.{{pdf,png}}")
