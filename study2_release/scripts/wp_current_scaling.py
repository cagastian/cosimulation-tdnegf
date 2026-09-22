#!/usr/bin/env python3
"""
wp_current_scaling.py -- quantitatively relate <W_p>'s decay rate to the
pumped spin current, across the completed J_K scan (J_I=0.0025 fixed) and
J_I scan (J_K=0.02 fixed). Both scans share one anchor point
(J_K=0.02, J_I=0.0025) -- literally the same underlying run
(AFM_jK0.02_theta_env0-5.2_kondo0.2_T10_scan), loaded into both sweeps' rows.

Per run: a decay rate dWp/dt (least-squares slope of <W_p> over the
Kondo-connected-and-precessing window) and two current summary stats (peak
|I^Salpha_L| and its FFT power at omega0/2*omega0), all restricted to that
same window so every number describes the same stretch of time.

Method for the I ~ J_K^a * J_I^b scaling-collapse attempt: with only two 1D
sweeps, (a, b) are NOT jointly determined by one 2D regression on the
combined points (poorly conditioned). Instead fit `a` from the J_K sweep's
own log-log slope and `b` from the J_I sweep's own log-log slope, each
normalized through the shared anchor point -- see fit_power_laws().

    python wp_current_scaling.py

writes img_coupled/summary/wp_current_scaling.csv and
img_coupled/summary/wp_current_scaling_{raw,collapse}.{pdf,png}
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from run_coupled import load_run, _kondo_window, setup_style, _savefig, log

DATA_DIR = Path("./data_coupled")
IMG_DIR = Path("./img_coupled/summary")

JK_SWEEP = ["0.02", "0.04", "0.06", "0.08", "0.09", "0.10", "0.11", "0.12"]   # J_I=0.0025 fixed
JK_TAG_FMT = "AFM_jK{jk}_theta_env0-5.2_kondo0.2_T10_scan"

JI_SWEEP = ["0.001", "0.0025", "0.005", "0.0075", "0.01", "0.02"]            # J_K=0.02 fixed
JI_ANCHOR = "0.0025"
ANCHOR_TAG = "AFM_jK0.02_theta_env0-5.2_kondo0.2_T10_scan"   # shared with the J_K sweep

J_K_FIXED_FOR_JI_SWEEP = 0.02
J_I_FIXED_FOR_JK_SWEEP = 0.0025


def ji_tag(ji_label):
    return ANCHOR_TAG if ji_label == JI_ANCHOR else f"AFM_jK0.02_jI{ji_label}_theta_env0-5.2_kondo0.2_T10_scan"


def decay_rate(out, meta):
    """Least-squares slope of <W_p>(t) (averaged over the 3 plaquettes) over
    the Kondo-connected-and-precessing window."""
    win = _kondo_window(out, meta, phase="precessing")
    if win is None:
        return np.nan
    i0, i1 = win
    t = out["t"][i0:i1 + 1]
    wp = out["Wp"][i0:i1 + 1].mean(axis=1)
    if len(t) < 2:
        return np.nan
    slope, _ = np.polyfit(t, wp, 1)
    return slope


def current_stats(out, meta):
    """Peak |I^S_alpha_L| and FFT power at omega0/2*omega0, all restricted to
    the same connected-and-precessing window decay_rate uses -- same
    rfft/Hann-window construction as fig_spectrum/fig_compare_spin_current."""
    win = _kondo_window(out, meta, phase="precessing")
    if win is None:
        return dict(I_peak=np.nan, I_fft_w0=np.nan, I_fft_2w0=np.nan)
    i0, i1 = win
    Is = out["Is"][i0:i1 + 1, 0, :]           # (n_t, 3) -- alpha = x,y,z, lead L
    I_peak = float(np.abs(Is).max())

    t = out["t"][i0:i1 + 1]
    w0 = meta["omega0"]
    if len(t) < 8:
        return dict(I_peak=I_peak, I_fft_w0=np.nan, I_fft_2w0=np.nan)
    dt = t[1] - t[0]
    freq = np.fft.rfftfreq(len(t), d=dt) * 2 * np.pi / w0   # in units of omega0

    power_at = {1.0: 0.0, 2.0: 0.0}
    for a in range(3):
        y = Is[:, a] - Is[:, a].mean()
        A = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
        for target in power_at:
            idx = np.argmin(np.abs(freq - target))
            power_at[target] = max(power_at[target], A[idx])

    return dict(I_peak=I_peak, I_fft_w0=power_at[1.0], I_fft_2w0=power_at[2.0])


def build_dataframe():
    rows = []
    for jk in JK_SWEEP:
        tag = JK_TAG_FMT.format(jk=jk)
        if not (DATA_DIR / f"{tag}_coupled.npz").exists():
            log(f"  [skip] J_K sweep, J_K={jk}: no finished data")
            continue
        out, meta = load_run(tag, str(DATA_DIR))
        rows.append(dict(sweep="J_K", J_K=float(jk), J_I=J_I_FIXED_FOR_JK_SWEEP,
                          tag=tag, dWp_dt=decay_rate(out, meta), **current_stats(out, meta)))

    for ji in JI_SWEEP:
        tag = ji_tag(ji)
        if not (DATA_DIR / f"{tag}_coupled.npz").exists():
            log(f"  [skip] J_I sweep, J_I={ji}: no finished data")
            continue
        out, meta = load_run(tag, str(DATA_DIR))
        rows.append(dict(sweep="J_I", J_K=J_K_FIXED_FOR_JI_SWEEP, J_I=float(ji),
                          tag=tag, dWp_dt=decay_rate(out, meta), **current_stats(out, meta)))

    return pd.DataFrame(rows)


def fit_power_laws(df, stat="I_fft_w0"):
    """a from the J_K sweep's log-log slope, b from the J_I sweep's,
    normalized through the shared anchor point. Returns (a, b, I_anchor,
    JK_anchor, JI_anchor) or None if either sweep has <2 usable points."""
    anchor = df[df["tag"] == ANCHOR_TAG]
    if anchor.empty:
        log("  [fit] no anchor row found -- can't normalize the two slopes")
        return None
    JK_anchor = float(anchor["J_K"].iloc[0])
    JI_anchor = float(anchor["J_I"].iloc[0])
    I_anchor = float(anchor[stat].iloc[0])
    if not (I_anchor > 0):
        log(f"  [fit] anchor {stat} is not positive -- can't fit a log-log slope")
        return None

    def slope(sub, xcol):
        sub = sub[(sub[stat] > 0) & (sub[xcol] > 0)]
        if len(sub) < 2:
            return None
        x, y = np.log(sub[xcol].values), np.log(sub[stat].values)
        return np.polyfit(x, y, 1)[0]

    a = slope(df[df["sweep"] == "J_K"], "J_K")
    b = slope(df[df["sweep"] == "J_I"], "J_I")
    if a is None or b is None:
        log("  [fit] not enough positive points in one sweep to fit a slope")
        return None
    return a, b, I_anchor, JK_anchor, JI_anchor


def main():
    setup_style()
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    df = build_dataframe()
    if df.empty:
        log("no data at all -- nothing to compute")
        return
    df.to_csv(IMG_DIR / "wp_current_scaling.csv", index=False)
    log(f"  wrote {IMG_DIR / 'wp_current_scaling.csv'}  ({len(df)} rows)")

    markers = {"J_K": "o", "J_I": "s"}
    colors = {"J_K": "tab:blue", "J_I": "tab:orange"}

    # --- fig 1: raw current stat vs dWp/dt, no fit -------------------------
    fig, ax = plt.subplots(figsize=(4.6, 4.0), layout="constrained")
    for sweep, sub in df.groupby("sweep"):
        ax.scatter(np.abs(sub["dWp_dt"]), sub["I_fft_w0"], marker=markers[sweep],
                  color=colors[sweep], label=f"{sweep} swept", zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$|d\langle\hat{W}_p\rangle/dt|$")
    ax.set_ylabel(r"FFT power at $\omega_0$")
    ax.legend()
    _savefig(fig, IMG_DIR / "wp_current_scaling_raw")
    log(f"  wrote {IMG_DIR / 'wp_current_scaling_raw'}.{{pdf,png}}")

    # --- fig 2: scaling-collapse against I_pred = J_K^a * J_I^b -----------
    fit = fit_power_laws(df, stat="I_fft_w0")
    if fit is None:
        log("  [collapse] skipped -- see [fit] message above")
        return
    a, b, I_anchor, JK_anchor, JI_anchor = fit
    log(f"  fitted a={a:.3f} (J_K slope), b={b:.3f} (J_I slope), "
        f"anchor J_K={JK_anchor:g} J_I={JI_anchor:g} I_fft_w0={I_anchor:.3e}")

    df["I_pred"] = I_anchor * (df["J_K"] / JK_anchor) ** a * (df["J_I"] / JI_anchor) ** b

    fig, ax = plt.subplots(figsize=(4.6, 4.0), layout="constrained")
    for sweep, sub in df.groupby("sweep"):
        ax.scatter(sub["I_pred"], np.abs(sub["dWp_dt"]), marker=markers[sweep],
                  color=colors[sweep], label=f"{sweep} swept", zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(rf"$I_\mathrm{{pred}} = J_K^{{{a:.2f}}} J_I^{{{b:.2f}}}$ (normalized at anchor)")
    ax.set_ylabel(r"$|d\langle\hat{W}_p\rangle/dt|$")
    ax.legend()
    _savefig(fig, IMG_DIR / "wp_current_scaling_collapse")
    log(f"  wrote {IMG_DIR / 'wp_current_scaling_collapse'}.{{pdf,png}}")


if __name__ == "__main__":
    main()
