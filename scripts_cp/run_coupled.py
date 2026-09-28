#!/usr/bin/env python3
"""
run_coupled.py -- self-consistent FI / QSL / NM coupled dynamics, cluster runner.

    H_QSL(t) = J_a H_Kitaev + J_I sum_i S_i.M(t) + s*J_K sum_i S_i.<sigma_i>(t)
    H_TB(t)  = -gamma sum_<ij> c+c        + s*J_K sum_i c+_i sigma c_i.<S_i>(t)

Both S_i and sigma_i are PAULI vectors (|.| <= 1): the draft defines
S_i = I (x) .. (x) sigma (x) .. (x) I, and TDNEGF's spin_density returns
Tr[rho_loc sigma].  No factor of 2 anywhere.  If you move to S = sigma/2,
halve J_I and J_K.

SIGN: the draft has +J_K in Eq.(1) and -J_K in Eq.(2).  A mean-field
decoupling of one J_K sum_i S_i.s_i gives both the SAME sign.  Default here is
both-minus (--kondo-sign -1); use +1 to reproduce the draft verbatim.

Units: gamma = 1, hbar = 1, time in hbar/gamma.

WHAT GETS SAVED
---------------
Everything the figures are built from lands in <out-dir>/<tag>_coupled.npz, so
the plots can be restyled later without re-running:

    t, S, sigma, Wp, Ic, Is, E      as before
    E_N                             bipartite log-negativity, hexagon vs rest
    M, theta_t                      the FI drive vector and cone angle
    rho_gmn, t_gmn, gmn_sites       reduced 6-spin rho at the GMN target times
    (optional) <tag>_states.npz     psi(t), strided -- for anything else later

    python run_coupled.py --replot --tag FM_jK0.05 --out-dir ./data_coupled

regenerates every figure from that file alone.  --export-gmn writes the
_r.txt/_i.txt pairs GMN.jl reads (see export_rho_for_gmn).

Examples
--------
    # smoke test, no Julia needed, ~1 min
    python run_coupled.py --backend mock --n-periods 1 --dt 1.0 --tag smoke

    # production, FM, 10 drive periods
    python run_coupled.py --backend julia --jl-dir ./jl --project ~/TDNEGF \
        --model FM --J-K 0.05 --n-periods 10 --dt 0.5 --tag FM_jK0.05

    # with smooth switch-on/off (drive on for periods 1..6)
    python run_coupled.py --backend mock --n-periods 9 --envelope \
        --t-on-periods 1 --t-off-periods 6 --tag envelope_test

    # dt convergence check
    for d in 2.0 1.0 0.5; do
        python run_coupled.py --backend mock --dt $d --n-periods 1 --tag dt$d
    done

Slurm sketch
------------
    #SBATCH -c 8
    #SBATCH --mem=16G
    export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
    python run_coupled.py --backend julia --jl-dir ./jl --model AFM --tag $SLURM_JOB_ID
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                     # no display on compute nodes
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LogNorm
import numpy as np
import qutip as qt
from scipy.sparse.linalg import expm_multiply
from scipy.signal import spectrogram

N = 14
PAULI = {"x": qt.sigmax(), "y": qt.sigmay(), "z": qt.sigmaz()}

# Hexagon p=1 of Fig. 1 -- the loopy subregion the draft computes GMN on.
HEX_SITES = (2, 3, 4, 9, 10, 11)

BONDS = [(0, 1, "x"), (2, 3, "x"), (4, 5, "x"), (7, 8, "x"), (9, 10, "x"), (11, 12, "x"),
         (1, 2, "y"), (3, 4, "y"), (5, 6, "y"), (8, 9, "y"), (10, 11, "y"), (12, 13, "y"),
         (0, 13, "z"), (2, 11, "z"), (4, 9, "z"), (6, 7, "z")]


def log(*a):
    print(*a, flush=True)                 # flush: slurm buffers stdout hard


# ---------------------------------------------------------------------------
# paper style
# ---------------------------------------------------------------------------
#
# Computer Modern, matching dashboard.py.  The OTFs live in
# ~/.local/share/fonts/cm-unicode (user-level, no sudo); if they are missing
# matplotlib silently falls back to DejaVu and the figures quietly stop being
# Computer Modern -- so we check and say so instead of failing silently.
#
# To (re)install without sudo:
#     curl -L -o /tmp/cmu.zip https://mirrors.ctan.org/fonts/cm-unicode.zip
#     mkdir -p ~/.local/share/fonts/cm-unicode
#     unzip -j /tmp/cmu.zip 'cm-unicode/fonts/otf/*.otf' \
#           -d ~/.local/share/fonts/cm-unicode
#     fc-cache -f ~/.local/share/fonts
#     rm -f ~/.cache/matplotlib/fontlist-*.json

SERIF_STACK = ["CMU Serif", "STIX Two Text", "DejaVu Serif"]

# Computer Modern has no U+210F: neither matplotlib's bundled cm*.ttf nor the
# CMU Serif OTFs contain it, so \hbar is the ONE glyph mathtext borrows from
# STIXGeneral -- and it sits in the x-label of every figure.  Everything else
# (including \langle, \rangle, \mathcal) is genuine CM.  Set HBAR = r"\bar{h}"
# for a figure that is 100% Computer Modern.
HBAR = r"\hbar"
TIME_LABEL = rf"Time  $({HBAR}/\gamma)$"

PAPER_RC = {
    "font.family": "serif",
    "font.serif": SERIF_STACK,
    "mathtext.fontset": "cm",
    "mathtext.fallback": "stix",          # deliberate, and only for \hbar
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "lines.linewidth": 0.9,
    "axes.linewidth": 0.7,
    "axes.formatter.use_mathtext": True,
    "axes.formatter.useoffset": True,
    "axes.unicode_minus": True,           # CMU Serif has U+2212; keep real minus
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "legend.frameon": False,
    "legend.handlelength": 0.9,
    "legend.labelspacing": 0.2,
    "legend.columnspacing": 0.6,
    "legend.handletextpad": 0.4,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.transparent": False,
    "pdf.fonttype": 42,                   # embed real glyphs, not Type-3
    "ps.fonttype": 42,
}

# component colours, same assignment as dashboard.py (x blue, y green, z red)
C_AXIS = {"x": "tab:blue", "y": "tab:green", "z": "tab:red"}
C_LEAD = {"L": "black", "R": "tab:red"}
C_WP = ["gray", "darkorange", "black"]
LS_WP = ["-", "-", ":"]
LW_WP = [1.0, 1.0, 1.2]


def setup_style(verbose=True):
    """Apply PAPER_RC and report which serif we actually got."""
    plt.rcParams.update(PAPER_RC)
    # fontTools chatters on every PDF write about the CM fonts' odd head
    # timestamps and their private "TeX" table (which it drops -- the font
    # itself is still embedded and subset). Cosmetic; keep the log readable.
    import logging
    logging.getLogger("fontTools").setLevel(logging.ERROR)
    if verbose:
        from matplotlib.font_manager import FontProperties, findfont
        got = findfont(FontProperties(family=SERIF_STACK), fallback_to_default=True)
        name = Path(got).stem
        if "cmun" not in name.lower():
            log(f"  [style] WARNING: Computer Modern not found, using {name}. "
                f"See the install note at the top of run_coupled.py.")
        elif verbose == "loud":
            log(f"  [style] serif = {name}")


def _panel_letter(ax, letter, xy=(-0.02, 1.02)):
    """
    Above the frame, top left.  Inside the axes it collides with either the
    data or the legend strip depending on the run; outside it never does, and
    the centred column titles leave the corner free.
    """
    ax.text(*xy, f"({letter})", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9, fontweight="bold")


def _drive_marks(ax, meta, T):
    """Dashed rules where the FI drive starts/stops precessing; dotted where
    J_K connects (see make_kondo_ramp)."""
    if meta.get("envelope"):
        ax.axvline(meta.get("t_on_periods", 0.0) * T,
                  color="0.4", ls="--", lw=0.8, zorder=0)
        t_off_periods = meta.get("t_off_periods", np.inf)
        if np.isfinite(t_off_periods):
            ax.axvline(t_off_periods * T, color="0.4", ls="--", lw=0.8, zorder=0)
    if meta.get("kondo_ramp"):
        ax.axvline(meta.get("t_on_kondo_periods", 0.0) * T,
                  color="0.4", ls=":", lw=0.9, zorder=0)


def _tidy(ax, t, T, meta, letter=None, nbins=3):
    _drive_marks(ax, meta, T)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=nbins))
    ax.margins(x=0.01, y=0.2)
    # the scientific multiplier ("x10^-7") defaults to the top-left corner,
    # which is where the panel letter goes -- send it to the right instead
    off = ax.yaxis.get_offset_text()
    off.set_ha("right")
    off.set_position((1.0, 1.0))
    off.set_size(8)
    if letter is not None:
        _panel_letter(ax, letter)


TRANSIENT_FRAC = 0.05          # fraction of the record treated as switch-on


def _after_transient(t, frac=TRANSIENT_FRAC):
    """
    Index of the first sample past the switch-on transient.

    The NM starts empty and the Kondo field switches on abruptly at t=0, so the
    current overshoots the pumped signal by orders of magnitude: in the
    FM_jK0.05 run |I_q| falls from 2.6e-5 to 1.6e-12 between a 3% and a 10%
    cut.  Drawing that spike clipped just paints a solid band over the first
    part of the panel, so the current panels start after it instead.  Set
    TRANSIENT_FRAC = 0 to draw all. (fig_spectrum uses _connected_window
    instead -- see there.)
    """
    return max(1, int(frac * len(t))) if frac else 0


def _connected_window(out, meta, thresh_frac=0.5):
    """
    (i0, i1) bounding the stretch of `out["t"]` where the NM is actually
    Kondo-connected to the QSL, i.e. where the recorded J_K_t (see
    run_coupled's `record`) is at least `thresh_frac` of its own max --
    matching the J_K/2 crossing the dotted --kondo-ramp marker is drawn at.
    Reflects the recorded coupling directly rather than trusting meta flags,
    so it's correct even for a constant (non-ramped) J_K (window = whole
    record) or old files with no J_K_t (falls back to the whole record, with
    a warning -- can't reconstruct it after the fact).
    """
    t = out["t"]
    jk = out.get("J_K_t")
    if jk is None or not np.size(jk) or not np.any(jk):
        log("    (spectrum: no J_K_t recorded, using the whole record)")
        return 0, len(t) - 1
    on = jk >= thresh_frac * jk.max()
    if not on.any():
        log("    (spectrum: J_K_t never crosses the threshold, using the whole record)")
        return 0, len(t) - 1
    idx = np.flatnonzero(on)
    return int(idx[0]), int(idx[-1])


def _kondo_window(out, meta, phase="precessing"):
    """
    (i0, i1) bounding the stretch of `out["t"]` where the NM is
    Kondo-connected (_connected_window) AND, per `phase`:

      "precessing" -- the drive is actively precessing: the whole run if
                       there's no --envelope, else [t_on,t_off]*T -- the same
                       window _drive_marks' dashed lines bound, regardless of
                       whether the envelope targets theta or the phase.
      "static"      -- the drive has already stopped precessing (t > t_off).

    Combining these into one FFT window mixes two different regimes (driven
    vs. relaxing-under-a-static-field); this is how the two are told apart.
    Returns None if the intersection is empty (e.g. "static" when the drive
    never stops, i.e. no envelope or t_off = inf).
    """
    t = out["t"]
    c0, c1 = _connected_window(out, meta)
    T = 2 * np.pi / meta["omega0"]
    if not meta.get("envelope"):
        t_on, t_off = 0.0, t[-1]
    else:
        t_on, t_off = meta["t_on_periods"] * T, meta["t_off_periods"] * T

    if phase == "precessing":
        lo, hi = max(t[c0], t_on), min(t[c1], t_off)
    elif phase == "static":
        if not np.isfinite(t_off):
            return None
        lo, hi = max(t[c0], t_off), t[c1]
    else:
        raise ValueError(f"phase must be 'precessing' or 'static', got {phase!r}")

    mask = (t >= lo) & (t <= hi)
    if not mask.any():
        return None
    idx = np.flatnonzero(mask)
    return int(idx[0]), int(idx[-1])


def _legend_top(ax, ncol=None, pad=0.30):
    """
    Put the legend in a strip reserved at the top of the panel instead of
    letting it sit on the data.  Call after the axes limits are settled.
    """
    h, l = ax.get_legend_handles_labels()
    if not h:
        return
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y1 + pad * (y1 - y0))
    ax.legend(h, l, loc="upper center", ncol=ncol or len(h), fontsize=8,
              borderaxespad=0.2, columnspacing=0.8, handlelength=1.1)


# ---------------------------------------------------------------------------
# reduced density matrices / entanglement
#
# qutip's psi.ptrace() on a 14-qubit ket takes ~20 s; the Schmidt reshape below
# is exact to 1e-17 against it and takes ~5 ms, which is what makes it viable
# to compute these every single coupled step instead of post-hoc.
# ---------------------------------------------------------------------------

def make_reducer(sites, n=N):
    """
    Returns split(psi) -> T, the (2^k, 2^(n-k)) Schmidt matrix for the
    bipartition (sites | rest).  Subsystem ordering is `sorted(sites)`, i.e.
    the same convention as qutip's ptrace and as qt.tensor (site 0 = leftmost,
    most significant), so the exported rho matches GMN.jl's dims = [2,2,...].
    """
    keep = sorted(sites)
    rest = [k for k in range(n) if k not in keep]
    perm = keep + rest
    dk = 2 ** len(keep)

    def split(psi):
        return psi.reshape([2] * n).transpose(perm).reshape(dk, -1)

    return split


def reduced_rho(psi, split):
    """rho_A = Tr_B |psi><psi|, Hermitised to kill 1e-19 asymmetry."""
    T = split(psi)
    rho = T @ T.conj().T
    return 0.5 * (rho + rho.conj().T)


def log_negativity(psi, split):
    """
    E_N between A and the rest for a PURE state: the Schmidt shortcut
        E_N = 2 log2 sum_i sqrt(lambda_i) = 2 log2 ||T||_*  (nuclear norm)
    Same quantity as dashboard.calculate_log_negativity_pure_bipartite.
    """
    s = np.linalg.svd(split(psi), compute_uv=False)
    return 2.0 * np.log2(s.sum())


# ---------------------------------------------------------------------------
# operators
# ---------------------------------------------------------------------------

def op_at(op, i, n=N):
    return qt.tensor([op if k == i else qt.qeye(2) for k in range(n)])


def two_site(a, i, b, j, n=N):
    ops = [qt.qeye(2)] * n
    ops[i], ops[j] = a, b
    return qt.tensor(ops)


def spm(q):
    """Qobj -> scipy CSR, sorted indices, never going dense (2^14 dense = 4 GB)."""
    m = q.to("CSR").data.as_scipy().tocsr()
    m.sort_indices()
    return m


def linidx(m):
    rows = np.repeat(np.arange(m.shape[0]), np.diff(m.indptr)).astype(np.int64)
    return rows * m.shape[1] + m.indices


def hamiltonian_kitaev(J=(1, 1, 1), n=N):
    Jx, Jy, Jz = J
    Jk = {"x": -Jx, "y": -Jy, "z": -Jz}
    return sum(Jk[a] * two_site(PAULI[a], i, PAULI[a], j, n) for i, j, a in BONDS)


def wilson_loops(n=N):
    out, m = [], n - 1
    for i in range(3):
        d = {"y": [0 + 2 * i, m - 2 - 2 * i],
             "z": [1 + 2 * i, m - 1 - 2 * i],
             "x": [2 + 2 * i, m - 0 - 2 * i]}
        ops = [qt.qeye(2)] * n
        for lbl, sites in d.items():
            for s in sites:
                ops[s] = PAULI[lbl]
        out.append(qt.tensor(ops))
    return out


# ---------------------------------------------------------------------------
# fast rebuildable Hamiltonian:  H(c) = H0 + sum_{i,a} c[i,a] sigma^a_i
# ---------------------------------------------------------------------------

def make_field_hamiltonian(H0, n=N):
    """
    Returns (set_H, H) where set_H(coeffs) writes into H's CSR data array and
    returns H.  All 3n single-site Paulis and H0 are pre-conformed to one
    shared sparsity pattern, so an update is a single (3n, nnz) matvec:
    ~1 ms instead of ~210 ms for a naive sparse sum.  Verified against the
    naive sum to 6e-17.
    """
    H0s = spm(H0)
    ops = [spm(op_at(PAULI[a], i, n)) for i in range(n) for a in "xyz"]

    # irrational weights so no structural entry cancels while building the union
    T = H0s * np.pi + sum(o * (1.0 + 0.37 * k) for k, o in enumerate(ops))
    T = T.tocsr()
    T.sort_indices()
    H = T.astype(complex)
    tlin = linidx(H)

    h0 = np.zeros(H.nnz, complex)
    h0[np.searchsorted(tlin, linidx(H0s))] = H0s.data
    D = np.zeros((3 * n, H.nnz), complex)
    for k, o in enumerate(ops):
        D[k, np.searchsorted(tlin, linidx(o))] = o.data

    def set_H(coeffs):
        H.data[:] = h0 + np.asarray(coeffs, float).ravel() @ D
        return H

    return set_H, H


# ---------------------------------------------------------------------------
# FI drive
# ---------------------------------------------------------------------------

def make_drive(J_I=0.005, omega0=0.01, theta0=np.pi / 8, envelope=False,
               envelope_target="theta", t_on=0.0, t_off=np.inf, k_ramp=0.05):
    """
    Returns (M, theta): M(t) is the unit magnetisation vector.

    envelope=False -> Eq.(1) of the draft: theta fixed, phase = omega0*t.

    envelope=True, envelope_target="theta" (default) -> theta(t) ramps
    0 -> theta0 -> 0 over [t_on, t_off]; the precession phase still runs on
    the bare clock, phase = omega0*t. Sigmoids go through tanh so large
    k*(t-t_on) does not overflow.

    envelope=True, envelope_target="phase" -> theta stays FIXED at theta0
    (e.g. theta0=arccos(1/sqrt(3)) for an [111]-type tilt) and instead the
    precession clock freezes outside [t_on, t_off]:

        tau(t) = t_on + (1/k)[softplus(k(t-t_on)) - softplus(k(t-t_off))]
        phase(t) = omega0 * tau(t)

    so the field sits static at that fixed tilt (azimuth = omega0*t_on, not
    separately settable) until t_on, then precesses, instead of the cone
    angle ramping up.
    """
    sig = lambda x: 0.5 * (1.0 + np.tanh(0.5 * x))
    phase_env = envelope and envelope_target == "phase"

    def theta(t):
        if not envelope or phase_env:
            return theta0
        off = sig(k_ramp * (t - t_off)) if np.isfinite(t_off) else 0.0
        return theta0 * (sig(k_ramp * (t - t_on)) - off)

    def tau(t):
        a = np.logaddexp(0.0, k_ramp * (t - t_on)) / k_ramp
        b = (np.logaddexp(0.0, k_ramp * (t - t_off)) / k_ramp
             if np.isfinite(t_off) else 0.0)
        return t_on + a - b

    def M(t):
        th = theta(t)
        ph = omega0 * (tau(t) if phase_env else t)
        return np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])

    return dict(M=M, theta=theta, J_I=J_I, omega0=omega0, theta0=theta0)


def make_kondo_ramp(J_K, t_on=0.0, k_ramp=0.02):
    """
    Smooth, one-way switch-on for the Kondo coupling: J_K(t) = J_K * sigmoid
    (k_ramp*(t-t_on)).  No off-ramp -- once connected, J_K stays connected,
    including after the FI drive's envelope (see make_drive) has switched
    back off.  Meant to be turned on well AFTER the drive starts precessing
    (t_on here > drive's t_on), so the Kondo coupling attaches adiabatically
    to an already-precessing state rather than quenching a static one.

    NOTE psi0 in run_coupled is the ground state with Kondo OFF (see its
    docstring) -- that is only a good t=0 state if J_K(0) ~ 0, i.e. t_on is
    several 1/k_ramp after t=0.  Do not set t_on ~ 0 with this ramp; use a
    plain float J_K (the run_coupled default) for that instead.
    """
    sig = lambda x: 0.5 * (1.0 + np.tanh(0.5 * x))
    return lambda t: J_K * sig(k_ramp * (t - t_on))


# ---------------------------------------------------------------------------
# NM backends -- each returns a dict of closures:
#   apply_S(S14, j_sd) / step(dt) -> t / readout() -> (t, sigma14, Ic, Is) / relax(T)
# ---------------------------------------------------------------------------

def make_mock_nm(relax_rate=0.05):
    """
    Backend stub, no Julia.  Local moments relaxing toward the applied exchange
    field plus precession; "currents" are a crude dsigma/dt proxy.
    PHYSICALLY MEANINGLESS -- it exists only to exercise the loop, the
    bookkeeping and the plotting.  Never use it for results.
    """
    st = {"t": 0.0, "sig": np.zeros((N, 3)), "h": np.zeros((N, 3)),
          "last": np.zeros((N, 3))}

    def apply_S(S14, j_sd):
        st["h"] = -j_sd * np.asarray(S14, float)

    def step(dt):
        st["last"] = st["sig"].copy()
        st["sig"] = st["sig"] + dt * (relax_rate * (st["h"] - st["sig"])
                                      + np.cross(st["sig"], st["h"]))
        st["t"] += dt
        return st["t"]

    def readout():
        d = (st["sig"] - st["last"]).sum(axis=0)
        return st["t"], st["sig"].copy(), np.array([d[2], -d[2]]), np.stack([d, -d])

    def relax(T, dt=0.5):
        for _ in range(max(1, int(round(T / dt)))):
            step(dt)
        return st["t"]

    return dict(apply_S=apply_S, step=step, readout=readout, relax=relax, kind="mock")


def make_julia_nm(jl_dir, project=None, **cfg):
    """
    Real backend: TDNEGF ladder through juliacall.  Needs `pip install juliacall`
    and a Julia env with TDNEGF, DifferentialEquations, StaticArrays.  jl_dir
    must hold both square_electrons_lattice.jl and qsl_nm_bridge.jl.
    """
    from juliacall import Main as jl
    if project:
        jl.seval(f'import Pkg; Pkg.activate(raw"{Path(project).resolve()}")')
    d = Path(jl_dir).resolve()
    jl.seval(f'include(raw"{d / "square_electrons_lattice.jl"}")')
    jl.seval(f'include(raw"{d / "qsl_nm_bridge.jl"}")')
    if not jl.check_registry():
        raise RuntimeError("honeycomb <-> ladder registry check failed")
    kw = ", ".join(f"{k}={v!r}" for k, v in cfg.items())
    dev = jl.seval(f"nm_setup(NMConfig({kw}))")

    def apply_S(S14, j_sd):
        jl.apply_S_honeycomb_b(dev, np.ascontiguousarray(S14, float), float(j_sd))

    def step(dt):
        return float(jl.step_b(dev, float(dt)))

    def readout():
        t, s, Ic, Is = jl.nm_readout(dev)
        return float(t), np.asarray(s), np.asarray(Ic), np.asarray(Is)

    def relax(T, dt=0.5):
        return float(jl.relax_b(dev, float(T), Δt=float(dt)))

    return dict(apply_S=apply_S, step=step, readout=readout, relax=relax, kind="julia")


# ---------------------------------------------------------------------------
# the coupled evolution
# ---------------------------------------------------------------------------

def run_coupled(nm, *, H_kitaev=None, J_alpha=0.1, J_K=0.05, J_K_of_t=None,
                drive=None, dt=0.5, t_final=628.3, sample_every=1, nm_relax=20.0,
                kondo_sign=-1, corrector=True, checkpoint=None, ckpt_every=500,
                hex_sites=HEX_SITES, n_gmn=9, save_states=False, state_every=50):
    """
    Exponential midpoint with one predictor-corrector pass: over each slice both
    Hamiltonians are frozen at their t+dt/2 values.  Second-order, versus the
    first-order left-endpoint freeze the Julia docstring assumes.

    kondo_sign s enters as H_QSL += s*J_K sum S.<sigma> and the matching term in
    H_TB.  The Julia side hard-codes H = H0 - j_sd*(S.sigma), so we hand it
    j_sd = -s*J_K.

    J_K_of_t, if given, overrides the constant J_K with a callable J_K(t) --
    see make_kondo_ramp for the smooth-connect helper.  J_K is still used for
    psi0 (t=0 is always Kondo-off, see below) and for the run's metadata/tag.

    Alongside the expectation values it records, per sample, the bipartite
    log-negativity of `hex_sites` against the rest (~5 ms, see make_reducer),
    the drive vector, and -- at `n_gmn` times spread over the run -- the reduced
    6-spin density matrix GMN.jl consumes.  `save_states` additionally keeps
    psi(t) every `state_every` steps (256 kB per snapshot).

    Returns a dict of arrays.
    """
    if H_kitaev is None:
        H_kitaev = hamiltonian_kitaev()
    if drive is None:
        drive = make_drive()
    M_of_t, J_I = drive["M"], drive["J_I"]
    JK = J_K_of_t if J_K_of_t is not None else (lambda _t, _v=J_K: _v)

    Hk = J_alpha * H_kitaev
    set_H, _ = make_field_hamiltonian(Hk)
    Wops = [spm(w) for w in wilson_loops()]
    sops = [[spm(op_at(PAULI[a], i)) for a in "xyz"] for i in range(N)]

    # --- psi0: GS of H_Kitaev + FI drive at t=0, Kondo OFF.
    # <sigma> is undefined before the NM has electrons, so psi0 cannot include
    # the Kondo field self-consistently -- the NM switches on suddenly at t=0.
    # nm_relax limits, but does not remove, the resulting transient.
    log("Computing QSL ground state at t=0 ...")
    M0 = M_of_t(0.0)
    Stot = {a: sum(op_at(PAULI[a], i) for i in range(N)) for a in "xyz"}
    H0 = Hk + J_I * sum(M0[k] * Stot[a] for k, a in enumerate("xyz"))
    _, psi_q = H0.groundstate(sparse=True, tol=1e-8, maxiter=100000)
    psi = psi_q.full().ravel().astype(complex)
    psi /= np.linalg.norm(psi)

    def expect_S(v):
        return np.array([[np.vdot(v, sops[i][a] @ v).real for a in range(3)]
                         for i in range(N)])

    def coeffs(t, sg):
        return J_I * M_of_t(t)[None, :] + kondo_sign * JK(t) * sg

    nm["apply_S"](np.zeros((N, 3)), 0.0)
    if nm_relax > 0:
        log(f"Relaxing NM for {nm_relax} hbar/gamma (decoupled) ...")
        nm["relax"](nm_relax)

    _, sg, Ic, Is = nm["readout"]()
    S = expect_S(psi)

    split = make_reducer(hex_sites)

    theta_of_t = drive.get("theta", lambda _t: drive.get("theta0", np.nan))

    rec = {k: [] for k in ("t", "S", "sigma", "Wp", "Ic", "Is", "E", "E_N",
                           "M", "theta_t", "J_K_t")}

    def record(t, S, sg, Ic, Is, psi, H):
        rec["t"].append(t)
        rec["S"].append(S.copy())
        rec["sigma"].append(np.asarray(sg).copy())
        rec["Wp"].append([np.vdot(psi, w @ psi).real for w in Wops])
        rec["Ic"].append(np.asarray(Ic).copy())
        rec["Is"].append(np.asarray(Is).copy())
        rec["E"].append(np.vdot(psi, H @ psi).real)
        rec["E_N"].append(log_negativity(psi, split))
        rec["M"].append(M_of_t(t))
        rec["theta_t"].append(float(theta_of_t(t)))   # varies when --envelope
        rec["J_K_t"].append(float(JK(t)))              # varies when J_K_of_t is set

    n_steps = int(round(t_final / dt))

    # GMN snapshots: n_gmn times spread over the run, stored as reduced rho so
    # the export needs neither psi nor a second pass.  Step 0 is included.
    gmn_steps = np.unique(np.round(np.linspace(0, n_steps, n_gmn)).astype(int))
    rho_gmn, t_gmn = [], []
    states, t_states = [], []

    def snapshot(step, t, psi):
        if step in gmn_steps:
            rho_gmn.append(reduced_rho(psi, split))
            t_gmn.append(t)
        if save_states and step % state_every == 0:
            states.append(psi.astype(np.complex128).copy())
            t_states.append(t)

    t = 0.0
    record(t, S, sg, Ic, Is, psi, set_H(coeffs(0.0, sg)))
    snapshot(0, t, psi)

    sg_prev = sg.copy()
    t0 = time.time()
    log(f"Starting {n_steps} coupled steps, dt = {dt}, t_final = {t_final:.1f}")

    for n in range(n_steps):
        tm = t + 0.5 * dt

        sg_mid = sg + 0.5 * (sg - sg_prev)                 # predictor
        H = set_H(coeffs(tm, sg_mid))
        psi_n = expm_multiply(-1j * dt * H, psi)
        psi_n /= np.linalg.norm(psi_n)
        S_n = expect_S(psi_n)

        nm["apply_S"](0.5 * (S + S_n), -kondo_sign * JK(tm))  # H = H0 - j_sd*(S.sigma)
        nm["step"](dt)
        _, sg_n, Ic, Is = nm["readout"]()

        if corrector:
            H = set_H(coeffs(tm, 0.5 * (sg + sg_n)))
            psi_n = expm_multiply(-1j * dt * H, psi)
            psi_n /= np.linalg.norm(psi_n)
            S_n = expect_S(psi_n)

        psi, S = psi_n, S_n
        sg_prev, sg = sg, sg_n
        t += dt

        if (n + 1) % sample_every == 0:
            record(t, S, sg, Ic, Is, psi, H)
        snapshot(n + 1, t, psi)

        if (n + 1) % max(1, n_steps // 20) == 0:
            el = time.time() - t0
            log(f"  {n+1:6d}/{n_steps}  t={t:9.1f}  <W_0>={rec['Wp'][-1][0]: .5f}  "
                f"I_L={np.asarray(Ic)[0]: .3e}  "
                f"[{el:.0f}s, ~{el/(n+1)*(n_steps-n-1):.0f}s left]")

        if checkpoint and (n + 1) % ckpt_every == 0:
            np.savez_compressed(checkpoint, **{k: np.array(v) for k, v in rec.items()})

    log(f"Done in {time.time()-t0:.0f}s")

    out = {k: np.array(v) for k, v in rec.items()}
    out["rho_gmn"] = np.array(rho_gmn)                    # (n_gmn, 64, 64) complex
    out["t_gmn"] = np.array(t_gmn)
    out["gmn_sites"] = np.array(sorted(hex_sites))
    if save_states:
        out["states"] = np.array(states)                  # (n_snap, 2^N) complex
        out["t_states"] = np.array(t_states)
    return out


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def save_run(out, tag, out_dir, meta):
    """
    npz  : every array the figures use, so --replot never needs the solvers.
    csv  : wide form, one row per sample -- for external plotting/Origin.
    json : the full parameter set.
    psi(t), if recorded, goes to its own file so the main npz stays small.
    """
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)

    arrays = {k: v for k, v in out.items() if k not in ("states", "t_states")}
    np.savez_compressed(d / f"{tag}_coupled.npz", **arrays,
                        **{k: v for k, v in meta.items() if np.isscalar(v)})
    (d / f"{tag}_meta.json").write_text(json.dumps(meta, indent=2, default=str))

    if "states" in out:
        p = d / f"{tag}_states.npz"
        np.savez_compressed(p, states=out["states"], t=out["t_states"])
        log(f"Saved {p}  ({out['states'].shape[0]} snapshots, "
            f"{p.stat().st_size / 1e9:.2f} GB)")

    import pandas as pd
    cols = {"t": out["t"]}
    for i in range(N):
        for a, lbl in enumerate("xyz"):
            cols[f"S_{lbl}_site{i}"] = out["S"][:, i, a]
            cols[f"sig_{lbl}_site{i}"] = out["sigma"][:, i, a]
    for p in range(out["Wp"].shape[1]):
        cols[f"W_{p}"] = out["Wp"][:, p]
    for q, lbl in enumerate(("L", "R")):
        cols[f"Ic_{lbl}"] = out["Ic"][:, q]
        for a, al in enumerate("xyz"):
            cols[f"Is_{lbl}_{al}"] = out["Is"][:, q, a]
    cols["E_QSL"] = out["E"]
    if "E_N" in out:
        cols["E_N"] = out["E_N"]              # same column name dashboard.py reads
    if "M" in out:
        for a, al in enumerate("xyz"):
            cols[f"M_{al}"] = out["M"][:, a]
    pd.DataFrame(cols).to_csv(d / f"{tag}_coupled_wide.csv", index=False)
    log(f"Saved {d / f'{tag}_coupled_wide.csv'}")


def load_run(tag, out_dir):
    """Inverse of save_run: (out, meta) from disk, for --replot."""
    d = Path(out_dir)
    with np.load(d / f"{tag}_coupled.npz") as z:
        out = {k: z[k] for k in z.files}
    meta_p = d / f"{tag}_meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    meta.setdefault("omega0", float(out.get("omega0", 0.01)))
    meta.setdefault("title", tag.replace("_", " "))
    return out, meta


# ---------------------------------------------------------------------------
# GMN export -- the file contract GMN.jl's load_rho() expects
# ---------------------------------------------------------------------------

def export_rho_for_gmn(out, prefix, out_dir, theta=0.0, ind=1):
    """
    Port of dashboard.export_rho_for_julia_gmn, fed from the rho snapshots the
    run already recorded (no psi, no second ptrace pass).

    Writes, into <out_dir>/GMN_results/ ,

        {prefix}_{j}_{ind}_theta{theta}_r.txt      real part, tab-separated
        {prefix}_{j}_{ind}_theta{theta}_i.txt      imaginary part

    for j = 1..n_gmn, which is exactly what GMN.jl's

        load_rho("GMN_results/"*filename*"_r.txt")

    reads with readdlm.  Run GMN.jl from the directory holding GMN_results/.
    dims there must be [2,2,2,2,2,2] for the default six-site hexagon.
    """
    rho = out["rho_gmn"]
    if rho.size == 0:
        log("  [gmn] nothing to export: no rho snapshots in this run")
        return []

    d = Path(out_dir) / "GMN_results"
    d.mkdir(parents=True, exist_ok=True)
    sites = out.get("gmn_sites", np.array(HEX_SITES))
    n_spin = int(round(np.log2(rho.shape[-1])))

    written = []
    for j, (r, t_val) in enumerate(zip(rho, out["t_gmn"]), start=1):
        base = f"{prefix}_{j}_{ind}_theta{theta}"
        np.savetxt(d / f"{base}_r.txt", r.real, fmt="%.18e", delimiter="\t")
        np.savetxt(d / f"{base}_i.txt", r.imag, fmt="%.18e", delimiter="\t")
        written.append(base)
        log(f"  [gmn] j={j}  t={t_val:9.2f}  -> {base}_[ri].txt")

    log(f"  [gmn] {len(written)} slices in {d}")
    log(f"  [gmn] sites {[int(s) for s in sites]}  ->  "
        f"GMN.jl dims = {[2] * n_spin}")
    return written


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def fig_dashboard(out, meta, path):
    """
    Two-row overview, laid out like Fig. 2 of the draft: the QSL quantities on
    top, what they pump into the NM underneath.  Column titles carry the
    quantity (no y-labels), which is what keeps the panels narrow enough to
    drop straight into a two-column page.
    """
    t, T = out["t"], 2 * np.pi / meta["omega0"]
    fig, ax = plt.subplots(2, 4, figsize=(7.8, 3.4), sharex=True,
                           layout="constrained")

    # --- row 0: QSL ---------------------------------------------------------
    Stot = out["S"].sum(axis=1) / N
    for a, lbl in enumerate("xyz"):
        ax[0, 0].plot(t, Stot[:, a], color=C_AXIS[lbl], lw=0.75,
                      label=rf"$\alpha={lbl}$")
    ax[0, 0].set_title(r"$\langle\hat{S}^{\alpha}_{\mathrm{tot}}\rangle/N$")

    # one curve per site: a sequential ramp reads as a family, and avoids
    # borrowing the x/y/z colours that mean something else in this figure
    site_colors = plt.cm.viridis(np.linspace(0.0, 0.9, N))
    for i in range(N):
        ax[0, 1].plot(t, out["S"][:, i, 2], lw=0.5, alpha=0.9,
                      color=site_colors[i])
    ax[0, 1].set_title(r"$\langle\hat{S}^z_i\rangle$")

    for p in range(out["Wp"].shape[1]):
        ax[0, 2].plot(t, out["Wp"][:, p], color=C_WP[p % len(C_WP)],
                      ls=LS_WP[p % len(LS_WP)], lw=LW_WP[p % len(LW_WP)],
                      label=rf"$p={p}$")
    ax[0, 2].set_ylim(-0.1, 1.15)
    ax[0, 2].set_title(r"$\langle\hat{W}_p\rangle$")

    if "E_N" in out and np.size(out["E_N"]):
        ax[0, 3].plot(t, out["E_N"], color="black", lw=1.0)
        ax[0, 3].set_title(r"$\mathcal{N}$")
    else:
        ax[0, 3].text(0.5, 0.5, "no $\\mathcal{N}$\nin this run", ha="center",
                      va="center", transform=ax[0, 3].transAxes, fontsize=8)
        ax[0, 3].set_xticks([]); ax[0, 3].set_yticks([])

    # --- row 1: NM ----------------------------------------------------------
    sig_tot = out["sigma"].sum(axis=1) / N
    for a, lbl in enumerate("xyz"):
        ax[1, 0].plot(t, sig_tot[:, a], color=C_AXIS[lbl], lw=0.75,
                      label=rf"$\alpha={lbl}$")
    ax[1, 0].set_title(r"$\langle\hat{\sigma}^{\alpha}\rangle_{\mathrm{NM}}/N$")

    i0 = _after_transient(t)                       # currents only
    for q, lbl in enumerate(("L", "R")):
        ax[1, 1].plot(t[i0:], out["Ic"][i0:, q], color=C_LEAD[lbl], lw=0.8,
                      ls="-" if q == 0 else (0, (4, 2)), label=rf"$q={lbl}$")
    ax[1, 1].set_title(r"$I_q$")

    for a, lbl in enumerate("xyz"):
        ax[1, 2].plot(t[i0:], out["Is"][i0:, 0, a], color=C_AXIS[lbl], lw=0.8)
    ax[1, 2].set_title(r"$I^{S_\alpha}_L$")

    ax[1, 3].plot(t, out["E"], color="black", lw=0.9)
    ax[1, 3].set_title(r"$\langle\hat{H}_{\mathrm{QSL}}\rangle$")

    letters = iter("abcdefgh")
    for r in range(2):
        for c in range(4):
            _tidy(ax[r, c], t, T, meta, letter=next(letters))
            if r == 1:
                ax[r, c].set_xlabel(TIME_LABEL)
                ax[r, c].xaxis.set_major_locator(MaxNLocator(nbins=4))

    # legends last, once limits are settled.  The alpha legend in (a) also
    # explains (e) and (g); the q legend in (f) also explains (b) of currents.
    _legend_top(ax[0, 0], ncol=3)
    _legend_top(ax[0, 2], ncol=3)
    _legend_top(ax[1, 1], ncol=2)
    _savefig(fig, path)


def fig_currents(out, meta, path):
    """Charge and spin currents per lead -- the deliverable the draft is missing."""
    t, T = out["t"], 2 * np.pi / meta["omega0"]
    # sharey per row: L and R are meant to be compared, and a legend that
    # stretches one panel's limits must not make them silently different
    fig, ax = plt.subplots(2, 2, figsize=(7.0, 3.6), sharex=True, sharey="row",
                           layout="constrained")
    letters = iter("abcd")
    i0 = _after_transient(t)
    for q, lbl in enumerate(("L", "R")):
        ax[0, q].plot(t[i0:], out["Ic"][i0:, q], color="black", lw=0.8)
        ax[0, q].set_title(rf"$I_q$,  $q={lbl}$")

        for a, al in enumerate("xyz"):
            ax[1, q].plot(t[i0:], out["Is"][i0:, q, a], color=C_AXIS[al],
                          lw=0.8, label=rf"$\alpha={al}$")
        ax[1, q].set_title(rf"$I^{{S_\alpha}}_q$,  $q={lbl}$")
        ax[1, q].set_xlabel(TIME_LABEL)
        ax[1, q].xaxis.set_major_locator(MaxNLocator(nbins=4))

    for r in range(2):
        for c in range(2):
            _tidy(ax[r, c], t, T, meta, letter=next(letters))
    _legend_top(ax[1, 0], ncol=3)
    _savefig(fig, path)


def fig_spectrum(out, meta, path, phase="precessing"):
    """
    FFT of the pumped currents in units of the drive frequency, taken ONLY
    over the window where the NM is Kondo-connected AND, per `phase`, the
    drive is either actively "precessing" or has already gone "static" (see
    _kondo_window) -- combining both into one FFT mixes a driven regime with
    a relaxing one. Call this twice (see make_figures) for the two windows.

    HYPOTHESIS UNDER TEST (not a result): uniform precession, as in the QAFI
    case, pumps at omega0 and 2*omega0 and little else; the inhomogeneous,
    entangled QSL dynamics should show a richer harmonic content.
    """
    t, w0 = out["t"], meta["omega0"]
    T = 2 * np.pi / w0
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6), layout="constrained")

    win = _kondo_window(out, meta, phase=phase)
    if win is None:
        for a in ax:
            a.text(0.5, 0.5, f"no '{phase}' window\n(Kondo-connected)",
                   ha="center", va="center", transform=a.transAxes, fontsize=9)
            a.set_xticks([]); a.set_yticks([])
        _savefig(fig, path)
        log(f"    (spectrum: no '{phase}' window)")
        return

    i0, i1 = win
    tt, dt = t[i0:i1 + 1], t[1] - t[0]

    # Resolving harmonics needs several drive periods of connected window:
    # frequency resolution is 1/(span) = omega0/n_periods_kept.
    n_kept = (tt[-1] - tt[0]) / T
    if n_kept < 3:
        for a in ax:
            a.text(0.5, 0.5, f"only {n_kept:.1f} drive periods are "
                             f"Kondo-connected and {phase} --\n"
                             "harmonics are not resolved",
                   ha="center", va="center", transform=a.transAxes, fontsize=9)
            a.set_xticks([]); a.set_yticks([])
        _savefig(fig, path)
        log(f"    (spectrum: only {n_kept:.1f} {phase} connected periods, "
            f"harmonics unresolved)")
        return

    freq = np.fft.rfftfreq(len(tt), d=dt) * 2 * np.pi / w0    # in units of omega0
    peak = 0.0

    for q, lbl in enumerate(("L", "R")):
        y = out["Ic"][i0:i1 + 1, q] - out["Ic"][i0:i1 + 1, q].mean()
        A = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
        peak = max(peak, A.max())
        ax[0].semilogy(freq, A, color=C_LEAD[lbl], lw=0.8,
                       ls="-" if q == 0 else (0, (4, 2)), label=rf"$q={lbl}$")
    ax[0].set_title(r"$|I_q(\omega)|^2$")

    for a, al in enumerate("xyz"):
        y = out["Is"][i0:i1 + 1, 0, a] - out["Is"][i0:i1 + 1, 0, a].mean()
        A = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
        peak = max(peak, A.max())
        ax[1].semilogy(freq, A, color=C_AXIS[al], lw=0.8, label=rf"$\alpha={al}$")
    ax[1].set_title(r"$|I^{S_\alpha}_L(\omega)|^2$")

    for k, a in enumerate(ax):
        a.set_xlim(0, 8)
        a.set_ylim(peak * 1e-6, peak * 3)
        for h in range(1, 9):
            a.axvline(h, color="0.85", ls=":", lw=0.6, zorder=0)
        a.set_xlabel(r"$\omega/\omega_0$")
        a.legend(loc="upper right", ncol=1)
        a.yaxis.set_minor_locator(ticker.NullLocator())
        _panel_letter(a, "ab"[k])
    fig.suptitle("Kondo-connected, drive precessing" if phase == "precessing"
                else "Kondo-connected, drive static (post-precession)",
                fontsize=9, y=1.02)
    _savefig(fig, path)


def fig_spectrogram(out, meta, path, win_periods=2.0, overlap_frac=0.9):
    """
    (omega/omega0, t) heatmap of |FFT(current)|^2 in a sliding Hann window
    (win_periods drive periods wide, overlap_frac overlap between hops) --
    the continuous-time complement to fig_spectrum's precessing/static
    split: instead of picking one window or the other, this shows the
    precessing -> static transition (and the t=0 switch-on, and the Kondo
    connect) directly as the spectrum evolves. Dashed markers = drive
    on/off (--envelope), dotted = Kondo connect (--kondo-ramp), same as
    _drive_marks/_kondo_window.
    """
    t, w0 = out["t"], meta["omega0"]
    T = 2 * np.pi / w0
    dt = t[1] - t[0]
    nperseg = min(len(t), max(16, int(win_periods * T / dt)))
    noverlap = int(nperseg * overlap_frac)

    # I_q dropped: it's noise-floor flat everywhere except the t=0 transient
    # (see fig_spectrum) -- nothing for a spectrogram to show.
    panels = [(r"$I^{S_\alpha}_L,\ \alpha=x$", out["Is"][:, 0, 0]),
             (r"$I^{S_\alpha}_L,\ \alpha=y$", out["Is"][:, 0, 1]),
             (r"$I^{S_\alpha}_L,\ \alpha=z$", out["Is"][:, 0, 2])]

    specs, vmax = [], 0.0
    for title, y in panels:
        f, tt, Sxx = spectrogram(y - y.mean(), fs=1 / dt, nperseg=nperseg,
                                 noverlap=noverlap, window="hann",
                                 scaling="spectrum")
        specs.append((title, f, tt, Sxx))
        vmax = max(vmax, Sxx.max())
    norm = LogNorm(vmin=max(vmax * 1e-8, 1e-300), vmax=vmax)

    fig, ax = plt.subplots(3, 1, figsize=(6.0, 6.6), sharex=True, sharey=True,
                           layout="constrained")
    mesh = None
    for k, (title, f, tt, Sxx) in enumerate(specs):
        a = ax[k]
        mesh = a.pcolormesh(tt, 2 * np.pi * f / w0, Sxx, norm=norm,
                            cmap="inferno", shading="auto")
        a.set_ylim(0, 8)
        a.set_title(title)
        a.set_ylabel(r"$\omega/\omega_0$")
        _panel_letter(a, "abc"[k], xy=(0.01, 1.05))
        if meta.get("envelope"):
            a.axvline(meta["t_on_periods"] * T, color="w", ls="--", lw=0.8, alpha=0.85)
            t_off = meta["t_off_periods"] * T
            if np.isfinite(t_off):
                a.axvline(t_off, color="w", ls="--", lw=0.8, alpha=0.85)
        if meta.get("kondo_ramp"):
            a.axvline(meta["t_on_kondo_periods"] * T, color="w", ls=":",
                     lw=0.9, alpha=0.85)
    ax[-1].set_xlabel(TIME_LABEL)

    fig.colorbar(mesh, ax=ax, shrink=0.85, pad=0.02, label="power (log scale)")
    _savefig(fig, path)


def fig_wp_current(out, meta, path):
    """
    <W_p>(t) and the pumped current on one time axis: does the current change
    character when the flux collapses?  If it does, that link IS the paper.
    """
    t, T = out["t"], 2 * np.pi / meta["omega0"]
    fig, ax1 = plt.subplots(figsize=(3.4, 2.4), layout="constrained")

    # This figure asks one question: does the current change character when the
    # flux collapses?  So the flux goes on a neutral grey ramp and the accent
    # colour is reserved for the current alone -- the shared darkorange of
    # C_WP would otherwise be indistinguishable from the current trace.
    wp_grey = ["0.60", "0.30", "black"]
    wp_ls = ["-", "--", ":"]
    for p in range(out["Wp"].shape[1]):
        ax1.plot(t, out["Wp"][:, p], color=wp_grey[p % len(wp_grey)],
                 ls=wp_ls[p % len(wp_ls)], lw=1.1, label=rf"$p={p}$", zorder=3)
    ax1.set_ylim(-0.1, 1.3)
    ax1.set_ylabel(r"$\langle\hat{W}_p\rangle$")
    ax1.set_xlabel(TIME_LABEL)
    ax1.legend(loc="lower left", ncol=3, frameon=True, framealpha=0.85,
               edgecolor="none", facecolor="white")
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
    ax1.xaxis.set_major_locator(MaxNLocator(nbins=4))
    _drive_marks(ax1, meta, T)

    accent = "#f86503"                    # same accent as dashboard.py's GMN axis
    i0 = _after_transient(t)
    ax2 = ax1.twinx()
    ax2.plot(t[i0:], out["Is"][i0:, 0, 2], color=accent, lw=0.7, alpha=0.9,
             zorder=0)
    ax2.set_ylabel(r"$I^{S_z}_L$", color=accent)
    ax2.tick_params(axis="y", colors=accent, direction="in")
    ax2.spines["right"].set_color(accent)
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
    ax2.yaxis.set_minor_locator(ticker.NullLocator())
    fmt = ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    ax2.yaxis.set_major_formatter(fmt)
    ax2.yaxis.get_offset_text().set(color=accent, size=8, ha="left")
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)
    _savefig(fig, path)


FIG_FORMATS = ("pdf", "png")


IMG_SUFFIXES = {".pdf", ".png", ".svg", ".eps", ".ps", ".jpg", ".jpeg",
                ".tif", ".tiff"}


def _savefig(fig, path, formats=None):
    """
    Vector for the manuscript, raster for quick looks -- same basename.

    Strip only a real image extension: tags carry dots ("FM_jK0.05_w0.01_dt0.5")
    and Path.with_suffix("") would cut at the LAST one, collapsing every figure
    of a run onto the same filename.
    """
    p = Path(path)
    if p.suffix.lower() in IMG_SUFFIXES:
        p = p.with_suffix("")
    for ext in (formats or FIG_FORMATS):
        fig.savefig(f"{p}.{ext}")
    plt.close(fig)


def make_figures(out, meta, tag, img_dir, formats=FIG_FORMATS):
    global FIG_FORMATS
    FIG_FORMATS = tuple(formats)
    setup_style()
    d = Path(img_dir)
    d.mkdir(parents=True, exist_ok=True)
    for name, fn in (("dashboard", fig_dashboard), ("currents", fig_currents),
                     ("wp_current", fig_wp_current),
                     ("spectrogram", fig_spectrogram)):
        stem = d / f"{tag}_{name}"
        fn(out, meta, stem)
        log(f"  wrote {stem}.{{{','.join(formats)}}}")
    for phase in ("precessing", "static"):
        stem = d / f"{tag}_spectrum_{phase}"
        fig_spectrum(out, meta, stem, phase=phase)
        log(f"  wrote {stem}.{{{','.join(formats)}}}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

MODELS = {"FM": (1, 1, 1), "AFM": (-1, -1, -1)}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=("mock", "julia"), default="mock")
    p.add_argument("--jl-dir", default="./jl", help="dir with the two .jl files")
    p.add_argument("--project", default=None, help="Julia project to activate")
    p.add_argument("--model", choices=tuple(MODELS), default="FM")
    p.add_argument("--J-alpha", type=float, default=0.1)
    p.add_argument("--J-I", type=float, default=0.0025)
    p.add_argument("--J-K", type=float, default=0.05,
                   help="PLACEHOLDER: the draft leaves J_K undefined. At 0.05 "
                        "this is not a weak perturbation on J_alpha=0.1.")
    p.add_argument("--omega0", type=float, default=0.01)
    p.add_argument("--theta", type=float, default=np.pi / 8)
    p.add_argument("--kondo-sign", type=int, choices=(-1, 1), default=-1)
    p.add_argument("--dt", type=float, default=0.5)
    p.add_argument("--n-periods", type=float, default=10.0)
    p.add_argument("--sample-every", type=int, default=1)
    p.add_argument("--nm-relax", type=float, default=20.0)
    p.add_argument("--no-corrector", action="store_true")
    p.add_argument("--envelope", action="store_true")
    p.add_argument("--envelope-target", choices=("theta", "phase"), default="theta",
                   help="what --t-on/off-periods and --k-ramp envelope: the "
                        "default 'theta' ramps the cone angle 0->theta0->0 "
                        "with the phase always running. 'phase' instead "
                        "holds theta fixed at --theta and freezes the "
                        "precession phase outside [t_on,t_off] -- the field "
                        "sits static at that fixed tilt until t_on, then "
                        "starts precessing. See make_drive.")
    p.add_argument("--t-on-periods", type=float, default=1.0)
    p.add_argument("--t-off-periods", type=float, default=6.0)
    p.add_argument("--k-ramp", type=float, default=0.02)
    p.add_argument("--kondo-ramp", action="store_true",
                   help="connect J_K smoothly (sigmoid) instead of as a step "
                        "at t=0 -- see make_kondo_ramp")
    p.add_argument("--t-on-kondo-periods", type=float, default=0.0,
                   help="drive periods at which J_K(t) crosses J_K/2; should "
                        "be well after --t-on-periods so Kondo connects to an "
                        "already-precessing drive, not a static one")
    p.add_argument("--k-ramp-kondo", type=float, default=None,
                   help="sigmoid rate for the J_K ramp (default: same as --k-ramp)")
    p.add_argument("--tag", default=None)
    p.add_argument("--out-dir", default="./data_coupled")
    p.add_argument("--img-dir", default="./img_coupled")
    p.add_argument("--no-figures", action="store_true")

    g = p.add_argument_group("output / replotting")
    g.add_argument("--replot", action="store_true",
                   help="skip the simulation: reload <tag>_coupled.npz and "
                        "rebuild the figures only")
    g.add_argument("--replot-ckpt", action="store_true",
                   help="like --replot, but from the mid-run <tag>_ckpt.npz "
                        "(updated periodically as the sim advances): a "
                        "snapshot of a run still in progress elsewhere, not "
                        "the final file. Run against the same --tag/--out-dir "
                        "(and drive/kondo flags, for the vertical markers) as "
                        "the live run, any time, without touching it.")
    g.add_argument("--fig-formats", default="pdf,png",
                   help="comma-separated savefig formats (default pdf,png)")
    g.add_argument("--hex-sites", default=",".join(map(str, HEX_SITES)),
                   help="subsystem for the negativity and the GMN export")
    g.add_argument("--n-gmn", type=int, default=9,
                   help="number of rho snapshots (GMN.jl loops j=1..n)")
    g.add_argument("--export-gmn", action="store_true",
                   help="also write the _r.txt/_i.txt pairs GMN.jl reads")
    g.add_argument("--gmn-prefix", default=None,
                   help="filename prefix for the GMN export (default: tag)")
    g.add_argument("--save-states", action="store_true",
                   help="keep psi(t) every --state-every steps (256 kB each)")
    g.add_argument("--state-every", type=int, default=50)
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    tag = a.tag or f"{a.model}_jK{a.J_K:g}_w{a.omega0:g}_dt{a.dt:g}"
    T = 2 * np.pi / a.omega0
    t_final = a.n_periods * T
    fig_formats = tuple(s.strip() for s in a.fig_formats.split(",") if s.strip())
    hex_sites = tuple(int(s) for s in a.hex_sites.split(",") if s.strip() != "")

    # --- replot path: no solvers, no Julia, just the saved arrays ------------
    if a.replot:
        log(f"Replotting {tag} from {a.out_dir} (no simulation)")
        out, meta = load_run(tag, a.out_dir)
        if a.export_gmn:
            export_rho_for_gmn(out, a.gmn_prefix or tag, a.out_dir,
                               theta=round(float(meta.get("theta", a.theta)), 3))
        if not a.no_figures:
            make_figures(out, meta, tag, a.img_dir, formats=fig_formats)
        return 0

    # --- replot-ckpt path: same, but from a still-running job's checkpoint ---
    if a.replot_ckpt:
        ckpt_path = Path(a.out_dir) / f"{tag}_ckpt.npz"
        d = np.load(ckpt_path)
        out = {k: d[k] for k in d.files}
        n_done = len(out["t"])
        t_last = float(out["t"][-1]) if n_done else 0.0
        log(f"Replotting {tag} from checkpoint {ckpt_path} "
            f"({n_done} samples, t={t_last:.1f} = {t_last/T:.2f}T of "
            f"{t_final:.1f} = {a.n_periods:g}T -- run still in progress)")
        meta = dict(vars(a), tag=tag, T=T, t_final=t_final, N=N,
                    hex_sites=list(hex_sites),
                    title=rf"Kitaev {a.model},  $J_K={a.J_K:g}$,  $\omega_0={a.omega0:g}$")
        if not a.no_figures:
            make_figures(out, meta, tag, a.img_dir, formats=fig_formats)
        return 0

    log("=" * 70)
    log(f"tag       : {tag}")
    log(f"model     : Kitaev {a.model}   J_alpha={a.J_alpha}  J_I={a.J_I}  J_K={a.J_K}")
    log(f"drive     : omega0={a.omega0} (T={T:.1f})  theta={a.theta:.4f}  "
        f"envelope={a.envelope}"
        + (f"  target={a.envelope_target}" if a.envelope else ""))
    if a.kondo_ramp:
        k_ramp_kondo = a.k_ramp_kondo if a.k_ramp_kondo is not None else a.k_ramp
        log(f"kondo_ramp: on, t_on={a.t_on_kondo_periods}T  k_ramp={k_ramp_kondo}")
    log(f"stepping  : dt={a.dt}  t_final={t_final:.1f}  ({int(t_final/a.dt)} steps)")
    log(f"kondo_sign: {a.kondo_sign}  ({'consistent' if a.kondo_sign==-1 else 'draft Eq.(1) as written'})")
    log(f"backend   : {a.backend}")
    if a.backend == "mock":
        log("  *** MOCK BACKEND: currents are meaningless, loop test only ***")
    log("=" * 70)

    drive = make_drive(J_I=a.J_I, omega0=a.omega0, theta0=a.theta,
                       envelope=a.envelope, envelope_target=a.envelope_target,
                       t_on=a.t_on_periods * T,
                       t_off=a.t_off_periods * T if a.envelope else np.inf,
                       k_ramp=a.k_ramp)

    J_K_of_t = None
    if a.kondo_ramp:
        k_ramp_kondo = a.k_ramp_kondo if a.k_ramp_kondo is not None else a.k_ramp
        J_K_of_t = make_kondo_ramp(a.J_K, t_on=a.t_on_kondo_periods * T,
                                   k_ramp=k_ramp_kondo)

    nm = (make_mock_nm() if a.backend == "mock"
          else make_julia_nm(a.jl_dir, project=a.project))

    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    out = run_coupled(
        nm, H_kitaev=hamiltonian_kitaev(MODELS[a.model]),
        J_alpha=a.J_alpha, J_K=a.J_K, J_K_of_t=J_K_of_t, drive=drive,
        dt=a.dt, t_final=t_final,
        sample_every=a.sample_every, nm_relax=a.nm_relax,
        kondo_sign=a.kondo_sign, corrector=not a.no_corrector,
        checkpoint=Path(a.out_dir) / f"{tag}_ckpt.npz",
        hex_sites=hex_sites, n_gmn=a.n_gmn,
        save_states=a.save_states, state_every=a.state_every,
    )

    meta = dict(vars(a), tag=tag, T=T, t_final=t_final, N=N,
                hex_sites=list(hex_sites),
                title=rf"Kitaev {a.model},  $J_K={a.J_K:g}$,  $\omega_0={a.omega0:g}$")
    save_run(out, tag, a.out_dir, meta)

    if a.export_gmn:
        export_rho_for_gmn(out, a.gmn_prefix or tag, a.out_dir,
                           theta=round(float(a.theta), 3))

    log(f"<W_p>(0)   = {np.round(out['Wp'][0], 6)}")
    log(f"<W_p>(end) = {np.round(out['Wp'][-1], 6)}")
    log(f"E(0) = {out['E'][0]:.8f}   E(end) = {out['E'][-1]:.8f}   "
        f"dE = {out['E'][-1]-out['E'][0]:+.3e}")
    log(f"E_N(0) = {out['E_N'][0]:.6f}   E_N(end) = {out['E_N'][-1]:.6f}   "
        f"(sites {list(hex_sites)} vs rest)")

    if not a.no_figures:
        make_figures(out, meta, tag, a.img_dir, formats=fig_formats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
