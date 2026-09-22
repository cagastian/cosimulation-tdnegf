# cosimulation -- project context

Last updated: 2026-09-17 (Study 3 timing fix). Written so a fresh agent (or a human) can pick this
up cold. If you're an agent reading this at the start of a session: read this
whole file before touching anything, then check "Current state" at the
bottom, which is the part most likely to be stale.

## What this is

A coupled-dynamics simulation: a ferromagnetic-insulator (FI) drive precessing
next to a Kitaev quantum spin liquid (QSL), Kondo-coupled to a normal-metal
(NM) tight-binding ladder (via TDNEGF, in Julia). The physics:

    H_QSL(t) = J_a H_Kitaev + J_I sum_i S_i.M(t) + s*J_K sum_i S_i.<sigma_i>(t)
    H_TB(t)  = -gamma sum_<ij> c+c        + s*J_K sum_i c+_i sigma c_i.<S_i>(t)

QSL side (14-site Kitaev honeycomb) is exact diagonalization in Python/qutip;
NM side is the Julia TDNEGF ladder. The two are stepped together with an
exponential-midpoint predictor-corrector. Read `run_coupled.py`'s module
docstring for the full sign/units conventions -- it's careful about them and
worth trusting over anything summarized here.

## File map

| File | Role |
|---|---|
| `run_coupled.py` | Everything: QSL Hamiltonian (Kitaev `hamiltonian_kitaev` and isotropic `hamiltonian_heisenberg`, select via `--hamiltonian`), the FI drive (`make_drive`), the coupled stepper (`run_coupled`), all figures, the CLI. This is the file you run. Spectrogram computation/drawing is split into reusable helpers `_spectrogram_specs`/`_plot_spectrogram_panel` (used by both `fig_spectrogram` and `compare_figures.py`) -- see "Spectrogram / compare_spin_current" below. |
| `square_electrons_lattice.jl` | NM layer -- the TDNEGF ladder + stepper interface (Julia). |
| `qsl_nm_bridge.jl` | Bridges the NM ladder to the QSL's honeycomb site indexing. |
| `compare_figures.py` | AFM-vs-FM(-vs-Heisenberg) comparison figures, reusing `run_coupled.py`'s style helpers. Tags are hardcoded near the top -- edit `FM_TAG`/`AFM_TAG`/`HEISENBERG_TAG` before running. Writes `compare_{dashboard,charge_and_diff,moments}` plus `compare_spectrogram_{AFM,FM}` (standalone, full-size, `w_max`-adjustable per-model spectrograms), `compare_spin_current` (AFM+FM, the paper-quality one, see below), and -- only if the Heisenberg data is present -- `spin_current_heisenberg` (Heisenberg alone) and `compare_spin_current_3model` (AFM+FM+Heisenberg together, `w_max=8`). All three spin-current figures are the same generalized `fig_compare_spin_current(models, path, w_max=...)` called with a 1/2/3-element `models` list -- see below. |
| `dashboard.py`, `mock_data.py` | Older/adjacent tooling; `mock_data.py` writes synthetic `_coupled.npz` files for iterating on figure code without a real run. |
| `gmn_core.jl`, `gmn_para.jl`, `GMN.jl` | Genuine multipartite negativity (GMN) calculation via Mosek/JuMP, copied in from `~/code/qutip_honeycomb/GMN/data_kitaev_test/`. `gmn_core.jl`+`gmn_para.jl` is the real, parallel-capable driver; `GMN.jl` is a 3-line single-point demo, not useful on its own. **Read the memory warning at the top of `gmn_para.jl` before touching `GMN_WORKERS`** -- see "GMN / Mosek" below. |
| `run_examples.sh` | Reference commands with every flag explained. Safe to `bash run_examples.sh` as-is (only a ~90s mock smoke test is live; everything else is `if false; then...fi` guarded). **Start here for how to actually run things.** |
| `jk_scan_heatmap.py`, `ji_scan_heatmap.py` | `(J_K, t)` / `(J_I, t)` heatmaps of `<W_p>` across the respective scan. Small, hardcoded-constants scripts (this project's convention -- see `compare_figures.py`'s hardcoded tags too), deliberately *not* generalized into one `--axis` script. Safe to re-run mid-scan -- fall back from the finished npz to the live checkpoint. `run_ji_scan.sh` is the J_I scan's saved launcher (sequential, never concurrent `--backend julia`); the J_K scan's launcher was run ad hoc and never saved as a file. |
| `wp_current_scaling.py` | Quantitative Wp-vs-current relation across both scans -- see "Wp-vs-current analysis" below. |
| `data_coupled/` | `<tag>_coupled.npz` (full results), `<tag>_ckpt.npz` (live checkpoint, updated periodically during a run), `<tag>_meta.json`, `<tag>_coupled_wide.csv`, `GMN_results/` (exported rho snapshots for GMN.jl). |
| `img_coupled/` | Six figures per run -- `dashboard`, `currents`, `wp_current`, `spectrogram`, `spectrum_precessing`, `spectrum_static` -- `.pdf`+`.png`. `img_coupled/summary/` holds `compare_figures.py`'s output. |
| `GMN_calculation_res/` | GMN.jl's output. |
| `paper_results/` | Two long-running paper-specific studies, self-contained (own `data/`/`img/` per study, not mixed into `data_coupled/`/`img_coupled/`) -- see "Paper results" below, **check there first if resuming cold**, it has PIDs/timing for runs that may still be in flight. |

## Environment gotcha you cannot skip

`--backend julia` goes through `juliacall`, which resolves to Julia 1.12.7.
That build's `libjulia-internal` wants a newer `GLIBCXX` than the system
`libstdc++`, so it fails immediately with a `GLIBCXX_3.4.30` error unless you
preload Julia's own libstdc++ first:

```bash
export LD_PRELOAD=/home/sebasgom/.julia/juliaup/julia-1.12.7+0.x64.linux.gnu/lib/julia/libstdc++.so.6
```

`--backend mock` (loop/plumbing smoke test, currents are physically
meaningless) doesn't need this.

## The simulation protocol, and why it looks the way it does

Early runs used a **constant J_K from t=0** with `psi0` computed as the
Kondo-*off* ground state (necessarily -- `<sigma>` doesn't exist before the
NM is coupled). That's a real quench: `<W_p>` (plaquette flux) collapsed
almost entirely in the first ~1000 time units, *before* the FI drive ever did
anything, contaminating every "baseline" reading. Fixed in two stages, both
still in the code and both used together in every recent production run:

1. **`--envelope`** on the FI drive: instead of a constant cone angle from
   t=0, `theta(t)` ramps `0 -> theta0 -> 0` smoothly over `[t_on, t_off]`
   (`--t-on-periods`/`--t-off-periods`/`--k-ramp`, in units of the drive
   period `T = 2*pi/omega0`). The precession phase always runs at
   `omega0*t` in this mode.
2. **`--kondo-ramp`**: `J_K(t)` connects smoothly (sigmoid, one-way, never
   disconnects) via `--t-on-kondo-periods`/`--k-ramp-kondo`, set to connect
   *after* the drive is already precessing (`t_on_kondo > t_on`) so Kondo
   attaches to a moving target instead of quenching a static one.

The standard production window that's been used repeatedly:
`--envelope --t-on-periods 5 --t-off-periods 40 --k-ramp 0.1 --kondo-ramp
--t-on-kondo-periods 15 --k-ramp-kondo 0.02`, over `--n-periods 50`.

**A third mode was added most recently: `--envelope-target phase`.** Instead
of enveloping theta, this holds theta *fixed* at `--theta` and freezes the
*precession phase* outside `[t_on, t_off]` instead (same flags, reused --
that's deliberate, ask before adding a parallel flag set). The field then
sits static at a fixed tilt (e.g. `--theta 0.9553166181245093` =
`arccos(1/sqrt(3))`, the "[111]-like" polar angle) instead of at `theta=0`,
until `t_on`, then precesses, and (if `t_off` is finite) freezes again.
**There is deliberately no `phi`/phase-offset flag** (removed on request) --
the frozen azimuth is whatever `omega0*t_on` works out to, which is always 0
for integer `t_on_periods`, so the actual static direction is
`(sin(theta0), 0, cos(theta0))`, not literally `(1,1,1)/sqrt(3)` (that would
need azimuth = 45 deg). Known and accepted tradeoff.

Plots mark `t_on`/`t_off` (dashed) and, if `--kondo-ramp`, the Kondo
connect point (dotted) automatically -- see `_drive_marks` in
`run_coupled.py`.

## Runs completed so far (tags in `data_coupled/`)

| Tag | Notes |
|---|---|
| `FM_jK0.05_w0.01_dt0.5[_julia]`, `FM_jK0_w0.01_dt0.5_julia`, `FM_jK0.2_w0.01_dt0.5_julia` | Early, constant-J_K (pre-fix), various J_K. Quench artifact present. |
| `FM_jK0.2_env5-15_k0.1_julia`, `AFM_jK0.2_env5-15_k0.1_julia` | First envelope runs (20T), still constant J_K (pre-Kondo-ramp fix). |
| `FM_jK0.05_kondoRampTest_julia` | 2-period sanity test of the new Kondo-ramp mechanism -- confirmed no quench. |
| `FM_jK0.05_env5-40_kondo15_julia`, `AFM_jK0.05_env5-40_kondo15_julia` | **The reference pair.** 50T, both fixes active, J_K=0.05. `compare_figures.py` is hardcoded to this pair. |
| `AFM_jK0.1_env5-40_kondo15_julia`, `AFM_jK0.15_env5-40_kondo15_julia` | Same protocol, J_K sweep on AFM. At 0.15 the flux collapses right at the Kondo-connect line -- strong enough that adiabatic-in-time connection alone doesn't save it. |
| `FM_jK0.05_phase5-35_kondo15_julia` | First use of `--envelope-target phase`. Done. |
| `FM_Jalpha0_theta_env0-8_kondo5_T10`, `FM_Jalpha0_theta_env0-5.2_kondo0.2_T10` | `--J-alpha 0` control runs (Kitaev term off -- isolates drive+Kondo-only currents). Fast T=10 protocol (see below), the template the scans reuse. |
| `AFM_jK{0.02,0.04,0.06,0.08,0.09,0.10,0.11,0.12}_theta_env0-5.2_kondo0.2_T10_scan` | **J_K scan, complete (8/8).** AFM, `J_I=0.0025` fixed, fast T=10 protocol (`--t-on-periods 0 --t-off-periods 5.2 --kondo-ramp --t-on-kondo-periods 0.2`). Aggregated by `jk_scan_heatmap.py` -> `img_coupled/summary/jk_scan_wp_heatmap.png`. Flux stays essentially flat through 0.08, then real decay onset around 0.09-0.12. |
| `AFM_jK0.02_jI{0.001,0.005,0.0075,0.01,0.02}_theta_env0-5.2_kondo0.2_T10_scan` | **J_I scan, complete (6/6).** Same protocol, `J_K=0.02` fixed. `J_I=0.0025` point is *not* a separate tag -- it reuses `AFM_jK0.02_theta_env0-5.2_kondo0.2_T10_scan` (shared anchor with the J_K scan). Aggregated by `ji_scan_heatmap.py` -> `img_coupled/summary/ji_scan_wp_heatmap.png`; the auto-refresh watcher regenerated it after each point and exited cleanly once all 6 landed. |
| `FM_heisenberg_jK0.05_theta_env0-5.2_kondo0.2_T10` | **Done.** First use of `--hamiltonian heisenberg` (isotropic Heisenberg on the same 16-bond lattice, `hamiltonian_heisenberg()` in `run_coupled.py`; `J=(1,1,1)` is FM, verified numerically -- ground energy exactly -16.0 for 16 bonds). `J_alpha=0.1` here, same as every Kitaev run -- **not** zero (don't confuse this with the `J_alpha=0` control runs above). Results: `<W_p>` stays ~0 throughout and `E_N` stays exactly 0, `dE=-1.3e-2` (real dynamics, confirmed in the current/dashboard figures). `W_p` isn't a meaningful observable for Heisenberg -- it's a Kitaev-specific flux operator with nothing in the isotropic Heisenberg Hamiltonian to excite or protect it, so it reading near-zero is *not* evidence the coupling is off, just that it's the wrong lens for this Hamiltonian; current/spin observables are what actually reflect the dynamics here. |

`julia_calib`, `julia_smoketest`, `smoke` are throwaway calibration/smoke
runs, safe to ignore or delete.

## Wp-vs-current analysis

`wp_current_scaling.py` reads both the J_K and J_I scans (via `_kondo_window`
and `load_run`) and extracts, per run: a decay rate `dWp/dt` (least-squares
slope of `<W_p>` over the Kondo-connected-and-precessing window) and current
summary stats (peak `|I^Salpha_L|`, FFT power at `omega0`/`2*omega0`).
Attempts a scaling-collapse fit `I ~ J_K^a * J_I^b`, fitting `a` and `b`
*separately* from each sweep's own log-log slope and normalizing through the
shared `(J_K=0.02, J_I=0.0025)` anchor point (a single 2D regression on the
combined ~14 points would be poorly conditioned with only two 1D sweeps).
Writes `img_coupled/summary/wp_current_scaling.csv` and
`_{raw,collapse}.png`. The anchor point loads identically from both sweeps by
construction -- a free correctness check on the script (verified: the two
rows are byte-identical). **Re-run against the now-complete 8/8 J_K + 6/6
J_I data (2026-09-03, chained automatically off the last J_I point's PID) --
`wp_current_scaling.csv` has all 14 rows, both sweeps show current growing
monotonically with their respective coupling.** Results not yet reviewed
in detail with the user -- check `img_coupled/summary/wp_current_scaling_
{raw,collapse}.png` before treating the scaling-collapse fit as final.

## Spectrogram / compare_spin_current (paper figure)

`fig_spectrogram` ((omega/omega0, t) heatmap of `|FFT(I^Salpha)|^2`, 3 panels
x/y/z) is smoothed for *readability*, not more information -- true
time/frequency resolution is still set by `win_periods`/`dt`:
Gaussian STFT window, 80-90% overlap, `nfft=4*nperseg` zero-padding, and
`imshow(..., interpolation='bicubic')` instead of `pcolormesh`. **Floor the
array to the color norm's `vmin` before `imshow`** -- `LogNorm` masks exact
zeros (common pre-Kondo-connect), and bicubic interpolation across masked/NaN
cells renders as salt-and-pepper speckle instead of clean black. The
computation/drawing is factored into `_spectrogram_specs`/
`_plot_spectrogram_panel` in `run_coupled.py` specifically so `fig_spectrogram`
and `compare_figures.py` share it rather than duplicating.

**Gaussian window width (`gauss_std_frac`, in `_spectrogram_specs`) default
is `0.5`** (both there and in `fig_spectrogram`'s own forwarding default),
changed from the original `1/8` this session after an explicit visual
comparison (`gauss_std_frac` in `{0.125, 0.25, 0.5, 1.0}`, one panel each) --
`0.5` gave noticeably sharper/better-resolved harmonic bands than `0.125`
without the fine vertical striping (spectral-leakage/sidelobe ringing) that
showed up at `1.0`. Mechanism: `gauss_std_frac` sets the taper's std as a
fraction of `nperseg`; small values roll off fast (using less of the window,
blurring frequency content but suppressing sidelobes), values approaching 1
flatten the taper toward rectangular (sharper bands, more leakage). This is
a real, user-confirmed tuning choice, not an arbitrary default -- don't
revert it without checking.

`compare_figures.fig_compare_spin_current(models, path, w_max=20.0)` is the
**paper figure** -- built to that standard, don't casually simplify it back
down. `models` is a list of `(label, out, meta)` tuples, one row each -- 1
row for a single model (Heisenberg alone), 2 for an AFM/FM pair, or more
(AFM+FM+Heisenberg together) for a combined comparison; this generalized
form replaced an earlier hardcoded-AFM/FM-only signature and a separate
duplicate single-row function this session. Layout: N rows (one per model)
x 4 columns -- col 0 is the full `I^Salpha(t)` time series (context), cols
1-3 are the spectrogram heatmap per alpha=x,y,z. Model is the row (not
column) specifically so the omega/omega0 axis can be shared across a row's
3 adjacent spectrogram columns: **only col 1 (the leftmost spectrogram
column) keeps omega/omega0 tick labels/ylabel**, cols 2-3 hide them
(`tick_params(labelleft=False)` + blank ylabel) since the range is always
the fixed `[0, w_max]` regardless of model -- nothing is lost. Each alpha's
heatmap uses a per-alpha monochrome colormap (`ALPHA_CMAP` in
`compare_figures.py`, black -> `C_AXIS[alpha]` -> white) instead of a shared
`inferno`, so the row-0 legend's x=blue/y=green/z=red colors directly tell
you which column is which -- this is why the per-panel alpha titles could be
dropped (`_plot_spectrogram_panel`'s `title` arg is `None` for these calls;
it now accepts `title=None`/`""` to skip `set_title` entirely, and a `cmap`
kwarg, both added this session, default-compatible so `fig_spectrogram`'s
own calls are unaffected). Each ROW gets its own color normalization and
colorbar (current magnitudes differ by orders of magnitude across models,
so one shared scale would flatten the smaller ones -- same reasoning as
col 0's independent y-axes).

**x-axis (time) sharing is grouped by matching run duration, not blanket
`sharex=True` across every row.** AFM/FM use the same 50T envelope protocol
(`t[-1]` matches), so they share one x-axis and only the bottom-of-group row
carries tick labels/xlabel, as before. But the Heisenberg run uses the much
shorter fast T10 protocol (~6300 hbar/gamma vs. AFM/FM's ~31400) -- sharing
x blindly across all three rows (an earlier attempt this session) squashed
the Heisenberg row into an unreadable sliver at the left edge of the long
AFM/FM range. Fixed by grouping rows by `t[-1]` (1% relative tolerance),
linking x-axes only within a group (`ax[r,c].sharex(ax[group_base,c])`), and
giving each group's own bottom row its own tick labels/xlabel. This is a
real, confirmed-by-rendering fix, not speculative -- if a 4th model with yet
another duration gets added, it'll automatically get its own ungrouped
x-axis rather than needing special-casing.

Wrapped in `plt.rc_context({...larger fonts...})` scoped to just this
figure (doesn't touch the global `PAPER_RC` other figures use). Panel
letters sit above the frame (`xy=(0.01, 1.05)`) on every row including 0,
matching the spectrogram rows. `_legend_top` and `_panel_letter` in
`run_coupled.py` gained optional `fontsize`/`handlelength` kwargs
(default-compatible) so this figure can override them without touching
other callers.

**`_cap_drive_marks` (local to `compare_figures.py`) is row-0-only, and
that's load-bearing, not a stylistic choice.** `_tidy` draws the drive/Kondo
axvlines full-height in axes-fraction (0-1) via a blended transform that
auto-follows the axis's ylim. Row 0's col-0 axis gets its ylim explicitly
fixed by `_legend_top` (which also disables autoscale for that axis) *before*
`_cap_drive_marks` runs, so redrawing its markers with a fixed data-coordinate
`ymax` is safe -- `y_cap` is captured dynamically as `ax[0,0].get_ylim()[1]`
*before* `_legend_top` pads it, rather than a hardcoded value, so this keeps
working regardless of which model ends up in row 0. **Other rows' axes are
never explicitly fixed** (they have no legend to reserve space for) --
autoscale is still live when `_cap_drive_marks` would run, so a fixed
`ymax` gets treated as real data and blows that row's y-range out to that
value, squashing its actual signal flat. Real incident, not
hypothetical -- confirmed by rendering it. If another row ever needs the
same treatment, fix its ylim explicitly first (e.g. via its own
`_legend_top` or an explicit `set_ylim`) before calling `_cap_drive_marks`
on it.

## GMN / Mosek -- read before running

A single GMN solve (64-dim, 31 bipartition constraints) can apparently
approach or exceed **~300-700+ GB RSS on its own**. Confirmed by two
incidents: one prior single-model job (9 solves reusing one model) peaked at
~742 GB; an attempt at 4 *concurrent* single-solve jobs OOM-killed one worker
within under a minute and left the other 3 ballooning past 947 GB combined,
filling all 63 GB of swap and nearly taking down a co-running
`run_coupled.py` job. **This machine has 1 TiB total RAM. `GMN_WORKERS=1`
(strictly sequential, one Mosek instance at a time) is the only configuration
confirmed safe.** The full incident writeup and reasoning is in the comment
block at the top of `gmn_para.jl` -- read it before changing `GMN_WORKERS`.

`gmn_para.jl` is currently configured (`RUNS` constant) for the FM/AFM
`jK0.05` reference pair, `GMN_ONLY` env var lets you restrict to just one
name. Export a run's rho snapshots first with `--replot --export-gmn
--no-figures` (writes into `data_coupled/GMN_results/`) before GMN.jl can use
them.

## Logarithmic negativity formula (verified correct)

`run_coupled.log_negativity(psi, split)` (`2.0 * np.log2(s.sum())` where `s`
are the singular values of the Schmidt-reshaped `psi`) was checked against
the standard pure-state closed form this session and is correct:
`E_N = log2||rho^T_B||_1 = 2*log2(sum_i sqrt(lambda_i))` for Schmidt
coefficients `lambda_i` (Vidal & Werner, PRA 65, 032314, 2002). The SVD
singular values of the reshaped Schmidt matrix `T` are already
`sqrt(lambda_i)` (since `rho_A = T T^dagger`), so `s.sum()` directly gives
`sum_i sqrt(lambda_i)` without ever forming or diagonalizing `rho_A` -- the
~5 ms shortcut the code's docstring references. This is algebraically the
same quantity `dashboard.calculate_log_negativity_pure_bipartite` computes
the slower way (explicit `ptrace` + `eigenenergies()` + sqrt + sum), just
via SVD instead. Preconditions (both satisfied by how `run_coupled` uses
it): `psi` must be normalized (true throughout -- `psi0` is explicitly
normalized and the stepper only ever applies the norm-preserving unitary
`expm_multiply(-1j*dt*H, psi)`, never renormalizes because nothing drifts),
and the bipartition must match `make_reducer`'s `split` (`hex_sites` vs.
rest, by construction).

## Known benign noise

Every `--backend julia` run prints a `SciMLBasePythonCallExt`/`PythonCall`
`ArgumentError` during Julia package precompilation. This is a missing
optional SciMLBase<->PythonCall extension, unrelated to anything this project
uses -- harmless, every real run has completed fine despite it.

## Operational lessons

- **Never run two `--backend julia` jobs (or two heavy Mosek jobs) at the
  same time** -- see the GMN incident above. Run sequentially.
- Long runs (50T is ~9-10h) should go in `tmux` (or `nohup`) so a dropped
  connection doesn't kill them.
- `--replot-ckpt` (pass the same flags as the live run) rebuilds figures from
  a still-running job's periodically-updated checkpoint, without touching it
  -- use this to check progress instead of waiting for completion.
- If a background job's process disappears from your task tracker but you
  didn't kill it, check `ps` directly before assuming it died -- task-tracker
  bookkeeping has been observed to lose a job's completion record across a
  session restart while the actual OS process kept running fine.
- **Don't chain a queued run with `while pgrep -f "<tag>"; do sleep; done`.**
  A real incident: a wait script wrote itself to disk via a heredoc
  containing that exact tag string, then launched itself via `bash
  <that file>` -- the heredoc text stayed visible in the *parent* shell
  wrapper's own argv, which `pgrep -f` then matched forever, silently
  stalling a queued run for hours. Wait on the specific PID instead
  (`while kill -0 "$PID" 2>/dev/null; do sleep 30; done`).
- **A fixed `axvline`/`vlines` `ymax` on an axis with autoscale still live
  gets treated as real data**, not just a drawing instruction -- it can blow
  the axis's y-range out to that value and flatten the actual signal. Only
  safe once that axis's ylim has been explicitly fixed (e.g. by `set_ylim`,
  which also disables autoscale). See `_cap_drive_marks` above for the
  concrete incident.

## Paper results (`paper_results/`) -- three studies, started 2026-09-08

A top-level folder, separate from `data_coupled/`/`img_coupled/`, for
paper-specific deliverables the user requested. All three studies use
scripts following this project's established conventions
(hardcoded-constants scan scripts, resumable, `--out-dir`/`--img-dir`
redirected into the study's own `data/`/`img/` subfolders, never two
`--backend julia` jobs concurrently).

**Study 1** (`paper_results/study1_updated_protocol/`) -- **DONE**
(2026-09-12). One AFM run + one FM run, new timing protocol: field static
along z, Kondo connects at 1T, drive precesses 5T->15T, run ends at 23T.
`J_K=0.05, J_I=0.014, J_alpha=0.1, omega0=0.001` (10x slower than usual).
`run_study1.sh` launched both (~289,000 steps/run, ~2 days/run, ~4 days
total); `make_figures.py` builds `study1_dashboard`/`study1_spin_current`
(reusing `compare_figures.py`'s functions, `w_max=20`); `refresh_figures.sh`
kept `img/` current every 15 min while it ran (no longer needed, both
finished). **Result**: AFM's `<W_p>` stays essentially flat (~1.05) through
the whole protocol including after precession stops; FM's is much noisier
and dips toward 0 during precession -- a clean qualitative AFM/FM contrast.

**Study 2** (`paper_results/study2_JI_omega0_scan/`) -- **IN PROGRESS,
currently PAUSED, 64/100 done**. A (J_I, omega0) grid, AFM+FM, `J_K=0.05`
fixed, fast-ish timing (Kondo connects 0.5T, precesses 1T->12T, ends at
20T). **The user's original spec (`J_I` 0.005-0.05 step 0.005 = 10 values,
`omega0` 0.001-0.05 step 0.001 = 50 values) had a typo** -- literally as
specified it's 1000 runs / ~22.6M steps / an estimated ~5 months (cost
scales as 1/omega0, the omega0=0.001 tail dominates). Corrected, confirmed
grid: `J_I` unchanged (10 values), **`omega0` = {0.01, 0.02, 0.03, 0.04,
0.05}** (5 values) -- **100 runs total, ~1.15M steps, ~8 days estimated.**
`J_I_VALUES`/`OMEGA0_VALUES` are at the top of `run_study2.py` -- **don't
casually widen the omega0 range/step back toward the original spec without
redoing this cost estimate**, the 1/omega0 scaling makes it non-obvious how
expensive a seemingly-small grid change is.

Scripts: `run_study2.py` (the grid launcher -- resumable, skips any tag
whose final npz already exists, stops on first non-zero exit; `--dry-run`
prints point count/step estimate without running anything), `scan2d_heatmap.py`
(2D `(J_I, omega0)` heatmaps of final-window `<W_p>` and spin-current FFT
power at omega0, AFM/FM side by side, NaN/blank for not-yet-run points),
`make_spectrograms.py` (regenerates the `(omega/omega0, t)` spectrogram at
**`w_max=30`**, since `run_coupled.py`'s own automatic per-run figures use
the project-wide default `w_max=8`), and `dashboard_by_freq.py` -- a
derived analysis figure (not a run launcher) built on top of whatever grid
points exist: N-model-row x 4-col dashboard overlaying 3 omega0 curves
(solid/dashed/dotted = omega0/2*omega0/4*omega0) at one fixed J_I per
model (`--J-I-afm`/`--J-I-fm`, currently 0.015/0.010), `--component
{x,y,z}` picks which S_alpha goes in col 0 (colored via `C_AXIS`; cols
1-3 are always black, linestyle-only). Heavily iterated on with the user
this session -- current state: legend inset in panel (a) upper-right
(labeled omega_0/omega_1=2*omega_0/omega_2=4*omega_0), `J_K=0.05` as inset
text top-left (not in the legend), column titles
`$\langle\hat{S}^\alpha_{tot}\rangle/N$` / `$\langle\hat{\bar{W}}_p\rangle$`
(hat OUTSIDE the bar, not the other way -- got this backwards once) /
`$\mathcal{N}$` / `$\mathcal{N}_{\text{GM}}^{6,6}$` (GMN column always
blank, no GMN.jl solve exists for these runs), x-axis label
`Time/$T_0$ $(\gamma/\hbar)$`. **Only panel (a)'s first dashed (T_on) and
dotted (T_on_kondo) markers are shortened to 88% height** (so they clear
the inset text/legend) -- T_off's marker and every marker in every other
panel run full height; this was a deliberate, twice-corrected choice
(first pass wrongly shortened all dashed lines in all panels), don't
"fix" it back to uniform full-height without checking.

**Launched 2026-09-08 ~19:54 EDT** (Study 1), queued behind it via
`queue_and_run_study2.sh` ~19:56 EDT, PID **1391989** (direct-PID-wait
pattern on Study 1's PID, not `pgrep -f`), with a periodic
`scan2d_heatmap.py`+`make_spectrograms.py` refresh every 15 min. AFM's
half finished first (50/50); FM's half was still running when Study 3 (below)
needed the machine.

**Paused 2026-09-16 ~21:43 EDT via `SIGSTOP`** on the in-flight
`run_coupled.py` PID (was **3070939**, FM jI=0.015 w0=0.05 -- check current
PID with `pgrep -af "run_coupled.py.*scan2d"` since it changes every point)
so Study 3 could run without violating the never-two-julia-jobs rule, with
*zero lost work* (frozen mid-integration, not killed/restarted). The
handoff script (`paper_results/study3_T40_freq_compare/
handoff_pause_study2_run_study3.sh`) automatically sends `SIGCONT` to that
same PID once Study 3 finishes, after which Study 2's own `run_study2.py`
loop continues through its remaining ~35 points on its own, no manual
restart needed. **If resuming cold: check `ps -p <that PID> -o stat` --
`STAT=T` means still correctly paused (this is expected, not a hang);
`STAT=R`/`S` after Study 3 finished means it auto-resumed correctly.**

**Study 3** (`paper_results/study3_T40_freq_compare/`) -- **IN PROGRESS**,
see "Study 3 timing bug and fix" below before touching this study further.
6 runs, a *new* T=40 protocol (longer than Study 2's T=20): field static
along z, drive starts precessing at 5T, Kondo connects at 8T
(already-precessing target, same "connect after precession starts"
principle as Study 1/the original production window), precession stops at
30T, run ends at 40T
(`--t-on-periods 5 --t-on-kondo-periods 8 --t-off-periods 30 --n-periods 40`,
*for the omega0=0.01 reference point only* -- other omega0 need scaled
period-counts, see below). `omega0` in {0.01, 0.02, 0.04}, `J_K=0.05`,
`J_alpha=0.1`. **AFM uses J_I=0.015, FM uses a 10x smaller J_I=0.0015** --
deliberately different per model (FM's interesting/decay regime sits at
much smaller J_I than AFM's, same asymmetry pattern as
`dashboard_by_freq.py`'s AFM=0.015/FM=0.010 default). Purpose: extend the
`dashboard_by_freq.py`-style frequency comparison to this new, longer
protocol -- **but note Study 3's comparison is meant to be in *absolute*
time (same physical window for all three omega0), unlike Study 2's
`dashboard_by_freq.py` which deliberately compares in *periods* (t/T,
different absolute duration per omega0) -- these are different, both
legitimate questions, don't conflate them.** No dedicated figure script
written yet for this study's results -- ask before assuming
`dashboard_by_freq.py` should just be pointed at it, since tag naming
(`..._study3` not `..._scan2d`), the T=40 vs T=20 markers, and the
absolute-vs-periods x-axis choice all differ -- it needs its own
`tag_for`/marker constants and its own x-axis convention, not a blind
reuse.

### Study 3 timing bug and fix (found 2026-09-17)

`run_study3.sh` passed the **same period-counts to every omega0**
(`--t-on-periods 5 --t-off-periods 30 --t-on-kondo-periods 8
--n-periods 40` for all three), but `run_coupled.py` converts periods to
absolute time using *that run's own* `T = 2*pi/omega0`
(`run_coupled.py:1323`, `t_final = a.n_periods * T`). So the three omega0
runs never covered the same physical time window: omega0=0.02 got half
the absolute duration of omega0=0.01, omega0=0.04 got a quarter. This
defeats comparing the three frequencies "in the same time" -- which is
what the user actually wants for Study 3 (contrast with Study 2's
`dashboard_by_freq.py`, which is *correctly* in periods-space, see above).

**Fix**: express every non-reference omega0's periods in units of the
*reference* `T_ref = 2*pi/0.01`, not its own `T` -- i.e. scale the
period-counts by `omega0/0.01` (ratio 1, 2, 4 for omega0 = 0.01, 0.02,
0.04) so the resulting absolute `t_on`/`t_off`/`t_on_kondo`/`t_final` land
on the same physical times for all three (`t_on=3141.6`,
`t_on_kondo=5026.5`, `t_off=18849.6`, `t_final=25132.7`, all in
hbar/gamma). Concretely: omega0=0.01 unchanged (5/30/8/40, 50265 steps);
omega0=0.02 becomes 10/60/16/80 (was 5/30/8/40, 50265 steps instead of the
buggy 25133); omega0=0.04 becomes 20/120/32/160 (was 5/30/8/40, 50265
steps instead of the buggy 12567). **This roughly doubles Study 3's total
compute** (all 6 points now cost the same, ~7.8h each, ~46.8h total,
instead of the original ~27.3h estimate) since omega0=0.02/0.04 now need
as many steps as omega0=0.01 instead of half/a-quarter as many. User
confirmed this tradeoff is worth it (2026-09-17).

By the time the fix was built, the original (buggy) `run_study3.sh` had
already produced: AFM omega0=0.01 (correct, ratio=1, kept), AFM omega0=0.02
and omega0=0.04 (both wrong-window, needed rerun), FM omega0=0.01
(correct, ratio=1, kept), FM omega0=0.02 (wrong-window, finished, needed
rerun), FM omega0=0.04 (wrong-window, was still running when the fix was
ready -- killed mid-run, PID 3319145, no salvageable output). **User chose:
overwrite the wrong AFM/FM omega0=0.02/0.04 npz/figures in place (same
tags), don't keep the old wrong-window data on disk.**

`run_study3_fixed.sh` (new) runs exactly those 4 points (AFM omega0=0.02,
AFM omega0=0.04, FM omega0=0.02, FM omega0=0.04) with the scaled
period-counts, unconditionally (no skip-if-exists check -- it always
overwrites, unlike the grid-scan scripts). `omega0=0.01` for both models
is *not* rerun by this script (already correct). The old
`run_study3.sh`/`handoff_pause_study2_run_study3.sh` pair was killed
(PIDs 3077423, 3077417) rather than left to finish, since letting it
proceed would have launched more wrong-window runs and prematurely
SIGCONT'd Study 2 before the fix-up was done. Superseded by
`handoff_run_fixed_study3_resume_study2.sh`, which runs
`run_study3_fixed.sh` then does the same Study-2-resume handoff the
original script would have. **Launched 2026-09-17 ~23:06 EDT**, detached
(`nohup ... & disown`), logging to `handoff_fixed.log`. If resuming cold:
check `pgrep -af run_coupled.py` for the current point and
`handoff_fixed.log` for progress; if nothing is running and Study 3's 4
tags aren't all present with the corrected step counts (50265 steps each,
check via `_meta.json`'s `t_final`/`n_steps` or the npz), rerun
`run_study3_fixed.sh` directly (safe to rerun -- it's short, 4 points,
no skip logic to fight).

If resuming this cold and the PIDs above are gone: check whether
`paper_results/*/data/*_coupled.npz` files exist for the expected tags
before assuming failure. Order of trust: Study 1 (done, ignore), Study 3
(check its 6 tags, resume via `run_study3.sh` if incomplete and nothing is
running), Study 2 (check its PID's `STAT` -- if truly dead rather than
paused, resume via `queue_and_run_study2.sh` or `run_study2.py` directly).

## Current state (as of last update)

- **Study 3: DONE (6/6), including the fix.** `run_study3_fixed.sh`
  finished cleanly (`rc=0`) at 2026-09-19 ~06:10 EDT -- all 4 corrected
  points (AFM/FM omega0=0.02/0.04) now share the same absolute time window
  as the omega0=0.01 reference (`t_final=25132.7`, 50265 steps each, all
  6 tags confirmed). Figures regenerated automatically for the 4 rerun
  points. `handoff_run_fixed_study3_resume_study2.sh` then sent Study 2
  its `SIGCONT` immediately (same timestamp), exactly as designed --
  no manual intervention was needed.
- **Study 2: running again, 86/100 done** as of 2026-09-20 ~23:00 EDT
  (was 64/100 when paused). Currently on `FM_jK0.05_jI0.040_w0.020_scan2d`
  (~1h left); 13 points remain after that, all FM, `J_I` in
  {0.040 (w0=0.03/0.04/0.05 left), 0.045, 0.050} across all 5 omega0 for
  the last two. **Estimated ~22h remaining, finishing around
  2026-09-21 ~21:00 EDT.** No dedicated figure script has been pointed at
  Study 3's fixed results yet (see "Study 3 timing bug and fix" above --
  ask before assuming reuse of `dashboard_by_freq.py`).

Everything from the *pre-`paper_results/` era* (the 8-point J_K scan, the
6-point J_I scan, the Heisenberg run, `wp_current_scaling.py` re-run
against the complete data, and the `compare_figures.py`/`run_coupled.py`
plotting overhaul -- model-as-row transpose, `ALPHA_CMAP`, duration-grouped
x-axis sharing, `gauss_std_frac=0.5`) is done and stable; see "Spectrogram /
compare_spin_current" above for that work's detail if it needs revisiting.
None of it is in flux -- all current effort is on the three paper_results
studies above.

**Ad hoc scp snapshots exist but will be stale**: `~/for_scp/paper_results/`
(full PDF+PNG copy of Study 1 + Study 2's `img/`, 940 files/277MB as of
2026-09-16) and `~/for_scp/paper_results/images/` (PNG-only mirror of the
same, 470 files/201MB). Both are manual, one-off copies made on request,
not auto-refreshed -- they don't include Study 3 at all yet. Re-copy
(`cp -r <study>/img/. ~/for_scp/paper_results/<study>/`) if the user asks
for an updated snapshot.

**Next actions once Study 3 finishes:** none scripted yet -- the user will
likely want a `dashboard_by_freq.py`-style frequency-comparison figure for
Study 3's data, but that script is hardcoded to Study 2's tag format/T=20
markers (see "Paper results" above) and will need adapting, not blind
reuse. Ask before assuming the exact figure they want.
