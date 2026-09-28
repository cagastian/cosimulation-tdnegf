# cosimulation -- project context

Last updated: 2026-08-31. Written so a fresh agent (or a human) can pick this
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
| `run_coupled.py` | Everything: QSL Hamiltonian, the FI drive (`make_drive`), the coupled stepper (`run_coupled`), all figures, the CLI. This is the file you run. |
| `square_electrons_lattice.jl` | NM layer -- the TDNEGF ladder + stepper interface (Julia). |
| `qsl_nm_bridge.jl` | Bridges the NM ladder to the QSL's honeycomb site indexing. |
| `compare_figures.py` | Four curated AFM-vs-FM comparison figures, reusing `run_coupled.py`'s style helpers. Tags are hardcoded near the top -- edit `FM_TAG`/`AFM_TAG` before running. |
| `dashboard.py`, `mock_data.py` | Older/adjacent tooling; `mock_data.py` writes synthetic `_coupled.npz` files for iterating on figure code without a real run. |
| `gmn_core.jl`, `gmn_para.jl`, `GMN.jl` | Genuine multipartite negativity (GMN) calculation via Mosek/JuMP, copied in from `~/code/qutip_honeycomb/GMN/data_kitaev_test/`. `gmn_core.jl`+`gmn_para.jl` is the real, parallel-capable driver; `GMN.jl` is a 3-line single-point demo, not useful on its own. **Read the memory warning at the top of `gmn_para.jl` before touching `GMN_WORKERS`** -- see "GMN / Mosek" below. |
| `run_examples.sh` | Reference commands with every flag explained. Safe to `bash run_examples.sh` as-is (only a ~90s mock smoke test is live; everything else is `if false; then...fi` guarded). **Start here for how to actually run things.** |
| `data_coupled/` | `<tag>_coupled.npz` (full results), `<tag>_ckpt.npz` (live checkpoint, updated periodically during a run), `<tag>_meta.json`, `<tag>_coupled_wide.csv`, `GMN_results/` (exported rho snapshots for GMN.jl). |
| `img_coupled/` | Four figures per run (`dashboard`, `currents`, `spectrum`, `wp_current`), `.pdf`+`.png`. `img_coupled/summary/` holds `compare_figures.py`'s output. |
| `GMN_calculation_res/` | GMN.jl's output. |

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
| `FM_jK0.05_phase5-35_kondo15_julia` | **In progress** -- see below. First use of `--envelope-target phase`. |

`julia_calib`, `julia_smoketest`, `smoke` are throwaway calibration/smoke
runs, safe to ignore or delete.

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

## Current state (as of last update)

`FM_jK0.05_phase5-35_kondo15_julia` is running (`--backend julia`, PID may
have changed -- check `ps aux | grep run_coupled.py`). Started
2026-08-31 11:29 EDT, 50T, `--envelope-target phase`, `--theta
0.9553166181245093`, `--t-on-periods 5 --t-off-periods 35`, J_K=0.05,
J_I=0.0025, Kondo connects at 15T. Expect completion roughly 9-10h after
start. Once done: figures land in `img_coupled/` automatically; nothing else
is queued behind it.
