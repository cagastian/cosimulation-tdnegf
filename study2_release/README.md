# Study 2 + Wp/current scaling

Coupled QSL(Kitaev honeycomb)-NM(TDNEGF ladder) simulation, Kondo-coupled to
a precessing ferromagnetic-insulator drive. This release bundles two
related pieces of work: **Study 2** (a large (J_I, omega0) grid) and the
earlier, smaller **Wp-vs-current scaling analysis** (varies J_K and J_I
separately, at one fixed omega0) that it builds on conceptually.

## `grid/` -- Study 2: (J_I, omega0) scan

A 10x5 grid, AFM and FM, `J_K=0.05` fixed: `J_I` in {0.005, 0.010, ...,
0.050} (10 values) x `omega0` in {0.01, 0.02, 0.03, 0.04, 0.05} (5 values)
= 100 points. Protocol: field static along z, Kondo connects at 0.5T,
drive precesses 1T->12T, run ends at 20T (T=2*pi/omega0, so shorter
absolute duration at higher omega0 -- unlike Study 3, this scan was never
meant to compare frequencies in absolute time, see `grid/figures/
dashboard_by_freq_*` below for why periods-space is the right comparison
here).

**Status: 99/100 done** as of this release (2026-09-21); the last point
(FM, J_I=0.050, omega0=0.05) was still running. Missing point will need
adding once it lands -- check `paper_results/study2_JI_omega0_scan/data/`
in the working repo for `FM_jK0.05_jI0.050_w0.050_scan2d_coupled.npz`.

`grid/data/`: one `<tag>_meta.json` + `<tag>_coupled.npz` per point (99
currently). `grid/figures/`:
- `scan2d_wp_heatmap.*` -- (J_I, omega0) heatmap of final-window <W_p>,
  AFM/FM side by side.
- `scan2d_current_fft_heatmap.*` -- same grid, spin-current FFT power at
  omega0 instead of <W_p>.
- `dashboard_by_freq_S{x,y,z}_AFM-jI0.015_FM-jI0.010.*` -- AFM (J_I=0.015)
  vs FM (J_I=0.010) dashboards overlaying all 5 omega0 values (linestyle
  = omega0/2*omega0/4*omega0/6*omega0/8*omega0-ish grouping by index) at
  one representative J_I per model. **x-axis is t/T (periods), not raw
  time** -- deliberate: the 5 omega0 runs have different absolute
  durations under this identical periods-based protocol, so periods-space
  is what makes them directly comparable on one axis (contrast with
  Study 3, where all omega0 share one fixed absolute window instead).

## `wp_current_scaling/` -- Wp-decay-rate vs. coupling scaling

Two independent 1D scans (AFM only), both fixed omega0=0.01, fast T10
protocol (`--t-on-periods 0 --t-off-periods 5.2 --kondo-ramp
--t-on-kondo-periods 0.2`):
- **J_K scan** (8 points): `J_K` in {0.02, 0.04, 0.06, 0.08, 0.09, 0.10,
  0.11, 0.12}, `J_I=0.0025` fixed.
- **J_I scan** (6 points): `J_I` in {0.001, 0.005, 0.0075, 0.01, 0.02,
  0.0025} (last is the shared anchor with the J_K scan, not a separate
  run), `J_K=0.02` fixed.

`wp_current_scaling/data/`: 13 unique `<tag>_meta.json` +
`<tag>_coupled.npz` (8 + 6 - 1 shared anchor). `wp_current_scaling/
figures/`:
- `jk_scan_wp_heatmap.*`, `ji_scan_wp_heatmap.*` -- (coupling, t) heatmaps
  of <W_p> across each respective scan. Flux stays essentially flat
  through J_K~0.08, real decay onset around J_K=0.09-0.12.
- `wp_current_scaling_raw.*` -- per-run decay rate dWp/dt (least-squares
  slope over the Kondo-connected-and-precessing window) and current
  summary stats (peak |I^Salpha_L|, FFT power at omega0/2*omega0) vs. the
  scanned coupling.
- `wp_current_scaling_collapse.*` -- scaling-collapse fit `I ~ J_K^a *
  J_I^b`, `a` and `b` fit separately from each sweep's own log-log slope,
  normalized through the shared anchor point (`J_K=0.02, J_I=0.0025`).
  Both sweeps show current growing monotonically with their respective
  coupling. **Not yet reviewed in detail against the final fit** -- treat
  as a working result, not a finished one.
- `wp_current_scaling.csv` -- the underlying per-point table (14 rows,
  the anchor point loads identically from both sweeps by construction).

## `scripts/`

- `run_study2.py` -- the grid launcher (resumable, skips completed tags).
- `scan2d_heatmap.py` -- builds `scan2d_{wp,current_fft}_heatmap`.
- `dashboard_by_freq.py` -- builds `dashboard_by_freq_S{x,y,z}_*`.
- `make_spectrograms.py` -- per-point (omega/omega0, t) spectrograms at
  w_max=30 (not included in this release -- 90+ files; regenerate from
  `grid/data/` if needed).
- `jk_scan_heatmap.py`, `ji_scan_heatmap.py` -- build the two scan
  heatmaps above.
- `wp_current_scaling.py` -- the scaling-fit analysis, reads both scans
  via `_kondo_window`/`load_run`, writes the csv + both scaling figures.

All scripts import from `run_coupled.py` (QSL/NM stepper, style helpers)
in the parent project -- they're not standalone outside that repo.
