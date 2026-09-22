# Reference AFM/FM pair + Heisenberg comparison

Coupled QSL-NM simulation. This release bundles the project's original
**reference pair** -- the first envelope+Kondo-ramp production runs, which
most of the project's plotting/style conventions were built and validated
against -- together with the **isotropic Heisenberg control** run that
tests whether the Kitaev-specific flux operator <W_p> is doing real work.

## The three runs

- `AFM_jK0.05_env5-40_kondo15_julia`, `FM_jK0.05_env5-40_kondo15_julia`
  -- **the reference pair.** 50T, `J_K=0.05`, `--envelope --t-on-periods 5
  --t-off-periods 40 --k-ramp 0.1 --kondo-ramp --t-on-kondo-periods 15
  --k-ramp-kondo 0.02` -- the standard production window used repeatedly
  early in the project, Kitaev Hamiltonian on both models.
- `FM_heisenberg_jK0.05_theta_env0-5.2_kondo0.2_T10` -- isotropic
  Heisenberg (`J=(1,1,1)`, FM, verified numerically: ground energy exactly
  -16.0 for 16 bonds) on the same 16-bond lattice, fast T10 protocol,
  `J_alpha=0.1` (same as every Kitaev run, *not* zero -- don't confuse
  with the separate `J_alpha=0` control runs elsewhere in the project).

## Result

`<W_p>` stays ~0 throughout the Heisenberg run and `E_N` stays exactly 0,
while `dE=-1.3e-2` shows real dynamics is happening (visible in
`compare_charge_and_diff`/`compare_moments`). This is *expected*, not a
bug: `<W_p>` is a Kitaev-specific plaquette-flux operator with nothing in
the isotropic Heisenberg Hamiltonian to excite or protect, so its reading
near-zero is not evidence the coupling is off -- current/spin observables
are the correct lens for the Heisenberg case, and `<W_p>` is confirmed to
behave sensibly (non-trivial) on the Kitaev models it was designed for.

## Figures

- `compare_dashboard.*` -- AFM-vs-FM reference-pair dashboard (2x4:
  S_tot, W_p per plaquette, log-negativity, GMN).
- `compare_spin_current.*` -- AFM-vs-FM spin-current spectrograms (the
  paper-quality figure this project iterated on most).
- `compare_spin_current_3model.*` -- AFM + FM + Heisenberg together,
  w_max=8 (x-axis grouped by matching run duration -- Heisenberg's T10
  protocol is much shorter than the reference pair's 50T, so it gets its
  own x-axis group rather than being squashed into a sliver).
- `spin_current_heisenberg.*` -- Heisenberg alone.
- `compare_spectrogram_AFM.*`, `compare_spectrogram_FM.*` -- standalone,
  full-size, per-model spectrograms.
- `compare_charge_and_diff.*`, `compare_moments.*` -- charge current /
  lead-difference and higher-moment comparisons, AFM vs FM.

## Data

`data/`: `<tag>_meta.json` + `<tag>_coupled.npz` for all three runs
(33MB, 35MB, 5.8MB -- all included, none near GitHub's 100MB limit).

## Scripts

`scripts/compare_figures.py` builds every figure in this release (and
several more not included here -- see its own docstring). Imports from
`run_coupled.py` in the parent project; not standalone outside that repo.
