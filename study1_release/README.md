# Study 1 -- updated protocol AFM/FM contrast

Coupled QSL(Kitaev honeycomb)-NM(TDNEGF ladder) simulation, Kondo-coupled to
a precessing ferromagnetic-insulator drive. **Study 1** is the first run of
a new timing protocol (later reused, with different absolute timing, as
the template for Studies 2 and 3): field static along z, Kondo connects at
1T, drive precesses 5T->15T, run ends at 23T. `J_K=0.05`, `J_I=0.014`,
`J_alpha=0.1`, `omega0=0.001` (10x slower than the project's other
protocols -- ~289,000 steps/run).

One AFM run and one FM run, otherwise identical parameters. **Done**
(2026-09-12).

## Result

AFM's <W_p> stays essentially flat (~1.05) through the whole protocol,
including after precession stops. FM's is much noisier and dips toward 0
during precession. A clean qualitative AFM/FM contrast -- see
`figures/study1_dashboard.*` and `figures/study1_spin_current.*`.

## Folder structure

```
figures/
  study1_dashboard.*        Combined AFM-vs-FM dashboard (built by
                             make_figures.py, reusing compare_figures.py's
                             functions).
  study1_spin_current.*     Combined AFM-vs-FM spin-current spectrogram
                             (w_max=20).
  per_run/                  Each model's own 6 standard figures (spins,
                             currents, spectrogram, spectrum
                             precessing/static, Wp-vs-current) -- 2 runs
                             x 6 figure types.

data/
  <tag>_meta.json            Run parameters/provenance, 2 files (AFM, FM).

scripts/
  make_figures.py            Builds both combined figures above.
  run_study1.sh               Original launcher.
```

## Data note

**Raw trajectory data (`*_coupled.npz`) is NOT included** -- each run is
143-150MB, over GitHub's 100MB per-file limit (would need git-lfs). Only
the small `*_meta.json` provenance files are here. The full `.npz` files
live in `paper_results/study1_updated_protocol/data/` in the working
repo if needed.
