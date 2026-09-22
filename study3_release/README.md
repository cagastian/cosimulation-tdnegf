# Study 3 -- T=40 frequency comparison

Coupled QSL(Kitaev honeycomb)-NM(TDNEGF ladder) simulation, Kondo-coupled to
a precessing ferromagnetic-insulator drive. This release covers **Study
3**: a frequency comparison at three drive frequencies omega0 in
{0.01, 0.02, 0.04}, for both the AFM and FM Kitaev models, over a common
T=40-period protocol.

## Protocol

Field static along z, then:
- t=5T: drive starts precessing
- t=8T: Kondo connects (already-precessing target)
- t=30T: precession stops, field freezes again
- t=40T: run ends

`J_K=0.05`, `J_alpha=0.1` (both fixed). AFM uses `J_I=0.015`; FM uses a
10x smaller `J_I=0.0015` (FM's interesting/decay regime sits at much
smaller J_I than AFM's).

## The timing fix (2026-09-17)

The first version of this study scaled `t_on`/`t_off`/`t_on_kondo`/`t_final`
by each run's *own* period `T = 2*pi/omega0`, so the three omega0 runs
never actually covered the same absolute time window -- omega0=0.02 got
half the physical duration of omega0=0.01, omega0=0.04 got a quarter. That
defeats the point of comparing frequencies "in the same time."

**Fix**: every omega0's period-counts are scaled by `omega0/0.01` (the
omega0=0.01 run is the reference) so `t_on`, `t_off`, `t_on_kondo`, and
`t_final` land on the *same absolute time* for all three:
`t_on=3141.6`, `t_on_kondo=5026.5`, `t_off=18849.6`, `t_final=25132.7`
(all in hbar/gamma). All 6 data files in `data/` are from the corrected
runs. `scripts/run_study3_fixed.sh` is what produced the 4 points that
needed rerunning after the bug was found (omega0=0.01 for both models was
already correct, ratio=1, and wasn't rerun).

## Results summary

- The fundamental drive peak at omega/omega0=1 (alpha=y, in-plane)
  dominates the pumped spin-current spectrum in every run; the charge
  current stays at the noise floor throughout (no charge pumping).
- Harmonic content sharpens and grows richer as omega0 increases, since
  the same absolute window now contains more drive periods to resolve
  them (up to 4x more at omega0=0.04) -- see `figures/spectra/` and
  `figures/fft_comparison/`.
- AFM's plaquette flux <W_p> is markedly more frequency-selective than
  FM's: at omega0=0.01 AFM's flux barely moves, while 2*omega0 and
  4*omega0 drive deep, slow oscillations down to ~0.3-0.5. FM's flux
  dips substantially at every frequency tested, with much less
  separation between the three -- see `figures/freq_comparison/`.

## Folder structure

```
figures/
  fft_comparison/       AFM-vs-FM spin-current spectrograms, one figure
                         per omega0 (dashboard_fft_study3_w<omega0>.*).
                         Col 0 = <S^alpha_tot>/N, cols 1-3 = (omega/omega0, t)
                         FFT heatmaps of the pumped spin current I^{S_alpha}_L,
                         one per alpha=x,y,z.
  freq_comparison/      AFM-vs-FM, all three omega0 overlaid per panel
                         (solid/dashed/dotted = omega0/2*omega0/4*omega0).
                         2x4: <S^z_tot>/N, <W_p>, log-negativity N, GMN
                         (blank -- no GMN solve exists for Study 3).
  per_run_dashboards/   Full 8-panel dashboard per run (spins, W_p per
                         plaquette, log-negativity, NM polarization,
                         charge/spin current, QSL energy) -- 6 runs.
  spectrograms/         Per-run (omega/omega0, t) spectrogram, all three
                         alpha in one 3-panel figure -- 6 runs.
  spectra/               Static FFT snapshots (Kondo-connected window only),
                         split into precessing/static phase -- 6 runs x 2.
  currents/              Charge + spin current time series -- 6 runs.
  wp_current/            <W_p>-vs-current relation -- 6 runs.

data/
  <tag>_meta.json        Run parameters/provenance (model, J_I, J_K, omega0,
                         timing protocol, etc).
  <tag>_coupled.npz      Full trajectory: t, S, sigma, Wp, Ic, Is, E, E_N, M,
                         theta_t, J_K_t. Load with numpy.load.

scripts/
  dashboard_fft_study3.py            Builds figures/fft_comparison/.
  dashboard_by_freq_study3.py        Builds figures/freq_comparison/.
  run_study3_fixed.sh                Reran the 4 points affected by the
                                     timing bug, with corrected periods.
  run_study3.sh                      Original (buggy) launcher -- kept for
                                     provenance/reference only, not used to
                                     produce anything in this release.
  handoff_run_fixed_study3_resume_study2.sh
                                      Orchestration: ran the fix-up, then
                                     resumed the (unrelated) Study 2 scan
                                     that was paused to free the machine.
```

Six `<tag>` values, `{AFM,FM}_jK0.05_jI{0.015,0.0015}_w{0.01,0.02,0.04}_study3`
(AFM uses J_I=0.015, FM uses J_I=0.0015):

- `AFM_jK0.05_jI0.015_w0.01_study3`
- `AFM_jK0.05_jI0.015_w0.02_study3`
- `AFM_jK0.05_jI0.015_w0.04_study3`
- `FM_jK0.05_jI0.0015_w0.01_study3`
- `FM_jK0.05_jI0.0015_w0.02_study3`
- `FM_jK0.05_jI0.0015_w0.04_study3`

Raw checkpoint files (`*_ckpt.npz`, periodic snapshots of an in-progress
run) and the wide-format CSV export (`*_coupled_wide.csv`, ~103MB/run --
over GitHub's 100MB file limit) are intentionally excluded from this
release; `*_coupled.npz` is the complete, final record of each run.
