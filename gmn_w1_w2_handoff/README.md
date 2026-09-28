# GMN batch handoff -- omega_1 (0.02) and omega_2 (0.04)

Genuine multipartite negativity (GMN) for Study 3's frequency-comparison
runs (`paper_results/study3_T40_freq_compare/` in the main repo), split
across two machines to parallelize safely:

- **This machine (wherever this folder ends up)**: the omega_1/omega_2
  third of the batch -- AFM at J_I=0.015, omega0 in {0.02, 0.04}, and FM
  at J_I=0.0015, omega0=0.04. **27 solves total** (3 runs x 9 time-point
  snapshots each).
- **The original machine**: kept AFM omega0=0.01 (J_I=0.015) running
  there, already partway through as of this handoff.

**Why two machines instead of one with more workers:** a single GMN solve
for this 64-dim/31-bipartition problem can approach or exceed ~700GB RSS
on its own (see the safety comment at the top of `scripts/gmn_para.jl`) --
running multiple solves *concurrently on one machine* already caused a
near-OOM incident once (documented in that same comment). Two separate
physical machines don't share a RAM pool, so one strictly-sequential
(`GMN_WORKERS=1`) batch per machine is exactly as safe as running one
alone, just parallelized across hardware instead of within it. **Do not
raise `GMN_WORKERS` above 1 on either machine.**

## What's here

```
GMN_results/    54 files -- 3 runs x 9 snapshots x [real, imag] parts,
                already exported from the original run's coupled.npz via
                run_coupled.py --replot --export-gmn --no-figures.
                Filenames: {tag}_{j}_1_theta0.393_[ri].txt, j=1..9, where
                tag is one of:
                  AFM_jK0.05_jI0.015_w0.02_study3   (omega_1 = 2*omega_0)
                  AFM_jK0.05_jI0.015_w0.04_study3   (omega_2 = 4*omega_0)
                  FM_jK0.05_jI0.0015_w0.04_study3   (omega_2, FM)
scripts/
  gmn_core.jl   Computational core (JuMP/Mosek model build + solve).
                Unmodified from the main repo.
  gmn_para.jl   Driver -- trimmed to exactly these 3 runs (see its own
                header). Unlike the main repo's copy, no GMN_ONLY
                filtering is needed since every entry in its ALL_RUNS has
                its data included here.
```

## Requirements on the target machine

- Julia (this project uses 1.12.x) with **JuMP**, **Mosek**, and
  **MosekTools** installed in whatever environment gets activated.
- **A valid Mosek license reachable from this machine.** If the license
  is node-locked to the original machine, it will NOT validate here --
  check this before running anything. A floating/network license server
  works fine as long as this machine can reach it.

## Running it

From this folder:

```bash
mkdir -p GMN_calculation_res
GMN_WORKERS=1 GMN_THREADS=16 GMN_CHUNKS=9 \
  GMN_IN_DIR="$(pwd)/GMN_results" \
  GMN_OUT_DIR="$(pwd)/GMN_calculation_res" \
  julia scripts/gmn_para.jl
```

Run it detached (`nohup ... &`, or `tmux`/`screen`) -- at the observed
rate on the original machine (~2h45m/solve for this problem size), 27
solves is **~74 hours (~3.1 days)**, though this machine's actual per-core
speed may differ.

Progress checkpoints land in `GMN_calculation_res/partial_<name>j<k>-<k>.txt`
as each solve finishes (one line: `<j>  <GMN value>`) -- safe to inspect
mid-run. Final per-run results land in `GMN_calculation_res/<name>GMN.txt`
once every chunk for that run has finished.

## Bringing results back

Once done, copy `GMN_calculation_res/*.txt` (both the final `*GMN.txt`
files and any `partial_*.txt`, small text files) back to the original
machine's `paper_results/study3_T40_freq_compare/GMN_calculation_res/`
directory to merge with the omega0=0.01 results computed there.
