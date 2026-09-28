#!/bin/bash
# run_examples.sh -- reference commands for run_coupled.py.
#
# Nothing in this file runs automatically: every real command below is
# guarded by `if false; then ... fi` so you can `bash run_examples.sh` safely
# (it just prints the "smoke test" section) and copy-paste whichever block
# you actually want, or flip its `false` to `true`.
#
# Everything here assumes you're in this directory:
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

# =============================================================================
# 0. The one thing you can't skip for --backend julia
# =============================================================================
# juliacall resolves to Julia 1.12.7, whose bundled libjulia-internal wants a
# newer GLIBCXX than the system libstdc++. Fix: preload Julia's own
# libstdc++ before it loads the system one. Without this, --backend julia
# fails immediately with a GLIBCXX_3.4.30 error. --backend mock doesn't need
# it (loop/plumbing test only, currents are meaningless).
export LD_PRELOAD=/home/sebasgom/.julia/juliaup/julia-1.12.7+0.x64.linux.gnu/lib/julia/libstdc++.so.6

# =============================================================================
# 1. Smoke test -- no Julia, ~1 min, just checks the loop/plumbing runs
# =============================================================================
python run_coupled.py --backend mock --n-periods 1 --dt 1.0 --tag smoke

# =============================================================================
# 2. A real production run (this is the shape every FM/AFM run in this
#    project has used): FM or AFM Kitaev model, FI drive with a smooth
#    switch-on/off envelope on the cone angle, Kondo coupling connecting
#    smoothly partway through so it attaches to an ALREADY-precessing drive
#    instead of quenching a static one.
# =============================================================================
if false; then
  python run_coupled.py \
    --backend julia --jl-dir . --project /home/sebasgom/TDNEGF \
    --model FM \
    --J-K 0.05 --J-I 0.0025 \
    --n-periods 50 --dt 0.5 \
    --envelope --t-on-periods 5 --t-off-periods 40 --k-ramp 0.1 \
    --kondo-ramp --t-on-kondo-periods 15 --k-ramp-kondo 0.02 \
    --tag FM_jK0.05_env5-40_kondo15_julia

  # then swap --model FM -> --model AFM (and the --tag) to run the other one.
  # Run them one after another, NOT at the same time -- see section 6.
fi

# What each flag means:
#   --model            FM or AFM Kitaev sign structure (see MODELS in the script)
#   --J-alpha           Kitaev coupling (default 0.1)
#   --J-I               FI-drive <-> QSL coupling (default 0.0025)
#   --J-K               QSL <-> NM Kondo coupling -- the one that matters most
#                        for how hard the flux sector gets hit; 0.05 is gentle,
#                        0.15 is enough to collapse it on its own (see the AFM
#                        J_K sweep results)
#   --n-periods N       total run length in units of the drive period T=2*pi/omega0
#   --dt                integrator step (0.5 is what every real run used; try
#                        --dt 1.0 for a faster/rougher look)
#   --envelope           turn on the theta (cone-angle) ramp; see section 4 for
#                        the alternative --envelope-target phase
#   --t-on-periods /
#   --t-off-periods      when the envelope ramps up / back down, in units of T
#   --k-ramp             sigmoid rate of that ramp (bigger = sharper)
#   --kondo-ramp         connect J_K smoothly instead of as a step at t=0
#   --t-on-kondo-periods when J_K crosses J_K/2 -- keep this LATER than
#                        --t-on-periods so Kondo attaches to a precessing
#                        drive, not a static one
#   --k-ramp-kondo       sigmoid rate for the J_K ramp (defaults to --k-ramp)
#   --tag                names every output file; if you don't set it you get
#                        an auto-generated one from model/J_K/omega0/dt

# =============================================================================
# 3. Watching a run that's still going, without touching it
# =============================================================================
# run_coupled.py checkpoints its recorded arrays every 500 steps into
# <out-dir>/<tag>_ckpt.npz while it runs. --replot-ckpt reads that (not the
# final npz) and rebuilds the four figures from wherever the sim currently
# is -- pass the SAME model/J_K/envelope/kondo flags (needed for the
# vertical markers and axis scaling), plus --replot-ckpt:
if false; then
  python run_coupled.py --replot-ckpt \
    --model FM --J-K 0.05 --J-I 0.0025 --n-periods 50 --dt 0.5 \
    --envelope --t-on-periods 5 --t-off-periods 40 --k-ramp 0.1 \
    --kondo-ramp --t-on-kondo-periods 15 --k-ramp-kondo 0.02 \
    --tag FM_jK0.05_env5-40_kondo15_julia
fi

# =============================================================================
# 4. NEW: freezing the PRECESSION instead of the cone angle
# =============================================================================
# --envelope-target picks what --t-on/off-periods and --k-ramp apply to:
#   theta (default) -- the cone angle ramps 0 -> theta0 -> 0, phase always runs
#   phase            -- theta stays FIXED at --theta; instead the precession
#                        phase itself freezes outside [t_on, t_off]. The field
#                        sits static at a fixed tilt (azimuth is whatever
#                        omega0*t_on works out to -- there's no separate phase
#                        offset flag) until t_on, then starts precessing, and
#                        (if t_off is finite) freezes again at t_off.
#
# Example: field starts static at a ~54.7 degree tilt (theta = arccos(1/sqrt3),
# the "[111]-like" cone angle), sits there for 5 periods, then precesses for
# the rest of a 20-period run (t_off left at its default 6 -- OVERRIDE this or
# it'll freeze again after just 1 more period; use a huge --t-off-periods for
# "never re-freeze"):
if false; then
  python run_coupled.py \
    --backend julia --jl-dir . --project /home/sebasgom/TDNEGF \
    --model FM --J-K 0.05 --J-I 0.0025 --n-periods 20 --dt 0.5 \
    --theta 0.9553166181245093 \
    --envelope --envelope-target phase \
    --t-on-periods 5 --t-off-periods 1e9 --k-ramp 0.02 \
    --tag FM_phase_envelope_test
fi

# =============================================================================
# 5. Exporting GMN.jl input from a finished run
# =============================================================================
# Every run records 9 reduced-density-matrix snapshots (--n-gmn, default 9)
# spread across the whole run automatically -- no flag needed to RECORD them.
# --export-gmn writes them out as the _r.txt/_i.txt pairs GMN.jl reads,
# straight from the finished <tag>_coupled.npz (--replot, not a live run):
if false; then
  python run_coupled.py --replot --tag FM_jK0.05_env5-40_kondo15_julia \
    --out-dir ./data_coupled --export-gmn --no-figures
fi
# Then see gmn_para.jl in this same directory for actually running GMN.jl's
# entanglement calculation on the exported files (slow -- hours per point,
# and MEMORY-LIMITED to one Mosek solve at a time on this machine, see the
# big warning comment at the top of that file before touching GMN_WORKERS).

# =============================================================================
# 6. Long runs: background + tmux, and NEVER run two --backend julia jobs
#    at once
# =============================================================================
# A 50-period run takes ~9-10 hours. Two big lessons from doing this for
# real:
#   - Launch inside tmux (or `nohup ... &`) so a dropped connection doesn't
#     kill it: `tmux new -s myrun`, run the command, `Ctrl-b d` to detach,
#     `tmux attach -t myrun` to check back in.
#   - Don't run FM and AFM (or any two --backend julia jobs) AT THE SAME
#     TIME -- run one, then the other, e.g.:
if false; then
  for MODEL in FM AFM; do
    python run_coupled.py \
      --backend julia --jl-dir . --project /home/sebasgom/TDNEGF \
      --model "$MODEL" --J-K 0.05 --J-I 0.0025 --n-periods 50 --dt 0.5 \
      --envelope --t-on-periods 5 --t-off-periods 40 --k-ramp 0.1 \
      --kondo-ramp --t-on-kondo-periods 15 --k-ramp-kondo 0.02 \
      --tag "${MODEL}_jK0.05_env5-40_kondo15_julia"
  done
fi

# =============================================================================
# 7. Comparing a matched FM/AFM pair once both are done
# =============================================================================
# compare_figures.py is hardcoded to FM_TAG/AFM_TAG near its top -- edit those
# two lines to point at whichever pair you just ran, then:
if false; then
  python compare_figures.py
fi
