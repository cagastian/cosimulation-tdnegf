#!/bin/bash
# Study 1 -- updated timing protocol for the paper.
#
# Timeline (in drive periods T = 2*pi/omega0):
#   t=0    field static along z (theta=0), Kondo OFF
#   t=1T   Kondo connects (--kondo-ramp, --t-on-kondo-periods 1)
#   t=5T   drive starts precessing (--t-on-periods 5)
#   t=15T  drive stops precessing, freezes again (--t-off-periods 15,
#          i.e. 5T + 10T of precessing)
#   t=23T  run ends (15T + 8T of static evolution after precession stops)
#
# J_K=0.05, J_I=0.014, J_alpha=0.1, omega0=0.001 (10x slower than this
# project's usual omega0=0.01 -- each run is ~289k steps, an estimated
# ~2 days at this backend's measured throughput; see CONTEXT.md /
# the conversation this was set up in for the estimate). Run strictly
# sequentially -- never two --backend julia jobs at once (see CONTEXT.md's
# "Operational lessons"). Safe to re-run: each run either completes or you
# can resume from its checkpoint by re-running the same command.
set -e
export LD_PRELOAD=/home/sebasgom/.julia/juliaup/julia-1.12.7+0.x64.linux.gnu/lib/julia/libstdc++.so.6

REPO=/home/sebasgom/code/cosimulation
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

COMMON="--backend julia --jl-dir . --project /home/sebasgom/TDNEGF \
  --J-alpha 0.1 --J-K 0.05 --J-I 0.014 --omega0 0.001 --dt 0.5 \
  --n-periods 23 --nm-relax 20 \
  --envelope --t-on-periods 5 --t-off-periods 15 --k-ramp 0.1 \
  --kondo-ramp --t-on-kondo-periods 1 --k-ramp-kondo 0.02 \
  --out-dir $HERE/data --img-dir $HERE/img"

for MODEL in AFM FM; do
  TAG="${MODEL}_jK0.05_jI0.014_w0.001_study1"
  echo "### $MODEL start $(date)"
  python run_coupled.py $COMMON --model $MODEL --tag "$TAG"
  rc=$?
  echo "### $MODEL done rc=$rc $(date)"
done
