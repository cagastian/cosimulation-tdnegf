#!/bin/bash
# Study 3 -- 6 runs, T=40 protocol, frequency comparison at fixed J_I per
# model (AFM=0.015, FM=0.0015), omega0 in {0.01, 0.02, 0.04}, J_K=0.05,
# J_alpha=0.1 (same as every other run in this project).
#
# Timeline (in drive periods T = 2*pi/omega0):
#   t=0    field static along z (theta=0), Kondo OFF
#   t=5T   drive starts precessing (--t-on-periods 5)
#   t=8T   Kondo connects, already-precessing target (--t-on-kondo-periods 8)
#   t=30T  drive stops precessing, freezes again (--t-off-periods 30)
#   t=40T  run ends (--n-periods 40)
#
# Run strictly sequentially -- never two --backend julia jobs at once (see
# CONTEXT.md's "Operational lessons"). Safe to re-run: skips a tag whose
# final npz already exists, so this can be interrupted and restarted
# without losing completed points or duplicating work.
set -e
export LD_PRELOAD=/home/sebasgom/.julia/juliaup/julia-1.12.7+0.x64.linux.gnu/lib/julia/libstdc++.so.6

REPO=/home/sebasgom/code/cosimulation
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

COMMON="--backend julia --jl-dir . --project /home/sebasgom/TDNEGF \
  --J-alpha 0.1 --J-K 0.05 --dt 0.5 --n-periods 40 --nm-relax 20 \
  --envelope --t-on-periods 5 --t-off-periods 30 --k-ramp 0.1 \
  --kondo-ramp --t-on-kondo-periods 8 --k-ramp-kondo 0.02 \
  --out-dir $HERE/data --img-dir $HERE/img"

for SPEC in "AFM:0.015" "FM:0.0015"; do
  MODEL="${SPEC%%:*}"
  JI="${SPEC##*:}"
  for W0 in 0.01 0.02 0.04; do
    TAG="${MODEL}_jK0.05_jI${JI}_w${W0}_study3"
    if [ -f "$HERE/data/${TAG}_coupled.npz" ]; then
      echo "### $TAG already done, skipping"
      continue
    fi
    echo "### $TAG start $(date)"
    python run_coupled.py $COMMON --model "$MODEL" --J-I "$JI" --omega0 "$W0" --tag "$TAG"
    rc=$?
    echo "### $TAG done rc=$rc $(date)"
  done
done
