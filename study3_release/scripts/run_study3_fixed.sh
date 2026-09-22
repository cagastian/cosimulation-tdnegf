#!/bin/bash
# Study 3 fix-up -- reruns the 4 points whose absolute time window was wrong.
#
# The original run_study3.sh passed the SAME period-counts (--t-on-periods 5
# --t-off-periods 30 --t-on-kondo-periods 8 --n-periods 40) to every omega0,
# but run_coupled.py converts periods to absolute time using *that run's
# own* T = 2*pi/omega0 (run_coupled.py:1323). So the three omega0 runs never
# covered the same physical time window -- omega0=0.02 got half the absolute
# duration of omega0=0.01, omega0=0.04 got a quarter. That defeats the
# purpose of comparing the three frequencies "in the same time."
#
# Fix: scale the period-counts by (omega0 / 0.01) -- i.e. express everything
# in periods of the REFERENCE T_ref = 2*pi/0.01, not the run's own T. omega0
# =0.01 is unchanged (ratio 1, already correct, not rerun here); 0.02 and
# 0.04 get their period-counts doubled/quadrupled so t_on, t_off,
# t_on_kondo, t_final all land on the same absolute times as the omega0=0.01
# run (t_on=3141.6, t_on_kondo=5026.5, t_off=18849.6, t_final=25132.7 for
# all three). This roughly doubles this fix-up's total compute vs. the
# original (buggy) plan, since omega0=0.02/0.04 now need as many steps as
# omega0=0.01 instead of half/a-quarter as many.
#
# Reruns, unconditionally (overwriting in place, same tags as the original
# run_study3.sh -- deliberate, per user instruction):
#   AFM  jI=0.015  omega0=0.02  (was t_final=12566.4, 25133 steps -- now 25132.7, 50265 steps)
#   AFM  jI=0.015  omega0=0.04  (was t_final=6283.2,  12567 steps -- now 25132.7, 50265 steps)
#   FM   jI=0.0015 omega0=0.02  (was t_final=12566.4, 25133 steps -- now 25132.7, 50265 steps)
#   FM   jI=0.0015 omega0=0.04  (was t_final=6283.2,  12567 steps -- now 25132.7, 50265 steps)
# NOT rerun: AFM/FM omega0=0.01 -- ratio=1, already correct as originally run.
set -e
export LD_PRELOAD=/home/sebasgom/.julia/juliaup/julia-1.12.7+0.x64.linux.gnu/lib/julia/libstdc++.so.6

REPO=/home/sebasgom/code/cosimulation
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

COMMON="--backend julia --jl-dir . --project /home/sebasgom/TDNEGF \
  --J-alpha 0.1 --J-K 0.05 --dt 0.5 --envelope --k-ramp 0.1 \
  --kondo-ramp --k-ramp-kondo 0.02 \
  --out-dir $HERE/data --img-dir $HERE/img"

# t_on_periods t_off_periods t_on_kondo_periods n_periods, all in units of
# THIS run's own T -- pre-scaled by (omega0/0.01) so the absolute times
# match the omega0=0.01 reference (t_on=5T_ref, t_off=30T_ref,
# t_on_kondo=8T_ref, n_periods=40T_ref).
for SPEC in "AFM:0.015" "FM:0.0015"; do
  MODEL="${SPEC%%:*}"
  JI="${SPEC##*:}"
  for POINT in "0.02:10:60:16:80" "0.04:20:120:32:160"; do
    W0="${POINT%%:*}"
    REST="${POINT#*:}"
    T_ON="${REST%%:*}"; REST="${REST#*:}"
    T_OFF="${REST%%:*}"; REST="${REST#*:}"
    T_ON_KONDO="${REST%%:*}"
    N_PERIODS="${REST#*:}"
    TAG="${MODEL}_jK0.05_jI${JI}_w0.${W0#0.}_study3"
    echo "### $TAG start $(date) (t_on=${T_ON}T t_off=${T_OFF}T t_on_kondo=${T_ON_KONDO}T n_periods=${N_PERIODS}T, T=2*pi/${W0})"
    python run_coupled.py $COMMON --model "$MODEL" --J-I "$JI" --omega0 "$W0" \
      --t-on-periods "$T_ON" --t-off-periods "$T_OFF" \
      --t-on-kondo-periods "$T_ON_KONDO" --n-periods "$N_PERIODS" \
      --tag "$TAG"
    rc=$?
    echo "### $TAG done rc=$rc $(date)"
  done
done
