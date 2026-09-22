#!/bin/bash
# Runs the Study 3 fix-up (run_study3_fixed.sh -- 4 corrected points), then
# resumes Study 2 (SIGCONT on its frozen point) exactly like the original
# handoff_pause_study2_run_study3.sh would have, once everything true is
# actually done. Supersedes that original handoff/run_study3.sh pairing,
# which was killed mid-flight on 2026-09-17 (it had already produced two
# wrong-window points -- FM omega0=0.02 finished, FM omega0=0.04 in
# progress -- because it scaled each run's periods by its OWN omega0
# instead of a fixed 0.01 reference; see run_study3_fixed.sh's header).
#
# Study 2's in-flight point (PID 3070939) has been SIGSTOP'd since
# 2026-09-16 ~21:43 EDT and is untouched by any of this.
set -e
STUDY2_RUN_PID=3070939
STUDY3_DIR=/home/sebasgom/code/cosimulation/paper_results/study3_T40_freq_compare

echo "$(date): running Study 3 fix-up (4 points: AFM/FM omega0=0.02,0.04)"
bash "$STUDY3_DIR/run_study3_fixed.sh"
rc=$?
echo "$(date): Study 3 fix-up finished rc=$rc"

echo "$(date): resuming Study 2 (PID $STUDY2_RUN_PID)"
kill -CONT "$STUDY2_RUN_PID"
ps -p "$STUDY2_RUN_PID" -o pid,stat,etime,cmd
echo "$(date): Study 2 resumed, its own queue script will continue through the rest of the grid"
