#!/bin/bash
# Run one MPI rank's command under perf.
#
# Every rank is counted (task clock, page faults, context switches); the
# ranks named in FREECAM_PERF_RECORD_RANKS are also sampled, user space
# only, at FREECAM_PERF_HZ samples a second.  Output goes to
# FREECAM_PERF_DIR as stat.<rank>.csv and record.<rank>.data, which
# tools/report_pi_cam_perf.py reads.  The rank comes from the launcher's
# environment (PALS on Derecho, PMI elsewhere).
set -euo pipefail

rank=${PALS_RANKID:-${PMI_RANK:-${OMPI_COMM_WORLD_RANK:-0}}}
dir=${FREECAM_PERF_DIR:?FREECAM_PERF_DIR is not set}
hz=${FREECAM_PERF_HZ:-299}
# Counting and sampling start after this many milliseconds, so that the
# model's initialization -- reading inputs, building tables -- stays out of
# a profile that is meant to describe the step.
delay=${FREECAM_PERF_DELAY_MS:-0}
events=task-clock,page-faults,minor-faults,major-faults,context-switches
mkdir -p "${dir}"

record=0
IFS=, read -r -a wanted <<< "${FREECAM_PERF_RECORD_RANKS:-}"
for candidate in "${wanted[@]:-}"; do
  [ "${candidate}" = "${rank}" ] && record=1
done

if [ "${record}" = 1 ]; then
  exec perf stat -x, -D "${delay}" -e "${events}" -o "${dir}/stat.${rank}.csv" -- \
    perf record -D "${delay}" -e cpu-clock:u -F "${hz}" -o "${dir}/record.${rank}.data" -- "$@"
fi
exec perf stat -x, -D "${delay}" -e "${events}" -o "${dir}/stat.${rank}.csv" -- "$@"
