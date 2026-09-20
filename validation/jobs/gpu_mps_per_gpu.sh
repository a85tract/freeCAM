#!/bin/bash
# One NVIDIA MPS control daemon per GPU on this node, in the node's default compute mode.
# An MPS server serves at most 48 client contexts, so 128 ranks over four GPUs need four
# servers, each seeing one device; without MPS, one CUDA context per rank does not fit 32
# ranks on a 40 GB device.  Run once per node (mpiexec -ppn 1), with CUDA_VISIBLE_DEVICES
# unset so the daemon sees the node's own GPUs.
#   gpu_mps_per_gpu.sh start | stop
action=${1:?start or stop}
base=${TMPDIR:-/tmp}/freecam-mps-${PBS_JOBID%%.*}
n=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
for k in $(seq 0 $((n - 1))); do
  export CUDA_MPS_PIPE_DIRECTORY=${base}/pipe-${k} CUDA_MPS_LOG_DIRECTORY=${base}/log-${k}
  if [ "${action}" = start ]; then
    mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
    echo "${n}" > "${base}/gpus"
    # the daemon must not keep the launcher's stdio, or the launcher waits for it
    if CUDA_VISIBLE_DEVICES=${k} nvidia-cuda-mps-control -d >/dev/null 2>&1 </dev/null; then
      echo "$(hostname): MPS daemon for GPU ${k} started"
    else
      echo "$(hostname): MPS daemon for GPU ${k} FAILED"
    fi
  else
    servers=$(echo get_server_list | nvidia-cuda-mps-control 2>&1 | tr '\n' ' ')
    disconnects=$(grep -c disconnected "${CUDA_MPS_LOG_DIRECTORY}/server.log" 2>/dev/null || echo 0)
    faults=$(grep -c -i 'error\|fail' "${CUDA_MPS_LOG_DIRECTORY}/server.log" 2>/dev/null || echo 0)
    echo "$(hostname): GPU ${k} servers [${servers}] client disconnects ${disconnects} log faults ${faults}"
    echo quit | nvidia-cuda-mps-control >/dev/null 2>&1
  fi
done
if [ "${action}" = stop ]; then sleep 1; rm -rf "${base}"; fi
exit 0
