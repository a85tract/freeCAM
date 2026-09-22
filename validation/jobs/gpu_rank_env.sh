#!/bin/bash
# Give this rank its node-local GPU, then run the command.  The GPU is the node-local rank
# modulo the node's GPU count, the same spread freecam.physics.native_model.local_gpu_index
# makes.  CUDA_VISIBLE_DEVICES must not arrive from the launching shell: mpiexec would hand
# every node the head node's GPU UUIDs, which match no device elsewhere.
#   mpiexec -n 512 bash gpu_rank_env.sh python -m freecam.pi_cam.cli ...
# Without MPS (the default) the rank names its GPU by index and owns a CUDA context on it.
# When gpu_mps_per_gpu.sh has started one MPS server a GPU on this node, the rank is pointed
# at its GPU's server instead, and names device index 0: an MPS client's CUDA_VISIBLE_DEVICES
# is read against the devices its server has (one), so index k>0 would find no device.
# Either way the model's device index inside the rank is 0.
local_rank=${PMI_LOCAL_RANK:-${SLURM_LOCALID:-${OMPI_COMM_WORLD_LOCAL_RANK:-0}}}
base=${FREECAM_MPS_BASE:-/tmp/freecam-mps-${PBS_JOBID%%.*}}     # the same path the daemons were started with, on every node
n=${FREECAM_GPUS_PER_NODE:-4}
if [ -f "${base}/gpus" ]; then
  n=$(cat "${base}/gpus")
  k=$((local_rank % n))
  export CUDA_VISIBLE_DEVICES=0 CUDA_MPS_PIPE_DIRECTORY=${base}/pipe-${k} CUDA_MPS_LOG_DIRECTORY=${base}/log-${k}
else
  export CUDA_VISIBLE_DEVICES=$((local_rank % n))
fi
# 32 contexts share one 40 GB device: load modules lazily and keep cuBLAS's workspace at 32 MB
# (torch's default reserves 136 MB a context)
export CUDA_MODULE_LOADING=LAZY CUBLAS_WORKSPACE_CONFIG=:4096:8
exec "$@"
