#!/bin/bash
# Build the batched-GPU hook plugin (tools/gpu_batch/fcb_plugin.c) against the device build of
# FTorch and the MPI ABI library.  Run inside the case environment on a login node:
#   CC=/path/to/gcc-12/bin/gcc tools/gpu_batch/build.sh [ftorch-root]
# The library lands in build/gpu_batch/fcb_plugin.so; FCB_PLUGIN_SO names it for fcb_binding.py.
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
ftorch=${1:-${FREECAM_FTORCH_ROOT:-$repo/build/ftorch-cuda}}
source "$repo/validation/jobs/common.sh"; source "${FREECAM_STATE_CASE}/.env_mach_specific.sh" >/dev/null 2>&1 || true
mpi_root=${CRAY_MPICH_DIR:?the MPI root (CRAY_MPICH_DIR) must be set}
cc=${CC:-gcc}
mkdir -p "$repo/build/gpu_batch"
"$cc" -O2 -fPIC -shared -std=c11 -Wall -Wextra \
  -DGPU_DEVICE_NONE=0 -DGPU_DEVICE_CUDA=1 -DGPU_DEVICE_HIP=1 -DGPU_DEVICE_XPU=12 -DGPU_DEVICE_MPS=13 \
  -I"$repo/build/ftorch-src/src" -I"$mpi_root/include" \
  "$repo/tools/gpu_batch/fcb_plugin.c" -o "$repo/build/gpu_batch/fcb_plugin.so" \
  -L"$ftorch/lib64" -lftorch -Wl,-rpath,"$ftorch/lib64" \
  -L"$mpi_root/lib-abi-mpich" -l:libmpi.so.12 -Wl,-rpath,"$mpi_root/lib-abi-mpich"
echo "built $repo/build/gpu_batch/fcb_plugin.so"
