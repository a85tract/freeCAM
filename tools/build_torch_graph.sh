#!/bin/bash
# Build libpycam_torch_graph.so into an FTorch installation, beside libftorch.so, where the image
# build links it: the CUDA graph runner (native/pi_cam/support/pycam_torch_graph.cpp) for an
# FTorch built with a GPU device, a stub that reports "no CUDA graphs" for a CPU one.
#
#   tools/build_torch_graph.sh [<FTorch prefix>]    (default build/ftorch; build/ftorch-cuda for the GPU one)
#
# The C++ compiler FTorch itself was built with (its CMake cache, or the GCC beside the C++ runtime
# it records): the library shares FTorch's libtorch and C++ runtime.  TORCH_GRAPH_CXX overrides it.
# A CUDA libtorch's CMake needs a CUDA toolkit in the environment (module load cuda); building
# needs no GPU.  tools/build_ftorch.sh runs this after installing FTorch.
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
prefix=$(realpath "${1:-${repo}/build/ftorch}")
if [ ! -f "${prefix}/lib64/libftorch.so" ] && [ ! -f "${prefix}/lib/libftorch.so" ]; then
  echo "no FTorch installed under ${prefix}" >&2
  exit 2
fi
gpu_device=$(cat "${prefix}/gpu_device" 2>/dev/null || echo NONE)
# the libtorch FTorch was built against (tools/build_ftorch.sh records it); only a CUDA build needs it
torch_lib=$(cat "${prefix}/torch_lib_dir" 2>/dev/null || true)
if [ "${gpu_device}" != "NONE" ] && [ -z "${torch_lib}" ]; then
  echo "${prefix} is a ${gpu_device} FTorch that records no torch_lib_dir: rebuild it with tools/build_ftorch.sh" >&2
  exit 2
fi
cxx_runtime=$(cat "${prefix}/cxx_runtime_dir" 2>/dev/null || true)
work=${prefix}-graph-build

cxx=${TORCH_GRAPH_CXX:-}
if [ -z "${cxx}" ] && [ -f "${prefix}-build/CMakeCache.txt" ]; then
  cxx=$(sed -n 's/^CMAKE_CXX_COMPILER:[A-Z]*=//p' "${prefix}-build/CMakeCache.txt")
fi
if [ -z "${cxx}" ] && [ -n "${cxx_runtime}" ] && [ -x "${cxx_runtime}/../bin/g++" ]; then
  cxx=$(realpath "${cxx_runtime}/../bin/g++")
fi
if [ -z "${cxx}" ]; then
  echo "cannot tell which C++ compiler built the FTorch under ${prefix}: set TORCH_GRAPH_CXX" >&2
  exit 2
fi
cuda=OFF
if [ "${gpu_device}" != "NONE" ]; then
  cuda=ON
  if ! command -v nvcc >/dev/null 2>&1; then
    echo "${prefix} is a ${gpu_device} FTorch: its libtorch's CMake needs a CUDA toolkit (module load cuda)" >&2
    exit 2
  fi
fi
# the CUDA libraries a CUDA wheel bundles (nvidia/<component>/lib), as FTorch's own runpath has them
cuda_libs=
if [ "${cuda}" = ON ]; then
  cuda_libs=$(find "${torch_lib}/../../nvidia" -mindepth 2 -maxdepth 2 -type d -name lib 2>/dev/null | sort | paste -sd ';' -)
fi
rpath="${cxx_runtime}${torch_lib:+;${torch_lib}}${cuda_libs:+;${cuda_libs}}"
rm -rf "${work}" && mkdir -p "${work}"
cmake -S "${repo}/native/pi_cam/support/torch_graph" -B "${work}" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="${cxx}" \
  -DPYCAM_TORCH_GRAPH_CUDA="${cuda}" \
  ${torch_lib:+-DCMAKE_PREFIX_PATH="${torch_lib}/../share/cmake"} \
  -DCMAKE_INSTALL_RPATH="${rpath}" -DCMAKE_BUILD_RPATH="${rpath}" \
  -DCMAKE_INSTALL_PREFIX="${prefix}"
cmake --build "${work}" -j 8
cmake --install "${work}"
echo "${cuda}" > "${prefix}/torch_graph"
echo "libpycam_torch_graph (CUDA graphs ${cuda}, ${cxx}) installed under ${prefix}/lib64"
