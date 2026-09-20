#!/bin/bash
# Build FTorch (Cambridge-ICCS) against the libtorch of this checkout's own torch
# package and install it under build/ftorch, where build_pi_cam_devices.py links it
# into the image so hooks can run bound TorchScript models.
#
#   tools/build_ftorch.sh [<FTorch git ref>]
#
# Environment overrides, for an FTorch that runs models on a GPU:
#   FTORCH_PYTHON      the interpreter whose torch package supplies libtorch (default .venv);
#                      a CUDA torch build gives a CUDA libtorch
#   FTORCH_GPU_DEVICE  FTorch's GPU_DEVICE option, e.g. CUDA (default NONE); a CUDA libtorch's
#                      cmake also needs a CUDA toolkit in the environment (module load cuda)
#   FTORCH_PREFIX      where to install (default build/ftorch; use build/ftorch-cuda for the GPU one)
# The prefix records the libtorch directory it was built against (torch_lib_dir), which the image
# build links and rpaths, so a CUDA image finds the CUDA libtorch and not the checkout's CPU one.
#
# The Fortran side is compiled inside the CESM case environment the image build
# uses (the same Intel compiler: a .mod file written by a newer compiler is not
# readable by an older one), the C++ side by a GCC that speaks C++20, which the
# torch headers require -- CXX in the environment names it when the case
# environment's g++ is too old.  FTorch's CMakeLists pins C++17; it is raised.
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
ref=${1:-main}
src=${repo}/build/ftorch-src
prefix=${FTORCH_PREFIX:-${repo}/build/ftorch}
work=${prefix}-build
python=${FTORCH_PYTHON:-${repo}/.venv/bin/python}
gpu_device=${FTORCH_GPU_DEVICE:-NONE}

# the caller's compiler choice, kept across the case environment (which sets CXX itself)
requested_cxx=${CXX:-}
requested_cc=${CC:-}
source "${repo}/validation/jobs/common.sh"
source "${FREECAM_STATE_CASE}/.env_mach_specific.sh" >/dev/null 2>&1 || true
cxx=${requested_cxx:-g++}
cc=${requested_cc:-${cxx%g++}gcc}
major=$("${cxx}" -dumpversion | cut -d. -f1)
if [ "${major}" -lt 10 ]; then
  echo "CXX=${cxx} is GCC ${major}; the torch headers need C++20 (GCC >= 10). Set CXX to a newer g++." >&2
  exit 2
fi
cxx_runtime=$(realpath "$(dirname "$("${cxx}" -print-file-name=libstdc++.so)")")
if [ ! -e "${cxx_runtime}/libstdc++.so.6" ] && [ -e "${cxx_runtime}/../lib64/libstdc++.so.6" ]; then
  cxx_runtime=$(realpath "${cxx_runtime}/../lib64")     # the 64-bit runtime beside a 32-bit lib/
fi
torch_cmake=$("${python}" -c 'import torch; print(torch.utils.cmake_prefix_path)')
torch_lib=$("${python}" -c 'import torch, pathlib; print(pathlib.Path(torch.__file__).resolve().parent / "lib")')
# the CUDA libraries a CUDA wheel bundles (nvidia/<component>/lib), ahead of the toolkit CMake adds
# to the runpath: the NVRTC found there must be the one whose builtins the run can load
cuda_libs=
if [ "${gpu_device}" != "NONE" ]; then
  cuda_libs=$("${python}" -c 'import torch, pathlib; n = pathlib.Path(torch.__file__).resolve().parent.parent / "nvidia"; print(";".join(str(d) for d in sorted(n.glob("*/lib")) if d.is_dir()) if n.is_dir() else "")')
fi
rpath="${cxx_runtime};${torch_lib}${cuda_libs:+;${cuda_libs}}"

if [ ! -d "${src}/.git" ]; then
  git clone --depth 1 --branch "${ref}" https://github.com/Cambridge-ICCS/FTorch.git "${src}"
fi
sed -i -E 's/^set\(CMAKE_CXX_STANDARD 17\)/set(CMAKE_CXX_STANDARD 20)/' "${src}/CMakeLists.txt"
# FTorch sets its install rpath itself ($ORIGIN and the libtorch directory); a device build
# needs the wheel's CUDA libraries on it too, ahead of the toolkit CMake appends from the link path
sed -i -E 's|^set\(CMAKE_INSTALL_RPATH "\$ORIGIN/\$\{relDir\};\$\{TORCH_LIBRARY_DIR\}"\)|set(CMAKE_INSTALL_RPATH "$ORIGIN/${relDir};${TORCH_LIBRARY_DIR};${FREECAM_EXTRA_RPATH}")|' "${src}/CMakeLists.txt"
grep -q 'FREECAM_EXTRA_RPATH' "${src}/CMakeLists.txt" || { echo "FTorch's CMakeLists no longer sets CMAKE_INSTALL_RPATH as expected; adjust tools/build_ftorch.sh" >&2; exit 2; }
rm -rf "${work}" "${prefix}" && mkdir -p "${work}"
cmake -S "${src}" -B "${work}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_Fortran_COMPILER=ifort -DCMAKE_CXX_COMPILER="${cxx}" -DCMAKE_C_COMPILER="${cc}" \
  -DCMAKE_PREFIX_PATH="${torch_cmake}" \
  -DGPU_DEVICE="${gpu_device}" \
  -DCMAKE_INSTALL_RPATH="${rpath}" -DCMAKE_BUILD_RPATH="${rpath}" -DFREECAM_EXTRA_RPATH="${cuda_libs}" \
  -DCMAKE_INSTALL_PREFIX="${prefix}"
cmake --build "${work}" -j 8
cmake --install "${work}"
# the C++ runtime the library was built with, for the image's own rpath
echo "${cxx_runtime}" > "${prefix}/cxx_runtime_dir"
# the libtorch this FTorch was built against, for the image's own link and rpath
echo "${torch_lib}" > "${prefix}/torch_lib_dir"
echo "${gpu_device}" > "${prefix}/gpu_device"
ls "${prefix}"/lib*/libftorch.so "${prefix}/include/ftorch/ftorch.mod"
echo "Fortran: $(ifort --version | head -1); C++: $("${cxx}" --version | head -1); C++ runtime: ${cxx_runtime}"
echo "FTorch installed under ${prefix}"
