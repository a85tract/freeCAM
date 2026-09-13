#!/bin/bash
# Build FTorch (Cambridge-ICCS) against the libtorch of this checkout's own torch
# package and install it under build/ftorch, where build_pi_cam_devices.py links it
# into the image so hooks can run bound TorchScript models.
#
#   tools/build_ftorch.sh [<FTorch git ref>]
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
work=${repo}/build/ftorch-build
prefix=${repo}/build/ftorch
python=${repo}/.venv/bin/python

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

if [ ! -d "${src}/.git" ]; then
  git clone --depth 1 --branch "${ref}" https://github.com/Cambridge-ICCS/FTorch.git "${src}"
fi
sed -i -E 's/^set\(CMAKE_CXX_STANDARD 17\)/set(CMAKE_CXX_STANDARD 20)/' "${src}/CMakeLists.txt"
rm -rf "${work}" "${prefix}" && mkdir -p "${work}"
cmake -S "${src}" -B "${work}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_Fortran_COMPILER=ifort -DCMAKE_CXX_COMPILER="${cxx}" -DCMAKE_C_COMPILER="${cc}" \
  -DCMAKE_PREFIX_PATH="${torch_cmake}" \
  -DCMAKE_INSTALL_RPATH="${cxx_runtime}" -DCMAKE_BUILD_RPATH="${cxx_runtime}" \
  -DCMAKE_INSTALL_PREFIX="${prefix}"
cmake --build "${work}" -j 8
cmake --install "${work}"
# the C++ runtime the library was built with, for the image's own rpath
echo "${cxx_runtime}" > "${prefix}/cxx_runtime_dir"
ls "${prefix}"/lib*/libftorch.so "${prefix}/include/ftorch/ftorch.mod"
echo "Fortran: $(ifort --version | head -1); C++: $("${cxx}" --version | head -1); C++ runtime: ${cxx_runtime}"
echo "FTorch installed under ${prefix}"
