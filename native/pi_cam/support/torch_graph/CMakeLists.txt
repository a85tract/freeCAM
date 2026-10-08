# libpycam_torch_graph: one forward of a bound TorchScript model captured as a CUDA graph
# (../pycam_torch_graph.cpp).  With PYCAM_TORCH_GRAPH_CUDA it is built against the CUDA libtorch
# FTorch uses; without it, a stub that answers "no CUDA graphs", so a CPU image links the same
# entries.  Built and installed beside libftorch.so by tools/build_torch_graph.sh.
cmake_minimum_required(VERSION 3.18)
project(pycam_torch_graph LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(PYCAM_TORCH_GRAPH_CUDA "capture CUDA graphs (needs a CUDA libtorch)" OFF)
add_library(pycam_torch_graph SHARED ${CMAKE_CURRENT_SOURCE_DIR}/../pycam_torch_graph.cpp)
if (PYCAM_TORCH_GRAPH_CUDA)
  find_package(Torch REQUIRED)
  target_link_libraries(pycam_torch_graph PRIVATE ${TORCH_LIBRARIES})
  target_compile_definitions(pycam_torch_graph PRIVATE PYCAM_TORCH_GRAPH_CUDA)
endif()
install(TARGETS pycam_torch_graph LIBRARY DESTINATION lib64)
