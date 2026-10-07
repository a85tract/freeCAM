// One forward of a TorchScript model FTorch has loaded, captured as a CUDA graph and replayed.
//
// A bound model's forward is hundreds of small kernels, each prepared by TorchScript's
// interpreter and launched by the CPU one at a time; at 27 columns a call that costs more than
// the GPU's arithmetic, and with 32 processes on one GPU the kernels queue at its scheduler.
// A graph records the forward's kernels once and replays them with one launch.
//
// The hook keeps its inputs and its packed output in arrays of fixed shape and address (a
// rank's batch), so the graph's inputs are device copies of them made once, refilled from the
// host before each replay, and its output is copied back after.  The device inputs keep the
// shapes and the column-major strides FTorch gives the same arrays (torch_tensor_from_array),
// so the model sees exactly what it sees without the graph.  Before a graph is used it is
// replayed once against an ordinary forward on the same inputs and refused unless the two
// answers agree bit for bit.
//
// Built against a CUDA libtorch with PYCAM_TORCH_GRAPH_CUDA defined; built without it, every
// entry reports that this build has no CUDA graphs (status 1), and the hook keeps the ordinary
// forward.  No floating-point work of its own: copies, a replay, a comparison.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#ifdef PYCAM_TORCH_GRAPH_CUDA
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/script.h>
#include <memory>
#endif

namespace {

// status codes, shared with the hooks module
constexpr int kOk = 0;
constexpr int kUnavailable = 1;       // this build has no CUDA graphs
constexpr int kBadArguments = 2;
constexpr int kCaptureFailed = 4;     // the forward could not be captured (a host sync, an error)
constexpr int kDiffers = 5;           // the replayed answer is not the ordinary forward's
constexpr int kShapesChanged = 6;     // the arrays are not the ones the graph was made for
constexpr int kRunFailed = 7;

thread_local std::string last_message;

int fail(int status, const std::string& message) {
  last_message = message;
  return status;
}

std::vector<int64_t> column_major_strides(const std::vector<int64_t>& sizes) {
  std::vector<int64_t> strides(sizes.size());
  int64_t step = 1;
  for (size_t i = 0; i < sizes.size(); ++i) {
    strides[i] = step;
    step *= sizes[i];
  }
  return strides;
}

// the shapes of nin arrays, ndim[i] extents each, flattened in order
bool read_shapes(int nin, const int* ndim, const int64_t* shapes, std::vector<std::vector<int64_t>>& out) {
  if (nin <= 0 || ndim == nullptr || shapes == nullptr) return false;
  out.clear();
  size_t at = 0;
  for (int i = 0; i < nin; ++i) {
    if (ndim[i] <= 0) return false;
    std::vector<int64_t> sizes(shapes + at, shapes + at + ndim[i]);
    for (int64_t extent : sizes)
      if (extent <= 0) return false;
    out.push_back(sizes);
    at += static_cast<size_t>(ndim[i]);
  }
  return true;
}

#ifdef PYCAM_TORCH_GRAPH_CUDA
struct Runner {
  torch::jit::script::Module* module = nullptr;
  c10::Device device{c10::kCUDA, 0};
  std::vector<std::vector<int64_t>> in_sizes;
  std::vector<int64_t> out_sizes;
  std::vector<torch::Tensor> inputs;    // the graph's inputs, on the device
  torch::Tensor output;                 // its answer, in the graph's memory pool
  std::unique_ptr<at::cuda::CUDAGraph> graph;
  c10::cuda::CUDAStream stream;
  Runner(c10::cuda::CUDAStream s) : stream(s) {}
};

torch::TensorOptions host_options() { return torch::TensorOptions().dtype(torch::kFloat64); }

torch::Tensor host_view(void* data, const std::vector<int64_t>& sizes) {
  return torch::from_blob(data, sizes, column_major_strides(sizes), host_options());
}

// equal element by element, a NaN equal to a NaN in the same place
bool same_answer(const torch::Tensor& a, const torch::Tensor& b) {
  if (a.sizes() != b.sizes() || a.scalar_type() != b.scalar_type()) return false;
  auto both_nan = torch::logical_and(torch::isnan(a), torch::isnan(b));
  return torch::logical_or(torch::eq(a, b), both_nan).all().item<bool>();
}
#endif

}  // namespace

extern "C" {

// 1 when this build captures CUDA graphs, 0 when it is the stub
int pycam_torch_graph_available() {
#ifdef PYCAM_TORCH_GRAPH_CUDA
  return 1;
#else
  return 0;
#endif
}

// the message of the last failure, null-terminated within length bytes; its full length
int pycam_torch_graph_message(char* buffer, int length) {
  if (buffer != nullptr && length > 0) {
    size_t n = std::min(static_cast<size_t>(length - 1), last_message.size());
    std::memcpy(buffer, last_message.data(), n);
    buffer[n] = '\0';
  }
  return static_cast<int>(last_message.size());
}

// Capture one forward of module (an FTorch torch_jit_script_module_t on cuda:device_index) over
// the nin host arrays host_in (float64, column-major, shapes as ndim and shapes give them) into
// the host array host_out.  warmup ordinary forwards first, so the TorchScript executor has
// settled and every lazy initialisation is done before the capture.  Returns the runner, or null
// with status set.
void* pycam_torch_graph_open(void* module, int device_index, int nin, void** host_in, const int* ndim,
                             const int64_t* shapes, void* host_out, int out_ndim, const int64_t* out_shape,
                             int warmup, int* status) {
  *status = kOk;
#ifndef PYCAM_TORCH_GRAPH_CUDA
  (void)module; (void)device_index; (void)nin; (void)host_in; (void)ndim; (void)shapes;
  (void)host_out; (void)out_ndim; (void)out_shape; (void)warmup;
  *status = fail(kUnavailable, "this image's libtorch has no CUDA: no CUDA graphs");
  return nullptr;
#else
  std::vector<std::vector<int64_t>> in_sizes, out_sizes;
  if (module == nullptr || host_in == nullptr || host_out == nullptr || device_index < 0 ||
      !read_shapes(nin, ndim, shapes, in_sizes) || !read_shapes(1, &out_ndim, out_shape, out_sizes)) {
    *status = fail(kBadArguments, "a CUDA graph needs the model, a CUDA device and every array with its shape");
    return nullptr;
  }
  try {
    c10::Device device(c10::kCUDA, static_cast<c10::DeviceIndex>(device_index));
    c10::cuda::CUDAGuard device_guard(device);
    torch::AutoGradMode enable_grad(false);             // as FTorch's forward runs
    auto runner = std::make_unique<Runner>(c10::cuda::getStreamFromPool(false, device.index()));
    runner->module = static_cast<torch::jit::script::Module*>(module);
    runner->device = device;
    runner->in_sizes = in_sizes;
    runner->out_sizes = out_sizes[0];
    std::vector<c10::IValue> args;
    for (int i = 0; i < nin; ++i) {
      // FTorch's own conversion: the host array's shape and strides, copied to the device
      runner->inputs.push_back(host_view(host_in[i], in_sizes[i]).to(device));
      args.emplace_back(runner->inputs.back());
    }
    {
      c10::cuda::CUDAStreamGuard stream_guard(runner->stream);
      for (int k = 0; k < warmup; ++k) (void)runner->module->forward(args);
      runner->stream.synchronize();
      runner->graph = std::make_unique<at::cuda::CUDAGraph>();
      runner->graph->capture_begin();
      runner->output = runner->module->forward(args).toTensor();
      runner->graph->capture_end();
      runner->stream.synchronize();
      // the replay against an ordinary forward on the same inputs, bit for bit
      auto ordinary = runner->module->forward(args).toTensor().clone();
      runner->graph->replay();
      runner->stream.synchronize();
      if (!runner->output.sizes().equals(runner->out_sizes) || !same_answer(ordinary, runner->output)) {
        *status = fail(kDiffers, "the replayed forward does not answer as the ordinary forward does");
        return nullptr;
      }
    }
    last_message.clear();
    return runner.release();
  } catch (const std::exception& error) {
    *status = fail(kCaptureFailed, std::string("capturing the forward failed: ") + error.what());
    return nullptr;
  }
#endif
}

// Refill the graph's inputs from host_in, replay, and copy the answer into host_out; the arrays
// must have the shapes the graph was made for.
int pycam_torch_graph_run(void* handle, int nin, void** host_in, const int* ndim, const int64_t* shapes,
                          void* host_out, int out_ndim, const int64_t* out_shape) {
#ifndef PYCAM_TORCH_GRAPH_CUDA
  (void)handle; (void)nin; (void)host_in; (void)ndim; (void)shapes; (void)host_out; (void)out_ndim; (void)out_shape;
  return fail(kUnavailable, "this image's libtorch has no CUDA: no CUDA graphs");
#else
  auto* runner = static_cast<Runner*>(handle);
  std::vector<std::vector<int64_t>> in_sizes, out_sizes;
  if (runner == nullptr || host_in == nullptr || host_out == nullptr || !read_shapes(nin, ndim, shapes, in_sizes) ||
      !read_shapes(1, &out_ndim, out_shape, out_sizes))
    return fail(kBadArguments, "a replay needs the runner and every array with its shape");
  if (in_sizes != runner->in_sizes || out_sizes[0] != runner->out_sizes)
    return fail(kShapesChanged, "the arrays are not the shapes the graph was captured for");
  try {
    c10::cuda::CUDAGuard device_guard(runner->device);
    c10::cuda::CUDAStreamGuard stream_guard(runner->stream);
    torch::AutoGradMode enable_grad(false);
    for (int i = 0; i < nin; ++i) runner->inputs[i].copy_(host_view(host_in[i], in_sizes[i]));
    runner->graph->replay();
    host_view(host_out, runner->out_sizes).copy_(runner->output);   // waits for the replay
    return kOk;
  } catch (const std::exception& error) {
    return fail(kRunFailed, std::string("replaying the forward failed: ") + error.what());
  }
#endif
}

// Release a runner (its graph, its device inputs and memory pool); null is ignored.
void pycam_torch_graph_close(void* handle) {
#ifdef PYCAM_TORCH_GRAPH_CUDA
  delete static_cast<Runner*>(handle);
#else
  (void)handle;
#endif
}

}  // extern "C"
