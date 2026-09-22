/* Batched GPU plugin for the compute_uwshcu_inv hook: the cross-rank batching experiment of
 * 2026-09-22 (records validation/pi_cam_pausable_g35-uwshcu-sub-shadow-gpu-batch*_50step.json).
 *
 * The hook's plugin interface is
 *   int plugin(int n_in, void **in_ptrs, int64_t *in_shapes, int n_out, void **out_ptrs, int64_t *out_shapes)
 * with three int64 dims per array (0-padded) and Fortran (column-major) arrays of 16 columns.
 *
 * The ranks that share one GPU form a group.  Each call, every member copies its 16 columns
 * of the 20 inputs into its slot of a node-shared window (leading dimension G*16, so the
 * slots stack along the column axis), the group barriers, the leader runs one forward of
 * the TorchScript model over all G*16 columns on the GPU and writes the packed output into
 * the shared output area, the group barriers again, and every member copies its 16 columns
 * of the output back into the hook's array.  Only the leader ever touches CUDA.
 *
 * Measured in the model, the two rendezvous a step cost far more than the forward they save:
 * the 32 ranks reach the hook up to a tenth of a second apart (their physics is load-
 * imbalanced within a step), so a call waits about 120 ms for the slowest, against 3.86 ms
 * for the model on the rank's own core.  Kept as the reproducible record of that result.
 */
#define _POSIX_C_SOURCE 200809L
#include <mpi.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "ctorch.h"

#define NIN 20
#define NCOL 16
#define NOUT_W 1190

/* dims beyond the column axis of each input: (d1, d2); 0 = absent.  dt is a scalar. */
static const int64_t IN_D1[NIN] = {0, 31, 31, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 31, 30, 30, 0, 0, 30};
static const int64_t IN_D2[NIN] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 57, 0, 0, 0, 0, 0, 0};

static MPI_Comm group_comm = MPI_COMM_NULL;
static MPI_Win win = MPI_WIN_NULL;
static int G = 0, grank = -1, is_leader = 0, initialised = 0;
static int64_t N = 0;                 /* G * NCOL, the batched column count */
static double *shm = NULL;            /* leader's window base: inputs then the output */
static double *in_base[NIN];          /* per input, the start of its batched array */
static double *out_base = NULL;       /* the batched packed output (N, 1190), column-major */
static torch_jit_script_module_t model = NULL;
static torch_device_t dev_type = torch_kCUDA;   /* device_index < 0 at init: the host, for a login-node check */
static int dev_index = 0;
static torch_tensor_t out_tensor = NULL;
static torch_tensor_t zero_t[NIN];
static int skip_input[NIN];           /* inputs the model ignores: fed as a zero device tensor, never copied */
static torch_jit_script_module_t cpu_model = NULL;   /* the like-for-like host reference */
static torch_tensor_t cpu_out_tensor = NULL;
static double cpu_out[NCOL * NOUT_W];

static double t_gather = 0, t_wait_in = 0, t_forward = 0, t_wait_out = 0, t_scatter = 0, t_total = 0;
static long ncalls = 0;

static double now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int64_t input_elements(int j, int64_t ncol) {
  int64_t n = ncol;
  if (IN_D1[j]) n *= IN_D1[j];
  if (IN_D2[j]) n *= IN_D2[j];
  return j == 0 ? 1 : n;   /* dt: one scalar in the leader's slot */
}

/* fcb_init: fcomm is the Fortran handle of the world communicator; gpus_per_node groups the
 * node's ranks by node-local rank modulo that count (1 = one group a node). */
int fcb_init(MPI_Fint fcomm, int gpus_per_node, const char *model_path, int device_index) {
  MPI_Comm world = MPI_Comm_f2c(fcomm), node_comm;
  memset(skip_input, 0, sizeof(skip_input));
  const char *skip = getenv("FCB_SKIP_INPUTS");
  if (skip && *skip) {
    char buf[256]; strncpy(buf, skip, sizeof(buf) - 1); buf[sizeof(buf) - 1] = 0;
    for (char *tok = strtok(buf, ","); tok; tok = strtok(NULL, ",")) { int j = atoi(tok); if (j > 0 && j < NIN) skip_input[j] = 1; }
  }
  int node_rank;
  MPI_Comm_split_type(world, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);
  MPI_Comm_rank(node_comm, &node_rank);
  MPI_Comm_split(node_comm, node_rank % gpus_per_node, node_rank, &group_comm);
  MPI_Comm_size(group_comm, &G);
  MPI_Comm_rank(group_comm, &grank);
  is_leader = (grank == 0);
  N = (int64_t)G * NCOL;

  int64_t total = 0;
  for (int j = 0; j < NIN; j++) total += input_elements(j, N);
  total += N * NOUT_W;
  MPI_Aint bytes = is_leader ? (MPI_Aint)(total * sizeof(double)) : 0;
  double *mine = NULL;
  if (MPI_Win_allocate_shared(bytes, sizeof(double), MPI_INFO_NULL, group_comm, &mine, &win) != MPI_SUCCESS) return 1;
  MPI_Aint qsize; int qdisp;
  MPI_Win_shared_query(win, 0, &qsize, &qdisp, &shm);
  double *p = shm;
  for (int j = 0; j < NIN; j++) { in_base[j] = p; p += input_elements(j, N); }
  out_base = p;
  if (is_leader) {
    memset(shm, 0, (size_t)(total * sizeof(double)));
    dev_type = device_index < 0 ? torch_kCPU : torch_kCUDA; dev_index = device_index < 0 ? -1 : device_index;
    model = torch_jit_load(model_path, dev_type, dev_index, false, false);
    if (!model) return 2;
    int64_t shape[2] = {N, NOUT_W}, strides[2] = {NOUT_W, 1};
    out_tensor = torch_from_blob(out_base, 2, shape, strides, torch_kFloat64, torch_kCPU, -1, false);
    if (!out_tensor) return 3;
    for (int j = 1; j < NIN; j++) {
      if (!skip_input[j]) continue;
      int64_t zshape[3] = {N, IN_D1[j] ? IN_D1[j] : 1, IN_D2[j] ? IN_D2[j] : 1};
      int nd = IN_D2[j] ? 3 : (IN_D1[j] ? 2 : 1);
      zero_t[j] = torch_zeros(nd, zshape, torch_kFloat64, dev_type, dev_index, false);
      if (!zero_t[j]) return 4;
    }
  }
  MPI_Barrier(group_comm);
  initialised = 1;
  return 0;
}

int fcb_group_size(void) { return G; }
int fcb_group_rank(void) { return grank; }

/* copy this rank's 16 columns of input j into its slot.  The batched arrays are row-major
 * (column index slowest): rank r's 16 rows form one contiguous block, and the leader's
 * host-to-device copy is one straight memcpy instead of a strided gather.  The hook's array
 * is Fortran-ordered (16, d1, d2): element (c, l, k) at c + 16*(l + d1*k). */
static void gather_input(int j, const double *src) {
  if (j == 0) { if (is_leader) in_base[0][0] = src[0]; return; }
  if (skip_input[j]) return;
  int64_t d1 = IN_D1[j] ? IN_D1[j] : 1, d2 = IN_D2[j] ? IN_D2[j] : 1, per = d1 * d2;
  double *dst = in_base[j] + (int64_t)grank * NCOL * per;
  for (int64_t c = 0; c < NCOL; c++)
    for (int64_t l = 0; l < d1; l++)
      for (int64_t k = 0; k < d2; k++)
        dst[c * per + l * d2 + k] = src[c + NCOL * (l + d1 * k)];
}

int fcb_plugin(int n_in, void **in_ptrs, int64_t *in_shapes, int n_out, void **out_ptrs, int64_t *out_shapes) {
  (void)in_shapes; (void)out_shapes;
  if (!initialised || n_in != NIN || n_out != 1) return 10;
  double t0 = now();
  for (int j = 0; j < NIN; j++) gather_input(j, (const double *)in_ptrs[j]);
  MPI_Win_sync(win);
  double t1 = now();
  MPI_Barrier(group_comm);
  double t2 = now();
  if (is_leader) {
    MPI_Win_sync(win);
    torch_tensor_t in_t[NIN];
    for (int j = 0; j < NIN; j++) {
      if (skip_input[j]) { in_t[j] = zero_t[j]; continue; }
      int64_t shape[3], strides[3]; int nd;
      if (j == 0) { nd = 1; shape[0] = 1; strides[0] = 1; }
      else if (IN_D2[j]) { nd = 3; shape[0] = N; shape[1] = IN_D1[j]; shape[2] = IN_D2[j]; strides[2] = 1; strides[1] = IN_D2[j]; strides[0] = IN_D1[j] * IN_D2[j]; }
      else if (IN_D1[j]) { nd = 2; shape[0] = N; shape[1] = IN_D1[j]; strides[1] = 1; strides[0] = IN_D1[j]; }
      else { nd = 1; shape[0] = N; strides[0] = 1; }
      in_t[j] = torch_from_blob(in_base[j], nd, shape, strides, torch_kFloat64, dev_type, dev_index, false);
      if (!in_t[j]) return 20 + j;
    }
    torch_jit_module_forward(model, in_t, NIN, &out_tensor, 1, false);
    for (int j = 0; j < NIN; j++) if (!skip_input[j]) torch_tensor_delete(in_t[j]);
    MPI_Win_sync(win);
  }
  double t3 = now();
  MPI_Barrier(group_comm);
  MPI_Win_sync(win);
  double t4 = now();
  double *dst = (double *)out_ptrs[0];
  const double *src = out_base + (int64_t)grank * NCOL * NOUT_W;
  for (int64_t c = 0; c < NCOL; c++)
    for (int64_t w = 0; w < NOUT_W; w++) dst[c + NCOL * w] = src[c * NOUT_W + w];
  double t5 = now();
  t_gather += t1 - t0; t_wait_in += t2 - t1; t_forward += t3 - t2; t_wait_out += t4 - t3; t_scatter += t5 - t4; t_total += t5 - t0;
  ncalls++;
  return 0;
}

/* the host reference: this rank alone runs the same model on its own 16 columns on the CPU,
 * as the FTorch hook does on the host path */
int fcb_cpu_reference(int n_in, void **in_ptrs, void *out_ptr, const char *model_path) {
  if (n_in != NIN) return 10;
  if (!cpu_model) {
    cpu_model = torch_jit_load(model_path, torch_kCPU, -1, false, false);
    if (!cpu_model) return 2;
    int64_t shape[2] = {NCOL, NOUT_W}, strides[2] = {1, NCOL};
    cpu_out_tensor = torch_from_blob(cpu_out, 2, shape, strides, torch_kFloat64, torch_kCPU, -1, false);
  }
  torch_tensor_t in_t[NIN];
  for (int j = 0; j < NIN; j++) {
    int64_t shape[3], strides[3]; int nd;
    if (j == 0) { nd = 1; shape[0] = 1; strides[0] = 1; }
    else if (IN_D2[j]) { nd = 3; shape[0] = NCOL; shape[1] = IN_D1[j]; shape[2] = IN_D2[j]; strides[0] = 1; strides[1] = NCOL; strides[2] = NCOL * IN_D1[j]; }
    else if (IN_D1[j]) { nd = 2; shape[0] = NCOL; shape[1] = IN_D1[j]; strides[0] = 1; strides[1] = NCOL; }
    else { nd = 1; shape[0] = NCOL; strides[0] = 1; }
    in_t[j] = torch_from_blob(in_ptrs[j], nd, shape, strides, torch_kFloat64, torch_kCPU, -1, false);
  }
  torch_jit_module_forward(cpu_model, in_t, NIN, &cpu_out_tensor, 1, false);
  for (int j = 0; j < NIN; j++) torch_tensor_delete(in_t[j]);
  memcpy(out_ptr, cpu_out, sizeof(cpu_out));
  return 0;
}

/* means in milliseconds: gather, wait for the group, forward (leader; others ~0), wait for
 * the result, scatter, total; then the call count */
void fcb_stats(double *out7) {
  double n = ncalls ? (double)ncalls : 1.0;
  out7[0] = t_gather / n * 1e3; out7[1] = t_wait_in / n * 1e3; out7[2] = t_forward / n * 1e3;
  out7[3] = t_wait_out / n * 1e3; out7[4] = t_scatter / n * 1e3; out7[5] = t_total / n * 1e3; out7[6] = (double)ncalls;
}

void fcb_reset_stats(void) { t_gather = t_wait_in = t_forward = t_wait_out = t_scatter = t_total = 0; ncalls = 0; }

void fcb_finalize(void) {
  if (is_leader && out_tensor) torch_tensor_delete(out_tensor);
  if (is_leader && model) torch_jit_module_delete(model);
  if (win != MPI_WIN_NULL) MPI_Win_free(&win);
  if (group_comm != MPI_COMM_NULL) MPI_Comm_free(&group_comm);
  initialised = 0;
}
