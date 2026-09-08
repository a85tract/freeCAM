/* Rank-local kernel execution counters and their process context.
 *
 * A counting image redirects a kernel's symbol to a generated tail-jump
 * trampoline (tools/generate_pi_cam_kcount_trampolines.py) that increments
 * counts[kernel_index][context] and jumps to the original definition with
 * every register and stack byte untouched: no argument is copied, no memory
 * is allocated, no floating-point operation changes, and no call returns to
 * Python.  The kernel index is the candidate's index in the committed
 * observability record, stable across image scopes.
 *
 * Python owns the context: the driver sets the slot at process boundaries
 * (paired set/restore, exception-safe) and reads the table as a zero-copy
 * NumPy view.  Slot meanings are recorded in the runtime coverage record,
 * not here.  Counters are plain 64-bit increments: valid for the admitted
 * single-threaded-per-rank configuration only; the CLI refuses to observe
 * with OpenMP threads.
 */
#include <stdint.h>
#include <string.h>

enum {
    PYCAM_KCOUNT_SLOTS = 64,
    PYCAM_KCOUNT_MAX_KERNELS = 1024,
};

int64_t pycam_kcount_table[PYCAM_KCOUNT_MAX_KERNELS * PYCAM_KCOUNT_SLOTS];
int32_t pycam_kcount_context = 0;

/* The trampoline object defines the strong count; an image without
 * trampolines reports zero instrumented kernels. */
__attribute__((weak)) int32_t pycam_kcount_kernel_count = 0;

int32_t pycam_kcount_info_v1(int32_t *kernels, int32_t *slots, int32_t *max_kernels)
{
    if (kernels == 0 || slots == 0 || max_kernels == 0) {
        return 1;
    }
    *kernels = pycam_kcount_kernel_count;
    *slots = PYCAM_KCOUNT_SLOTS;
    *max_kernels = PYCAM_KCOUNT_MAX_KERNELS;
    return 0;
}

int64_t *pycam_kcount_data_v1(void)
{
    return pycam_kcount_table;
}

int32_t pycam_kcount_context_get_v1(void)
{
    return pycam_kcount_context;
}

/* Set the context slot; returns the previous slot so the caller can restore
 * it in a finally block.  An out-of-range slot falls back to 0 (unattributed)
 * rather than corrupting the table. */
int32_t pycam_kcount_context_set_v1(int32_t slot)
{
    int32_t previous = pycam_kcount_context;
    pycam_kcount_context = (slot >= 0 && slot < PYCAM_KCOUNT_SLOTS) ? slot : 0;
    return previous;
}

void pycam_kcount_reset_v1(void)
{
    memset(pycam_kcount_table, 0, sizeof pycam_kcount_table);
    pycam_kcount_context = 0;
}
