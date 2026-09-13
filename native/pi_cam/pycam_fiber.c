/* A second execution stack for a pausable stage, so that a hook deep in the
 * Fortran call stack can hand control back to Python without unwinding.
 *
 * The stage's runner starts its state machine on this stack.  When a hooked
 * kernel is reached with a replacement installed, the hook yields: the main
 * context resumes inside the runner's start or resume entry, which returns
 * NEEDS_PYTHON_KERNEL to Python exactly as a runner-level pause does.  Python
 * later calls resume, which switches back and the hook returns to its caller.
 * Fortran still never calls Python; it only switches stacks.  One fiber per
 * process suffices: stages run one at a time.
 *
 * The stack is reserved lazily (MAP_NORESERVE) and touched only as deep as the
 * Fortran needs, so its cost in resident memory is what the run actually uses.
 */
#include <stddef.h>
#include <stdint.h>
#include <sys/mman.h>
#include <ucontext.h>

static ucontext_t pycam_main_context;
static ucontext_t pycam_fiber_context;
static void *pycam_fiber_stack = NULL;
static size_t pycam_fiber_stack_bytes = 0;
static void (*pycam_fiber_body)(void) = NULL;
static volatile int pycam_fiber_event = 0;
static volatile int pycam_fiber_alive = 0;   /* the body has started and not returned */
static volatile int pycam_fiber_running = 0; /* the fiber stack is the one executing now */

static void pycam_fiber_trampoline(void)
{
    pycam_fiber_body();
    pycam_fiber_alive = 0;
    /* uc_link returns to the main context when the body returns */
}

/* Start body() on the fiber stack; returns when it yields or returns. */
int pycam_fiber_start_v1(void (*body)(void), int64_t stack_bytes, int32_t *event)
{
    if (pycam_fiber_alive || body == NULL || stack_bytes < (1 << 20)) {
        return 1;
    }
    if (pycam_fiber_stack != NULL && pycam_fiber_stack_bytes != (size_t)stack_bytes) {
        munmap(pycam_fiber_stack, pycam_fiber_stack_bytes);
        pycam_fiber_stack = NULL;
    }
    if (pycam_fiber_stack == NULL) {
        void *stack = mmap(NULL, (size_t)stack_bytes, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (stack == MAP_FAILED) {
            return 2;
        }
        pycam_fiber_stack = stack;
        pycam_fiber_stack_bytes = (size_t)stack_bytes;
    }
    if (getcontext(&pycam_fiber_context) != 0) {
        return 3;
    }
    pycam_fiber_context.uc_stack.ss_sp = pycam_fiber_stack;
    pycam_fiber_context.uc_stack.ss_size = pycam_fiber_stack_bytes;
    pycam_fiber_context.uc_link = &pycam_main_context;
    pycam_fiber_body = body;
    pycam_fiber_event = 0;
    makecontext(&pycam_fiber_context, pycam_fiber_trampoline, 0);
    pycam_fiber_alive = 1;
    pycam_fiber_running = 1;
    if (swapcontext(&pycam_main_context, &pycam_fiber_context) != 0) {
        pycam_fiber_alive = 0;
        pycam_fiber_running = 0;
        return 4;
    }
    pycam_fiber_running = 0;
    *event = pycam_fiber_event;
    return 0;
}

/* Continue the fiber after a yield; returns at its next yield or return. */
int pycam_fiber_resume_v1(int32_t *event)
{
    if (!pycam_fiber_alive || pycam_fiber_running) {
        return 1;
    }
    pycam_fiber_running = 1;
    if (swapcontext(&pycam_main_context, &pycam_fiber_context) != 0) {
        pycam_fiber_running = 0;
        return 4;
    }
    pycam_fiber_running = 0;
    *event = pycam_fiber_event;
    return 0;
}

/* Called on the fiber stack: hand `event` to the main context and wait. */
void pycam_fiber_yield_v1(int32_t event)
{
    if (!pycam_fiber_running) {
        return;
    }
    pycam_fiber_event = event;
    swapcontext(&pycam_fiber_context, &pycam_main_context);
}

/* Called on the fiber stack just before the body returns: the event start or
 * resume will report once the body has finished. */
void pycam_fiber_finish_v1(int32_t event)
{
    pycam_fiber_event = event;
}

/* Whether a body is suspended on the fiber stack. */
int32_t pycam_fiber_alive_v1(void)
{
    return pycam_fiber_alive ? 1 : 0;
}

/* Whether the caller is executing on the fiber stack. */
int32_t pycam_fiber_running_v1(void)
{
    return pycam_fiber_running ? 1 : 0;
}

/* Forget a suspended body (after an error): its stack is reused by the next start. */
void pycam_fiber_abandon_v1(void)
{
    pycam_fiber_alive = 0;
    pycam_fiber_running = 0;
}

/* Resident and reserved size of the fiber stack, for the run's memory record. */
int64_t pycam_fiber_stack_bytes_v1(void)
{
    return (int64_t)pycam_fiber_stack_bytes;
}
