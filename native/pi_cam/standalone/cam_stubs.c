/*
 * Link-time stubs for a standalone physics-function image.
 *
 * A minimal image is linked from the oracle's own numerical objects, and a
 * few host-service symbols those objects reference have no business in a
 * single-column call: history output, namelist broadcast, the MPI shorthand
 * module, unit bookkeeping.  Each gets one of three treatments, chosen per
 * symbol in the function's reviewed YAML and instantiated here through
 * stub_list.h, which the builder generates from that YAML:
 *
 *   INERT        the call is a no-op on the numerical path (history output
 *                disabled, masterproc false).  Only symbols proven not to
 *                affect the routine's outputs may be inert.
 *   FAIL_CLOSED  must never be reached from a standalone call.  Reaching one
 *                is an implementation error, not a bad sample: report the
 *                symbol on stderr and exit 87.
 *   ABORT        the Fortran abort bridge.  CAM's endrun resolves to
 *                shr_sys_abort; the message is reported on stderr and the
 *                process exits 86, so a host can tell an aborted sample from
 *                every other failure.
 *
 * Nothing here touches the routine's numerics; the original objects are
 * linked unchanged.
 */

#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

#define FREECAM_EXIT_ABORT 86
#define FREECAM_EXIT_STUB_CALLED 87

static void freecam_write_all(int fd, const char *text, size_t length) {
    while (length > 0) {
        ssize_t written = write(fd, text, length);
        if (written <= 0) {
            return;
        }
        text += written;
        length -= (size_t)written;
    }
}

static void freecam_report(const char *prefix, const char *text, size_t length) {
    freecam_write_all(2, prefix, strlen(prefix));
    freecam_write_all(2, text, length);
    freecam_write_all(2, "\n", 1);
}

/* Inert: void procedure, any argument list (the caller cleans up). */
#define FREECAM_INERT_VOID(name) void name(void) {}

/* Inert: logical function that answers .false. */
#define FREECAM_INERT_FALSE(name) int32_t name(void) { return 0; }

/* Inert: module storage the objects read but the routine does not use. */
#define FREECAM_INERT_DATA_INT32(name) int32_t name = 0;

/* Fail closed: report which stub was reached and end the process. */
#define FREECAM_FAIL_CLOSED(name)                                          \
    int32_t name(void) {                                                   \
        freecam_report("FREECAM_STUB_CALLED: ", #name, strlen(#name));     \
        _exit(FREECAM_EXIT_STUB_CALLED);                                   \
        return 0;                                                          \
    }

/*
 * Abort bridge for `subroutine shr_sys_abort(string, rc)` with both dummies
 * optional: ifort passes the two addresses (NULL when absent) and then the
 * hidden character length.  The build's abort probe proves this convention.
 */
#define FREECAM_ABORT(name)                                                \
    void name(const char *string, const int32_t *rc, size_t string_len) {  \
        (void)rc;                                                          \
        size_t length = 0;                                                 \
        if (string != NULL && string_len < 4096) {                         \
            length = string_len;                                           \
            while (length > 0 && string[length - 1] == ' ') {              \
                length -= 1;                                               \
            }                                                              \
        }                                                                  \
        freecam_report("FREECAM_FORTRAN_ABORT: ", string == NULL ? "" : string, length); \
        _exit(FREECAM_EXIT_ABORT);                                         \
    }

#include "stub_list.h"
