/* Python-owned backing storage for unmodified Intel Fortran ALLOCATE calls.
 *
 * Intel Fortran's serial allocatable runtime reaches the exported
 * _mm_malloc/_mm_free boundary after it has prepared the array descriptor.
 * The original non-PIC CESM objects call that boundary without being rebuilt.
 * When a Python callback is installed, this file obtains aligned storage from
 * Python and returns its address to the unchanged Fortran runtime.
 */

#define _GNU_SOURCE

#include <dlfcn.h>
#include <execinfo.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef void *(*pycesm_heap_allocate_callback_v1)(
    const char *source_id,
    int64_t allocation_id,
    int64_t byte_count,
    int64_t alignment,
    int32_t *status);

typedef void (*pycesm_heap_release_callback_v1)(void *address, int32_t *status);

typedef struct pycesm_owned_pointer_v1 {
  void *address;
  struct pycesm_owned_pointer_v1 *next;
} pycesm_owned_pointer_v1;

static pycesm_heap_allocate_callback_v1 pycesm_allocate_callback = NULL;
static pycesm_heap_release_callback_v1 pycesm_release_callback = NULL;
static pycesm_owned_pointer_v1 *pycesm_owned_pointers = NULL;
static int64_t pycesm_next_allocation_id = 1;
static int64_t pycesm_live_allocations = 0;
static int64_t pycesm_total_allocations = 0;
static int32_t pycesm_callback_error = 0;
static _Thread_local int pycesm_inside_callback = 0;

/* Stable Intel 2023 runtime entrypoints used when Python ownership is off. */
extern int32_t for_allocate_handle(size_t, void **, int32_t, void *);
extern int32_t for_alloc_allocatable_handle(size_t, void **, int32_t, void *);
extern int32_t for_deallocate_handle(void *, int32_t, void *);

static void *pycesm_fallback_allocate(size_t byte_count, size_t alignment) {
  void *address = NULL;
  size_t actual_size = byte_count == 0 ? 1 : byte_count;
  size_t actual_alignment = alignment;
  if (actual_alignment < sizeof(void *)) {
    actual_alignment = sizeof(void *);
  }
  if ((actual_alignment & (actual_alignment - 1)) != 0) {
    actual_alignment = sizeof(void *);
  }
  if (posix_memalign(&address, actual_alignment, actual_size) != 0) {
    return NULL;
  }
  return address;
}

static void pycesm_source_id(char *buffer, size_t buffer_length) {
  void *frames[24];
  int frame_count;
  int index;
  if (buffer_length == 0) {
    return;
  }
  snprintf(buffer, buffer_length, "intel-fortran:unknown");
  frame_count = backtrace(frames, (int)(sizeof(frames) / sizeof(frames[0])));
  for (index = 1; index < frame_count; ++index) {
    Dl_info info;
    uintptr_t offset;
    const char *symbol;
    const char *image;
    if (dladdr(frames[index], &info) == 0) {
      continue;
    }
    symbol = info.dli_sname == NULL ? "" : info.dli_sname;
    image = info.dli_fname == NULL ? "" : info.dli_fname;
    if (strcmp(symbol, "_mm_malloc") == 0 ||
        strncmp(symbol, "for_alloc", 9) == 0 ||
        strncmp(symbol, "for_dealloc", 11) == 0 ||
        strncmp(symbol, "for_realloc", 11) == 0 ||
        strncmp(symbol, "pycesm_", 8) == 0 ||
        strstr(image, "libifcore") != NULL ||
        strstr(image, "libirc") != NULL) {
      continue;
    }
    if (info.dli_saddr != NULL && symbol[0] != '\0') {
      offset = (uintptr_t)frames[index] - (uintptr_t)info.dli_saddr;
      snprintf(buffer, buffer_length, "%s+0x%" PRIxPTR, symbol, offset);
    } else if (info.dli_fbase != NULL) {
      offset = (uintptr_t)frames[index] - (uintptr_t)info.dli_fbase;
      snprintf(buffer, buffer_length, "%s+0x%" PRIxPTR, image, offset);
    }
    return;
  }
}

static int pycesm_register_owned_pointer(void *address) {
  pycesm_owned_pointer_v1 *entry =
      (pycesm_owned_pointer_v1 *)malloc(sizeof(*entry));
  if (entry == NULL) {
    return 1;
  }
  entry->address = address;
  entry->next = pycesm_owned_pointers;
  pycesm_owned_pointers = entry;
  pycesm_live_allocations += 1;
  pycesm_total_allocations += 1;
  return 0;
}

static int pycesm_remove_owned_pointer(void *address) {
  pycesm_owned_pointer_v1 **link = &pycesm_owned_pointers;
  while (*link != NULL) {
    pycesm_owned_pointer_v1 *entry = *link;
    if (entry->address == address) {
      *link = entry->next;
      free(entry);
      pycesm_live_allocations -= 1;
      return 1;
    }
    link = &entry->next;
  }
  return 0;
}

static int pycesm_is_owned_pointer(void *address) {
  pycesm_owned_pointer_v1 *entry = pycesm_owned_pointers;
  while (entry != NULL) {
    if (entry->address == address) {
      return 1;
    }
    entry = entry->next;
  }
  return 0;
}

void pycesm_fortran_heap_set_allocator_v1(
    pycesm_heap_allocate_callback_v1 allocate_callback,
    pycesm_heap_release_callback_v1 release_callback,
    int32_t *status) {
  if (status == NULL) {
    return;
  }
  if (allocate_callback == NULL || release_callback == NULL ||
      pycesm_allocate_callback != NULL || pycesm_live_allocations != 0) {
    *status = 1;
    return;
  }
  pycesm_allocate_callback = allocate_callback;
  pycesm_release_callback = release_callback;
  pycesm_callback_error = 0;
  pycesm_next_allocation_id = 1;
  pycesm_total_allocations = 0;
  *status = 0;
}

void pycesm_fortran_heap_clear_allocator_v1(int64_t *live_allocations,
                                             int64_t *total_allocations,
                                             int32_t *status) {
  if (live_allocations != NULL) {
    *live_allocations = pycesm_live_allocations;
  }
  if (total_allocations != NULL) {
    *total_allocations = pycesm_total_allocations;
  }
  if (status == NULL) {
    return;
  }
  while (pycesm_owned_pointers != NULL && pycesm_release_callback != NULL) {
    pycesm_owned_pointer_v1 *entry = pycesm_owned_pointers;
    int32_t release_status = 0;
    pycesm_owned_pointers = entry->next;
    pycesm_inside_callback = 1;
    pycesm_release_callback(entry->address, &release_status);
    pycesm_inside_callback = 0;
    if (release_status != 0) {
      pycesm_callback_error = 1;
    }
    free(entry);
    pycesm_live_allocations -= 1;
  }
  if (pycesm_live_allocations != 0 || pycesm_callback_error != 0) {
    *status = 1;
    return;
  }
  pycesm_allocate_callback = NULL;
  pycesm_release_callback = NULL;
  *status = 0;
}

void *_mm_malloc(size_t byte_count, size_t alignment) {
  char source_id[512];
  int32_t status = 0;
  int64_t allocation_id;
  void *address;
  if (pycesm_allocate_callback == NULL || pycesm_inside_callback) {
    return pycesm_fallback_allocate(byte_count, alignment);
  }
  pycesm_inside_callback = 1;
  pycesm_source_id(source_id, sizeof(source_id));
  allocation_id = pycesm_next_allocation_id++;
  address = pycesm_allocate_callback(source_id, allocation_id,
                                     (int64_t)byte_count, (int64_t)alignment,
                                     &status);
  /* Status 2 is an explicit policy decision: this allocation is transient
   * scratch or an opaque native resource and therefore stays native. */
  if (status == 2) {
    pycesm_inside_callback = 0;
    return pycesm_fallback_allocate(byte_count, alignment);
  }
  if (status != 0 || address == NULL ||
      ((uintptr_t)address % (alignment < sizeof(void *) ? sizeof(void *)
                                                        : alignment)) != 0 ||
      pycesm_register_owned_pointer(address) != 0) {
    pycesm_callback_error = 1;
    address = NULL;
  }
  pycesm_inside_callback = 0;
  return address;
}

void _mm_free(void *address) {
  int32_t status = 0;
  if (address == NULL) {
    return;
  }
  if (pycesm_remove_owned_pointer(address)) {
    if (pycesm_release_callback == NULL || pycesm_inside_callback) {
      pycesm_callback_error = 1;
      return;
    }
    pycesm_inside_callback = 1;
    pycesm_release_callback(address, &status);
    if (status != 0) {
      pycesm_callback_error = 1;
    }
    pycesm_inside_callback = 0;
    return;
  }
  free(address);
}

/* The original CESM objects reference these symbols directly.  Defining them
 * in the non-PIC executable is therefore a stronger and deterministic
 * interposition point than relying on symbol lookup from inside libifcore.
 */
int32_t for_allocate(size_t byte_count, void **descriptor, int32_t flags) {
  char source_id[512];
  void *address;
  int32_t status = 0;
  int64_t allocation_id;
  int32_t alignment_code = (flags >> 16) & 0x1f;
  size_t alignment =
      alignment_code >= 5 ? ((size_t)1 << alignment_code) : (size_t)32;
  if (pycesm_allocate_callback == NULL || pycesm_inside_callback) {
    return for_allocate_handle(byte_count, descriptor, flags, NULL);
  }
  pycesm_inside_callback = 1;
  pycesm_source_id(source_id, sizeof(source_id));
  allocation_id = pycesm_next_allocation_id++;
  address = pycesm_allocate_callback(source_id, allocation_id,
                                     (int64_t)byte_count, (int64_t)alignment,
                                     &status);
  /* A schema decision to keep an opaque resource or transient scratch object
   * native must delegate the complete allocatable operation, not merely its
   * raw byte allocation.  Intel's handle routine also initializes descriptor
   * metadata required by allocatable derived-type components. */
  if (status == 2) {
    pycesm_inside_callback = 0;
    return for_allocate_handle(byte_count, descriptor, flags, NULL);
  }
  if (status != 0 || address == NULL ||
      ((uintptr_t)address % alignment) != 0 ||
      pycesm_register_owned_pointer(address) != 0) {
    pycesm_callback_error = 1;
    address = NULL;
  }
  pycesm_inside_callback = 0;
  if (descriptor != NULL) {
    *descriptor = address;
  }
  return address == NULL ? 41 : 0;
}

/* Intel emits this checked entrypoint for ordinary ALLOCATE statements.  The
 * original implementation first rejects an already allocated descriptor and
 * then tail-calls for_allocate_handle.  Preserve the diagnostic path by using
 * the original handle when the descriptor is non-null; only fresh storage is
 * redirected to the Python allocator.
 */
int32_t for_alloc_allocatable(size_t byte_count, void **descriptor,
                              int32_t flags) {
  if (descriptor == NULL || *descriptor != NULL) {
    return for_alloc_allocatable_handle(byte_count, descriptor, flags, NULL);
  }
  return for_allocate(byte_count, descriptor, flags);
}

int32_t for_dealloc_allocatable(void *address, int32_t flags) {
  if (pycesm_is_owned_pointer(address)) {
    _mm_free(address);
    return pycesm_callback_error == 0 ? 0 : 41;
  }
  return for_deallocate_handle(address, flags, NULL);
}


/* Some generated call sites use the unchecked deallocator rather than the
 * allocatable-specific spelling.  It has the same address/flags ABI and must
 * release Python-owned storage through the matching callback.
 */
int32_t for_deallocate(void *address, int32_t flags) {
  if (pycesm_is_owned_pointer(address)) {
    _mm_free(address);
    return pycesm_callback_error == 0 ? 0 : 41;
  }
  return for_deallocate_handle(address, flags, NULL);
}
