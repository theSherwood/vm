/* Shared guest `errno` cell for the Postgres shims (slice CC). glibc's `errno` macro expands to
 * `*__errno_location()`; every shim (`os_shim.c` syscalls, `libc_shim.c` parsing) sets it, and a
 * driver reads it — so the accessor must exist exactly once. This header is include-guarded, so a
 * driver that pulls in several shims (the eventual whole-Postgres build `#include`s them all into
 * one translation unit) gets a single definition, while a probe that pulls in just one still has it.
 */

#ifndef SHIM_ERRNO_H
#define SHIM_ERRNO_H

int shim_errno; /* the guest's errno storage (one tentative definition per combined TU) */
int *__errno_location(void) { return &shim_errno; }

#endif
