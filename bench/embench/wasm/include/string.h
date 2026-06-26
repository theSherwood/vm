/* Minimal freestanding-wasm `<string.h>` shim for the Embench cross-engine build (see
 * bench/embench/README.md). The host SVM build uses real libc headers + the on-ramp's synthesized
 * helpers; the wasm32 build is `-nostdlib`, so it gets these prototypes plus the definitions in
 * `../defs.h`. `memcpy`/`memmove`/`memset` are clang intrinsics lowered to wasm bulk-memory
 * (`-mbulk-memory`); `memcmp`/`bcmp` are defined by `wrapper.c`'s `SVM_BUILD` block. */
#ifndef _EMBENCH_WASM_STRING_H
#define _EMBENCH_WASM_STRING_H
#include <stddef.h>
void *memcpy(void *, const void *, size_t);
void *memmove(void *, const void *, size_t);
void *memset(void *, int, size_t);
int memcmp(const void *, const void *, size_t);
void *memchr(const void *, int, size_t);
size_t strlen(const char *);
char *strchr(const char *, int);
int strcmp(const char *, const char *);
int strncmp(const char *, const char *, size_t);
#endif
