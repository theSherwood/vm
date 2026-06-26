/* Minimal freestanding-wasm `<stdlib.h>` shim (see string.h header comment). Only prototypes — any
 * heap use in the timed `run` path goes through the BEEBS array allocator (`calloc_beebs`), so real
 * `malloc`/`qsort`/etc. are referenced only by dead code that the linker's `--gc-sections` drops. */
#ifndef _EMBENCH_WASM_STDLIB_H
#define _EMBENCH_WASM_STDLIB_H
#include <stddef.h>
void *malloc(size_t);
void *calloc(size_t, size_t);
void *realloc(void *, size_t);
void free(void *);
void abort(void);
void exit(int);
void qsort(void *, size_t, size_t, int (*)(const void *, const void *));
int atoi(const char *);
long atol(const char *);
#endif
