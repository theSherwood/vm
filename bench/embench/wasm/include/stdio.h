/* Minimal freestanding-wasm `<stdio.h>` shim (see string.h header comment). The kernels' `printf`/
 * `fprintf` calls live in their `benchmark()`/`main()` (unused by the SVM/wasm `run` entry), so these
 * prototypes only satisfy compilation — `--gc-sections` drops the dead references at link. */
#ifndef _EMBENCH_WASM_STDIO_H
#define _EMBENCH_WASM_STDIO_H
#include <stdarg.h>
typedef struct _EMBENCH_FILE FILE;
extern FILE *stdout, *stderr;
int printf(const char *, ...);
int fprintf(FILE *, const char *, ...);
int putchar(int);
int puts(const char *);
#endif
