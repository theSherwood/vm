#ifndef __STDARG_H
#define __STDARG_H

// SVM flat-buffer varargs ABI (DESIGN.md §3d). The caller marshals the promoted variadic
// arguments into a contiguous buffer — one 8-byte slot each — and passes a pointer to it
// as a hidden trailing argument; inside the callee that pointer is `__va_area__`. This
// replaces chibicc's x86-64 SysV register-save-area stdarg.h, which does not match our
// (clang-wasm-style) calling convention.

typedef char *va_list[1];

#define va_start(ap, last) ((ap)[0] = __va_area__)
#define va_end(ap) ((void)0)
#define va_copy(dest, src) ((dest)[0] = (src)[0])

// Read the current 8-byte slot as `ty`, then advance to the next slot.
#define va_arg(ap, ty) (*(ty *)(((ap)[0] += 8) - 8))

#define __GNUC_VA_LIST 1
typedef va_list __gnuc_va_list;

#endif
