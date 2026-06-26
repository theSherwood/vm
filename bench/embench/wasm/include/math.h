/* Minimal freestanding-wasm `<math.h>` shim (see string.h header comment). `sqrt`/`fabs`/`floor`/
 * `ceil` map to clang builtins that lower to native wasm fp instructions; the transcendentals are
 * prototype-only (referenced only by kernels' dead code, dropped by `--gc-sections`). */
#ifndef _EMBENCH_WASM_MATH_H
#define _EMBENCH_WASM_MATH_H
static inline double sqrt(double x) { return __builtin_sqrt(x); }
static inline float sqrtf(float x) { return __builtin_sqrtf(x); }
static inline double fabs(double x) { return __builtin_fabs(x); }
static inline double floor(double x) { return __builtin_floor(x); }
static inline double ceil(double x) { return __builtin_ceil(x); }
double exp(double);
double log(double);
double pow(double, double);
double sin(double);
double cos(double);
#endif
