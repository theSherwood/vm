/* Generic cross-engine wrapper for an Embench-IoT benchmark — see bench/embench/README.md.
 *
 * Embench source is NOT vendored here (mixed per-benchmark licenses); point the driver at a checkout
 * via $EMBENCH. This file `#include`s one benchmark's `.c` (so `run` can call its *static*
 * `benchmark_body`) and exposes a single entry the SVM frontend translates:
 *
 *   long run(long n)  — run `n` Embench "iterations" (benchmark_body(n, GLOBAL_SCALE_FACTOR)) and
 *                       return verify_benchmark()'s strict pass/fail (1 == matched Embench's expected
 *                       output). Used both as the timed kernel and as the cross-engine correctness
 *                       oracle (every engine must return the same 1).
 *
 * The benchmark's own `benchmark()` (which needs the scale-factor macros) is compiled but unused; we
 * call `benchmark_body` directly with our own `n`. `main` is compiled only for the native build
 * (timing harness); the SVM build defines SVM_BUILD so the bitcode carries no libc-calling `main`.
 *
 * Required -D: BENCH_SRC="\"<abs path to the benchmark .c>\"". Optional: BEEBS_SRC for benchmarks that
 * use the BEEBS rand/heap (e.g. crc32). Always pass -DNDEBUG (drops asserts → no __assert_fail extern).
 */
#ifndef GLOBAL_SCALE_FACTOR
#define GLOBAL_SCALE_FACTOR 1
#endif
#ifndef CPU_MHZ
#define CPU_MHZ 1
#endif
#ifndef WARMUP_HEAT
#define WARMUP_HEAT 0
#endif

#include BENCH_SRC
#ifdef BEEBS_SRC
#include BEEBS_SRC
#endif

/* `verify_benchmark` compares result arrays with memcmp, which clang lowers to a `memcmp`/`bcmp`
 * libcall the SVM on-ramp has no definition for. Provide them in-module for the SVM build (compiled
 * with -fno-builtin-memcmp/-bcmp so clang doesn't fold these back into self-calls); the native build
 * uses libc. memcpy/memset stay clang intrinsics the on-ramp already lowers. */
#ifdef SVM_BUILD
#include <stddef.h>
int
memcmp (const void *a, const void *b, size_t n)
{
  const unsigned char *x = a, *y = b;
  for (size_t i = 0; i < n; i++)
    if (x[i] != y[i])
      return (int) x[i] - (int) y[i];
  return 0;
}
int
bcmp (const void *a, const void *b, size_t n)
{
  return memcmp (a, b, n);
}
#endif

long
run (long n)
{
  initialise_benchmark ();
  int r = benchmark_body ((unsigned int) n, (unsigned int) GLOBAL_SCALE_FACTOR);
  return (long) verify_benchmark (r);
}

#ifndef SVM_BUILD
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double
now (void)
{
  struct timespec t;
  clock_gettime (CLOCK_MONOTONIC, &t);
  return (double) t.tv_sec * 1e9 + (double) t.tv_nsec;
}
int
main (int argc, char **argv)
{
  long small = atol (argv[1]), large = atol (argv[2]), vn = atol (argv[3]);
  volatile long sink = 0;
  sink += run (large);
  double bs = 1e18, bl = 1e18;
  for (int r = 0; r < 10; r++) { double a = now (); sink += run (small); double e = now (); if (e - a < bs) bs = e - a; }
  for (int r = 0; r < 10; r++) { double a = now (); sink += run (large); double e = now (); if (e - a < bl) bl = e - a; }
  printf ("%.6f\n%ld\n", (bl - bs) / (double) (large - small), run (vn));
  return (int) sink;
}
#endif
