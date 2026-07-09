// Self-timing native driver for the confine harness (linked with each `kernel.c`, which exports
// `long run(long n)`). Mirrors the Rust `per_iter`: per-iteration ns = (min t(large) - min t(small))
// / (large - small), min over `reps`. stdout is two lines — "<per_iter_ns>" then "<run(small)>" (the
// correctness oracle) — matching the node runner's format so both parse the same way.
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

extern long run(long);

static double now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

static double best(long n, int reps) {
  volatile long warm = run(n);
  (void)warm;
  double b = 1e30;
  for (int r = 0; r < reps; r++) {
    double t = now_ns();
    volatile long x = run(n);
    (void)x;
    double e = now_ns() - t;
    if (e < b) b = e;
  }
  return b;
}

int main(int argc, char **argv) {
  long small = atol(argv[1]), large = atol(argv[2]);
  int reps = argc > 3 ? atoi(argv[3]) : 25;
  double per = (best(large, reps) - best(small, reps)) / (double)(large - small);
  printf("%.6f\n%ld\n", per, run(small));
  return 0;
}
