#include <stdio.h>
#include <stdint.h>
#include <time.h>
// Every kernel is int32_t(int32_t) now (i32-LCG / i32 accumulators), so one bench path covers all.
int32_t alu(int32_t), call(int32_t), call_indirect(int32_t), mem(int32_t), chase(int32_t),
    chase_rand(int32_t), fnv(int32_t), fma_k(int32_t), vsum(int32_t);
static double now() { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec*1e9 + t.tv_nsec; }
static volatile int32_t sink;
static double min_run(int32_t (*k)(int32_t), int32_t n) {
  sink += k(n); // warm up
  double best = 1e18;
  for (int r = 0; r < 25; r++) { double a = now(); sink += k(n); double b = now(); if (b-a < best) best = b-a; }
  return best;
}
static void bench(const char *name, int32_t (*k)(int32_t)) {
  double s = min_run(k, 1000), l = min_run(k, 201000);
  printf("native,%s,%.4f\n", name, (l - s) / 200000.0);
}
int main() {
  bench("alu", alu); bench("call", call); bench("call_indirect", call_indirect); bench("mem", mem);
  bench("chase", chase); bench("chase_rand", chase_rand);
  bench("fnv", fnv); bench("fma", fma_k); bench("vsum", vsum);
  return 0;
}
