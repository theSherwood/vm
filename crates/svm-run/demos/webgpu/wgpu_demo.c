/* wgpu_demo.c — the headless WebGPU **compute → readback → assert-vs-CPU** demo (LLVM.md WebGPU
 * demo 1). Proves the whole data plane — upload → WGSL compute → readback — with zero windowing, on
 * the bare powerbox plus the granted `webgpu` capability. Two kernels:
 *   (1) an in-place `u32` map (`b[i] = b[i]*3 + 7`) over one storage buffer;
 *   (2) a two-buffer kernel (`c[i] = a[i]*a[i] + i`) — a read-only input at binding 0 and a
 *       read_write output at binding 1 — exercising multi-buffer bind groups.
 * Each result is re-computed on the CPU **in-guest** and compared element-by-element, so a wrong
 * dispatch or a mis-marshaled buffer is caught here; the program prints OK/FAIL + an FNV digest and
 * exits 0 only if both kernels match. There is no native oracle (the capability is SVM-only, like the
 * async/JIT demos), so the test asserts this self-check. */

#include <stdio.h>

#ifdef SVM_GUEST
#include "wgpu_shim.c"
#endif

#define N 1024

static unsigned fnv(const unsigned *a, int n) {
  unsigned h = 2166136261u;
  for (int i = 0; i < n; i++) h = (h ^ a[i]) * 16777619u;
  return h;
}

static unsigned in[N], out[N], a[N], c[N];

int main(void) {
  int fail = 0;

  /* --- kernel 1: in-place mul-add over a single storage buffer --------------------------------- */
  for (int i = 0; i < N; i++) in[i] = (unsigned)i * 2654435761u >> 8;
  static const char sh1[] =
      "@group(0) @binding(0) var<storage, read_write> b: array<u32>;\n"
      "@compute @workgroup_size(64)\n"
      "fn main(@builtin(global_invocation_id) g: vec3<u32>) {\n"
      "  let i = g.x;\n"
      "  if (i < arrayLength(&b)) { b[i] = b[i] * 3u + 7u; }\n"
      "}\n";
  int buf = gpu_buffer(N * 4);
  gpu_write(buf, in, N * 4);
  int p1 = gpu_shader(sh1, sizeof sh1 - 1);
  gpu_dispatch(p1, buf, -1, (N + 63) / 64);
  gpu_read(buf, out, N * 4);
  int ok1 = (buf >= 0 && p1 >= 0);
  for (int i = 0; i < N && ok1; i++) {
    if (out[i] != in[i] * 3u + 7u) ok1 = 0;
  }
  printf("compute1 (mul-add, 1 buffer): %s digest=%08x\n", ok1 ? "OK" : "FAIL", fnv(out, N));
  fail |= !ok1;

  /* --- kernel 2: two-buffer kernel c[i] = a[i]*a[i] + i ---------------------------------------- */
  for (int i = 0; i < N; i++) a[i] = (unsigned)(i * 7 + 1);
  static const char sh2[] =
      "@group(0) @binding(0) var<storage, read> a: array<u32>;\n"
      "@group(0) @binding(1) var<storage, read_write> c: array<u32>;\n"
      "@compute @workgroup_size(64)\n"
      "fn main(@builtin(global_invocation_id) g: vec3<u32>) {\n"
      "  let i = g.x;\n"
      "  if (i < arrayLength(&a)) { c[i] = a[i] * a[i] + i; }\n"
      "}\n";
  int ba = gpu_buffer(N * 4), bc = gpu_buffer(N * 4);
  gpu_write(ba, a, N * 4);
  int p2 = gpu_shader(sh2, sizeof sh2 - 1);
  gpu_dispatch(p2, ba, bc, (N + 63) / 64);
  gpu_read(bc, c, N * 4);
  int ok2 = (ba >= 0 && bc >= 0 && p2 >= 0);
  for (int i = 0; i < N && ok2; i++) {
    if (c[i] != a[i] * a[i] + (unsigned)i) ok2 = 0;
  }
  printf("compute2 (a*a+i, 2 buffers): %s digest=%08x\n", ok2 ? "OK" : "FAIL", fnv(c, N));
  fail |= !ok2;

  printf("webgpu compute: %s\n", fail ? "FAIL" : "ALL MATCH cpu");
  return fail ? 1 : 0;
}
