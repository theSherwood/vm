/* Guest WebGPU compute shim — bridges a minimal C compute API to the host `webgpu` capability
 * (svm-webgpu crate) via the generic §7 host-defined-capability surface (`__vm_cap_resolve` +
 * `__vm_host_call`), exactly like the fs/LMDB shims. The guest never holds a GPU pointer: it names
 * buffers/pipelines by small integer ids the host hands back, and only *data* (upload bytes, WGSL
 * source, readback bytes) crosses the window boundary.
 *
 * `#include`d into a driver under `-DSVM_GUEST`. */

extern int __vm_cap_resolve(const char *name, long len);
extern long __vm_host_call(int h, int op, long a, long b, long c, long d);

enum { GPU_CREATE_BUF = 0, GPU_WRITE = 1, GPU_SHADER = 2, GPU_DISPATCH = 3, GPU_READ = 4 };

static int g_gpu = -2; /* -2 = unresolved */
static int gpu(void) {
  if (g_gpu == -2) g_gpu = __vm_cap_resolve("webgpu", 6);
  return g_gpu;
}

/* Create a storage buffer of `size` bytes; returns its id (>= 0) or -1. */
static int gpu_buffer(long size) { return (int)__vm_host_call(gpu(), GPU_CREATE_BUF, size, 0, 0, 0); }
/* Upload `n` bytes from the window at `p` into buffer `buf`. */
static long gpu_write(int buf, const void *p, long n) {
  return __vm_host_call(gpu(), GPU_WRITE, buf, (long)p, n, 0);
}
/* Compile a WGSL compute shader (entry `main`); returns a pipeline id (>= 0) or -1 on a compile error. */
static int gpu_shader(const char *wgsl, long n) {
  return (int)__vm_host_call(gpu(), GPU_SHADER, (long)wgsl, n, 0, 0);
}
/* Run `pipe` over `groups` workgroups, binding `b0`@0 (and `b1`@1 when `b1 >= 0`). */
static long gpu_dispatch(int pipe, int b0, int b1, int groups) {
  return __vm_host_call(gpu(), GPU_DISPATCH, pipe, b0, b1, groups);
}
/* Read `n` bytes of buffer `buf` back into the window at `p`. */
static long gpu_read(int buf, void *p, long n) {
  return __vm_host_call(gpu(), GPU_READ, buf, (long)p, n, 0);
}
