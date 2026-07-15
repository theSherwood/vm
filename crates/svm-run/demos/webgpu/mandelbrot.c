/* mandelbrot.c — the WebGPU **compute → image → PNG** demo (LLVM.md WebGPU demo 3): a WGSL compute
 * shader renders a Mandelbrot set into an RGBA buffer, the guest reads it back, self-validates it,
 * and writes the raw pixels out through the granted `fs` capability (the host test encodes the PNG).
 * Exercises the `webgpu` compute cap and the `fs` cap **together**, on the bare powerbox.
 *
 * Self-validation without a fragile hardcoded hash: the imaginary-axis mapping is exactly symmetric
 * (`y0(py) = -y0(H-1-py)`), and the Mandelbrot set is symmetric about the real axis, so row `py` must
 * equal row `H-1-py` **bit-for-bit** — a structural invariant robust to any float implementation.
 * Plus a sanity pair: the origin is in the set (max iterations), a far corner escapes fast.
 *
 * `#include`d into a driver under `-DSVM_GUEST`. */

#include <stdio.h>

#ifdef SVM_GUEST
#include "wgpu_shim.c"

/* Minimal `fs` cap file write (the same §7 host-call surface, op protocol from svm-run/src/fs.rs). */
extern int __vm_cap_resolve(const char *name, long len);
extern long __vm_host_call(int h, int op, long a, long b, long c, long d);
enum { FS_OPEN = 0, FS_WRITE = 2, FS_CLOSE = 4 };
enum { FS_O_WRITE = 2, FS_O_TRUNC = 8, FS_O_CREATE = 16 };
static int g_fs = -2;
static int fs(void) {
  if (g_fs == -2) g_fs = __vm_cap_resolve("fs", 2);
  return g_fs;
}
static long strlen_c(const char *s) {
  long n = 0;
  while (s[n]) n++;
  return n;
}
static int fs_write_file(const char *name, const void *buf, long len) {
  long fd = __vm_host_call(fs(), FS_OPEN, (long)name, strlen_c(name), FS_O_WRITE | FS_O_CREATE | FS_O_TRUNC, 0);
  if (fd < 0) return -1;
  long w = __vm_host_call(fs(), FS_WRITE, fd, (long)buf, len, 0);
  __vm_host_call(fs(), FS_CLOSE, fd, 0, 0, 0);
  return w == len ? 0 : -1;
}
#endif

#define W 320
#define H 240

static unsigned img[W * H];

static unsigned fnv(const unsigned *a, int n) {
  unsigned h = 2166136261u;
  for (int i = 0; i < n; i++) h = (h ^ a[i]) * 16777619u;
  return h;
}

/* The WGSL is a compile-time constant; W/H are baked in as `const`s. Escape iterations map to an
 * RGBA `u32` (little-endian 0xAABBGGRR). The imaginary mapping is symmetric around 0 so the image is
 * mirror-symmetric top/bottom (the self-check relies on this). */
static const char shader[] =
    "@group(0) @binding(0) var<storage, read_write> img: array<u32>;\n"
    "const W: u32 = 320u;\n"
    "const H: u32 = 240u;\n"
    "@compute @workgroup_size(64)\n"
    "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n"
    "  let idx = gid.x;\n"
    "  if (idx >= W * H) { return; }\n"
    "  let px = idx % W;\n"
    "  let py = idx / W;\n"
    "  let x0 = (f32(px) + 0.5) / f32(W) * 3.5 - 2.5;\n"
    /* center-relative so mirrored rows are exact IEEE negatives: y0(H-1-py) == -y0(py) bit-for-bit,
       and the Mandelbrot conjugate symmetry then makes the two rows identical (the self-check). */
    "  let y0 = (f32(py) - 119.5) * 0.01;\n"
    "  var x = 0.0;\n"
    "  var y = 0.0;\n"
    "  var it = 0u;\n"
    "  let maxit = 255u;\n"
    "  loop {\n"
    "    if (it >= maxit) { break; }\n"
    "    let xt = x * x - y * y + x0;\n"
    "    y = 2.0 * x * y + y0;\n"
    "    x = xt;\n"
    "    if (x * x + y * y > 4.0) { break; }\n"
    "    it = it + 1u;\n"
    "  }\n"
    "  let c = it;\n"
    "  let r = c;\n"
    "  let g = (c * 3u) & 255u;\n"
    "  let b = 255u - c;\n"
    "  img[idx] = (255u << 24u) | (b << 16u) | (g << 8u) | r;\n"
    "}\n";

int main(void) {
  int buf = gpu_buffer(W * H * 4);
  int pipe = gpu_shader(shader, sizeof shader - 1);
  gpu_dispatch(pipe, buf, -1, (W * H + 63) / 64);
  gpu_read(buf, img, W * H * 4);

  int ok = (buf >= 0 && pipe >= 0);
  /* structural self-check: exact top/bottom mirror symmetry (float-implementation independent) */
  int symmetric = 1;
  for (int py = 0; py < H / 2 && symmetric; py++) {
    for (int px = 0; px < W; px++) {
      if (img[py * W + px] != img[(H - 1 - py) * W + px]) { symmetric = 0; break; }
    }
  }
  /* sanity: the origin (x0≈0,y0≈0) is deep in the set (max iterations → r channel 255); a corner
   * escapes fast (low iteration → r channel small). */
  unsigned center = img[(H / 2) * W + (int)((2.5 / 3.5) * W)]; /* complex 0+0i */
  unsigned corner = img[0];
  int center_in = (center & 0xff) >= 250;   /* r ≈ maxit */
  int corner_out = (corner & 0xff) <= 8;    /* r small */
  ok = ok && symmetric && center_in && corner_out;

  printf("mandelbrot %dx%d: %s (symmetric=%d center_in=%d corner_out=%d) digest=%08x\n",
         W, H, ok ? "OK" : "FAIL", symmetric, center_in, corner_out, fnv(img, W * H));

#ifdef SVM_GUEST
  int wrote = fs_write_file("mandel.rgba", img, W * H * 4);
  printf("wrote mandel.rgba: %s\n", wrote == 0 ? "OK" : "FAIL");
  ok = ok && (wrote == 0);
#endif

  printf("webgpu mandelbrot: %s\n", ok ? "ALL OK" : "FAIL");
  return ok ? 0 : 1;
}
