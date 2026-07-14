/* Framebuffer demo: render a deterministic RGBA gradient and present one frame through the
 * `display` capability — the browser blits it to a <canvas>. There is no stdout; the *frame* is the
 * output. This is the first guest to drive the framebuffer output waist (the path Doom rides): the
 * §7 host-defined `display` capability, resolved by name like Lua's `io` / SQLite's VFS `fs`.
 *
 *   d = __vm_cap_resolve("display", 7);         // the granted framebuffer handle (< 0 if ungranted)
 *   __vm_host_call(d, 0, ptr, w, h, 0);         // op 0 = present(ptr, w, h): w*h*4 RGBA bytes
 *
 * The image is a pure function of (x, y) — fully deterministic, so the captured bytes are the
 * differential anchor (`browser/tests/display.rs` asserts every pixel against the same formula).
 * RGBA, row-major, top row first (the canvas ImageData layout, so the host presents it with one
 * putImageData, no flip). */
extern int __vm_cap_resolve(const char *name, long len);
extern long __vm_host_call(int h, int op, long a, long b, long c, long d);

#define W 128
#define H 128
static unsigned char fb[W * H * 4];

int main(void) {
  for (int y = 0; y < H; y++) {
    for (int x = 0; x < W; x++) {
      unsigned char *px = &fb[(y * W + x) * 4];
      px[0] = (unsigned char)(x * 255 / (W - 1)); /* R ramps left→right  */
      px[1] = (unsigned char)(y * 255 / (H - 1)); /* G ramps top→bottom  */
      px[2] = (unsigned char)(((x ^ y) & 63) * 4); /* B: a subtle xor weave */
      px[3] = 255;                                /* A: opaque            */
    }
  }
  int d = __vm_cap_resolve("display", 7);
  if (d < 0) return 1; /* no framebuffer granted — nothing to present */
  __vm_host_call(d, 0, (long)fb, W, H, 0);
  return 0;
}
