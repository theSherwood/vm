/* Interactive Mandelbrot zoom (a playground framebuffer demo). A reactor guest: the host calls the
 * exported `tick()` once per animation frame, and per-frame state (the view rectangle) persists in
 * globals. Each frame slowly zooms toward a fixed interesting point, computes the Mandelbrot escape
 * for every pixel **on the CPU in-guest** (double-precision `z = z² + c`), colors it with a cycling
 * integer rainbow palette, and presents the RGBA frame through the `display` capability — the same
 * output waist `gradient`/`life`/Doom ride. The arrow keys steer the zoom target (nudges scale with
 * the zoom, so exploring feels natural); Up/Down also zoom faster/slower.
 *
 *   d = __vm_cap_resolve("display", 7);   __vm_host_call(d, 0, ptr, w, h, 0);  // present(ptr,w,h)
 *   k = __vm_cap_resolve("keyboard", 8);  e = __vm_host_call(k, 0, 0,0,0,0);   // poll() → event | -1
 *
 * A key event is packed `(pressed << 16) | keycode` (JS keyCodes: Left 37, Up 38, Right 39, Down 40).
 * No libm: the escape iteration is plain f64 arithmetic and the palette is integer, so it lowers
 * through the on-ramp with nothing bundled. The view is a pure function of the frame count + the key
 * events, so a fixed input script yields a deterministic frame sequence. */

extern int __vm_cap_resolve(const char *name, long len);
extern long __vm_host_call(int h, int op, long a, long b, long c, long d);

#define W 240
#define H 180
#define MAXIT 110

static unsigned char fb[W * H * 4];
static int disp, kbd;

/* view: center (cx, cy) and half-height `scale` in the complex plane */
static double cx, cy, scale;
static double tx, ty;   /* the zoom target the auto-zoom homes toward (steerable) */
static int frame;
static int zooming = 1; /* auto-zoom on by default */

/* A cycling 6-segment rainbow: t in 0..255 → an RGB on the color wheel (no trig). */
static void rainbow(int t, unsigned char *rgb) {
  t &= 255;
  int i = t / 43, f = (t % 43) * 6;
  if (f > 255) f = 255;
  switch (i) {
    case 0:  rgb[0] = 255;     rgb[1] = f;       rgb[2] = 0;       break;
    case 1:  rgb[0] = 255 - f; rgb[1] = 255;     rgb[2] = 0;       break;
    case 2:  rgb[0] = 0;       rgb[1] = 255;     rgb[2] = f;       break;
    case 3:  rgb[0] = 0;       rgb[1] = 255 - f; rgb[2] = 255;     break;
    case 4:  rgb[0] = f;       rgb[1] = 0;       rgb[2] = 255;     break;
    default: rgb[0] = 255;     rgb[1] = 0;       rgb[2] = 255 - f; break;
  }
}

int main(void) {
  disp = __vm_cap_resolve("display", 7);
  kbd = __vm_cap_resolve("keyboard", 8);
  /* a classic "seahorse valley" point — endless structure to zoom into */
  tx = -0.743643887037151;
  ty = 0.131825904205330;
  cx = -0.5;
  cy = 0.0;
  scale = 1.25;
  frame = 0;
  return 0;
}

int tick(void) {
  /* drain input: arrows nudge the target (scaled to the zoom), Up/Down also toggle zoom direction */
  for (;;) {
    long e = __vm_host_call(kbd, 0, 0, 0, 0, 0);
    if (e < 0) break;
    if ((e >> 16) & 1) {
      int code = (int)(e & 0xffff);
      double step = scale * 0.15;
      if (code == 37) tx -= step;      /* Left  */
      else if (code == 39) tx += step; /* Right */
      else if (code == 38) ty -= step; /* Up    */
      else if (code == 40) ty += step; /* Down  */
    }
  }

  /* ease the center toward the target, then zoom in; loop back out when very deep */
  cx += (tx - cx) * 0.06;
  cy += (ty - cy) * 0.06;
  if (zooming) {
    scale *= 0.975;
    if (scale < 2.0e-13) scale = 1.25; /* f64 detail runs out — reset the zoom loop */
  }

  double aspect = (double)W / (double)H;
  for (int py = 0; py < H; py++) {
    double y0 = cy + ((double)py - H / 2.0) / (H / 2.0) * scale;
    for (int px = 0; px < W; px++) {
      double x0 = cx + ((double)px - W / 2.0) / (W / 2.0) * scale * aspect;
      double x = 0.0, y = 0.0;
      int it = 0;
      while (it < MAXIT) {
        double xt = x * x - y * y + x0;
        y = 2.0 * x * y + y0;
        x = xt;
        if (x * x + y * y > 4.0) break;
        it++;
      }
      unsigned char *p = &fb[(py * W + px) * 4];
      if (it >= MAXIT) {
        p[0] = p[1] = p[2] = 0; /* inside the set: black */
      } else {
        rainbow(it * 3 + frame * 2, p); /* escape count → cycling rainbow */
      }
      p[3] = 255;
    }
  }

  __vm_host_call(disp, 0, (long)fb, W, H, 0);
  frame++;
  return 0;
}
