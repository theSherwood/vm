/* Shakedown driver: run Sean Barrett's stb_perlin (public domain) in the sandbox.
 *
 * This is the first *floating-point-heavy* shakedown. Perlin noise is dense f32
 * arithmetic — gradient dot products (grad[0]*x + grad[1]*y + grad[2]*z), the
 * quintic ease polynomial ((a*6-15)*a+10)*a*a*a, and trilinear lerps a+(b-a)*t —
 * plus int<->float conversions (fastfloor) and, in the fbm/turbulence/ridge
 * variants, multiply/accumulate chains over octaves. It exercises the IR's f32
 * path end-to-end through the frontend for the first time.
 *
 * To get byte-exact parity we avoid float *formatting* (printf %f rounding is its
 * own risk): each noise value is scaled to a fixed-point integer and printed. So
 * any divergence in the actual f32 arithmetic between native cc and our JIT shows
 * up directly in the digits. */

#include <stddef.h>

/* stb_perlin's ridge/turbulence/fbm use fabs(); the sandbox build has no libm. */
double fabs(double x) { return x < 0 ? -x : x; }

#define STB_PERLIN_IMPLEMENTATION
#include "stb_perlin.h"

int write(int fd, char *buf, long n);

static void puts_(const char *s) {
  int n = 0;
  while (s[n]) n++;
  write(1, (char *)s, n);
}

/* Print a signed integer in decimal followed by `sep`. */
static void puti(int v, char sep) {
  char buf[16];
  int i = sizeof(buf);
  buf[--i] = sep;
  unsigned u = v < 0 ? (unsigned)(-v) : (unsigned)v;
  if (u == 0) buf[--i] = '0';
  while (u) {
    buf[--i] = (char)('0' + u % 10);
    u /= 10;
  }
  if (v < 0) buf[--i] = '-';
  write(1, &buf[i], (long)(sizeof(buf) - i));
}

/* Scale a noise value into fixed point (5 fractional digits) and emit it. */
static void emit(float v, char sep) { puti((int)(v * 100000.0f), sep); }

int main(void) {
  /* A small 3D grid of plain Perlin noise. */
  puts_("noise3:\n");
  for (int yi = 0; yi < 4; yi++) {
    for (int xi = 0; xi < 8; xi++) {
      float x = xi * 0.35f, y = yi * 0.35f, z = 1.5f;
      emit(stb_perlin_noise3(x, y, z, 0, 0, 0), xi == 7 ? '\n' : ' ');
    }
  }
  /* Wrapped noise (power-of-two tiling) along a line. */
  puts_("wrap:\n");
  for (int xi = 0; xi < 8; xi++)
    emit(stb_perlin_noise3(xi * 0.5f, 0.25f, 0.75f, 4, 4, 4), xi == 7 ? '\n' : ' ');
  /* The octave-accumulating variants — multiply/accumulate chains + fabs. */
  puts_("fbm/turb/ridge:\n");
  for (int xi = 0; xi < 6; xi++) {
    float x = xi * 0.6f + 0.1f, y = 2.3f, z = 0.7f;
    emit(stb_perlin_fbm_noise3(x, y, z, 2.0f, 0.5f, 6), ' ');
    emit(stb_perlin_turbulence_noise3(x, y, z, 2.0f, 0.5f, 6), ' ');
    emit(stb_perlin_ridge_noise3(x, y, z, 2.0f, 0.5f, 1.0f, 6), '\n');
  }
  return 0;
}
