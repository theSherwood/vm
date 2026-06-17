/* Shakedown driver: a tiny ASCII sphere raytracer, to exercise **transcendental libm** — a small
 * `libm` bundled *as guest code* (the on-ramp keeps math in the sandbox: no host math capability).
 * One unit sphere, orthographic rays, diffuse lighting + sinusoidal surface bands + an exponential
 * rim falloff, rendered to a character ramp and written row-by-row. `sqrt`/`floor` lower to SVM
 * float ops (slices F/L); `g_sin`/`g_exp` are guest polynomial approximations. The differential is
 * clean *because* the transcendentals are guest code: native `cc` compiles the same `g_sin`/`g_exp`,
 * so every value is bit-identical (the only machine ops in play — `sqrt`, `fmuladd`, +−*∕ — are
 * IEEE / unfused on both sides). svm-run's output must match a native `cc` build byte-for-byte. */
#include <stddef.h>

long write(int fd, const void *buf, long n);
double sqrt(double);
double floor(double);

/* bundled guest libm (poly approximations; runs entirely in the guest) */
static double g_sin(double x) {
  const double TAU = 6.28318530717958647692;
  x -= TAU * floor(x / TAU + 0.5); /* range-reduce to [-pi, pi] */
  double t = x * x;
  return x * (1.0 + t * (-1.0 / 6 + t * (1.0 / 120 + t * (-1.0 / 5040))));
}
static double g_exp(double x) {
  const double LN2 = 0.69314718055994530942;
  double kf = floor(x / LN2 + 0.5);
  int k = (int)kf;
  double r = x - kf * LN2; /* r in [-ln2/2, ln2/2] */
  double er = 1.0 + r * (1.0 + r * (0.5 + r * (1.0 / 6 + r * (1.0 / 24 + r / 120))));
  double p = 1.0;
  if (k >= 0)
    for (int i = 0; i < k; i++) p *= 2.0;
  else
    for (int i = 0; i < -k; i++) p *= 0.5;
  return er * p; /* 2^k * exp(r) */
}

int main(void) {
  const int W = 56, H = 22;
  const char ramp[] = " .:-=+*#%@";
  const int RN = (int)sizeof(ramp) - 2;
  char row[80];
  double lx = -0.7, ly = 0.7, lz = -0.6; /* light direction */
  double ll = sqrt(lx * lx + ly * ly + lz * lz);
  lx /= ll;
  ly /= ll;
  lz /= ll;
  for (int j = 0; j < H; j++) {
    for (int i = 0; i < W; i++) {
      double px = ((double)i / (W - 1) - 0.5) * 3.0;
      double py = (0.5 - (double)j / (H - 1)) * 3.0 * ((double)H / W) * 2.0;
      double ox = px, oy = py, oz = -4.0; /* ray origin; dir = +z */
      double b = 2.0 * oz;                /* d=(0,0,1) ⇒ b=2 o·d, c=|o|^2-1 */
      double c = ox * ox + oy * oy + oz * oz - 1.0;
      double disc = b * b - 4.0 * c;
      char ch = ' ';
      if (disc >= 0.0) {
        double t = (-b - sqrt(disc)) * 0.5;
        double hx = ox, hy = oy, hz = oz + t; /* hit = normal (unit sphere) */
        double diff = hx * lx + hy * ly + hz * lz;
        if (diff < 0) diff = 0;
        double stripe = 0.5 + 0.5 * g_sin(hy * 9.0); /* sinusoidal bands */
        double rim = g_exp(hz * 1.2);                /* hz<0 facing cam: dim rim */
        double v = diff * (0.45 + 0.55 * stripe) * (0.4 + 0.6 * rim);
        int idx = (int)(v * RN + 0.5);
        if (idx < 0) idx = 0;
        if (idx > RN) idx = RN;
        ch = ramp[idx];
      }
      row[i] = ch;
    }
    row[W] = '\n';
    write(1, row, W + 1);
  }
  return 0;
}
