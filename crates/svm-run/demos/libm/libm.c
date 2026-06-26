/* A small, self-contained **guest `libm`** for the SVM LLVM on-ramp.
 *
 * The on-ramp keeps math *in the sandbox*: a program that needs the transcendentals brings them as
 * ordinary guest C, which translate as normal functions (no host math capability, no translator
 * intrinsic). A guest definition also **shadows** any libc-name binding the on-ramp would otherwise
 * synthesize — so linking this file into a program (e.g. Lua: `llvm-link lua_core.bc libm.bc`)
 * replaces the `pow`/`exp`/`log` trap stubs with these real implementations.
 *
 * These are faithful transcriptions of Sun's **fdlibm** (`__ieee754_exp`/`log`/`pow`) — genuinely
 * accurate (sub-ULP), not poly approximations — using only IEEE `+ - * /`, comparisons, and double
 * word access via a union. Every op is bit-deterministic and identical on a native `cc` build and on
 * the on-ramp (baseline x86-64 has no FMA, so `a*b+c` is unfused on both sides), so a differential
 * demo built from the same source is byte-identical across all three engines and native.
 *
 * Build for linking: `clang -O2 -fno-builtin -emit-llvm -c libm.c -o libm.bc`. (`-fno-builtin` keeps
 * clang from rewriting the bodies in terms of the very functions they define.)
 */

/* ── double word access (little-endian; both lanes are x86-64) ───────────────────────────────── */
typedef union {
  double d;
  unsigned long long u;
} libm_du;

static unsigned libm_hi(double x) {
  libm_du t;
  t.d = x;
  return (unsigned)(t.u >> 32);
}
static unsigned libm_lo(double x) {
  libm_du t;
  t.d = x;
  return (unsigned)t.u;
}
static double libm_words(unsigned hi, unsigned lo) {
  libm_du t;
  t.u = ((unsigned long long)hi << 32) | lo;
  return t.d;
}
static double libm_set_hi(double x, unsigned hi) {
  libm_du t;
  t.d = x;
  t.u = ((unsigned long long)hi << 32) | (unsigned)t.u;
  return t.d;
}
static double libm_set_lo(double x, unsigned lo) {
  libm_du t;
  t.d = x;
  t.u = (t.u & 0xffffffff00000000ULL) | lo;
  return t.d;
}

/* `sqrt` lowers to the SVM `f64.sqrt` op and `scalbn` to the synthesized `__svm_ldexp` (both
 * bit-exact to libc); native links them from `-lm`. So `pow` can use them without a guest body. */
double sqrt(double);
double scalbn(double, int);

/* ── exp (fdlibm __ieee754_exp) ──────────────────────────────────────────────────────────────── */
double exp(double x) {
  static const double one = 1.0, halF[2] = {0.5, -0.5},
                      o_threshold = 7.09782712893383973096e+02,
                      u_threshold = -7.45133219101941108420e+02,
                      ln2HI[2] = {6.93147180369123816490e-01, -6.93147180369123816490e-01},
                      ln2LO[2] = {1.90821492927058770002e-10, -1.90821492927058770002e-10},
                      invln2 = 1.44269504088896338700e+00, P1 = 1.66666666666666019037e-01,
                      P2 = -2.77777777770155933842e-03, P3 = 6.61375632143793436117e-05,
                      P4 = -1.65339022054652515390e-06, P5 = 4.13813679705723846039e-08,
                      huge = 1.0e+300, twom1000 = 9.33263618503218878990e-302;
  double y, hi = 0, lo = 0, c, t, twopk;
  int k = 0, xsb;
  unsigned hx;

  hx = libm_hi(x);
  xsb = (hx >> 31) & 1; /* sign bit of x */
  hx &= 0x7fffffff;     /* high word of |x| */

  /* filter out non-finite argument */
  if (hx >= 0x40862E42) {  /* if |x| >= 709.78... */
    if (hx >= 0x7ff00000) { /* inf or NaN */
      unsigned lx = libm_lo(x);
      if (((hx & 0xfffff) | lx) != 0) return x + x;  /* NaN */
      return (xsb == 0) ? x : 0.0;                   /* exp(+-inf) = {inf, 0} */
    }
    if (x > o_threshold) return huge * huge;            /* overflow */
    if (x < u_threshold) return twom1000 * twom1000;    /* underflow */
  }

  /* argument reduction: x = k*ln2 + r,  |r| <= ln2/2 */
  if (hx > 0x3fd62e42) {     /* |x| > 0.5 ln2 */
    if (hx < 0x3FF0A2B2) {   /* and |x| < 1.5 ln2 */
      hi = x - ln2HI[xsb];
      lo = ln2LO[xsb];
      k = 1 - xsb - xsb;
    } else {
      k = (int)(invln2 * x + halF[xsb]);
      t = k;
      hi = x - t * ln2HI[0]; /* t*ln2HI is exact here */
      lo = t * ln2LO[0];
    }
    x = hi - lo;
  } else if (hx < 0x3e300000) { /* |x| < 2**-28 */
    if (huge + x > one) return one + x; /* trigger inexact */
  } else {
    k = 0;
  }

  /* x is now in the primary range */
  t = x * x;
  if (k >= -1021)
    twopk = libm_words((unsigned)(0x3ff + k) << 20, 0);
  else
    twopk = libm_words((unsigned)(0x3ff + (k + 1000)) << 20, 0);
  c = x - t * (P1 + t * (P2 + t * (P3 + t * (P4 + t * P5))));
  if (k == 0) return one - ((x * c) / (c - 2.0) - x);
  y = one - ((lo - (x * c) / (2.0 - c)) - hi);
  if (k >= -1021) {
    if (k == 1024) return y * 2.0 * 0x1p1023;
    return y * twopk;
  }
  return y * twopk * twom1000;
}

/* ── log (fdlibm __ieee754_log) ──────────────────────────────────────────────────────────────── */
double log(double x) {
  static const double ln2_hi = 6.93147180369123816490e-01, ln2_lo = 1.90821492927058770002e-10,
                      two54 = 1.80143985094819840000e+16, Lg1 = 6.666666666666735130e-01,
                      Lg2 = 3.999999999940941908e-01, Lg3 = 2.857142874366239149e-01,
                      Lg4 = 2.222219843214978396e-01, Lg5 = 1.818357216161805012e-01,
                      Lg6 = 1.531383769920937332e-01, Lg7 = 1.479819860511658591e-01, zero = 0.0;
  double hfsq, f, s, z, R, w, t1, t2, dk;
  int k, i, j;
  int hx;
  unsigned lx;

  hx = (int)libm_hi(x);
  lx = libm_lo(x);

  k = 0;
  if (hx < 0x00100000) {                       /* x < 2**-1022 (subnormal or 0/neg) */
    if (((hx & 0x7fffffff) | (int)lx) == 0) return -two54 / zero; /* log(+-0) = -inf */
    if (hx < 0) return (x - x) / zero;          /* log(-#) = NaN */
    k -= 54;
    x *= two54; /* subnormal: scale up */
    hx = (int)libm_hi(x);
  }
  if (hx >= 0x7ff00000) return x + x; /* inf or NaN */
  k += (hx >> 20) - 1023;
  hx &= 0x000fffff;
  i = (hx + 0x95f64) & 0x100000;
  x = libm_set_hi(x, (unsigned)(hx | (i ^ 0x3ff00000))); /* normalize x or x/2 */
  k += (i >> 20);
  f = x - 1.0;
  if ((0x000fffff & (2 + hx)) < 3) { /* -2**-20 <= f < 2**-20 */
    if (f == zero) {
      if (k == 0) return zero;
      dk = (double)k;
      return dk * ln2_hi + dk * ln2_lo;
    }
    R = f * f * (0.5 - 0.33333333333333333 * f);
    if (k == 0) return f - R;
    dk = (double)k;
    return dk * ln2_hi - ((R - dk * ln2_lo) - f);
  }
  s = f / (2.0 + f);
  dk = (double)k;
  z = s * s;
  i = hx - 0x6147a;
  w = z * z;
  j = 0x6b851 - hx;
  t1 = w * (Lg2 + w * (Lg4 + w * Lg6));
  t2 = z * (Lg1 + w * (Lg3 + w * (Lg5 + w * Lg7)));
  i |= j;
  R = t2 + t1;
  if (i > 0) {
    hfsq = 0.5 * f * f;
    if (k == 0) return f - (hfsq - s * (hfsq + R));
    return dk * ln2_hi - ((hfsq - (s * (hfsq + R) + dk * ln2_lo)) - f);
  }
  if (k == 0) return f - s * (f - R);
  return dk * ln2_hi - ((s * (f - R) - dk * ln2_lo) - f);
}

/* ── pow (fdlibm __ieee754_pow) ──────────────────────────────────────────────────────────────── */
double pow(double x, double y) {
  static const double bp[] = {1.0, 1.5},
                      dp_h[] = {0.0, 5.84962487220764160156e-01},
                      dp_l[] = {0.0, 1.35003920212974897128e-08}, zero = 0.0, one = 1.0, two = 2.0,
                      two53 = 9007199254740992.0, huge = 1.0e300, tiny = 1.0e-300,
                      L1 = 5.99999999999994648725e-01, L2 = 4.28571428578550184252e-01,
                      L3 = 3.33333329818377432918e-01, L4 = 2.72728123808534006489e-01,
                      L5 = 2.30660745775561366331e-01, L6 = 2.06975017800338417784e-01,
                      P1 = 1.66666666666666019037e-01, P2 = -2.77777777770155933842e-03,
                      P3 = 6.61375632143793436117e-05, P4 = -1.65339022054652515390e-06,
                      P5 = 4.13813679705723846039e-08, lg2 = 6.93147180559945286227e-01,
                      lg2_h = 6.93147182464599609375e-01, lg2_l = -1.90465429995776804525e-09,
                      ovt = 8.0085662595372944372e-0017, cp = 9.61796693925975554329e-01,
                      cp_h = 9.61796700954437255859e-01, cp_l = -7.02846165095275826516e-09,
                      ivln2 = 1.44269504088896338700e+00, ivln2_h = 1.44269502162933349609e+00,
                      ivln2_l = 1.92596299112661746887e-08;
  double z, ax, z_h, z_l, p_h, p_l, y1, t1, t2, r, s, t, u, v, w;
  int i, j, k, yisint, n;
  int hx, hy, ix, iy;
  unsigned lx, ly;

  hx = (int)libm_hi(x);
  lx = libm_lo(x);
  hy = (int)libm_hi(y);
  ly = libm_lo(y);
  ix = hx & 0x7fffffff;
  iy = hy & 0x7fffffff;

  if ((iy | (int)ly) == 0) return one; /* x**0 = 1 */

  /* +-NaN return x+y */
  if (ix > 0x7ff00000 || ((ix == 0x7ff00000) && (lx != 0)) || iy > 0x7ff00000 ||
      ((iy == 0x7ff00000) && (ly != 0)))
    return x + y;

  /* yisint: 0 = not int, 1 = odd int, 2 = even int (only matters when x < 0) */
  yisint = 0;
  if (hx < 0) {
    if (iy >= 0x43400000)
      yisint = 2;
    else if (iy >= 0x3ff00000) {
      k = (iy >> 20) - 0x3ff;
      if (k > 20) {
        j = (int)(ly >> (52 - k));
        if ((unsigned)(j << (52 - k)) == ly) yisint = 2 - (j & 1);
      } else if (ly == 0) {
        j = iy >> (20 - k);
        if ((j << (20 - k)) == iy) yisint = 2 - (j & 1);
      }
    }
  }

  /* special value of y */
  if (ly == 0) {
    if (iy == 0x7ff00000) { /* y is +-inf */
      if (((ix - 0x3ff00000) | (int)lx) == 0)
        return one; /* (-1)**+-inf = 1 */
      else if (ix >= 0x3ff00000)
        return (hy >= 0) ? y : zero; /* (|x|>1)**+-inf */
      else
        return (hy < 0) ? -y : zero; /* (|x|<1)**+-inf */
    }
    if (iy == 0x3ff00000) { /* y is +-1 */
      if (hy < 0)
        return one / x;
      else
        return x;
    }
    if (hy == 0x40000000) return x * x;            /* y is 2 */
    if (hy == 0x3fe00000 && hx >= 0) return sqrt(x); /* y is 0.5, x >= +0 */
  }

  ax = libm_words((unsigned)ix, lx); /* |x| */

  /* special value of x: +-0, +-inf, +-1 */
  if (lx == 0) {
    if (ix == 0x7ff00000 || ix == 0 || ix == 0x3ff00000) {
      z = ax;
      if (hy < 0) z = one / z;
      if (hx < 0) {
        if (((ix - 0x3ff00000) | yisint) == 0)
          z = (z - z) / (z - z); /* (-1)**non-int = NaN */
        else if (yisint == 1)
          z = -z; /* (x<0)**odd = -(|x|**odd) */
      }
      return z;
    }
  }

  n = (hx >> 31) + 1;
  if ((n | yisint) == 0) return (x - x) / (x - x); /* (x<0)**(non-int) = NaN */

  s = one;
  if ((n | (yisint - 1)) == 0) s = -one; /* (-ve)**(odd int) */

  /* |y| is huge */
  if (iy > 0x41e00000) {   /* |y| > 2**31 */
    if (iy > 0x43f00000) { /* |y| > 2**64: must over/underflow */
      if (ix <= 0x3fefffff) return (hy < 0) ? huge * huge : tiny * tiny;
      if (ix >= 0x3ff00000) return (hy > 0) ? huge * huge : tiny * tiny;
    }
    if (ix < 0x3fefffff) return (hy < 0) ? s * huge * huge : s * tiny * tiny;
    if (ix > 0x3ff00000) return (hy > 0) ? s * huge * huge : s * tiny * tiny;
    /* now |1-x| is tiny <= 2**-20: compute log(x) by the series */
    t = ax - one;
    w = (t * t) * (0.5 - t * (0.3333333333333333333333 - t * 0.25));
    u = ivln2_h * t;
    v = t * ivln2_l - w * ivln2;
    t1 = u + v;
    t1 = libm_set_lo(t1, 0);
    t2 = v - (t1 - u);
  } else {
    double ss, s2, s_h, s_l, t_h, t_l;
    n = 0;
    if (ix < 0x00100000) { /* subnormal x */
      ax *= two53;
      n -= 53;
      ix = (int)libm_hi(ax);
    }
    n += (ix >> 20) - 0x3ff;
    j = ix & 0x000fffff;
    ix = j | 0x3ff00000; /* normalize ix into [1,2) */
    if (j <= 0x3988E)
      k = 0; /* |x| < sqrt(3/2) */
    else if (j < 0xBB67A)
      k = 1; /* |x| < sqrt(3) */
    else {
      k = 0;
      n += 1;
      ix -= 0x00100000;
    }
    ax = libm_set_hi(ax, (unsigned)ix);

    /* ss = (x-1)/(x+1) or (x-1.5)/(x+1.5) */
    u = ax - bp[k];
    v = one / (ax + bp[k]);
    ss = u * v;
    s_h = ss;
    s_h = libm_set_lo(s_h, 0);
    t_h = zero;
    t_h = libm_set_hi(t_h, (unsigned)(((ix >> 1) | 0x20000000) + 0x00080000 + (k << 18)));
    t_l = ax - (t_h - bp[k]);
    s_l = v * ((u - s_h * t_h) - s_h * t_l);
    /* log(ax) */
    s2 = ss * ss;
    r = s2 * s2 * (L1 + s2 * (L2 + s2 * (L3 + s2 * (L4 + s2 * (L5 + s2 * L6)))));
    r += s_l * (s_h + ss);
    s2 = s_h * s_h;
    t_h = 3.0 + s2 + r;
    t_h = libm_set_lo(t_h, 0);
    t_l = r - ((t_h - 3.0) - s2);
    u = s_h * t_h;
    v = s_l * t_h + t_l * ss;
    p_h = u + v;
    p_h = libm_set_lo(p_h, 0);
    p_l = v - (p_h - u);
    z_h = cp_h * p_h;
    z_l = cp_l * p_h + p_l * cp + dp_l[k];
    t = (double)n;
    t1 = (((z_h + z_l) + dp_h[k]) + t);
    t1 = libm_set_lo(t1, 0);
    t2 = z_l - (((t1 - t) - dp_h[k]) - z_h);
  }

  /* split y into y1+y2 and compute (y1+y2)*(t1+t2) */
  y1 = y;
  y1 = libm_set_lo(y1, 0);
  p_l = (y - y1) * t1 + y * t2;
  p_h = y1 * t1;
  z = p_l + p_h;
  j = (int)libm_hi(z);
  i = (int)libm_lo(z);
  if (j >= 0x40900000) { /* z >= 1024 */
    if (((j - 0x40900000) | i) != 0)
      return s * huge * huge; /* overflow */
    if (p_l + ovt > z - p_h) return s * huge * huge;
  } else if ((j & 0x7fffffff) >= 0x4090cc00) { /* z <= -1075 */
    if (((j - (int)0xc090cc00) | i) != 0)
      return s * tiny * tiny; /* underflow */
    if (p_l <= z - p_h) return s * tiny * tiny;
  }

  /* compute 2**(p_h+p_l) */
  i = j & 0x7fffffff;
  k = (i >> 20) - 0x3ff;
  n = 0;
  if (i > 0x3fe00000) { /* |z| > 0.5: set n = [z+0.5] */
    n = j + (0x00100000 >> (k + 1));
    k = ((n & 0x7fffffff) >> 20) - 0x3ff;
    t = zero;
    t = libm_set_hi(t, (unsigned)(n & ~(0x000fffff >> k)));
    n = ((n & 0x000fffff) | 0x00100000) >> (20 - k);
    if (j < 0) n = -n;
    p_h -= t;
  }
  t = p_l + p_h;
  t = libm_set_lo(t, 0);
  u = t * lg2_h;
  v = (p_l - (t - p_h)) * lg2 + t * lg2_l;
  z = u + v;
  w = v - (z - u);
  t = z * z;
  t1 = z - t * (P1 + t * (P2 + t * (P3 + t * (P4 + t * P5))));
  r = (z * t1) / (t1 - two) - (w + z * w);
  z = one - (r - z);
  j = (int)libm_hi(z);
  j += (n << 20);
  if ((j >> 20) <= 0)
    z = scalbn(z, n); /* subnormal output */
  else
    z = libm_set_hi(z, (unsigned)j);
  return s * z;
}

/* ── sin/cos (fdlibm __kernel_sin/__kernel_cos + medium-path argument reduction) ──────────────────
 *
 * Reduce x to r in [-pi/4, pi/4] plus a quadrant n (mod 4); each kernel is a minimax polynomial over
 * [-pi/4, pi/4]. The reduction is fdlibm's **medium path**, accurate for |x| <= 2^20*(pi/2) ~ 1.65e6
 * (the bound where `n*pio2_1` stays an exact double product) — covering all realistic use. We drop
 * fdlibm's `npio2_hw[]` fast-path table (a pure optimization: always running the cancellation-
 * correction iterations is equally correct), so there is no big table to transcribe. For |x| beyond
 * the bound the reduction is **reduced precision** (a documented limitation; the full Payne-Hanek
 * `__kernel_rem_pio2` table is future work) but the functions stay total. */

static double libm_kernel_sin(double x, double y, int iy) {
  static const double half = 5.00000000000000000000e-01, S1 = -1.66666666666666324348e-01,
                      S2 = 8.33333333332248946124e-03, S3 = -1.98412698298579493134e-04,
                      S4 = 2.75573137070700676789e-06, S5 = -2.50507602534068634195e-08,
                      S6 = 1.58969099521155010221e-10;
  double z, r, v;
  int ix = (int)libm_hi(x) & 0x7fffffff;
  if (ix < 0x3e400000) {       /* |x| < 2**-27 */
    if ((int)x == 0) return x; /* generate inexact */
  }
  z = x * x;
  v = z * x;
  r = S2 + z * (S3 + z * (S4 + z * (S5 + z * S6)));
  if (iy == 0) return x + v * (S1 + z * r);
  return x - ((z * (half * y - v * r) - y) - v * S1);
}

static double libm_kernel_cos(double x, double y) {
  static const double one = 1.00000000000000000000e+00, C1 = 4.16666666666666019037e-02,
                      C2 = -1.38888888888741095749e-03, C3 = 2.48015872894767294178e-05,
                      C4 = -2.75573143513906633035e-07, C5 = 2.08757232129817482790e-09,
                      C6 = -1.13596475577881948265e-11;
  double a, hz, z, r, qx;
  int ix = (int)libm_hi(x) & 0x7fffffff;
  if (ix < 0x3e400000) { /* |x| < 2**-27 */
    if ((int)x == 0) return one;
  }
  z = x * x;
  r = z * (C1 + z * (C2 + z * (C3 + z * (C4 + z * (C5 + z * C6)))));
  if (ix < 0x3FD33333) /* |x| < 0.3 */
    return one - (0.5 * z - (z * r - x * y));
  if (ix > 0x3fe90000)
    qx = 0.28125; /* x > 0.78125 */
  else
    qx = libm_words((unsigned)(ix - 0x00200000), 0); /* x/4 */
  hz = 0.5 * z - qx;
  a = one - qx;
  return a - (hz - (z * r - x * y));
}

/* Reduce x: write r = y[0]+y[1] in [-pi/4, pi/4], return the quadrant count n (n & 3 = quadrant). */
static int libm_rem_pio2(double x, double *y) {
  static const double half = 0.5, invpio2 = 6.36619772367581382433e-01,
                      pio2_1 = 1.57079632673412561417e+00, pio2_1t = 6.07710050650619224932e-11,
                      pio2_2 = 6.07710050630396597660e-11, pio2_2t = 2.02226624879595063154e-21,
                      pio2_3 = 2.02226624871116645580e-21, pio2_3t = 8.47842766036889956997e-32,
                      toint = 6755399441055744.0; /* 2^52+2^51, round-to-nearest magic */
  double z, w, t, r, fn;
  int i, j, n, hx, ix;
  unsigned high;

  hx = (int)libm_hi(x);
  ix = hx & 0x7fffffff;
  if (ix <= 0x3fe921fb) { /* |x| <= pi/4: no reduction */
    y[0] = x;
    y[1] = 0;
    return 0;
  }
  if (ix < 0x4002d97c) { /* |x| < 3pi/4: n = +-1 */
    if (hx > 0) {
      z = x - pio2_1;
      if (ix != 0x3ff921fb) {
        y[0] = z - pio2_1t;
        y[1] = (z - y[0]) - pio2_1t;
      } else {
        z -= pio2_2;
        y[0] = z - pio2_2t;
        y[1] = (z - y[0]) - pio2_2t;
      }
      return 1;
    }
    z = x + pio2_1;
    if (ix != 0x3ff921fb) {
      y[0] = z + pio2_1t;
      y[1] = (z - y[0]) + pio2_1t;
    } else {
      z += pio2_2;
      y[0] = z + pio2_2t;
      y[1] = (z - y[0]) + pio2_2t;
    }
    return -1;
  }
  if (ix <= 0x413921fb) {                       /* |x| <= 2^20*(pi/2): medium, full accuracy */
    t = libm_words((unsigned)ix, libm_lo(x));   /* |x| */
    n = (int)(t * invpio2 + half);
    fn = (double)n;
    r = t - fn * pio2_1;
    w = fn * pio2_1t; /* 1st round, ~85-bit */
    j = ix >> 20;
    y[0] = r - w;
    high = libm_hi(y[0]);
    i = j - (int)((high >> 20) & 0x7ff);
    if (i > 16) { /* 2nd iteration, ~118-bit */
      t = r;
      w = fn * pio2_2;
      r = t - w;
      w = fn * pio2_2t - ((t - r) - w);
      y[0] = r - w;
      high = libm_hi(y[0]);
      i = j - (int)((high >> 20) & 0x7ff);
      if (i > 49) { /* 3rd iteration, ~151-bit */
        t = r;
        w = fn * pio2_3;
        r = t - w;
        w = fn * pio2_3t - ((t - r) - w);
        y[0] = r - w;
      }
    }
    y[1] = (r - y[0]) - w;
    if (hx < 0) {
      y[0] = -y[0];
      y[1] = -y[1];
      return -n;
    }
    return n;
  }
  /* |x| > 2^20*(pi/2): out of the validated domain — a reduced-precision reduction (1 extra part)
   * so the function is total. Realistic Lua never reaches here; the full Payne-Hanek table is the
   * future addition that would restore <=2 ULP for huge arguments. */
  t = libm_words((unsigned)ix, libm_lo(x)); /* |x| */
  fn = (t * invpio2 + toint) - toint;       /* nearest integer (degrades past 2^52) */
  r = t - fn * pio2_1;
  w = fn * pio2_1t;
  z = r - w;
  w = fn * pio2_2 - ((r - z) - w);
  r = z - w;
  y[0] = r;
  y[1] = (z - r) - w;
  z = fn * 0.25;
  n = (int)(fn - 4.0 * (double)(long long)z); /* fn mod 4 */
  if (hx < 0) {
    y[0] = -y[0];
    y[1] = -y[1];
    n = (4 - n) & 3;
  }
  return n & 3;
}

double sin(double x) {
  double y[2], z = 0.0;
  int n, ix = (int)libm_hi(x) & 0x7fffffff;
  if (ix <= 0x3fe921fb) return libm_kernel_sin(x, z, 0); /* |x| < pi/4 */
  if (ix >= 0x7ff00000) return x - x;                    /* sin(Inf or NaN) = NaN */
  n = libm_rem_pio2(x, y);
  switch (n & 3) {
    case 0:
      return libm_kernel_sin(y[0], y[1], 1);
    case 1:
      return libm_kernel_cos(y[0], y[1]);
    case 2:
      return -libm_kernel_sin(y[0], y[1], 1);
    default:
      return -libm_kernel_cos(y[0], y[1]);
  }
}

double cos(double x) {
  double y[2], z = 0.0;
  int n, ix = (int)libm_hi(x) & 0x7fffffff;
  if (ix <= 0x3fe921fb) return libm_kernel_cos(x, z); /* |x| < pi/4 */
  if (ix >= 0x7ff00000) return x - x;                 /* cos(Inf or NaN) = NaN */
  n = libm_rem_pio2(x, y);
  switch (n & 3) {
    case 0:
      return libm_kernel_cos(y[0], y[1]);
    case 1:
      return -libm_kernel_sin(y[0], y[1], 1);
    case 2:
      return -libm_kernel_cos(y[0], y[1]);
    default:
      return libm_kernel_sin(y[0], y[1], 1);
  }
}
