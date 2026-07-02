/* Guest inverse-trig + modf for the Lua test-suite build, faithful transcriptions of Sun's fdlibm
 * (`s_atan`/`e_atan2`/`e_asin`/`e_acos`/`s_modf`) — sub-ULP accurate, using only IEEE `+ - * /` and
 * double-word access. The bundled guest libm (`demos/libm/libm.c`) provides sin/cos/exp/log/pow/sqrt;
 * these fill the functions Lua's `math` library needs that it lacks (`math.atan`/`asin`/`acos`/
 * `modf`), shadowing the on-ramp's fail-closed stubs. `math.tan`/`log10`/`log2` stay in the small shim
 * (derived from sin/cos/log). Compile with `-fno-strict-aliasing -fno-builtin` (the word-access macros
 * type-pun through int pointers, exactly as fdlibm requires natively). */

/* double-word access (little-endian; both native x86-64 and the on-ramp are LE) */
#define __HI(x)  *(1 + (int *)&(x))
#define __LO(x)  *(int *)&(x)
#define __HIp(x) *(1 + (int *)(x))
#define __LOp(x) *(int *)(x)

extern double sqrt(double);

static double fabs(double x) {
  __HI(x) &= 0x7fffffff;
  return x;
}

static const double
    atanhi[] = {4.63647609000806093515e-01, 7.85398163397448278999e-01,
                9.82793723247329054082e-01, 1.57079632679489655800e+00},
    atanlo[] = {2.26987774529616870924e-17, 3.06161699786838301793e-17,
                1.39033110312309984516e-17, 6.12323399573676603587e-17},
    aT[] = {3.33333333333329318027e-01,  -1.99999999998764832476e-01,
            1.42857142725034663711e-01,  -1.11111104054623557880e-01,
            9.09088713343650656196e-02,  -7.69187620504482999495e-02,
            6.66107313738753120669e-02,  -5.83357013379057348645e-02,
            4.97687799461593236017e-02,  -3.65315727442169155270e-02,
            1.62858201153657823623e-02};
static const double one = 1.0, huge = 1.0e300;

double atan(double x) {
  double w, s1, s2, z;
  int ix, hx, id;
  hx = __HI(x);
  ix = hx & 0x7fffffff;
  if (ix >= 0x44100000) { /* if |x| >= 2^66 */
    if (ix > 0x7ff00000 || (ix == 0x7ff00000 && (__LO(x) != 0)))
      return x + x; /* NaN */
    if (hx > 0)
      return atanhi[3] + atanlo[3];
    else
      return -atanhi[3] - atanlo[3];
  }
  if (ix < 0x3fdc0000) { /* |x| < 0.4375 */
    if (ix < 0x3e200000) { /* |x| < 2^-29 */
      if (huge + x > one) return x; /* raise inexact */
    }
    id = -1;
  } else {
    x = fabs(x);
    if (ix < 0x3ff30000) { /* |x| < 1.1875 */
      if (ix < 0x3fe60000) { /* 7/16 <= |x| < 11/16 */
        id = 0;
        x = (2.0 * x - one) / (2.0 + x);
      } else { /* 11/16 <= |x| < 19/16 */
        id = 1;
        x = (x - one) / (x + one);
      }
    } else {
      if (ix < 0x40038000) { /* |x| < 2.4375 */
        id = 2;
        x = (x - 1.5) / (one + 1.5 * x);
      } else { /* 2.4375 <= |x| < 2^66 */
        id = 3;
        x = -1.0 / x;
      }
    }
  }
  z = x * x;
  w = z * z;
  s1 = z * (aT[0] + w * (aT[2] + w * (aT[4] + w * (aT[6] + w * (aT[8] + w * aT[10])))));
  s2 = w * (aT[1] + w * (aT[3] + w * (aT[5] + w * (aT[7] + w * aT[9]))));
  if (id < 0) return x - x * (s1 + s2);
  z = atanhi[id] - ((x * (s1 + s2) - atanlo[id]) - x);
  return (hx < 0) ? -z : z;
}

static const double tiny = 1.0e-300, zero = 0.0,
                    pi_o_4 = 7.8539816339744827900e-01,
                    pi_o_2 = 1.5707963267948965580e+00,
                    pi = 3.1415926535897931160e+00,
                    pi_lo = 1.2246467991473531772e-16;

double atan2(double y, double x) {
  double z;
  int k, m, hx, hy, ix, iy;
  unsigned lx, ly;
  hx = __HI(x);
  ix = hx & 0x7fffffff;
  lx = __LO(x);
  hy = __HI(y);
  iy = hy & 0x7fffffff;
  ly = __LO(y);
  if (((ix | ((lx | -lx) >> 31)) > 0x7ff00000) ||
      ((iy | ((ly | -ly) >> 31)) > 0x7ff00000)) /* x or y is NaN */
    return x + y;
  if (((hx - 0x3ff00000) | lx) == 0) return atan(y); /* x = 1.0 */
  m = ((hy >> 31) & 1) | ((hx >> 30) & 2);           /* 2*sign(x)+sign(y) */
  if ((iy | ly) == 0) {                              /* when y = 0 */
    switch (m) {
      case 0:
      case 1: return y;             /* atan(+-0, +anything) = +-0 */
      case 2: return pi + tiny;     /* atan(+0, -anything) = pi */
      case 3: return -pi - tiny;    /* atan(-0, -anything) = -pi */
    }
  }
  if ((ix | lx) == 0) return (hy < 0) ? -pi_o_2 - tiny : pi_o_2 + tiny; /* x = 0 */
  if (ix == 0x7ff00000) {                                               /* x is INF */
    if (iy == 0x7ff00000) {
      switch (m) {
        case 0: return pi_o_4 + tiny;
        case 1: return -pi_o_4 - tiny;
        case 2: return 3.0 * pi_o_4 + tiny;
        case 3: return -3.0 * pi_o_4 - tiny;
      }
    } else {
      switch (m) {
        case 0: return zero;
        case 1: return -zero;
        case 2: return pi + tiny;
        case 3: return -pi - tiny;
      }
    }
  }
  if (iy == 0x7ff00000) return (hy < 0) ? -pi_o_2 - tiny : pi_o_2 + tiny; /* y is INF */
  k = (iy - ix) >> 20;                                                    /* compute y/x */
  if (k > 60)
    z = pi_o_2 + 0.5 * pi_lo; /* |y/x| > 2^60 */
  else if (hx < 0 && k < -60)
    z = 0.0; /* |y|/x < -2^60 */
  else
    z = atan(fabs(y / x));
  switch (m) {
    case 0: return z;                    /* atan(+,+) */
    case 1: __HI(z) ^= 0x80000000; return z; /* atan(-,+) */
    case 2: return pi - (z - pi_lo);     /* atan(+,-) */
    default: return (z - pi_lo) - pi;    /* atan(-,-) */
  }
}

static const double
    pio2_hi = 1.57079632679489655800e+00, pio2_lo = 6.12323399573676603587e-17,
    pio4_hi = 7.85398163397448278999e-01, pS0 = 1.66666666666666657415e-01,
    pS1 = -3.25565818622400915405e-01, pS2 = 2.01212532134862925881e-01,
    pS3 = -4.00555345006794114027e-02, pS4 = 7.91534994289814532176e-04,
    pS5 = 3.47933107596021167570e-05, qS1 = -2.40339491173441421878e+00,
    qS2 = 2.02094576023350569471e+00, qS3 = -6.88283971605453293030e-01,
    qS4 = 7.70381505559019352791e-02;

double asin(double x) {
  double t, w, p, q, c, r, s;
  int hx, ix;
  t = 0.0;
  hx = __HI(x);
  ix = hx & 0x7fffffff;
  if (ix >= 0x3ff00000) { /* |x| >= 1 */
    if (((ix - 0x3ff00000) | __LO(x)) == 0)
      return x * pio2_hi + x * pio2_lo; /* asin(1) = +-pi/2 */
    return (x - x) / (x - x);           /* asin(|x|>1) is NaN */
  } else if (ix < 0x3fe00000) {         /* |x| < 0.5 */
    if (ix < 0x3e400000) {              /* |x| < 2^-27 */
      if (huge + x > one) return x;
    } else {
      t = x * x;
      p = t * (pS0 + t * (pS1 + t * (pS2 + t * (pS3 + t * (pS4 + t * pS5)))));
      q = one + t * (qS1 + t * (qS2 + t * (qS3 + t * qS4)));
      w = p / q;
      return x + x * w;
    }
  }
  w = one - fabs(x);
  t = w * 0.5;
  p = t * (pS0 + t * (pS1 + t * (pS2 + t * (pS3 + t * (pS4 + t * pS5)))));
  q = one + t * (qS1 + t * (qS2 + t * (qS3 + t * qS4)));
  s = sqrt(t);
  if (ix >= 0x3FEF3333) { /* |x| > 0.975 */
    w = p / q;
    t = pio2_hi - (2.0 * (s + s * w) - pio2_lo);
  } else {
    w = s;
    __LO(w) = 0;
    c = (t - w * w) / (s + w);
    r = p / q;
    p = 2.0 * s * r - (pio2_lo - 2.0 * c);
    q = pio4_hi - 2.0 * w;
    t = pio4_hi - (p - q);
  }
  return (hx > 0) ? t : -t;
}

static const double acos_pi = 3.14159265358979311600e+00;

double acos(double x) {
  double z, p, q, r, w, s, c, df;
  int hx, ix;
  hx = __HI(x);
  ix = hx & 0x7fffffff;
  if (ix >= 0x3ff00000) { /* |x| >= 1 */
    if (((ix - 0x3ff00000) | __LO(x)) == 0) {
      if (hx > 0)
        return 0.0; /* acos(1) = 0 */
      else
        return acos_pi + 2.0 * pio2_lo; /* acos(-1) = pi */
    }
    return (x - x) / (x - x); /* acos(|x|>1) is NaN */
  }
  if (ix < 0x3fe00000) { /* |x| < 0.5 */
    if (ix <= 0x3c600000) return pio2_hi + pio2_lo; /* |x| < 2^-57 */
    z = x * x;
    p = z * (pS0 + z * (pS1 + z * (pS2 + z * (pS3 + z * (pS4 + z * pS5)))));
    q = one + z * (qS1 + z * (qS2 + z * (qS3 + z * qS4)));
    r = p / q;
    return pio2_hi - (x - (pio2_lo - x * r));
  } else if (hx < 0) { /* x < -0.5 */
    z = (one + x) * 0.5;
    p = z * (pS0 + z * (pS1 + z * (pS2 + z * (pS3 + z * (pS4 + z * pS5)))));
    q = one + z * (qS1 + z * (qS2 + z * (qS3 + z * qS4)));
    s = sqrt(z);
    r = p / q;
    w = r * s - pio2_lo;
    return acos_pi - 2.0 * (s + w);
  } else { /* x > 0.5 */
    z = (one - x) * 0.5;
    s = sqrt(z);
    df = s;
    __LO(df) = 0;
    c = (z - df * df) / (s + df);
    p = z * (pS0 + z * (pS1 + z * (pS2 + z * (pS3 + z * (pS4 + z * pS5)))));
    q = one + z * (qS1 + z * (qS2 + z * (qS3 + z * qS4)));
    r = p / q;
    w = r * s + c;
    return 2.0 * (df + w);
  }
}

double modf(double x, double *iptr) {
  int i0, i1, j0;
  unsigned i;
  i0 = __HI(x);
  i1 = __LO(x);
  j0 = ((i0 >> 20) & 0x7ff) - 0x3ff; /* exponent of x */
  if (j0 < 20) {                     /* integer part in high x */
    if (j0 < 0) {                    /* |x| < 1 */
      __HIp(iptr) = i0 & 0x80000000;
      __LOp(iptr) = 0; /* *iptr = +-0 */
      return x;
    } else {
      i = (0x000fffff) >> j0;
      if (((i0 & i) | i1) == 0) { /* x is integral */
        *iptr = x;
        __HI(x) &= 0x80000000;
        __LO(x) = 0;
        return x;
      } else {
        __HIp(iptr) = i0 & (~i);
        __LOp(iptr) = 0;
        return x - *iptr;
      }
    }
  } else if (j0 > 51) { /* no fraction part */
    *iptr = x * one;
    __HI(x) &= 0x80000000;
    __LO(x) = 0;
    return x;
  } else { /* fraction part in low x */
    i = ((unsigned)(0xffffffff)) >> (j0 - 20);
    if ((i1 & i) == 0) { /* x is integral */
      *iptr = x;
      __HI(x) &= 0x80000000;
      __LO(x) = 0;
      return x;
    } else {
      __HIp(iptr) = i0;
      __LOp(iptr) = i1 & (~i);
      return x - *iptr;
    }
  }
}
