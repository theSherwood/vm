/* A correctly-rounded **guest `strtod`** (decimal string → `double`) for the SVM LLVM on-ramp.
 *
 * Like the guest `libm`, a program that needs `strtod` brings it as ordinary guest C; a guest
 * definition **shadows** the on-ramp's would-be trap stub, so `llvm-link lua_core.bc strtod.bc`
 * makes Lua's numeric-literal parsing real. The keystone for *float* Lua: every float literal
 * (`3.14`, `1e10`, …) is parsed here.
 *
 * Correctly-rounded decimal→`f64` is **unique** (round-to-nearest-even), so this matches any other
 * correctly-rounded `strtod` — incl. glibc's — bit-for-bit. The method is the rigorous one, with no
 * precomputed power-of-ten table (nothing to mis-transcribe): parse every significant digit into a
 * big integer `num` with a decimal exponent `e` (value = num·10^e), form the exact rational
 * `N / Dn`, and take the nearest double by an **exact big-integer division** with round-to-nearest-
 * even. Handles normal, subnormal (incl. the boundary), and overflow→±inf.
 *
 * Scope: decimal floats with optional sign and `e`/`E` exponent, plus **hex floats**
 * (`0x1.8p3`, `0x.ABCDEFp+24`, …): the hex mantissa is exact in base 2, so it accumulates into the
 * same big integer with a binary exponent and rounds via `strtod_assemble2` (round-to-nearest-even) —
 * matching glibc bit-for-bit, incl. Lua's own hex-float literals. `endptr` is set to the first
 * unconsumed character (NULL `endptr` is allowed); an empty hex mantissa (`0x`, `0x.`) recognizes only
 * the leading "0" (glibc semantics), so a caller sees the leftover 'x'. Not handled (callers that need
 * them define their own): the `inf`/`nan` spellings and `errno`/`ERANGE`. Significant digits past 800
 * (decimal) / 200 (hex) are ignored — far more than the ~17/13 that determine any double.
 */

/* ── a minimal base-2^32 big integer ─────────────────────────────────────────────────────────── */
#define BN_CAP 256
typedef struct {
  unsigned v[BN_CAP]; /* little-endian limbs */
  int n;              /* number of significant limbs (0 ⇒ value 0) */
} bn;

static void bn_zero(bn *a) { a->n = 0; }
static int bn_is_zero(const bn *a) { return a->n == 0; }

static void bn_from_u64(bn *a, unsigned long long x) {
  a->n = 0;
  while (x) {
    a->v[a->n++] = (unsigned)x;
    x >>= 32;
  }
}

/* a = a*m + add  (m, add fit in 32 bits) — the digit-accumulation and ×10 primitive. */
static void bn_mul_add_small(bn *a, unsigned m, unsigned add) {
  unsigned long long carry = add;
  for (int i = 0; i < a->n; i++) {
    unsigned long long p = (unsigned long long)a->v[i] * m + carry;
    a->v[i] = (unsigned)p;
    carry = p >> 32;
  }
  while (carry) {
    a->v[a->n++] = (unsigned)carry;
    carry >>= 32;
  }
}

static int bn_bitlen(const bn *a) {
  if (a->n == 0) return 0;
  unsigned hi = a->v[a->n - 1];
  int b = (a->n - 1) * 32;
  while (hi) {
    b++;
    hi >>= 1;
  }
  return b;
}

static int bn_getbit(const bn *a, int i) {
  int limb = i >> 5, bit = i & 31;
  if (limb >= a->n) return 0;
  return (a->v[limb] >> bit) & 1;
}

/* a <<= bits */
static void bn_shl(bn *a, int bits) {
  if (a->n == 0 || bits == 0) return;
  int limbs = bits >> 5, sh = bits & 31;
  int newn = a->n + limbs + 1;
  for (int i = newn - 1; i >= 0; i--) {
    unsigned long long acc = 0;
    int src = i - limbs;
    if (src >= 0 && src < a->n) acc |= (unsigned long long)a->v[src] << sh;
    if (sh && src - 1 >= 0 && src - 1 < a->n) acc |= (unsigned long long)a->v[src - 1] >> (32 - sh);
    a->v[i] = (unsigned)acc;
  }
  a->n = newn;
  while (a->n > 0 && a->v[a->n - 1] == 0) a->n--;
}

static int bn_cmp(const bn *a, const bn *b) {
  if (a->n != b->n) return a->n < b->n ? -1 : 1;
  for (int i = a->n - 1; i >= 0; i--)
    if (a->v[i] != b->v[i]) return a->v[i] < b->v[i] ? -1 : 1;
  return 0;
}

/* a -= b  (requires a >= b) */
static void bn_sub(bn *a, const bn *b) {
  long long borrow = 0;
  for (int i = 0; i < a->n; i++) {
    long long d = (long long)a->v[i] - (i < b->n ? b->v[i] : 0u) - borrow;
    if (d < 0) {
      d += 0x100000000LL;
      borrow = 1;
    } else {
      borrow = 0;
    }
    a->v[i] = (unsigned)d;
  }
  while (a->n > 0 && a->v[a->n - 1] == 0) a->n--;
}

/* a *= b  (school multiply into a scratch, then copy back) */
static void bn_mul(bn *a, const bn *b) {
  if (a->n == 0 || b->n == 0) {
    a->n = 0;
    return;
  }
  unsigned out[BN_CAP];
  int on = a->n + b->n;
  for (int i = 0; i < on; i++) out[i] = 0;
  for (int i = 0; i < a->n; i++) {
    unsigned long long carry = 0;
    unsigned long long ai = a->v[i];
    for (int j = 0; j < b->n; j++) {
      unsigned long long cur = (unsigned long long)out[i + j] + ai * b->v[j] + carry;
      out[i + j] = (unsigned)cur;
      carry = cur >> 32;
    }
    int k = i + b->n;
    while (carry) {
      unsigned long long cur = (unsigned long long)out[k] + carry;
      out[k] = (unsigned)cur;
      carry = cur >> 32;
      k++;
    }
  }
  while (on > 0 && out[on - 1] == 0) on--;
  for (int i = 0; i < on; i++) a->v[i] = out[i];
  a->n = on;
}

/* a *= 10^e  (e >= 0), by repeated ×(10^9) chunks then the remainder. */
static void bn_mul_pow10(bn *a, int e) {
  static const unsigned p10[10] = {1u,      10u,      100u,      1000u,      10000u,
                                   100000u, 1000000u, 10000000u, 100000000u, 1000000000u};
  while (e >= 9) {
    bn_mul_add_small(a, 1000000000u, 0);
    e -= 9;
  }
  if (e > 0) bn_mul_add_small(a, p10[e], 0);
}

/* ── the f64 image ───────────────────────────────────────────────────────────────────────────── */
typedef union {
  double d;
  unsigned long long u;
} strtod_du;

static double strtod_bits(unsigned long long u) {
  strtod_du t;
  t.u = u;
  return t.d;
}

/* Assemble the nearest double to (num · 10^e10), num != 0, from a big-integer division. */
static double strtod_assemble(const bn *num, int e10, int neg) {
  /* Exact value V = N / Dn, with the powers of ten split into 2 and 5 so the 2-part folds into the
     binary scale F below. N = num·5^max(e10,0), Dn = 5^max(-e10,0); the 2^e10 factor is applied as
     part of F. */
  bn N = *num, Dn;
  bn_from_u64(&Dn, 1);
  int p2 = e10; /* the 2^e10 factor handled via the binary scale */
  if (e10 > 0) {
    bn five;
    bn_from_u64(&five, 5);
    for (int i = 0; i < e10; i++) bn_mul(&N, &five);
  } else if (e10 < 0) {
    bn five;
    bn_from_u64(&five, 5);
    for (int i = 0; i < -e10; i++) bn_mul(&Dn, &five);
  }
  (void)p2;

  /* Choose F so that q = round(V / 2^F) lands in [2^52, 2^53): F ≈ ilog2(V) - 52, where
     ilog2(V) ≈ bitlen(N) - bitlen(Dn) + e10 (the 2^e10 factor). */
  int F = bn_bitlen(&N) - bn_bitlen(&Dn) + e10 - 53;

  /* Build A and B so that q = floor(A/B): A = N·2^(e10 - F) when (e10 - F) >= 0, else B = Dn·2^(F - e10).
     A = N·2^a2 / (Dn·2^F)·... — fold both 2-powers. Net scale shift = e10 - F. */
  for (int pass = 0; pass < 2; pass++) {
    bn A = N, B = Dn;
    int shift = e10 - F;
    if (shift >= 0)
      bn_shl(&A, shift);
    else
      bn_shl(&B, -shift);

    /* binary long division: q = floor(A/B), r = A mod B */
    unsigned long long q = 0;
    bn r;
    bn_zero(&r);
    int top = bn_bitlen(&A);
    for (int i = top - 1; i >= 0; i--) {
      bn_shl(&r, 1);
      if (bn_getbit(&A, i)) bn_mul_add_small(&r, 1, 1); /* r |= 1 (r is even after <<1) */
      q <<= 1;
      if (bn_cmp(&r, &B) >= 0) {
        bn_sub(&r, &B);
        q |= 1;
      }
    }

    /* q should be in [2^52, 2^53). If the estimate was off by one bit, bump F and redo once. */
    if (q < (1ULL << 52)) {
      F -= 1;
      continue;
    }
    if (q >= (1ULL << 53)) {
      F += 1;
      continue;
    }

    /* round-to-nearest-even on the remainder: compare 2r vs B */
    bn r2 = r;
    bn_shl(&r2, 1);
    int c = bn_cmp(&r2, &B);
    if (c > 0 || (c == 0 && (q & 1)))
      q += 1;
    if (q >= (1ULL << 53)) { /* rounding carried out of the 53-bit window */
      q >>= 1;
      F += 1;
    }

    /* now value ≈ q·2^F with q in [2^52, 2^53): the unbiased binary exponent is F+52. */
    int be = F + 52;
    if (be > 1023) /* overflow */
      return neg ? strtod_bits(0xFFF0000000000000ULL) : strtod_bits(0x7FF0000000000000ULL);
    if (be >= -1022) { /* normal */
      unsigned long long mant = q & ((1ULL << 52) - 1);
      unsigned long long bits = ((unsigned long long)(be + 1023) << 52) | mant;
      if (neg) bits |= 0x8000000000000000ULL;
      return strtod_bits(bits);
    }
    /* subnormal: recompute the mantissa at the fixed scale 2^-1074 (F = -1074, q < 2^52). */
    break;
  }

  /* Subnormal path: q' = round(V / 2^-1074). Re-divide with F = -1074. */
  {
    int F2 = -1074;
    bn A = N, B = Dn;
    int shift = e10 - F2;
    if (shift >= 0)
      bn_shl(&A, shift);
    else
      bn_shl(&B, -shift);
    unsigned long long q = 0;
    bn r;
    bn_zero(&r);
    int top = bn_bitlen(&A);
    for (int i = top - 1; i >= 0; i--) {
      bn_shl(&r, 1);
      if (bn_getbit(&A, i)) bn_mul_add_small(&r, 1, 1);
      q <<= 1;
      if (bn_cmp(&r, &B) >= 0) {
        bn_sub(&r, &B);
        q |= 1;
      }
    }
    bn r2 = r;
    bn_shl(&r2, 1);
    int c = bn_cmp(&r2, &B);
    if (c > 0 || (c == 0 && (q & 1))) q += 1;
    /* q == 2^52 (rounded up to the smallest normal) assembles correctly as exp field 1, mant 0. */
    unsigned long long bits = q & ((1ULL << 53) - 1);
    if (neg) bits |= 0x8000000000000000ULL;
    return strtod_bits(bits);
  }
}

/* Assemble the nearest double to (num · 2^e2), num != 0 — the hex-float path. A pure power-of-two
   scale, so (unlike the decimal case) no factor of five: normalize `num` to a 53-bit q with
   round-to-nearest-even, then place it at binary exponent `be`. */
static double strtod_assemble2(const bn *num, int e2, int neg) {
  bn N = *num;
  int bl = bn_bitlen(&N);  /* >= 1 since num != 0 */
  int be = bl + e2 - 1;    /* unbiased exponent of the MSB (bit bl-1) */
  unsigned long long q = 0;
  int s = bl - 53; /* bits to drop from the bottom of N (may be <= 0) */
  if (s <= 0) {
    for (int i = bl - 1; i >= 0; i--) q = (q << 1) | (unsigned long long)(bn_getbit(&N, i) ? 1 : 0);
    q <<= (unsigned)(-s); /* left-align the MSB to bit 52 (exact) */
  } else {
    for (int i = bl - 1; i >= s; i--) q = (q << 1) | (unsigned long long)(bn_getbit(&N, i) ? 1 : 0);
    int guard = bn_getbit(&N, s - 1);
    int sticky = 0;
    for (int i = s - 2; i >= 0; i--) {
      if (bn_getbit(&N, i)) { sticky = 1; break; }
    }
    if (guard && (sticky || (q & 1))) {
      q += 1;
      if (q >> 53) { q >>= 1; be += 1; } /* rounding carried out of the 53-bit window */
    }
  }
  /* value = q · 2^(be - 52), q in [2^52, 2^53) */
  if (be > 1023)
    return neg ? strtod_bits(0xFFF0000000000000ULL) : strtod_bits(0x7FF0000000000000ULL);
  if (be >= -1022) { /* normal */
    unsigned long long mant = q & ((1ULL << 52) - 1);
    unsigned long long bits = ((unsigned long long)(be + 1023) << 52) | mant;
    if (neg) bits |= 0x8000000000000000ULL;
    return strtod_bits(bits);
  }
  /* subnormal: q' = round(q / 2^d) at the fixed scale 2^-1074, d = -1022 - be > 0 */
  int d = -1022 - be;
  unsigned long long qd, guard, sticky;
  if (d >= 64) {
    qd = 0;
    guard = 0;
    sticky = (q != 0);
  } else {
    qd = q >> d;
    guard = (q >> (d - 1)) & 1ULL;
    sticky = (q & ((1ULL << (d - 1)) - 1)) ? 1 : 0;
  }
  if (guard && (sticky || (qd & 1))) qd += 1;
  unsigned long long bits = qd & ((1ULL << 53) - 1); /* qd==2^52 assembles as the smallest normal */
  if (neg) bits |= 0x8000000000000000ULL;
  return strtod_bits(bits);
}

static int strtod_hexval(int c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

static int strtod_isspace(int c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\v' || c == '\f' || c == '\r';
}

double strtod(const char *nptr, char **endptr) {
  const char *p = nptr;
  while (strtod_isspace((unsigned char)*p)) p++;

  int neg = 0;
  if (*p == '+' || *p == '-') {
    neg = (*p == '-');
    p++;
  }

  /* hex float: 0x<hexdigits>[.<hexdigits>][p±<dec>]. The mantissa is exact in base 2, so accumulate
     the hex digits into a big integer and track a binary exponent (each fraction digit is 2^-4, the
     'p' exponent adds directly); `strtod_assemble2` rounds to nearest. An empty mantissa (`0x`, `0x.`)
     is not a hex number — consume just the leading "0" so the caller sees the leftover 'x' and
     reports a malformed number, matching glibc. */
  if (p[0] == '0' && (p[1] == 'x' || p[1] == 'X')) {
    const char *hp = p + 2;
    bn hm;
    bn_zero(&hm);
    int e2 = 0, anyhex = 0, hseen_dot = 0, hsig = 0;
    for (;; hp++) {
      if (*hp == '.') {
        if (hseen_dot) break;
        hseen_dot = 1;
        continue;
      }
      int d = strtod_hexval((unsigned char)*hp);
      if (d < 0) break;
      anyhex = 1;
      if (hsig < 200) {
        bn_mul_add_small(&hm, 16, (unsigned)d);
        hsig++;
        if (hseen_dot) e2 -= 4;
      } else if (!hseen_dot) {
        e2 += 4; /* integer digits past precision keep the magnitude */
      }
    }
    if (!anyhex) { /* "0x" with no mantissa: recognize just the "0" */
      if (endptr) *endptr = (char *)(p + 1);
      return neg ? strtod_bits(0x8000000000000000ULL) : 0.0;
    }
    if (*hp == 'p' || *hp == 'P') {
      const char *e = hp + 1;
      int esign = 0, eval = 0, edig = 0;
      if (*e == '+' || *e == '-') {
        esign = (*e == '-');
        e++;
      }
      while (*e >= '0' && *e <= '9') {
        if (eval < 100000) eval = eval * 10 + (*e - '0');
        e++;
        edig = 1;
      }
      if (edig) {
        e2 += esign ? -eval : eval;
        hp = e;
      }
    }
    if (endptr) *endptr = (char *)hp;
    if (bn_is_zero(&hm)) return neg ? strtod_bits(0x8000000000000000ULL) : 0.0;
    return strtod_assemble2(&hm, e2, neg);
  }

  bn num;
  bn_zero(&num);
  int any = 0;     /* saw at least one digit */
  int sig = 0;     /* significant digits accumulated into num */
  int e10 = 0;     /* decimal exponent of num */
  int seen_dot = 0;
  int lead_zero = 0; /* skip leading zeros (keep num small) */

  for (;; p++) {
    if (*p == '.') {
      if (seen_dot) break;
      seen_dot = 1;
      continue;
    }
    if (*p < '0' || *p > '9') break;
    any = 1;
    int dig = *p - '0';
    if (dig == 0 && sig == 0) {
      /* leading zeros contribute nothing; if after the dot, they push the exponent down */
      lead_zero = 1;
      if (seen_dot) e10--;
      continue;
    }
    if (sig < 800) {
      bn_mul_add_small(&num, 10, (unsigned)dig);
      sig++;
      if (seen_dot) e10--;
    } else {
      /* past the precision that can affect a double: keep the magnitude, drop the digit */
      if (!seen_dot) e10++;
    }
  }
  (void)lead_zero;

  if (!any) { /* no conversion */
    if (endptr) *endptr = (char *)nptr;
    return 0.0;
  }

  /* optional exponent */
  if (*p == 'e' || *p == 'E') {
    const char *e = p + 1;
    int esign = 0, eval = 0, edig = 0;
    if (*e == '+' || *e == '-') {
      esign = (*e == '-');
      e++;
    }
    while (*e >= '0' && *e <= '9') {
      if (eval < 100000) eval = eval * 10 + (*e - '0');
      e++;
      edig = 1;
    }
    if (edig) {
      e10 += esign ? -eval : eval;
      p = e;
    }
  }

  if (endptr) *endptr = (char *)p;

  if (bn_is_zero(&num)) return neg ? strtod_bits(0x8000000000000000ULL) : 0.0;

  return strtod_assemble(&num, e10, neg);
}
