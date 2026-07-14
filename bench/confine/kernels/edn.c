// Faithful embench `edn` shape — the DSP suite (vec_mpy1/mac/fir/fir_no_red_ld/latsynth/iir1/
// codebook/jpegdct) on `short` (16-bit) arrays with `long` accumulators, vendored verbatim from
// embench-iot src/edn/libedn.c. Unlike the idealized `fir` confine kernel (single dot, int data),
// this exercises edn's real construct mix: strided 16-bit loads, `long`-accumulated dot products,
// and the in-place filters — the workload embench reports svm-jit ~1.4x behind wt64 on (which the
// best-of-25 harness is here to confirm or refute, as it did for matmult).
#define N 100
#define ORDER 50

static short a[200];
static short b[200];
static short c;
static long d;
static int e;
static long output[200];

static void vec_mpy1(short y[], const short x[], short scaler) {
  long i;
  for (i = 0; i < 150; i++) y[i] += ((scaler * x[i]) >> 15);
}

static long mac(const short *a, const short *b, long sqr, long *sum) {
  long i;
  long dotp = *sum;
  for (i = 0; i < 150; i++) { dotp += b[i] * a[i]; sqr += b[i] * b[i]; }
  *sum = dotp;
  return sqr;
}

static void fir(const short array1[], const short coeff[], long output[]) {
  long i, j, sum;
  for (i = 0; i < N - ORDER; i++) {
    sum = 0;
    for (j = 0; j < ORDER; j++) sum += array1[i + j] * coeff[j];
    output[i] = sum >> 15;
  }
}

static void fir_no_red_ld(const short x[], const short h[], long y[]) {
  long i, j, sum0, sum1;
  short x0, x1, h0, h1;
  for (j = 0; j < 100; j += 2) {
    sum0 = 0; sum1 = 0;
    x0 = x[j];
    for (i = 0; i < 32; i += 2) {
      x1 = x[j + i + 1]; h0 = h[i];
      sum0 += x0 * h0; sum1 += x1 * h0;
      x0 = x[j + i + 2]; h1 = h[i + 1];
      sum0 += x1 * h1; sum1 += x0 * h1;
    }
    y[j] = sum0 >> 15; y[j + 1] = sum1 >> 15;
  }
}

static long latsynth(short b[], const short k[], long n, long f) {
  long i;
  f -= b[n - 1] * k[n - 1];
  for (i = n - 2; i >= 0; i--) {
    f -= b[i] * k[i];
    b[i + 1] = b[i] + ((k[i] * (f >> 16)) >> 16);
  }
  b[0] = f >> 16;
  return f;
}

static void iir1(const short *coefs, const short *input, long *optr, long *state) {
  long x, t, n;
  x = input[0];
  for (n = 0; n < 50; n++) {
    t = x + ((coefs[2] * state[0] + coefs[3] * state[1]) >> 15);
    x = t + ((coefs[0] * state[0] + coefs[1] * state[1]) >> 15);
    state[1] = state[0]; state[0] = t;
    coefs += 4; state += 2;
  }
  *optr++ = x;
}

static long codebook(long mask, long bitchanged, long numbasis, long codeword,
                     long g, const short *d, short ddim, short theta) {
  long j;
  for (j = bitchanged + 1; j <= numbasis; j++) {}
  return g;
}

static void jpegdct(short *d, short *r) {
  long t[12];
  short i, j, k, m, n, p;
  for (k = 1, m = 0, n = 13, p = 8; k <= 8; k += 7, m += 3, n += 3, p -= 7, d -= 64) {
    for (i = 0; i < 8; i++, d += p) {
      for (j = 0; j < 4; j++) {
        t[j] = d[k * j] + d[k * (7 - j)];
        t[7 - j] = d[k * j] - d[k * (7 - j)];
      }
      t[8] = t[0] + t[3]; t[9] = t[0] - t[3];
      t[10] = t[1] + t[2]; t[11] = t[1] - t[2];
      d[0] = (t[8] + t[10]) >> m;
      d[4 * k] = (t[8] - t[10]) >> m;
      t[8] = (short)(t[11] + t[9]) * r[10];
      d[2 * k] = t[8] + (short)((t[9] * r[9]) >> n);
      d[6 * k] = t[8] + (short)((t[11] * r[11]) >> n);
      t[0] = (short)(t[4] + t[7]) * r[2];
      t[1] = (short)(t[5] + t[6]) * r[0];
      t[2] = t[4] + t[6]; t[3] = t[5] + t[7];
      t[8] = (short)(t[2] + t[3]) * r[8];
      t[2] = (short)t[2] * r[1] + t[8];
      t[3] = (short)t[3] * r[3] + t[8];
      d[7 * k] = (short)(t[4] * r[4] + t[0] + t[2]) >> n;
      d[5 * k] = (short)(t[5] * r[6] + t[1] + t[3]) >> n;
      d[3 * k] = (short)(t[6] * r[5] + t[1] + t[2]) >> n;
      d[1 * k] = (short)(t[7] * r[7] + t[0] + t[3]) >> n;
    }
  }
}

long run(long iters) {
  long h = 0;
  for (long t = 0; t < iters; t++) {
    // Re-init the working buffers each iteration (the filters mutate a/b), matching edn's body.
    for (int i = 0; i < 200; i++) {
      int p = i & 7;
      a[i] = (short)(0x0000 + p * 0x0111 + (short)t);
      b[i] = (short)(0x0c60 - p * 0x0140 + (short)t);
    }
    c = 0x3; d = 0xAAAA; e = 0xEEEE;
    vec_mpy1(a, b, c);
    c = mac(a, b, (long)c, output);
    fir(a, b, output);
    fir_no_red_ld(a, b, output);
    d = latsynth(a, b, N, d);
    iir1(a, b, &output[100], output);
    e = codebook(d, 1, 17, e, d, a, c, 1);
    jpegdct(a, b);
    h += c + d + e + output[0] + output[50] + output[100];
  }
  return h;
}
