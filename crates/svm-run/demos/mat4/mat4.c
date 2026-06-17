/* Shakedown driver: a 4×4 matrix × vec4 transform using GCC/Clang vector extensions, to exercise
 * 128-bit SIMD (`<4 x float>`). The matrix is 4 column vectors; `matvec` broadcasts each component
 * of `v` and accumulates the columns. Prints the (int-truncated) result components via `printf`.
 * svm-run's output must match a native `cc` build byte-for-byte. The vector ops map to SVM's §17
 * v128 (`f32x4` add/mul, splat, extract/insert lane). */
#include <stddef.h>

int printf(const char *fmt, ...);
typedef float float4 __attribute__((vector_size(16)));

__attribute__((noinline)) static float4 matvec(const float4 *col, float4 v) {
  return col[0] * v[0] + col[1] * v[1] + col[2] * v[2] + col[3] * v[3];
}

int main(void) {
  /* column-major affine transform: scale by 2, then translate by (10,20,30) */
  float4 col[4] = {
      {2, 0, 0, 0}, {0, 2, 0, 0}, {0, 0, 2, 0}, {10, 20, 30, 1}};
  float4 pts[3] = {{3, 4, 5, 1}, {-1, 0, 2, 1}, {100, 100, 100, 1}};
  for (int i = 0; i < 3; i++) {
    float4 r = matvec(col, pts[i]);
    printf("(%d, %d, %d, %d)\n", (int)r[0], (int)r[1], (int)r[2], (int)r[3]);
  }
  return 0;
}
