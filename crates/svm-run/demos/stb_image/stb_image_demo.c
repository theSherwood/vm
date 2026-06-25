/* Corpus shakedown: run Sean Barrett's stb_image (public domain) PNG decoder in the sandbox.
 *
 * This is a real-parser shakedown — stb_image decodes an *embedded* PNG (compiled-in bytes,
 * no file I/O) and the demo writes the raw decoded RGBA pixels to stdout. The native cc build
 * decodes the same bytes, so the two pixel streams are the byte-exact oracle for each other.
 *
 * It exercises the decoder end-to-end: stb's built-in zlib inflate (Huffman + LZ77 back-refs),
 * the PNG row unfilters (None/Sub/Up/Average/Paeth — the test image cycles all five, so the
 * narrow `unsigned char` predictor arithmetic gets hit), the IDAT/IHDR/CRC chunk walk, and
 * heap traffic through the on-ramp's synthesized malloc/realloc/free (slices from heapgrow/
 * sortvec). Configured PNG-only (no JPEG/GIF/…) and with no HDR/linear path, so it needs no
 * float libm and no setjmp — pure integer parsing.
 *
 * Built with auto-vectorization off for the on-ramp (the inflate/unfilter loops clang would
 * SIMD-vectorize are the §17 vector lane); the native oracle keeps vectorizing, and exact
 * integer decoding agrees scalar-vs-vectorized. */

#include <stddef.h>

/* stb_image force-includes <string.h>/<stdlib.h>, so we do NOT define memcpy/memset/memmove or
 * malloc/realloc/free here (that would clash with the libc decls in the native oracle build).
 * Instead: clang -O2 lowers stb's mem* calls to `llvm.mem*` intrinsics the on-ramp handles, and
 * malloc/realloc/free are recognized libc names bound to the on-ramp's powerbox heap. */

/* PNG-only, no file I/O, no HDR/linear (keeps libm + setjmp out); asserts compiled out. */
#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_STDIO
#define STBI_ONLY_PNG
#define STBI_NO_HDR
#define STBI_NO_LINEAR
/* stb auto-makes its failure-reason string `_Thread_local` when the compiler supports it, which
 * pulls in `llvm.threadlocal.address` (TLS is out of the scalar on-ramp's scope). The strings are
 * only for error reporting, which this demo doesn't read — disable thread-locals outright. */
#define STBI_NO_THREAD_LOCALS
#define STBI_ASSERT(x) ((void)0)
#include "stb_image.h"

#include "test_image.inc"

int write(int fd, char *buf, long n);

int main(void) {
  int w = 0, h = 0, comp = 0;
  unsigned char *pixels =
      stbi_load_from_memory(PNG, (int)PNG_LEN, &w, &h, &comp, 4); /* force RGBA out */
  if (!pixels || w != IMG_W || h != IMG_H) return 1;

  /* Emit the decoded image as raw RGBA bytes — the differential payload. */
  write(1, (char *)pixels, (long)w * h * 4);

  stbi_image_free(pixels);
  return 0;
}
