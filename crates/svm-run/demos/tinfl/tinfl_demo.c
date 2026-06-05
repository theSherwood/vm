/* Shakedown driver: run miniz's `tinfl` DEFLATE/zlib *inflate* engine in the sandbox.
 *
 * tinfl is a single-file Lehmer-style inflate state machine — a deeply nested switch
 * driven by a "coroutine" macro (TINFL_CR_*), bit-buffer shifts, Huffman fast/slow
 * lookup tables, and a 32 KiB LZ77 dictionary carried inside `tinfl_decompressor`.
 * It is a very different shape from the earlier shakedowns and a good stress test of
 * goto/switch lowering, struct layout, and unaligned-free byte loads.
 *
 * `blob.inc` is a zlib stream (generated with stock zlib) of the line
 *   "The quick brown fox jumps over the lazy dog.\n" repeated six times (270 bytes,
 *   compressed to 55). We inflate it back and write it to stdout; `svm-run`'s output
 *   must match a native `cc` build byte-for-byte. */

#include <stddef.h>

/* The whole-program sandbox build has no libc; provide the two mem ops tinfl uses. */
void *memcpy(void *d, const void *s, size_t n) {
  unsigned char *p = (unsigned char *)d;
  const unsigned char *q = (const unsigned char *)s;
  for (size_t i = 0; i < n; i++) p[i] = q[i];
  return d;
}
void *memset(void *s, int c, size_t n) {
  unsigned char *p = (unsigned char *)s;
  for (size_t i = 0; i < n; i++) p[i] = (unsigned char)c;
  return s;
}

#define MINIZ_NO_STDIO
#define MINIZ_NO_TIME
#define MINIZ_NO_MALLOC
#ifndef NDEBUG
#define NDEBUG
#endif
#include "miniz_tinfl.c"
#include "blob.inc"

int write(int fd, char *buf, long n);

int main(void) {
  static unsigned char out[4096];
  tinfl_decompressor decomp;
  tinfl_init(&decomp);
  size_t in_len = BLOB_LEN, out_len = sizeof(out);
  tinfl_status st = tinfl_decompress(
      &decomp, BLOB, &in_len, out, out, &out_len,
      TINFL_FLAG_PARSE_ZLIB_HEADER | TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF);
  if (st != TINFL_STATUS_DONE || out_len != ORIG_LEN) {
    char *e = "DECOMPRESS FAILED\n";
    int n = 0;
    while (e[n]) n++;
    write(1, e, n);
    return 1;
  }
  write(1, (char *)out, (long)out_len);
  return 0;
}
