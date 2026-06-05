#include <stddef.h>
/* The whole-program build has no libc, so provide the one libc function B-Con's sha256.c uses. */
void *memset(void *s, int c, size_t n) {
  unsigned char *p = (unsigned char *)s;
  for (size_t i = 0; i < n; i++) p[i] = (unsigned char)c;
  return s;
}
#include "sha256.c"

int write(int fd, char *buf, long n);
static void puthex(BYTE *h, int n) {
  char *hx = "0123456789abcdef"; char out[2];
  for (int i = 0; i < n; i++) { out[0] = hx[h[i] >> 4]; out[1] = hx[h[i] & 15]; write(1, out, 2); }
  char nl = '\n'; write(1, &nl, 1);
}
static int slen(const char *s) { int n = 0; while (s[n]) n++; return n; }
static void hash_str(const char *msg) {
  SHA256_CTX ctx; BYTE hash[SHA256_BLOCK_SIZE];
  sha256_init(&ctx);
  sha256_update(&ctx, (const BYTE *)msg, slen(msg));
  sha256_final(&ctx, hash);
  puthex(hash, SHA256_BLOCK_SIZE);
}
int main(void) {
  hash_str("");                                          /* e3b0c442... */
  hash_str("abc");                                       /* ba7816bf... */
  hash_str("The quick brown fox jumps over the lazy dog"); /* d7a8fbb3... */
  return 0;
}
