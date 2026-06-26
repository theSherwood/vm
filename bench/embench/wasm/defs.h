/* Definitions of the few libc functions the Embench `run` path actually *calls* (as opposed to the
 * `memcpy`/`memset`/`memmove` that `-mbulk-memory` lowers to wasm instructions, or `memcmp`/`bcmp`
 * that `wrapper.c`'s `SVM_BUILD` block defines). Force-included ahead of the kernel sources for the
 * wasm32 build (`clang -include`), which is `-nostdlib` and so has no real libc. The host SVM build
 * gets these from the on-ramp's synthesized helpers instead and never includes this file. */
#include <stddef.h>
size_t strlen(const char *s) {
  const char *p = s;
  while (*p) p++;
  return (size_t)(p - s);
}
char *strchr(const char *s, int c) {
  for (;; s++) {
    if (*s == (char)c) return (char *)s;
    if (!*s) return 0;
  }
}
void *memchr(const void *s, int c, size_t n) {
  const unsigned char *p = s;
  while (n--) {
    if (*p == (unsigned char)c) return (void *)p;
    p++;
  }
  return 0;
}
int strcmp(const char *a, const char *b) {
  while (*a && *a == *b) { a++; b++; }
  return (int)(unsigned char)*a - (int)(unsigned char)*b;
}
int strncmp(const char *a, const char *b, size_t n) {
  while (n--) {
    if (*a != *b) return (int)(unsigned char)*a - (int)(unsigned char)*b;
    if (!*a) break;
    a++; b++;
  }
  return 0;
}
