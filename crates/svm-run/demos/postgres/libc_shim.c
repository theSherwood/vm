/* Guest pure-libc shim — the computational libc surface Postgres needs that isn't a syscall and
 * isn't already synthesized by the on-ramp (slice CB, Postgres runtime gap #11c). Companion to
 * `os_shim.c` (the file/directory syscalls): this file holds the *pure* functions — no capability,
 * no host call, just deterministic computation the guest carries itself.
 *
 * First inhabitant: **ctype** (`__ctype_b_loc`/`__ctype_tolower_loc`/`__ctype_toupper_loc`). glibc's
 * `<ctype.h>` `isalpha`/`isdigit`/… macros expand to `(*__ctype_b_loc())[c] & _ISbit` — a direct
 * index into a locale table; Postgres's scanner/parser classify every input byte this way, so the
 * SQL front end is dead without them. We reproduce the **C/POSIX-locale** tables *exactly* (the
 * locale Postgres bootstraps in), as static compile-time literals so there is no runtime init and
 * no chance of an unresolved table pointer. Validated byte-for-byte against glibc by the
 * `ctype_probe.c` differential (all twelve classes + case mapping over every byte 0..255).
 *
 * The tables span the index range [-128, 255] (glibc allows `table[EOF]` and signed-`char` indices);
 * each `__ctype_*_loc` returns a pointer to element 0, so `ptr[c]` is valid across that range. The
 * `unsigned short` class bits are glibc's `_ISbit` ABI (bits/ctype-info.h): `_ISupper`=0x0100,
 * `_ISlower`=0x0200, `_ISalpha`=0x0400, `_ISdigit`=0x0800, `_ISxdigit`=0x1000, `_ISspace`=0x2000,
 * `_ISprint`=0x4000, `_ISgraph`=0x8000, `_ISblank`=0x0001, `_IScntrl`=0x0002, `_ISpunct`=0x0004,
 * `_ISalnum`=0x0008.
 *
 * `#include`d into a driver under `-DSVM_GUEST`, like `os_shim.c`.
 */

static const unsigned short shim_ctype_b[384] = {
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0002, 0x0002, 0x0002, 0x0002,
  0x0002, 0x0002, 0x0002, 0x0002, 0x0002, 0x2003, 0x2002, 0x2002, 0x2002, 0x2002, 0x0002, 0x0002,
  0x0002, 0x0002, 0x0002, 0x0002, 0x0002, 0x0002, 0x0002, 0x0002, 0x0002, 0x0002, 0x0002, 0x0002,
  0x0002, 0x0002, 0x0002, 0x0002, 0x6001, 0xc004, 0xc004, 0xc004, 0xc004, 0xc004, 0xc004, 0xc004,
  0xc004, 0xc004, 0xc004, 0xc004, 0xc004, 0xc004, 0xc004, 0xc004, 0xd808, 0xd808, 0xd808, 0xd808,
  0xd808, 0xd808, 0xd808, 0xd808, 0xd808, 0xd808, 0xc004, 0xc004, 0xc004, 0xc004, 0xc004, 0xc004,
  0xc004, 0xd508, 0xd508, 0xd508, 0xd508, 0xd508, 0xd508, 0xc508, 0xc508, 0xc508, 0xc508, 0xc508,
  0xc508, 0xc508, 0xc508, 0xc508, 0xc508, 0xc508, 0xc508, 0xc508, 0xc508, 0xc508, 0xc508, 0xc508,
  0xc508, 0xc508, 0xc508, 0xc004, 0xc004, 0xc004, 0xc004, 0xc004, 0xc004, 0xd608, 0xd608, 0xd608,
  0xd608, 0xd608, 0xd608, 0xc608, 0xc608, 0xc608, 0xc608, 0xc608, 0xc608, 0xc608, 0xc608, 0xc608,
  0xc608, 0xc608, 0xc608, 0xc608, 0xc608, 0xc608, 0xc608, 0xc608, 0xc608, 0xc608, 0xc608, 0xc004,
  0xc004, 0xc004, 0xc004, 0x0002, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000
};
static const int shim_ctype_tolower[384] = {
  -128, -127, -126, -125, -124, -123, -122, -121, -120, -119, -118, -117,
  -116, -115, -114, -113, -112, -111, -110, -109, -108, -107, -106, -105,
  -104, -103, -102, -101, -100, -99, -98, -97, -96, -95, -94, -93,
  -92, -91, -90, -89, -88, -87, -86, -85, -84, -83, -82, -81,
  -80, -79, -78, -77, -76, -75, -74, -73, -72, -71, -70, -69,
  -68, -67, -66, -65, -64, -63, -62, -61, -60, -59, -58, -57,
  -56, -55, -54, -53, -52, -51, -50, -49, -48, -47, -46, -45,
  -44, -43, -42, -41, -40, -39, -38, -37, -36, -35, -34, -33,
  -32, -31, -30, -29, -28, -27, -26, -25, -24, -23, -22, -21,
  -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9,
  -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3,
  4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
  16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
  28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
  40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
  52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
  64, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107,
  108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
  120, 121, 122, 91, 92, 93, 94, 95, 96, 97, 98, 99,
  100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
  112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
  124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135,
  136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147,
  148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
  160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171,
  172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183,
  184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195,
  196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207,
  208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219,
  220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231,
  232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243,
  244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255
};
static const int shim_ctype_toupper[384] = {
  -128, -127, -126, -125, -124, -123, -122, -121, -120, -119, -118, -117,
  -116, -115, -114, -113, -112, -111, -110, -109, -108, -107, -106, -105,
  -104, -103, -102, -101, -100, -99, -98, -97, -96, -95, -94, -93,
  -92, -91, -90, -89, -88, -87, -86, -85, -84, -83, -82, -81,
  -80, -79, -78, -77, -76, -75, -74, -73, -72, -71, -70, -69,
  -68, -67, -66, -65, -64, -63, -62, -61, -60, -59, -58, -57,
  -56, -55, -54, -53, -52, -51, -50, -49, -48, -47, -46, -45,
  -44, -43, -42, -41, -40, -39, -38, -37, -36, -35, -34, -33,
  -32, -31, -30, -29, -28, -27, -26, -25, -24, -23, -22, -21,
  -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9,
  -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3,
  4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
  16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
  28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
  40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
  52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
  64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
  76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87,
  88, 89, 90, 91, 92, 93, 94, 95, 96, 65, 66, 67,
  68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
  80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 123,
  124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135,
  136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147,
  148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
  160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171,
  172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183,
  184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195,
  196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207,
  208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219,
  220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231,
  232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243,
  244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255
};

/* Each pointer is initialized *at load time* to element 128 (code point 0) of its table — a
 * compile-time-constant address, so no initializer function runs in the guest. */
static const unsigned short *shim_ctype_b_ptr = shim_ctype_b + 128;
static const int *shim_ctype_tolower_ptr = shim_ctype_tolower + 128;
static const int *shim_ctype_toupper_ptr = shim_ctype_toupper + 128;

const unsigned short **__ctype_b_loc(void) { return &shim_ctype_b_ptr; }
const int **__ctype_tolower_loc(void) { return &shim_ctype_tolower_ptr; }
const int **__ctype_toupper_loc(void) { return &shim_ctype_toupper_ptr; }

/* ============================================================================================== *
 * string.h — the members Postgres uses that the on-ramp does NOT already synthesize.
 * (Synthesized already: strlen/strcmp/strcpy/strchr/strrchr/strspn/strcspn/strpbrk/strncmp/
 * strcoll/memcmp/memchr/bcmp — those are NOT redefined here.)
 * ============================================================================================== */

#include <locale.h>
#include <stddef.h>
#include <stdlib.h> /* malloc, for strdup */
#include <string.h> /* strchr (used by getopt); the shim's own str* match these declarations */

static size_t shim_strlen(const char *s) {
  const char *p = s;
  while (*p) p++;
  return (size_t)(p - s);
}

char *strcat(char *dst, const char *src) {
  char *d = dst + shim_strlen(dst);
  while ((*d++ = *src++)) {
  }
  return dst;
}
char *strncat(char *dst, const char *src, size_t n) {
  char *d = dst + shim_strlen(dst);
  while (n-- && *src) *d++ = *src++;
  *d = 0;
  return dst;
}
char *strncpy(char *dst, const char *src, size_t n) {
  size_t i = 0;
  for (; i < n && src[i]; i++) dst[i] = src[i];
  for (; i < n; i++) dst[i] = 0; /* POSIX: NUL-pad the remainder */
  return dst;
}
size_t strnlen(const char *s, size_t n) {
  size_t i = 0;
  while (i < n && s[i]) i++;
  return i;
}
char *strstr(const char *hay, const char *needle) {
  if (!*needle) return (char *)hay;
  for (; *hay; hay++) {
    const char *h = hay, *n = needle;
    while (*h && *n && *h == *n) {
      h++;
      n++;
    }
    if (!*n) return (char *)hay;
  }
  return (char *)0;
}
char *strchrnul(const char *s, int c) {
  while (*s && *s != (char)c) s++;
  return (char *)s; /* points at the match, or at the terminating NUL */
}
char *strdup(const char *s) {
  size_t n = shim_strlen(s) + 1;
  char *p = (char *)malloc(n);
  if (p) {
    for (size_t i = 0; i < n; i++) p[i] = s[i];
  }
  return p;
}
/* BSD strlcpy/strlcat: size-bounded, always NUL-terminate, return the length they *tried* to build. */
size_t strlcpy(char *dst, const char *src, size_t size) {
  size_t sl = shim_strlen(src);
  if (size) {
    size_t n = sl < size - 1 ? sl : size - 1;
    for (size_t i = 0; i < n; i++) dst[i] = src[i];
    dst[n] = 0;
  }
  return sl;
}
size_t strlcat(char *dst, const char *src, size_t size) {
  size_t dl = 0;
  while (dl < size && dst[dl]) dl++;
  size_t sl = shim_strlen(src);
  if (dl == size) return size + sl; /* dst not NUL-terminated within size */
  size_t n = sl < size - dl - 1 ? sl : size - dl - 1;
  for (size_t i = 0; i < n; i++) dst[dl + i] = src[i];
  dst[dl + n] = 0;
  return dl + sl;
}
char *strtok_r(char *s, const char *delim, char **save) {
  if (!s) s = *save;
  /* skip leading delimiters */
  for (; *s; s++) {
    const char *d = delim;
    int hit = 0;
    for (; *d; d++)
      if (*s == *d) {
        hit = 1;
        break;
      }
    if (!hit) break;
  }
  if (!*s) {
    *save = s;
    return (char *)0;
  }
  char *tok = s;
  for (; *s; s++) {
    const char *d = delim;
    for (; *d; d++)
      if (*s == *d) {
        *s = 0;
        *save = s + 1;
        return tok;
      }
  }
  *save = s;
  return tok;
}
static char *shim_strtok_save;
char *strtok(char *s, const char *delim) { return strtok_r(s, delim, &shim_strtok_save); }
/* In the C/POSIX locale collation is byte order and transform is identity, so strxfrm is a bounded
 * copy returning the source length and strcoll/_l are strcmp. */
size_t strxfrm(char *dst, const char *src, size_t n) {
  size_t sl = shim_strlen(src);
  if (n) {
    size_t c = sl < n - 1 ? sl : n - 1;
    for (size_t i = 0; i < c; i++) dst[i] = src[i];
    dst[c] = 0;
  }
  return sl;
}
int strcoll_l(const char *a, const char *b, locale_t loc) {
  (void)loc;
  for (; *a && *a == *b; a++, b++) {
  }
  return (int)(unsigned char)*a - (int)(unsigned char)*b;
}

/* ============================================================================================== *
 * stdlib.h — integer parsing (`strtol`/`strtoul`). The float `strtod`/`strtof`/`atof` are the
 * correctly-rounded bignum parser in `../strtod/strtod.c` (linked via `pg_shims.c`). glibc's C23 build
 * renames the callers to `__isoc23_strtol`/`__isoc23_strtoul` (identical semantics), resolved here too.
 * ============================================================================================== */

#include <errno.h>
#include <limits.h>

#include "shim_errno.h" /* provides errno's __errno_location (shared with os_shim.c) */

static int shim_digit(int c, int base) {
  int d;
  if (c >= '0' && c <= '9')
    d = c - '0';
  else if (c >= 'a' && c <= 'z')
    d = c - 'a' + 10;
  else if (c >= 'A' && c <= 'Z')
    d = c - 'A' + 10;
  else
    return -1;
  return d < base ? d : -1;
}
/* Shared core: parse [ws][sign][0x/0]digits in `base` (0 = autodetect), honoring endptr and ERANGE.
 * `is_signed` selects the LONG_MIN/LONG_MAX vs ULONG_MAX clamp; the unsigned path still accepts a
 * leading '-' and negates modulo 2^64, exactly as C requires. */
static unsigned long shim_strtox(const char *s, char **end, int base, int is_signed, int *neg_out) {
  const char *p = s;
  while (*p == ' ' || (*p >= '\t' && *p <= '\r')) p++; /* isspace, C locale */
  int neg = 0;
  if (*p == '+' || *p == '-') neg = (*p++ == '-');
  if ((base == 0 || base == 16) && p[0] == '0' && (p[1] == 'x' || p[1] == 'X') &&
      shim_digit(p[2], 16) >= 0) {
    p += 2;
    base = 16;
  } else if (base == 0 && p[0] == '0') {
    base = 8;
  } else if (base == 0) {
    base = 10;
  }
  unsigned long acc = 0;
  int any = 0, overflow = 0;
  unsigned long cutoff = is_signed ? (neg ? (unsigned long)LONG_MAX + 1UL : (unsigned long)LONG_MAX)
                                   : ULONG_MAX;
  for (int d; (d = shim_digit(*p, base)) >= 0; p++) {
    any = 1;
    if (acc > (cutoff - (unsigned long)d) / (unsigned long)base) overflow = 1;
    acc = acc * (unsigned long)base + (unsigned long)d;
  }
  if (end) *end = (char *)(any ? p : s);
  *neg_out = neg;
  if (overflow) {
    errno = ERANGE;
    if (is_signed) return neg ? (unsigned long)LONG_MIN : (unsigned long)LONG_MAX;
    return ULONG_MAX;
  }
  return acc;
}
long strtol(const char *s, char **end, int base) {
  int neg;
  unsigned long v = shim_strtox(s, end, base, 1, &neg);
  if (v == (unsigned long)LONG_MIN || v == (unsigned long)LONG_MAX) return (long)v; /* clamped */
  return neg ? -(long)v : (long)v;
}
unsigned long strtoul(const char *s, char **end, int base) {
  int neg;
  unsigned long v = shim_strtox(s, end, base, 0, &neg);
  return neg ? (unsigned long)(-(long)v) : v;
}
long __isoc23_strtol(const char *s, char **end, int base) { return strtol(s, end, base); }
unsigned long __isoc23_strtoul(const char *s, char **end, int base) { return strtoul(s, end, base); }
int atoi(const char *s) { return (int)strtol(s, (char **)0, 10); }
long atol(const char *s) { return strtol(s, (char **)0, 10); }

/* ============================================================================================== *
 * wchar — the C/POSIX locale is a byte↔wchar identity map (each byte value 0..255 is its own wide
 * character), so mbstowcs/wcstombs are widening/narrowing copies. That's exactly glibc's C-locale
 * behavior; Postgres uses them in a few encoding-conversion fallbacks.
 * ============================================================================================== */

#include <wchar.h>

size_t mbstowcs(wchar_t *dst, const char *src, size_t n) {
  size_t i = 0;
  if (!dst) { /* just count (up to the terminating NUL) */
    while (src[i]) i++;
    return i;
  }
  for (; i < n; i++) {
    dst[i] = (wchar_t)(unsigned char)src[i];
    if (!src[i]) return i; /* stop at NUL (which is written but not counted) */
  }
  return i; /* filled n wide chars without hitting NUL */
}
/* The C environment. The powerbox passes no ambient env, so it starts empty — but Postgres both
 * *reads* it (`getenv`, `save_ps_display_args` walking `environ`) and *writes* it (`setenv` for
 * locale/service vars in early startup), so provide a small mutable vector with coherent
 * `getenv`/`setenv`/`unsetenv`/`putenv`. Defining `getenv` here shadows the on-ramp's synthesized
 * (always-NULL) one so a read sees what a prior write set. Fixed capacity, NULL-terminated; entries
 * are heap "NAME=VALUE" strings (an overwrite leaks the old — fine for a single boot). */
#define SHIM_ENV_MAX 64
static char *shim_environ[SHIM_ENV_MAX + 1] = {(char *)0};
char **environ = shim_environ;

static int env_find(const char *name, size_t nlen) {
  for (int i = 0; shim_environ[i]; i++) {
    if (!strncmp(shim_environ[i], name, nlen) && shim_environ[i][nlen] == '=') return i;
  }
  return -1;
}
static int env_count(void) {
  int n = 0;
  while (shim_environ[n]) n++;
  return n;
}
char *getenv(const char *name) {
  if (!name) return (char *)0;
  int i = env_find(name, strlen(name));
  if (i < 0) return (char *)0;
  char *eq = strchr(shim_environ[i], '=');
  return eq ? eq + 1 : (char *)0;
}
int setenv(const char *name, const char *value, int overwrite) {
  if (!name || !*name || strchr(name, '=')) {
    shim_errno = 22; /* EINVAL */
    return -1;
  }
  size_t nlen = strlen(name), vlen = strlen(value);
  int i = env_find(name, nlen);
  if (i >= 0 && !overwrite) return 0;
  char *entry = (char *)malloc(nlen + 1 + vlen + 1);
  if (!entry) {
    shim_errno = 12; /* ENOMEM */
    return -1;
  }
  memcpy(entry, name, nlen);
  entry[nlen] = '=';
  memcpy(entry + nlen + 1, value, vlen + 1);
  if (i >= 0) {
    shim_environ[i] = entry;
    return 0;
  }
  int n = env_count();
  if (n >= SHIM_ENV_MAX) {
    shim_errno = 12;
    return -1;
  }
  shim_environ[n] = entry;
  shim_environ[n + 1] = (char *)0;
  return 0;
}
int unsetenv(const char *name) {
  if (!name || !*name || strchr(name, '=')) {
    shim_errno = 22;
    return -1;
  }
  int i = env_find(name, strlen(name));
  if (i < 0) return 0;
  int n = env_count();
  shim_environ[i] = shim_environ[n - 1]; /* swap the last into the hole */
  shim_environ[n - 1] = (char *)0;
  return 0;
}
int putenv(char *str) {
  /* POSIX: `str` becomes part of the environment directly (not copied). */
  char *eq = strchr(str, '=');
  if (!eq) return unsetenv(str);
  size_t nlen = (size_t)(eq - str);
  int i = env_find(str, nlen);
  if (i >= 0) {
    shim_environ[i] = str;
    return 0;
  }
  int n = env_count();
  if (n >= SHIM_ENV_MAX) {
    shim_errno = 12;
    return -1;
  }
  shim_environ[n] = str;
  shim_environ[n + 1] = (char *)0;
  return 0;
}

size_t wcstombs(char *dst, const wchar_t *src, size_t n) {
  size_t i = 0;
  if (!dst) {
    while (src[i]) i++;
    return i;
  }
  for (; i < n; i++) {
    dst[i] = (char)src[i];
    if (!src[i]) return i;
  }
  return i;
}

/* ---- pseudo-random (`srandom`/`random`, `srand`/`rand`) --------------------------------------- *
 * Postgres seeds `srandom` at startup (e.g. the postmaster's random-seed init) and draws `random()`
 * for non-cryptographic values (cancel keys, backoff jitter). The exact glibc TYPE_3 sequence isn't
 * required — only a deterministic, decently-mixed 31-bit stream — so this is a small xorshift over a
 * 64-bit state, folded to POSIX's `[0, RAND_MAX]`. Deterministic given the seed (sandbox
 * reproducibility). `initstate`/`setstate` are accepted for completeness. */
static unsigned long long g_rng = 0x2545F4914F6CDD1DULL;
void srandom(unsigned int seed) { g_rng = seed ? (unsigned long long)seed : 1ULL; }
long random(void) {
  unsigned long long x = g_rng;
  x ^= x << 13;
  x ^= x >> 7;
  x ^= x << 17;
  g_rng = x;
  return (long)((x >> 33) & 0x7fffffffUL); /* top bits, 31-bit range */
}
void srand(unsigned int seed) { srandom(seed); }
int rand(void) { return (int)random(); }
char *initstate(unsigned int seed, char *state, size_t n) {
  (void)state;
  (void)n;
  srandom(seed);
  return state;
}
char *setstate(char *state) { return state; }

/* ============================================================================================== *
 * getopt (+ strsignal). `strerror` lives in locale_shim.c; the GNU `char *strerror_r` needs its own
 * `_GNU_SOURCE`-isolated TU (`strerror_shim.c`) so its prototype doesn't clash with the POSIX one here.
 * ============================================================================================== */

char *strsignal(int sig) {
  (void)sig;
  return (char *)"Signal";
}

/* Standard `getopt` (no `getopt_long`) + its globals — Postgres's `--single` arg parsing uses it. */
char *optarg = (char *)0;
int optind = 1, opterr = 1, optopt = 0;
static int shim_optpos = 1;
int getopt(int argc, char *const argv[], const char *optstring) {
  if (optind >= argc || !argv[optind] || argv[optind][0] != '-' || argv[optind][1] == 0) return -1;
  if (argv[optind][1] == '-' && argv[optind][2] == 0) { /* "--" ends option parsing */
    optind++;
    return -1;
  }
  int c = (unsigned char)argv[optind][shim_optpos];
  const char *p = (c == ':') ? (char *)0 : strchr(optstring, c);
  if (!p) {
    optopt = c;
    if (argv[optind][++shim_optpos] == 0) {
      optind++;
      shim_optpos = 1;
    }
    return '?';
  }
  if (p[1] == ':') { /* option takes an argument */
    if (argv[optind][shim_optpos + 1] != 0) {
      optarg = (char *)&argv[optind][shim_optpos + 1];
      optind++;
    } else if (optind + 1 < argc) {
      optarg = argv[optind + 1];
      optind += 2;
    } else {
      optopt = c;
      optind++;
      shim_optpos = 1;
      return optstring[0] == ':' ? ':' : '?';
    }
    shim_optpos = 1;
    return c;
  }
  if (argv[optind][++shim_optpos] == 0) {
    optind++;
    shim_optpos = 1;
  }
  return c;
}
