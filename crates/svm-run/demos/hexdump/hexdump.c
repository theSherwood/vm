/* Shakedown driver: a classic `hexdump -C`-style tool, to exercise the on-ramp's varargs `printf`.
 *
 * Reads stdin in 16-byte rows and prints each as `OFFSET  HH HH … HH  |ascii|` via `printf`, the
 * dominant general-C output path. Exercises the format engine: `%08lx`/`%02x` (hex with width +
 * zero-pad), `%c`, `%s`, plain literals, and a length modifier (`l`). svm-run's output must match a
 * native `cc` build byte-for-byte. The format engine runs **in the guest** (parsed at translate
 * time, lowered to int→string + `Stream.write`); only the bytes cross the capability boundary. */
#include <stddef.h>

long read(int fd, char *buf, long n);
int printf(const char *fmt, ...);

int main(void) {
  unsigned char buf[16];
  long off = 0, n;
  while ((n = read(0, (char *)buf, sizeof buf)) > 0) {
    printf("%08lx  ", off);
    for (int i = 0; i < 16; i++) {
      if (i < n)
        printf("%02x ", buf[i]);
      else
        printf("   ");
      if (i == 7) printf(" ");
    }
    printf(" |");
    for (long i = 0; i < n; i++) {
      unsigned char c = buf[i];
      printf("%c", (c >= 32 && c < 127) ? c : '.');
    }
    printf("|\n");
    off += n;
  }
  return 0;
}
