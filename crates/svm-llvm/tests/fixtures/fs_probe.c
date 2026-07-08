/* Probe for the **configurable Fs capability** (svm-run's `fs::mem_fs`/`fs::host_fs`): resolve the
 * embedder-granted capability by name (`__vm_cap_resolve`, §7 cap.self.resolve) and drive the whole
 * op protocol through `__vm_host_call` (§7 host-defined capability, the wasm-import analogue) —
 * open/write/close, reopen/seek/read-back, rename, append, remove, and the attenuation refusals
 * (`..`/absolute paths). Exit 0 iff every step behaved; any failure returns its step number.
 *
 * Phase B (host-fs verification) is keyed on a `seed.txt` the embedder may pre-place in the granted
 * root: when present, its content is verified and an `out.txt` is written AND LEFT BEHIND, so the
 * Rust test can assert the bytes really landed in the real directory. Under `mem_fs` (fresh, empty)
 * the phase is skipped — the same binary runs against both backends. */

extern int __vm_cap_resolve(const char *name, long len);
extern long __vm_host_call(int h, int op, long a, long b, long c, long d);
int printf(const char *, ...);

enum { OPEN = 0, READ = 1, WRITE = 2, SEEK = 3, CLOSE = 4, REMOVE = 5, RENAME = 6, TRUNCATE = 7, SYNC = 8 };
enum { O_READ = 1, O_WRITE = 2, O_APPEND = 4, O_TRUNC = 8, O_CREATE = 16 };

static int fs;
static long hc(int op, long a, long b, long c, long d) { return __vm_host_call(fs, op, a, b, c, d); }

int main(void) {
  fs = __vm_cap_resolve("fs", 2);
  if (fs < 0) return 1;

  /* create + write */
  long fd = hc(OPEN, (long)"hello.txt", 9, O_WRITE | O_CREATE | O_TRUNC, 0);
  if (fd < 0) return 2;
  if (hc(WRITE, fd, (long)"hello, fs!", 10, 0) != 10) return 3;
  if (hc(CLOSE, fd, 0, 0, 0) != 0) return 4;

  /* reopen read-only: size via seek-end, absolute seek, read-back */
  fd = hc(OPEN, (long)"hello.txt", 9, O_READ, 0);
  if (fd < 0) return 5;
  if (hc(SEEK, fd, 2, 0, 0) != 10) return 6;
  if (hc(SEEK, fd, 0, 7, 0) != 7) return 7;
  char buf[16];
  long n = hc(READ, fd, (long)buf, 16, 0);
  if (n != 3) return 8;
  if (buf[0] != 'f' || buf[1] != 's' || buf[2] != '!') return 9;
  if (hc(READ, fd, (long)buf, 16, 0) != 0) return 10; /* EOF */
  if (hc(WRITE, fd, (long)"x", 1, 0) >= 0) return 11; /* read-only fd refuses writes */
  if (hc(CLOSE, fd, 0, 0, 0) != 0) return 12;

  /* rename: old name gone, new name readable */
  if (hc(RENAME, (long)"hello.txt", 9, (long)"world.txt", 9) != 0) return 13;
  if (hc(OPEN, (long)"hello.txt", 9, O_READ, 0) >= 0) return 14;
  fd = hc(OPEN, (long)"world.txt", 9, O_READ, 0);
  if (fd < 0) return 15;
  if (hc(CLOSE, fd, 0, 0, 0) != 0) return 16;

  /* append grows the file at its end */
  fd = hc(OPEN, (long)"world.txt", 9, O_APPEND | O_CREATE, 0);
  if (fd < 0) return 17;
  if (hc(WRITE, fd, (long)"++", 2, 0) != 2) return 18;
  if (hc(CLOSE, fd, 0, 0, 0) != 0) return 19;
  fd = hc(OPEN, (long)"world.txt", 9, O_READ, 0);
  if (hc(SEEK, fd, 2, 0, 0) != 12) return 20;
  if (hc(CLOSE, fd, 0, 0, 0) != 0) return 21;

  /* remove: gone afterwards; missing files stay errors */
  if (hc(REMOVE, (long)"world.txt", 9, 0, 0) != 0) return 22;
  if (hc(OPEN, (long)"world.txt", 9, O_READ, 0) >= 0) return 23;
  if (hc(REMOVE, (long)"world.txt", 9, 0, 0) >= 0) return 24;

  /* truncate: shrink discards, grow zero-fills; read-only fds refuse; sync succeeds on any fd */
  fd = hc(OPEN, (long)"trunc.txt", 9, O_READ | O_WRITE | O_CREATE | O_TRUNC, 0);
  if (fd < 0) return 34;
  if (hc(WRITE, fd, (long)"0123456789", 10, 0) != 10) return 35;
  if (hc(TRUNCATE, fd, 4, 0, 0) != 0) return 36;
  if (hc(SYNC, fd, 0, 0, 0) != 0) return 37;
  if (hc(SEEK, fd, 2, 0, 0) != 4) return 38; /* shrunk */
  if (hc(TRUNCATE, fd, 6, 0, 0) != 0) return 39;
  if (hc(SEEK, fd, 0, 3, 0) != 3) return 40;
  n = hc(READ, fd, (long)buf, 16, 0);
  if (n != 3) return 41;
  if (buf[0] != '3' || buf[1] != 0 || buf[2] != 0) return 42; /* grow zero-filled */
  if (hc(CLOSE, fd, 0, 0, 0) != 0) return 43;
  fd = hc(OPEN, (long)"trunc.txt", 9, O_READ, 0);
  if (fd < 0) return 44;
  if (hc(TRUNCATE, fd, 0, 0, 0) >= 0) return 45; /* read-only fd refuses truncate */
  if (hc(CLOSE, fd, 0, 0, 0) != 0) return 46;
  if (hc(REMOVE, (long)"trunc.txt", 9, 0, 0) != 0) return 47;

  /* attenuation: `..` and absolute paths are refused by protocol, on every backend */
  if (hc(OPEN, (long)"../escape", 9, O_WRITE | O_CREATE, 0) >= 0) return 25;
  if (hc(OPEN, (long)"/etc/pass", 9, O_READ, 0) >= 0) return 26;
  if (hc(OPEN, (long)"a/../b.txt", 10, O_WRITE | O_CREATE, 0) >= 0) return 27;

  /* phase B (host-fs only): verify a pre-seeded file, leave `out.txt` for the host to inspect */
  fd = hc(OPEN, (long)"seed.txt", 8, O_READ, 0);
  if (fd >= 0) {
    n = hc(READ, fd, (long)buf, 16, 0);
    if (n != 4) return 28;
    if (buf[0] != 'S' || buf[1] != 'E' || buf[2] != 'E' || buf[3] != 'D') return 29;
    if (hc(CLOSE, fd, 0, 0, 0) != 0) return 30;
    fd = hc(OPEN, (long)"out.txt", 7, O_WRITE | O_CREATE | O_TRUNC, 0);
    if (fd < 0) return 31;
    if (hc(WRITE, fd, (long)"GUEST", 5, 0) != 5) return 32;
    if (hc(CLOSE, fd, 0, 0, 0) != 0) return 33;
  }

  printf("fs probe ok\n");
  return 0;
}
