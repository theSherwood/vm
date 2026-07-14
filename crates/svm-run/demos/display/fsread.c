/* fs-capability reactor demo (Doom slice 4): read a file through the `fs` capability at init, then
 * present its bytes as a framebuffer every frame. This is the last waist Doom needs — the WAD read
 * path. doomgeneric reads its IWAD through stock C file I/O at `doomgeneric_Create`; here we exercise
 * the same open/seek/read shape directly against the `fs` cap so the reactor's fs plumbing is proven
 * on its own, without the whole Doom module.
 *
 *   f = __vm_cap_resolve("fs", 2);
 *   fd = __vm_host_call(f, 0, nameptr, namelen, flags, 0);   // open  → fd | -2 (ENOENT)
 *   n  = __vm_host_call(f, 1, fd, bufptr, len, 0);           // read  → bytes read
 *   p  = __vm_host_call(f, 3, fd, whence, off, 0);           // seek  → new pos (whence 0=SET,1=CUR,2=END)
 *
 * `_start → main` resolves the caps and reads "data.bin" once (open, seek to END for the size, seek
 * back, read) — the reactor keeps this guest instance alive, so the bytes stashed here persist for
 * every `tick`. Each `tick` renders the file bytes as a 16×16 grayscale image (pixel i's R=G=B is
 * byte i, or 0 past the end) and presents it through `display`. The frame is a pure function of the
 * served file, so a fixed blob yields a deterministic frame — the differential anchor
 * (browser/tests/reactor_fs.rs). */
extern int __vm_cap_resolve(const char *name, long len);
extern long __vm_host_call(int h, int op, long a, long b, long c, long d);

#define W 16
#define H 16
#define CAP (W * H) /* at most one byte per pixel */

static unsigned char fb[W * H * 4];
static unsigned char data[CAP];
static int disp, fs; /* resolved capability handles (stashed once, in _start → main) */
static int len;      /* bytes actually read from "data.bin" */

int main(void) {
  disp = __vm_cap_resolve("display", 7);
  fs = __vm_cap_resolve("fs", 2);
  len = 0;
  if (fs >= 0) {
    static const char name[] = "data.bin";
    long fd = __vm_host_call(fs, 0, (long)name, 8, 0, 0);
    if (fd >= 0) {
      long size = __vm_host_call(fs, 3, fd, 2, 0, 0); /* seek END → file size */
      __vm_host_call(fs, 3, fd, 0, 0, 0);             /* seek back to start   */
      if (size > CAP) size = CAP;
      long n = __vm_host_call(fs, 1, fd, (long)data, size, 0); /* read the bytes in */
      if (n > 0) len = (int)n;
      __vm_host_call(fs, 4, fd, 0, 0, 0); /* close */
    }
  }
  return 0;
}

/* One frame: render the file bytes as a grayscale image and present it. */
int tick(void) {
  for (int i = 0; i < W * H; i++) {
    unsigned char v = i < len ? data[i] : 0;
    unsigned char *p = &fb[i * 4];
    p[0] = v;   /* R */
    p[1] = v;   /* G */
    p[2] = v;   /* B */
    p[3] = 255; /* A */
  }
  if (disp >= 0)
    __vm_host_call(disp, 0, (long)fb, W, H, 0);
  return 0; /* keep going */
}
