/* Conway's Game of Life (Doom slice 3): the heap-persistence demo. The cell grid lives in the
 * malloc heap — which the on-ramp grows into the reserved tail ABOVE the mapped window, exactly where
 * Doom's zone allocator will sit. Each tick computes the next generation *from the current one*, so
 * the demo only advances if the guest's whole memory (heap included) persists between frames. Under a
 * snapshot reactor that round-trips just the low window, the heap resets every frame and the glider
 * freezes at generation 0; with the live-window reactor it evolves. Deterministic (a fixed glider on a
 * toroidal grid) → the differential anchor (browser/tests/reactor.rs).
 *
 *   d = __vm_cap_resolve("display", 7);   __vm_host_call(d, 0, ptr, w, h, 0);  // present(ptr,w,h)
 *
 * One RGBA pixel per cell (the page scales it up); live = amber, dead = near-black. */
extern int __vm_cap_resolve(const char *name, long len);
extern long __vm_host_call(int h, int op, long a, long b, long c, long d);
extern void *calloc(unsigned long n, unsigned long sz);

#define W 96
#define H 64

static unsigned char *cur, *nxt; /* W*H cell grids, on the heap (above the mapped window) */
static unsigned char *fb;        /* W*H*4 RGBA frame, also on the heap */
static int disp;

static int wrap(int v, int n) { return (v % n + n) % n; }
static int at(unsigned char *g, int x, int y) { return g[wrap(y, H) * W + wrap(x, W)]; }

int main(void) {
  disp = __vm_cap_resolve("display", 7);
  cur = calloc(W * H, 1);
  nxt = calloc(W * H, 1);
  fb = calloc((unsigned long)W * H * 4, 1);
  if (!cur || !nxt || !fb) return 1;
  /* Seed a single glider near the top-left — it travels down-right forever on the torus. */
  int gx = 2, gy = 2;
  cur[(gy + 0) * W + (gx + 1)] = 1;
  cur[(gy + 1) * W + (gx + 2)] = 1;
  cur[(gy + 2) * W + (gx + 0)] = 1;
  cur[(gy + 2) * W + (gx + 1)] = 1;
  cur[(gy + 2) * W + (gx + 2)] = 1;
  return 0;
}

int tick(void) {
  /* Next generation from the current grid (the persistence-critical read). */
  for (int y = 0; y < H; y++) {
    for (int x = 0; x < W; x++) {
      int n = at(cur, x - 1, y - 1) + at(cur, x, y - 1) + at(cur, x + 1, y - 1) +
              at(cur, x - 1, y) + at(cur, x + 1, y) +
              at(cur, x - 1, y + 1) + at(cur, x, y + 1) + at(cur, x + 1, y + 1);
      int alive = cur[y * W + x];
      nxt[y * W + x] = (n == 3 || (alive && n == 2)) ? 1 : 0;
    }
  }
  unsigned char *t = cur; cur = nxt; nxt = t; /* swap (pointers live in globals, persisted) */

  for (int i = 0; i < W * H; i++) {
    unsigned char *p = &fb[i * 4];
    if (cur[i]) { p[0] = 255; p[1] = 200; p[2] = 40; }   /* live: amber */
    else        { p[0] = 12;  p[1] = 12;  p[2] = 24; }   /* dead: near-black */
    p[3] = 255;
  }
  __vm_host_call(disp, 0, (long)fb, W, H, 0);
  return 0;
}
