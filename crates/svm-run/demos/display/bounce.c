/* Interactive framebuffer demo (Doom slice 2): a bouncing box you steer with the arrow keys. This is
 * the first guest to drive the *reactor* run model — the host calls the exported `tick()` once per
 * frame (not `main()` to completion), and per-frame state lives in globals that persist across calls.
 * Each tick drains the `keyboard` capability, moves the box (bouncing off the walls), renders an RGBA
 * frame, and presents it through the `display` capability; the browser blits it to a <canvas> in a
 * requestAnimationFrame loop. The same two waists Doom needs — a framebuffer out, key events in.
 *
 *   d = __vm_cap_resolve("display", 7);   __vm_host_call(d, 0, ptr, w, h, 0);  // present(ptr,w,h)
 *   k = __vm_cap_resolve("keyboard", 8);  e = __vm_host_call(k, 0, 0,0,0,0);   // poll() → event | -1
 *
 * A key event is packed `(pressed << 16) | keycode` (pressed 1=down/0=up); `poll` returns -1 when the
 * queue is empty. Keycodes are JS `keyCode`s: Left 37, Up 38, Right 39, Down 40. The motion is a pure
 * function of the initial state + the key events, so a fixed input script yields a deterministic frame
 * sequence — the differential anchor (browser/tests/reactor.rs). */
extern int __vm_cap_resolve(const char *name, long len);
extern long __vm_host_call(int h, int op, long a, long b, long c, long d);

#define W 160
#define H 120
#define BOX 8
#define SPEED 2

static unsigned char fb[W * H * 4];
static int disp, kbd;    /* resolved capability handles (stashed once, in _start → main) */
static int bx, by;       /* box top-left, in pixels */
static int vx, vy;       /* velocity */

/* Fill one pixel (x,y) with (r,g,b), opaque. */
static void put(int x, int y, int r, int g, int b) {
  unsigned char *p = &fb[(y * W + x) * 4];
  p[0] = (unsigned char)r; p[1] = (unsigned char)g; p[2] = (unsigned char)b; p[3] = 255;
}

int main(void) {
  disp = __vm_cap_resolve("display", 7);
  kbd = __vm_cap_resolve("keyboard", 8);
  bx = (W - BOX) / 2; by = (H - BOX) / 2; /* centered */
  vx = SPEED; vy = SPEED;                 /* moving down-right to start */
  return 0;
}

/* One frame: drain input, step the box (bounce off walls), draw, present. */
int tick(void) {
  for (;;) {
    long e = __vm_host_call(kbd, 0, 0, 0, 0, 0);
    if (e < 0) break;                     /* queue empty */
    if ((e >> 16) & 1) {                  /* key-down: steer */
      int code = (int)(e & 0xffff);
      if (code == 37) vx = -SPEED;        /* Left  */
      else if (code == 39) vx = SPEED;    /* Right */
      else if (code == 38) vy = -SPEED;   /* Up    */
      else if (code == 40) vy = SPEED;    /* Down  */
    }
  }

  bx += vx; by += vy;
  if (bx < 0) { bx = 0; vx = -vx; }
  if (bx > W - BOX) { bx = W - BOX; vx = -vx; }
  if (by < 0) { by = 0; vy = -vy; }
  if (by > H - BOX) { by = H - BOX; vy = -vy; }

  for (int y = 0; y < H; y++)             /* clear to dark blue */
    for (int x = 0; x < W; x++)
      put(x, y, 16, 16, 40);
  for (int y = 0; y < BOX; y++)           /* draw the box, bright amber */
    for (int x = 0; x < BOX; x++)
      put(bx + x, by + y, 255, 220, 40);

  __vm_host_call(disp, 0, (long)fb, W, H, 0);
  return 0;                               /* keep going */
}
