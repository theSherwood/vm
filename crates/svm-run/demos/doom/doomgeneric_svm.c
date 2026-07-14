/* doomgeneric platform layer for the SVM sandbox (Doom slice 3b/4).
 *
 * doomgeneric isolates all platform code to the six DG_* hooks below + the DG_ScreenBuffer it draws
 * into. This maps them onto the on-ramp capabilities the interactive playground already exposes:
 *   - DG_DrawFrame  → the `display` capability (present an RGBA frame; the page blits it to a canvas)
 *   - DG_GetKey     → the `keyboard` capability (poll packed key events; mapped to Doom key codes)
 *   - DG_GetTicksMs → a DETERMINISTIC frame clock (fixed ms per frame), so the headless differential
 *                     against a native `cc` build is reproducible frame-for-frame (no wall-clock)
 *   - DG_SleepMs    → no-op (the host — the reactor / requestAnimationFrame loop — paces frames)
 *   - DG_Init / DG_SetWindowTitle → no-ops
 *
 * The reactor run model (slice 2/3a) drives this: `main()` runs `doomgeneric_Create` once (reads the
 * WAD through the `fs` capability), then the exported `tick()` calls `doomgeneric_Tick()` once per
 * frame over the persistent guest window (globals/BSS + the multi-MB zone heap all survive — slice 3a).
 */
#include "doomgeneric.h"
#include "doomkeys.h"

extern int __vm_cap_resolve(const char *name, long len);
extern long __vm_host_call(int h, int op, long a, long b, long c, long d);

/* DG_ScreenBuffer is DOOMGENERIC_RESX*RESY XRGB pixels (0x00RRGGBB in a uint32). The canvas wants
 * RGBA bytes, so we swizzle into this side buffer once per frame before presenting. */
#define NPIX (DOOMGENERIC_RESX * DOOMGENERIC_RESY)
static unsigned char rgba[NPIX * 4];

static int g_disp = -1;
static int g_kbd = -1;
static unsigned int g_ticks = 0; /* deterministic ms clock: advanced a fixed step per frame */

#define MS_PER_FRAME 28 /* ~35.7 fps — Doom's native tic rate is 35 Hz; fixed for reproducibility */

void DG_Init(void) {
  g_disp = __vm_cap_resolve("display", 7);
  g_kbd = __vm_cap_resolve("keyboard", 8);
}

void DG_DrawFrame(void) {
  for (int i = 0; i < NPIX; i++) {
    unsigned int p = DG_ScreenBuffer[i]; /* 0x00RRGGBB */
    unsigned char *o = &rgba[i * 4];
    o[0] = (unsigned char)(p >> 16); /* R */
    o[1] = (unsigned char)(p >> 8);  /* G */
    o[2] = (unsigned char)(p);       /* B */
    o[3] = 255;                      /* A */
  }
  if (g_disp >= 0)
    __vm_host_call(g_disp, 0, (long)rgba, DOOMGENERIC_RESX, DOOMGENERIC_RESY, 0);
}

void DG_SleepMs(uint32_t ms) { (void)ms; /* the host paces frames */ }

uint32_t DG_GetTicksMs(void) { return g_ticks; }

/* Map a browser keyCode (the `keyboard` cap's event codes) to a Doom key. Covers movement, fire, use,
 * enter/escape/tab and space — enough to walk through the shareware demo and menus. */
static unsigned char map_key(int code) {
  switch (code) {
    case 37: return KEY_LEFTARROW;   /* ArrowLeft  */
    case 38: return KEY_UPARROW;     /* ArrowUp    */
    case 39: return KEY_RIGHTARROW;  /* ArrowRight */
    case 40: return KEY_DOWNARROW;   /* ArrowDown  */
    case 17: return KEY_FIRE;        /* Ctrl       */
    case 32: return KEY_USE;         /* Space      */
    case 13: return KEY_ENTER;       /* Enter      */
    case 27: return KEY_ESCAPE;      /* Escape     */
    case 9:  return KEY_TAB;         /* Tab        */
    case 16: return KEY_RSHIFT;      /* Shift (run) */
    default:
      if (code >= 65 && code <= 90) return (unsigned char)(code + 32); /* A-Z → lowercase ascii */
      return (unsigned char)code;
  }
}

/* doomgeneric pumps this until it returns 0. Each call dequeues one event from the keyboard cap:
 * poll() returns a packed (pressed<<16)|keycode, or -1 when the queue is empty. Advance the frame
 * clock here too (once per DG_GetKey pump cycle would be wrong; we advance it in tick() — see main). */
int DG_GetKey(int *pressed, unsigned char *key) {
  if (g_kbd < 0) return 0;
  long e = __vm_host_call(g_kbd, 0, 0, 0, 0, 0);
  if (e < 0) return 0; /* queue empty */
  *pressed = (int)((e >> 16) & 1);
  *key = map_key((int)(e & 0xffff));
  return 1;
}

void DG_SetWindowTitle(const char *title) { (void)title; }

/* Called once per frame by the reactor's tick() (see main.c) to advance the deterministic clock. */
void DG_SVM_AdvanceClock(void) { g_ticks += MS_PER_FRAME; }
