/* Headless frame-hash differential platform (Doom slice 3c). Doom's renderer is pure fixed-point, so
 * given the same WAD + a deterministic clock + input, the guest (translated through the on-ramp) and a
 * native `cc` build must produce byte-identical framebuffers. This platform makes that observable
 * without a display capability: DG_DrawFrame hashes DG_ScreenBuffer and prints the hash, so the whole
 * run's stdout is a frame-hash stream — compare the guest's to native's (the §18 oracle, the way the
 * SQLite differential compares stdout).
 *
 * Compiled BOTH ways from this one source: as the guest (the libc shim routes fopen→`fs`, printf→the
 * Stream capability) and as a native binary (`cc`, real libc + a real doom1.wad on disk). No `__vm_*`
 * here — it is portable C; the storage/stdout injection happens at the libc boundary. */
#include "doomgeneric.h"

extern int printf(const char *fmt, ...);

#define NPIX (DOOMGENERIC_RESX * DOOMGENERIC_RESY)
#ifndef DOOM_DIFF_FRAMES
#define DOOM_DIFF_FRAMES 8
#endif

/* Deterministic **virtual** ms clock: no wall-clock, so guest and native agree bit-for-bit. Doom's
 * TryRunTics busy-waits (`I_Sleep(1)`) until the clock advances a tic; a frozen clock spins forever,
 * so DG_SleepMs advances the virtual clock — the wait loop makes forward progress by exactly the
 * number of sleeps Doom performs (identical on both builds), and no real time is consulted. */
static unsigned g_ticks;
static int g_frame;

void DG_Init(void) {}
void DG_DrawFrame(void) {
  unsigned h = 2166136261u; /* FNV-1a over the XRGB framebuffer */
  for (int i = 0; i < NPIX; i++) { h ^= DG_ScreenBuffer[i]; h *= 16777619u; }
  printf("frame %d %08x\n", g_frame, h);
}
void DG_SleepMs(uint32_t ms) { g_ticks += ms ? ms : 1; } /* advance the virtual clock (never real sleep) */
uint32_t DG_GetTicksMs(void) { return g_ticks; }
int DG_GetKey(int *pressed, unsigned char *key) { (void)pressed; (void)key; return 0; }
void DG_SetWindowTitle(const char *title) { (void)title; }

static char a0[] = "doom";
static char a1[] = "-iwad";
static char a2[] = "doom1.wad";
static char *g_argv[] = {a0, a1, a2};

int main(void) {
  doomgeneric_Create(3, g_argv);
  for (g_frame = 0; g_frame < DOOM_DIFF_FRAMES; g_frame++) {
    g_ticks += 28; /* nudge ~1 tic (35 Hz) between frames so gameplay advances deterministically */
    doomgeneric_Tick();
  }
  return 0;
}
