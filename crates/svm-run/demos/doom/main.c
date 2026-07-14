/* Doom guest entry for the reactor run model (Doom slice 3b/4).
 *
 * `main()` runs `doomgeneric_Create` once at `_start` — it reads the WAD through the `fs` capability
 * and builds the game state in the zone heap. Then the reactor calls the exported `tick()` once per
 * frame; the persistent guest window (slice 3a) keeps the multi-MB zone across frames. This is the
 * same shape as `bounce`/`life`, scaled up to a real game. */
#include "doomgeneric.h"

extern void DG_SVM_AdvanceClock(void);

/* argv for a shareware run: the `fs` capability serves `doom1.wad`. */
static char a0[] = "doom";
static char a1[] = "-iwad";
static char a2[] = "doom1.wad";
static char *g_argv[] = {a0, a1, a2};

int main(void) {
  doomgeneric_Create(3, g_argv);
  return 0;
}

int tick(void) {
  DG_SVM_AdvanceClock();
  doomgeneric_Tick();
  return 0;
}
