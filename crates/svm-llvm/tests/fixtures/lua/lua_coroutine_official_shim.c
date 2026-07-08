/* Guest-libc addendum for the debug-library build: `fgets`, referenced only by ldblib.c's interactive
 * `debug.debug()` (never called by coroutine.lua). A no-filesystem stub returning EOF is sufficient. */
#include <stddef.h>
typedef struct FILE FILE;
char *fgets(char *s, int n, FILE *f) { (void)s; (void)n; (void)f; return NULL; }
