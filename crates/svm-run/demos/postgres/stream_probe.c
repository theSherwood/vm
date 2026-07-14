/* stream_probe.c — the stdout/stderr FILE* stream path + fs-cap file path coexist (slice CE).
 *
 * Exercises the on-ramp's `__vm_stream_write` fd-dispatch: `fwrite(…, stdout)`/`fputc(…, stdout)`
 * reach the powerbox Stream cap (via os_shim's `write(1, …)`), while a `fopen`'d file flows through
 * the fs cap — two disjoint fd namespaces (streams 0/1/2, files ≥3). The guest byte-matches the
 * native glibc oracle. All stdout output is via the buffered stdio/`printf` surface (glibc flushes
 * at exit in program order), so ordering matches the guest's unbuffered streams.
 */

#include <stdio.h>

#ifdef SVM_GUEST
#include "os_shim.c"
#include "stdio_shim.c"
#endif

int main(void) {
  fwrite("hello stream\n", 1, 13, stdout); /* stdout FILE* → write(1) → Stream cap */
  fputc('A', stdout);
  fputc('\n', stdout);

  FILE *f = fopen("f.txt", "w"); /* a real file → fs cap (fd >= 3) */
  fwrite("filedata", 1, 8, f);
  fclose(f);
  f = fopen("f.txt", "r");
  char buf[32];
  size_t n = fread(buf, 1, sizeof buf, f);
  buf[n] = 0;
  fclose(f);
  printf("file=[%s]\n", buf); /* printf → Stream cap too, after the file round-trip */
  return 0;
}
