/* stdio_probe.c — byte-exact differential for the guest file-backed stdio shim (slice CD).
 *
 * Guest (`-DSVM_GUEST`, os_shim.c + stdio_shim.c → the fs cap) vs native glibc over a real file:
 * fopen/fwrite/fputc/ftell → close → reopen → fgets/fgetc/ungetc/fread/feof/fseek/ftell. Console
 * status is printed with the on-ramp's synthesized `printf` (→ stdout stream); the *file* bytes flow
 * through the capability. Runs on both `mem_fs` and `host_fs`.
 */

#include <stdio.h>
#include <string.h>

#ifdef SVM_GUEST
#include "os_shim.c"
#include "stdio_shim.c"
#endif

int main(void) {
  FILE *f = fopen("t.txt", "w");
  printf("fopen_w=%d\n", f != NULL);
  const char *lines = "alpha\nbeta\ngamma\n";
  printf("fwrite=%d\n", (int)fwrite(lines, 1, strlen(lines), f));
  fputc('!', f);
  printf("wtell=%d\n", (int)ftell(f));
  fclose(f);

  f = fopen("t.txt", "r");
  printf("fopen_r=%d\n", f != NULL);
  char buf[64];
  printf("fgets1=%s", fgets(buf, sizeof buf, f)); /* "alpha\n" */
  int c = fgetc(f);
  printf("fgetc=%c\n", c); /* 'b' */
  ungetc(c, f);
  size_t rd = fread(buf, 1, sizeof buf, f);
  buf[rd] = 0;
  printf("fread=%d[%s]\n", (int)rd, buf); /* "beta\ngamma\n!" */
  printf("feof=%d\n", feof(f));

  fseek(f, 0, SEEK_SET);
  printf("tell0=%d\n", (int)ftell(f));
  fseek(f, 6, SEEK_SET);
  printf("at6=%s", fgets(buf, sizeof buf, f)); /* "beta\n" */
  fseek(f, 0, SEEK_END);
  printf("size=%d\n", (int)ftell(f));
  clearerr(f);
  printf("ferr=%d fileno_ok=%d\n", ferror(f), fileno(f) >= 0);
  fclose(f);
  return 0;
}
