/* os_probe.c — a deterministic exerciser for the guest OS-shim (slice CA).
 *
 * One source, two builds (the SQLite-Phase-B pattern):
 *  - **guest** (`-DSVM_GUEST`): `#include`s `os_shim.c`, so every POSIX call below is bridged to the
 *    `fs` capability; translated and run in the sandbox under a granted `fs` cap.
 *  - **native oracle**: plain `cc`, using the real glibc against a real temp directory.
 * Same sequence → byte-identical stdout, which is the whole test: the shim reproduces libc's
 * file/directory semantics over the capability.
 *
 * Everything printed is deterministic across runs and platforms: rc values, `stat` type bits (the
 * `S_IF*` masks are the same ABI on the reference host), sizes, and — crucially — directory entries
 * are **sorted** and the `.`/`..` entries filtered, so the arbitrary `readdir` order of the native
 * unix backend matches the cap's (already sorted) order.
 */

#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef SVM_GUEST
#include "os_shim.c"
#endif

int main(void) {
  char buf[128];
  struct stat st;
  int fd;

  printf("mkdir d=%d\n", mkdir("d", 0755));
  printf("mkdir d again=%d\n", mkdir("d", 0755)); /* EEXIST → -1 */

  fd = open("d/f", O_CREAT | O_WRONLY | O_TRUNC, 0644);
  printf("open ok=%d\n", fd >= 0);
  printf("write=%d\n", (int)write(fd, "hello world", 11));
  close(fd);

  /* Each stat() is its own statement: C leaves argument-evaluation order unspecified, so reading
   * `st` in the same printf arg list that calls stat() would race the fill. */
  int rc = stat("d/f", &st);
  printf("stat f=%d mode=%d size=%d\n", rc, (int)(st.st_mode & S_IFMT), (int)st.st_size);
  rc = stat("d", &st);
  printf("stat d=%d isdir=%d\n", rc, (int)((st.st_mode & S_IFMT) == S_IFDIR));
  printf("stat nope=%d\n", stat("d/nope", &st));

  fd = open("d/f", O_RDONLY, 0);
  int n = (int)read(fd, buf, 5);
  buf[n] = 0;
  printf("read5=%s\n", buf);
  n = (int)pread(fd, buf, 5, 6);
  buf[n] = 0;
  printf("pread6=%s\n", buf);
  fstat(fd, &st);
  printf("fstat size=%d\n", (int)st.st_size);
  close(fd);

  printf("access f=%d none=%d\n", access("d/f", 0), access("d/none", 0));

  /* three more files, then a sorted, dotfile-filtered directory walk */
  for (int i = 0; i < 3; i++) {
    char p[8];
    p[0] = 'd';
    p[1] = '/';
    p[2] = 'g';
    p[3] = (char)('0' + i);
    p[4] = 0;
    fd = open(p, O_CREAT | O_WRONLY, 0644);
    close(fd);
  }
  DIR *dp = opendir("d");
  char names[64][64];
  int cnt = 0;
  struct dirent *de;
  while ((de = readdir(dp))) {
    if (!strcmp(de->d_name, ".") || !strcmp(de->d_name, "..")) continue;
    if (cnt < 64) {
      int j = 0;
      while (de->d_name[j] && j < 63) {
        names[cnt][j] = de->d_name[j];
        j++;
      }
      names[cnt][j] = 0;
      cnt++;
    }
  }
  closedir(dp);
  /* insertion sort (avoid depending on qsort) */
  for (int i = 1; i < cnt; i++) {
    char tmp[64];
    strcpy(tmp, names[i]);
    int j = i - 1;
    while (j >= 0 && strcmp(names[j], tmp) > 0) {
      strcpy(names[j + 1], names[j]);
      j--;
    }
    strcpy(names[j + 1], tmp);
  }
  printf("dir count=%d:", cnt);
  for (int i = 0; i < cnt; i++) printf(" %s", names[i]);
  printf("\n");

  printf("rename=%d\n", rename("d/f", "d/f2"));
  printf("unlink=%d\n", unlink("d/f2"));
  printf("rmdir nonempty=%d\n", rmdir("d")); /* g0..g2 remain → -1 */
  unlink("d/g0");
  unlink("d/g1");
  unlink("d/g2");
  printf("rmdir empty=%d\n", rmdir("d"));
  printf("stat gone=%d\n", stat("d", &st));

  fd = open("t", O_CREAT | O_RDWR | O_TRUNC, 0644);
  write(fd, "0123456789", 10);
  printf("ftruncate=%d\n", ftruncate(fd, 4));
  fstat(fd, &st);
  printf("trunc size=%d\n", (int)st.st_size);
  close(fd);
  unlink("t");

  return 0;
}
