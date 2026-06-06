/* Shakedown driver: run kokke/tiny-regex-c in the sandbox.
 *
 * tiny-regex-c is a Rob-Pike-style backtracking matcher: re_match recurses through
 * matchpattern -> matchstar/matchplus/matchquestion -> matchpattern, retrying on
 * failure. That recursion-with-backtracking is a new control-flow shape for the
 * shakedown series (the data-stack threading + general goto/branch lowering get a
 * workout), distinct from the earlier integer/struct/float libraries.
 *
 * We compile it freestanding (RE_FREESTANDING drops the libc <stdio.h>/<ctype.h>
 * includes and the printf-only re_print debug helper), provide the three ctype
 * predicates it uses, and run a table of (pattern, text) cases — printing the match
 * index and length for each. svm-run's output must match a native cc build. */

#include <stddef.h>

/* The library needs only these three ctype predicates (matchdigit/alpha/whitespace). */
int isdigit(int c) { return c >= '0' && c <= '9'; }
int isalpha(int c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
int isspace(int c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

#define RE_FREESTANDING
#include "re.c"

int write(int fd, char *buf, long n);

static void puts_(const char *s) {
  int n = 0;
  while (s[n]) n++;
  write(1, (char *)s, n);
}
static void puti(int v) {
  char buf[16];
  int i = sizeof(buf);
  unsigned u = v < 0 ? (unsigned)(-v) : (unsigned)v;
  if (u == 0) buf[--i] = '0';
  while (u) {
    buf[--i] = (char)('0' + u % 10);
    u /= 10;
  }
  if (v < 0) buf[--i] = '-';
  write(1, &buf[i], (long)(sizeof(buf) - i));
}

static void run(const char *pat, const char *text) {
  int len = 0;
  int idx = re_match(pat, text, &len);
  puts_("re_match(\"");
  puts_(pat);
  puts_("\", \"");
  puts_(text);
  puts_("\") -> idx=");
  puti(idx);
  puts_(" len=");
  puti(len);
  puts_("\n");
}

int main(void) {
  run("\\d+", "abc 12345 xyz");           /* digits run */
  run("[a-z]+", "  Hello World");          /* lowercase run, skips caps/space */
  run("^\\w+", "hello_world!");            /* anchored word */
  run("\\s", "no_spaces_here");            /* no match -> idx=-1 */
  run("a.c", "xxabcyy");                   /* dot */
  run("colou?r", "my favorite colour");    /* optional */
  run("[0-9]+\\.[0-9]+", "pi is 3.14159"); /* escaped dot, ranges */
  run("^[A-Z][a-z]*$", "Title");           /* both anchors */
  run("ab*c", "ac");                        /* star, zero reps */
  run("ab+c", "ac");                        /* plus, needs >=1 -> -1 */
  run("\\w+@\\w+", "send to user@host ok"); /* a tiny email-ish pattern */
  run(".*", "anything");                    /* greedy dot-star */
  return 0;
}
