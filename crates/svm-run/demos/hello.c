// The same greeting as hello.svm, but in C — compiled through the chibicc frontend and run
// sandboxed:  svm-run crates/svm-run/demos/hello.c
//
// `write` is a powerbox builtin (the Stream capability, §3e); the frontend lowers it to a
// `cap.call` on the granted stdout handle.

int write(int fd, char *buf, long n);

int main(void) {
  char *msg = "hello, sandbox!\n";
  long n = 0;
  while (msg[n]) n++;
  write(1, msg, n);
  return 0;
}
