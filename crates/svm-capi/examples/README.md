# svm-capi C examples

`hello.c` is a complete C embedding via [`../include/svm.h`](../include/svm.h): it parses hand-written
IR, binds a built-in `write` (stdout) capability **and** a custom host-defined `meaning` capability by
name, instantiates, runs on the JIT, and reads back the captured stdout and the entry's return value.

## Build & run

From the repo root, build the static library, then compile and link the example against it:

```sh
cargo build -p svm-capi
cc crates/svm-capi/examples/hello.c \
   -I crates/svm-capi/include \
   target/debug/libsvm_capi.a \
   -lpthread -ldl -lm \
   -o /tmp/svm_hello
/tmp/svm_hello
```

Expected output:

```
guest stdout: Hello from C!
entry returned: 42
OK
```

Notes:
- Pass the `.a` path **directly** (not `-L target/debug -lsvm_capi`) so the linker takes the static
  library rather than the `.so` (which would need `LD_LIBRARY_PATH` at run time).
- The trailing `-lpthread -ldl -lm` are the system libraries the Rust staticlib needs on Linux; on
  macOS they are unnecessary (replace with nothing, and the `.a` is `libsvm_capi.a` all the same).
- For a release build use `cargo build -p svm-capi --release` and `target/release/libsvm_capi.a`.

The portable ABI test (`cargo test -p svm-capi`) exercises the same surface from Rust on every CI
platform; this C example is the human-facing proof of real C linkage.
