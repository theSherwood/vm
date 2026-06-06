# heapgrow demo — a guest that grows its own heap via `map`

A program that **consumes the Memory capability** (§3e/§4) through the ordinary
`#include <stdlib.h>`: the sandbox's shipped guest libc (`frontend/chibicc/include/stdlib.h`)
provides a `malloc`/`free`/`calloc`/`realloc` whose heap grows into the *reserved tail* of the
window on demand — committing pages with the `__vm_map` builtin (`cap.call` on the granted Memory
handle) — instead of a fixed bump region. This is the §1a differentiator ("large/sparse programs
that fight wasm's flat linear memory"), shown end to end, with the sandboxed output byte-identical
to a native `cc` build (which uses the platform `malloc`).

`heapgrow.c` is plain portable C — `#include <stdlib.h>`, then allocate eight 128 KiB int blocks
(1 MiB total, ~16× the 64 KiB initial window), fill/sum/free each, and print the running totals.
Nothing in the source is SVM-specific; the growth happens entirely inside the shipped `malloc`.

```sh
cargo run -p svm-run -- crates/svm-run/demos/heapgrow/heapgrow.c
```

The sandboxed run commits ~1 MiB of reserved-tail pages through the Memory cap (interp page map /
JIT real `mprotect`, the kernel demand-paging the physical backing) and matches a native build
(`demo_heapgrow_matches_native`). It exercises the whole growth stack: powerbox Memory grant →
`malloc` → `__vm_map` builtin → `cap.call` → `MprotectWindow` growth → masked access to the
freshly-committed tail.
