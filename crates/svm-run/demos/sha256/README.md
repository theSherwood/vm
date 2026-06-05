# SHA-256 demo

`sha256.c` / `sha256.h` are from Brad Conte's [crypto-algorithms](https://github.com/B-Con/crypto-algorithms)
(**public domain**), a classic compact SHA-256. `sha256.c` is vendored with two vestigial
includes removed — `<stdlib.h>` (unused) and the legacy `<memory.h>` (only declared `memset`,
which `sha_demo.c` provides) — so the demo is self-contained (no libc), and the algorithm is
otherwise unmodified.

`sha_demo.c` hashes a few strings and prints their hex digests. Run it sandboxed:

```sh
cargo run -p svm-run -- crates/svm-run/demos/sha256/sha_demo.c
```

A pure integer/bit-manipulation shape (32-bit wrapping arithmetic, rotates synthesized from
shifts, a constant round-key table) — output matches a native build against the standard test
vectors. The shakedown turned a `func_index` null-token crash on an undefined-function call
into a clean error.
