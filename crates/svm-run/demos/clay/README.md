# Clay layout demo

`clay.h` is the [Clay](https://github.com/nicbarker/clay) UI layout library by Nic Barker
(v0.14, **zlib/libpng license** — see the license block at the end of `clay.h`), vendored
unmodified as a real-world C shakedown target and demo.

`clay_demo.c` builds a small layout and prints the resulting render commands. Run it sandboxed:

```sh
cargo run -p svm-run -- crates/svm-run/demos/clay/clay_demo.c
```

It compiles through the chibicc frontend (`#define CLAY_DISABLE_SIMD`) to ~93k lines of our IR,
verifies, and runs on the JIT — output identical to a native build. This exercised and fixed
several frontend/IR gaps (see the commits / `HANDOFF.md`).
