# jsmn JSON tokenizer demo

`jsmn.h` is [jsmn](https://github.com/zserge/jsmn) by Serge Zaitsev (**MIT license** — see the
header block at the top of `jsmn.h`), a minimal zero-allocation JSON tokenizer, vendored
unmodified as a real-world C shakedown target and demo.

`jsmn_demo.c` tokenizes a JSON string and prints each token's type, child count, and source
span. Run it sandboxed:

```sh
cargo run -p svm-run -- crates/svm-run/demos/jsmn/jsmn_demo.c
```

A different shape from the Clay demo (pure char/state-machine string scanning, no structs-as-
geometry, no allocations) — it ran identically to a native build with no new fixes needed,
exercising string handling, escapes/unicode, nesting, and the error-code paths end-to-end.
