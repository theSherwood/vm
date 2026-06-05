# xxHash demo

`xxhash.h` is [xxHash](https://github.com/Cyan4973/xxHash) by Yann Collet (**BSD-2-Clause** —
see the header block in `xxhash.h`), vendored unmodified. `xxh_demo.c` configures it for a
self-contained scalar build — `XXH_INLINE_ALL`, `XXH_NO_XXH3` (skip the SIMD path),
`XXH_NO_STREAM` (skip the malloc-using streaming API), `XXH_VECTOR XXH_SCALAR` — and provides the
`memcpy`/`memset` it uses, then prints XXH32/XXH64 of a few strings.

```sh
cargo run -p svm-run -- crates/svm-run/demos/xxhash/xxh_demo.c
```

Another integer/bit shape (32- and 64-bit multiply/rotate hashing), matching the standard test
vectors and a native build. The shakedown added `_Static_assert` (C11) support to the frontend.
