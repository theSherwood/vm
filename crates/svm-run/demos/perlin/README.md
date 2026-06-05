# Perlin noise demo (first floating-point shakedown)

`stb_perlin.h` is [stb_perlin](https://github.com/nothings/stb) by Sean Barrett
(**public domain** — see the license block at the end of the header), vendored unmodified.
It is Ken Perlin's improved 3D noise plus the fbm / turbulence / ridge octave variants.

`perlin_demo.c` defines `STB_PERLIN_IMPLEMENTATION`, provides the one libc function the
octave variants use (`fabs` — no libm in the sandbox), and evaluates noise over a small grid.

```sh
cargo run -p svm-run -- crates/svm-run/demos/perlin/perlin_demo.c
```

This is the series' first **floating-point-heavy** shakedown — every earlier library (Clay,
jsmn, SHA-256, xxHash, tinfl) was integer/pointer/struct shaped, so the IR's f32 path had real
differential-fuzz coverage but no real-program coverage. Perlin noise is dense f32 arithmetic:
gradient dot products (`grad[0]*x + grad[1]*y + grad[2]*z`), the quintic ease polynomial, and
trilinear `lerp`s, plus int↔float conversion (`fastfloor`) and multiply/accumulate chains over
octaves.

To get byte-exact parity without depending on float *formatting* (printf `%f` rounding is its
own risk), the driver scales each noise value to a fixed-point integer and prints that — so any
divergence in the actual f32 arithmetic between native `cc` and our JIT would show up directly in
the digits. Output matches a native build byte-for-byte, with no new fixes.
