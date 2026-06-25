# Monocypher demo (crypto / 64-bit-carry shakedown)

`monocypher.c` / `monocypher.h` are [Monocypher](https://monocypher.org) 4.0.2 by Loup
Vaillant (**dual-licensed BSD-2-Clause OR CC0-1.0** — public domain; see the license block at
the top of each file), vendored unmodified. Monocypher is a small, self-contained,
constant-time crypto library (BLAKE2b, ChaCha20, Poly1305, X25519, EdDSA, Argon2). It is
fully freestanding — `monocypher.c` includes only its own header, with no libc dependency.

`monocypher_demo.c` provides the one powerbox primitive it uses (`write`) and drives four
headline primitives over fixed inputs:

- **BLAKE2b** — hash a fixed message, print the 64-byte digest.
- **ChaCha20** (djb variant) — encrypt a fixed plaintext, print the ciphertext.
- **Poly1305** — MAC the ciphertext, print the 16-byte tag.
- **X25519** — Alice/Bob ECDH: derive public keys and independently compute the shared secret.

```sh
cargo run -p svm-run -- crates/svm-run/demos/monocypher/monocypher_demo.c
```

This is the corpus's **crypto / 64-bit-arithmetic** shakedown. The earlier integer libraries
(SHA-256, xxHash, crc32) were hash/checksum shaped; Monocypher adds modern AEAD + an elliptic
curve, whose field arithmetic (25.5-bit limbs, `i32 × i32 → i64` products with carry
propagation) and constant-time bit-twiddling are brutal on the 64-bit shift/rotate/multiply
paths — exactly the differential-fuzz coverage that has no real-program coverage otherwise.

Output is hex (no float formatting, so no `%f` rounding risk), matched byte-for-byte against a
native `cc` build. The X25519 section is additionally a **known-answer test**: the two ECDH
shared secrets must be identical (the curve's correctness invariant), and the program's exit
code is the mismatch count — so the elliptic-curve arithmetic is self-validated, not just
differenced against native.
