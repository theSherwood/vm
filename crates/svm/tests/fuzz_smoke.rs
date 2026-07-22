//! Stable-toolchain smoke fuzzing — the day-one security invariants from AGENTS.md
//! and `DESIGN.md` §2a, runnable without nightly/cargo-fuzz. A real libFuzzer target
//! lives in `fuzz/` (nightly); this catches the same panics in CI on stable.
//!
//! Invariants asserted (the "fail-closed, never crash" property of the escape-TCB):
//!   1. `decode_module` never panics / OOMs / hangs on arbitrary bytes.
//!   2. `verify_module` never panics on any decoded module.
//!   3. A *verified* module never panics the interpreter (bounded by fuel).
//!   4. Binary round-trip is identity on every decodable module; text round-trip is
//!      identity on every verified module (mirrors the `roundtrip` fuzz target).

use svm::default_args;
use svm_encode::{decode_module, encode_module};
use svm_interp::run;
use svm_verify::verify_module;

/// Tiny deterministic PRNG (xorshift64*) — no external deps.
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn byte(&mut self) -> u8 {
        (self.next_u64() & 0xff) as u8
    }
    fn range(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next_u64() % n as u64) as usize
        }
    }
}

fn drive(bytes: &[u8]) {
    // 1. Decode must fail-closed, never panic.
    if let Ok(m) = decode_module(bytes) {
        // 4a. Binary round-trip is identity on every decodable module.
        assert_eq!(
            decode_module(&encode_module(&m)),
            Ok(m.clone()),
            "binary round-trip"
        );
        // 2. Verify must never panic.
        if verify_module(&m).is_ok() {
            // 4b. Text round-trip is identity on every verified module.
            assert_eq!(
                svm_text::parse_module(&svm_text::print_module(&m)),
                Ok(m.clone()),
                "text round-trip"
            );
            // 3. A verified module is safe to interpret (bounded by fuel).
            for (fi, f) in m.funcs.iter().enumerate() {
                let args = default_args(&f.params);
                let mut fuel = 10_000u64;
                let _ = run(&m, fi as u32, &args, &mut fuel);
            }
        }
    }
}

#[test]
fn decode_verify_interp_never_panic_on_random_bytes() {
    let mut rng = Rng(0x9E3779B97F4A7C15);
    for _ in 0..200_000 {
        let len = rng.range(64);
        let mut buf = Vec::with_capacity(len);
        for _ in 0..len {
            buf.push(rng.byte());
        }
        drive(&buf);
    }
}

#[test]
fn mutated_valid_modules_never_panic() {
    // Start from valid encodings and flip bytes — exercises the decoder's recovery
    // paths closer to the valid manifold than pure random noise reaches.
    let seeds = [
        r#"func (i32,i32)->(i32){
block 0 (v0:i32,v1:i32) { v2 = i32.add v0 v1
 return v2 } }"#,
        r#"func ()->(i64){
block 0 () { v0 = i64.const 7
 return v0 } }"#,
    ];
    let mut rng = Rng(0xD1B54A32D192ED03);
    for seed in seeds {
        let m = svm_text::parse_module(seed).expect("seed parses");
        let base = encode_module(&m);
        for _ in 0..50_000 {
            let mut buf = base.clone();
            // Apply 1..=4 random single-byte mutations.
            let muts = 1 + rng.range(4);
            for _ in 0..muts {
                if !buf.is_empty() {
                    let i = rng.range(buf.len());
                    buf[i] = rng.byte();
                }
            }
            drive(&buf);
        }
    }
}
