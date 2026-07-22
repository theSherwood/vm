//! Cross-engine micro-benchmark for the SVM backends — **tree-walker**, **bytecode engine**, and
//! **JIT** — over the same compute kernels, with per-iteration compute isolated by large/small-`n`
//! subtraction and taken as the min over repetitions (the methodology of `src/bin/bench.rs`). It is an
//! *example* (not the `svm-bench` binary) because the JIT lives in `svm-jit`, a dev-dependency.
//!
//! Output is machine-readable CSV on stdout — `engine,kernel,ns_per_iter` — so an external driver can
//! merge it with native / wasm / python numbers into one table. Run:
//!   cargo run --release --example megabench -p svm

use std::time::Instant;

use svm::{ir, text};
use svm_interp::{bytecode, Value};

fn main() {
    // `chase` (16 KiB, L1, constant-stride) and `chase_rand` (4 MiB, LCG permutation) are
    // dependent-load pointer chases — each load's address is the previous load's value, so the
    // access can't be forwarded/hoisted/vectorized (unlike `mem`, which every compiler deletes).
    let chase = chase_src(16, 4096, false); // memory 2^16 = 64 KiB window holds the 16 KiB array
    let chase_rand = chase_src(22, 1 << 20, true); // memory 2^22 = 4 MiB window holds the 4 MiB array
    let kernels: [(&str, &str, i32, i32); 9] = [
        ("alu", ALU, 1_000, 201_000),
        ("call", CALL, 1_000, 201_000),
        ("call_indirect", CALL_INDIRECT, 1_000, 201_000),
        ("mem", MEM, 1_000, 201_000),
        ("chase", &chase, 1_000, 201_000),
        ("chase_rand", &chase_rand, 1_000, 201_000),
        ("fnv", FNV, 1_000, 201_000),
        ("fma", FMA, 1_000, 201_000),
        ("vsum", VSUM, 1_000, 201_000),
    ];
    for (name, src, small, large) in kernels {
        let m = text::parse_module(src).expect("kernel parses");

        let tw = per_iter(small, large, |n| {
            let mut fuel = u64::MAX;
            let r = svm_interp::run(&m, 0, &[Value::I32(n)], &mut fuel);
            std::hint::black_box(&r);
        });
        println!("svm-tree-walk,{name},{tw:.4}");

        let bc = per_iter(small, large, |n| {
            let mut fuel = u64::MAX;
            let r = bytecode::compile_and_run(&m, 0, &[Value::I32(n)], &mut fuel)
                .expect("bytecode drives the kernel");
            std::hint::black_box(&r);
        });
        println!("svm-bytecode,{name},{bc:.4}");

        let jit = per_iter(small, large, |n| {
            let r = svm_jit::compile_and_run(&m, 0, &[n as i64]).expect("jit compiles + runs");
            std::hint::black_box(&r);
        });
        println!("svm-jit,{name},{jit:.4}");
    }
}

/// Per-iteration compute (ns) for `run_one(n)`, isolated by large/small-`n` subtraction and taken as
/// the min over repetitions (compute is deterministic; min rejects scheduler/noise spikes).
fn per_iter(small: i32, large: i32, run_one: impl Fn(i32)) -> f64 {
    let t_small = min_run(small, &run_one);
    let t_large = min_run(large, &run_one);
    (t_large - t_small) / (large - small) as f64
}

fn min_run(n: i32, run_one: &impl Fn(i32)) -> f64 {
    run_one(n); // warm up (the JIT's compile, the caches)
    let reps = 25;
    let mut best = f64::MAX;
    for _ in 0..reps {
        let start = Instant::now();
        run_one(n);
        best = best.min(start.elapsed().as_nanos() as f64);
    }
    best
}

// The kernels mirror `src/bin/bench.rs` exactly (so the SVM numbers are comparable run-to-run), and
// the external C / wasm / python drivers replicate the *same* computation.

/// `acc += n; n -= 1` until zero — a pure scalar/branch recurrence (sum 1..n, i32).
const ALU: &str = r#"
func (i32) -> (i32) {
block 0 (v0: i32) {
  v1 = i32.const 0
  br 1(v0, v1)
}
block 1 (v2: i32, v3: i32) {
  v4 = i32.add v3 v2
  v5 = i32.const 1
  v6 = i32.sub v2 v5
  br_if v6 1(v6, v4) 2(v4)
}
block 2 (v7: i32) {
  return v7
  }
}
"#;

/// Each iteration calls a leaf `+1` function — the call/return kernel (window open/close cost).
const CALL: &str = r#"
func (i32) -> (i32) {
block 0 (v0: i32) {
  v1 = i32.const 0
  br 1(v0, v1)
}
block 1 (v2: i32, v3: i32) {
  v4 = call 1(v3)
  v5 = i32.const 1
  v6 = i32.sub v2 v5
  br_if v6 1(v6, v4) 2(v4)
}
block 2 (v7: i32) {
  return v7
  }
}
func (i32) -> (i32) {
block 0 (v0: i32) {
  v1 = i32.const 1
  v2 = i32.add v0 v1
  return v2
  }
}
"#;

/// Each iteration dispatches through the `call_indirect` table — mask + slot read + type-check.
const CALL_INDIRECT: &str = r#"
func (i32) -> (i32) {
block 0 (v0: i32) {
  v1 = i32.const 0
  br 1(v0, v1)
}
block 1 (v2: i32, v3: i32) {
  v4 = i32.const 1
  v5 = call_indirect (i32) -> (i32) v4 (v3)
  v6 = i32.const 1
  v7 = i32.sub v2 v6
  br_if v7 1(v7, v5) 2(v5)
}
block 2 (v8: i32) {
  return v8
  }
}
func (i32) -> (i32) {
block 0 (v0: i32) {
  v1 = i32.const 1
  v2 = i32.add v0 v1
  return v2
  }
}
"#;

/// Each iteration does one `i32.store` + one `i32.load` at a fixed address — the memory kernel.
const MEM: &str = r#"memory 16
func (i32) -> (i32) {
block 0 (v0: i32) {
  v1 = i32.const 0
  br 1(v0, v1)
}
block 1 (v2: i32, v3: i32) {
  v4 = i64.const 0
  i32.store v4 v3
  v5 = i32.load v4
  v6 = i32.const 1
  v7 = i32.add v5 v6
  v8 = i32.const 1
  v9 = i32.sub v2 v8
  br_if v9 1(v9, v7) 2(v7)
}
block 2 (v10: i32) {
  return v10
  }
}
"#;

/// Generate a dependent-load **pointer-chase** kernel (`func (i32 n) -> (i64)`): rebuild a chain of
/// `size` i32 slots in linear memory (a fixed prelude that cancels in the large/small-`n` subtraction),
/// then chase it `n` times — `idx = mem[idx*4]` — accumulating the visited indices. Because each load's
/// address is the previous load's value, the access can't be forwarded, hoisted, or vectorized (mirrors
/// `bench/cross-engine/kernels.c`). `lcg=false` builds a constant-stride cycle (prefetcher-friendly →
/// load-issue path); `lcg=true` builds a full-period LCG permutation (prefetcher-defeating → cache
/// latency). `mem_log2` sizes the window to hold the `size*4`-byte array.
fn chase_src(mem_log2: u32, size: u32, lcg: bool) -> String {
    let mask = size - 1;
    // next = the value stored at slot `vi` (the slot it points to).
    let next = if lcg {
        // (vi * 1103515245 + 12345) & mask  — Hull-Dobell full-period permutation mod 2^k.
        "  vmul = i32.const 1103515245\n  vinc = i32.const 12345\n  \
         vm = i32.mul vi vmul\n  va = i32.add vm vinc\n  vnext = i32.and va vmaskc\n"
    } else {
        // (vi + 1789) & mask  — a constant-stride cycle.
        "  vstride = i32.const 1789\n  va = i32.add vi vstride\n  vnext = i32.and va vmaskc\n"
    };
    format!(
        "memory {mem_log2}
func (i32) -> (i64) {{
block 0 (v0: i32) {{
  vi0 = i32.const 0
  vrem0 = i32.const {size}
  br binit(vi0, vrem0, v0)
binit(vi: i32, vrem: i32, vn: i32):
  vfour = i64.const 4
  vidx64 = i64.extend_i32_u vi
  vaddr = i64.mul vidx64 vfour
  vmaskc = i32.const {mask}
{next}  i32.store vaddr vnext
  vone = i32.const 1
  vi2 = i32.add vi vone
  vrem2 = i32.sub vrem vone
  vzero = i32.const 0
  vhops0 = i64.const 0
  br_if vrem2 binit(vi2, vrem2, vn) bchase(vzero, vhops0, vn)
bchase(vidx: i32, vhops: i64, vk: i32):
  vfour2 = i64.const 4
  vc64 = i64.extend_i32_u vidx
  vcaddr = i64.mul vc64 vfour2
  vloaded = i32.load vcaddr
  vle = i64.extend_i32_u vloaded
  vhops2 = i64.add vhops vle
  vkone = i32.const 1
  vk2 = i32.sub vk vkone
  br_if vk2 bchase(vloaded, vhops2, vk2) bret(vhops2)
bret(vh: i64):
  return vh
  }}
}}
"
    )
}

/// Realistic composite: **FNV-1a-32** over a 4 KiB byte buffer, hashing `n` bytes (wrapping). The hash
/// chain is serial (`h` feeds the next iteration) so it can't be vectorized or closed-form-folded —
/// byte-load + xor + mul + branch, like a real hashing inner loop. The buffer is rebuilt inside (a
/// fixed prelude that cancels in the subtraction).
const FNV: &str = r#"memory 16
func (i32) -> (i32) {
block 0 (v0: i32) {
  fi0 = i32.const 0
  frem0 = i32.const 4096
  br finit(fi0, frem0, v0)
finit(fi: i32, frem: i32, fcount: i32):
  faddr = i64.extend_i32_u fi
  fseven = i32.const 7
  fm = i32.mul fi fseven
  fone = i32.const 1
  fp = i32.add fm fone
  fff = i32.const 255
  fb = i32.and fp fff
  i32.store8 faddr fb
  fi2 = i32.add fi fone
  frem2 = i32.sub frem fone
  fbasis = i32.const 2166136261
  br_if frem2 finit(fi2, frem2, fcount) fhash(fcount, fbasis)
fhash(hrem: i32, hh: i32):
  hmask = i32.const 4095
  hidx = i32.and hrem hmask
  haddr = i64.extend_i32_u hidx
  hbyte = i32.load8_u haddr
  hxor = i32.xor hh hbyte
  hprime = i32.const 16777619
  hh2 = i32.mul hxor hprime
  hone = i32.const 1
  hrem3 = i32.sub hrem hone
  br_if hrem3 fhash(hrem3, hh2) fret(hh2)
fret(hf: i32):
  return hf
  }
}
"#;

/// Float: a scalar **f64 FMA recurrence** `acc = acc*C + D`, `n` times. A serial floating-point
/// dependency (latency-bound) — not vectorizable, not closed-form-foldable (FP isn't reassociated).
/// Returns `trunc(acc)` so every backend returns an `i32` (no f64-return plumbing in the drivers).
const FMA: &str = r#"
func (i32) -> (i32) {
block 0 (v0: i32) {
  pacc0 = f64.const 1.0
  br ploop(v0, pacc0)
ploop(pk: i32, pacc: f64):
  pc = f64.const 0.9999999
  pd = f64.const 1.0
  pmul = f64.mul pacc pc
  pacc2 = f64.add pmul pd
  pone = i32.const 1
  pk2 = i32.sub pk pone
  br_if pk2 ploop(pk2, pacc2) pdone(pacc2)
pdone(paccf: f64):
  pr = i32.trunc_f64_s paccf
  return pr
  }
}
"#;

/// SIMD probe: a contiguous **i32 reduction** `sum += arr[k]` over `n` elements of a 1 MiB array. The
/// access is a clean affine sweep, so an auto-vectorizing backend (native AVX, wasm SIMD) collapses it
/// to vector adds, while a scalar backend (the SVM JIT's Cranelift, and the interpreters) stays scalar
/// — exposing the vectorization gap. Array rebuilt inside (cancels in the subtraction).
const VSUM: &str = r#"memory 20
func (i32) -> (i32) {
block 0 (v0: i32) {
  si0 = i32.const 0
  srem0 = i32.const 262144
  br vinit(si0, srem0, v0)
vinit(vi: i32, vrem: i32, vn: i32):
  vfour = i64.const 4
  vi64 = i64.extend_i32_u vi
  vaddr = i64.mul vi64 vfour
  vone = i32.const 1
  vval = i32.add vi vone
  i32.store vaddr vval
  vi2 = i32.add vi vone
  vrem2 = i32.sub vrem vone
  szero = i32.const 0
  br_if vrem2 vinit(vi2, vrem2, vn) vsumloop(szero, szero, vn)
vsumloop(vk: i32, vsum: i32, vc: i32):
  vfour2 = i64.const 4
  vk64 = i64.extend_i32_u vk
  vaddr2 = i64.mul vk64 vfour2
  vld = i32.load vaddr2
  vsum2 = i32.add vsum vld
  vsone = i32.const 1
  vk2 = i32.add vk vsone
  vsrem = i32.sub vc vk2
  br_if vsrem vsumloop(vk2, vsum2, vc) vsret(vsum2)
vsret(vsf: i32):
  return vsf
  }
}
"#;

// Keep `ir` referenced (the parser returns `ir::Module`) without an unused-import warning if the
// signature ever changes — a no-op the optimizer drops.
#[allow(dead_code)]
fn _ir_ref(_m: &ir::Module) {}
