//! Structured generator of **well-defined, terminating** LLVM-IR functions for the on-ramp
//! differential harness (`onramp_diff.rs` stable test + `fuzz/onramp_diff` target).
//!
//! It emits a single `define i64 @run()` whose body is a straight-line sequence of ops over a
//! typed SSA value pool, **computing each value's concrete result as it emits** — so the final
//! `ret i64` value is known by construction (the *oracle*), with no separate interpreter to carry
//! its own bugs. The generated program is UB-free by construction (shift amounts `< width`, array
//! indices masked in-bounds, no div, no poison flags), so it must **never trap**, and all three
//! svm backends (tree-walker, bytecode, JIT) must return the oracle value. A divergence is an
//! **on-ramp translation bug** — the I23 class, where every backend agrees with the *wrong* IR
//! (so an interp-vs-JIT differential can't see it; the source-semantics oracle can).
//!
//! Coverage is biased to the translation surface that has bitten us: `getelementptr` with distinct
//! source element types (`i8` vs `[N x i32]`), 2-lane (`<2 x i32>`, packed-i64) *and* 128-bit
//! vector min/max, vector widen/narrow, and integer width conversions.
#![allow(dead_code)]

/// Entropy: consume libFuzzer bytes first (coverage-guided), then a deterministic xorshift so a
/// seed with no bytes is reproducible. Mirrors `svm`'s `irgen::Gen`.
pub struct Gen {
    data: Vec<u8>,
    pos: usize,
    rng: u64,
}

impl Gen {
    pub fn from_bytes(data: &[u8]) -> Gen {
        let mut seed = 0x9e3779b97f4a7c15u64 ^ (data.len() as u64).wrapping_mul(0x100000001b3);
        for &b in data.iter().take(16) {
            seed = seed.wrapping_mul(31).wrapping_add(b as u64);
        }
        Gen {
            data: data.to_vec(),
            pos: 0,
            rng: seed | 1,
        }
    }
    pub fn from_seed(seed: u64) -> Gen {
        Gen {
            data: Vec::new(),
            pos: 0,
            rng: seed | 1,
        }
    }
    fn byte(&mut self) -> u8 {
        if self.pos < self.data.len() {
            let b = self.data[self.pos];
            self.pos += 1;
            b
        } else {
            // xorshift64
            let mut x = self.rng;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.rng = x;
            (x >> 24) as u8
        }
    }
    fn u(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.byte() as usize) % n
        }
    }
    fn u64v(&mut self) -> u64 {
        let mut v = 0u64;
        for _ in 0..8 {
            v = (v << 8) | self.byte() as u64;
        }
        v
    }
    fn boundary_scalar(&mut self, bits: u32) -> u64 {
        // Bias constants toward boundary cases so overflow/sign paths are hit.
        let choices: [u64; 8] = [0, 1, u64::MAX, 2, 7, 0x8000_0000, 0x7fff_ffff, self.u64v()];
        mask(choices[self.u(8)], bits)
    }
}

/// Mask a raw value to `bits` (its canonical unsigned image).
fn mask(v: u64, bits: u32) -> u64 {
    if bits >= 64 {
        v
    } else {
        v & ((1u64 << bits) - 1)
    }
}
/// The signed `i64` value of a `bits`-wide raw pattern.
fn sext(v: u64, bits: u32) -> i64 {
    if bits >= 64 {
        v as i64
    } else {
        ((v << (64 - bits)) as i64) >> (64 - bits)
    }
}

/// A vector shape the generator uses. `lanes * lane_bits <= 128` for the native-v128 shapes;
/// `<2 x i64>` and `<2 x i32>` also cover the packed-i64 2-lane path.
#[derive(Clone, Copy, PartialEq)]
struct Shape {
    lanes: u32,
    lane_bits: u32,
}

#[derive(Clone)]
enum Kind {
    Scalar { bits: u32, raw: u64 },
    Vec { shape: Shape, lanes: Vec<u64> },
}

#[derive(Clone)]
struct Val {
    name: String,
    kind: Kind,
}

pub struct Prog {
    body: String,
    pool: Vec<Val>,
    n: usize,
    /// Shadow of the `[16 x i32]` stack array (concrete bytes as u32 lanes).
    mem: [u32; 16],
    have_mem: bool,
    /// Contents of the module-level `@g = constant [16 x i32]` (known to the oracle), read via
    /// **constexpr** GEPs — the path where I23's const-GEP-stride bug lived (a `getelementptr
    /// (i8, ptr @g, …)` whose source element type differs from `@g`'s pointee).
    glob: [u32; 16],
}

/// A per-lane binary op on raw lane values `(x, y, lane_bits) -> result`.
type LaneOp = fn(u64, u64, u32) -> u64;

const SCALAR_TYS: [u32; 2] = [32, 64];
const SHAPES: [Shape; 3] = [
    Shape {
        lanes: 4,
        lane_bits: 32,
    }, // <4 x i32>
    Shape {
        lanes: 2,
        lane_bits: 32,
    }, // <2 x i32>  (packed-i64 path)
    Shape {
        lanes: 2,
        lane_bits: 64,
    }, // <2 x i64>
];

fn ty_str_scalar(bits: u32) -> String {
    format!("i{bits}")
}
fn ty_str_vec(s: Shape) -> String {
    format!("<{} x i{}>", s.lanes, s.lane_bits)
}

impl Prog {
    fn fresh(&mut self) -> String {
        let n = self.n;
        self.n += 1;
        format!("%v{n}")
    }
    fn emit(&mut self, line: &str) {
        self.body.push_str("  ");
        self.body.push_str(line);
        self.body.push('\n');
    }
    fn push_scalar(&mut self, name: String, bits: u32, raw: u64) {
        self.pool.push(Val {
            name,
            kind: Kind::Scalar {
                bits,
                raw: mask(raw, bits),
            },
        });
    }
    fn push_vec(&mut self, name: String, shape: Shape, lanes: Vec<u64>) {
        let lanes = lanes.iter().map(|&l| mask(l, shape.lane_bits)).collect();
        self.pool.push(Val {
            name,
            kind: Kind::Vec { shape, lanes },
        });
    }
    /// A pool scalar of exactly `bits`, or a fresh constant if none exists.
    fn any_scalar(&mut self, g: &mut Gen, bits: u32) -> (String, u64) {
        let idxs: Vec<usize> = self
            .pool
            .iter()
            .enumerate()
            .filter(|(_, v)| matches!(v.kind, Kind::Scalar { bits: b, .. } if b == bits))
            .map(|(i, _)| i)
            .collect();
        if !idxs.is_empty() && !g.byte().is_multiple_of(4) {
            let i = idxs[g.u(idxs.len())];
            if let Kind::Scalar { raw, .. } = self.pool[i].kind {
                return (self.pool[i].name.clone(), raw);
            }
        }
        let c = g.boundary_scalar(bits);
        (format!("{}", sext(c, bits)), c) // signed decimal literal (LLVM accepts negatives)
    }
    /// A pool vector of exactly `shape`, or a freshly built constant vector if none exists.
    fn any_vec(&mut self, g: &mut Gen, shape: Shape) -> (String, Vec<u64>) {
        let idxs: Vec<usize> = self
            .pool
            .iter()
            .enumerate()
            .filter(|(_, v)| matches!(&v.kind, Kind::Vec { shape: s, .. } if *s == shape))
            .map(|(i, _)| i)
            .collect();
        if !idxs.is_empty() && !g.byte().is_multiple_of(3) {
            let i = idxs[g.u(idxs.len())];
            if let Kind::Vec { lanes, .. } = &self.pool[i].kind {
                return (self.pool[i].name.clone(), lanes.clone());
            }
        }
        // A constant vector literal.
        let lanes: Vec<u64> = (0..shape.lanes)
            .map(|_| g.boundary_scalar(shape.lane_bits))
            .collect();
        let elems: Vec<String> = lanes
            .iter()
            .map(|&l| format!("i{} {}", shape.lane_bits, sext(l, shape.lane_bits)))
            .collect();
        (format!("<{}>", elems.join(", ")), lanes)
    }
}

/// The set of well-defined ops. Each computes its concrete result to keep the oracle exact.
pub fn gen_program(g: &mut Gen) -> (String, i64) {
    let mut p = Prog {
        body: String::new(),
        pool: Vec::new(),
        n: 0,
        mem: [0; 16],
        have_mem: false,
        glob: [0; 16],
    };
    for slot in p.glob.iter_mut() {
        *slot = g.boundary_scalar(32) as u32;
    }

    // A stack array for GEP/load/store coverage.
    p.emit("%arr = alloca [16 x i32], align 16");
    p.have_mem = true;

    let steps = 12 + g.u(40);
    for _ in 0..steps {
        match g.u(16) {
            // ---- scalar integer binops (wrapping / bitwise) ----
            0 => {
                let bits = SCALAR_TYS[g.u(2)];
                let (a, av) = p.any_scalar(g, bits);
                let (b, bv) = p.any_scalar(g, bits);
                let (op, r) = match g.u(6) {
                    0 => ("add", av.wrapping_add(bv)),
                    1 => ("sub", av.wrapping_sub(bv)),
                    2 => ("mul", av.wrapping_mul(bv)),
                    3 => ("and", av & bv),
                    4 => ("or", av | bv),
                    _ => ("xor", av ^ bv),
                };
                let d = p.fresh();
                p.emit(&format!("{d} = {op} i{bits} {a}, {b}"));
                p.push_scalar(d, bits, r);
            }
            // ---- shifts (amount < width, no poison) ----
            1 => {
                let bits = SCALAR_TYS[g.u(2)];
                let (a, av) = p.any_scalar(g, bits);
                let amt = (g.byte() as u32) % bits;
                let (op, r) = match g.u(3) {
                    0 => ("shl", mask(av << amt, bits)),
                    1 => ("lshr", mask(av, bits) >> amt),
                    _ => ("ashr", mask((sext(av, bits) >> amt) as u64, bits)),
                };
                let d = p.fresh();
                p.emit(&format!("{d} = {op} i{bits} {a}, {amt}"));
                p.push_scalar(d, bits, r);
            }
            // ---- icmp + select ----
            2 => {
                let bits = SCALAR_TYS[g.u(2)];
                let (a, av) = p.any_scalar(g, bits);
                let (b, bv) = p.any_scalar(g, bits);
                let (pred, cond) = match g.u(6) {
                    0 => ("eq", av == bv),
                    1 => ("ne", av != bv),
                    2 => ("slt", sext(av, bits) < sext(bv, bits)),
                    3 => ("sgt", sext(av, bits) > sext(bv, bits)),
                    4 => ("ult", mask(av, bits) < mask(bv, bits)),
                    _ => ("ugt", mask(av, bits) > mask(bv, bits)),
                };
                let c = p.fresh();
                p.emit(&format!("{c} = icmp {pred} i{bits} {a}, {b}"));
                let (x, xv) = p.any_scalar(g, bits);
                let (y, yv) = p.any_scalar(g, bits);
                let d = p.fresh();
                p.emit(&format!("{d} = select i1 {c}, i{bits} {x}, i{bits} {y}"));
                p.push_scalar(d, bits, if cond { xv } else { yv });
            }
            // ---- width conversions ----
            3 => {
                let (a, av) = p.any_scalar(g, 32);
                let d = p.fresh();
                match g.u(2) {
                    0 => {
                        p.emit(&format!("{d} = sext i32 {a} to i64"));
                        p.push_scalar(d, 64, sext(av, 32) as u64);
                    }
                    _ => {
                        p.emit(&format!("{d} = zext i32 {a} to i64"));
                        p.push_scalar(d, 64, mask(av, 32));
                    }
                }
            }
            4 => {
                let (a, av) = p.any_scalar(g, 64);
                let d = p.fresh();
                p.emit(&format!("{d} = trunc i64 {a} to i32"));
                p.push_scalar(d, 32, mask(av, 32));
            }
            // ---- vector binops ----
            5 => {
                let s = SHAPES[g.u(3)];
                let (a, av) = p.any_vec(g, s);
                let (b, bv) = p.any_vec(g, s);
                let (op, f): (&str, LaneOp) = match g.u(6) {
                    0 => ("add", |x, y, _| x.wrapping_add(y)),
                    1 => ("sub", |x, y, _| x.wrapping_sub(y)),
                    2 => ("mul", |x, y, _| x.wrapping_mul(y)),
                    3 => ("and", |x, y, _| x & y),
                    4 => ("or", |x, y, _| x | y),
                    _ => ("xor", |x, y, _| x ^ y),
                };
                let r: Vec<u64> = av
                    .iter()
                    .zip(&bv)
                    .map(|(&x, &y)| f(x, y, s.lane_bits))
                    .collect();
                let d = p.fresh();
                p.emit(&format!("{d} = {op} {} {a}, {b}", ty_str_vec(s)));
                p.push_vec(d, s, r);
            }
            // ---- vector min/max intrinsics (the vec2/128 hinge) ----
            6 => {
                let s = SHAPES[g.u(3)];
                let (a, av) = p.any_vec(g, s);
                let (b, bv) = p.any_vec(g, s);
                let (nm, signed, is_max) = match g.u(4) {
                    0 => ("smax", true, true),
                    1 => ("smin", true, false),
                    2 => ("umax", false, true),
                    _ => ("umin", false, false),
                };
                let lb = s.lane_bits;
                let r: Vec<u64> = av
                    .iter()
                    .zip(&bv)
                    .map(|(&x, &y)| {
                        let pick_x = if signed {
                            if is_max {
                                sext(x, lb) >= sext(y, lb)
                            } else {
                                sext(x, lb) <= sext(y, lb)
                            }
                        } else if is_max {
                            mask(x, lb) >= mask(y, lb)
                        } else {
                            mask(x, lb) <= mask(y, lb)
                        };
                        if pick_x {
                            x
                        } else {
                            y
                        }
                    })
                    .collect();
                let tv = ty_str_vec(s);
                let d = p.fresh();
                p.emit(&format!(
                    "{d} = call {tv} @llvm.{nm}.v{}i{}({tv} {a}, {tv} {b})",
                    s.lanes, s.lane_bits
                ));
                p.push_vec(d, s, r);
            }
            // ---- vector widen: sext/zext <L x i32> -> <L x i64> (L=2 keeps <=128) ----
            7 => {
                let s = Shape {
                    lanes: 2,
                    lane_bits: 32,
                };
                let (a, av) = p.any_vec(g, s);
                let d = p.fresh();
                let out = Shape {
                    lanes: 2,
                    lane_bits: 64,
                };
                let (kw, r): (&str, Vec<u64>) = match g.u(2) {
                    0 => ("sext", av.iter().map(|&x| sext(x, 32) as u64).collect()),
                    _ => ("zext", av.iter().map(|&x| mask(x, 32)).collect()),
                };
                p.emit(&format!(
                    "{d} = {kw} {} {a} to {}",
                    ty_str_vec(s),
                    ty_str_vec(out)
                ));
                p.push_vec(d, out, r);
            }
            // ---- vector narrow: trunc <2 x i64> -> <2 x i32> ----
            8 => {
                let s = Shape {
                    lanes: 2,
                    lane_bits: 64,
                };
                let (a, av) = p.any_vec(g, s);
                let out = Shape {
                    lanes: 2,
                    lane_bits: 32,
                };
                let r: Vec<u64> = av.iter().map(|&x| mask(x, 32)).collect();
                let d = p.fresh();
                p.emit(&format!(
                    "{d} = trunc {} {a} to {}",
                    ty_str_vec(s),
                    ty_str_vec(out)
                ));
                p.push_vec(d, out, r);
            }
            // ---- extractelement: vector lane -> scalar ----
            9 => {
                let s = SHAPES[g.u(3)];
                let (a, av) = p.any_vec(g, s);
                let lane = g.u(s.lanes as usize);
                let d = p.fresh();
                p.emit(&format!(
                    "{d} = extractelement {} {a}, i32 {lane}",
                    ty_str_vec(s)
                ));
                p.push_scalar(d, s.lane_bits, av[lane]);
            }
            // ---- store i32 via [16 x i32] GEP (variable in-bounds index) ----
            10 | 11 => {
                let (v, vv) = p.any_scalar(g, 32);
                let idx = g.u(16);
                let ptr = p.fresh();
                p.emit(&format!(
                    "{ptr} = getelementptr inbounds [16 x i32], ptr %arr, i64 0, i64 {idx}"
                ));
                p.emit(&format!("store i32 {v}, ptr {ptr}, align 4"));
                p.mem[idx] = mask(vv, 32) as u32;
            }
            // ---- load i32 via an i8-element GEP (distinct source element type — the I23 path) ----
            12 => {
                let idx = g.u(16);
                let byteoff = idx * 4;
                let ptr = p.fresh();
                p.emit(&format!(
                    "{ptr} = getelementptr inbounds i8, ptr %arr, i64 {byteoff}"
                ));
                let d = p.fresh();
                p.emit(&format!("{d} = load i32, ptr {ptr}, align 4"));
                p.push_scalar(d, 32, p.mem[idx] as u64);
            }
            // ---- load @g[idx] via a **constexpr** array-typed GEP (source element `[16 x i32]`) ----
            13 => {
                let idx = g.u(16);
                let d = p.fresh();
                p.emit(&format!(
                    "{d} = load i32, ptr getelementptr inbounds ([16 x i32], ptr @g, i64 0, i64 {idx}), align 4"
                ));
                p.push_scalar(d, 32, p.glob[idx] as u64);
            }
            // ---- load @g[idx] via a **constexpr** i8-typed GEP (source element `i8`, byte offset —
            //      the exact I23 const-GEP-stride path: strides by `i8`, not by `@g`'s pointee) ----
            14 => {
                let idx = g.u(16);
                let byteoff = idx * 4;
                let d = p.fresh();
                p.emit(&format!(
                    "{d} = load i32, ptr getelementptr (i8, ptr @g, i64 {byteoff}), align 4"
                ));
                p.push_scalar(d, 32, p.glob[idx] as u64);
            }
            // ---- build a vector by insertelement from scalars ----
            _ => {
                let s = SHAPES[g.u(3)];
                let mut cur = String::from("undef");
                let mut lanes = Vec::new();
                for lane in 0..s.lanes {
                    let (v, vv) = p.any_scalar(g, s.lane_bits);
                    lanes.push(vv);
                    let d = p.fresh();
                    p.emit(&format!(
                        "{d} = insertelement {} {cur}, i{} {v}, i32 {lane}",
                        ty_str_vec(s),
                        s.lane_bits
                    ));
                    cur = d;
                }
                // `cur` names the final vector.
                p.pool.push(Val {
                    name: cur,
                    kind: Kind::Vec { shape: s, lanes },
                });
            }
        }
    }

    // Reduce the whole pool to one i64 by xor-folding (scalars widened, vector lanes extracted).
    let mut acc_name = String::from("0");
    let mut acc_val: u64 = 0;
    // Snapshot the pool names/kinds to iterate (we append new insts as we fold).
    let snapshot: Vec<Val> = p.pool.clone();
    for v in snapshot {
        match v.kind {
            Kind::Scalar { bits, raw } => {
                let w = if bits == 64 {
                    (v.name.clone(), raw)
                } else {
                    let d = p.fresh();
                    p.emit(&format!("{d} = zext i32 {} to i64", v.name));
                    (d, mask(raw, 32))
                };
                let d = p.fresh();
                p.emit(&format!("{d} = xor i64 {acc_name}, {}", w.0));
                acc_val ^= w.1;
                acc_name = d;
            }
            Kind::Vec { shape, lanes } => {
                for (li, &lv) in lanes.iter().enumerate() {
                    let e = p.fresh();
                    p.emit(&format!(
                        "{e} = extractelement {} {}, i32 {li}",
                        ty_str_vec(shape),
                        v.name
                    ));
                    let w = if shape.lane_bits == 64 {
                        (e, lv)
                    } else {
                        let d = p.fresh();
                        p.emit(&format!("{d} = zext i32 {e} to i64"));
                        (d, mask(lv, 32))
                    };
                    let d = p.fresh();
                    p.emit(&format!("{d} = xor i64 {acc_name}, {}", w.0));
                    acc_val ^= w.1;
                    acc_name = d;
                }
            }
        }
    }
    p.emit(&format!("ret i64 {acc_name}"));

    // Assemble the module: the read-only global (read via constexpr GEPs), the min/max intrinsic
    // declarations, then define @run.
    let mut m = String::new();
    let gelems: Vec<String> = p
        .glob
        .iter()
        .map(|&v| format!("i32 {}", sext(v as u64, 32)))
        .collect();
    m.push_str(&format!(
        "@g = internal constant [16 x i32] [{}]\n",
        gelems.join(", ")
    ));
    for s in SHAPES {
        for nm in ["smax", "smin", "umax", "umin"] {
            let tv = ty_str_vec(s);
            m.push_str(&format!(
                "declare {tv} @llvm.{nm}.v{}i{}({tv}, {tv})\n",
                s.lanes, s.lane_bits
            ));
        }
    }
    m.push_str("define i64 @run() {\nentry:\n");
    m.push_str(&p.body);
    m.push_str("}\n");
    (m, acc_val as i64)
}

/// Which fail-closed-capable backends actually executed a program (for the stable test's
/// coverage guard). The tree-walker always runs (the on-ramp's reference oracle).
pub struct Ran {
    pub translated: bool,
    pub bc: bool,
    pub jit: bool,
}

/// Translate `ll` and assert every backend that can execute it returns `oracle` (the source
/// semantics). `Err(reason)` on any divergence — a miscompile. Shared by the stable test and the
/// libFuzzer target so both check identically.
///
/// - translate `unsup` → not a miscompile (a construct a later slice will support) → skipped;
/// - the tree-walker is the always-on oracle-check (it supports everything translate emits and
///   these programs never trap);
/// - bytecode `None` / JIT `Unsupported` = fail-closed on a construct → that backend skipped;
/// - any *other* outcome (wrong value, trap, other error) is a miscompile.
pub fn check(ll: &str, oracle: i64) -> Result<Ran, String> {
    use svm_interp::{bytecode, Value};
    let mut ran = Ran {
        translated: false,
        bc: false,
        jit: false,
    };
    let t = match svm_llvm::translate_ll_str(ll) {
        Ok(t) => t,
        Err(_) => return Ok(ran),
    };
    ran.translated = true;
    svm_verify::verify_module(&t.module).map_err(|e| format!("verify failed: {e:?}"))?;
    let run = t
        .exports
        .iter()
        .find(|(n, _)| n == "run")
        .map(|(_, i)| *i)
        .ok_or("no `run` export")?;
    let sp = t.entry_sp as i64;

    let mut fuel = 50_000_000u64;
    match svm_interp::run(&t.module, run, &[Value::I64(sp)], &mut fuel) {
        Ok(v) if v == vec![Value::I64(oracle)] => {}
        other => return Err(format!("tree-walk = {other:?}, oracle = {oracle}")),
    }
    let mut fuel = 50_000_000u64;
    match bytecode::compile_and_run(&t.module, run, &[Value::I64(sp)], &mut fuel) {
        None => {}
        Some(Ok(v)) if v == vec![Value::I64(oracle)] => ran.bc = true,
        other => return Err(format!("bytecode = {other:?}, oracle = {oracle}")),
    }
    match svm_jit::compile_and_run(&t.module, run, &[sp]) {
        Err(svm_jit::JitError::Unsupported(_)) => {}
        Ok(svm_jit::JitOutcome::Returned(v)) if v == vec![oracle] => ran.jit = true,
        other => return Err(format!("jit = {other:?}, oracle = {oracle}")),
    }
    Ok(ran)
}
