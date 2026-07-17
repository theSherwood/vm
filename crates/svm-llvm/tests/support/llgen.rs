//! Structured generator of **well-defined, terminating** LLVM-IR modules for the on-ramp
//! differential harness (`onramp_diff.rs` stable test + `fuzz/onramp_diff` target).
//!
//! It emits `define i64 @run()` (plus a few pure helper functions and a read-only global) while
//! **computing each value's concrete result as it emits** — so the final `ret i64` value is known
//! by construction (the *oracle*), with no separate interpreter to carry its own bugs. Every
//! program is UB-free by construction (shift amounts `< width`, in-bounds array indices, fixed loop
//! bounds, no div, no poison flags), so it must **never trap**, and all three svm backends
//! (tree-walker, bytecode, JIT) must return the oracle value. A divergence is an **on-ramp
//! translation bug** — the I23 class, where every backend agrees with the *wrong* IR (so an
//! interp-vs-JIT differential can't see it; the source-semantics oracle can).
//!
//! Coverage spans the translation surface that has bitten us and its neighbours:
//! - `getelementptr` with distinct source element types — instruction-form over an `alloca` **and**
//!   **constexpr** GEPs over a global (`i8` vs `[N x i32]` — the I23 const-GEP-stride path);
//! - 2-lane (`<2 x i32>`, packed-i64) *and* 128-bit vector min/max, widen/narrow, and width
//!   conversions (the I23 vec2-minmax path);
//! - scalar bit-manipulation intrinsics (`ctpop`/`ctlz`/`cttz`/`bswap`/`bitreverse`/`abs`) and
//!   funnel-shift rotates (`fshl`/`fshr`);
//! - vector shifts, vector `icmp`+`select` (`<N x i1>` masks), and `shufflevector`;
//! - control flow — phi-merge diamonds and fixed-bound counted loops (back-edges, phi lowering,
//!   loop-variant GEP indices); and
//! - function calls to pure helpers (the call ABI: threaded data-SP, arg passing, scalar return).
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

// ---- exact oracles for the intrinsic ops (must match the backends bit-for-bit) ----------------

/// `llvm.ctlz(x, is_zero_poison=false)` over a `bits`-wide value: leading zeros, `bits` when zero.
fn ctlz_bits(v: u64, bits: u32) -> u64 {
    let m = mask(v, bits);
    if m == 0 {
        bits as u64
    } else {
        (m.leading_zeros() - (64 - bits)) as u64
    }
}
/// `llvm.cttz(x, is_zero_poison=false)`: trailing zeros, `bits` when zero.
fn cttz_bits(v: u64, bits: u32) -> u64 {
    let m = mask(v, bits);
    if m == 0 {
        bits as u64
    } else {
        m.trailing_zeros() as u64
    }
}
/// `llvm.bswap`: reverse the `bits`-wide byte order (`bits` is a byte multiple: 32 or 64 here).
fn bswap_bits(v: u64, bits: u32) -> u64 {
    match bits {
        32 => (v as u32).swap_bytes() as u64,
        64 => v.swap_bytes(),
        _ => unreachable!("bswap only over i32/i64"),
    }
}
/// `llvm.bitreverse`: reverse all `bits` bits.
fn bitrev_bits(v: u64, bits: u32) -> u64 {
    match bits {
        32 => (v as u32).reverse_bits() as u64,
        64 => v.reverse_bits(),
        _ => unreachable!("bitreverse only over i32/i64"),
    }
}
/// `llvm.abs(x, is_int_min_poison=false)`: two's-complement absolute value; `abs(INT_MIN)=INT_MIN`.
fn abs_bits(v: u64, bits: u32) -> u64 {
    mask(sext(v, bits).wrapping_abs() as u64, bits)
}
/// `llvm.fshl(a, b, c)` over `bits`: high `bits` of `(a:b) << (c mod bits)`.
fn fshl_bits(a: u64, b: u64, c: u32, bits: u32) -> u64 {
    let c = c % bits;
    if c == 0 {
        mask(a, bits)
    } else {
        mask((mask(a, bits) << c) | (mask(b, bits) >> (bits - c)), bits)
    }
}
/// `llvm.fshr(a, b, c)` over `bits`: low `bits` of `(a:b) >> (c mod bits)`.
fn fshr_bits(a: u64, b: u64, c: u32, bits: u32) -> u64 {
    let c = c % bits;
    if c == 0 {
        mask(b, bits)
    } else {
        mask((mask(a, bits) << (bits - c)) | (mask(b, bits) >> c), bits)
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
    /// Fresh basic-block label counter (for the phi-merge diamonds and counted loops).
    blk: usize,
    /// The label of the block currently being generated into — a phi's predecessor edge from the
    /// straight-line code before a construct.
    cur_block: String,
    /// Pure `i64,i64 -> i64` helpers `@run` can call (exercises the call ABI).
    helpers: Vec<Helper>,
}

/// A per-lane binary op on raw lane values `(x, y, lane_bits) -> result`.
type LaneOp = fn(u64, u64, u32) -> u64;

/// A wrapping i64 binop selector (`0=add 1=sub 2=mul _=xor`) — shared by generated helper bodies
/// and their Rust oracle so both agree exactly.
fn apply_i64(op: u8, a: u64, b: u64) -> u64 {
    match op {
        0 => a.wrapping_add(b),
        1 => a.wrapping_sub(b),
        2 => a.wrapping_mul(b),
        _ => a ^ b,
    }
}
fn op_name(op: u8) -> &'static str {
    ["add", "sub", "mul", "xor"][(op & 3) as usize]
}

/// A pure `i64 f(i64 a, i64 b)` helper: `(a op1 c1) op2 (b op3 c2)` (all wrapping). Emitted as its
/// own `define` and callable from `@run` — exercises the on-ramp's call ABI (the threaded data-SP,
/// arg passing, scalar return). Small and closed-form so the oracle evaluates it directly.
#[derive(Clone)]
struct Helper {
    name: String,
    op1: u8,
    c1: i64,
    op2: u8,
    op3: u8,
    c2: i64,
}
impl Helper {
    fn eval(&self, a: u64, b: u64) -> u64 {
        let x = apply_i64(self.op1, a, self.c1 as u64);
        let y = apply_i64(self.op3, b, self.c2 as u64);
        apply_i64(self.op2, x, y)
    }
    fn define(&self) -> String {
        format!(
            "define i64 @{}(i64 %a, i64 %b) {{\n  \
             %x = {} i64 %a, {}\n  \
             %y = {} i64 %b, {}\n  \
             %r = {} i64 %x, %y\n  \
             ret i64 %r\n}}\n",
            self.name,
            op_name(self.op1),
            self.c1,
            op_name(self.op3),
            self.c2,
            op_name(self.op2),
        )
    }
}

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
    /// Emit a block label (`name:` at column 0 — starts a new basic block) and make it current.
    fn emit_label(&mut self, name: &str) {
        self.body.push_str(name);
        self.body.push_str(":\n");
        self.cur_block = name.to_string();
    }
    /// Emit one scalar arithmetic op into the *current* block, returning its `(name, conc)`. Used to
    /// give a phi-merge branch a computed (not merely forwarded) value. Local to its block — the
    /// caller truncates the pool afterward so the value never leaks past its dominance region.
    fn branch_expr(&mut self, g: &mut Gen, bits: u32) -> (String, u64) {
        let (a, av) = self.any_scalar(g, bits);
        let (b, bv) = self.any_scalar(g, bits);
        let (op, r) = match g.u(4) {
            0 => ("add", av.wrapping_add(bv)),
            1 => ("sub", av.wrapping_sub(bv)),
            2 => ("mul", av.wrapping_mul(bv)),
            _ => ("xor", av ^ bv),
        };
        let d = self.fresh();
        self.emit(&format!("{d} = {op} i{bits} {a}, {b}"));
        self.push_scalar(d.clone(), bits, r);
        (d, mask(r, bits))
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
        blk: 0,
        cur_block: String::from("entry"),
        helpers: Vec::new(),
    };
    for slot in p.glob.iter_mut() {
        *slot = g.boundary_scalar(32) as u32;
    }
    for h in 0..(1 + g.u(3)) {
        p.helpers.push(Helper {
            name: format!("f{h}"),
            op1: g.byte() & 3,
            c1: sext(g.boundary_scalar(64), 64),
            op2: g.byte() & 3,
            op3: g.byte() & 3,
            c2: sext(g.boundary_scalar(64), 64),
        });
    }

    // A stack array for GEP/load/store coverage.
    p.emit("%arr = alloca [16 x i32], align 16");
    p.have_mem = true;

    let steps = 12 + g.u(40);
    for _ in 0..steps {
        match g.u(24) {
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
            // ---- control flow: a diamond with a phi merge (forward-only ⇒ still terminating) ----
            15 => {
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
                let k = p.blk;
                p.blk += 1;
                let (tl, el, ml) = (format!("t{k}"), format!("e{k}"), format!("m{k}"));
                p.emit(&format!("br i1 {c}, label %{tl}, label %{el}"));
                // Values in each arm are block-local: snapshot the pool, generate, then truncate so
                // nothing escapes its dominance region (the merge only sees the phi).
                let base = p.pool.len();
                p.emit_label(&tl);
                let (tv, tvc) = p.branch_expr(g, bits);
                p.emit(&format!("br label %{ml}"));
                p.pool.truncate(base);
                p.emit_label(&el);
                let (ev, evc) = p.branch_expr(g, bits);
                p.emit(&format!("br label %{ml}"));
                p.pool.truncate(base);
                p.emit_label(&ml);
                let phi = p.fresh();
                p.emit(&format!(
                    "{phi} = phi i{bits} [ {tv}, %{tl} ], [ {ev}, %{el} ]"
                ));
                p.push_scalar(phi, bits, if cond { tvc } else { evc });
            }
            // ---- a fixed-bound counted loop: induction + accumulator phis, a loop-variant load
            //      from @g[i] (exercises back-edges, phi lowering, and a variable GEP index) ----
            16 => {
                let n = 2 + g.u(5) as u64; // 2..=6 iterations — in-bounds for @g[i] and terminating
                let (init, initc) = p.any_scalar(g, 64);
                let pre = p.cur_block.clone();
                let k = p.blk;
                p.blk += 1;
                let (hl, xl) = (format!("h{k}"), format!("x{k}"));
                let (iv, accv, inext, accnext) = (p.fresh(), p.fresh(), p.fresh(), p.fresh());
                p.emit(&format!("br label %{hl}"));
                p.emit_label(&hl);
                p.emit(&format!("{iv} = phi i64 [ 0, %{pre} ], [ {inext}, %{hl} ]"));
                p.emit(&format!(
                    "{accv} = phi i64 [ {init}, %{pre} ], [ {accnext}, %{hl} ]"
                ));
                let gp = p.fresh();
                p.emit(&format!(
                    "{gp} = getelementptr inbounds [16 x i32], ptr @g, i64 0, i64 {iv}"
                ));
                let gi = p.fresh();
                p.emit(&format!("{gi} = load i32, ptr {gp}, align 4"));
                let gis = p.fresh();
                p.emit(&format!("{gis} = sext i32 {gi} to i64"));
                p.emit(&format!("{accnext} = add i64 {accv}, {gis}"));
                p.emit(&format!("{inext} = add i64 {iv}, 1"));
                let lc = p.fresh();
                p.emit(&format!("{lc} = icmp ult i64 {inext}, {n}"));
                p.emit(&format!("br i1 {lc}, label %{hl}, label %{xl}"));
                p.emit_label(&xl);
                // The final accumulator (`{accnext}` in the header) dominates the exit (header is the
                // exit's only predecessor). Compute the oracle by running the loop.
                let mut acc = initc;
                for i in 0..n {
                    acc = acc.wrapping_add(sext(p.glob[i as usize] as u64, 32) as u64);
                }
                p.push_scalar(accnext, 64, acc);
            }
            // ---- call a pure helper (the call ABI: threaded data-SP, arg passing, scalar return) ----
            17 => {
                let hi = g.u(p.helpers.len());
                let h = p.helpers[hi].clone();
                let (a, av) = p.any_scalar(g, 64);
                let (b, bv) = p.any_scalar(g, 64);
                let d = p.fresh();
                p.emit(&format!("{d} = call i64 @{}(i64 {a}, i64 {b})", h.name));
                p.push_scalar(d, 64, h.eval(av, bv));
            }
            // ---- scalar bit-manipulation intrinsics (ctpop/ctlz/cttz/bswap/bitreverse/abs) ----
            18 => {
                let bits = SCALAR_TYS[g.u(2)];
                let (a, av) = p.any_scalar(g, bits);
                let m = mask(av, bits);
                let (nm, r): (&str, u64) = match g.u(6) {
                    0 => ("ctpop", m.count_ones() as u64),
                    1 => ("ctlz", ctlz_bits(m, bits)),
                    2 => ("cttz", cttz_bits(m, bits)),
                    3 => ("bswap", bswap_bits(m, bits)),
                    4 => ("bitreverse", bitrev_bits(m, bits)),
                    _ => ("abs", abs_bits(m, bits)),
                };
                let d = p.fresh();
                // ctlz/cttz take an `is_zero_poison` flag, abs an `is_int_min_poison` flag — both false.
                match nm {
                    "ctlz" | "cttz" | "abs" => p.emit(&format!(
                        "{d} = call i{bits} @llvm.{nm}.i{bits}(i{bits} {a}, i1 false)"
                    )),
                    _ => p.emit(&format!(
                        "{d} = call i{bits} @llvm.{nm}.i{bits}(i{bits} {a})"
                    )),
                }
                p.push_scalar(d, bits, r);
            }
            // ---- funnel shift / rotate (fshl/fshr) ----
            19 => {
                let bits = SCALAR_TYS[g.u(2)];
                let (a, av) = p.any_scalar(g, bits);
                let (b, bv) = p.any_scalar(g, bits);
                let amt = (g.byte() as u32) % bits;
                let (nm, r) = if g.byte().is_multiple_of(2) {
                    ("fshl", fshl_bits(av, bv, amt, bits))
                } else {
                    ("fshr", fshr_bits(av, bv, amt, bits))
                };
                let d = p.fresh();
                p.emit(&format!(
                    "{d} = call i{bits} @llvm.{nm}.i{bits}(i{bits} {a}, i{bits} {b}, i{bits} {amt})"
                ));
                p.push_scalar(d, bits, r);
            }
            // ---- vector shifts (per-lane amount < lane width, no poison) ----
            20 => {
                let s = SHAPES[g.u(3)];
                let (a, av) = p.any_vec(g, s);
                let lb = s.lane_bits;
                let amts: Vec<u32> = (0..s.lanes).map(|_| (g.byte() as u32) % lb).collect();
                let opc = g.u(3);
                let r: Vec<u64> = av
                    .iter()
                    .zip(&amts)
                    .map(|(&x, &amt)| match opc {
                        0 => mask(mask(x, lb) << amt, lb),
                        1 => mask(x, lb) >> amt,
                        _ => mask((sext(x, lb) >> amt) as u64, lb),
                    })
                    .collect();
                let op = ["shl", "lshr", "ashr"][opc];
                let amtvec: Vec<String> = amts.iter().map(|&a| format!("i{lb} {a}")).collect();
                let d = p.fresh();
                p.emit(&format!(
                    "{d} = {op} {} {a}, <{}>",
                    ty_str_vec(s),
                    amtvec.join(", ")
                ));
                p.push_vec(d, s, r);
            }
            // ---- vector icmp + vector select (`<N x i1>` mask feeding a per-lane select) ----
            21 => {
                let s = SHAPES[g.u(3)];
                let (a, av) = p.any_vec(g, s);
                let (b, bv) = p.any_vec(g, s);
                let lb = s.lane_bits;
                let pi = g.u(6);
                let pred = ["eq", "ne", "slt", "sgt", "ult", "ugt"][pi];
                let conds: Vec<bool> = av
                    .iter()
                    .zip(&bv)
                    .map(|(&x, &y)| match pi {
                        0 => x == y,
                        1 => x != y,
                        2 => sext(x, lb) < sext(y, lb),
                        3 => sext(x, lb) > sext(y, lb),
                        4 => mask(x, lb) < mask(y, lb),
                        _ => mask(x, lb) > mask(y, lb),
                    })
                    .collect();
                let tv = ty_str_vec(s);
                let c = p.fresh();
                p.emit(&format!("{c} = icmp {pred} {tv} {a}, {b}"));
                let (x, xv) = p.any_vec(g, s);
                let (y, yv) = p.any_vec(g, s);
                let d = p.fresh();
                p.emit(&format!(
                    "{d} = select <{} x i1> {c}, {tv} {x}, {tv} {y}",
                    s.lanes
                ));
                let r: Vec<u64> = (0..s.lanes as usize)
                    .map(|i| if conds[i] { xv[i] } else { yv[i] })
                    .collect();
                p.push_vec(d, s, r);
            }
            // ---- shufflevector: pick each result lane from the a:b concatenation (constant mask) ----
            22 => {
                let s = SHAPES[g.u(3)];
                let (a, av) = p.any_vec(g, s);
                let (b, bv) = p.any_vec(g, s);
                let n = s.lanes as usize;
                let sel: Vec<usize> = (0..n).map(|_| g.u(2 * n)).collect();
                let maskv: Vec<String> = sel.iter().map(|&i| format!("i32 {i}")).collect();
                let r: Vec<u64> = sel
                    .iter()
                    .map(|&i| if i < n { av[i] } else { bv[i - n] })
                    .collect();
                let d = p.fresh();
                p.emit(&format!(
                    "{d} = shufflevector {tv} {a}, {tv} {b}, <{n} x i32> <{}>",
                    maskv.join(", "),
                    tv = ty_str_vec(s)
                ));
                p.push_vec(d, s, r);
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
    // Scalar bit-manip / funnel-shift intrinsics (unused declares are harmless).
    for bits in SCALAR_TYS {
        m.push_str(&format!("declare i{bits} @llvm.ctpop.i{bits}(i{bits})\n"));
        m.push_str(&format!(
            "declare i{bits} @llvm.ctlz.i{bits}(i{bits}, i1)\n"
        ));
        m.push_str(&format!(
            "declare i{bits} @llvm.cttz.i{bits}(i{bits}, i1)\n"
        ));
        m.push_str(&format!("declare i{bits} @llvm.bswap.i{bits}(i{bits})\n"));
        m.push_str(&format!(
            "declare i{bits} @llvm.bitreverse.i{bits}(i{bits})\n"
        ));
        m.push_str(&format!("declare i{bits} @llvm.abs.i{bits}(i{bits}, i1)\n"));
        m.push_str(&format!(
            "declare i{bits} @llvm.fshl.i{bits}(i{bits}, i{bits}, i{bits})\n"
        ));
        m.push_str(&format!(
            "declare i{bits} @llvm.fshr.i{bits}(i{bits}, i{bits}, i{bits})\n"
        ));
    }
    for h in &p.helpers {
        m.push_str(&h.define());
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
