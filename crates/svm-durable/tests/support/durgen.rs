//! Generative property for the freeze/thaw transform (DURABILITY.md §7/§12.6, R11).
//!
//! Shared between the stable `cargo test` (`tests/durable_fuzz.rs`) and the libFuzzer
//! target (`fuzz/fuzz_targets/durable.rs`), mirroring `irgen`. The generator emits
//! **in-scope** durable modules (single block, single `cap.call`, `return`) so the
//! property exercises a real input space instead of the arbitrary-IR generator (which
//! the Phase-1 transform would reject almost everywhere).
//!
//! Each module is checked for two properties:
//!   1. **inert in NORMAL** — the instrumented module run in `NORMAL` state produces
//!      the same result as the original, un-instrumented module;
//!   2. **round-trip** — freeze → serialize window → restore → thaw equals the
//!      uninterrupted run, on a *fresh* host (so a buggy re-issue of the `cap.call`
//!      instead of reloading the saved value would diverge).

#![allow(dead_code)] // not every helper is used by both includers

use svm_durable::{
    init_durable_window, read_state, transform_module, write_state, SHADOW_BASE, SHADOW_SP_OFF,
    STATE_NORMAL, STATE_REWINDING, STATE_UNWINDING,
};
use svm_interp::{run_capture_reserved_with_host, run_with_host, Host, Value};
use svm_ir::{BinOp, Block, Func, FuncType, Inst, IntTy, Memory, Module, Terminator, ValType};

pub const SIZE_LOG2: u8 = 18;
pub const WINDOW: usize = 1 << SIZE_LOG2;

// Clock capability (type_id 2, op 0): `(i32) -> (i64)`, deterministic per host.
const CLOCK_TYPE_ID: u32 = 2;
const CLOCK_OP: u32 = 0;

/// A tiny xorshift-backed generator: consumes input bytes when available, falls back
/// to the PRNG when exhausted (same shape as `irgen::Gen`).
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
    fn raw(&mut self) -> u64 {
        let mut x = self.rng;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn byte(&mut self) -> u8 {
        if self.pos < self.data.len() {
            let b = self.data[self.pos];
            self.pos += 1;
            b
        } else {
            (self.raw() & 0xff) as u8
        }
    }
    pub fn u64v(&mut self) -> u64 {
        let mut v = 0u64;
        for _ in 0..8 {
            v = (v << 8) | self.byte() as u64;
        }
        v
    }
    /// A value in `0..n` (0 if `n == 0`).
    fn below(&mut self, n: u32) -> u32 {
        if n == 0 {
            0
        } else {
            (self.u64v() % n as u64) as u32
        }
    }
}

/// A total i64 binary op (no `div`/`rem` → never traps), so a generated program always
/// runs to completion and the equivalence comparison is meaningful.
fn total_binop(g: &mut Gen) -> BinOp {
    match g.below(9) {
        0 => BinOp::Add,
        1 => BinOp::Sub,
        2 => BinOp::Mul,
        3 => BinOp::And,
        4 => BinOp::Or,
        5 => BinOp::Xor,
        6 => BinOp::Shl,
        7 => BinOp::ShrS,
        _ => BinOp::ShrU,
    }
}

/// The single may-suspend op a generated function suspends on.
enum Suspend {
    /// Leaf: `cap.call` to the clock (the deepest frame).
    Cap,
    /// Propagated: a `call` to function `callee` (a deeper may-suspend function).
    Call(u32),
}

/// Build one `func (i32) -> (i64)`: a randomized i64 prefix, the `suspend` op (whose i64
/// result is `cap_result`), then a suffix whose result chains off `cap_result` — so the
/// saved/reloaded value is genuinely exercised — and a `return`. The single param `v0` is
/// the clock handle, threaded down the chain as the call/`cap.call` argument.
fn gen_func(g: &mut Gen, suspend: Suspend) -> Func {
    let mut insts: Vec<Inst> = Vec::new();
    let mut i64_vals: Vec<u32> = Vec::new();
    let mut next: u32 = 1; // v0 is the i32 handle param

    // Prefix: a few i64 consts / total binops (live across the suspend op for a wrapper).
    for _ in 0..g.below(6) {
        if i64_vals.len() < 2 || g.below(3) == 0 {
            insts.push(Inst::ConstI64(g.u64v() as i64));
        } else {
            let a = i64_vals[g.below(i64_vals.len() as u32) as usize];
            let b = i64_vals[g.below(i64_vals.len() as u32) as usize];
            insts.push(Inst::IntBin { ty: IntTy::I64, op: total_binop(g), a, b });
        }
        i64_vals.push(next);
        next += 1;
    }

    // The may-suspend op, producing an i64 in `cap_result`.
    let cap_result = match suspend {
        Suspend::Cap => {
            insts.push(Inst::ConstI32(g.u64v() as i32)); // the i32 clock arg
            let arg = next;
            next += 1;
            insts.push(Inst::CapCall {
                type_id: CLOCK_TYPE_ID,
                op: CLOCK_OP,
                sig: FuncType {
                    params: vec![ValType::I32],
                    results: vec![ValType::I64],
                },
                handle: 0,
                args: vec![arg],
            });
            let r = next;
            next += 1;
            r
        }
        Suspend::Call(callee) => {
            insts.push(Inst::Call { func: callee, args: vec![0] }); // pass the handle down
            let r = next;
            next += 1;
            r
        }
    };
    i64_vals.push(cap_result);

    // Suffix: chain off the call result so `return` depends on the resumed value.
    let mut acc = cap_result;
    for _ in 0..g.below(6) {
        let b = if g.below(2) == 0 {
            insts.push(Inst::ConstI64(g.u64v() as i64));
            let c = next;
            next += 1;
            c
        } else {
            i64_vals[g.below(i64_vals.len() as u32) as usize]
        };
        insts.push(Inst::IntBin { ty: IntTy::I64, op: total_binop(g), a: acc, b });
        acc = next;
        next += 1;
        i64_vals.push(acc);
    }

    Func {
        params: vec![ValType::I32],
        results: vec![ValType::I64],
        blocks: vec![Block {
            params: vec![ValType::I32],
            insts,
            term: Terminator::Return(vec![acc]),
        }],
    }
}

/// Build an in-scope durable module: a call chain `func0 → func1 → … → leaf`, of a
/// randomized depth `1..=4`. The entry (func 0) and every wrapper propagates the suspend
/// through a `call`; only the deepest function holds the `cap.call`. At depth 1 this is
/// the original single-frame leaf shape.
pub fn gen_module(g: &mut Gen) -> Module {
    let depth = 1 + g.below(4); // 1..=4 stacked frames
    let funcs: Vec<Func> = (0..depth)
        .map(|i| {
            let suspend = if i == depth - 1 {
                Suspend::Cap
            } else {
                Suspend::Call(i + 1)
            };
            gen_func(g, suspend)
        })
        .collect();

    Module {
        funcs,
        memory: Some(Memory {
            size_log2: SIZE_LOG2,
        }),
        data: Vec::new(),
    }
}

fn read_sp(w: &[u8]) -> u64 {
    let mut b = [0u8; 8];
    b.copy_from_slice(&w[SHADOW_SP_OFF as usize..SHADOW_SP_OFF as usize + 8]);
    u64::from_le_bytes(b)
}

/// Check the two properties on one generated module.
pub fn fuzz_one(g: &mut Gen) {
    let m = gen_module(g);
    let inst = transform_module(&m).expect("an in-scope module must transform");
    svm_verify::verify_module(&inst).expect("instrumented IR must verify");

    let clock_v = g.u64v() as i64;

    // --- (1) inert in NORMAL: instrumented (NORMAL) == un-instrumented ---
    let r_orig = {
        let mut h = Host::new();
        h.clock_ns = clock_v;
        let clk = h.grant_clock();
        let mut fuel = 1_000_000u64;
        run_with_host(&m, 0, &[Value::I32(clk)], &mut fuel, &mut h)
    };
    let (r_base, _) = {
        let mut h = Host::new();
        h.clock_ns = clock_v;
        let clk = h.grant_clock();
        let mut fuel = 1_000_000u64;
        run_capture_reserved_with_host(
            &inst,
            0,
            &[Value::I32(clk)],
            &mut fuel,
            &init_durable_window(WINDOW),
            SIZE_LOG2,
            &mut h,
        )
    };
    assert_eq!(
        r_orig, r_base,
        "instrumentation must be inert in NORMAL state"
    );
    let base = r_base.expect("generated programs are trap-free");

    // --- (2) freeze: the poll after the call unwinds out to the host ---
    let (r_freeze, snap) = {
        let mut h = Host::new();
        h.clock_ns = clock_v; // same initial conditions as the baseline
        let clk = h.grant_clock();
        let mut win = init_durable_window(WINDOW);
        write_state(&mut win, STATE_UNWINDING);
        let mut fuel = 1_000_000u64;
        run_capture_reserved_with_host(
            &inst,
            0,
            &[Value::I32(clk)],
            &mut fuel,
            &win,
            SIZE_LOG2,
            &mut h,
        )
    };
    assert!(r_freeze.is_ok(), "freeze returns a placeholder, not a trap");
    assert_eq!(
        read_state(&snap),
        STATE_UNWINDING,
        "artifact is still UNWINDING (the stack unwound, did not complete)"
    );
    assert!(read_sp(&snap) > SHADOW_BASE, "a shadow frame was pushed");

    // --- (3) thaw on a FRESH host (clock now 0): must reload, not re-issue ---
    let (r_thaw, final_win) = {
        let mut win = snap.clone();
        write_state(&mut win, STATE_REWINDING);
        let mut h = Host::new();
        let clk = h.grant_clock();
        let mut fuel = 1_000_000u64;
        run_capture_reserved_with_host(
            &inst,
            0,
            &[Value::I32(clk)],
            &mut fuel,
            &win,
            SIZE_LOG2,
            &mut h,
        )
    };
    assert_eq!(
        r_thaw,
        Ok(base),
        "thawed run must equal the uninterrupted run (saved cap result reloaded, not re-issued)"
    );
    assert_eq!(
        read_state(&final_win),
        STATE_NORMAL,
        "thaw must flip the state word back to NORMAL"
    );
}
