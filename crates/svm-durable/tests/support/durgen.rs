//! Generative property for the freeze/thaw transform (DURABILITY.md §7/§12.6, R11).
//!
//! Shared between the stable `cargo test` (`tests/durable_fuzz.rs`) and the libFuzzer
//! targets (`fuzz/fuzz_targets/durable.rs`, `durable_fiber.rs`), mirroring `irgen`. Two
//! generators, each emitting **in-scope** durable modules so the properties exercise a real
//! input space instead of the arbitrary-IR generator (which the transform would reject):
//!
//!   * [`gen_module`] / [`fuzz_one`] — call-chain modules (leaf `cap.call` / propagated `Call`,
//!     1..=4 frames, multi-point, multi-block);
//!   * [`gen_fiber_module`] / [`fuzz_fiber_one`] — root+fiber modules (§12.8 Phase 3.1): a root
//!     resuming one fiber that `suspend`s 1..=3 times, values live across each suspend.
//!
//! Each module is checked for two properties:
//!   1. **inert in NORMAL** — the instrumented module run in `NORMAL` state produces
//!      the same result as the original, un-instrumented module;
//!   2. **round-trip** — freeze → thaw equals the uninterrupted run on a *fresh* host (a buggy
//!      re-issue of the `cap.call` / `cont.resume` instead of reloading/redelivering would diverge).

#![allow(dead_code)] // not every helper is used by both includers

use svm_durable::{
    arm_freeze_after, begin_thaw, init_durable_window, read_state, read_thaw_state,
    transform_module, write_state, SHADOW_BASE, STATE_NORMAL, STATE_UNWINDING,
};
use svm_interp::{run_capture_reserved_with_host, run_with_host, Host, Value};
use svm_ir::{
    BinOp, Block, CmpOp, Func, FuncType, Inst, IntTy, Memory, Module, Terminator, ValType,
};

// 128 KiB: the durable region needs `DURABLE_RESERVE` (64 KiB), and a smaller window keeps the
// per-run commit footprint modest — the JIT commits a window per compile, and on a memory-tight
// Windows CI runner the cumulative commit of many compiles can hit the limit (os error 1455).
pub const SIZE_LOG2: u8 = 17;
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

/// What a generated function suspends on.
enum Suspend {
    /// Leaf: `npoints` sequential `cap.call`s to the clock (the deepest frame). `npoints`
    /// > 1 gives a single function multiple resume points (multi-arm `br_table`).
    Cap { npoints: u32 },
    /// Propagated: a single `call` to function `callee` (a deeper may-suspend function).
    Call(u32),
}

/// Append a few i64 consts / total binops to `insts`, tracking value indices in
/// `i64_vals` / `next`. Used for both the prefix and the inter-/post-suspend suffixes.
fn gen_straightline(
    g: &mut Gen,
    insts: &mut Vec<Inst>,
    i64_vals: &mut Vec<u32>,
    next: &mut u32,
    acc: u32,
) {
    for _ in 0..g.below(4) {
        let b = if i64_vals.len() < 2 || g.below(2) == 0 {
            insts.push(Inst::ConstI64(g.u64v() as i64));
            let c = *next;
            *next += 1;
            i64_vals.push(c);
            c
        } else {
            i64_vals[g.below(i64_vals.len() as u32) as usize]
        };
        insts.push(Inst::IntBin {
            ty: IntTy::I64,
            op: total_binop(g),
            a: acc,
            b,
        });
        let r = *next;
        *next += 1;
        i64_vals.push(r);
    }
}

/// Append the suspend op(s) + their folding/suffix to `insts`, given the handle at value
/// index 0 and a starting accumulator `acc`. Returns the final accumulator. Each op's i64
/// result is folded into `acc`, so every saved/reloaded value is exercised.
fn emit_suspend_body(
    g: &mut Gen,
    suspend: &Suspend,
    insts: &mut Vec<Inst>,
    i64_vals: &mut Vec<u32>,
    next: &mut u32,
    mut acc: u32,
) -> u32 {
    match *suspend {
        Suspend::Cap { npoints } => {
            for _ in 0..npoints {
                insts.push(Inst::ConstI32(g.u64v() as i32)); // the i32 clock arg
                let arg = *next;
                *next += 1;
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
                let cap_result = *next;
                *next += 1;
                i64_vals.push(cap_result);
                insts.push(Inst::IntBin {
                    ty: IntTy::I64,
                    op: total_binop(g),
                    a: acc,
                    b: cap_result,
                });
                acc = *next;
                *next += 1;
                i64_vals.push(acc);
                gen_straightline(g, insts, i64_vals, next, acc);
                acc = *i64_vals.last().unwrap();
            }
        }
        Suspend::Call(callee) => {
            insts.push(Inst::Call {
                func: callee,
                args: vec![0],
            }); // pass the handle down
            let call_result = *next;
            *next += 1;
            i64_vals.push(call_result);
            insts.push(Inst::IntBin {
                ty: IntTy::I64,
                op: total_binop(g),
                a: acc,
                b: call_result,
            });
            acc = *next;
            *next += 1;
            i64_vals.push(acc);
            gen_straightline(g, insts, i64_vals, next, acc);
            acc = *i64_vals.last().unwrap();
        }
    }
    acc
}

/// Build one `func (i32) -> (i64)`. The single param `v0` is the clock handle, threaded as
/// the call/`cap.call` argument. When `split`, the prefix lands in the entry block and the
/// suspend body in a *non-entry* block (reached by an unconditional branch carrying the
/// handle + accumulator as block params) — exercising the multi-block transform: the live
/// values cross as branch args and must be spilled/reloaded, not recovered from the entry.
fn gen_func(g: &mut Gen, suspend: Suspend, split: bool) -> Func {
    if !split {
        let mut insts: Vec<Inst> = Vec::new();
        let mut i64_vals: Vec<u32> = Vec::new();
        let mut next: u32 = 1; // v0 is the i32 handle param
        insts.push(Inst::ConstI64(g.u64v() as i64)); // seed the accumulator
        let mut acc = 1;
        next += 1;
        i64_vals.push(acc);
        gen_straightline(g, &mut insts, &mut i64_vals, &mut next, acc);
        acc = *i64_vals.last().unwrap();
        acc = emit_suspend_body(g, &suspend, &mut insts, &mut i64_vals, &mut next, acc);
        return Func {
            params: vec![ValType::I32],
            results: vec![ValType::I64],
            blocks: vec![Block {
                params: vec![ValType::I32],
                insts,
                term: Terminator::Return(vec![acc]),
            }],
        };
    }

    // Entry block: prefix → accumulator, then branch to block1 carrying [handle, acc].
    let mut b0: Vec<Inst> = Vec::new();
    let mut i64_vals = vec![1u32];
    let mut next = 2u32; // v0 = handle param, v1 = the seed const below
    b0.push(Inst::ConstI64(g.u64v() as i64));
    let mut acc = 1u32;
    gen_straightline(g, &mut b0, &mut i64_vals, &mut next, acc);
    acc = *i64_vals.last().unwrap();
    let entry = Block {
        params: vec![ValType::I32],
        insts: b0,
        term: Terminator::Br {
            target: 1,
            args: vec![0, acc],
        },
    };

    // block1(handle: i32, acc: i64): the suspend body, then return. Value indices restart:
    // v0 = handle, v1 = acc.
    let mut b1: Vec<Inst> = Vec::new();
    let mut i64_vals = vec![1u32];
    let mut next = 2u32;
    let acc1 = emit_suspend_body(g, &suspend, &mut b1, &mut i64_vals, &mut next, 1);
    let body = Block {
        params: vec![ValType::I32, ValType::I64],
        insts: b1,
        term: Terminator::Return(vec![acc1]),
    };

    Func {
        params: vec![ValType::I32],
        results: vec![ValType::I64],
        blocks: vec![entry, body],
    }
}

/// Build an in-scope durable module: a call chain `func0 → func1 → … → leaf`, of a
/// randomized depth `1..=4`. Every wrapper propagates the suspend through a `call`; only
/// the deepest function holds the `cap.call`(s) — `1..=3` of them, so the leaf exercises
/// multiple resume points. At depth 1 / one point this is the original single-frame shape.
pub fn gen_module(g: &mut Gen) -> Module {
    let depth = 1 + g.below(4); // 1..=4 stacked frames
    let leaf_points = 1 + g.below(3); // 1..=3 resume points in the leaf
    let funcs: Vec<Func> = (0..depth)
        .map(|i| {
            let suspend = if i == depth - 1 {
                Suspend::Cap {
                    npoints: leaf_points,
                }
            } else {
                Suspend::Call(i + 1)
            };
            // ~half the functions split their body across two blocks, so the suspend op
            // lands in a non-entry block (multi-block segmentation + branch remapping).
            let split = g.below(2) == 0;
            gen_func(g, suspend, split)
        })
        .collect();

    Module {
        funcs,
        memory: Some(Memory {
            size_log2: SIZE_LOG2,
        }),
        data: Vec::new(),
        imports: Vec::new(),
        exports: Vec::new(),
        debug_info: None,
    }
}

/// Build an in-scope module with a **poll-free loop ahead of the `cap.call`** — exercising the
/// Phase-4 Slice A loop-header back-edge poll (`SuspendKind::LoopHeader`). `block1` is a loop header
/// (a back-edge target) whose body is pure compute (consts + total binops + an `i+1 < trip`
/// counter), so the only freeze site inside it is the inserted header poll; the `cap.call` lands in
/// `block2` *after* the loop, so a freeze-from-start lands on the header poll first. The handle is a
/// loop-carried block param (an `i32` spilled/reloaded across the header), so this also covers
/// mixed-type header-param spilling. Oracle: `seed (op step)×trip`, then folded with the clock.
pub fn gen_loop_module(g: &mut Gen) -> Module {
    let trip = 1 + g.below(8) as i64; // 1..=8 bounded iterations (poll-free, always terminates)
    let seed = g.u64v() as i64;
    let step = g.u64v() as i64;
    let body_op = total_binop(g);
    let fold_op = total_binop(g);
    let clock_arg = g.u64v() as i32;

    // block0(v0: i32 handle): seed the accumulator + counter, enter the loop carrying the handle.
    let b0 = Block {
        params: vec![ValType::I32],
        insts: vec![
            Inst::ConstI64(seed), // v1 = acc
            Inst::ConstI64(0),    // v2 = i
        ],
        term: Terminator::Br {
            target: 1,
            args: vec![0, 1, 2], // [handle, acc, i]
        },
    };

    // block1(v0: i32, v1: i64 acc, v2: i64 i): poll-free body; back-edge to self or exit.
    let b1 = Block {
        params: vec![ValType::I32, ValType::I64, ValType::I64],
        insts: vec![
            Inst::ConstI64(step), // v3
            Inst::IntBin {
                ty: IntTy::I64,
                op: body_op,
                a: 1,
                b: 3,
            }, // v4 = acc `op` step
            Inst::ConstI64(1),    // v5
            Inst::IntBin {
                ty: IntTy::I64,
                op: BinOp::Add,
                a: 2,
                b: 5,
            }, // v6 = i + 1
            Inst::ConstI64(trip), // v7
            Inst::IntCmp {
                ty: IntTy::I64,
                op: CmpOp::LtS,
                a: 6,
                b: 7,
            }, // v8 = (i+1) < trip
        ],
        term: Terminator::BrIf {
            cond: 8,
            then_blk: 1,
            then_args: vec![0, 4, 6], // continue: [handle, acc', i']
            else_blk: 2,
            else_args: vec![0, 4], // exit: [handle, acc']
        },
    };

    // block2(v0: i32 handle, v1: i64 acc): the `cap.call` after the loop, folded into the result.
    let b2 = Block {
        params: vec![ValType::I32, ValType::I64],
        insts: vec![
            Inst::ConstI32(clock_arg), // v2 = the i32 clock arg
            Inst::CapCall {
                type_id: CLOCK_TYPE_ID,
                op: CLOCK_OP,
                sig: FuncType {
                    params: vec![ValType::I32],
                    results: vec![ValType::I64],
                },
                handle: 0, // v0
                args: vec![2],
            }, // v3 = clock
            Inst::IntBin {
                ty: IntTy::I64,
                op: fold_op,
                a: 1,
                b: 3,
            }, // v4 = acc `op` clock
        ],
        term: Terminator::Return(vec![4]),
    };

    Module {
        funcs: vec![Func {
            params: vec![ValType::I32],
            results: vec![ValType::I64],
            blocks: vec![b0, b1, b2],
        }],
        memory: Some(Memory {
            size_log2: SIZE_LOG2,
        }),
        data: Vec::new(),
        imports: Vec::new(),
        exports: Vec::new(),
        debug_info: None,
    }
}

// §12.8 4A.5: the root context's shadow-SP word is the first 8 bytes of its region (at `SHADOW_BASE`),
// not the legacy global `SHADOW_SP_OFF`. Frames grow just past it, so an empty root stack reads
// `SHADOW_BASE + 8`.
fn read_sp(w: &[u8]) -> u64 {
    let mut b = [0u8; 8];
    b.copy_from_slice(&w[SHADOW_BASE as usize..SHADOW_BASE as usize + 8]);
    u64::from_le_bytes(b)
}

/// The empty (no-frames) root shadow-SP under the 4A.5 per-context layout.
const ROOT_EMPTY_SP: u64 = SHADOW_BASE + 8;

// ---- Fiber generator + freeze/thaw property (Phase 3.1 hardening) ----
//
// A fiber'd module the §12.8 single-fiber freeze/thaw path round-trips: a root that creates one
// fiber and resumes it `S+1` times, and a fiber that `suspend`s `S` times then returns. Pure
// arithmetic (no caps) so the run is deterministic — freeze→thaw equals the uninterrupted run with
// no clock bookkeeping. Both functions are single-block multi-point (the transform splits each at
// its `cont.resume` / `suspend` ops). The fiber keeps its entry `arg` live across *every* suspend
// (used in the final result), so each suspend point spills/reloads a live value; the root keeps the
// fiber handle live across *every* resume, so each resume point reloads it for the re-issue.

/// The fiber `(i64 sp, i64 arg) -> (i64)`: suspend `S` times, folding `arg` in each round (so it is
/// live across all of them), then return.
fn gen_fiber_func(g: &mut Gen, suspends: u32) -> Func {
    let arg = 1u32; // v1 — kept live to the end
    let mut insts: Vec<Inst> = Vec::new();
    let mut next = 2u32; // v0 = sp, v1 = arg
    insts.push(Inst::ConstI64(g.u64v() as i64));
    let c0 = next;
    next += 1;
    insts.push(Inst::IntBin {
        ty: IntTy::I64,
        op: total_binop(g),
        a: c0,
        b: arg,
    });
    let mut acc = next;
    next += 1;
    for _ in 0..suspends {
        insts.push(Inst::Suspend { value: acc });
        let r = next; // the next resume's delivered value
        next += 1;
        insts.push(Inst::IntBin {
            ty: IntTy::I64,
            op: total_binop(g),
            a: r,
            b: arg, // re-use `arg` → it stays live across this suspend
        });
        acc = next;
        next += 1;
    }
    insts.push(Inst::ConstI64(g.u64v() as i64));
    let cf = next;
    next += 1;
    insts.push(Inst::IntBin {
        ty: IntTy::I64,
        op: total_binop(g),
        a: acc,
        b: cf,
    });
    let ret = next;
    Func {
        params: vec![ValType::I64, ValType::I64],
        results: vec![ValType::I64],
        blocks: vec![Block {
            params: vec![ValType::I64, ValType::I64],
            insts,
            term: Terminator::Return(vec![ret]),
        }],
    }
}

/// The root `() -> (i64)`: create fiber (func 1), resume it `S+1` times threading the handle, and
/// sum every delivered value. The handle is live across every resume (spilled at each point).
fn gen_fiber_root(g: &mut Gen, suspends: u32) -> Func {
    let resumes = suspends + 1;
    let mut insts: Vec<Inst> = Vec::new();
    let mut next = 0u32;
    insts.push(Inst::RefFunc { func: 1 });
    let vf = next;
    next += 1;
    insts.push(Inst::ConstI64(4096)); // the fiber's data-stack base (unused by the interp)
    let vsp = next;
    next += 1;
    insts.push(Inst::ContNew { func: vf, sp: vsp });
    let k = next;
    next += 1;
    let mut vals: Vec<u32> = Vec::new();
    for _ in 0..resumes {
        insts.push(Inst::ConstI64(g.u64v() as i64));
        let a = next;
        next += 1;
        insts.push(Inst::ContResume { k, arg: a });
        next += 1; // status (i32)
        let val = next; // delivered value (i64)
        next += 1;
        vals.push(val);
    }
    let mut acc = vals[0];
    for &v in &vals[1..] {
        insts.push(Inst::IntBin {
            ty: IntTy::I64,
            op: total_binop(g),
            a: acc,
            b: v,
        });
        acc = next;
        next += 1;
    }
    Func {
        params: vec![],
        results: vec![ValType::I64],
        blocks: vec![Block {
            params: vec![],
            insts,
            term: Terminator::Return(vec![acc]),
        }],
    }
}

/// A fiber'd durable module: `[root, fiber]`, fiber suspends `1..=3` times.
pub fn gen_fiber_module(g: &mut Gen) -> Module {
    let suspends = 1 + g.below(3); // 1..=3
    Module {
        funcs: vec![gen_fiber_root(g, suspends), gen_fiber_func(g, suspends)],
        memory: Some(Memory {
            size_log2: SIZE_LOG2,
        }),
        data: Vec::new(),
        imports: Vec::new(),
        exports: Vec::new(),
        debug_info: None,
    }
}

/// The §12.8 single-fiber freeze/thaw property: instrumentation is inert in NORMAL, and freezing
/// (with the driver flattening the parked fiber + exporting its `FrozenFiber` residue) then thawing
/// (re-seeding the residue + re-entering under REWINDING) reproduces the uninterrupted run.
pub fn fuzz_fiber_one(g: &mut Gen) {
    let m = gen_fiber_module(g);
    let inst = transform_module(&m).expect("an in-scope fiber module must transform");
    svm_verify::verify_module(&inst).expect("instrumented fiber IR must verify");

    // (1) inert in NORMAL: un-instrumented == instrumented (NORMAL).
    let r_orig = {
        let mut h = Host::new();
        let mut fuel = 1_000_000u64;
        run_with_host(&m, 0, &[], &mut fuel, &mut h)
    };
    let r_base = {
        let mut h = Host::new();
        h.set_durable(true);
        let mut fuel = 1_000_000u64;
        let (r, _) = run_capture_reserved_with_host(
            &inst,
            0,
            &[],
            &mut fuel,
            &init_durable_window(WINDOW),
            SIZE_LOG2,
            &mut h,
        );
        r
    };
    assert_eq!(r_orig, r_base, "instrumentation must be inert in NORMAL");
    let base = r_base.expect("generated fiber programs are trap-free");

    // (2) freeze: UNWINDING from the start unwinds the root at resume #1 (fiber parked), then the
    // driver flattens the fiber. Capture the window + the exported fiber residue.
    let (r_freeze, snap, frozen) = {
        let mut h = Host::new();
        h.set_durable(true);
        let mut win = init_durable_window(WINDOW);
        write_state(&mut win, STATE_UNWINDING);
        let mut fuel = 1_000_000u64;
        let (r, snap) =
            run_capture_reserved_with_host(&inst, 0, &[], &mut fuel, &win, SIZE_LOG2, &mut h);
        (r, snap, h.frozen_fibers().to_vec())
    };
    assert!(r_freeze.is_ok(), "freeze returns a placeholder, not a trap");
    assert_eq!(frozen.len(), 1, "the single fiber was flattened");
    assert!(
        read_sp(&snap) >= ROOT_EMPTY_SP,
        "the root's shadow-SP is in-reserve"
    );

    // (3) thaw: re-seed the fiber residue, flip to REWINDING, re-enter; must equal the baseline.
    let (r_thaw, final_win) = {
        let mut win = snap.clone();
        begin_thaw(&mut win, 0);
        let mut h = Host::new();
        h.set_durable(true);
        h.set_frozen_fibers(frozen);
        let mut fuel = 1_000_000u64;
        run_capture_reserved_with_host(&inst, 0, &[], &mut fuel, &win, SIZE_LOG2, &mut h)
    };
    assert_eq!(
        r_thaw,
        Ok(base),
        "thawed fiber run must equal the uninterrupted run"
    );
    assert_eq!(
        read_thaw_state(&final_win, 0),
        STATE_NORMAL,
        "thaw must flip the state word back to NORMAL"
    );
}

// ---- Recycling churn generator + freeze/thaw property (recycling step 4) ----
//
// A fiber'd module that **recycles a slot before freezing**: the root creates + finishes
// `throwaways` (1..=3) fibers — each reuses the lowest free slot (slot 0) and returns without
// suspending, bumping that slot's generation — so the *next* `cont.new` lands the real fiber B in
// slot 0 at generation `throwaways`. The freeze is then armed (`arm_freeze_after`) to land with B
// parked after its first suspend, so the residue carries a **recycled** (generation > 0) fiber. This
// is the path the freeze-before-start harness can't reach (a recycled parked fiber needs a prior
// fiber-finish, i.e. a prior safepoint). A = func 2 (finishes), B = func 1 (the §12.8 fiber).

/// A generated recycling-churn module plus the freeze arming the generator computed for it.
pub struct RecycleModule {
    pub module: Module,
    /// `arm_freeze_after` count that lands the freeze with B parked after its first suspend:
    /// `throwaways` resume safepoints (the throwaways finish, no suspend) + B's resume#1 + B's
    /// suspend#1.
    pub arm: i64,
    /// B's slot generation at the freeze (== `throwaways`): slot 0 was recycled that many times.
    pub generation: u64,
}

/// The root for a recycling-churn module `() -> (i64)` (see the section comment): create+finish
/// `throwaways` fibers (func 2), then create + resume the real fiber B (func 1) `suspends + 1`
/// times, summing every delivered value (the throwaways' returns + B's), all live across the
/// resumes so the freeze/thaw must reload them.
fn gen_recycle_root(g: &mut Gen, throwaways: u32, suspends: u32) -> Func {
    let mut insts: Vec<Inst> = Vec::new();
    let mut next = 0u32;
    insts.push(Inst::RefFunc { func: 2 }); // A: finishes immediately (recycles its slot)
    let v_fa = next;
    next += 1;
    insts.push(Inst::RefFunc { func: 1 }); // B: the real fiber
    let v_fb = next;
    next += 1;
    insts.push(Inst::ConstI64(4096)); // shared data-stack base (unused by the interp)
    let v_sp = next;
    next += 1;

    let mut vals: Vec<u32> = Vec::new();
    // Throwaway cycles: each creates A in the lowest free slot (slot 0) and resumes it once; A
    // returns without suspending, so the slot frees and its generation bumps before the next cycle.
    for _ in 0..throwaways {
        insts.push(Inst::ContNew {
            func: v_fa,
            sp: v_sp,
        });
        let k = next;
        next += 1;
        insts.push(Inst::ConstI64(g.u64v() as i64));
        let a = next;
        next += 1;
        insts.push(Inst::ContResume { k, arg: a });
        next += 1; // status (i32)
        let val = next; // delivered value (A's return)
        next += 1;
        vals.push(val);
    }
    // The real fiber B reuses slot 0 at generation `throwaways`; resume it `suspends + 1` times.
    insts.push(Inst::ContNew {
        func: v_fb,
        sp: v_sp,
    });
    let kb = next;
    next += 1;
    for _ in 0..(suspends + 1) {
        insts.push(Inst::ConstI64(g.u64v() as i64));
        let a = next;
        next += 1;
        insts.push(Inst::ContResume { k: kb, arg: a });
        next += 1; // status (i32)
        let val = next; // delivered value (i64)
        next += 1;
        vals.push(val);
    }
    // Sum every delivered value (kept live across all the resumes).
    let mut acc = vals[0];
    for &v in &vals[1..] {
        insts.push(Inst::IntBin {
            ty: IntTy::I64,
            op: total_binop(g),
            a: acc,
            b: v,
        });
        acc = next;
        next += 1;
    }
    Func {
        params: vec![],
        results: vec![ValType::I64],
        blocks: vec![Block {
            params: vec![],
            insts,
            term: Terminator::Return(vec![acc]),
        }],
    }
}

/// A recycling-churn module `[root, fiberB, fiberA]` (B suspends `1..=3` times; the slot is
/// recycled `1..=3` times first), with the freeze arming + expected generation the driver needs.
pub fn gen_recycle_fiber_module(g: &mut Gen) -> RecycleModule {
    let throwaways = 1 + g.below(3); // 1..=3 recycle cycles → B lands at generation `throwaways`
    let suspends = 1 + g.below(3); // 1..=3 (B's suspend count)
    let root = gen_recycle_root(g, throwaways, suspends);
    let fiber_b = gen_fiber_func(g, suspends); // func 1
    let fiber_a = gen_fiber_func(g, 0); // func 2: zero suspends ⇒ finishes immediately
    RecycleModule {
        module: Module {
            funcs: vec![root, fiber_b, fiber_a],
            memory: Some(Memory {
                size_log2: SIZE_LOG2,
            }),
            data: Vec::new(),
            imports: Vec::new(),
            exports: Vec::new(),
            debug_info: None,
        },
        arm: throwaways as i64 + 2,
        generation: throwaways as u64,
    }
}

/// The recycling freeze/thaw property (recycling step 4, interpreter): a slot is recycled to
/// generation `throwaways`, the real fiber reuses it, and the **mid-run** armed freeze flattens that
/// recycled (generation > 0) parked fiber; the residue carries the generation, and the thaw re-seeds
/// + re-enters to reproduce the uninterrupted run.
pub fn fuzz_recycle_fiber_one(g: &mut Gen) {
    let RecycleModule {
        module: m,
        arm,
        generation,
    } = gen_recycle_fiber_module(g);
    let inst = transform_module(&m).expect("an in-scope recycling fiber module must transform");
    svm_verify::verify_module(&inst).expect("instrumented recycling IR must verify");

    // (1) inert in NORMAL: un-instrumented == instrumented (NORMAL).
    let r_orig = {
        let mut h = Host::new();
        let mut fuel = 1_000_000u64;
        run_with_host(&m, 0, &[], &mut fuel, &mut h)
    };
    let r_base = {
        let mut h = Host::new();
        h.set_durable(true);
        let mut fuel = 1_000_000u64;
        let (r, _) = run_capture_reserved_with_host(
            &inst,
            0,
            &[],
            &mut fuel,
            &init_durable_window(WINDOW),
            SIZE_LOG2,
            &mut h,
        );
        r
    };
    assert_eq!(r_orig, r_base, "instrumentation must be inert in NORMAL");
    let base = r_base.expect("generated recycling programs are trap-free");

    // (2) freeze, armed mid-run so the recycled fiber B is parked when the root unwinds.
    let (r_freeze, snap, frozen) = {
        let mut h = Host::new();
        h.set_durable(true);
        let mut win = init_durable_window(WINDOW);
        arm_freeze_after(&mut win, arm);
        let mut fuel = 1_000_000u64;
        let (r, snap) =
            run_capture_reserved_with_host(&inst, 0, &[], &mut fuel, &win, SIZE_LOG2, &mut h);
        (r, snap, h.frozen_fibers().to_vec())
    };
    assert!(r_freeze.is_ok(), "freeze returns a placeholder, not a trap");
    assert_eq!(
        read_state(&snap),
        STATE_UNWINDING,
        "the armed run promoted itself to UNWINDING at the recycled fiber's suspend"
    );
    assert_eq!(frozen.len(), 1, "the recycled fiber B was flattened");
    assert_eq!(
        (frozen[0].slot, frozen[0].generation),
        (0, generation),
        "B = recycled slot 0 at the bumped generation"
    );

    // (3) thaw: re-seed the residue (at its generation), re-enter under REWINDING; equals baseline.
    let (r_thaw, final_win) = {
        let mut win = snap.clone();
        begin_thaw(&mut win, 0);
        let mut h = Host::new();
        h.set_durable(true);
        h.set_frozen_fibers(frozen);
        let mut fuel = 1_000_000u64;
        run_capture_reserved_with_host(&inst, 0, &[], &mut fuel, &win, SIZE_LOG2, &mut h)
    };
    assert_eq!(
        r_thaw,
        Ok(base),
        "thawed recycled-fiber run must equal the uninterrupted run"
    );
    assert_eq!(
        read_thaw_state(&final_win, 0),
        STATE_NORMAL,
        "thaw must flip the state word back to NORMAL"
    );
}

/// Check the two properties on one generated module.
/// `gen_module` call-chain round-trip property.
pub fn fuzz_one(g: &mut Gen) {
    let m = gen_module(g);
    let clock_v = g.u64v() as i64;
    check_roundtrip(&m, clock_v);
}

/// `gen_loop_module` poll-free-loop round-trip property (Phase-4 Slice A back-edge poll): the
/// freeze-from-start lands on the inserted loop-header poll, and the thaw re-enters the loop body.
pub fn fuzz_loop_one(g: &mut Gen) {
    let m = gen_loop_module(g);
    let clock_v = g.u64v() as i64;
    check_roundtrip(&m, clock_v);
}

/// The shared freeze/thaw round-trip property over an in-scope durable module: inert in NORMAL,
/// freeze-from-start unwinds out to the host, and thaw on a continued-clock host equals the
/// uninterrupted run (a buggy re-issue instead of reload/redeliver would diverge).
fn check_roundtrip(m: &Module, clock_v: i64) {
    let inst = transform_module(m).expect("an in-scope module must transform");
    svm_verify::verify_module(&inst).expect("instrumented IR must verify");

    // --- (1) inert in NORMAL: instrumented (NORMAL) == un-instrumented ---
    let r_orig = {
        let mut h = Host::new();
        h.clock_ns = clock_v;
        let clk = h.grant_clock();
        let mut fuel = 1_000_000u64;
        run_with_host(m, 0, &[Value::I32(clk)], &mut fuel, &mut h)
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

    // --- (2) freeze: unwinding from the start, the poll after the first suspend point
    // unwinds out to the host. Record how far the clock advanced (the suspend points
    // reached during freeze each consumed a tick). ---
    let (r_freeze, snap, clock_after) = {
        let mut h = Host::new();
        h.clock_ns = clock_v; // same initial conditions as the baseline
        let clk = h.grant_clock();
        let mut win = init_durable_window(WINDOW);
        write_state(&mut win, STATE_UNWINDING);
        let mut fuel = 1_000_000u64;
        let (r, snap) = run_capture_reserved_with_host(
            &inst,
            0,
            &[Value::I32(clk)],
            &mut fuel,
            &win,
            SIZE_LOG2,
            &mut h,
        );
        (r, snap, h.clock_ns)
    };
    assert!(r_freeze.is_ok(), "freeze returns a placeholder, not a trap");
    assert_eq!(
        read_state(&snap),
        STATE_UNWINDING,
        "artifact is still UNWINDING (the stack unwound, did not complete)"
    );
    assert!(read_sp(&snap) > ROOT_EMPTY_SP, "a shadow frame was pushed");

    // --- (3) thaw on a fresh host whose clock *continues* from the freeze (D-scope: the
    // host clock is not in the artifact). The frozen suspend point's result must be
    // reloaded — a re-issue would consume the next tick and diverge — while any later
    // suspend points re-perform against the continued clock, matching the baseline. ---
    let (r_thaw, final_win) = {
        let mut win = snap.clone();
        begin_thaw(&mut win, 0);
        let mut h = Host::new();
        h.clock_ns = clock_after;
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
        "thawed run must equal the uninterrupted run (frozen result reloaded, not re-issued)"
    );
    assert_eq!(
        read_thaw_state(&final_win, 0),
        STATE_NORMAL,
        "thaw must flip the state word back to NORMAL"
    );
}
