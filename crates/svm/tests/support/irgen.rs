//! Structured generator of **verifier-valid** IR modules, shared by the stable-CI
//! differential test (`jit_fuzz.rs`) and the libFuzzer `diff` target. It builds
//! well-typed, defined-before-use, **terminating** modules *by construction*:
//!
//! - operands are drawn from a typed value pool (block params + earlier results), with a
//!   fresh constant synthesized when no value of the needed type exists — so every op is
//!   type-correct and references only already-defined values;
//! - block-branch / return arguments are generated to match the target's exact param
//!   types;
//! - the CFG uses **forward-only edges** and the call graph uses **forward-only calls**,
//!   so both are DAGs — no loops, no recursion — and execution always halts.
//!
//! Therefore any interpreter-vs-JIT divergence on a generated module is a real backend
//! bug, not malformed input. Constant values are biased toward boundary cases (0, ±1,
//! INT_MIN/MAX, NaN, ±inf) so div-by-zero, overflow, and bad-conversion traps are hit.

#![allow(dead_code)] // each includer (test / fuzz target) uses a subset

use svm_ir::*;

/// Entropy source: consume the libFuzzer input first (for coverage-guided exploration),
/// then fall back to a deterministic xorshift PRNG (so the stable test is reproducible
/// from a seed with no input bytes).
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
    fn u32v(&mut self) -> u32 {
        let mut v = 0u32;
        for _ in 0..4 {
            v = (v << 8) | self.byte() as u32;
        }
        v
    }
    fn u64v(&mut self) -> u64 {
        ((self.u32v() as u64) << 32) | self.u32v() as u64
    }
    /// A value in `0..n` (0 if `n == 0`).
    fn below(&mut self, n: u32) -> u32 {
        if n == 0 {
            0
        } else {
            self.u32v() % n
        }
    }
    fn boolean(&mut self) -> bool {
        self.byte() & 1 == 1
    }

    fn valtype(&mut self) -> ValType {
        match self.below(4) {
            0 => ValType::I32,
            1 => ValType::I64,
            2 => ValType::F32,
            _ => ValType::F64,
        }
    }
    fn inttype(&mut self) -> IntTy {
        if self.boolean() {
            IntTy::I64
        } else {
            IntTy::I32
        }
    }
    fn floattype(&mut self) -> FloatTy {
        if self.boolean() {
            FloatTy::F64
        } else {
            FloatTy::F32
        }
    }
    fn i32c(&mut self) -> i32 {
        match self.below(8) {
            0 => 0,
            1 => 1,
            2 => -1,
            3 => i32::MIN,
            4 => i32::MAX,
            _ => self.u32v() as i32,
        }
    }
    fn i64c(&mut self) -> i64 {
        match self.below(8) {
            0 => 0,
            1 => 1,
            2 => -1,
            3 => i64::MIN,
            4 => i64::MAX,
            _ => self.u64v() as i64,
        }
    }
    fn f32bits(&mut self) -> u32 {
        match self.below(8) {
            0 => 0,
            1 => 0x3f80_0000, // 1.0
            2 => 0xbf80_0000, // -1.0
            3 => 0x7fc0_0000, // NaN
            4 => 0x7f80_0000, // +inf
            5 => 0xff80_0000, // -inf
            _ => self.u32v(),
        }
    }
    fn f64bits(&mut self) -> u64 {
        match self.below(8) {
            0 => 0,
            1 => 0x3ff0_0000_0000_0000, // 1.0
            2 => 0xbff0_0000_0000_0000, // -1.0
            3 => 0x7ff8_0000_0000_0000, // NaN
            4 => 0x7ff0_0000_0000_0000, // +inf
            5 => 0xfff0_0000_0000_0000, // -inf
            _ => self.u64v(),
        }
    }
}

/// Per-block builder: the typed value pool, the instruction list, and the next SSA index.
struct BB<'g> {
    g: &'g mut Gen,
    insts: Vec<Inst>,
    pool: Vec<(ValIdx, ValType)>,
    nv: u32,
}

impl<'g> BB<'g> {
    fn new(g: &'g mut Gen, params: &[ValType]) -> BB<'g> {
        BB {
            pool: params
                .iter()
                .enumerate()
                .map(|(i, &t)| (i as u32, t))
                .collect(),
            nv: params.len() as u32,
            insts: Vec::new(),
            g,
        }
    }
    fn push(&mut self, inst: Inst, ty: ValType) -> ValIdx {
        let idx = self.nv;
        self.nv += 1;
        self.insts.push(inst);
        self.pool.push((idx, ty));
        idx
    }
    fn push0(&mut self, inst: Inst) {
        self.insts.push(inst);
    }
    fn push_multi(&mut self, inst: Inst, results: &[ValType]) {
        self.insts.push(inst);
        for &t in results {
            let idx = self.nv;
            self.nv += 1;
            self.pool.push((idx, t));
        }
    }
    /// A value of type `ty`: a random one from the pool, or a fresh constant.
    fn want(&mut self, ty: ValType) -> ValIdx {
        let cands: Vec<u32> = self
            .pool
            .iter()
            .filter(|(_, t)| *t == ty)
            .map(|(i, _)| *i)
            .collect();
        if !cands.is_empty() {
            return cands[self.g.below(cands.len() as u32) as usize];
        }
        let inst = match ty {
            ValType::I32 => Inst::ConstI32(self.g.i32c()),
            ValType::I64 => Inst::ConstI64(self.g.i64c()),
            ValType::F32 => Inst::ConstF32(self.g.f32bits()),
            ValType::F64 => Inst::ConstF64(self.g.f64bits()),
        };
        self.push(inst, ty)
    }
    fn edge_args(&mut self, params: &[ValType]) -> Vec<ValIdx> {
        params.iter().map(|&t| self.want(t)).collect()
    }
    fn pick_fwd(&mut self, bi: usize, nblocks: usize) -> usize {
        bi + 1 + self.g.below((nblocks - bi - 1) as u32) as usize
    }
}

const CONVS: [ConvOp; 3] = [ConvOp::ExtendI32S, ConvOp::ExtendI32U, ConvOp::WrapI64];

/// Append one random instruction to `bb`, valid in this context.
fn gen_inst(bb: &mut BB, fi: usize, sigs: &[(Vec<ValType>, Vec<ValType>)], has_mem: bool) {
    let nfuncs = sigs.len();
    let can_call = fi + 1 < nfuncs;
    loop {
        match bb.g.below(17) {
            0 => {
                let ty = bb.g.inttype();
                let op = BinOp::from_index(bb.g.below(15) as u8).unwrap();
                let a = bb.want(ty.val());
                let b = bb.want(ty.val());
                bb.push(Inst::IntBin { ty, op, a, b }, ty.val());
            }
            1 => {
                let ty = bb.g.inttype();
                let op = CmpOp::from_index(bb.g.below(10) as u8).unwrap();
                let a = bb.want(ty.val());
                let b = bb.want(ty.val());
                bb.push(Inst::IntCmp { ty, op, a, b }, ValType::I32);
            }
            2 => {
                let ty = bb.g.inttype();
                let op = IntUnOp::from_index(bb.g.below(6) as u8).unwrap();
                let a = bb.want(ty.val());
                bb.push(Inst::IntUn { ty, op, a }, ty.val());
            }
            3 => {
                let ty = bb.g.inttype();
                let a = bb.want(ty.val());
                bb.push(Inst::Eqz { ty, a }, ValType::I32);
            }
            4 => {
                let op = CONVS[bb.g.below(3) as usize];
                let (_, from, to) = op.sig();
                let a = bb.want(from);
                bb.push(Inst::Convert { op, a }, to);
            }
            5 => {
                let ty = bb.g.valtype();
                let cond = bb.want(ValType::I32);
                let a = bb.want(ty);
                let b = bb.want(ty);
                bb.push(Inst::Select { cond, a, b }, ty);
            }
            6 => {
                let ty = bb.g.floattype();
                let op = FBinOp::from_index(bb.g.below(7) as u8).unwrap();
                let a = bb.want(ty.val());
                let b = bb.want(ty.val());
                bb.push(Inst::FBin { ty, op, a, b }, ty.val());
            }
            7 => {
                let ty = bb.g.floattype();
                let op = FUnOp::from_index(bb.g.below(7) as u8).unwrap();
                let a = bb.want(ty.val());
                bb.push(Inst::FUn { ty, op, a }, ty.val());
            }
            8 => {
                let ty = bb.g.floattype();
                let op = FCmpOp::from_index(bb.g.below(6) as u8).unwrap();
                let a = bb.want(ty.val());
                let b = bb.want(ty.val());
                bb.push(Inst::FCmp { ty, op, a, b }, ValType::I32);
            }
            9 => {
                let op = FToI::from_index(bb.g.below(8) as u8).unwrap();
                let (from, to, _) = op.parts();
                let a = bb.want(from.val());
                if bb.g.boolean() {
                    bb.push(Inst::FToISat { op, a }, to.val());
                } else {
                    bb.push(Inst::FToITrap { op, a }, to.val());
                }
            }
            10 => {
                let op = IToF::from_index(bb.g.below(8) as u8).unwrap();
                let (from, to, _) = op.parts();
                let a = bb.want(from.val());
                bb.push(Inst::IToFConv { op, a }, to.val());
            }
            11 => {
                let op = CastOp::ALL[bb.g.below(6) as usize];
                let (_, from, to) = op.sig();
                let a = bb.want(from);
                bb.push(Inst::Cast { op, a }, to);
            }
            12 => {
                let a = bb.want(ValType::I64);
                let b = bb.want(ValType::I64);
                bb.push(Inst::PtrAdd { a, b }, ValType::I64);
            }
            13 => {
                let to_int = bb.g.boolean();
                let a = bb.want(ValType::I64);
                bb.push(Inst::PtrCast { to_int, a }, ValType::I64);
            }
            14 if has_mem => {
                let op = LoadOp::from_index(bb.g.below(14) as u8).unwrap();
                let (_, rty, _, _) = op.info();
                let addr = bb.want(ValType::I64);
                let offset = bb.g.below(256) as u64;
                bb.push(
                    Inst::Load {
                        op,
                        addr,
                        offset,
                        align: 0,
                    },
                    rty,
                );
            }
            15 if has_mem => {
                let op = StoreOp::from_index(bb.g.below(9) as u8).unwrap();
                let (_, vty, _) = op.info();
                let addr = bb.want(ValType::I64);
                let value = bb.want(vty);
                let offset = bb.g.below(256) as u64;
                bb.push0(Inst::Store {
                    op,
                    addr,
                    value,
                    offset,
                    align: 0,
                });
            }
            16 if can_call => {
                let j = fi + 1 + bb.g.below((nfuncs - fi - 1) as u32) as usize;
                let (cp, cr) = (sigs[j].0.clone(), sigs[j].1.clone());
                let args: Vec<ValIdx> = cp.iter().map(|&t| bb.want(t)).collect();
                bb.push_multi(
                    Inst::Call {
                        func: j as u32,
                        args,
                    },
                    &cr,
                );
            }
            _ => continue, // a mem/call kind that isn't available here — re-roll
        }
        break;
    }
}

fn gen_term(
    bb: &mut BB,
    bi: usize,
    nblocks: usize,
    bparams: &[Vec<ValType>],
    results: &[ValType],
) -> Terminator {
    if bb.g.below(24) == 0 {
        return Terminator::Unreachable;
    }
    let forward = bi + 1 < nblocks;
    match if forward { bb.g.below(4) } else { 0 } {
        0 => Terminator::Return(results.iter().map(|&t| bb.want(t)).collect()),
        1 => {
            let t = bb.pick_fwd(bi, nblocks);
            let args = bb.edge_args(&bparams[t]);
            Terminator::Br {
                target: t as u32,
                args,
            }
        }
        2 => {
            let cond = bb.want(ValType::I32);
            let t1 = bb.pick_fwd(bi, nblocks);
            let then_args = bb.edge_args(&bparams[t1]);
            let t2 = bb.pick_fwd(bi, nblocks);
            let else_args = bb.edge_args(&bparams[t2]);
            Terminator::BrIf {
                cond,
                then_blk: t1 as u32,
                then_args,
                else_blk: t2 as u32,
                else_args,
            }
        }
        _ => {
            let idx = bb.want(ValType::I32);
            let nt = 1 + bb.g.below(3) as usize;
            let targets: Vec<Edge> = (0..nt)
                .map(|_| {
                    let t = bb.pick_fwd(bi, nblocks);
                    (t as u32, bb.edge_args(&bparams[t]))
                })
                .collect();
            let dt = bb.pick_fwd(bi, nblocks);
            let default = (dt as u32, bb.edge_args(&bparams[dt]));
            Terminator::BrTable {
                idx,
                targets,
                default,
            }
        }
    }
}

fn gen_func(g: &mut Gen, fi: usize, sigs: &[(Vec<ValType>, Vec<ValType>)], has_mem: bool) -> Func {
    let (params, results) = sigs[fi].clone();
    let nblocks = 1 + g.below(4) as usize;
    let mut bparams: Vec<Vec<ValType>> = vec![params.clone()];
    for _ in 1..nblocks {
        let k = g.below(3) as usize;
        bparams.push((0..k).map(|_| g.valtype()).collect());
    }
    let mut blocks = Vec::with_capacity(nblocks);
    for bi in 0..nblocks {
        let params_i = bparams[bi].clone();
        let mut bb = BB::new(g, &params_i);
        let ninsts = bb.g.below(8);
        for _ in 0..ninsts {
            gen_inst(&mut bb, fi, sigs, has_mem);
        }
        let term = gen_term(&mut bb, bi, nblocks, &bparams, &results);
        blocks.push(Block {
            params: params_i,
            insts: bb.insts,
            term,
        });
    }
    Func {
        params,
        results,
        blocks,
    }
}

/// Generate a complete, verifier-valid module.
pub fn gen_module(g: &mut Gen) -> Module {
    let nfuncs = 1 + g.below(3) as usize;
    let sigs: Vec<(Vec<ValType>, Vec<ValType>)> = (0..nfuncs)
        .map(|_| {
            let params = (0..g.below(4)).map(|_| g.valtype()).collect();
            let results = (0..g.below(3)).map(|_| g.valtype()).collect();
            (params, results)
        })
        .collect();
    let has_mem = g.boolean();
    let memory = has_mem.then_some(Memory { size_log2: 16 });
    let funcs = (0..nfuncs)
        .map(|fi| gen_func(g, fi, &sigs, has_mem))
        .collect();
    Module { funcs, memory }
}

/// Random argument `Value`s matching `params` (for invoking the entry function). Defined
/// here so the interpreter and JIT receive identical inputs.
pub fn gen_args(g: &mut Gen, params: &[ValType]) -> Vec<svm_interp::Value> {
    use svm_interp::Value;
    params
        .iter()
        .map(|&t| match t {
            ValType::I32 => Value::I32(g.i32c()),
            ValType::I64 => Value::I64(g.i64c()),
            ValType::F32 => Value::F32(f32::from_bits(g.f32bits())),
            ValType::F64 => Value::F64(f64::from_bits(g.f64bits())),
        })
        .collect()
}

// ---- shared differential check (used by jit_fuzz.rs and the libFuzzer `diff` target) ----

use svm_interp::{run_capture, Trap, Value};
use svm_jit::{compile_and_run_capture, JitError, JitOutcome, TrapKind};
use svm_verify::verify_module;

fn to_slot(v: Value) -> i64 {
    match v {
        Value::I32(x) => x as i64,
        Value::I64(x) => x,
        Value::F32(x) => x.to_bits() as i64,
        Value::F64(x) => x.to_bits() as i64,
    }
}
fn from_slot(t: ValType, s: i64) -> Value {
    match t {
        ValType::I32 => Value::I32(s as i32),
        ValType::I64 => Value::I64(s),
        ValType::F32 => Value::F32(f32::from_bits(s as u32)),
        ValType::F64 => Value::F64(f64::from_bits(s as u64)),
    }
}
fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        // NaNs compare equal — the IR does not pin a NaN bit-pattern across backends.
        (Value::F32(x), Value::F32(y)) => x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()),
        (Value::F64(x), Value::F64(y)) => x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()),
        _ => a == b,
    }
}
/// Trap kinds the scalar JIT models (others — fuel/stack/guard — it need not).
fn interp_trap_kind(t: &Trap) -> Option<TrapKind> {
    match t {
        Trap::DivByZero => Some(TrapKind::DivByZero),
        Trap::IntOverflow => Some(TrapKind::IntOverflow),
        Trap::BadConversion => Some(TrapKind::BadConversion),
        Trap::Unreachable => Some(TrapKind::Unreachable),
        Trap::IndirectCallType => Some(TrapKind::IndirectCallType),
        Trap::CapFault => Some(TrapKind::CapFault),
        _ => None,
    }
}

fn is_float(t: ValType) -> bool {
    matches!(t, ValType::F32 | ValType::F64)
}

/// True if `m` involves any floating-point value anywhere. The escape-oracle byte-compares
/// the final window, but the IR does **not** pin NaN bit-patterns across backends (that's
/// why [`values_equal`] is NaN-insensitive), so a computed NaN that reaches memory could
/// differ *legitimately* — a false escape. Confinement is about *addresses*, which integer
/// modules exercise fully, so the memory oracle runs on float-free modules only; float
/// coverage stays at the (NaN-insensitive) value level.
fn has_float(m: &Module) -> bool {
    let any = |ts: &[ValType]| ts.iter().copied().any(is_float);
    m.funcs.iter().any(|f| {
        any(&f.params)
            || any(&f.results)
            || f.blocks.iter().any(|b| {
                any(&b.params)
                    || b.insts.iter().any(|inst| match inst {
                        Inst::ConstF32(_)
                        | Inst::ConstF64(_)
                        | Inst::FBin { .. }
                        | Inst::FUn { .. }
                        | Inst::FCmp { .. }
                        | Inst::FToISat { .. }
                        | Inst::FToITrap { .. }
                        | Inst::IToFConv { .. } => true,
                        Inst::Cast { op, .. } => {
                            let (_, from, to) = op.sig();
                            is_float(from) || is_float(to)
                        }
                        Inst::Load { op, .. } => is_float(op.info().1),
                        Inst::Store { op, .. } => is_float(op.info().1),
                        _ => false,
                    })
            })
    })
}

/// Run `m`'s entry on both backends and assert they agree (result value-equal, or same
/// modelled trap kind) — **and**, for a float-free module with memory, that they leave a
/// byte-identical window (the escape-oracle, §18). Panics with the offending module.
pub fn run_differential(m: &Module, args: &[Value]) {
    let results = m.funcs[0].results.clone();

    // Escape-oracle seed: a non-zero, varied pattern over the window, so a divergent or
    // under-masked load/store shows up as a final-memory mismatch (zero-init could hide a
    // bad read that returns 0). Only for float-free modules (NaN bits aren't pinned across
    // backends), and capped so a huge *declared* window doesn't allocate here (the JIT also
    // rejects windows above its backing cap → `Unsupported`, skipped below).
    let mem_oracle = !has_float(m) && matches!(m.memory, Some(mc) if mc.size_log2 <= 20);
    let init: Vec<u8> = if mem_oracle {
        let size = 1usize << m.memory.unwrap().size_log2;
        (0..size)
            .map(|i| (i as u8).wrapping_mul(31) ^ 0xa5)
            .collect()
    } else {
        Vec::new()
    };

    let mut fuel = 5_000_000u64;
    let (interp, imem) = run_capture(m, 0, args, &mut fuel, &init);
    let slots: Vec<i64> = args.iter().copied().map(to_slot).collect();
    let (jit, jmem) = match compile_and_run_capture(m, 0, &slots, &init) {
        Ok(o) => o,
        Err(JitError::Unsupported(_)) => return, // generator only emits lowered ops; be safe
        Err(e) => panic!("JIT failed to compile a verified module: {e:?}\n{m:#?}"),
    };
    match (interp, jit) {
        (Ok(want), JitOutcome::Returned(s)) => {
            let got: Vec<Value> = results
                .iter()
                .zip(s)
                .map(|(t, v)| from_slot(*t, v))
                .collect();
            assert_eq!(want.len(), got.len(), "result arity\n{m:#?}");
            for (w, g) in want.iter().zip(&got) {
                assert!(
                    values_equal(w, g),
                    "interp/JIT disagree: {want:?} vs {got:?}\n{m:#?}"
                );
            }
            // Escape-oracle: both ran to completion, so the interpreter (the masking
            // reference, §4) confined every access to `[0, size)` — the JIT, lowering the
            // same masking arithmetic on the same inputs, must therefore leave an identical
            // window. A mismatch means a JIT access escaped or was mis-masked.
            if mem_oracle {
                if let Some(i) = imem.iter().zip(&jmem).position(|(a, b)| a != b) {
                    panic!(
                        "escape-oracle: interp/JIT final memory differs at byte {i} \
                         (interp={:#04x} jit={:#04x}) — an access not masked into [0,size)\n{m:#?}",
                        imem[i], jmem[i]
                    );
                }
                assert_eq!(imem.len(), jmem.len(), "window snapshot length\n{m:#?}");
            }
        }
        (Err(trap), JitOutcome::Trapped(kind)) => {
            if let Some(want) = interp_trap_kind(&trap) {
                assert_eq!(
                    kind, want,
                    "trap kind: JIT {kind:?} vs interp {trap:?}\n{m:#?}"
                );
            }
        }
        (Err(trap), JitOutcome::Returned(_)) => assert!(
            interp_trap_kind(&trap).is_none(),
            "interp trapped {trap:?} but JIT returned\n{m:#?}"
        ),
        (i, j) => panic!("outcome mismatch: {i:?} vs {j:?}\n{m:#?}"),
    }
}

/// Generate one module from `g`, verify it, and differential-test it. The single entry
/// point shared by the stable seed loop and the libFuzzer target.
pub fn fuzz_one(g: &mut Gen) {
    let m = gen_module(g);
    verify_module(&m).unwrap_or_else(|e| panic!("generator emitted invalid IR: {e:?}\n{m:#?}"));
    let params = m.funcs[0].params.clone();
    let args = gen_args(g, &params);
    run_differential(&m, &args);
}
