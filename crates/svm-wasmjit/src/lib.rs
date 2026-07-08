//! **SVM IR → WebAssembly emitter** — slice 1 of the browser wasm-JIT tier (`BROWSER.md`
//! § "wasm-JIT tier — design & implementation plan").
//!
//! Compiles a verified [`svm_ir::Module`]'s functions into one WebAssembly module (binary bytes,
//! hand-encoded — no dependencies, like the other escape-TCB-adjacent crates), so hot guest compute
//! can run on a wasm engine's optimizing tiers instead of the bytecode dispatch loop. **Fail-closed
//! like `svm-jit`**: anything outside the supported subset returns [`Error::Unsupported`] and the
//! caller keeps the function on the bytecode-interpreter tier.
//!
//! ## Emitted shape
//!
//! One exported wasm function `f{i}` per SVM function `i`. Each takes two prepended environment
//! params ahead of the SVM signature:
//!
//! - `win: i32` — the guest window's base address in linear memory (import `env.memory`);
//! - `env: i32` — the engine-side environment cell: an `i64` **fuel counter** at offset 0, debited
//!   once per dispatcher iteration (coarser than the interpreter's per-op fuel — a bound, not an
//!   observable; DESIGN.md §5) and trapped on exhaustion.
//!
//! ## Confinement (the load-bearing part)
//!
//! Every guest access replicates `svm_mask::Window::checked` **exactly** (§4): with
//! `mask = (1 << DEFAULT_RESERVED_LOG2) - 1` and `mapped = 1 << size_log2`,
//!
//! ```text
//! rel = (addr + offset) & mask;   if rel > mapped - width { trap(MemoryFault) }
//! access linear memory at win + rel
//! ```
//!
//! both constants baked at compile time. SVM-specific traps (memory fault, fuel) route through the
//! imported `env.trap(code)` (the host records the code; the following `unreachable` aborts);
//! div/rem-by-zero, signed-overflow, and `unreachable` map to wasm's own identical traps.
//!
//! ## Control flow
//!
//! v1 is the **block dispatcher**: SSA values live in wasm locals (one per block-scoped value —
//! block params first, then each instruction's results, mirroring the verifier's numbering); a
//! `loop` re-dispatches on a `$next` local via `br_table` over one wasm `block` per SVM block.
//! Branch arguments are pushed onto the operand stack *then* popped into the target's param locals
//! (reverse order), so a self-branch that permutes its own params can't read an already-overwritten
//! local. A relooper for reducible CFGs is a planned upgrade, not a correctness need.
//!
//! Proven by `tests/differential.rs`: every kernel runs on the bytecode engine (the oracle) and on
//! the emitted wasm under `wasmi`, comparing results **and trap kinds**.

#![forbid(unsafe_code)]

use svm_ir::{
    BinOp, Block, CmpOp, ConvOp, Func, Inst, IntTy, IntUnOp, LoadOp, Module, StoreOp, Terminator,
    ValType, DEFAULT_RESERVED_LOG2,
};

/// Trap code delivered through `env.trap` when the per-dispatch fuel counter goes negative.
pub const TRAP_OUT_OF_FUEL: i32 = 1;
/// Trap code delivered through `env.trap` when a confined access fails the guard check
/// (`rel + width > mapped` — the §4 guard-region fault).
pub const TRAP_MEMORY_FAULT: i32 = 2;

/// Why a module was refused. Fail-closed: the caller runs the module on the interpreter tier.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Error {
    /// An instruction / terminator / type outside the v1 subset (the payload names it).
    Unsupported(&'static str),
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::Unsupported(what) => write!(f, "unsupported by the wasm tier: {what}"),
        }
    }
}

const MASK: u64 = (1u64 << DEFAULT_RESERVED_LOG2) - 1;

// ---- wasm binary encoding primitives -------------------------------------------------------------

fn uleb(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(b);
            return;
        }
        out.push(b | 0x80);
    }
}

fn sleb64(out: &mut Vec<u8>, mut v: i64) {
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        let done = (v == 0 && b & 0x40 == 0) || (v == -1 && b & 0x40 != 0);
        if done {
            out.push(b);
            return;
        }
        out.push(b | 0x80);
    }
}

fn sleb32(out: &mut Vec<u8>, v: i32) {
    sleb64(out, v as i64);
}

fn valtype_byte(t: ValType) -> Result<u8, Error> {
    match t {
        ValType::I32 => Ok(0x7f),
        ValType::I64 => Ok(0x7e),
        // Floats/v128/ref are interpreter-tier for now (the compute subset is integer).
        _ => Err(Error::Unsupported("non-integer value type")),
    }
}

// A handful of opcode groups the emitter uses; everything else is written as raw bytes at the
// emission site with a comment.
const OP_UNREACHABLE: u8 = 0x00;
const OP_LOOP: u8 = 0x03;
const OP_IF: u8 = 0x04;
const OP_ELSE: u8 = 0x05;
const OP_END: u8 = 0x0b;
const OP_BR: u8 = 0x0c;
const OP_BR_TABLE: u8 = 0x0e;
const OP_RETURN: u8 = 0x0f;
const OP_CALL: u8 = 0x10;
const OP_BLOCK: u8 = 0x02;
const OP_LOCAL_GET: u8 = 0x20;
const OP_LOCAL_SET: u8 = 0x21;
const OP_LOCAL_TEE: u8 = 0x22;
const OP_SELECT: u8 = 0x1b;
const OP_I32_CONST: u8 = 0x41;
const OP_I64_CONST: u8 = 0x42;
const BLOCKTYPE_VOID: u8 = 0x40;

/// `IntBin` opcodes are contiguous in wasm in exactly [`BinOp`]'s declaration order.
fn intbin_opcode(ty: IntTy, op: BinOp) -> u8 {
    let idx = BinOp::ALL.iter().position(|o| *o == op).unwrap() as u8;
    match ty {
        IntTy::I32 => 0x6a + idx,
        IntTy::I64 => 0x7c + idx,
    }
}

fn intcmp_opcode(ty: IntTy, op: CmpOp) -> u8 {
    // wasm orders lt/gt before le/ge; CmpOp orders le before gt — map explicitly.
    let (i32op, i64op) = match op {
        CmpOp::Eq => (0x46, 0x51),
        CmpOp::Ne => (0x47, 0x52),
        CmpOp::LtS => (0x48, 0x53),
        CmpOp::LtU => (0x49, 0x54),
        CmpOp::LeS => (0x4c, 0x57),
        CmpOp::LeU => (0x4d, 0x58),
        CmpOp::GtS => (0x4a, 0x55),
        CmpOp::GtU => (0x4b, 0x56),
        CmpOp::GeS => (0x4e, 0x59),
        CmpOp::GeU => (0x4f, 0x5a),
    };
    match ty {
        IntTy::I32 => i32op,
        IntTy::I64 => i64op,
    }
}

fn intun_opcode(ty: IntTy, op: IntUnOp) -> Result<u8, Error> {
    Ok(match (ty, op) {
        (IntTy::I32, IntUnOp::Clz) => 0x67,
        (IntTy::I32, IntUnOp::Ctz) => 0x68,
        (IntTy::I32, IntUnOp::Popcnt) => 0x69,
        (IntTy::I32, IntUnOp::Extend8S) => 0xc0,
        (IntTy::I32, IntUnOp::Extend16S) => 0xc1,
        (IntTy::I32, IntUnOp::Extend32S) => {
            return Err(Error::Unsupported("i32.extend32_s"));
        }
        (IntTy::I64, IntUnOp::Clz) => 0x79,
        (IntTy::I64, IntUnOp::Ctz) => 0x7a,
        (IntTy::I64, IntUnOp::Popcnt) => 0x7b,
        (IntTy::I64, IntUnOp::Extend8S) => 0xc2,
        (IntTy::I64, IntUnOp::Extend16S) => 0xc3,
        (IntTy::I64, IntUnOp::Extend32S) => 0xc4,
    })
}

/// `(opcode, access width, result type)` for a load.
fn load_op(op: LoadOp) -> Result<(u8, u64, ValType), Error> {
    Ok(match op {
        LoadOp::I32 => (0x28, 4, ValType::I32),
        LoadOp::I64 => (0x29, 8, ValType::I64),
        LoadOp::F32 | LoadOp::F64 => return Err(Error::Unsupported("float load")),
        LoadOp::I32_8S => (0x2c, 1, ValType::I32),
        LoadOp::I32_8U => (0x2d, 1, ValType::I32),
        LoadOp::I32_16S => (0x2e, 2, ValType::I32),
        LoadOp::I32_16U => (0x2f, 2, ValType::I32),
        LoadOp::I64_8S => (0x30, 1, ValType::I64),
        LoadOp::I64_8U => (0x31, 1, ValType::I64),
        LoadOp::I64_16S => (0x32, 2, ValType::I64),
        LoadOp::I64_16U => (0x33, 2, ValType::I64),
        LoadOp::I64_32S => (0x34, 4, ValType::I64),
        LoadOp::I64_32U => (0x35, 4, ValType::I64),
    })
}

/// `(opcode, access width)` for a store.
fn store_op(op: StoreOp) -> Result<(u8, u64), Error> {
    Ok(match op {
        StoreOp::I32 => (0x36, 4),
        StoreOp::I64 => (0x37, 8),
        StoreOp::F32 | StoreOp::F64 => return Err(Error::Unsupported("float store")),
        StoreOp::I32_8 => (0x3a, 1),
        StoreOp::I32_16 => (0x3b, 2),
        StoreOp::I64_8 => (0x3c, 1),
        StoreOp::I64_16 => (0x3d, 2),
        StoreOp::I64_32 => (0x3e, 4),
    })
}

// ---- per-function value typing (mirrors the verifier's block-scoped numbering) -------------------

/// The types of one block's value list: params first, then each instruction's results in order.
/// Only the v1 subset is typed; anything else is `Unsupported` (fail-closed — the module was
/// verified, so no `unwrap` here can be reached by a malformed operand index).
fn block_value_types(m: &Module, b: &Block) -> Result<Vec<ValType>, Error> {
    let mut tys: Vec<ValType> = b.params.clone();
    for inst in &b.insts {
        match inst {
            Inst::ConstI32(_) => tys.push(ValType::I32),
            Inst::ConstI64(_) => tys.push(ValType::I64),
            Inst::IntBin { ty, .. } => tys.push(ty.val()),
            Inst::IntCmp { .. } | Inst::Eqz { .. } => tys.push(ValType::I32),
            Inst::IntUn { ty, .. } => tys.push(ty.val()),
            Inst::Convert { op, .. } => tys.push(op.sig().2),
            Inst::Select { a, .. } => {
                let t = *tys
                    .get(*a as usize)
                    .ok_or(Error::Unsupported("select operand"))?;
                tys.push(t);
            }
            Inst::Load { op, .. } => tys.push(load_op(*op)?.2),
            Inst::Store { .. } => {}
            Inst::Call { func, .. } => {
                let callee = m
                    .funcs
                    .get(*func as usize)
                    .ok_or(Error::Unsupported("call target"))?;
                tys.extend(callee.results.iter().copied());
            }
            _ => return Err(Error::Unsupported("instruction outside the v1 subset")),
        }
    }
    Ok(tys)
}

// ---- tiering analysis (slice 3): which functions the JIT tier emits vs. leaves to the interp ------

/// Per-function tiering classification for a module (`BROWSER.md` § "wasm-JIT tier", slice 3).
/// The JIT tier emits the **in-subset** functions and routes a call to an **interp-callable** one
/// through the engine (a cross-tier call); a guest is `mixed_ok` when func 0 and everything it
/// reaches is one or the other (and nothing reachable suspends — a JITted frame can't unwind, so
/// suspension anywhere forces the whole guest to the interpreter).
#[derive(Clone, Debug)]
pub struct Analysis {
    /// `in_subset[i]` — function `i` is entirely within the integer compute subset the emitter
    /// lowers directly (it becomes an emitted `f{i}`).
    pub in_subset: Vec<bool>,
    /// `interp_leaf[i]` — function `i` is **not** in-subset but is safe to run on the bytecode
    /// engine as a cross-tier leaf: an all-integer signature (so the call ABI marshals only i64
    /// slots), **memory-free**, makes no calls (a true leaf, so no transitive window/state to
    /// share), and no concurrency / capability ops. A JITted caller reaches it via `env.call_interp`.
    pub interp_leaf: Vec<bool>,
    /// `reachable[i]` — function `i` is reachable from func 0 through call edges.
    pub reachable: Vec<bool>,
    /// Every reachable function is in-subset or an interp leaf, func 0 is in-subset, and nothing
    /// reachable uses concurrency — i.e. the guest can run on the JIT tier (with cross-tier calls).
    pub mixed_ok: bool,
}

/// A block terminator the emitter lowers (tail calls are not in the v1 subset).
fn term_in_subset(t: &Terminator) -> bool {
    matches!(
        t,
        Terminator::Br { .. }
            | Terminator::BrIf { .. }
            | Terminator::BrTable { .. }
            | Terminator::Return(_)
            | Terminator::Unreachable
    )
}

/// Whether every instruction, terminator, and value type of `f` is in the emitter's integer compute
/// subset — reusing [`block_value_types`] (which errors on any out-of-subset instruction) as the
/// single source of truth, plus a type check (all values i32/i64) and the terminator check.
fn func_in_subset(m: &Module, f: &Func) -> bool {
    f.blocks.iter().all(|b| {
        block_value_types(m, b).is_ok_and(|tys| tys.iter().all(|t| valtype_byte(*t).is_ok()))
            && term_in_subset(&b.term)
    })
}

/// The function indices `f` calls (direct `Call`s + tail-call terminators — the latter keeps the
/// reachability sound even though a tail call itself isn't emitted).
fn func_callees(f: &Func) -> Vec<u32> {
    let mut out = Vec::new();
    for b in &f.blocks {
        for inst in &b.insts {
            if let Inst::Call { func, .. } = inst {
                out.push(*func);
            }
        }
        if let Terminator::ReturnCall { func, .. } = &b.term {
            out.push(*func);
        }
    }
    out
}

/// Whether `f` is safe to run as a cross-tier interpreter leaf (see [`Analysis::interp_leaf`]).
fn interp_leaf(f: &Func) -> bool {
    let int_sig = f
        .params
        .iter()
        .chain(&f.results)
        .all(|t| matches!(t, ValType::I32 | ValType::I64));
    if !int_sig || f.uses_concurrency() {
        return false;
    }
    f.blocks.iter().all(|b| {
        !matches!(
            b.term,
            Terminator::ReturnCall { .. } | Terminator::ReturnCallIndirect { .. }
        ) && b.insts.iter().all(|i| {
            !matches!(
                i,
                // memory ops (a leaf's fresh window would diverge from the shared one),
                Inst::Load { .. }
                        | Inst::Store { .. }
                        | Inst::AtomicLoad { .. }
                        | Inst::AtomicStore { .. }
                        | Inst::AtomicRmw { .. }
                        | Inst::AtomicCmpxchg { .. }
                        | Inst::V128Load { .. }
                        | Inst::V128Store { .. }
                        // calls (a true leaf only — transitive tiers are a later refinement),
                        | Inst::Call { .. }
                        | Inst::CallIndirect { .. }
                        // and host/capability ops (no powerbox in the cross-tier callback).
                        | Inst::CapCall { .. }
                        | Inst::CallImport { .. }
            )
        })
    })
}

/// Classify every function of a **verified** `m` for tiering (see [`Analysis`]).
pub fn analyze(m: &Module) -> Analysis {
    let n = m.funcs.len();
    let in_subset: Vec<bool> = m.funcs.iter().map(|f| func_in_subset(m, f)).collect();
    let interp_leaf: Vec<bool> = m
        .funcs
        .iter()
        .enumerate()
        .map(|(i, f)| !in_subset[i] && interp_leaf(f))
        .collect();

    // Reachability from func 0 through call edges.
    let mut reachable = vec![false; n];
    if n > 0 {
        let mut stack = vec![0u32];
        reachable[0] = true;
        while let Some(fi) = stack.pop() {
            for c in func_callees(&m.funcs[fi as usize]) {
                if (c as usize) < n && !reachable[c as usize] {
                    reachable[c as usize] = true;
                    stack.push(c);
                }
            }
        }
    }

    let mixed_ok =
        n > 0 && in_subset[0] && (0..n).all(|i| !reachable[i] || in_subset[i] || interp_leaf[i]);

    Analysis {
        in_subset,
        interp_leaf,
        reachable,
        mixed_ok,
    }
}

// ---- the emitter ---------------------------------------------------------------------------------

/// Compile every function of a **verified** `m` into one wasm module, importing a **non-shared**
/// `env.memory`. See [`compile_module_shared`] for the browser (threads-build) link target.
pub fn compile_module(m: &Module) -> Result<Vec<u8>, Error> {
    compile_module_with(m, false)
}

/// Like [`compile_module`] but the imported `env.memory` is declared **shared** — the browser
/// wasm-JIT tier links the emitted module against the cdylib's shared linear memory (the threads
/// build), and wasm requires the import's shared flag to match the provided memory's. The emitted
/// code is otherwise byte-identical to the non-shared form (only the memory-import limits differ),
/// so the `compile_module` differential (`tests/differential.rs`, under `wasmi`, which has no
/// shared-memory support) fully covers this variant's codegen.
pub fn compile_module_shared(m: &Module) -> Result<Vec<u8>, Error> {
    compile_module_with(m, true)
}

/// Two imported functions precede every emitted function, so a defined function's wasm index is
/// `IMPORTED_FUNCS + its position among the emitted functions`.
const IMPORTED_FUNCS: u32 = 2;
/// `env.call_interp` scratch: the cross-tier call marshals its i64 arg/result slots starting at
/// this byte offset in the `env` cell (past the `i64` fuel counter at 0). The host must allocate the
/// `env` cell at least [`ENV_CELL_BYTES`] large.
const ENV_SCRATCH_OFF: u64 = 16;
/// Max i64 slots the cross-tier scratch holds (a call with more args-or-results than this is refused
/// — 64 is absurdly generous for a function signature).
const XCALL_MAX_SLOTS: usize = 64;
/// Bytes the host must allocate for the `env` cell: the `i64` fuel counter + the cross-tier scratch.
pub const ENV_CELL_BYTES: usize = ENV_SCRATCH_OFF as usize + XCALL_MAX_SLOTS * 8;

/// Compile every function of a **verified** `m` into one wasm module (whole-module, all-integer).
/// Exports `f{i}` per SVM function; imports `env.memory` (shared iff `shared_memory`), `env.trap`,
/// and `env.call_interp`. Returns [`Error::Unsupported`] if *any* function is outside the v1 subset
/// — for a mixed integer/interp guest use [`compile_module_mixed`].
pub fn compile_module_with(m: &Module, shared_memory: bool) -> Result<Vec<u8>, Error> {
    let a = analyze(m);
    if !a.in_subset.iter().all(|&s| s) {
        return Err(Error::Unsupported(
            "a function is outside the integer subset",
        ));
    }
    let n = m.funcs.len();
    let emitted: Vec<usize> = (0..n).collect();
    let wasm_of: Vec<Option<u32>> = (0..n).map(|i| Some(IMPORTED_FUNCS + i as u32)).collect();
    emit_module(m, shared_memory, &emitted, &wasm_of, &a.interp_leaf)
}

/// Compile a **mixed-tier** guest (`BROWSER.md` § "wasm-JIT tier", slice 3): emit the in-subset
/// functions and route a call to an interp leaf through `env.call_interp` (the engine runs it on the
/// bytecode interpreter — see [`Analysis`]). [`Error::Unsupported`] unless [`Analysis::mixed_ok`].
pub fn compile_module_mixed(m: &Module, shared_memory: bool) -> Result<Vec<u8>, Error> {
    let a = analyze(m);
    if !a.mixed_ok {
        return Err(Error::Unsupported("guest is not mixed-tier runnable"));
    }
    // Emit the in-subset functions in SVM-index order; each gets the next wasm index. Interp leaves
    // (and unreachable non-subset functions) get no wasm index — a call to a leaf goes to the import.
    let mut wasm_of: Vec<Option<u32>> = vec![None; m.funcs.len()];
    let mut emitted: Vec<usize> = Vec::new();
    for (i, &in_sub) in a.in_subset.iter().enumerate() {
        if in_sub {
            wasm_of[i] = Some(IMPORTED_FUNCS + emitted.len() as u32);
            emitted.push(i);
        }
    }
    emit_module(m, shared_memory, &emitted, &wasm_of, &a.interp_leaf)
}

/// Assemble the wasm module: emit the functions listed in `emitted` (SVM indices, in the order they
/// take wasm indices), routing each `Call` via `wasm_of` (a direct wasm call) or, for an interp
/// leaf, through `env.call_interp`. See the module docs for the emitted shape.
fn emit_module(
    m: &Module,
    shared_memory: bool,
    emitted: &[usize],
    wasm_of: &[Option<u32>],
    interp_leaf: &[bool],
) -> Result<Vec<u8>, Error> {
    if !m.imports.is_empty() {
        return Err(Error::Unsupported("unresolved imports"));
    }
    if !m.data.is_empty() {
        return Err(Error::Unsupported("data segments"));
    }
    let mapped: u64 = match &m.memory {
        Some(mc) => 1u64 << mc.size_log2,
        None => 0,
    };

    // Types: 0 = env.trap `(i32) -> ()`, 1 = env.call_interp `(i32 func, i32 args_ptr) -> ()`, then
    // one per emitted function (dedup'd).
    let mut types: Vec<(Vec<u8>, Vec<u8>)> = vec![(vec![0x7f], vec![]), (vec![0x7f, 0x7f], vec![])];
    let mut fn_type_idx: Vec<u32> = Vec::with_capacity(emitted.len());
    for &fi in emitted {
        let f = &m.funcs[fi];
        let mut params = vec![0x7f, 0x7f]; // win: i32, env: i32
        for p in &f.params {
            params.push(valtype_byte(*p)?);
        }
        let mut results = Vec::with_capacity(f.results.len());
        for r in &f.results {
            results.push(valtype_byte(*r)?);
        }
        let ty = (params, results);
        let idx = match types.iter().position(|t| *t == ty) {
            Some(i) => i,
            None => {
                types.push(ty);
                types.len() - 1
            }
        };
        fn_type_idx.push(idx as u32);
    }

    let mut bodies: Vec<Vec<u8>> = Vec::with_capacity(emitted.len());
    for &fi in emitted {
        bodies.push(emit_func(m, &m.funcs[fi], mapped, wasm_of, interp_leaf)?);
    }

    // ---- assemble the module ----
    let mut out = vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00]; // \0asm v1

    let mut sec = Vec::new(); // type section (1)
    uleb(&mut sec, types.len() as u64);
    for (params, results) in &types {
        sec.push(0x60);
        uleb(&mut sec, params.len() as u64);
        sec.extend_from_slice(params);
        uleb(&mut sec, results.len() as u64);
        sec.extend_from_slice(results);
    }
    section(&mut out, 1, &sec);

    let mut sec = Vec::new(); // import section (2): env.memory, env.trap (type 0), env.call_interp (type 1)
    uleb(&mut sec, 3);
    import_name(&mut sec, "env", "memory");
    sec.push(0x02); // memory
    if shared_memory {
        // Flag 0x03 = shared + has-max; min 0, max 65536 (wasm32's 4 GiB / 64 KiB-page ceiling). A
        // min-0/max-ceiling import is satisfied by any shared memory the host provides, and the
        // shared flag must match the provided memory's (the browser threads build's shared memory).
        sec.push(0x03);
        uleb(&mut sec, 0);
        uleb(&mut sec, 65536);
    } else {
        // Flag 0x00 = min-only, non-shared (the `wasmi` differential + a plain cdylib build).
        sec.push(0x00);
        uleb(&mut sec, 0);
    }
    import_name(&mut sec, "env", "trap");
    sec.push(0x00); // func
    uleb(&mut sec, 0); // type index 0
    import_name(&mut sec, "env", "call_interp");
    sec.push(0x00); // func
    uleb(&mut sec, 1); // type index 1
    section(&mut out, 2, &sec);

    let mut sec = Vec::new(); // function section (3)
    uleb(&mut sec, emitted.len() as u64);
    for ti in &fn_type_idx {
        uleb(&mut sec, *ti as u64);
    }
    section(&mut out, 3, &sec);

    let mut sec = Vec::new(); // export section (7): "f{svm_idx}" → its wasm index
    uleb(&mut sec, emitted.len() as u64);
    for &fi in emitted {
        let name = format!("f{fi}");
        uleb(&mut sec, name.len() as u64);
        sec.extend_from_slice(name.as_bytes());
        sec.push(0x00);
        uleb(&mut sec, wasm_of[fi].unwrap() as u64);
    }
    section(&mut out, 7, &sec);

    let mut sec = Vec::new(); // code section (10)
    uleb(&mut sec, bodies.len() as u64);
    for b in &bodies {
        uleb(&mut sec, b.len() as u64);
        sec.extend_from_slice(b);
    }
    section(&mut out, 10, &sec);

    Ok(out)
}

fn section(out: &mut Vec<u8>, id: u8, payload: &[u8]) {
    out.push(id);
    uleb(out, payload.len() as u64);
    out.extend_from_slice(payload);
}

fn import_name(out: &mut Vec<u8>, module: &str, name: &str) {
    uleb(out, module.len() as u64);
    out.extend_from_slice(module.as_bytes());
    uleb(out, name.len() as u64);
    out.extend_from_slice(name.as_bytes());
}

/// Per-function emission state: the (block, value) → wasm-local map plus the scratch locals.
struct FnCtx {
    /// `local_of[block][value]` — wasm local index of each block-scoped SSA value.
    local_of: Vec<Vec<u32>>,
    next_l: u32,
    ea_l: u32,
    fuel_l: u32,
    /// Open label count inside the body; the dispatcher `loop` is the first label opened, so a
    /// branch back to it from depth `d` is `br (d - 1)`.
    depth: u32,
}

impl FnCtx {
    fn br_dispatch(&self, code: &mut Vec<u8>) {
        code.push(OP_BR);
        uleb(code, (self.depth - 1) as u64);
    }
}

fn emit_func(
    m: &Module,
    f: &Func,
    mapped: u64,
    wasm_of: &[Option<u32>],
    interp_leaf: &[bool],
) -> Result<Vec<u8>, Error> {
    let n_params = 2 + f.params.len() as u32; // win, env, then the SVM params

    // Allocate locals: every block's value list, then $next/$ea/$fuel.
    let mut local_types: Vec<ValType> = Vec::new();
    let mut local_of: Vec<Vec<u32>> = Vec::with_capacity(f.blocks.len());
    let mut per_block_types: Vec<Vec<ValType>> = Vec::with_capacity(f.blocks.len());
    for b in &f.blocks {
        let tys = block_value_types(m, b)?;
        let mut idxs = Vec::with_capacity(tys.len());
        for t in &tys {
            idxs.push(n_params + local_types.len() as u32);
            local_types.push(*t);
        }
        local_of.push(idxs);
        per_block_types.push(tys);
    }
    let next_l = n_params + local_types.len() as u32;
    local_types.push(ValType::I32);
    let ea_l = n_params + local_types.len() as u32;
    local_types.push(ValType::I64);
    let fuel_l = n_params + local_types.len() as u32;
    local_types.push(ValType::I64);

    let mut cx = FnCtx {
        local_of,
        next_l,
        ea_l,
        fuel_l,
        depth: 0,
    };

    let mut code = Vec::new();
    // Copy the SVM params into the entry block's param locals ($next defaults to 0 = entry).
    for (i, _) in f.params.iter().enumerate() {
        code.push(OP_LOCAL_GET);
        uleb(&mut code, 2 + i as u64);
        code.push(OP_LOCAL_SET);
        uleb(&mut code, cx.local_of[0][i] as u64);
    }

    // The dispatcher: loop { fuel; block .. block { br_table $next } code_0 .. code_{N-1} }.
    code.push(OP_LOOP);
    code.push(BLOCKTYPE_VOID);
    cx.depth += 1;
    emit_fuel_check(&mut cx, &mut code);
    let n = f.blocks.len();
    for _ in 0..n {
        code.push(OP_BLOCK);
        code.push(BLOCKTYPE_VOID);
        cx.depth += 1;
    }
    code.push(OP_LOCAL_GET);
    uleb(&mut code, cx.next_l as u64);
    code.push(OP_BR_TABLE);
    uleb(&mut code, n as u64); // n labels + default
    for k in 0..n {
        uleb(&mut code, k as u64); // depth k exits block k → lands at code_k
    }
    uleb(&mut code, (n - 1) as u64); // default: unreachable by construction; any valid label

    for (k, b) in f.blocks.iter().enumerate() {
        code.push(OP_END); // close block k; code_k follows
        cx.depth -= 1;
        emit_block_body(
            m,
            f,
            &mut cx,
            &mut code,
            k,
            b,
            &per_block_types[k],
            mapped,
            wasm_of,
            interp_leaf,
        )?;
    }
    code.push(OP_END); // close the loop
    cx.depth -= 1;
    code.push(OP_UNREACHABLE); // every path returned / trapped / re-dispatched
    code.push(OP_END); // function body end

    // Prepend the locals vector (grouped runs of one type).
    let mut body = Vec::new();
    let mut groups: Vec<(u32, u8)> = Vec::new();
    for t in &local_types {
        let byte = valtype_byte(*t)?;
        match groups.last_mut() {
            Some((count, b)) if *b == byte => *count += 1,
            _ => groups.push((1, byte)),
        }
    }
    uleb(&mut body, groups.len() as u64);
    for (count, byte) in groups {
        uleb(&mut body, count as u64);
        body.push(byte);
    }
    body.extend_from_slice(&code);
    Ok(body)
}

/// Debit one fuel unit from the `i64` cell at `env` and trap `TRAP_OUT_OF_FUEL` when it goes
/// negative. Runs once per dispatcher iteration — a coarser debit than the interpreter's per-op
/// fuel, deliberately (fuel is a §5 bound, not an observable).
fn emit_fuel_check(cx: &mut FnCtx, code: &mut Vec<u8>) {
    code.push(OP_LOCAL_GET); // [env]        (store address)
    uleb(code, 1);
    code.push(OP_LOCAL_GET); // [env, env]
    uleb(code, 1);
    code.extend_from_slice(&[0x29, 0x03, 0x00]); // i64.load align=8 → [env, fuel]
    code.push(OP_I64_CONST);
    sleb64(code, 1);
    code.push(0x7d); // i64.sub → [env, fuel-1]
    code.push(OP_LOCAL_TEE);
    uleb(code, cx.fuel_l as u64);
    code.extend_from_slice(&[0x37, 0x03, 0x00]); // i64.store align=8 → []
    code.push(OP_LOCAL_GET);
    uleb(code, cx.fuel_l as u64);
    code.push(OP_I64_CONST);
    sleb64(code, 0);
    code.push(0x53); // i64.lt_s
    code.push(OP_IF);
    code.push(BLOCKTYPE_VOID);
    cx.depth += 1;
    emit_trap(code, TRAP_OUT_OF_FUEL);
    code.push(OP_END);
    cx.depth -= 1;
}

/// `call env.trap(code); unreachable` — the host records the SVM trap kind, the `unreachable`
/// aborts execution.
fn emit_trap(code: &mut Vec<u8>, trap_code: i32) {
    code.push(OP_I32_CONST);
    sleb32(code, trap_code);
    code.push(OP_CALL);
    uleb(code, 0); // func 0 = the env.trap import
    code.push(OP_UNREACHABLE);
}

/// Confine + guard-check the effective address for a `width`-byte access, leaving the confined
/// 32-bit linear-memory address on the stack: `win + ((addr + offset) & MASK)`, trapping
/// `MemoryFault` unless `rel <= mapped - width` (exactly `svm_mask::Window::checked`).
fn emit_confine(
    cx: &mut FnCtx,
    code: &mut Vec<u8>,
    addr_local: u32,
    offset: u64,
    width: u64,
    mapped: u64,
) {
    code.push(OP_LOCAL_GET);
    uleb(code, addr_local as u64);
    code.push(OP_I64_CONST);
    sleb64(code, offset as i64);
    code.push(0x7c); // i64.add
    code.push(OP_I64_CONST);
    sleb64(code, MASK as i64);
    code.push(0x83); // i64.and → rel
    code.push(OP_LOCAL_TEE);
    uleb(code, cx.ea_l as u64);
    code.push(OP_I64_CONST);
    sleb64(code, mapped.wrapping_sub(width) as i64);
    code.push(0x56); // i64.gt_u: rel > mapped - width ?
    code.push(OP_IF);
    code.push(BLOCKTYPE_VOID);
    cx.depth += 1;
    emit_trap(code, TRAP_MEMORY_FAULT);
    code.push(OP_END);
    cx.depth -= 1;
    code.push(OP_LOCAL_GET);
    uleb(code, cx.ea_l as u64);
    code.push(0xa7); // i32.wrap_i64
    code.push(OP_LOCAL_GET);
    uleb(code, 0); // win
    code.push(0x6a); // i32.add → the confined linear-memory address
}

/// Push branch args onto the operand stack, then pop them into the target block's param locals in
/// reverse — stack copies make a param-permuting self-branch safe.
fn emit_edge(
    cx: &FnCtx,
    code: &mut Vec<u8>,
    from_block: usize,
    target: u32,
    args: &[svm_ir::ValIdx],
) {
    for a in args {
        code.push(OP_LOCAL_GET);
        uleb(code, cx.local_of[from_block][*a as usize] as u64);
    }
    for i in (0..args.len()).rev() {
        code.push(OP_LOCAL_SET);
        uleb(code, cx.local_of[target as usize][i] as u64);
    }
    code.push(OP_I32_CONST);
    sleb32(code, target as i32);
    code.push(OP_LOCAL_SET);
    uleb(code, cx.next_l as u64);
}

#[allow(clippy::too_many_arguments)]
fn emit_block_body(
    m: &Module,
    f: &Func,
    cx: &mut FnCtx,
    code: &mut Vec<u8>,
    k: usize,
    b: &Block,
    value_types: &[ValType],
    mapped: u64,
    wasm_of: &[Option<u32>],
    interp_leaf: &[bool],
) -> Result<(), Error> {
    let mut next_val = b.params.len(); // where the next instruction's results land
    let get = |code: &mut Vec<u8>, cx: &FnCtx, v: svm_ir::ValIdx| {
        code.push(OP_LOCAL_GET);
        uleb(code, cx.local_of[k][v as usize] as u64);
    };
    for inst in &b.insts {
        match inst {
            Inst::ConstI32(v) => {
                code.push(OP_I32_CONST);
                sleb32(code, *v);
                set_result(cx, code, k, &mut next_val);
            }
            Inst::ConstI64(v) => {
                code.push(OP_I64_CONST);
                sleb64(code, *v);
                set_result(cx, code, k, &mut next_val);
            }
            Inst::IntBin { ty, op, a, b: rb } => {
                get(code, cx, *a);
                get(code, cx, *rb);
                code.push(intbin_opcode(*ty, *op));
                set_result(cx, code, k, &mut next_val);
            }
            Inst::IntCmp { ty, op, a, b: rb } => {
                get(code, cx, *a);
                get(code, cx, *rb);
                code.push(intcmp_opcode(*ty, *op));
                set_result(cx, code, k, &mut next_val);
            }
            Inst::IntUn { ty, op, a } => {
                get(code, cx, *a);
                code.push(intun_opcode(*ty, *op)?);
                set_result(cx, code, k, &mut next_val);
            }
            Inst::Eqz { ty, a } => {
                get(code, cx, *a);
                code.push(match ty {
                    IntTy::I32 => 0x45,
                    IntTy::I64 => 0x50,
                });
                set_result(cx, code, k, &mut next_val);
            }
            Inst::Convert { op, a } => {
                get(code, cx, *a);
                code.push(match op {
                    ConvOp::ExtendI32S => 0xac,
                    ConvOp::ExtendI32U => 0xad,
                    ConvOp::WrapI64 => 0xa7,
                });
                set_result(cx, code, k, &mut next_val);
            }
            Inst::Select { cond, a, b: rb } => {
                get(code, cx, *a);
                get(code, cx, *rb);
                get(code, cx, *cond);
                code.push(OP_SELECT);
                set_result(cx, code, k, &mut next_val);
            }
            Inst::Load {
                op, addr, offset, ..
            } => {
                let (opcode, width, _) = load_op(*op)?;
                emit_confine(
                    cx,
                    code,
                    cx.local_of[k][*addr as usize],
                    *offset,
                    width,
                    mapped,
                );
                code.extend_from_slice(&[opcode, 0x00, 0x00]); // align=1, offset=0
                set_result(cx, code, k, &mut next_val);
            }
            Inst::Store {
                op,
                addr,
                value,
                offset,
                ..
            } => {
                let (opcode, width) = store_op(*op)?;
                emit_confine(
                    cx,
                    code,
                    cx.local_of[k][*addr as usize],
                    *offset,
                    width,
                    mapped,
                );
                get(code, cx, *value);
                code.extend_from_slice(&[opcode, 0x00, 0x00]); // align=1, offset=0
            }
            Inst::Call { func, args } => {
                let callee = &m.funcs[*func as usize];
                let n_results = callee.results.len();
                match wasm_of[*func as usize] {
                    // Same-tier: a direct wasm call to the emitted function (win/env threaded).
                    Some(widx) => {
                        code.push(OP_LOCAL_GET);
                        uleb(code, 0); // win
                        code.push(OP_LOCAL_GET);
                        uleb(code, 1); // env
                        for a in args {
                            get(code, cx, *a);
                        }
                        code.push(OP_CALL);
                        uleb(code, widx as u64);
                        // Results pushed in order; pop into destination locals in reverse.
                        for i in (0..n_results).rev() {
                            code.push(OP_LOCAL_SET);
                            uleb(code, cx.local_of[k][next_val + i] as u64);
                        }
                    }
                    // Cross-tier: `func` is an interp leaf. Marshal args as i64 slots into the env
                    // scratch, call `env.call_interp(func, args_ptr)` (the engine runs it on the
                    // bytecode interpreter and writes results back to the same slots), then reload.
                    None => {
                        if !interp_leaf[*func as usize] {
                            return Err(Error::Unsupported("call to a non-emitted, non-leaf func"));
                        }
                        if args.len().max(n_results) > XCALL_MAX_SLOTS {
                            return Err(Error::Unsupported("cross-tier call arity too large"));
                        }
                        // Store each arg to env + ENV_SCRATCH_OFF + i*8 (widen an i32 to the slot).
                        for (i, a) in args.iter().enumerate() {
                            code.push(OP_LOCAL_GET);
                            uleb(code, 1); // env
                            code.push(OP_I32_CONST);
                            sleb32(code, (ENV_SCRATCH_OFF + i as u64 * 8) as i32);
                            code.push(0x6a); // i32.add → slot addr
                            get(code, cx, *a);
                            if callee.params[i] == ValType::I32 {
                                code.push(0xad); // i64.extend_i32_u
                            }
                            code.extend_from_slice(&[0x37, 0x03, 0x00]); // i64.store align=8
                        }
                        // env.call_interp(func_svm_idx, args_ptr = env + ENV_SCRATCH_OFF).
                        code.push(OP_I32_CONST);
                        sleb32(code, *func as i32);
                        code.push(OP_LOCAL_GET);
                        uleb(code, 1); // env
                        code.push(OP_I32_CONST);
                        sleb32(code, ENV_SCRATCH_OFF as i32);
                        code.push(0x6a); // i32.add
                        code.push(OP_CALL);
                        uleb(code, 1); // func 1 = env.call_interp
                                       // Load results back from the scratch slots (narrow to i32 where needed).
                        for i in 0..n_results {
                            code.push(OP_LOCAL_GET);
                            uleb(code, 1); // env
                            code.push(OP_I32_CONST);
                            sleb32(code, (ENV_SCRATCH_OFF + i as u64 * 8) as i32);
                            code.push(0x6a); // i32.add
                            code.extend_from_slice(&[0x29, 0x03, 0x00]); // i64.load align=8
                            if callee.results[i] == ValType::I32 {
                                code.push(0xa7); // i32.wrap_i64
                            }
                            code.push(OP_LOCAL_SET);
                            uleb(code, cx.local_of[k][next_val + i] as u64);
                        }
                    }
                }
                next_val += n_results;
            }
            _ => return Err(Error::Unsupported("instruction outside the v1 subset")),
        }
    }
    debug_assert_eq!(next_val, value_types.len());

    match &b.term {
        Terminator::Br { target, args } => {
            emit_edge(cx, code, k, *target, args);
            cx.br_dispatch(code);
        }
        Terminator::BrIf {
            cond,
            then_blk,
            then_args,
            else_blk,
            else_args,
        } => {
            get(code, cx, *cond);
            code.push(OP_IF);
            code.push(BLOCKTYPE_VOID);
            cx.depth += 1;
            emit_edge(cx, code, k, *then_blk, then_args);
            code.push(OP_ELSE);
            emit_edge(cx, code, k, *else_blk, else_args);
            code.push(OP_END);
            cx.depth -= 1;
            cx.br_dispatch(code);
        }
        Terminator::BrTable {
            idx,
            targets,
            default,
        } => {
            // One landing block per edge (targets then default); each edge assigns its own args.
            let arms: Vec<&svm_ir::Edge> =
                targets.iter().chain(core::iter::once(default)).collect();
            for _ in &arms {
                code.push(OP_BLOCK);
                code.push(BLOCKTYPE_VOID);
                cx.depth += 1;
            }
            get(code, cx, *idx);
            code.push(OP_BR_TABLE);
            uleb(code, targets.len() as u64);
            for j in 0..targets.len() {
                uleb(code, j as u64);
            }
            uleb(code, targets.len() as u64); // default = the outermost landing block
            for (j, (target, args)) in arms.iter().enumerate() {
                code.push(OP_END);
                cx.depth -= 1;
                emit_edge(cx, code, k, *target, args);
                cx.br_dispatch(code);
                // The br above leaves this position unreachable; the next `end` (or code) follows.
                let _ = j;
            }
        }
        Terminator::Return(vals) => {
            for v in vals {
                get(code, cx, *v);
            }
            code.push(OP_RETURN);
        }
        Terminator::Unreachable => {
            code.push(OP_UNREACHABLE);
        }
        Terminator::ReturnCall { .. } | Terminator::ReturnCallIndirect { .. } => {
            return Err(Error::Unsupported("tail call"));
        }
    }
    let _ = f;
    Ok(())
}

fn set_result(cx: &FnCtx, code: &mut Vec<u8>, k: usize, next_val: &mut usize) {
    code.push(OP_LOCAL_SET);
    uleb(code, cx.local_of[k][*next_val] as u64);
    *next_val += 1;
}
