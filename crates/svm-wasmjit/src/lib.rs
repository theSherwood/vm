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
//! Every guest access replicates the trap-confinement `svm_mask::Window::checked` **exactly**
//! (§4, D38): with `mask = (1 << DEFAULT_RESERVED_LOG2) - 1` and `mapped = 1 << size_log2`,
//!
//! ```text
//! eff = addr + offset;   if eff > mapped - width { trap(MemoryFault) }   // unmasked check
//! access linear memory at win + (eff & mask)   // clamp: no-op past the check
//! ```
//!
//! both constants baked at compile time. An out-of-window address **faults at the offending
//! access** — it is never wrapped back into the window (the `& mask` after the check mirrors the
//! native JIT's check+clamp lowering and cannot change a passing address). SVM-specific traps (memory fault, fuel) route through the
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
/// Trap code delivered through `env.trap` when an access fails the trap-confinement bounds
/// check (`addr + offset + width > mapped` — the §4 `MemoryFault` at the offending access).
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

// ---- the emitter ---------------------------------------------------------------------------------

/// Compile every function of a **verified** `m` into one wasm module. Exports `f{i}` per SVM
/// function; imports `env.memory` and `env.trap (i32) -> ()`. Returns the binary bytes, or
/// [`Error::Unsupported`] if *any* function uses something outside the v1 subset (module-granular
/// fail-closed, like `compile_module`'s `None` — per-function tiering is slice 3).
pub fn compile_module(m: &Module) -> Result<Vec<u8>, Error> {
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

    // Type section entries: index 0 = env.trap's (i32) -> (); then one per function (dedup'd).
    let mut types: Vec<(Vec<u8>, Vec<u8>)> = vec![(vec![0x7f], vec![])];
    let mut fn_type_idx: Vec<u32> = Vec::with_capacity(m.funcs.len());
    for f in &m.funcs {
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

    let mut bodies: Vec<Vec<u8>> = Vec::with_capacity(m.funcs.len());
    for f in &m.funcs {
        bodies.push(emit_func(m, f, mapped)?);
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

    let mut sec = Vec::new(); // import section (2): env.memory (min 0), env.trap (type 0)
    uleb(&mut sec, 2);
    import_name(&mut sec, "env", "memory");
    sec.extend_from_slice(&[0x02, 0x00, 0x00]); // memory, limits: min-only, min 0
    import_name(&mut sec, "env", "trap");
    sec.push(0x00); // func
    uleb(&mut sec, 0); // type index 0
    section(&mut out, 2, &sec);

    let mut sec = Vec::new(); // function section (3)
    uleb(&mut sec, m.funcs.len() as u64);
    for ti in &fn_type_idx {
        uleb(&mut sec, *ti as u64);
    }
    section(&mut out, 3, &sec);

    let mut sec = Vec::new(); // export section (7): "f{i}" → func 1+i (trap import is func 0)
    uleb(&mut sec, m.funcs.len() as u64);
    for i in 0..m.funcs.len() {
        let name = format!("f{i}");
        uleb(&mut sec, name.len() as u64);
        sec.extend_from_slice(name.as_bytes());
        sec.push(0x00);
        uleb(&mut sec, 1 + i as u64);
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

fn emit_func(m: &Module, f: &Func, mapped: u64) -> Result<Vec<u8>, Error> {
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
        emit_block_body(m, f, &mut cx, &mut code, k, b, &per_block_types[k], mapped)?;
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

/// Confine the effective address for a `width`-byte access under **trap-confinement** (§4, D38),
/// leaving the confined 32-bit linear-memory address on the stack: bounds-check the *unmasked*
/// `eff = addr + offset` (trap `MemoryFault` unless `eff <= mapped - width` — exactly the
/// trap-confinement `svm_mask::Window::checked`, so an out-of-window address faults instead of
/// wrapping back in), then compute `win + (eff & MASK)`. The `& MASK` clamp is a no-op past the
/// check (`eff < mapped ≤ reserved`), kept to mirror the native JIT's check+clamp lowering and to
/// keep the following `i32.wrap` in-window as defense-in-depth.
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
    code.push(0x7c); // i64.add → eff (unmasked)
    code.push(OP_LOCAL_TEE);
    uleb(code, cx.ea_l as u64);
    code.push(OP_I64_CONST);
    sleb64(code, mapped.wrapping_sub(width) as i64);
    code.push(0x56); // i64.gt_u: eff > mapped - width ?
    code.push(OP_IF);
    code.push(BLOCKTYPE_VOID);
    cx.depth += 1;
    emit_trap(code, TRAP_MEMORY_FAULT);
    code.push(OP_END);
    cx.depth -= 1;
    code.push(OP_LOCAL_GET);
    uleb(code, cx.ea_l as u64);
    code.push(OP_I64_CONST);
    sleb64(code, MASK as i64);
    code.push(0x83); // i64.and → clamp (no-op past the check)
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
                code.push(OP_LOCAL_GET); // thread win
                uleb(code, 0);
                code.push(OP_LOCAL_GET); // thread env
                uleb(code, 1);
                for a in args {
                    get(code, cx, *a);
                }
                code.push(OP_CALL);
                uleb(code, 1 + *func as u64); // imports (trap) precede defined funcs
                let n_results = m.funcs[*func as usize].results.len();
                // Results are pushed in order; pop into the destination locals in reverse.
                for i in (0..n_results).rev() {
                    code.push(OP_LOCAL_SET);
                    uleb(code, cx.local_of[k][next_val + i] as u64);
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
