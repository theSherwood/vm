//! Cranelift JIT backend (`DESIGN.md` §9, §18).
//!
//! Cranelift is the chosen codegen **by design** (§1): it is the security mechanism,
//! not a liability — we share Wasmtime's most security-critical component, so the
//! escape-TCB *delta* we own is just this CLIF generation plus the §4 masking
//! lowering. Correctness is established by **differential testing against the
//! reference interpreter** (§18, invariants I1/I4), the oracle in `svm-interp`.
//!
//! ## Status: integer slice + the §4 memory masking lowering
//! Lowers `i32`/`i64` scalars: constants, the non-trapping integer arithmetic/
//! bitwise/shift/rotate ops, comparisons, `eqz`, `select`, `clz/ctz/popcnt`, the
//! `br`/`br_if`/`return` terminators, and **integer loads/stores with confinement
//! masking** — the security-critical I1 lowering. Anything else returns
//! [`JitError::Unsupported`] so the differential harness skips it. Calls, `br_table`,
//! floats, conversions, and trapping ops grow from here under the same check.
//!
//! ## The masking lowering (§4, invariant I1)
//! Every access masks the **final effective address** into the window —
//! `(addr + offset) & (size - 1)` — then adds the window base. This is exactly
//! [`svm_mask::Window::confine`] (the isolated, separately-fuzzed spec), so the JIT
//! and that unit lower the same arithmetic. The window allocation carries a small
//! guard margin so a masked base near the top plus the access width never escapes the
//! allocation (a real deployment uses guard *pages* + a fault for the width overrun).
//!
//! ## Calling convention
//! To support any arity behind one native signature, a compiled function is
//! `extern "C" fn(args: *const i64, results: *mut i64, mem_base: *mut u8)`: the entry
//! loads the IR parameters from `args` (one `i64` slot each), `return` stores results
//! to `results`, and loads/stores are relative to `mem_base`. The caller
//! ([`compile_and_run`]) marshals values and owns the window.

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::types::{I16, I32, I64, I8};
use cranelift_codegen::ir::{
    AbiParam, BlockArg, Endianness, Function, InstBuilder, MemFlags, Type, UserFuncName, Value,
};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use svm_ir::{
    BinOp, Block, Func, FuncIdx, Inst, IntUnOp, LoadOp, Module as IrModule, StoreOp, Terminator,
    ValType,
};

/// Largest window the reference JIT will back with a flat host allocation. Real
/// deployments reserve a huge guard-paged virtual range (§4); for the differential
/// harness we allocate `1 << size_log2` bytes (+ a guard margin), so cap it.
const MAX_JIT_WINDOW_LOG2: u8 = 26; // 64 MiB
/// Guard margin past the window so a masked base near the top plus an access width
/// never reads/writes outside the allocation (models the §4 guard region; a real
/// impl uses guard *pages* + a fault). 8 = widest scalar access.
const GUARD: usize = 8;

/// Why the JIT could not compile (or run) a function. The integer slice rejects
/// anything it does not yet lower with [`JitError::Unsupported`]; the differential
/// harness treats that as "skip", not "fail".
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum JitError {
    /// An instruction/terminator the current slice does not lower yet.
    Unsupported(&'static str),
    /// Structurally invalid (a verified module never hits this; defensive only).
    Malformed,
    /// Cranelift rejected the generated CLIF or failed to compile it.
    Backend(String),
}

/// The CLIF type backing an IR value type.
fn clif_ty(t: ValType) -> Type {
    match t {
        ValType::I32 => I32,
        ValType::I64 => I64,
        // Floats are not in this slice; lowering rejects them before this is reached.
        ValType::F32 | ValType::F64 => I64,
    }
}

/// Compile `func` and run it on slot-encoded `args` (each `i64` is one parameter
/// slot; `i32` params occupy the low 32 bits). Returns the result slots. Intended for
/// the differential harness (see the `svm` crate's JIT tests).
pub fn compile_and_run(m: &IrModule, func: FuncIdx, args: &[i64]) -> Result<Vec<i64>, JitError> {
    let f = m.funcs.get(func as usize).ok_or(JitError::Malformed)?;
    // Reject the not-yet-lowered surface up front so we never emit partial CLIF.
    ensure_supported(f)?;

    // Allocate the guest window (zeroed, + guard margin) if the module declares memory.
    // `mask` is the §4 confinement mask; `mem_base` is null when there is no window.
    let (window, mask): (Vec<u8>, u64) = match m.memory {
        Some(mc) => {
            if mc.size_log2 > MAX_JIT_WINDOW_LOG2 {
                return Err(JitError::Unsupported(
                    "window too large for the reference JIT",
                ));
            }
            let size = 1usize << mc.size_log2;
            (vec![0u8; size + GUARD], (size as u64) - 1)
        }
        None => (Vec::new(), 0),
    };

    let mut flags = settings::builder();
    // A JIT'd function is called directly, not relocated into a shared object.
    let _ = flags.set("is_pic", "false");
    let isa = cranelift_native::builder()
        .map_err(|e| JitError::Backend(e.to_string()))?
        .finish(settings::Flags::new(flags))
        .map_err(|e| JitError::Backend(e.to_string()))?;
    let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    let mut module = JITModule::new(builder);

    let mut ctx = module.make_context();
    build_clif(&mut ctx.func, f, mask)?;

    let id = module
        .declare_function("f", Linkage::Export, &ctx.func.signature)
        .map_err(|e| JitError::Backend(e.to_string()))?;
    module
        .define_function(id, &mut ctx)
        .map_err(|e| JitError::Backend(e.to_string()))?;
    module.clear_context(&mut ctx);
    module
        .finalize_definitions()
        .map_err(|e| JitError::Backend(e.to_string()))?;

    let code = module.get_finalized_function(id);
    // SAFETY: `code` points at a finalized function with exactly this signature
    // (`build_clif` set it). `module` owns the executable page and stays alive until
    // after the call below.
    let run: extern "C" fn(*const i64, *mut i64, *mut u8) = unsafe { std::mem::transmute(code) };

    let mut window = window;
    let mem_base = if window.is_empty() {
        std::ptr::null_mut()
    } else {
        window.as_mut_ptr()
    };
    let mut results = vec![0i64; f.results.len()];
    // SAFETY: `run` reads `f.params.len()` arg slots, writes `f.results.len()` result
    // slots, and accesses only `[mem_base, mem_base + size + GUARD)` (the masking
    // lowering confines every effective address to `< size`, plus the guard margin).
    // All three buffers outlive the call.
    run(args.as_ptr(), results.as_mut_ptr(), mem_base);
    drop(window);
    drop(module); // frees the executable memory after the call has returned
    Ok(results)
}

/// Reject functions using any op outside the integer slice, so `build_clif` can lower
/// the remainder totally. Keeping the check separate keeps the lowering readable.
fn ensure_supported(f: &Func) -> Result<(), JitError> {
    for ty in f.params.iter().chain(&f.results) {
        if matches!(ty, ValType::F32 | ValType::F64) {
            return Err(JitError::Unsupported("float types"));
        }
    }
    for blk in &f.blocks {
        for ty in &blk.params {
            if matches!(ty, ValType::F32 | ValType::F64) {
                return Err(JitError::Unsupported("float block params"));
            }
        }
        for inst in &blk.insts {
            match inst {
                Inst::ConstI32(_) | Inst::ConstI64(_) | Inst::Select { .. } => {}
                Inst::IntBin { op, .. } => match op {
                    // Trapping ops need fault plumbing (deferred).
                    BinOp::DivS | BinOp::DivU | BinOp::RemS | BinOp::RemU => {
                        return Err(JitError::Unsupported("trapping div/rem"))
                    }
                    _ => {}
                },
                Inst::IntCmp { .. } | Inst::Eqz { .. } => {}
                Inst::IntUn { op, .. } => match op {
                    IntUnOp::Clz | IntUnOp::Ctz | IntUnOp::Popcnt => {}
                    _ => return Err(JitError::Unsupported("int extend ops")),
                },
                Inst::Load { op, .. } => {
                    if matches!(op.info().1, ValType::F32 | ValType::F64) {
                        return Err(JitError::Unsupported("float load"));
                    }
                }
                Inst::Store { op, .. } => {
                    if matches!(op.info().1, ValType::F32 | ValType::F64) {
                        return Err(JitError::Unsupported("float store"));
                    }
                }
                _ => return Err(JitError::Unsupported("instruction")),
            }
        }
        match &blk.term {
            Terminator::Br { .. }
            | Terminator::BrIf { .. }
            | Terminator::Return(_)
            | Terminator::Unreachable => {}
            _ => return Err(JitError::Unsupported("terminator")),
        }
    }
    Ok(())
}

/// Per-function lowering context shared across blocks.
struct Lower {
    /// Holds `results_ptr` so any block's `return` can store to it.
    results_var: Variable,
    /// Holds `mem_base` (the window base) for load/store lowering.
    mem_var: Variable,
    /// The §4 confinement mask (`size - 1`); `0` when the module has no memory.
    mask: u64,
}

/// Build the CLIF for one IR function into `clif`.
fn build_clif(clif: &mut Function, f: &Func, mask: u64) -> Result<(), JitError> {
    if f.blocks.is_empty() {
        return Err(JitError::Malformed);
    }
    // Native signature: (args_ptr: i64, results_ptr: i64, mem_base: i64) -> ().
    clif.signature.params.push(AbiParam::new(I64));
    clif.signature.params.push(AbiParam::new(I64));
    clif.signature.params.push(AbiParam::new(I64));
    clif.name = UserFuncName::user(0, 0);

    let mut fbctx = FunctionBuilderContext::new();
    let mut b = FunctionBuilder::new(clif, &mut fbctx);

    // One CLIF block per IR block, with params mirroring the IR block params. A
    // separate CLIF entry block holds the native params and jumps into IR block 0
    // with the loaded function arguments.
    let blocks: Vec<_> = f.blocks.iter().map(|_| b.create_block()).collect();
    for (i, blk) in f.blocks.iter().enumerate() {
        for p in &blk.params {
            b.append_block_param(blocks[i], clif_ty(*p));
        }
    }
    let entry = b.create_block();
    b.append_block_param(entry, I64); // args_ptr
    b.append_block_param(entry, I64); // results_ptr
    b.append_block_param(entry, I64); // mem_base
    b.switch_to_block(entry);
    b.seal_block(entry);
    let args_ptr = b.block_params(entry)[0];
    let results_ptr = b.block_params(entry)[1];
    let mem_base = b.block_params(entry)[2];

    // `results_ptr` / `mem_base` are needed across blocks; stash them in variables so
    // any block can read them without threading them through block params.
    let results_var = b.declare_var(I64);
    b.def_var(results_var, results_ptr);
    let mem_var = b.declare_var(I64);
    b.def_var(mem_var, mem_base);
    let lower = Lower {
        results_var,
        mem_var,
        mask,
    };

    // Load the function arguments from the args buffer, narrowing i32 params.
    let mut entry_args: Vec<BlockArg> = Vec::with_capacity(f.params.len());
    for (i, p) in f.params.iter().enumerate() {
        let slot = b
            .ins()
            .load(I64, MemFlags::trusted(), args_ptr, (i * 8) as i32);
        let v = if clif_ty(*p) == I32 {
            b.ins().ireduce(I32, slot)
        } else {
            slot
        };
        entry_args.push(BlockArg::from(v));
    }
    b.ins().jump(blocks[0], &entry_args);

    for (i, blk) in f.blocks.iter().enumerate() {
        lower_block(&mut b, blk, blocks[i], &blocks, &lower)?;
    }

    b.seal_all_blocks();
    b.finalize();
    Ok(())
}

/// Lower one IR block's body + terminator into its CLIF block.
fn lower_block(
    b: &mut FunctionBuilder,
    blk: &Block,
    cb: cranelift_codegen::ir::Block,
    blocks: &[cranelift_codegen::ir::Block],
    lower: &Lower,
) -> Result<(), JitError> {
    b.switch_to_block(cb);
    // The CLIF block params are the IR block params; seed the value map with them.
    let mut vals: Vec<Value> = b.block_params(cb).to_vec();

    for inst in &blk.insts {
        let v = match inst {
            Inst::ConstI32(c) => b.ins().iconst(I32, *c as i64),
            Inst::ConstI64(c) => b.ins().iconst(I64, *c),
            Inst::IntBin { op, a, b: rb, .. } => {
                let (x, y) = (get(&vals, *a)?, get(&vals, *rb)?);
                int_bin(b, *op, x, y)
            }
            Inst::IntUn { op, a, .. } => {
                let x = get(&vals, *a)?;
                match op {
                    IntUnOp::Clz => b.ins().clz(x),
                    IntUnOp::Ctz => b.ins().ctz(x),
                    IntUnOp::Popcnt => b.ins().popcnt(x),
                    _ => return Err(JitError::Unsupported("int extend ops")),
                }
            }
            Inst::IntCmp { op, a, b: rb, .. } => {
                let (x, y) = (get(&vals, *a)?, get(&vals, *rb)?);
                let c = b.ins().icmp(int_cc(*op), x, y);
                b.ins().uextend(I32, c) // bool (I8) -> i32 0/1
            }
            Inst::Eqz { a, .. } => {
                let x = get(&vals, *a)?;
                let c = b.ins().icmp_imm(IntCC::Equal, x, 0);
                b.ins().uextend(I32, c)
            }
            Inst::Select { cond, a, b: rb } => {
                let (c, x, y) = (get(&vals, *cond)?, get(&vals, *a)?, get(&vals, *rb)?);
                b.ins().select(c, x, y)
            }
            Inst::Load {
                op, addr, offset, ..
            } => {
                let phys = mask_addr(b, lower, get(&vals, *addr)?, *offset);
                lower_load(b, *op, phys)
            }
            Inst::Store {
                op,
                addr,
                value,
                offset,
                ..
            } => {
                let phys = mask_addr(b, lower, get(&vals, *addr)?, *offset);
                lower_store(b, *op, phys, get(&vals, *value)?);
                continue; // store produces no value
            }
            _ => return Err(JitError::Unsupported("instruction")),
        };
        vals.push(v);
    }

    match &blk.term {
        Terminator::Br { target, args } => {
            let ba = map_args(&vals, args)?;
            let t = *blocks.get(*target as usize).ok_or(JitError::Malformed)?;
            b.ins().jump(t, &ba);
        }
        Terminator::BrIf {
            cond,
            then_blk,
            then_args,
            else_blk,
            else_args,
        } => {
            let c = get(&vals, *cond)?;
            let ta = map_args(&vals, then_args)?;
            let ea = map_args(&vals, else_args)?;
            let tb = *blocks.get(*then_blk as usize).ok_or(JitError::Malformed)?;
            let eb = *blocks.get(*else_blk as usize).ok_or(JitError::Malformed)?;
            b.ins().brif(c, tb, &ta, eb, &ea);
        }
        Terminator::Return(outs) => {
            let results_ptr = b.use_var(lower.results_var);
            for (i, o) in outs.iter().enumerate() {
                let v = get(&vals, *o)?;
                // Widen i32 results to fill the i64 slot (sign-extend; the harness
                // reads back only the low 32 bits for an i32 result type).
                let slot = if b.func.dfg.value_type(v) == I32 {
                    b.ins().sextend(I64, v)
                } else {
                    v
                };
                b.ins()
                    .store(MemFlags::trusted(), slot, results_ptr, (i * 8) as i32);
            }
            b.ins().return_(&[]);
        }
        Terminator::Unreachable => {
            b.ins()
                .trap(cranelift_codegen::ir::TrapCode::user(1).unwrap());
        }
        _ => return Err(JitError::Unsupported("terminator")),
    }
    Ok(())
}

fn get(vals: &[Value], i: u32) -> Result<Value, JitError> {
    vals.get(i as usize).copied().ok_or(JitError::Malformed)
}

/// The §4 confinement masking lowering (invariant I1): compute the physical address
/// `mem_base + ((addr + offset) & mask)`. The `(addr + offset) & mask` is exactly
/// `svm_mask::Window::confine`, so the JIT and the isolated masking unit agree.
fn mask_addr(b: &mut FunctionBuilder, lower: &Lower, addr: Value, offset: u64) -> Value {
    let off = b.ins().iconst(I64, offset as i64);
    let eff = b.ins().iadd(addr, off);
    let m = b.ins().iconst(I64, lower.mask as i64);
    let masked = b.ins().band(eff, m);
    let base = b.use_var(lower.mem_var);
    b.ins().iadd(base, masked)
}

/// Little-endian, may-trap memory access flags (the window is host memory; the guard
/// margin absorbs width overrun, so this never faults in practice).
fn mem_flags() -> MemFlags {
    let mut mf = MemFlags::new();
    mf.set_endianness(Endianness::Little);
    mf
}

/// The CLIF type holding `width` raw bytes.
fn width_ty(width: u32) -> Type {
    match width {
        1 => I8,
        2 => I16,
        4 => I32,
        _ => I64,
    }
}

fn lower_load(b: &mut FunctionBuilder, op: LoadOp, phys: Value) -> Value {
    let (_, rty, width, signed) = op.info();
    let load_ty = width_ty(width);
    let raw = b.ins().load(load_ty, mem_flags(), phys, 0);
    let result_ty = clif_ty(rty);
    if load_ty == result_ty {
        raw
    } else if signed {
        b.ins().sextend(result_ty, raw) // narrow signed load: sign-extend
    } else {
        b.ins().uextend(result_ty, raw) // narrow unsigned load: zero-extend
    }
}

fn lower_store(b: &mut FunctionBuilder, op: StoreOp, phys: Value, value: Value) {
    let (_, _, width) = op.info();
    let store_ty = width_ty(width);
    // Narrow stores keep only the low `width` bytes (matches the interpreter).
    let v = if b.func.dfg.value_type(value) == store_ty {
        value
    } else {
        b.ins().ireduce(store_ty, value)
    };
    b.ins().store(mem_flags(), v, phys, 0);
}

/// Map IR edge args to CLIF block-call args (`BlockArg`, the 0.132 block-call type).
fn map_args(vals: &[Value], args: &[u32]) -> Result<Vec<BlockArg>, JitError> {
    args.iter()
        .map(|a| get(vals, *a).map(BlockArg::from))
        .collect()
}

fn int_bin(b: &mut FunctionBuilder, op: BinOp, x: Value, y: Value) -> Value {
    match op {
        BinOp::Add => b.ins().iadd(x, y),
        BinOp::Sub => b.ins().isub(x, y),
        BinOp::Mul => b.ins().imul(x, y),
        BinOp::And => b.ins().band(x, y),
        BinOp::Or => b.ins().bor(x, y),
        BinOp::Xor => b.ins().bxor(x, y),
        BinOp::Shl => b.ins().ishl(x, y),
        BinOp::ShrS => b.ins().sshr(x, y),
        BinOp::ShrU => b.ins().ushr(x, y),
        BinOp::Rotl => b.ins().rotl(x, y),
        BinOp::Rotr => b.ins().rotr(x, y),
        // Trapping div/rem are rejected by `ensure_supported`.
        BinOp::DivS | BinOp::DivU | BinOp::RemS | BinOp::RemU => unreachable!("rejected earlier"),
    }
}

fn int_cc(op: svm_ir::CmpOp) -> IntCC {
    use svm_ir::CmpOp::*;
    match op {
        Eq => IntCC::Equal,
        Ne => IntCC::NotEqual,
        LtS => IntCC::SignedLessThan,
        LtU => IntCC::UnsignedLessThan,
        LeS => IntCC::SignedLessThanOrEqual,
        LeU => IntCC::UnsignedLessThanOrEqual,
        GtS => IntCC::SignedGreaterThan,
        GtU => IntCC::UnsignedGreaterThan,
        GeS => IntCC::SignedGreaterThanOrEqual,
        GeU => IntCC::UnsignedGreaterThanOrEqual,
    }
}
