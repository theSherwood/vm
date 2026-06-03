//! Cranelift JIT backend (`DESIGN.md` §9, §18).
//!
//! Cranelift is the chosen codegen **by design** (§1): it is the security mechanism,
//! not a liability — we share Wasmtime's most security-critical component, so the
//! escape-TCB *delta* we own is just this CLIF generation plus the §4 masking
//! lowering. Correctness is established by **differential testing against the
//! reference interpreter** (§18, invariants I1/I4), the oracle in `svm-interp`.
//!
//! ## Status
//! Lowers the full scalar surface: `i32`/`i64`/`f32`/`f64` consts, all integer and
//! float arithmetic/bitwise/shift/rotate/compare ops (incl. trapping `div`/`rem`),
//! `eqz`/`select`/`clz`/`ctz`/`popcnt`, every conversion (extend/wrap/demote/promote/
//! reinterpret, int↔float, saturating **and** trapping `trunc`), **integer loads/
//! stores with confinement masking** (the security-critical I1 lowering), and the
//! `br`/`br_if`/`br_table`/`return`/`return_call`/`unreachable` terminators incl.
//! direct and tail calls. Anything else returns [`JitError::Unsupported`] so the
//! differential harness skips it. Still ahead: indirect calls + `cap.call` (the
//! function/handle tables). Trapping ops abort the process if executed (no
//! trap-catching infra yet); the harness only runs the JIT on non-trapping inputs.
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

use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::types::{F32, F64, I16, I32, I64, I8};
use cranelift_codegen::ir::{
    AbiParam, BlockArg, BlockCall, Endianness, Function, InstBuilder, JumpTableData, MemFlags,
    Type, UserFuncName, Value,
};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module};
use svm_ir::{
    BinOp, Block, CastOp, ConvOp, FBinOp, FCmpOp, FUnOp, FloatTy, Func, FuncIdx, Inst, IntTy,
    IntUnOp, LoadOp, Module as IrModule, StoreOp, Terminator, ValType,
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
        ValType::F32 => F32,
        ValType::F64 => F64,
    }
}

/// The CLIF type for an integer-class IR type (operands to int↔float conversions).
fn int_clif_ty(t: IntTy) -> Type {
    match t {
        IntTy::I32 => I32,
        IntTy::I64 => I64,
    }
}

/// The CLIF type for a float-class IR type.
fn float_clif_ty(t: FloatTy) -> Type {
    match t {
        FloatTy::F32 => F32,
        FloatTy::F64 => F64,
    }
}

/// Compile the whole module and run `func` on slot-encoded `args` (each `i64` is one
/// parameter slot; `i32`/`f32` occupy the low 32 bits). Returns the result slots.
/// Intended for the differential harness (see the `svm` crate's JIT tests).
///
/// All functions are compiled with a **natural CLIF ABI** — `(mem_base, params…) ->
/// (results…)` — so direct/tail calls are ordinary CLIF calls; the entry is wrapped
/// in a fixed buffer-ABI trampoline so any arity is callable from Rust.
pub fn compile_and_run(m: &IrModule, func: FuncIdx, args: &[i64]) -> Result<Vec<i64>, JitError> {
    let entry = m.funcs.get(func as usize).ok_or(JitError::Malformed)?;
    // Calls can reach any function, so every function must be lowerable.
    for f in &m.funcs {
        ensure_supported(f)?;
    }

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
    // Cranelift's x64 `return_call` (tail calls, §3b) lowering requires frame pointers.
    let _ = flags.set("preserve_frame_pointers", "true");
    let isa = cranelift_native::builder()
        .map_err(|e| JitError::Backend(e.to_string()))?
        .finish(settings::Flags::new(flags))
        .map_err(|e| JitError::Backend(e.to_string()))?;
    let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    let mut module = JITModule::new(builder);

    // Declare every function (natural ABI) up front so calls can reference any of them.
    let ids: Vec<_> = m
        .funcs
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let sig = natural_sig(&mut module, f);
            module
                .declare_function(&format!("f{i}"), Linkage::Local, &sig)
                .map_err(|e| JitError::Backend(e.to_string()))
        })
        .collect::<Result<_, _>>()?;

    // Define each function body. `clear_context` after each define resets the cached
    // CFG/domtree so the next function never compiles against a stale CFG.
    let mut ctx = module.make_context();
    for (f, id) in m.funcs.iter().zip(&ids) {
        build_clif(&mut module, &ids, &mut ctx.func, f, mask)?;
        module
            .define_function(*id, &mut ctx)
            .map_err(|e| JitError::Backend(e.to_string()))?;
        module.clear_context(&mut ctx);
    }

    // The buffer-ABI trampoline for the entry, exported so Rust can call it.
    build_trampoline(&mut module, &mut ctx.func, ids[func as usize], entry);
    let tramp = module
        .declare_function("trampoline", Linkage::Export, &ctx.func.signature)
        .map_err(|e| JitError::Backend(e.to_string()))?;
    module
        .define_function(tramp, &mut ctx)
        .map_err(|e| JitError::Backend(e.to_string()))?;
    module.clear_context(&mut ctx);
    module
        .finalize_definitions()
        .map_err(|e| JitError::Backend(e.to_string()))?;

    let code = module.get_finalized_function(tramp);
    // SAFETY: `code` is the finalized trampoline with exactly this signature
    // (`build_trampoline` set it). `module` owns the executable page until dropped.
    let run: extern "C" fn(*const i64, *mut i64, *mut u8) = unsafe { std::mem::transmute(code) };

    let mut window = window;
    let mem_base = if window.is_empty() {
        std::ptr::null_mut()
    } else {
        window.as_mut_ptr()
    };
    let mut results = vec![0i64; entry.results.len()];
    // SAFETY: `run` reads `entry.params.len()` arg slots, writes `entry.results.len()`
    // result slots, and accesses only `[mem_base, mem_base + size + GUARD)` (masking
    // confines every effective address to `< size`, plus the guard margin). All three
    // buffers outlive the call.
    run(args.as_ptr(), results.as_mut_ptr(), mem_base);
    drop(window);
    drop(module); // frees the executable memory after the call has returned
    Ok(results)
}

/// The natural CLIF signature for an IR function: `(mem_base: i64, params…) ->
/// (results…)`. `mem_base` is threaded through every call so loads/stores reach the
/// window without a global.
fn natural_sig(module: &mut JITModule, f: &Func) -> cranelift_codegen::ir::Signature {
    let mut sig = module.make_signature();
    // The `tail` calling convention so `return_call` (guaranteed tail calls, §3b) is
    // available; a normal `call` from the trampoline works against it too.
    sig.call_conv = cranelift_codegen::isa::CallConv::Tail;
    sig.params.push(AbiParam::new(I64)); // mem_base
    for p in &f.params {
        sig.params.push(AbiParam::new(clif_ty(*p)));
    }
    for r in &f.results {
        sig.returns.push(AbiParam::new(clif_ty(*r)));
    }
    sig
}

/// Reject functions using any op outside the integer slice, so `build_clif` can lower
/// the remainder totally. Keeping the check separate keeps the lowering readable.
fn ensure_supported(f: &Func) -> Result<(), JitError> {
    for blk in &f.blocks {
        for inst in &blk.insts {
            match inst {
                Inst::ConstI32(_)
                | Inst::ConstI64(_)
                | Inst::ConstF32(_)
                | Inst::ConstF64(_)
                | Inst::Select { .. }
                | Inst::IntCmp { .. }
                | Inst::Eqz { .. }
                | Inst::FBin { .. }
                | Inst::FUn { .. }
                | Inst::FCmp { .. }
                | Inst::FToISat { .. }
                | Inst::FToITrap { .. }
                | Inst::IToFConv { .. }
                | Inst::Cast { .. }
                | Inst::Load { .. }
                | Inst::Store { .. }
                | Inst::Call { .. }
                | Inst::IntBin { .. }
                | Inst::Convert { .. } => {}
                Inst::IntUn { op, .. } => match op {
                    IntUnOp::Clz | IntUnOp::Ctz | IntUnOp::Popcnt => {}
                    _ => return Err(JitError::Unsupported("int extend ops")),
                },
                _ => return Err(JitError::Unsupported("instruction")),
            }
        }
        match &blk.term {
            Terminator::Br { .. }
            | Terminator::BrIf { .. }
            | Terminator::BrTable { .. }
            | Terminator::Return(_)
            | Terminator::ReturnCall { .. }
            | Terminator::Unreachable => {}
            _ => return Err(JitError::Unsupported("terminator")),
        }
    }
    Ok(())
}

/// Per-function lowering context shared across blocks.
struct Lower<'a> {
    /// Holds `mem_base` (the window base) for load/store lowering and call threading.
    mem_var: Variable,
    /// The §4 confinement mask (`size - 1`); `0` when the module has no memory.
    mask: u64,
    /// Every function's `FuncId`, so `call`/`return_call` can reference callees.
    ids: &'a [FuncId],
}

/// Build the natural-ABI CLIF for one IR function: `(mem_base, params…) ->
/// (results…)`. The CLIF entry block holds the native params and jumps into IR
/// block 0 passing the parameters as its block args.
fn build_clif(
    module: &mut JITModule,
    ids: &[FuncId],
    clif: &mut Function,
    f: &Func,
    mask: u64,
) -> Result<(), JitError> {
    if f.blocks.is_empty() {
        return Err(JitError::Malformed);
    }
    clif.signature = natural_sig(module, f);
    clif.name = UserFuncName::user(0, 0);

    let mut fbctx = FunctionBuilderContext::new();
    let mut b = FunctionBuilder::new(clif, &mut fbctx);

    // One CLIF block per IR block, with params mirroring the IR block params. A
    // separate CLIF entry block holds the native params and jumps into IR block 0.
    let blocks: Vec<_> = f.blocks.iter().map(|_| b.create_block()).collect();
    for (i, blk) in f.blocks.iter().enumerate() {
        for p in &blk.params {
            b.append_block_param(blocks[i], clif_ty(*p));
        }
    }
    let entry = b.create_block();
    b.append_block_param(entry, I64); // mem_base
    for p in &f.params {
        b.append_block_param(entry, clif_ty(*p));
    }
    b.switch_to_block(entry);
    b.seal_block(entry);
    let mem_base = b.block_params(entry)[0];

    // `mem_base` is needed across blocks; stash it in a variable.
    let mem_var = b.declare_var(I64);
    b.def_var(mem_var, mem_base);
    let lower = Lower { mem_var, mask, ids };

    // Jump into IR block 0 passing the function parameters (entry params after mem_base).
    let entry_args: Vec<BlockArg> = b.block_params(entry)[1..]
        .iter()
        .map(|v| BlockArg::from(*v))
        .collect();
    b.ins().jump(blocks[0], &entry_args);

    for (i, blk) in f.blocks.iter().enumerate() {
        lower_block(module, &mut b, blk, blocks[i], &blocks, &lower)?;
    }

    b.seal_all_blocks();
    b.finalize();
    Ok(())
}

/// Build the fixed buffer-ABI trampoline `fn(args_ptr, results_ptr, mem_base)` that
/// decodes the entry function's args from `args_ptr`, calls it (natural ABI), and
/// stores its results to `results_ptr`. This is what Rust calls, so any arity works.
fn build_trampoline(module: &mut JITModule, clif: &mut Function, entry_id: FuncId, entry: &Func) {
    clif.signature.params.push(AbiParam::new(I64)); // args_ptr
    clif.signature.params.push(AbiParam::new(I64)); // results_ptr
    clif.signature.params.push(AbiParam::new(I64)); // mem_base
    clif.name = UserFuncName::user(0, 1);

    let mut fbctx = FunctionBuilderContext::new();
    let mut b = FunctionBuilder::new(clif, &mut fbctx);
    let blk = b.create_block();
    b.append_block_param(blk, I64);
    b.append_block_param(blk, I64);
    b.append_block_param(blk, I64);
    b.switch_to_block(blk);
    b.seal_block(blk);
    let args_ptr = b.block_params(blk)[0];
    let results_ptr = b.block_params(blk)[1];
    let mem_base = b.block_params(blk)[2];

    // Decode args (mem_base first), call the entry, store results.
    let mut call_args = vec![mem_base];
    for (i, p) in entry.params.iter().enumerate() {
        let slot = b
            .ins()
            .load(I64, MemFlags::trusted(), args_ptr, (i * 8) as i32);
        call_args.push(decode_slot(&mut b, slot, *p));
    }
    let callee = module.declare_func_in_func(entry_id, b.func);
    let call = b.ins().call(callee, &call_args);
    let rets: Vec<Value> = b.inst_results(call).to_vec();
    for (i, r) in rets.iter().enumerate() {
        let slot = encode_slot(&mut b, *r);
        b.ins()
            .store(MemFlags::trusted(), slot, results_ptr, (i * 8) as i32);
    }
    b.ins().return_(&[]);
    b.seal_all_blocks();
    b.finalize();
}

/// Lower one IR block's body + terminator into its CLIF block.
fn lower_block(
    module: &mut JITModule,
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
        // `call` appends 0..N results — handle it before the single-value match.
        if let Inst::Call { func, args } = inst {
            let callee_id = *lower.ids.get(*func as usize).ok_or(JitError::Malformed)?;
            let callee = module.declare_func_in_func(callee_id, b.func);
            let mut cargs = vec![b.use_var(lower.mem_var)];
            for a in args {
                cargs.push(get(&vals, *a)?);
            }
            let call = b.ins().call(callee, &cargs);
            vals.extend_from_slice(b.inst_results(call));
            continue;
        }
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
            Inst::ConstF32(bits) => {
                // Materialize via the exact bit pattern (NaN-safe), then bitcast.
                let i = b.ins().iconst(I32, *bits as i64);
                b.ins().bitcast(F32, MemFlags::new(), i)
            }
            Inst::ConstF64(bits) => {
                let i = b.ins().iconst(I64, *bits as i64);
                b.ins().bitcast(F64, MemFlags::new(), i)
            }
            Inst::FBin { op, a, b: rb, .. } => {
                let (x, y) = (get(&vals, *a)?, get(&vals, *rb)?);
                float_bin(b, *op, x, y)
            }
            Inst::FUn { op, a, .. } => {
                let x = get(&vals, *a)?;
                float_un(b, *op, x)
            }
            Inst::FCmp { op, a, b: rb, .. } => {
                let (x, y) = (get(&vals, *a)?, get(&vals, *rb)?);
                let c = b.ins().fcmp(float_cc(*op), x, y);
                b.ins().uextend(I32, c) // bool (I8) -> i32 0/1
            }
            Inst::Convert { op, a } => {
                let x = get(&vals, *a)?;
                match op {
                    ConvOp::ExtendI32S => b.ins().sextend(I64, x),
                    ConvOp::ExtendI32U => b.ins().uextend(I64, x),
                    ConvOp::WrapI64 => b.ins().ireduce(I32, x),
                }
            }
            Inst::Cast { op, a } => {
                let x = get(&vals, *a)?;
                match op {
                    CastOp::Demote => b.ins().fdemote(F32, x),
                    CastOp::Promote => b.ins().fpromote(F64, x),
                    CastOp::ReinterpI32F32 => b.ins().bitcast(F32, MemFlags::new(), x),
                    CastOp::ReinterpF32I32 => b.ins().bitcast(I32, MemFlags::new(), x),
                    CastOp::ReinterpI64F64 => b.ins().bitcast(F64, MemFlags::new(), x),
                    CastOp::ReinterpF64I64 => b.ins().bitcast(I64, MemFlags::new(), x),
                }
            }
            Inst::IToFConv { op, a } => {
                let x = get(&vals, *a)?;
                let (_, to, signed) = op.parts();
                let fty = float_clif_ty(to);
                if signed {
                    b.ins().fcvt_from_sint(fty, x)
                } else {
                    b.ins().fcvt_from_uint(fty, x)
                }
            }
            Inst::FToISat { op, a } => {
                let x = get(&vals, *a)?;
                let (_, to, signed) = op.parts();
                let ity = int_clif_ty(to);
                // Saturating (wasm trunc_sat): NaN→0, out-of-range→clamp — exactly
                // Cranelift's saturating fcvt, so it matches the interpreter.
                if signed {
                    b.ins().fcvt_to_sint_sat(ity, x)
                } else {
                    b.ins().fcvt_to_uint_sat(ity, x)
                }
            }
            Inst::FToITrap { op, a } => {
                let x = get(&vals, *a)?;
                let (_, to, signed) = op.parts();
                let ity = int_clif_ty(to);
                // Trapping (wasm trunc): NaN/out-of-range trap — Cranelift's
                // non-saturating fcvt traps on exactly those, matching the interpreter.
                if signed {
                    b.ins().fcvt_to_sint(ity, x)
                } else {
                    b.ins().fcvt_to_uint(ity, x)
                }
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
        Terminator::BrTable {
            idx,
            targets,
            default,
        } => {
            let index = get(&vals, *idx)?;
            // Build a BlockCall (target block + its edge args) for each table entry
            // and the default; Cranelift masks the index and selects, default on OOB.
            let mut entries = Vec::with_capacity(targets.len());
            for (t, args) in targets {
                let ba = map_args(&vals, args)?;
                let blk = *blocks.get(*t as usize).ok_or(JitError::Malformed)?;
                entries.push(BlockCall::new(
                    blk,
                    ba.iter().copied(),
                    &mut b.func.dfg.value_lists,
                ));
            }
            let (dt, dargs) = default;
            let dba = map_args(&vals, dargs)?;
            let dblk = *blocks.get(*dt as usize).ok_or(JitError::Malformed)?;
            let dcall = BlockCall::new(dblk, dba.iter().copied(), &mut b.func.dfg.value_lists);
            let jt = b.create_jump_table(JumpTableData::new(dcall, &entries));
            b.ins().br_table(index, jt);
        }
        Terminator::Return(outs) => {
            // Natural ABI: return the result values directly (CLIF multi-return).
            let rets: Vec<Value> = outs
                .iter()
                .map(|o| get(&vals, *o))
                .collect::<Result<_, _>>()?;
            b.ins().return_(&rets);
        }
        Terminator::ReturnCall { func, args } => {
            // Tail call (§3b): replace this frame with the callee, threading mem_base.
            let callee_id = *lower.ids.get(*func as usize).ok_or(JitError::Malformed)?;
            let callee = module.declare_func_in_func(callee_id, b.func);
            let mut cargs = vec![b.use_var(lower.mem_var)];
            for a in args {
                cargs.push(get(&vals, *a)?);
            }
            b.ins().return_call(callee, &cargs);
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
    // Float loads read the float type directly (no extension).
    if matches!(rty, ValType::F32 | ValType::F64) {
        return b.ins().load(clif_ty(rty), mem_flags(), phys, 0);
    }
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
    let (_, vty, width) = op.info();
    // Float stores write the float bits directly.
    if matches!(vty, ValType::F32 | ValType::F64) {
        b.ins().store(mem_flags(), value, phys, 0);
        return;
    }
    let store_ty = width_ty(width);
    // Narrow stores keep only the low `width` bytes (matches the interpreter).
    let v = if b.func.dfg.value_type(value) == store_ty {
        value
    } else {
        b.ins().ireduce(store_ty, value)
    };
    b.ins().store(mem_flags(), v, phys, 0);
}

/// Decode an `i64` calling-convention slot to a value of IR type `ty`.
fn decode_slot(b: &mut FunctionBuilder, slot: Value, ty: ValType) -> Value {
    match ty {
        ValType::I64 => slot,
        ValType::I32 => b.ins().ireduce(I32, slot),
        ValType::F32 => {
            let i = b.ins().ireduce(I32, slot);
            b.ins().bitcast(F32, MemFlags::new(), i)
        }
        ValType::F64 => b.ins().bitcast(F64, MemFlags::new(), slot),
    }
}

/// Encode a value into its `i64` calling-convention slot (the harness reads back the
/// low 32 bits for i32/f32 results).
fn encode_slot(b: &mut FunctionBuilder, v: Value) -> Value {
    match b.func.dfg.value_type(v) {
        I64 => v,
        I32 => b.ins().uextend(I64, v),
        F32 => {
            let i = b.ins().bitcast(I32, MemFlags::new(), v);
            b.ins().uextend(I64, i)
        }
        F64 => b.ins().bitcast(I64, MemFlags::new(), v),
        _ => v,
    }
}

fn float_bin(b: &mut FunctionBuilder, op: FBinOp, x: Value, y: Value) -> Value {
    match op {
        FBinOp::Add => b.ins().fadd(x, y),
        FBinOp::Sub => b.ins().fsub(x, y),
        FBinOp::Mul => b.ins().fmul(x, y),
        FBinOp::Div => b.ins().fdiv(x, y),
        FBinOp::Min => b.ins().fmin(x, y),
        FBinOp::Max => b.ins().fmax(x, y),
        FBinOp::Copysign => b.ins().fcopysign(x, y),
    }
}

fn float_un(b: &mut FunctionBuilder, op: FUnOp, x: Value) -> Value {
    match op {
        FUnOp::Abs => b.ins().fabs(x),
        FUnOp::Neg => b.ins().fneg(x),
        FUnOp::Sqrt => b.ins().sqrt(x),
        FUnOp::Ceil => b.ins().ceil(x),
        FUnOp::Floor => b.ins().floor(x),
        FUnOp::Trunc => b.ins().trunc(x),
        FUnOp::Nearest => b.ins().nearest(x),
    }
}

fn float_cc(op: FCmpOp) -> FloatCC {
    match op {
        FCmpOp::Eq => FloatCC::Equal,
        FCmpOp::Ne => FloatCC::NotEqual, // unordered ≠ (NaN ne x is true), wasm semantics
        FCmpOp::Lt => FloatCC::LessThan,
        FCmpOp::Le => FloatCC::LessThanOrEqual,
        FCmpOp::Gt => FloatCC::GreaterThan,
        FCmpOp::Ge => FloatCC::GreaterThanOrEqual,
    }
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
        // Trapping div/rem: Cranelift's sdiv/udiv trap on /0 (and sdiv on INT_MIN/-1
        // overflow); srem/urem trap on /0 only and define INT_MIN%-1 = 0 — exactly
        // the interpreter's semantics. Trapping inputs are never JIT-run (the harness
        // skips them), so a trap here would be a genuine divergence.
        BinOp::DivS => b.ins().sdiv(x, y),
        BinOp::DivU => b.ins().udiv(x, y),
        BinOp::RemS => b.ins().srem(x, y),
        BinOp::RemU => b.ins().urem(x, y),
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
