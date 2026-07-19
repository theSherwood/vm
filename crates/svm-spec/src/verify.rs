//! The **reference verifier** (SPEC.md suite 2): an independent second implementation
//! of the DESIGN.md §3b/§3c validity rules, written from the prose and the `svm-ir`
//! type documentation — deliberately *not* from `svm-verify`'s code. The conformance
//! suite (`crates/svm/tests/spec_verify.rs`) asserts the two implementations agree on
//! accept/reject over the row modules, directed rule mutations, and an `irgen` sweep —
//! closing the accept-direction gap: a production-verifier bug now needs the *same*
//! bug here, independently, to survive.
//!
//! Errors are plain strings naming the violated rule — agreement is on accept/reject;
//! pinning `svm-verify`'s precise error *variants* is the directed mutation tests' job.

use svm_ir::*;

type R = Result<(), String>;

/// Verify a whole module against the §3b/§3c rules. `Ok(())` ⇔ the module is valid.
pub fn verify(m: &Module) -> R {
    // §3a: a declared window must have a representable power-of-two size.
    if let Some(mem) = &m.memory {
        if mem.size_log2 >= 64 {
            return Err(format!(
                "memory size_log2 {} unrepresentable",
                mem.size_log2
            ));
        }
    }
    // §3a/D40: every data segment needs a window and must fit `[0, size)` without
    // offset+len overflow.
    for (i, d) in m.data.iter().enumerate() {
        let Some(mem) = &m.memory else {
            return Err(format!("data segment {i} without declared memory"));
        };
        match d.offset.checked_add(d.bytes.len() as u64) {
            Some(end) if end <= mem.size() => {}
            _ => return Err(format!("data segment {i} outside the window")),
        }
    }
    // §7 / IMPORTS.md phase 1: import names must be uniquely resolvable (the instantiation
    // policy binds by name), mirroring the export-name rule below.
    for (i, imp) in m.imports.iter().enumerate() {
        if m.imports[..i].iter().any(|o| o.name == imp.name) {
            return Err(format!("duplicate import name {:?}", imp.name));
        }
    }
    for (fi, f) in m.funcs.iter().enumerate() {
        verify_func(f, &m.funcs, &m.imports, m.memory.is_some())
            .map_err(|e| format!("fn{fi}: {e}"))?;
    }
    // Exports name real functions, uniquely.
    for (i, e) in m.exports.iter().enumerate() {
        if e.func as usize >= m.funcs.len() {
            return Err(format!("export {i} names function {} out of range", e.func));
        }
        if m.exports[..i].iter().any(|o| o.name == e.name) {
            return Err(format!("duplicate export name {:?}", e.name));
        }
    }
    Ok(())
}

fn verify_func(f: &Func, funcs: &[Func], imports: &[Import], has_memory: bool) -> R {
    // §3b rule 2: the entry block's params equal the function signature's params.
    match f.blocks.first() {
        Some(entry) if entry.params == f.params => {}
        _ => return Err("entry block params != function params (or no blocks)".into()),
    }
    let fn_results: Vec<usize> = funcs.iter().map(|f| f.results.len()).collect();
    for b in &f.blocks {
        let mut types: Vec<ValType> = b.params.clone();
        for inst in &b.insts {
            check_inst(inst, &mut types, funcs, imports, has_memory, b, &fn_results)?;
        }
        check_term(&b.term, &types, f, funcs)?;
    }
    Ok(())
}

/// §3b rule 3: operand `v` is defined strictly earlier and has exactly type `want`.
fn want(types: &[ValType], v: ValIdx, want: ValType) -> R {
    match types.get(v as usize) {
        None => Err(format!(
            "value v{v} not defined yet ({} defined)",
            types.len()
        )),
        Some(t) if *t == want => Ok(()),
        Some(t) => Err(format!("v{v}: expected {want:?}, found {t:?}")),
    }
}

fn need_memory(has_memory: bool) -> R {
    if has_memory {
        Ok(())
    } else {
        Err("memory op without a declared window".into())
    }
}

/// One instruction: check operands, append result type(s). Exhaustive over `Inst`, so
/// a new instruction forces a rule decision here (the same forcing function as
/// [`crate::coverage`]).
fn check_inst(
    inst: &Inst,
    types: &mut Vec<ValType>,
    funcs: &[Func],
    imports: &[Import],
    has_memory: bool,
    block: &Block,
    fn_results: &[usize],
) -> R {
    use ValType as V;
    let w = |v: ValIdx, t: V| want(types, v, t);
    // The type(s) this instruction appends; computed before mutating `types`.
    let push: Vec<V> = match inst {
        Inst::ConstI32(_) => vec![V::I32],
        Inst::ConstI64(_) => vec![V::I64],
        Inst::ConstF32(_) => vec![V::F32],
        Inst::ConstF64(_) => vec![V::F64],
        Inst::IntBin { ty, a, b, .. } => {
            w(*a, ty.val())?;
            w(*b, ty.val())?;
            vec![ty.val()]
        }
        Inst::IntCmp { ty, a, b, .. } => {
            w(*a, ty.val())?;
            w(*b, ty.val())?;
            vec![V::I32]
        }
        Inst::IntUn { ty, a, .. } => {
            w(*a, ty.val())?;
            vec![ty.val()]
        }
        Inst::Eqz { ty, a } => {
            w(*a, ty.val())?;
            vec![V::I32]
        }
        Inst::Convert { op, a } => {
            let (_, src, dst) = op.sig();
            w(*a, src)?;
            vec![dst]
        }
        // §3b: the one polymorphic op — `a` fixes the type, `b` must match exactly.
        Inst::Select { cond, a, b } => {
            w(*cond, V::I32)?;
            let t = *types
                .get(*a as usize)
                .ok_or_else(|| format!("select a v{a} not defined"))?;
            w(*b, t)?;
            vec![t]
        }
        Inst::FBin { ty, a, b, .. } => {
            w(*a, ty.val())?;
            w(*b, ty.val())?;
            vec![ty.val()]
        }
        Inst::FUn { ty, a, .. } => {
            w(*a, ty.val())?;
            vec![ty.val()]
        }
        Inst::Fma { ty, a, b, c } => {
            w(*a, ty.val())?;
            w(*b, ty.val())?;
            w(*c, ty.val())?;
            vec![ty.val()]
        }
        Inst::FCmp { ty, a, b, .. } => {
            w(*a, ty.val())?;
            w(*b, ty.val())?;
            vec![V::I32]
        }
        Inst::FToISat { op, a } | Inst::FToITrap { op, a } => {
            let (from, to, _) = op.parts();
            w(*a, from.val())?;
            vec![to.val()]
        }
        Inst::IToFConv { op, a } => {
            let (from, to, _) = op.parts();
            w(*a, from.val())?;
            vec![to.val()]
        }
        Inst::Cast { op, a } => {
            let (_, src, dst) = op.sig();
            w(*a, src)?;
            vec![dst]
        }
        Inst::PtrAdd { a, b } => {
            w(*a, V::I64)?;
            w(*b, V::I64)?;
            vec![V::I64]
        }
        Inst::PtrCast { a, .. } => {
            w(*a, V::I64)?;
            vec![V::I64]
        }

        // ----- memory (§3b/§4): addresses are i64; every access needs a window -----
        Inst::Load { op, addr, .. } => {
            need_memory(has_memory)?;
            w(*addr, V::I64)?;
            vec![op.info().1]
        }
        Inst::Store {
            op, addr, value, ..
        } => {
            need_memory(has_memory)?;
            w(*addr, V::I64)?;
            w(*value, op.info().1)?;
            vec![]
        }
        Inst::MemCopy { dst, src, len } | Inst::MemMove { dst, src, len } => {
            need_memory(has_memory)?;
            w(*dst, V::I64)?;
            w(*src, V::I64)?;
            w(*len, V::I64)?;
            vec![]
        }
        Inst::MemFill { dst, val, len } => {
            need_memory(has_memory)?;
            w(*dst, V::I64)?;
            w(*val, V::I32)?;
            w(*len, V::I64)?;
            vec![]
        }

        // ----- atomics (§12): a load may not be release-flavored, a store may not be
        // acquire-flavored; a fence takes any ordering -----
        Inst::AtomicLoad {
            ty, addr, order, ..
        } => {
            need_memory(has_memory)?;
            if !order.valid_for_load() {
                return Err("atomic load with release ordering".into());
            }
            w(*addr, V::I64)?;
            vec![ty.val()]
        }
        Inst::AtomicStore {
            ty,
            addr,
            value,
            order,
            ..
        } => {
            need_memory(has_memory)?;
            if !order.valid_for_store() {
                return Err("atomic store with acquire ordering".into());
            }
            w(*addr, V::I64)?;
            w(*value, ty.val())?;
            vec![]
        }
        Inst::AtomicRmw {
            ty, addr, value, ..
        } => {
            need_memory(has_memory)?;
            w(*addr, V::I64)?;
            w(*value, ty.val())?;
            vec![ty.val()]
        }
        Inst::AtomicCmpxchg {
            ty,
            addr,
            expected,
            replacement,
            ..
        } => {
            need_memory(has_memory)?;
            w(*addr, V::I64)?;
            w(*expected, ty.val())?;
            w(*replacement, ty.val())?;
            vec![ty.val()]
        }
        Inst::AtomicFence { .. } => vec![],
        Inst::MemoryWait {
            ty,
            addr,
            expected,
            timeout,
        } => {
            need_memory(has_memory)?;
            w(*addr, V::I64)?;
            w(*expected, ty.val())?;
            w(*timeout, V::I64)?;
            vec![V::I32]
        }
        Inst::MemoryNotify { addr, count } => {
            need_memory(has_memory)?;
            w(*addr, V::I64)?;
            w(*count, V::I32)?;
            vec![V::I32]
        }

        // ----- calls (§3b rule 5 / §3c): args match the signature exactly -----
        Inst::Call { func, args } => {
            let callee = funcs
                .get(*func as usize)
                .ok_or_else(|| format!("call to out-of-range fn{func}"))?;
            check_args(types, args, &callee.params)?;
            callee.results.clone()
        }
        Inst::CallIndirect { ty, idx, args } => {
            w(*idx, V::I32)?;
            check_args(types, args, &ty.params)?;
            ty.results.clone()
        }
        Inst::RefFunc { func } => {
            if *func as usize >= funcs.len() {
                return Err(format!("ref.func to out-of-range fn{func}"));
            }
            vec![V::I32] // a funcref is a plain i32 table index (§3c/D37)
        }
        Inst::CapCall {
            sig, handle, args, ..
        } => {
            w(*handle, V::I32)?; // forgeable index; safety is the runtime check (D37)
            check_args(types, args, &sig.params)?;
            sig.results.clone()
        }
        // §7 / IMPORTS.md phase 1: a `call.import` is executable when its index names a declared
        // import and its self-describing sig equals the manifest's (the canonical interface);
        // out-of-range (including the empty-manifest legacy shape) or a sig disagreement is
        // fail-closed. Operand typing mirrors `cap.call` (handle i32 + args per sig).
        Inst::CallImport {
            import,
            sig,
            handle,
            args,
        } => {
            let Some(decl) = imports.get(*import as usize) else {
                return Err(format!(
                    "unresolved import {import} (out of manifest range)"
                ));
            };
            if decl.sig != *sig {
                return Err(format!("import {import} signature mismatch with manifest"));
            }
            w(*handle, V::I32)?;
            check_args(types, args, &sig.params)?;
            sig.results.clone()
        }
        Inst::CapSelfCount => vec![V::I32],
        Inst::CapSelfAttest => vec![V::I32],
        Inst::CapSelfGet { idx } => {
            w(*idx, V::I32)?;
            vec![V::I32, V::I32]
        }
        Inst::CapSelfResolve { name_ptr, name_len } => {
            w(*name_ptr, V::I64)?;
            w(*name_len, V::I64)?;
            vec![V::I32]
        }
        Inst::CapSelfLabel {
            handle,
            buf_ptr,
            buf_cap,
        } => {
            w(*handle, V::I32)?;
            w(*buf_ptr, V::I64)?;
            w(*buf_cap, V::I64)?;
            vec![V::I32]
        }

        // ----- fibers / threads / TLS (§12) -----
        Inst::ContNew { func, sp } => {
            w(*func, V::I32)?;
            w(*sp, V::I64)?;
            vec![V::I64]
        }
        Inst::ContResume { k, arg } => {
            w(*k, V::I64)?;
            w(*arg, V::I64)?;
            vec![V::I32, V::I64] // (status, value)
        }
        Inst::Suspend { value } => {
            w(*value, V::I64)?;
            vec![V::I64]
        }
        Inst::ThreadSpawn { func, sp, arg } => {
            let callee = funcs
                .get(*func as usize)
                .ok_or_else(|| format!("thread.spawn of out-of-range fn{func}"))?;
            // §12: the thread entry signature is fixed — (i64 sp, i64 arg) -> i64.
            if callee.params != [V::I64, V::I64] || callee.results != [V::I64] {
                return Err(format!("thread.spawn entry fn{func} has wrong signature"));
            }
            w(*sp, V::I64)?;
            w(*arg, V::I64)?;
            vec![V::I32]
        }
        Inst::ThreadJoin { handle } => {
            w(*handle, V::I32)?;
            vec![V::I64]
        }
        Inst::VcpuTlsGet => vec![V::I64],
        Inst::VcpuTlsSet { val } => {
            w(*val, V::I64)?;
            vec![]
        }
        Inst::DurableShadowBase => vec![V::I64],

        // ----- setjmp/longjmp: touch the guest jmp_buf, so they need a window -----
        Inst::SetJmp { buf } => {
            need_memory(has_memory)?;
            w(*buf, V::I64)?;
            vec![V::I32]
        }
        Inst::LongJmp { buf, val } => {
            need_memory(has_memory)?;
            w(*buf, V::I64)?;
            w(*val, V::I32)?;
            vec![]
        }

        // ----- GC root enumeration (GC.md): writes into the window; a *constant*
        // payload mask may only clear the top byte -----
        Inst::GcRoots {
            heap_lo,
            heap_hi,
            mask,
            buf,
            cap,
        } => {
            need_memory(has_memory)?;
            for v in [heap_lo, heap_hi, mask, buf, cap] {
                w(*v, V::I64)?;
            }
            if let Some(m) = const_i64(block, fn_results, *mask) {
                if (m as u64) | 0xFF00_0000_0000_0000 != u64::MAX {
                    return Err(format!("gc.roots constant mask {m:#x} clears low bits"));
                }
            }
            vec![V::I64]
        }

        // ----- SIMD (§17/D58): total lane typing -----
        Inst::ConstV128(_) => vec![V::V128],
        Inst::V128Load { addr, .. } => {
            need_memory(has_memory)?;
            w(*addr, V::I64)?;
            vec![V::V128]
        }
        Inst::V128Store { addr, value, .. } => {
            need_memory(has_memory)?;
            w(*addr, V::I64)?;
            w(*value, V::V128)?;
            vec![]
        }
        Inst::Splat { shape, a } => {
            w(*a, shape.lane_val())?;
            vec![V::V128]
        }
        Inst::ExtractLane { shape, lane, a, .. } => {
            if *lane >= shape.lanes() {
                return Err(format!("lane {lane} out of range for {shape:?}"));
            }
            w(*a, V::V128)?;
            vec![shape.lane_val()]
        }
        Inst::ReplaceLane {
            shape, lane, a, b, ..
        } => {
            if *lane >= shape.lanes() {
                return Err(format!("lane {lane} out of range for {shape:?}"));
            }
            w(*a, V::V128)?;
            w(*b, shape.lane_val())?;
            vec![V::V128]
        }
        Inst::VIntBin { shape, a, b, .. } | Inst::VIntCmp { shape, a, b, .. } => {
            if shape.is_float() {
                return Err("integer-lane op with a float shape".into());
            }
            w(*a, V::V128)?;
            w(*b, V::V128)?;
            vec![V::V128]
        }
        Inst::VShift { shape, a, amt, .. } => {
            if shape.is_float() {
                return Err("shift with a float shape".into());
            }
            w(*a, V::V128)?;
            w(*amt, V::I32)?;
            vec![V::V128]
        }
        Inst::VIntUn { shape, a, .. } => {
            if shape.is_float() {
                return Err("integer-lane op with a float shape".into());
            }
            w(*a, V::V128)?;
            vec![V::V128]
        }
        Inst::VFloatBin { shape, a, b, .. }
        | Inst::VFloatCmp { shape, a, b, .. }
        | Inst::VPMinMax { shape, a, b, .. } => {
            if !shape.is_float() {
                return Err("float-lane op with an integer shape".into());
            }
            w(*a, V::V128)?;
            w(*b, V::V128)?;
            vec![V::V128]
        }
        Inst::VFloatUn { shape, a, .. } => {
            if !shape.is_float() {
                return Err("float-lane op with an integer shape".into());
            }
            w(*a, V::V128)?;
            vec![V::V128]
        }
        Inst::VFma { shape, a, b, c, .. } => {
            if !shape.is_float() {
                return Err("float-lane op with an integer shape".into());
            }
            w(*a, V::V128)?;
            w(*b, V::V128)?;
            w(*c, V::V128)?;
            vec![V::V128]
        }
        // Saturating add/sub and rounding average exist for i8x16/i16x8 only.
        Inst::VSatBin { shape, a, b, .. } | Inst::VAvgr { shape, a, b } => {
            if !matches!(shape, VShape::I8x16 | VShape::I16x8) {
                return Err("sat/avgr op with a shape other than i8x16/i16x8".into());
            }
            w(*a, V::V128)?;
            w(*b, V::V128)?;
            vec![V::V128]
        }
        // Fixed-shape binary vector ops: nothing beyond v128 operands to enforce.
        Inst::VDot { a, b }
        | Inst::VDotI8 { a, b }
        | Inst::VQ15MulrSat { a, b }
        | Inst::Swizzle { a, b }
        | Inst::VBitBin { a, b, .. } => {
            w(*a, V::V128)?;
            w(*b, V::V128)?;
            vec![V::V128]
        }
        // Widen/extmul: the result shape must have a half-width source.
        Inst::VWiden { shape, a, .. } => {
            if shape.narrower().is_none() {
                return Err("widen to a shape with no half-width source".into());
            }
            w(*a, V::V128)?;
            vec![V::V128]
        }
        Inst::VExtMul { shape, a, b, .. } => {
            if shape.narrower().is_none() {
                return Err("extmul to a shape with no half-width source".into());
            }
            w(*a, V::V128)?;
            w(*b, V::V128)?;
            vec![V::V128]
        }
        Inst::VExtAddPairwise { shape, a, .. } => {
            if !matches!(shape, VShape::I16x8 | VShape::I32x4) {
                return Err("extadd_pairwise shape must be i16x8/i32x4".into());
            }
            w(*a, V::V128)?;
            vec![V::V128]
        }
        Inst::VNarrow { shape, a, b, .. } => {
            if !matches!(shape, VShape::I8x16 | VShape::I16x8) {
                return Err("narrow to a shape other than i8x16/i16x8".into());
            }
            w(*a, V::V128)?;
            w(*b, V::V128)?;
            vec![V::V128]
        }
        Inst::VConvert { a, .. } | Inst::VPopcnt { a } | Inst::VNot { a } => {
            w(*a, V::V128)?;
            vec![V::V128]
        }
        Inst::VAnyTrue { a } => {
            w(*a, V::V128)?;
            vec![V::I32]
        }
        Inst::VAllTrue { shape, a } | Inst::VBitmask { shape, a } => {
            if shape.is_float() {
                return Err("boolean reduction with a float shape".into());
            }
            w(*a, V::V128)?;
            vec![V::I32]
        }
        Inst::Bitselect { a, b, mask } => {
            w(*a, V::V128)?;
            w(*b, V::V128)?;
            w(*mask, V::V128)?;
            vec![V::V128]
        }
        Inst::Shuffle { lanes, a, b } => {
            if lanes.iter().any(|&l| l >= 32) {
                return Err("shuffle lane index >= 32".into());
            }
            w(*a, V::V128)?;
            w(*b, V::V128)?;
            vec![V::V128]
        }
        Inst::SimdWidthBytes => vec![V::I32],
    };
    types.extend_from_slice(&push);
    Ok(())
}

fn check_args(types: &[ValType], args: &[ValIdx], params: &[ValType]) -> R {
    if args.len() != params.len() {
        return Err(format!(
            "call arg count {} != param count {}",
            args.len(),
            params.len()
        ));
    }
    for (a, p) in args.iter().zip(params) {
        want(types, *a, *p)?;
    }
    Ok(())
}

/// §3b rule 4 + terminator typing: exactly-one-terminator is structural in `svm-ir`
/// (`Block::term`), so what remains is edge/return/tail-call checking.
fn check_term(term: &Terminator, types: &[ValType], f: &Func, funcs: &[Func]) -> R {
    use ValType as V;
    let edge = |target: BlockIdx, args: &[ValIdx]| -> R {
        let tb = f
            .blocks
            .get(target as usize)
            .ok_or_else(|| format!("branch to out-of-range block{target}"))?;
        if args.len() != tb.params.len() {
            return Err(format!(
                "branch to block{target}: {} args for {} params",
                args.len(),
                tb.params.len()
            ));
        }
        for (a, p) in args.iter().zip(&tb.params) {
            want(types, *a, *p)?;
        }
        Ok(())
    };
    match term {
        Terminator::Br { target, args } => edge(*target, args),
        Terminator::BrIf {
            cond,
            then_blk,
            then_args,
            else_blk,
            else_args,
        } => {
            want(types, *cond, V::I32)?;
            edge(*then_blk, then_args)?;
            edge(*else_blk, else_args)
        }
        Terminator::BrTable {
            idx,
            targets,
            default,
        } => {
            want(types, *idx, V::I32)?;
            for (t, args) in targets {
                edge(*t, args)?;
            }
            edge(default.0, &default.1)
        }
        Terminator::Return(vals) => {
            if vals.len() != f.results.len() {
                return Err(format!(
                    "return of {} values from a {}-result function",
                    vals.len(),
                    f.results.len()
                ));
            }
            for (v, r) in vals.iter().zip(&f.results) {
                want(types, *v, *r)?;
            }
            Ok(())
        }
        // A tail call's args match the callee, and the callee's results must equal
        // THIS function's results exactly (§3b — the callee returns on our behalf).
        Terminator::ReturnCall { func, args } => {
            let callee = funcs
                .get(*func as usize)
                .ok_or_else(|| format!("return_call to out-of-range fn{func}"))?;
            check_args(types, args, &callee.params)?;
            if callee.results != f.results {
                return Err("tail-callee results != function results".into());
            }
            Ok(())
        }
        Terminator::ReturnCallIndirect { ty, idx, args } => {
            want(types, *idx, V::I32)?;
            check_args(types, args, &ty.params)?;
            if ty.results != f.results {
                return Err("tail-callee results != function results".into());
            }
            Ok(())
        }
        Terminator::Unreachable => Ok(()),
    }
}

/// The constant an operand resolves to when its defining instruction is an earlier
/// `i64.const` in this block (params and non-consts give `None`). Value numbering:
/// params take `0..params.len()`, then each instruction its `result_count` indices.
fn const_i64(b: &Block, fn_results: &[usize], v: ValIdx) -> Option<i64> {
    let mut idx = b.params.len() as u32;
    for inst in &b.insts {
        let n = inst.result_count(fn_results) as u32;
        if v >= idx && v < idx + n {
            return match inst {
                Inst::ConstI64(c) => Some(*c),
                _ => None,
            };
        }
        idx += n;
    }
    None
}
