//! Slice 7 — **completeness closure**: typing + encoding rows for the remaining
//! control / host / concurrency ops, the atomics, and the terminators. These carry no
//! semantic `eval` (host- and interleaving-dependent — the SPEC.md scope fence leaves
//! their behavior to `explore_all` / loom / the existing differentials), but they get:
//!
//! - a **minimal verifiable module** witnessing the op's typing rule (accepted by both
//!   `svm-verify` and the reference verifier, `spec_structural.rs`);
//! - a **`decode∘encode` round-trip** and an **opcode-byte pin** against the restated
//!   map, exactly like the value-op encoding suite.
//!
//! With these, every one of the 86 `Inst` variants and all 7 `Terminator`s is homed by
//! a spec row (see the tally in [`struct_rows`] and the `structural_row_tally` test),
//! finishing the exhaustive walk `coverage()` began — adding an op now forces a row.
//!
//! `CallImport` is the one op with `verifies: false`: its row module carries no import
//! **manifest**, and a `call.import` whose index names no declared import is fail-closed
//! (IMPORTS.md phase 1 — with a matching manifest it *is* executable; the manifest-bearing
//! accept/reject legs live in the `spec_verify` directed cases). Its row still round-trips
//! and pins its byte; only the verifier-accept leg is skipped for it.

use svm_ir::*;

use crate::Enc;

/// A structural (typing + encoding, no `eval`) op row.
pub struct StructRow {
    pub id: String,
    pub encoding: Enc,
    /// Whether the op's **row module** verifies (false only for `CallImport`, whose row
    /// carries no import manifest — a manifest-bearing `call.import` is valid, IMPORTS.md).
    pub verifies: bool,
    /// The op sits at `funcs[0].blocks[0].term` (true) or `.insts[0]` (false) — the
    /// encoding suite derives its opcode-pin baseline accordingly.
    pub is_term: bool,
    /// A module containing the op; verifiable when `verifies`.
    pub module: Module,
}

/// An entry function whose block-0 params are `params` and whose only instruction is
/// `op`, terminated by `unreachable` (which references nothing, so the op's result may
/// go unused without tripping the verifier). `extra` functions follow at indices `1..`.
fn inst_module(params: Vec<ValType>, op: Inst, needs_mem: bool, extra: Vec<Func>) -> Module {
    let mut funcs = vec![Func {
        params: params.clone(),
        results: vec![],
        blocks: vec![Block {
            params,
            insts: vec![op],
            term: Terminator::Unreachable,
        }],
    }];
    funcs.extend(extra);
    Module {
        funcs,
        memory: needs_mem.then_some(Memory { size_log2: 16 }),
        ..Default::default()
    }
}

/// An entry function whose block-0 (params `params`, no instructions) ends in `term`,
/// with `extra_blocks` appended (branch targets) and `extra_funcs` at indices `1..`.
fn term_module(
    params: Vec<ValType>,
    results: Vec<ValType>,
    term: Terminator,
    extra_blocks: Vec<Block>,
    extra_funcs: Vec<Func>,
) -> Module {
    let mut blocks = vec![Block {
        params: params.clone(),
        insts: vec![],
        term,
    }];
    blocks.extend(extra_blocks);
    let mut funcs = vec![Func {
        params,
        results,
        blocks,
    }];
    funcs.extend(extra_funcs);
    Module {
        funcs,
        ..Default::default()
    }
}

/// A trap-terminated `()->()` block, used as a branch target / call sink.
fn sink_block(params: Vec<ValType>) -> Block {
    Block {
        params,
        insts: vec![],
        term: Terminator::Unreachable,
    }
}

fn inst_row(id: &str, enc: Enc, params: Vec<ValType>, op: Inst, needs_mem: bool) -> StructRow {
    StructRow {
        id: id.into(),
        encoding: enc,
        verifies: true,
        is_term: false,
        module: inst_module(params, op, needs_mem, vec![]),
    }
}

/// Every remaining op, as a structural row. The tally (asserted by
/// `structural_row_tally`): 7 memory/atomic (4 atomics + fence + 2 `v128` mem); 3 calls
/// (`call`, `call_indirect`, `ref.func`); 7 host (`cap.call`, 4 `cap.self.*`,
/// `call_import`, `import_attach`); 7 concurrency; 6 misc control
/// (`setjmp`/`longjmp`/`gc.roots`/2 tls/`durable_shadow_base`); 7 terminators =
/// **37 rows**, which with the 80 scalar + 70 float + 26 memory + SIMD rows homes all
/// 87 `Inst` variants and 7 `Terminator`s.
// Sequential `push`es (not a `vec![]` literal) so the `call`/`thread.spawn` rows can
// name their intermediate callee `Func`s between pushes.
#[allow(clippy::vec_init_then_push)]
pub fn struct_rows() -> Vec<StructRow> {
    let i32t = ValType::I32;
    let i64t = ValType::I64;
    let v128t = ValType::V128;
    let ft_void = FuncType {
        params: vec![],
        results: vec![],
    };
    let mut rows = Vec::new();

    // ----- atomics + fence + v128 memory (§12/§17): all need a window -----
    rows.push(inst_row(
        "atomic.load",
        Enc::Byte(0xC6),
        vec![i64t],
        Inst::AtomicLoad {
            ty: IntTy::I32,
            addr: 0,
            offset: 0,
            order: Ordering::SeqCst,
        },
        true,
    ));
    rows.push(inst_row(
        "atomic.store",
        Enc::Byte(0xC7),
        vec![i64t, i32t],
        Inst::AtomicStore {
            ty: IntTy::I32,
            addr: 0,
            value: 1,
            offset: 0,
            order: Ordering::SeqCst,
        },
        true,
    ));
    rows.push(inst_row(
        "atomic.rmw",
        Enc::Byte(0xC8),
        vec![i64t, i32t],
        Inst::AtomicRmw {
            ty: IntTy::I32,
            op: AtomicRmwOp::Add,
            addr: 0,
            value: 1,
            offset: 0,
            order: Ordering::SeqCst,
        },
        true,
    ));
    rows.push(inst_row(
        "atomic.cmpxchg",
        Enc::Byte(0xC9),
        vec![i64t, i32t, i32t],
        Inst::AtomicCmpxchg {
            ty: IntTy::I32,
            addr: 0,
            expected: 1,
            replacement: 2,
            offset: 0,
            order: Ordering::SeqCst,
        },
        true,
    ));
    rows.push(inst_row(
        "atomic.fence",
        Enc::Byte(0xE9),
        vec![],
        Inst::AtomicFence {
            order: Ordering::SeqCst,
        },
        false,
    ));
    rows.push(inst_row(
        "v128.load",
        Enc::Prefixed(0xFE, 0x01),
        vec![i64t],
        Inst::V128Load {
            addr: 0,
            offset: 0,
            align: 0,
        },
        true,
    ));
    rows.push(inst_row(
        "v128.store",
        Enc::Prefixed(0xFE, 0x02),
        vec![i64t, v128t],
        Inst::V128Store {
            addr: 0,
            value: 1,
            offset: 0,
            align: 0,
        },
        true,
    ));

    // ----- calls + ref.func (§3c) -----
    // `call`/`ref.func`/`return_call` reference a real callee at index 1.
    let callee_void = Func {
        params: vec![],
        results: vec![],
        blocks: vec![sink_block(vec![])],
    };
    rows.push(StructRow {
        id: "call".into(),
        encoding: Enc::Byte(0x73),
        verifies: true,
        is_term: false,
        module: inst_module(
            vec![],
            Inst::Call {
                func: 1,
                args: vec![],
            },
            false,
            vec![callee_void.clone()],
        ),
    });
    rows.push(inst_row(
        "call_indirect",
        Enc::Byte(0x74),
        vec![i32t],
        Inst::CallIndirect {
            ty: ft_void.clone(),
            idx: 0,
            args: vec![],
        },
        false,
    ));
    rows.push(StructRow {
        id: "ref.func".into(),
        encoding: Enc::Byte(0x75),
        verifies: true,
        is_term: false,
        module: inst_module(vec![], Inst::RefFunc { func: 1 }, false, vec![callee_void]),
    });

    // ----- host: cap.call, cap.self.*, and the unresolved call_import (§7) -----
    rows.push(inst_row(
        "cap.call",
        Enc::Byte(0x79),
        vec![i32t],
        Inst::CapCall {
            type_id: 0,
            op: 0,
            sig: ft_void.clone(),
            handle: 0,
            args: vec![],
        },
        false,
    ));
    rows.push(inst_row(
        "cap.self.count",
        Enc::Byte(0x7A),
        vec![],
        Inst::CapSelfCount,
        false,
    ));
    rows.push(inst_row(
        "cap.self.get",
        Enc::Byte(0x7B),
        vec![i32t],
        Inst::CapSelfGet { idx: 0 },
        false,
    ));
    rows.push(inst_row(
        "cap.self.resolve",
        Enc::Byte(0x7E),
        vec![i64t, i64t],
        Inst::CapSelfResolve {
            name_ptr: 0,
            name_len: 1,
        },
        false,
    ));
    rows.push(inst_row(
        "cap.self.label",
        Enc::Byte(0x7F),
        vec![i32t, i64t, i64t],
        Inst::CapSelfLabel {
            handle: 0,
            buf_ptr: 1,
            buf_cap: 2,
        },
        false,
    ));
    // The pre-resolution import form: no valid module contains it (verifier rejects an
    // unresolved import), so `verifies: false` — round-trip + byte pin only.
    rows.push(StructRow {
        id: "call_import".into(),
        encoding: Enc::Byte(0x7C),
        verifies: false,
        is_term: false,
        module: inst_module(
            vec![i32t],
            Inst::CallImport {
                import: 0,
                op: 0,
                sig: ft_void.clone(),
                args: vec![],
            },
            false,
            vec![],
        ),
    });

    // v8 §7/§22 link-form symbolic call: the loader-ABI placeholder. Never verifies by
    // design (a surviving `call.sym` is an unresolved symbol) — round-trip + byte pin only.
    rows.push(StructRow {
        id: "call_sym".into(),
        encoding: Enc::Byte(0x0E),
        verifies: false,
        is_term: false,
        module: inst_module(
            vec![i32t],
            Inst::CallSym {
                import: 0,
                sig: ft_void.clone(),
                handle: 0,
                args: vec![],
            },
            false,
            vec![],
        ),
    });

    // Phase-2 `import.attach` (IMPORTS.md): like `call_import`, the row module carries no
    // manifest, so the op fails verification (out-of-range import) — round-trip + byte pin
    // only; the manifest-bearing accept/reject legs are `spec_verify` directed cases.
    rows.push(StructRow {
        id: "import_attach".into(),
        encoding: Enc::Byte(0x63),
        verifies: false,
        is_term: false,
        module: inst_module(
            vec![i32t],
            Inst::ImportAttach {
                import: 0,
                handle: 0,
            },
            false,
            vec![],
        ),
    });

    // ----- concurrency: fibers, threads, futex (§12) -----
    rows.push(inst_row(
        "cont.new",
        Enc::Byte(0xCA),
        vec![i32t, i64t],
        Inst::ContNew { func: 0, sp: 1 },
        false,
    ));
    rows.push(inst_row(
        "cont.resume",
        Enc::Byte(0xCB),
        vec![i64t, i64t],
        Inst::ContResume { k: 0, arg: 1 },
        false,
    ));
    rows.push(inst_row(
        "suspend",
        Enc::Byte(0xCC),
        vec![i64t],
        Inst::Suspend { value: 0 },
        false,
    ));
    // `thread.spawn`'s callee must have the fixed thread-entry signature (i64,i64)->i64.
    let thread_entry = Func {
        params: vec![i64t, i64t],
        results: vec![i64t],
        blocks: vec![Block {
            params: vec![i64t, i64t],
            insts: vec![],
            term: Terminator::Return(vec![0]),
        }],
    };
    rows.push(StructRow {
        id: "thread.spawn".into(),
        encoding: Enc::Byte(0xCD),
        verifies: true,
        is_term: false,
        module: inst_module(
            vec![i64t, i64t],
            Inst::ThreadSpawn {
                func: 1,
                sp: 0,
                arg: 1,
            },
            false,
            vec![thread_entry],
        ),
    });
    rows.push(inst_row(
        "thread.join",
        Enc::Byte(0xCE),
        vec![i32t],
        Inst::ThreadJoin { handle: 0 },
        false,
    ));
    rows.push(inst_row(
        "memory.wait",
        Enc::Byte(0xCF),
        vec![i64t, i32t, i64t],
        Inst::MemoryWait {
            ty: IntTy::I32,
            addr: 0,
            expected: 1,
            timeout: 2,
        },
        true,
    ));
    rows.push(inst_row(
        "memory.notify",
        Enc::Byte(0xE8),
        vec![i64t, i32t],
        Inst::MemoryNotify { addr: 0, count: 1 },
        true,
    ));

    // ----- misc control: setjmp/longjmp, gc.roots, TLS, durable-shadow -----
    rows.push(inst_row(
        "setjmp",
        Enc::Byte(0xEE),
        vec![i64t],
        Inst::SetJmp { buf: 0 },
        true,
    ));
    rows.push(inst_row(
        "longjmp",
        Enc::Byte(0xEF),
        vec![i64t, i32t],
        Inst::LongJmp { buf: 0, val: 1 },
        true,
    ));
    // A non-constant `mask` (a block param) is unconstrained by the verifier (the
    // top-byte-strip rule fires only on a *constant* fold-down mask).
    rows.push(inst_row(
        "gc.roots",
        Enc::Byte(0xEA),
        vec![i64t, i64t, i64t, i64t, i64t],
        Inst::GcRoots {
            heap_lo: 0,
            heap_hi: 1,
            mask: 2,
            buf: 3,
            cap: 4,
        },
        true,
    ));
    rows.push(inst_row(
        "vcpu.tls.get",
        Enc::Byte(0xEB),
        vec![],
        Inst::VcpuTlsGet,
        false,
    ));
    rows.push(inst_row(
        "vcpu.tls.set",
        Enc::Byte(0xEC),
        vec![i64t],
        Inst::VcpuTlsSet { val: 0 },
        false,
    ));
    rows.push(inst_row(
        "durable.shadow_base",
        Enc::Byte(0xED),
        vec![],
        Inst::DurableShadowBase,
        false,
    ));

    // ----- terminators (§3b rule 4) -----
    rows.push(StructRow {
        id: "br".into(),
        encoding: Enc::Byte(0x80),
        verifies: true,
        is_term: true,
        module: term_module(
            vec![],
            vec![],
            Terminator::Br {
                target: 1,
                args: vec![],
            },
            vec![sink_block(vec![])],
            vec![],
        ),
    });
    rows.push(StructRow {
        id: "br_if".into(),
        encoding: Enc::Byte(0x81),
        verifies: true,
        is_term: true,
        module: term_module(
            vec![i32t],
            vec![],
            Terminator::BrIf {
                cond: 0,
                then_blk: 1,
                then_args: vec![],
                else_blk: 2,
                else_args: vec![],
            },
            vec![sink_block(vec![]), sink_block(vec![])],
            vec![],
        ),
    });
    rows.push(StructRow {
        id: "br_table".into(),
        encoding: Enc::Byte(0x82),
        verifies: true,
        is_term: true,
        module: term_module(
            vec![i32t],
            vec![],
            Terminator::BrTable {
                idx: 0,
                targets: vec![(1, vec![])],
                default: (2, vec![]),
            },
            vec![sink_block(vec![]), sink_block(vec![])],
            vec![],
        ),
    });
    rows.push(StructRow {
        id: "return".into(),
        encoding: Enc::Byte(0x83),
        verifies: true,
        is_term: true,
        module: term_module(
            vec![i32t],
            vec![i32t],
            Terminator::Return(vec![0]),
            vec![],
            vec![],
        ),
    });
    // `return_call`'s callee results must equal this function's results (both `()`).
    let callee_void2 = Func {
        params: vec![],
        results: vec![],
        blocks: vec![sink_block(vec![])],
    };
    rows.push(StructRow {
        id: "return_call".into(),
        encoding: Enc::Byte(0x85),
        verifies: true,
        is_term: true,
        module: term_module(
            vec![],
            vec![],
            Terminator::ReturnCall {
                func: 1,
                args: vec![],
            },
            vec![],
            vec![callee_void2],
        ),
    });
    rows.push(StructRow {
        id: "return_call_indirect".into(),
        encoding: Enc::Byte(0x86),
        verifies: true,
        is_term: true,
        module: term_module(
            vec![i32t],
            vec![],
            Terminator::ReturnCallIndirect {
                ty: ft_void,
                idx: 0,
                args: vec![],
            },
            vec![],
            vec![],
        ),
    });
    rows.push(StructRow {
        id: "unreachable".into(),
        encoding: Enc::Byte(0x8F),
        verifies: true,
        is_term: true,
        module: term_module(vec![], vec![], Terminator::Unreachable, vec![], vec![]),
    });

    rows
}

/// Which slice's rows home an instruction (SPEC.md completeness). **Exhaustive over
/// `Inst`** — a new variant is a compile error here until the spec decides where it
/// lives, the third exhaustive forcing function (alongside `coverage()` and the
/// reference verifier's `check_inst`). Every arm is a *real* home: there is no
/// "uncovered" case, so the walk is closed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RowHome {
    /// A `scalar_rows`/`float_rows` value row with an `eval`.
    Value,
    /// A `mem_rows` window-model row.
    Memory,
    /// A `simd::simd_rows` lane row.
    Simd,
    /// A `struct_rows` typing+encoding row.
    Structural,
}

pub fn row_home(inst: &Inst) -> RowHome {
    match inst {
        // Value rows (slices 1–2): scalar int + float.
        Inst::ConstI32(_)
        | Inst::ConstI64(_)
        | Inst::ConstF32(_)
        | Inst::ConstF64(_)
        | Inst::IntBin { .. }
        | Inst::IntCmp { .. }
        | Inst::IntUn { .. }
        | Inst::Eqz { .. }
        | Inst::Convert { .. }
        | Inst::Select { .. }
        | Inst::FBin { .. }
        | Inst::FUn { .. }
        | Inst::Fma { .. }
        | Inst::FCmp { .. }
        | Inst::FToISat { .. }
        | Inst::FToITrap { .. }
        | Inst::IToFConv { .. }
        | Inst::Cast { .. }
        | Inst::PtrAdd { .. }
        | Inst::PtrCast { .. } => RowHome::Value,

        // Memory-window rows (slice 5).
        Inst::Load { .. }
        | Inst::Store { .. }
        | Inst::MemCopy { .. }
        | Inst::MemMove { .. }
        | Inst::MemFill { .. } => RowHome::Memory,

        // SIMD lane rows (slice 6).
        Inst::ConstV128(_)
        | Inst::Splat { .. }
        | Inst::ExtractLane { .. }
        | Inst::ReplaceLane { .. }
        | Inst::VIntBin { .. }
        | Inst::VIntCmp { .. }
        | Inst::VFloatCmp { .. }
        | Inst::VShift { .. }
        | Inst::VIntUn { .. }
        | Inst::VSatBin { .. }
        | Inst::VWiden { .. }
        | Inst::VNarrow { .. }
        | Inst::VConvert { .. }
        | Inst::VPMinMax { .. }
        | Inst::VPopcnt { .. }
        | Inst::VAvgr { .. }
        | Inst::VDot { .. }
        | Inst::VDotI8 { .. }
        | Inst::VExtMul { .. }
        | Inst::VExtAddPairwise { .. }
        | Inst::VQ15MulrSat { .. }
        | Inst::VFma { .. }
        | Inst::VAnyTrue { .. }
        | Inst::VAllTrue { .. }
        | Inst::VBitmask { .. }
        | Inst::VFloatBin { .. }
        | Inst::VFloatUn { .. }
        | Inst::VBitBin { .. }
        | Inst::VNot { .. }
        | Inst::Bitselect { .. }
        | Inst::Shuffle { .. }
        | Inst::Swizzle { .. }
        | Inst::SimdWidthBytes => RowHome::Simd,

        // Structural rows (slice 7): atomics + v128 memory, calls, host, concurrency,
        // misc control.
        Inst::AtomicLoad { .. }
        | Inst::AtomicStore { .. }
        | Inst::AtomicRmw { .. }
        | Inst::AtomicCmpxchg { .. }
        | Inst::AtomicFence { .. }
        | Inst::V128Load { .. }
        | Inst::V128Store { .. }
        | Inst::Call { .. }
        | Inst::CallIndirect { .. }
        | Inst::RefFunc { .. }
        | Inst::CapCall { .. }
        | Inst::CallImport { .. }
        | Inst::CallImportDyn { .. }
        | Inst::CallSym { .. }
        | Inst::ExportHandle { .. }
        | Inst::ImportAttach { .. }
        | Inst::CapSelfCount
        | Inst::CapSelfAttest
        | Inst::CapSelfGet { .. }
        | Inst::CapSelfResolve { .. }
        | Inst::CapSelfLabel { .. }
        | Inst::CapSelfTypeId { .. }
        | Inst::CapSelfCovers { .. }
        | Inst::ContNew { .. }
        | Inst::ContResume { .. }
        | Inst::Suspend { .. }
        | Inst::ThreadSpawn { .. }
        | Inst::ThreadJoin { .. }
        | Inst::MemoryWait { .. }
        | Inst::MemoryNotify { .. }
        | Inst::SetJmp { .. }
        | Inst::LongJmp { .. }
        | Inst::GcRoots { .. }
        | Inst::VcpuTlsGet
        | Inst::VcpuTlsSet { .. }
        | Inst::DurableShadowBase => RowHome::Structural,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn structural_row_tally() {
        let rows = struct_rows();
        assert_eq!(rows.len(), 38, "structural row count (update on new ops)");
        let mut ids: Vec<&str> = rows.iter().map(|r| r.id.as_str()).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), rows.len(), "duplicate structural row id");
        // Exactly three ops have row modules that fail verification: `call_import` +
        // `import_attach` (valid only against a declared manifest, IMPORTS.md; the
        // manifest-bearing accept legs live in `spec_verify`) and `call_sym` (the v8
        // link-form placeholder, which *never* verifies by design).
        assert_eq!(rows.iter().filter(|r| !r.verifies).count(), 3);
        assert_eq!(rows.iter().filter(|r| r.is_term).count(), 7, "terminators");
    }

    /// Cross-check that `row_home` and `struct_rows` agree on the structural set: every
    /// non-terminator structural row's op is `RowHome::Structural`. (Terminators aren't
    /// `Inst`s, so they're excluded here — `structural_row_tally` counts them instead.)
    #[test]
    fn structural_rows_are_structural_homed() {
        for row in struct_rows().iter().filter(|r| !r.is_term) {
            let op = &row.module.funcs[0].blocks[0].insts[0];
            assert_eq!(
                row_home(op),
                RowHome::Structural,
                "{} is not Structural-homed",
                row.id
            );
        }
    }
}
