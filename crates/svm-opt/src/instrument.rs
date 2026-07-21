//! **Memory-access instrumentation** (hooks): rewrite a module so that every guest linear-memory
//! access is announced to a host capability *before* it executes.
//!
//! The pass inserts, ahead of each memory op, a `cap.call` to an embedder-bound hook capability
//! carrying the event kind (the cap-call `op`) and the access coordinates (effective address +
//! width, or span operands for bulk ops). The engines are untouched: an instrumented module is an
//! ordinary module, so all three backends run it — and, by the §3 parity invariant, produce the
//! same event stream — with zero cost to modules that are not instrumented (they are byte-for-byte
//! unchanged and never see this pass).
//!
//! Like the rest of this crate, the pass is **untrusted for escape** (§2a/§20a): callers re-verify
//! the output (`svm_verify::verify_module`) before running it, so a bug here is a clean verify
//! error, never an escape.
//!
//! Event contract (the hook fires **pre-access, pre-confinement-check**, so the final event of a
//! faulting run is the *attempted* faulting access):
//! - scalar / v128 / atomic ops: `op = kind`, `args = [effective_addr, width]` — the effective
//!   address is the base operand plus the op's immediate offset (materialized as a wrapping i64
//!   add, matching both backends' address fold);
//! - `mem.copy` / `mem.move`: `op = COPY`, `args = [dst, src, len]` (one event per bulk op —
//!   consumers expand the span themselves);
//! - `mem.fill`: `op = FILL`, `args = [dst, len]`.
//!
//! Deliberately **not** hooked (runtime-internal or host-side accesses, not guest data ops):
//! futex `wait`/`notify` word touches, `setjmp`/`longjmp` jmp_buf traffic, `gc.roots` scans,
//! `cap.self.resolve`/`label` name/label buffers, and host-side `GuestMem` access from other
//! capability handlers. Accesses removed by a frontend's SSA promotion never reach the IR, so no
//! hook design can see them; the trace is of the post-promotion module.
//!
//! Fuel note: the inserted instructions consume fuel like any others, so an instrumented module's
//! fuel consumption differs from the pristine module's. [`MemHookStats::inserted_insts`] lets an
//! embedder scale its fuel limit. `debug_info` is dropped (its `(func, block, inst)` positions
//! would be stale after insertion).

use alloc::vec;
use alloc::vec::Vec;

use crate::{map_operands, map_term_operands};
use svm_ir::{BinOp, Func, FuncType, Inst, IntTy, Module, ValIdx, ValType};

/// The hook capability's event kinds — the `op` immediate of each inserted `cap.call`. An
/// embedder's handler dispatches on these (the first argument of the `HostFn` ABI).
pub mod mem_hook_op {
    /// `[addr, width]` — plain and v128 loads (v128 is width 16).
    pub const LOAD: u32 = 0;
    /// `[addr, width]` — plain and v128 stores.
    pub const STORE: u32 = 1;
    /// `[addr, width]`.
    pub const ATOMIC_LOAD: u32 = 2;
    /// `[addr, width]`.
    pub const ATOMIC_STORE: u32 = 3;
    /// `[addr, width]` — read-modify-write (one event, not a load + store pair).
    pub const ATOMIC_RMW: u32 = 4;
    /// `[addr, width]` — compare-exchange (one event; fires whether or not the swap happens).
    pub const ATOMIC_CMPXCHG: u32 = 5;
    /// `[dst, src, len]` — `mem.copy` and `mem.move`.
    pub const COPY: u32 = 6;
    /// `[dst, len]` — `mem.fill`.
    pub const FILL: u32 = 7;
}

/// Where the inserted `cap.call`s point: the hook capability's interface `type_id` (the embedder's
/// host-fn interface) and the concrete `handle` the host will have granted by the time the module
/// runs. The handle is baked as a constant, so the granting side must mint exactly this value
/// (grants are deterministic; svm-run grants the hook first on a fresh host and asserts).
#[derive(Clone, Copy, Debug)]
pub struct MemHookSpec {
    pub type_id: u32,
    pub handle: i32,
}

/// What the pass did — [`inserted_insts`](MemHookStats::inserted_insts) is the extra per-run op
/// count an embedder can use to scale fuel limits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemHookStats {
    /// Memory ops that got a hook call.
    pub hooked_ops: usize,
    /// Total instructions inserted (consts + address adds + cap.calls).
    pub inserted_insts: usize,
}

/// Instrument every guest memory access of `m` with a pre-access `cap.call` to the hook capability
/// described by `spec`. Pure `Module -> Module`; the input is untouched. Callers must re-verify
/// the result before running it (fail-closed, like every pass in this crate).
pub fn instrument_mem_hooks(m: &Module, spec: MemHookSpec) -> (Module, MemHookStats) {
    let fn_results: Vec<usize> = m.funcs.iter().map(|f| f.results.len()).collect();
    let mut out = m.clone();
    // Positions key on (func, block, inst) — stale after insertion, so drop rather than mislead.
    out.debug_info = None;
    let mut stats = MemHookStats::default();
    for f in &mut out.funcs {
        instrument_func(f, &fn_results, spec, &mut stats);
    }
    (out, stats)
}

/// The hook event a memory op announces, with operands still as **old** (pre-renumbering)
/// block-local indices.
enum Ev {
    Access {
        kind: u32,
        addr: ValIdx,
        offset: u64,
        width: u32,
    },
    Copy {
        dst: ValIdx,
        src: ValIdx,
        len: ValIdx,
    },
    Fill {
        dst: ValIdx,
        len: ValIdx,
    },
}

fn int_width(ty: IntTy) -> u32 {
    match ty {
        IntTy::I32 => 4,
        IntTy::I64 => 8,
    }
}

/// Classify a memory op into its hook event, or `None` for every other instruction. Exhaustive on
/// the memory-op set on purpose — the excluded runtime-internal accesses are listed in the module
/// docs; a *new* guest data-memory op must be added here (see the `classification_is_exhaustive`
/// test, which cross-checks against [`Inst::effects`]).
fn mem_event_of(inst: &Inst) -> Option<Ev> {
    use mem_hook_op as k;
    Some(match *inst {
        Inst::Load {
            op, addr, offset, ..
        } => Ev::Access {
            kind: k::LOAD,
            addr,
            offset,
            width: op.info().2,
        },
        Inst::Store {
            op, addr, offset, ..
        } => Ev::Access {
            kind: k::STORE,
            addr,
            offset,
            width: op.info().2,
        },
        Inst::V128Load { addr, offset, .. } => Ev::Access {
            kind: k::LOAD,
            addr,
            offset,
            width: 16,
        },
        Inst::V128Store { addr, offset, .. } => Ev::Access {
            kind: k::STORE,
            addr,
            offset,
            width: 16,
        },
        Inst::AtomicLoad {
            ty, addr, offset, ..
        } => Ev::Access {
            kind: k::ATOMIC_LOAD,
            addr,
            offset,
            width: int_width(ty),
        },
        Inst::AtomicStore {
            ty, addr, offset, ..
        } => Ev::Access {
            kind: k::ATOMIC_STORE,
            addr,
            offset,
            width: int_width(ty),
        },
        Inst::AtomicRmw {
            ty, addr, offset, ..
        } => Ev::Access {
            kind: k::ATOMIC_RMW,
            addr,
            offset,
            width: int_width(ty),
        },
        Inst::AtomicCmpxchg {
            ty, addr, offset, ..
        } => Ev::Access {
            kind: k::ATOMIC_CMPXCHG,
            addr,
            offset,
            width: int_width(ty),
        },
        Inst::MemCopy { dst, src, len } | Inst::MemMove { dst, src, len } => {
            Ev::Copy { dst, src, len }
        }
        Inst::MemFill { dst, len, .. } => Ev::Fill { dst, len },
        _ => return None,
    })
}

fn instrument_func(
    f: &mut Func,
    fn_results: &[usize],
    spec: MemHookSpec,
    stats: &mut MemHookStats,
) {
    for blk in &mut f.blocks {
        if !blk.insts.iter().any(|i| mem_event_of(i).is_some()) {
            continue;
        }
        // Rebuild the block with hook calls inserted. The wire form is block-local SSA (params
        // first, then one index per instruction result), so insertion shifts every later index:
        // `remap[old] = new` carries the fixup, applied through the exhaustive operand remapper.
        let params = blk.params.len() as u32;
        let old_insts = core::mem::take(&mut blk.insts);
        let mut insts: Vec<Inst> = Vec::with_capacity(old_insts.len());
        let mut remap: Vec<ValIdx> = (0..params).collect();
        let mut next: ValIdx = params;

        // Push a single-result instruction, returning its new value index.
        let push1 = |insts: &mut Vec<Inst>, next: &mut ValIdx, i: Inst| -> ValIdx {
            insts.push(i);
            let v = *next;
            *next += 1;
            v
        };

        for inst in old_insts {
            if let Some(ev) = mem_event_of(&inst) {
                let before = insts.len();
                let handle = push1(&mut insts, &mut next, Inst::ConstI32(spec.handle));
                let (op, args): (u32, Vec<ValIdx>) = match ev {
                    Ev::Access {
                        kind,
                        addr,
                        offset,
                        width,
                    } => {
                        // The effective address the access will attempt: base + immediate offset,
                        // as a wrapping i64 add (the same fold both backends confine).
                        let base = remap[addr as usize];
                        let eff = if offset == 0 {
                            base
                        } else {
                            let off = push1(&mut insts, &mut next, Inst::ConstI64(offset as i64));
                            push1(
                                &mut insts,
                                &mut next,
                                Inst::IntBin {
                                    ty: IntTy::I64,
                                    op: BinOp::Add,
                                    a: base,
                                    b: off,
                                },
                            )
                        };
                        let w = push1(&mut insts, &mut next, Inst::ConstI64(width as i64));
                        (kind, vec![eff, w])
                    }
                    Ev::Copy { dst, src, len } => (
                        mem_hook_op::COPY,
                        vec![
                            remap[dst as usize],
                            remap[src as usize],
                            remap[len as usize],
                        ],
                    ),
                    Ev::Fill { dst, len } => (
                        mem_hook_op::FILL,
                        vec![remap[dst as usize], remap[len as usize]],
                    ),
                };
                insts.push(Inst::CapCall {
                    type_id: spec.type_id,
                    op,
                    sig: FuncType {
                        params: vec![ValType::I64; args.len()],
                        results: vec![],
                    },
                    handle,
                    args,
                });
                stats.hooked_ops += 1;
                stats.inserted_insts += insts.len() - before;
            }
            let n = inst.result_count(fn_results);
            let mut ni = inst;
            map_operands(&mut ni, &mut |v| remap[v as usize]);
            insts.push(ni);
            for _ in 0..n {
                remap.push(next);
                next += 1;
            }
        }
        map_term_operands(&mut blk.term, &mut |v| remap[v as usize]);
        blk.insts = insts;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use svm_ir::{Block, LoadOp, Memory, StoreOp, Terminator};

    // `handle` = what the first grant on a fresh `Host` mints (generation 1 << CAP_LOG2, slot 0).
    // Embedders discover it with a scratch grant rather than hard-coding; this pins the test's host
    // setup below.
    const SPEC: MemHookSpec = MemHookSpec {
        type_id: 13,
        handle: 256,
    };

    /// `func() -> i64 { store i64 7 at 16+8; load i64 at 16+8 }` — one store, one load, both with
    /// a non-zero immediate offset, plus a bulk fill; exercises renumbering across insertions.
    fn sample() -> Module {
        let insts = vec![
            Inst::ConstI64(16), // v0
            Inst::ConstI64(7),  // v1
            Inst::Store {
                op: StoreOp::I64,
                addr: 0,
                value: 1,
                offset: 8,
                align: 3,
            },
            Inst::ConstI32(0),  // v2
            Inst::ConstI64(32), // v3
            Inst::MemFill {
                dst: 0,
                val: 2,
                len: 3,
            },
            Inst::Load {
                op: LoadOp::I64,
                addr: 0,
                offset: 8,
                align: 3,
            }, // v4
        ];
        Module {
            interfaces: vec![],
            funcs: vec![Func {
                params: vec![],
                results: vec![ValType::I64],
                blocks: vec![Block {
                    params: vec![],
                    insts,
                    term: Terminator::Return(vec![4]),
                }],
            }],
            memory: Some(Memory { size_log2: 16 }),
            data: vec![],
            imports: vec![],
            exports: vec![],
            impl_exports: vec![],
            debug_info: None,
        }
    }

    #[test]
    fn inserts_one_capcall_per_memory_op_and_reverifies() {
        let m = sample();
        svm_verify::verify_module(&m).expect("sample verifies");
        let (out, stats) = instrument_mem_hooks(&m, SPEC);
        assert_eq!(stats.hooked_ops, 3);
        let caps = out.funcs[0].blocks[0]
            .insts
            .iter()
            .filter(|i| matches!(i, Inst::CapCall { .. }))
            .count();
        assert_eq!(caps, 3);
        svm_verify::verify_module(&out).expect("instrumented module re-verifies");
        // The original module is untouched (pure transform).
        assert_eq!(m, sample());
    }

    #[test]
    fn instrumented_module_computes_the_same_result_and_reports_the_accesses() {
        let m = sample();
        let (out, _) = instrument_mem_hooks(&m, SPEC);
        svm_verify::verify_module(&out).expect("re-verify");

        // Reference run of the pristine module.
        let mut h0 = svm_interp::Host::new();
        let mut fuel0 = 1_000_000u64;
        let r0 = svm_interp::run_with_host(&m, 0, &[], &mut fuel0, &mut h0);

        // Instrumented run: grant the recording hook FIRST so its handle matches `SPEC.handle`.
        let events = std::sync::Arc::new(std::sync::Mutex::new(Vec::<(u32, Vec<i64>)>::new()));
        let sink = events.clone();
        let mut h = svm_interp::Host::new();
        let handle = h.grant_host_fn(Box::new(move |op, args, _mem| {
            sink.lock().unwrap().push((op, args.to_vec()));
            Ok(vec![])
        }));
        assert_eq!(
            handle, SPEC.handle,
            "first grant on a fresh Host mints the baked handle"
        );
        let mut fuel = 1_000_000u64;
        let r = svm_interp::run_with_host(&out, 0, &[], &mut fuel, &mut h);

        assert_eq!(
            r0, r,
            "instrumentation must not perturb the guest-visible result"
        );
        let got = events.lock().unwrap().clone();
        assert_eq!(
            got,
            vec![
                (mem_hook_op::STORE, vec![24, 8]),
                (mem_hook_op::FILL, vec![16, 32]),
                (mem_hook_op::LOAD, vec![24, 8]),
            ]
        );
    }

    /// Every instruction [`Inst::effects`] classifies as touching guest memory is either hooked by
    /// [`mem_event_of`] or on the documented exclusion list — so a future guest data-memory op
    /// can't silently go untraced.
    #[test]
    fn classification_is_exhaustive() {
        let excluded = |i: &Inst| {
            matches!(
                i,
                // Runtime-internal / control accesses, not guest data ops (module docs).
                Inst::MemoryWait { .. }
                    | Inst::MemoryNotify { .. }
                    | Inst::SetJmp { .. }
                    | Inst::LongJmp { .. }
                    | Inst::GcRoots { .. }
                    | Inst::CapSelfResolve { .. }
                    | Inst::CapSelfLabel { .. }
                    // Calls clobber conservatively; the callee's own ops carry the hooks.
                    | Inst::Call { .. }
                    | Inst::CallIndirect { .. }
                    | Inst::CallImport { .. }
                    | Inst::CapCall { .. }
                    // Stack switches clobber conservatively, like calls.
                    | Inst::ContResume { .. }
                    | Inst::Suspend { .. }
                    | Inst::ThreadSpawn { .. }
                    | Inst::ThreadJoin { .. }
            )
        };
        // A representative of every memory-effect instruction; `effects()` is the oracle.
        for inst in representative_insts() {
            let fx = inst.effects();
            if (fx.reads_mem || fx.writes_mem) && !excluded(&inst) {
                assert!(
                    mem_event_of(&inst).is_some(),
                    "memory-effect inst not hooked and not excluded: {inst:?}"
                );
            }
        }
    }

    /// One value of each `Inst` variant that reads or writes guest memory per [`Inst::effects`]
    /// (plus the excluded ones), so `classification_is_exhaustive` exercises the full set.
    fn representative_insts() -> Vec<Inst> {
        use svm_ir::{AtomicRmwOp, Ordering};
        vec![
            Inst::Load {
                op: LoadOp::I32,
                addr: 0,
                offset: 0,
                align: 2,
            },
            Inst::Store {
                op: StoreOp::I32,
                addr: 0,
                value: 1,
                offset: 0,
                align: 2,
            },
            Inst::V128Load {
                addr: 0,
                offset: 0,
                align: 4,
            },
            Inst::V128Store {
                addr: 0,
                value: 1,
                offset: 0,
                align: 4,
            },
            Inst::MemCopy {
                dst: 0,
                src: 1,
                len: 2,
            },
            Inst::MemMove {
                dst: 0,
                src: 1,
                len: 2,
            },
            Inst::MemFill {
                dst: 0,
                val: 1,
                len: 2,
            },
            Inst::AtomicLoad {
                ty: IntTy::I32,
                addr: 0,
                offset: 0,
                order: Ordering::SeqCst,
            },
            Inst::AtomicStore {
                ty: IntTy::I32,
                addr: 0,
                value: 1,
                offset: 0,
                order: Ordering::SeqCst,
            },
            Inst::AtomicRmw {
                ty: IntTy::I32,
                op: AtomicRmwOp::Add,
                addr: 0,
                value: 1,
                offset: 0,
                order: Ordering::SeqCst,
            },
            Inst::AtomicCmpxchg {
                ty: IntTy::I32,
                addr: 0,
                expected: 1,
                replacement: 2,
                offset: 0,
                order: Ordering::SeqCst,
            },
            Inst::MemoryWait {
                ty: IntTy::I32,
                addr: 0,
                expected: 1,
                timeout: 2,
            },
            Inst::MemoryNotify { addr: 0, count: 1 },
            Inst::SetJmp { buf: 0 },
            Inst::LongJmp { buf: 0, val: 1 },
            Inst::GcRoots {
                heap_lo: 0,
                heap_hi: 1,
                mask: 2,
                buf: 3,
                cap: 4,
            },
            Inst::CapSelfResolve {
                name_ptr: 0,
                name_len: 1,
            },
            Inst::CapSelfLabel {
                handle: 0,
                buf_ptr: 1,
                buf_cap: 2,
            },
        ]
    }
}
