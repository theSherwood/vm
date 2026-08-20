//! The **op catalog**: the full mnemonic fan-out (`i32.add`, `f64.sqrt`, `i8x16.add`, …), each
//! paired with a minimal verifiable [`Module`] that exercises it. One enumeration serves both the
//! rendered matrix ([`super::render_markdown`]) and the conformance pin (`tests/conformance.rs`), so
//! the row set and the tested set can never drift apart.
//!
//! Mnemonics are formed from `svm-ir`'s own sub-op vocabulary (`.name()`/`.prefix()`/`.sig()`), the
//! same source the text printer uses — so a renamed op renames its catalog row automatically.
//!
//! **Granularity note.** Rows fan out over the *semantic* immediates (type prefix, sub-op, lane
//! shape). The memory-access `align` hint and the load/store/rmw/cmpxchg atomic `Ordering` (both
//! write-only — every backend executed seq-cst regardless; DESIGN §12) left the wire at the wire
//! rev, so there is nothing left to collapse for them. `AtomicFence` keeps its `Ordering`, but it
//! is not fanned out (all orderings share support/result).

use svm_ir::*;

/// What a catalogued row classifies — an instruction or a block terminator.
#[derive(Clone, Debug)]
pub enum Focus {
    Inst(Inst),
    Term(Terminator),
}

/// One catalogued op: its mnemonic, family, the [`Focus`] the matrix classifies, and a minimal
/// verifiable module exercising it (entry = function [`Op::entry`]).
#[derive(Clone, Debug)]
pub struct Op {
    pub mnemonic: String,
    pub family: &'static str,
    pub focus: Focus,
    pub module: Module,
    pub entry: FuncIdx,
    /// `Some(reason)` ⇒ the conformance test does not auto-synthesize a backend check for this op —
    /// its support is target-conditional (fibers/threads/`setjmp` need the x86-64-unix substrate) or
    /// needs host wiring the differential harnesses already exercise (cap/import/export ops). The row
    /// still appears in the matrix with its manifest classification; only the end-to-end pin is
    /// skipped. Kept explicit and visible so the coverage gap is auditable, never silent.
    pub conf_skip: Option<&'static str>,
}

impl Op {
    /// The four-backend classification for this row (from the manifest).
    pub fn cells(&self) -> [super::Cell; 4] {
        match &self.focus {
            Focus::Inst(i) => super::parity(i),
            Focus::Term(t) => super::parity_term(t),
        }
    }
}

// ---- minimal-module builders --------------------------------------------------------------------

const MEM: Memory = Memory { size_log2: 16 };

fn func(params: Vec<ValType>, results: Vec<ValType>, insts: Vec<Inst>, term: Terminator) -> Func {
    Func {
        params: params.clone(),
        results,
        blocks: vec![Block {
            params,
            insts,
            term,
        }],
    }
}

fn module1(f: Func, memory: bool) -> Module {
    Module {
        funcs: vec![f],
        memory: memory.then_some(MEM),
        ..Module::default()
    }
}

/// A self-contained unit: block params feed `inst`'s operands (which reference `v0..`), and the op's
/// results (if any) are returned. `memory` declares a window for memory/atomic ops.
fn unit(operands: &[ValType], inst: Inst, results: &[ValType], memory: bool) -> Module {
    let n = operands.len() as ValIdx;
    let ret: Vec<ValIdx> = (0..results.len() as ValIdx).map(|k| n + k).collect();
    module1(
        func(
            operands.to_vec(),
            results.to_vec(),
            vec![inst],
            Terminator::Return(ret),
        ),
        memory,
    )
}

// ---- mnemonic helpers ---------------------------------------------------------------------------

fn widen_suffix(op: VWidenOp) -> &'static str {
    match op {
        VWidenOp::LowS => "low_s",
        VWidenOp::LowU => "low_u",
        VWidenOp::HighS => "high_s",
        VWidenOp::HighU => "high_u",
    }
}

// ---- the catalog --------------------------------------------------------------------------------

/// Build the full catalog. Order is family-grouped, matching the rendered matrix's sections.
pub fn catalog() -> Vec<Op> {
    let mut ops = Vec::new();
    scalar_int(&mut ops);
    scalar_float(&mut ops);
    conversions(&mut ops);
    memory(&mut ops);
    atomics(&mut ops);
    simd(&mut ops);
    control_and_calls(&mut ops);
    caps_and_reflection(&mut ops);
    process(&mut ops);
    fibers_threads(&mut ops);
    terminators(&mut ops);
    ops
}

fn push(ops: &mut Vec<Op>, mnemonic: String, family: &'static str, inst: Inst, module: Module) {
    ops.push(Op {
        mnemonic,
        family,
        focus: Focus::Inst(inst),
        module,
        entry: 0,
        conf_skip: None,
    });
}

fn push_skip(
    ops: &mut Vec<Op>,
    mnemonic: String,
    family: &'static str,
    inst: Inst,
    module: Module,
    reason: &'static str,
) {
    ops.push(Op {
        mnemonic,
        family,
        focus: Focus::Inst(inst),
        module,
        entry: 0,
        conf_skip: Some(reason),
    });
}

fn scalar_int(ops: &mut Vec<Op>) {
    const FAM: &str = "scalar integer";
    for ty in [IntTy::I32, IntTy::I64] {
        let t = ty.val();
        push(
            ops,
            format!("{}.const", ty.prefix()),
            FAM,
            if ty == IntTy::I32 {
                Inst::ConstI32(0)
            } else {
                Inst::ConstI64(0)
            },
            unit(
                &[],
                if ty == IntTy::I32 {
                    Inst::ConstI32(0)
                } else {
                    Inst::ConstI64(0)
                },
                &[t],
                false,
            ),
        );
        for op in BinOp::ALL {
            let i = Inst::IntBin { ty, op, a: 0, b: 1 };
            push(
                ops,
                format!("{}.{}", ty.prefix(), op.name()),
                FAM,
                i.clone(),
                unit(&[t, t], i, &[t], false),
            );
        }
        for op in CmpOp::ALL {
            let i = Inst::IntCmp { ty, op, a: 0, b: 1 };
            push(
                ops,
                format!("{}.{}", ty.prefix(), op.name()),
                FAM,
                i.clone(),
                unit(&[t, t], i, &[ValType::I32], false),
            );
        }
        for op in IntUnOp::ALL {
            let i = Inst::IntUn { ty, op, a: 0 };
            push(
                ops,
                format!("{}.{}", ty.prefix(), op.name()),
                FAM,
                i.clone(),
                unit(&[t], i, &[t], false),
            );
        }
        let eqz = Inst::Eqz { ty, a: 0 };
        push(
            ops,
            format!("{}.eqz", ty.prefix()),
            FAM,
            eqz.clone(),
            unit(&[t], eqz, &[ValType::I32], false),
        );
    }
    // Select is type-generic; enumerate the representative i32 form (support is shape-agnostic).
    let sel = Inst::Select {
        cond: 0,
        a: 1,
        b: 2,
    };
    push(
        ops,
        "select".into(),
        FAM,
        sel.clone(),
        unit(
            &[ValType::I32, ValType::I32, ValType::I32],
            sel,
            &[ValType::I32],
            false,
        ),
    );
}

fn scalar_float(ops: &mut Vec<Op>) {
    const FAM: &str = "scalar float";
    for ty in [FloatTy::F32, FloatTy::F64] {
        let t = ty.val();
        push(
            ops,
            format!("{}.const", ty.prefix()),
            FAM,
            if ty == FloatTy::F32 {
                Inst::ConstF32(0)
            } else {
                Inst::ConstF64(0)
            },
            unit(
                &[],
                if ty == FloatTy::F32 {
                    Inst::ConstF32(0)
                } else {
                    Inst::ConstF64(0)
                },
                &[t],
                false,
            ),
        );
        for op in FBinOp::ALL {
            let i = Inst::FBin { ty, op, a: 0, b: 1 };
            push(
                ops,
                format!("{}.{}", ty.prefix(), op.name()),
                FAM,
                i.clone(),
                unit(&[t, t], i, &[t], false),
            );
        }
        for op in FUnOp::ALL {
            let i = Inst::FUn { ty, op, a: 0 };
            push(
                ops,
                format!("{}.{}", ty.prefix(), op.name()),
                FAM,
                i.clone(),
                unit(&[t], i, &[t], false),
            );
        }
        for op in FCmpOp::ALL {
            let i = Inst::FCmp { ty, op, a: 0, b: 1 };
            push(
                ops,
                format!("{}.{}", ty.prefix(), op.name()),
                FAM,
                i.clone(),
                unit(&[t, t], i, &[ValType::I32], false),
            );
        }
        let fma = Inst::Fma {
            ty,
            a: 0,
            b: 1,
            c: 2,
        };
        push(
            ops,
            format!("{}.fma", ty.prefix()),
            FAM,
            fma.clone(),
            unit(&[t, t, t], fma, &[t], false),
        );
    }
}

fn conversions(ops: &mut Vec<Op>) {
    const FAM: &str = "conversions";
    for op in [ConvOp::ExtendI32S, ConvOp::ExtendI32U, ConvOp::WrapI64] {
        let (name, src, dst) = op.sig();
        let i = Inst::Convert { op, a: 0 };
        push(
            ops,
            name.into(),
            FAM,
            i.clone(),
            unit(&[src], i, &[dst], false),
        );
    }
    for op in FToI::ALL {
        let (fty, ity, _) = op.parts();
        let sat = Inst::FToISat { op, a: 0 };
        push(
            ops,
            op.name().into(),
            FAM,
            sat.clone(),
            unit(&[fty.val()], sat, &[ity.val()], false),
        );
        let trap = Inst::FToITrap { op, a: 0 };
        push(
            ops,
            op.trap_name().into(),
            FAM,
            trap.clone(),
            unit(&[fty.val()], trap, &[ity.val()], false),
        );
    }
    for op in IToF::ALL {
        let (ity, fty, _) = op.parts();
        let i = Inst::IToFConv { op, a: 0 };
        push(
            ops,
            op.name().into(),
            FAM,
            i.clone(),
            unit(&[ity.val()], i, &[fty.val()], false),
        );
    }
    for op in CastOp::ALL {
        let (name, src, dst) = op.sig();
        let i = Inst::Cast { op, a: 0 };
        push(
            ops,
            name.into(),
            FAM,
            i.clone(),
            unit(&[src], i, &[dst], false),
        );
    }
}

fn memory(ops: &mut Vec<Op>) {
    const FAM: &str = "memory";
    for op in LoadOp::ALL {
        let (name, res, _, _) = op.info();
        let i = Inst::Load {
            op,
            addr: 0,
            offset: 0,
        };
        push(
            ops,
            name.into(),
            FAM,
            i.clone(),
            unit(&[ValType::I64], i, &[res], true),
        );
    }
    for op in StoreOp::ALL {
        let (name, val, _) = op.info();
        let i = Inst::Store {
            op,
            addr: 0,
            value: 1,
            offset: 0,
        };
        push(
            ops,
            name.into(),
            FAM,
            i.clone(),
            unit(&[ValType::I64, val], i, &[], true),
        );
    }
    let i64 = ValType::I64;
    for (name, i) in [
        (
            "mem.copy",
            Inst::MemCopy {
                dst: 0,
                src: 1,
                len: 2,
            },
        ),
        (
            "mem.move",
            Inst::MemMove {
                dst: 0,
                src: 1,
                len: 2,
            },
        ),
    ] {
        push(
            ops,
            name.into(),
            FAM,
            i.clone(),
            unit(&[i64, i64, i64], i, &[], true),
        );
    }
    // `mem.fill`'s fill byte (`val`) is an i32, not i64.
    let fill = Inst::MemFill {
        dst: 0,
        val: 1,
        len: 2,
    };
    push(
        ops,
        "mem.fill".into(),
        FAM,
        fill.clone(),
        unit(&[i64, ValType::I32, i64], fill, &[], true),
    );
}

fn atomics(ops: &mut Vec<Op>) {
    const FAM: &str = "atomics";
    for ty in [IntTy::I32, IntTy::I64] {
        let t = ty.val();
        let load = Inst::AtomicLoad {
            ty,
            addr: 0,
            offset: 0,
        };
        push(
            ops,
            format!("{}.atomic.load", ty.prefix()),
            FAM,
            load.clone(),
            unit(&[ValType::I64], load, &[t], true),
        );
        let store = Inst::AtomicStore {
            ty,
            addr: 0,
            value: 1,
            offset: 0,
        };
        push(
            ops,
            format!("{}.atomic.store", ty.prefix()),
            FAM,
            store.clone(),
            unit(&[ValType::I64, t], store, &[], true),
        );
        for op in AtomicRmwOp::ALL {
            let i = Inst::AtomicRmw {
                ty,
                op,
                addr: 0,
                value: 1,
                offset: 0,
            };
            push(
                ops,
                format!("{}.atomic.rmw.{}", ty.prefix(), op.name()),
                FAM,
                i.clone(),
                unit(&[ValType::I64, t], i, &[t], true),
            );
        }
        let cx = Inst::AtomicCmpxchg {
            ty,
            addr: 0,
            expected: 1,
            replacement: 2,
            offset: 0,
        };
        push(
            ops,
            format!("{}.atomic.cmpxchg", ty.prefix()),
            FAM,
            cx.clone(),
            unit(&[ValType::I64, t, t], cx, &[t], true),
        );
    }
    let fence = Inst::AtomicFence {
        order: Ordering::SeqCst,
    };
    push(
        ops,
        "atomic.fence".into(),
        FAM,
        fence.clone(),
        unit(&[], fence, &[], false),
    );
    // Futex ops are concurrency (target-conditional on Cranelift, leaf-declined on wasm) — skip the pin.
    let wait = Inst::MemoryWait {
        ty: IntTy::I32,
        addr: 0,
        expected: 1,
        timeout: 2,
    };
    push_skip(
        ops,
        "i32.atomic.wait".into(),
        FAM,
        wait.clone(),
        unit(
            &[ValType::I64, ValType::I32, ValType::I64],
            wait,
            &[ValType::I32],
            true,
        ),
        "concurrency: target-conditional / leaf-declined",
    );
    let notify = Inst::MemoryNotify { addr: 0, count: 1 };
    push_skip(
        ops,
        "atomic.notify".into(),
        FAM,
        notify.clone(),
        unit(&[ValType::I64, ValType::I32], notify, &[ValType::I32], true),
        "concurrency: target-conditional / leaf-declined",
    );
}

fn simd(ops: &mut Vec<Op>) {
    const FAM: &str = "simd v128";
    let v = ValType::V128;
    let ints = [VShape::I8x16, VShape::I16x8, VShape::I32x4, VShape::I64x2];
    let floats = [VShape::F32x4, VShape::F64x2];

    push(
        ops,
        "v128.const".into(),
        FAM,
        Inst::ConstV128([0; 16]),
        unit(&[], Inst::ConstV128([0; 16]), &[v], false),
    );
    let ld = Inst::V128Load { addr: 0, offset: 0 };
    push(
        ops,
        "v128.load".into(),
        FAM,
        ld.clone(),
        unit(&[ValType::I64], ld, &[v], true),
    );
    let st = Inst::V128Store {
        addr: 0,
        value: 1,
        offset: 0,
    };
    push(
        ops,
        "v128.store".into(),
        FAM,
        st.clone(),
        unit(&[ValType::I64, v], st, &[], true),
    );

    for shape in VShape::ALL {
        let lane = shape.lane_val();
        let sp = Inst::Splat { shape, a: 0 };
        push(
            ops,
            format!("{}.splat", shape.name()),
            FAM,
            sp.clone(),
            unit(&[lane], sp, &[v], false),
        );
        let narrow = matches!(shape, VShape::I8x16 | VShape::I16x8);
        let ex = Inst::ExtractLane {
            shape,
            lane: 0,
            signed: narrow,
            a: 0,
        };
        push(
            ops,
            format!("{}.extract_lane", shape.name()),
            FAM,
            ex.clone(),
            unit(&[v], ex, &[lane], false),
        );
        let rl = Inst::ReplaceLane {
            shape,
            lane: 0,
            a: 0,
            b: 1,
        };
        push(
            ops,
            format!("{}.replace_lane", shape.name()),
            FAM,
            rl.clone(),
            unit(&[v, lane], rl, &[v], false),
        );
    }
    for shape in ints {
        for op in VIntBinOp::ALL {
            let i = Inst::VIntBin {
                shape,
                op,
                a: 0,
                b: 1,
            };
            push(
                ops,
                format!("{}.{}", shape.name(), op.name()),
                FAM,
                i.clone(),
                unit(&[v, v], i, &[v], false),
            );
        }
        for op in VICmpOp::ALL {
            let i = Inst::VIntCmp {
                shape,
                op,
                a: 0,
                b: 1,
            };
            push(
                ops,
                format!("{}.{}", shape.name(), op.name()),
                FAM,
                i.clone(),
                unit(&[v, v], i, &[v], false),
            );
        }
        let sh_amt = ValType::I32;
        for op in VShiftOp::ALL {
            let i = Inst::VShift {
                shape,
                op,
                a: 0,
                amt: 1,
            };
            push(
                ops,
                format!("{}.{}", shape.name(), op.name()),
                FAM,
                i.clone(),
                unit(&[v, sh_amt], i, &[v], false),
            );
        }
        for op in VIntUnOp::ALL {
            let i = Inst::VIntUn { shape, op, a: 0 };
            push(
                ops,
                format!("{}.{}", shape.name(), op.name()),
                FAM,
                i.clone(),
                unit(&[v], i, &[v], false),
            );
        }
        let at = Inst::VAllTrue { shape, a: 0 };
        push(
            ops,
            format!("{}.all_true", shape.name()),
            FAM,
            at.clone(),
            unit(&[v], at, &[ValType::I32], false),
        );
        let bm = Inst::VBitmask { shape, a: 0 };
        push(
            ops,
            format!("{}.bitmask", shape.name()),
            FAM,
            bm.clone(),
            unit(&[v], bm, &[ValType::I32], false),
        );
    }
    // Saturating add/sub: i8x16/i16x8 only.
    for shape in [VShape::I8x16, VShape::I16x8] {
        for op in VSatBinOp::ALL {
            let i = Inst::VSatBin {
                shape,
                op,
                a: 0,
                b: 1,
            };
            push(
                ops,
                format!("{}.{}", shape.name(), op.name()),
                FAM,
                i.clone(),
                unit(&[v, v], i, &[v], false),
            );
        }
        let avgr = Inst::VAvgr { shape, a: 0, b: 1 };
        push(
            ops,
            format!("{}.avgr_u", shape.name()),
            FAM,
            avgr.clone(),
            unit(&[v, v], avgr, &[v], false),
        );
    }
    // Widen: result shapes i16x8/i32x4/i64x2.
    for shape in [VShape::I16x8, VShape::I32x4, VShape::I64x2] {
        for op in VWidenOp::ALL {
            let i = Inst::VWiden { shape, op, a: 0 };
            push(
                ops,
                format!("{}.{}", shape.name(), op.name()),
                FAM,
                i.clone(),
                unit(&[v], i, &[v], false),
            );
            let em = Inst::VExtMul {
                shape,
                op,
                a: 0,
                b: 1,
            };
            push(
                ops,
                format!("{}.extmul_{}", shape.name(), widen_suffix(op)),
                FAM,
                em.clone(),
                unit(&[v, v], em, &[v], false),
            );
        }
    }
    // Narrow: result shapes i8x16/i16x8.
    for shape in [VShape::I8x16, VShape::I16x8] {
        for op in VNarrowOp::ALL {
            let i = Inst::VNarrow {
                shape,
                op,
                a: 0,
                b: 1,
            };
            push(
                ops,
                format!("{}.{}", shape.name(), op.name()),
                FAM,
                i.clone(),
                unit(&[v, v], i, &[v], false),
            );
        }
    }
    // Ext-add-pairwise: result shapes i16x8/i32x4.
    for shape in [VShape::I16x8, VShape::I32x4] {
        for signed in [true, false] {
            let i = Inst::VExtAddPairwise {
                shape,
                signed,
                a: 0,
            };
            push(
                ops,
                format!(
                    "{}.extadd_pairwise_{}",
                    shape.name(),
                    if signed { "s" } else { "u" }
                ),
                FAM,
                i.clone(),
                unit(&[v], i, &[v], false),
            );
        }
    }
    for op in VCvtOp::ALL {
        let i = Inst::VConvert { op, a: 0 };
        push(
            ops,
            op.name().into(),
            FAM,
            i.clone(),
            unit(&[v], i, &[v], false),
        );
    }
    for shape in floats {
        for op in VFloatBinOp::ALL {
            let i = Inst::VFloatBin {
                shape,
                op,
                a: 0,
                b: 1,
            };
            push(
                ops,
                format!("{}.{}", shape.name(), op.name()),
                FAM,
                i.clone(),
                unit(&[v, v], i, &[v], false),
            );
        }
        for op in VFloatUnOp::ALL {
            let i = Inst::VFloatUn { shape, op, a: 0 };
            push(
                ops,
                format!("{}.{}", shape.name(), op.name()),
                FAM,
                i.clone(),
                unit(&[v], i, &[v], false),
            );
        }
        for op in VFCmpOp::ALL {
            let i = Inst::VFloatCmp {
                shape,
                op,
                a: 0,
                b: 1,
            };
            push(
                ops,
                format!("{}.{}", shape.name(), op.name()),
                FAM,
                i.clone(),
                unit(&[v, v], i, &[v], false),
            );
        }
        for op in VPMinMaxOp::ALL {
            let i = Inst::VPMinMax {
                shape,
                op,
                a: 0,
                b: 1,
            };
            push(
                ops,
                format!("{}.{}", shape.name(), op.name()),
                FAM,
                i.clone(),
                unit(&[v, v], i, &[v], false),
            );
        }
        for neg in [false, true] {
            let i = Inst::VFma {
                shape,
                neg,
                a: 0,
                b: 1,
                c: 2,
            };
            push(
                ops,
                format!(
                    "{}.{}",
                    shape.name(),
                    if neg { "relaxed_nmadd" } else { "relaxed_madd" }
                ),
                FAM,
                i.clone(),
                unit(&[v, v, v], i, &[v], false),
            );
        }
    }
    for op in VBitBinOp::ALL {
        let i = Inst::VBitBin { op, a: 0, b: 1 };
        push(
            ops,
            format!("v128.{}", op.name()),
            FAM,
            i.clone(),
            unit(&[v, v], i, &[v], false),
        );
    }
    for (name, i) in [
        ("v128.not", Inst::VNot { a: 0 }),
        ("v128.any_true", Inst::VAnyTrue { a: 0 }),
        ("i8x16.popcnt", Inst::VPopcnt { a: 0 }),
    ] {
        let res = if name == "v128.any_true" {
            ValType::I32
        } else {
            v
        };
        push(
            ops,
            name.into(),
            FAM,
            i.clone(),
            unit(&[v], i, &[res], false),
        );
    }
    let bs = Inst::Bitselect {
        a: 0,
        b: 1,
        mask: 2,
    };
    push(
        ops,
        "v128.bitselect".into(),
        FAM,
        bs.clone(),
        unit(&[v, v, v], bs, &[v], false),
    );
    let dot = Inst::VDot { a: 0, b: 1 };
    push(
        ops,
        "i32x4.dot_i16x8_s".into(),
        FAM,
        dot.clone(),
        unit(&[v, v], dot, &[v], false),
    );
    let dot8 = Inst::VDotI8 { a: 0, b: 1 };
    push(
        ops,
        "i16x8.dot_i8x16_s".into(),
        FAM,
        dot8.clone(),
        unit(&[v, v], dot8, &[v], false),
    );
    let q15 = Inst::VQ15MulrSat { a: 0, b: 1 };
    push(
        ops,
        "i16x8.q15mulr_sat_s".into(),
        FAM,
        q15.clone(),
        unit(&[v, v], q15, &[v], false),
    );
    let shuf = Inst::Shuffle {
        lanes: [0; 16],
        a: 0,
        b: 1,
    };
    push(
        ops,
        "i8x16.shuffle".into(),
        FAM,
        shuf.clone(),
        unit(&[v, v], shuf, &[v], false),
    );
    let swz = Inst::Swizzle { a: 0, b: 1 };
    push(
        ops,
        "i8x16.swizzle".into(),
        FAM,
        swz.clone(),
        unit(&[v, v], swz, &[v], false),
    );
}

fn control_and_calls(ops: &mut Vec<Op>) {
    const FAM: &str = "calls & control";
    // ref.func: a funcref is a plain i32 (the function index).
    let rf = Inst::RefFunc { func: 0 };
    push(
        ops,
        "ref.func".into(),
        FAM,
        rf.clone(),
        unit(&[], rf, &[ValType::I32], false),
    );
    // Direct call: func0 calls func1 (a `() -> ()` nop).
    let nop = func(vec![], vec![], vec![], Terminator::Return(vec![]));
    let caller = func(
        vec![],
        vec![],
        vec![Inst::Call {
            func: 1,
            args: vec![],
        }],
        Terminator::Return(vec![]),
    );
    ops.push(Op {
        mnemonic: "call".into(),
        family: FAM,
        focus: Focus::Inst(Inst::Call {
            func: 1,
            args: vec![],
        }),
        module: Module {
            funcs: vec![caller, nop.clone()],
            ..Module::default()
        },
        entry: 0,
        conf_skip: None,
    });
    // call_indirect through the function table (idx is an i32; targets funcs[idx]).
    let ci = Inst::CallIndirect {
        ty: FuncType::default(),
        idx: 0,
        args: vec![],
    };
    let ci_caller = func(
        vec![],
        vec![],
        vec![Inst::ConstI32(1), ci.clone()],
        Terminator::Return(vec![]),
    );
    ops.push(Op {
        mnemonic: "call_indirect".into(),
        family: FAM,
        focus: Focus::Inst(ci),
        module: Module {
            funcs: vec![ci_caller, nop],
            ..Module::default()
        },
        entry: 0,
        conf_skip: None,
    });
}

fn caps_and_reflection(ops: &mut Vec<Op>) {
    const FAM: &str = "capabilities & reflection";
    let skip = "host cap/import wiring — covered by the differential harnesses, not auto-synth";
    // Cap-call family: build a verifiable module, but skip the end-to-end pin (needs host wiring).
    let cc = Inst::CapCall {
        type_id: 1,
        op: 0,
        sig: FuncType::default(),
        handle: 1,
        args: vec![],
    };
    push_skip(
        ops,
        "cap.call".into(),
        FAM,
        cc.clone(),
        module1(
            func(
                vec![],
                vec![],
                vec![Inst::ConstI32(0), cc],
                Terminator::Return(vec![]),
            ),
            false,
        ),
        skip,
    );
    // Reflection intrinsics. `cap.self.count`/`get`/`resolve`/`label`/`attest` are no longer distinct
    // ops — they are `cap.call CAP_SELF op N`, covered by the generic `cap.call` parity row.
    let tls_get = Inst::VcpuTlsGet;
    push(
        ops,
        "vcpu.tls.get".into(),
        FAM,
        tls_get.clone(),
        unit(&[], tls_get, &[ValType::I64], false),
    );
    let tls_set = Inst::VcpuTlsSet { val: 0 };
    push(
        ops,
        "vcpu.tls.set".into(),
        FAM,
        tls_set.clone(),
        unit(&[ValType::I64], tls_set, &[], false),
    );
    for (name, inst) in [
        ("export.handle", Inst::ExportHandle { export: 0 }),
        (
            "import.attach",
            Inst::ImportAttach {
                import: 0,
                handle: 0,
            },
        ),
        ("cap.self.type_id", Inst::CapSelfTypeId { ty: 0 }),
        ("cap.self.covers", Inst::CapSelfCovers { handle: 0, ty: 0 }),
        (
            "call.import",
            Inst::CallImport {
                import: 0,
                op: 0,
                sig: FuncType::default(),
                args: vec![],
            },
        ),
        (
            "call.import.dyn",
            Inst::CallImportDyn {
                ty: 0,
                op: 0,
                sig: FuncType::default(),
                handle: 0,
                args: vec![],
            },
        ),
        (
            "call.sym",
            Inst::CallSym {
                import: 0,
                sig: FuncType::default(),
                handle: 0,
                args: vec![],
            },
        ),
        ("durable.shadow_base", Inst::DurableShadowBase),
    ] {
        // Listed for the matrix; module is a placeholder (never compiled — conf_skip).
        push_skip(ops, name.into(), FAM, inst, Module::default(), skip);
    }
}

fn process(ops: &mut Vec<Op>) {
    const FAM: &str = "process, serve & fork";
    // Every op here is a `cap.call` sub-op that needs scheduler + host wiring (a live child, a serve
    // point, a parked caller) to run — so the end-to-end pin is skipped, exactly like the `cap.call`
    // and import/export rows. Their support is instead exercised by the fork/serve differential
    // harnesses (`clone_caller.rs`, `svc_serve_loop.rs`, `c_fork.rs`, `fork_manager.rs`); this
    // section makes their **classification** visible in the matrix, where a single `cap.call` row
    // used to hide the fork gap. `parity()` classifies each by `(type_id, op)`.
    let skip = "process/serve substrate — scheduler + host wiring; exercised by the fork/serve \
                differential harnesses, not auto-synth";
    let self_ty = svm_ir::CAP_SELF_TYPE_ID;
    let inst = crate::capcall::INSTANTIATOR;
    let capcall = |type_id: u32, op: u32| Inst::CapCall {
        type_id,
        op,
        sig: FuncType::default(),
        handle: 0,
        args: vec![],
    };
    for (name, type_id, op) in [
        // §14 executor children — the process-spawn nouns.
        ("instantiate", inst, 0u32),
        ("join", inst, 1),
        ("instantiate_module", inst, 5),
        ("child_offer", inst, 14),
        // §3.6 service points + the FORK.md self-namespace ops.
        ("svc.poll", self_ty, crate::capcall::SVC_POLL),
        ("svc.wait", self_ty, crate::capcall::SVC_WAIT),
        ("clone_caller", self_ty, crate::capcall::CLONE_CALLER),
        ("reap", self_ty, crate::capcall::REAP),
        ("fuel.remaining", self_ty, crate::capcall::FUEL_REMAINING),
        ("exec_module", self_ty, crate::capcall::EXEC),
    ] {
        push_skip(
            ops,
            name.into(),
            FAM,
            capcall(type_id, op),
            Module::default(),
            skip,
        );
    }
}

fn fibers_threads(ops: &mut Vec<Op>) {
    const FAM: &str = "fibers, threads & non-local control";
    let skip = "target-conditional (needs x86-64-unix stack-switch / setjmp substrate)";
    for (name, inst) in [
        ("cont.new", Inst::ContNew { func: 0, sp: 1 }),
        (
            "cont.resume",
            Inst::ContResume {
                k: 0,
                arg: 1,
                block: false,
            },
        ),
        // I48 advisory blocking resume (`block: true`): same parity shape as `cont.resume`; the
        // idling backends park while the OS-thread-parallel drivers take the FIBER_PARKED
        // downgrade (DESIGN §12).
        (
            "cont.resume.block",
            Inst::ContResume {
                k: 0,
                arg: 1,
                block: true,
            },
        ),
        ("suspend", Inst::Suspend { value: 0 }),
        (
            "thread.spawn",
            Inst::ThreadSpawn {
                func: 1,
                sp: 0,
                arg: 0,
            },
        ),
        ("thread.join", Inst::ThreadJoin { handle: 0 }),
        (
            "gc.roots",
            Inst::GcRoots {
                heap_lo: 0,
                heap_hi: 1,
                mask: 2,
                buf: 3,
                cap: 4,
            },
        ),
        ("setjmp", Inst::SetJmp { buf: 0 }),
        ("longjmp", Inst::LongJmp { buf: 0, val: 1 }),
    ] {
        push_skip(ops, name.into(), FAM, inst, Module::default(), skip);
    }
}

fn terminators(ops: &mut Vec<Op>) {
    const FAM: &str = "terminators";
    // return
    ops.push(Op {
        mnemonic: "return".into(),
        family: FAM,
        focus: Focus::Term(Terminator::Return(vec![0])),
        module: module1(
            func(
                vec![ValType::I32],
                vec![ValType::I32],
                vec![],
                Terminator::Return(vec![0]),
            ),
            false,
        ),
        entry: 0,
        conf_skip: None,
    });
    // unreachable
    ops.push(Op {
        mnemonic: "unreachable".into(),
        family: FAM,
        focus: Focus::Term(Terminator::Unreachable),
        module: module1(func(vec![], vec![], vec![], Terminator::Unreachable), false),
        entry: 0,
        conf_skip: None,
    });
    // br: block0 -> block1
    let br_fn = Func {
        params: vec![ValType::I32],
        results: vec![ValType::I32],
        blocks: vec![
            Block {
                params: vec![ValType::I32],
                insts: vec![],
                term: Terminator::Br {
                    target: 1,
                    args: vec![0],
                },
            },
            Block {
                params: vec![ValType::I32],
                insts: vec![],
                term: Terminator::Return(vec![0]),
            },
        ],
    };
    ops.push(Op {
        mnemonic: "br".into(),
        family: FAM,
        focus: Focus::Term(Terminator::Br {
            target: 1,
            args: vec![0],
        }),
        module: module1(br_fn, false),
        entry: 0,
        conf_skip: None,
    });
    // br_if
    let brif_fn = Func {
        params: vec![ValType::I32],
        results: vec![ValType::I32],
        blocks: vec![
            Block {
                params: vec![ValType::I32],
                insts: vec![],
                term: Terminator::BrIf {
                    cond: 0,
                    then_blk: 1,
                    then_args: vec![],
                    else_blk: 2,
                    else_args: vec![],
                },
            },
            Block {
                params: vec![],
                insts: vec![Inst::ConstI32(1)],
                term: Terminator::Return(vec![0]),
            },
            Block {
                params: vec![],
                insts: vec![Inst::ConstI32(0)],
                term: Terminator::Return(vec![0]),
            },
        ],
    };
    ops.push(Op {
        mnemonic: "br_if".into(),
        family: FAM,
        focus: Focus::Term(Terminator::BrIf {
            cond: 0,
            then_blk: 1,
            then_args: vec![],
            else_blk: 2,
            else_args: vec![],
        }),
        module: module1(brif_fn, false),
        entry: 0,
        conf_skip: None,
    });
    // br_table
    let brt_fn = Func {
        params: vec![ValType::I32],
        results: vec![ValType::I32],
        blocks: vec![
            Block {
                params: vec![ValType::I32],
                insts: vec![],
                term: Terminator::BrTable {
                    idx: 0,
                    targets: vec![(1, vec![])],
                    default: (2, vec![]),
                },
            },
            Block {
                params: vec![],
                insts: vec![Inst::ConstI32(1)],
                term: Terminator::Return(vec![0]),
            },
            Block {
                params: vec![],
                insts: vec![Inst::ConstI32(0)],
                term: Terminator::Return(vec![0]),
            },
        ],
    };
    ops.push(Op {
        mnemonic: "br_table".into(),
        family: FAM,
        focus: Focus::Term(Terminator::BrTable {
            idx: 0,
            targets: vec![(1, vec![])],
            default: (2, vec![]),
        }),
        module: module1(brt_fn, false),
        entry: 0,
        conf_skip: None,
    });
    // return_call: func0 tail-calls func1 (both `() -> ()`).
    let rc_target = func(vec![], vec![], vec![], Terminator::Return(vec![]));
    let rc_caller = Func {
        params: vec![],
        results: vec![],
        blocks: vec![Block {
            params: vec![],
            insts: vec![],
            term: Terminator::ReturnCall {
                func: 1,
                args: vec![],
            },
        }],
    };
    ops.push(Op {
        mnemonic: "return_call".into(),
        family: FAM,
        focus: Focus::Term(Terminator::ReturnCall {
            func: 1,
            args: vec![],
        }),
        module: Module {
            funcs: vec![rc_caller, rc_target.clone()],
            ..Module::default()
        },
        entry: 0,
        conf_skip: None,
    });
    // return_call_indirect
    let rci_caller = Func {
        params: vec![],
        results: vec![],
        blocks: vec![Block {
            params: vec![],
            insts: vec![Inst::ConstI32(1)],
            term: Terminator::ReturnCallIndirect {
                ty: FuncType::default(),
                idx: 0,
                args: vec![],
            },
        }],
    };
    ops.push(Op {
        mnemonic: "return_call_indirect".into(),
        family: FAM,
        focus: Focus::Term(Terminator::ReturnCallIndirect {
            ty: FuncType::default(),
            idx: 0,
            args: vec![],
        }),
        module: Module {
            funcs: vec![rci_caller, rc_target],
            ..Module::default()
        },
        entry: 0,
        conf_skip: None,
    });
}
