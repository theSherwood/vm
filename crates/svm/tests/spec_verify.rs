//! SPEC.md suite 2 — verifier conformance. Two independent implementations of the
//! §3b/§3c validity rules — `svm-verify` (production, the TCB contract) and
//! `svm_spec::verify` (the reference, written from the prose) — must agree:
//!
//! 1. **Accept:** every spec-row module verifies under both.
//! 2. **Reject, directed:** for each typing rule, a minimal module violating exactly
//!    that rule is rejected by both — and by the production verifier with the
//!    *specific* `VerifyError` variant, where the rule→variant mapping is one-to-one.
//! 3. **Agreement sweep:** `irgen`'s verifier-valid modules (all accepted by both),
//!    plus deterministic structural mutations of them (accept/reject must agree —
//!    a mutation is allowed to leave the module valid).
//!
//! This closes the verifier's accept-direction gap: an accept-direction bug in
//! `svm-verify` now needs the same bug, independently written, in the reference.

#[path = "support/irgen.rs"]
mod irgen;

use irgen::Gen;
use svm::verify::{verify_module, VerifyError};
use svm_ir::*;
use svm_spec::{all_rows, mem_rows, module_for, module_for_mem, vectors_for, Shape};

/// Assert both verifiers reject `m`, the production one matching `pat`.
#[track_caller]
fn reject(m: &Module, what: &str, pat: impl Fn(&VerifyError) -> bool) {
    match verify_module(m) {
        Err(e) if pat(&e) => {}
        r => panic!("[{what}] production verifier returned {r:?}"),
    }
    assert!(
        svm_spec::verify::verify(m).is_err(),
        "[{what}] reference verifier accepted an invalid module"
    );
}

/// Assert both verifiers accept `m`.
#[track_caller]
fn accept(m: &Module, what: &str) {
    verify_module(m).unwrap_or_else(|e| panic!("[{what}] production verifier rejected: {e:?}"));
    svm_spec::verify::verify(m)
        .unwrap_or_else(|e| panic!("[{what}] reference verifier rejected: {e}"));
}

/// `func (params) -> (results) {{ block0(params): insts; term }}`.
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

fn module(funcs: Vec<Func>) -> Module {
    Module {
        funcs,
        ..Default::default()
    }
}

/// Part 1 — accept: every spec-row module passes both verifiers.
#[test]
fn spec_row_modules_verify_under_both() {
    for row in all_rows() {
        let m = match row.shape {
            Shape::Operands => module_for(&row, &[]),
            Shape::Immediate => {
                let sample = vectors_for(&row).into_iter().next().unwrap();
                module_for(&row, &sample)
            }
        };
        accept(&m, &row.id);
    }
    for row in mem_rows() {
        let m = module_for_mem(&row, 0);
        accept(&m, &row.id);

        // Per-row directed rejects: every memory op needs a declared window, an `i64`
        // address, and defined operands.
        let mut no_mem = m.clone();
        no_mem.memory = None;
        reject(&no_mem, &format!("{} without memory", row.id), |e| {
            matches!(e, VerifyError::MemoryNotDeclared { .. })
        });
        let mut wrong_addr = m.clone();
        wrong_addr.funcs[0].params[0] = ValType::I32;
        wrong_addr.funcs[0].blocks[0].params[0] = ValType::I32;
        reject(&wrong_addr, &format!("{} i32 address", row.id), |e| {
            matches!(e, VerifyError::TypeMismatch { .. })
        });
        let mut oob = m.clone();
        let idx: Vec<ValIdx> = row.operands.iter().map(|_| 100).collect();
        oob.funcs[0].blocks[0].insts[0] = (row.build)(&idx, 0);
        reject(&oob, &format!("{} undefined operand", row.id), |e| {
            matches!(e, VerifyError::ValueOutOfRange { .. })
        });
    }
}

/// Part 2a — generic per-row rejects: a wrongly-typed first operand and an undefined
/// operand index, derived from every table row (SPEC.md suite 2 "systematic
/// mutations").
#[test]
fn spec_row_mutations_reject_under_both() {
    for row in all_rows() {
        if row.shape != Shape::Operands || row.operands.is_empty() {
            continue;
        }
        // Wrong first-operand type: retype param 0 (in both the signature and the
        // entry block, keeping the entry rule satisfied) so exactly the op's own
        // operand check fails. (For `select`, param 0 is the `cond`, so the
        // polymorphic arms stay consistent.)
        let mut m = module_for(&row, &[]);
        let flip = |t: ValType| {
            if t == ValType::I32 {
                ValType::I64
            } else {
                ValType::I32
            }
        };
        m.funcs[0].params[0] = flip(m.funcs[0].params[0]);
        m.funcs[0].blocks[0].params[0] = flip(m.funcs[0].blocks[0].params[0]);
        reject(&m, &format!("{} wrong operand type", row.id), |e| {
            matches!(e, VerifyError::TypeMismatch { .. })
        });

        // Undefined operand: indices far past the defined count (§3b rule 3
        // defined-earlier, no forward refs).
        let mut m = module_for(&row, &[]);
        let idx: Vec<ValIdx> = row.operands.iter().map(|_| 100).collect();
        m.funcs[0].blocks[0].insts[0] = (row.build)(&idx, &[]);
        reject(&m, &format!("{} undefined operand", row.id), |e| {
            matches!(e, VerifyError::ValueOutOfRange { .. })
        });
    }
}

/// Part 2b — directed structural/rule rejects, one minimal module per rule, pinned to
/// the production verifier's error variant.
#[test]
fn directed_rule_rejects() {
    use Terminator as T;
    use ValType as V;

    // Entry rule: block0 params must equal the function's params (§3b rule 2).
    let mut f = func(vec![V::I32], vec![V::I32], vec![], T::Return(vec![0]));
    f.blocks[0].params.clear();
    reject(&module(vec![f]), "entry params mismatch", |e| {
        matches!(e, VerifyError::EntryParamsMismatch { .. })
    });

    // A function with no blocks cannot return.
    let f = Func {
        params: vec![],
        results: vec![],
        blocks: vec![],
    };
    reject(&module(vec![f]), "no blocks", |e| {
        matches!(e, VerifyError::EntryParamsMismatch { .. })
    });

    // Branch target out of range.
    let f = func(
        vec![],
        vec![],
        vec![],
        T::Br {
            target: 9,
            args: vec![],
        },
    );
    reject(&module(vec![f]), "branch target out of range", |e| {
        matches!(e, VerifyError::BlockOutOfRange { .. })
    });

    // Branch arg count must equal the target block's params (§3b rule 4).
    let mut f = func(
        vec![V::I32],
        vec![V::I32],
        vec![],
        T::Br {
            target: 1,
            args: vec![],
        },
    );
    f.blocks.push(Block {
        params: vec![V::I32],
        insts: vec![],
        term: T::Return(vec![0]),
    });
    reject(&module(vec![f]), "branch arg count", |e| {
        matches!(e, VerifyError::ArgCountMismatch { .. })
    });

    // Branch arg type must match the target param exactly.
    let mut f = func(
        vec![V::I64],
        vec![V::I32],
        vec![],
        T::Br {
            target: 1,
            args: vec![0],
        },
    );
    f.blocks.push(Block {
        params: vec![V::I32],
        insts: vec![],
        term: T::Return(vec![0]),
    });
    reject(&module(vec![f]), "branch arg type", |e| {
        matches!(e, VerifyError::TypeMismatch { .. })
    });

    // Return arity and type match the signature exactly.
    let f = func(vec![V::I32], vec![V::I32], vec![], T::Return(vec![]));
    reject(&module(vec![f]), "return arity", |e| {
        matches!(e, VerifyError::ResultCountMismatch { .. })
    });
    let f = func(vec![V::I64], vec![V::I32], vec![], T::Return(vec![0]));
    reject(&module(vec![f]), "return type", |e| {
        matches!(e, VerifyError::TypeMismatch { .. })
    });

    // Calls: callee index in range; arg count/types match its signature (§3b rule 5).
    let f = func(
        vec![],
        vec![],
        vec![Inst::Call {
            func: 7,
            args: vec![],
        }],
        T::Return(vec![]),
    );
    reject(&module(vec![f]), "call out of range", |e| {
        matches!(e, VerifyError::CallFuncOutOfRange { .. })
    });
    let callee = func(vec![V::I32], vec![], vec![], T::Return(vec![]));
    let caller = func(
        vec![],
        vec![],
        vec![Inst::Call {
            func: 1,
            args: vec![],
        }],
        T::Return(vec![]),
    );
    reject(
        &module(vec![caller, callee.clone()]),
        "call arg count",
        |e| matches!(e, VerifyError::CallArgCountMismatch { .. }),
    );
    let caller = func(
        vec![V::I64],
        vec![],
        vec![Inst::Call {
            func: 1,
            args: vec![0],
        }],
        T::Return(vec![]),
    );
    reject(&module(vec![caller, callee]), "call arg type", |e| {
        matches!(e, VerifyError::TypeMismatch { .. })
    });

    // Tail call: the callee's results must equal this function's results.
    let callee = func(
        vec![],
        vec![V::I64],
        vec![Inst::ConstI64(0)],
        T::Return(vec![0]),
    );
    let caller = func(
        vec![],
        vec![V::I32],
        vec![],
        T::ReturnCall {
            func: 1,
            args: vec![],
        },
    );
    reject(&module(vec![caller, callee]), "tail-call results", |e| {
        matches!(e, VerifyError::ResultCountMismatch { .. })
    });

    // Memory ops need a declared window.
    let f = func(
        vec![V::I64],
        vec![V::I32],
        vec![Inst::Load {
            op: LoadOp::I32,
            addr: 0,
            offset: 0,
            align: 0,
        }],
        T::Return(vec![1]),
    );
    reject(&module(vec![f]), "load without memory", |e| {
        matches!(e, VerifyError::MemoryNotDeclared { .. })
    });

    // Window size must be representable.
    let mut m = module(vec![func(vec![], vec![], vec![], T::Return(vec![]))]);
    m.memory = Some(Memory { size_log2: 64 });
    reject(&m, "memory too large", |e| {
        matches!(e, VerifyError::MemorySizeTooLarge { .. })
    });

    // Data segments: need memory, and must fit the window (incl. offset overflow).
    let mut m = module(vec![func(vec![], vec![], vec![], T::Return(vec![]))]);
    m.data.push(Data {
        offset: 0,
        readonly: false,
        bytes: vec![1],
    });
    reject(&m, "data without memory", |e| {
        matches!(e, VerifyError::DataWithoutMemory { .. })
    });
    let mut m = module(vec![func(vec![], vec![], vec![], T::Return(vec![]))]);
    m.memory = Some(Memory { size_log2: 12 });
    m.data.push(Data {
        offset: u64::MAX, // offset+len overflows — must fail closed, not wrap
        readonly: false,
        bytes: vec![1],
    });
    reject(&m, "data offset overflow", |e| {
        matches!(e, VerifyError::DataOutOfWindow { .. })
    });

    // §12 thread entry signature is fixed: (i64, i64) -> i64.
    let entry = func(vec![], vec![], vec![], T::Return(vec![]));
    let spawner = func(
        vec![V::I64],
        vec![V::I32],
        vec![Inst::ThreadSpawn {
            func: 1,
            sp: 0,
            arg: 0,
        }],
        T::Return(vec![1]),
    );
    reject(
        &module(vec![spawner, entry]),
        "thread entry signature",
        |e| matches!(e, VerifyError::ThreadEntrySignature { .. }),
    );

    // §12 atomic orderings: a store may not carry acquire semantics.
    let mut m = module(vec![func(
        vec![V::I64, V::I32],
        vec![],
        vec![Inst::AtomicStore {
            ty: IntTy::I32,
            addr: 0,
            value: 1,
            offset: 0,
            order: Ordering::Acquire,
        }],
        T::Return(vec![]),
    )]);
    m.memory = Some(Memory { size_log2: 12 });
    reject(&m, "atomic store acquire", |e| {
        matches!(e, VerifyError::BadAtomicOrdering { .. })
    });

    // §17 SIMD: lane indices bounded by the shape; op families constrain shapes.
    let f = func(
        vec![V::V128],
        vec![V::I32],
        vec![Inst::ExtractLane {
            shape: VShape::I32x4,
            lane: 9,
            signed: false,
            a: 0,
        }],
        T::Return(vec![1]),
    );
    reject(&module(vec![f]), "simd lane out of range", |e| {
        matches!(e, VerifyError::BadSimdLane { .. })
    });
    let f = func(
        vec![V::V128],
        vec![V::V128],
        vec![Inst::VIntBin {
            shape: VShape::F32x4,
            op: VIntBinOp::Add,
            a: 0,
            b: 0,
        }],
        T::Return(vec![1]),
    );
    reject(&module(vec![f]), "simd shape mismatch", |e| {
        matches!(e, VerifyError::BadSimdShape { .. })
    });

    // §7: an unresolved named import must be rejected fail-closed.
    let f = func(
        vec![],
        vec![],
        vec![Inst::CallImport {
            import: 0,
            sig: FuncType {
                params: vec![],
                results: vec![],
            },
            handle: 0,
            args: vec![],
        }],
        T::Return(vec![]),
    );
    reject(&module(vec![f]), "unresolved import", |e| {
        matches!(e, VerifyError::UnresolvedImport { .. })
    });

    // §7 / IMPORTS.md phase 1 — the manifest-bearing legs. A `call.import` whose index names a
    // declared import with a matching sig is VALID (executable, no rewrite); a sig disagreement
    // and a duplicate manifest name are each rejected, by both verifiers.
    let import_call = |sig: FuncType| {
        func(
            vec![],
            vec![],
            vec![
                Inst::ConstI32(0), // vestigial handle operand (IMPORTS.md §2.5)
                Inst::CallImport {
                    import: 0,
                    sig,
                    handle: 0,
                    args: vec![],
                },
            ],
            T::Return(vec![]),
        )
    };
    let unit_sig = FuncType {
        params: vec![],
        results: vec![],
    };
    let mut m = module(vec![import_call(unit_sig.clone())]);
    m.imports = vec![svm_ir::Import {
        name: "ping".into(),
        sig: unit_sig.clone(),
        mode: svm_ir::ImportMode::Required,
    }];
    accept(&m, "manifest-bearing call.import");
    // Same module, call-site sig disagrees with the manifest's declaration.
    let mut m = module(vec![import_call(FuncType {
        params: vec![],
        results: vec![V::I32],
    })]);
    m.imports = vec![svm_ir::Import {
        name: "ping".into(),
        sig: unit_sig.clone(),
        mode: svm_ir::ImportMode::Required,
    }];
    reject(&m, "import sig mismatch", |e| {
        matches!(e, VerifyError::ImportSigMismatch { .. })
    });
    // Two manifest entries sharing a name.
    let mut m = module(vec![import_call(unit_sig.clone())]);
    m.imports = vec![
        svm_ir::Import {
            name: "ping".into(),
            sig: unit_sig.clone(),
            mode: svm_ir::ImportMode::Required,
        },
        svm_ir::Import {
            name: "ping".into(),
            sig: unit_sig,
            mode: svm_ir::ImportMode::Required,
        },
    ];
    reject(&m, "duplicate import name", |e| {
        matches!(e, VerifyError::DuplicateImport { .. })
    });

    // Phase-2 `import.attach` (IMPORTS.md): valid against a rebindable declaration; rejected
    // against a required one and past the manifest — by both verifiers.
    let attach_fn = func(
        vec![],
        vec![],
        vec![
            Inst::ConstI32(0),
            Inst::ImportAttach {
                import: 0,
                handle: 0,
            },
        ],
        T::Return(vec![]),
    );
    let stream_sig = FuncType {
        params: vec![V::I64, V::I64],
        results: vec![V::I64],
    };
    let mut m = module(vec![attach_fn.clone()]);
    m.imports = vec![svm_ir::Import {
        name: "out".into(),
        sig: stream_sig.clone(),
        mode: svm_ir::ImportMode::Rebindable,
    }];
    accept(&m, "attach to a rebindable import");
    let mut m = module(vec![attach_fn.clone()]);
    m.imports = vec![svm_ir::Import {
        name: "out".into(),
        sig: stream_sig,
        mode: svm_ir::ImportMode::Required,
    }];
    reject(&m, "attach to a required import", |e| {
        matches!(e, VerifyError::AttachNotRebindable { .. })
    });
    let m = module(vec![attach_fn]);
    reject(&m, "attach past the manifest", |e| {
        matches!(e, VerifyError::UnresolvedImport { .. })
    });

    // GC.md: a constant gc.roots mask may only clear the top byte.
    let mut m = module(vec![func(
        vec![V::I64],
        vec![V::I64],
        vec![
            Inst::ConstI64(0), // a fold-down mask (clears low bits) — unsafe
            Inst::GcRoots {
                heap_lo: 0,
                heap_hi: 0,
                mask: 1,
                buf: 0,
                cap: 0,
            },
        ],
        T::Return(vec![2]),
    )]);
    m.memory = Some(Memory { size_log2: 12 });
    reject(&m, "gc.roots unsafe mask", |e| {
        matches!(e, VerifyError::GcRootsMaskUnsafe { .. })
    });

    // Exports: real targets, unique names.
    let mut m = module(vec![func(vec![], vec![], vec![], T::Return(vec![]))]);
    m.exports.push(Export {
        name: "main".into(),
        func: 5,
    });
    reject(&m, "export out of range", |e| {
        matches!(e, VerifyError::ExportFuncOutOfRange { .. })
    });
    let mut m = module(vec![func(vec![], vec![], vec![], T::Return(vec![]))]);
    m.exports.push(Export {
        name: "main".into(),
        func: 0,
    });
    m.exports.push(Export {
        name: "main".into(),
        func: 0,
    });
    reject(&m, "duplicate export", |e| {
        matches!(e, VerifyError::DuplicateExport { .. })
    });

    // Select is polymorphic but exact: `b` must match `a`'s type.
    let f = func(
        vec![V::I32, V::I32, V::I64],
        vec![V::I32],
        vec![Inst::Select {
            cond: 0,
            a: 1,
            b: 2,
        }],
        T::Return(vec![3]),
    );
    reject(&module(vec![f]), "select type mismatch", |e| {
        matches!(e, VerifyError::TypeMismatch { .. })
    });
}

/// Deterministic structural mutations for the agreement sweep. A mutation may leave
/// the module valid — the assertion is *agreement*, not rejection.
fn mutate(m: &mut Module, kind: u64) {
    match kind % 6 {
        // Retarget the first unconditional branch far out of range.
        0 => {
            for f in &mut m.funcs {
                for b in &mut f.blocks {
                    if let Terminator::Br { target, .. } = &mut b.term {
                        *target = 999;
                        return;
                    }
                }
            }
        }
        // Empty the first return's argument list.
        1 => {
            for f in &mut m.funcs {
                for b in &mut f.blocks {
                    if let Terminator::Return(vals) = &mut b.term {
                        if !vals.is_empty() {
                            vals.clear();
                            return;
                        }
                    }
                }
            }
        }
        // Rotate the first block parameter's type (leaves the func signature alone,
        // so this usually trips the entry rule or an operand type).
        2 => {
            for f in &mut m.funcs {
                for b in &mut f.blocks {
                    if let Some(p) = b.params.first_mut() {
                        *p = match *p {
                            ValType::I32 => ValType::I64,
                            ValType::I64 => ValType::F32,
                            ValType::F32 => ValType::F64,
                            ValType::F64 => ValType::V128,
                            ValType::V128 => ValType::Ref,
                            ValType::Ref => ValType::I32,
                        };
                        return;
                    }
                }
            }
        }
        // Drop the memory declaration (any load/store must then be rejected).
        3 => m.memory = None,
        // Grow the entry function's result arity.
        4 => {
            if let Some(f) = m.funcs.first_mut() {
                f.results.push(ValType::I32);
            }
        }
        // Prepend a const to a nonempty block: shifts the value numbering under every
        // later operand index.
        5 => {
            for f in &mut m.funcs {
                for b in &mut f.blocks {
                    if !b.insts.is_empty() {
                        b.insts.insert(0, Inst::ConstI32(0));
                        return;
                    }
                }
            }
        }
        _ => unreachable!(),
    }
}

/// Part 3 — the agreement sweep: `irgen` modules are accepted by both verifiers, and
/// structural mutations of them keep the two in accept/reject agreement.
#[test]
fn verifier_agreement_on_generated_modules() {
    for seed in 0..300u64 {
        let mut g = Gen::from_seed(seed);
        let m = irgen::gen_module(&mut g);
        accept(&m, &format!("irgen seed {seed}"));
        for kind in 0..6u64 {
            let mut mm = m.clone();
            mutate(&mut mm, kind);
            let prod = verify_module(&mm).is_ok();
            let refv = svm_spec::verify::verify(&mm).is_ok();
            assert_eq!(
                prod,
                refv,
                "verifier disagreement (seed {seed}, mutation {kind}): production={prod}, \
                 reference={refv}\n{}",
                svm::text::print_module(&mm)
            );
        }
    }
}
