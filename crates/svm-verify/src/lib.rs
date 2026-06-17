//! The verifier — security-critical TCB (`DESIGN.md` §2a, invariants I2/I3/I4;
//! §3b "verifier validity rules").
//!
//! A single linear forward pass, O(module size), no dominance analysis and no
//! fixups (block parameters make cross-block dataflow explicit). For each block we
//! seed a local type vector with the block's declared parameter types, walk the
//! instructions (checking each operand is defined *earlier* and exactly the right
//! type, then appending the result type), and finally check the terminator's branch
//! arguments against each target block's declared parameter types.
//!
//! Result types are computed here from opcode + operand types (§3a "inferred result
//! types"); for the one polymorphic op (`select`) the result is the operand type.
//!
//! **Fail-closed:** any violation returns `Err`; the verifier never panics on any
//! input (that property is fuzzed — see the `svm` crate). A module that verifies is
//! the precondition for the escape-freedom contract (§2a); soundness of *this code*
//! is the separate hard problem (§18).
#![forbid(unsafe_code)]

use svm_ir::{BlockIdx, Func, Inst, Module, Terminator, VShape, ValIdx, ValType};

/// Why verification rejected a module. Carries enough location to debug, never
/// enough to be load-bearing for safety (the boolean accept/reject is the contract).
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum VerifyError {
    /// Entry block parameter types must equal the function signature's parameters.
    EntryParamsMismatch { func: u32 },
    /// A branch/return references a block index that does not exist.
    BlockOutOfRange { func: u32, block: u32, target: u32 },
    /// An operand references a value index not yet defined in this block.
    ValueOutOfRange {
        func: u32,
        block: u32,
        value: ValIdx,
        defined: u32,
    },
    /// An operand had the wrong type for its opcode.
    TypeMismatch {
        func: u32,
        block: u32,
        expected: ValType,
        found: ValType,
    },
    /// Branch argument count did not match the target block's parameter count.
    ArgCountMismatch {
        func: u32,
        block: u32,
        target: u32,
        expected: usize,
        found: usize,
    },
    /// Return value count did not match the function's result count.
    ResultCountMismatch {
        func: u32,
        block: u32,
        expected: usize,
        found: usize,
    },
    /// A load/store appeared but the module declares no linear memory.
    MemoryNotDeclared { func: u32, block: u32 },
    /// The declared window size (`1 << size_log2`) is not representable.
    MemorySizeTooLarge { size_log2: u8 },
    /// A `data` segment was declared but the module has no linear memory to place it in.
    DataWithoutMemory { seg: u32 },
    /// A `data` segment's `[offset, offset+len)` does not fit within the declared window.
    DataOutOfWindow { seg: u32 },
    /// A `call` referenced a function index that does not exist.
    CallFuncOutOfRange { func: u32, block: u32, callee: u32 },
    /// A `call`'s argument count did not match the callee's parameter count.
    CallArgCountMismatch {
        func: u32,
        block: u32,
        expected: usize,
        found: usize,
    },
    /// A `thread.spawn` named a function whose signature is not the fixed thread entry type
    /// `(i64 sp, i64 arg) -> i64` (§12).
    ThreadEntrySignature { func: u32, block: u32, callee: u32 },
    /// An atomic carried an ordering its op can't have: a load with release semantics, or a store
    /// with acquire semantics (§12 / C11).
    BadAtomicOrdering { func: u32, block: u32 },
    /// A `<shape>.extract_lane`/`replace_lane` named a lane index `>= shape.lanes()`, or an
    /// `i8x16.shuffle` byte index `>= 32` (§17). Lane indices are immediates, so this is a
    /// structural check.
    BadSimdLane { func: u32, block: u32 },
    /// A lane-wise op was given a shape of the wrong category — an integer op on a float shape
    /// or a float op on an integer shape (§17).
    BadSimdShape { func: u32, block: u32 },
    /// A [`Inst::CallImport`] reached the verifier (§7). Named imports must be lowered to
    /// concrete `cap.call`s by `svm_ir::resolve_imports` at instantiation *before*
    /// verification; an unresolved import in a module presented for execution is fail-closed.
    UnresolvedImport { func: u32, block: u32, import: u32 },
}

/// Verify an entire module. `Ok(())` is the only "accept".
pub fn verify_module(m: &Module) -> Result<(), VerifyError> {
    // A declared window must have a representable size (`1 << size_log2`, with the
    // mask `size - 1` well-defined). `size_log2 == 63` is the largest window.
    if let Some(mem) = &m.memory {
        if mem.size_log2 >= 64 {
            return Err(VerifyError::MemorySizeTooLarge {
                size_log2: mem.size_log2,
            });
        }
    }
    // Data segments must fit within the declared window `[0, size)` (§3a / D40). The runtime
    // copies them in (and protects `readonly` ones) at instantiation, so an out-of-window or
    // memory-less segment is rejected here, fail-closed.
    for (i, d) in m.data.iter().enumerate() {
        let seg = i as u32;
        let Some(mem) = &m.memory else {
            return Err(VerifyError::DataWithoutMemory { seg });
        };
        let end = d.offset.checked_add(d.bytes.len() as u64);
        if end.is_none_or(|e| e > mem.size()) {
            return Err(VerifyError::DataOutOfWindow { seg });
        }
    }
    let has_memory = m.memory.is_some();
    for (fi, f) in m.funcs.iter().enumerate() {
        verify_func(fi as u32, f, &m.funcs, has_memory)?;
    }
    Ok(())
}

fn verify_func(fi: u32, f: &Func, funcs: &[Func], has_memory: bool) -> Result<(), VerifyError> {
    // Per function: the entry block's parameters are the function's parameters.
    match f.blocks.first() {
        Some(entry) if entry.params == f.params => {}
        Some(_) => return Err(VerifyError::EntryParamsMismatch { func: fi }),
        // A function with no blocks cannot return; treat as ill-formed.
        None => return Err(VerifyError::EntryParamsMismatch { func: fi }),
    }

    let nblocks = f.blocks.len() as u32;
    for (bi, b) in f.blocks.iter().enumerate() {
        let bi = bi as u32;
        // Seed the local type vector with the block's declared parameter types.
        let mut types: Vec<ValType> = b.params.clone();

        for inst in &b.insts {
            // `Call` needs the whole-module signatures and appends 0..N results, so
            // it is checked here rather than in `check_inst`.
            if let Inst::Call { func, args } = inst {
                let callee = funcs
                    .get(*func as usize)
                    .ok_or(VerifyError::CallFuncOutOfRange {
                        func: fi,
                        block: bi,
                        callee: *func,
                    })?;
                if args.len() != callee.params.len() {
                    return Err(VerifyError::CallArgCountMismatch {
                        func: fi,
                        block: bi,
                        expected: callee.params.len(),
                        found: args.len(),
                    });
                }
                {
                    let cx = Cx {
                        fi,
                        bi,
                        types: &types,
                    };
                    for (a, want) in args.iter().zip(&callee.params) {
                        cx.expect(*a, *want)?;
                    }
                }
                types.extend_from_slice(&callee.results);
                continue;
            }
            if let Inst::RefFunc { func } = inst {
                if *func as usize >= funcs.len() {
                    return Err(VerifyError::CallFuncOutOfRange {
                        func: fi,
                        block: bi,
                        callee: *func,
                    });
                }
                types.push(ValType::I32); // a funcref is a plain i32 index (§3c)
                continue;
            }
            if let Inst::CallIndirect { ty, idx, args } = inst {
                {
                    let cx = Cx {
                        fi,
                        bi,
                        types: &types,
                    };
                    cx.expect(*idx, ValType::I32)?;
                    if args.len() != ty.params.len() {
                        return Err(VerifyError::CallArgCountMismatch {
                            func: fi,
                            block: bi,
                            expected: ty.params.len(),
                            found: args.len(),
                        });
                    }
                    for (a, want) in args.iter().zip(&ty.params) {
                        cx.expect(*a, *want)?;
                    }
                }
                types.extend_from_slice(&ty.results);
                continue;
            }
            if let Inst::CapCall {
                sig, handle, args, ..
            } = inst
            {
                {
                    let cx = Cx {
                        fi,
                        bi,
                        types: &types,
                    };
                    // The handle is a forgeable i32 index; safety is the runtime
                    // use-site check (host-owned table type_id/generation), not typing.
                    cx.expect(*handle, ValType::I32)?;
                    if args.len() != sig.params.len() {
                        return Err(VerifyError::CallArgCountMismatch {
                            func: fi,
                            block: bi,
                            expected: sig.params.len(),
                            found: args.len(),
                        });
                    }
                    for (a, want) in args.iter().zip(&sig.params) {
                        cx.expect(*a, *want)?;
                    }
                }
                types.extend_from_slice(&sig.results);
                continue;
            }
            // §7 named imports must be resolved to `cap.call`s before verification — reject
            // a stray `CallImport` fail-closed (it carries no `type_id`/`op` to check).
            if let Inst::CallImport { import, .. } = inst {
                return Err(VerifyError::UnresolvedImport {
                    func: fi,
                    block: bi,
                    import: *import,
                });
            }
            // §12 `cont.resume` appends two results `(status: i32, value: i64)`, so —
            // like `call` — it is checked here rather than in `check_inst`.
            if let Inst::ContResume { k, arg } = inst {
                let cx = Cx {
                    fi,
                    bi,
                    types: &types,
                };
                cx.expect(*k, ValType::I32)?; // forgeable fiber handle
                cx.expect(*arg, ValType::I64)?;
                types.push(ValType::I32); // status
                types.push(ValType::I64); // value
                continue;
            }
            // §7 reflection: `cap.self.count` appends an `i32`; `cap.self.get` reads an `i32`
            // index and appends `(handle: i32, type_id: i32)`. Always valid (no module/memory
            // dependency) — the runtime bounds the index against the live table.
            if let Inst::CapSelfCount = inst {
                types.push(ValType::I32);
                continue;
            }
            if let Inst::CapSelfGet { idx } = inst {
                let cx = Cx {
                    fi,
                    bi,
                    types: &types,
                };
                cx.expect(*idx, ValType::I32)?;
                types.push(ValType::I32); // handle
                types.push(ValType::I32); // type_id
                continue;
            }
            // §12 `thread.spawn` resolves a static `funcidx` whose signature must be the fixed
            // thread-entry type `(i64 sp, i64 arg) -> i64`, so — like `call` — it needs whole-module
            // info.
            if let Inst::ThreadSpawn { func, sp, arg } = inst {
                let callee = funcs
                    .get(*func as usize)
                    .ok_or(VerifyError::CallFuncOutOfRange {
                        func: fi,
                        block: bi,
                        callee: *func,
                    })?;
                if callee.params != [ValType::I64, ValType::I64] || callee.results != [ValType::I64]
                {
                    return Err(VerifyError::ThreadEntrySignature {
                        func: fi,
                        block: bi,
                        callee: *func,
                    });
                }
                let cx = Cx {
                    fi,
                    bi,
                    types: &types,
                };
                cx.expect(*sp, ValType::I64)?;
                cx.expect(*arg, ValType::I64)?;
                types.push(ValType::I32); // the thread handle
                continue;
            }
            // A value-producing instruction appends its result type; `Store` does not.
            if let Some(result) = check_inst(fi, bi, inst, &types, has_memory)? {
                types.push(result);
            }
        }

        check_terminator(fi, bi, &b.term, &types, nblocks, f, funcs)?;
    }
    Ok(())
}

/// Check one instruction's operands against the running type vector and return the
/// result type to append (`None` for `Store`). Operands must reference
/// strictly-earlier indices.
fn check_inst(
    fi: u32,
    bi: u32,
    inst: &Inst,
    types: &[ValType],
    has_memory: bool,
) -> Result<Option<ValType>, VerifyError> {
    let cx = Cx { fi, bi, types };
    // `Store` is the only instruction that yields no value; handle it up front so the
    // main match can produce a single result type.
    if let Inst::Store {
        op, addr, value, ..
    } = inst
    {
        if !has_memory {
            return Err(VerifyError::MemoryNotDeclared {
                func: fi,
                block: bi,
            });
        }
        cx.expect(*addr, ValType::I64)?;
        cx.expect(*value, op.info().1)?;
        return Ok(None);
    }
    // §12 atomic store — the other no-result memory op.
    if let Inst::AtomicStore {
        ty,
        addr,
        value,
        order,
        ..
    } = inst
    {
        if !has_memory {
            return Err(VerifyError::MemoryNotDeclared {
                func: fi,
                block: bi,
            });
        }
        if !order.valid_for_store() {
            return Err(VerifyError::BadAtomicOrdering {
                func: fi,
                block: bi,
            });
        }
        cx.expect(*addr, ValType::I64)?;
        cx.expect(*value, ty.val())?;
        return Ok(None);
    }
    // §17 `v128.store` — the third no-result memory op (a 16-byte masked access).
    if let Inst::V128Store { addr, value, .. } = inst {
        if !has_memory {
            return Err(VerifyError::MemoryNotDeclared {
                func: fi,
                block: bi,
            });
        }
        cx.expect(*addr, ValType::I64)?;
        cx.expect(*value, ValType::V128)?;
        return Ok(None);
    }
    let ty = match inst {
        Inst::ConstI32(_) => ValType::I32,
        Inst::ConstI64(_) => ValType::I64,
        // §7 named imports are rejected above (the multi-result/call section); unreachable here.
        Inst::CallImport { .. } => {
            unreachable!("CallImport handled before check_inst's value match")
        }
        // §7 reflection appends its results in the multi-result section above; unreachable here.
        Inst::CapSelfCount | Inst::CapSelfGet { .. } => {
            unreachable!("cap.self.* handled before check_inst's value match")
        }
        Inst::IntBin { ty, a, b, .. } => {
            let t = ty.val();
            cx.expect(*a, t)?;
            cx.expect(*b, t)?;
            t
        }
        Inst::IntCmp { ty, a, b, .. } => {
            let t = ty.val();
            cx.expect(*a, t)?;
            cx.expect(*b, t)?;
            ValType::I32
        }
        Inst::IntUn { ty, a, .. } => {
            cx.expect(*a, ty.val())?;
            ty.val()
        }
        Inst::Eqz { ty, a } => {
            cx.expect(*a, ty.val())?;
            ValType::I32
        }
        Inst::Convert { op, a } => {
            let (_, src, dst) = op.sig();
            cx.expect(*a, src)?;
            dst
        }
        Inst::Select { cond, a, b } => {
            cx.expect(*cond, ValType::I32)?;
            // Polymorphic: `a` defines the result type, `b` must match it.
            let t = cx.type_of(*a)?;
            cx.expect(*b, t)?;
            t
        }
        Inst::ConstF32(_) => ValType::F32,
        Inst::ConstF64(_) => ValType::F64,
        Inst::FBin { ty, a, b, .. } => {
            let t = ty.val();
            cx.expect(*a, t)?;
            cx.expect(*b, t)?;
            t
        }
        Inst::FUn { ty, a, .. } => {
            cx.expect(*a, ty.val())?;
            ty.val()
        }
        Inst::FCmp { ty, a, b, .. } => {
            let t = ty.val();
            cx.expect(*a, t)?;
            cx.expect(*b, t)?;
            ValType::I32
        }
        Inst::FToISat { op, a } | Inst::FToITrap { op, a } => {
            let (from, to, _) = op.parts();
            cx.expect(*a, from.val())?;
            to.val()
        }
        Inst::PtrAdd { a, b } => {
            cx.expect(*a, ValType::I64)?;
            cx.expect(*b, ValType::I64)?;
            ValType::I64
        }
        Inst::PtrCast { a, .. } => {
            cx.expect(*a, ValType::I64)?;
            ValType::I64
        }
        Inst::IToFConv { op, a } => {
            let (from, to, _) = op.parts();
            cx.expect(*a, from.val())?;
            to.val()
        }
        Inst::Cast { op, a } => {
            let (_, src, dst) = op.sig();
            cx.expect(*a, src)?;
            dst
        }
        Inst::Load { op, addr, .. } => {
            if !has_memory {
                return Err(VerifyError::MemoryNotDeclared {
                    func: fi,
                    block: bi,
                });
            }
            cx.expect(*addr, ValType::I64)?;
            op.info().1
        }
        Inst::AtomicLoad {
            ty, addr, order, ..
        } => {
            if !has_memory {
                return Err(VerifyError::MemoryNotDeclared {
                    func: fi,
                    block: bi,
                });
            }
            if !order.valid_for_load() {
                return Err(VerifyError::BadAtomicOrdering {
                    func: fi,
                    block: bi,
                });
            }
            cx.expect(*addr, ValType::I64)?;
            ty.val()
        }
        Inst::AtomicRmw {
            ty, addr, value, ..
        } => {
            if !has_memory {
                return Err(VerifyError::MemoryNotDeclared {
                    func: fi,
                    block: bi,
                });
            }
            cx.expect(*addr, ValType::I64)?;
            cx.expect(*value, ty.val())?;
            ty.val()
        }
        Inst::AtomicCmpxchg {
            ty,
            addr,
            expected,
            replacement,
            ..
        } => {
            if !has_memory {
                return Err(VerifyError::MemoryNotDeclared {
                    func: fi,
                    block: bi,
                });
            }
            cx.expect(*addr, ValType::I64)?;
            cx.expect(*expected, ty.val())?;
            cx.expect(*replacement, ty.val())?;
            ty.val()
        }
        // §12 fibers. `cont.new` takes an i32 funcref, yields an i32 handle; `suspend`
        // takes an i64, yields the i64 of the next resume. (`cont.resume` is multi-result
        // and handled in the main loop.)
        Inst::ContNew { func, sp } => {
            cx.expect(*func, ValType::I32)?;
            cx.expect(*sp, ValType::I64)?; // the fiber's data-stack base
            ValType::I32
        }
        Inst::Suspend { value } => {
            cx.expect(*value, ValType::I64)?;
            ValType::I64
        }
        // §GC conservative root enumeration: i64 heap_lo, heap_hi, buf, cap ⇒ i64 count. Writes the
        // candidate words into guest memory at `buf`, so it requires a declared window.
        Inst::GcRoots {
            heap_lo,
            heap_hi,
            buf,
            cap,
        } => {
            if !has_memory {
                return Err(VerifyError::MemoryNotDeclared {
                    func: fi,
                    block: bi,
                });
            }
            cx.expect(*heap_lo, ValType::I64)?;
            cx.expect(*heap_hi, ValType::I64)?;
            cx.expect(*buf, ValType::I64)?;
            cx.expect(*cap, ValType::I64)?;
            ValType::I64
        }
        // §12 thread join: an i32 thread handle in, the joined vCPU's i64 result out. (The handle
        // is forgeable; safety is the runtime use-site check, like a fiber/capability handle.)
        Inst::ThreadJoin { handle } => {
            cx.expect(*handle, ValType::I32)?;
            ValType::I64
        }
        // §12 futex wait: i64 addr, `ty` expected value, i64 timeout ⇒ i32 status. Touches memory.
        Inst::MemoryWait {
            ty,
            addr,
            expected,
            timeout,
        } => {
            if !has_memory {
                return Err(VerifyError::MemoryNotDeclared {
                    func: fi,
                    block: bi,
                });
            }
            cx.expect(*addr, ValType::I64)?;
            cx.expect(*expected, ty.val())?;
            cx.expect(*timeout, ValType::I64)?;
            ValType::I32
        }
        // §12 futex notify: i64 addr, i32 count ⇒ i32 woken. Requires declared memory.
        Inst::MemoryNotify { addr, count } => {
            if !has_memory {
                return Err(VerifyError::MemoryNotDeclared {
                    func: fi,
                    block: bi,
                });
            }
            cx.expect(*addr, ValType::I64)?;
            cx.expect(*count, ValType::I32)?;
            ValType::I32
        }
        // A standalone fence produces no value and needs no memory or operands (any ordering is
        // valid for a fence) — accept it directly.
        Inst::AtomicFence { .. } => return Ok(None),

        // ----- §17 SIMD (D58): total lane-typing rules -----
        Inst::ConstV128(_) => ValType::V128,
        Inst::V128Load { addr, .. } => {
            if !has_memory {
                return Err(VerifyError::MemoryNotDeclared {
                    func: fi,
                    block: bi,
                });
            }
            cx.expect(*addr, ValType::I64)?;
            ValType::V128
        }
        Inst::Splat { shape, a } => {
            cx.expect(*a, shape.lane_val())?;
            ValType::V128
        }
        Inst::ExtractLane { shape, lane, a, .. } => {
            if *lane >= shape.lanes() {
                return Err(VerifyError::BadSimdLane {
                    func: fi,
                    block: bi,
                });
            }
            cx.expect(*a, ValType::V128)?;
            shape.lane_val()
        }
        Inst::ReplaceLane {
            shape, lane, a, b, ..
        } => {
            if *lane >= shape.lanes() {
                return Err(VerifyError::BadSimdLane {
                    func: fi,
                    block: bi,
                });
            }
            cx.expect(*a, ValType::V128)?;
            cx.expect(*b, shape.lane_val())?;
            ValType::V128
        }
        Inst::VIntBin { shape, a, b, .. } => {
            if shape.is_float() {
                return Err(VerifyError::BadSimdShape {
                    func: fi,
                    block: bi,
                });
            }
            cx.expect(*a, ValType::V128)?;
            cx.expect(*b, ValType::V128)?;
            ValType::V128
        }
        Inst::VIntCmp { shape, a, b, .. } => {
            if shape.is_float() {
                return Err(VerifyError::BadSimdShape {
                    func: fi,
                    block: bi,
                });
            }
            cx.expect(*a, ValType::V128)?;
            cx.expect(*b, ValType::V128)?;
            ValType::V128
        }
        Inst::VShift { shape, a, amt, .. } => {
            if shape.is_float() {
                return Err(VerifyError::BadSimdShape {
                    func: fi,
                    block: bi,
                });
            }
            cx.expect(*a, ValType::V128)?;
            cx.expect(*amt, ValType::I32)?;
            ValType::V128
        }
        Inst::VFloatBin { shape, a, b, .. }
        | Inst::VFloatCmp { shape, a, b, .. }
        | Inst::VPMinMax { shape, a, b, .. } => {
            if !shape.is_float() {
                return Err(VerifyError::BadSimdShape {
                    func: fi,
                    block: bi,
                });
            }
            cx.expect(*a, ValType::V128)?;
            cx.expect(*b, ValType::V128)?;
            ValType::V128
        }
        Inst::VFloatUn { shape, a, .. } => {
            if !shape.is_float() {
                return Err(VerifyError::BadSimdShape {
                    func: fi,
                    block: bi,
                });
            }
            cx.expect(*a, ValType::V128)?;
            ValType::V128
        }
        Inst::VIntUn { shape, a, .. } => {
            if shape.is_float() {
                return Err(VerifyError::BadSimdShape {
                    func: fi,
                    block: bi,
                });
            }
            cx.expect(*a, ValType::V128)?;
            ValType::V128
        }
        // `i8x16.popcnt`: shape is fixed (i8x16), so there is no lane rule to enforce.
        Inst::VPopcnt { a } => {
            cx.expect(*a, ValType::V128)?;
            ValType::V128
        }
        // Saturating add/sub is `i8x16`/`i16x8` only (the wasm spec has no wider sat).
        Inst::VSatBin { shape, a, b, .. } => {
            if !matches!(shape, VShape::I8x16 | VShape::I16x8) {
                return Err(VerifyError::BadSimdShape {
                    func: fi,
                    block: bi,
                });
            }
            cx.expect(*a, ValType::V128)?;
            cx.expect(*b, ValType::V128)?;
            ValType::V128
        }
        // Unsigned rounding average: `i8x16`/`i16x8` only (the only shapes wasm defines `avgr_u`).
        Inst::VAvgr { shape, a, b } => {
            if !matches!(shape, VShape::I8x16 | VShape::I16x8) {
                return Err(VerifyError::BadSimdShape {
                    func: fi,
                    block: bi,
                });
            }
            cx.expect(*a, ValType::V128)?;
            cx.expect(*b, ValType::V128)?;
            ValType::V128
        }
        // Dot product: fixed shapes (i16x8 → i32x4), so there is no lane rule to enforce.
        Inst::VDot { a, b } => {
            cx.expect(*a, ValType::V128)?;
            cx.expect(*b, ValType::V128)?;
            ValType::V128
        }
        // Widen: the result shape must be an integer shape that has a (half-width) source.
        Inst::VWiden { shape, a, .. } => {
            if shape.narrower().is_none() {
                return Err(VerifyError::BadSimdShape {
                    func: fi,
                    block: bi,
                });
            }
            cx.expect(*a, ValType::V128)?;
            ValType::V128
        }
        // Narrow: `i8x16`/`i16x8` results only (the wasm spec has no wider narrow).
        Inst::VNarrow { shape, a, b, .. } => {
            if !matches!(shape, VShape::I8x16 | VShape::I16x8) {
                return Err(VerifyError::BadSimdShape {
                    func: fi,
                    block: bi,
                });
            }
            cx.expect(*a, ValType::V128)?;
            cx.expect(*b, ValType::V128)?;
            ValType::V128
        }
        // Lane conversions: `v128` → `v128`, fully described by the op.
        Inst::VConvert { a, .. } => {
            cx.expect(*a, ValType::V128)?;
            ValType::V128
        }
        // Boolean reductions: a `v128` → an `i32`. `all_true`/`bitmask` carry an integer shape;
        // `any_true` is shape-agnostic.
        Inst::VAnyTrue { a } => {
            cx.expect(*a, ValType::V128)?;
            ValType::I32
        }
        Inst::VAllTrue { shape, a } | Inst::VBitmask { shape, a } => {
            if shape.is_float() {
                return Err(VerifyError::BadSimdShape {
                    func: fi,
                    block: bi,
                });
            }
            cx.expect(*a, ValType::V128)?;
            ValType::I32
        }
        Inst::VBitBin { a, b, .. } => {
            cx.expect(*a, ValType::V128)?;
            cx.expect(*b, ValType::V128)?;
            ValType::V128
        }
        Inst::VNot { a } => {
            cx.expect(*a, ValType::V128)?;
            ValType::V128
        }
        Inst::Bitselect { a, b, mask } => {
            cx.expect(*a, ValType::V128)?;
            cx.expect(*b, ValType::V128)?;
            cx.expect(*mask, ValType::V128)?;
            ValType::V128
        }
        Inst::Shuffle { lanes, a, b } => {
            // Each byte index selects from the 32-byte `a ++ b`; ≥32 is structurally invalid.
            if lanes.iter().any(|&l| l >= 32) {
                return Err(VerifyError::BadSimdLane {
                    func: fi,
                    block: bi,
                });
            }
            cx.expect(*a, ValType::V128)?;
            cx.expect(*b, ValType::V128)?;
            ValType::V128
        }
        Inst::Swizzle { a, b } => {
            cx.expect(*a, ValType::V128)?;
            cx.expect(*b, ValType::V128)?;
            ValType::V128
        }
        Inst::SimdWidthBytes => ValType::I32,

        // Handled before/around the match; listed for exhaustiveness (no panic).
        Inst::Store { .. }
        | Inst::AtomicStore { .. }
        | Inst::V128Store { .. }
        | Inst::Call { .. }
        | Inst::RefFunc { .. }
        | Inst::CallIndirect { .. }
        | Inst::CapCall { .. }
        | Inst::ContResume { .. }
        | Inst::ThreadSpawn { .. } => return Ok(None),
    };
    Ok(Some(ty))
}

fn check_terminator(
    fi: u32,
    bi: u32,
    term: &Terminator,
    types: &[ValType],
    nblocks: u32,
    f: &Func,
    funcs: &[Func],
) -> Result<(), VerifyError> {
    let cx = Cx { fi, bi, types };
    match term {
        Terminator::Br { target, args } => {
            check_edge(&cx, *target, args, nblocks, f)?;
        }
        Terminator::BrIf {
            cond,
            then_blk,
            then_args,
            else_blk,
            else_args,
        } => {
            cx.expect(*cond, ValType::I32)?;
            check_edge(&cx, *then_blk, then_args, nblocks, f)?;
            check_edge(&cx, *else_blk, else_args, nblocks, f)?;
        }
        Terminator::BrTable {
            idx,
            targets,
            default,
        } => {
            cx.expect(*idx, ValType::I32)?;
            for (t, args) in targets {
                check_edge(&cx, *t, args, nblocks, f)?;
            }
            let (t, args) = default;
            check_edge(&cx, *t, args, nblocks, f)?;
        }
        Terminator::Return(vals) => {
            if vals.len() != f.results.len() {
                return Err(VerifyError::ResultCountMismatch {
                    func: fi,
                    block: bi,
                    expected: f.results.len(),
                    found: vals.len(),
                });
            }
            for (v, want) in vals.iter().zip(&f.results) {
                cx.expect(*v, *want)?;
            }
        }
        Terminator::ReturnCall { func, args } => {
            let callee = funcs
                .get(*func as usize)
                .ok_or(VerifyError::CallFuncOutOfRange {
                    func: fi,
                    block: bi,
                    callee: *func,
                })?;
            check_tail_call(&cx, args, &callee.params, &callee.results, &f.results)?;
        }
        Terminator::ReturnCallIndirect { ty, idx, args } => {
            cx.expect(*idx, ValType::I32)?;
            check_tail_call(&cx, args, &ty.params, &ty.results, &f.results)?;
        }
        // Aborts unconditionally; references nothing, so nothing to check.
        Terminator::Unreachable => {}
    }
    Ok(())
}

/// Shared checks for `return_call`/`return_call_indirect`: the args match the
/// callee's parameters, and the callee's results equal *this* function's results
/// (a tail call returns the callee's results as our own).
fn check_tail_call(
    cx: &Cx,
    args: &[ValIdx],
    callee_params: &[ValType],
    callee_results: &[ValType],
    func_results: &[ValType],
) -> Result<(), VerifyError> {
    if args.len() != callee_params.len() {
        return Err(VerifyError::CallArgCountMismatch {
            func: cx.fi,
            block: cx.bi,
            expected: callee_params.len(),
            found: args.len(),
        });
    }
    for (a, want) in args.iter().zip(callee_params) {
        cx.expect(*a, *want)?;
    }
    if callee_results != func_results {
        return Err(VerifyError::ResultCountMismatch {
            func: cx.fi,
            block: cx.bi,
            expected: func_results.len(),
            found: callee_results.len(),
        });
    }
    Ok(())
}

/// Check a single branch edge: target in range, arg count + types match the target
/// block's declared parameters exactly.
fn check_edge(
    cx: &Cx,
    target: BlockIdx,
    args: &[ValIdx],
    nblocks: u32,
    f: &Func,
) -> Result<(), VerifyError> {
    if target >= nblocks {
        return Err(VerifyError::BlockOutOfRange {
            func: cx.fi,
            block: cx.bi,
            target,
        });
    }
    let target_params = &f.blocks[target as usize].params;
    if args.len() != target_params.len() {
        return Err(VerifyError::ArgCountMismatch {
            func: cx.fi,
            block: cx.bi,
            target,
            expected: target_params.len(),
            found: args.len(),
        });
    }
    for (v, want) in args.iter().zip(target_params) {
        cx.expect(*v, *want)?;
    }
    Ok(())
}

/// Bundles the location + running type vector for concise operand checks.
struct Cx<'a> {
    fi: u32,
    bi: u32,
    types: &'a [ValType],
}

impl Cx<'_> {
    /// The type of an earlier-defined operand, or `ValueOutOfRange`.
    fn type_of(&self, v: ValIdx) -> Result<ValType, VerifyError> {
        self.types
            .get(v as usize)
            .copied()
            .ok_or(VerifyError::ValueOutOfRange {
                func: self.fi,
                block: self.bi,
                value: v,
                defined: self.types.len() as u32,
            })
    }

    /// An operand must be defined earlier in this block and have exactly `want`'s type.
    fn expect(&self, v: ValIdx, want: ValType) -> Result<(), VerifyError> {
        let found = self.type_of(v)?;
        if found != want {
            return Err(VerifyError::TypeMismatch {
                func: self.fi,
                block: self.bi,
                expected: want,
                found,
            });
        }
        Ok(())
    }
}
