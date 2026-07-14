//! SPEC.md slice 6 — SIMD semantic vectors. Every §17 `v128` value op runs its
//! boundary-pattern vectors on all three backends against the spec's lane semantics
//! (`svm_spec::simd`), plus the two memory-facing SIMD ops (`v128.load`/`store`) on
//! the window-boundary lattice — the widened 16-byte masked access is the one
//! escape-TCB delta SIMD adds (D58), so its admission rule gets the same treatment as
//! the scalar memory ops.
//!
//! `v128` can't cross the JIT entry ABI, so inputs are baked as consts and a `v128`
//! result is observed as two `i64x2.extract_lane`s; vectors are batched many-per-module
//! to amortize JIT compiles (SIMD value ops are total — a batch always completes).
//!
//! NaN policy (D58): a computed float lane's NaN bit pattern is unpinned — rows carry
//! `nan_lanes`, and those lanes compare as "is a NaN"; everything else is bit-exact.

use svm_interp::{bytecode, run, run_capture_reserved, Host, Trap, Value};
use svm_ir::{Block, Func, Inst, Memory, Module, Terminator, VShape, ValType};
use svm_jit::{compile, JitOutcome, TrapKind};
use svm_spec::simd::{
    encode_probe_module, module_for_simd_batch, simd_rows, simd_vectors_for, SimdRow,
};
use svm_spec::{SpecVal, MEM_LOG2, MEM_SIZE};

const BATCH: usize = 16;

fn value_slots(v: &Value) -> Vec<i64> {
    match v {
        Value::I32(x) => vec![*x as i64],
        Value::I64(x) => vec![*x],
        Value::F32(x) => vec![x.to_bits() as i64],
        Value::F64(x) => vec![x.to_bits() as i64],
        Value::V128(b) => vec![
            i64::from_le_bytes(b[..8].try_into().unwrap()),
            i64::from_le_bytes(b[8..].try_into().unwrap()),
        ],
        Value::Ref(x) => vec![*x as i64],
    }
}

/// Compare one expected value against its observation slots, lane-NaN-aware.
fn expected_matches(expected: SpecVal, nan_lanes: Option<VShape>, slots: &[i64]) -> bool {
    match expected {
        SpecVal::V128(want) => {
            let mut got = [0u8; 16];
            got[..8].copy_from_slice(&slots[0].to_le_bytes());
            got[8..].copy_from_slice(&slots[1].to_le_bytes());
            match nan_lanes {
                None => got == want,
                Some(shape) => {
                    let w = shape.lane_bytes() as usize;
                    (0..16 / w).all(|i| {
                        let (gw, ww) = (&got[i * w..(i + 1) * w], &want[i * w..(i + 1) * w]);
                        if gw == ww {
                            return true;
                        }
                        // Both NaN of the lane's float width ⇒ unpinned bits, accept.
                        match w {
                            4 => {
                                f32::from_le_bytes(gw.try_into().unwrap()).is_nan()
                                    && f32::from_le_bytes(ww.try_into().unwrap()).is_nan()
                            }
                            _ => {
                                f64::from_le_bytes(gw.try_into().unwrap()).is_nan()
                                    && f64::from_le_bytes(ww.try_into().unwrap()).is_nan()
                            }
                        }
                    })
                }
            }
        }
        SpecVal::I32(e) => e == slots[0] as i32,
        SpecVal::I64(e) => e == slots[0],
        SpecVal::F32(e) => {
            e == slots[0] as u32
                || (f32::from_bits(e).is_nan() && f32::from_bits(slots[0] as u32).is_nan())
        }
        SpecVal::F64(e) => {
            e == slots[0] as u64
                || (f64::from_bits(e).is_nan() && f64::from_bits(slots[0] as u64).is_nan())
        }
    }
}

/// Slots each vector's observation occupies in the batch's result list.
fn slots_per_vector(row: &SimdRow) -> usize {
    if row.result == ValType::V128 {
        2
    } else {
        1
    }
}

fn check_batch(row: &SimdRow, batch: &[Vec<SpecVal>]) {
    let m = module_for_simd_batch(row, batch);
    svm::verify::verify_module(&m)
        .unwrap_or_else(|e| panic!("simd batch module for {} fails verify: {e:?}", row.id));
    // The reference verifier (suite 2) must agree on the accept side here too — this
    // is where its SIMD lane-typing arms get exercised over every op family.
    svm_spec::verify::verify(&m)
        .unwrap_or_else(|e| panic!("reference verifier rejects the {} batch: {e}", row.id));
    let expected: Vec<SpecVal> = batch.iter().map(|v| (row.eval)(v)).collect();
    let spv = slots_per_vector(row);

    let check_slots = |backend: &str, slots: &[i64]| {
        for (i, e) in expected.iter().enumerate() {
            let s = &slots[i * spv..(i + 1) * spv];
            assert!(
                expected_matches(*e, row.nan_lanes, s),
                "spec-simd divergence [{backend}] op={} vector={:?}\n expected={e:?}\n got slots={s:?}",
                row.id,
                batch[i]
            );
        }
    };

    // Tree-walk interpreter.
    let mut fuel = 100_000u64;
    let ir = run(&m, 0, &[], &mut fuel);
    let vs = ir.unwrap_or_else(|t| panic!("interp trapped a total SIMD batch ({}): {t:?}", row.id));
    let slots: Vec<i64> = vs.iter().flat_map(value_slots).collect();
    check_slots("interp", &slots);

    // Bytecode interpreter.
    let mut fuel = 100_000u64;
    let bc = bytecode::compile_and_run(&m, 0, &[], &mut fuel)
        .unwrap_or_else(|| panic!("bytecode does not support the {} batch", row.id));
    let vs =
        bc.unwrap_or_else(|t| panic!("bytecode trapped a total SIMD batch ({}): {t:?}", row.id));
    let slots: Vec<i64> = vs.iter().flat_map(value_slots).collect();
    check_slots("bytecode", &slots);

    // Cranelift JIT — one compile per batch. `i64x2` min/max is a **documented**
    // JIT bail (svm-jit's `ensure_supported`: no legalizable Cranelift lowering on
    // the target ISAs; wasm never emits it; the interpreters remain the oracle) —
    // exactly those four rows skip the JIT leg. Any other `Unsupported` is a finding.
    const DOCUMENTED_JIT_BAILS: &[&str] = &["i64x2.mins", "i64x2.minu", "i64x2.maxs", "i64x2.maxu"];
    let mut cm = match compile(&m, 0) {
        Ok(cm) => cm,
        Err(svm_jit::JitError::Unsupported(_))
            if DOCUMENTED_JIT_BAILS.contains(&row.id.as_str()) =>
        {
            return;
        }
        Err(e) => panic!("simd batch for {} fails JIT compile: {e:?}", row.id),
    };
    let (out, _mem) = cm.run(&[], None, None, None).unwrap();
    match out {
        JitOutcome::Returned(slots) => check_slots("jit", &slots),
        other => panic!(
            "jit did not return a total SIMD batch ({}): {other:?}",
            row.id
        ),
    }
}

#[test]
fn spec_simd_vectors_match_all_backends() {
    let next = std::sync::atomic::AtomicUsize::new(0);
    let vectors_run = std::sync::atomic::AtomicUsize::new(0);
    let n_rows = simd_rows().len();
    let workers = std::thread::available_parallelism().map_or(4, |n| n.get().min(8));
    std::thread::scope(|s| {
        for _ in 0..workers {
            s.spawn(|| {
                // Rows carry non-Sync boxed closures; construction is deterministic,
                // so each worker builds its own copy and claims rows by index.
                let rows = simd_rows();
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(row) = rows.get(i) else { return };
                    let vectors = simd_vectors_for(row);
                    for batch in vectors.chunks(BATCH) {
                        check_batch(row, batch);
                    }
                    vectors_run.fetch_add(vectors.len(), std::sync::atomic::Ordering::Relaxed);
                }
            });
        }
    });
    let vectors_run = vectors_run.into_inner();
    assert!(n_rows > 200, "suspiciously few SIMD rows: {n_rows}");
    assert!(
        vectors_run > 8_000,
        "suspiciously few SIMD vectors ran: {vectors_run}"
    );
}

// --- v128.load / v128.store: the widened 16-byte masked access (D58) -------------------

fn interp_trap_mem() -> Trap {
    Trap::MemoryFault
}

fn init_window() -> Vec<u8> {
    (0..MEM_SIZE as usize)
        .map(|i| (i as u8).wrapping_mul(31) ^ 0xa5)
        .collect()
}

/// The 16-byte admission lattice: fits exactly, one past, wrap-around.
fn v128_mem_lattice() -> Vec<(u64, u64)> {
    let mut cases = Vec::new();
    for addr in [
        0,
        8,
        MEM_SIZE - 16,
        MEM_SIZE - 15,
        MEM_SIZE - 1,
        MEM_SIZE,
        u64::MAX,
    ] {
        for offset in [0, MEM_SIZE, u64::MAX] {
            cases.push((addr, offset));
        }
    }
    cases
}

fn admit16(addr: u64, offset: u64) -> Option<usize> {
    match addr.checked_add(offset).and_then(|e| e.checked_add(16)) {
        Some(end) if end <= MEM_SIZE => Some((end - 16) as usize),
        _ => None,
    }
}

#[test]
fn spec_v128_load_store_boundary_lattice() {
    let init = init_window();
    let value: [u8; 16] = core::array::from_fn(|i| 0x40 + i as u8);
    for (addr, offset) in v128_mem_lattice() {
        // v128.load: const addr → load → two i64 extracts.
        let mut insts = vec![
            Inst::ConstI64(addr as i64),
            Inst::V128Load {
                addr: 0,
                offset,
                align: 0,
            },
        ];
        for lane in [0u8, 1] {
            insts.push(Inst::ExtractLane {
                shape: VShape::I64x2,
                lane,
                signed: false,
                a: 1,
            });
        }
        let m = Module {
            funcs: vec![Func {
                params: vec![],
                results: vec![ValType::I64, ValType::I64],
                blocks: vec![Block {
                    params: vec![],
                    insts,
                    term: Terminator::Return(vec![2, 3]),
                }],
            }],
            memory: Some(Memory {
                size_log2: MEM_LOG2,
            }),
            ..Default::default()
        };
        svm::verify::verify_module(&m).unwrap();
        let expected = admit16(addr, offset);
        run_all_mem(
            &m,
            &init,
            "v128.load",
            addr,
            offset,
            |outcome, _mem| match (expected, outcome) {
                (Some(ea), Ok(slots)) => {
                    let mut got = [0u8; 16];
                    got[..8].copy_from_slice(&slots[0].to_le_bytes());
                    got[8..].copy_from_slice(&slots[1].to_le_bytes());
                    assert_eq!(&got[..], &init[ea..ea + 16], "v128.load bytes");
                }
                (None, Err(())) => {}
                (e, o) => {
                    panic!("v128.load addr={addr:#x} offset={offset:#x}: expected {e:?}, got {o:?}")
                }
            },
        );

        // v128.store: const addr + const value → store.
        let m = Module {
            funcs: vec![Func {
                params: vec![],
                results: vec![],
                blocks: vec![Block {
                    params: vec![],
                    insts: vec![
                        Inst::ConstI64(addr as i64),
                        Inst::ConstV128(value),
                        Inst::V128Store {
                            addr: 0,
                            value: 1,
                            offset,
                            align: 0,
                        },
                    ],
                    term: Terminator::Return(vec![]),
                }],
            }],
            memory: Some(Memory {
                size_log2: MEM_LOG2,
            }),
            ..Default::default()
        };
        svm::verify::verify_module(&m).unwrap();
        let mut model = init.clone();
        if let Some(ea) = expected {
            model[ea..ea + 16].copy_from_slice(&value);
        }
        run_all_mem(
            &m,
            &init,
            "v128.store",
            addr,
            offset,
            |outcome, mem| match (expected, outcome) {
                (Some(_), Ok(_)) => assert_eq!(mem, &model[..], "v128.store window"),
                (None, Err(())) => assert_eq!(mem, &init[..], "faulting v128.store mutated"),
                (e, o) => panic!(
                    "v128.store addr={addr:#x} offset={offset:#x}: expected {e:?}, got {o:?}"
                ),
            },
        );
    }
}

/// Run a memory-observing module on all three backends; `check(outcome, window)` gets
/// `Ok(slots)` or `Err(())` for a MemoryFault (any other trap panics). The JIT's
/// faulting window is not checked (see spec_mem's I21 note; scalar-path v128 accesses
/// don't partial-write, but keep the two suites' policies aligned).
fn run_all_mem(
    m: &Module,
    init: &[u8],
    what: &str,
    addr: u64,
    offset: u64,
    check: impl Fn(Result<&[i64], ()>, &[u8]),
) {
    let mut fuel = 10_000u64;
    let (ir, imem) = run_capture_reserved(m, 0, &[], &mut fuel, init, MEM_LOG2);
    match &ir {
        Ok(vs) => {
            let slots: Vec<i64> = vs.iter().flat_map(value_slots).collect();
            check(Ok(&slots), &imem);
        }
        Err(t) if *t == interp_trap_mem() => check(Err(()), &imem),
        Err(t) => panic!("[interp] {what} addr={addr:#x} offset={offset:#x}: {t:?}"),
    }

    let mut fuel = 10_000u64;
    let mut host = Host::new();
    let (bc, bmem) = bytecode::compile_and_run_capture_reserved_with_host(
        m,
        0,
        &[],
        &mut fuel,
        init,
        MEM_LOG2,
        &mut host,
    )
    .unwrap_or_else(|| panic!("[bytecode] {what}: unsupported module"));
    match &bc {
        Ok(vs) => {
            let slots: Vec<i64> = vs.iter().flat_map(value_slots).collect();
            check(Ok(&slots), &bmem);
        }
        Err(t) if *t == interp_trap_mem() => check(Err(()), &bmem),
        Err(t) => panic!("[bytecode] {what} addr={addr:#x} offset={offset:#x}: {t:?}"),
    }

    let mut cm = compile(m, 0).unwrap();
    let (out, jmem) = cm.run(&[], Some(init), None, None).unwrap();
    match out {
        JitOutcome::Returned(slots) => check(Ok(&slots), &jmem[..init.len().min(jmem.len())]),
        JitOutcome::Trapped(TrapKind::MemoryFault) => check(Err(()), init), // window unchecked on JIT trap
        other => panic!("[jit] {what} addr={addr:#x} offset={offset:#x}: {other:?}"),
    }
}

// --- encoding + verifier integration for the SIMD rows ---------------------------------

#[test]
fn spec_simd_encoding_conformance() {
    use svm::encode::{decode_module, encode_module};
    use svm_spec::Enc;
    for row in simd_rows() {
        // A raw single-op probe (dangling indices are fine — encode/decode are
        // structural); first divergence vs a const baseline is the 0xFE prefix.
        let inst = if row.id == "v128.const" {
            Inst::ConstV128([7u8; 16])
        } else {
            let idx: Vec<u32> = (0..row.inputs.len() as u32).collect();
            (row.build)(&idx)
        };
        let m = encode_probe_module(inst);
        let bytes = encode_module(&m);
        let back =
            decode_module(&bytes).unwrap_or_else(|e| panic!("decode failed for {}: {e:?}", row.id));
        assert_eq!(back, m, "decode∘encode changed the IR for {}", row.id);

        let mut base = m.clone();
        base.funcs[0].blocks[0].insts[0] = Inst::ConstI32(0);
        let base_bytes = encode_module(&base);
        let i = bytes
            .iter()
            .zip(&base_bytes)
            .position(|(a, b)| a != b)
            .unwrap_or_else(|| panic!("no divergence vs const baseline for {}", row.id));
        match row.encoding {
            Enc::Prefixed(p, sub) => {
                assert_eq!(
                    bytes[i], p,
                    "escape prefix for {}: spec says {p:#04x}, encoder wrote {:#04x}",
                    row.id, bytes[i]
                );
                assert_eq!(
                    bytes[i + 1],
                    sub,
                    "sub-opcode for {}: spec says {sub:#04x}, encoder wrote {:#04x}",
                    row.id,
                    bytes[i + 1]
                );
            }
            Enc::Byte(_) => panic!("SIMD rows are always prefix-encoded ({})", row.id),
        }
    }
}
