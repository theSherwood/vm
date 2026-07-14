//! SPEC.md slice 5 — memory-op semantic vectors against the spec **window model**.
//! For every load/store/bulk row, run the window-boundary vector lattice on all three
//! backends and assert each matches the model: same loaded value, same trap kind, and
//! — on completing runs — a **byte-identical final window** (the spec-level analogue
//! of the §18 escape oracle, with the *expected* bytes computed by the model rather
//! than only cross-compared between backends).
//!
//! The model pins trap-confinement (§4 / TRAP_CONFINEMENT.md): the whole access span,
//! computed without wraparound, must lie in `[0, mapped)`; a wrapping effective
//! address faults rather than aliasing; zero-length bulk ops are no-ops even at wild
//! pointers; a faulting access mutates nothing.
//!
//! Window bytes are compared on **completing** runs. On faulting runs the trap kind is
//! pinned on all three backends, and the interpreters' windows must additionally match
//! the model (untouched) — the JIT is exempt from the faulting-window comparison
//! because a faulting bulk *libcall* may have written a prefix before the fault
//! (checked against `reserved`, backed only to `mapped` — see D62); whether that
//! partial write should be pinned is an open SPEC.md question.

use svm_interp::{bytecode, run_capture_reserved, Host, Trap, Value};
use svm_jit::{compile, JitOutcome, TrapKind};
use svm_spec::{
    mem_rows, mem_vectors_for, module_for_mem, MemRow, SpecTrap, SpecVal, MEM_LOG2, MEM_OFFSETS,
    MEM_SIZE,
};

fn to_value(v: SpecVal) -> Value {
    match v {
        SpecVal::I32(x) => Value::I32(x),
        SpecVal::I64(x) => Value::I64(x),
        SpecVal::F32(b) => Value::F32(f32::from_bits(b)),
        SpecVal::F64(b) => Value::F64(f64::from_bits(b)),
        SpecVal::V128(_) => unreachable!("no memory row takes a v128"),
    }
}

fn to_slot(v: SpecVal) -> i64 {
    match v {
        SpecVal::I32(x) => x as i64,
        SpecVal::I64(x) => x,
        SpecVal::F32(b) => b as i64,
        SpecVal::F64(b) => b as i64,
        SpecVal::V128(_) => unreachable!("no memory row takes a v128"),
    }
}

fn value_matches(expected: SpecVal, got: &Value) -> bool {
    match (expected, got) {
        (SpecVal::I32(e), Value::I32(g)) => e == *g,
        (SpecVal::I64(e), Value::I64(g)) => e == *g,
        // Pure moves — loads reproduce stored bits exactly, NaNs included.
        (SpecVal::F32(e), Value::F32(g)) => e == g.to_bits(),
        (SpecVal::F64(e), Value::F64(g)) => e == g.to_bits(),
        _ => false,
    }
}

fn slot_matches(expected: SpecVal, slot: i64) -> bool {
    match expected {
        SpecVal::I32(e) => e == slot as i32,
        SpecVal::I64(e) => e == slot,
        SpecVal::F32(e) => e == slot as u32,
        SpecVal::F64(e) => e == slot as u64,
        SpecVal::V128(_) => unreachable!("no memory row yields a v128"),
    }
}

/// The seeded window pattern (nonzero + varied, like the escape oracle's, so a wrong
/// or missing write shows up instead of hiding in zeroes).
fn init_window() -> Vec<u8> {
    (0..MEM_SIZE as usize)
        .map(|i| (i as u8).wrapping_mul(31) ^ 0xa5)
        .collect()
}

/// One vector on all three backends against the model. `first_diff` keeps window
/// mismatch panics readable.
fn first_diff(a: &[u8], b: &[u8]) -> Option<usize> {
    if a.len() != b.len() {
        return Some(a.len().min(b.len()));
    }
    a.iter().zip(b).position(|(x, y)| x != y)
}

fn check_vector(
    row: &MemRow,
    m: &svm_ir::Module,
    cm: &mut svm_jit::CompiledModule,
    offset: u64,
    vector: &[SpecVal],
    init: &[u8],
    model: &mut [u8], // scratch, reset from `init` here (reused to avoid re-alloc)
) {
    // The model outcome: expected value/trap + expected final window. Load rows never
    // mutate the window (`mutates == false`), so their per-backend window compares are
    // skipped — a load that corrupted memory would already fail the escape oracle.
    let mutates = row.result.is_none();
    model.copy_from_slice(init);
    let expected = (row.eval)(vector, offset, model);
    let ctx = |backend: &str, got: &dyn std::fmt::Debug| {
        format!(
            "spec-mem divergence [{backend}] op={} offset={offset:#x} vector={vector:?}\n \
             expected={expected:?}\n got={got:?}",
            row.id
        )
    };
    let win_ctx = |backend: &str, got: &[u8], want: &[u8]| {
        let i = first_diff(got, want).unwrap_or(0);
        format!(
            "spec-mem window mismatch [{backend}] op={} offset={offset:#x} vector={vector:?}\n \
             first diff at byte {i}: got {:?}, model {:?}",
            row.id,
            got.get(i),
            want.get(i)
        )
    };

    let args: Vec<Value> = vector.iter().copied().map(to_value).collect();

    // Tree-walk interpreter. `reserved_log2 = MEM_LOG2` ⇒ a fully-mapped window
    // (`mapped == reserved`), exactly the spec model — and a 64 KiB reservation
    // instead of the default terabyte-scale one, which dominated the suite's runtime
    // when created per vector.
    let mut fuel = 10_000u64;
    let (ir, imem) = run_capture_reserved(m, 0, &args, &mut fuel, init, MEM_LOG2);
    match (&expected, &ir) {
        (Ok(Some(e)), Ok(vs)) if vs.len() == 1 && value_matches(*e, &vs[0]) => {}
        (Ok(None), Ok(vs)) if vs.is_empty() => {}
        (Err(t), Err(tr)) if *tr == interp_trap(*t) => {}
        _ => panic!("{}", ctx("interp", &ir)),
    }
    // Faulting access mutates nothing, too.
    assert!(
        !mutates || imem == model,
        "{}",
        win_ctx("interp", &imem, model)
    );

    // Bytecode interpreter, same fully-mapped reservation (empty powerbox host —
    // these modules make no cap.calls).
    let mut fuel = 10_000u64;
    let mut host = Host::new();
    let (bc, bmem) = bytecode::compile_and_run_capture_reserved_with_host(
        m, 0, &args, &mut fuel, init, MEM_LOG2, &mut host,
    )
    .unwrap_or_else(|| panic!("{}", ctx("bytecode", &"unsupported module")));
    match (&expected, &bc) {
        (Ok(Some(e)), Ok(vs)) if vs.len() == 1 && value_matches(*e, &vs[0]) => {}
        (Ok(None), Ok(vs)) if vs.is_empty() => {}
        (Err(t), Err(tr)) if *tr == interp_trap(*t) => {}
        _ => panic!("{}", ctx("bytecode", &bc)),
    }
    assert!(
        !mutates || bmem == model,
        "{}",
        win_ctx("bytecode", &bmem, model)
    );

    // Cranelift JIT — except the ISSUES.md **I21** carve-out: a *bulk* op whose span
    // overruns `mapped` but stays inside the reservation reaches the libcall, where
    // the trap depends on the libcall touching the overrun (lost entirely for
    // `dst == src`, partial-write-y otherwise). Interp/bytecode above stay fully
    // pinned on these vectors; the JIT leg is skipped until I21 is fixed.
    if expected == Err(SpecTrap::MemoryFault) && !row.has_offset {
        let reserved = 1u64 << svm_ir::DEFAULT_RESERVED_LOG2;
        let in_reserved = |ptr: SpecVal, len: SpecVal| {
            let (p, l) = (to_slot(ptr) as u64, to_slot(len) as u64);
            l == 0 || (l <= reserved && p <= reserved - l)
        };
        let reaches_libcall = match row.id.as_str() {
            "mem.fill" => in_reserved(vector[0], vector[2]),
            _ => in_reserved(vector[0], vector[2]) && in_reserved(vector[1], vector[2]),
        };
        if reaches_libcall {
            return;
        }
    }
    let slots: Vec<i64> = vector.iter().copied().map(to_slot).collect();
    let (out, jmem) = cm
        .run(&slots, Some(init), None, None)
        .unwrap_or_else(|e| panic!("{}", ctx("jit", &e)));
    match (&expected, &out) {
        (Ok(Some(e)), JitOutcome::Returned(rs)) if rs.len() == 1 && slot_matches(*e, rs[0]) => {}
        (Ok(None), JitOutcome::Returned(rs)) if rs.is_empty() => {}
        (Err(t), JitOutcome::Trapped(k)) if *k == jit_trap(*t) => {}
        _ => panic!("{}", ctx("jit", &out)),
    }
    // Window pinned on completing runs; see the module comment for the faulting-run
    // exemption (bulk-libcall partial writes are not yet pinned).
    if mutates && expected.is_ok() {
        assert!(
            jmem[..model.len()] == model[..],
            "{}",
            win_ctx("jit", &jmem[..model.len().min(jmem.len())], model)
        );
    }
}

fn interp_trap(t: SpecTrap) -> Trap {
    match t {
        SpecTrap::DivByZero => Trap::DivByZero,
        SpecTrap::IntOverflow => Trap::IntOverflow,
        SpecTrap::BadConversion => Trap::BadConversion,
        SpecTrap::MemoryFault => Trap::MemoryFault,
    }
}

fn jit_trap(t: SpecTrap) -> TrapKind {
    match t {
        SpecTrap::DivByZero => TrapKind::DivByZero,
        SpecTrap::IntOverflow => TrapKind::IntOverflow,
        SpecTrap::BadConversion => TrapKind::BadConversion,
        SpecTrap::MemoryFault => TrapKind::MemoryFault,
    }
}

#[test]
fn spec_mem_vectors_match_all_backends() {
    let init = init_window();
    // Work units = (row, offset) modules, spread over worker threads: the interpreter
    // legs pay a per-run window setup/teardown (~ms at a 64 KiB window), so the suite
    // is wall-clock-bound on them — the units are independent (each thread owns its
    // module, compiled JIT, and model buffer), and parallelism keeps the FULL
    // boundary lattice within the CI budget instead of thinning it.
    let rows = mem_rows();
    let units: Vec<(usize, u64)> = rows
        .iter()
        .enumerate()
        .flat_map(|(i, row)| {
            let offsets: &[u64] = if row.has_offset { MEM_OFFSETS } else { &[0] };
            offsets.iter().map(move |&o| (i, o)).collect::<Vec<_>>()
        })
        .collect();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let vectors_run = std::sync::atomic::AtomicUsize::new(0);
    let workers = std::thread::available_parallelism().map_or(4, |n| n.get().min(8));
    std::thread::scope(|s| {
        for _ in 0..workers {
            s.spawn(|| {
                // Rows carry non-Sync boxed closures; construction is cheap and
                // deterministic, so each worker builds its own copy and indexes it
                // with the shared unit list.
                let rows = mem_rows();
                let mut model = init.clone();
                loop {
                    let u = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(&(ri, offset)) = units.get(u) else {
                        return;
                    };
                    let row = &rows[ri];
                    let m = module_for_mem(row, offset);
                    svm::verify::verify_module(&m).unwrap_or_else(|e| {
                        panic!("spec mem module for {} fails verify: {e:?}", row.id)
                    });
                    let mut cm = compile(&m, 0).unwrap_or_else(|e| {
                        panic!("spec mem module for {} fails JIT compile: {e:?}", row.id)
                    });
                    for vector in mem_vectors_for(row) {
                        check_vector(row, &m, &mut cm, offset, &vector, &init, &mut model);
                        vectors_run.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                }
            });
        }
    });
    let vectors_run = vectors_run.into_inner();
    assert!(
        vectors_run > 3_000,
        "suspiciously few spec memory vectors ran: {vectors_run}"
    );
}
