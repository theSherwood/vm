//! §14 co-fiber **resume/suspend on the JIT**, differentially vs. the interpreter. A guest holding
//! an `Instantiator` `spawn_coroutine`s (op 2) a child confined to a power-of-two sub-window and
//! drives it with `resume` (op 3). On the JIT the child is a **suspended native continuation** — an
//! `svm-fiber` stack running the child's own compilation over its own guarded window, with its
//! `Yielder` (iface 7) baked as the child's `cap.call` thunk; the parent's window slice and the
//! child's window are synced at every switch (the cooperative equivalent of the interpreter's live
//! shared backing). Both backends must agree on every yielded/returned value, the status sequence,
//! the guest-visible `Yielder` handle, trap propagation, and the final parent window bytes.
//!
//! Gated on `fiber_supported()` (the stack-switch substrate); elsewhere a coroutine op CapFaults.

use svm_interp::{run_capture_reserved_with_host, Host, Trap, Value};
use svm_jit::{compile_and_run_capture_reserved_with_host, JitOutcome};
use svm_text::parse_module;
use svm_verify::verify_module;

const SUSPENDED: i64 = 0;
const RETURNED: i64 = 1;
const FAULTED: i64 = 2;

type BothOut = (Result<Vec<Value>, Trap>, Vec<u8>, JitOutcome, Vec<u8>);

/// Run `src`'s entry (the granted `Instantiator` handle is `v0`) on **both** backends over a
/// fully-mapped, identically seeded `1<<win_log2`-byte window with an `Instantiator` over the whole
/// window; return both results and final windows.
fn both(src: &str, win_log2: u8) -> BothOut {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    let win = 1u64 << win_log2;
    let init: Vec<u8> = (0..win)
        .map(|i| (i as u8).wrapping_mul(31) ^ 0xa5)
        .collect();

    let mut hi = Host::new();
    let ih = hi.grant_instantiator(0, win);
    let mut hj = Host::new();
    let jh = hj.grant_instantiator(0, win);
    assert_eq!(ih, jh, "the Instantiator grant must encode identically");

    let mut fuel = 5_000_000u64;
    let (ir, imem) =
        run_capture_reserved_with_host(&m, 0, &[Value::I32(ih)], &mut fuel, &init, 0, &mut hi);
    let (jo, jmem) = compile_and_run_capture_reserved_with_host(
        &m,
        0,
        &[jh as i64],
        &init,
        0,
        svm_run::cap_thunk,
        &mut hj as *mut Host as *mut core::ffi::c_void,
    )
    .expect("jit");
    (ir, imem, jo, jmem)
}

/// The shared ping-pong program (mirrors `coroutine.rs`): the parent spawns a 64 KiB coroutine at
/// 64 KiB, resumes it three times (delivering 0, 10, 20); the child stores a marker, yields 100,
/// yields `200 + r1`, and returns `999 + r2`. The parent folds the three values + the final status
/// into one `i64`.
fn coro_src() -> &'static str {
    "memory 17
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i64.const 1
  v2 = i64.const 65536
  v3 = i64.const 16
  v4 = i64.const 0
  v5 = cap.call 6 2 (i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4)
  v6 = i64.const 0
  v7, v8 = cap.call 6 3 (i32, i64) -> (i32, i64) v0 (v5, v6)
  v9 = i64.const 10
  v10, v11 = cap.call 6 3 (i32, i64) -> (i32, i64) v0 (v5, v9)
  v12 = i64.const 20
  v13, v14 = cap.call 6 3 (i32, i64) -> (i32, i64) v0 (v5, v12)
  v15 = i64.add v8 v11
  v16 = i64.add v15 v14
  v17 = i64.extend_i32_s v13
  v18 = i64.const 1000000
  v19 = i64.mul v17 v18
  v20 = i64.add v16 v19
  return v20
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i32.wrap_i64 v0
  v2 = i64.const 0
  v3 = i32.const 7
  i32.store8 v2 v3
  v4 = i64.const 100
  v5 = cap.call 7 0 (i64) -> (i64) v1 (v4)
  v6 = i64.const 200
  v7 = i64.add v6 v5
  v8 = cap.call 7 0 (i64) -> (i64) v1 (v7)
  v9 = i64.const 999
  v10 = i64.add v9 v8
  return v10
  }
}
"
}

#[test]
fn jit_coroutine_matches_interp() {
    if !svm_jit::fiber_supported() {
        return; // no stack-switch substrate → no JIT coroutine runtime on this target
    }
    let (ir, imem, jo, jmem) = both(coro_src(), 17);

    // y1=100, y2=200+10, y3=999+20; final status RETURNED — identical on both backends.
    let want = 100 + 210 + 1019 + RETURNED * 1_000_000;
    let ival = ir.expect("interp ran ok").pop().expect("one result");
    assert_eq!(ival, Value::I64(want), "interp round-trip");
    assert!(
        matches!(jo, JitOutcome::Returned(ref s) if s == &[want]),
        "jit: {jo:?}"
    );

    // The escape-oracle across suspensions: byte-identical final parent windows — the child's marker
    // in its slice (synced at switches on the JIT, live-shared on the interp), everything else seeded.
    assert_eq!(imem, jmem, "interp/JIT parent windows diverge");
    const CHILD: u64 = 64 << 10;
    assert_eq!(jmem[CHILD as usize], 7, "child marker missing on the JIT");
    let init: Vec<u8> = (0..(128u64 << 10))
        .map(|i| (i as u8).wrapping_mul(31) ^ 0xa5)
        .collect();
    for i in 0..(128u64 << 10) {
        if !(CHILD..CHILD + (64 << 10)).contains(&i) {
            assert_eq!(
                jmem[i as usize], init[i as usize],
                "JIT coroutine escaped to parent byte {i}"
            );
        }
    }
}

/// The `Yielder` handle is guest-visible data (the child's entry argument), so the two backends must
/// mint the **same value** (the reference `Host`'s first-grant encoding) — pinned by a child that
/// returns its argument without yielding.
#[test]
fn jit_coroutine_yielder_handle_matches_interp() {
    if !svm_jit::fiber_supported() {
        return;
    }
    let src = "memory 17
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i64.const 1
  v2 = i64.const 0
  v3 = i64.const 16
  v4 = i64.const 0
  v5 = cap.call 6 2 (i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4)
  v6 = i64.const 0
  v7, v8 = cap.call 6 3 (i32, i64) -> (i32, i64) v0 (v5, v6)
  return v8
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  return v0
  }
}
";
    let (ir, _imem, jo, _jmem) = both(src, 17);
    let ival = ir.expect("interp ran ok").pop().expect("one result");
    let jval = match jo {
        JitOutcome::Returned(ref s) => s[0],
        ref other => panic!("jit: {other:?}"),
    };
    assert_eq!(
        Value::I64(jval),
        ival,
        "the Yielder handle the child receives must encode identically on both backends"
    );
}

/// §14 **fault-driven yield on the JIT**, differentially: `spawn_demand_coroutine` (op 4) starts the
/// child with its whole window unmapped — on the JIT, real `PROT_NONE`/uncommitted pages whose first
/// touch raises a **hardware fault**; the handler suspends the child's fiber *from the fault
/// handler* to the parent (status FAULTED, value = the fault address in parent-window coordinates).
/// The parent supplies the byte and resumes; the child's faulting load **re-executes** against the
/// freshly supplied page. Both backends must agree on the status sequence, the fault address, the
/// supplied byte, and the final parent window.
#[test]
fn jit_demand_coroutine_matches_interp() {
    if !svm_jit::fiber_supported() {
        return;
    }
    // Mirrors `coroutine.rs`'s demand tests: parent spawns a demand-paged 64 KiB child at 64 KiB;
    // first resume FAULTs at the child's first page; the parent stores 123 at the reported address
    // and resumes; the child's load re-executes and returns the byte (RETURNED).
    let src = "memory 17
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i64.const 1
  v2 = i64.const 65536
  v3 = i64.const 16
  v4 = i64.const 0
  v5 = cap.call 6 4 (i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4)
  v6 = i64.const 0
  v7, v8 = cap.call 6 3 (i32, i64) -> (i32, i64) v0 (v5, v6)
  v9 = i32.const 123
  i32.store8 v8 v9
  v10 = i64.const 0
  v11, v12 = cap.call 6 3 (i32, i64) -> (i32, i64) v0 (v5, v10)
  v13 = i64.extend_i32_s v7
  v14 = i64.const 1000000
  v15 = i64.mul v13 v14
  v16 = i64.extend_i32_s v11
  v17 = i64.const 1000
  v18 = i64.mul v16 v17
  v19 = i64.add v12 v15
  v20 = i64.add v19 v18
  return v20
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 0
  v2 = i32.load8_u v1
  v3 = i64.extend_i32_u v2
  return v3
  }
}
";
    let (ir, imem, jo, jmem) = both(src, 17);
    // status1 = FAULTED, status2 = RETURNED, value2 = the supplied 123 — on both backends.
    let want = 123 + FAULTED * 1_000_000 + RETURNED * 1000;
    let ival = ir.expect("interp ran ok").pop().expect("one result");
    assert_eq!(ival, Value::I64(want), "interp demand round-trip");
    assert!(
        matches!(jo, JitOutcome::Returned(ref s) if s == &[want]),
        "jit: {jo:?}"
    );
    assert_eq!(
        imem, jmem,
        "interp/JIT parent windows diverge after demand paging"
    );
}

/// The fault address handed to the parent is in *parent-window* coordinates on both backends (the
/// child faults at its offset 0 → parent address 64 KiB, its sub-window base).
#[test]
fn jit_demand_coroutine_fault_address_matches_interp() {
    if !svm_jit::fiber_supported() {
        return;
    }
    let src = "memory 17
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i64.const 1
  v2 = i64.const 65536
  v3 = i64.const 16
  v4 = i64.const 0
  v5 = cap.call 6 4 (i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4)
  v6 = i64.const 0
  v7, v8 = cap.call 6 3 (i32, i64) -> (i32, i64) v0 (v5, v6)
  return v8
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 0
  v2 = i32.load8_u v1
  v3 = i64.extend_i32_u v2
  return v3
  }
}
";
    let (ir, _imem, jo, _jmem) = both(src, 17);
    let ival = ir.expect("interp ran ok").pop().expect("one result");
    assert_eq!(ival, Value::I64(65536), "interp fault address");
    assert!(
        matches!(jo, JitOutcome::Returned(ref s) if s == &[65536]),
        "jit fault address: {jo:?}"
    );
}

/// The first resume of a yielding coroutine reports SUSPENDED on both backends, and a coroutine
/// child trap (`unreachable` after a yield) propagates to the parent on both.
#[test]
fn jit_coroutine_suspends_and_propagates_traps() {
    if !svm_jit::fiber_supported() {
        return;
    }
    // (a) first resume → SUSPENDED.
    let src = "memory 17
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i64.const 1
  v2 = i64.const 0
  v3 = i64.const 16
  v4 = i64.const 0
  v5 = cap.call 6 2 (i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4)
  v6 = i64.const 0
  v7, v8 = cap.call 6 3 (i32, i64) -> (i32, i64) v0 (v5, v6)
  v9 = i64.extend_i32_s v7
  return v9
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i32.wrap_i64 v0
  v2 = i64.const 42
  v3 = cap.call 7 0 (i64) -> (i64) v1 (v2)
  return v3
  }
}
";
    let (ir, _i, jo, _j) = both(src, 17);
    let ival = ir.expect("interp ran ok").pop().expect("one result");
    assert_eq!(ival, Value::I64(SUSPENDED), "interp first-resume status");
    assert!(
        matches!(jo, JitOutcome::Returned(ref s) if s == &[SUSPENDED]),
        "jit: {jo:?}"
    );

    // (b) a child trap after a yield propagates at the *second* resume, on both backends.
    let src = "memory 17
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i64.const 1
  v2 = i64.const 0
  v3 = i64.const 16
  v4 = i64.const 0
  v5 = cap.call 6 2 (i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4)
  v6 = i64.const 0
  v7, v8 = cap.call 6 3 (i32, i64) -> (i32, i64) v0 (v5, v6)
  v9, v10 = cap.call 6 3 (i32, i64) -> (i32, i64) v0 (v5, v6)
  return v10
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i32.wrap_i64 v0
  v2 = i64.const 1
  v3 = cap.call 7 0 (i64) -> (i64) v1 (v2)
  unreachable
  }
}
";
    let (ir, _i, jo, _j) = both(src, 17);
    assert!(
        ir.is_err(),
        "interp: child trap should propagate, got {ir:?}"
    );
    assert!(matches!(jo, JitOutcome::Trapped(_)), "jit: {jo:?}");
}
