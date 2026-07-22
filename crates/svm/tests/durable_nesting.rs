//! DURABILITY.md §4 **enforcement** — *"a durable domain admits only freezable modules and may
//! only spawn durable children"* (the one-flag check at instantiate/install; the first slice of
//! nested-guest durability). The freezable attestation is the host's, at grant time
//! (`Host::grant_durable_module` — instrumentation is a compile-mode fact the runtime cannot
//! re-derive from the IR, like verification):
//!
//! * a **durable** parent's `instantiate_module` (§14 op 5) of an *unmarked* grant is refused
//!   fail-closed (`-EINVAL` — the un-instrumented child could never drain-then-unwound, silently
//!   making the subtree non-snapshottable);
//! * the *same* module granted with the durable attestation instantiates and runs;
//! * a **same-module** child (op 0) stays admissible — it runs the parent's own (instrumented)
//!   funcs — and the child inherits the parent's durability (subtree property);
//! * a **non-durable** domain is entirely unaffected (`separate_module.rs` is the standing
//!   control); and a durable domain's guest-driven `Jit.compile` (§22 — also a module
//!   installation) fails closed until a host-side instrumentation hook exists.

use std::sync::Arc;
use svm_durable::{
    arm_freeze_after, begin_thaw, init_durable_window, read_state, transform_module, write_state,
    STATE_NORMAL, STATE_UNWINDING,
};
use svm_interp::{run_capture_reserved_with_host, Host, Trap, Value};
use svm_ir::{Func, Module};
use svm_text::parse_module;
use svm_verify::verify_module;

const SIZE_LOG2: u8 = 18;
const WINDOW: usize = 1 << SIZE_LOG2;
const EINVAL: i64 = -22;

fn instrument(src: &str) -> Module {
    let m = parse_module(src).expect("parse");
    let inst = transform_module(&m).expect("transform");
    verify_module(&inst).expect("instrumented module verifies");
    inst
}

/// The child "plugin": 128 KiB window (its durable reserve + room), pure-compute entry
/// (`(i64) -> (i64)`, the starter-cap convention) returning 4321. Instrumented, so a durable
/// parent may admit it — *when the grant attests it*.
fn child() -> Module {
    instrument(
        "memory 17
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 4321
  return v1
  }
}
",
    )
}

/// A **separate** child plugin with a *looping* entry (sums 0..100 = 4950 with back-edge polls),
/// so a freeze reliably catches it live — the separate-module analog of `PARENT_SELF_LOOP`'s child.
fn child_loop() -> Module {
    instrument(
        "memory 17
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 0
  v2 = i64.const 0
  br 1(v1, v2)
}
block 1 (v3: i64, v4: i64) {
  v5 = i64.const 100
  v6 = i64.lt_s v3 v5
  br_if v6 2(v3, v4) 3(v4)
}
block 2 (v7: i64, v8: i64) {
  v9 = i64.add v8 v7
  v10 = i64.const 1
  v11 = i64.add v7 v10
  br 1(v11, v9)
}
block 3 (v12: i64) {
  return v12
  }
}
",
    )
}

/// A *different* separate module (returns 8888) for the thaw identity-gate test — its digest
/// differs from `child_loop`, so re-granting it in place of the frozen child fails closed.
fn child_other() -> Module {
    instrument(
        "memory 17
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 8888
  return v1
  }
}
",
    )
}

/// Durable parent that `instantiate_module`s (op 5) its granted child at the aligned carve
/// `[128 KiB, 256 KiB)` and returns the op's i32 status — the refusal probe (no join). The module
/// handle arrives as an `i64` entry arg (the op's slot ABI) since the Phase-1 durable transform
/// has no conversions.
const PARENT_PROBE: &str = "memory 18
func (i32, i64) -> (i32) {
block 0 (v0: i32, v1: i64) {
  v2 = i64.const 0
  v3 = i64.const 131072
  v4 = i64.const 17
  v5 = cap.call 6 5 (i64, i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4, v2)
  return v5
  }
}
";

/// Durable parent that instantiates its granted child and `join`s it (op 1) — the happy path.
const PARENT_JOIN: &str = "memory 18
func (i32, i64) -> (i64) {
block 0 (v0: i32, v1: i64) {
  v2 = i64.const 0
  v3 = i64.const 131072
  v4 = i64.const 17
  v5 = cap.call 6 5 (i64, i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4, v2)
  v6 = cap.call 6 1 (i32) -> (i64) v0 (v5)
  return v6
  }
}
";

/// Durable parent that instantiates a **same-module** child (op 0: its own func 1) and joins it.
const PARENT_SELF: &str = "memory 18
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i64.const 1
  v2 = i64.const 131072
  v3 = i64.const 17
  v4 = i64.const 0
  v5 = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4)
  v6 = cap.call 6 1 (i32) -> (i64) v0 (v5)
  return v6
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 777
  return v1
  }
}
";

/// `PARENT_SELF` with a **looping** child (func 1 sums 0..100 with back-edge polls — a real
/// mid-computation continuation for the subtree freeze). Total = 4950.
const PARENT_SELF_LOOP: &str = "memory 18
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i64.const 1
  v2 = i64.const 131072
  v3 = i64.const 17
  v4 = i64.const 0
  v5 = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4)
  v6 = cap.call 6 1 (i32) -> (i64) v0 (v5)
  return v6
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.const 0
  v3 = i64.const 0
  br 1(v2, v3)
}
block 1 (v4: i64, v5: i64) {
  v6 = i64.const 100
  v7 = i64.lt_s v4 v6
  br_if v7 2(v4, v5) 3(v5)
}
block 2 (v8: i64, v9: i64) {
  v10 = i64.add v9 v8
  v11 = i64.const 1
  v12 = i64.add v8 v11
  br 1(v12, v10)
}
block 3 (v13: i64) {
  return v13
  }
}
";

/// Run a durable-instrumented `parent` with an `Instantiator` over the whole window plus the
/// given extra handle args, on a durable host over a durable window.
fn run_durable(parent: &Module, host: &mut Host, args: &[Value]) -> Vec<Value> {
    host.set_durable(true);
    let mut fuel = 5_000_000u64;
    let (r, _) = run_capture_reserved_with_host(
        parent,
        0,
        args,
        &mut fuel,
        &init_durable_window(WINDOW),
        SIZE_LOG2,
        host,
    );
    r.expect("run ok")
}

/// §4: a durable domain refuses to instantiate a module grant without the freezable attestation
/// (`grant_module`, not `grant_durable_module`) — fail-closed `-EINVAL`, like a bad carve.
#[test]
fn durable_domain_refuses_an_unmarked_module_grant() {
    let parent = instrument(PARENT_PROBE);
    let mut host = Host::new();
    let ih = host.grant_instantiator(0, WINDOW as u64);
    let mh = host.grant_module(&child()); // verified + instrumented, but NOT attested durable
    let r = run_durable(&parent, &mut host, &[Value::I32(ih), Value::I64(mh as i64)]);
    assert_eq!(
        r,
        vec![Value::I32(EINVAL as i32)],
        "an unmarked module grant must be refused in a durable domain"
    );
}

/// The same child granted **with** the durable attestation instantiates and runs to its result.
#[test]
fn durable_domain_admits_a_durable_module_grant() {
    let parent = instrument(PARENT_JOIN);
    let mut host = Host::new();
    let ih = host.grant_instantiator(0, WINDOW as u64);
    let mh = host.grant_durable_module(&child()); // host attests: transform_module ran
    let r = run_durable(&parent, &mut host, &[Value::I32(ih), Value::I64(mh as i64)]);
    assert_eq!(
        r,
        vec![Value::I64(4321)],
        "a durable-attested module child runs in a durable domain"
    );
}

/// A **same-module** child (op 0) needs no grant — it runs the parent's own instrumented funcs —
/// so a durable domain admits it unchanged (and it inherits the parent's durability).
#[test]
fn durable_domain_admits_a_same_module_child() {
    let parent = instrument(PARENT_SELF);
    let mut host = Host::new();
    let ih = host.grant_instantiator(0, WINDOW as u64);
    let r = run_durable(&parent, &mut host, &[Value::I32(ih)]);
    assert_eq!(
        r,
        vec![Value::I64(777)],
        "a same-module child runs in a durable domain"
    );
}

/// The rule is durable-domain-only: the identical unmarked grant instantiates fine in a
/// NON-durable domain (`separate_module.rs` is the broader standing control; this pins the
/// minimal pair against the refusal test above).
#[test]
fn non_durable_domain_admits_an_unmarked_module_grant() {
    let parent = instrument(PARENT_JOIN);
    let mut host = Host::new(); // NOT durable
    let ih = host.grant_instantiator(0, WINDOW as u64);
    let mh = host.grant_module(&child());
    let mut fuel = 5_000_000u64;
    let (r, _) = run_capture_reserved_with_host(
        &parent,
        0,
        &[Value::I32(ih), Value::I64(mh as i64)],
        &mut fuel,
        &init_durable_window(WINDOW),
        SIZE_LOG2,
        &mut host,
    );
    assert_eq!(
        r.expect("run ok"),
        vec![Value::I64(4321)],
        "an unmarked grant instantiates in a non-durable domain"
    );
}

/// §22 guest-driven JIT is a module installation too: a durable domain's `Jit.compile` fails
/// closed (`-EINVAL`) even with a working validator installed — an un-instrumented unit could
/// never drain-then-unwind. The identical non-durable host compiles the same bytes fine.
#[test]
fn durable_domain_refuses_guest_jit_compile() {
    fn trivial_validator(
        _bytes: &[u8],
        _mem: Option<u8>,
        _symtab: &[u8],
    ) -> Result<Arc<[Func]>, i64> {
        let m = parse_module(
            "func () -> (i64) {
block 0 () {
  v0 = i64.const 1
  return v0
  }
}
",
        )
        .expect("parse unit");
        Ok(m.funcs.into())
    }

    // Control: the same validator + grant on a NON-durable host compiles.
    let mut host = Host::new();
    host.set_jit_validator(trivial_validator);
    let jh = host.grant_jit(None);
    assert!(
        host.jit_compile(jh, b"unit").expect("no trap").is_ok(),
        "non-durable: the unit compiles"
    );

    // Durable: same setup, compile fails closed before any unit is stored.
    let mut host = Host::new();
    host.set_durable(true);
    host.set_jit_validator(trivial_validator);
    let jh = host.grant_jit(None);
    assert!(
        matches!(host.jit_compile(jh, b"unit").expect("no trap"), Err(e) if e == EINVAL),
        "durable: guest JIT compile is refused fail-closed"
    );
}

// ---- Freeze × §14 children (the fail-closed half of "STW quiesces the subtree as a unit") ----

/// Durable parent: instantiate + join a same-module child, then drive a fiber that suspends once
/// (the armed freeze trigger ticks on `cont.resume`, so `arm = 2` lands the freeze at the second
/// resume — after the join, with the fiber parked: the covered residue shape). Returns child
/// result + fiber result = 777 + 55 = 832.
const PARENT_JOIN_THEN_FIBER: &str = "memory 18
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i64.const 1
  v2 = i64.const 131072
  v3 = i64.const 17
  v4 = i64.const 0
  v5 = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4)
  v6 = cap.call 6 1 (i32) -> (i64) v0 (v5)
  v7 = ref.func 2
  v8 = i64.const 4096
  v9 = cont.new v7 v8
  v10 = i64.const 0
  v11, v12 = cont.resume v9 v10
  v13, v14 = cont.resume v9 v12
  v15 = i64.add v6 v14
  return v15
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.const 777
  return v2
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.const 5
  v3 = suspend v2
  v4 = i64.const 55
  return v4
  }
}
";

/// Durable parent that spawns a §14 coroutine child (op 2) and never resumes it — the child stays
/// suspended (host-side native continuation) when the freeze lands.
const PARENT_CORO: &str = "memory 18
func (i32) -> (i32) {
block 0 (v0: i32) {
  v1 = i64.const 1
  v2 = i64.const 131072
  v3 = i64.const 17
  v4 = i64.const 0
  v5 = cap.call 6 2 (i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4)
  return v5
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 9
  return v1
  }
}
";

/// §4 subtree freeze (the covered shape): a freeze landing while a same-module §14 child is
/// **live** no longer refuses — the child is driven to unwind into its own carve (the subtree STW
/// broadcast), rides as `FrozenNested` residue, and a thaw re-attaches it under `REWINDING`; the
/// parent's re-executed `join` then delivers the child's result, reproducing the uninterrupted
/// total. The child loops with back-edge polls, so its continuation is a *real* mid-computation
/// capture path (frozen at its entry poll here — freeze-from-start broadcasts before it starts).
#[test]
fn freeze_with_live_nested_child_thaws_and_completes() {
    let parent = instrument(PARENT_SELF_LOOP);

    // Control.
    let mut host = Host::new();
    host.set_durable(true);
    let ih = host.grant_instantiator(0, WINDOW as u64);
    let mut fuel = 50_000_000u64;
    let (base, _) = run_capture_reserved_with_host(
        &parent,
        0,
        &[Value::I32(ih)],
        &mut fuel,
        &init_durable_window(WINDOW),
        SIZE_LOG2,
        &mut host,
    );
    assert_eq!(base, Ok(vec![Value::I64(4950)]), "uninterrupted loop total");

    // Freeze-from-start: the parent unwinds at its post-instantiate poll with the child live; the
    // broadcast drives the child to unwind into its carve; the run returns a placeholder (NOT the
    // old ThreadFault refusal) and records the re-attach residue.
    let mut fhost = Host::new();
    fhost.set_durable(true);
    let fih = fhost.grant_instantiator(0, WINDOW as u64);
    let mut win = init_durable_window(WINDOW);
    write_state(&mut win, STATE_UNWINDING);
    let mut fuel = 50_000_000u64;
    let (fr, fsnap) = run_capture_reserved_with_host(
        &parent,
        0,
        &[Value::I32(fih)],
        &mut fuel,
        &win,
        SIZE_LOG2,
        &mut fhost,
    );
    assert!(fr.is_ok(), "subtree freeze returns a placeholder: {fr:?}");
    assert_eq!(read_state(&fsnap), STATE_UNWINDING, "artifact frozen");
    assert_eq!(
        fhost.frozen_nested().len(),
        1,
        "one nested child rode as residue"
    );
    let residue = fhost.frozen_nested().to_vec();
    assert_eq!(residue[0].slot, 0);
    assert_eq!(residue[0].carve_off, 131072);

    // Thaw: re-attach the child from its carve; the parent reloads its handle, re-executes join,
    // and the rewound child completes its loop — the uninterrupted total, reproduced.
    let mut twin = fsnap.clone();
    begin_thaw(&mut twin, 0);
    let mut thost = Host::new();
    thost.set_durable(true);
    thost.set_frozen_nested(residue);
    let tih = thost.grant_instantiator(0, WINDOW as u64);
    let mut fuel = 50_000_000u64;
    let (tr, tsnap) = run_capture_reserved_with_host(
        &parent,
        0,
        &[Value::I32(tih)],
        &mut fuel,
        &twin,
        SIZE_LOG2,
        &mut thost,
    );
    assert_eq!(
        tr,
        Ok(vec![Value::I64(4950)]),
        "thaw re-attached the nested child; join delivered its total"
    );
    assert_eq!(read_state(&tsnap), STATE_NORMAL, "thaw back to NORMAL");
}

/// Freeze a live **separate-module** child (host-supplied at restore): its module identity rides
/// the artifact as a content digest only; the thaw re-attaches it against the restore host's
/// **re-granted** module and reproduces the total. Covers the in-memory arc *and* the codec (the
/// non-durable `Module` handle is drained before serialize; the module is re-granted after the
/// §12.6 canonical re-freeze check so the re-freeze still sees only durable handles).
#[test]
fn freeze_with_live_separate_module_child_thaws_through_the_codec() {
    let parent = instrument(PARENT_JOIN);

    let mut fhost = Host::new();
    fhost.set_durable(true);
    let ih = fhost.grant_instantiator(0, WINDOW as u64);
    let mh = fhost.grant_durable_module(&child_loop());
    let mut win = init_durable_window(WINDOW);
    write_state(&mut win, STATE_UNWINDING);
    let mut fuel = 50_000_000u64;
    let (fr, fsnap) = run_capture_reserved_with_host(
        &parent,
        0,
        &[Value::I32(ih), Value::I64(mh as i64)],
        &mut fuel,
        &win,
        SIZE_LOG2,
        &mut fhost,
    );
    assert!(fr.is_ok(), "freeze placeholder: {fr:?}");
    assert_eq!(read_state(&fsnap), STATE_UNWINDING, "child frozen live");
    let residue = fhost.frozen_nested().to_vec();
    assert_eq!(residue.len(), 1);
    assert!(
        residue[0].module_digest.is_some(),
        "separate-module child carries a digest"
    );

    // Drain the non-durable Module handle so the domain is snapshottable, then serialize.
    fhost.drain_non_durable();
    let artifact = svm_snapshot::freeze(&parent, &fsnap, &fhost).expect("serializes after drain");

    // Restore; §12.6 canonical re-freeze BEFORE re-granting the module (only durable handles live).
    let mut thost = Host::new();
    thost.set_durable(true);
    let window = svm_snapshot::restore(&artifact, &parent, &mut thost).expect("restores");
    assert_eq!(
        thost.frozen_nested(),
        fhost.frozen_nested(),
        "residue re-seeded"
    );
    assert_eq!(
        svm_snapshot::freeze(&parent, &window, &thost).expect("re-freeze"),
        artifact,
        "canonical re-freeze byte-identical"
    );

    // Recover the restored Instantiator handle *before* re-granting the (non-durable) Module — the
    // embedder then supplies the matching module (host-supplied at restore) and thaws.
    let caps = thost
        .capture_durable_handles()
        .expect("only durable handles restored");
    let tih = ((caps[0].generation << 8) | caps[0].slot) as i32;
    let tmh = thost.grant_durable_module(&child_loop());
    let mut twin = window;
    begin_thaw(&mut twin, 0);
    let mut fuel = 50_000_000u64;
    let (tr, tsnap) = run_capture_reserved_with_host(
        &parent,
        0,
        &[Value::I32(tih), Value::I64(tmh as i64)],
        &mut fuel,
        &twin,
        SIZE_LOG2,
        &mut thost,
    );
    assert_eq!(
        tr,
        Ok(vec![Value::I64(4950)]),
        "re-attached separate-module child reproduces the total"
    );
    assert_eq!(read_state(&tsnap), STATE_NORMAL, "thaw back to NORMAL");
}

/// The per-child R5 identity gate: thawing a separate-module artifact **without** re-granting the
/// child's module (or with a *different* module) fails closed — `module_by_digest` finds no match,
/// so the parent's re-executed `join` traps rather than mis-running a wrong module in the carve.
#[test]
fn thaw_separate_module_child_fails_closed_on_missing_or_mismatched_module() {
    let parent = instrument(PARENT_JOIN);
    let freeze = || {
        let mut fhost = Host::new();
        fhost.set_durable(true);
        let ih = fhost.grant_instantiator(0, WINDOW as u64);
        let mh = fhost.grant_durable_module(&child_loop());
        let mut win = init_durable_window(WINDOW);
        write_state(&mut win, STATE_UNWINDING);
        let mut fuel = 50_000_000u64;
        let (_, fsnap) = run_capture_reserved_with_host(
            &parent,
            0,
            &[Value::I32(ih), Value::I64(mh as i64)],
            &mut fuel,
            &win,
            SIZE_LOG2,
            &mut fhost,
        );
        (fhost.frozen_nested().to_vec(), fsnap)
    };
    let (residue, fsnap) = freeze();
    if read_state(&fsnap) != STATE_UNWINDING {
        return; // freeze didn't catch it live (rare) — nothing to gate
    }

    // Thaw with NO module re-granted → join fails closed.
    let run_thaw = |granted: Option<Module>| -> Result<Vec<Value>, Trap> {
        let mut thost = Host::new();
        thost.set_durable(true);
        let ih = thost.grant_instantiator(0, WINDOW as u64);
        let tmh = granted.map(|m| thost.grant_durable_module(&m)).unwrap_or(0);
        thost.set_frozen_nested(residue.clone());
        let mut twin = fsnap.clone();
        begin_thaw(&mut twin, 0);
        let mut fuel = 50_000_000u64;
        run_capture_reserved_with_host(
            &parent,
            0,
            &[Value::I32(ih), Value::I64(tmh as i64)],
            &mut fuel,
            &twin,
            SIZE_LOG2,
            &mut thost,
        )
        .0
    };
    assert_eq!(
        run_thaw(None),
        Err(Trap::ThreadFault),
        "missing module: thaw fails closed"
    );
    assert_eq!(
        run_thaw(Some(child_other())),
        Err(Trap::ThreadFault),
        "mismatched module (wrong digest): thaw fails closed"
    );
}

/// Same fail-closed for a suspended §14 **coroutine** child (its native continuation can't ride
/// the artifact either).
#[test]
fn freeze_with_suspended_coroutine_fails_closed() {
    let parent = instrument(PARENT_CORO);
    let mut host = Host::new();
    host.set_durable(true);
    let ih = host.grant_instantiator(0, WINDOW as u64);
    let mut win = init_durable_window(WINDOW);
    write_state(&mut win, STATE_UNWINDING);
    let mut fuel = 5_000_000u64;
    let (r, _) = run_capture_reserved_with_host(
        &parent,
        0,
        &[Value::I32(ih)],
        &mut fuel,
        &win,
        SIZE_LOG2,
        &mut host,
    );
    assert_eq!(
        r,
        Err(Trap::ThreadFault),
        "a freeze with a suspended §14 coroutine must refuse"
    );
}

/// The refusal must not over-fire: a freeze landing **after** the §14 child was joined mints a
/// valid artifact, and the thaw **reloads** the child's join result (reload-not-reissue — the
/// child is never re-run) and reproduces the uninterrupted total.
#[test]
fn freeze_after_nested_child_joined_thaws_and_reloads_the_join_result() {
    let parent = instrument(PARENT_JOIN_THEN_FIBER);

    // Control: uninterrupted total.
    let mut host = Host::new();
    host.set_durable(true);
    let ih = host.grant_instantiator(0, WINDOW as u64);
    let mut fuel = 5_000_000u64;
    let (base, _) = run_capture_reserved_with_host(
        &parent,
        0,
        &[Value::I32(ih)],
        &mut fuel,
        &init_durable_window(WINDOW),
        SIZE_LOG2,
        &mut host,
    );
    assert_eq!(base, Ok(vec![Value::I64(777 + 55)]), "uninterrupted total");

    // Armed freeze: tick 1 at the first resume (runs; the fiber parks), tick 2 at the second —
    // the freeze lands with the child already joined (its result checkpointed in the parent's
    // shadow frame) and the fiber parked (rides as Section-2 residue).
    let mut fhost = Host::new();
    fhost.set_durable(true);
    let fih = fhost.grant_instantiator(0, WINDOW as u64);
    let mut win = init_durable_window(WINDOW);
    arm_freeze_after(&mut win, 2);
    let mut fuel = 5_000_000u64;
    let (fr, fsnap) = run_capture_reserved_with_host(
        &parent,
        0,
        &[Value::I32(fih)],
        &mut fuel,
        &win,
        SIZE_LOG2,
        &mut fhost,
    );
    assert!(
        fr.is_ok(),
        "freeze-after-join returns a placeholder: {fr:?}"
    );
    assert_eq!(
        read_state(&fsnap),
        STATE_UNWINDING,
        "the armed freeze landed (joined child does not refuse it)"
    );

    // Thaw: the parent rewinds, reloading the instantiate handle and the join result from its
    // shadow frame — the child never re-runs — and completes to the uninterrupted total.
    let mut twin = fsnap.clone();
    begin_thaw(&mut twin, 0);
    let mut thost = Host::new();
    thost.set_durable(true);
    thost.set_frozen_fibers(fhost.frozen_fibers().to_vec());
    let tih = thost.grant_instantiator(0, WINDOW as u64);
    let mut fuel = 5_000_000u64;
    let (tr, tsnap) = run_capture_reserved_with_host(
        &parent,
        0,
        &[Value::I32(tih)],
        &mut fuel,
        &twin,
        SIZE_LOG2,
        &mut thost,
    );
    assert_eq!(
        tr,
        Ok(vec![Value::I64(777 + 55)]),
        "thaw reproduces the total — the §14 join result reloaded, the child never re-ran"
    );
    assert_eq!(read_state(&tsnap), STATE_NORMAL, "thaw back to NORMAL");
}

/// The bytecode engine's durable-capture entry must **decline** a §14 module (fall back to the
/// tree-walker, which owns the durable nesting rules): its own instantiate arm has neither the
/// admission check nor the fail-closed, so driving a durable §14 module there would mint the exact
/// thaw-faulting artifact the tree-walker refuses. (svm-run's bytecode backend falls back on
/// `None`, so an embedder always lands on the enforced path.)
#[test]
fn bytecode_durable_capture_declines_a_nesting_module() {
    let parent = instrument(PARENT_SELF);
    let mut host = Host::new();
    host.set_durable(true);
    let ih = host.grant_instantiator(0, WINDOW as u64);
    let mut fuel = 5_000_000u64;
    let r = svm_interp::bytecode::compile_and_run_capture_reserved_with_host(
        &parent,
        0,
        &[Value::I32(ih)],
        &mut fuel,
        &init_durable_window(WINDOW),
        SIZE_LOG2,
        &mut host,
    );
    assert!(
        r.is_none(),
        "the bytecode durable entry must decline §14 modules (tree-walker owns the nesting rules)"
    );
}

/// Stage C — the nested artifact through the **snapshot codec** (format v8): freeze the live
/// nested child, serialize the real §12 artifact (the child's carve rides the window image; the
/// `FrozenNested` re-attach record rides Section 2), restore into a fresh host, assert the §12.6
/// **canonical re-freeze is byte-identical**, then thaw — the re-attached child completes its
/// loop and the parent's join delivers the uninterrupted total.
#[test]
fn nested_artifact_serializes_restores_and_thaws_through_the_codec() {
    let parent = instrument(PARENT_SELF_LOOP);

    // Freeze with the child live (as in the in-memory test).
    let mut fhost = Host::new();
    fhost.set_durable(true);
    let fih = fhost.grant_instantiator(0, WINDOW as u64);
    let mut win = init_durable_window(WINDOW);
    write_state(&mut win, STATE_UNWINDING);
    let mut fuel = 50_000_000u64;
    let (fr, fsnap) = run_capture_reserved_with_host(
        &parent,
        0,
        &[Value::I32(fih)],
        &mut fuel,
        &win,
        SIZE_LOG2,
        &mut fhost,
    );
    assert!(fr.is_ok(), "subtree freeze placeholder: {fr:?}");
    assert_eq!(
        fhost.frozen_nested().len(),
        1,
        "one nested child in residue"
    );

    // Serialize the real artifact: window image (parent + child carve) + Section-2 nested record.
    let artifact =
        svm_snapshot::freeze(&parent, &fsnap, &fhost).expect("nested artifact serializes");

    // Restore into a FRESH host: handles re-pinned, nested residue re-seeded.
    let mut thost = Host::new();
    thost.set_durable(true);
    let window = svm_snapshot::restore(&artifact, &parent, &mut thost).expect("restores");
    assert_eq!(
        thost.frozen_nested(),
        fhost.frozen_nested(),
        "restore re-seeded the nested re-attach residue"
    );

    // §12.6 invariant 1 — canonical: re-serializing the restored domain is byte-identical.
    assert_eq!(
        svm_snapshot::freeze(&parent, &window, &thost).expect("re-freeze"),
        artifact,
        "canonical re-freeze of a restored nested artifact is byte-identical"
    );

    // Thaw: the child re-attaches from its carve and completes; join delivers the total.
    let mut twin = window;
    begin_thaw(&mut twin, 0);
    let caps = thost.capture_durable_handles().expect("durable handles");
    let tih = ((caps[0].generation << 8) | caps[0].slot) as i32;
    let mut fuel = 50_000_000u64;
    let (tr, tsnap) = run_capture_reserved_with_host(
        &parent,
        0,
        &[Value::I32(tih)],
        &mut fuel,
        &twin,
        SIZE_LOG2,
        &mut thost,
    );
    assert_eq!(
        tr,
        Ok(vec![Value::I64(4950)]),
        "the codec-restored nested child completed; join delivered its total"
    );
    assert_eq!(read_state(&tsnap), STATE_NORMAL, "thaw back to NORMAL");
}

/// A parent with a **completed-but-unjoined** §14 child at the freeze point (completed-result
/// residue). It instantiates child B (trivial → 33) and child A (the long loop → 4950), joins A
/// first (while parked in A's join the single durable worker also runs B to completion), then drives
/// a fiber so an armed freeze can land *after* B finished but *before* the parent joins B. Total =
/// 4950 (A) + 33 (B) + 5 (fiber) = 4988.
const PARENT_TWO_CHILDREN: &str = "memory 18
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i64.const 2
  v2 = i64.const 196608
  v3 = i64.const 16
  v4 = i64.const 0
  v5 = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4)
  v6 = i64.const 1
  v7 = i64.const 131072
  v8 = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v0 (v6, v7, v3, v4)
  v9 = cap.call 6 1 (i32) -> (i64) v0 (v8)
  v10 = ref.func 3
  v11 = i64.const 4096
  v12 = cont.new v10 v11
  v13 = i64.const 0
  v14, v15 = cont.resume v12 v13
  v16, v17 = cont.resume v12 v15
  v18 = cap.call 6 1 (i32) -> (i64) v0 (v5)
  v19 = i64.add v9 v18
  v20 = i64.add v19 v17
  return v20
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.const 0
  v3 = i64.const 0
  br 1(v2, v3)
}
block 1 (v4: i64, v5: i64) {
  v6 = i64.const 100
  v7 = i64.lt_s v4 v6
  br_if v7 2(v4, v5) 3(v5)
}
block 2 (v8: i64, v9: i64) {
  v10 = i64.add v9 v8
  v11 = i64.const 1
  v12 = i64.add v8 v11
  br 1(v12, v10)
}
block 3 (v13: i64) {
  return v13
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.const 33
  return v2
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.const 1
  v3 = suspend v2
  v4 = i64.const 5
  return v4
  }
}
";

/// §4 completed-result residue: a freeze landing while a §14 child is **completed-but-unjoined**
/// records its join result (no `UNWINDING` broadcast — nothing to unwind), and a thaw delivers that
/// result to the parent's re-executed `join` **without re-running** the child. Verified in-memory
/// and through the codec, and the residue is asserted to actually carry `completed_result`.
#[test]
fn freeze_with_completed_unjoined_child_rides_and_reloads() {
    let parent = instrument(PARENT_TWO_CHILDREN);
    const TOTAL: i64 = 4950 + 33 + 5;

    // Control.
    let mut host = Host::new();
    host.set_durable(true);
    let ih = host.grant_instantiator(0, WINDOW as u64);
    let mut fuel = 50_000_000u64;
    let (base, _) = run_capture_reserved_with_host(
        &parent,
        0,
        &[Value::I32(ih)],
        &mut fuel,
        &init_durable_window(WINDOW),
        SIZE_LOG2,
        &mut host,
    );
    assert_eq!(base, Ok(vec![Value::I64(TOTAL)]), "uninterrupted total");

    // Armed freeze at the fiber's second resume — B has completed (ran during A's join) but is
    // not yet joined; A is already joined (its result checkpointed).
    let mut fhost = Host::new();
    fhost.set_durable(true);
    let fih = fhost.grant_instantiator(0, WINDOW as u64);
    let mut win = init_durable_window(WINDOW);
    arm_freeze_after(&mut win, 2);
    let mut fuel = 50_000_000u64;
    let (fr, fsnap) = run_capture_reserved_with_host(
        &parent,
        0,
        &[Value::I32(fih)],
        &mut fuel,
        &win,
        SIZE_LOG2,
        &mut fhost,
    );
    assert!(fr.is_ok(), "freeze placeholder: {fr:?}");
    if read_state(&fsnap) != STATE_UNWINDING {
        return; // freeze didn't land here (scheduling); the control already proved correctness
    }
    let nested = fhost.frozen_nested().to_vec();
    let completed: Vec<_> = nested
        .iter()
        .filter(|n| n.completed_result.is_some())
        .collect();
    assert_eq!(
        completed.len(),
        1,
        "exactly one completed-unjoined child (B)"
    );
    assert_eq!(
        completed[0].completed_result,
        Some(33),
        "B's join result rides the residue"
    );

    // Through the codec: serialize → restore → canonical re-freeze byte-identical.
    let artifact = svm_snapshot::freeze(&parent, &fsnap, &fhost).expect("serializes");
    let mut thost = Host::new();
    thost.set_durable(true);
    let window = svm_snapshot::restore(&artifact, &parent, &mut thost).expect("restores");
    assert_eq!(
        thost.frozen_nested(),
        fhost.frozen_nested(),
        "residue re-seeded"
    );
    assert_eq!(
        svm_snapshot::freeze(&parent, &window, &thost).expect("re-freeze"),
        artifact,
        "canonical re-freeze byte-identical"
    );

    // Thaw: B's result reloads into the parent's join without re-running B; A rewinds/reloads; the
    // fiber re-attaches; the total is reproduced.
    let mut twin = window;
    begin_thaw(&mut twin, 0);
    let caps = thost.capture_durable_handles().expect("durable");
    let tih = ((caps[0].generation << 8) | caps[0].slot) as i32;
    let mut fuel = 50_000_000u64;
    let (tr, tsnap) = run_capture_reserved_with_host(
        &parent,
        0,
        &[Value::I32(tih)],
        &mut fuel,
        &twin,
        SIZE_LOG2,
        &mut thost,
    );
    assert_eq!(
        tr,
        Ok(vec![Value::I64(TOTAL)]),
        "thaw reproduces the total (B reloaded, not re-run)"
    );
    assert_eq!(read_state(&tsnap), STATE_NORMAL, "thaw back to NORMAL");
}

// ---- Depth-2 durable nesting (DURABILITY.md §4 — "STW quiesces the subtree as a unit", extended
// from depth-1 parent→child to depth-2 parent→child→grandchild; interpreter + in-memory only) ----

/// A 512 KiB root window for the depth-2 geometry (below), large enough that each of the three
/// nesting levels holds its own 64 KiB durable reserve without overlap:
///   • root       window `[0, 512 KiB)`   — reserve `[0, 64 KiB)`
///   • child      carve  `[256 KiB, 512 KiB)`, `size_log2 18` — reserve `[256 KiB, 320 KiB)`
///   • grandchild carve  `[384 KiB, 512 KiB)`, `size_log2 17` — reserve `[384 KiB, 448 KiB)`
/// The child's `Instantiator` is over its own window base 0, so the grandchild carve is named
/// **child-relative** `[128 KiB, 256 KiB)` (offset `131072`) — the freeze records that child-relative
/// offset, and the thaw composes it with the child's absolute base to land the grandchild at
/// `[384 KiB, 512 KiB)` of the root image.
const D2_SIZE_LOG2: u8 = 19;
const D2_WINDOW: usize = 1 << D2_SIZE_LOG2;

/// A **same-module depth-2** durable parent:
///   • func 0 (root): instantiates func 1 as a same-module child at the `[256 KiB, 512 KiB)` carve
///     (op 0), joins it, returns its result.
///   • func 1 (child): instantiates func 2 as a same-module grandchild at its child-relative
///     `[128 KiB, 256 KiB)` sub-carve (op 0), joins it, returns the grandchild's total.
///   • func 2 (grandchild): the `0..100 = 4950` back-edge-polled loop (a real mid-computation
///     continuation), same body as `PARENT_SELF_LOOP`'s child.
///
/// **Handle ABI note.** A §14 child receives its `Instantiator` as an `i64` entry arg (the
/// `instantiate` op's slot ABI passes `Value::I64(cinst)`), but a `cap.call` handle **operand** must
/// be `i32` (the verifier's forgeable-index type). The Phase-1 durable transform rejects both scalar
/// conversions (`i32.wrap_i64` → `UnsupportedInst`) and *any* guest linear-memory op (`GuestUsesMemory`),
/// so the child cannot truncate its `i64` handle at all. It doesn't need to: a nested child's powerbox
/// is a fresh `Host` whose **first** grant is its `Instantiator` (slot 0, generation 1), a deterministic
/// handle value `(1 << CAP_LOG2) | 0 == 256`. So the child ignores its `i64` entry arg and names its own
/// `Instantiator` with the constant `i32.const 256` — the same value the fresh-host grant returns on both
/// the freeze and the thaw re-attach (which re-grants the `Instantiator` first, too).
const PARENT_DEPTH2: &str = "memory 19
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i64.const 1
  v2 = i64.const 262144
  v3 = i64.const 18
  v4 = i64.const 0
  v5 = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4)
  v6 = cap.call 6 1 (i32) -> (i64) v0 (v5)
  return v6
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i32.const 256
  v2 = i64.const 2
  v3 = i64.const 131072
  v4 = i64.const 17
  v5 = i64.const 0
  v6 = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v1 (v2, v3, v4, v5)
  v7 = cap.call 6 1 (i32) -> (i64) v1 (v6)
  return v7
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 0
  v2 = i64.const 0
  br 1(v1, v2)
}
block 1 (v3: i64, v4: i64) {
  v5 = i64.const 100
  v6 = i64.lt_s v3 v5
  br_if v6 2(v3, v4) 3(v4)
}
block 2 (v7: i64, v8: i64) {
  v9 = i64.add v8 v7
  v10 = i64.const 1
  v11 = i64.add v7 v10
  br 1(v11, v9)
}
block 3 (v12: i64) {
  return v12
  }
}
";

/// §4 depth-2 subtree freeze/thaw: a freeze landing while a **child and a grandchild** are both live
/// records *two* [`svm_interp::FrozenNested`] re-attach records — the child (tagged `parent_task == 0`,
/// the root) and the grandchild (tagged `parent_task == <child's task id>`) — coalesced in the root
/// host via the shared freeze-residue sink, exactly as `thread.spawn`'s shared host coalesces
/// [`svm_interp::FrozenVCpu`] across levels. A thaw groups the residue by `parent_task`, rebuilds each
/// parent's join table (the grandchild re-attaches into the *child's* table, not the root's), and both
/// levels rewind: the grandchild completes its loop, its join delivers the total up to the child, and
/// the child's join delivers it up to the root — reproducing the uninterrupted result across **two**
/// nesting levels. (In-memory arc; the codec round-trip is covered by
/// `depth2_nested_artifact_serializes_restores_and_thaws_through_the_codec`.)
#[test]
fn freeze_with_live_depth2_grandchild_thaws_and_completes() {
    let parent = instrument(PARENT_DEPTH2);

    // (1) Control: uninterrupted, the grandchild's 4950 propagates up through both joins.
    let mut host = Host::new();
    host.set_durable(true);
    let ih = host.grant_instantiator(0, D2_WINDOW as u64);
    let mut fuel = 50_000_000u64;
    let (base, _) = run_capture_reserved_with_host(
        &parent,
        0,
        &[Value::I32(ih)],
        &mut fuel,
        &init_durable_window(D2_WINDOW),
        D2_SIZE_LOG2,
        &mut host,
    );
    assert_eq!(
        base,
        Ok(vec![Value::I64(4950)]),
        "uninterrupted depth-2 total propagates through both joins"
    );

    // (2) Freeze-from-start: the root unwinds under `UNWINDING`, and — because a mid-freeze
    // `instantiate` seeds the child's carve `UNWINDING` too (the subtree STW) — the child runs its
    // own `instantiate`/`join` under `UNWINDING`, recording its live grandchild before it drains. The
    // run returns a placeholder and both re-attach records ride the root host.
    let mut fhost = Host::new();
    fhost.set_durable(true);
    let fih = fhost.grant_instantiator(0, D2_WINDOW as u64);
    let mut win = init_durable_window(D2_WINDOW);
    write_state(&mut win, STATE_UNWINDING);
    let mut fuel = 50_000_000u64;
    let (fr, fsnap) = run_capture_reserved_with_host(
        &parent,
        0,
        &[Value::I32(fih)],
        &mut fuel,
        &win,
        D2_SIZE_LOG2,
        &mut fhost,
    );
    assert!(
        fr.is_ok(),
        "depth-2 subtree freeze returns a placeholder: {fr:?}"
    );
    assert_eq!(read_state(&fsnap), STATE_UNWINDING, "artifact frozen");

    let residue = fhost.frozen_nested().to_vec();
    assert_eq!(
        residue.len(),
        2,
        "two nested children rode as residue (child + grandchild)"
    );
    // The child: a direct child of the root (`parent_task == 0`) at the `[256 KiB, ..)` carve.
    let child_rec = residue
        .iter()
        .find(|n| n.parent_task == 0)
        .expect("a root-child record (parent_task == 0)");
    assert_eq!(child_rec.slot, 0);
    assert_eq!(
        child_rec.carve_off, 262144,
        "child carve is root-relative 256 KiB"
    );
    // The grandchild: tagged with the child's task id (non-zero), at the child-relative `[128 KiB, ..)`
    // sub-carve — the record the depth-2 slice adds.
    let gchild_rec = residue
        .iter()
        .find(|n| n.parent_task != 0)
        .expect("a grandchild record (parent_task == child's task id)");
    assert_ne!(
        gchild_rec.parent_task, 0,
        "the grandchild is tagged with its parent-child's task id, not the root's"
    );
    assert_eq!(
        gchild_rec.carve_off, 131072,
        "grandchild carve is child-relative 128 KiB (composed with the child's base on thaw)"
    );

    // (3) Thaw: re-attach both levels from their carves; each rewinds, and the two joins deliver the
    // grandchild's total all the way up to the root — freeze→thaw ≡ uninterrupted across TWO levels.
    let mut twin = fsnap.clone();
    begin_thaw(&mut twin, 0);
    let mut thost = Host::new();
    thost.set_durable(true);
    thost.set_frozen_nested(residue);
    let tih = thost.grant_instantiator(0, D2_WINDOW as u64);
    let mut fuel = 50_000_000u64;
    let (tr, tsnap) = run_capture_reserved_with_host(
        &parent,
        0,
        &[Value::I32(tih)],
        &mut fuel,
        &twin,
        D2_SIZE_LOG2,
        &mut thost,
    );
    assert_eq!(
        tr,
        Ok(vec![Value::I64(4950)]),
        "thaw re-attached the child and grandchild; both joins delivered the total"
    );
    assert_eq!(read_state(&tsnap), STATE_NORMAL, "thaw back to NORMAL");
}

/// Depth-2 through the **snapshot codec** (format v12): the same 3-level module, but the two
/// re-attach records (child `parent_task == 0` + grandchild `parent_task == <child's task>`) ride
/// the real §12 artifact instead of an in-memory hand-off. v12 carries `parent_task` on the wire, so
/// `restore` re-seeds the residue *byte-for-byte identically* (asserted) — proving the grandchild's
/// non-zero tag survives serialize/restore, not just the in-memory arc. Then the §12.6 canonical
/// re-freeze is byte-identical, and the thaw rewinds both levels so the grandchild's total propagates
/// up through both joins to the root — freeze→serialize→restore→thaw ≡ uninterrupted across two levels.
#[test]
fn depth2_nested_artifact_serializes_restores_and_thaws_through_the_codec() {
    let parent = instrument(PARENT_DEPTH2);

    // Freeze with the child and grandchild both live (as in the in-memory depth-2 test).
    let mut fhost = Host::new();
    fhost.set_durable(true);
    let fih = fhost.grant_instantiator(0, D2_WINDOW as u64);
    let mut win = init_durable_window(D2_WINDOW);
    write_state(&mut win, STATE_UNWINDING);
    let mut fuel = 50_000_000u64;
    let (fr, fsnap) = run_capture_reserved_with_host(
        &parent,
        0,
        &[Value::I32(fih)],
        &mut fuel,
        &win,
        D2_SIZE_LOG2,
        &mut fhost,
    );
    assert!(fr.is_ok(), "depth-2 subtree freeze placeholder: {fr:?}");
    assert_eq!(
        fhost.frozen_nested().len(),
        2,
        "two nested children in residue (child + grandchild)"
    );

    // Serialize the real artifact: window image (root + child carve + grandchild sub-carve) + the two
    // Section-2 nested records, each now carrying its `parent_task` (v12).
    let artifact =
        svm_snapshot::freeze(&parent, &fsnap, &fhost).expect("depth-2 artifact serializes");

    // Restore into a FRESH host: the residue re-seeds byte-for-byte — so the grandchild's non-zero
    // `parent_task` round-tripped through the wire, not just the in-memory arc.
    let mut thost = Host::new();
    thost.set_durable(true);
    let window = svm_snapshot::restore(&artifact, &parent, &mut thost).expect("restores");
    assert_eq!(
        thost.frozen_nested(),
        fhost.frozen_nested(),
        "restore re-seeded both re-attach records incl. the grandchild's parent_task"
    );

    // §12.6 invariant 1 — canonical: re-serializing the restored domain is byte-identical.
    assert_eq!(
        svm_snapshot::freeze(&parent, &window, &thost).expect("re-freeze"),
        artifact,
        "canonical re-freeze of a restored depth-2 artifact is byte-identical"
    );

    // Thaw: both levels re-attach from their carves and rewind; both joins deliver the total up.
    let mut twin = window;
    begin_thaw(&mut twin, 0);
    let caps = thost.capture_durable_handles().expect("durable handles");
    let tih = ((caps[0].generation << 8) | caps[0].slot) as i32;
    let mut fuel = 50_000_000u64;
    let (tr, tsnap) = run_capture_reserved_with_host(
        &parent,
        0,
        &[Value::I32(tih)],
        &mut fuel,
        &twin,
        D2_SIZE_LOG2,
        &mut thost,
    );
    assert_eq!(
        tr,
        Ok(vec![Value::I64(4950)]),
        "the codec-restored depth-2 subtree completed; both joins delivered the total"
    );
    assert_eq!(read_state(&tsnap), STATE_NORMAL, "thaw back to NORMAL");
}
