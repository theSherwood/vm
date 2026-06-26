//! Phase-3 slice 3.1.1 (DURABILITY.md §12.8): a **durable** run keeps the active shadow-SP word
//! pointing at the *running* context's per-fiber shadow region, swapping it on every fiber switch
//! (D-fiber-cont **option A** — the swap lives in the interpreter's `cont.*` execution, where the
//! resume chain is known, not in emitted IR). This is the invariant the freeze path rests on: a
//! poll that fires while a fiber runs must spill into that fiber's own region, never a sibling's.
//!
//! Without a real freeze driver yet (slices 3.1.3–4), the swap is observed directly: a host-fn
//! capability reads the active shadow-SP each time it is called, and we drive a root that probes,
//! runs two fibers (each probes, then **suspends** so both stay concurrently live in their own
//! slots), and probes again — proving each context sees a distinct region base and control is
//! restored to the root's region. (The fibers suspend rather than return so neither slot is recycled
//! mid-run — §12.8 recycling step 3 reuses a *finished* fiber's slot, which would otherwise route the
//! second fiber into the first's freed region.)

use std::sync::{Arc, Mutex};
use svm_interp::{
    run_capture_reserved_with_host, Host, Value, DURABLE_RESERVE, SHADOW_BASE, SHADOW_SP_OFF,
    SHADOW_STRIDE,
};
use svm_text::parse_module;
use svm_verify::verify_module;

const WINDOW_LOG2: u8 = 17; // 128 KiB ≥ DURABLE_RESERVE (64 KiB)
const WINDOW: usize = 1 << WINDOW_LOG2;

#[test]
fn durable_fiber_switch_routes_shadow_sp_per_context() {
    // func0 (root, v0 = host-fn handle): probe; create+resume fiber A; create+resume fiber B;
    // probe. func1 (fiber): the resume `arg` carries the handle (truncated back to i32); probe.
    // §12.8 4A.5: each probe passes `durable.shadow_base` (the active context's own region base, from
    // the runtime-private register) to the host fn, which records it — directly exercising per-context
    // routing (vs. the legacy single swapped `SHADOW_SP_OFF` word, now retired).
    let src = "memory 17\n\
        func (i32) -> (i64) {\n\
        block0(v0: i32):\n\
        \x20 v1 = durable.shadow_base\n\
        \x20 v2 = cap.call 13 0 (i64) -> (i64) v0 (v1)\n\
        \x20 v3 = ref.func 1\n\
        \x20 v4 = i64.const 4096\n\
        \x20 v5 = cont.new v3 v4\n\
        \x20 v6 = i64.extend_i32_u v0\n\
        \x20 v7, v8 = cont.resume v5 v6\n\
        \x20 v9 = i64.const 8192\n\
        \x20 v10 = cont.new v3 v9\n\
        \x20 v11, v12 = cont.resume v10 v6\n\
        \x20 v13 = durable.shadow_base\n\
        \x20 v14 = cap.call 13 0 (i64) -> (i64) v0 (v13)\n\
        \x20 return v2\n\
        }\n\
        func (i64, i64) -> (i64) {\n\
        block0(v0: i64, v1: i64):\n\
        \x20 v2 = i32.wrap_i64 v1\n\
        \x20 v3 = durable.shadow_base\n\
        \x20 v4 = cap.call 13 0 (i64) -> (i64) v2 (v3)\n\
        \x20 v5 = suspend v4\n\
        \x20 return v5\n\
        }\n";
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");

    // Each host-fn call records the `durable.shadow_base` the running context passed (its own region).
    let probes: Arc<Mutex<Vec<u64>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&probes);
    let mut host = Host::new();
    host.set_durable(true);
    let hf = host.grant_host_fn(Box::new(move |_op, args, _mem| {
        sink.lock().unwrap().push(args[0] as u64);
        Ok(vec![0])
    }));

    // A zeroed window (state = NORMAL); the per-context shadow-base comes from the runtime register,
    // not the window, so no seed is needed.
    let init = vec![0u8; WINDOW];

    let mut fuel = 1_000_000u64;
    let (res, _snap) = run_capture_reserved_with_host(
        &m,
        0,
        &[Value::I32(hf)],
        &mut fuel,
        &init,
        WINDOW_LOG2,
        &mut host,
    );
    assert!(res.is_ok(), "durable fiber run trapped: {res:?}");

    let seen = probes.lock().unwrap().clone();
    assert_eq!(seen.len(), 4, "four probes: root, fiber A, fiber B, root");
    let root = SHADOW_BASE; // context 0
    let a = SHADOW_BASE + SHADOW_STRIDE; // fiber slot 0 → context 1
    let b = SHADOW_BASE + 2 * SHADOW_STRIDE; // fiber slot 1 → context 2
    assert_eq!(seen[0], root, "root runs in context 0's region");
    assert_eq!(seen[1], a, "fiber A unwinds into its own region");
    assert_eq!(seen[2], b, "fiber B unwinds into a distinct region");
    assert_eq!(
        seen[3], root,
        "the swap restored the root's region on return"
    );
    assert!(
        a != root && b != root && a != b,
        "per-context regions are distinct (no collision)"
    );
    assert!(
        b + SHADOW_STRIDE <= DURABLE_RESERVE,
        "every assigned region fits within the durable reserve"
    );
}

#[test]
fn non_durable_fiber_run_leaves_the_reserve_untouched() {
    // The same module run **without** `set_durable` must not touch the shadow-SP word — fibers
    // still work, and a non-durable guest's byte 8 stays whatever it was (here, a sentinel).
    let src = "memory 17\n\
        func (i32) -> (i64) {\n\
        block0(v0: i32):\n\
        \x20 v2 = ref.func 1\n\
        \x20 v3 = i64.const 4096\n\
        \x20 v4 = cont.new v2 v3\n\
        \x20 v5 = i64.extend_i32_u v0\n\
        \x20 v6, v7 = cont.resume v4 v5\n\
        \x20 return v7\n\
        }\n\
        func (i64, i64) -> (i64) {\n\
        block0(v0: i64, v1: i64):\n\
        \x20 v2 = i64.const 42\n\
        \x20 return v2\n\
        }\n";
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");

    const SENTINEL: u64 = 0xDEAD_BEEF_CAFE_F00D;
    let mut init = vec![0u8; WINDOW];
    init[SHADOW_SP_OFF as usize..SHADOW_SP_OFF as usize + 8]
        .copy_from_slice(&SENTINEL.to_le_bytes());

    let mut host = Host::new(); // durable left false
    let mut fuel = 1_000_000u64;
    let (res, snap) = run_capture_reserved_with_host(
        &m,
        0,
        &[Value::I32(0)],
        &mut fuel,
        &init,
        WINDOW_LOG2,
        &mut host,
    );
    assert_eq!(
        res.expect("run ok"),
        vec![Value::I64(42)],
        "fiber returns 42"
    );
    let word = u64::from_le_bytes(
        snap[SHADOW_SP_OFF as usize..SHADOW_SP_OFF as usize + 8]
            .try_into()
            .unwrap(),
    );
    assert_eq!(word, SENTINEL, "a non-durable run never writes the reserve");
}
