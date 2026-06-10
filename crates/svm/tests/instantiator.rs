//! §14 `Instantiator` (iface 6) — VM-in-VM nesting on the interpreter's §12 executor. A parent
//! guest holding an `Instantiator` capability `instantiate`s a child confined to a power-of-two
//! sub-window of its own window, then `join`s it (parking only the calling fiber). The child runs
//! the same module's entry on the same M:N executor, confined by masking to its slice, with an
//! attenuated powerbox (its own `AddressSpace`) and a fuel quota. These tests pin: the child runs
//! and its result returns through `join`; its writes land **only** in its sub-window of the shared
//! parent backing (confinement — the parent sees the superset); an out-of-range carve is rejected;
//! and a child trap propagates to the parent on `join`.
//!
//! The JIT path is deferred (it has no in-process executor to spawn a child vCPU into) — an
//! `instantiate` there resolves to a `CapFault`, like any unsupported capability.

use svm_interp::{run_capture_reserved_with_host, Host, Trap, Value};
use svm_text::parse_module;
use svm_verify::verify_module;

/// A module whose func 0 (parent) instantiates func 1 (child) in a 4 KiB sub-window at `off`,
/// joins it, and returns the child's result. The child stores a marker at its offset 0 and again
/// through a far address that the mask folds back into its 4 KiB window, then returns 42.
fn nest_src(off: u64, size_log2: u64) -> String {
    format!(
        "memory 17\n\
         func (i32) -> (i64) {{\n\
         block0(v0: i32):\n\
         \x20 v1 = i64.const 1\n\
         \x20 v2 = i64.const {off}\n\
         \x20 v3 = i64.const {size_log2}\n\
         \x20 v4 = i64.const 0\n\
         \x20 v5 = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4)\n\
         \x20 v6 = cap.call 6 1 (i32) -> (i64) v0 (v5)\n\
         \x20 return v6\n\
         }}\n\
         func (i64) -> (i64) {{\n\
         block0(v0: i64):\n\
         \x20 v1 = i64.const 0\n\
         \x20 v2 = i32.const 171\n\
         \x20 i32.store8 v1 v2\n\
         \x20 v3 = i64.const 99999\n\
         \x20 v4 = i32.const 200\n\
         \x20 i32.store8 v3 v4\n\
         \x20 v5 = i64.const 42\n\
         \x20 return v5\n\
         }}\n"
    )
}

/// Run `src`'s entry with an `Instantiator` granted over the whole `1<<size_log2`-byte window,
/// seeded with a non-zero pattern, returning the entry result and the final window snapshot.
fn run_nested(src: &str, win_log2: u8) -> (Result<Vec<Value>, Trap>, Vec<u8>) {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    let mut host = Host::new();
    let h = host.grant_instantiator(0, 1u64 << win_log2);
    let init: Vec<u8> = (0..(1u64 << win_log2))
        .map(|i| (i as u8).wrapping_mul(31) ^ 0xa5)
        .collect();
    let mut fuel = 5_000_000u64;
    run_capture_reserved_with_host(&m, 0, &[Value::I32(h)], &mut fuel, &init, 0, &mut host)
}

#[test]
fn instantiate_child_runs_confined_and_joins() {
    const OFF: u64 = 64 << 10; // a 64 KiB-aligned slot in the 128 KiB window
    const SIZE: u64 = 4096; // size_log2 = 12
    let (res, mem) = run_nested(&nest_src(OFF, 12), 17);

    // The child ran on the executor while the parent was parked on `join`, and its result returned.
    assert_eq!(
        res.expect("run ok"),
        vec![Value::I64(42)],
        "child result via join"
    );

    // Its two stores landed in the child's sub-window of the *shared parent backing* — at child
    // offset 0 (→ window OFF) and at the far address folded back in (99999 & 4095 = 1695).
    let init: Vec<u8> = (0..(128u64 << 10))
        .map(|i| (i as u8).wrapping_mul(31) ^ 0xa5)
        .collect();
    assert_eq!(
        mem[OFF as usize], 171,
        "child store at its offset 0 missing"
    );
    assert_eq!(
        mem[(OFF + 1695) as usize],
        200,
        "child far store didn't fold into its window"
    );

    // Confinement: every byte *outside* the child's slice is exactly as the parent seeded it — the
    // child could not reach the parent's other memory.
    for i in 0..(128u64 << 10) {
        if !(OFF..OFF + SIZE).contains(&i) {
            assert_eq!(
                mem[i as usize], init[i as usize],
                "child escaped to parent byte {i}"
            );
        }
    }
}

#[test]
fn instantiate_rejects_out_of_range_carve() {
    // A child window at offset 128 KiB does not fit in the 128 KiB holder → instantiate returns a
    // negative handle; the parent returns it without joining (joining a bad handle would fault).
    let src = "memory 17\n\
         func (i32) -> (i64) {\n\
         block0(v0: i32):\n\
         \x20 v1 = i64.const 1\n\
         \x20 v2 = i64.const 131072\n\
         \x20 v3 = i64.const 12\n\
         \x20 v4 = i64.const 0\n\
         \x20 v5 = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4)\n\
         \x20 v6 = i64.extend_i32_s v5\n\
         \x20 return v6\n\
         }\n\
         func (i64) -> (i64) {\n\
         block0(v0: i64):\n\
         \x20 v1 = i64.const 0\n\
         \x20 return v1\n\
         }\n";
    let (res, _mem) = run_nested(src, 17);
    assert_eq!(
        res.expect("run ok"),
        vec![Value::I64(-22)],
        "an out-of-range carve must return -EINVAL, not a live child"
    );
}

#[test]
fn child_trap_propagates_on_join() {
    // The child traps (`unreachable`); `join` must surface it as the parent's trap, not a value.
    let src = "memory 17\n\
         func (i32) -> (i64) {\n\
         block0(v0: i32):\n\
         \x20 v1 = i64.const 1\n\
         \x20 v2 = i64.const 0\n\
         \x20 v3 = i64.const 12\n\
         \x20 v4 = i64.const 0\n\
         \x20 v5 = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4)\n\
         \x20 v6 = cap.call 6 1 (i32) -> (i64) v0 (v5)\n\
         \x20 return v6\n\
         }\n\
         func (i64) -> (i64) {\n\
         block0(v0: i64):\n\
         \x20 unreachable\n\
         }\n";
    let (res, _mem) = run_nested(src, 17);
    assert!(
        res.is_err(),
        "a child trap must propagate through join, got {res:?}"
    );
}
