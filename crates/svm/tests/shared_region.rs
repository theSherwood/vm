//! §13 SharedRegion — end-to-end through the real `cap.call` dispatch (iface 4): a host-granted
//! shared region mapped into the window at *two* offsets aliases the same bytes, so a store at one
//! offset is visible at a load from the other. This is the interpreter reference slice (the JIT side
//! — real shared mappings in `MprotectWindow` — and the interp↔JIT differential are the next §13
//! increment). The guest queries `page_size` (op 3) and works in whole pages, so it is host-agnostic
//! (4 KiB / 16 KiB alike).

use svm_interp::{run_capture_reserved_with_host, Host, Value};
use svm_text::parse_module;
use svm_verify::verify_module;

const MARKER: i64 = 0x0123_4567_89ab_cdef;

/// Map the region at window offset 0 and again at window offset `page_size`, store `MARKER` at 0,
/// then return the i64 loaded from `page_size` — which must read back `MARKER` *iff* the two
/// mappings alias the same backing.
fn alias_probe_src() -> String {
    format!(
        "memory 16\n\
         func (i32) -> (i64) {{\n\
         block0(v0: i32):\n\
         \x20 v1 = cap.call 4 3 () -> (i64) v0 ()\n\
         \x20 v2 = i64.const 0\n\
         \x20 v3 = i32.const 3\n\
         \x20 v4 = cap.call 4 0 (i64, i64, i64, i32) -> (i64) v0 (v2, v2, v1, v3)\n\
         \x20 v5 = cap.call 4 0 (i64, i64, i64, i32) -> (i64) v0 (v1, v2, v1, v3)\n\
         \x20 v6 = i64.const {MARKER}\n\
         \x20 i64.store v2 v6\n\
         \x20 v7 = i64.load v1\n\
         \x20 return v7\n\
         }}\n"
    )
}

#[test]
fn shared_region_two_offsets_alias_through_cap_call() {
    let src = alias_probe_src();
    let m = parse_module(&src).expect("parse");
    verify_module(&m).expect("verify");

    let mut host = Host::new();
    // A region comfortably larger than any host page (covers 16 KiB pages too).
    let h = host.grant_shared_region(1 << 16);
    let mut fuel = 1_000_000u64;
    let (res, _snap) =
        run_capture_reserved_with_host(&m, 0, &[Value::I32(h)], &mut fuel, &[], 0, &mut host);

    assert_eq!(
        res.expect("run ok"),
        vec![Value::I64(MARKER)],
        "the second mapping must alias the first: a store at offset 0 must be visible at page_size"
    );
}

#[test]
fn shared_region_without_second_mapping_is_not_aliased() {
    // Control: map the region only at offset 0, store MARKER there, and read at page_size — which is
    // an ordinary (unmapped/zero) window page, *not* aliased. Proves the positive test is non-vacuous.
    let src = format!(
        "memory 16\n\
         func (i32) -> (i64) {{\n\
         block0(v0: i32):\n\
         \x20 v1 = cap.call 4 3 () -> (i64) v0 ()\n\
         \x20 v2 = i64.const 0\n\
         \x20 v3 = i32.const 3\n\
         \x20 v4 = cap.call 4 0 (i64, i64, i64, i32) -> (i64) v0 (v2, v2, v1, v3)\n\
         \x20 v6 = i64.const {MARKER}\n\
         \x20 i64.store v2 v6\n\
         \x20 v7 = i64.load v1\n\
         \x20 return v7\n\
         }}\n"
    );
    let m = parse_module(&src).expect("parse");
    verify_module(&m).expect("verify");

    let mut host = Host::new();
    let h = host.grant_shared_region(1 << 16);
    let mut fuel = 1_000_000u64;
    let (res, _snap) =
        run_capture_reserved_with_host(&m, 0, &[Value::I32(h)], &mut fuel, &[], 0, &mut host);

    assert_eq!(
        res.expect("run ok"),
        vec![Value::I64(0)],
        "an un-aliased window page must read zero, not the marker"
    );
}

/// Anti-rot pin for the Windows §13 gap (tracking issue #1): the JIT's shared mapping is not yet
/// wired on Windows (it needs placeholder reservations — see the `MprotectWindow::map_region` TODO),
/// so `map` over a `SharedRegion` returns `-EINVAL` (-22) there. This test asserts that contract so
/// the gap stays *tested and intentional* — it will fail loudly the moment Windows support lands,
/// at which point flip it to the real `jit_diff` differential.
#[cfg(windows)]
#[test]
fn shared_region_map_is_einval_on_windows_until_placeholders() {
    use core::ffi::c_void;
    use svm_jit::{compile_and_run_capture_reserved_with_host, JitOutcome};

    let src = "memory 16\n\
         func (i32) -> (i64) {\n\
         block0(v0: i32):\n\
         \x20 v1 = cap.call 4 3 () -> (i64) v0 ()\n\
         \x20 v2 = i64.const 0\n\
         \x20 v3 = i32.const 3\n\
         \x20 v4 = cap.call 4 0 (i64, i64, i64, i32) -> (i64) v0 (v2, v2, v1, v3)\n\
         \x20 return v4\n\
         }\n";
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");

    let mut host = Host::new();
    let h = host.grant_shared_region(1 << 16); // pure-Rust backing (no os_fd on Windows)
    let (jit, _mem) = compile_and_run_capture_reserved_with_host(
        &m,
        0,
        &[h as i64],
        &[],
        0,
        svm_run::cap_thunk,
        &mut host as *mut Host as *mut c_void,
    )
    .expect("jit compiles");
    assert!(
        matches!(jit, JitOutcome::Returned(ref s) if s == &[-22]),
        "Windows JIT map_region must return -EINVAL until placeholder mappings land (issue #1): {jit:?}"
    );
}
