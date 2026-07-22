//! §7 capability **reflection** (`cap.self.count` / `cap.self.get`): an always-available, read-only
//! intrinsic by which a guest discovers the capabilities its host granted *this* domain — the live
//! handle-table entries as `(handle, type_id)`. It confers no authority (the guest already holds
//! every handle it sees), so it is ambient and safe (DESIGN.md §7). These run on **both** backends:
//! the interpreter services `cap.self.*` directly, the JIT lowers them to a `cap.call` thunk with the
//! reserved `CAP_SELF_TYPE_ID`, and both share one host `self_dispatch` — so they agree.

use core::ffi::c_void;

use svm_interp::{run_with_host, Host, StreamRole, Value};
use svm_jit::{compile_and_run_with_host, JitOutcome};
use svm_text::parse_module;
use svm_verify::verify_module;

/// Grant the standard 3-handle powerbox (stdout, stdin, exit) in order; the handle values are the
/// entry's arguments. Granting into a fresh `Host` is deterministic, so the interp and JIT hosts get
/// identical handle encodings.
fn grant3(host: &mut Host) -> Vec<Value> {
    vec![
        Value::I32(host.grant_stream(StreamRole::Out)), // slot 0 — Stream (stdout)
        Value::I32(host.grant_stream(StreamRole::In)),  // slot 1 — Stream (stdin)
        Value::I32(host.grant_exit()),                  // slot 2 — Exit
    ]
}

fn as_i64(v: &Value) -> i64 {
    match v {
        Value::I32(x) => *x as i64,
        Value::I64(x) => *x,
        other => panic!("unexpected value {other:?}"),
    }
}

/// Parse + verify, round-trip the text form, then run entry 0 on **both** backends with the 3-handle
/// powerbox; assert they agree and return the interpreter's results. (The single-`i32` results these
/// tests use compare directly.)
fn run(src: &str) -> Vec<Value> {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    assert_eq!(
        parse_module(&svm_text::print_module(&m)).expect("reparse"),
        m,
        "cap.self.* must round-trip through the text form"
    );

    let mut hi = Host::new();
    let iargs = grant3(&mut hi);
    let mut fuel = 10_000_000u64;
    let interp = run_with_host(&m, 0, &iargs, &mut fuel, &mut hi).expect("interp run");

    let mut hj = Host::new();
    let jslots: Vec<i64> = grant3(&mut hj).iter().map(as_i64).collect();
    let jit = match compile_and_run_with_host(&m, 0, &jslots, svm_run::cap_thunk, host_ptr(&mut hj))
        .expect("jit compile")
    {
        JitOutcome::Returned(v) => v,
        other => panic!("jit did not return: {other:?}"),
    };
    assert_eq!(
        interp.iter().map(as_i64).collect::<Vec<_>>(),
        jit,
        "interp and JIT must agree on cap.self.*"
    );
    interp
}

fn host_ptr(h: &mut Host) -> *mut c_void {
    h as *mut Host as *mut c_void
}

/// `cap.self.count` reports exactly the number of capabilities the host granted this domain.
#[test]
fn count_reflects_the_granted_powerbox() {
    let src = "func (i32, i32, i32) -> (i32) {\n\
               block 0 (v0: i32, v1: i32, v2: i32) {\n\
               \x20 v3 = cap.self.count\n\
               \x20 return v3\n\
                 }\n\
               }\n";
    assert_eq!(run(src), vec![Value::I32(3)], "three handles were granted");
}

/// `cap.self.get(i)` yields the i-th held capability's `(handle, type_id)`. Entry 2 (the 3rd grant)
/// is Exit, so its `type_id` is `iface::EXIT == 1`.
#[test]
fn get_reports_the_interface_type_id() {
    let src = "func (i32, i32, i32) -> (i32) {\n\
               block 0 (v0: i32, v1: i32, v2: i32) {\n\
               \x20 v3 = i32.const 2\n\
               \x20 v4, v5 = cap.self.get v3\n\
               \x20 return v5\n\
                 }\n\
               }\n";
    assert_eq!(run(src), vec![Value::I32(1)], "entry 2 is Exit (type_id 1)");
}

/// The *handle* `cap.self.get` returns is the very handle the host granted — so a guest can use what
/// it discovers. Entry 2's handle must equal `v2` (the exit handle); the program returns their
/// difference, so a passing run is `0`.
#[test]
fn get_returns_the_usable_granted_handle() {
    let src = "func (i32, i32, i32) -> (i32) {\n\
               block 0 (v0: i32, v1: i32, v2: i32) {\n\
               \x20 v3 = i32.const 2\n\
               \x20 v4, v5 = cap.self.get v3\n\
               \x20 v6 = i32.sub v4 v2\n\
               \x20 return v6\n\
                 }\n\
               }\n";
    assert_eq!(
        run(src),
        vec![Value::I32(0)],
        "the discovered handle equals the granted one"
    );
}

/// An out-of-range index is fail-closed on **both** backends (the guest bounds it by the count).
#[test]
fn get_out_of_range_traps() {
    let src = "func (i32, i32, i32) -> (i32) {\n\
               block 0 (v0: i32, v1: i32, v2: i32) {\n\
               \x20 v3 = i32.const 99\n\
               \x20 v4, v5 = cap.self.get v3\n\
               \x20 return v5\n\
                 }\n\
               }\n";
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");

    let mut hi = Host::new();
    let iargs = grant3(&mut hi);
    let mut fuel = 10_000_000u64;
    assert!(
        run_with_host(&m, 0, &iargs, &mut fuel, &mut hi).is_err(),
        "out-of-range cap.self.get must fail closed (interp)"
    );

    let mut hj = Host::new();
    let jslots: Vec<i64> = grant3(&mut hj).iter().map(as_i64).collect();
    let outcome = compile_and_run_with_host(&m, 0, &jslots, svm_run::cap_thunk, host_ptr(&mut hj))
        .expect("jit compile");
    assert!(
        matches!(outcome, JitOutcome::Trapped(_)),
        "out-of-range cap.self.get must trap on the JIT too, got {outcome:?}"
    );
}

/// The reflection intrinsics survive the binary form (`svm-encode`) — opcodes 0x7A/0x7B/0x7E
/// (`count`/`get`/`resolve`) round-trip, through text and binary.
#[test]
fn binary_round_trip() {
    let src = "memory 15\n\
               func (i32) -> (i32) {\n\
               block 0 (v0: i32) {\n\
               \x20 v1 = cap.self.count\n\
               \x20 v2 = i32.const 0\n\
               \x20 v3, v4 = cap.self.get v2\n\
               \x20 v5 = i64.const 0\n\
               \x20 v6 = i64.const 4\n\
               \x20 v7 = cap.self.resolve v5 v6\n\
               \x20 v8 = cap.self.label v3 v5 v6\n\
               \x20 return v1\n\
                 }\n\
               }\n";
    let m = parse_module(src).expect("parse");
    // Text print → re-parse is identity (covers `cap.self.resolve`'s grammar).
    assert_eq!(
        parse_module(&svm_text::print_module(&m)).expect("reparse"),
        m,
        "cap.self.* must round-trip through the text form"
    );
    let bytes = svm_encode::encode_module(&m);
    let back = svm_encode::decode_module(&bytes).expect("decode");
    assert_eq!(
        back, m,
        "cap.self.* must round-trip through the binary form"
    );
}

/// `Host::cap_label` (F9) is the host-side reverse of the name directory: a registered handle resolves
/// to its label, an unregistered/forged one to `None` — the accessor an embedder uses in diagnostics.
#[test]
fn host_cap_label_round_trips_and_misses() {
    let mut h = Host::new();
    let handle = h.grant_stream(StreamRole::Out);
    assert_eq!(h.cap_label(handle), None, "a grant is unlabeled by default");
    h.register_cap_name("stdout", handle);
    assert_eq!(
        h.cap_label(handle),
        Some("stdout"),
        "labeled after register"
    );
    assert_eq!(
        h.cap_label(handle + 999),
        None,
        "a handle with no registered label resolves to None"
    );
}

/// The capstone: a guest **discovers a capability it holds and uses it** at runtime, with no
/// compile-time knowledge of which handle it is — on **both** backends. It reflects entry 0, confirms
/// it's a `Stream` (`type_id == 0`), and writes "hi" through the *discovered* handle; the bytes
/// reaching each host's stdout prove the discovered handle is a usable capability.
#[test]
fn discover_then_use_a_granted_capability() {
    let src = "memory 16\n\
               data 0 \"hi\"\n\
               func (i32, i32, i32) -> (i32) {\n\
               block 0 (v0: i32, v1: i32, v2: i32) {\n\
               \x20 v3 = i32.const 0\n\
               \x20 v4, v5 = cap.self.get v3\n\
               \x20 v6 = i64.const 0\n\
               \x20 v7 = i64.const 2\n\
               \x20 v8 = cap.call 0 1 (i64, i64) -> (i64) v4(v6, v7)\n\
               \x20 return v5\n\
                 }\n\
               }\n";
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");

    let mut hi = Host::new();
    let iargs = grant3(&mut hi);
    let mut fuel = 10_000_000u64;
    let interp = run_with_host(&m, 0, &iargs, &mut fuel, &mut hi).expect("interp run");
    assert_eq!(interp, vec![Value::I32(0)], "entry 0 reflects as a Stream");
    assert_eq!(
        hi.stdout, b"hi",
        "wrote through the discovered handle (interp)"
    );

    let mut hj = Host::new();
    let jslots: Vec<i64> = grant3(&mut hj).iter().map(as_i64).collect();
    let jit = compile_and_run_with_host(&m, 0, &jslots, svm_run::cap_thunk, host_ptr(&mut hj))
        .expect("jit compile");
    assert!(
        matches!(jit, JitOutcome::Returned(ref v) if v[0] == 0),
        "jit: {jit:?}"
    );
    assert_eq!(
        hj.stdout, b"hi",
        "wrote through the discovered handle (JIT)"
    );
}
