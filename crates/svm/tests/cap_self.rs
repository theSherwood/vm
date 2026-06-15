//! §7 capability **reflection** (`cap.self.count` / `cap.self.get`): an always-available, read-only
//! intrinsic by which a guest discovers the capabilities its host granted *this* domain — the live
//! handle-table entries as `(handle, type_id)`. It confers no authority (the guest already holds
//! every handle it sees), so it is ambient and safe (DESIGN.md §7). Interp-first: the JIT bails
//! `Unsupported` for now, so these drive the reference interpreter directly.

use svm_interp::{run_with_host, Host, StreamRole, Value};
use svm_text::parse_module;
use svm_verify::verify_module;

/// Parse + verify a text-IR program, grant the standard 3-handle powerbox (stdout, stdin, exit) in
/// order, and run entry 0 on the interpreter with those handles as its arguments.
fn run(src: &str) -> Vec<Value> {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    // Round-trip the text form while we're here (the intrinsics must print + re-parse identically).
    assert_eq!(
        parse_module(&svm_text::print_module(&m)).expect("reparse"),
        m,
        "cap.self.* must round-trip through the text form"
    );
    let mut host = Host::new();
    let args = vec![
        Value::I32(host.grant_stream(StreamRole::Out)), // slot 0 — Stream (stdout)
        Value::I32(host.grant_stream(StreamRole::In)),  // slot 1 — Stream (stdin)
        Value::I32(host.grant_exit()),                  // slot 2 — Exit
    ];
    let mut fuel = 10_000_000u64;
    run_with_host(&m, 0, &args, &mut fuel, &mut host).expect("interp run")
}

/// `cap.self.count` reports exactly the number of capabilities the host granted this domain.
#[test]
fn count_reflects_the_granted_powerbox() {
    let src = "func (i32, i32, i32) -> (i32) {\n\
               block0(v0: i32, v1: i32, v2: i32):\n\
               \x20 v3 = cap.self.count\n\
               \x20 return v3\n\
               }\n";
    assert_eq!(run(src), vec![Value::I32(3)], "three handles were granted");
}

/// `cap.self.get(i)` yields the i-th held capability's `(handle, type_id)`. Entry 2 (the 3rd grant)
/// is Exit, so its `type_id` is `iface::EXIT == 1`.
#[test]
fn get_reports_the_interface_type_id() {
    let src = "func (i32, i32, i32) -> (i32) {\n\
               block0(v0: i32, v1: i32, v2: i32):\n\
               \x20 v3 = i32.const 2\n\
               \x20 v4, v5 = cap.self.get v3\n\
               \x20 return v5\n\
               }\n";
    assert_eq!(run(src), vec![Value::I32(1)], "entry 2 is Exit (type_id 1)");
}

/// The *handle* `cap.self.get` returns is the very handle the host granted — so a guest can use what
/// it discovers. Entry 2's handle must equal `v2` (the exit handle passed into `_start`); the
/// program returns their difference, so a passing run is `0`. Entry 0's type_id is `Stream == 0`.
#[test]
fn get_returns_the_usable_granted_handle() {
    let src = "func (i32, i32, i32) -> (i32) {\n\
               block0(v0: i32, v1: i32, v2: i32):\n\
               \x20 v3 = i32.const 2\n\
               \x20 v4, v5 = cap.self.get v3\n\
               \x20 v6 = i32.sub v4 v2\n\
               \x20 return v6\n\
               }\n";
    assert_eq!(
        run(src),
        vec![Value::I32(0)],
        "the discovered handle equals the granted one"
    );
}

/// An out-of-range index is fail-closed (the guest is expected to bound it by `cap.self.count`).
#[test]
fn get_out_of_range_traps() {
    let src = "func (i32, i32, i32) -> (i32) {\n\
               block0(v0: i32, v1: i32, v2: i32):\n\
               \x20 v3 = i32.const 99\n\
               \x20 v4, v5 = cap.self.get v3\n\
               \x20 return v5\n\
               }\n";
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    let mut host = Host::new();
    let args = vec![
        Value::I32(host.grant_stream(StreamRole::Out)),
        Value::I32(host.grant_stream(StreamRole::In)),
        Value::I32(host.grant_exit()),
    ];
    let mut fuel = 10_000_000u64;
    assert!(
        run_with_host(&m, 0, &args, &mut fuel, &mut host).is_err(),
        "out-of-range cap.self.get must fail closed"
    );
}

/// The reflection intrinsics survive the binary form (`svm-encode`) — opcodes 0x7A/0x7B round-trip.
#[test]
fn binary_round_trip() {
    let src = "func (i32) -> (i32) {\n\
               block0(v0: i32):\n\
               \x20 v1 = cap.self.count\n\
               \x20 v2 = i32.const 0\n\
               \x20 v3, v4 = cap.self.get v2\n\
               \x20 return v1\n\
               }\n";
    let m = parse_module(src).expect("parse");
    let bytes = svm_encode::encode_module(&m);
    let back = svm_encode::decode_module(&bytes).expect("decode");
    assert_eq!(
        back, m,
        "cap.self.* must round-trip through the binary form"
    );
}

/// Interp-first: the JIT does not yet lower `cap.self.*`, so it bails `Unsupported` (fail-closed)
/// rather than miscompiling — the reference interpreter is the whole story for now (cf. fibers).
#[test]
fn jit_bails_unsupported() {
    let src = "func (i32) -> (i32) {\n\
               block0(v0: i32):\n\
               \x20 v1 = cap.self.count\n\
               \x20 return v1\n\
               }\n";
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    assert!(
        svm_jit::compile_and_run(&m, 0, &[0]).is_err(),
        "the JIT must bail Unsupported on cap.self.*, not miscompile"
    );
}
