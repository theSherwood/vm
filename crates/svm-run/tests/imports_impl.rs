//! IMPORTS.md **§3.2** — provider-side interface offers wired into import slots, end to end:
//! a provider module declares `export "adder" impl <funcidx>...`, the embedder binds a
//! consumer's import slot to one op of that offer (`HostCap::impl_offer`), and the consumer's
//! `call.import` executes the offered guest function — identically on all three backends,
//! through the one shared generic dispatch (v1 pure dispatch: the impl computes over its
//! arguments alone — no window, no capabilities).

use svm_run::{instantiate_with_imports, Backend, HostCap, Imports, Outcome, RunConfig};
use svm_text::parse_module;

/// The provider: func 1 implements `add(a, b) = a + b`; the offer's op 0 names it.
const PROVIDER: &str = "\
type 0 func (i64, i64) -> (i64)
type 1 interface { add: 0 }
export 0 interface \"adder\" 1 { add: 1 }

func (i64) -> (i64) {
block 0 (v0: i64) {
  return v0
  }
}

func (i64, i64) -> (i64) {
block 0 (va: i64, vb: i64) {
  vs = i64.add va vb
  return vs
  }
}
";

/// The consumer: `_start` calls the wired `add` op with (40, 2) and exits with the sum.
const CONSUMER: &str = "\
import 0 \"add\" (i64, i64) -> (i64)
import 1 \"exit\" (i32) -> ()

func () -> () {
block 0 () {
  vh = i32.const 0
  va = i64.const 40
  vb = i64.const 2
  vr = call.import 0 (va, vb)
  vc = i32.wrap_i64 vr
  call.import 1 (vc)
  unreachable
  }
}

export 0 func \"_start\" 0
";

#[test]
fn a_wired_offer_runs_identically_on_all_three_backends() {
    let provider = parse_module(PROVIDER).expect("provider parses");
    svm_verify::verify_module(&provider).expect("provider verifies");
    let consumer = parse_module(CONSUMER).expect("consumer parses");

    let registry = Imports::new()
        .provide(
            "add",
            HostCap::impl_offer(&provider, "adder", 0).expect("offer resolves"),
        )
        .provide("exit", HostCap::exit());
    let inst = instantiate_with_imports(consumer.clone(), registry).expect("instantiate");
    assert_eq!(inst.module(), &consumer, "no rewrite (phase-1 invariant)");

    for backend in [Backend::TreeWalk, Backend::Bytecode, Backend::Jit] {
        let r = inst
            .run(backend, &RunConfig::default())
            .unwrap_or_else(|e| panic!("{backend:?}: {e}"));
        assert_eq!(
            r.outcome,
            Outcome::Exited(42),
            "{backend:?}: the wired guest impl must compute 40 + 2"
        );
    }
}

#[test]
fn offer_signature_mismatch_fails_instantiation_closed() {
    let provider = parse_module(PROVIDER).expect("provider parses");
    // The consumer declares the wrong signature for the wired op: (i64) -> (i64) vs the
    // offer's (i64, i64) -> (i64). Structural, fail-closed, at instantiation — before any run.
    let consumer = parse_module(
        "import 0 \"add\" (i64) -> (i64)\n\
         import 1 \"exit\" (i32) -> ()\n\
         func () -> () {\n\
         block 0 () {\n\
           vh = i32.const 0\n\
           vc = i32.const 0\n\
           call.import 1 (vc)\n\
           unreachable\n\
           }\n\
         }\n\
         export 0 func \"_start\" 0\n",
    )
    .expect("consumer parses");
    let registry = Imports::new()
        .provide(
            "add",
            HostCap::impl_offer(&provider, "adder", 0).expect("offer resolves"),
        )
        .provide("exit", HostCap::exit());
    let err = match instantiate_with_imports(consumer, registry) {
        Err(e) => e,
        Ok(_) => panic!("a signature mismatch must refuse instantiation"),
    };
    assert!(
        err.contains("§3.2"),
        "the refusal names the wiring rule: {err}"
    );
}

/// A stateful provider: `bump() -> i64` increments a counter in the provider's OWN window —
/// the §3.2 v2 exporter-domain-state service, offered as `export "counter" impl 0`.
const STATEFUL_PROVIDER: &str = "\
memory 16
type 0 func () -> (i64)
type 1 interface { add: 0 }
export 0 interface \"counter\" 1 { add: 0 }

func () -> (i64) {
block 0 () {
  va = i64.const 0
  vc = i64.load va
  v1 = i64.const 1
  vn = i64.add vc v1
  i64.store va vn
  return vn
  }
}
";

/// The consumer: calls `bump` three times and exits with the third count — 3 only if the
/// provider's window state persisted across the calls.
const STATEFUL_CONSUMER: &str = "\
import 0 \"bump\" () -> (i64)
import 1 \"exit\" (i32) -> ()

func () -> () {
block 0 () {
  vh = i32.const 0
  v1 = call.import 0 ()
  v2 = call.import 0 ()
  v3 = call.import 0 ()
  vc = i32.wrap_i64 v3
  call.import 1 (vc)
  unreachable
  }
}

export 0 func \"_start\" 0
";

#[test]
fn an_instanced_offer_keeps_state_across_calls_on_all_three_backends() {
    let provider = parse_module(STATEFUL_PROVIDER).expect("provider parses");
    svm_verify::verify_module(&provider).expect("provider verifies");
    let consumer = parse_module(STATEFUL_CONSUMER).expect("consumer parses");

    let registry = Imports::new()
        .provide(
            "bump",
            HostCap::impl_service(&provider, "counter", 0).expect("offer resolves"),
        )
        .provide("exit", HostCap::exit());
    let inst = instantiate_with_imports(consumer, registry).expect("instantiate");

    for backend in [Backend::TreeWalk, Backend::Bytecode, Backend::Jit] {
        let r = inst
            .run(backend, &RunConfig::default())
            .unwrap_or_else(|e| panic!("{backend:?}: {e}"));
        assert_eq!(
            r.outcome,
            Outcome::Exited(3),
            "{backend:?}: the provider's window state must persist across the three calls"
        );
    }
}

#[test]
fn impl_offer_fails_closed_on_unknown_offer_or_op() {
    let provider = parse_module(PROVIDER).expect("provider parses");
    assert!(HostCap::impl_offer(&provider, "nope", 0).is_none());
    assert!(HostCap::impl_offer(&provider, "adder", 1).is_none());
}

// --- §3.5 grouped host-native providers (HostCap::iface) --------------------------------------

/// A consumer that imports a **whole interface** and uses only one op: a grouped import `log`
/// declaring `interface { write }` (a subset of the host `Stream`'s read/write/close), called
/// as `call.import 0.write`. `write(buf, len) -> nwritten` writes the 3 data bytes and returns
/// 3, which the guest passes to `exit` — the cross-backend observable.
const GROUPED_CONSUMER: &str = "\
memory 16
data 0 \"hi\\n\"
type 0 func (i64, i64) -> (i64)
type 1 interface { write: 0 }
type 2 func (i32) -> ()
import 0 interface \"log\" 1
import 1 func \"exit\" 2
func 0 () -> () {
block 0 () {
  v0 = i64.const 0
  v1 = i64.const 3
  v2 = call.import 0.write (v0, v1)
  v3 = i32.wrap_i64 v2
  call.import 1 (v3)
  unreachable
  }
}
export 0 func \"_start\" 0
";

/// The provided host interface, in the `Stream` handle's **native op order**: read=0, write=1,
/// close=2. A consumer needing only `write` covers it (subset); the frozen remap sends the
/// consumer's op 0 to the handle's native op 1.
fn stream_shape() -> svm_run::IfaceShape {
    use svm_ir::{FuncType, ValType};
    let rw = FuncType {
        params: vec![ValType::I64, ValType::I64],
        results: vec![ValType::I64],
    };
    svm_run::IfaceShape::new()
        .op("read", rw.clone())
        .op("write", rw)
        .op(
            "close",
            FuncType {
                params: vec![],
                results: vec![],
            },
        )
}

#[test]
fn a_grouped_host_interface_binds_a_subset_and_dispatches_across_backends() {
    let consumer = parse_module(GROUPED_CONSUMER).expect("consumer parses");
    let shape = stream_shape();
    let registry = Imports::new()
        .provide(
            "log",
            // A host-native `Stream` (stdout) offered as a whole interface; the consumer binds a
            // subset and the remap routes its `write` to the stream's native op 1.
            HostCap::iface(&shape, |h, _| h.grant_stream(svm_interp::StreamRole::Out)),
        )
        .provide("exit", HostCap::exit());
    let inst = instantiate_with_imports(consumer, registry).expect("instantiate");
    for backend in [Backend::TreeWalk, Backend::Bytecode, Backend::Jit] {
        let r = inst
            .run(backend, &RunConfig::default())
            .unwrap_or_else(|e| panic!("{backend:?}: {e}"));
        assert_eq!(
            r.outcome,
            Outcome::Exited(3),
            "{backend:?}: the grouped write dispatched through the remap and returned nwritten"
        );
    }
}

#[test]
fn a_grouped_host_interface_that_does_not_cover_fails_instantiation() {
    // The consumer requires `read`, which the provided shape has — but with a *different*
    // signature than the consumer declared, so coverage fails closed at instantiation.
    let consumer = parse_module(
        "type 0 func (i32) -> (i32)\n\
         type 1 interface { read: 0 }\n\
         import 0 interface \"log\" 1\n\
         func 0 () -> () {\n\
         block 0 () {\n\
           return\n\
           }\n\
         }\n\
         export 0 func \"_start\" 0\n",
    )
    .expect("consumer parses");
    let shape = stream_shape(); // read is (i64,i64)->(i64), not (i32)->(i32)
    let registry = Imports::new().provide(
        "log",
        HostCap::iface(&shape, |h, _| h.grant_stream(svm_interp::StreamRole::Out)),
    );
    let err = match instantiate_with_imports(consumer, registry) {
        Err(e) => e,
        Ok(_) => panic!("a non-covering grouped import must refuse instantiation"),
    };
    assert!(
        err.contains("§3.5") && err.contains("not covered"),
        "the refusal names the coverage rule: {err}"
    );
}
