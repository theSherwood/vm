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
export \"adder\" impl 1

func (i64) -> (i64) {
block0(v0: i64):
  return v0
}

func (i64, i64) -> (i64) {
block0(va: i64, vb: i64):
  vs = i64.add va vb
  return vs
}
";

/// The consumer: `_start` calls the wired `add` op with (40, 2) and exits with the sum.
const CONSUMER: &str = "\
import 0 \"add\" (i64, i64) -> (i64)
import 1 \"exit\" (i32) -> ()

func () -> () {
block0():
  vh = i32.const 0
  va = i64.const 40
  vb = i64.const 2
  vr = call.import 0 vh (va, vb)
  vc = i32.wrap_i64 vr
  call.import 1 vh (vc)
  unreachable
}

export \"_start\" 0
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
         block0():\n\
           vh = i32.const 0\n\
           vc = i32.const 0\n\
           call.import 1 vh (vc)\n\
           unreachable\n\
         }\n\
         export \"_start\" 0\n",
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
export \"counter\" impl 0

func () -> (i64) {
block0():
  va = i64.const 0
  vc = i64.load va
  v1 = i64.const 1
  vn = i64.add vc v1
  i64.store va vn
  return vn
}
";

/// The consumer: calls `bump` three times and exits with the third count — 3 only if the
/// provider's window state persisted across the calls.
const STATEFUL_CONSUMER: &str = "\
import 0 \"bump\" () -> (i64)
import 1 \"exit\" (i32) -> ()

func () -> () {
block0():
  vh = i32.const 0
  v1 = call.import 0 vh ()
  v2 = call.import 0 vh ()
  v3 = call.import 0 vh ()
  vc = i32.wrap_i64 v3
  call.import 1 vh (vc)
  unreachable
}

export \"_start\" 0
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
