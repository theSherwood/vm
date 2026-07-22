//! IMPORTS.md **phase 2** — `rebindable` import slots + `import.attach`: the
//! reflect-then-attach discovery pattern over executable imports, with one call convention.
//!
//! A guest declares a typed-but-empty slot (`rebindable` + a template-only registry binding),
//! discovers a granted capability at runtime (`cap.self.resolve`), attaches it into the slot,
//! and drives it through the same static-mode `call.import` as any required import — on all
//! three backends, with no module rewriting anywhere.

use svm_interp::iface;
use svm_run::{instantiate_with_imports, Backend, HostCap, Imports, Outcome, RunConfig};
use svm_text::{parse_module, print_module};

/// Paramless `_start`: stores `"logger"` at 0..6 and `"hi"` at 32..34, resolves the
/// runtime-granted `logger` capability by name, **attaches** it into rebindable slot 0
/// (`"out"`), then writes through the slot and exits 7 via required slot 1 (`"exit"`).
const ATTACH_START: &str = "memory 17\n\
import 0 \"out\" (i64, i64) -> (i64) rebindable\n\
import 1 \"exit\" (i32) -> ()\n\
func () -> () {\n\
block 0 () {\n\
  vc0 = i32.const 108\n\
  va0 = i64.const 0\n\
  i32.store8 va0 vc0\n\
  vc1 = i32.const 111\n\
  va1 = i64.const 1\n\
  i32.store8 va1 vc1\n\
  vc2 = i32.const 103\n\
  va2 = i64.const 2\n\
  i32.store8 va2 vc2\n\
  vc3 = i32.const 103\n\
  va3 = i64.const 3\n\
  i32.store8 va3 vc3\n\
  vc4 = i32.const 101\n\
  va4 = i64.const 4\n\
  i32.store8 va4 vc4\n\
  vc5 = i32.const 114\n\
  va5 = i64.const 5\n\
  i32.store8 va5 vc5\n\
  vc6 = i32.const 104\n\
  va6 = i64.const 32\n\
  i32.store8 va6 vc6\n\
  vc7 = i32.const 105\n\
  va7 = i64.const 33\n\
  i32.store8 va7 vc7\n\
  vp = i64.const 0\n\
  vl = i64.const 6\n\
  vh = cap.self.resolve vp vl\n\
  vst = import.attach 0 vh\n\
  vbuf = i64.const 32\n\
  vn = i64.const 2\n\
  vw = call.import 0 (vbuf, vn)\n\
  vcode = i32.const 7\n\
  call.import 1 (vcode)\n\
  unreachable\n\
  }\n\
}\n\
export 0 func \"_start\" 0\n";

fn registry() -> Imports {
    Imports::new()
        .provide("out", HostCap::template(iface::STREAM, 1))
        .provide("exit", HostCap::exit())
}

/// Text + wire round-trips: the `rebindable` suffix and `import.attach` survive
/// print→parse and encode→decode (v4 mode byte + the 0x63 opcode).
#[test]
fn mode_and_attach_round_trip() {
    let m = parse_module(ATTACH_START).expect("parse");
    assert_eq!(m.imports[0].mode, svm_ir::ImportMode::Rebindable);
    assert_eq!(m.imports[1].mode, svm_ir::ImportMode::Required);
    let m2 = parse_module(&print_module(&m)).expect("reparse");
    assert_eq!(m, m2, "text round-trip");
    let m3 = svm_encode::decode_module(&svm_encode::encode_module(&m)).expect("decode");
    assert_eq!(m, m3, "wire round-trip");
}

/// The discovery pattern end-to-end, identically on all three backends: template slot starts
/// empty, the guest resolves the run-granted `logger`, attaches, writes through the slot.
#[test]
fn attach_then_call_runs_identically_on_all_three_backends() {
    let m = parse_module(ATTACH_START).expect("parse");
    let inst = instantiate_with_imports(m.clone(), registry()).expect("instantiate");
    assert_eq!(inst.module(), &m, "no rewrite (phase-1 invariant holds)");
    for backend in [Backend::TreeWalk, Backend::Bytecode, Backend::Jit] {
        let r = inst
            .run_with_caps(
                backend,
                &RunConfig::default(),
                &[("logger", HostCap::stdout())],
            )
            .unwrap_or_else(|e| panic!("{backend:?}: {e}"));
        assert_eq!(r.stdout, b"hi", "{backend:?} stdout");
        assert_eq!(r.outcome, Outcome::Exited(7), "{backend:?} outcome");
    }
}

/// Attaching a wrong-interface handle is a probeable `-EINVAL`, not a trap: the guest attaches
/// the (Exit-typed) `exit` handle into the Stream-typed slot and exits with the status.
#[test]
fn attach_wrong_type_returns_einval() {
    let guest = "memory 17\n\
import 0 \"out\" (i64, i64) -> (i64) rebindable\n\
import 1 \"exit\" (i32) -> ()\n\
func () -> () {\n\
block 0 () {\n\
  vc0 = i32.const 101\n\
  va0 = i64.const 0\n\
  i32.store8 va0 vc0\n\
  vc1 = i32.const 120\n\
  va1 = i64.const 1\n\
  i32.store8 va1 vc1\n\
  vc2 = i32.const 105\n\
  va2 = i64.const 2\n\
  i32.store8 va2 vc2\n\
  vc3 = i32.const 116\n\
  va3 = i64.const 3\n\
  i32.store8 va3 vc3\n\
  vp = i64.const 0\n\
  vl = i64.const 4\n\
  vh = cap.self.resolve vp vl\n\
  vst = import.attach 0 vh\n\
  call.import 1 (vst)\n\
  unreachable\n\
  }\n\
}\n\
export 0 func \"_start\" 0\n";
    let m = parse_module(guest).expect("parse");
    let inst = instantiate_with_imports(m, registry()).expect("instantiate");
    for backend in [Backend::TreeWalk, Backend::Bytecode, Backend::Jit] {
        let r = inst
            .run(backend, &RunConfig::default())
            .unwrap_or_else(|e| panic!("{backend:?}: {e}"));
        assert_eq!(
            r.outcome,
            Outcome::Exited(-22),
            "{backend:?}: attach must return -EINVAL for a wrong-type handle"
        );
    }
}

/// Calling through a declared-but-never-attached rebindable slot is fail-closed (CapFault).
#[test]
fn unattached_rebindable_slot_traps() {
    let guest = "memory 17\n\
import 0 \"out\" (i64, i64) -> (i64) rebindable\n\
import 1 \"exit\" (i32) -> ()\n\
func () -> () {\n\
block 0 () {\n\
  vph = i32.const 0\n\
  vbuf = i64.const 32\n\
  vn = i64.const 2\n\
  vw = call.import 0 (vbuf, vn)\n\
  vcode = i32.const 0\n\
  call.import 1 (vcode)\n\
  unreachable\n\
  }\n\
}\n\
export 0 func \"_start\" 0\n";
    let m = parse_module(guest).expect("parse");
    let inst = instantiate_with_imports(m, registry()).expect("instantiate");
    for backend in [Backend::TreeWalk, Backend::Bytecode, Backend::Jit] {
        let Err(err) = inst.run(backend, &RunConfig::default()) else {
            panic!("{backend:?}: unattached slot must trap");
        };
        assert!(
            err.contains("CapFault"),
            "{backend:?}: expected a cap fault, got: {err}"
        );
    }
}

/// Verifier rules: attach to a `required` slot and attach out of manifest range are rejected
/// statically; a `required` import bound to a template-only capability fails instantiation.
#[test]
fn attach_static_rules_fail_closed() {
    // Attach to a required slot.
    let m = parse_module(
        "import 0 \"x\" (i64) -> (i64)\n\
func () -> () {\n\
block 0 () {\n\
  vh = i32.const 0\n\
  vs = import.attach 0 vh\n\
  return\n\
  }\n\
}\n",
    )
    .expect("parse");
    assert!(
        matches!(
            svm_verify::verify_module(&m),
            Err(svm_verify::VerifyError::AttachNotRebindable { .. })
        ),
        "attach to a required slot must be rejected"
    );
    // Attach past the manifest.
    let m = parse_module(
        "func () -> () {\n\
block 0 () {\n\
  vh = i32.const 0\n\
  vs = import.attach 3 vh\n\
  return\n\
  }\n\
}\n",
    )
    .expect("parse");
    assert!(
        matches!(
            svm_verify::verify_module(&m),
            Err(svm_verify::VerifyError::UnresolvedImport { .. })
        ),
        "attach out of manifest range must be rejected"
    );
    // Required import bound to a template-only cap.
    let m = parse_module(ATTACH_START).expect("parse");
    let bad = Imports::new()
        .provide("out", HostCap::template(iface::STREAM, 1))
        .provide("exit", HostCap::template(iface::EXIT, 0)); // required "exit" gets a template
    let Err(err) = instantiate_with_imports(m, bad) else {
        panic!("required-import-to-template must fail instantiation");
    };
    assert!(err.contains("template-only"), "error explains it: {err}");
}

/// The manifest-completeness bit (IMPORTS.md §2.2): true for a module whose only capability
/// dispatch is `call.import`; false the moment a dynamic-mode `cap.call` appears.
#[test]
fn manifest_completeness_bit() {
    let m = parse_module(ATTACH_START).expect("parse");
    assert!(
        svm_verify::manifest_complete(&m),
        "call.import-only module is manifest-complete"
    );
    let dynamic = parse_module(
        "func (i32) -> () {\n\
block 0 (v0: i32) {\n\
  v1 = i64.const 0\n\
  v2 = cap.call 0 1 (i64, i64) -> (i64) v0 (v1, v1)\n\
  return\n\
  }\n\
}\n",
    )
    .expect("parse");
    assert!(
        !svm_verify::manifest_complete(&dynamic),
        "a cap.call makes the module open-world"
    );
    // The reserved self namespace is exempt (§3.1): its dispatch-form ops (e.g. provenance)
    // are authority-neutral reflection, statically identifiable from the type_id immediate —
    // querying them must not cost the completeness bit.
    let self_query = parse_module(
        "func (i32) -> (i32) {\n\
block 0 (v0: i32) {\n\
  v1 = i64.const 0\n\
  v2 = cap.call 4294967295 5 (i64) -> (i32) v0 (v1)\n\
  return v2\n\
  }\n\
}\n",
    )
    .expect("parse");
    assert!(
        svm_verify::manifest_complete(&self_query),
        "a cap.self dispatch (reserved immediate) keeps the completeness bit"
    );
}
