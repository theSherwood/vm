//! §3.6 **behavioral parity** — a serving domain behaves identically on all three backends.
//! The serve loop is a single implementation on the reference interpreter (the oracle);
//! bytecode reaches it by declining the serving ops at compile and falling back, and the JIT
//! by the `module_serves` fold in svm-run. This pins the end-to-end §3.6 story — spawn a
//! serving child, mint a live-callee offer (`Instantiator.child_offer`), call through it
//! (park), serve (`svc.wait`), reply-wake, join — producing the SAME observable on
//! TreeWalk, Bytecode, and Jit.

use svm_run::{instantiate_with_imports, Backend, HostCap, Imports, Outcome, RunConfig};
use svm_text::parse_module;

/// `_start`: resolve the granted `"vm"` Instantiator by name, spawn the serving child
/// (func 1), mint `child_offer(child, export 0)`, call `add(40, 2)` through the live cap
/// (parking until the child's `svc.wait` serves it), join, and exit with the reply — 42.
const SERVING_PROGRAM: &str = "\
memory 17
data 0 \"vm\"
type 0 func (i64, i64) -> (i64)
type 1 interface { add: 0 }
export 0 interface \"adder\" 1 { add: 2 }
import 0 \"exit\" (i32) -> ()

func 0 () -> () {
block 0 () {
  vp = i64.const 0
  vl = i64.const 2
  vh = cap.self.resolve vp vl
  v1 = i64.const 1
  v2 = i64.const 65536
  v3 = i64.const 12
  v4 = i64.const 0
  v5 = cap.call 6 0 (i64, i64, i64, i64) -> (i32) vh (v1, v2, v3, v4)
  v6 = i64.const 0
  v7 = cap.call 6 14 (i32, i64) -> (i32) vh (v5, v6)
  va = i64.const 40
  vb = i64.const 2
  vr = cap.call 268435456 0 (i64, i64) -> (i64) v7 (va, vb)
  vj = cap.call 6 1 (i32) -> (i64) vh (v5)
  vc = i32.wrap_i64 vr
  call.import 0 (vc)
  unreachable
  }
}

func 1 (i64) -> (i64) {
block 0 (v0: i64) {
  vz = i32.const 0
  vn = svc.wait vz
  return vn
  }
}

func 2 (i64, i64) -> (i64) {
block 0 (va: i64, vb: i64) {
  vs = i64.add va vb
  return vs
  }
}
";

#[test]
fn a_serving_domain_behaves_identically_on_all_three_backends() {
    let m = parse_module(SERVING_PROGRAM).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    let registry = Imports::new().provide("exit", HostCap::exit());
    let inst = instantiate_with_imports(m, registry).expect("instantiate");
    for backend in [Backend::TreeWalk, Backend::Bytecode, Backend::Jit] {
        let r = inst
            .run_with_caps(
                backend,
                &RunConfig::default(),
                &[(
                    "vm",
                    HostCap::custom(6, 0, |h, win| h.grant_instantiator(0, win)),
                )],
            )
            .unwrap_or_else(|e| panic!("{backend:?}: {e}"));
        assert_eq!(
            r.outcome,
            Outcome::Exited(42),
            "{backend:?}: spawn → child_offer → park → svc.wait-serve → reply → join → 42"
        );
    }
}
