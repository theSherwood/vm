//! **Phase 3 — uniform run config across backends.** The same powerbox program runs on the
//! tree-walker, the bytecode engine, and the JIT through one `RunConfig`, and the resource limits
//! (fuel, spawn quota, window size) apply uniformly where each backend supports them. Proves the
//! "pick a backend, set the knobs, run" interface from `svm_run::Instance::run` / `run_diff`.
//!
//! Gated `#![cfg(unix)]` like the other JIT differential suites.
#![cfg(unix)]

use svm_run::{instantiate, Backend, Instance, Limits, Outcome, RunConfig, Value};

/// A minimal fixed-powerbox program: loads the stashed stdout handle (slot 0) and writes a RO string.
const HELLO: &str = "\
memory 15
data ro 16384 \"hello, powerbox\\n\"
export \"entry\" 0
func (i64) -> (i32) {
block0(v0: i64):
  v1 = i64.const 0
  v2 = i32.load v1
  v3 = i64.const 16384
  v4 = i64.const 16
  v5 = call.import \"write\" (i64, i64) -> (i64) v2 (v3, v4)
  v6 = i32.const 0
  return v6
}
";

fn hello_instance() -> Instance {
    let module = svm_text::parse_module(HELLO).expect("parse");
    let with_start = svm_ir::synth_powerbox_start(module, 0, 3, false).expect("synth");
    instantiate(with_start).expect("instantiate")
}

/// All three backends run the same program through one `RunConfig` and produce identical output.
#[test]
fn every_backend_runs_under_one_config() {
    for backend in [Backend::TreeWalk, Backend::Bytecode, Backend::Jit] {
        let run = hello_instance()
            .run(backend, &RunConfig::default())
            .unwrap_or_else(|e| panic!("{backend:?}: {e}"));
        assert_eq!(run.stdout, b"hello, powerbox\n", "{backend:?} stdout");
        assert_eq!(
            run.outcome,
            Outcome::Returned(vec![Value::I32(0)]),
            "{backend:?} outcome"
        );
    }
}

/// The differential entry (`run_diff`) cross-checks tree-walk vs JIT under the config.
#[test]
fn run_diff_under_config() {
    let run = hello_instance()
        .run_diff(&RunConfig::default())
        .expect("diff");
    assert_eq!(run.stdout, b"hello, powerbox\n");
    assert_eq!(run.outcome, Outcome::Returned(vec![Value::I32(0)]));
}

/// `fuel` bounds the **interpreters** (per-op budget) but is ignored by the JIT (which bounds runaway
/// guests with a `deadline` instead) — the one documented backend-specific knob, shown uniform-API but
/// honest about who honors it.
#[test]
fn fuel_bounds_interpreters_not_the_jit() {
    let tight = RunConfig {
        limits: Limits {
            fuel: Some(1),
            ..Limits::default()
        },
        ..RunConfig::default()
    };
    // A 1-op budget out-of-fuels the tree-walker and the bytecode engine before the program finishes.
    assert!(
        hello_instance().run(Backend::TreeWalk, &tight).is_err(),
        "fuel=1 must out-of-fuel the tree-walker"
    );
    assert!(
        hello_instance().run(Backend::Bytecode, &tight).is_err(),
        "fuel=1 must out-of-fuel the bytecode engine"
    );
    // The JIT has no per-op counter, so it ignores `fuel` and runs to completion.
    let run = hello_instance()
        .run(Backend::Jit, &tight)
        .expect("the JIT ignores per-op fuel");
    assert_eq!(run.stdout, b"hello, powerbox\n");
}

/// The "amount of memory available" knob (`memory_size_log2`) overrides the module's declared window
/// uniformly across backends.
#[test]
fn memory_window_override_applies_to_every_backend() {
    let cfg = RunConfig {
        memory_size_log2: Some(22), // 4 MiB — larger than the synthesized default
        ..RunConfig::default()
    };
    for backend in [Backend::TreeWalk, Backend::Bytecode, Backend::Jit] {
        let run = hello_instance()
            .run(backend, &cfg)
            .unwrap_or_else(|e| panic!("{backend:?}: {e}"));
        assert_eq!(
            run.stdout, b"hello, powerbox\n",
            "{backend:?} under 4 MiB window"
        );
    }
}
