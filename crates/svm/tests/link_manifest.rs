//! **Separate-artifact linking with a retained import manifest** (`svm_ir::link_with_manifest` +
//! `svm_ir::synth_manifest_start`) — the direct-IR frontend on-ramp (first consumer: jacl). A
//! separately-compiled *runtime* unit carries live §7 capability imports (`call.import` slots);
//! a *program* unit reaches the runtime by `call.sym`. The linker resolves the cross-unit symbol
//! to a direct call, **retains** the host-bound names in the merged manifest (DESIGN §7: "the
//! same named-import mechanism generalizes to cross-unit linking"), and `synth_manifest_start`
//! wraps the program entry in the §3e paramless `_start` — a runnable powerbox module through
//! public API, no rewrite at instantiation, no internal reindexing reimplemented by the frontend.
//!
//! Gated `#![cfg(unix)]` like the other differential suites (`Instance::call` runs interp + JIT).
#![cfg(unix)]

use svm_ir::{link, link_with_manifest, synth_manifest_start, LinkError, LinkUnit};
use svm_run::{
    instantiate, instantiate_with_imports, is_named_powerbox_entry, HostCap, Imports, Outcome,
    Value,
};

fn unit(src: &str, exports: &[(&str, u32)]) -> LinkUnit {
    LinkUnit {
        module: svm_text::parse_module(src).expect("parse unit"),
        exports: exports.iter().map(|(n, i)| (n.to_string(), *i)).collect(),
        ..Default::default()
    }
}

/// The separately-compiled **runtime library**: declares the capability import `write` as a
/// manifest slot and exposes `rt_emit(sp)`, which streams its own (read-only) banner through it.
const RUNTIME_UNIT: &str = "\
memory 15
import 0 \"write\" (i64, i64) -> (i64)
data ro 16384 \"hi from rt\\n\"
func (i64) -> () {
block 0 (v0: i64) {
  v1 = i64.const 16384
  v2 = i64.const 11
  v3 = call.import 0 (v1, v2)
  return
  }
}
";

/// The separately-compiled **program**: reaches the runtime purely by name (`call.sym "rt_emit"`),
/// like any cross-unit symbol.
const PROGRAM_UNIT: &str = "\
func (i64) -> (i32) {
block 0 (v0: i64) {
  v1 = i32.const 0
  call.sym \"rt_emit\" (i64) -> () v1 (v0)
  v2 = i32.const 0
  return v2
  }
}
";

/// The core jacl shape: runtime (live capability imports) + program link into one module whose
/// manifest still carries `write`; `synth_manifest_start` makes it a powerbox module; the host
/// binds the name at instantiation and the banner reaches stdout. Public API end to end.
#[test]
fn runtime_with_capability_imports_links_into_a_powerbox() {
    let linked = link_with_manifest(&[
        unit(RUNTIME_UNIT, &[("rt_emit", 0)]),
        unit(PROGRAM_UNIT, &[("main", 0)]),
    ])
    .expect("link runtime + program");
    // The cross-unit symbol resolved; the capability import survived as the one manifest slot.
    assert_eq!(
        linked
            .imports
            .iter()
            .map(|i| i.name.as_str())
            .collect::<Vec<_>>(),
        ["write"],
        "the host-bound name is retained, the cross-unit one is resolved away"
    );
    let main = linked.resolve_export("main").expect("main exported");
    assert_eq!(main, 1, "program funcs reindex after the runtime's");

    let module = synth_manifest_start(linked, main, false).expect("synthesize _start");
    assert!(
        is_named_powerbox_entry(&module),
        "paramless `_start` at function 0, exported by name"
    );
    assert_eq!(module.resolve_export("rt_emit"), Some(1), "exports shifted");
    assert_eq!(module.resolve_export("main"), Some(2), "exports shifted");
    svm_verify::verify_module(&module).expect("linked powerbox module verifies");

    let imports = Imports::new().provide("write", HostCap::stdout());
    let instance = instantiate_with_imports(module, imports).expect("instantiate by name");
    let run = instance.call("_start", &[]).expect("run");
    assert_eq!(run.stdout, b"hi from rt\n", "the runtime's slot was bound");
    assert_eq!(run.outcome, Outcome::Returned(vec![Value::I32(0)]));
}

/// A program with its **own** capability imports, in a different local order than the runtime's:
/// its local slot indices must reindex into the merged, name-deduped manifest. The program's
/// local import 0 (`exit`) lands at merged index 1 — if the reindex were wrong, verification
/// (self-describing sig vs. the slot's declared shape) or the observed outcome would catch it —
/// and its duplicate `write` declaration dedups onto the runtime's slot.
const PROGRAM_WITH_CAPS_UNIT: &str = "\
import 0 \"exit\" (i32) -> ()
import 1 \"write\" (i64, i64) -> (i64)
func (i64) -> () {
block 0 (v0: i64) {
  v1 = i32.const 0
  call.sym \"rt_emit\" (i64) -> () v1 (v0)
  v2 = i32.const 7
  call.import 0 (v2)
  unreachable
  }
}
";

#[test]
fn retained_slots_reindex_into_the_merged_manifest() {
    let linked = link_with_manifest(&[
        unit(RUNTIME_UNIT, &[("rt_emit", 0)]),
        unit(PROGRAM_WITH_CAPS_UNIT, &[("main", 0)]),
    ])
    .expect("link");
    assert_eq!(
        linked
            .imports
            .iter()
            .map(|i| i.name.as_str())
            .collect::<Vec<_>>(),
        ["write", "exit"],
        "first-occurrence order, deduped by name"
    );
    let main = linked.resolve_export("main").expect("main exported");
    let module = synth_manifest_start(linked, main, false).expect("synthesize _start");
    svm_verify::verify_module(&module).expect("verifies — slot indices reindexed consistently");

    let imports = Imports::new()
        .provide("write", HostCap::stdout())
        .provide("exit", HostCap::exit());
    let instance = instantiate_with_imports(module, imports).expect("instantiate");
    let run = instance.call("_start", &[]).expect("run");
    assert_eq!(run.stdout, b"hi from rt\n");
    assert_eq!(
        run.outcome,
        Outcome::Exited(7),
        "the program's local `exit` slot dispatched through its merged index"
    );
}

/// Units disagreeing on a shared import's structural shape fail the link closed — a name never
/// silently binds under two meanings (D59: identity is structural).
#[test]
fn conflicting_import_shapes_fail_closed() {
    let clash = "\
import 0 \"write\" (i32) -> ()
func (i64) -> () {
block 0 (v0: i64) {
  v1 = i32.const 0
  call.import 0 (v1)
  return
  }
}
";
    let err = link_with_manifest(&[
        unit(RUNTIME_UNIT, &[("rt_emit", 0)]),
        unit(clash, &[("clash", 0)]),
    ])
    .expect_err("two structural meanings for one name");
    assert!(
        matches!(err, LinkError::ImportShapeMismatch(ref m) if m.contains("write")),
        "got {err:?}"
    );
}

/// The classic [`link`] keeps its contract: an import no unit exports is fail-closed at link —
/// manifest retention is opt-in ([`link_with_manifest`]), never a silent default.
#[test]
fn plain_link_still_fails_closed_on_capability_imports() {
    let err = link(&[unit(RUNTIME_UNIT, &[("rt_emit", 0)])]).expect_err("nothing exports write");
    assert_eq!(err, LinkError::Unresolved("write".into()));
}

/// `synth_manifest_start` validates its contract fail-closed: the entry must exist, take a single
/// `i64` (the data-stack pointer), and the module must not already carry a `_start` bootstrap.
#[test]
fn synth_rejects_bad_entries() {
    let m = svm_text::parse_module(PROGRAM_UNIT).expect("parse");
    assert!(
        synth_manifest_start(m.clone(), 9, false)
            .expect_err("out of range")
            .contains("out of range"),
        "entry index is validated"
    );
    let no_sp =
        svm_text::parse_module("func () -> () {\nblock 0 () {\n  return\n  }\n}\n").expect("parse");
    assert!(
        synth_manifest_start(no_sp, 0, false)
            .expect_err("wrong entry signature")
            .contains("single i64"),
        "entry must take the data-stack pointer"
    );
    let synthesized = synth_manifest_start(m, 0, false).expect("first synth");
    assert!(
        synth_manifest_start(synthesized, 1, false)
            .expect_err("already has a bootstrap")
            .contains("_start"),
        "double synthesis is rejected"
    );
}

/// `seed_heap` seeds the guest heap words ([`svm_ir::POWERBOX_HEAP_BRK`]/`_TOP`) to the window's
/// mapped boundary, and the window grows to cover the data-stack reserve — observed from inside
/// the guest, through the ordinary powerbox run path.
#[test]
fn seed_heap_seeds_the_heap_words() {
    let src = "\
memory 15
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 32
  v2 = i64.load v1
  return v2
  }
}
";
    let m = svm_text::parse_module(src).expect("parse");
    let module = synth_manifest_start(m, 0, true).expect("synth");
    let size_log2 = module.memory.expect("memory grown").size_log2;
    assert_eq!(
        size_log2, 21,
        "64 KiB stack base + 1 MiB reserve needs a 2 MiB window"
    );
    let instance = instantiate(module).expect("instantiate (no imports)");
    let run = instance.call("_start", &[]).expect("run");
    assert_eq!(
        run.outcome,
        Outcome::Returned(vec![Value::I64(1 << 21)]),
        "the heap bump pointer was seeded to the mapped boundary"
    );
}
