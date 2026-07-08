//! DURABILITY.md §4 **enforcement** — *"a durable domain admits only freezable modules and may
//! only spawn durable children"* (the one-flag check at instantiate/install; the first slice of
//! nested-guest durability). The freezable attestation is the host's, at grant time
//! (`Host::grant_durable_module` — instrumentation is a compile-mode fact the runtime cannot
//! re-derive from the IR, like verification):
//!
//! * a **durable** parent's `instantiate_module` (§14 op 5) of an *unmarked* grant is refused
//!   fail-closed (`-EINVAL` — the un-instrumented child could never drain-then-unwound, silently
//!   making the subtree non-snapshottable);
//! * the *same* module granted with the durable attestation instantiates and runs;
//! * a **same-module** child (op 0) stays admissible — it runs the parent's own (instrumented)
//!   funcs — and the child inherits the parent's durability (subtree property);
//! * a **non-durable** domain is entirely unaffected (`separate_module.rs` is the standing
//!   control); and a durable domain's guest-driven `Jit.compile` (§22 — also a module
//!   installation) fails closed until a host-side instrumentation hook exists.

use std::sync::Arc;
use svm_durable::{init_durable_window, transform_module};
use svm_interp::{run_capture_reserved_with_host, Host, Value};
use svm_ir::{Func, Module};
use svm_text::parse_module;
use svm_verify::verify_module;

const SIZE_LOG2: u8 = 18;
const WINDOW: usize = 1 << SIZE_LOG2;
const EINVAL: i64 = -22;

fn instrument(src: &str) -> Module {
    let m = parse_module(src).expect("parse");
    let inst = transform_module(&m).expect("transform");
    verify_module(&inst).expect("instrumented module verifies");
    inst
}

/// The child "plugin": 128 KiB window (its durable reserve + room), pure-compute entry
/// (`(i64) -> (i64)`, the starter-cap convention) returning 4321. Instrumented, so a durable
/// parent may admit it — *when the grant attests it*.
fn child() -> Module {
    instrument(
        "memory 17
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 4321
  return v1
}
",
    )
}

/// Durable parent that `instantiate_module`s (op 5) its granted child at the aligned carve
/// `[128 KiB, 256 KiB)` and returns the op's i32 status — the refusal probe (no join). The module
/// handle arrives as an `i64` entry arg (the op's slot ABI) since the Phase-1 durable transform
/// has no conversions.
const PARENT_PROBE: &str = "memory 18
func (i32, i64) -> (i32) {
block0(v0: i32, v1: i64):
  v2 = i64.const 0
  v3 = i64.const 131072
  v4 = i64.const 17
  v5 = cap.call 6 5 (i64, i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4, v2)
  return v5
}
";

/// Durable parent that instantiates its granted child and `join`s it (op 1) — the happy path.
const PARENT_JOIN: &str = "memory 18
func (i32, i64) -> (i64) {
block0(v0: i32, v1: i64):
  v2 = i64.const 0
  v3 = i64.const 131072
  v4 = i64.const 17
  v5 = cap.call 6 5 (i64, i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4, v2)
  v6 = cap.call 6 1 (i32) -> (i64) v0 (v5)
  return v6
}
";

/// Durable parent that instantiates a **same-module** child (op 0: its own func 1) and joins it.
const PARENT_SELF: &str = "memory 18
func (i32) -> (i64) {
block0(v0: i32):
  v1 = i64.const 1
  v2 = i64.const 131072
  v3 = i64.const 17
  v4 = i64.const 0
  v5 = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4)
  v6 = cap.call 6 1 (i32) -> (i64) v0 (v5)
  return v6
}
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 777
  return v1
}
";

/// Run a durable-instrumented `parent` with an `Instantiator` over the whole window plus the
/// given extra handle args, on a durable host over a durable window.
fn run_durable(parent: &Module, host: &mut Host, args: &[Value]) -> Vec<Value> {
    host.set_durable(true);
    let mut fuel = 5_000_000u64;
    let (r, _) = run_capture_reserved_with_host(
        parent,
        0,
        args,
        &mut fuel,
        &init_durable_window(WINDOW),
        SIZE_LOG2,
        host,
    );
    r.expect("run ok")
}

/// §4: a durable domain refuses to instantiate a module grant without the freezable attestation
/// (`grant_module`, not `grant_durable_module`) — fail-closed `-EINVAL`, like a bad carve.
#[test]
fn durable_domain_refuses_an_unmarked_module_grant() {
    let parent = instrument(PARENT_PROBE);
    let mut host = Host::new();
    let ih = host.grant_instantiator(0, WINDOW as u64);
    let mh = host.grant_module(&child()); // verified + instrumented, but NOT attested durable
    let r = run_durable(&parent, &mut host, &[Value::I32(ih), Value::I64(mh as i64)]);
    assert_eq!(
        r,
        vec![Value::I32(EINVAL as i32)],
        "an unmarked module grant must be refused in a durable domain"
    );
}

/// The same child granted **with** the durable attestation instantiates and runs to its result.
#[test]
fn durable_domain_admits_a_durable_module_grant() {
    let parent = instrument(PARENT_JOIN);
    let mut host = Host::new();
    let ih = host.grant_instantiator(0, WINDOW as u64);
    let mh = host.grant_durable_module(&child()); // host attests: transform_module ran
    let r = run_durable(&parent, &mut host, &[Value::I32(ih), Value::I64(mh as i64)]);
    assert_eq!(
        r,
        vec![Value::I64(4321)],
        "a durable-attested module child runs in a durable domain"
    );
}

/// A **same-module** child (op 0) needs no grant — it runs the parent's own instrumented funcs —
/// so a durable domain admits it unchanged (and it inherits the parent's durability).
#[test]
fn durable_domain_admits_a_same_module_child() {
    let parent = instrument(PARENT_SELF);
    let mut host = Host::new();
    let ih = host.grant_instantiator(0, WINDOW as u64);
    let r = run_durable(&parent, &mut host, &[Value::I32(ih)]);
    assert_eq!(
        r,
        vec![Value::I64(777)],
        "a same-module child runs in a durable domain"
    );
}

/// The rule is durable-domain-only: the identical unmarked grant instantiates fine in a
/// NON-durable domain (`separate_module.rs` is the broader standing control; this pins the
/// minimal pair against the refusal test above).
#[test]
fn non_durable_domain_admits_an_unmarked_module_grant() {
    let parent = instrument(PARENT_JOIN);
    let mut host = Host::new(); // NOT durable
    let ih = host.grant_instantiator(0, WINDOW as u64);
    let mh = host.grant_module(&child());
    let mut fuel = 5_000_000u64;
    let (r, _) = run_capture_reserved_with_host(
        &parent,
        0,
        &[Value::I32(ih), Value::I64(mh as i64)],
        &mut fuel,
        &init_durable_window(WINDOW),
        SIZE_LOG2,
        &mut host,
    );
    assert_eq!(
        r.expect("run ok"),
        vec![Value::I64(4321)],
        "an unmarked grant instantiates in a non-durable domain"
    );
}

/// §22 guest-driven JIT is a module installation too: a durable domain's `Jit.compile` fails
/// closed (`-EINVAL`) even with a working validator installed — an un-instrumented unit could
/// never drain-then-unwind. The identical non-durable host compiles the same bytes fine.
#[test]
fn durable_domain_refuses_guest_jit_compile() {
    fn trivial_validator(
        _bytes: &[u8],
        _mem: Option<u8>,
        _symtab: &[u8],
    ) -> Result<Arc<[Func]>, i64> {
        let m = parse_module(
            "func () -> (i64) {
block0():
  v0 = i64.const 1
  return v0
}
",
        )
        .expect("parse unit");
        Ok(m.funcs.into())
    }

    // Control: the same validator + grant on a NON-durable host compiles.
    let mut host = Host::new();
    host.set_jit_validator(trivial_validator);
    let jh = host.grant_jit(None);
    assert!(
        host.jit_compile(jh, b"unit").expect("no trap").is_ok(),
        "non-durable: the unit compiles"
    );

    // Durable: same setup, compile fails closed before any unit is stored.
    let mut host = Host::new();
    host.set_durable(true);
    host.set_jit_validator(trivial_validator);
    let jh = host.grant_jit(None);
    assert!(
        matches!(host.jit_compile(jh, b"unit").expect("no trap"), Err(e) if e == EINVAL),
        "durable: guest JIT compile is refused fail-closed"
    );
}
