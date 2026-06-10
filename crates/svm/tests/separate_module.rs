//! §14 **separate-module children** — the "plugin-in-plugin" story (interpreter side; the interp↔JIT
//! differential lives in `jit_separate_module.rs`). The host verifies a *different* module and grants
//! the parent a **`Module` capability** (iface 8); the parent passes it to the `Instantiator`'s
//! module ops (5 `instantiate_module` / 6 `spawn_coroutine_module` / 7
//! `spawn_demand_coroutine_module`) to spawn a child domain running *that* module, confined to a
//! carve of the parent's window. The child's **data segments materialize into the carve at spawn**
//! (its string literals just work), the carve must **equal the module's declared memory** (§14
//! transparency: the plugin behaves exactly as it would standalone), and a forged module handle is an
//! inert `CapFault`. A demand-paged module child gets its data segments **supplied lazily** — its
//! first touch of a segment page faults to the parent, which simply resumes (the bytes are already in
//! the shared backing): the §14 parent-as-pager model with zero extra machinery.

use svm_interp::{run_capture_reserved_with_host, Host, Trap, Value};
use svm_text::parse_module;
use svm_verify::verify_module;

const RETURNED: i64 = 1;
const FAULTED: i64 = 2;

/// The child ("plugin") module: 64 KiB window, a data segment `"VM"` at offset 100. Its entry
/// (`(i64) -> (i64)`, the starter-cap convention) loads its own data byte at 100, stores a marker at
/// offset 0, and returns `byte + 1000` — exercising code, data, and window writes of a foreign module.
fn child_src() -> &'static str {
    "memory 16
data 100 \"VM\"
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 100
  v2 = i32.load8_u v1
  v3 = i64.const 0
  v4 = i32.const 7
  i32.store8 v3 v4
  v5 = i64.extend_i32_u v2
  v6 = i64.const 1000
  v7 = i64.add v5 v6
  return v7
}
"
}

/// Run `parent_src`'s entry with `(instantiator over the whole window, module handle for child_src)`
/// as its two args, over a seeded 128 KiB window; return the result and final window.
fn run_with_child(parent_src: &str) -> (Result<Vec<Value>, Trap>, Vec<u8>) {
    let parent = parse_module(parent_src).expect("parse parent");
    verify_module(&parent).expect("verify parent");
    let child = parse_module(child_src()).expect("parse child");
    verify_module(&child).expect("verify child");

    let mut host = Host::new();
    let ih = host.grant_instantiator(0, 128 << 10);
    let mh = host.grant_module(&child);
    let init: Vec<u8> = (0..(128u64 << 10))
        .map(|i| (i as u8).wrapping_mul(31) ^ 0xa5)
        .collect();
    let mut fuel = 5_000_000u64;
    run_capture_reserved_with_host(
        &parent,
        0,
        &[Value::I32(ih), Value::I32(mh)],
        &mut fuel,
        &init,
        0,
        &mut host,
    )
}

#[test]
fn module_child_runs_with_its_data_segments() {
    // instantiate_module(module, entry 0, off 64 KiB, size_log2 16, fuel 0) → join → child's result.
    let parent = "memory 17
func (i32, i32) -> (i64) {
block0(v0: i32, v1: i32):
  v2 = i64.extend_i32_s v1
  v3 = i64.const 0
  v4 = i64.const 65536
  v5 = i64.const 16
  v6 = cap.call 6 5 (i64, i64, i64, i64, i64) -> (i32) v0 (v2, v3, v4, v5, v3)
  v7 = cap.call 6 1 (i32) -> (i64) v0 (v6)
  return v7
}
";
    let (res, mem) = run_with_child(parent);
    // The child read its own data segment ('V' = 86) — a foreign module's code + data ran confined.
    assert_eq!(
        res.expect("run ok"),
        vec![Value::I64(86 + 1000)],
        "module child result"
    );
    // Its data segment materialized into the carve (the parent sees it — the §14 superset)…
    const CHILD: u64 = 64 << 10;
    assert_eq!(&mem[(CHILD + 100) as usize..(CHILD + 102) as usize], b"VM");
    // …its marker landed at its offset 0, and nothing outside the carve changed.
    assert_eq!(mem[CHILD as usize], 7, "child marker missing");
    let init: Vec<u8> = (0..(128u64 << 10))
        .map(|i| (i as u8).wrapping_mul(31) ^ 0xa5)
        .collect();
    for i in 0..CHILD {
        assert_eq!(
            mem[i as usize], init[i as usize],
            "module child escaped to byte {i}"
        );
    }
}

#[test]
fn module_child_carve_must_match_declared_memory() {
    // A 4 KiB carve for a module that declares 64 KiB → -EINVAL (§14 transparency: the plugin runs
    // with exactly its declared window, or not at all).
    let parent = "memory 17
func (i32, i32) -> (i64) {
block0(v0: i32, v1: i32):
  v2 = i64.extend_i32_s v1
  v3 = i64.const 0
  v4 = i64.const 65536
  v5 = i64.const 12
  v6 = cap.call 6 5 (i64, i64, i64, i64, i64) -> (i32) v0 (v2, v3, v4, v5, v3)
  v7 = i64.extend_i32_s v6
  return v7
}
";
    let (res, _mem) = run_with_child(parent);
    assert_eq!(
        res.expect("run ok"),
        vec![Value::I64(-22)],
        "a carve ≠ the module's declared memory must be -EINVAL"
    );
}

#[test]
fn forged_module_handle_capfaults() {
    // Passing the *Instantiator* handle where a Module handle is expected (wrong iface) → CapFault.
    let parent = "memory 17
func (i32, i32) -> (i64) {
block0(v0: i32, v1: i32):
  v2 = i64.extend_i32_s v0
  v3 = i64.const 0
  v4 = i64.const 65536
  v5 = i64.const 16
  v6 = cap.call 6 5 (i64, i64, i64, i64, i64) -> (i32) v0 (v2, v3, v4, v5, v3)
  v7 = i64.extend_i32_s v6
  return v7
}
";
    let (res, _mem) = run_with_child(parent);
    assert!(
        matches!(res, Err(Trap::CapFault)),
        "a forged module handle must CapFault, got {res:?}"
    );
}

#[test]
fn demand_module_child_gets_data_segments_lazily() {
    // spawn_demand_coroutine_module: the child's data segments are in the shared backing, but its
    // pages start unmapped — its first read of the segment FAULTs to the parent, which supplies the
    // page by simply resuming (the bytes are already there). Lazy plugin loading, for free.
    let parent = "memory 17
func (i32, i32) -> (i64) {
block0(v0: i32, v1: i32):
  v2 = i64.extend_i32_s v1
  v3 = i64.const 0
  v4 = i64.const 65536
  v5 = i64.const 16
  v6 = cap.call 6 7 (i64, i64, i64, i64, i64) -> (i32) v0 (v2, v3, v4, v5, v3)
  v7, v8 = cap.call 6 3 (i32, i64) -> (i32, i64) v0 (v6, v3)
  v9, v10 = cap.call 6 3 (i32, i64) -> (i32, i64) v0 (v6, v3)
  v11 = i64.extend_i32_s v7
  v12 = i64.const 1000000
  v13 = i64.mul v11 v12
  v14 = i64.extend_i32_s v9
  v15 = i64.const 100000
  v16 = i64.mul v14 v15
  v17 = i64.add v10 v13
  v18 = i64.add v17 v16
  return v18
}
";
    let (res, _mem) = run_with_child(parent);
    // First resume: FAULTED (status 2) at the child's first touched page; second: RETURNED (1) with
    // the child's result 1086 ('V' + 1000) — the segment byte arrived lazily. The fault *address*
    // (v8) is intentionally unasserted here; the interp↔JIT differential pins it byte-exactly.
    let want = 1086 + FAULTED * 1_000_000 + RETURNED * 100_000;
    assert_eq!(
        res.expect("run ok"),
        vec![Value::I64(want)],
        "lazy module-child data supply"
    );
}
