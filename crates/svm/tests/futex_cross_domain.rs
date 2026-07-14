//! PROCESS.md S0 spike (O2): do futex `wait`/`notify` rendezvous **across domains** on shared
//! backing? The §12 futex key is the *confined absolute* address (`Mem::confine_checked` returns
//! `window.base() + offset`, indexing the possibly-parent-sized backing), and a §14 nested child
//! shares the parent's backing `Arc` (`Mem::nested_view`) and the parent's executor — so a parent
//! and its nested child should rendezvous on the *same byte* under *different guest addresses*
//! (parent: `carve_off + a`; child: `a`). These tests pin that this works by construction on the
//! interpreter — the load-bearing fact for building endpoints as a **library** over shared memory
//! + futex (PROCESS.md §4 "library first") instead of new runtime rendezvous machinery.
//!
//! Scope note (the S0 boundary, recorded): this covers the **nested** (same-backing) case only.
//! Two *siblings* aliasing a §13 `SharedRegion` mapped at different window offsets would compute
//! *different* keys (each keys on its own window-absolute alias address, not the region's canonical
//! byte) — the sibling-pipe case needs its own slice if library endpoints are to span it.

use svm_interp::{run_capture_reserved_with_host, Host, Trap, Value};
use svm_text::parse_module;
use svm_verify::verify_module;

/// Run `src`'s func 0 with an `Instantiator` over the whole `1<<win_log2` window.
fn run(src: &str, win_log2: u8, fuel: u64) -> Result<Vec<Value>, Trap> {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    let mut host = Host::new();
    let h = host.grant_instantiator(0, 1u64 << win_log2);
    let init = vec![0u8; 1usize << win_log2];
    let mut f = fuel;
    run_capture_reserved_with_host(&m, 0, &[Value::I32(h)], &mut f, &init, 0, &mut host).0
}

/// Deterministic rendezvous — the key-space proof. The parent instantiates a child in a 64 KiB
/// carve at 64 KiB, then parks on `atomic.wait` at **parent** address `64K + 4096` (expected 0,
/// no store ever changes it). The child spin-`notify`s **its** address `4096` — the same backing
/// byte under a different guest address — until the notify reports a waiter woken. If the keys
/// did not collide, the parent would sleep to the wait cap and the child would spin to fuel
/// exhaustion; instead the parent wakes (status 0 = woken, never 1 = not-equal since the value
/// stays 0) and joins the child's 7.
///
/// Ordering is safe on every executor shape: on a single cooperative worker the parent runs
/// until it parks (instantiate enqueues, no switch), so the child's first notify finds it; with
/// real parallel workers the child spins until the park lands.
#[test]
fn parent_waits_child_notifies_across_domains() {
    const SRC: &str = "memory 17\n\
func (i32) -> (i64) {\n\
block0(v0: i32):\n\
  v1 = i64.const 1\n\
  v2 = i64.const 65536\n\
  v3 = i64.const 16\n\
  v4 = i64.const 0\n\
  v5 = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4)\n\
  v6 = i64.const 69632\n\
  v7 = i32.const 0\n\
  v8 = i64.const -1\n\
  v9 = i32.atomic.wait v6 v7 v8\n\
  v10 = cap.call 6 1 (i32) -> (i64) v0 (v5)\n\
  v11 = i64.extend_i32_u v9\n\
  v12 = i64.const 100\n\
  v13 = i64.mul v11 v12\n\
  v14 = i64.add v13 v10\n\
  return v14\n\
}\n\
func (i64) -> (i64) {\n\
block0(v0: i64):\n\
  br block1()\n\
block1():\n\
  v1 = i64.const 4096\n\
  v2 = i32.const 1\n\
  v3 = atomic.notify v1 v2\n\
  v4 = i32.const 0\n\
  v5 = i32.lt_u v4 v3\n\
  br_if v5 block2() block1()\n\
block2():\n\
  v6 = i64.const 7\n\
  return v6\n\
}\n";
    // status 0 (woken) * 100 + child result 7 — any other status means the keys diverged.
    assert_eq!(
        run(SRC, 17, 50_000_000).expect("run ok"),
        vec![Value::I64(7)],
        "parent must be woken (status 0) by the child's notify of the same backing byte"
    );
}

/// The inverse direction, racy by nature (the parent cannot spin-notify without starving a
/// single-worker executor, so it stores + notifies once). Child `atomic.wait`s its address 4096
/// (expected 0); parent stores 1 and notifies once at its alias `64K + 4096`, then joins. The
/// futex compare-and-park makes every interleaving safe, and both legal outcomes prove the
/// cross-domain semantics:
///   child parked first  → notify wakes it:   child status 0, parent saw 1 woken → result  1
///   store landed first  → compare sees 1:    child status 1, notify woke 0      → result 10
/// A hang or any other value means the shared-backing futex contract is broken.
#[test]
fn child_waits_parent_notifies_across_domains() {
    const SRC: &str = "memory 17\n\
func (i32) -> (i64) {\n\
block0(v0: i32):\n\
  v1 = i64.const 1\n\
  v2 = i64.const 65536\n\
  v3 = i64.const 16\n\
  v4 = i64.const 0\n\
  v5 = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4)\n\
  v6 = i64.const 69632\n\
  v7 = i32.const 1\n\
  i32.atomic.store.release v6 v7\n\
  v8 = atomic.notify v6 v7\n\
  v9 = cap.call 6 1 (i32) -> (i64) v0 (v5)\n\
  v10 = i64.const 10\n\
  v11 = i64.mul v9 v10\n\
  v12 = i64.extend_i32_u v8\n\
  v13 = i64.add v11 v12\n\
  return v13\n\
}\n\
func (i64) -> (i64) {\n\
block0(v0: i64):\n\
  v1 = i64.const 4096\n\
  v2 = i32.const 0\n\
  v3 = i64.const -1\n\
  v4 = i32.atomic.wait v1 v2 v3\n\
  v5 = i64.extend_i32_u v4\n\
  return v5\n\
}\n";
    let r = run(SRC, 17, 50_000_000).expect("run ok");
    assert!(
        r == vec![Value::I64(1)] || r == vec![Value::I64(10)],
        "expected (woken,1)=1 or (not-equal,0)=10, got {r:?}"
    );
}
