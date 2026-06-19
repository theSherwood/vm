//! Phase-3.2 — **vCPU-context recycling** (interpreter). A durable domain assigns each spawned
//! `thread.spawn` child a shadow context growing down from `MAX_SHADOW_CTX` (~15); before recycling
//! those grew monotonically, capping a durable domain at ~15 **lifetime** spawns. Now a child's
//! context is freed back to the registry when it genuinely finishes (a `u16` occupancy mask + a
//! free-on-finish hook), so the bound is *peak concurrent* vCPUs.
//!
//! **Freeze/thaw note.** The thaw reconstruction is now gap-tolerant *by construction* — it derives
//! each re-spawned child's context from its restored shadow-SP and seeds the registry's occupancy mask
//! from that, rather than assuming the live children occupy the top `n` contexts densely. But a freeze
//! that *captures a recycled-context child* is not reachable today: a freeze-from-start drives every
//! vCPU to `UNWINDING` at t=0 (so no child finishes-and-recycles — the residue is always dense), and a
//! *mid-run* multi-vCPU freeze would need a true stop-the-world (the `arm_freeze_after` trigger only
//! promotes the **running** vCPU's per-context state word, so a sibling keeps its own state and never
//! freezes). That STW is Phase-4 work; until it lands, the gap path is exercised only on the dense path
//! (`multivcpu.rs`, `roundtrip.rs`), and this file pins the reachable headline: the lifted cap.

use svm_durable::{init_durable_window, transform_module_assume_confined};
use svm_interp::{run_capture_reserved_with_host, Host, Value};
use svm_ir::{Memory, Module};

const SIZE_LOG2: u8 = 17;
const WINDOW: usize = 1 << SIZE_LOG2;

fn instrument(src: &str) -> Module {
    let mut m = svm_text::parse_module(src).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
    });
    let inst = transform_module_assume_confined(&m).expect("transform");
    svm_verify::verify_module(&inst).expect("instrumented IR verifies");
    inst
}

// A durable root that spawns + joins a trivial child 20 times in a loop (well past the ~15 lifetime
// ceiling). Each child returns 1 and finishes, freeing its shadow context; the next spawn reuses it.
// Without recycling the 16th `thread.spawn` would `ThreadFault` (the vCPU pool growing down meets the
// fiber pool growing up). Result = 20.
const LOOP_SRC: &str = r#"
func () -> (i64) {
block0():
  v0 = i64.const 0
  v1 = i64.const 0
  br block1(v0, v1)
block1(v2: i64, v3: i64):
  v4 = i64.const 20
  v5 = i64.lt_u v2 v4
  br_if v5 block2(v2, v3) block3(v3)
block2(v6: i64, v7: i64):
  v8 = i64.const 0
  v9 = i64.const 0
  v10 = thread.spawn 1 v8 v9
  v11 = thread.join v10
  v12 = i64.add v7 v11
  v13 = i64.const 1
  v14 = i64.add v6 v13
  br block1(v14, v12)
block3(v15: i64):
  return v15
}
func (i64, i64) -> (i64) {
block0(v0: i64, v1: i64):
  v2 = i64.const 1
  return v2
}
"#;

#[test]
fn recycling_lifts_the_lifetime_spawn_cap() {
    let inst = instrument(LOOP_SRC);
    let mut h = Host::new();
    h.set_durable(true); // durable ⇒ each spawn reserves a shadow context (the capped resource)
    let mut fuel = 10_000_000u64;
    let (r, _) = run_capture_reserved_with_host(
        &inst,
        0,
        &[],
        &mut fuel,
        &init_durable_window(WINDOW),
        SIZE_LOG2,
        &mut h,
    );
    assert_eq!(
        r,
        Ok(vec![Value::I64(20)]),
        "20 sequential spawn/join cycles complete — joined children's contexts recycle \
         (without recycling the 16th spawn would ThreadFault)"
    );
}

// Two children live at once (peak concurrent = 2), 8 iterations ⇒ 16 lifetime spawns. Each iteration
// reserves two top contexts and frees them on join, so the next iteration reuses them — exercising the
// multi-bit occupancy mask (and the lowest-occupied frontier) as well as the recycling. Without
// recycling the 16th lifetime spawn would `ThreadFault`. Handles ride registers (no memory). Result =
// 16.
const PAIR_SRC: &str = r#"
func () -> (i64) {
block0():
  v0 = i64.const 0
  v1 = i64.const 0
  br block1(v0, v1)
block1(v2: i64, v3: i64):
  v4 = i64.const 8
  v5 = i64.lt_u v2 v4
  br_if v5 block2(v2, v3) block3(v3)
block2(v6: i64, v7: i64):
  v8 = i64.const 0
  v9 = thread.spawn 1 v8 v8
  v10 = thread.spawn 1 v8 v8
  v11 = thread.join v9
  v12 = thread.join v10
  v13 = i64.add v7 v11
  v14 = i64.add v13 v12
  v15 = i64.const 1
  v16 = i64.add v6 v15
  br block1(v16, v14)
block3(v17: i64):
  return v17
}
func (i64, i64) -> (i64) {
block0(v0: i64, v1: i64):
  v2 = i64.const 1
  return v2
}
"#;

#[test]
fn recycling_handles_concurrent_contexts() {
    let inst = instrument(PAIR_SRC);
    let mut h = Host::new();
    h.set_durable(true);
    let mut fuel = 50_000_000u64;
    let (r, _) = run_capture_reserved_with_host(
        &inst,
        0,
        &[],
        &mut fuel,
        &init_durable_window(WINDOW),
        SIZE_LOG2,
        &mut h,
    );
    assert_eq!(
        r,
        Ok(vec![Value::I64(16)]),
        "8 iterations × 2 concurrent children (16 lifetime, 2 peak) — each iteration reuses the \
         previous pair's freed contexts (without recycling the 16th spawn would ThreadFault)"
    );
}
