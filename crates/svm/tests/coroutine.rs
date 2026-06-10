//! §14 co-fiber **resume/suspend** (the `Yielder`, iface 7) — a guest holding an `Instantiator`
//! `spawn_coroutine`s a child confined to a sub-window and drives it cooperatively: each `resume`
//! runs the child until it `yield`s (status SUSPENDED, handing back a value) or returns (status
//! RETURNED). The child and parent ping-pong on the **same** thread (the child runs inline, never on
//! the executor) — the cooperative-coroutine primitive the §14 parent-virtualized-fault / lazy-paging
//! model is built on. This pins the value hand-off both ways and the suspended/returned status
//! sequence, plus that the child stays confined to its slice across suspensions.

use svm_interp::{run_capture_reserved_with_host, Host, Trap, Value};
use svm_text::parse_module;
use svm_verify::verify_module;

const SUSPENDED: i32 = 0;
const RETURNED: i32 = 1;

/// func 0 (parent) spawns func 1 as a coroutine in a 4 KiB window at 64 KiB, then resumes it three
/// times — delivering 0, then 10, then 20 — collecting the value the child yields/returns each time,
/// plus the status of the last resume. Returns `y1*1 + y2*1 + y3 + status3*1_000_000` so a single
/// `i64` pins all four observations. The child yields `100`, then `200 + r1`, then returns `999 + r2`
/// (where `rN` is the value the parent passed on resume N+1). It also stores a marker into its own
/// window between yields, to confirm it stays confined across suspensions.
fn coro_src() -> &'static str {
    "memory 17
func (i32) -> (i64) {
block0(v0: i32):
  v1 = i64.const 1
  v2 = i64.const 65536
  v3 = i64.const 12
  v4 = i64.const 0
  v5 = cap.call 6 2 (i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4)
  v6 = i64.const 0
  v7, v8 = cap.call 6 3 (i32, i64) -> (i32, i64) v0 (v5, v6)
  v9 = i64.const 10
  v10, v11 = cap.call 6 3 (i32, i64) -> (i32, i64) v0 (v5, v9)
  v12 = i64.const 20
  v13, v14 = cap.call 6 3 (i32, i64) -> (i32, i64) v0 (v5, v12)
  v15 = i64.add v8 v11
  v16 = i64.add v15 v14
  v17 = i64.extend_i32_s v13
  v18 = i64.const 1000000
  v19 = i64.mul v17 v18
  v20 = i64.add v16 v19
  return v20
}
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i32.wrap_i64 v0
  v2 = i64.const 0
  v3 = i32.const 7
  i32.store8 v2 v3
  v4 = i64.const 100
  v5 = cap.call 7 0 (i64) -> (i64) v1 (v4)
  v6 = i64.const 200
  v7 = i64.add v6 v5
  v8 = cap.call 7 0 (i64) -> (i64) v1 (v7)
  v9 = i64.const 999
  v10 = i64.add v9 v8
  return v10
}
"
}

fn run(src: &str) -> (Result<Vec<Value>, Trap>, Vec<u8>) {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    let mut host = Host::new();
    let h = host.grant_instantiator(0, 128 << 10);
    let init: Vec<u8> = (0..(128u64 << 10))
        .map(|i| (i as u8).wrapping_mul(31) ^ 0xa5)
        .collect();
    let mut fuel = 5_000_000u64;
    run_capture_reserved_with_host(&m, 0, &[Value::I32(h)], &mut fuel, &init, 0, &mut host)
}

#[test]
fn coroutine_resume_suspend_round_trips_values() {
    let (res, mem) = run(coro_src());
    // y1 = 100 (first yield), y2 = 200 + 10 = 210 (second yield, r1 = 10), y3 = 999 + 20 = 1019
    // (return, r2 = 20); the last resume's status is RETURNED (1).
    let want = 100 + 210 + 1019 + (RETURNED as i64) * 1_000_000;
    assert_eq!(
        res.expect("run ok"),
        vec![Value::I64(want)],
        "coroutine value/status round-trip"
    );

    // The child's marker landed in *its* slice (window offset 64 KiB), and nowhere else outside it —
    // confinement held across the suspensions.
    const CHILD: u64 = 64 << 10;
    let init: Vec<u8> = (0..(128u64 << 10))
        .map(|i| (i as u8).wrapping_mul(31) ^ 0xa5)
        .collect();
    assert_eq!(mem[CHILD as usize], 7, "child marker missing");
    for i in 0..(128u64 << 10) {
        if !(CHILD..CHILD + 4096).contains(&i) {
            assert_eq!(
                mem[i as usize], init[i as usize],
                "coroutine escaped to parent byte {i}"
            );
        }
    }
}

/// The first resume's status is SUSPENDED (the child yields before returning) — pin it directly, so
/// the round-trip test above can't pass with a degenerate "child returns immediately" coroutine.
#[test]
fn coroutine_first_resume_suspends() {
    let src = "memory 17
func (i32) -> (i64) {
block0(v0: i32):
  v1 = i64.const 1
  v2 = i64.const 0
  v3 = i64.const 12
  v4 = i64.const 0
  v5 = cap.call 6 2 (i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4)
  v6 = i64.const 0
  v7, v8 = cap.call 6 3 (i32, i64) -> (i32, i64) v0 (v5, v6)
  v9 = i64.extend_i32_s v7
  return v9
}
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i32.wrap_i64 v0
  v2 = i64.const 42
  v3 = cap.call 7 0 (i64) -> (i64) v1 (v2)
  return v3
}
";
    let (res, _mem) = run(src);
    assert_eq!(
        res.expect("run ok"),
        vec![Value::I64(SUSPENDED as i64)],
        "first resume of a yielding coroutine must report SUSPENDED"
    );
}
