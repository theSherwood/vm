//! §12/D57 **migratable fibers on the interpreter** (step 3b-i, SCHEDULING.md "Integration
//! design"): the per-vCPU fiber tables are replaced by one **run-shared registry**, so a fiber
//! created (and even part-run) on one vCPU can be claimed and continued by another — a safe
//! `Vec<Frame>` hand-off, exactly like the vCPUs the scheduler already migrates. The registry's
//! claim is the **single-owner arbiter**: of any racing `cont.resume`s, exactly one wins and a
//! loser gets a clean `FiberFault`.
//!
//! These pin the reference semantics — the oracle the JIT's lock-free registry (3b-ii) and
//! cross-thread asm resume (3c) will be differentially tested against. Each behavior runs on the
//! real M:N pool (`run`), the seeded explorer (`run_scheduled`), and the exhaustive checker
//! (`explore_all`), and the exhaustive outcome sets are cross-checked against the **unreduced**
//! brute-force enumerator — proving the DPOR `MemAccess::Fiber` conflict rule (fiber ops don't
//! commute) loses no interleaving.

use svm_interp::{explore_all, explore_all_bruteforce, run, run_scheduled, run_with_host};
use svm_interp::{Host, Quota, Trap, Value};
use svm_text::parse_module;
use svm_verify::verify_module;

fn module(src: &str) -> svm_ir::Module {
    let m = parse_module(src).unwrap_or_else(|e| panic!("parse failed: {e:?}\n{src}"));
    verify_module(&m).unwrap_or_else(|e| panic!("verify failed: {e:?}\n{src}"));
    m
}

/// Run func 0 on the real M:N pool and return the single i64 result (or the trap).
fn run_i64(src: &str) -> Result<i64, Trap> {
    let m = module(src);
    let mut fuel = 10_000_000u64;
    match run(&m, 0, &[], &mut fuel) {
        Ok(vals) => match vals.as_slice() {
            [Value::I64(v)] => Ok(*v),
            other => panic!("expected one i64 result, got {other:?}"),
        },
        Err(t) => Err(t),
    }
}

/// Exhaustively explore `src` and assert the outcome set — and that the DPOR checker and the
/// unreduced brute-force enumerator (the reduction-soundness oracle) agree on it exactly.
fn assert_outcomes(src: &str, want: &[Result<i64, Trap>]) {
    let m = module(src);
    let to_set = |ex: svm_interp::Exhaustive| -> Vec<Result<i64, Trap>> {
        assert!(ex.complete, "exploration must complete");
        let mut got: Vec<Result<i64, Trap>> = ex
            .outcomes
            .into_iter()
            .map(|r| {
                r.map(|vals| match vals.as_slice() {
                    [Value::I64(v)] => *v,
                    other => panic!("expected one i64 result, got {other:?}"),
                })
            })
            .collect();
        got.sort_by_key(|r| format!("{r:?}"));
        got
    };
    let mut want: Vec<Result<i64, Trap>> = want.to_vec();
    want.sort_by_key(|r| format!("{r:?}"));
    let dpor = to_set(explore_all(&m, 0, &[], 1_000_000, 200_000));
    let brute = to_set(explore_all_bruteforce(&m, 0, &[], 1_000_000, 200_000));
    assert_eq!(dpor, want, "DPOR outcome set\n{src}");
    assert_eq!(
        brute, want,
        "brute-force outcome set (DPOR reduction oracle)\n{src}"
    );
}

/// **The headline: a mid-life fiber migrates across vCPUs.** The root creates fiber F and resumes
/// it once — F suspends, capturing its first-resume argument (5) in its parked stack. The root
/// then hands F's *handle* to a spawned vCPU, which resumes it there: F continues past its
/// `suspend` **on the other vCPU** and returns `10*x + arg = 10*7 + 5 = 75` — the `+ 5` proving
/// the stack state captured on the root survived the migration intact (a restart would lose it).
const MIGRATE: &str = r#"
func () -> (i64) {
block0():
  v0 = ref.func 2
  v1 = i64.const 4096
  v2 = cont.new v0 v1
  v3 = i64.const 5
  v4, v5 = cont.resume v2 v3
  v6 = i64.extend_i32_u v2
  v7 = thread.spawn 1 v6 v6
  v8 = thread.join v7
  return v8
}
func (i64, i64) -> (i64) {
block0(vsp: i64, varg: i64):
  v0 = i32.wrap_i64 varg
  v1 = i64.const 7
  v2, v3 = cont.resume v0 v1
  return v3
}
func (i64, i64) -> (i64) {
block0(vsp: i64, varg: i64):
  v0 = suspend varg
  v1 = i64.const 10
  v2 = i64.mul v0 v1
  v3 = i64.add v2 varg
  return v3
}
"#;

#[test]
fn fiber_suspended_on_root_resumes_on_spawned_vcpu() {
    assert_eq!(run_i64(MIGRATE), Ok(75), "real M:N pool");
    for seed in 0..32 {
        let m = module(MIGRATE);
        assert_eq!(
            run_scheduled(&m, 0, &[], 10_000_000, seed),
            Ok(vec![Value::I64(75)]),
            "seeded explorer, seed {seed}"
        );
    }
    assert_outcomes(MIGRATE, &[Ok(75)]);
}

/// **Racing resumes: exactly one claimant wins.** The root creates one `Pending` fiber and spawns
/// two workers that both `cont.resume` it (the fiber returns `arg + 41 = 42` to whichever wins).
/// In *every* interleaving exactly one worker wins and the other's lost claim is a `FiberFault`
/// that propagates through the root's join — so the outcome set is exactly `{FiberFault}`.
/// Non-vacuous: if both claims could win, both joins would succeed and `Ok(84)` would appear; if
/// neither could, the fiber's 42 would never be computed (covered by the single-worker control).
const RACE: &str = r#"
func () -> (i64) {
block0():
  v0 = ref.func 2
  v1 = i64.const 0
  v2 = cont.new v0 v1
  v3 = i64.extend_i32_u v2
  v4 = thread.spawn 1 v3 v3
  v5 = thread.spawn 1 v3 v3
  v6 = thread.join v4
  v7 = thread.join v5
  v8 = i64.add v6 v7
  return v8
}
func (i64, i64) -> (i64) {
block0(vsp: i64, varg: i64):
  v0 = i32.wrap_i64 varg
  v1 = i64.const 1
  v2, v3 = cont.resume v0 v1
  return v3
}
func (i64, i64) -> (i64) {
block0(vsp: i64, varg: i64):
  v0 = i64.const 41
  v1 = i64.add varg v0
  return v1
}
"#;

#[test]
fn racing_resumes_have_exactly_one_winner() {
    assert_outcomes(RACE, &[Err(Trap::FiberFault)]);
    // The real pool agrees (whichever worker loses, the join propagates its fault).
    assert_eq!(run_i64(RACE), Err(Trap::FiberFault));
}

/// The single-worker control for [`RACE`]: with no competitor, the foreign claim **wins** — a
/// fiber created on the root starts and completes on the spawned vCPU (`1 + 41 = 42`). This is
/// the non-vacuity half: a foreign resume is genuinely a successful claim, not an always-fault.
const NO_RACE: &str = r#"
func () -> (i64) {
block0():
  v0 = ref.func 2
  v1 = i64.const 0
  v2 = cont.new v0 v1
  v3 = i64.extend_i32_u v2
  v4 = thread.spawn 1 v3 v3
  v5 = thread.join v4
  return v5
}
func (i64, i64) -> (i64) {
block0(vsp: i64, varg: i64):
  v0 = i32.wrap_i64 varg
  v1 = i64.const 1
  v2, v3 = cont.resume v0 v1
  return v3
}
func (i64, i64) -> (i64) {
block0(vsp: i64, varg: i64):
  v0 = i64.const 41
  v1 = i64.add varg v0
  return v1
}
"#;

#[test]
fn foreign_vcpu_claim_succeeds_without_a_race() {
    assert_eq!(run_i64(NO_RACE), Ok(42));
    assert_outcomes(NO_RACE, &[Ok(42)]);
}

/// **The fiber quota is per-run now** (the registry is run-shared; SCHEDULING.md §6): with
/// `max_fibers = 2` (the root computation + one creation), the root's `cont.new` fills the run's
/// budget, so a *spawned vCPU's* `cont.new` trips it — under the old per-vCPU tables the child's
/// fresh table would have admitted it (this is the non-vacuous pin of the semantic change).
#[test]
fn fiber_quota_spans_vcpus() {
    let src = r#"
func () -> (i64) {
block0():
  v0 = ref.func 2
  v1 = i64.const 0
  v2 = cont.new v0 v1
  v3 = thread.spawn 1 v1 v1
  v4 = thread.join v3
  return v4
}
func (i64, i64) -> (i64) {
block0(vsp: i64, varg: i64):
  v0 = ref.func 2
  v1 = cont.new v0 varg
  v2 = i64.extend_i32_u v1
  return v2
}
func (i64, i64) -> (i64) {
block0(vsp: i64, varg: i64):
  return varg
}
"#;
    let m = module(src);
    let run_quota = |max_fibers: usize| -> Result<Vec<Value>, Trap> {
        let mut host = Host::new();
        host.set_quota(Quota {
            max_fibers,
            max_vcpus: 1 << 16,
        });
        let mut fuel = 10_000_000u64;
        run_with_host(&m, 0, &[], &mut fuel, &mut host)
    };
    assert_eq!(
        run_quota(2),
        Err(Trap::FiberFault),
        "the child's cont.new must trip the run-wide quota the root already filled"
    );
    assert_eq!(
        run_quota(3),
        Ok(vec![Value::I64(1)]),
        "one more slot admits it — and the child's handle (1) continues the run's numbering"
    );
}
