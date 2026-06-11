//! Spin-loop handling in the exhaustive model checker ([`explore_all`]). A busy-wait spinlock is the
//! classic pathological case for stateless model checking: a thread that keeps retrying a `cmpxchg`
//! (or re-reading a flag) makes every retry a fresh scheduling decision, so the interleaving tree is
//! unbounded — and an *unfair* schedule that keeps picking the spinner starves the lock holder until
//! fuel runs out, producing a spurious `OutOfFuel` outcome. Neither DPOR nor sleep sets help (the spin
//! read genuinely *conflicts* with the holder's release, so they're dependent).
//!
//! The checker now detects this dynamically: a visible op that changes no memory and returns the vCPU
//! to the same local configuration is a pure busy-wait, so the vCPU is **parked** (off the runnable
//! set) until another vCPU writes the address it was reading — exactly the semantics of the spin, with
//! none of the redundant decision points and no starvation. Sound (a stuttering thread's future is
//! fixed until shared memory it reads changes), which these tests check by reasoning about the
//! outcomes a spinlock-serialized program *must* produce.

use svm_interp::{explore_all, Value};
use svm_text::parse_module;
use svm_verify::verify_module;

const FUEL: u64 = 50_000_000;
const MAX: u64 = 5_000_000;

fn explore(src: &str) -> svm_interp::Exhaustive {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    explore_all(&m, 0, &[], FUEL, MAX)
}

/// A worker that takes the spinlock at `mem[0]` (a `cmpxchg`-retry loop — `block1` loops to itself on
/// failure, the single-visible-op spin the detector collapses), runs `body`, then releases. `$id` is
/// the function index its `body` text should not collide with.
macro_rules! spin_worker {
    ($body:literal) => {
        concat!(
            "func (i64, i64) -> (i64) {\n",
            "block0(vsp: i64, varg: i64):\n  br block1()\n",
            "block1():\n",
            "  vz = i32.const 0\n  vo = i32.const 1\n  vlock = i64.const 0\n",
            "  vold = i32.atomic.cmpxchg vlock vz vo\n",
            "  vok = i32.eq vold vz\n  br_if vok block2() block1()\n",
            "block2():\n",
            $body,
            "  vlock2 = i64.const 0\n  vrel = i32.const 0\n  i32.atomic.store vlock2 vrel\n",
            "  vret = i64.const 0\n  return vret\n}\n"
        )
    };
}

/// Two workers each increment the shared counter at `mem[8]` once under the lock; `main` returns it.
/// Mutual exclusion ⇒ no lost update ⇒ exactly 2 on **every** interleaving. Without spin-loop handling
/// this program is intractable (the spinner's retries are unbounded and an unfair schedule starves the
/// holder into `OutOfFuel`); with it, the whole tree is a handful of schedules.
const SPIN_COUNTER: &str = concat!(
    "memory 16\n",
    "func () -> (i64) {\n",
    "block0():\n",
    "  vsp = i64.const 100\n  va = i64.const 1\n",
    "  vh0 = thread.spawn 1 vsp va\n  vh1 = thread.spawn 1 vsp va\n",
    "  vj0 = thread.join vh0\n  vj1 = thread.join vh1\n",
    "  vc = i64.const 8\n  vr = i64.load vc\n  return vr\n}\n",
    spin_worker!("  vc = i64.const 8\n  vcur = i64.load vc\n  v1 = i64.const 1\n  vnew = i64.add vcur v1\n  i64.store vc vnew\n"),
    spin_worker!("  vc = i64.const 8\n  vcur = i64.load vc\n  v1 = i64.const 1\n  vnew = i64.add vcur v1\n  i64.store vc vnew\n"),
);

/// Asymmetric workers under the same lock: worker 1 does `counter += 1`, worker 2 does `counter *= 3`.
/// Mutual exclusion serializes them, but the **acquisition order** is nondeterministic:
/// 1-then-2 = `(0+1)*3 = 3`; 2-then-1 = `(0*3)+1 = 1`. The outcome set must be exactly {1, 3} — so this
/// proves the spin-park preserves the lock-acquisition-order nondeterminism (it drops no reachable
/// outcome, it only suppresses the redundant spinning).
const SPIN_ASYM: &str = concat!(
    "memory 16\n",
    "func () -> (i64) {\n",
    "block0():\n",
    "  vsp = i64.const 100\n  va = i64.const 1\n",
    "  vh0 = thread.spawn 1 vsp va\n  vh1 = thread.spawn 2 vsp va\n",
    "  vj0 = thread.join vh0\n  vj1 = thread.join vh1\n",
    "  vc = i64.const 8\n  vr = i64.load vc\n  return vr\n}\n",
    spin_worker!("  vc = i64.const 8\n  vcur = i64.load vc\n  v1 = i64.const 1\n  vnew = i64.add vcur v1\n  i64.store vc vnew\n"),
    spin_worker!("  vc = i64.const 8\n  vcur = i64.load vc\n  v3 = i64.const 3\n  vnew = i64.mul vcur v3\n  i64.store vc vnew\n"),
);

#[test]
fn spinlock_exhaustively_verifiable() {
    let r = explore(SPIN_COUNTER);
    assert!(
        r.complete,
        "spinlock exploration did not terminate ({} schedules) — spin-park regressed",
        r.schedules
    );
    assert_eq!(
        r.outcomes,
        vec![Ok(vec![Value::I64(2)])],
        "spinlock must mutually exclude (counter == 2 on every interleaving)"
    );
    // Spin-park collapses the busy-wait to a handful of schedules. Without it this is unbounded.
    assert!(
        r.schedules < 100,
        "expected the spin to collapse, got {} schedules",
        r.schedules
    );
}

#[test]
fn spinlock_preserves_acquisition_order_outcomes() {
    let r = explore(SPIN_ASYM);
    assert!(r.complete, "asymmetric spinlock did not terminate");
    let mut got: Vec<i64> = r
        .outcomes
        .iter()
        .map(|o| match o {
            Ok(v) => match v[0] {
                Value::I64(x) => x,
                _ => panic!("unexpected value {v:?}"),
            },
            Err(e) => panic!("unexpected trap {e:?} — spin-park starved a thread?"),
        })
        .collect();
    got.sort_unstable();
    assert_eq!(
        got,
        vec![1, 3],
        "both lock-acquisition orders must appear; spin-park dropped a reachable outcome"
    );
}
