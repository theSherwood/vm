//! Reference stop-the-world **quiesce barrier** for guest-coordinated GC (GC.md §2.1).
//!
//! svm provides **no** world-stop primitive — STW is pure guest code over the existing window
//! atomics + futex (`i32.atomic.load.acquire` / `i32.atomic.store.release` / `i32.atomic.wait` /
//! `atomic.notify`) and `thread.spawn`/`thread.join`. This test *is* the reference: it builds the
//! barrier parametrically in `N` vCPUs and exercises it end-to-end on **both** backends, so a guest
//! (e.g. JACL) has a tested pattern to copy. svm wraps only the collector↔vCPU rendezvous; the
//! safepoint poll + park lives in the guest scheduler (here, the mutator vCPUs).
//!
//! Protocol (single-writer i32 window slots, so **no atomic read-modify-write** is needed):
//!   - The collector bumps `EPOCH` and waits until every mutator's `parked[i] == 1`.
//!   - Each mutator runs a bounded "work" loop (incrementing `work[i]`, polling `STOPPED` + `EPOCH`
//!     at the top — its safepoint); on an `EPOCH` change (or loop end) it sets `parked[i]`, notifies,
//!     and blocks on `RELEASE`.
//!   - In the stopped window the collector raises `STOPPED` (a real collector would call `gc.roots`
//!     here), then clears it, stores `RELEASE`, and notifies. Mutators wake, exit, and are joined.
//!
//! Two soundness checks, both expected `0`: `VIOLATION` (a mutator sets it if it ever observes
//! `STOPPED == 1` during work — impossible if the barrier holds, since all mutators are blocked on
//! `RELEASE` before the collector raises `STOPPED`), and `WORKFAIL` (the collector sets it if a
//! parked mutator's `work[i]` is `0` — i.e. the `parked[i]` release/acquire failed to publish the
//! mutator's work, a happens-before bug). Run for N=2 and N=4. Both backends (soundness, not
//! interp↔JIT equality — GC.md §3.2); gated to the stack-switch substrate like the cross-vCPU
//! `gc.roots` test.

#![cfg(any(
    all(unix, target_arch = "x86_64"),
    all(unix, target_arch = "aarch64"),
    all(windows, target_arch = "x86_64")
))]

use std::fmt::Write as _;

use svm_interp::{run, Value};
use svm_text::parse_module;
use svm_verify::verify_module;

// Window byte offsets of the barrier's i32 slots (each a single-writer flag; `HANDLE`/`parked`/`work`
// are per-mutator arrays). All well within the 64 KiB window (`memory 16`).
const EPOCH: i64 = 256; // collector → mutators: a stop is requested (0 → 1)
const RELEASE: i64 = 260; // collector → mutators: resume (0 → 1)
const STOPPED: i64 = 264; // collector: 1 only inside the stopped window (mutual-exclusion probe)
const VIOLATION: i64 = 268; // mutator → collector: a mutator ran work while STOPPED == 1 (must stay 0)
const WORKFAIL: i64 = 272; // collector: a parked mutator's work[i] was 0 (must stay 0)
const HANDLE_BASE: i64 = 320; // collector: thread handles, HANDLE_BASE + i*4
const PARKED_BASE: i64 = 512; // mutator i → collector: parked[i]
const WORK_BASE: i64 = 1024; // mutator i: work[i] progress counter
                             // Mutators run a bounded work loop (the literal `64` in `MUTATOR`'s `block4`) so loop-end is a
                             // fallback safepoint if the collector's `EPOCH` bump is somehow missed — keeps the test hang-proof.
const TIMEOUT: i64 = 1_000_000_000; // 1s futex timeout — every wait re-checks its condition (hang-proof)
const WAKE_ALL: i64 = 1_000_000_000; // notify count: wake every waiter

/// A fresh, function-unique value name (the text parser requires every `vN`/name in a function to be
/// distinct, even across blocks).
fn fresh(n: &mut usize) -> String {
    let s = format!("c{n}");
    *n += 1;
    s
}

/// The shared mutator body (func 1, the thread entry `(i64 sp, i64 me) -> i64`). `me` indexes the
/// per-mutator `parked[me]`/`work[me]` slots. Blocks: 0 entry, 1 work-top (safepoint), 2 set-violation,
/// 3 work-body, 4 loop-check, 5 park, 6 release-wait, 7 release-wait-block, 8 done.
const MUTATOR: &str = "\
func (i64, i64) -> (i64) {
block 0 (msp: i64, mme: i64) {
  mfour = i64.const 4
  moff = i64.mul mme mfour
  mwb = i64.const 1024
  mwa = i64.add mwb moff
  mpb = i64.const 512
  mpa = i64.add mpb moff
  mc0 = i32.const 0
  br 1(mwa, mpa, mc0)
}
block 1 (twa: i64, tpa: i64, tc: i32) {
  tsa = i64.const 264
  tst = i32.atomic.load.acquire tsa
  tone = i32.const 1
  tis = i32.eq tst tone
  br_if tis 2(twa, tpa, tc) 3(twa, tpa, tc)
}
block 2 (swa: i64, spa: i64, sc: i32) {
  sva = i64.const 268
  sone = i32.const 1
  i32.atomic.store.release sva sone
  br 3(swa, spa, sc)
}
block 3 (bwa: i64, bpa: i64, bc: i32) {
  bone = i32.const 1
  bcv = i32.add bc bone
  i32.atomic.store.release bwa bcv
  bea = i64.const 256
  be = i32.atomic.load.acquire bea
  bz = i32.const 0
  bchg = i32.ne be bz
  br_if bchg 5(bpa) 4(bwa, bpa, bcv)
}
block 4 (kwa: i64, kpa: i64, kc: i32) {
  kn = i32.const 64
  klt = i32.lt_s kc kn
  br_if klt 1(kwa, kpa, kc) 5(kpa)
}
block 5 (ppa: i64) {
  pone = i32.const 1
  i32.atomic.store.release ppa pone
  pcnt = i32.const 1
  pnfy = atomic.notify ppa pcnt
  br 6()
}
block 6 () {
  rra = i64.const 260
  rr = i32.atomic.load.acquire rra
  rone = i32.const 1
  riseq = i32.eq rr rone
  br_if riseq 8() 7(rr)
}
block 7 (wr: i32) {
  wra = i64.const 260
  wto = i64.const 1000000000
  wres = i32.atomic.wait wra wr wto
  br 6()
}
block 8 () {
  dz = i64.const 0
  return dz
  }
}
";

/// Build the collector (func 0, entry `() -> (i64, i64)` returning `(VIOLATION, WORKFAIL)`) for
/// `m = n_vcpus - 1` mutators, followed by the shared [`MUTATOR`]. The collector's blocks are
/// self-contained (state lives in window slots, not block params), so each just recomputes its
/// constant addresses and branches. Block indices: 0 = spawn; per mutator i a 4-block wait-parked
/// group at `1 + 4i`; `stw` at `1 + 4m`; per mutator i a join block at `2 + 4m + i`; final at `2 + 5m`.
fn build_quiesce_module(n_vcpus: usize) -> String {
    assert!(n_vcpus >= 2);
    let m = (n_vcpus - 1) as i64; // mutator count
    let mut nv = 0usize;
    let mut s = String::from("memory 16\nfunc () -> (i64, i64) {\n");

    let wp = |i: i64| 1 + 4 * i; // wait-parked head for mutator i
    let wpw = |i: i64| 2 + 4 * i; // wait-parked block (futex)
    let wpc = |i: i64| 3 + 4 * i; // work>0 check
    let setwf = |i: i64| 4 + 4 * i; // set WORKFAIL
    let stw = 1 + 4 * m; // stopped-the-world block
    let join = |i: i64| 2 + 4 * m + i; // join block for mutator i
    let final_blk = 2 + 5 * m;
    // The block reached after mutator i's parked-check succeeds (next mutator, or the STW block).
    let next_after = |i: i64| if i < m - 1 { wp(i + 1) } else { stw };

    // ---- block0: spawn the mutators, request the stop, wake any early waiters. ----
    s.push_str("block 0 () {\n");
    let sp = fresh(&mut nv);
    writeln!(s, "  {sp} = i64.const 4096").unwrap();
    for i in 0..m {
        let arg = fresh(&mut nv);
        let h = fresh(&mut nv);
        let ha = fresh(&mut nv);
        writeln!(s, "  {arg} = i64.const {i}").unwrap();
        writeln!(s, "  {h} = thread.spawn 1 {sp} {arg}").unwrap();
        writeln!(s, "  {ha} = i64.const {}", HANDLE_BASE + i * 4).unwrap();
        writeln!(s, "  i32.atomic.store.release {ha} {h}").unwrap();
    }
    let ea = fresh(&mut nv);
    let one = fresh(&mut nv);
    writeln!(s, "  {ea} = i64.const {EPOCH}").unwrap();
    writeln!(s, "  {one} = i32.const 1").unwrap();
    writeln!(s, "  i32.atomic.store.release {ea} {one}").unwrap();
    let ea2 = fresh(&mut nv);
    let wake = fresh(&mut nv);
    let nfy = fresh(&mut nv);
    writeln!(s, "  {ea2} = i64.const {EPOCH}").unwrap();
    writeln!(s, "  {wake} = i32.const {WAKE_ALL}").unwrap();
    writeln!(s, "  {nfy} = atomic.notify {ea2} {wake}").unwrap();
    writeln!(s, "  br {}()", wp(0)).unwrap();
    writeln!(s, "  }}").unwrap();

    // ---- per-mutator: wait until parked[i] == 1, then confirm work[i] != 0. ----
    for i in 0..m {
        let pa_off = PARKED_BASE + i * 4;
        let wa_off = WORK_BASE + i * 4;

        // wp(i): spin/wait until parked[i] == 1.
        writeln!(s, "block {} () {{", wp(i)).unwrap();
        let pa = fresh(&mut nv);
        let p = fresh(&mut nv);
        let one = fresh(&mut nv);
        let iseq = fresh(&mut nv);
        writeln!(s, "  {pa} = i64.const {pa_off}").unwrap();
        writeln!(s, "  {p} = i32.atomic.load.acquire {pa}").unwrap();
        writeln!(s, "  {one} = i32.const 1").unwrap();
        writeln!(s, "  {iseq} = i32.eq {p} {one}").unwrap();
        writeln!(s, "  br_if {iseq} {}() {}()", wpc(i), wpw(i)).unwrap();

        // wpw(i): block on parked[i] (futex), then re-check.
        writeln!(s, "  }}").unwrap();
        writeln!(s, "block {} () {{", wpw(i)).unwrap();
        let pa2 = fresh(&mut nv);
        let z = fresh(&mut nv);
        let to = fresh(&mut nv);
        let wres = fresh(&mut nv);
        writeln!(s, "  {pa2} = i64.const {pa_off}").unwrap();
        writeln!(s, "  {z} = i32.const 0").unwrap();
        writeln!(s, "  {to} = i64.const {TIMEOUT}").unwrap();
        writeln!(s, "  {wres} = i32.atomic.wait {pa2} {z} {to}").unwrap();
        writeln!(s, "  br {}()", wp(i)).unwrap();

        // wpc(i): parked ⟹ work[i] must be non-zero (the release/acquire published it).
        writeln!(s, "  }}").unwrap();
        writeln!(s, "block {} () {{", wpc(i)).unwrap();
        let wa = fresh(&mut nv);
        let w = fresh(&mut nv);
        let z2 = fresh(&mut nv);
        let nz = fresh(&mut nv);
        writeln!(s, "  {wa} = i64.const {wa_off}").unwrap();
        writeln!(s, "  {w} = i32.atomic.load.acquire {wa}").unwrap();
        writeln!(s, "  {z2} = i32.const 0").unwrap();
        writeln!(s, "  {nz} = i32.ne {w} {z2}").unwrap();
        writeln!(s, "  br_if {nz} {}() {}()", next_after(i), setwf(i)).unwrap();

        // setwf(i): publish the work-publish failure, then continue.
        writeln!(s, "  }}").unwrap();
        writeln!(s, "block {} () {{", setwf(i)).unwrap();
        let wf = fresh(&mut nv);
        let one2 = fresh(&mut nv);
        writeln!(s, "  {wf} = i64.const {WORKFAIL}").unwrap();
        writeln!(s, "  {one2} = i32.const 1").unwrap();
        writeln!(s, "  i32.atomic.store.release {wf} {one2}").unwrap();
        writeln!(s, "  br {}()", next_after(i)).unwrap();
        writeln!(s, "  }}").unwrap();
    }

    // ---- stw: the stopped window (raise STOPPED, then release everyone). ----
    writeln!(s, "block {stw} () {{").unwrap();
    let sa = fresh(&mut nv);
    let one = fresh(&mut nv);
    writeln!(s, "  {sa} = i64.const {STOPPED}").unwrap();
    writeln!(s, "  {one} = i32.const 1").unwrap();
    writeln!(s, "  i32.atomic.store.release {sa} {one}").unwrap();
    // (a real collector calls gc.roots(…) here — see gc_roots.rs; orthogonal to the barrier.)
    let sa2 = fresh(&mut nv);
    let z = fresh(&mut nv);
    writeln!(s, "  {sa2} = i64.const {STOPPED}").unwrap();
    writeln!(s, "  {z} = i32.const 0").unwrap();
    writeln!(s, "  i32.atomic.store.release {sa2} {z}").unwrap();
    let ra = fresh(&mut nv);
    let one2 = fresh(&mut nv);
    writeln!(s, "  {ra} = i64.const {RELEASE}").unwrap();
    writeln!(s, "  {one2} = i32.const 1").unwrap();
    writeln!(s, "  i32.atomic.store.release {ra} {one2}").unwrap();
    let ra2 = fresh(&mut nv);
    let wake = fresh(&mut nv);
    let nfy = fresh(&mut nv);
    writeln!(s, "  {ra2} = i64.const {RELEASE}").unwrap();
    writeln!(s, "  {wake} = i32.const {WAKE_ALL}").unwrap();
    writeln!(s, "  {nfy} = atomic.notify {ra2} {wake}").unwrap();
    writeln!(s, "  br {}()", join(0)).unwrap();

    // ---- per-mutator joins. ----
    for i in 0..m {
        writeln!(s, "  }}").unwrap();
        writeln!(s, "block {} () {{", join(i)).unwrap();
        let ha = fresh(&mut nv);
        let h = fresh(&mut nv);
        let jr = fresh(&mut nv);
        writeln!(s, "  {ha} = i64.const {}", HANDLE_BASE + i * 4).unwrap();
        writeln!(s, "  {h} = i32.atomic.load.acquire {ha}").unwrap();
        writeln!(s, "  {jr} = thread.join {h}").unwrap();
        let nxt = if i < m - 1 { join(i + 1) } else { final_blk };
        writeln!(s, "  br {nxt}()").unwrap();
    }

    // ---- final: return (VIOLATION, WORKFAIL), each extended i32 → i64. ----
    writeln!(s, "  }}").unwrap();
    writeln!(s, "block {final_blk} () {{").unwrap();
    let va = fresh(&mut nv);
    let vi = fresh(&mut nv);
    let viol = fresh(&mut nv);
    let wfa = fresh(&mut nv);
    let wfi = fresh(&mut nv);
    let wfl = fresh(&mut nv);
    writeln!(s, "  {va} = i64.const {VIOLATION}").unwrap();
    writeln!(s, "  {vi} = i32.atomic.load.acquire {va}").unwrap();
    writeln!(s, "  {viol} = i64.extend_i32_u {vi}").unwrap();
    writeln!(s, "  {wfa} = i64.const {WORKFAIL}").unwrap();
    writeln!(s, "  {wfi} = i32.atomic.load.acquire {wfa}").unwrap();
    writeln!(s, "  {wfl} = i64.extend_i32_u {wfi}").unwrap();
    writeln!(s, "  return {viol} {wfl}").unwrap();
    s.push_str("  }\n");

    s.push_str("}\n");
    s.push_str(MUTATOR);
    s
}

/// Run the collector (entry 0) on a backend and return `(violations, workfails)` — both must be 0.
fn run_interp_pair(m: &svm_ir::Module) -> (i64, i64) {
    let mut fuel = 200_000_000u64;
    let out = run(m, 0, &[], &mut fuel).unwrap_or_else(|t| panic!("interp trapped: {t:?}"));
    let got: Vec<i64> = out
        .iter()
        .map(|v| match v {
            Value::I64(x) => *x,
            other => panic!("expected i64, got {other:?}"),
        })
        .collect();
    (got[0], got[1])
}

fn run_jit_pair(m: &svm_ir::Module) -> (i64, i64) {
    use svm_jit::{compile_and_run, JitOutcome};
    match compile_and_run(m, 0, &[]).expect("jit compile/run") {
        JitOutcome::Returned(s) => (s[0], s[1]),
        other => panic!("jit did not return: {other:?}"),
    }
}

/// The reference barrier completes correctly on both backends for N=2 and N=4 vCPUs: every mutator
/// parks (so the collector's `parked == N-1` rendezvous fires), no mutator runs work inside the
/// stopped window (`VIOLATION == 0`), the parked mutators' work is published to the collector
/// (`WORKFAIL == 0`), and all vCPUs resume and join (the run returns rather than hanging).
#[test]
fn quiesce_barrier_stops_and_resumes_all_vcpus() {
    for &n in &[2usize, 4] {
        let m = parse_module(&build_quiesce_module(n))
            .unwrap_or_else(|e| panic!("parse (N={n}): {e:?}"));
        verify_module(&m).unwrap_or_else(|e| panic!("verify (N={n}): {e:?}"));

        let (vi, wf) = run_interp_pair(&m);
        assert_eq!(
            (vi, wf),
            (0, 0),
            "interp (N={n}): VIOLATION/WORKFAIL must be 0 (got violation={vi}, workfail={wf})"
        );

        let (vj, wfj) = run_jit_pair(&m);
        assert_eq!(
            (vj, wfj),
            (0, 0),
            "jit (N={n}): VIOLATION/WORKFAIL must be 0 (got violation={vj}, workfail={wfj})"
        );
    }
}
