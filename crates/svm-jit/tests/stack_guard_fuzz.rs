//! Soundness fuzz for the software stack-overflow guard (feature `stack-check`, STACK_GUARD.md §
//! "Security contract"; tracker: STACK_GUARD_FLIP.md blocker #1).
//!
//! The guard becomes **escape-TCB** under the arena stack model: with `arena-stacks` there is no
//! hardware guard page between adjacent fibers, so a prologue check that fires *late* lets a frame
//! write below the running fiber's low bound and corrupt a *neighbour* fiber's stack. Its soundness
//! must therefore be **evidenced**, not just argued.
//!
//! ## Oracle
//!
//! We run the **default guard-page** fiber backend (i.e. NOT `arena-stacks`) with `stack-check` on,
//! so BOTH guards are live at once. A `PROT_NONE` guard page sits exactly at the fiber's `usable_low`;
//! the software check is designed to fire ~`RED_ZONE` *above* it. So for any generated recursion a
//! real overflow MUST surface as `StackOverflow` — the software check fired first, cleanly, before the
//! stack pointer ever reached the guard page.
//!
//! Any *other* outcome is a **guard hole**: a frame slipped past the software check and reached the
//! guard page. Empirically that surfaces two ways, and this test fails on both:
//!
//! * `Trapped(MemoryFault)` — the SIGSEGV handler had enough stack to run (asserted against below);
//! * a **hard process crash** (`SIGSEGV`, signal 11) — the fault happens *at stack exhaustion*, and
//!   the trap handler (`trap_shim.c`, `SA_ONSTACK` set but no `sigaltstack` installed) double-faults
//!   on the exhausted stack. A crash of the test binary is a cargo test failure, so the invariant is
//!   still enforced — just abruptly. (This double-fault is exactly why the guard page *alone* does
//!   not survivably catch fiber overflow, and why the software check — which traps through
//!   `trap_out`, no signal — is load-bearing even before the arena drops the page. See
//!   STACK_GUARD_FLIP.md "sigaltstack finding".)
//!
//! The guard page is thus a ground-truth oracle for "wrote below `usable_low`", and needs no arena —
//! so this runs on stock Linux/macOS CI.
//!
//! ## Forcing genuinely large frames (the hard part)
//!
//! The scary dimension is **frame size vs `RED_ZONE`** (16 KiB): a single frame larger than the red
//! zone — or Cranelift stack-probing a large frame — could touch below `usable_low` before the check
//! runs. To exercise it we must make a frame carry `N` real spill slots. A naive chain of pure values
//! folded only *after* the recursive call does **not** work: Cranelift sinks the pure computation past
//! the call (all uses are post-call), so frames stay tiny and the "past-RED_ZONE" claim is hollow (the
//! `frame_scaling_*` control below exists to catch exactly that regression).
//!
//! Instead each recursive body builds a dependency chain `a_1..a_N` and uses every `a_i` on **both
//! sides** of the call: a Horner fold `h1` of all `a_i` is passed *as the recursive call argument*
//! (pinning the `a_i` before the call — the arg can't be sunk past it), and a *different* Horner fold
//! `h2` of all `a_i` is combined with the return value *after* the call (keeping them live across it).
//! The two folds use different multipliers so GVN can't collapse them into one reduction. Net: `N`
//! values are live across the call ⇒ `N` real spill slots ⇒ frame ≈ `N * 8` bytes. The chain is a
//! genuine `a_i = a_{i-1}*k + seed` dependency, so `a_i` can't be single-instruction rematerialized.
#![cfg(all(
    feature = "stack-check",
    any(
        all(unix, target_arch = "x86_64"),
        all(unix, target_arch = "aarch64"),
        all(windows, target_arch = "x86_64")
    )
))]

use svm_jit::{compile_and_run, JitOutcome, TrapKind};
use svm_text::parse_module;

/// splitmix64 — deterministic, dependency-free PRNG so the CI seeds are reproducible.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn upto(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// Emit, into `s` at SSA counter `k` (returned advanced), the `N`-slot chain and the two Horner
/// folds. `seed` is the SSA name feeding the chain; returns `(k, h1, a)` where `h1` is the SSA name of
/// the pre-call fold (the recursive call argument) and `a` is the list of chain-value SSA indices to
/// fold *after* the call. All `a_i` are live from here to the post-call fold, so the frame gets `N`
/// spill slots.
fn emit_chain_and_h1(
    s: &mut String,
    mut k: usize,
    seed: usize,
    n: usize,
) -> (usize, usize, Vec<usize>) {
    let c3 = k;
    s.push_str(&format!("  v{c3} = i64.const 3\n"));
    k += 1;
    let mut prev = seed;
    let mut a = Vec::with_capacity(n);
    for _ in 0..n {
        let t = k;
        s.push_str(&format!("  v{t} = i64.mul v{prev} v{c3}\n"));
        k += 1;
        let ai = k;
        s.push_str(&format!("  v{ai} = i64.add v{t} v{seed}\n"));
        k += 1;
        a.push(ai);
        prev = ai;
    }
    // h1 = Horner fold over a_i with multiplier 5 (distinct from the post-call fold's 7), seeded by
    // `seed`. This is the recursive call argument, so every a_i must be computed before the call.
    let k5 = k;
    s.push_str(&format!("  v{k5} = i64.const 5\n"));
    k += 1;
    let mut h = seed;
    for &ai in &a {
        let m = k;
        s.push_str(&format!("  v{m} = i64.mul v{h} v{k5}\n"));
        k += 1;
        let add = k;
        s.push_str(&format!("  v{add} = i64.add v{m} v{ai}\n"));
        k += 1;
        h = add;
    }
    (k, h, a)
}

/// Emit the post-call Horner fold `h2` (multiplier 7) over `a`, seeded by `rc` (the call result), into
/// `s`. Returns `(k, result)`. Using every `a_i` here — with a multiplier different from `h1` — keeps
/// them live across the call without letting GVN share the two reductions.
fn emit_h2(s: &mut String, mut k: usize, rc: usize, a: &[usize]) -> (usize, usize) {
    let k7 = k;
    s.push_str(&format!("  v{k7} = i64.const 7\n"));
    k += 1;
    let mut h = rc;
    for &ai in a {
        let m = k;
        s.push_str(&format!("  v{m} = i64.mul v{h} v{k7}\n"));
        k += 1;
        let add = k;
        s.push_str(&format!("  v{add} = i64.add v{m} v{ai}\n"));
        k += 1;
        h = add;
    }
    (k, h)
}

/// Func 2 `(i64 depth, i64 seed) -> (i64)`. `bounded == false`: no base case, recurses forever
/// (`depth` passed through unchanged) — overflows regardless of frame size. `bounded == true`: base
/// case at `depth == 0` returns `seed`, else recurses on `depth - 1`. Either way the body carries an
/// `n`-slot frame via the both-sides chain (see module docs).
fn build_func2(n: usize, bounded: bool) -> String {
    let mut s = String::from("func (i64, i64) -> (i64) {\nblock0(v0: i64, v1: i64):\n");
    // v0 = depth, v1 = seed. Counter starts at 2.
    let mut k = 2usize;
    if bounded {
        let zero = k;
        s.push_str(&format!("  v{zero} = i64.const 0\n"));
        k += 1;
        let iseq = k;
        s.push_str(&format!("  v{iseq} = i64.eq v0 v{zero}\n"));
        k += 1;
        // base: return seed (v1); recursive: fall through to block1 carrying (depth, seed).
        s.push_str(&format!("  br_if v{iseq} block2(v1) block1(v0, v1)\n"));
        let dp = k;
        k += 1;
        let sp = k;
        k += 1;
        s.push_str(&format!("block1(v{dp}: i64, v{sp}: i64):\n"));
        let (k2, h1, a) = emit_chain_and_h1(&mut s, k, sp, n);
        k = k2;
        let one = k;
        s.push_str(&format!("  v{one} = i64.const 1\n"));
        k += 1;
        let nd = k;
        s.push_str(&format!("  v{nd} = i64.sub v{dp} v{one}\n"));
        k += 1;
        let rc = k;
        s.push_str(&format!("  v{rc} = call 2 (v{nd}, v{h1})\n"));
        k += 1;
        let (k3, res) = emit_h2(&mut s, k, rc, &a);
        k = k3;
        s.push_str(&format!("  return v{res}\n"));
        let bp = k;
        k += 1;
        let _ = k;
        s.push_str(&format!("block2(v{bp}: i64):\n"));
        s.push_str(&format!("  return v{bp}\n"));
    } else {
        let (k2, h1, a) = emit_chain_and_h1(&mut s, k, 1 /* seed = v1 */, n);
        k = k2;
        let rc = k;
        // recurse forever: depth (v0) passed through unchanged, seed = h1.
        s.push_str(&format!("  v{rc} = call 2 (v0, v{h1})\n"));
        k += 1;
        let (k3, res) = emit_h2(&mut s, k, rc, &a);
        k = k3;
        let _ = k;
        s.push_str(&format!("  return v{res}\n"));
    }
    s.push_str("}\n");
    s
}

/// Wrap func 2 in the root→fiber harness: func 0 (root) creates a fiber over func 1 and resumes it;
/// func 1 (the fiber entry) calls func 2 with `(initial_depth, seed=1)`. Only fibers get a real stack
/// limit (the root runs `limit = 0`, inert), so the recursion must run *inside* the fiber.
fn build_module(func2: &str, initial_depth: i64) -> String {
    let root = "\
func () -> (i64) {
block0():
  v0 = ref.func 1
  v1 = i64.const 4096
  v2 = cont.new v0 v1
  v3 = i64.const 0
  v4, v5 = cont.resume v2 v3
  return v5
}
";
    let entry = format!(
        "\
func (i64, i64) -> (i64) {{
block0(v0: i64, v1: i64):
  v2 = i64.const {initial_depth}
  v3 = i64.const 1
  v4 = call 2 (v2, v3)
  return v4
}}
"
    );
    format!("{root}{entry}{func2}")
}

/// Core soundness property: an overflowing fiber, at *every* frame size, must trap `StackOverflow`
/// and never `MemoryFault` (nor crash) — either would prove a frame reached the guard page past the
/// software check.
#[test]
fn overflow_always_traps_stack_overflow_never_memory_fault() {
    let mut rng = Rng(0xF1BE_57AC_C0DE_0001);
    // Always exercise the past-RED_ZONE regime (2048 i64 slots ≈ 16 KiB) plus a spread below it. The
    // random samples stay <= 1024 slots: regalloc over the large-N frames dominates compile time, and
    // the two past-RED_ZONE anchors (plus the frame-scaling control) already cover the > 16 KiB case.
    let anchors = [2usize, 32, 256, 1024, 2048, 2200];
    let mut n_big = 0usize;
    for i in 0..18usize {
        let n = if i < anchors.len() {
            anchors[i]
        } else {
            1 + rng.upto(1024) as usize
        };
        if n >= 2048 {
            n_big += 1;
        }
        let src = build_module(&build_func2(n, false), 0);
        let m = parse_module(&src).unwrap_or_else(|e| panic!("N={n}: parse failed: {e:?}"));
        match compile_and_run(&m, 0, &[]).expect("jit compile/run") {
            JitOutcome::Trapped(TrapKind::StackOverflow) => {}
            JitOutcome::Trapped(TrapKind::MemoryFault) => panic!(
                "GUARD HOLE (N={n}): overflow hit the hardware guard page (MemoryFault) — a frame \
                 wrote below usable_low before the software prologue check fired"
            ),
            other => panic!("N={n}: expected StackOverflow, got {other:?}"),
        }
    }
    assert!(
        n_big > 0,
        "sweep must include frames at/past RED_ZONE (>= 2048 slots)"
    );
}

/// Frame-scaling positive control (refutes sinking/rematerialization): a *terminating* recursion with
/// base case at depth 40 but a 2200-slot frame must overflow — 40 × (a multi-KiB frame) exceeds the
/// 256 KiB control stack. If the chain were sunk past the call / collapsed (frame ≈ 0), 40 shallow
/// frames would fit and this would *return*. So `StackOverflow` here is evidence the `N`-slot chain
/// occupies real frame space — i.e. the sweep above genuinely reaches the past-`RED_ZONE` regime.
#[test]
fn frame_scaling_large_frame_at_modest_depth_overflows() {
    let src = build_module(&build_func2(2200, true), 40);
    let m = parse_module(&src).expect("parse");
    match compile_and_run(&m, 0, &[]).expect("jit compile/run") {
        JitOutcome::Trapped(TrapKind::StackOverflow) => {}
        JitOutcome::Returned(_) => panic!(
            "frame did NOT scale with N: depth-40 recursion with a 2200-slot chain returned, so the \
             chain was sunk/collapsed rather than spilled — the frame-size sweep is not reaching \
             large frames and its RED_ZONE coverage claim is hollow"
        ),
        other => panic!("expected StackOverflow, got {other:?}"),
    }
}

/// No false positives: a legitimate bounded recursion (depth × frame kept safely under the 256 KiB
/// control stack) must run to completion — the check must not fire early.
#[test]
fn bounded_recursion_does_not_false_trap() {
    let mut rng = Rng(0xB0DE_D00D_1234_5678);
    for _ in 0..40usize {
        let n = 1 + rng.upto(32) as usize; // small frame
        let depth = 1 + rng.upto(40) as i64; // modest depth; depth * frame << 256 KiB
        let src = build_module(&build_func2(n, true), depth);
        let m = parse_module(&src).unwrap_or_else(|e| panic!("depth={depth} N={n}: parse: {e:?}"));
        match compile_and_run(&m, 0, &[]).expect("jit compile/run") {
            JitOutcome::Returned(_) => {}
            JitOutcome::Trapped(TrapKind::StackOverflow) => panic!(
                "FALSE TRAP: bounded recursion depth={depth} frame~{n} tripped the guard early"
            ),
            other => panic!("depth={depth} N={n}: expected a normal return, got {other:?}"),
        }
    }
}
