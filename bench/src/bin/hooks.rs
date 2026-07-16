//! **Memory-hooks overhead probe** (HOOKS.md §5): what does an opted-in run actually pay per
//! reported event, on each backend?
//!
//! The zero-cost claim for programs that *don't* opt in is structural (the engines are untouched —
//! see HOOKS.md §2/§5), so it isn't what this measures. This times the **opted-in** path, the way
//! a real consumer reaches it — `svm_run::Instance::with_mem_hooks` with a minimal counting hook —
//! against the same pristine module, on all three backends. The published quantity is
//! **overhead per event** (hooked − pristine, divided by the events per iteration) and the
//! resulting hooked **events/sec**: the numbers the HOOKS.md design so far only estimated
//! (~50–100 ns/event on the interpreters, ~10–20 ns on the JIT — the consumer's own handler is on
//! top of this probe's `AtomicU64` increment).
//!
//! Methodology matches the sibling harnesses (`crates/svm/src/bin/bench.rs`): per-iteration cost
//! is isolated by **subtraction** — `(time(N_LARGE) − time(N_SMALL)) / (N_LARGE − N_SMALL)` — which
//! cancels each backend's fixed per-run cost (frame setup, bytecode compile, JIT compile: identical
//! at both counts), and times are the **min** over repetitions. The event count is asserted, not
//! assumed (never benchmark a miscompile). Absolute ns are machine-dependent: this is a
//! watch-it-over-time probe, not a `--check` ratio gate.
//!
//! Run from `bench/`:  `cargo run --release --bin hooks`

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use svm_run::{instantiate, Backend, Instance, MemHookFn, RunConfig};
use svm_text::parse_module;

/// Loop-count pair for the subtraction; the kernel body does [`EVENTS_PER_ITER`] memory accesses.
const N_SMALL: u64 = 1_000;
const N_LARGE: u64 = 201_000;
/// One store + one load per loop iteration.
const EVENTS_PER_ITER: u64 = 2;
/// Min-of repetitions (robust to a noisy box).
const REPS: usize = 5;

/// The mem kernel with its loop count baked in (a hooked instance runs through the powerbox entry,
/// which takes no caller args): `n` iterations of store-acc / load-back / accumulate.
fn kernel(n: u64) -> String {
    format!(
        r#"memory 16
func () -> (i64) {{
block0():
  v0 = i32.const {n}
  v1 = i64.const 0
  br block1(v0, v1)
block1(v2: i32, v3: i64):
  v4 = i64.const 64
  i64.store v4 v3
  v5 = i64.load v4
  v6 = i64.add v3 v5
  v7 = i32.const 1
  v8 = i32.sub v2 v7
  br_if v8 block1(v8, v6) block2(v6)
block2(v9: i64):
  return v9
}}
"#
    )
}

fn instance(n: u64) -> Instance {
    instantiate(parse_module(&kernel(n)).expect("parse")).expect("instantiate")
}

/// A counting hook: the cheapest useful consumer (one relaxed `fetch_add` per event), so the
/// number reported is the *mechanism's* cost; a real cache model adds its own work on top.
fn counting_hook(count: Arc<AtomicU64>) -> impl Fn() -> MemHookFn + Send + Sync + 'static {
    move || -> MemHookFn {
        let count = count.clone();
        Box::new(move |_ev| {
            count.fetch_add(1, Ordering::Relaxed);
            Ok(())
        })
    }
}

/// Min-of-[`REPS`] wall time of one full run (per-run compile included — subtraction cancels it).
fn time_run(inst: &Instance, backend: Backend) -> f64 {
    let config = RunConfig::default();
    let mut best = f64::INFINITY;
    for _ in 0..REPS {
        let t = Instant::now();
        let r = inst.run(backend, &config).expect("run");
        let dt = t.elapsed().as_secs_f64();
        std::hint::black_box(&r);
        best = best.min(dt);
    }
    best
}

/// Per-iteration ns on `backend`, subtraction-isolated across the two loop counts.
fn per_iter(small: &Instance, large: &Instance, backend: Backend) -> f64 {
    let ts = time_run(small, backend);
    let tl = time_run(large, backend);
    (tl - ts) * 1e9 / (N_LARGE - N_SMALL) as f64
}

fn main() {
    // Pristine and hooked instances at both loop counts. The hooked pair shares one counter,
    // reset per timing; the count is asserted against the exact expected event total.
    let pristine_small = instance(N_SMALL);
    let pristine_large = instance(N_LARGE);
    let count = Arc::new(AtomicU64::new(0));
    let hooked_small = instance(N_SMALL)
        .with_mem_hooks(counting_hook(count.clone()))
        .expect("hooks");
    let hooked_large = instance(N_LARGE)
        .with_mem_hooks(counting_hook(count.clone()))
        .expect("hooks");

    let stats = hooked_large.mem_hook_stats().expect("hooked");
    println!(
        "mem-hooks overhead probe (counting hook, {EVENTS_PER_ITER} events/iter; \
         {} hooked ops, {} inserted insts in the kernel)\n",
        stats.hooked_ops, stats.inserted_insts
    );

    // Sanity: one hooked run reports exactly the expected events on every backend.
    for backend in [Backend::TreeWalk, Backend::Bytecode, Backend::Jit] {
        count.store(0, Ordering::Relaxed);
        hooked_small
            .run(backend, &RunConfig::default())
            .expect("hooked run");
        let got = count.load(Ordering::Relaxed);
        assert_eq!(
            got,
            N_SMALL * EVENTS_PER_ITER,
            "{backend:?} must report exactly the kernel's accesses"
        );
    }

    println!(
        "{:>10}  {:>14}  {:>14}  {:>16}  {:>12}",
        "backend", "pristine/iter", "hooked/iter", "overhead/event", "hooked ev/s"
    );
    for backend in [Backend::TreeWalk, Backend::Bytecode, Backend::Jit] {
        let base = per_iter(&pristine_small, &pristine_large, backend);
        let hooked = per_iter(&hooked_small, &hooked_large, backend);
        let per_event = (hooked - base) / EVENTS_PER_ITER as f64;
        let events_per_sec = EVENTS_PER_ITER as f64 * 1e9 / hooked;
        println!(
            "{:>10}  {:>12.2}ns  {:>12.2}ns  {:>14.2}ns  {:>10.1}M/s",
            format!("{backend:?}"),
            base,
            hooked,
            per_event,
            events_per_sec / 1e6
        );
    }
}
