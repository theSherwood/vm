//! Generative concurrent oracle (§12/§18) — drive the deterministic interleaving explorer from a
//! structured **program generator**, not just a handful of hand-written modules.
//!
//! The verification problem for threaded code is that a run has no single "expected" value to diff
//! against: the outcome depends on the interleaving. We sidestep that by generating only programs
//! whose result is **interleaving-invariant by construction**, so the oracle is an exact number we
//! compute on the host:
//!
//! Each generated program spawns N worker threads. Worker `t` performs `iters_t` iterations of
//! `i64.atomic.rmw.add cell_t, amount_t` on one of a few shared cells. Atomic RMW-add is
//! linearizable, and integer addition is commutative and associative, so the final contents of every
//! cell are a pure function of the *multiset* of adds — independent of how the threads interleave.
//! `main` then folds the cells into a single `i64` checksum, weighting cell `c` by `c+1` so that an
//! add landing in the wrong cell (a routing/address bug) also perturbs the result.
//!
//!   expected = Σ_t (cell_t + 1) · amount_t · iters_t
//!
//! Any deviation is therefore a real bug — a lost update (scheduler double-running or dropping a
//! vCPU, or a non-atomic RMW), a misrouted store, or an explorer that realizes an impossible
//! interleaving. We check each program two ways, mirroring `concurrent.rs`:
//!
//! - **Deterministic sweep:** [`svm_interp::run_scheduled`] across many scheduler seeds — systematic,
//!   reproducible coverage. A failure is replayable from `(program_seed, scheduler_seed)`.
//! - **Real-executor stress:** [`svm_interp::run`] on the M:N pool — OS scheduling supplies extra
//!   interleaving variety and the run is ThreadSanitizer-clean.

use svm_interp::{run, run_scheduled, Trap, Value};
use svm_text::parse_module;
use svm_verify::verify_module;

/// Reproducible xorshift PRNG (same family as the `irgen` generator), seeded per program.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Rng {
        Rng(seed ^ 0x9e3779b97f4a7c15 | 1)
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// A value in `lo..=hi`.
    fn range(&mut self, lo: u64, hi: u64) -> u64 {
        lo + self.next() % (hi - lo + 1)
    }
}

/// One worker's script: do `iters` atomic adds of `amount` to shared `cell`.
struct Worker {
    cell: u64,
    amount: u64,
    iters: u64,
}

struct Program {
    text: String,
    /// Σ_t (cell_t + 1) · amount_t · iters_t — the interleaving-invariant checksum `main` returns.
    expected: i64,
}

/// Build a verifier-valid concurrent module from a program seed.
///
/// The single worker function (func 1) unpacks its `(cell, amount, iters)` script from the spawn
/// `arg` — packed `(cell << 32) | (amount << 16) | iters` — so every thread runs the *same* code with
/// per-thread parameters, and the generator only has to vary constants. `main` (func 0) spawns all
/// workers, joins them all, then returns the weighted checksum.
fn gen_program(seed: u64) -> Program {
    let mut r = Rng::new(seed);
    let cells = r.range(1, 4);
    let n = r.range(2, 6);
    let workers: Vec<Worker> = (0..n)
        .map(|_| Worker {
            cell: r.range(0, cells - 1),
            amount: r.range(1, 8),
            iters: r.range(1, 150),
        })
        .collect();

    let expected: i64 = workers
        .iter()
        .map(|w| ((w.cell + 1) * w.amount * w.iters) as i64)
        .sum();

    // ---- main (func 0): straight-line spawn-all, join-all, then fold the cells ----
    let mut main = String::from("func () -> (i64) {\nblock0():\n  vsp = i64.const 0\n");
    for (t, w) in workers.iter().enumerate() {
        let arg = (w.cell << 32) | (w.amount << 16) | w.iters;
        main.push_str(&format!("  varg{t} = i64.const {arg}\n"));
        main.push_str(&format!("  vh{t} = thread.spawn 1 vsp varg{t}\n"));
    }
    for t in 0..workers.len() {
        main.push_str(&format!("  vj{t} = thread.join vh{t}\n"));
    }
    main.push_str("  vacc0 = i64.const 0\n");
    for c in 0..cells {
        let addr = c * 8;
        let w = c + 1;
        main.push_str(&format!("  vaddr{c} = i64.const {addr}\n"));
        main.push_str(&format!("  vld{c} = i64.atomic.load vaddr{c}\n"));
        main.push_str(&format!("  vwt{c} = i64.const {w}\n"));
        main.push_str(&format!("  vm{c} = i64.mul vld{c} vwt{c}\n"));
        main.push_str(&format!("  vacc{} = i64.add vacc{c} vm{c}\n", c + 1));
    }
    main.push_str(&format!("  return vacc{cells}\n}}\n"));

    // ---- worker (func 1): unpack script from arg, then a counted RMW-add loop ----
    // The text IR uses a per-block value scope, so loop-carried values (the counter and the unpacked
    // `off`/`amount`/`iters`) are threaded explicitly as block parameters.
    let worker = r#"func (i64, i64) -> (i64) {
block0(vsp: i64, varg: i64):
  c32 = i64.const 32
  hi = i64.shr_u varg c32
  mask = i64.const 65535
  cell = i64.and hi mask
  c16 = i64.const 16
  mid = i64.shr_u varg c16
  amount = i64.and mid mask
  iters = i64.and varg mask
  eight = i64.const 8
  off = i64.mul cell eight
  zero = i64.const 0
  br block1(zero, off, amount, iters)
block1(i: i64, off1: i64, amount1: i64, iters1: i64):
  cmp = i64.lt_u i iters1
  br_if cmp block2(i, off1, amount1, iters1) block3()
block2(i2: i64, off2: i64, amount2: i64, iters2: i64):
  rmw = i64.atomic.rmw.add off2 amount2
  one = i64.const 1
  inext = i64.add i2 one
  br block1(inext, off2, amount2, iters2)
block3():
  ret = i64.const 0
  return ret
}
"#;

    let text = format!("memory 16\n{main}{worker}");
    Program { text, expected }
}

fn one_i64(vals: Result<Vec<Value>, Trap>) -> i64 {
    match vals {
        Ok(v) => match v.as_slice() {
            [Value::I64(x)] => *x,
            other => panic!("expected one i64, got {other:?}"),
        },
        Err(t) => panic!("unexpected trap {t:?}"),
    }
}

/// Generate many programs; check each against its computed checksum on the deterministic explorer
/// (a sweep of scheduler seeds) and on the real M:N executor.
#[test]
fn generated_commutative_programs_match_oracle() {
    const PROGRAMS: u64 = 256;
    const SCHED_SEEDS: u64 = 12;
    const REAL_RUNS: u64 = 2;

    for pseed in 0..PROGRAMS {
        let prog = gen_program(pseed);
        let m = parse_module(&prog.text)
            .unwrap_or_else(|e| panic!("parse (program seed {pseed}): {e:?}\n{}", prog.text));
        verify_module(&m)
            .unwrap_or_else(|e| panic!("verify (program seed {pseed}): {e:?}\n{}", prog.text));

        // Deterministic explorer: systematic, reproducible interleavings.
        for sseed in 0..SCHED_SEEDS {
            let got = one_i64(run_scheduled(&m, 0, &[], 50_000_000, sseed));
            assert_eq!(
                got, prog.expected,
                "explorer mismatch: program seed {pseed}, scheduler seed {sseed}\n{}",
                prog.text
            );
        }

        // Real M:N executor: OS-scheduled, TSan-clean.
        for run_i in 0..REAL_RUNS {
            let mut fuel = 50_000_000u64;
            let got = one_i64(run(&m, 0, &[], &mut fuel));
            assert_eq!(
                got, prog.expected,
                "real-executor mismatch: program seed {pseed}, run #{run_i}\n{}",
                prog.text
            );
        }
    }
}

/// The generator + explorer pair is fully deterministic: a `(program_seed, scheduler_seed)` pair
/// always yields the same result, so any failure above is replayable.
#[test]
fn generated_programs_are_reproducible() {
    for pseed in [0u64, 1, 17, 99, 200] {
        let prog = gen_program(pseed);
        let m = parse_module(&prog.text).expect("parse");
        verify_module(&m).expect("verify");
        for sseed in [0u64, 3, 41] {
            let a = one_i64(run_scheduled(&m, 0, &[], 50_000_000, sseed));
            let b = one_i64(run_scheduled(&m, 0, &[], 50_000_000, sseed));
            assert_eq!(a, b, "program {pseed} / scheduler {sseed} not reproducible");
        }
    }
}
