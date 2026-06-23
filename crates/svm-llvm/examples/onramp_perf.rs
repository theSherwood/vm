//! On-ramp perf harness — the baseline guardrail for the `setjmp`/`longjmp` slice (LLVM.md).
//!
//! Run with:  `cd crates/svm-llvm && cargo run --release --example onramp_perf`
//!
//! It times representative on-ramp workloads on **both** backends (the reference interpreter and the
//! Cranelift JIT) and reports per-iteration cost. The point is the **non-`setjmp` baseline**: when
//! `setjmp`/`longjmp` lands (gated on the program actually calling it), these rows must not move —
//! that is the measured proof of the "non-users pay nothing" claim. A `setjmp` micro-row slots in
//! beside them once the ops exist (a happy-path `setjmp`-in-a-loop with no `longjmp`, and a
//! `longjmp`-taken row), so the cost of the feature itself is measured, not assumed.
//!
//! Recorded pre-`setjmp` baseline (release, one dev box — for regression eyeballing, not a CI gate):
//!   loop_lcg interp ~4.1 ns/it  |  calls interp ~25.6 ns/it  |  computed_goto interp ~106 ns/it
//! The **interpreter** rows are the low-noise guardrail (what must not move when `setjmp` lands); the
//! JIT rows are compile-dominated/near-noise here because `svm_jit` only exposes compile-and-run
//! together, so the large/small-n subtraction leaves execution below the timing floor.
//!
//! Methodology: each workload is `int work(int n)` whose body runs `n` fold-resistant iterations
//! (a serial recurrence, so the optimizer can't collapse the loop). We time it at a large and a
//! small `n` and subtract — cancelling the fixed per-call cost (JIT compile + entry), since
//! `svm_jit` only exposes compile-and-run together — then divide by the iteration delta and take the
//! min over reps (least-noise). interp == JIT is asserted, so the work is never optimized away.

use std::process::Command;
use std::time::Instant;

use svm_interp::Value;
use svm_jit::JitOutcome;

const REPS: u32 = 9;
const N_LARGE: i32 = 4_000_000;
const N_SMALL: i32 = 1_000;

/// `(name, C source defining `int work(int)`, short note)`.
const WORKLOADS: &[(&str, &str, &str)] = &[
    (
        "loop_lcg",
        "int work(int n){ unsigned a = 1u; for (int i = 0; i < n; i++) a = a*1664525u + 1013904223u; return (int)a; }",
        "tight loop, serial LCG recurrence (loop throughput)",
    ),
    (
        "calls",
        "static unsigned mix(unsigned x){ return x*2654435761u + 0x9e3779b9u; } \
         int work(int n){ unsigned a = 1u; for (int i = 0; i < n; i++) a = mix(a) ^ (a >> 13); return (int)a; }",
        "call/return per iteration (call-stack overhead — the setjmp-relevant axis)",
    ),
    (
        "computed_goto",
        // A threaded (indirectbr/blockaddress) dispatch loop — the interpreter category setjmp targets.
        "int work(int n){ static const void *const tbl[]={&&A,&&B,&&C}; unsigned a=(unsigned)n; \
         int pc=0; for(int i=0;i<n;i++){ goto *tbl[pc]; \
         A: a=a*1664525u+1u; pc=1; continue; \
         B: a^=a>>13; pc=2; continue; \
         C: a+=0x9e3779b9u; pc=0; continue; } return (int)a; }",
        "computed-goto dispatch (indirectbr → br_table)",
    ),
];

fn compile(name: &str, src: &str) -> Option<std::path::PathBuf> {
    let dir = std::env::temp_dir();
    let c = dir.join(format!("onramp_perf_{}_{}.c", std::process::id(), name));
    let bc = dir.join(format!("onramp_perf_{}_{}.bc", std::process::id(), name));
    std::fs::write(&c, src).ok()?;
    let ok = Command::new("clang")
        .args(["-O2", "-emit-llvm", "-c"])
        .arg(&c)
        .arg("-o")
        .arg(&bc)
        .status()
        .ok()?
        .success();
    ok.then_some(bc)
}

/// Run `work(n)` on the interpreter, returning (result, elapsed).
fn time_interp(m: &svm_ir::Module, entry_sp: u64, n: i32) -> (i32, std::time::Duration) {
    let args = [Value::I64(entry_sp as i64), Value::I32(n)];
    let mut fuel = u64::MAX;
    let t = Instant::now();
    let r = svm_interp::run_fast(m, 0, &args, &mut fuel).expect("interp run");
    let dt = t.elapsed();
    (
        match r[0] {
            Value::I32(x) => x,
            _ => panic!("expected i32"),
        },
        dt,
    )
}

/// Run `work(n)` on the JIT (compile + run together), returning (result, elapsed).
fn time_jit(m: &svm_ir::Module, entry_sp: u64, n: i32) -> (i32, std::time::Duration) {
    let slots = [entry_sp as i64, n as i64];
    let t = Instant::now();
    let out = svm_jit::compile_and_run(m, 0, &slots).expect("jit run");
    let dt = t.elapsed();
    let v = match out {
        JitOutcome::Returned(s) => s[0] as i32,
        other => panic!("unexpected JIT outcome {other:?}"),
    };
    (v, dt)
}

/// ns per iteration via the large/small-n subtraction, min over reps.
fn ns_per_iter(run: impl Fn(i32) -> (i32, std::time::Duration)) -> (i32, f64) {
    let mut best = f64::INFINITY;
    let mut result = 0;
    for _ in 0..REPS {
        let (r_lo, t_lo) = run(N_SMALL);
        let (r_hi, t_hi) = run(N_LARGE);
        result = r_hi;
        let _ = r_lo;
        let delta = t_hi.as_nanos() as f64 - t_lo.as_nanos() as f64;
        let per = delta / (N_LARGE - N_SMALL) as f64;
        if per < best {
            best = per;
        }
    }
    (result, best)
}

fn main() {
    println!(
        "{:<16} {:>14} {:>14}   note",
        "workload", "interp ns/it", "jit ns/it"
    );
    println!("{}", "-".repeat(72));
    for (name, src, note) in WORKLOADS {
        let Some(bc) = compile(name, src) else {
            eprintln!("note: skipping {name} (clang unavailable)");
            continue;
        };
        let t = svm_llvm::translate_bc_path(&bc).expect("translate");
        svm_verify::verify_module(&t.module).expect("verify");
        let m = &t.module;
        let sp = t.entry_sp;

        let (ri, interp) = ns_per_iter(|n| time_interp(m, sp, n));
        let (rj, jit) = ns_per_iter(|n| time_jit(m, sp, n));
        assert_eq!(ri, rj, "{name}: interp ({ri}) vs JIT ({rj}) disagree");

        println!("{name:<16} {interp:>14.2} {jit:>14.2}   {note}");
    }
}
