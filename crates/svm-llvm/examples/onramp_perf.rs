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
//! Guardrail use: the absolute ns vary by machine/load, so compare **same-box before/after a change**.
//! The non-`setjmp` rows (`loop_lcg`/`calls`/`computed_goto`) are the baseline that must not move now
//! that `setjmp`/`longjmp` lowering exists — and it can't, because it is **gated on use**: those
//! workloads' IR is byte-identical to before (the translate suite asserts it). The `setjmp_*` rows
//! (interp-only; the JIT bails `Unsupported` for now) measure the feature's own cost: capture is an
//! O(live-values) frame snapshot, the long-jump an O(frames-unwound) truncate — both paid only when
//! executed (a typical `setjmp` capture lands a few `calls`-rows' worth; a `longjmp` round-trip ~2–3×
//! that). The interpreter rows are the low-noise signal; the JIT rows are compile-dominated/near-noise
//! because `svm_jit` only exposes compile-and-run together.
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

/// `(name, C source defining `int work(int)`, short note, runs_on_jit)`. All rows run on the JIT now
/// that its native-stack `setjmp`/`longjmp` has landed (libc `_setjmp`/`_longjmp` inline from JITted
/// code, LLVM.md §"JIT `longjmp`"): the `setjmp_*` rows measure the JIT's capture / round-trip cost
/// alongside the interpreters'. The first three are the **non-`setjmp` baseline** — they must not move
/// now that `setjmp`/`longjmp` lowering exists (gated on use; their IR is byte-identical to before).
const WORKLOADS: &[(&str, &str, &str, bool)] = &[
    (
        "loop_lcg",
        "int work(int n){ unsigned a = 1u; for (int i = 0; i < n; i++) a = a*1664525u + 1013904223u; return (int)a; }",
        "tight loop, serial LCG recurrence (loop throughput)",
        true,
    ),
    (
        "calls",
        "static unsigned mix(unsigned x){ return x*2654435761u + 0x9e3779b9u; } \
         int work(int n){ unsigned a = 1u; for (int i = 0; i < n; i++) a = mix(a) ^ (a >> 13); return (int)a; }",
        "call/return per iteration (call-stack overhead — the setjmp-relevant axis)",
        true,
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
        true,
    ),
    (
        "setjmp_happy",
        // `setjmp` each iteration, never long-jumped (the pcall-entry happy path): measures capture.
        "#include <setjmp.h>\nstatic jmp_buf e;\n\
         int work(int n){ unsigned a=1u; for(int i=0;i<n;i++){ setjmp(e); a=a*1664525u+1u; } return (int)a; }",
        "setjmp capture per iteration, no longjmp",
        true,
    ),
    (
        "setjmp_longjmp",
        // setjmp + a one-frame longjmp re-entry each iteration: measures capture + unwind.
        "#include <setjmp.h>\nstatic jmp_buf e;\n\
         int work(int n){ unsigned a=1u; for(int i=0;i<n;i++){ volatile int x=setjmp(e); \
         if(x==0){ a=a*1664525u+1u; longjmp(e,1); } } return (int)a; }",
        "setjmp + longjmp round-trip per iteration",
        true,
    ),
];

fn compile(name: &str, src: &str) -> Option<std::path::PathBuf> {
    let dir = std::env::temp_dir();
    let c = dir.join(format!("onramp_perf_{}_{}.c", std::process::id(), name));
    let bc = dir.join(format!("onramp_perf_{}_{}.ll", std::process::id(), name));
    std::fs::write(&c, src).ok()?;
    let ok = Command::new("clang")
        .args(["-O2", "-emit-llvm", "-S"])
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
    for (name, src, note, runs_on_jit) in WORKLOADS {
        let Some(bc) = compile(name, src) else {
            eprintln!("note: skipping {name} (clang unavailable)");
            continue;
        };
        let t = svm_llvm::translate_ll_path(&bc).expect("translate");
        svm_verify::verify_module(&t.module).expect("verify");
        let m = &t.module;
        let sp = t.entry_sp;

        let (ri, interp) = ns_per_iter(|n| time_interp(m, sp, n));
        if *runs_on_jit {
            let (rj, jit) = ns_per_iter(|n| time_jit(m, sp, n));
            assert_eq!(ri, rj, "{name}: interp ({ri}) vs JIT ({rj}) disagree");
            println!("{name:<16} {interp:>14.2} {jit:>14.2}   {note}");
        } else {
            println!("{name:<16} {interp:>14.2} {:>14}   {note}", "n/a");
        }
    }
}
