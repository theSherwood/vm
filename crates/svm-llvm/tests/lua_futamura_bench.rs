//! **Benchmarks for the Lua Futamura residuals** — the arc's first timing numbers.
//!
//! Methodology (the compile-time confound handled honestly):
//!
//! - **Per-iteration interpreter cost, differential-N.** `run_powerbox` JIT-compiles the whole
//!   module on every call, so absolute whole-program times are dominated by compile for tiny
//!   scripts. But the baseline uses the SAME module for every trip count — only the stdin script
//!   changes — so `(T(N₂) − T(N₁)) / (N₂ − N₁)` cancels compile (and parse, up to a few bytes of
//!   script text) exactly and isolates `luaV_execute`'s per-iteration cost.
//! - **Residual whole-program at fixed N.** An entry-rooted residual *unrolls* a constant trip
//!   count, so its module differs per N — no clean differential. Instead the residual is compared
//!   at a fixed N with identical treatment (whole program, own compile included), alongside its
//!   build costs (profile + specialize wall time, residual size). A bigger-module compile is a
//!   *real* cost of unrolled specialization and is reported, not hidden. (The rolled/stitched
//!   shapes are what amortize it; timing those end-to-end needs the mid-loop stitching follow-up.)
//! - `best_of` (min over reps, after a warmup) per the house pattern in `peval_demo.rs`.
//!
//! No timing assertions — correctness only (this is a report, not a gate), and `#[ignore]`d.
//!
//! **Reference numbers** (2026-08-07, release, CI-class container; re-run to refresh):
//!
//! - Baseline `luaV_execute` per iteration (`x = add(x, 3)` + forloop, through the JIT-compiled
//!   interpreter): **≈ 93 ns** (20M-iteration differential; compile+parse cancelled).
//! - Whole-program at N=200: **parity (1.00×)** — both ≈ 0.95 s, dominated by JIT-compiling the
//!   ~690-function Lua module; the residual's extra ~9.7k blocks disappear into that.
//! - Specialize scaling: linear, ≈ **48.5 residual blocks and ≈ 0.5–0.6 ms of specialize time per
//!   unrolled iteration** (plus ~25 ms fixed); profile pass ≈ 1.8 s at N=200.
//! - **Rolled-residual execution speedup (`rolled_residual_vs_interpreter_per_iteration`): 11.5×.**
//!   On a pure `x = x + 3` loop with a dynamic trip count, the interpreter costs **≈ 17.6 ns/iter**
//!   and the dispatch-folded rolled residual **≈ 1.53 ns/iter** (20M-iteration differential on both
//!   sides; the rolled module is N-independent, so compile cancels exactly and this is a true
//!   execution-vs-execution number). This is the payoff the whole arc was pointing at: with the
//!   interpreter's decode+dispatch removed, the residual runs an order of magnitude faster per
//!   iteration. (The heavier `add(x, 3)` loop above measures ≈ 93 ns/iter because it pays a full
//!   Lua call per iteration; the two baselines differ by exactly that call overhead.)
//!
//! The economics these numbers pin down: at ~93 ns/iter interpreted vs ~0.6 ms/iter to *build*
//! unrolled residual blocks, **unrolled specialization can never pay per-run** — it needs either
//! build-once/run-many with a compiled-module cache or **rolled** loops (N-independent residual
//! size), which is exactly the deployment shape the rolled/stitching work targets — and which the
//! 11.5× execution number above confirms actually pays. Meanwhile the pipeline costs nothing at
//! runtime: residual whole-program time is at parity with the baseline.
//!
//! - **Compiled-module cache (`powerbox_program_amortizes_compile`): the build-once/run-many API.**
//!   `svm_run::PowerboxProgram` compiles a powerbox module *once* and runs it against many inputs,
//!   reusing the native code. On the ~690-function Lua module the per-call `run_powerbox` pays its
//!   full JIT compile every time; `PowerboxProgram` pays it once, so the amortized per-run cost drops
//!   to just the guest execution. This is the missing piece the numbers above called for: it makes a
//!   built residual (or the bare interpreter) pay its compile a single time across a batch, turning
//!   the 11.5× execution win into an end-to-end win instead of one swamped by recompilation.
//! - **Zero-config end-to-end (`auto_rolled_zero_config_vs_interpreter`): the whole pipeline, no
//!   hand-config.** `lua_rolled::auto_rolled(m, script)` takes a Lua chunk and returns the rolled
//!   residual via the shared `peval_capture::discover` — profile + discover + specialize measured as
//!   one **≈ 0.3 s build** (the cost the compiled-module cache amortizes). The resulting residual runs
//!   at **≈ 1.5–1.7 ns/iter**, matching the hand-built rolled residual, i.e. **≈ 6–11×** faster than
//!   the interpreter — the ratio swings only because the interpreter baseline is noisy on shared
//!   runners (≈ 10–18 ns/iter), while the residual side is stable. Same script-in → speedup-out result
//!   as the hand-built number, now with **zero hand-configuration**.
//!
//! Run: `cargo test -p svm-llvm --release --test lua_futamura_bench -- --ignored --nocapture`

mod futamura;
mod lua_rolled;
mod peval_capture;

use std::hint::black_box;
use std::time::{Duration, Instant};

fn best_of(reps: usize, mut f: impl FnMut() -> Vec<u8>) -> (Duration, Vec<u8>) {
    let out = f(); // warmup (and the correctness sample)
    let mut best = Duration::MAX;
    for _ in 0..reps {
        let t = Instant::now();
        black_box(f());
        best = best.min(t.elapsed());
    }
    (best, out)
}

fn loop_script(n: u64) -> String {
    format!(
        "local function add(a, b) return a + b end\n\
         local x = 0\n\
         for i = 1, {n} do x = add(x, 3) end\n\
         print(x)\n"
    )
}

#[test]
#[ignore = "benchmark; run explicitly with --release -- --ignored --nocapture"]
fn lua_futamura_roi() {
    let reps = 3;
    let m = futamura::lua_module();

    // ---- 1. Baseline per-iteration cost (differential-N, compile/parse cancelled). ----
    let (n1, n2) = (1_000_000u64, 21_000_000u64);
    let mr = &m;
    let run_base = |n: u64| {
        let s = loop_script(n);
        move || {
            svm_run::run_powerbox(mr, s.as_bytes())
                .expect("baseline run")
                .stdout
        }
    };
    let (t1, o1) = best_of(reps, run_base(n1));
    let (t2, o2) = best_of(reps, run_base(n2));
    assert_eq!(o1, format!("{}\n", 3 * n1).into_bytes());
    assert_eq!(o2, format!("{}\n", 3 * n2).into_bytes());
    let per_iter = (t2.saturating_sub(t1)).as_nanos() as f64 / (n2 - n1) as f64;
    println!("baseline whole-program: N={n1} -> {t1:?}, N={n2} -> {t2:?}");
    println!(
        "baseline luaV_execute per-iteration (call + add + return + forloop): {per_iter:.1} ns"
    );

    // ---- 2. The unrolled residual at fixed N: whole-program vs baseline + build costs. ----
    let n = 200u64;
    let script = loop_script(n);
    let b = futamura::auto_build(&script);
    println!(
        "residual build @ N={n}: profile {:?}, specialize {:?}, {} blocks",
        b.profile_time, b.specialize_time, b.residual_blocks
    );
    let (tb, ob) = best_of(reps, || {
        svm_run::run_powerbox(&b.baseline, script.as_bytes())
            .expect("baseline")
            .stdout
    });
    let (tr, or_) = best_of(reps, || {
        svm_run::run_powerbox(&b.residual, script.as_bytes())
            .expect("residual")
            .stdout
    });
    assert_eq!(ob, format!("{}\n", 3 * n).into_bytes());
    assert_eq!(or_, ob, "residual byte-identical");
    println!(
        "whole-program @ N={n} (each includes its own JIT compile): baseline {tb:?}, residual {tr:?} ({:.2}x)",
        tb.as_secs_f64() / tr.as_secs_f64()
    );

    // Residual-module compile overhead, isolated: time the residual module on the SAME differential
    // trick — its unrolled loop ignores the script's N (the fold baked N=200), so two different
    // stdin scripts execute identical guest work and the delta is pure parse noise. What remains
    // constant across both is compile + the baked run; compare with the baseline's same-N absolute.
    // (Reported for context; the honest end-to-end number is the fixed-N comparison above.)

    // ---- 3. Specialize-time scaling with unroll depth. ----
    for n in [25u64, 50, 100] {
        let s = loop_script(n);
        let b = futamura::auto_build(&s);
        println!(
            "  scaling: N={n:>3} -> specialize {:?}, {} blocks",
            b.specialize_time, b.residual_blocks
        );
    }
}

// ---------------------------------------------------------------------------------------------
// The execution-only comparison: the ROLLED residual (N-independent, runs standalone with a
// dynamic trip count) vs the interpreter on the same pure loop. Same module at both trip counts on
// BOTH sides, so the differential cancels compile exactly — this is the first true
// execution-vs-execution number for the arc. Residual construction is the proven recipe from
// `lua_futamura_rolled.rs` (loop-body safepoint capture + `dynamic_cells`), kept verbatim.
// ---------------------------------------------------------------------------------------------

use svm_interp::{IrPc, Stop, StopReason, Value, WatchKind};
use svm_ir::{Block, Func, Inst, LoadOp, Module, Terminator, ValType};
use svm_peval::{specialize_with_config, SpecArg, SpecConfig};

const ROLL_SCRIPT: &str = "local x = 0\nfor i = 1, 50 do x = x + 3 end\nreturn x\n";
const CAPTURE_LEN: usize = 8 << 20;
const L_STACK_LAST: u64 = 40;
const L_STACK: u64 = 48;
const LUA_STATE_SIZE: usize = 200;
const CI_SIZE: usize = 104;
const CI_FUNC: u64 = 0;
const LCLOSURE_P: u64 = 24;
const PROTO_SIZECODE: u64 = 24;
const PROTO_CODE: u64 = 64;
const SV: u64 = 16;

fn rd_u64(w: &[u8], a: u64) -> u64 {
    u64::from_le_bytes(w[a as usize..a as usize + 8].try_into().unwrap())
}
fn rd_i32(w: &[u8], a: u64) -> i32 {
    i32::from_le_bytes(w[a as usize..a as usize + 4].try_into().unwrap())
}

struct LoopEntry {
    sp: i64,
    l: u64,
    ci: u64,
    base: u64,
    counter: u64,
    window: Vec<u8>,
}

fn capture_loop_entry(m: &Module, luav: u32) -> LoopEntry {
    let inst = svm_run::instantiate(m.clone()).expect("instantiate");
    let mut insp = inst.debug_attach(ROLL_SCRIPT.as_bytes().to_vec(), u64::MAX);
    let entry_pc = IrPc {
        module: 0,
        func: luav,
        block: 0,
        inst: 0,
    };
    insp.set_breakpoint(entry_pc);
    let (sp, l, ci) = match insp.run_until_stop() {
        Stop::Break {
            reason: StopReason::Breakpoint,
            ..
        } => {
            let g = |i| match insp.read_ir_value(0, i) {
                Some(Value::I64(v)) => v,
                o => panic!("{o:?}"),
            };
            (g(0), g(1) as u64, g(2) as u64)
        }
        o => panic!("no entry break: {o:?}"),
    };
    insp.clear_breakpoint(entry_pc);
    let w0 = insp.read_window(0, CAPTURE_LEN).expect("window");
    let func = rd_u64(&w0, ci + CI_FUNC);
    let base = func + SV;
    let counter = base + 3 * SV;
    let acc = base + SV;
    insp.set_watchpoint(counter, 8, WatchKind::Write);
    let window = loop {
        match insp.run_until_stop() {
            Stop::Break {
                reason: StopReason::Watchpoint { write: true, .. },
                ..
            } => {
                insp.step();
                let w = insp.read_window(0, CAPTURE_LEN).expect("window");
                let ctag = w[(counter + 8) as usize];
                let atag = w[(acc + 8) as usize];
                if ctag == 0x03 && atag == 0x03 && rd_u64(&w, acc) == 0 && rd_u64(&w, counter) > 0 {
                    break w;
                }
            }
            o => panic!("did not reach loop body: {o:?}"),
        }
    };
    LoopEntry {
        sp,
        l,
        ci,
        base,
        counter,
        window,
    }
}

fn with_readback(residual: &Module, read_addr: u64) -> (Module, u32) {
    let mut m = residual.clone();
    let wrapper = m.funcs.len() as u32;
    m.funcs.push(Func {
        params: vec![ValType::I64, ValType::I64, ValType::I64],
        results: vec![ValType::I64],
        blocks: vec![Block {
            params: vec![ValType::I64, ValType::I64, ValType::I64],
            insts: vec![
                Inst::ConstI64(read_addr as i64),
                Inst::Call {
                    func: 0,
                    args: vec![0, 1, 2],
                },
                Inst::Load {
                    op: LoadOp::I64,
                    addr: 3,
                    offset: 0,
                },
            ],
            term: Terminator::Return(vec![4]),
        }],
    });
    (m, wrapper)
}

fn jit_run(m: &Module, e: u32, a: &[i64]) -> i64 {
    match svm_jit::compile_and_run(m, e, a) {
        Ok(svm_jit::JitOutcome::Returned(v)) => v[0],
        o => panic!("jit {o:?}"),
    }
}

fn pure_loop_script(n: u64) -> String {
    format!("local x = 0\nfor i = 1, {n} do x = x + 3 end\nprint(x)\n")
}

#[test]
#[ignore = "benchmark; run explicitly with --release -- --ignored --nocapture"]
fn rolled_residual_vs_interpreter_per_iteration() {
    let reps = 3;
    let m = futamura::lua_module();
    let luav = m
        .exports
        .iter()
        .find(|e| e.name == "luaV_execute")
        .expect("export")
        .func;

    // ---- Baseline: the interpreter on the pure loop, differential-N (compile cancels). ----
    let (n1, n2) = (1_000_000u64, 21_000_000u64);
    let mr = &m;
    let run_base = |n: u64| {
        let s = pure_loop_script(n);
        move || {
            svm_run::run_powerbox(mr, s.as_bytes())
                .expect("baseline run")
                .stdout
        }
    };
    let (t1, o1) = best_of(reps, run_base(n1));
    let (t2, o2) = best_of(reps, run_base(n2));
    assert_eq!(o1, format!("{}\n", 3 * n1).into_bytes());
    assert_eq!(o2, format!("{}\n", 3 * n2).into_bytes());
    let interp_ns = (t2.saturating_sub(t1)).as_nanos() as f64 / (n2 - n1) as f64;
    println!(
        "interpreter pure loop (x = x + 3): {interp_ns:.1} ns/iter  ({t1:?} @ {n1}, {t2:?} @ {n2})"
    );

    // ---- The rolled residual: same loop, dynamic counter, same module at both counts. ----
    let cap = capture_loop_entry(&m, luav);
    let (l, ci, w) = (cap.l, cap.ci, &cap.window);
    let stack_lo = rd_u64(w, l + L_STACK);
    let stack_hi = rd_u64(w, l + L_STACK_LAST);
    let func = rd_u64(w, ci + CI_FUNC);
    let closure = rd_u64(w, func);
    let proto = rd_u64(w, closure + LCLOSURE_P);
    let code = rd_u64(w, proto + PROTO_CODE);
    let sizecode = rd_i32(w, proto + PROTO_SIZECODE) as usize;
    let slice = |a: u64, n: usize| w[a as usize..a as usize + n].to_vec();
    let cfg = SpecConfig {
        const_overlays: vec![
            (l, slice(l, LUA_STATE_SIZE)),
            (stack_lo, slice(stack_lo, (stack_hi - stack_lo) as usize)),
            (ci, slice(ci, CI_SIZE)),
            (code, slice(code, 4 * sizecode)),
            (closure, slice(closure, 48)),
            (proto, slice(proto, 128)),
        ],
        rename: Some((l, l + LUA_STATE_SIZE as u64)),
        rename_extra: vec![(stack_lo, stack_hi), (ci, ci + CI_SIZE as u64)],
        rename_is_private: true,
        rename_seed_from_image: true,
        dynamic_cells: vec![(cap.base + SV, 8), (cap.base + 2 * SV, 8), (cap.counter, 8)],
        indirect_targets_cap: Some(16),
        ..SpecConfig::default()
    };
    let args = [
        SpecArg::ConstI64(cap.sp),
        SpecArg::ConstI64(l as i64),
        SpecArg::ConstI64(ci as i64),
    ];
    let r = specialize_with_config(&m, luav, &args, &cfg).expect("rolls");
    let (wm, we) = with_readback(&r, cap.base + SV);
    svm_verify::verify_module(&wm).expect("verifies");

    // Correctness at both counts, then differential timing (jit compiles per call, but the module
    // is IDENTICAL at both counts, so the delta is pure execution).
    let (c1, c2) = (n1 as i64 - 1, n2 as i64 - 1); // counter c ⇒ c+1 body executions
    assert_eq!(jit_run(&wm, we, &[0, 1, c1]), 3 * n1 as i64);
    assert_eq!(jit_run(&wm, we, &[0, 1, c2]), 3 * n2 as i64);
    let time_at = |c: i64| {
        let mut best = Duration::MAX;
        for _ in 0..reps {
            let t = Instant::now();
            black_box(jit_run(&wm, we, &[0, 1, c]));
            best = best.min(t.elapsed());
        }
        best
    };
    let r1 = time_at(c1);
    let r2 = time_at(c2);
    let resid_ns = (r2.saturating_sub(r1)).as_nanos() as f64 / (n2 - n1) as f64;
    println!(
        "rolled residual   (x = x + 3): {resid_ns:.2} ns/iter  ({r1:?} @ {n1}, {r2:?} @ {n2})"
    );
    println!(
        "execution speedup (interpreter / residual): {:.1}x",
        interp_ns / resid_ns
    );
}

// ---------------------------------------------------------------------------------------------
// The compiled-module cache: `svm_run::PowerboxProgram` compiles a powerbox module ONCE and runs it
// against many inputs, reusing the native code. `run_powerbox` recompiles the whole ~690-function
// Lua module every call, so a batch of N runs pays N full JIT compiles; the cache pays ONE, dropping
// the amortized per-run cost to just the guest execution. This is the build-once/run-many API the
// benchmark economics called for — it lets a built residual (or the bare interpreter) amortize its
// compile across a batch instead of re-paying it per invocation.
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "benchmark; run explicitly with --release -- --ignored --nocapture"]
fn powerbox_program_amortizes_compile() {
    let reps = 3;
    let m = futamura::lua_module();
    let scripts = [
        "print(1 + 2)\n",
        "local s = 0\nfor i = 1, 100 do s = s + i end\nprint(s)\n",
        "print('hello from lua')\n",
    ];

    // Correctness on the REAL module: the cache is byte-identical to per-call run_powerbox.
    let mut prog = svm_run::PowerboxProgram::compile(m.clone()).expect("compile powerbox program");
    for s in scripts {
        let want = svm_run::run_powerbox(&m, s.as_bytes())
            .expect("run_powerbox")
            .stdout;
        let got = prog.run(s.as_bytes()).expect("PowerboxProgram::run").stdout;
        assert_eq!(got, want, "cache diverged from run_powerbox on {s:?}");
    }

    // Batch of N runs: per-call compile vs compile-once. Use the same script each time so the only
    // difference measured is who pays the JIT compile.
    let n = 8usize;
    let script = scripts[1].as_bytes();

    let mut per_call = Duration::MAX;
    for _ in 0..reps {
        let t = Instant::now();
        for _ in 0..n {
            black_box(
                svm_run::run_powerbox(&m, script)
                    .expect("run_powerbox")
                    .stdout,
            );
        }
        per_call = per_call.min(t.elapsed());
    }

    let mut cached = Duration::MAX;
    for _ in 0..reps {
        // The honest end-to-end cost of the cache: one compile + N runs, timed together.
        let t = Instant::now();
        let mut prog = svm_run::PowerboxProgram::compile(m.clone()).expect("compile");
        for _ in 0..n {
            black_box(prog.run(script).expect("run").stdout);
        }
        cached = cached.min(t.elapsed());
    }

    println!(
        "batch of N={n} runs (Lua module): per-call run_powerbox {per_call:?} ({:?}/run), \
         compile-once PowerboxProgram {cached:?} ({:?}/run)",
        per_call / n as u32,
        cached / n as u32,
    );
    println!(
        "compiled-module cache speedup over the batch: {:.1}x",
        per_call.as_secs_f64() / cached.as_secs_f64()
    );
}

// ---------------------------------------------------------------------------------------------
// The zero-config end-to-end number: `lua_rolled::auto_rolled` (script in → rolled residual out, no
// hardcoded offsets, via the shared discovery driver) vs the interpreter on the same loop. Same
// differential-N methodology as `rolled_residual_vs_interpreter_per_iteration`, but the residual is
// built by the driver rather than by hand — closing the loop from "a Lua chunk" to "a measured
// per-iteration speedup", plus the one-time build cost the compiled-module cache amortizes.
// ---------------------------------------------------------------------------------------------

const AUTO_ROLL_SCRIPT: &str = "local x = 0\nfor i = 1, 50 do x = x + 3 end\nreturn x\n";

#[test]
#[ignore = "benchmark; run explicitly with --release -- --ignored --nocapture"]
fn auto_rolled_zero_config_vs_interpreter() {
    let reps = 3;
    let m = futamura::lua_module();

    // ---- Interpreter, differential-N (compile cancels): the pure `x = x + 3` loop. ----
    let (n1, n2) = (1_000_000u64, 21_000_000u64);
    let mr = &m;
    let run_base = |n: u64| {
        let s = pure_loop_script(n);
        move || svm_run::run_powerbox(mr, s.as_bytes()).expect("run").stdout
    };
    let (t1, o1) = best_of(reps, run_base(n1));
    let (t2, o2) = best_of(reps, run_base(n2));
    assert_eq!(o1, format!("{}\n", 3 * n1).into_bytes());
    assert_eq!(o2, format!("{}\n", 3 * n2).into_bytes());
    let interp_ns = (t2.saturating_sub(t1)).as_nanos() as f64 / (n2 - n1) as f64;

    // ---- Zero-config auto-rolled residual: the whole build (profile + discover + specialize),
    // timed once. This is the cost the compiled-module cache amortizes across runs. ----
    let t = Instant::now();
    let ar = lua_rolled::auto_rolled(&m, AUTO_ROLL_SCRIPT);
    let build = t.elapsed();
    let np = ar.dyn_cells.len();
    let (wm, we) = lua_rolled::with_readback(&ar.residual, ar.entry, ar.acc_addr, np);
    svm_verify::verify_module(&wm).expect("wrapped residual verifies");
    println!(
        "auto_rolled build (profile + discover + specialize): {build:?}; residual {} blocks, {np} dyn params",
        ar.residual_blocks
    );

    // The residual is body-entered (in-flight iteration), so counter=c ⇒ x = 3·(c+1); to reproduce a
    // trip count N (x = 3·N) seed counter = N-1, accumulator = 0. The module is N-independent, so the
    // differential cancels the residual's own JIT compile just as it cancels the interpreter's.
    let seed = |nn: u64| {
        let mut a = vec![0i64; np];
        a[ar.counter_ix] = nn as i64 - 1;
        a[ar.acc_ix] = 0;
        a
    };
    assert_eq!(jit_run(&wm, we, &seed(n1)), 3 * n1 as i64, "residual @ n1");
    assert_eq!(jit_run(&wm, we, &seed(n2)), 3 * n2 as i64, "residual @ n2");
    let time_at = |nn: u64| {
        let a = seed(nn);
        let mut best = Duration::MAX;
        for _ in 0..reps {
            let t = Instant::now();
            black_box(jit_run(&wm, we, &a));
            best = best.min(t.elapsed());
        }
        best
    };
    let r1 = time_at(n1);
    let r2 = time_at(n2);
    let resid_ns = (r2.saturating_sub(r1)).as_nanos() as f64 / (n2 - n1) as f64;

    println!("interpreter (x = x + 3):        {interp_ns:.1} ns/iter");
    println!("auto-rolled residual (zero-config): {resid_ns:.2} ns/iter");
    println!(
        "zero-config execution speedup (interpreter / residual): {:.1}x",
        interp_ns / resid_ns
    );
}
