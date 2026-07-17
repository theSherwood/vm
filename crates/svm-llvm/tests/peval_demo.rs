//! End-to-end **Futamura** demo: a real **Brainfuck interpreter written in C**, compiled
//! `clang -O2 → LLVM → svm-llvm → svm-IR`, then **partial-evaluated** by `svm-peval` against a fixed
//! BF program. The residual is the *compiled* BF program — the dispatch loop, opcode decode, and
//! bracket-matching all folded away — leaving just the data-dependent loop over the tape.
//!
//! The setup mirrors weval's real use case: the program is a **runtime pointer** the C compiler
//! cannot see into (so clang `-O2` must, and does, emit a *generic* interpreter), while *we* declare
//! that pointer constant and its bytes readonly — the §20c caller contract — so the specializer
//! folds what the compiler couldn't. (If `prog` were a compile-time `const` instead, clang's own
//! constant propagation would specialize it and there'd be nothing left to show.)
//!
//! Run:        `cargo test -p svm-llvm --test peval_demo -- --nocapture`            (correctness)
//! Bench:      `cargo test -p svm-llvm --test peval_demo -- --ignored --nocapture`  (size + speed)
//! (svm-llvm is workspace-excluded; run from `crates/svm-llvm`.)

use std::hint::black_box;
use std::process::Command;
use std::time::{Duration, Instant};

use svm_ir::{Module, ValType};
use svm_peval::{optimize_module, specialize, SpecArg};
use svm_verify::verify_module;

/// A tiny Brainfuck interpreter. `long` cells, so a `,` reads a large runtime input and a loop can
/// run many times — a meaningful Futamura workload. `prog` is an **opaque pointer parameter**
/// (clang can't fold it); the tape is interpreter state. `run(prog, input)` returns the `.` output.
const BF_C: &str = r#"
long run(const char *prog, long input) {
    static long tape[4096];
    long pc = 0, dp = 0, out = 0;
    for (;;) {
        char c = prog[pc];
        if (c == 0) break;
        if (c == '>') dp++;
        else if (c == '<') dp--;
        else if (c == '+') tape[dp]++;
        else if (c == '-') tape[dp]--;
        else if (c == ',') tape[dp] = input;
        else if (c == '.') out += tape[dp];
        else if (c == '[') {
            if (tape[dp] == 0) {
                long d = 1;
                while (d) { pc++; char k = prog[pc]; if (k == '[') d++; else if (k == ']') d--; }
            }
        } else if (c == ']') {
            if (tape[dp] != 0) {
                long d = 1;
                while (d) { pc--; char k = prog[pc]; if (k == ']') d++; else if (k == '[') d--; }
            }
        }
        pc++;
    }
    return out;
}
/* The BF program lives in a readonly segment; `run` reads it through the opaque pointer.
   `,[>+++<-]>.` reads the input into cell 0, then loops it down adding 3 to cell 1 each time,
   and outputs cell 1 — i.e. out = 3 * input, via a runtime-count loop. */
const char bfprog[] = ",[>+++<-]>.";
int main(void) { return (int)run(bfprog, 0); }
"#;

/// The BF program bytes (with the C string's trailing NUL), as they appear in the readonly segment.
const BF_PROG: &[u8] = b",[>+++<-]>.\0";

/// Compile C → LLVM bitcode (`clang -O2`) → svm-IR. Returns the module and the frontend's entry
/// data-stack pointer (`sp`). Mirrors `frontend_bench.rs`.
fn compile_c_to_svm(name: &str, src: &str) -> (Module, i64) {
    let base = std::env::temp_dir().join(format!("peval_demo_{name}"));
    let cf = base.with_extension("c");
    let bc = base.with_extension("ll");
    std::fs::write(&cf, src).unwrap();
    let ok = Command::new("clang")
        .args([
            "-O2",
            "-emit-llvm",
            "-S",
            "-fno-vectorize",
            "-fno-slp-vectorize",
        ])
        .arg(&cf)
        .arg("-o")
        .arg(&bc)
        .status()
        .unwrap()
        .success();
    assert!(ok, "clang failed");
    let t = svm_llvm::translate_ll_path(&bc).expect("svm-llvm translate");
    let _ = std::fs::remove_file(&cf);
    let _ = std::fs::remove_file(&bc);
    (t.module, t.entry_sp as i64)
}

/// The `run(sp, prog, input) -> i64` entry (svm-llvm prepends the `sp` data-stack pointer).
fn run_entry(m: &Module) -> u32 {
    m.funcs
        .iter()
        .position(|f| {
            f.params == [ValType::I64, ValType::I64, ValType::I64] && f.results == [ValType::I64]
        })
        .expect("run(i64,i64,i64)->i64 entry") as u32
}

/// The window address of the readonly segment holding the BF program.
fn prog_addr(m: &Module) -> i64 {
    m.data
        .iter()
        .find(|d| d.readonly && d.bytes.starts_with(BF_PROG))
        .map(|d| d.offset as i64)
        .expect("readonly BF program segment")
}

/// Specialize the interpreter against its program: `sp` and the `prog` pointer static, `input`
/// dynamic. The readonly program folds; the dispatch + bracket-matching collapse.
fn specialize_bf(m: &Module, entry: u32, sp: i64) -> Result<Module, svm_peval::SpecError> {
    specialize(
        m,
        entry,
        &[
            SpecArg::ConstI64(sp),
            SpecArg::ConstI64(prog_addr(m)),
            SpecArg::Dynamic,
        ],
    )
}

fn run_jit(m: &Module, entry: u32, args: &[i64]) -> i64 {
    match svm_jit::compile_and_run(m, entry, args) {
        Ok(svm_jit::JitOutcome::Returned(v)) => v[0],
        o => panic!("unexpected jit outcome {o:?}"),
    }
}

fn sizes(m: &Module) -> (usize, usize) {
    let blocks: usize = m.funcs.iter().map(|f| f.blocks.len()).sum();
    let bytes = svm_encode::encode_module(m).len();
    (blocks, bytes)
}

#[test]
fn recon_bf_interpreter_translates() {
    let (m, sp) = compile_c_to_svm("recon", BF_C);
    let entry = run_entry(&m);
    verify_module(&m).expect("translated interpreter verifies");
    eprintln!("\n=== BF interpreter (clang -O2 -> svm-llvm) ===");
    eprintln!(
        "entry func {entry} (sp={sp:#x}), {} functions",
        m.funcs.len()
    );
    eprintln!("program segment at {:#x}", prog_addr(&m));
    for (i, d) in m.data.iter().enumerate() {
        eprintln!(
            "  data[{i}]: offset={:#x} len={} readonly={}",
            d.offset,
            d.bytes.len(),
            d.readonly
        );
    }
    eprintln!("  run: {} blocks", m.funcs[entry as usize].blocks.len());
}

#[test]
fn bf_specializes_and_matches_interpreter() {
    let (m, sp) = compile_c_to_svm("correct", BF_C);
    let entry = run_entry(&m);
    verify_module(&m).expect("interpreter verifies");

    let residual = specialize_bf(&m, entry, sp).expect("specializes the BF interpreter");
    verify_module(&residual).expect("residual re-verifies");
    let opt = optimize_module(&residual);
    verify_module(&opt).expect("optimized residual re-verifies");

    eprintln!(
        "\nresidual: {} block(s) (interpreter run() had {}); optimized: {} block(s)",
        residual.funcs[0].blocks.len(),
        m.funcs[entry as usize].blocks.len(),
        opt.funcs[0].blocks.len(),
    );
    // The dispatch is gone: no opcode load survives in the residual (the program folded away).
    // (The tape loads/stores remain — that's the runtime data.)

    for input in [0i64, 1, 2, 7, 100, 1000, 100_000] {
        let want = run_jit(&m, entry, &[sp, prog_addr(&m), input]);
        assert_eq!(want, 3 * input, "interpreter itself wrong at {input}");
        assert_eq!(
            run_jit(&residual, 0, &[input]),
            want,
            "residual diverged at input {input}"
        );
        assert_eq!(
            run_jit(&opt, 0, &[input]),
            want,
            "optimized diverged at {input}"
        );
    }
}

fn best_of(reps: usize, mut f: impl FnMut() -> i64) -> Duration {
    f();
    let mut best = Duration::MAX;
    for _ in 0..reps {
        let t = Instant::now();
        black_box(f());
        best = best.min(t.elapsed());
    }
    best
}

#[test]
#[ignore = "perf demo — run with --ignored --nocapture"]
fn bf_futamura_roi() {
    let (m, sp) = compile_c_to_svm("roi", BF_C);
    let entry = run_entry(&m);
    verify_module(&m).expect("verifies");
    let pa = prog_addr(&m);
    let residual = optimize_module(&specialize_bf(&m, entry, sp).expect("specializes"));
    verify_module(&residual).expect("residual verifies");

    let n: i64 = 2_000_000; // loop trip count (the runtime input)
    let expect = 3 * n;
    assert_eq!(run_jit(&m, entry, &[sp, pa, n]), expect);
    assert_eq!(run_jit(&residual, 0, &[n]), expect);

    // interp(interpreter): the reference interpreter running the BF interpreter — the honest
    // "interpreted interpreter" baseline.
    let interp_interp = || {
        let mut fuel = u64::MAX;
        match svm_interp::run(
            &m,
            entry,
            &[
                svm_interp::Value::I64(sp),
                svm_interp::Value::I64(pa),
                svm_interp::Value::I64(n),
            ],
            &mut fuel,
        ) {
            Ok(v) => match v.as_slice() {
                [svm_interp::Value::I64(x)] => *x,
                o => panic!("{o:?}"),
            },
            Err(t) => panic!("{t:?}"),
        }
    };

    let reps = 5;
    let t_ii = best_of(1, interp_interp); // very slow; one timed rep
    let t_ji = best_of(reps, || run_jit(&m, entry, &[sp, pa, n]));
    let t_jr = best_of(reps, || run_jit(&residual, 0, &[n]));

    let (ib, iby) = sizes(&m);
    let (rb, rby) = sizes(&residual);
    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    eprintln!("\n=== Brainfuck Futamura ROI (out = 3*n, n = {n}) ===");
    eprintln!("size  interpreter: {ib} blocks, {iby} bytes");
    eprintln!("size  residual:    {rb} blocks, {rby} bytes");
    eprintln!(
        "{:<28} {:>12} {:>10}",
        "configuration", "time(ms)", "speedup"
    );
    let base = ms(t_ii);
    for (name, d) in [
        ("interp(interpreter)", t_ii),
        ("jit(interpreter)", t_ji),
        ("jit(residual)", t_jr),
    ] {
        eprintln!("{:<28} {:>12.3} {:>9.1}x", name, ms(d), base / ms(d));
    }
    eprintln!(
        "\nspecialization win, JIT backend: {:.1}x",
        ms(t_ji) / ms(t_jr)
    );
}
