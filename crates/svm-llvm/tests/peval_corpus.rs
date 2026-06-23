//! **Partial-evaluation corpus benchmark.** Where `peval_demo.rs` (Brainfuck) and `lisp_demo.rs`
//! (Lisp) each show off *one* real interpreter, this drives them as a **corpus-as-data** through a
//! single metric matrix, so we can see *where specialization pays off and where it can't*:
//!
//!   * **size**       — interpreter vs residual vs optimized residual (blocks / insts / bytes).
//!   * **PE time**    — the compile-time cost of specialization itself (specialize + optimize).
//!   * **jit compile**— time to JIT the interpreter vs the residual, measured *separately* from run
//!     (the `*_futamura_roi` demos call `compile_and_run` every rep, folding compile time into every
//!     "run" number; here we compile once and run many).
//!   * **run + scaling** — workload run time across several sizes. A *flat* speedup curve means the
//!     win is interpretive overhead we deleted outright (durable); a *shrinking* curve means real
//!     work dominates and PE can't help past a point.
//!   * **amortization** — break-even run count: PE compute cost ÷ per-run runtime saving.
//!
//! The corpus spans the two interpreter shapes real frontends emit: a flat **bytecode dispatch loop**
//! (BF) and a recursive **tree-walker** (Lisp — finite AST that fully unrolls, and guest recursion
//! that needs selective outlining).
//!
//! Run:   `cargo test -p svm-llvm --test peval_corpus -- --nocapture`            (correctness + size)
//! Bench: `cargo test -p svm-llvm --test peval_corpus -- --ignored --nocapture`  (full matrix)
//! (svm-llvm is workspace-excluded; run from `crates/svm-llvm`.)

use std::hint::black_box;
use std::process::Command;
use std::time::Instant;

use svm_ir::{Module, ValType, DEFAULT_RESERVED_LOG2};
use svm_jit::{CompiledModule, JitOutcome, Quota, INERT_CAP_THUNK};
use svm_peval::{
    optimize_module, specialize, specialize_with_config, SpecArg, SpecConfig, SpecError,
};
use svm_verify::verify_module;

// ===========================================================================================
// The C interpreters (mirrors of the demos — kept self-contained so this harness is the one place
// the corpus is described). Both take the program as an opaque pointer clang can't fold, which we
// then declare constant + readonly so the specializer folds what the compiler couldn't.
// ===========================================================================================

/// A Brainfuck interpreter: a flat bytecode dispatch loop over a readonly program, `long` tape cells.
/// `__PROG__` is replaced by the guest BF program (see [`bf_c`]) so one interpreter serves a range of
/// programs — e.g. `+++++.` (a constant), `,[>+++<-]>.` (out = 3*input), `,[>+++++<-]>.` (5*input).
const BF_C_TEMPLATE: &str = r#"
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
        else if (c == '[') { if (tape[dp] == 0) { long d=1; while(d){pc++; char k=prog[pc]; if(k=='[')d++; else if(k==']')d--;} } }
        else if (c == ']') { if (tape[dp] != 0) { long d=1; while(d){pc--; char k=prog[pc]; if(k==']')d++; else if(k=='[')d--;} } }
        pc++;
    }
    return out;
}
const char bfprog[] = "__PROG__";
int main(void) { return (int)run(bfprog, 0); }
"#;

/// The BF interpreter C with `prog` baked as the readonly program.
fn bf_c(prog: &str) -> String {
    BF_C_TEMPLATE.replace("__PROG__", prog)
}

/// The program's bytes as they appear in the readonly segment (the C string's trailing NUL included),
/// so its segment can be located by content.
fn bf_prog_bytes(prog: &str) -> Vec<u8> {
    let mut b = prog.as_bytes().to_vec();
    b.push(0);
    b
}

/// A Lisp/Scheme-subset recursive tree-walker over a readonly AST of 16-byte nodes. `expr_ast` is a
/// finite formula (unrolls); `fib_ast` defines fib in the AST (guest recursion → selective outline).
const LISP_C: &str = r#"
typedef struct { int tag, a, b, c; } Node;
enum { LIT, VAR, ADD, SUB, MUL, LT, EQ, IF, LET, CALL };
#define NSLOTS 4
static long ev(const Node *p, int n, long *env) {
    int tag = p[n].tag, a = p[n].a, b = p[n].b, c = p[n].c;
    switch (tag) {
        case LIT:  return a;
        case VAR:  return env[a];
        case ADD:  return ev(p, a, env) + ev(p, b, env);
        case SUB:  return ev(p, a, env) - ev(p, b, env);
        case MUL:  return ev(p, a, env) * ev(p, b, env);
        case LT:   return ev(p, a, env) < ev(p, b, env);
        case EQ:   return ev(p, a, env) == ev(p, b, env);
        case IF:   return ev(p, a, env) ? ev(p, b, env) : ev(p, c, env);
        case LET:  env[a] = ev(p, b, env); return ev(p, c, env);
        case CALL: { long ne[NSLOTS]; ne[0] = ev(p, b, env); return ev(p, a, ne); }
    }
    return 0;
}
long run(const Node *prog, long x, long y) {
    long env[NSLOTS];
    env[0] = x; env[1] = y;
    return ev(prog, 0, env);
}
static const Node expr_ast[] = {
    {LET, 2, 1, 3}, {MUL, 2, 2, 0}, {VAR, 0, 0, 0}, {IF, 4, 6, 9},
    {LT, 2, 5, 0}, {VAR, 1, 0, 0}, {ADD, 7, 8, 0}, {VAR, 2, 0, 0},
    {MUL, 10, 2, 0}, {SUB, 11, 5, 0}, {LIT, 3, 0, 0}, {MUL, 7, 7, 0},
};
static const Node fib_ast[] = {
    {CALL, 1, 2, 0}, {IF, 3, 4, 5}, {VAR, 0, 0, 0}, {LT, 4, 6, 0},
    {VAR, 0, 0, 0}, {ADD, 7, 8, 0}, {LIT, 2, 0, 0}, {CALL, 1, 9, 0},
    {CALL, 1, 10, 0}, {SUB, 4, 11, 0}, {SUB, 4, 6, 0}, {LIT, 1, 0, 0},
};
int main(void) { return (int)(run(expr_ast, 3, 5) + run(fib_ast, 10, 0)); }
"#;

// ----- frontend + lookup helpers -----------------------------------------------------------

/// Compile C → LLVM bitcode (`clang -O2`) → svm-IR. `extra` carries per-interpreter clang flags
/// (the tree-walker needs `-fno-optimize-sibling-calls` to keep its recursion as real calls).
fn compile_c_to_svm(name: &str, src: &str, extra: &[&str]) -> (Module, i64, Vec<(String, u32)>) {
    let base = std::env::temp_dir().join(format!("peval_corpus_{name}"));
    let cf = base.with_extension("c");
    let bc = base.with_extension("bc");
    std::fs::write(&cf, src).unwrap();
    let mut args = vec![
        "-O2",
        "-emit-llvm",
        "-c",
        "-fno-vectorize",
        "-fno-slp-vectorize",
    ];
    args.extend_from_slice(extra);
    let ok = Command::new("clang")
        .args(&args)
        .arg(&cf)
        .arg("-o")
        .arg(&bc)
        .status()
        .unwrap()
        .success();
    assert!(ok, "clang failed for {name}");
    let t = svm_llvm::translate_bc_path(&bc).expect("svm-llvm translate");
    let _ = std::fs::remove_file(&cf);
    let _ = std::fs::remove_file(&bc);
    (t.module, t.entry_sp as i64, t.exports)
}

/// The `run(...)` entry: by signature for BF (`run(sp,prog,input)`), or by export name for Lisp.
fn run_entry(m: &Module, exports: &[(String, u32)]) -> u32 {
    if let Some((_, idx)) = exports.iter().find(|(n, _)| n == "run") {
        return *idx;
    }
    m.funcs
        .iter()
        .position(|f| {
            f.params == [ValType::I64, ValType::I64, ValType::I64] && f.results == [ValType::I64]
        })
        .expect("run entry") as u32
}

/// The window address of the readonly segment whose bytes begin with `prefix`.
fn prog_addr(m: &Module, prefix: &[u8]) -> i64 {
    m.data
        .iter()
        .find(|d| d.readonly && d.bytes.starts_with(prefix))
        .map(|d| d.offset as i64)
        .expect("readonly program segment")
}

/// The first AST node's 16 bytes (4 little-endian i32s), to locate a Lisp program's segment.
fn node_bytes(tag: i32, a: i32, b: i32, c: i32) -> [u8; 16] {
    let mut out = [0u8; 16];
    for (i, v) in [tag, a, b, c].into_iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    out
}

// ----- jit: compile once, run many ---------------------------------------------------------

/// JIT-compile a module's entry into a long-lived `CompiledModule` (no host powerbox), so we can run
/// it repeatedly and time compilation separately from execution.
fn jit_compile(m: &Module, entry: u32) -> CompiledModule {
    CompiledModule::compile(
        m,
        entry,
        INERT_CAP_THUNK,
        std::ptr::null_mut(),
        DEFAULT_RESERVED_LOG2,
        None,
        None,
        None,
        None,
        Quota::default(),
        0,
    )
    .expect("jit compile")
}

/// Run an already-compiled module over a fresh guest window (state resets each call).
fn jit_call(cm: &mut CompiledModule, args: &[i64]) -> i64 {
    match cm.run(args, None, None, None) {
        Ok((JitOutcome::Returned(v), _mem)) => v[0],
        o => panic!("unexpected jit outcome {o:?}"),
    }
}

/// One-shot compile+run, for the cheap correctness pass.
fn jit_once(m: &Module, entry: u32, args: &[i64]) -> i64 {
    match svm_jit::compile_and_run(m, entry, args) {
        Ok(JitOutcome::Returned(v)) => v[0],
        o => panic!("unexpected jit outcome {o:?}"),
    }
}

/// Reference interpreter (svm-interp) — the honest, slow "interpreted interpreter" baseline.
fn interp_call(m: &Module, entry: u32, args: &[i64]) -> i64 {
    let vals: Vec<_> = args.iter().map(|&a| svm_interp::Value::I64(a)).collect();
    let mut fuel = u64::MAX;
    match svm_interp::run(m, entry, &vals, &mut fuel) {
        Ok(v) => match v.as_slice() {
            [svm_interp::Value::I64(x)] => *x,
            o => panic!("bad interp result {o:?}"),
        },
        Err(t) => panic!("interp trapped: {t:?}"),
    }
}

// ----- timing + sizing ---------------------------------------------------------------------

/// Best (min) wall time of `reps` runs of `f`, in seconds; warms up once first.
fn best(reps: usize, mut f: impl FnMut()) -> f64 {
    f();
    let mut b = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        f();
        b = b.min(t.elapsed().as_secs_f64());
    }
    b
}

struct Sizes {
    blocks: usize,
    insts: usize,
    bytes: usize,
}
fn sizes(m: &Module) -> Sizes {
    Sizes {
        blocks: m.funcs.iter().map(|f| f.blocks.len()).sum(),
        insts: m
            .funcs
            .iter()
            .flat_map(|f| &f.blocks)
            .map(|b| b.insts.len())
            .sum(),
        bytes: svm_encode::encode_module(m).len(),
    }
}

/// Emit one machine-readable metric row when `SVM_BENCH_CSV` is set: `CSV,<bench>,<case>,<metric>,
/// <value>` on stdout (run with `--nocapture` and `grep '^CSV,'`). Off by default.
fn csv(case: &str, metric: &str, value: f64) {
    if std::env::var_os("SVM_BENCH_CSV").is_some() {
        println!("CSV,peval_corpus,{case},{metric},{value}");
    }
}

// ===========================================================================================
// The corpus, as data.
// ===========================================================================================

/// `n -> argument vector` for an entry (interpreter args, or residual args, for workload size `n`).
type ArgsFn = Box<dyn Fn(i64) -> Vec<i64>>;
/// Produce the residual from a (fresh) copy of the interpreter.
type SpecializeFn = Box<dyn Fn(&Module) -> Result<Module, SpecError>>;

struct Case {
    name: &'static str,
    /// Interpreter shape, for the report.
    kind: &'static str,
    interpreter: Module,
    entry: u32,
    /// Args to the *interpreter* for workload size `n`.
    interp_args: ArgsFn,
    /// Args to the *residual* (entry func 0) for workload size `n`.
    residual_args: ArgsFn,
    /// The reference result for workload size `n`.
    expect: Box<dyn Fn(i64) -> i64>,
    /// Produce the residual from a (fresh) copy of the interpreter.
    specialize: SpecializeFn,
    /// Small, varied inputs for the correctness pass.
    correctness: Vec<i64>,
    /// Workload sizes for the timing sweep (last = headline).
    bench_scales: Vec<i64>,
    /// A near-zero-work input (empty loop / trivial recursion). Run time at this point is essentially
    /// the per-call JIT setup floor (guest-window allocation/zeroing); subtracting it isolates the
    /// marginal *compute*, so speedups reflect work removed rather than fixed overhead.
    floor_input: i64,
    /// Whether run time grows with `n` (a loop / recursion). `false` ⇒ size/structure win only, so a
    /// scaling sweep is meaningless and we time a single representative point.
    scales_with_work: bool,
}

fn fib_ref(n: i64) -> i64 {
    if n < 2 {
        n
    } else {
        fib_ref(n - 1) + fib_ref(n - 2)
    }
}
fn expr_ref(x: i64, y: i64) -> i64 {
    let a = x * x;
    if x < y {
        a + 3 * x
    } else {
        a * a - y
    }
}

/// Build a corpus `Case` for the BF interpreter running `prog` (`sp`/`prog` baked, input dynamic).
/// `scaling` ⇒ the program has a runtime-count loop (out grows with input); else it folds.
fn bf_case(
    name: &'static str,
    kind: &'static str,
    slug: &str,
    prog: &str,
    expect: Box<dyn Fn(i64) -> i64>,
    scaling: bool,
) -> Case {
    let (m, sp, exports) = compile_c_to_svm(slug, &bf_c(prog), &[]);
    let entry = run_entry(&m, &exports);
    let pa = prog_addr(&m, &bf_prog_bytes(prog));
    Case {
        name,
        kind,
        interpreter: m,
        entry,
        interp_args: Box::new(move |n| vec![sp, pa, n]),
        residual_args: Box::new(|n| vec![n]),
        expect,
        specialize: Box::new(move |m| {
            specialize(
                m,
                entry,
                &[
                    SpecArg::ConstI64(sp),
                    SpecArg::ConstI64(pa),
                    SpecArg::Dynamic,
                ],
            )
        }),
        // These multipliers count the tape cell down from the input, so they terminate only for
        // input >= 0; the constant program ignores the input entirely.
        correctness: vec![0, 1, 2, 7, 100, 1000],
        bench_scales: if scaling {
            vec![10_000, 100_000, 1_000_000]
        } else {
            vec![7]
        },
        floor_input: 0, // input 0 → tape cell is 0 → the BF loop body is skipped
        scales_with_work: scaling,
    }
}

fn corpus() -> Vec<Case> {
    let mut cases = Vec::new();

    // --- BF, a range of guest programs spanning the gain curve on one real interpreter -------------
    // Folds to a constant (no input loop): the whole program collapses to `return 5`.
    cases.push(bf_case(
        "bf: constant (out=5)",
        "bytecode (folds)",
        "bf_const",
        "+++++.",
        Box::new(|_| 5),
        false,
    ));
    // Runtime-count loop, light body (+3 per iteration).
    cases.push(bf_case(
        "bf: 3*n (light loop)",
        "bytecode loop",
        "bf_3x",
        ",[>+++<-]>.",
        Box::new(|n| 3 * n),
        true,
    ));
    // Runtime-count loop, heavier body (+5 per iteration) — more real work per dispatch.
    cases.push(bf_case(
        "bf: 5*n (heavier loop)",
        "bytecode loop",
        "bf_5x",
        ",[>+++++<-]>.",
        Box::new(|n| 5 * n),
        true,
    ));

    // --- Lisp expr: finite AST → fully unrolled formula (size/structure win, run ~constant) -------
    {
        let (m, sp, exports) =
            compile_c_to_svm("lisp_expr", LISP_C, &["-fno-optimize-sibling-calls"]);
        let entry = run_entry(&m, &exports);
        let pa = prog_addr(&m, &node_bytes(8, 2, 1, 3)); // expr_ast (LET, 2, 1, 3)
        const Y: i64 = 5; // hold y fixed; sweep the first input as "n"
        cases.push(Case {
            name: "lisp-expr: x<y ? a+3x : a*a-y",
            kind: "tree-walk (unrolled)",
            interpreter: m,
            entry,
            interp_args: Box::new(move |x| vec![sp, pa, x, Y]),
            residual_args: Box::new(|x| vec![x, Y]),
            expect: Box::new(|x| expr_ref(x, Y)),
            specialize: Box::new(move |m| {
                specialize(
                    m,
                    entry,
                    &[
                        SpecArg::ConstI64(sp),
                        SpecArg::ConstI64(pa),
                        SpecArg::Dynamic,
                        SpecArg::Dynamic,
                    ],
                )
            }),
            correctness: vec![-5, 0, 2, 3, 7, 50],
            bench_scales: vec![7],
            floor_input: 7, // unused (no scaling): the formula's work is constant
            scales_with_work: false,
        });
    }

    // --- Lisp fib: guest recursion → selective outlining (a compiled recursive function) ----------
    {
        let (m, sp, exports) =
            compile_c_to_svm("lisp_fib", LISP_C, &["-fno-optimize-sibling-calls"]);
        let entry = run_entry(&m, &exports);
        let pa = prog_addr(&m, &node_bytes(9, 1, 2, 0)); // fib_ast (CALL, 1, 2, 0)
        cases.push(Case {
            name: "lisp-fib: fib(n)",
            kind: "tree-walk (recursion)",
            interpreter: m,
            entry,
            interp_args: Box::new(move |n| vec![sp, pa, n, 0]),
            // The residual threads the data-stack pointer at runtime (sp dynamic), so entry is run(sp,n,y).
            residual_args: Box::new(move |n| vec![sp, n, 0]),
            expect: Box::new(fib_ref),
            specialize: Box::new(move |m| {
                specialize_with_config(
                    m,
                    entry,
                    &[
                        SpecArg::Dynamic,
                        SpecArg::ConstI64(pa),
                        SpecArg::Dynamic,
                        SpecArg::Dynamic,
                    ],
                    &SpecConfig {
                        selective_outline: true,
                        ..SpecConfig::default()
                    },
                )
            }),
            correctness: vec![0, 1, 2, 5, 10, 15, 20],
            bench_scales: vec![18, 22, 26],
            floor_input: 0, // fib(0) → the base case, essentially no recursion
            scales_with_work: true,
        });
    }

    cases
}

// ===========================================================================================
// Tests.
// ===========================================================================================

/// Cheap, always-on guard: every case specializes, the residual re-verifies, and across a range of
/// inputs interpreter == residual == optimized residual. Prints the size table (a size-regression
/// guard, like `svm-peval`'s `size_corpus`).
#[test]
fn corpus_specializes_and_matches() {
    println!(
        "\n=== PE size corpus (i=interpreter, r=residual, o=optimized) ===\n{:<32} {:<22} {:>12} {:>14} {:>18} {:>7}",
        "case", "shape", "blocks i/r/o", "insts i/r/o", "bytes i/r/o", "bytes"
    );
    for c in corpus() {
        verify_module(&c.interpreter).expect("interpreter verifies");
        let residual = (c.specialize)(&c.interpreter).expect("specializes");
        verify_module(&residual).expect("residual verifies");
        let opt = optimize_module(&residual);
        verify_module(&opt).expect("optimized residual verifies");

        for &n in &c.correctness {
            let want = (c.expect)(n);
            let got_interp = jit_once(&c.interpreter, c.entry, &(c.interp_args)(n));
            assert_eq!(got_interp, want, "{}: interpreter wrong at n={n}", c.name);
            assert_eq!(
                jit_once(&residual, 0, &(c.residual_args)(n)),
                want,
                "{}: residual diverged at n={n}",
                c.name
            );
            assert_eq!(
                jit_once(&opt, 0, &(c.residual_args)(n)),
                want,
                "{}: optimized diverged at n={n}",
                c.name
            );
        }

        let (i, r, o) = (sizes(&c.interpreter), sizes(&residual), sizes(&opt));
        println!(
            "{:<32} {:<22} {:>12} {:>14} {:>18} {:>6.0}%",
            c.name,
            c.kind,
            format!("{}/{}/{}", i.blocks, r.blocks, o.blocks),
            format!("{}/{}/{}", i.insts, r.insts, o.insts),
            format!("{}/{}/{}", i.bytes, r.bytes, o.bytes),
            100.0 * o.bytes as f64 / i.bytes as f64,
        );
    }
}

/// The full metric matrix (slow): PE time, JIT compile time (interpreter vs residual), a run-time
/// scaling sweep, and amortization break-even. Compile-once / run-many keeps compile time out of the
/// run numbers.
#[test]
#[ignore = "benchmark — run with --ignored --nocapture"]
fn corpus_metric_matrix() {
    let ms = |s: f64| s * 1e3;
    for c in corpus() {
        verify_module(&c.interpreter).expect("interpreter verifies");

        // PE cost: specialize + optimize (the work to produce the deployable residual).
        let t_pe = best(3, || {
            black_box(optimize_module(
                &(c.specialize)(&c.interpreter).expect("specializes"),
            ));
        });
        let residual = optimize_module(&(c.specialize)(&c.interpreter).expect("specializes"));
        verify_module(&residual).expect("residual verifies");

        // JIT compile time, interpreter vs residual (separate from run).
        let t_cc_i = best(5, || {
            black_box(jit_compile(&c.interpreter, c.entry));
        });
        let t_cc_r = best(5, || {
            black_box(jit_compile(&residual, 0));
        });

        let (i, r) = (sizes(&c.interpreter), sizes(&residual));
        println!("\n=== {} [{}] ===", c.name, c.kind);
        println!(
            "size  interpreter: {} blocks, {} insts, {} bytes",
            i.blocks, i.insts, i.bytes
        );
        println!(
            "size  residual:    {} blocks, {} insts, {} bytes  ({:.0}% of interpreter bytes)",
            r.blocks,
            r.insts,
            r.bytes,
            100.0 * r.bytes as f64 / i.bytes as f64
        );
        println!("PE time (specialize+optimize): {:.3} ms", ms(t_pe));
        println!(
            "jit compile: interpreter {:.3} ms, residual {:.3} ms",
            ms(t_cc_i),
            ms(t_cc_r)
        );
        csv(c.name, "interp_bytes", i.bytes as f64);
        csv(c.name, "residual_bytes", r.bytes as f64);
        csv(
            c.name,
            "residual_pct",
            100.0 * r.bytes as f64 / i.bytes as f64,
        );
        csv(c.name, "pe_ms", ms(t_pe));
        csv(c.name, "jit_compile_interp_ms", ms(t_cc_i));
        csv(c.name, "jit_compile_residual_ms", ms(t_cc_r));

        // Compile each once, then run many.
        let mut cm_i = jit_compile(&c.interpreter, c.entry);
        let mut cm_r = jit_compile(&residual, 0);

        // Sanity: agree on the headline workload before timing.
        let head = *c.bench_scales.last().unwrap();
        let want = (c.expect)(head);
        assert_eq!(
            jit_call(&mut cm_i, &(c.interp_args)(head)),
            want,
            "{}",
            c.name
        );
        assert_eq!(
            jit_call(&mut cm_r, &(c.residual_args)(head)),
            want,
            "{}",
            c.name
        );

        let reps = 5;
        if c.scales_with_work {
            // Subtract the per-call setup floor (run at the near-zero-work input) to isolate compute.
            let fa_i = (c.interp_args)(c.floor_input);
            let fa_r = (c.residual_args)(c.floor_input);
            let floor_i = best(reps, || {
                black_box(jit_call(&mut cm_i, &fa_i));
            });
            let floor_r = best(reps, || {
                black_box(jit_call(&mut cm_r, &fa_r));
            });
            println!(
                "per-call JIT setup floor: interpreter {:.4} ms, residual {:.4} ms (subtracted below)",
                ms(floor_i),
                ms(floor_r)
            );
            println!(
                "{:>12} {:>16} {:>16} {:>9}",
                "workload", "interp compute(ms)", "residual(ms)", "speedup"
            );
            let (mut head_i, mut head_r) = (0.0, 0.0);
            for &n in &c.bench_scales {
                let ia = (c.interp_args)(n);
                let ra = (c.residual_args)(n);
                let ti = (best(reps, || {
                    black_box(jit_call(&mut cm_i, &ia));
                }) - floor_i)
                    .max(0.0);
                let tr = (best(reps, || {
                    black_box(jit_call(&mut cm_r, &ra));
                }) - floor_r)
                    .max(0.0);
                // Below ~1 µs the subtracted compute is at the floor's noise level; don't report a ratio.
                let speedup = if tr > 1e-6 {
                    csv(c.name, &format!("speedup@{n}"), ti / tr);
                    format!("{:.1}x", ti / tr)
                } else {
                    "n/a".to_string()
                };
                println!("{:>12} {:>16.4} {:>16.4} {:>9}", n, ms(ti), ms(tr), speedup);
                head_i = ti;
                head_r = tr;
            }
            // Amortization: runs at the headline workload before PE's compute cost pays for itself in
            // per-run compute saved (interpreter vs residual).
            let saving = head_i - head_r;
            if saving > 0.0 {
                println!(
                    "amortization: break-even at ~{:.0} runs (PE {:.3} ms ÷ saving {:.4} ms/run @ {})",
                    t_pe / saving,
                    ms(t_pe),
                    ms(saving),
                    head
                );
            } else {
                println!("amortization: n/a (no per-run compute win at workload {head})");
            }

            // The honest baseline: the reference interpreter executing the interpreter ("interpreted
            // interpreter"), at the smallest workload (one rep — orders of magnitude slower), against
            // jit(interpreter) at the same size (same program → a clean backend speedup).
            let small = c.bench_scales[0];
            let t_ii = best(1, || {
                black_box(interp_call(
                    &c.interpreter,
                    c.entry,
                    &(c.interp_args)(small),
                ));
            });
            let sa = (c.interp_args)(small);
            let t_ji_small = best(reps, || {
                black_box(jit_call(&mut cm_i, &sa));
            });
            println!(
                "interp(interpreter) @ {}: {:.3} ms  ({:.0}x slower than jit(interpreter) @ same size)",
                small,
                ms(t_ii),
                t_ii / t_ji_small
            );
        } else {
            // No runtime loop: run time is dominated by the fixed per-call floor and is ~constant, so
            // the win is *size* and *JIT-compile time*, not run time. Report the one point plainly.
            let ia = (c.interp_args)(head);
            let ra = (c.residual_args)(head);
            let ti = best(reps, || {
                black_box(jit_call(&mut cm_i, &ia));
            });
            let tr = best(reps, || {
                black_box(jit_call(&mut cm_r, &ra));
            });
            println!("(no runtime loop — run time ~constant; the win is size + compile time)");
            println!(
                "run @ {}: interpreter {:.4} ms, residual {:.4} ms ({:.1}x)  |  jit-compile win {:.1}x",
                head,
                ms(ti),
                ms(tr),
                ti / tr,
                t_cc_i / t_cc_r
            );
        }
    }
}
