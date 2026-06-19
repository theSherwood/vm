//! End-to-end **Futamura** demo, take two: a small **Lisp / Scheme-subset evaluator written in C**,
//! compiled `clang -O2 → LLVM → svm-llvm → svm-IR`, then **partial-evaluated** by `svm-peval` against
//! a fixed Lisp program. Where the Brainfuck demo (`peval_demo.rs`) had a flat dispatch loop, this
//! is a *recursive tree-walking interpreter* over an AST — the shape real language frontends emit —
//! and it exercises two different residual strategies:
//!
//!  * **Expression program** (`let`/`if`/arithmetic over runtime inputs): the AST is finite, so the
//!    recursive `ev` fully **unrolls** (plain inlining). The dispatch `switch`, the node decode, and
//!    the whole AST collapse to a straight-line/branchy arithmetic kernel — the *compiled formula*.
//!  * **Recursive program** (`fib` defined *in the Lisp AST*): the guest recursion has dynamic depth,
//!    so inlining would diverge. With **selective outlining** (`SpecConfig::selective_outline`) the
//!    leaves and structure inline as usual and *only* the recursive self-call outlines, folding into a
//!    **tight self-recursive residual** — the *compiled function* (a 2-function fib, not one tiny
//!    function per AST node). This is exactly weval's trick.
//!
//! As in the BF demo, the program is an **opaque pointer parameter** clang can't fold (so `-O2` emits
//! a *generic* evaluator), while *we* declare that pointer constant and its bytes readonly — the
//! §20c caller contract — so the specializer folds what the compiler couldn't.
//!
//! Run:   `cargo test -p svm-llvm --test lisp_demo -- --nocapture`            (correctness)
//! Bench: `cargo test -p svm-llvm --test lisp_demo -- --ignored --nocapture`  (size + speed)
//! (svm-llvm is workspace-excluded; run from `crates/svm-llvm`.)

use std::hint::black_box;
use std::process::Command;
use std::time::{Duration, Instant};

use svm_ir::Module;
use svm_peval::{optimize_module, specialize, specialize_with_config, SpecArg, SpecConfig};
use svm_verify::verify_module;

/// A tiny Lisp/Scheme-subset tree-walking interpreter. The AST is a flat array of 16-byte `Node`s in
/// a readonly segment; `ev(prog, node, env)` recurses over it. `prog` is an **opaque pointer
/// parameter** (clang can't fold it); `env` is an integer environment (the variable slots). `CALL`
/// makes a fresh environment, so user functions — including **recursion** — work. `run(prog, x, y)`
/// seeds slots 0/1 with the runtime inputs and evaluates node 0.
const LISP_C: &str = r#"
typedef struct { int tag, a, b, c; } Node;
enum { LIT, VAR, ADD, SUB, MUL, LT, EQ, IF, LET, CALL };
#define NSLOTS 4

static long ev(const Node *p, int n, long *env) {
    /* Read fields directly from the (readonly) program so the specializer folds them — copying the
       whole node to a stack temp would hide the tag/operands behind a stack load and the node index
       would go dynamic. */
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

/* expr: (let ((a (* x x))) (if (< x y) (+ a (* 3 x)) (- (* a a) y)))
   i.e. a = x*x; result = x<y ? a + 3x : a*a - y. A finite AST -> fully unrolls. */
static const Node expr_ast[] = {
    {LET, 2, 1, 3},   /* 0: let slot2 = (x*x) in IF...        */
    {MUL, 2, 2, 0},   /* 1: x * x                              */
    {VAR, 0, 0, 0},   /* 2: x                                  */
    {IF,  4, 6, 9},   /* 3: if (x<y) ... else ...             */
    {LT,  2, 5, 0},   /* 4: x < y                              */
    {VAR, 1, 0, 0},   /* 5: y                                  */
    {ADD, 7, 8, 0},   /* 6: a + 3x                             */
    {VAR, 2, 0, 0},   /* 7: a (slot 2)                         */
    {MUL,10, 2, 0},   /* 8: 3 * x                              */
    {SUB,11, 5, 0},   /* 9: a*a - y                            */
    {LIT, 3, 0, 0},   /* 10: 3                                 */
    {MUL, 7, 7, 0},   /* 11: a * a                             */
};

/* fib, defined *in the Lisp AST*: (define (fib v) (if (< v 2) v (+ (fib (- v 1)) (fib (- v 2)))))
   the program is (fib x). CALL nodes 7 and 8 point back at the body (node 1) -> guest recursion. */
static const Node fib_ast[] = {
    {CALL, 1, 2, 0},  /* 0: (fib x)                            */
    {IF,   3, 4, 5},  /* 1: fib body: if (v<2) v else ...      */
    {VAR,  0, 0, 0},  /* 2: x  (the call argument)             */
    {LT,   4, 6, 0},  /* 3: v < 2                              */
    {VAR,  0, 0, 0},  /* 4: v  (param, slot 0 of the frame)    */
    {ADD,  7, 8, 0},  /* 5: fib(v-1) + fib(v-2)                */
    {LIT,  2, 0, 0},  /* 6: 2                                  */
    {CALL, 1, 9, 0},  /* 7: fib(v-1)                           */
    {CALL, 1,10, 0},  /* 8: fib(v-2)                           */
    {SUB,  4,11, 0},  /* 9: v - 1                              */
    {SUB,  4, 6, 0},  /* 10: v - 2                             */
    {LIT,  1, 0, 0},  /* 11: 1                                 */
};

int main(void) { return (int)(run(expr_ast, 3, 5) + run(fib_ast, 10, 0)); }
"#;

/// The first node's 16 bytes, as they appear in the readonly segment (4 little-endian i32s),
/// so each program's segment can be located by content.
fn node_bytes(tag: i32, a: i32, b: i32, c: i32) -> [u8; 16] {
    let mut out = [0u8; 16];
    for (i, v) in [tag, a, b, c].into_iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    out
}

/// Compile C → LLVM bitcode (`clang -O2`) → svm-IR. Returns the module, the frontend's entry
/// data-stack pointer (`sp`), and the exported `name → func index` table. Mirrors `peval_demo.rs`.
fn compile_c_to_svm(name: &str, src: &str) -> (Module, i64, Vec<(String, u32)>) {
    let base = std::env::temp_dir().join(format!("lisp_demo_{name}"));
    let cf = base.with_extension("c");
    let bc = base.with_extension("bc");
    std::fs::write(&cf, src).unwrap();
    let ok = Command::new("clang")
        .args([
            "-O2",
            "-emit-llvm",
            "-c",
            "-fno-vectorize",
            "-fno-slp-vectorize",
            // Keep the evaluator's recursion as real recursive calls: without this, clang's
            // tail-recursion elimination loopifies `ev` and the `if` branches collapse to a
            // `select` of node indices feeding a loop phi — which makes the node index *dynamic*
            // (data-dependent), defeating the dispatch-folding the whole demo is about.
            "-fno-optimize-sibling-calls",
        ])
        .arg(&cf)
        .arg("-o")
        .arg(&bc)
        .status()
        .unwrap()
        .success();
    assert!(ok, "clang failed");
    let t = svm_llvm::translate_bc_path(&bc).expect("svm-llvm translate");
    let _ = std::fs::remove_file(&cf);
    let _ = std::fs::remove_file(&bc);
    (t.module, t.entry_sp as i64, t.exports)
}

/// The index of an exported function by name (here, `run`); svm-llvm prepends the `sp` data-stack
/// pointer, so the entry is `run(sp, prog, x, y)`.
fn func_named(exports: &[(String, u32)], name: &str) -> u32 {
    exports
        .iter()
        .find(|(n, _)| n == name)
        .unwrap_or_else(|| panic!("export {name:?} not found"))
        .1
}

/// The window address of the readonly segment whose bytes begin with `prefix` (the program's first
/// AST node) — i.e. where a given Lisp program lives.
fn prog_addr(m: &Module, prefix: &[u8; 16]) -> i64 {
    m.data
        .iter()
        .find(|d| d.readonly && d.bytes.starts_with(prefix))
        .map(|d| d.offset as i64)
        .expect("readonly AST segment")
}

fn run_jit(m: &Module, entry: u32, args: &[i64]) -> i64 {
    match svm_jit::compile_and_run(m, entry, args) {
        Ok(svm_jit::JitOutcome::Returned(v)) => v[0],
        o => panic!("unexpected jit outcome {o:?}"),
    }
}

fn run_interp(m: &Module, entry: u32, args: &[i64]) -> i64 {
    let vals: Vec<_> = args.iter().map(|&a| svm_interp::Value::I64(a)).collect();
    let mut fuel = u64::MAX;
    match svm_interp::run(m, entry, &vals, &mut fuel) {
        Ok(v) => match v.as_slice() {
            [svm_interp::Value::I64(x)] => *x,
            o => panic!("{o:?}"),
        },
        Err(t) => panic!("{t:?}"),
    }
}

fn sizes(m: &Module) -> (usize, usize) {
    let blocks: usize = m.funcs.iter().map(|f| f.blocks.len()).sum();
    let bytes = svm_encode::encode_module(m).len();
    (blocks, bytes)
}

/// expr(x,y) = let a=x*x in (x<y ? a+3x : a*a-y) — the reference the residual must match.
fn expr_ref(x: i64, y: i64) -> i64 {
    let a = x * x;
    if x < y {
        a + 3 * x
    } else {
        a * a - y
    }
}

fn fib_ref(n: i64) -> i64 {
    if n < 2 {
        n
    } else {
        fib_ref(n - 1) + fib_ref(n - 2)
    }
}

#[test]
fn recon_lisp_interpreter_translates() {
    let (m, sp, exports) = compile_c_to_svm("recon", LISP_C);
    let entry = func_named(&exports, "run");
    verify_module(&m).expect("translated interpreter verifies");
    eprintln!("\n=== Lisp interpreter (clang -O2 -> svm-llvm) ===");
    eprintln!(
        "entry func {entry} (sp={sp:#x}), {} functions",
        m.funcs.len()
    );
    for (i, d) in m.data.iter().enumerate() {
        eprintln!(
            "  data[{i}]: offset={:#x} len={} readonly={}",
            d.offset,
            d.bytes.len(),
            d.readonly
        );
    }
    eprintln!(
        "  expr program at {:#x}, fib program at {:#x}",
        prog_addr(&m, &node_bytes(8, 2, 1, 3)),
        prog_addr(&m, &node_bytes(9, 1, 2, 0)),
    );
    eprintln!("  run: {} blocks", m.funcs[entry as usize].blocks.len());
}

#[test]
fn expr_specializes_and_matches_interpreter() {
    let (m, sp, exports) = compile_c_to_svm("expr", LISP_C);
    let entry = func_named(&exports, "run");
    verify_module(&m).expect("interpreter verifies");
    let pa = prog_addr(&m, &node_bytes(8, 2, 1, 3)); // expr_ast

    // Finite AST -> plain inlining fully unrolls `ev`; sp is a baked constant (as in the BF demo).
    let residual = specialize(
        &m,
        entry,
        &[
            SpecArg::ConstI64(sp),
            SpecArg::ConstI64(pa),
            SpecArg::Dynamic,
            SpecArg::Dynamic,
        ],
    )
    .expect("specializes the expression program");
    verify_module(&residual).expect("residual re-verifies");
    let opt = optimize_module(&residual);
    verify_module(&opt).expect("optimized residual re-verifies");

    eprintln!(
        "\nexpr residual: {} block(s) (whole interpreter was {} blocks across {} fns); optimized: {}",
        residual.funcs[0].blocks.len(),
        m.funcs.iter().map(|f| f.blocks.len()).sum::<usize>(),
        m.funcs.len(),
        opt.funcs[0].blocks.len(),
    );

    for x in [-5i64, 0, 2, 3, 7, 50] {
        for y in [-1i64, 5, 100] {
            let want = run_jit(&m, entry, &[sp, pa, x, y]);
            assert_eq!(
                want,
                expr_ref(x, y),
                "interpreter itself wrong at ({x},{y})"
            );
            assert_eq!(
                run_jit(&residual, 0, &[x, y]),
                want,
                "residual at ({x},{y})"
            );
            assert_eq!(run_jit(&opt, 0, &[x, y]), want, "optimized at ({x},{y})");
        }
    }
}

#[test]
fn fib_specializes_and_matches_interpreter() {
    let (m, sp, exports) = compile_c_to_svm("fib", LISP_C);
    let entry = func_named(&exports, "run");
    verify_module(&m).expect("interpreter verifies");
    let pa = prog_addr(&m, &node_bytes(9, 1, 2, 0)); // fib_ast

    // Guest recursion has dynamic depth -> selective outlining: inline the leaves/structure, outline
    // only the recursive self-call (sp must be dynamic so the residual threads the data stack at
    // runtime; a baked-constant sp would grow per recursion level and diverge).
    let cfg = SpecConfig {
        selective_outline: true,
        ..SpecConfig::default()
    };
    let residual = specialize_with_config(
        &m,
        entry,
        &[
            SpecArg::Dynamic,
            SpecArg::ConstI64(pa),
            SpecArg::Dynamic,
            SpecArg::Dynamic,
        ],
        &cfg,
    )
    .expect("specializes the recursive fib program");
    verify_module(&residual).expect("residual re-verifies");
    let opt = optimize_module(&residual);
    verify_module(&opt).expect("optimized residual re-verifies");

    eprintln!(
        "\nfib residual: {} function(s), {} block(s) (interpreter: {} fns, {} blocks); optimized: {} fns",
        residual.funcs.len(),
        residual.funcs.iter().map(|f| f.blocks.len()).sum::<usize>(),
        m.funcs.len(),
        m.funcs.iter().map(|f| f.blocks.len()).sum::<usize>(),
        opt.funcs.len(),
    );

    for n in [0i64, 1, 2, 3, 5, 10, 15, 20] {
        let want = run_jit(&m, entry, &[sp, pa, n, 0]);
        assert_eq!(want, fib_ref(n), "interpreter itself wrong at fib({n})");
        // The residual entry threads sp; pass the frontend's entry sp as the first argument.
        assert_eq!(
            run_jit(&residual, 0, &[sp, n, 0]),
            want,
            "residual fib({n})"
        );
        assert_eq!(run_jit(&opt, 0, &[sp, n, 0]), want, "optimized fib({n})");
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
fn lisp_futamura_roi() {
    let (m, sp, exports) = compile_c_to_svm("roi", LISP_C);
    let entry = func_named(&exports, "run");
    verify_module(&m).expect("verifies");
    let ms = |d: Duration| d.as_secs_f64() * 1e3;

    // --- expr: the dispatch/decode fold away, leaving the compiled formula (a size/structure win) -
    let epa = prog_addr(&m, &node_bytes(8, 2, 1, 3));
    let expr_res = optimize_module(
        &specialize(
            &m,
            entry,
            &[
                SpecArg::ConstI64(sp),
                SpecArg::ConstI64(epa),
                SpecArg::Dynamic,
                SpecArg::Dynamic,
            ],
        )
        .expect("expr specializes"),
    );
    verify_module(&expr_res).expect("expr residual verifies");

    // --- fib: a compiled recursive function (selective outlining) --------------------------------
    let fpa = prog_addr(&m, &node_bytes(9, 1, 2, 0));
    let cfg = SpecConfig {
        selective_outline: true,
        ..SpecConfig::default()
    };
    let fib_res = optimize_module(
        &specialize_with_config(
            &m,
            entry,
            &[
                SpecArg::Dynamic,
                SpecArg::ConstI64(fpa),
                SpecArg::Dynamic,
                SpecArg::Dynamic,
            ],
            &cfg,
        )
        .expect("fib specializes"),
    );
    verify_module(&fib_res).expect("fib residual verifies");

    let reps = 5;

    eprintln!("\n=== Lisp Futamura ROI ===");

    // expr: the whole tree-walker (dispatch switch + node decode + AST) collapses to the compiled
    // formula. This is the *structure/size* win — there's no loop, so nothing to time meaningfully;
    // the speed story is fib below. (A counted host loop over many evaluations would not help: an
    // online partial evaluator *unrolls* a 0..n counted loop — its induction variable looks constant
    // each step — so the guest's only foldable looping construct is recursion, which fib exercises.)
    {
        let (ib, iby) = sizes(&m);
        let (rb, rby) = sizes(&expr_res);
        eprintln!("\n-- expr formula (single evaluation; structure/size) --");
        eprintln!(
            "interpreter: {} fns, {ib} blocks, {iby} bytes",
            m.funcs.len()
        );
        eprintln!(
            "residual:    {} fn,  {rb} blocks, {rby} bytes  (the whole tree-walker is one straight-line formula)",
            expr_res.funcs.len()
        );
        eprintln!("(bytes carry the unchanged AST data segments in both, so they understate the code collapse)");
    }

    // fib: a single recursive evaluation; the residual is the compiled function.
    {
        let n: i64 = 32;
        let want = fib_ref(n);
        assert_eq!(run_jit(&m, entry, &[sp, fpa, n, 0]), want);
        assert_eq!(run_jit(&fib_res, 0, &[sp, n, 0]), want);

        let t_ii = best_of(1, || run_interp(&m, entry, &[sp, fpa, n, 0]));
        let t_ji = best_of(reps, || run_jit(&m, entry, &[sp, fpa, n, 0]));
        let t_jr = best_of(reps, || run_jit(&fib_res, 0, &[sp, n, 0]));
        let (ib, iby) = sizes(&m);
        let (rb, rby) = sizes(&fib_res);
        eprintln!("\n-- fib({n}) recursion --");
        eprintln!("size  interpreter: {ib} blocks, {iby} bytes");
        eprintln!(
            "size  residual:    {rb} blocks ({} fns), {rby} bytes",
            fib_res.funcs.len()
        );
        let base = ms(t_ii);
        for (name, d) in [
            ("interp(interpreter)", t_ii),
            ("jit(interpreter)", t_ji),
            ("jit(residual)", t_jr),
        ] {
            eprintln!("{:<28} {:>12.3} ms {:>9.1}x", name, ms(d), base / ms(d));
        }
        eprintln!(
            "specialization win, JIT backend: {:.1}x",
            ms(t_ji) / ms(t_jr)
        );
    }
}
