//! **A gallery of more-real Lua programs through the zero-config auto-rolled pipeline.**
//!
//! Each program is a genuine numeric loop — the loop body *uses the loop variable* and does real
//! arithmetic (multiplies, multiple accumulators, a swap), not the trivial `x = x + 3`. For each one we
//! print the program's result **plus the per-iteration speed of the interpreter and of the rolled
//! residual on that program** (differential-N on both sides cancels JIT-compile / parse). Result and
//! both speeds, side by side.
//!
//! The division family `%` / `//` is now in scope too: `auto_rolled` deopts the integer div/mod
//! divide-by-zero cold arm to the baseline (marking the divisor + result dynamic, and resuming off the
//! metamethod trailer — see its comments), so `i % d` / `i // d` roll like any other arithmetic on the
//! hot path. The last two programs below exercise them (divisor a local, so it is a register operand —
//! the K-form `i % 2` still calls an un-inlined helper and stays a wall) and are checked against the
//! interpreter the same way. Everything here is straight-line integer arithmetic per iteration.
//!
//! Run: `cargo test -p svm-llvm --release --test lua_real_programs -- --ignored --nocapture`

mod lua_rolled;
mod peval_capture;

use std::hint::black_box;
use std::time::{Duration, Instant};

use lua_rolled::{auto_rolled, has_br_table, lua_module};
use svm_ir::{Block, Func, Inst, LoadOp, Module, Terminator, ValType};

fn jit_run(m: &Module, e: u32, a: &[i64]) -> i64 {
    match svm_jit::compile_and_run(m, e, a) {
        Ok(svm_jit::JitOutcome::Returned(v)) => v[0],
        o => panic!("jit {o:?}"),
    }
}

/// Wrap the rolled residual (params = dynamic cells) with a `(cells…) -> i64` that calls it then loads
/// the accumulator it wrote back.
fn with_readback(residual: &Module, entry: u32, read_addr: u64, nparams: usize) -> (Module, u32) {
    let mut m = residual.clone();
    let wrapper = m.funcs.len() as u32;
    let params: Vec<ValType> = vec![ValType::I64; nparams];
    m.funcs.push(Func {
        params: params.clone(),
        results: vec![ValType::I64],
        blocks: vec![Block {
            params,
            insts: vec![
                Inst::ConstI64(read_addr as i64),
                Inst::Call {
                    func: entry,
                    args: (0..nparams as u32).collect(),
                },
                Inst::Load {
                    op: LoadOp::I64,
                    addr: nparams as u32,
                    offset: 0,
                },
            ],
            term: Terminator::Return(vec![nparams as u32 + 1]),
        }],
    });
    (m, wrapper)
}

/// A real program: a name, the loop body, the initial-`local` declarations, and the accumulator name.
struct Prog {
    name: &'static str,
    decls: &'static str, // e.g. "local s = 0"
    body: &'static str,  // e.g. "s = s + i * i"
    ret: &'static str,   // the variable printed / returned (e.g. "s")
}

fn return_script(p: &Prog, n: u64) -> String {
    format!(
        "{}\nfor i = 1, {n} do {} end\nreturn {}\n",
        p.decls, p.body, p.ret
    )
}
fn print_script(p: &Prog, n: u64) -> String {
    format!(
        "{}\nfor i = 1, {n} do {} end\nprint({})\n",
        p.decls, p.body, p.ret
    )
}

fn best(reps: usize, mut f: impl FnMut()) -> Duration {
    f(); // warmup
    let mut b = Duration::MAX;
    for _ in 0..reps {
        let t = Instant::now();
        f();
        b = b.min(t.elapsed());
    }
    b
}

#[test]
#[ignore = "benchmark; run with --release -- --ignored --nocapture"]
fn real_program_gallery() {
    let m = lua_module();
    let reps = 5;
    // differential-N trip counts (cancels compile + parse on both sides). A 20M-iteration delta keeps
    // the per-iteration number stable against shared-runner noise (the interpreter side especially).
    let (n1, n2) = (2_000_000u64, 22_000_000u64);
    let sample = 50u64; // a small trip count for the displayed result

    let progs = [
        Prog {
            name: "sum 1..n",
            decls: "local s = 0",
            body: "s = s + i",
            ret: "s",
        },
        Prog {
            name: "sum of squares",
            decls: "local s = 0",
            body: "s = s + i * i",
            ret: "s",
        },
        Prog {
            name: "polynomial i²-i",
            decls: "local s = 0",
            body: "s = s + i * i - i",
            ret: "s",
        },
        Prog {
            name: "powers of two",
            decls: "local p = 1",
            body: "p = p + p",
            ret: "p",
        },
        Prog {
            name: "fibonacci",
            decls: "local a = 0\nlocal b = 1",
            body: "local t = a + b a = b b = t",
            ret: "b",
        },
        Prog {
            name: "count odds (i%d)",
            decls: "local d = 2\nlocal s = 0",
            body: "s = s + i % d",
            ret: "s",
        },
        Prog {
            name: "sum thirds (i//d)",
            decls: "local d = 3\nlocal s = 0",
            body: "s = s + i // d",
            ret: "s",
        },
    ];

    println!(
        "\n{:<18} {:>14} {:>16} {:>16} {:>10}",
        "program", "output(n=50)", "interp ns/iter", "residual ns/iter", "speedup"
    );
    println!("{}", "-".repeat(78));

    for p in &progs {
        // Zero-config rolled residual (specialized once at a fixed trip count; it is N-independent).
        let ar = auto_rolled(&m, &return_script(p, sample));
        assert!(
            !has_br_table(&ar.residual.funcs[ar.entry as usize]),
            "{}: dispatch folds",
            p.name
        );
        let np = ar.dyn_cells.len();
        // `ar.acc_addr` is the returned register (decoded from the chunk's RETURN1 bytecode) — correct
        // even for multi-accumulator chunks like fibonacci, whose returned `b` is not the lowest cell.
        let (wm, we) = with_readback(&ar.residual, ar.entry, ar.acc_addr, np);
        svm_verify::verify_module(&wm).expect("residual verifies");

        // ---- Program output at n=sample: residual vs interpreter must agree. ----
        let resid_out = jit_run(&wm, we, &ar.captured);
        let interp_stdout = svm_run::run_powerbox(&m, print_script(p, sample).as_bytes())
            .expect("interp")
            .stdout;
        let interp_out: i64 = String::from_utf8_lossy(&interp_stdout)
            .trim()
            .parse()
            .expect("int output");
        assert_eq!(
            resid_out, interp_out,
            "{}: residual == interpreter output",
            p.name
        );

        // ---- Interpreter per-iteration, differential-N (whole-program, compile+parse cancel). ----
        let mr = &m;
        let run_interp = |n: u64| {
            let s = print_script(p, n);
            move || {
                black_box(svm_run::run_powerbox(mr, s.as_bytes()).expect("run").stdout);
            }
        };
        let ti1 = best(reps, run_interp(n1));
        let ti2 = best(reps, run_interp(n2));
        let interp_ns = (ti2.saturating_sub(ti1)).as_nanos() as f64 / (n2 - n1) as f64;

        // ---- Residual per-iteration, differential-N (same module, only the counter arg changes; the
        // in-flight-iteration offset is constant so it cancels in the delta). ----
        let seed_at = |c: u64| {
            let mut a = ar.captured.clone();
            a[ar.counter_ix] = c as i64;
            a
        };
        let (a1, a2) = (seed_at(n1), seed_at(n2));
        let tr1 = best(reps, || {
            black_box(jit_run(&wm, we, &a1));
        });
        let tr2 = best(reps, || {
            black_box(jit_run(&wm, we, &a2));
        });
        let resid_ns = (tr2.saturating_sub(tr1)).as_nanos() as f64 / (n2 - n1) as f64;

        println!(
            "{:<18} {:>14} {:>13.1} ns {:>13.2} ns {:>8.1}x",
            p.name,
            resid_out,
            interp_ns,
            resid_ns,
            interp_ns / resid_ns
        );
    }
    println!("\n(residual = zero-config auto_rolled; both per-iteration via differential-N)");
}
