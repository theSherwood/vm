//! **Auto-rolled residual: zero-config, via the driver-level `lua_rolled::auto_rolled`.**
//!
//! The rolled residual (the shape that delivers the 11.5× per-iteration win) is now built by a
//! reusable driver function — `lua_rolled::auto_rolled(m, script)` — on top of the shared
//! interpreter-agnostic `peval_capture::discover`. This test drives it with no hardcoded register
//! offsets: it recovers the loop's carried cells, identifies the counter and accumulator, and the
//! real `luaV_execute` loop rolls (dispatch folds, byte-identical across backends).
//!
//! Run: `cargo test -p svm-llvm --test lua_futamura_auto_rolled -- --nocapture`

mod lua_rolled;
mod peval_capture;

use lua_rolled::{auto_rolled, has_br_table, lua_module, with_readback};
use svm_interp::Value;
use svm_ir::Module;

const SCRIPT: &str = "local x = 0\nfor i = 1, 50 do x = x + 3 end\nreturn x\n";

fn tw(m: &Module, e: u32, a: &[i64]) -> i64 {
    let mut fuel = u64::MAX;
    let vs: Vec<Value> = a.iter().map(|&x| Value::I64(x)).collect();
    match svm_interp::run(m, e, &vs, &mut fuel)
        .expect("no trap")
        .as_slice()
    {
        [Value::I64(x)] => *x,
        o => panic!("bad {o:?}"),
    }
}
fn jit(m: &Module, e: u32, a: &[i64]) -> i64 {
    match svm_jit::compile_and_run(m, e, a) {
        Ok(svm_jit::JitOutcome::Returned(v)) => v[0],
        o => panic!("jit {o:?}"),
    }
}

/// The real Lua answer for a `return <int>` chunk, via the reference interpreter (rewrite the trailing
/// `return X` to `print(X)` and parse stdout). Used as the ground-truth oracle for the coverage test.
fn real_lua(m: &Module, script: &str) -> i64 {
    let mut prog = script.replacen("return ", "print(", 1).trim_end().to_string();
    prog.push_str(")\n");
    let out = svm_run::run_powerbox(m, prog.as_bytes())
        .expect("interp run")
        .stdout;
    String::from_utf8_lossy(&out).trim().parse().expect("int stdout")
}

#[test]
fn auto_discovers_the_hand_built_loop_cells() {
    let m = lua_module();
    let r = auto_rolled(&m, SCRIPT);
    let as_reg = |a: u64| (a - r.reg_base) / r.reg_stride;
    println!(
        "\nauto_rolled cells (as R[n]): {:?}  counter=R{} acc=R{}",
        r.dyn_cells.iter().map(|&a| as_reg(a)).collect::<Vec<_>>(),
        as_reg(r.dyn_cells[r.counter_ix]),
        as_reg(r.acc_addr),
    );
    // The hand-built rolled test hardcodes exactly x=R[1], i=R[2], counter=R[3]; auto-discovery must
    // recover that set (an extra internal FORLOOP register is harmless).
    for reg in [1u64, 2, 3] {
        let a = r.reg_base + reg * r.reg_stride;
        assert!(r.dyn_cells.contains(&a), "auto discovery missed R[{reg}]");
    }
    assert_eq!(as_reg(r.dyn_cells[r.counter_ix]), 3, "counter is R[3]");
    assert_eq!(
        as_reg(r.acc_addr),
        1,
        "accumulator is R[1] (first local before the loop)"
    );
}

#[test]
fn auto_rolled_residual_rolls_and_is_correct() {
    let m = lua_module();
    let r = auto_rolled(&m, SCRIPT);
    let f = &r.residual.funcs[r.entry as usize];
    let n = r.dyn_cells.len();
    println!(
        "\nAUTO-ROLLED: {} blocks -> {} blocks, {} params (dyn cells), br_table={}",
        r.base_blocks,
        r.residual_blocks,
        f.params.len(),
        has_br_table(f)
    );
    assert!(!has_br_table(f), "the dispatch must fold");
    assert_eq!(
        f.params.len(),
        n,
        "residual takes the discovered dynamic cells"
    );
    assert!(
        r.residual_blocks <= 160,
        "the loop rolled ({} blocks)",
        r.residual_blocks
    );

    let (wm, we) = with_readback(&r.residual, r.entry, r.acc_addr, n);
    svm_verify::verify_module(&wm).expect("wrapped residual verifies");

    // Captured seed reproduces the real Lua answer (`for i=1,50 do x=x+3 end` resumed at the body with
    // counter=49 ⇒ x=150), on all backends.
    let counter0 = r.captured[r.counter_ix];
    let want_captured = 3 * (counter0 + 1);
    assert_eq!(
        tw(&wm, we, &r.captured),
        want_captured,
        "captured tree-walk"
    );
    assert_eq!(jit(&wm, we, &r.captured), want_captured, "captured jit");

    // The loop truly ROLLS: sweep the dynamic trip counter, same fixed module, x0 + 3·(c+1).
    for c in [0i64, 1, 7, 100, 1000] {
        let mut a = r.captured.clone();
        a[r.counter_ix] = c;
        let want = 3 * (c + 1);
        assert_eq!(tw(&wm, we, &a), want, "tree-walk c={c}");
        assert_eq!(jit(&wm, we, &a), want, "jit c={c}");
    }
    println!(
        "rolled + correct across a trip sweep (counter param index {})",
        r.counter_ix
    );
}

/// Integer `%` and `//` fold and roll through the zero-config path: `auto_rolled` marks each op's
/// divisor + result registers dynamic and deopts the divide-by-zero cold arm to the carried baseline,
/// so the division stays a residual op on the hot path while the dispatch still folds. The divisor is
/// a *local* (`d`) — a register operand, so the guard is inline in `luaV_execute` (the constant K-form
/// `i % 2` calls an un-inlined helper and is out of scope). Correctness is checked against real Lua on
/// all backends, across a trip sweep, for both operators.
#[test]
fn auto_rolled_div_and_mod_fold_roll_and_are_correct() {
    let m = lua_module();
    // (name, script, closed-form result for trip count n). `d` is a local ⇒ register-operand div/mod.
    type Closed = fn(i64) -> i64;
    let cases: [(&str, &str, Closed); 2] = [
        (
            "i % d",
            "local d = 3\nlocal s = 0\nfor i = 1, 50 do s = s + i % d end\nreturn s\n",
            |n| (1..=n).map(|i| i % 3).sum(),
        ),
        (
            "i // d",
            "local d = 3\nlocal s = 0\nfor i = 1, 50 do s = s + i // d end\nreturn s\n",
            |n| (1..=n).map(|i| i / 3).sum(),
        ),
    ];
    for (name, script, want_fn) in cases {
        let r = auto_rolled(&m, script);
        let f = &r.residual.funcs[r.entry as usize];
        assert!(!has_br_table(f), "{name}: the dispatch folds");
        let n = r.dyn_cells.len();
        let (wm, we) = with_readback(&r.residual, r.entry, r.acc_addr, n);
        svm_verify::verify_module(&wm).expect("wrapped residual verifies");

        // The captured seed reproduces the real Lua answer at the observed trip count, and the loop
        // rolls: sweeping the counter reproduces the closed form on both backends.
        let counter0 = r.captured[r.counter_ix];
        assert_eq!(
            tw(&wm, we, &r.captured),
            want_fn(counter0 + 1),
            "{name}: captured tree-walk"
        );
        for c in [0i64, 1, 6, 49, 200] {
            let mut a = r.captured.clone();
            a[r.counter_ix] = c;
            let want = want_fn(c + 1);
            assert_eq!(tw(&wm, we, &a), want, "{name}: tree-walk c={c}");
            assert_eq!(jit(&wm, we, &a), want, "{name}: jit c={c}");
        }
        println!(
            "{name}: folds + rolls + correct ({} residual blocks)",
            f.blocks.len()
        );
    }
}

/// **Control-flow breadth.** The zero-config path was built for a straight numeric-`for` accumulator, but
/// the specializer already handles richer control flow *inside* that loop with no extra machinery: a
/// data-dependent `if` in the body (the branch stays dynamic, dispatch still folds), nested `for` loops
/// (the inner loop rolls within the outer safepoint, using both induction variables), and an early
/// `break`. Each folds its dispatch, verifies, and reproduces the reference interpreter on both backends
/// at the captured trip count. This locks that breadth in against regression. (Two known walls stay out
/// of scope: `while`/`repeat` — the FORLOOP-shaped `discover` finds no counter — and the constant K-form
/// `i % 2` — an un-inlined helper, per the div/mod test above.)
#[test]
fn auto_rolled_covers_conditionals_nesting_and_break() {
    let m = lua_module();
    let cases = [
        (
            "if-in-body",
            "local s = 0\nfor i = 1, 50 do if i < 25 then s = s + i end end\nreturn s\n",
        ),
        (
            "nested-for",
            "local s = 0\nfor i = 1, 10 do for j = 1, 10 do s = s + 1 end end\nreturn s\n",
        ),
        (
            "nested-uses-both",
            "local s = 0\nfor i = 1, 10 do for j = 1, 10 do s = s + i * j end end\nreturn s\n",
        ),
        (
            "break",
            "local s = 0\nfor i = 1, 50 do s = s + i if i >= 25 then break end end\nreturn s\n",
        ),
    ];
    for (name, script) in cases {
        let want = real_lua(&m, script);
        let r = auto_rolled(&m, script);
        let f = &r.residual.funcs[r.entry as usize];
        assert!(!has_br_table(f), "{name}: the dispatch must fold");
        let n = r.dyn_cells.len();
        let (wm, we) = with_readback(&r.residual, r.entry, r.acc_addr, n);
        svm_verify::verify_module(&wm).expect("wrapped residual verifies");
        assert_eq!(jit(&wm, we, &r.captured), want, "{name}: jit == real Lua");
        assert_eq!(tw(&wm, we, &r.captured), want, "{name}: tree-walk == real Lua");
        println!("{name}: folds + verifies + correct (= {want})");
    }
}
