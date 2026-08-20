//! **Rolling the real `luaV_execute` with a runtime-input register.**
//!
//! `lua_futamura_vmstate.rs` projects `luaV_execute` fully for a fixed script — the loop *unrolls* to
//! a constant, because every seeded cell folds. A *rolled, input-taking* residual (the shape #613
//! documented as the hard target) needs the loop's trip count to stay dynamic. This drives that:
//!
//!  1. Capture the VM at the **loop body** (a real safepoint: after `FORPREP`, `ci->u.l.savedpc`
//!     naturally points at the first body opcode, the loop control registers are live). This is the
//!     same "resume a paused VM" model as the `savedpc` tests — the capture *is* a valid resume image.
//!  2. Seed the whole VM state (`L` / stack / `ci` + the read-only heap) so the dispatch folds — but
//!     mark the **FORLOOP counter cell** a [`SpecConfig::dynamic_cells`] runtime input, so the loop
//!     condition stays dynamic and the loop **rolls** instead of unrolling.
//!
//! The result is a rolled, dispatch-free residual of the *real* `luaV_execute`: 841 blocks collapse
//! to ~53, with **no `br_table`** (the 83-way dispatch folded), **no loads** (all VM state folded to
//! SSA), and **no calls / `call_indirect`** — it takes the loop's three live-in cells (the accumulator
//! `x`, the loop variable `i`, the trip counter) as dynamic parameters and rolls. Correctness: the
//! residual writes the final accumulator back into the window, so a small appended wrapper reads it
//! out; for the captured inputs it reproduces the real Lua answer (`x = 3*50 = 150`, the value
//! `lua_futamura_vmstate.rs` folds for this script), and across a sweep of the dynamic trip count it
//! matches `x0 + 3*(c0+1)` byte-identically on all three backends (tree-walk / bytecode / svm-jit).
//!
//! *Scope.* This is a pure integer loop, so its exit continuation folds cleanly (a seeded `ci`, no
//! allocation ⇒ no GC step, no error raise) and the cut-set / guard-and-deopt machinery is **not**
//! needed here — the missing piece was purely `dynamic_cells` (the runtime-input register). A loop
//! whose body allocates (string/table ops → `luaC_checkGC`) would reach the GC/error call-outs #613
//! flagged; cutting/deopting *those* is the next target, and the machinery for it already exists
//! (`cut_calls` / `deopt_targets`, exercised on ported C in `peval_savedpc_ported.rs`).
//!
//! Run: `cargo test -p svm-llvm --test lua_futamura_rolled -- --nocapture`
//! (add `--features svm-peval/trace` to print the first specializer blocker while iterating).

use svm_interp::{IrPc, Stop, StopReason, Value, WatchKind};
use svm_ir::{Block, Func, Inst, LoadOp, Module, Terminator, ValType};
use svm_peval::{specialize_with_config, SpecArg, SpecConfig};

const SCRIPT: &str = "local x = 0\nfor i = 1, 50 do x = x + 3 end\nreturn x\n";
const CAPTURE_LEN: usize = 8 << 20;

// Lua 5.4.7 offsets.
const L_STACK_LAST: u64 = 40;
const L_STACK: u64 = 48;
const L_CI: u64 = 32;
const LUA_STATE_SIZE: usize = 200;
const CI_SIZE: usize = 104;
const CI_FUNC: u64 = 0;
const LCLOSURE_P: u64 = 24;
const PROTO_SIZECODE: u64 = 24;
const PROTO_CODE: u64 = 64;
const STACKVALUE_SIZE: u64 = 16;

fn lua_module() -> Module {
    let path = format!(
        "{}/tests/fixtures/lua/lua_eval.ll",
        env!("CARGO_MANIFEST_DIR")
    );
    svm_llvm::translate_ll_path(&path)
        .expect("translate")
        .module
}
fn luav_execute(m: &Module) -> u32 {
    m.exports
        .iter()
        .find(|e| e.name == "luaV_execute")
        .expect("export")
        .func
}
fn rd_u64(w: &[u8], a: u64) -> u64 {
    u64::from_le_bytes(w[a as usize..a as usize + 8].try_into().unwrap())
}
fn rd_i32(w: &[u8], a: u64) -> i32 {
    i32::from_le_bytes(w[a as usize..a as usize + 4].try_into().unwrap())
}

/// The captured loop-body safepoint: the entry `sp`, the `L`/`ci` pointers, the register base, the
/// FORLOOP counter cell address, and the full window image at that point.
struct LoopEntry {
    sp: i64,
    l: u64,
    ci: u64,
    base: u64,
    counter: u64,
    window: Vec<u8>,
}

/// Break at `luaV_execute` entry (read `sp`/`L`/`ci` + the register base), then advance to the loop
/// body via a write-watchpoint on the counter cell, stopping once the loop is set up (the counter is
/// a small positive integer with the VNUMINT tag and the accumulator is still 0).
fn capture_loop_entry(m: &Module, luav: u32) -> LoopEntry {
    let inst = svm_run::instantiate(m.clone()).expect("instantiate");
    let mut insp = inst.debug_attach(SCRIPT.as_bytes().to_vec(), u64::MAX);
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

    // ci->func -> register base. R[0] = ci->func + 1 TValue; the FORLOOP counter is R[3] here (the
    // `for i=1,50` loop's internal countdown), captured empirically as the cell that counts down.
    let w0 = insp.read_window(0, CAPTURE_LEN).expect("window");
    let func = rd_u64(&w0, ci + CI_FUNC);
    let base = func + STACKVALUE_SIZE;
    let counter = base + 3 * STACKVALUE_SIZE;

    // Advance to the loop body: watch writes to the counter cell; stop the first time it holds the
    // freshly-computed trip count (VNUMINT, and the accumulator still 0 — the body hasn't run). The
    // accumulator `x` sits at R[1] (base+16) after VARARGPREP shifts the main chunk's registers.
    let acc = base + STACKVALUE_SIZE; // x
    insp.set_watchpoint(counter, 8, WatchKind::Write);
    let window = loop {
        match insp.run_until_stop() {
            Stop::Break {
                reason: StopReason::Watchpoint { write: true, .. },
                ..
            } => {
                insp.step(); // apply the write
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

fn has_br_table(f: &svm_ir::Func) -> bool {
    f.blocks
        .iter()
        .any(|b| matches!(b.term, Terminator::BrTable { .. }))
}

/// The rolled `luaV_execute` residual returns void — its result is the loop-carried accumulator it
/// *writes back* into the window. To read it out we append a wrapper `(x0,i0,c0) -> i64` that calls
/// the residual (entry 0) and loads the accumulator cell it spilled, then run that wrapper on each
/// backend. (A pure residual — loads=0 — writes back the final VM state on return.)
fn with_readback(residual: &Module, read_addr: u64) -> (Module, u32) {
    let mut m = residual.clone();
    let wrapper = m.funcs.len() as u32;
    let block = Block {
        params: vec![ValType::I64, ValType::I64, ValType::I64], // x0, i0, c0
        insts: vec![
            Inst::ConstI64(read_addr as i64), // v3
            Inst::Call {
                func: 0,
                args: vec![0, 1, 2],
            }, // void
            Inst::Load {
                op: LoadOp::I64,
                addr: 3,
                offset: 0,
            }, // v4
        ],
        term: Terminator::Return(vec![4]),
    };
    m.funcs.push(Func {
        params: vec![ValType::I64, ValType::I64, ValType::I64],
        results: vec![ValType::I64],
        blocks: vec![block],
    });
    (m, wrapper)
}
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
fn bc(m: &Module, e: u32, a: &[i64]) -> i64 {
    let mut fuel = u64::MAX;
    let vs: Vec<Value> = a.iter().map(|&x| Value::I64(x)).collect();
    match svm_interp::bytecode::compile_and_run(m, e, &vs, &mut fuel)
        .expect("subset")
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

#[test]
fn dynamic_counter_rolls_the_real_luav_execute_loop() {
    let m = lua_module();
    let luav = luav_execute(&m);
    let cap = capture_loop_entry(&m, luav);
    let (l, ci, w) = (cap.l, cap.ci, &cap.window);

    let stack_lo = rd_u64(w, l + L_STACK);
    let stack_hi = rd_u64(w, l + L_STACK_LAST);
    assert_eq!(rd_u64(w, l + L_CI), ci, "L->ci != captured ci");
    assert!(
        stack_lo <= cap.counter && cap.counter < stack_hi,
        "counter on stack"
    );

    // The read-only heap chased by the (already-run) prologue: ci->func -> LClosure -> Proto -> code.
    let func = rd_u64(w, ci + CI_FUNC);
    let closure = rd_u64(w, func);
    let proto = rd_u64(w, closure + LCLOSURE_P);
    let code = rd_u64(w, proto + PROTO_CODE);
    let sizecode = rd_i32(w, proto + PROTO_SIZECODE) as usize;

    let saved = rd_u64(w, ci + 32); // ci->u.l.savedpc — points at the loop body after FORPREP
    let counter0 = rd_u64(w, cap.counter);
    println!("\n=== loop-body capture ===");
    println!(
        "L={l:#x} ci={ci:#x} base={:#x} counter@{:#x}={counter0}",
        cap.base, cap.counter
    );
    println!(
        "savedpc={saved:#x} code={code:#x} (body pc = {})",
        (saved - code) / 4
    );

    let slice = |a: u64, n: usize| w[a as usize..a as usize + n].to_vec();
    let stack_len = (stack_hi - stack_lo) as usize;

    let cfg = SpecConfig {
        const_overlays: vec![
            (l, slice(l, LUA_STATE_SIZE)),
            (stack_lo, slice(stack_lo, stack_len)),
            (ci, slice(ci, CI_SIZE)),
            (code, slice(code, 4 * sizecode)),
            (closure, slice(closure, 48)),
            (proto, slice(proto, 128)),
        ],
        rename: Some((l, l + LUA_STATE_SIZE as u64)),
        rename_extra: vec![(stack_lo, stack_hi), (ci, ci + CI_SIZE as u64)],
        rename_is_private: true,
        rename_seed_from_image: true,
        // Every loop-carried cell must be dynamic for the memo to converge (a cell that folds to a
        // *different constant each iteration* forces the loop to unroll). Here that is the accumulator
        // `x` (R1), the loop variable `i` (R2), and the FORLOOP counter (R3); the step (R4) is loop-
        // invariant and stays folded.
        dynamic_cells: vec![
            (cap.base + STACKVALUE_SIZE, 8),     // x
            (cap.base + 2 * STACKVALUE_SIZE, 8), // i
            (cap.counter, 8),                    // FORLOOP counter
        ],
        indirect_targets_cap: Some(16),
        ..SpecConfig::default()
    };

    let args = [
        SpecArg::ConstI64(cap.sp),
        SpecArg::ConstI64(l as i64),
        SpecArg::ConstI64(ci as i64),
    ];
    let r = specialize_with_config(&m, luav, &args, &cfg)
        .expect("real luaV_execute rolls (run with svm-peval/trace to see any wall)");
    svm_verify::verify_module(&r).expect("residual re-verifies");

    let f = &r.funcs[0];
    let (mut loads, mut calls, mut indirect) = (0, 0, 0);
    for b in &f.blocks {
        for i in &b.insts {
            match i {
                Inst::Load { .. } => loads += 1,
                Inst::Call { .. } => calls += 1,
                Inst::CallIndirect { .. } => indirect += 1,
                _ => {}
            }
        }
        if matches!(b.term, Terminator::ReturnCallIndirect { .. }) {
            indirect += 1;
        }
    }
    println!(
        "\nROLLED: real luaV_execute {} blocks -> residual {} blocks, {} params, results={:?}",
        m.funcs[luav as usize].blocks.len(),
        f.blocks.len(),
        f.params.len(),
        f.results
    );
    println!(
        "  br_table={}  loads={loads} calls={calls} indirect={indirect}",
        has_br_table(f)
    );

    // Structure: the 83-way dispatch folded, the loop rolled to a bounded block count, and the residual
    // takes the loop's three live-in cells (x, i, trip counter) as dynamic parameters.
    assert!(!has_br_table(f), "the dispatch must fold");
    assert_eq!(f.params.len(), 3, "residual takes (x0, i0, counter0)");
    assert_eq!(indirect, 0, "no residual call_indirect (cold paths pruned)");
    assert!(
        f.blocks.len() <= 120,
        "the loop rolled (got {} blocks)",
        f.blocks.len()
    );

    // Correctness. The captured run is `for i=1,50 do x=x+3 end` resumed at the loop body with counter
    // = 49 (49 back-edges + the in-flight iteration = 50 body executions), so the real Lua result is
    // x = 3*50 = 150 — the value `lua_futamura_vmstate.rs` folds for this exact script. Reading the
    // residual's written-back accumulator for the captured inputs must reproduce it, on all backends.
    let x_addr = cap.base + STACKVALUE_SIZE;
    let (wm, we) = with_readback(&r, x_addr);
    svm_verify::verify_module(&wm).expect("wrapped residual verifies");
    let cap_args = [0i64, 1, counter0 as i64];
    assert_eq!(
        tw(&wm, we, &cap_args),
        150,
        "tree-walk residual @ captured inputs"
    );
    assert_eq!(
        bc(&wm, we, &cap_args),
        150,
        "bytecode residual @ captured inputs"
    );
    assert_eq!(
        jit(&wm, we, &cap_args),
        150,
        "svm-jit residual @ captured inputs"
    );

    // Rolled correctness across the *dynamic* trip count: with counter0 = c, the loop runs c+1 body
    // executions, so x = x0 + 3*(c+1). Byte-identical across the three backends over a sweep proves the
    // loop genuinely rolled and computes correctly parameterized by the runtime input (a baked constant
    // would give one fixed answer regardless of the argument).
    for (x0, i0, c0) in [
        (0i64, 1i64, 0i64),
        (0, 1, 7),
        (10, 1, 20),
        (0, 1, 99),
        (5, 1, 1000),
    ] {
        let want = x0 + 3 * (c0 + 1);
        let a = [x0, i0, c0];
        let (t, b, j) = (tw(&wm, we, &a), bc(&wm, we, &a), jit(&wm, we, &a));
        assert_eq!(
            (t, b, j),
            (want, want, want),
            "residual @ (x0={x0}, c0={c0})"
        );
    }
    println!(
        "  verified: 150 @ captured inputs; x0+3*(c0+1) across a trip-count sweep, all 3 backends"
    );
}
