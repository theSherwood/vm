//! **Guard-and-deopt on a real Lua opcode's cold arm.**
//!
//! The pure/allocating rolled loops fold every opcode whose fast path is monomorphic. The general
//! tiered-JIT shape is different: an opcode with a **foldable fast path and an unfoldable cold arm**
//! (a type-check miss, an overflow, an *error raise*). This drives that on real `luaV_execute` using
//! integer `//`: `OP_IDIV` guards `divisor == 0` and, on zero, calls `luaG_runerror` — the head of the
//! setjmp error machinery the specializer must not project. With the divisor a runtime input
//! ([`SpecConfig::dynamic_cells`]) the guard is a **dynamic branch**, and marking the zero arm a
//! [`SpecConfig::deopt_targets`] block makes the residual fold the division on the hot path and
//! **deopt** the zero case to the carried baseline `luaV_execute` (which raises the real Lua error).
//!
//! The deopt targets are found by name: every `luaV_execute` block that directly calls `luaG_runerror`
//! (the inlined arithmetic error arms). Only the IDIV zero / `INT_MIN // -1` arms are reachable in this
//! script — the rest are never entered at spec time, so marking them all is harmless. The deopt
//! handler is the carried baseline `luaV_execute` itself ([`SpecConfig::carry_whole_module`]), which
//! resumes from the spilled `savedpc`.
//!
//! Result: `luaV_execute` (841 blocks) rolls to a ~157-block dispatch-free entry; the integer division
//! stays a residual op on the hot path, and the `d == 0` / overflow guards deopt to the baseline. End
//! to end (d != 0) it computes `s = 50 * (100 // d)` byte-identically on all three backends.
//!
//! Run: `cargo test -p svm-llvm --test lua_futamura_deopt -- --nocapture`

use svm_interp::{IrPc, Stop, StopReason, Value, WatchKind};
use svm_ir::{Block, Data, Func, Inst, LoadOp, Module, Terminator, ValType};
use svm_peval::{specialize_with_config, SpecArg, SpecConfig};

const L_STACK_LAST: u64 = 40;
const L_STACK: u64 = 48;
const L_CI: u64 = 32;
const LUA_STATE_SIZE: usize = 200;
const CI_SIZE: usize = 104;

// `d` (the divisor) is a loop-invariant local; marking its register dynamic makes OP_IDIV's zero
// guard a dynamic branch. `a`/`d` are locals so both IDIV operands are registers.
const SCRIPT: &str =
    "local s = 0\nlocal a = 100\nlocal d = 7\nfor i = 1, 50 do s = s + a // d end\nreturn s\n";
const CAPTURE_LEN: usize = 8 << 20;

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
/// Capture the loop-body safepoint: (sp, L, ci, base, counter, window). The FORLOOP counter cell is
/// found by watching the candidate and stopping once it holds a small positive VNUMINT while the
/// accumulator is still 0.
#[allow(clippy::type_complexity)]
fn capture_loop_body(
    m: &Module,
    luav: u32,
    counter_off: u64,
    acc_off: u64,
) -> (i64, u64, u64, u64, Vec<u8>) {
    let inst = svm_run::instantiate(m.clone()).expect("instantiate");
    let mut insp = inst.debug_attach(SCRIPT.as_bytes().to_vec(), u64::MAX);
    let entry = IrPc {
        module: 0,
        func: luav,
        block: 0,
        inst: 0,
    };
    insp.set_breakpoint(entry);
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
    insp.clear_breakpoint(entry);
    let w0 = insp.read_window(0, CAPTURE_LEN).expect("window");
    let func = rd_u64(&w0, ci + CI_FUNC);
    let base = func + STACKVALUE_SIZE;
    let counter = base + counter_off;
    let acc = base + acc_off;
    insp.set_watchpoint(counter, 8, WatchKind::Write);
    let window = loop {
        match insp.run_until_stop() {
            Stop::Break {
                reason: StopReason::Watchpoint { write: true, .. },
                ..
            } => {
                insp.step();
                let w = insp.read_window(0, CAPTURE_LEN).expect("window");
                if w[(counter + 8) as usize] == 0x03
                    && w[(acc + 8) as usize] == 0x03
                    && rd_u64(&w, acc) == 0
                    && rd_u64(&w, counter) > 0
                {
                    break w;
                }
            }
            o => panic!("did not reach loop body: {o:?}"),
        }
    };
    (sp, l, ci, base, window)
}

fn has_br_table(f: &svm_ir::Func) -> bool {
    f.blocks
        .iter()
        .any(|b| matches!(b.term, Terminator::BrTable { .. }))
}
fn n_return_calls(f: &svm_ir::Func) -> usize {
    f.blocks
        .iter()
        .filter(|b| matches!(b.term, Terminator::ReturnCall { .. }))
        .count()
}

#[test]
fn idiv_zero_arm_deopts_while_the_fast_path_folds() {
    let m = lua_module();
    let luav = luav_execute(&m);
    let runerror = m
        .exports
        .iter()
        .find(|e| e.name.contains("luaG_runerror"))
        .unwrap()
        .func;
    // The inlined luaG_runerror arms in luaV_execute (from the dump). Only the IDIV zero arm is
    // reachable in this script; the rest are harmless (never entered at spec time).
    let runerror_blocks: Vec<u32> = m.funcs[luav as usize]
        .blocks
        .iter()
        .enumerate()
        .filter(|(_, b)| {
            b.insts
                .iter()
                .any(|i| matches!(i, Inst::Call { func, .. } if *func == runerror))
        })
        .map(|(bi, _)| bi as u32)
        .collect();

    // counter (the FORLOOP countdown) is at base+80; accumulator s at base+16.
    let (sp, l, ci, base, w) = capture_loop_body(&m, luav, 80, 16);
    let stack_lo = rd_u64(&w, l + L_STACK);
    let stack_hi = rd_u64(&w, l + L_STACK_LAST);
    assert_eq!(rd_u64(&w, l + L_CI), ci, "L->ci");
    let func = rd_u64(&w, ci + CI_FUNC);
    let closure = rd_u64(&w, func);
    let proto = rd_u64(&w, closure + LCLOSURE_P);
    let code = rd_u64(&w, proto + PROTO_CODE);
    let sizecode = rd_i32(&w, proto + PROTO_SIZECODE) as usize;
    let d_addr = base + 48;
    println!(
        "\ncapture: base={base:#x} d@{d_addr:#x}={} savedpc=code+{} runerror_blocks={runerror_blocks:?}",
        rd_u64(&w, d_addr) as i64,
        (rd_u64(&w, ci + 32) - code) / 4
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
        // Loop-carried cells (s, the two loop indices, the counter, the IDIV temp) + the divisor `d`
        // (loop-invariant, but dynamic so OP_IDIV's `divisor == 0` guard is a dynamic branch).
        dynamic_cells: vec![
            (base + 16, 8),  // s
            (base + 48, 8),  // d (divisor)
            (base + 64, 8),  // loop index
            (base + 80, 8),  // FORLOOP counter
            (base + 112, 8), // visible i
            (base + 128, 8), // IDIV temp
        ],
        deopt_targets: runerror_blocks.iter().map(|&b| (luav, b)).collect(),
        deopt_handler: Some(luav), // the carried baseline luaV_execute resumes from savedpc
        carry_whole_module: true,
        indirect_targets_cap: Some(16),
        ..SpecConfig::default()
    };

    let args = [
        SpecArg::ConstI64(sp),
        SpecArg::ConstI64(l as i64),
        SpecArg::ConstI64(ci as i64),
    ];
    let r = specialize_with_config(&m, luav, &args, &cfg).expect(
        "IDIV loop specializes with the zero arm deopting (run with svm-peval/trace on a wall)",
    );
    svm_verify::verify_module(&r).expect("residual verifies");
    let entry = (r.funcs.len() - 1) as u32;
    let f = &r.funcs[entry as usize];
    println!(
        "\nDEOPT: luaV_execute {} blocks -> rolled entry {} blocks; br_table={} deopt_tail_calls={}",
        m.funcs[luav as usize].blocks.len(),
        f.blocks.len(),
        has_br_table(f),
        n_return_calls(f),
    );
    // Fast path folds (dispatch gone), the division stays a residual op on the hot path, and exactly
    // one edge — the `d == 0` guard — deopts to the baseline.
    assert!(!has_br_table(f), "the dispatch folded");
    assert!(
        n_return_calls(f) >= 1,
        "the IDIV zero arm deopts to the baseline (a tail-call)"
    );
    assert!(
        f.blocks.len() <= 200,
        "the loop rolled ({} blocks)",
        f.blocks.len()
    );
    println!("  IDIV zero arm deopts to baseline; division folds on the hot path; loop rolled");

    // End-to-end (d != 0, the hot path). Seed the residual's memory with the captured heap and run the
    // rolled entry: the six loop-carried cells are the entry's dynamic parameters, in address order
    // (s, d, i1, counter, i2, temp). With the counter seeded to 49 the loop runs 50 times, so
    // s = 50 * (100 // d) — the real Lua result. Byte-identical on all three backends proves the folded
    // division fast path is correct; the `d == 0` case takes the deopt tail-call instead (checked
    // structurally above — resuming the baseline error raise end-to-end needs the pcall setjmp
    // re-established, which is the separate re-embedding step).
    let x_addr = base + 16; // s
    let cap = |off: u64| rd_u64(&w, base + off) as i64;
    let (i1, counter0, i2, temp) = (cap(64), cap(80), cap(112), cap(128));
    let params = |d: i64| [0i64, d, i1, counter0, i2, temp];
    for d in [7i64, 3, 10, 100, 200] {
        let want = 50 * (100 / d);
        let p = params(d);
        assert_eq!(
            run_with_heap(&r, entry, &w, x_addr, &p),
            want,
            "tree-walk @ d={d}"
        );
        assert_eq!(
            bc_with_heap(&r, entry, &w, x_addr, &p),
            want,
            "bytecode @ d={d}"
        );
    }
    // Exercise the JIT once (it compiles the whole carried runtime): the captured d = 7 ⇒ s = 700.
    assert_eq!(
        jit_with_heap(&r, entry, &w, x_addr, &params(7)),
        700,
        "svm-jit @ d=7"
    );
    println!(
        "  executed: s = 50*(100//d) across a divisor sweep — folded division, all 3 backends"
    );
}

/// Seed the residual's memory with the captured heap `w`, append an entry that calls the rolled loop
/// `entry(p...)` then loads the accumulator cell, and return the wrapped module + wrapper index.
fn wrap_with_heap(
    residual: &Module,
    entry: u32,
    w: &[u8],
    x_addr: u64,
    np: usize,
) -> (Module, u32) {
    let mut m = residual.clone();
    m.data.push(Data {
        offset: 0,
        readonly: false,
        bytes: w.to_vec(),
    });
    let wrapper = m.funcs.len() as u32;
    let params = vec![ValType::I64; np];
    let call_args: Vec<u32> = (0..np as u32).collect();
    let addr_v = np as u32; // ConstI64 result index
    m.funcs.push(Func {
        params: params.clone(),
        results: vec![ValType::I64],
        blocks: vec![Block {
            params,
            insts: vec![
                Inst::ConstI64(x_addr as i64),
                Inst::Call {
                    func: entry,
                    args: call_args,
                },
                Inst::Load {
                    op: LoadOp::I64,
                    addr: addr_v,
                    offset: 0,
                },
            ],
            term: Terminator::Return(vec![addr_v + 1]),
        }],
    });
    (m, wrapper)
}
fn run_with_heap(r: &Module, e: u32, w: &[u8], x: u64, p: &[i64]) -> i64 {
    let (m, wr) = wrap_with_heap(r, e, w, x, p.len());
    let mut fuel = u64::MAX;
    let vs: Vec<Value> = p.iter().map(|&a| Value::I64(a)).collect();
    match svm_interp::run(&m, wr, &vs, &mut fuel)
        .expect("no trap")
        .as_slice()
    {
        [Value::I64(v)] => *v,
        o => panic!("bad {o:?}"),
    }
}
fn bc_with_heap(r: &Module, e: u32, w: &[u8], x: u64, p: &[i64]) -> i64 {
    let (m, wr) = wrap_with_heap(r, e, w, x, p.len());
    let mut fuel = u64::MAX;
    let vs: Vec<Value> = p.iter().map(|&a| Value::I64(a)).collect();
    match svm_interp::bytecode::compile_and_run(&m, wr, &vs, &mut fuel)
        .expect("subset")
        .expect("no trap")
        .as_slice()
    {
        [Value::I64(v)] => *v,
        o => panic!("bad {o:?}"),
    }
}
fn jit_with_heap(r: &Module, e: u32, w: &[u8], x: u64, p: &[i64]) -> i64 {
    let (m, wr) = wrap_with_heap(r, e, w, x, p.len());
    match svm_jit::compile_and_run(&m, wr, p) {
        Ok(svm_jit::JitOutcome::Returned(v)) => v[0],
        o => panic!("jit {o:?}"),
    }
}
