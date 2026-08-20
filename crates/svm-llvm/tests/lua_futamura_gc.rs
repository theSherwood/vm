//! **GC-cut: rolling an *allocating* real `luaV_execute` loop.**
//!
//! `lua_futamura_rolled.rs` rolls a pure integer loop — its exit continuation folds cleanly because
//! nothing allocates. This drives the harder case #613 flagged: a loop whose body **allocates** (a
//! table per iteration → `luaH_new` + `luaC_checkGC` → a possible GC step), which drags in the
//! allocator/GC/`setjmp` machinery the specializer must not project. Two features carry it:
//!
//!  - [`SpecConfig::cut_calls_read_state`] keeps the allocator/GC as **opaque call-outs**: the live
//!    register file is spilled to the window before each (so a non-moving GC scans typed registers)
//!    but *not* reloaded (the folded tag constants stay folded, so the dispatch stays collapsed).
//!  - [`SpecConfig::carry_whole_module`] carries the runtime at its **original indices**, because the
//!    allocator/GC closure dispatches through the function table (`(*g->frealloc)(…)`, `g->panic`)
//!    and the runtime funcref values are original indices with no table to remap.
//!
//! The residual is a rolled, dispatch-free fast path whose only calls are the opaque allocator/GC
//! call-outs; end-to-end it makes *real* allocations (the carried arena allocator runs each iteration)
//! and computes the loop result byte-identically on all three backends.
//!
//! Run: `cargo test -p svm-llvm --test lua_futamura_gc -- --nocapture`
//! (add `--features svm-peval/trace` to print the first specializer blocker while iterating).

use svm_interp::{IrPc, Stop, StopReason, Value, WatchKind};
use svm_ir::{Block, Data, Func, Inst, LoadOp, Module, Terminator, ValType};
use svm_peval::{specialize_with_config, SpecArg, SpecConfig};

const SCRIPT: &str = "local x = 0\nfor i = 1, 50 do local t = {}\n x = x + 1 end\nreturn x\n";
const CAPTURE_LEN: usize = 8 << 20;

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
fn export_name(m: &Module, f: u32) -> String {
    m.exports
        .iter()
        .find(|e| e.func == f)
        .map(|e| e.name.clone())
        .unwrap_or_else(|| format!("f{f}"))
}
fn rd_u64(w: &[u8], a: u64) -> u64 {
    u64::from_le_bytes(w[a as usize..a as usize + 8].try_into().unwrap())
}
fn rd_i32(w: &[u8], a: u64) -> i32 {
    i32::from_le_bytes(w[a as usize..a as usize + 4].try_into().unwrap())
}
fn rd_u32(w: &[u8], a: u64) -> u32 {
    u32::from_le_bytes(w[a as usize..a as usize + 4].try_into().unwrap())
}

#[test]
fn dump_allocating_bytecode() {
    let m = lua_module();
    let luav = luav_execute(&m);
    let inst = svm_run::instantiate(m.clone()).expect("instantiate");
    let mut insp = inst.debug_attach(SCRIPT.as_bytes().to_vec(), u64::MAX);
    insp.set_breakpoint(IrPc {
        module: 0,
        func: luav,
        block: 0,
        inst: 0,
    });
    let ci = match insp.run_until_stop() {
        Stop::Break { .. } => match insp.read_ir_value(0, 2) {
            Some(Value::I64(v)) => v as u64,
            o => panic!("{o:?}"),
        },
        o => panic!("no break: {o:?}"),
    };
    let w = insp.read_window(0, CAPTURE_LEN).expect("window");
    let func = rd_u64(&w, ci + CI_FUNC);
    let closure = rd_u64(&w, func);
    let proto = rd_u64(&w, closure + LCLOSURE_P);
    let code = rd_u64(&w, proto + PROTO_CODE);
    let sizecode = rd_i32(&w, proto + PROTO_SIZECODE) as u64;
    println!("\n=== bytecode ({sizecode} instrs) ===");
    for pc in 0..sizecode {
        let i = rd_u32(&w, code + 4 * pc);
        println!(
            "  [{pc:2}] op={:2} A={:3} Bx={:6}",
            i & 0x7f,
            (i >> 7) & 0xff,
            i >> 15
        );
    }

    // Which functions in luaV_execute's closure allocate / step the GC / raise — the cut candidates.
    let mut stack = vec![luav];
    let mut seen = std::collections::BTreeSet::new();
    while let Some(f) = stack.pop() {
        if !seen.insert(f) {
            continue;
        }
        for b in &m.funcs[f as usize].blocks {
            for i in &b.insts {
                if let Inst::Call { func, .. } = i {
                    stack.push(*func);
                }
            }
            if let Terminator::ReturnCall { func, .. } = &b.term {
                stack.push(*func);
            }
        }
    }
    println!("\n=== alloc/GC/error functions in the closure ===");
    for &f in &seen {
        let n = export_name(&m, f);
        if n.contains("luaC_")
            || n.contains("luaM_")
            || n.contains("luaH_new")
            || n.contains("luaH_resize")
            || n.contains("luaD_throw")
            || n.contains("luaD_growstack")
        {
            println!("  f{f}: {n}");
        }
    }

    // The functions luaV_execute *directly* calls (the cut points must be direct callees).
    println!("\n=== luaV_execute direct callees ===");
    let mut direct = std::collections::BTreeSet::new();
    for b in &m.funcs[luav as usize].blocks {
        for i in &b.insts {
            if let Inst::Call { func, .. } = i {
                direct.insert(*func);
            }
        }
        if let Terminator::ReturnCall { func, .. } = &b.term {
            direct.insert(*func);
        }
    }
    for f in &direct {
        println!("  f{f}: {}", export_name(&m, *f));
    }
}

/// Break at entry, read sp/L/ci + register base, then advance to the loop body (watch the counter
/// cell). Returns (sp, L, ci, base, counter_addr, window).
#[allow(clippy::type_complexity)]
fn capture_loop_entry(m: &Module, luav: u32) -> (i64, u64, u64, u64, u64, Vec<u8>) {
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
    let counter = base + 3 * STACKVALUE_SIZE; // R3 cell — the FORLOOP countdown (as in the pure loop)
    let acc = base + STACKVALUE_SIZE; // x
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
    (sp, l, ci, base, counter, window)
}

fn byname(m: &Module, name: &str) -> u32 {
    m.exports
        .iter()
        .find(|e| e.name == name)
        .unwrap_or_else(|| panic!("no export {name}"))
        .func
}

#[test]
fn gc_cut_rolls_the_allocating_loop() {
    let m = lua_module();
    let luav = luav_execute(&m);
    let (sp, l, ci, base, counter, w) = capture_loop_entry(&m, luav);

    let stack_lo = rd_u64(&w, l + L_STACK);
    let stack_hi = rd_u64(&w, l + L_STACK_LAST);
    assert_eq!(rd_u64(&w, l + L_CI), ci, "L->ci mismatch");
    let func = rd_u64(&w, ci + CI_FUNC);
    let closure = rd_u64(&w, func);
    let proto = rd_u64(&w, closure + LCLOSURE_P);
    let code = rd_u64(&w, proto + PROTO_CODE);
    let sizecode = rd_i32(&w, proto + PROTO_SIZECODE) as usize;
    let saved = rd_u64(&w, ci + 32);
    println!(
        "\ncapture: base={base:#x} counter@{counter:#x}={} savedpc=code+{}",
        rd_u64(&w, counter),
        (saved - code) / 4
    );

    let slice = |a: u64, n: usize| w[a as usize..a as usize + n].to_vec();
    let stack_len = (stack_hi - stack_lo) as usize;

    // Cut the allocator + GC + error functions so their deeply-stateful bodies (heap link-in, GC
    // stack scan, the OOM setjmp) are carried opaque instead of projected. They read L / scan the
    // stack, so they are state-touching cuts (spill the rename regions, call, reload).
    let cut_names = [
        "luaH_new",
        "luaH_resize",
        "luaH_resizearray",
        "luaC_newobj",
        "luaC_step",
        "luaC_barrier_",
        "luaC_barrierback_",
        "luaM_malloc_",
        "luaM_realloc_",
        "luaM_free_",
        "luaD_throw",
        "luaD_growstack",
    ];
    let cut: Vec<u32> = cut_names.iter().map(|n| byname(&m, n)).collect();

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
        dynamic_cells: vec![
            (base + STACKVALUE_SIZE, 8),     // x
            (base + 2 * STACKVALUE_SIZE, 8), // i
            (counter, 8),                    // FORLOOP counter
        ],
        cut_calls_read_state: cut,
        carry_whole_module: true, // the alloc/GC closure has indirect calls (frealloc, panic)
        indirect_targets_cap: Some(16),
        ..SpecConfig::default()
    };

    let args = [
        SpecArg::ConstI64(sp),
        SpecArg::ConstI64(l as i64),
        SpecArg::ConstI64(ci as i64),
    ];
    let r = specialize_with_config(&m, luav, &args, &cfg)
        .expect("allocating loop rolls with the alloc/GC cut (run with svm-peval/trace on a wall)");
    svm_verify::verify_module(&r).expect("residual re-verifies");
    // carry_whole_module appends the rolled entry after the carried runtime (index funcs.len()-1).
    let entry = (r.funcs.len() - 1) as u32;
    let f = &r.funcs[entry as usize];
    let (mut brt, mut calls, mut indirect) = (0, 0, 0);
    for b in &f.blocks {
        for i in &b.insts {
            match i {
                Inst::Call { .. } => calls += 1,
                Inst::CallIndirect { .. } => indirect += 1,
                _ => {}
            }
        }
        match b.term {
            Terminator::BrTable { .. } => brt += 1,
            Terminator::ReturnCallIndirect { .. } => indirect += 1,
            _ => {}
        }
    }
    println!(
        "\nROLLED (alloc): luaV_execute {} blocks -> rolled entry {} blocks, {} params; br_table={brt} calls={calls} indirect={indirect}",
        m.funcs[luav as usize].blocks.len(),
        f.blocks.len(),
        f.params.len(),
    );

    // The dispatch folded and the loop rolled; the allocator/GC survive as opaque `call`s on the hot
    // path (one per allocating opcode), not projected. `indirect==0` in the *entry* — the indirect
    // calls live only in the carried runtime, reached through those opaque calls.
    assert!(!has_br_table_f(f), "the dispatch folded");
    assert_eq!(f.params.len(), 3, "residual takes (x0, i0, counter0)");
    assert_eq!(indirect, 0, "no indirect call on the rolled fast path");
    assert!(
        calls > 0,
        "the allocator/GC are opaque call-outs on the fast path"
    );
    assert!(
        f.blocks.len() <= 200,
        "the loop rolled (got {} blocks)",
        f.blocks.len()
    );
    println!("  allocator/GC kept as {calls} opaque call-out(s); loop rolled, dispatch folded");

    // End-to-end execution. The cut allocator/GC read the live heap (`global_State`, the arena), so we
    // seed the residual's memory with the captured window snapshot (a writable data segment), then run
    // the rolled entry: each iteration makes a *real* opaque allocation via the carried arena allocator
    // and the dispatch/arithmetic fold around it. The loop `x = x + 1` runs counter0+1 times, so the
    // written-back accumulator is x0 + (counter0 + 1). Correct on all three backends proves the rolled
    // fast path and the opaque allocator/GC call-outs compose correctly.
    let x_addr = base + STACKVALUE_SIZE;
    let counter0 = rd_u64(&w, counter) as i64;
    // The captured run is `for i=1,50 do local t={} x=x+1 end`, resumed at the loop body with
    // counter0 = 49 (50 body executions), so x = 0 + 50 = 50 — the real Lua answer for this script.
    assert_eq!(
        run_with_heap(&r, entry, &w, x_addr, &[0, 1, counter0]),
        50,
        "captured run must reproduce x = 50"
    );
    // The JIT compiles the whole carried runtime (~840 funcs), so exercise it once, on the captured
    // inputs; the cheaper tree-walk / bytecode backends carry the dynamic-trip-count sweep.
    assert_eq!(
        jit_with_heap(&r, entry, &w, x_addr, &[0, 1, counter0]),
        50,
        "svm-jit @ captured"
    );
    for (x0, i0, c0) in [(0i64, 1i64, counter0), (0, 1, 9), (7, 1, 25), (3, 1, 100)] {
        let want = x0 + (c0 + 1);
        let a = [x0, i0, c0];
        assert_eq!(
            run_with_heap(&r, entry, &w, x_addr, &a),
            want,
            "tree-walk @ (x0={x0}, c0={c0})"
        );
        assert_eq!(
            bc_with_heap(&r, entry, &w, x_addr, &a),
            want,
            "bytecode @ (x0={x0}, c0={c0})"
        );
    }
    println!("  executed: x=50 @ captured (all 3 backends); x0+(c0+1) across a trip sweep — real opaque allocations");
}

fn has_br_table_f(f: &svm_ir::Func) -> bool {
    f.blocks
        .iter()
        .any(|b| matches!(b.term, Terminator::BrTable { .. }))
}

/// Wrap the residual so its result is observable: seed memory with the captured heap `w` (a writable
/// data segment), append an entry that calls the rolled loop `entry(x0,i0,c0)` then loads the
/// accumulator cell it wrote back, and return the wrapped module + wrapper index.
fn wrap_with_heap(residual: &Module, entry: u32, w: &[u8], x_addr: u64) -> (Module, u32) {
    let mut m = residual.clone();
    m.data.push(Data {
        offset: 0,
        readonly: false,
        bytes: w.to_vec(),
    });
    let wrapper = m.funcs.len() as u32;
    m.funcs.push(Func {
        params: vec![ValType::I64, ValType::I64, ValType::I64],
        results: vec![ValType::I64],
        blocks: vec![Block {
            params: vec![ValType::I64, ValType::I64, ValType::I64],
            insts: vec![
                Inst::ConstI64(x_addr as i64), // v3
                Inst::Call {
                    func: entry,
                    args: vec![0, 1, 2],
                }, // void
                Inst::Load {
                    op: LoadOp::I64,
                    addr: 3,
                    offset: 0,
                }, // v4
            ],
            term: Terminator::Return(vec![4]),
        }],
    });
    (m, wrapper)
}
fn run_with_heap(residual: &Module, entry: u32, w: &[u8], x_addr: u64, a: &[i64]) -> i64 {
    let (m, e) = wrap_with_heap(residual, entry, w, x_addr);
    let mut fuel = u64::MAX;
    let vs: Vec<Value> = a.iter().map(|&x| Value::I64(x)).collect();
    match svm_interp::run(&m, e, &vs, &mut fuel)
        .expect("no trap")
        .as_slice()
    {
        [Value::I64(x)] => *x,
        o => panic!("bad {o:?}"),
    }
}
fn bc_with_heap(residual: &Module, entry: u32, w: &[u8], x_addr: u64, a: &[i64]) -> i64 {
    let (m, e) = wrap_with_heap(residual, entry, w, x_addr);
    let mut fuel = u64::MAX;
    let vs: Vec<Value> = a.iter().map(|&x| Value::I64(x)).collect();
    match svm_interp::bytecode::compile_and_run(&m, e, &vs, &mut fuel)
        .expect("subset")
        .expect("no trap")
        .as_slice()
    {
        [Value::I64(x)] => *x,
        o => panic!("bad {o:?}"),
    }
}
fn jit_with_heap(residual: &Module, entry: u32, w: &[u8], x_addr: u64, a: &[i64]) -> i64 {
    let (m, e) = wrap_with_heap(residual, entry, w, x_addr);
    match svm_jit::compile_and_run(&m, e, a) {
        Ok(svm_jit::JitOutcome::Returned(v)) => v[0],
        o => panic!("jit {o:?}"),
    }
}
