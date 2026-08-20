//! **The guarded stitch: closure-identity check with interpreter fallback.**
//!
//! `lua_futamura_call_stitch` transfers to the continuation unconditionally — valid only because the
//! captured run is deterministic. A *deployable* stitch must guard: at the boundary, check that the
//! callee closure in the frame's func slot is the one the continuation was specialized for, and fall
//! back to the **baseline interpreter** (resumed from the same spilled window state) on mismatch.
//!
//! The guard needs no engine surface either: residual B's stitch handler is a hand-built `() -> ()`
//! function — exactly what a stitch compiler would emit — that loads the closure pointer from the
//! spilled window, compares it against the expected constant, and tail-calls the continuation on a
//! match or the pristine `luaV_execute` copy (with the materialized `(sp, L, ci)`) on a mismatch.
//!
//! Both paths are proven to actually run by **poisoning the arm that must not be taken** (an
//! empty function that silently drops execution, so a misroute produces no output and fails the
//! stdout diff):
//!
//! - **match variant**: expected = the real closure; fallback arm poisoned. Prints `50` only if the
//!   guard takes the continuation.
//! - **mismatch variant**: expected = a bogus pointer; continuation arm poisoned; fallback = the real
//!   baseline copy. Prints `50` only if the guard falls back to the interpreter — which resumes from
//!   the spilled state at the callee frame and completes the whole program.
//!
//! Run: `cargo test -p svm-llvm --test lua_futamura_call_stitch_guard -- --nocapture`

use svm_interp::{IrPc, Stop, StopReason, Value};
use svm_ir::{Block, CmpOp, Func, Inst, IntTy, LoadOp, Module, Terminator, ValType};
use svm_peval::{specialize_with_config, LuaSite, PoscallModel, PrecallModel, SpecArg, SpecConfig};

const SCRIPT: &str = "local function add(a, b) return a + b end\nprint(add(2, 3) * 10)\n";
const CAP: usize = 8 << 20;

const L_CI: u64 = 32;
const L_STACK_LAST: u64 = 40;
const L_STACK: u64 = 48;
const LUA_STATE_SIZE: usize = 200;
// The real CallInfo allocation stride (see I71 facet b).
const CI_SIZE: usize = 64;
const CI_FUNC: u64 = 0;
const LCLOSURE_P: u64 = 24;
const PROTO_SIZEK: u64 = 20;
const PROTO_SIZECODE: u64 = 24;
const PROTO_SIZEP: u64 = 32;
const PROTO_K: u64 = 56;
const PROTO_CODE: u64 = 64;
const PROTO_P: u64 = 72;
const VNUMINT: u8 = 0x03;

fn lua_module() -> Module {
    let p = format!(
        "{}/tests/fixtures/lua/lua_eval.ll",
        env!("CARGO_MANIFEST_DIR")
    );
    svm_llvm::translate_ll_path(&p).expect("translate").module
}
fn byname(m: &Module, n: &str) -> Option<u32> {
    m.exports.iter().find(|e| e.name == n).map(|e| e.func)
}
fn rd_u64(w: &[u8], a: u64) -> u64 {
    u64::from_le_bytes(w[a as usize..a as usize + 8].try_into().unwrap())
}
fn rd_i32(w: &[u8], a: u64) -> i32 {
    i32::from_le_bytes(w[a as usize..a as usize + 4].try_into().unwrap())
}
fn dispatch_block(m: &Module, luav: u32) -> u32 {
    let mut best = (0u32, 0usize);
    for (bi, b) in m.funcs[luav as usize].blocks.iter().enumerate() {
        if let Terminator::BrTable { targets, .. } = &b.term {
            if targets.len() > best.1 {
                best = (bi as u32, targets.len());
            }
        }
    }
    best.0
}
/// The precall branch block, its `ra` SSA index, and the **startfunc arm** — the branch target taken
/// for a Lua callee (precall returned non-NULL), i.e. the frame-entry block the stitch deopts at.
fn precall_site(m: &Module, luav: u32, precall: u32) -> (u32, u32, u32) {
    for (bi, b) in m.funcs[luav as usize].blocks.iter().enumerate() {
        let ra = b.insts.iter().find_map(|i| match i {
            Inst::Call { func, args } if *func == precall => Some(args[2]),
            _ => None,
        });
        if let Some(ra) = ra {
            if let Terminator::BrIf {
                then_blk, else_blk, ..
            } = b.term
            {
                // The startfunc block re-derives the frame then falls into the shared setup chain; it
                // is the arm whose block just forwards (a plain `Br`).
                let is_startfunc = |t: u32| {
                    matches!(
                        m.funcs[luav as usize].blocks[t as usize].term,
                        Terminator::Br { .. }
                    ) && m.funcs[luav as usize].blocks[t as usize].insts.is_empty()
                };
                let sf = if is_startfunc(then_blk) {
                    then_blk
                } else if is_startfunc(else_blk) {
                    else_blk
                } else {
                    panic!("neither precall arm looks like startfunc");
                };
                return (bi as u32, ra, sf);
            }
        }
    }
    panic!("no precall branch site");
}
fn entry_capture(insp: &mut svm_interp::Inspector, luav: u32) -> (i64, u64, u64) {
    let pc = IrPc {
        module: 0,
        func: luav,
        block: 0,
        inst: 0,
    };
    insp.set_breakpoint(pc);
    let r = match insp.run_until_stop() {
        Stop::Break { .. } => {
            let g = |i| match insp.read_ir_value(0, i) {
                Some(Value::I64(v)) => v,
                o => panic!("v{i}: {o:?}"),
            };
            (g(0), g(1) as u64, g(2) as u64)
        }
        o => panic!("no entry break: {o:?}"),
    };
    insp.clear_breakpoint(pc);
    r
}

fn cut_lists(m: &Module) -> (Vec<u32>, Vec<u32>) {
    let read: Vec<u32> = [
        "luaH_get",
        "luaH_getshortstr",
        "luaH_getstr",
        "luaH_getint",
        "luaV_finishget",
        "luaC_step",
        "luaC_newobj",
        "luaH_new",
        "luaH_resize",
        "luaF_newLclosure",
        "luaF_newCclosure",
        "luaC_barrier_",
        "luaC_barrierback_",
        "luaM_realloc_",
        "luaM_saferealloc_",
        "luaM_malloc_",
        "luaM_growaux_",
        "luaS_resize",
        "luaS_newlstr",
        "luaC_fullgc",
        "luaF_findupval",
        "luaF_closeupval",
        "luaF_close",
        "luaF_initupvals",
    ]
    .iter()
    .filter_map(|n| byname(m, n))
    .collect();
    let touch: Vec<u32> = [
        "luaD_precall",
        "precallC",
        "luaD_call",
        "luaD_callnoyield",
        "luaD_poscall",
        "luaD_growstack",
        "luaD_reallocstack",
    ]
    .iter()
    .filter_map(|n| byname(m, n))
    .collect();
    (read, touch)
}

#[test]
fn guarded_stitch_takes_the_right_arm() {
    let m = lua_module();
    let luav = byname(&m, "luaV_execute").expect("luaV_execute");
    let precall = byname(&m, "luaD_precall").expect("precall");
    let poscall = byname(&m, "luaD_poscall").expect("poscall");
    let dispatch = dispatch_block(&m, luav);
    let (pblock, ra_val, startfunc) = precall_site(&m, luav, precall);
    println!("precall block b{pblock}, startfunc arm b{startfunc}");

    // ---- Capture: entry state, the two call-site ras, and add's frame at its first dispatch. ----
    let inst = svm_run::instantiate(m.clone()).expect("inst");
    let mut insp = inst.debug_attach(SCRIPT.as_bytes().to_vec(), u64::MAX);
    let (sp, l, main_ci) = entry_capture(&mut insp, luav);
    let w = insp.read_window(0, CAP).expect("w");
    let stack_lo = rd_u64(&w, l + L_STACK);
    let stack_hi = rd_u64(&w, l + L_STACK_LAST);
    let stack_len = (stack_hi - stack_lo) as usize;
    let main_func = rd_u64(&w, main_ci + CI_FUNC);
    let main_cl = rd_u64(&w, main_func);
    let main_proto = rd_u64(&w, main_cl + LCLOSURE_P);
    let main_code = rd_u64(&w, main_proto + PROTO_CODE);
    let main_sizecode = rd_i32(&w, main_proto + PROTO_SIZECODE) as usize;
    let main_pp = rd_u64(&w, main_proto + PROTO_P);
    let main_sizep = rd_i32(&w, main_proto + PROTO_SIZEP) as usize;
    let main_k = rd_u64(&w, main_proto + PROTO_K);
    let main_sizek = rd_i32(&w, main_proto + PROTO_SIZEK) as usize;

    let pbp = IrPc {
        module: 0,
        func: luav,
        block: pblock as usize,
        inst: 0,
    };
    insp.set_breakpoint(pbp);
    let mut ras = Vec::new();
    while ras.len() < 2 {
        match insp.run_until_stop() {
            Stop::Break {
                reason: StopReason::Breakpoint,
                ..
            } => match insp.read_ir_value(0, ra_val as usize) {
                Some(Value::I64(v)) => ras.push(v as u64),
                o => panic!("ra: {o:?}"),
            },
            o => panic!("expected precall break: {o:?}"),
        }
    }
    let (ra_add, ra_print) = (ras[0], ras[1]);

    // add's frame + the window at its first dispatch — the exact state residual A resumes from.
    let (cci, wb) = {
        let inst2 = svm_run::instantiate(m.clone()).expect("inst2");
        let mut in2 = inst2.debug_attach(SCRIPT.as_bytes().to_vec(), u64::MAX);
        let (_s, l2, _c) = entry_capture(&mut in2, luav);
        let dbp = IrPc {
            module: 0,
            func: luav,
            block: dispatch as usize,
            inst: 0,
        };
        in2.set_breakpoint(dbp);
        loop {
            match in2.run_until_stop() {
                Stop::Break {
                    reason: StopReason::Breakpoint,
                    ..
                } => {
                    let wv = in2.read_window(0, CAP).expect("w");
                    let ci = rd_u64(&wv, l2 + L_CI);
                    let f = rd_u64(&wv, ci + CI_FUNC);
                    let cl = rd_u64(&wv, f);
                    let p = rd_u64(&wv, cl + LCLOSURE_P);
                    if p != main_proto {
                        break (ci, wv);
                    }
                }
                o => panic!("callee frame not reached: {o:?}"),
            }
        }
    };
    let cf = rd_u64(&wb, cci + CI_FUNC);
    let ccl = rd_u64(&wb, cf);
    let cp = rd_u64(&wb, ccl + LCLOSURE_P);
    let cc = rd_u64(&wb, cp + PROTO_CODE);
    let csc = rd_i32(&wb, cp + PROTO_SIZECODE) as usize;
    let ck = rd_u64(&wb, cp + PROTO_K);
    let csk = rd_i32(&wb, cp + PROTO_SIZEK) as usize;
    assert_eq!(cf, ra_add, "callee ci->func == the add call ra");
    let ci_previous_off = (0..CI_SIZE as u64)
        .step_by(8)
        .find(|&off| rd_u64(&wb, cci + off) == main_ci)
        .expect("ci->previous");

    let slice = |src: &[u8], a: u64, n: usize| src[a as usize..a as usize + n].to_vec();
    let (read, touch) = cut_lists(&m);

    // ---- Pass A: the continuation residual, rooted at add's dispatch (the reroot shape). Mutable
    // state (L, stack, main's ci with its advanced savedpc) from add's dispatch-moment window. ----
    let mut overlays_a = vec![
        (l, slice(&wb, l, LUA_STATE_SIZE)),
        (stack_lo, slice(&wb, stack_lo, stack_len)),
        (main_ci, slice(&wb, main_ci, CI_SIZE)),
        (main_code, slice(&w, main_code, 4 * main_sizecode)),
        (main_cl, slice(&w, main_cl, 48)),
        (main_proto, slice(&w, main_proto, 128)),
        (main_pp, slice(&w, main_pp, 8 * main_sizep)),
        (cci, slice(&wb, cci, CI_SIZE)),
        (ccl, slice(&wb, ccl, 48)),
        (cp, slice(&wb, cp, 128)),
        (cc, slice(&wb, cc, 4 * csc)),
    ];
    if main_sizek > 0 {
        overlays_a.push((main_k, slice(&w, main_k, 16 * main_sizek)));
    }
    if csk > 0 && ck != main_k {
        overlays_a.push((ck, slice(&wb, ck, 16 * csk)));
    }
    let cfg_a = SpecConfig {
        const_overlays: overlays_a,
        rename: Some((l, l + LUA_STATE_SIZE as u64)),
        rename_extra: vec![
            (stack_lo, stack_hi),
            (main_ci, main_ci + CI_SIZE as u64),
            (cci, cci + CI_SIZE as u64),
        ],
        rename_is_private: true,
        rename_seed_from_image: true,
        cut_calls_read_state: read.clone(),
        cut_calls_touch_state: touch.clone(),
        carry_whole_module: true,
        carry_keep_imports: true,
        indirect_targets_cap: Some(16),
        precall_model: Some(PrecallModel {
            precall,
            ra_arg: 2,
            l_ci_addr: l + L_CI,
            lua_sites: vec![],
            c_sites: vec![ra_print],
            poscall: Some(PoscallModel {
                poscall,
                ci_previous_off,
                ci_func_off: CI_FUNC,
                tag_off: 8,
                selective: vec![(cci, VNUMINT)],
            }),
        }),
        ..SpecConfig::default()
    };
    let args_a = [
        SpecArg::ConstI64(sp),
        SpecArg::ConstI64(l as i64),
        SpecArg::ConstI64(cci as i64),
    ];
    let r_a = specialize_with_config(&m, luav, &args_a, &cfg_a)
        .expect("residual A: add-rooted continuation");
    let entry_a = r_a.funcs.last().expect("entry A").clone();
    assert!(
        entry_a.params.is_empty() && entry_a.results.is_empty(),
        "the continuation resumes purely from the window: () -> ()"
    );
    println!("residual A (continuation): {} blocks", entry_a.blocks.len());

    // ---- Pass B: the caller residual, on a module padded with a `() -> ()` placeholder handler at
    // index N. The identity carry keeps it at N; residual A's entry replaces it afterwards. ----
    let handler_idx = m.funcs.len() as u32; // N: the guard
    let cont_idx = handler_idx + 1; // N+1: the continuation (or poison)
    let base_idx = handler_idx + 2; // N+2: the baseline fallback (or poison)
    let mut m_padded = m.clone();
    for _ in 0..3 {
        m_padded.funcs.push(Func {
            params: vec![],
            results: vec![],
            blocks: vec![Block {
                params: vec![],
                insts: vec![],
                term: Terminator::Return(vec![]),
            }],
        });
    }
    let mut overlays_b = vec![
        (l, slice(&w, l, LUA_STATE_SIZE)),
        (stack_lo, slice(&w, stack_lo, stack_len)),
        (main_ci, slice(&w, main_ci, CI_SIZE)),
        (main_code, slice(&w, main_code, 4 * main_sizecode)),
        (main_cl, slice(&w, main_cl, 48)),
        (main_proto, slice(&w, main_proto, 128)),
        (main_pp, slice(&w, main_pp, 8 * main_sizep)),
        // add's proto: main's `OP_CLOSURE` reads `proto->p[bx]` and folds its `sizeupvalues` to bound
        // the upvalue-init loop — needed even though this residual never enters add's body.
        (cp, slice(&wb, cp, 128)),
    ];
    if main_sizek > 0 {
        overlays_b.push((main_k, slice(&w, main_k, 16 * main_sizek)));
    }
    let cfg_b = SpecConfig {
        const_overlays: overlays_b,
        rename: Some((l, l + LUA_STATE_SIZE as u64)),
        rename_extra: vec![(stack_lo, stack_hi), (main_ci, main_ci + CI_SIZE as u64)],
        rename_is_private: true,
        rename_seed_from_image: true,
        cut_calls_read_state: read,
        cut_calls_touch_state: touch,
        carry_whole_module: true,
        carry_keep_imports: true,
        indirect_targets_cap: Some(16),
        // The stitch: entering the callee's frame bails to the continuation instead of folding on.
        deopt_targets: vec![(luav, startfunc)],
        deopt_handler: Some(handler_idx),
        precall_model: Some(PrecallModel {
            precall,
            ra_arg: 2,
            l_ci_addr: l + L_CI,
            lua_sites: vec![LuaSite {
                ra: ra_add,
                callee_ci: cci,
                pins: vec![(cf, ccl)],
            }],
            c_sites: vec![ra_print],
            poscall: None,
        }),
        ..SpecConfig::default()
    };
    let args_b = [
        SpecArg::ConstI64(sp),
        SpecArg::ConstI64(l as i64),
        SpecArg::ConstI64(main_ci as i64),
    ];
    let r = specialize_with_config(&m_padded, luav, &args_b, &cfg_b)
        .expect("residual B: entry-rooted caller");
    let entry_b = (r.funcs.len() - 1) as u32;
    let fb = &r.funcs[entry_b as usize];
    let stitches = fb
        .blocks
        .iter()
        .filter(|b| matches!(b.term, Terminator::ReturnCall { func, .. } if func == handler_idx))
        .count();
    println!(
        "residual B (caller): {} blocks, {} stitch tail-call(s) to the continuation",
        fb.blocks.len(),
        stitches
    );
    assert!(
        stitches >= 1,
        "the call boundary must tail-call the continuation residual"
    );

    // ---- The guard: a hand-built `() -> ()` handler. Loads the callee closure pointer from the
    // spilled window's func slot, compares against `expected`, and tail-calls the continuation on a
    // match or the baseline interpreter (materialized `(sp, L, ci)`) on a mismatch. ----
    let guard = |expected: u64| Func {
        params: vec![],
        results: vec![],
        blocks: vec![
            Block {
                params: vec![],
                insts: vec![
                    Inst::ConstI64(cf as i64),
                    Inst::Load {
                        op: LoadOp::I64,
                        addr: 0,
                        offset: 0,
                    },
                    Inst::ConstI64(expected as i64),
                    Inst::IntCmp {
                        ty: IntTy::I64,
                        op: CmpOp::Eq,
                        a: 1,
                        b: 2,
                    },
                ],
                term: Terminator::BrIf {
                    cond: 3,
                    then_blk: 1,
                    then_args: vec![],
                    else_blk: 2,
                    else_args: vec![],
                },
            },
            Block {
                params: vec![],
                insts: vec![],
                term: Terminator::ReturnCall {
                    func: cont_idx,
                    args: vec![],
                },
            },
            Block {
                params: vec![],
                insts: vec![
                    Inst::ConstI64(sp),
                    Inst::ConstI64(l as i64),
                    Inst::ConstI64(cci as i64),
                ],
                term: Terminator::ReturnCall {
                    func: base_idx,
                    args: vec![0, 1, 2],
                },
            },
        ],
    };
    // Poison: silently drops execution, so a misrouted guard produces no output and fails the diff.
    let poison = |params: Vec<ValType>| Func {
        params: params.clone(),
        results: vec![],
        blocks: vec![Block {
            params,
            insts: vec![],
            term: Terminator::Return(vec![]),
        }],
    };
    let embed = Func {
        params: vec![ValType::I64, ValType::I64, ValType::I64],
        results: vec![],
        blocks: vec![Block {
            params: vec![ValType::I64, ValType::I64, ValType::I64],
            insts: vec![Inst::Call {
                func: entry_b,
                args: vec![],
            }],
            term: Terminator::Return(vec![]),
        }],
    };
    let base = svm_run::run_powerbox(&m, SCRIPT.as_bytes()).expect("baseline");
    assert_eq!(base.stdout, b"50\n", "baseline prints 50");

    // Variant 1 — MATCH: real expected closure, real continuation, poisoned fallback.
    let mut r1 = r.clone();
    r1.funcs[handler_idx as usize] = guard(ccl);
    r1.funcs[cont_idx as usize] = entry_a.clone();
    r1.funcs[base_idx as usize] = poison(vec![ValType::I64, ValType::I64, ValType::I64]);
    r1.imports = m.imports.clone();
    r1.types = m.types.clone();
    r1.funcs[luav as usize] = embed.clone();
    svm_verify::verify_module(&r1).expect("match variant verifies");
    let out1 = svm_run::run_powerbox(&r1, SCRIPT.as_bytes()).expect("match variant");
    println!(
        "match variant (guard→continuation): {:?}",
        String::from_utf8_lossy(&out1.stdout)
    );
    assert_eq!(
        out1.stdout, base.stdout,
        "guard must take the continuation (the fallback is poisoned)"
    );

    // Variant 2 — MISMATCH: bogus expected closure, poisoned continuation, REAL baseline fallback
    // (a pristine copy of luaV_execute — the embed replaces the original index).
    let mut r2 = r.clone();
    r2.funcs[handler_idx as usize] = guard(ccl ^ 0x40);
    r2.funcs[cont_idx as usize] = poison(vec![]);
    r2.funcs[base_idx as usize] = m.funcs[luav as usize].clone();
    r2.imports = m.imports.clone();
    r2.types = m.types.clone();
    r2.funcs[luav as usize] = embed;
    svm_verify::verify_module(&r2).expect("mismatch variant verifies");
    let out2 = svm_run::run_powerbox(&r2, SCRIPT.as_bytes()).expect("mismatch variant");
    println!(
        "mismatch variant (guard→baseline): {:?}",
        String::from_utf8_lossy(&out2.stdout)
    );
    assert_eq!(
        out2.stdout, base.stdout,
        "guard must fall back to the interpreter (the continuation is poisoned)"
    );
    println!("  ✓ guarded stitch: match → continuation, mismatch → interpreter fallback, both 50");
}
