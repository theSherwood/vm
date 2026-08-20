//! **Zero-config auto-rolled Lua residual — the driver-level `auto_rolled`.**
//!
//! `lua_futamura_auto_rolled.rs` first showed the rolled residual could be built with no hardcoded
//! offsets; this promotes that to a reusable driver function on top of the shared
//! [`peval_capture::discover`], so a caller hands it a Lua chunk and gets back the rolled residual +
//! the metadata to run/verify it (dynamic cells, the trip counter, the accumulator). Both the focused
//! correctness test and the end-to-end benchmark call this — one code path, script in → residual out.
//!
//! Scope: a single top-level numeric-`for` accumulator chunk (`local acc = …; for i = a,b do acc = … end`).
//! The result cell is identified generically by decoding the chunk's `RETURN`/`RETURN1` bytecode (its
//! `A` operand is the returned register), so multi-accumulator chunks like fibonacci work too. Integer
//! `%` / `//` are in scope in both forms: a **register** divisor (`local d`) deopts its divide-by-zero
//! cold arm to the carried baseline, and a **constant** divisor (`i % 2`, an `OP_MODK`/`OP_IDIVK`) folds
//! outright — freezing the proto's constants pool proves the divisor non-zero, so the error arm is dead
//! (see `auto_rolled`).

#![allow(dead_code)]

use super::peval_capture::{self, Located, TargetDesc};
use svm_interp::{IrPc, Stop, StopReason, Value};
use svm_ir::{Block, Func, Inst, LoadOp, Module, Terminator, ValType};
use svm_peval::{specialize_with_config, SpecArg, SpecConfig};

const CAPTURE_LEN: usize = 8 << 20;
// Lua 5.4.7 offsets (call-free loop ⇒ the 104-byte ci over-slice is safe: no adjacent-frame collision).
const L_STACK_LAST: u64 = 40;
const L_STACK: u64 = 48;
const LUA_STATE_SIZE: usize = 200;
const CI_SIZE: usize = 104;
const CI_FUNC: u64 = 0;
const CI_SAVEDPC: u64 = 32;
const LCLOSURE_P: u64 = 24;
const PROTO_SIZECODE: u64 = 24;
const PROTO_CODE: u64 = 64;
const PROTO_SIZEK: u64 = 20;
const PROTO_K: u64 = 56;
const STACKVALUE_SIZE: u64 = 16;
const VNUMINT_TAG: u8 = 0x03;

fn rd_u64(w: &[u8], a: u64) -> u64 {
    u64::from_le_bytes(w[a as usize..a as usize + 8].try_into().unwrap())
}
fn rd_i32(w: &[u8], a: u64) -> i32 {
    i32::from_le_bytes(w[a as usize..a as usize + 4].try_into().unwrap())
}

pub fn lua_module() -> Module {
    let p = format!(
        "{}/tests/fixtures/lua/lua_eval.ll",
        env!("CARGO_MANIFEST_DIR")
    );
    svm_llvm::translate_ll_path(&p).expect("translate").module
}
pub fn luav_execute(m: &Module) -> u32 {
    m.exports
        .iter()
        .find(|e| e.name == "luaV_execute")
        .expect("export")
        .func
}

fn read_entry(insp: &mut svm_interp::Inspector, luav: u32) -> (i64, u64, u64) {
    let entry = IrPc {
        module: 0,
        func: luav,
        block: 0,
        inst: 0,
    };
    insp.set_breakpoint(entry);
    let r = match insp.run_until_stop() {
        Stop::Break {
            reason: StopReason::Breakpoint,
            ..
        } => {
            let g = |i| match insp.read_ir_value(0, i) {
                Some(Value::I64(v)) => v,
                o => panic!("entry v: {o:?}"),
            };
            (g(0), g(1) as u64, g(2) as u64)
        }
        o => panic!("no entry break: {o:?}"),
    };
    insp.clear_breakpoint(entry);
    r
}

/// A zero-config rolled residual + everything needed to run and verify it.
pub struct AutoRolled {
    /// The rolled residual; params are the dynamic cells (ascending).
    pub residual: Module,
    /// Entry function of the residual. `0` on the fast path; on the whole-module-carry deopt fallback
    /// (used to deopt the div/mod cold arm for `%` / `//`) the entry lands at the last function, so
    /// callers must dispatch through this rather than assuming func 0.
    pub entry: u32,
    /// Dynamic-cell addresses (ascending) — the residual's parameters, positionally.
    pub dyn_cells: Vec<u64>,
    /// Index (in `dyn_cells` / residual params) of the trip counter — sweep this to vary the loop.
    pub counter_ix: usize,
    /// Index of the accumulator cell (the residual writes it back; read it for the result).
    pub acc_ix: usize,
    /// Address of the accumulator cell.
    pub acc_addr: u64,
    /// Captured seed value of each dynamic cell (positional) — reproduces the original run.
    pub captured: Vec<i64>,
    /// Residual and baseline `luaV_execute` block counts (for reporting).
    pub residual_blocks: usize,
    pub base_blocks: usize,
    /// Register base (`ci->func + 1 TValue`) — so callers can report cells as `R[n]`.
    pub reg_base: u64,
    /// Register stride in bytes.
    pub reg_stride: u64,
}

/// Build the rolled residual for a single numeric-`for` accumulator chunk, fully automatically.
pub fn auto_rolled(m: &Module, script: &str) -> AutoRolled {
    let luav = luav_execute(m);
    let dispatch = peval_capture::dispatch_block(m, luav);

    // Locate the register base + entry scalars.
    let inst = svm_run::instantiate(m.clone()).expect("instantiate");
    let mut insp0 = inst.debug_attach(script.as_bytes().to_vec(), u64::MAX);
    let (sp, l, ci) = read_entry(&mut insp0, luav);
    let w0 = insp0.read_window(0, CAPTURE_LEN).expect("window");
    drop(insp0);
    let base = rd_u64(&w0, ci + CI_FUNC) + STACKVALUE_SIZE;

    let loc = Located {
        reg_base: base,
        pc_addr: Some(ci + CI_SAVEDPC),
    };
    let desc = TargetDesc {
        reg_stride: STACKVALUE_SIZE,
        n_regs: 32,
        tag: Some((8, VNUMINT_TAG)),
        capture_len: CAPTURE_LEN,
        observe_hits: 48,
    };
    let script_owned = script.to_string();
    let make_insp = move || {
        let inst = svm_run::instantiate(m.clone()).expect("instantiate");
        let mut insp = inst.debug_attach(script_owned.as_bytes().to_vec(), u64::MAX);
        read_entry(&mut insp, luav);
        insp
    };
    let mut d = peval_capture::discover(&make_insp, luav, dispatch, &loc, &desc);
    assert!(d.counter != 0, "no trip counter discovered");
    assert!(
        d.varying.contains(&d.counter),
        "counter must be a dynamic cell"
    );

    let w = &d.window;
    let stack_lo = rd_u64(w, l + L_STACK);
    let stack_hi = rd_u64(w, l + L_STACK_LAST);
    let stack_len = (stack_hi - stack_lo) as usize;
    let func = rd_u64(w, ci + CI_FUNC);
    let closure = rd_u64(w, func);
    let proto = rd_u64(w, closure + LCLOSURE_P);
    let code = rd_u64(w, proto + PROTO_CODE);
    let sizecode = rd_i32(w, proto + PROTO_SIZECODE) as usize;
    let slice = |a: u64, n: usize| w[a as usize..a as usize + n].to_vec();

    // **Freeze the constants pool.** A constant-operand op like `i % 2` compiles to `OP_MODK`, whose
    // divisor is `K[C]` — a load from the proto's constants array (`Proto.k`), a separate heap allocation
    // the `proto` overlay doesn't cover. Un-frozen, the divisor stays `Dyn`, so the specializer can't
    // prove it non-zero and must project the divide-by-zero / metamethod machinery (which reaches the GC
    // and the allocator) — `Unsupported`. Freezing the (read-only) `K` array folds `K[C]` to the literal,
    // so `i % 2` folds with the error arm dead — no divisor cell, no deopt. `Proto.k` @+56, `sizek` @+20;
    // each constant is one 16-byte `TValue`.
    let kptr = rd_u64(w, proto + PROTO_K);
    let sizek = rd_i32(w, proto + PROTO_SIZEK) as usize;
    let k_overlay: Vec<(u64, Vec<u8>)> = if kptr != 0 && sizek > 0 {
        vec![(kptr, slice(kptr, sizek * STACKVALUE_SIZE as usize))]
    } else {
        Vec::new()
    };

    // **Freeze read-only tables.** A `local t = {…}` read in the loop (`t[i]`) needs the table object in
    // constant memory for the read to fold: the struct (so `array`, `alimit`, and the null `metatable`
    // fold), the array part (a constant-index read folds to an immediate; a dynamic-index read becomes a
    // masked load from it), and the node/hash part (so the slow-path `luaH_get` folds instead of exploring
    // the rehash machinery). Each frozen region goes in **both** `const_overlays` (spec-time folding) and a
    // readonly **data segment** (so a dynamic-index runtime load hits mapped, initialized memory — the
    // object lives on the heap above the register stack, outside every renamed region). Tables are found
    // generically by scanning the register frame for a collectable-table tag (`0x45`). Lua 5.4 `Table`:
    // `lsizenode` @+11, `alimit` @+12, `array` @+16, `node` @+24, `metatable` @+40, size 56; `Node` is 24
    // bytes. A table-free chunk freezes nothing, so this is inert on the numeric/control-flow paths.
    const TABLE_TAG: u8 = 0x45;
    let win_len = w.len() as u64;
    let mut table_overlay: Vec<(u64, Vec<u8>)> = Vec::new();
    let mut table_data: Vec<(u64, Vec<u8>)> = Vec::new();
    let mut seen: Vec<u64> = Vec::new();
    for reg in 0..32u64 {
        let addr = base + (reg + 1) * STACKVALUE_SIZE; // +1: VARARGPREP frame shift
        if addr + 16 > win_len || w[(addr + 8) as usize] != TABLE_TAG {
            continue;
        }
        let tptr = rd_u64(w, addr);
        if tptr == 0 || tptr + 56 > win_len || seen.contains(&tptr) {
            continue;
        }
        seen.push(tptr);
        let mut freeze = |lo: u64, len: usize| {
            table_overlay.push((lo, slice(lo, len)));
            table_data.push((lo, slice(lo, len)));
        };
        freeze(tptr, 56); // the Table struct
        let alimit = rd_i32(w, tptr + 12);
        let array = rd_u64(w, tptr + 16);
        if array != 0 && alimit > 0 && array + alimit as u64 * 16 <= win_len {
            freeze(array, alimit as usize * 16); // the array part
        }
        let node = rd_u64(w, tptr + 24);
        let nbytes = (1u64 << w[(tptr + 11) as usize]) * 24;
        if node != 0 && node + nbytes <= win_len && !seen.contains(&node) {
            seen.push(node);
            freeze(node, nbytes as usize); // the node / hash part (an array-only table: the shared dummy)
        }
    }

    // Integer `//` (OP_IDIV) and `%` (OP_MOD) are the only arithmetic ops that *raise* on a bad
    // operand — a zero divisor, or `INT_MIN // -1` overflow — and the register-register forms inline
    // that guard (and its `luaG_runerror` cold arm) directly in `luaV_execute`. The proven recipe
    // (`lua_futamura_deopt.rs`) is to make the **divisor a dynamic cell** so the `divisor == 0` guard is
    // a *dynamic* branch, then deopt its cold arm: the division folds on the hot path and the error case
    // deopts to the baseline. So decode every OP_IDIV / OP_MOD and mark its divisor register (operand C)
    // dynamic. A loop-invariant divisor (`local d = 2`) would otherwise sit in the renamed mutable stack
    // and load as an untracked `Dyn`, leaving the guard un-deoptable; forcing it a declared dynamic cell
    // is what threads it to the deopt edge. Lua 5.4 iABC: op=bits0..6, A=7..14, B=16..23, C=24..31; the
    // K-forms (OP_IDIVK / OP_MODK) instead *call* `luaV_idiv` / `luaV_mod`, whose error is inside the
    // helper (not inline, so not deoptable here) — those stay a scope wall, so a constant divisor must be
    // written as a local to become a register operand. Straight-line chunks contain no such op: inert.
    const OP_MOD: u32 = 37;
    const OP_IDIV: u32 = 40;
    for pc in 0..sizecode {
        let word = u32::from_le_bytes(
            w[(code + pc as u64 * 4) as usize..(code + pc as u64 * 4) as usize + 4]
                .try_into()
                .unwrap(),
        );
        let op = word & 0x7F;
        if op == OP_MOD || op == OP_IDIV {
            // Both the divisor (operand C) and the result temp (operand A) are dynamic: C so the
            // `divisor == 0` guard is a dynamic, deoptable branch; A because the division result is
            // dynamic and must be a declared cell for the residual to store it and thread it across the
            // deopt edge (the baseline resumes from the register file). Matches the hand-picked cells in
            // `lua_futamura_deopt.rs` (the divisor and the IDIV temp).
            let a = ((word >> 7) & 0xFF) as u64; // result register
            let c = ((word >> 24) & 0xFF) as u64; // divisor register
            for reg in [a, c] {
                let addr = base + (reg + 1) * STACKVALUE_SIZE; // +1: VARARGPREP frame shift
                if !d.varying.contains(&addr) {
                    d.varying.push(addr);
                }
            }
        }
    }
    d.varying.sort_unstable(); // params are positional-ascending; keep the order stable after the union

    // **Resume off the metamethod trailer.** Every Lua 5.4 arithmetic op is followed by an `OP_MMBIN*`
    // (the metamethod fallback for non-number operands). The generic discovery safepoint often lands on
    // that trailer rather than on the arithmetic op itself — and resuming a residual *at* `OP_MMBIN`
    // forces the specializer to project the metamethod-miss path (`luaT_trybinTM` → allocate an error →
    // the throw machinery), which is un-foldable and hits `Unsupported`. The arithmetic op one
    // instruction earlier is a clean resume point (its fast integer path folds; `OP_MMBIN` is then a
    // known no-op reached with a constant "no metamethod" flag). Rewinding is state-consistent: the op's
    // result register is a declared dynamic cell, so the resumed op simply recomputes it before the next
    // instruction reads it, and no other register moved. Lua 5.4 metamethod trailers: MMBIN=46,
    // MMBINI=47, MMBINK=48. (Only div/mod's safepoint lands here; straight-line chunks resume on the op.)
    let mut ci_image = slice(ci, CI_SIZE);
    let savedpc = rd_u64(w, ci + CI_SAVEDPC);
    let pc_ix = (savedpc - code) / 4;
    if pc_ix > 0 {
        let word = u32::from_le_bytes(
            w[savedpc as usize..savedpc as usize + 4]
                .try_into()
                .unwrap(),
        );
        let op = word & 0x7F;
        if matches!(op, 46..=48) {
            let rewound = savedpc - 4;
            ci_image[CI_SAVEDPC as usize..CI_SAVEDPC as usize + 8]
                .copy_from_slice(&rewound.to_le_bytes());
        }
    }

    // The error/throw cold arms: an arithmetic op like `%` / `//` guards its operands (a `divisor == 0`,
    // an `INT_MIN // -1` overflow, a non-number metamethod miss) and, on the cold side, raises a real
    // Lua error via the setjmp/longjmp machinery (`luaG_runerror` → `luaD_throw`). That unwind path is
    // inherently un-foldable, so projecting into it hits `Unsupported`. The fix mirrors the tiered-JIT
    // shape: **fold the hot arithmetic and deopt the cold arm to the carried baseline `luaV_execute`**,
    // which resumes from the spilled `savedpc` and raises the real error. Find those cold arms
    // structurally — every `luaV_execute` block that calls a function which reaches setjmp/longjmp — and
    // mark them `deopt_targets`. A straight-line integer chunk enters none of them, so this is inert
    // there (the fast path below handles it with no carry at all).
    let callees = |f: &Func| -> Vec<u32> {
        f.blocks
            .iter()
            .flat_map(|b| {
                b.insts.iter().filter_map(|i| match i {
                    Inst::Call { func, .. } => Some(*func),
                    _ => None,
                })
            })
            .collect()
    };
    // An **error raiser** is a function that never returns to its caller — every exit is a `longjmp`
    // (or a tail-call to another error raiser). Lua's `luaG_runerror` / `luaG_opinterror` /
    // `luaD_throw` are exactly these; an allocator or GC step *reaches* the throw machinery on its OOM
    // arm but **returns normally** on the hot path, so it is not a raiser. That distinction is the
    // whole game: deopting a block because it calls a raiser targets a genuine error arm; deopting one
    // because it calls a merely-can-throw helper (an allocator on the FORLOOP's hot path) would deopt
    // the loop itself. So the raiser test is "reaches setjmp **and** has no normal return".
    let n = m.funcs.len();
    let mut reaches = vec![false; n];
    for (f, func) in m.funcs.iter().enumerate() {
        reaches[f] = func.uses_setjmp();
    }
    loop {
        let mut changed = false;
        for f in 0..n {
            if reaches[f] {
                continue;
            }
            if callees(&m.funcs[f]).iter().any(|&g| reaches[g as usize]) {
                reaches[f] = true;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let never_returns = |f: &Func| -> bool {
        !f.blocks
            .iter()
            .any(|b| matches!(b.term, Terminator::Return(_)))
    };
    let is_raiser: Vec<bool> = m
        .funcs
        .iter()
        .enumerate()
        .map(|(f, func)| reaches[f] && never_returns(func) && func.blocks.len() > 1)
        .collect();
    // **Cold-arm deopt targets.** Two families of `luaV_execute` cold arm reach the allocator / throw
    // machinery and can't be projected, so they deopt to the carried baseline (never taken at runtime for
    // the hot integer/table path):
    //   1. **error raisers** — an op like `%` / `//` guarding a zero divisor calls `luaG_runerror` (below);
    //   2. **metamethod / slow-arm dispatchers** — a `t[i]` read produces a *dynamic* slot pointer, and a
    //      value *loaded* from a table has a tag the specializer can't prove numeric, so every consuming op
    //      (`+`, `<`, `#`, …) and the table read itself explore their metamethod arm (`luaV_finishget` /
    //      `luaT_trybinTM` / …), which invokes a metamethod (→ `luaD_call` → the allocator) or rehashes.
    //      For integer data in a dense array those arms are never taken, so deopting the `luaV_execute`
    //      blocks that call one folds the hot read + arithmetic and sends the (impossible) slow case to
    //      baseline — the same guard-and-deopt shape as `%`/`//`, scoped to the metamethod dispatchers
    //      (not every allocator-reaching block — the GC checkpoint on the hot loop back-edge folds and must
    //      not be deopted). A numeric/control-flow chunk reaches none of these arms: inert there.
    let mm_helpers: Vec<u32> = [
        "luaV_finishget",
        "luaV_finishset",
        "luaT_trybinTM",
        "luaT_trybiniTM",
        "luaT_trybinassocTM",
        "luaT_callorderTM",
        "luaT_callorderiTM",
        "luaV_equalobj",
        "luaV_objlen",
        "luaV_concat",
    ]
    .iter()
    .filter_map(|nm| m.exports.iter().find(|e| e.name == *nm).map(|e| e.func))
    .collect();
    let deopt_blocks: Vec<u32> = m.funcs[luav as usize]
        .blocks
        .iter()
        .enumerate()
        .filter(|(_, b)| {
            b.insts.iter().any(|i| {
                matches!(i, Inst::Call { func, .. }
                    if *func != luav && (is_raiser[*func as usize] || mm_helpers.contains(func)))
            })
        })
        .map(|(bi, _)| bi as u32)
        .collect();
    let args = [
        SpecArg::ConstI64(sp),
        SpecArg::ConstI64(l as i64),
        SpecArg::ConstI64(ci as i64),
    ];
    let base_cfg = || SpecConfig {
        const_overlays: [
            vec![
                (l, slice(l, LUA_STATE_SIZE)),
                (stack_lo, slice(stack_lo, stack_len)),
                (ci, ci_image.clone()),
                (code, slice(code, 4 * sizecode)),
                (closure, slice(closure, 48)),
                (proto, slice(proto, 128)),
            ],
            table_overlay.clone(),
            k_overlay.clone(),
        ]
        .concat(),
        rename: Some((l, l + LUA_STATE_SIZE as u64)),
        rename_extra: vec![(stack_lo, stack_hi), (ci, ci + CI_SIZE as u64)],
        rename_is_private: true,
        rename_seed_from_image: true,
        dynamic_cells: d.varying.iter().map(|&a| (a, 8)).collect(),
        indirect_targets_cap: Some(16),
        ..SpecConfig::default()
    };
    // Fast path: a straight-line integer chunk projects to a small standalone residual (entry = func 0,
    // calls nothing). If that hits `Unsupported` (e.g. `%` / `//` projects into the un-foldable
    // setjmp/longjmp error path), fall back to deopting every cold error arm to the carried baseline
    // `luaV_execute` — the hot arithmetic still folds and the residual entry lands at the last function.
    let (mut residual, entry) = match specialize_with_config(m, luav, &args, &base_cfg()) {
        Ok(r) => (r, 0u32),
        Err(_) => {
            let cfg = SpecConfig {
                deopt_targets: deopt_blocks.iter().map(|&b| (luav, b)).collect(),
                deopt_handler: Some(luav),
                carry_whole_module: true,
                ..base_cfg()
            };
            let r = specialize_with_config(m, luav, &args, &cfg)
                .expect("auto-rolled projection (with cold-arm deopt)");
            let e = (r.funcs.len() - 1) as u32;
            (r, e)
        }
    };
    for (off, bytes) in &table_data {
        if !residual.data.iter().any(|d| d.offset == *off) {
            residual.data.push(svm_ir::Data {
                offset: *off,
                readonly: true,
                bytes: bytes.clone(),
            });
        }
    }
    svm_verify::verify_module(&residual).expect("residual verifies");

    let counter_ix = d.varying.iter().position(|&a| a == d.counter).unwrap();
    // The **result register**: the chunk's `return <v>` compiles to a `RETURN1 A` (or `RETURN A ...`)
    // bytecode whose `A` is the returned register. Decoding it (a single lookup in the proto code) is
    // the generic, dataflow-free way to know which cell holds the program's result — unlike the
    // "lowest carried cell" heuristic, this is correct for multi-accumulator chunks (fibonacci returns
    // `b`, not the lowest register). The residual writes the register file back on return even though
    // it stops before executing the `RETURN`, so reading that register gives the result.
    // Lua 5.4.7 opcodes: RETURN=70, RETURN0=71, RETURN1=72. `A` is bits 7..15, `B` bits 16..24. The
    // real `return <v>` is `RETURN1` (always one value) or a `RETURN` with `B != 1` (B-1 values; the
    // implicit end-of-chunk `RETURN A 1` returns nothing). `A` is in the chunk's own register
    // numbering; VARARGPREP shifts the runtime frame up by one, so the cell address is
    // `base + (A + 1)*stride`.
    let mut result_reg: Option<u64> = None;
    for pc in 0..sizecode {
        let word = u32::from_le_bytes(
            w[(code + pc as u64 * 4) as usize..(code + pc as u64 * 4) as usize + 4]
                .try_into()
                .unwrap(),
        );
        let op = word & 0x7F;
        let a = ((word >> 7) & 0xFF) as u64;
        let b = (word >> 16) & 0xFF;
        if op == 72 || (op == 70 && b != 1) {
            result_reg = Some(a);
            break;
        }
    }
    // Accumulator = the returned register (+1 VARARGPREP shift) if it is a discovered carried cell,
    // else the lowest carried non-counter cell.
    let acc_addr = result_reg
        .map(|a| base + (a + 1) * STACKVALUE_SIZE)
        .filter(|a| d.varying.contains(a))
        .unwrap_or_else(|| {
            *d.varying
                .iter()
                .find(|&&a| a != d.counter)
                .expect("a carried cell distinct from the counter")
        });
    let acc_ix = d.varying.iter().position(|&a| a == acc_addr).unwrap_or(0);
    let captured: Vec<i64> = d.varying.iter().map(|&a| rd_u64(w, a) as i64).collect();

    AutoRolled {
        residual_blocks: residual.funcs[entry as usize].blocks.len(),
        base_blocks: m.funcs[luav as usize].blocks.len(),
        residual,
        entry,
        dyn_cells: d.varying,
        counter_ix,
        acc_ix,
        acc_addr,
        captured,
        reg_base: base,
        reg_stride: STACKVALUE_SIZE,
    }
}

/// Append a `(dyn0, dyn1, …) -> i64` wrapper that calls the rolled residual (its `entry` function)
/// then loads the accumulator cell it wrote back.
pub fn with_readback(
    residual: &Module,
    entry: u32,
    read_addr: u64,
    nparams: usize,
) -> (Module, u32) {
    let mut m = residual.clone();
    let wrapper = m.funcs.len() as u32;
    let params: Vec<ValType> = vec![ValType::I64; nparams];
    let call_args: Vec<u32> = (0..nparams as u32).collect();
    let addr_v = nparams as u32;
    let load_v = nparams as u32 + 1;
    m.funcs.push(Func {
        params: params.clone(),
        results: vec![ValType::I64],
        blocks: vec![Block {
            params,
            insts: vec![
                Inst::ConstI64(read_addr as i64),
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
            term: Terminator::Return(vec![load_v]),
        }],
    });
    (m, wrapper)
}

pub fn has_br_table(f: &Func) -> bool {
    f.blocks
        .iter()
        .any(|b| matches!(b.term, Terminator::BrTable { .. }))
}

/// Zero-config rolled residual for a `while` / `repeat` loop with a **register** limit (`while i <= n`).
///
/// A numeric-`for` re-saves `ci->savedpc` every iteration (FORLOOP), so [`auto_rolled`] finds the loop
/// and its resume point from that in-memory pc. A `while`/`repeat` loop does **not**: `savedpc` is only
/// written at calls/errors, so it stays stale at an early instruction — useless for locating the loop or
/// resuming. This path instead reads the **real** pc from the interpreter's live IR value (index 1 in
/// `luaV_execute`'s dispatch block) during observation, and:
///  1. finds the **loop-top** = the target of the backward `JMP` (the loop's back-edge) by decoding the
///     bytecode, then overrides `ci->savedpc` to it for the resume;
///  2. forces the loop's **limit** register dynamic. A while loop counts *up* toward a limit, so the
///     swept trip parameter is the limit itself (`counter_ix`), not a down-counter. The limit must be a
///     variable (`local n`); a constant literal (`while i <= 8`, an `LEI` immediate) bakes a fixed trip
///     count and cannot roll — this asserts it finds a register-register comparison.
///
/// Returns the same [`AutoRolled`] shape as [`auto_rolled`]; `counter_ix` is the limit (sweep it to vary
/// the trip count) and `acc_addr` the returned register.
pub fn while_rolled(m: &Module, script: &str) -> AutoRolled {
    let luav = luav_execute(m);
    let dispatch = peval_capture::dispatch_block(m, luav);

    // Entry capture (register base + code/proto pointers).
    let inst = svm_run::instantiate(m.clone()).expect("instantiate");
    let mut insp = inst.debug_attach(script.as_bytes().to_vec(), u64::MAX);
    let (sp, l, ci) = read_entry(&mut insp, luav);
    let w0 = insp.read_window(0, CAPTURE_LEN).expect("window");
    drop(insp);
    let func = rd_u64(&w0, ci + CI_FUNC);
    let base = func + STACKVALUE_SIZE;
    let closure = rd_u64(&w0, func);
    let proto = rd_u64(&w0, closure + LCLOSURE_P);
    let code = rd_u64(&w0, proto + PROTO_CODE);
    let sizecode = rd_i32(&w0, proto + PROTO_SIZECODE) as usize;
    let word_at = |pc: usize| {
        u32::from_le_bytes(
            w0[(code + pc as u64 * 4) as usize..(code + pc as u64 * 4) as usize + 4]
                .try_into()
                .unwrap(),
        )
    };

    // ---- Static decode: loop-top (backward-JMP target) + the loop's register comparison. ----
    // Lua 5.4 `OP_JMP` (56) carries a signed `sJ` (bits 7..31, bias `0xFFFFFF`); target = `pc + 1 + sJ`.
    const OP_JMP: u32 = 56;
    let mut loop_top: Option<usize> = None;
    for pc in 0..sizecode {
        let word = word_at(pc);
        if word & 0x7F == OP_JMP {
            let sj = ((word >> 7) & 0x1FFFFFF) as i32 - 0xFFFFFF;
            let tgt = pc as i32 + 1 + sj;
            if tgt >= 0 && (tgt as usize) < pc {
                loop_top = Some(loop_top.map_or(tgt as usize, |t| t.min(tgt as usize)));
            }
        }
    }
    let loop_top = loop_top.expect("no backward JMP (not a while/repeat loop)");
    let loop_top_ptr = code + loop_top as u64 * 4;
    // The loop's comparison: the first register-register `EQ`(57)/`LT`(58)/`LE`(59) at/after the loop-top.
    // (A constant-limit loop uses the immediate forms `LEI`/`LTI`/… — no register to sweep, so it can't
    // roll; the `expect` below fails closed on that.)
    let (cmp_a, cmp_b) = (loop_top..sizecode)
        .find_map(|pc| {
            let word = word_at(pc);
            matches!(word & 0x7F, 57..=59)
                .then(|| (((word >> 7) & 0xFF) as u64, ((word >> 16) & 0xFF) as u64))
        })
        .expect("no register-register comparison in the loop (a constant limit cannot roll)");
    let cmp_a_addr = base + (cmp_a + 1) * STACKVALUE_SIZE; // +1: VARARGPREP frame shift
    let cmp_b_addr = base + (cmp_b + 1) * STACKVALUE_SIZE;

    // ---- Pass 1: observe the register file at loop-top hits (keyed on the real pc = IR value 1). ----
    const N_REGS: u64 = 32;
    let observe = |target_hits: usize| -> (Vec<Vec<u64>>, Vec<u8>) {
        let inst = svm_run::instantiate(m.clone()).expect("instantiate");
        let mut insp = inst.debug_attach(script.as_bytes().to_vec(), u64::MAX);
        read_entry(&mut insp, luav);
        let disp = IrPc {
            module: 0,
            func: luav,
            block: dispatch as usize,
            inst: 0,
        };
        insp.set_breakpoint(disp);
        let mut series: Vec<Vec<u64>> = vec![Vec::new(); N_REGS as usize];
        let mut first_window = Vec::new();
        let mut hits = 0;
        for _ in 0..4000 {
            match insp.run_until_stop() {
                Stop::Break {
                    reason: StopReason::Breakpoint,
                    ..
                } => {
                    let pcptr = match insp.read_ir_value(0, 1) {
                        Some(Value::I64(v)) => v as u64,
                        _ => continue,
                    };
                    if pcptr == loop_top_ptr {
                        let w = insp.read_window(0, CAPTURE_LEN).expect("w");
                        if hits == 0 {
                            first_window = w.clone();
                        }
                        for r in 0..N_REGS {
                            series[r as usize].push(rd_u64(&w, base + r * STACKVALUE_SIZE));
                        }
                        hits += 1;
                        if hits >= target_hits {
                            break;
                        }
                    }
                }
                Stop::Finished { .. } => break,
                o => panic!("observe: {o:?}"),
            }
        }
        insp.clear_breakpoint(disp);
        (series, first_window)
    };
    // The first loop-top window is the clean seed (before the first iteration: accumulator 0, counter at
    // its start), and 16 loop-top hits are enough to tell varying cells from the invariant limit.
    let (series, window) = observe(16);
    let top_hits = series[0].len();
    assert!(top_hits >= 3, "too few loop-top hits ({top_hits})");
    assert!(!window.is_empty(), "no loop-top window captured");
    let w = &window;

    // Varying cells: ≥3 distinct values across loop-top hits (the loop variable + accumulator). The limit
    // is loop-invariant (1 distinct) so it is added explicitly; force it dynamic so `i <= n` stays a
    // dynamic guard and the loop rolls over a swept trip count instead of baking a fixed one.
    let mut varying: Vec<u64> = Vec::new();
    for r in 0..N_REGS {
        let mut d = series[r as usize].clone();
        d.sort_unstable();
        d.dedup();
        if d.len() >= 3 {
            varying.push(base + r * STACKVALUE_SIZE);
        }
    }
    let is_invariant = |addr: u64| -> bool {
        let r = ((addr - base) / STACKVALUE_SIZE) as usize;
        series[r].windows(2).all(|w| w[0] == w[1])
    };
    let limit_addr = if is_invariant(cmp_a_addr) {
        cmp_a_addr
    } else {
        cmp_b_addr
    };
    assert!(
        !varying.contains(&limit_addr) && is_invariant(limit_addr),
        "the loop limit must be a loop-invariant register operand"
    );
    varying.push(limit_addr);
    varying.sort_unstable();
    // Tag filter: keep only integer-tagged cells at the safepoint (drops scratch pointers/floats).
    varying.retain(|&a| w.get((a + 8) as usize).copied() == Some(VNUMINT_TAG));
    assert!(
        varying.contains(&limit_addr),
        "the limit cell was dropped by the tag filter"
    );

    let stack_lo = rd_u64(w, l + L_STACK);
    let stack_hi = rd_u64(w, l + L_STACK_LAST);
    let stack_len = (stack_hi - stack_lo) as usize;
    let slice = |a: u64, n: usize| w[a as usize..a as usize + n].to_vec();

    // Resume point: override `ci->savedpc` to the loop-top (the real pc is not in `savedpc` for a while
    // loop). Overriding `savedpc` is the same mechanism `auto_rolled` uses (it even rewinds it by one op).
    let mut ci_image = slice(ci, CI_SIZE);
    ci_image[CI_SAVEDPC as usize..CI_SAVEDPC as usize + 8]
        .copy_from_slice(&loop_top_ptr.to_le_bytes());

    let args = [
        SpecArg::ConstI64(sp),
        SpecArg::ConstI64(l as i64),
        SpecArg::ConstI64(ci as i64),
    ];
    let cfg = SpecConfig {
        const_overlays: vec![
            (l, slice(l, LUA_STATE_SIZE)),
            (stack_lo, slice(stack_lo, stack_len)),
            (ci, ci_image.clone()),
            (code, slice(code, 4 * sizecode)),
            (closure, slice(closure, 48)),
            (proto, slice(proto, 128)),
        ],
        rename: Some((l, l + LUA_STATE_SIZE as u64)),
        rename_extra: vec![(stack_lo, stack_hi), (ci, ci + CI_SIZE as u64)],
        rename_is_private: true,
        rename_seed_from_image: true,
        dynamic_cells: varying.iter().map(|&a| (a, 8)).collect(),
        indirect_targets_cap: Some(16),
        ..SpecConfig::default()
    };
    let residual = specialize_with_config(m, luav, &args, &cfg).expect("while/repeat specialize");
    svm_verify::verify_module(&residual).expect("residual verifies");

    // Result register = the `RETURN1 A` / `RETURN A (B!=1)` operand (decoded, as in `auto_rolled`).
    let mut result_reg: Option<u64> = None;
    for pc in 0..sizecode {
        let word = word_at(pc);
        let op = word & 0x7F;
        let a = ((word >> 7) & 0xFF) as u64;
        let b = (word >> 16) & 0xFF;
        if op == 72 || (op == 70 && b != 1) {
            result_reg = Some(a);
            break;
        }
    }
    let acc_addr = result_reg
        .map(|a| base + (a + 1) * STACKVALUE_SIZE)
        .filter(|a| varying.contains(a))
        .unwrap_or_else(|| {
            *varying
                .iter()
                .find(|&&a| a != limit_addr)
                .expect("a carried cell distinct from the limit")
        });
    let counter_ix = varying.iter().position(|&a| a == limit_addr).unwrap();
    let acc_ix = varying.iter().position(|&a| a == acc_addr).unwrap_or(0);
    let captured: Vec<i64> = varying.iter().map(|&a| rd_u64(w, a) as i64).collect();

    AutoRolled {
        residual_blocks: residual.funcs[0].blocks.len(),
        base_blocks: m.funcs[luav as usize].blocks.len(),
        residual,
        entry: 0,
        dyn_cells: varying,
        counter_ix,
        acc_ix,
        acc_addr,
        captured,
        reg_base: base,
        reg_stride: STACKVALUE_SIZE,
    }
}
