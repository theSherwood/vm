//! In-window dynamic linking, milestone 0: **compile-time (static) linking of a function symbol.**
//!
//! A unit `caller` references another unit `add` by *name* (`call.import "add"`). The loader resolves
//! the name to `add`'s function index and `svm_ir::resolve_imports_with` rewrites the `CallImport`
//! into a **direct `call`** — exactly what a static linker does (symbol → concrete call). By the time
//! the verifier and both backends see the module, it's an ordinary closed module; "linking" was a
//! source-to-source rewrite, above the TCB, re-verified like everything else. (Dynamic, separately-
//! compiled linking — `call_indirect` through a `Jit.install` slot — is the next milestone.)

use svm_interp::Value;
use svm_ir::{Resolved, ResolvedCap};

/// Two "units" in one module: `add(a,b)=a+b` at index 0, and `caller(a,b)` (index 1) that calls
/// `add` **by name**. The dummy `v2` is the (unused) capability-handle operand `call.import` carries;
/// resolving to a `Func` drops it.
const TWO_UNITS: &str = "\
func (i32, i32) -> (i32) {
block0(v0: i32, v1: i32):
  v2 = i32.add v0 v1
  return v2
}

func (i32, i32) -> (i32) {
block0(v0: i32, v1: i32):
  v2 = i32.const 0
  v3 = call.import \"add\" (i32, i32) -> (i32) v2 (v0, v1)
  return v3
}
";

/// Resolve + verify, then run `caller` (entry 1) on interp + JIT with `args`; assert they agree and
/// return the i32 result.
fn link_and_run(resolver: impl FnMut(&str) -> Option<Resolved>, args: &[i32]) -> i64 {
    let m = svm_text::parse_module(TWO_UNITS).expect("parse");
    assert_eq!(m.imports.len(), 1, "one named import: \"add\"");
    // The compile-time link step: rewrite call.import "add" → a direct call to add's index.
    let linked = svm_ir::resolve_imports_with(&m, resolver).expect("resolve");
    assert!(linked.imports.is_empty(), "imports lowered away");
    // No CallImport survives; it became a direct Call.
    assert!(
        linked.funcs[1].blocks[0]
            .insts
            .iter()
            .all(|i| !matches!(i, svm_ir::Inst::CallImport { .. })),
        "the import must be lowered to a direct call"
    );
    svm_verify::verify_module(&linked).expect("verify linked module");

    let ivals: Vec<Value> = args.iter().map(|&x| Value::I32(x)).collect();
    let mut fuel = 10_000_000u64;
    let interp = svm_interp::run(&linked, 1, &ivals, &mut fuel).expect("interp run");
    let jargs: Vec<i64> = args.iter().map(|&x| x as i64).collect();
    let jit = match svm_jit::compile_and_run(&linked, 1, &jargs).expect("jit compile") {
        svm_jit::JitOutcome::Returned(v) => v,
        other => panic!("jit did not return: {other:?}"),
    };
    let iv = match interp[0] {
        Value::I32(x) => x as i64,
        other => panic!("unexpected interp value {other:?}"),
    };
    assert_eq!(iv as u32 as u64, jit[0] as u32 as u64, "interp != jit");
    iv
}

/// The core: `caller` reaches `add` purely by name, resolved at link time to a direct call.
#[test]
fn caller_links_to_add_by_name() {
    assert_eq!(
        link_and_run(|n| (n == "add").then_some(Resolved::Func(0)), &[3, 4]),
        7
    );
    assert_eq!(
        link_and_run(|n| (n == "add").then_some(Resolved::Func(0)), &[100, -1]),
        99
    );
}

/// An **unresolved** symbol is fail-closed (the loader can't find `add`).
#[test]
fn unresolved_symbol_fails_closed() {
    let m = svm_text::parse_module(TWO_UNITS).expect("parse");
    let err = svm_ir::resolve_imports_with(&m, |_| None).expect_err("must fail closed");
    assert_eq!(err, svm_ir::ImportError::Unresolved("add".into()));
}

/// A **signature mismatch** can't produce a type-unsafe call: linking feeds the re-verifier, never
/// bypasses it. `sym` is declared `(i32,i32)->i32` but resolved to a `(i64)->i64` function, so the
/// rewritten direct call has the wrong arg count/types — and `verify_module` rejects the linked
/// module. (This is the link-time symbol-signature check, enforced by re-verification, not trust.)
#[test]
fn signature_mismatch_is_caught_by_reverify() {
    let src = "\
func (i64) -> (i64) {
block0(v0: i64):
  return v0
}

func (i32, i32) -> (i32) {
block0(v0: i32, v1: i32):
  v2 = i32.const 0
  v3 = call.import \"sym\" (i32, i32) -> (i32) v2 (v0, v1)
  return v3
}
";
    let m = svm_text::parse_module(src).expect("parse");
    let linked = svm_ir::resolve_imports_with(&m, |_| Some(Resolved::Func(0))).expect("resolve");
    assert!(
        svm_verify::verify_module(&linked).is_err(),
        "a signature-mismatched link must be rejected by re-verification"
    );
}

/// The generalized pass still does the §7 capability case (`Resolved::Cap`) — a sanity check that the
/// `resolve_imports` (cap-only) path is unchanged by delegating through `resolve_imports_with`.
#[test]
fn capability_resolution_still_works() {
    let src = "\
func (i32) -> (i32) {
block0(v0: i32):
  v1 = i32.const 0
  v2 = call.import \"write\" (i32) -> (i32) v0 (v1)
  return v2
}
";
    let m = svm_text::parse_module(src).expect("parse");
    let linked = svm_ir::resolve_imports_with(&m, |_| {
        Some(Resolved::Cap(ResolvedCap { type_id: 0, op: 1 }))
    })
    .expect("resolve");
    // The import lowered to a cap.call (not a direct call).
    assert!(linked.funcs[0].blocks[0].insts.iter().any(|i| matches!(
        i,
        svm_ir::Inst::CapCall {
            type_id: 0,
            op: 1,
            ..
        }
    )));
}

// ---------------------------------------------------------------------------------------------
// Milestone 1: the static linker — concatenate *separate* units into one program (svm_ir::link).
// ---------------------------------------------------------------------------------------------

use svm_ir::{link, LinkUnit};

/// Run entry `idx` of an already-verified module on interp + JIT, assert they agree, return the i32.
fn run_entry(m: &svm_ir::Module, idx: u32, args: &[i32]) -> i64 {
    let ivals: Vec<Value> = args.iter().map(|&x| Value::I32(x)).collect();
    let mut fuel = 10_000_000u64;
    let interp = svm_interp::run(m, idx, &ivals, &mut fuel).expect("interp run");
    let jargs: Vec<i64> = args.iter().map(|&x| x as i64).collect();
    let jit = match svm_jit::compile_and_run(m, idx, &jargs).expect("jit compile") {
        svm_jit::JitOutcome::Returned(v) => v,
        other => panic!("jit did not return: {other:?}"),
    };
    let iv = match interp[0] {
        Value::I32(x) => x as i64,
        other => panic!("unexpected interp value {other:?}"),
    };
    assert_eq!(iv as u32 as u64, jit[0] as u32 as u64, "interp != jit");
    iv
}

fn unit(src: &str, exports: &[(&str, u32)]) -> LinkUnit {
    LinkUnit {
        module: svm_text::parse_module(src).expect("parse unit"),
        exports: exports.iter().map(|(n, i)| (n.to_string(), *i)).collect(),
        ..Default::default()
    }
}

const MATH_UNIT: &str = "\
func (i32, i32) -> (i32) {
block0(v0: i32, v1: i32):
  v2 = i32.add v0 v1
  return v2
}
";

/// `app` calls `add` by name; it lives in a **separate** unit (`math`). The linker concatenates them
/// (app's functions reindexed after math's) and resolves the import to a direct call.
const APP_UNIT: &str = "\
func (i32, i32) -> (i32) {
block0(v0: i32, v1: i32):
  v2 = i32.const 0
  v3 = call.import \"add\" (i32, i32) -> (i32) v2 (v0, v1)
  return v3
}
";

#[test]
fn links_two_separate_units_into_one_program() {
    let linked = link(&[unit(MATH_UNIT, &[("add", 0)]), unit(APP_UNIT, &[])]).expect("link");
    // math's `add` is function 0; app's `main` is function 1 (reindexed after math).
    assert_eq!(linked.funcs.len(), 2);
    assert!(linked.imports.is_empty(), "all imports resolved");
    svm_verify::verify_module(&linked).expect("verify linked program");
    // app's main (entry 1) calls into math's add across the unit boundary.
    assert_eq!(run_entry(&linked, 1, &[3, 4]), 7);
    assert_eq!(run_entry(&linked, 1, &[40, 2]), 42);
}

/// A three-unit chain proves reindexing across more than two units: `app` → `add`, where `add` itself
/// lives after an unrelated `pad` unit, so its global index is shifted and the import still resolves.
#[test]
fn links_across_a_reindexing_offset() {
    let pad = "\
func (i32) -> (i32) {
block0(v0: i32):
  return v0
}
"; // an unrelated unit so `math` lands at a non-zero base
    let linked = link(&[
        unit(pad, &[("pad", 0)]),
        unit(MATH_UNIT, &[("add", 0)]), // global index 1
        unit(APP_UNIT, &[]),            // global index 2; its "add" → 1
    ])
    .expect("link");
    svm_verify::verify_module(&linked).expect("verify");
    assert_eq!(
        run_entry(&linked, 2, &[10, 5]),
        15,
        "app(entry 2) → add at global index 1"
    );
}

/// An import no unit exports is fail-closed.
#[test]
fn link_unresolved_symbol_fails_closed() {
    let err = link(&[unit(APP_UNIT, &[])]).expect_err("nothing exports add");
    assert_eq!(err, svm_ir::LinkError::Unresolved("add".into()));
}

/// Two units exporting the same symbol is fail-closed.
#[test]
fn link_duplicate_symbol_fails_closed() {
    let err = link(&[
        unit(MATH_UNIT, &[("add", 0)]),
        unit(MATH_UNIT, &[("add", 0)]),
    ])
    .expect_err("two `add`s");
    assert_eq!(err, svm_ir::LinkError::DuplicateSymbol("add".into()));
}

// ---------------------------------------------------------------------------------------------
// Milestone 2: data symbols + per-unit data relocation (LinkUnit data_exports + relocations).
// ---------------------------------------------------------------------------------------------

use svm_ir::{DataReloc, RelocKind};

/// 16 bytes of padding data so the unit that follows lands at a **non-zero** data base — making the
/// relocation observable (a coincidental base of 0 would prove nothing).
const PAD16: &str = "\
memory 16
data 0 \"\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\"
func (i32) -> (i32) {
block0(v0: i32):
  return v0
}
";

/// A **cross-unit data symbol**: `store` exports the byte 42 as data symbol "answer"; `load` reads it
/// by name. The linker places `store`'s data at a non-zero base (after `pad`), records "answer" at
/// that window address, and patches `load`'s address constant (left at 0) to it — so `load` reads the
/// byte wherever the linker put it. Proves the data moved and the reference followed.
#[test]
fn cross_unit_data_symbol_is_relocated() {
    let store = LinkUnit {
        module: svm_text::parse_module(
            "memory 16\ndata 0 \"\\x2a\"\nfunc (i32) -> (i32) {\nblock0(v0: i32):\n  return v0\n}\n",
        )
        .unwrap(),
        data_exports: vec![("answer".into(), 0)],
        ..Default::default()
    };
    let load = LinkUnit {
        module: svm_text::parse_module(
            "memory 16\n\
             func (i32) -> (i32) {\n\
             block0(v0: i32):\n\
             \x20 v1 = i64.const 0\n\
             \x20 v2 = i32.load8_u v1\n\
             \x20 return v2\n\
             }\n",
        )
        .unwrap(),
        // The address const (func 0, block 0, inst 0) is the address of "answer".
        relocations: vec![DataReloc {
            func: 0,
            block: 0,
            inst: 0,
            kind: RelocKind::DataSymbol("answer".into()),
        }],
        ..Default::default()
    };
    let linked = link(&[unit(PAD16, &[]), store, load]).expect("link");
    svm_verify::verify_module(&linked).expect("verify");
    // `load` is the 3rd unit's function → global index 2.
    assert_eq!(
        run_entry(&linked, 2, &[0]),
        42,
        "read the relocated cross-unit datum"
    );
}

/// **Self-data relocation**: a unit references its *own* data by a unit-local offset; linked after
/// `pad`, its data moves to a non-zero base and its own address const is shifted by the same base
/// (`SelfData`), so the reference still lands on its data. The const here is the local offset 0, so a
/// passing read proves the `+ base` was applied to both the segment and the reference identically.
#[test]
fn self_data_is_relocated() {
    let me = LinkUnit {
        module: svm_text::parse_module(
            "memory 16\n\
             data 0 \"\\x07\"\n\
             func (i32) -> (i32) {\n\
             block0(v0: i32):\n\
             \x20 v1 = i64.const 0\n\
             \x20 v2 = i32.load8_u v1\n\
             \x20 return v2\n\
             }\n",
        )
        .unwrap(),
        relocations: vec![DataReloc {
            func: 0,
            block: 0,
            inst: 0,
            kind: RelocKind::SelfData,
        }],
        ..Default::default()
    };
    let linked = link(&[unit(PAD16, &[]), me]).expect("link");
    svm_verify::verify_module(&linked).expect("verify");
    assert_eq!(
        run_entry(&linked, 1, &[0]),
        7,
        "own data ref follows the relocation"
    );
}

/// A relocation that points at a non-constant instruction is fail-closed.
#[test]
fn bad_relocation_fails_closed() {
    let u = LinkUnit {
        module: svm_text::parse_module("func (i32) -> (i32) {\nblock0(v0: i32):\n  return v0\n}\n")
            .unwrap(),
        // inst 0 of an empty block body doesn't exist → BadReloc.
        relocations: vec![DataReloc {
            func: 0,
            block: 0,
            inst: 0,
            kind: RelocKind::SelfData,
        }],
        ..Default::default()
    };
    assert!(matches!(link(&[u]), Err(svm_ir::LinkError::BadReloc(_))));
}

// ---------------------------------------------------------------------------------------------
// Milestone 3: dynamic linking — resolve a symbol to a call_indirect TABLE SLOT (Resolved::Slot).
// A separately-compiled unit reaches a function it doesn't share an index space with, by slot.
// ---------------------------------------------------------------------------------------------

/// `main` imports `F` by name and the loader resolves it to **table slot 1** — not a direct call but a
/// `call_indirect` through the shared function table (how a separately-compiled unit reaches another).
/// `F` (slot 1) is `a*2 + b`; a decoy `G` sits at slot 0. The handle placeholder const (`i32.const 0`)
/// is patched to `1` and reused as the index, so a passing `F(10,3)=23` (not `G`'s 7) proves the slot.
#[test]
fn import_resolves_to_a_call_indirect_slot() {
    let src = "\
func (i32, i32) -> (i32) {
block0(v0: i32, v1: i32):
  v2 = i32.sub v0 v1
  return v2
}

func (i32, i32) -> (i32) {
block0(v0: i32, v1: i32):
  v2 = i32.const 2
  v3 = i32.mul v0 v2
  v4 = i32.add v3 v1
  return v4
}

func (i32, i32) -> (i32) {
block0(v0: i32, v1: i32):
  v2 = i32.const 0
  v3 = call.import \"F\" (i32, i32) -> (i32) v2 (v0, v1)
  return v3
}
";
    let m = svm_text::parse_module(src).expect("parse");
    let linked =
        svm_ir::resolve_imports_with(&m, |n| (n == "F").then_some(svm_ir::Resolved::Slot(1)))
            .expect("resolve to slot");
    // The import became a `call_indirect`, and the handle const was patched to the slot (1).
    let insts = &linked.funcs[2].blocks[0].insts;
    assert!(
        matches!(insts[0], svm_ir::Inst::ConstI32(1)),
        "handle const patched to slot 1"
    );
    assert!(
        matches!(insts[1], svm_ir::Inst::CallIndirect { .. }),
        "import lowered to call_indirect, not a direct call"
    );
    svm_verify::verify_module(&linked).expect("verify");
    // main (entry 2) dispatches to slot 1 = F(a,b) = a*2+b; F(10,3) = 23 (G would give 7).
    assert_eq!(
        run_entry(&linked, 2, &[10, 3]),
        23,
        "reached F via the resolved slot"
    );
}

/// A `Slot` import whose handle operand isn't a `ConstI32` placeholder is fail-closed (the frontend
/// must emit one — it's patched to the slot and reused as the index).
#[test]
fn slot_import_requires_a_const_handle() {
    // The handle here is a block *parameter* (v0), not a const → SlotHandleNotConst.
    let src = "\
func (i32, i32) -> (i32) {
block0(v0: i32, v1: i32):
  v2 = call.import \"F\" (i32) -> (i32) v0 (v1)
  return v2
}
";
    let m = svm_text::parse_module(src).expect("parse");
    let err = svm_ir::resolve_imports_with(&m, |_| Some(svm_ir::Resolved::Slot(0)))
        .expect_err("non-const handle must fail closed");
    assert_eq!(err, svm_ir::ImportError::SlotHandleNotConst);
}

/// The linker merges the units' impl surfaces (IMPORTS.md §3.2/OQ3): interfaces concatenate
/// with a per-unit index offset, offers reindex their interface reference and op funcidxs
/// through the same offsets as function exports, and an offer name colliding with any symbol
/// fails closed like a duplicate export.
#[test]
fn link_merges_impl_surfaces_across_units() {
    let provider = unit(
        "type (i32, i32) -> (i32)\n\
         interface { add: 0 }\n\
         export \"adder\" impl 1 : 0\n\n\
         func (i32, i32) -> (i32) {\n\
         block0(v0: i32, v1: i32):\n\
           v2 = i32.add v0 v1\n\
           return v2\n\
         }\n",
        &[],
    );
    let other = unit(MATH_UNIT, &[("add", 0)]);
    // `other` first, so the provider's funcs and interface reindex across a nonzero offset.
    let m = svm_ir::link(&[other, provider]).expect("links");
    svm_verify::verify_module(&m).expect("merged module verifies");
    assert_eq!(
        m.types.len(),
        2,
        "type section merged (one Func + one Interface)"
    );
    let offer = m
        .resolve_impl_export("adder")
        .expect("offer survives the merge");
    assert_eq!(offer.interface, 1);
    assert_eq!(
        offer.ops,
        vec![1],
        "op funcidx reindexed past the first unit"
    );

    // An offer name colliding with a function export symbol fails closed.
    let clash = unit(
        "type (i32, i32) -> (i32)\n\
         interface { add: 0 }\n\
         export \"add\" impl 1 : 0\n\n\
         func (i32, i32) -> (i32) {\n\
         block0(v0: i32, v1: i32):\n\
           v2 = i32.add v0 v1\n\
           return v2\n\
         }\n",
        &[],
    );
    let named = unit(MATH_UNIT, &[("add", 0)]);
    assert!(
        matches!(
            svm_ir::link(&[named, clash]),
            Err(svm_ir::LinkError::DuplicateSymbol(n)) if n == "add"
        ),
        "offer/export name collision fails the link closed"
    );
}
