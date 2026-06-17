//! The wasm debug-info on-ramp (DEBUGGING.md §6/W4): a clang-compiled wasm carrying embedded DWARF
//! `.debug_line` transpiles to IR with the §6 debug-info waist populated — source locations mapped
//! onto IR pcs by a *second* producer (not chibicc), exercising the frontend-neutral boundary.

use std::collections::BTreeSet;

// Built with: clang --target=wasm32 -g -O0 -nostdlib -Wl,--no-entry -Wl,--export-all
//   int add(int a, int b) { int s = a + b; /*L2*/ return s + 1; /*L3*/ }
const DLINE: &[u8] = include_bytes!("fixtures/dline_clang.wasm");

#[test]
fn wasm_dwarf_line_maps_into_the_debug_info_waist() {
    let t = svm_wasm::transpile(DLINE).expect("transpile");
    // The debug section is strippable / untrusted-for-escape — it must not affect verification.
    svm_verify::verify_module(&t.module).expect("verify");

    let di = t
        .module
        .debug_info
        .as_ref()
        .expect("debug info populated from .debug_line");

    // The file table came from the DWARF line-program header.
    assert!(
        di.files.iter().any(|f| f.ends_with("dline.c")),
        "file table names the C source: {:?}",
        di.files
    );
    // Source locations are mapped onto IR pcs (variable ingest is asserted separately below).
    assert!(!di.locs.is_empty(), "some source locations were mapped");

    // Every `.debug_*` section is carried through verbatim as a §6 rich blob (for a future DWARF
    // re-emitter) — including the variable-bearing `.debug_info`, which the core doesn't yet parse.
    assert!(
        di.blobs.iter().any(|b| b.producer == ".debug_line"),
        "passes .debug_line through as a blob: {:?}",
        di.blobs.iter().map(|b| &b.producer).collect::<Vec<_>>()
    );
    let info = di
        .blobs
        .iter()
        .find(|b| b.producer == ".debug_info")
        .expect("carries .debug_info verbatim");
    assert!(!info.bytes.is_empty(), "the blob is the raw section bytes");

    // The body's source lines (2: `int s = a + b;`, 3: `return s + 1;`) are present, and every loc
    // resolves to an in-range IR pc — the cross-check that the wasm-offset→IR-pc mapping is sane.
    let lines: BTreeSet<u32> = di.locs.iter().map(|l| l.line).collect();
    assert!(
        lines.contains(&2) && lines.contains(&3),
        "body lines mapped: {lines:?}"
    );
    for l in &di.locs {
        assert!(l.file as usize == 0, "single source file");
        let f = l.func as usize;
        assert!(f < t.module.funcs.len(), "loc func {f} in range");
        let b = l.block as usize;
        assert!(b < t.module.funcs[f].blocks.len(), "loc block in range");
        assert!(
            (l.inst as usize) < t.module.funcs[f].blocks[b].insts.len(),
            "loc inst in range"
        );
    }
}

#[test]
fn wasm_dwarf_info_extracts_source_variables() {
    // The §6 variable-ingest foundation: the DWARF `.debug_info` reader recovers each source
    // variable (name, `DW_OP_fbreg` offset, type) and the subprogram's frame-base wasm local from
    // the real clang fixture. (Wiring these into `debug.var` is a follow-up slice.)
    let t = svm_wasm::transpile(DLINE).expect("transpile");
    let blobs = &t.module.debug_info.as_ref().unwrap().blobs;
    let sec = |name: &str| -> &[u8] {
        blobs
            .iter()
            .find(|b| b.producer == name)
            .map(|b| b.bytes.as_slice())
            .unwrap_or(&[])
    };
    let info =
        svm_wasm::dwarf_info::parse(sec(".debug_info"), sec(".debug_abbrev"), sec(".debug_str"))
            .expect("parse .debug_info");

    // The `add(int a, int b)` subprogram: frame base is wasm local 4, with a/b/s at fbreg +12/+8/+4.
    let add = info
        .subs
        .iter()
        .find(|s| s.vars.iter().any(|v| v.name == "a"))
        .expect("the add() subprogram with named vars");
    assert_eq!(add.frame_base_local, Some(4), "frame base is wasm local 4");

    let var = |n: &str| {
        add.vars
            .iter()
            .find(|v| v.name == n)
            .unwrap_or_else(|| panic!("var {n}"))
    };
    assert_eq!(var("a").fbreg, 12);
    assert_eq!(var("b").fbreg, 8);
    assert_eq!(var("s").fbreg, 4); // `int s = a + b;`

    // The variable type resolves to the `int` base type (signed = DW_ATE_signed = 5, 4 bytes).
    let int_ty = info
        .base_types
        .get(&var("s").type_ref)
        .expect("s's type DIE");
    assert_eq!(int_ty.name, "int");
    assert_eq!((int_ty.encoding, int_ty.size), (5, 4));
}

#[test]
fn wasm_dwarf_variables_ingested_into_the_waist() {
    // End-to-end: a clang wasm guest's source variables land in the §6 waist as named `debug.var`s
    // with a `WindowVia` location (the DWARF frame-base local resolved per pc + the `fbreg` offset)
    // and a structured `int` type — a second producer feeding the *variable* half of the waist.
    use svm_ir::{Encoding, TypeDef, VarLoc};

    let t = svm_wasm::transpile(DLINE).expect("transpile");
    svm_verify::verify_module(&t.module).expect("verify"); // debug info is escape-irrelevant
    let di = t.module.debug_info.as_ref().expect("debug info");

    // a, b, s are present, each a window-via-frame-base var into the C stack frame.
    let var = |n: &str| {
        di.vars.iter().find(|v| v.name == n).unwrap_or_else(|| {
            panic!(
                "var {n} ingested: {:?}",
                di.vars.iter().map(|v| &v.name).collect::<Vec<_>>()
            )
        })
    };
    for n in ["a", "b", "s"] {
        let v = var(n);
        let VarLoc::WindowVia { base, off } = &v.loc else {
            panic!("{n} is a WindowVia var, got {:?}", v.loc);
        };
        // The base loclist (the frame-base wasm local's SSA value per pc) is non-empty, and `off` is
        // the `DW_OP_fbreg` offset (a/b/s at +12/+8/+4).
        assert!(!base.is_empty(), "{n} has a frame-base location list");
        let expected_off = match n {
            "a" => 12,
            "b" => 8,
            _ => 4,
        };
        assert_eq!(*off, expected_off, "{n} fbreg offset");
        // All resolve to the same in-range IR function.
        let func = v.func as usize;
        assert!(func < t.module.funcs.len());
        for l in base {
            assert!(
                (l.block as usize) < t.module.funcs[func].blocks.len(),
                "base block in range"
            );
        }
        // Type is the structured `int` (signed, 4 bytes).
        let tid = v.type_id.expect("typed");
        assert!(matches!(
            &di.types[tid as usize],
            TypeDef::Base { name, encoding: Encoding::Signed, size: 4 } if name == "int"
        ));
    }
}
