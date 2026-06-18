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
    let svm_wasm::dwarf_info::DwarfType::Base {
        name,
        encoding,
        size,
    } = info.types.get(&var("s").type_ref).expect("s's type DIE")
    else {
        panic!("s is a base type");
    };
    assert_eq!(name, "int");
    assert_eq!((*encoding, *size), (5, 4));
}

// dist(n): struct Point{int x,y} p; int row[3]; struct Point *pp; — exercises aggregate/array/
// pointer DWARF type DIEs.
const AGG: &[u8] = include_bytes!("fixtures/agg_clang.wasm");

#[test]
fn wasm_dwarf_ingests_aggregate_pointer_and_array_types() {
    use svm_ir::{TypeDef, VarLoc};

    let t = svm_wasm::transpile(AGG).expect("transpile");
    svm_verify::verify_module(&t.module).expect("verify");
    let di = t.module.debug_info.as_ref().expect("debug info");
    let types = &di.types;
    let var = |n: &str| {
        di.vars
            .iter()
            .find(|v| v.name == n)
            .unwrap_or_else(|| panic!("var {n}"))
    };
    let var_type = |n: &str| &types[var(n).type_id.expect("typed") as usize];

    // Every local is a WindowVia into the C frame.
    for n in ["p", "row", "pp"] {
        assert!(
            matches!(var(n).loc, VarLoc::WindowVia { .. }),
            "{n} is WindowVia"
        );
    }

    // `struct Point p` — an aggregate with x@0, y@4, both 4-byte ints, size 8.
    let TypeDef::Aggregate { name, size, fields } = var_type("p") else {
        panic!("p is a struct, got {:?}", var_type("p"));
    };
    assert_eq!(name, "struct Point");
    assert_eq!(*size, 8);
    assert_eq!(
        fields
            .iter()
            .map(|f| (f.name.as_str(), f.offset))
            .collect::<Vec<_>>(),
        vec![("x", 0), ("y", 4)]
    );
    assert!(matches!(
        &types[fields[0].ty as usize],
        TypeDef::Base { size: 4, .. }
    ));

    // `int row[3]` — array of 3 ints.
    let TypeDef::Array { elem, count, .. } = var_type("row") else {
        panic!("row is an array");
    };
    assert_eq!(*count, 3);
    assert!(matches!(
        &types[*elem as usize],
        TypeDef::Base { size: 4, .. }
    ));

    // `struct Point *pp` — pointer whose pointee is the same aggregate as `p`.
    let TypeDef::Pointer { pointee, name, .. } = var_type("pp") else {
        panic!("pp is a pointer");
    };
    assert_eq!(name, "struct Point *");
    assert!(
        matches!(&types[*pointee as usize], TypeDef::Aggregate { name, .. } if name == "struct Point")
    );
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

// Built with: clang --target=wasm32 -g -O0 -nostdlib -Wl,--no-entry -Wl,--export-all
//   int counter = 7; struct Point { int x; int y; } origin = { 3, 4 };
//   int bump(int n) { counter = counter + n; return counter + origin.x; }
const GLOBALS: &[u8] = include_bytes!("fixtures/global_clang.wasm");

#[test]
fn wasm_dwarf_ingests_module_scoped_globals() {
    // The wasm DWARF producer as the *third* emitter of the §6 module-scoped-global primitive (slice
    // 28 / 30): a CU-level `DW_TAG_variable` at a fixed `DW_OP_addr` becomes a `GLOBAL_SCOPE`
    // `VarLoc::Fixed` var at that linear-memory (= window) address, with its structured type.
    use svm_ir::{TypeDef, VarLoc};

    let t = svm_wasm::transpile(GLOBALS).expect("transpile");
    svm_verify::verify_module(&t.module).expect("verify");
    let di = t.module.debug_info.as_ref().expect("debug info");

    let g = |n: &str| {
        di.vars
            .iter()
            .find(|v| v.name == n && v.func == svm_ir::GLOBAL_SCOPE)
            .unwrap_or_else(|| {
                panic!(
                    "global {n}: {:?}",
                    di.vars
                        .iter()
                        .map(|v| (&v.name, v.func))
                        .collect::<Vec<_>>()
                )
            })
    };

    // `int counter` — a fixed-address int global.
    let counter = g("counter");
    let VarLoc::Fixed { addr } = counter.loc else {
        panic!("counter is Fixed, got {:?}", counter.loc);
    };
    assert!(addr != 0, "a real linear-memory address");
    assert!(matches!(
        &di.types[counter.type_id.expect("typed") as usize],
        TypeDef::Base { size: 4, .. }
    ));

    // `struct Point origin` — a fixed-address aggregate.
    let origin = g("origin");
    assert!(matches!(origin.loc, VarLoc::Fixed { .. }));
    assert!(matches!(
        &di.types[origin.type_id.expect("typed") as usize],
        TypeDef::Aggregate { name, fields, .. } if name == "struct Point" && fields.len() == 2
    ));
}

#[test]
fn wasm_global_reads_its_value_at_runtime() {
    // The fixed linear address maps to the window directly, so the data-segment value reads back: at
    // `bump`'s entry the global `counter` holds its initializer 7.
    use svm_interp::{Inspector, IrPc, Stop, Value, VarValue};

    let t = svm_wasm::transpile(GLOBALS).expect("transpile");
    svm_verify::verify_module(&t.module).expect("verify");
    let bump = t
        .exports
        .iter()
        .find(|(n, _)| n == "bump")
        .expect("bump export")
        .1;

    let mut insp = Inspector::attach(&t.module, bump, &[Value::I32(10)], 5_000_000);
    insp.set_breakpoint(IrPc {
        module: 0,
        func: bump,
        block: 0,
        inst: 0,
    });
    assert!(
        matches!(insp.run_until_stop(), Stop::Break { .. }),
        "stopped at bump entry"
    );
    let v = insp.read_var(0, "counter", 4).expect("counter readable");
    let got = match v {
        VarValue::Bytes(b) => i32::from_le_bytes(b[..4].try_into().unwrap()),
        VarValue::Value(Value::I32(n)) => n,
        other => panic!("unexpected {other:?}"),
    };
    assert_eq!(got, 7, "counter's initializer read through the global");
}
