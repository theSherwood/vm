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
    // Only source locations are ingested as core (no types/vars yet).
    assert!(di.types.is_empty() && di.vars.is_empty());
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
