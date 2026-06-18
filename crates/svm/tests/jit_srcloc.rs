//! W5 JIT/DWARF tier, Stages 0–2 (DEBUGGING.md §7): the JIT threads each op's §6 `debug.loc` into a
//! Cranelift `SourceLoc`, after `finalize` builds a finalized-machine-address → source map that
//! `CompiledModule::symbolize` resolves (Stages 0–1), and synthesizes a DWARF `.debug_line` section
//! over that map (Stage 2) — round-tripped here through the existing `svm_wasm::dwarf_line` reader,
//! no debugger needed. Host-side tooling, off the runtime path (§2a).

use std::collections::BTreeSet;

use svm_ir::DEFAULT_RESERVED_LOG2;
use svm_jit::{CompiledModule, Quota, INERT_CAP_THUNK};
use svm_text::parse_module;

/// A pure-compute function with a hand-written §6 debug section: source lines 2, 3, 4 map onto its
/// three computing ops (the §6 `debug.loc` rows the JIT must carry into the machine-code address
/// map). `compute(a)`: line 2 `t = a + 1`, line 3 `u = t * 3`, line 4 `return u - 2`.
const COMPUTE_DBG: &str = r#"
func (i32) -> (i32) {
block0(v0: i32):
  v1 = i32.const 1
  v2 = i32.add v0 v1
  v3 = i32.const 3
  v4 = i32.mul v2 v3
  v5 = i32.const 2
  v6 = i32.sub v4 v5
  return v6
}

debug.file 0 "compute.c"
debug.loc 0 0 1 0 2 7
debug.loc 0 0 3 0 3 7
debug.loc 0 0 5 0 4 3
"#;

fn compile(src: &str) -> CompiledModule {
    let m = parse_module(src).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    CompiledModule::compile(
        &m,
        0,
        INERT_CAP_THUNK,
        core::ptr::null_mut(),
        DEFAULT_RESERVED_LOG2,
        None,
        None,
        None,
        None,
        Quota::default(),
        0,
    )
    .expect("jit compiles")
}

#[test]
fn jit_threads_source_locs_into_a_machine_address_map() {
    let cm = compile(COMPUTE_DBG);
    let ranges = cm.src_ranges();
    assert!(
        !ranges.is_empty(),
        "the JIT carried source locs into finalized code"
    );

    // The source lines the address map covers are exactly the `debug.loc` body lines (2, 3, 4) —
    // every source position survived lowering + finalize into a real machine-code range.
    let lines: BTreeSet<u32> = ranges.iter().map(|r| r.line).collect();
    assert_eq!(
        lines,
        BTreeSet::from([2, 3, 4]),
        "the three body lines are mapped"
    );

    // Each range is a non-empty machine-address span naming the C source.
    for r in ranges {
        assert!(r.lo < r.hi, "range covers real code: {r:?}");
        assert_eq!(r.func, 0);
    }

    // Symbolizing the start of each mapped range round-trips to its source line/file — the
    // machine-pc → source resolution the trap symbolizer / DWARF emitter will use.
    for r in ranges {
        let loc = cm
            .symbolize(r.lo as usize)
            .unwrap_or_else(|| panic!("symbolize {:#x}", r.lo));
        assert_eq!(loc.line, r.line, "symbolize matches the range's line");
        assert_eq!(loc.file, "compute.c", "file resolves to the C source");
    }

    // An address well outside the JIT'd code has no source mapping.
    assert!(cm.symbolize(0).is_none(), "unmapped address ⇒ None");
}

#[test]
fn jit_without_debug_info_has_no_source_map() {
    // No `debug.*` section ⇒ no source locs stamped (codegen byte-identical to before) and an empty
    // map, so `symbolize` always returns `None`.
    let cm = compile("func (i32) -> (i32) {\nblock0(v0: i32):\n  return v0\n}\n");
    assert!(cm.src_ranges().is_empty(), "no -g ⇒ no source map");
    assert!(cm
        .symbolize(cm.src_ranges().first().map_or(0x1000, |r| r.lo as usize))
        .is_none());
}

#[test]
fn jit_emits_debug_line_that_round_trips_through_the_reader() {
    // Stage 2: the synthesized `.debug_line` section, parsed back by the project's own DWARF
    // line-program reader, reconstructs the exact machine-address → (file, line) map — the
    // address→source table gdb/lldb will read once the section is registered.
    let cm = compile(COMPUTE_DBG);
    let bytes = cm.debug_line_section();
    assert!(!bytes.is_empty(), "a -g module emits a .debug_line section");

    let prog = svm_wasm::dwarf_line::parse(&bytes).expect("the emitted line program parses");
    // The 1-based file table names the C source (index 0 is the reader's placeholder).
    assert_eq!(prog.files.get(1).map(String::as_str), Some("compute.c"));

    // The non-`end_sequence` rows are exactly the JIT's source ranges: each range's `lo` address
    // carries its source line and file (1-based). Compare as sets, address-keyed.
    let emitted: BTreeSet<(u64, u32)> = prog
        .rows
        .iter()
        .filter(|r| !r.end_sequence)
        .map(|r| (r.address, r.line))
        .collect();
    let expected: BTreeSet<(u64, u32)> = cm.src_ranges().iter().map(|r| (r.lo, r.line)).collect();
    assert_eq!(
        emitted, expected,
        "line-program rows reconstruct the address→line map"
    );

    // Every row's file index resolves to the C source through the program's own table.
    for r in prog.rows.iter().filter(|r| !r.end_sequence) {
        assert_eq!(
            prog.files.get(r.file as usize).map(String::as_str),
            Some("compute.c")
        );
    }

    // A non-`-g` module emits no section.
    let bare = compile("func (i32) -> (i32) {\nblock0(v0: i32):\n  return v0\n}\n");
    assert!(bare.debug_line_section().is_empty());
}

#[test]
fn jit_emits_debug_info_subprograms_that_round_trip() {
    // Stage 2b: the synthesized `.debug_info` + `.debug_abbrev`, parsed back by the project's DWARF
    // info reader, recovers one `DW_TAG_subprogram` per function covering its machine extent — what
    // lets gdb/lldb map a stopped address to its function.
    let cm = compile(COMPUTE_DBG);
    let (info, abbrev) = cm.debug_info_sections();
    assert!(
        !info.is_empty() && !abbrev.is_empty(),
        "a -g module emits .debug_info"
    );

    let parsed = svm_wasm::dwarf_info::parse(&info, &abbrev, &[]).expect("the emitted CU parses");
    assert_eq!(parsed.subs.len(), 1, "one subprogram (the single function)");

    // Its [low_pc, high_pc) is the span of func 0's source-mapped machine ranges.
    let lo = cm.src_ranges().iter().map(|r| r.lo).min().unwrap();
    let hi = cm.src_ranges().iter().map(|r| r.hi).max().unwrap();
    assert_eq!(
        parsed.subs[0].low_pc, lo,
        "subprogram low_pc = function start"
    );
    assert_eq!(
        parsed.subs[0].high_pc, hi,
        "subprogram high_pc = function end"
    );

    // A non-`-g` module emits nothing.
    let bare = compile("func (i32) -> (i32) {\nblock0(v0: i32):\n  return v0\n}\n");
    let (bi, ba) = bare.debug_info_sections();
    assert!(bi.is_empty() && ba.is_empty());
}
