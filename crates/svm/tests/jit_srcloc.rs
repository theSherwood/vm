//! W5 JIT/DWARF tier, Stages 0–2 (DEBUGGING.md §7): the JIT threads each op's §6 `debug.loc` into a
//! Cranelift `SourceLoc`, after `finalize` builds a finalized-machine-address → source map that
//! `CompiledModule::symbolize` resolves (Stages 0–1), and synthesizes a DWARF `.debug_line` section
//! over that map (Stage 2) — round-tripped here through the existing `svm_wasm::dwarf_line` reader,
//! no debugger needed. Host-side tooling, off the runtime path (§2a).

use std::collections::BTreeSet;

use svm_ir::DEFAULT_RESERVED_LOG2;
use svm_jit::{CompiledModule, Quota, VarMachineLoc, INERT_CAP_THUNK};
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

#[test]
fn jit_wraps_dwarf_in_an_elf_whose_sections_round_trip() {
    // Stage 2c: the in-memory ELF the GDB JIT interface hands gdb embeds the synthesized DWARF and
    // a `.text` section pointing at the live code. Re-parse the ELF, pull the `.debug_*` sections
    // back out, and round-trip them through the readers — the same DWARF gdb will read, now proven
    // to survive the ELF wrapper. No debugger needed.
    let cm = compile(COMPUTE_DBG);
    let elf = cm.elf_object();
    assert!(!elf.is_empty(), "a -g module produces an ELF object");
    assert_eq!(&elf[0..4], b"\x7fELF", "it is an ELF");
    assert_eq!(elf[4], 2, "ELFCLASS64");

    // `.debug_line` extracted from the ELF reconstructs the same address→line map as the raw
    // section (so the wrapper preserved it byte-for-byte and the offsets are sound).
    let line = svm_jit::gdb::elf_section(&elf, ".debug_line").expect(".debug_line in the ELF");
    assert_eq!(
        line,
        cm.debug_line_section(),
        "the embedded .debug_line is the section verbatim"
    );
    let prog = svm_wasm::dwarf_line::parse(line).expect("embedded line program parses");
    let from_elf: BTreeSet<(u64, u32)> = prog
        .rows
        .iter()
        .filter(|r| !r.end_sequence)
        .map(|r| (r.address, r.line))
        .collect();
    let expected: BTreeSet<(u64, u32)> = cm.src_ranges().iter().map(|r| (r.lo, r.line)).collect();
    assert_eq!(from_elf, expected, "ELF-embedded line map matches the code");

    // `.debug_info` + `.debug_abbrev` round-trip out of the ELF to the same subprogram.
    let info = svm_jit::gdb::elf_section(&elf, ".debug_info").expect(".debug_info in the ELF");
    let abbrev =
        svm_jit::gdb::elf_section(&elf, ".debug_abbrev").expect(".debug_abbrev in the ELF");
    let parsed = svm_wasm::dwarf_info::parse(info, abbrev, &[]).expect("embedded CU parses");
    assert_eq!(
        parsed.subs.len(),
        1,
        "one subprogram survives the ELF wrapper"
    );

    // The DWARF addresses are *real* finalized-code addresses, and the `.text` section's extent
    // (the section gdb maps the DWARF onto) covers them — the link that makes the addresses
    // meaningful to the debugger.
    let lo = cm.src_ranges().iter().map(|r| r.lo).min().unwrap();
    let hi = cm.src_ranges().iter().map(|r| r.hi).max().unwrap();
    assert_eq!(parsed.subs[0].low_pc, lo);
    assert!(lo > 0x1000, "low_pc is a live mapped address, not a stub");

    // A non-`-g` module produces no ELF (nothing to register).
    let bare = compile("func (i32) -> (i32) {\nblock0(v0: i32):\n  return v0\n}\n");
    assert!(bare.elf_object().is_empty());
    let _ = hi;
}

#[test]
fn jit_registers_and_unregisters_with_the_gdb_jit_interface() {
    // Stage 2c: registering with the GDB JIT interface links a code entry onto the descriptor list
    // (action = JIT_REGISTER_FN) and dropping the guard unlinks it (action = JIT_UNREGISTER_FN) —
    // the linked-list/action-flag protocol gdb observes via its breakpoint on
    // `__jit_debug_register_code`. We can't drive a real gdb in CI, so we assert the descriptor
    // state directly.
    let cm = compile(COMPUTE_DBG);
    let (_, before) = svm_jit::gdb::descriptor_state();

    {
        let _reg = cm.register_with_gdb().expect("a -g module registers");
        let (action, during) = svm_jit::gdb::descriptor_state();
        assert_eq!(action, 1, "action_flag = JIT_REGISTER_FN");
        assert_eq!(during, before + 1, "one more object is registered");

        // A second registration nests; both are on the list at once.
        let _reg2 = cm.register_with_gdb().expect("registers again");
        let (_, two) = svm_jit::gdb::descriptor_state();
        assert_eq!(two, before + 2, "registrations stack on the list");
    }

    // Both guards dropped ⇒ both entries unlinked, last action was an unregister.
    let (action, after) = svm_jit::gdb::descriptor_state();
    assert_eq!(after, before, "dropping the guards unregisters the objects");
    assert_eq!(
        action, 2,
        "action_flag = JIT_UNREGISTER_FN after the last drop"
    );

    // A non-`-g` module has nothing to register.
    let bare = compile("func (i32) -> (i32) {\nblock0(v0: i32):\n  return v0\n}\n");
    assert!(bare.register_with_gdb().is_none());
}

const VAR_DBG: &str = r#"
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
debug.type 0 base "int" signed 4
debug.var 0 "a" ssalist 1 0 0 0 "int" 0
debug.var 0 "t" ssalist 1 0 1 2 "int" 0
"#;

#[test]
fn jit_tracks_source_variable_machine_locations() {
    // Stage 3a: the JIT labels the CLIF values backing SSA-resident source variables and reads back
    // Cranelift's `value_labels_ranges`, so each `-g` variable gets the machine ranges over which it
    // lives in a register or CFA-relative slot — the seed for the Stage 3c `DW_AT_location` loclists.
    let cm = compile(VAR_DBG);
    let vars = cm.var_locations();

    // Both declared source variables are present, attributed to function 0.
    let names: BTreeSet<&str> = vars.iter().map(|v| v.name.as_str()).collect();
    assert_eq!(
        names,
        BTreeSet::from(["a", "t"]),
        "both -g vars are tracked"
    );
    assert!(vars.iter().all(|v| v.func == 0));

    // At least one variable resolves to a real machine location (the local `t = a + 1` lives in a
    // register/slot across the multiply that uses it). A variable the optimizer folded away (`a`,
    // consumed immediately into the add) has empty ranges — the faithful `<optimized out>` case.
    assert!(
        vars.iter().any(|v| !v.ranges.is_empty()),
        "a -g variable is tracked to a machine location"
    );

    // Every tracked range is a non-empty machine span in a plausible location, and (sharing the
    // function's finalized base with the source map) sits within the JIT'd code window.
    let code_end = cm.src_ranges().iter().map(|r| r.hi).max().unwrap();
    let code_start = cm.src_ranges().iter().map(|r| r.lo).min().unwrap();
    for v in vars {
        for r in &v.ranges {
            assert!(r.lo < r.hi, "non-empty range for {:?}: {r:?}", v.name);
            assert!(
                r.hi <= code_end + 0x80 && r.lo + 0x80 >= code_start,
                "range {r:?} for {:?} lands in the function's code",
                v.name
            );
            match r.loc {
                VarMachineLoc::Reg(d) => assert!(d < 0x80, "plausible DWARF regnum {d}"),
                VarMachineLoc::CfaOffset(_) => {}
            }
        }
    }
}

#[test]
fn jit_without_debug_vars_tracks_no_variables() {
    // A `-g` module with source *lines* but no `debug.var` tracks no variables, and an ordinary
    // module tracks none either (codegen byte-identical — no `collect_debug_info`, no labels).
    assert!(
        compile(COMPUTE_DBG).var_locations().is_empty(),
        "lines without debug.var ⇒ no tracked vars"
    );
    let bare = compile("func (i32) -> (i32) {\nblock0(v0: i32):\n  return v0\n}\n");
    assert!(bare.var_locations().is_empty());
}

/// A module carrying the four structured `TypeDef` shapes that map to DWARF type DIEs (Stage 3b):
/// a base `int`, a pointer to it, a `[3]` array of it, and a two-field struct of it. Everything
/// references type 0 (`int`), exercising the inter-type `DW_FORM_ref4` fixups.
const TYPES_DBG: &str = r#"
func (i32) -> (i32) {
block0(v0: i32):
  v1 = i32.const 1
  v2 = i32.add v0 v1
  return v2
}

debug.file 0 "types.c"
debug.loc 0 0 1 0 2 7
debug.type 0 base "int" signed 4
debug.type 1 ptr "int *" 0 8
debug.type 2 array "int[3]" 0 3
debug.type 3 agg "Point" 8
debug.field 3 "x" 0 0
debug.field 3 "y" 4 0
"#;

#[test]
fn jit_emits_type_dies_that_round_trip() {
    // Stage 3b: the §6 `TypeDef` graph emitted as `DW_TAG_*_type` DIEs, parsed back by the project's
    // DWARF info reader — every type recovered, and the inter-type references resolve to the right
    // DIE (the `DW_FORM_ref4` fixups landed on the correct offsets). These are the type DIEs the
    // Stage 3c variable DIEs will point `DW_AT_type` at.
    use svm_wasm::dwarf_info::DwarfType;

    let cm = compile(TYPES_DBG);
    let (info, abbrev) = cm.debug_info_sections();
    let parsed = svm_wasm::dwarf_info::parse(&info, &abbrev, &[]).expect("the emitted CU parses");

    // The `int` base type is what every other type references; find its DIE offset.
    let int_off = *parsed
        .types
        .iter()
        .find_map(|(off, t)| {
            matches!(t, DwarfType::Base { name, .. } if name == "int").then_some(off)
        })
        .expect("int base type present");

    // Base: signed, 4 bytes (DW_ATE_signed = 0x05).
    match &parsed.types[&int_off] {
        DwarfType::Base {
            name,
            encoding,
            size,
        } => {
            assert_eq!(name, "int");
            assert_eq!(*encoding, 0x05, "DW_ATE_signed");
            assert_eq!(*size, 4);
        }
        _ => panic!("expected a base type at the int offset"),
    }

    // `int *` — an 8-byte pointer whose pointee resolves back to the `int` DIE.
    assert!(
        parsed.types.values().any(|t| matches!(
            t,
            DwarfType::Pointer { pointee: Some(p), size } if *p == int_off && *size == 8
        )),
        "int* pointer references the int type"
    );

    // `int[3]` — an array of three elements whose element type resolves to `int`.
    assert!(
        parsed.types.values().any(|t| matches!(
            t,
            DwarfType::Array { elem: Some(e), count } if *e == int_off && *count == 3
        )),
        "int[3] array references the int type"
    );

    // `struct Point { int x@0; int y@4; }`, 8 bytes — both members reference the `int` DIE.
    let (kw, size, members) = parsed
        .types
        .values()
        .find_map(|t| match t {
            DwarfType::Aggregate {
                kw,
                name,
                size,
                members,
            } if name == "Point" => Some((*kw, *size, members)),
            _ => None,
        })
        .expect("Point struct present");
    assert_eq!(kw, "struct");
    assert_eq!(size, 8);
    assert_eq!(members.len(), 2);
    assert_eq!((members[0].name.as_str(), members[0].offset), ("x", 0));
    assert_eq!((members[1].name.as_str(), members[1].offset), ("y", 4));
    assert!(
        members.iter().all(|m| m.type_ref == int_off),
        "both members reference the int type"
    );

    // The 2b subprogram still parses alongside the types.
    assert_eq!(parsed.subs.len(), 1, "the function subprogram survives");

    // A non-`-g` module emits nothing.
    let bare = compile("func (i32) -> (i32) {\nblock0(v0: i32):\n  return v0\n}\n");
    let (bi, ba) = bare.debug_info_sections();
    assert!(bi.is_empty() && ba.is_empty());
}

#[test]
fn jit_emits_variable_loclists_matching_the_value_locations() {
    // Stage 3c: each tracked source variable becomes a `DW_TAG_variable` whose `DW_AT_location` is a
    // `.debug_loc` list of `DW_OP_reg{N}` entries built from the Stage 3a machine ranges. The reader
    // is clang/wasm-shaped (fbreg/addr only), so we validate the loclist bytes directly — that they
    // encode exactly what `var_locations()` resolved — and confirm the DIE tree still parses. The
    // real `print x` check is the `gdb_attach` example under gdb (DEBUGGING.md §7, 3d).
    let cm = compile(VAR_DBG);
    let loc = cm.debug_loc_section();
    assert!(
        !loc.is_empty(),
        "a register-resident var produces a .debug_loc list"
    );

    // Exactly one location list — only `t` is register-resident; `a` was optimized out (empty
    // ranges ⇒ no list, gdb shows `<optimized out>`). Count the base-address-selection sentinels.
    let base_sel = u64::MAX.to_le_bytes();
    let lists = loc.windows(8).filter(|w| *w == base_sel).count();
    assert_eq!(lists, 1, "one loclist for `t`, none for the folded `a`");

    // The loclist carries `t`'s register range verbatim: `lo | hi | exprlen | DW_OP_reg{d}`.
    let t = cm
        .var_locations()
        .iter()
        .find(|v| v.name == "t")
        .expect("t is tracked");
    let r = *t
        .ranges
        .iter()
        .find(|r| matches!(r.loc, VarMachineLoc::Reg(_)))
        .expect("t has a register range");
    let VarMachineLoc::Reg(d) = r.loc else {
        unreachable!()
    };
    let expr: Vec<u8> = if d < 32 {
        vec![0x50 + d as u8] // DW_OP_reg0 + d
    } else {
        let mut e = vec![0x90u8]; // DW_OP_regx <uleb>
        let mut v = d as u64;
        loop {
            let mut b = (v & 0x7f) as u8;
            v >>= 7;
            if v != 0 {
                b |= 0x80;
            }
            e.push(b);
            if v == 0 {
                break;
            }
        }
        e
    };
    let mut entry = Vec::new();
    entry.extend_from_slice(&r.lo.to_le_bytes());
    entry.extend_from_slice(&r.hi.to_le_bytes());
    entry.extend_from_slice(&(expr.len() as u16).to_le_bytes());
    entry.extend_from_slice(&expr);
    assert!(
        loc.windows(entry.len()).any(|w| w == entry.as_slice()),
        "the loclist encodes t's [lo,hi) DW_OP_reg{d} entry from the value-location map"
    );

    // The `.debug_info` still parses with the variable DIE as a subprogram child (the per-subprogram
    // null termination is intact).
    let (info, abbrev) = cm.debug_info_sections();
    let parsed = svm_wasm::dwarf_info::parse(&info, &abbrev, &[]).expect("the CU parses");
    assert_eq!(
        parsed.subs.len(),
        1,
        "the subprogram survives the var children"
    );

    // A `-g` module with no register-resident vars (COMPUTE_DBG has lines but no `debug.var`)
    // produces no location lists.
    assert!(compile(COMPUTE_DBG).debug_loc_section().is_empty());
}

#[test]
fn jit_emits_debug_frame_cfi_for_unwinding() {
    // Stage 4a: a `.debug_frame` whose CIE carries the frame-pointer unwind rules and whose single
    // FDE covers the function — what lets gdb unwind a stopped JIT frame (`bt`) and compute the CFA
    // the subprograms' `DW_AT_frame_base` refers to. Validated structurally here (the readers have no
    // CFI parser); the real `bt` check is the `gdb_attach` example under gdb (DEBUGGING.md §7, 4a).
    let cm = compile(VAR_DBG);
    let f = cm.debug_frame_section();
    assert!(!f.is_empty(), "a -g module emits .debug_frame");

    // CIE: length, CIE_id sentinel, v4, empty augmentation, 8-byte addresses.
    let cie_len = u32::from_le_bytes(f[0..4].try_into().unwrap()) as usize;
    assert_eq!(&f[4..8], &[0xff, 0xff, 0xff, 0xff], "CIE_id sentinel");
    assert_eq!(f[8], 4, "CIE version 4");
    assert_eq!(f[9], 0, "empty augmentation");
    assert_eq!(f[10], 8, "address_size");
    assert_eq!(f[11], 0, "segment selector size");
    let cie_end = 4 + cie_len;
    // The CIE must define the frame-pointer CFA: DW_CFA_def_cfa r6 (rbp), offset 16.
    assert!(
        f[..cie_end].windows(3).any(|w| w == [0x0c, 0x06, 0x10]),
        "CIE defines CFA = rbp + 16"
    );

    // One FDE (one function), referencing the CIE at offset 0 and covering the function's extent.
    let fde = &f[cie_end..];
    let fde_len = u32::from_le_bytes(fde[0..4].try_into().unwrap()) as usize;
    assert_eq!(
        &fde[4..8],
        &[0, 0, 0, 0],
        "FDE CIE_pointer → CIE at offset 0"
    );
    let lo = u64::from_le_bytes(fde[8..16].try_into().unwrap());
    let range = u64::from_le_bytes(fde[16..24].try_into().unwrap());
    let exp_lo = cm.src_ranges().iter().map(|r| r.lo).min().unwrap();
    let exp_hi = cm.src_ranges().iter().map(|r| r.hi).max().unwrap();
    assert_eq!(lo, exp_lo, "FDE starts at the function's low_pc");
    assert_eq!(range, exp_hi - exp_lo, "FDE spans the function extent");
    assert_eq!(cie_end + 4 + fde_len, f.len(), "exactly one CIE + one FDE");

    // A non-`-g` module emits no frame info.
    let bare = compile("func (i32) -> (i32) {\nblock0(v0: i32):\n  return v0\n}\n");
    assert!(bare.debug_frame_section().is_empty());
}

/// A function that forces register spills (Stage 4b): thirteen `call` results kept live **across a
/// barrier `call`**, so only the callee-saved registers can carry them across — far fewer than
/// thirteen on every target (≈5 on x86-64, ≈10 on aarch64) — and the regalloc must park the rest in
/// CFA-relative stack slots, yielding `VarMachineLoc::CfaOffset` variable ranges (`DW_OP_fbreg`, not
/// `DW_OP_reg`). All thirteen are labeled, so at least one is guaranteed to spill regardless of
/// target register file. `func 1` is the trivial callee. Calls are used (not arithmetic on the param)
/// because the optimizer rematerializes/folds plain computed values, dropping their value labels.
const SPILL_DBG: &str = r#"
func (i32) -> (i32) {
block0(v0: i32):
  v1 = call 1(v0)
  v2 = call 1(v0)
  v3 = call 1(v0)
  v4 = call 1(v0)
  v5 = call 1(v0)
  v6 = call 1(v0)
  v7 = call 1(v0)
  v8 = call 1(v0)
  v9 = call 1(v0)
  v10 = call 1(v0)
  v11 = call 1(v0)
  v12 = call 1(v0)
  v13 = call 1(v0)
  v14 = call 1(v0)
  v15 = i32.add v1 v2
  v16 = i32.add v15 v3
  v17 = i32.add v16 v4
  v18 = i32.add v17 v5
  v19 = i32.add v18 v6
  v20 = i32.add v19 v7
  v21 = i32.add v20 v8
  v22 = i32.add v21 v9
  v23 = i32.add v22 v10
  v24 = i32.add v23 v11
  v25 = i32.add v24 v12
  v26 = i32.add v25 v13
  v27 = i32.add v26 v14
  return v27
}

func (i32) -> (i32) {
block0(v0: i32):
  v1 = i32.const 3
  v2 = i32.mul v0 v1
  return v2
}

debug.file 0 "spill.c"
debug.loc 0 0 1 0 2 5
debug.type 0 base "int" signed 4
debug.var 0 "s1" ssalist 1 0 0 1 "int" 0
debug.var 0 "s2" ssalist 1 0 0 2 "int" 0
debug.var 0 "s3" ssalist 1 0 0 3 "int" 0
debug.var 0 "s4" ssalist 1 0 0 4 "int" 0
debug.var 0 "s5" ssalist 1 0 0 5 "int" 0
debug.var 0 "s6" ssalist 1 0 0 6 "int" 0
debug.var 0 "s7" ssalist 1 0 0 7 "int" 0
debug.var 0 "s8" ssalist 1 0 0 8 "int" 0
debug.var 0 "s9" ssalist 1 0 0 9 "int" 0
debug.var 0 "s10" ssalist 1 0 0 10 "int" 0
debug.var 0 "s11" ssalist 1 0 0 11 "int" 0
debug.var 0 "s12" ssalist 1 0 0 12 "int" 0
debug.var 0 "s13" ssalist 1 0 0 13 "int" 0
"#;

#[test]
fn jit_emits_fbreg_loclists_for_spilled_variables() {
    // Stage 4b: a source variable Cranelift spilled to a CFA-relative stack slot
    // (`VarMachineLoc::CfaOffset`) becomes a `DW_OP_fbreg <off>` location-list entry — resolving
    // against the subprogram's `DW_AT_frame_base = DW_OP_call_frame_cfa` (Stage 4a) — while its
    // register ranges stay `DW_OP_reg`. Validated against real regalloc output: every range's loclist
    // entry is reconstructed and found verbatim in `.debug_loc`. The end-to-end `print` is the
    // `gdb_attach` example under gdb.
    let cm = compile(SPILL_DBG);
    let vars = cm.var_locations();
    let loc = cm.debug_loc_section();

    // The fixture is built to force spills: at least one variable range is a CFA-relative slot.
    assert!(
        vars.iter()
            .flat_map(|v| &v.ranges)
            .any(|r| matches!(r.loc, VarMachineLoc::CfaOffset(_))),
        "the spill fixture yields at least one CfaOffset range"
    );

    // Mirror the emitter's expression encoding so we can find each range's entry in `.debug_loc`.
    let expr = |l: VarMachineLoc| -> Vec<u8> {
        match l {
            VarMachineLoc::Reg(d) if d < 32 => vec![0x50 + d as u8],
            VarMachineLoc::Reg(d) => {
                let mut e = vec![0x90u8];
                let mut v = d as u64;
                loop {
                    let mut b = (v & 0x7f) as u8;
                    v >>= 7;
                    if v != 0 {
                        b |= 0x80;
                    }
                    e.push(b);
                    if v == 0 {
                        break;
                    }
                }
                e
            }
            VarMachineLoc::CfaOffset(off) => {
                let mut e = vec![0x91u8]; // DW_OP_fbreg
                let mut v = off;
                loop {
                    let mut b = (v & 0x7f) as u8;
                    v >>= 7; // arithmetic shift
                    let done = (v == 0 && b & 0x40 == 0) || (v == -1 && b & 0x40 != 0);
                    if !done {
                        b |= 0x80;
                    }
                    e.push(b);
                    if done {
                        break;
                    }
                }
                e
            }
        }
    };

    // Every range of every tracked variable is present in `.debug_loc` as `lo | hi | exprlen | expr`,
    // with `DW_OP_fbreg` for the spilled (CfaOffset) ranges and `DW_OP_reg` for the register ranges.
    let mut saw_fbreg = false;
    for v in vars {
        for r in &v.ranges {
            let e = expr(r.loc);
            if e[0] == 0x91 {
                saw_fbreg = true;
            }
            let mut entry = Vec::new();
            entry.extend_from_slice(&r.lo.to_le_bytes());
            entry.extend_from_slice(&r.hi.to_le_bytes());
            entry.extend_from_slice(&(e.len() as u16).to_le_bytes());
            entry.extend_from_slice(&e);
            assert!(
                loc.windows(entry.len()).any(|w| w == entry.as_slice()),
                "{}'s {:?} range is encoded in .debug_loc",
                v.name,
                r.loc
            );
        }
    }
    assert!(saw_fbreg, "a spilled variable produced a DW_OP_fbreg entry");

    // The `.debug_info` still parses with these richer loclists referenced (only `func 0` carries
    // source lines, so it is the lone subprogram).
    let (info, abbrev) = cm.debug_info_sections();
    let parsed = svm_wasm::dwarf_info::parse(&info, &abbrev, &[]).expect("the CU parses");
    assert_eq!(
        parsed.subs.len(),
        1,
        "the source-mapped function is a subprogram"
    );
}
