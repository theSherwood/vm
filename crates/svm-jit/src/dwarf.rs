//! Hand-rolled DWARF **emitter** for the W5 JIT/DWARF tier (DEBUGGING.md §7, Stage 2) — the inverse
//! of `svm-wasm`'s `dwarf_line`/`dwarf_info` readers. It synthesizes DWARF sections from the JIT's
//! finalized machine-address → source map ([`crate::SrcRange`], Stage 1) so gdb/lldb can resolve
//! JIT'd guest addresses to source lines. DWARF v4, DWARF32, 8-byte addresses.
//!
//! Strippable host-side tooling, untrusted-for-escape (§2a): a malformed section mis-renders in the
//! debugger, never affects the running guest. Hand-rolled (no `gimli`) to match the parsers' ethos
//! and because only a tiny, fixed subset of the format is needed.

use crate::SrcRange;
use svm_ir::{Encoding, TypeDef};

fn uleb(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let mut b = (v & 0x7f) as u8;
        v >>= 7;
        if v != 0 {
            b |= 0x80;
        }
        out.push(b);
        if v == 0 {
            break;
        }
    }
}

fn sleb(out: &mut Vec<u8>, mut v: i64) {
    loop {
        let mut b = (v & 0x7f) as u8;
        v >>= 7; // arithmetic shift
        let done = (v == 0 && b & 0x40 == 0) || (v == -1 && b & 0x40 != 0);
        if !done {
            b |= 0x80;
        }
        out.push(b);
        if done {
            break;
        }
    }
}

/// `DW_LNE_set_address <addr>` — an extended opcode whose operand length selects the 8-byte address
/// size the reader infers from `ext_end - p`.
fn set_address(out: &mut Vec<u8>, addr: u64) {
    out.push(0); // extended-opcode marker
    uleb(out, 1 + 8); // length: the sub-opcode byte + 8 address bytes
    out.push(2); // DW_LNE_set_address
    out.extend_from_slice(&addr.to_le_bytes());
}

/// Emit a `.debug_line` line-number program mapping the finalized machine addresses to source. Each
/// [`SrcRange`] becomes its own self-contained sequence — a row at `lo` carrying its `(file, line,
/// col)`, then `set_address(hi)` + `DW_LNE_end_sequence` to close `[lo, hi)` — so non-contiguous
/// ranges (prologue gaps, separate functions) never bleed one line into the next. `files` is the
/// 0-based source-path table ([`SrcRange::file`] indexes it); DWARF file indices are 1-based.
pub fn debug_line(ranges: &[SrcRange], files: &[String]) -> Vec<u8> {
    // Program header body (everything the `header_length` field covers):
    //   minimum_instruction_length=1, maximum_operations_per_instruction=1 (v4), default_is_stmt=1,
    //   line_base=-5, line_range=14, opcode_base=13 (line_base/line_range unused — we never emit
    //   special opcodes), then standard_opcode_lengths[1..=12], then an empty include_directories.
    let mut hdr = vec![1, 1, 1, (-5i8) as u8, 14, 13];
    hdr.extend_from_slice(&[0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1]); // standard_opcode_lengths[1..=12]
    hdr.push(0); // include_directories: empty (terminator)
    for f in files {
        hdr.extend_from_slice(f.as_bytes());
        hdr.push(0);
        uleb(&mut hdr, 0); // dir index
        uleb(&mut hdr, 0); // mtime
        uleb(&mut hdr, 0); // length
    }
    hdr.push(0); // file_names terminator

    // The line-number program: one independent sequence per range.
    let mut prog = Vec::new();
    for r in ranges {
        set_address(&mut prog, r.lo);
        prog.push(4); // DW_LNS_set_file
        uleb(&mut prog, r.file as u64 + 1); // 1-based
        prog.push(5); // DW_LNS_set_column
        uleb(&mut prog, r.col as u64);
        prog.push(3); // DW_LNS_advance_line (from the initial line == 1)
        sleb(&mut prog, r.line as i64 - 1);
        prog.push(1); // DW_LNS_copy → a row at `lo`
        set_address(&mut prog, r.hi);
        // DW_LNE_end_sequence (extended opcode 1) → closes [lo, hi) and resets the row registers.
        prog.push(0);
        uleb(&mut prog, 1);
        prog.push(1);
    }

    let mut out = Vec::new();
    let unit_len = 2 /* version */ + 4 /* header_length */ + hdr.len() + prog.len();
    out.extend_from_slice(&(unit_len as u32).to_le_bytes()); // unit_length (DWARF32)
    out.extend_from_slice(&4u16.to_le_bytes()); // version
    out.extend_from_slice(&(hdr.len() as u32).to_le_bytes()); // header_length
    out.extend_from_slice(&hdr);
    out.extend_from_slice(&prog);
    out
}

// DWARF tag / attribute / form constants for the `.debug_info` we emit (Stages 2b + 3b).
const DW_TAG_ARRAY_TYPE: u64 = 0x01;
const DW_TAG_MEMBER: u64 = 0x0d;
const DW_TAG_POINTER_TYPE: u64 = 0x0f;
const DW_TAG_COMPILE_UNIT: u64 = 0x11;
const DW_TAG_STRUCTURE_TYPE: u64 = 0x13;
const DW_TAG_SUBRANGE_TYPE: u64 = 0x21;
const DW_TAG_BASE_TYPE: u64 = 0x24;
const DW_TAG_SUBPROGRAM: u64 = 0x2e;
const DW_AT_NAME: u64 = 0x03;
const DW_AT_BYTE_SIZE: u64 = 0x0b;
const DW_AT_STMT_LIST: u64 = 0x10;
const DW_AT_LOW_PC: u64 = 0x11;
const DW_AT_HIGH_PC: u64 = 0x12;
const DW_AT_COUNT: u64 = 0x37;
const DW_AT_DATA_MEMBER_LOCATION: u64 = 0x38;
const DW_AT_ENCODING: u64 = 0x3e;
const DW_AT_TYPE: u64 = 0x49;
const DW_FORM_ADDR: u64 = 0x01;
const DW_FORM_DATA1: u64 = 0x0b;
const DW_FORM_DATA8: u64 = 0x07;
const DW_FORM_STRING: u64 = 0x08;
const DW_FORM_UDATA: u64 = 0x0f;
const DW_FORM_REF4: u64 = 0x13;
const DW_FORM_SEC_OFFSET: u64 = 0x17;

// DWARF base-type encodings (`DW_ATE_*`) — the inverse of `dwarf_info`'s `encoding` byte.
const DW_ATE_BOOLEAN: u8 = 0x02;
const DW_ATE_FLOAT: u8 = 0x04;
const DW_ATE_SIGNED: u8 = 0x05;
const DW_ATE_UNSIGNED: u8 = 0x07;

/// The CU header length (DWARF32 v4): `unit_length(4) + version(2) + debug_abbrev_offset(4) +
/// address_size(1)`. A DIE at byte `p` in the DIE buffer is at CU/section offset `CU_HEADER_LEN + p`
/// — which is what a `DW_FORM_ref4` (CU-relative) must carry to name it.
const CU_HEADER_LEN: usize = 11;

/// Map the §6 neutral [`Encoding`] to its DWARF `DW_ATE_*` byte.
fn dwarf_ate(e: Encoding) -> u8 {
    match e {
        Encoding::Signed => DW_ATE_SIGNED,
        Encoding::Unsigned => DW_ATE_UNSIGNED,
        Encoding::Float => DW_ATE_FLOAT,
        Encoding::Bool => DW_ATE_BOOLEAN,
    }
}

/// Append a NUL-terminated `DW_FORM_string`.
fn dw_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(s.as_bytes());
    out.push(0);
}

// Abbreviation codes for the DIEs we emit. Codes 1–2 (CU, subprogram) are Stage 2b; 3–8 are the
// Stage 3b type DIEs.
const ABBR_CU: u64 = 1;
const ABBR_SUBPROGRAM: u64 = 2;
const ABBR_BASE_TYPE: u64 = 3;
const ABBR_POINTER_TYPE: u64 = 4;
const ABBR_ARRAY_TYPE: u64 = 5;
const ABBR_SUBRANGE_TYPE: u64 = 6;
const ABBR_STRUCTURE_TYPE: u64 = 7;
const ABBR_MEMBER: u64 = 8;

/// Build the `.debug_abbrev` table shared by [`debug_info`]: the CU + subprogram entries (Stage 2b)
/// and the type-DIE entries (Stage 3b). The DIE forms here fix the byte layout `debug_info` writes.
fn abbrev_table() -> Vec<u8> {
    let mut a = Vec::new();
    let mut entry = |code: u64, tag: u64, children: bool, attrs: &[(u64, u64)]| {
        uleb(&mut a, code);
        uleb(&mut a, tag);
        a.push(if children { 1 } else { 0 });
        for &(at, form) in attrs {
            uleb(&mut a, at);
            uleb(&mut a, form);
        }
        uleb(&mut a, 0);
        uleb(&mut a, 0); // end of this entry's attrs
    };
    // CU: name + `DW_AT_stmt_list` (→ `.debug_line` offset 0, so gdb binds source lines).
    entry(
        ABBR_CU,
        DW_TAG_COMPILE_UNIT,
        true,
        &[
            (DW_AT_NAME, DW_FORM_STRING),
            (DW_AT_STMT_LIST, DW_FORM_SEC_OFFSET),
        ],
    );
    // Subprogram: name + low_pc + high_pc (DWARF4 offset form). Children (Stage 3c adds var DIEs).
    entry(
        ABBR_SUBPROGRAM,
        DW_TAG_SUBPROGRAM,
        true,
        &[
            (DW_AT_NAME, DW_FORM_STRING),
            (DW_AT_LOW_PC, DW_FORM_ADDR),
            (DW_AT_HIGH_PC, DW_FORM_DATA8),
        ],
    );
    entry(
        ABBR_BASE_TYPE,
        DW_TAG_BASE_TYPE,
        false,
        &[
            (DW_AT_NAME, DW_FORM_STRING),
            (DW_AT_ENCODING, DW_FORM_DATA1),
            (DW_AT_BYTE_SIZE, DW_FORM_UDATA),
        ],
    );
    entry(
        ABBR_POINTER_TYPE,
        DW_TAG_POINTER_TYPE,
        false,
        &[(DW_AT_BYTE_SIZE, DW_FORM_UDATA), (DW_AT_TYPE, DW_FORM_REF4)],
    );
    entry(
        ABBR_ARRAY_TYPE,
        DW_TAG_ARRAY_TYPE,
        true,
        &[(DW_AT_TYPE, DW_FORM_REF4)],
    );
    entry(
        ABBR_SUBRANGE_TYPE,
        DW_TAG_SUBRANGE_TYPE,
        false,
        &[(DW_AT_COUNT, DW_FORM_UDATA)],
    );
    entry(
        ABBR_STRUCTURE_TYPE,
        DW_TAG_STRUCTURE_TYPE,
        true,
        &[
            (DW_AT_NAME, DW_FORM_STRING),
            (DW_AT_BYTE_SIZE, DW_FORM_UDATA),
        ],
    );
    entry(
        ABBR_MEMBER,
        DW_TAG_MEMBER,
        false,
        &[
            (DW_AT_NAME, DW_FORM_STRING),
            (DW_AT_TYPE, DW_FORM_REF4),
            (DW_AT_DATA_MEMBER_LOCATION, DW_FORM_UDATA),
        ],
    );
    a.push(0); // end of the abbrev table
    a
}

/// Emit `.debug_info` + `.debug_abbrev`: one compile-unit DIE whose children are the §6 `types`
/// graph as `DW_TAG_*_type` DIEs (Stage 3b — the inverse of `dwarf_info`'s type reader) followed by
/// a `DW_TAG_subprogram` per function (Stage 2b: `(name, low_pc, high_pc)`, `high_pc` the DWARF4
/// offset form). `funcs` is `(func index, lo, hi)` machine ranges; `types` is indexed by `TypeId`,
/// and inter-type references (`pointee`/`elem`/field `ty`) become CU-relative `DW_FORM_ref4`s
/// resolved by a fixup pass once every type DIE's offset is known.
pub fn debug_info(funcs: &[(u32, u64, u64)], types: &[TypeDef]) -> (Vec<u8>, Vec<u8>) {
    let abbrev = abbrev_table();

    let mut dies = Vec::new();
    uleb(&mut dies, ABBR_CU); // CU DIE
    dw_str(&mut dies, "svm-jit"); // DW_AT_name
    dies.extend_from_slice(&0u32.to_le_bytes()); // DW_AT_stmt_list → .debug_line offset 0

    // Type DIEs. Record each `TypeId`'s CU offset as we go; `DW_AT_type` references are written as
    // placeholder zeros and patched afterward (a type may reference one defined later).
    let mut type_off = vec![0u32; types.len()];
    let mut fixups: Vec<(usize, u32)> = Vec::new(); // (byte position in `dies`, target TypeId)
    let type_ref = |dies: &mut Vec<u8>, fixups: &mut Vec<(usize, u32)>, target: u32| {
        fixups.push((dies.len(), target));
        dies.extend_from_slice(&0u32.to_le_bytes());
    };
    for (id, t) in types.iter().enumerate() {
        type_off[id] = (CU_HEADER_LEN + dies.len()) as u32;
        match t {
            TypeDef::Base {
                name,
                encoding,
                size,
            } => {
                uleb(&mut dies, ABBR_BASE_TYPE);
                dw_str(&mut dies, name);
                dies.push(dwarf_ate(*encoding));
                uleb(&mut dies, *size as u64);
            }
            TypeDef::Pointer { pointee, size, .. } => {
                uleb(&mut dies, ABBR_POINTER_TYPE);
                uleb(&mut dies, *size as u64);
                type_ref(&mut dies, &mut fixups, *pointee);
            }
            TypeDef::Array { elem, count, .. } => {
                uleb(&mut dies, ABBR_ARRAY_TYPE);
                type_ref(&mut dies, &mut fixups, *elem);
                // The element count rides on a child `DW_TAG_subrange_type`.
                uleb(&mut dies, ABBR_SUBRANGE_TYPE);
                uleb(&mut dies, *count as u64);
                uleb(&mut dies, 0); // close the array's children
            }
            TypeDef::Aggregate { name, size, fields } => {
                uleb(&mut dies, ABBR_STRUCTURE_TYPE);
                dw_str(&mut dies, name);
                uleb(&mut dies, *size as u64);
                for f in fields {
                    uleb(&mut dies, ABBR_MEMBER);
                    dw_str(&mut dies, &f.name);
                    type_ref(&mut dies, &mut fixups, f.ty);
                    uleb(&mut dies, f.offset as u64);
                }
                uleb(&mut dies, 0); // close the struct's children
            }
            // No structure carried — a name + size only (renders as an opaque struct).
            TypeDef::Opaque { name, size } => {
                uleb(&mut dies, ABBR_STRUCTURE_TYPE);
                dw_str(&mut dies, name);
                uleb(&mut dies, *size as u64);
                uleb(&mut dies, 0); // no members
            }
        }
    }

    // Subprogram DIEs.
    for &(func, lo, hi) in funcs {
        uleb(&mut dies, ABBR_SUBPROGRAM);
        dw_str(&mut dies, &format!("fn{func}")); // DW_AT_name (synthesized)
        dies.extend_from_slice(&lo.to_le_bytes()); // DW_AT_low_pc (8-byte address)
        dies.extend_from_slice(&hi.saturating_sub(lo).to_le_bytes()); // DW_AT_high_pc (offset)
    }
    uleb(&mut dies, 0); // end the CU's children

    // Patch every `DW_AT_type` reference now that all type offsets are known (an out-of-range id —
    // there should be none — stays the zero placeholder, which resolves to no type).
    for (pos, target) in fixups {
        if let Some(&off) = type_off.get(target as usize) {
            dies[pos..pos + 4].copy_from_slice(&off.to_le_bytes());
        }
    }

    let mut info = Vec::new();
    let unit_len = 2 /* version */ + 4 /* abbrev_offset */ + 1 /* addr_size */ + dies.len();
    info.extend_from_slice(&(unit_len as u32).to_le_bytes()); // unit_length (DWARF32)
    info.extend_from_slice(&4u16.to_le_bytes()); // version
    info.extend_from_slice(&0u32.to_le_bytes()); // debug_abbrev_offset (table starts at 0)
    info.push(8); // address_size
    info.extend_from_slice(&dies);
    (info, abbrev)
}
