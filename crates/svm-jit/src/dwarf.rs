//! Hand-rolled DWARF **emitter** for the W5 JIT/DWARF tier (DEBUGGING.md §7, Stage 2) — the inverse
//! of `svm-wasm`'s `dwarf_line`/`dwarf_info` readers. It synthesizes DWARF sections from the JIT's
//! finalized machine-address → source map ([`crate::SrcRange`], Stage 1) so gdb/lldb can resolve
//! JIT'd guest addresses to source lines. DWARF v4, DWARF32, 8-byte addresses.
//!
//! Strippable host-side tooling, untrusted-for-escape (§2a): a malformed section mis-renders in the
//! debugger, never affects the running guest. Hand-rolled (no `gimli`) to match the parsers' ethos
//! and because only a tiny, fixed subset of the format is needed.

use crate::SrcRange;

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

// DWARF tag / attribute / form constants for the minimal `.debug_info` we emit (Stage 2b).
const DW_TAG_COMPILE_UNIT: u64 = 0x11;
const DW_TAG_SUBPROGRAM: u64 = 0x2e;
const DW_AT_NAME: u64 = 0x03;
const DW_AT_STMT_LIST: u64 = 0x10;
const DW_AT_LOW_PC: u64 = 0x11;
const DW_AT_HIGH_PC: u64 = 0x12;
const DW_FORM_ADDR: u64 = 0x01;
const DW_FORM_DATA8: u64 = 0x07;
const DW_FORM_STRING: u64 = 0x08;
const DW_FORM_SEC_OFFSET: u64 = 0x17;

/// Emit a minimal `.debug_info` + `.debug_abbrev` pair (Stage 2b): one compile-unit DIE whose
/// children are a `DW_TAG_subprogram` per function — `(name, low_pc, high_pc)`, where `high_pc` is
/// the DWARF4 *offset* form (`hi - lo`). `funcs` is `(func index, lo, hi)` machine ranges. This is
/// what lets gdb/lldb map a stopped address to its function (and, with `.debug_line`, its source).
pub fn debug_info(funcs: &[(u32, u64, u64)]) -> (Vec<u8>, Vec<u8>) {
    // `.debug_abbrev`: code 1 = compile_unit (children), code 2 = subprogram (no children).
    let mut abbrev = Vec::new();
    uleb(&mut abbrev, 1);
    uleb(&mut abbrev, DW_TAG_COMPILE_UNIT);
    abbrev.push(1); // DW_CHILDREN_yes
    uleb(&mut abbrev, DW_AT_NAME);
    uleb(&mut abbrev, DW_FORM_STRING);
    // `DW_AT_stmt_list` points gdb at this CU's `.debug_line` program (offset 0 — the JIT gives each
    // module its own ELF with a single line program). Without it gdb loads the function but no source
    // lines, so a `break file.c:N` never binds.
    uleb(&mut abbrev, DW_AT_STMT_LIST);
    uleb(&mut abbrev, DW_FORM_SEC_OFFSET);
    uleb(&mut abbrev, 0);
    uleb(&mut abbrev, 0); // end of code-1 attrs
    uleb(&mut abbrev, 2);
    uleb(&mut abbrev, DW_TAG_SUBPROGRAM);
    abbrev.push(0); // DW_CHILDREN_no
    uleb(&mut abbrev, DW_AT_NAME);
    uleb(&mut abbrev, DW_FORM_STRING);
    uleb(&mut abbrev, DW_AT_LOW_PC);
    uleb(&mut abbrev, DW_FORM_ADDR);
    uleb(&mut abbrev, DW_AT_HIGH_PC);
    uleb(&mut abbrev, DW_FORM_DATA8);
    uleb(&mut abbrev, 0);
    uleb(&mut abbrev, 0); // end of code-2 attrs
    abbrev.push(0); // end of the abbrev table

    // `.debug_info` DIE tree: CU DIE (code 1) then a subprogram DIE (code 2) per function, closed
    // by a null DIE.
    let mut dies = Vec::new();
    uleb(&mut dies, 1); // CU DIE
    dies.extend_from_slice(b"svm-jit\0"); // DW_AT_name
    dies.extend_from_slice(&0u32.to_le_bytes()); // DW_AT_stmt_list → .debug_line offset 0
    for &(func, lo, hi) in funcs {
        uleb(&mut dies, 2); // subprogram DIE
        dies.extend_from_slice(format!("fn{func}\0").as_bytes()); // DW_AT_name (synthesized)
        dies.extend_from_slice(&lo.to_le_bytes()); // DW_AT_low_pc (8-byte address)
        dies.extend_from_slice(&hi.saturating_sub(lo).to_le_bytes()); // DW_AT_high_pc (offset)
    }
    uleb(&mut dies, 0); // end the CU's children

    let mut info = Vec::new();
    let unit_len = 2 /* version */ + 4 /* abbrev_offset */ + 1 /* addr_size */ + dies.len();
    info.extend_from_slice(&(unit_len as u32).to_le_bytes()); // unit_length (DWARF32)
    info.extend_from_slice(&4u16.to_le_bytes()); // version
    info.extend_from_slice(&0u32.to_le_bytes()); // debug_abbrev_offset (table starts at 0)
    info.push(8); // address_size
    info.extend_from_slice(&dies);
    (info, abbrev)
}
