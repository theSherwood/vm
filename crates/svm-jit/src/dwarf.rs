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
