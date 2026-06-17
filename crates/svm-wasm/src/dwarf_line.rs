//! A minimal DWARF (v2–v4, DWARF32) `.debug_line` reader — just enough to recover the
//! `(address → file, line, column)` rows that map wasm code offsets to source, for the §6
//! debug-info waist (DEBUGGING.md W4, the wasm producer on-ramp).
//!
//! It is deliberately small and **best-effort**: any malformed/unsupported input yields `None`, and
//! the caller simply ships no debug info — the section is strippable and untrusted-for-escape (§2a),
//! never load-bearing. DWARF64, the v5 directory/file format, and `max_ops_per_instruction > 1`
//! (irrelevant for wasm) are out of scope. Addresses are wasm code-section-relative offsets (the
//! convention `clang`/`wasm-ld` emit), which the caller matches against per-operator offsets.

/// One row of the line-number program (a `(file, line, col)` at a code `address`).
pub struct LineRow {
    pub address: u64,
    pub file: u64,
    pub line: u32,
    pub col: u32,
    /// A `DW_LNE_end_sequence` marker (no code maps here — the caller skips it).
    pub end_sequence: bool,
}

/// A decoded line program: the 1-based file table (index 0 is an unused placeholder) and its rows.
pub struct LineProgram {
    pub files: Vec<String>,
    pub rows: Vec<LineRow>,
}

struct Cur<'a> {
    b: &'a [u8],
    p: usize,
}

impl Cur<'_> {
    fn u8(&mut self) -> Option<u8> {
        let v = *self.b.get(self.p)?;
        self.p += 1;
        Some(v)
    }
    fn u16(&mut self) -> Option<u16> {
        let v = u16::from_le_bytes([*self.b.get(self.p)?, *self.b.get(self.p + 1)?]);
        self.p += 2;
        Some(v)
    }
    fn u32(&mut self) -> Option<u32> {
        let mut a = [0u8; 4];
        a.copy_from_slice(self.b.get(self.p..self.p + 4)?);
        self.p += 4;
        Some(u32::from_le_bytes(a))
    }
    fn uleb(&mut self) -> Option<u64> {
        let mut v = 0u64;
        let mut shift = 0;
        loop {
            let byte = self.u8()?;
            if shift < 64 {
                v |= ((byte & 0x7f) as u64) << shift;
            }
            shift += 7;
            if byte & 0x80 == 0 {
                return Some(v);
            }
            if shift >= 70 {
                return None; // runaway
            }
        }
    }
    fn sleb(&mut self) -> Option<i64> {
        let mut v = 0i64;
        let mut shift = 0u32;
        loop {
            let byte = self.u8()?;
            if shift < 64 {
                v |= ((byte & 0x7f) as i64) << shift;
            }
            shift += 7;
            if byte & 0x80 == 0 {
                if shift < 64 && byte & 0x40 != 0 {
                    v |= -1i64 << shift; // sign-extend
                }
                return Some(v);
            }
            if shift >= 70 {
                return None;
            }
        }
    }
    /// A NUL-terminated string (UTF-8 lossy — names are tooling, not load-bearing).
    fn cstr(&mut self) -> Option<String> {
        let start = self.p;
        while *self.b.get(self.p)? != 0 {
            self.p += 1;
        }
        let s = String::from_utf8_lossy(&self.b[start..self.p]).into_owned();
        self.p += 1; // consume the NUL
        Some(s)
    }
}

/// Parse a `.debug_line` section, returning its file table + line rows, or `None` if malformed or
/// using a feature outside the supported subset.
pub fn parse(data: &[u8]) -> Option<LineProgram> {
    let mut c = Cur { b: data, p: 0 };

    // Unit header (DWARF32 only).
    let unit_len = c.u32()? as usize;
    if unit_len == 0xffff_ffff {
        return None; // DWARF64
    }
    let unit_end = c.p.checked_add(unit_len)?;
    if unit_end > data.len() {
        return None;
    }
    let version = c.u16()?;
    if !(2..=4).contains(&version) {
        return None; // v5 reorganizes the dir/file tables; out of scope
    }
    let header_len = c.u32()? as usize;
    let program_start = c.p.checked_add(header_len)?;
    let min_inst_len = c.u8()? as u64;
    if version >= 4 {
        let _max_ops = c.u8()?; // assumed 1 for wasm
    }
    let default_is_stmt = c.u8()? != 0;
    let _ = default_is_stmt;
    let line_base = c.u8()? as i8 as i64;
    let line_range = c.u8()? as i64;
    if line_range == 0 {
        return None;
    }
    let opcode_base = c.u8()?;
    let mut std_lens = Vec::new();
    for _ in 1..opcode_base {
        std_lens.push(c.u8()?);
    }
    // include_directories: NUL-terminated strings, ended by an empty string. (We don't need them.)
    loop {
        if c.cstr()?.is_empty() {
            break;
        }
    }
    // file_names: {name, dir_index, mtime, length}, ended by an empty name. 1-based.
    let mut files = vec![String::new()];
    loop {
        let name = c.cstr()?;
        if name.is_empty() {
            break;
        }
        let _dir = c.uleb()?;
        let _mtime = c.uleb()?;
        let _len = c.uleb()?;
        files.push(name);
    }

    // The line-number program proper starts at `program_start` (after the header).
    if program_start > unit_end {
        return None;
    }
    c.p = program_start;

    let mut rows = Vec::new();
    let mut address = 0u64;
    let mut file = 1u64;
    let mut line = 1i64;
    let mut col = 0u64;
    let emit = |rows: &mut Vec<LineRow>, address, file, line: i64, col, end_sequence| {
        rows.push(LineRow {
            address,
            file,
            line: line.max(0) as u32,
            col: col as u32,
            end_sequence,
        });
    };

    while c.p < unit_end {
        let opcode = c.u8()?;
        if opcode == 0 {
            // Extended opcode: length, then a sub-opcode.
            let len = c.uleb()? as usize;
            let ext_end = c.p.checked_add(len)?;
            if ext_end > unit_end || len == 0 {
                return None;
            }
            let sub = c.u8()?;
            match sub {
                1 => {
                    // DW_LNE_end_sequence
                    emit(&mut rows, address, file, line, col, true);
                    address = 0;
                    file = 1;
                    line = 1;
                    col = 0;
                }
                2 => {
                    // DW_LNE_set_address: the remaining bytes, little-endian.
                    let n = ext_end - c.p;
                    let mut a = 0u64;
                    for i in 0..n {
                        a |= (c.u8()? as u64) << (8 * i.min(7));
                    }
                    address = a;
                }
                _ => {} // ignored extended opcode (e.g. set_discriminator)
            }
            c.p = ext_end; // skip any unread operand bytes
        } else if opcode < opcode_base {
            match opcode {
                1 => emit(&mut rows, address, file, line, col, false), // DW_LNS_copy
                2 => address += min_inst_len * c.uleb()?,              // advance_pc
                3 => line += c.sleb()?,                                // advance_line
                4 => file = c.uleb()?,                                 // set_file
                5 => col = c.uleb()?,                                  // set_column
                6 => {}                                                // negate_stmt
                7 => {}                                                // set_basic_block
                8 => {
                    // const_add_pc: advance by the address part of special opcode 255.
                    let adjusted = 255i64 - opcode_base as i64;
                    address += min_inst_len * (adjusted / line_range) as u64;
                }
                9 => address += c.u16()? as u64, // fixed_advance_pc (no min_inst_len scaling)
                10 | 11 => {}                    // set_prologue_end / set_epilogue_begin
                12 => {
                    c.uleb()?; // set_isa
                }
                _ => {
                    // Unknown standard opcode: skip its declared ULEB operands.
                    let nargs = std_lens.get((opcode - 1) as usize).copied().unwrap_or(0);
                    for _ in 0..nargs {
                        c.uleb()?;
                    }
                }
            }
        } else {
            // Special opcode: advance address + line, then emit a row.
            let adjusted = (opcode - opcode_base) as i64;
            address += min_inst_len * (adjusted / line_range) as u64;
            line += line_base + (adjusted % line_range);
            emit(&mut rows, address, file, line, col, false);
        }
    }

    Some(LineProgram { files, rows })
}
