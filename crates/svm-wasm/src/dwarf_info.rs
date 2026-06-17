//! A minimal DWARF v4 (DWARF32) `.debug_info` reader — just enough to recover **source variables**
//! (name, location, type) from a wasm guest's embedded debug info, for the §6 waist (the wasm
//! producer's variable-ingest on-ramp, DEBUGGING.md W4).
//!
//! It parses `.debug_abbrev` (abbreviation tables) + `.debug_info` (the DIE tree) + `.debug_str`
//! (string pool), extracting, per `DW_TAG_subprogram`: its PC range, its frame base (a wasm local,
//! from `DW_OP_WASM_location`), and its `DW_TAG_formal_parameter` / `DW_TAG_variable` children —
//! each a `(name, DW_OP_fbreg offset, type DIE)`. Base types (`DW_TAG_base_type`) are collected by
//! DIE offset so the caller can resolve `DW_AT_type`.
//!
//! Best-effort: any malformed/unsupported input yields `None` and the caller ships no variables
//! (the debug section is strippable / untrusted-for-escape, §2a). DWARF64, v5's reorganized tables,
//! and location forms other than the single `DW_OP_fbreg` / `DW_OP_WASM_location` clang `-O0` emits
//! are out of scope (such a variable is simply dropped).

use std::collections::BTreeMap;

// DWARF constants (the subset this reader needs).
mod tag {
    pub const SUBPROGRAM: u64 = 0x2e;
    pub const FORMAL_PARAMETER: u64 = 0x05;
    pub const VARIABLE: u64 = 0x34;
    pub const BASE_TYPE: u64 = 0x24;
}
mod at {
    pub const NAME: u64 = 0x03;
    pub const BYTE_SIZE: u64 = 0x0b;
    pub const ENCODING: u64 = 0x3e;
    pub const LOW_PC: u64 = 0x11;
    pub const HIGH_PC: u64 = 0x12;
    pub const FRAME_BASE: u64 = 0x40;
    pub const LOCATION: u64 = 0x02;
    pub const TYPE: u64 = 0x49;
}
mod form {
    pub const ADDR: u64 = 0x01;
    pub const DATA2: u64 = 0x05;
    pub const DATA4: u64 = 0x06;
    pub const DATA8: u64 = 0x07;
    pub const STRING: u64 = 0x08;
    pub const DATA1: u64 = 0x0b;
    pub const FLAG: u64 = 0x0c;
    pub const SDATA: u64 = 0x0d;
    pub const STRP: u64 = 0x0e;
    pub const UDATA: u64 = 0x0f;
    pub const REF4: u64 = 0x13;
    pub const REF_UDATA: u64 = 0x15;
    pub const SEC_OFFSET: u64 = 0x17;
    pub const EXPRLOC: u64 = 0x18;
    pub const FLAG_PRESENT: u64 = 0x19;
}
const DW_OP_FBREG: u8 = 0x91;
const DW_OP_WASM_LOCATION: u8 = 0xed;

/// A source variable recovered from a `DW_TAG_formal_parameter` / `DW_TAG_variable`.
pub struct DwarfVar {
    pub name: String,
    /// `DW_OP_fbreg` byte offset from the subprogram's frame base.
    pub fbreg: i64,
    /// CU-relative DIE offset of the variable's `DW_AT_type` (resolve via [`DwarfInfo::base_types`]).
    pub type_ref: u32,
}

/// A function's debug info: its PC range, frame-base wasm local, and variables.
pub struct DwarfSub {
    pub low_pc: u64,
    pub high_pc: u64,
    /// The wasm local index holding the frame base (`DW_OP_WASM_location 0x0 <n>`), if expressed so.
    pub frame_base_local: Option<u32>,
    pub vars: Vec<DwarfVar>,
}

/// A `DW_TAG_base_type`: a primitive's render name, DWARF encoding byte, and byte size.
pub struct DwarfBaseType {
    pub name: String,
    pub encoding: u8,
    pub size: u32,
}

/// The decoded subset: subprograms (in `.debug_info` order) + base types keyed by DIE offset.
pub struct DwarfInfo {
    pub subs: Vec<DwarfSub>,
    pub base_types: BTreeMap<u32, DwarfBaseType>,
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
    fn uint(&mut self, n: usize) -> Option<u64> {
        let mut v = 0u64;
        for i in 0..n {
            v |= (self.u8()? as u64) << (8 * i);
        }
        Some(v)
    }
    fn uleb(&mut self) -> Option<u64> {
        let (mut v, mut s) = (0u64, 0u32);
        loop {
            let byte = self.u8()?;
            if s < 64 {
                v |= ((byte & 0x7f) as u64) << s;
            }
            s += 7;
            if byte & 0x80 == 0 {
                return Some(v);
            }
            if s >= 70 {
                return None;
            }
        }
    }
    fn sleb(&mut self) -> Option<i64> {
        let (mut v, mut s) = (0i64, 0u32);
        loop {
            let byte = self.u8()?;
            if s < 64 {
                v |= ((byte & 0x7f) as i64) << s;
            }
            s += 7;
            if byte & 0x80 == 0 {
                if s < 64 && byte & 0x40 != 0 {
                    v |= -1i64 << s;
                }
                return Some(v);
            }
            if s >= 70 {
                return None;
            }
        }
    }
    fn take(&mut self, n: usize) -> Option<&[u8]> {
        let s = self.b.get(self.p..self.p + n)?;
        self.p += n;
        Some(s)
    }
}

/// A `(tag, has_children, attrs)` abbreviation declaration.
struct Abbrev {
    tag: u64,
    has_children: bool,
    attrs: Vec<(u64, u64)>, // (attribute, form)
}

/// Parse the abbreviation table at `offset` into a code→declaration map.
fn parse_abbrev(abbrev: &[u8], offset: usize) -> Option<BTreeMap<u64, Abbrev>> {
    let mut c = Cur {
        b: abbrev,
        p: offset,
    };
    let mut table = BTreeMap::new();
    loop {
        let code = c.uleb()?;
        if code == 0 {
            break; // end of this table
        }
        let tag = c.uleb()?;
        let has_children = c.u8()? != 0;
        let mut attrs = Vec::new();
        loop {
            let a = c.uleb()?;
            let f = c.uleb()?;
            if a == 0 && f == 0 {
                break;
            }
            attrs.push((a, f));
        }
        table.insert(
            code,
            Abbrev {
                tag,
                has_children,
                attrs,
            },
        );
    }
    Some(table)
}

/// A read attribute value (only the variants this reader inspects are distinguished; values of
/// forms we never look at are folded into `U`/`Flag` just to advance the cursor).
enum Val {
    U(u64),
    Str(String),
    /// A `DW_FORM_exprloc` block (a DWARF location expression).
    Expr(Vec<u8>),
    Flag,
}

/// Read one attribute `form`, returning its value and advancing the cursor.
fn read_form(c: &mut Cur, str_sec: &[u8], form: u64, addr_size: usize) -> Option<Val> {
    Some(match form {
        form::ADDR => Val::U(c.uint(addr_size)?),
        form::DATA1 | form::FLAG => Val::U(c.uint(1)?),
        form::DATA2 => Val::U(c.uint(2)?),
        form::DATA4 | form::REF4 | form::SEC_OFFSET => Val::U(c.uint(4)?),
        form::DATA8 => Val::U(c.uint(8)?),
        form::UDATA | form::REF_UDATA => Val::U(c.uleb()?),
        form::SDATA => Val::U(c.sleb()? as u64), // never inspected; just advance
        form::FLAG_PRESENT => Val::Flag,
        form::STRP => {
            let off = c.uint(4)? as usize;
            Val::Str(str_at(str_sec, off)?)
        }
        form::STRING => {
            let start = c.p;
            while *c.b.get(c.p)? != 0 {
                c.p += 1;
            }
            let s = String::from_utf8_lossy(&c.b[start..c.p]).into_owned();
            c.p += 1;
            Val::Str(s)
        }
        form::EXPRLOC => {
            let n = c.uleb()? as usize;
            Val::Expr(c.take(n)?.to_vec())
        }
        _ => return None, // an unsupported form — give up (best-effort)
    })
}

/// A NUL-terminated string from `.debug_str` at `offset`.
fn str_at(str_sec: &[u8], offset: usize) -> Option<String> {
    let rest = str_sec.get(offset..)?;
    let end = rest.iter().position(|&b| b == 0).unwrap_or(rest.len());
    Some(String::from_utf8_lossy(&rest[..end]).into_owned())
}

/// `DW_OP_fbreg <sleb>` → the offset (the only location form we ingest).
fn fbreg_offset(expr: &[u8]) -> Option<i64> {
    let mut c = Cur { b: expr, p: 0 };
    (c.u8()? == DW_OP_FBREG).then_some(())?;
    c.sleb()
}

/// `DW_OP_WASM_location 0x0 <local>` → the wasm local index (a frame pointer); else `None`.
fn frame_base_local(expr: &[u8]) -> Option<u32> {
    let mut c = Cur { b: expr, p: 0 };
    (c.u8()? == DW_OP_WASM_LOCATION).then_some(())?;
    (c.u8()? == 0x00).then_some(())?; // 0 = local
    Some(c.uleb()? as u32)
}

/// Parse `.debug_info` (with `.debug_abbrev` + `.debug_str`), extracting subprograms + base types.
pub fn parse(info: &[u8], abbrev: &[u8], str_sec: &[u8]) -> Option<DwarfInfo> {
    let mut c = Cur { b: info, p: 0 };
    // CU header (DWARF32 v2–4).
    let unit_len = c.uint(4)? as usize;
    if unit_len == 0xffff_ffff {
        return None; // DWARF64
    }
    let unit_end = c.p.checked_add(unit_len)?;
    if unit_end > info.len() {
        return None;
    }
    let version = c.uint(2)?;
    if !(2..=4).contains(&version) {
        return None; // v5 moves abbrev_offset/addr_size and reorganizes forms
    }
    let abbrev_off = c.uint(4)? as usize;
    let addr_size = c.u8()? as usize;
    if addr_size == 0 || addr_size > 8 {
        return None;
    }
    let abbrevs = parse_abbrev(abbrev, abbrev_off)?;

    let mut subs: Vec<DwarfSub> = Vec::new();
    let mut base_types: BTreeMap<u32, DwarfBaseType> = BTreeMap::new();
    let mut cur_sub: Option<usize> = None; // index into `subs` while walking its children

    while c.p < unit_end {
        let die_off = c.p as u32;
        let code = c.uleb()?;
        if code == 0 {
            // A null DIE ends a sibling chain; the next non-null returns to the parent level. We
            // only nest one level (subprogram → vars), so a null closes the current subprogram.
            cur_sub = None;
            continue;
        }
        let ab = abbrevs.get(&code)?;

        // Read every attribute (to advance), capturing the ones we care about.
        let (mut name, mut ty, mut byte_size, mut encoding) = (None, None, None, None);
        let (mut low_pc, mut high_pc, mut frame_base, mut location) = (None, None, None, None);
        for &(attr, f) in &ab.attrs {
            let v = read_form(&mut c, str_sec, f, addr_size)?;
            match (attr, v) {
                (at::NAME, Val::Str(s)) => name = Some(s),
                (at::TYPE, Val::U(n)) => ty = Some(n as u32),
                (at::BYTE_SIZE, Val::U(n)) => byte_size = Some(n as u32),
                (at::ENCODING, Val::U(n)) => encoding = Some(n as u8),
                (at::LOW_PC, Val::U(n)) => low_pc = Some(n),
                (at::HIGH_PC, Val::U(n)) => high_pc = Some(n),
                (at::FRAME_BASE, Val::Expr(e)) => frame_base = Some(e),
                (at::LOCATION, Val::Expr(e)) => location = Some(e),
                _ => {}
            }
        }

        match ab.tag {
            tag::SUBPROGRAM => {
                let low = low_pc.unwrap_or(0);
                subs.push(DwarfSub {
                    low_pc: low,
                    // `DW_AT_high_pc` is an offset from low_pc when it's a constant (DWARF4).
                    high_pc: low + high_pc.unwrap_or(0),
                    frame_base_local: frame_base.as_deref().and_then(frame_base_local),
                    vars: Vec::new(),
                });
                cur_sub = if ab.has_children {
                    Some(subs.len() - 1)
                } else {
                    None
                };
            }
            tag::FORMAL_PARAMETER | tag::VARIABLE => {
                if let (Some(si), Some(name), Some(loc), Some(ty)) = (cur_sub, name, location, ty) {
                    if let Some(fbreg) = fbreg_offset(&loc) {
                        subs[si].vars.push(DwarfVar {
                            name,
                            fbreg,
                            type_ref: ty,
                        });
                    }
                }
            }
            tag::BASE_TYPE => {
                if let (Some(name), Some(size)) = (name, byte_size) {
                    base_types.insert(
                        die_off,
                        DwarfBaseType {
                            name,
                            encoding: encoding.unwrap_or(0),
                            size,
                        },
                    );
                }
            }
            _ => {
                // A DIE with children we don't model (e.g. a lexical block) would break the
                // one-level subprogram nesting; clang `-O0` doesn't emit them between a subprogram
                // and its top-level vars, so this reader stays flat (deeper nesting ⇒ those vars
                // are simply not attached).
            }
        }
    }

    Some(DwarfInfo { subs, base_types })
}
