//! Binary encoding for the IR (`DESIGN.md` §3a). LEB128 + block-local indices; no
//! bespoke compression. The design goal is that decode and verify *fuse* into one
//! linear pass; this crate is the decode half (verification lives in `svm-verify`).
//!
//! **Decoding is escape-TCB and untrusted-input-facing:** it must reject malformed
//! input with `Err` and **never panic, never OOM, always terminate** on arbitrary
//! bytes (fuzzed in the `svm` crate). We therefore never pre-allocate from an
//! untrusted count, and we reject counts that cannot fit in the remaining bytes.
#![forbid(unsafe_code)]

use svm_ir::{Block, Func, Inst, Module, Terminator, ValType};

mod tag {
    // Value types.
    pub const T_I32: u8 = 0;
    pub const T_I64: u8 = 1;
    pub const T_F32: u8 = 2;
    pub const T_F64: u8 = 3;

    // Instruction opcodes.
    pub const I32_CONST: u8 = 0x10;
    pub const I64_CONST: u8 = 0x11;
    pub const I32_ADD: u8 = 0x20;
    pub const I64_ADD: u8 = 0x21;

    // Terminator opcodes.
    pub const BR: u8 = 0x40;
    pub const BR_IF: u8 = 0x41;
    pub const RETURN: u8 = 0x42;
}

const MAGIC: [u8; 4] = *b"SVM\x00";
const VERSION: u8 = 1;

/// Why decoding rejected a byte stream.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum DecodeError {
    UnexpectedEof,
    BadMagic,
    BadVersion(u8),
    BadType(u8),
    BadOpcode(u8),
    /// A LEB128-encoded integer did not fit its target width.
    IntTooLarge,
    /// A LEB128 sequence was longer than its target width allows.
    LebOverflow,
    /// A count exceeded the bytes that could possibly satisfy it (anti-OOM/DoS).
    CountTooLarge,
    /// Bytes remained after a complete module was decoded.
    TrailingBytes,
}

// ----------------------------------------------------------------------------
// Encoding
// ----------------------------------------------------------------------------

/// Encode a module to bytes. (Producer-side; not part of the untrusted-input TCB.)
pub fn encode_module(m: &Module) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&MAGIC);
    out.push(VERSION);
    write_uleb(&mut out, m.funcs.len() as u64);
    for f in &m.funcs {
        encode_func(&mut out, f);
    }
    out
}

fn encode_func(out: &mut Vec<u8>, f: &Func) {
    write_types(out, &f.params);
    write_types(out, &f.results);
    write_uleb(out, f.blocks.len() as u64);
    for b in &f.blocks {
        write_types(out, &b.params);
        write_uleb(out, b.insts.len() as u64);
        for inst in &b.insts {
            encode_inst(out, inst);
        }
        encode_term(out, &b.term);
    }
}

fn encode_inst(out: &mut Vec<u8>, inst: &Inst) {
    match inst {
        Inst::I32Const(c) => {
            out.push(tag::I32_CONST);
            write_sleb(out, *c as i64);
        }
        Inst::I64Const(c) => {
            out.push(tag::I64_CONST);
            write_sleb(out, *c);
        }
        Inst::I32Add(a, b) => {
            out.push(tag::I32_ADD);
            write_uleb(out, *a as u64);
            write_uleb(out, *b as u64);
        }
        Inst::I64Add(a, b) => {
            out.push(tag::I64_ADD);
            write_uleb(out, *a as u64);
            write_uleb(out, *b as u64);
        }
    }
}

fn encode_term(out: &mut Vec<u8>, t: &Terminator) {
    match t {
        Terminator::Br { target, args } => {
            out.push(tag::BR);
            write_uleb(out, *target as u64);
            write_idxs(out, args);
        }
        Terminator::BrIf {
            cond,
            then_blk,
            then_args,
            else_blk,
            else_args,
        } => {
            out.push(tag::BR_IF);
            write_uleb(out, *cond as u64);
            write_uleb(out, *then_blk as u64);
            write_idxs(out, then_args);
            write_uleb(out, *else_blk as u64);
            write_idxs(out, else_args);
        }
        Terminator::Return(vals) => {
            out.push(tag::RETURN);
            write_idxs(out, vals);
        }
    }
}

fn write_types(out: &mut Vec<u8>, ts: &[ValType]) {
    write_uleb(out, ts.len() as u64);
    for t in ts {
        out.push(type_tag(*t));
    }
}

fn write_idxs(out: &mut Vec<u8>, idxs: &[u32]) {
    write_uleb(out, idxs.len() as u64);
    for i in idxs {
        write_uleb(out, *i as u64);
    }
}

fn type_tag(t: ValType) -> u8 {
    match t {
        ValType::I32 => tag::T_I32,
        ValType::I64 => tag::T_I64,
        ValType::F32 => tag::T_F32,
        ValType::F64 => tag::T_F64,
    }
}

pub fn write_uleb(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let byte = (v & 0x7f) as u8;
        v >>= 7;
        if v != 0 {
            out.push(byte | 0x80);
        } else {
            out.push(byte);
            break;
        }
    }
}

pub fn write_sleb(out: &mut Vec<u8>, mut v: i64) {
    loop {
        let byte = (v & 0x7f) as u8;
        v >>= 7; // arithmetic shift (sign-extending)
        let sign_bit = byte & 0x40 != 0;
        if (v == 0 && !sign_bit) || (v == -1 && sign_bit) {
            out.push(byte);
            break;
        } else {
            out.push(byte | 0x80);
        }
    }
}

// ----------------------------------------------------------------------------
// Decoding (untrusted-input-facing)
// ----------------------------------------------------------------------------

/// Decode a module from bytes. Rejects malformed input; never panics/OOMs.
pub fn decode_module(bytes: &[u8]) -> Result<Module, DecodeError> {
    let mut c = Cursor::new(bytes);
    if c.take(4)? != MAGIC {
        return Err(DecodeError::BadMagic);
    }
    let v = c.byte()?;
    if v != VERSION {
        return Err(DecodeError::BadVersion(v));
    }
    let nfuncs = c.count()?;
    let mut funcs = Vec::new();
    for _ in 0..nfuncs {
        funcs.push(decode_func(&mut c)?);
    }
    if !c.at_end() {
        return Err(DecodeError::TrailingBytes);
    }
    Ok(Module { funcs })
}

fn decode_func(c: &mut Cursor) -> Result<Func, DecodeError> {
    let params = decode_types(c)?;
    let results = decode_types(c)?;
    let nblocks = c.count()?;
    let mut blocks = Vec::new();
    for _ in 0..nblocks {
        blocks.push(decode_block(c)?);
    }
    Ok(Func {
        params,
        results,
        blocks,
    })
}

fn decode_block(c: &mut Cursor) -> Result<Block, DecodeError> {
    let params = decode_types(c)?;
    let ninsts = c.count()?;
    let mut insts = Vec::new();
    for _ in 0..ninsts {
        insts.push(decode_inst(c)?);
    }
    let term = decode_term(c)?;
    Ok(Block {
        params,
        insts,
        term,
    })
}

fn decode_inst(c: &mut Cursor) -> Result<Inst, DecodeError> {
    let op = c.byte()?;
    Ok(match op {
        tag::I32_CONST => Inst::I32Const(c.sleb_i32()?),
        tag::I64_CONST => Inst::I64Const(c.sleb()?),
        tag::I32_ADD => Inst::I32Add(c.idx()?, c.idx()?),
        tag::I64_ADD => Inst::I64Add(c.idx()?, c.idx()?),
        other => return Err(DecodeError::BadOpcode(other)),
    })
}

fn decode_term(c: &mut Cursor) -> Result<Terminator, DecodeError> {
    let op = c.byte()?;
    Ok(match op {
        tag::BR => Terminator::Br {
            target: c.idx()?,
            args: decode_idxs(c)?,
        },
        tag::BR_IF => Terminator::BrIf {
            cond: c.idx()?,
            then_blk: c.idx()?,
            then_args: decode_idxs(c)?,
            else_blk: c.idx()?,
            else_args: decode_idxs(c)?,
        },
        tag::RETURN => Terminator::Return(decode_idxs(c)?),
        other => return Err(DecodeError::BadOpcode(other)),
    })
}

fn decode_types(c: &mut Cursor) -> Result<Vec<ValType>, DecodeError> {
    let n = c.count()?;
    let mut ts = Vec::new();
    for _ in 0..n {
        ts.push(decode_type(c)?);
    }
    Ok(ts)
}

fn decode_type(c: &mut Cursor) -> Result<ValType, DecodeError> {
    Ok(match c.byte()? {
        tag::T_I32 => ValType::I32,
        tag::T_I64 => ValType::I64,
        tag::T_F32 => ValType::F32,
        tag::T_F64 => ValType::F64,
        other => return Err(DecodeError::BadType(other)),
    })
}

fn decode_idxs(c: &mut Cursor) -> Result<Vec<u32>, DecodeError> {
    let n = c.count()?;
    let mut v = Vec::new();
    for _ in 0..n {
        v.push(c.idx()?);
    }
    Ok(v)
}

/// A bounds-checked forward cursor. All reads return `Err` past the end.
struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Cursor { bytes, pos: 0 }
    }

    fn at_end(&self) -> bool {
        self.pos >= self.bytes.len()
    }

    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.pos)
    }

    fn byte(&mut self) -> Result<u8, DecodeError> {
        let b = *self.bytes.get(self.pos).ok_or(DecodeError::UnexpectedEof)?;
        self.pos += 1;
        Ok(b)
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], DecodeError> {
        let end = self.pos.checked_add(n).ok_or(DecodeError::UnexpectedEof)?;
        let s = self
            .bytes
            .get(self.pos..end)
            .ok_or(DecodeError::UnexpectedEof)?;
        self.pos = end;
        Ok(s)
    }

    /// Read an unsigned LEB128 as `u64` (max 10 bytes; rejects overflow).
    fn uleb(&mut self) -> Result<u64, DecodeError> {
        let mut result: u64 = 0;
        let mut shift = 0u32;
        loop {
            let byte = self.byte()?;
            if shift >= 64 {
                return Err(DecodeError::LebOverflow);
            }
            let low = (byte & 0x7f) as u64;
            // Reject bits that would not fit in u64.
            if shift == 63 && low > 1 {
                return Err(DecodeError::LebOverflow);
            }
            result |= low << shift;
            if byte & 0x80 == 0 {
                return Ok(result);
            }
            shift += 7;
        }
    }

    /// Read a signed LEB128 as `i64` (sign-extended; rejects overflow).
    fn sleb(&mut self) -> Result<i64, DecodeError> {
        let mut result: i64 = 0;
        let mut shift = 0u32;
        loop {
            let byte = self.byte()?;
            if shift >= 64 {
                return Err(DecodeError::LebOverflow);
            }
            result |= ((byte & 0x7f) as i64) << shift;
            shift += 7;
            if byte & 0x80 == 0 {
                // Sign-extend if the sign bit of the last group is set.
                if shift < 64 && (byte & 0x40) != 0 {
                    result |= -1i64 << shift;
                }
                return Ok(result);
            }
        }
    }

    fn sleb_i32(&mut self) -> Result<i32, DecodeError> {
        let v = self.sleb()?;
        i32::try_from(v).map_err(|_| DecodeError::IntTooLarge)
    }

    /// Read a value index (`u32`).
    fn idx(&mut self) -> Result<u32, DecodeError> {
        let v = self.uleb()?;
        u32::try_from(v).map_err(|_| DecodeError::IntTooLarge)
    }

    /// Read a collection count, rejecting counts that cannot fit in the remaining
    /// bytes (each item needs >= 1 byte). Prevents OOM/DoS from a forged length.
    fn count(&mut self) -> Result<usize, DecodeError> {
        let v = self.uleb()?;
        let n = usize::try_from(v).map_err(|_| DecodeError::CountTooLarge)?;
        if n > self.remaining() {
            return Err(DecodeError::CountTooLarge);
        }
        Ok(n)
    }
}
