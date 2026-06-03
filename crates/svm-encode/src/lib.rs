//! Binary encoding for the IR (`DESIGN.md` §3a). LEB128 + block-local indices; no
//! bespoke compression. The design goal is that decode and verify *fuse* into one
//! linear pass; this crate is the decode half (verification lives in `svm-verify`).
//!
//! Opcode map (one byte): families are laid out in contiguous ranges so the encoder
//! is `base + op.index()` and the decoder is a range match:
//!   `0x10..` constants · `0x20..` i32 arith · `0x30` i32 eqz · `0x31..` i32 cmp ·
//!   `0x40..` i64 arith · `0x50` i64 eqz · `0x51..` i64 cmp · `0x60..` convert ·
//!   `0x70` select · `0x80..` terminators.
//!
//! **Decoding is escape-TCB and untrusted-input-facing:** it must reject malformed
//! input with `Err` and **never panic, never OOM, always terminate** on arbitrary
//! bytes (fuzzed in the `svm` crate). We therefore never pre-allocate from an
//! untrusted count, and we reject counts that cannot fit in the remaining bytes.
#![forbid(unsafe_code)]

use svm_ir::{
    BinOp, Block, CastOp, CmpOp, ConvOp, Edge, FBinOp, FCmpOp, FToI, FUnOp, FloatTy, Func,
    FuncType, IToF, Inst, IntTy, LoadOp, Memory, Module, StoreOp, Terminator, ValIdx, ValType,
};

mod op {
    // Value types.
    pub const T_I32: u8 = 0;
    pub const T_I64: u8 = 1;
    pub const T_F32: u8 = 2;
    pub const T_F64: u8 = 3;

    // Constants.
    pub const CONST_I32: u8 = 0x10;
    pub const CONST_I64: u8 = 0x11;
    pub const CONST_F32: u8 = 0x12; // + 4 raw bytes (LE bits)
    pub const CONST_F64: u8 = 0x13; // + 8 raw bytes (LE bits)

    // Family bases (each op is `base + op.index()`) and their inclusive range ends.
    pub const I32_BIN: u8 = 0x20; // + BinOp index (0..=12)
    pub const I32_BIN_END: u8 = 0x2C;
    pub const I32_EQZ: u8 = 0x30;
    pub const I32_CMP: u8 = 0x31; // + CmpOp index (0..=9)
    pub const I32_CMP_END: u8 = 0x3A;
    pub const I64_BIN: u8 = 0x40;
    pub const I64_BIN_END: u8 = 0x4C;
    pub const I64_EQZ: u8 = 0x50;
    pub const I64_CMP: u8 = 0x51;
    pub const I64_CMP_END: u8 = 0x5A;

    // Conversions.
    pub const EXTEND_I32_S: u8 = 0x60;
    pub const EXTEND_I32_U: u8 = 0x61;
    pub const WRAP_I64: u8 = 0x62;

    pub const SELECT: u8 = 0x70;
    pub const CALL: u8 = 0x73; // direct call: uleb funcidx, then arg idx-list
    pub const CALL_INDIRECT: u8 = 0x74; // sig (params,results), idx, arg idx-list
    pub const REF_FUNC: u8 = 0x75; // uleb funcidx -> i32 funcref

    // Memory ops. Each carries: address operand, [value operand for stores], an
    // immediate uleb offset, and an alignment-hint byte.
    pub const STORE: u8 = 0x84; // + StoreOp index (0..=8) -> 0x84..=0x8C
    pub const STORE_END: u8 = 0x8C;
    pub const LOAD: u8 = 0xF0; // + LoadOp index (0..=13) -> 0xF0..=0xFD
    pub const LOAD_END: u8 = 0xFD;

    // Float families.
    pub const F32_BIN: u8 = 0x90; // + FBinOp index (0..=6)
    pub const F32_BIN_END: u8 = 0x96;
    pub const F32_UN: u8 = 0x98; // + FUnOp index (0..=6)
    pub const F32_UN_END: u8 = 0x9E;
    pub const F64_BIN: u8 = 0xA0;
    pub const F64_BIN_END: u8 = 0xA6;
    pub const F64_UN: u8 = 0xA8;
    pub const F64_UN_END: u8 = 0xAE;
    pub const F32_CMP: u8 = 0xB0; // + FCmpOp index (0..=5)
    pub const F32_CMP_END: u8 = 0xB5;
    pub const F64_CMP: u8 = 0xB8;
    pub const F64_CMP_END: u8 = 0xBD;
    pub const CAST: u8 = 0xC0; // + CastOp index (0..=5)
    pub const CAST_END: u8 = 0xC5;
    pub const FTOI: u8 = 0xD0; // + FToI index (0..=7)
    pub const FTOI_END: u8 = 0xD7;
    pub const ITOF: u8 = 0xE0; // + IToF index (0..=7)
    pub const ITOF_END: u8 = 0xE7;

    // Terminators.
    pub const BR: u8 = 0x80;
    pub const BR_IF: u8 = 0x81;
    pub const BR_TABLE: u8 = 0x82;
    pub const RETURN: u8 = 0x83;
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
    /// The memory-presence flag byte was neither 0 nor 1.
    BadMemoryFlag(u8),
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
    // Memory descriptor: presence flag, then `size_log2` if present.
    match &m.memory {
        None => out.push(0),
        Some(mem) => {
            out.push(1);
            out.push(mem.size_log2);
        }
    }
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
        Inst::ConstI32(c) => {
            out.push(op::CONST_I32);
            write_sleb(out, *c as i64);
        }
        Inst::ConstI64(c) => {
            out.push(op::CONST_I64);
            write_sleb(out, *c);
        }
        Inst::IntBin { ty, op: o, a, b } => {
            out.push(bin_base(*ty) + o.index());
            write_uleb(out, *a as u64);
            write_uleb(out, *b as u64);
        }
        Inst::IntCmp { ty, op: o, a, b } => {
            out.push(cmp_base(*ty) + o.index());
            write_uleb(out, *a as u64);
            write_uleb(out, *b as u64);
        }
        Inst::Eqz { ty, a } => {
            out.push(match ty {
                IntTy::I32 => op::I32_EQZ,
                IntTy::I64 => op::I64_EQZ,
            });
            write_uleb(out, *a as u64);
        }
        Inst::Convert { op: o, a } => {
            out.push(match o {
                ConvOp::ExtendI32S => op::EXTEND_I32_S,
                ConvOp::ExtendI32U => op::EXTEND_I32_U,
                ConvOp::WrapI64 => op::WRAP_I64,
            });
            write_uleb(out, *a as u64);
        }
        Inst::Select { cond, a, b } => {
            out.push(op::SELECT);
            write_uleb(out, *cond as u64);
            write_uleb(out, *a as u64);
            write_uleb(out, *b as u64);
        }
        Inst::ConstF32(bits) => {
            out.push(op::CONST_F32);
            out.extend_from_slice(&bits.to_le_bytes());
        }
        Inst::ConstF64(bits) => {
            out.push(op::CONST_F64);
            out.extend_from_slice(&bits.to_le_bytes());
        }
        Inst::FBin { ty, op: o, a, b } => {
            out.push(fbin_base(*ty) + o.index());
            write_uleb(out, *a as u64);
            write_uleb(out, *b as u64);
        }
        Inst::FUn { ty, op: o, a } => {
            out.push(fun_base(*ty) + o.index());
            write_uleb(out, *a as u64);
        }
        Inst::FCmp { ty, op: o, a, b } => {
            out.push(fcmp_base(*ty) + o.index());
            write_uleb(out, *a as u64);
            write_uleb(out, *b as u64);
        }
        Inst::FToISat { op: o, a } => {
            out.push(op::FTOI + o.index());
            write_uleb(out, *a as u64);
        }
        Inst::IToFConv { op: o, a } => {
            out.push(op::ITOF + o.index());
            write_uleb(out, *a as u64);
        }
        Inst::Cast { op: o, a } => {
            out.push(op::CAST + o.index());
            write_uleb(out, *a as u64);
        }
        Inst::Load {
            op: o,
            addr,
            offset,
            align,
        } => {
            out.push(op::LOAD + o.index());
            write_uleb(out, *addr as u64);
            write_uleb(out, *offset);
            out.push(*align);
        }
        Inst::Store {
            op: o,
            addr,
            value,
            offset,
            align,
        } => {
            out.push(op::STORE + o.index());
            write_uleb(out, *addr as u64);
            write_uleb(out, *value as u64);
            write_uleb(out, *offset);
            out.push(*align);
        }
        Inst::Call { func, args } => {
            out.push(op::CALL);
            write_uleb(out, *func as u64);
            write_idxs(out, args);
        }
        Inst::RefFunc { func } => {
            out.push(op::REF_FUNC);
            write_uleb(out, *func as u64);
        }
        Inst::CallIndirect { ty, idx, args } => {
            out.push(op::CALL_INDIRECT);
            write_types(out, &ty.params);
            write_types(out, &ty.results);
            write_uleb(out, *idx as u64);
            write_idxs(out, args);
        }
    }
}

fn fbin_base(ty: FloatTy) -> u8 {
    match ty {
        FloatTy::F32 => op::F32_BIN,
        FloatTy::F64 => op::F64_BIN,
    }
}
fn fun_base(ty: FloatTy) -> u8 {
    match ty {
        FloatTy::F32 => op::F32_UN,
        FloatTy::F64 => op::F64_UN,
    }
}
fn fcmp_base(ty: FloatTy) -> u8 {
    match ty {
        FloatTy::F32 => op::F32_CMP,
        FloatTy::F64 => op::F64_CMP,
    }
}

fn encode_term(out: &mut Vec<u8>, t: &Terminator) {
    match t {
        Terminator::Br { target, args } => {
            out.push(op::BR);
            write_edge(out, *target, args);
        }
        Terminator::BrIf {
            cond,
            then_blk,
            then_args,
            else_blk,
            else_args,
        } => {
            out.push(op::BR_IF);
            write_uleb(out, *cond as u64);
            write_edge(out, *then_blk, then_args);
            write_edge(out, *else_blk, else_args);
        }
        Terminator::BrTable {
            idx,
            targets,
            default,
        } => {
            out.push(op::BR_TABLE);
            write_uleb(out, *idx as u64);
            write_uleb(out, targets.len() as u64);
            for (t, args) in targets {
                write_edge(out, *t, args);
            }
            write_edge(out, default.0, &default.1);
        }
        Terminator::Return(vals) => {
            out.push(op::RETURN);
            write_idxs(out, vals);
        }
    }
}

fn bin_base(ty: IntTy) -> u8 {
    match ty {
        IntTy::I32 => op::I32_BIN,
        IntTy::I64 => op::I64_BIN,
    }
}
fn cmp_base(ty: IntTy) -> u8 {
    match ty {
        IntTy::I32 => op::I32_CMP,
        IntTy::I64 => op::I64_CMP,
    }
}

fn write_edge(out: &mut Vec<u8>, target: u32, args: &[ValIdx]) {
    write_uleb(out, target as u64);
    write_idxs(out, args);
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
        ValType::I32 => op::T_I32,
        ValType::I64 => op::T_I64,
        ValType::F32 => op::T_F32,
        ValType::F64 => op::T_F64,
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
    let memory = match c.byte()? {
        0 => None,
        1 => Some(Memory {
            size_log2: c.byte()?,
        }),
        other => return Err(DecodeError::BadMemoryFlag(other)),
    };
    let nfuncs = c.count()?;
    let mut funcs = Vec::new();
    for _ in 0..nfuncs {
        funcs.push(decode_func(&mut c)?);
    }
    if !c.at_end() {
        return Err(DecodeError::TrailingBytes);
    }
    Ok(Module { funcs, memory })
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
    let b = c.byte()?;
    Ok(match b {
        op::CONST_I32 => Inst::ConstI32(c.sleb_i32()?),
        op::CONST_I64 => Inst::ConstI64(c.sleb()?),

        op::I32_BIN..=op::I32_BIN_END => int_bin(IntTy::I32, b - op::I32_BIN, c)?,
        op::I64_BIN..=op::I64_BIN_END => int_bin(IntTy::I64, b - op::I64_BIN, c)?,

        op::I32_EQZ => Inst::Eqz {
            ty: IntTy::I32,
            a: c.idx()?,
        },
        op::I64_EQZ => Inst::Eqz {
            ty: IntTy::I64,
            a: c.idx()?,
        },

        op::I32_CMP..=op::I32_CMP_END => int_cmp(IntTy::I32, b - op::I32_CMP, c)?,
        op::I64_CMP..=op::I64_CMP_END => int_cmp(IntTy::I64, b - op::I64_CMP, c)?,

        op::EXTEND_I32_S => Inst::Convert {
            op: ConvOp::ExtendI32S,
            a: c.idx()?,
        },
        op::EXTEND_I32_U => Inst::Convert {
            op: ConvOp::ExtendI32U,
            a: c.idx()?,
        },
        op::WRAP_I64 => Inst::Convert {
            op: ConvOp::WrapI64,
            a: c.idx()?,
        },

        op::SELECT => Inst::Select {
            cond: c.idx()?,
            a: c.idx()?,
            b: c.idx()?,
        },
        op::CALL => Inst::Call {
            func: c.idx()?,
            args: decode_idxs(c)?,
        },
        op::REF_FUNC => Inst::RefFunc { func: c.idx()? },
        op::CALL_INDIRECT => Inst::CallIndirect {
            ty: FuncType {
                params: decode_types(c)?,
                results: decode_types(c)?,
            },
            idx: c.idx()?,
            args: decode_idxs(c)?,
        },

        op::CONST_F32 => Inst::ConstF32(c.u32_le()?),
        op::CONST_F64 => Inst::ConstF64(c.u64_le()?),

        op::F32_BIN..=op::F32_BIN_END => fbin(FloatTy::F32, b - op::F32_BIN, c)?,
        op::F64_BIN..=op::F64_BIN_END => fbin(FloatTy::F64, b - op::F64_BIN, c)?,
        op::F32_UN..=op::F32_UN_END => fun(FloatTy::F32, b - op::F32_UN, c)?,
        op::F64_UN..=op::F64_UN_END => fun(FloatTy::F64, b - op::F64_UN, c)?,
        op::F32_CMP..=op::F32_CMP_END => fcmp(FloatTy::F32, b - op::F32_CMP, c)?,
        op::F64_CMP..=op::F64_CMP_END => fcmp(FloatTy::F64, b - op::F64_CMP, c)?,

        op::CAST..=op::CAST_END => Inst::Cast {
            op: CastOp::from_index(b - op::CAST).ok_or(DecodeError::BadOpcode(b))?,
            a: c.idx()?,
        },
        op::FTOI..=op::FTOI_END => Inst::FToISat {
            op: FToI::from_index(b - op::FTOI).ok_or(DecodeError::BadOpcode(b))?,
            a: c.idx()?,
        },
        op::ITOF..=op::ITOF_END => Inst::IToFConv {
            op: IToF::from_index(b - op::ITOF).ok_or(DecodeError::BadOpcode(b))?,
            a: c.idx()?,
        },

        op::LOAD..=op::LOAD_END => Inst::Load {
            op: LoadOp::from_index(b - op::LOAD).ok_or(DecodeError::BadOpcode(b))?,
            addr: c.idx()?,
            offset: c.uleb()?,
            align: c.byte()?,
        },
        op::STORE..=op::STORE_END => Inst::Store {
            op: StoreOp::from_index(b - op::STORE).ok_or(DecodeError::BadOpcode(b))?,
            addr: c.idx()?,
            value: c.idx()?,
            offset: c.uleb()?,
            align: c.byte()?,
        },

        other => return Err(DecodeError::BadOpcode(other)),
    })
}

fn fbin(ty: FloatTy, index: u8, c: &mut Cursor) -> Result<Inst, DecodeError> {
    let op = FBinOp::from_index(index).ok_or(DecodeError::BadOpcode(index))?;
    Ok(Inst::FBin {
        ty,
        op,
        a: c.idx()?,
        b: c.idx()?,
    })
}

fn fun(ty: FloatTy, index: u8, c: &mut Cursor) -> Result<Inst, DecodeError> {
    let op = FUnOp::from_index(index).ok_or(DecodeError::BadOpcode(index))?;
    Ok(Inst::FUn {
        ty,
        op,
        a: c.idx()?,
    })
}

fn fcmp(ty: FloatTy, index: u8, c: &mut Cursor) -> Result<Inst, DecodeError> {
    let op = FCmpOp::from_index(index).ok_or(DecodeError::BadOpcode(index))?;
    Ok(Inst::FCmp {
        ty,
        op,
        a: c.idx()?,
        b: c.idx()?,
    })
}

fn int_bin(ty: IntTy, index: u8, c: &mut Cursor) -> Result<Inst, DecodeError> {
    let op = BinOp::from_index(index).ok_or(DecodeError::BadOpcode(index))?;
    Ok(Inst::IntBin {
        ty,
        op,
        a: c.idx()?,
        b: c.idx()?,
    })
}

fn int_cmp(ty: IntTy, index: u8, c: &mut Cursor) -> Result<Inst, DecodeError> {
    let op = CmpOp::from_index(index).ok_or(DecodeError::BadOpcode(index))?;
    Ok(Inst::IntCmp {
        ty,
        op,
        a: c.idx()?,
        b: c.idx()?,
    })
}

fn decode_term(c: &mut Cursor) -> Result<Terminator, DecodeError> {
    let b = c.byte()?;
    Ok(match b {
        op::BR => {
            let (target, args) = decode_edge(c)?;
            Terminator::Br { target, args }
        }
        op::BR_IF => {
            let cond = c.idx()?;
            let (then_blk, then_args) = decode_edge(c)?;
            let (else_blk, else_args) = decode_edge(c)?;
            Terminator::BrIf {
                cond,
                then_blk,
                then_args,
                else_blk,
                else_args,
            }
        }
        op::BR_TABLE => {
            let idx = c.idx()?;
            let n = c.count()?;
            let mut targets = Vec::new();
            for _ in 0..n {
                targets.push(decode_edge(c)?);
            }
            let default = decode_edge(c)?;
            Terminator::BrTable {
                idx,
                targets,
                default,
            }
        }
        op::RETURN => Terminator::Return(decode_idxs(c)?),
        other => return Err(DecodeError::BadOpcode(other)),
    })
}

fn decode_edge(c: &mut Cursor) -> Result<Edge, DecodeError> {
    let target = c.idx()?;
    let args = decode_idxs(c)?;
    Ok((target, args))
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
        op::T_I32 => ValType::I32,
        op::T_I64 => ValType::I64,
        op::T_F32 => ValType::F32,
        op::T_F64 => ValType::F64,
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

    /// Read 4 raw little-endian bytes as `u32` (float-constant bits).
    fn u32_le(&mut self) -> Result<u32, DecodeError> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// Read 8 raw little-endian bytes as `u64` (float-constant bits).
    fn u64_le(&mut self) -> Result<u64, DecodeError> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
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
