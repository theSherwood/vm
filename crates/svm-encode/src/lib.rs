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
// `no_std` + `alloc` so the encoder runs **in-sandbox**: a guest specializes a module with
// `svm-peval` and serializes the residual here before submitting it to the §22 `Jit` capability
// (PEVAL.md Milestone 3 — guest-side specialization). The test harness still gets `std`; dependents
// are unaffected (they bring their own `std`). Matches `svm-ir`/`svm-verify`/`svm-peval`.
#![cfg_attr(not(test), no_std)]

extern crate alloc;

use alloc::borrow::ToOwned;
use alloc::string::String;
use alloc::vec::Vec;

use svm_ir::{
    AtomicRmwOp, BinOp, Block, CastOp, CmpOp, ConvOp, Data, DebugInfo, Edge, Encoding, Export,
    FBinOp, FCmpOp, FToI, FUnOp, Field, FloatTy, Func, FuncIdx, FuncName, FuncType, IToF, Import,
    Inst, IntTy, IntUnOp, LoadOp, Loc, Memory, Module, Ordering, ProducerBlob, SsaLoc, StoreOp,
    Terminator, TypeDef, VBitBinOp, VCvtOp, VFCmpOp, VFloatBinOp, VFloatUnOp, VICmpOp, VIntBinOp,
    VIntUnOp, VNarrowOp, VPMinMaxOp, VSatBinOp, VShape, VShiftOp, VWidenOp, ValIdx, ValType,
    VarInfo, VarLoc,
};

/// Decode the atomic/fence memory-ordering byte (its [`Ordering::index`]).
fn ord_from(b: u8, op: u8) -> Result<Ordering, DecodeError> {
    Ordering::from_index(b).ok_or(DecodeError::BadOpcode(op))
}

/// Encode an [`IntTy`] as the atomic `ty` byte (`0` = i32, `1` = i64).
fn int_ty_byte(ty: IntTy) -> u8 {
    match ty {
        IntTy::I32 => 0,
        IntTy::I64 => 1,
    }
}

/// Decode the atomic `ty` byte; any other value is a malformed opcode payload.
fn int_ty_from(b: u8, op: u8) -> Result<IntTy, DecodeError> {
    match b {
        0 => Ok(IntTy::I32),
        1 => Ok(IntTy::I64),
        _ => Err(DecodeError::BadOpcode(op)),
    }
}

mod op {
    // Value types.
    pub const T_I32: u8 = 0;
    pub const T_I64: u8 = 1;
    pub const T_F32: u8 = 2;
    pub const T_F64: u8 = 3;
    pub const T_V128: u8 = 4;
    pub const T_REF: u8 = 5; // opaque 64-bit reference (GC forward-compat reservation)

    // Constants.
    pub const CONST_I32: u8 = 0x10;
    pub const CONST_I64: u8 = 0x11;
    pub const CONST_F32: u8 = 0x12; // + 4 raw bytes (LE bits)
    pub const CONST_F64: u8 = 0x13; // + 8 raw bytes (LE bits)

    // Unary integer ops (`base + IntUnOp index`, 0..=5).
    pub const I32_UN: u8 = 0x14;
    pub const I32_UN_END: u8 = 0x19;
    pub const I64_UN: u8 = 0x1A;
    pub const I64_UN_END: u8 = 0x1F;

    // Family bases (each op is `base + op.index()`) and their inclusive range ends.
    pub const I32_BIN: u8 = 0x20; // + BinOp index (0..=14)
    pub const I32_BIN_END: u8 = 0x2E;
    pub const I32_EQZ: u8 = 0x30;
    pub const I32_CMP: u8 = 0x31; // + CmpOp index (0..=9)
    pub const I32_CMP_END: u8 = 0x3A;
    pub const I64_BIN: u8 = 0x40;
    pub const I64_BIN_END: u8 = 0x4E;
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
    pub const PTR_FROM_INT: u8 = 0x76; // i64 -> i64 (no-op provenance cast)
    pub const PTR_TO_INT: u8 = 0x77;
    pub const PTR_ADD: u8 = 0x78; // (i64, i64) -> i64
    pub const CAP_CALL: u8 = 0x79; // type_id, op, sig, handle, arg idx-list
    pub const CAP_SELF_COUNT: u8 = 0x7A; // §7 reflection: () -> i32 count
    pub const CAP_SELF_GET: u8 = 0x7B; // §7 reflection: idx -> (i32 handle, i32 type_id)
    pub const CALL_IMPORT: u8 = 0x7C; // §7 unresolved import: import idx, sig, handle, arg idx-list
    pub const FMA: u8 = 0x7D; // scalar fused multiply-add: ty byte (0=f32,1=f64), a, b, c
    pub const CAP_SELF_RESOLVE: u8 = 0x7E; // §7 reflection: (name_ptr, name_len) -> i32 handle|-errno
    pub const CAP_SELF_LABEL: u8 = 0x7F; // §7 reflection: (handle, buf_ptr, buf_cap) -> i32 label len

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
    // §12 atomics. Each is followed by a `ty` byte (0 = i32, 1 = i64), then operand idxs + offset.
    pub const ATOMIC_LOAD: u8 = 0xC6; // ty, addr, offset
    pub const ATOMIC_STORE: u8 = 0xC7; // ty, addr, value, offset
    pub const ATOMIC_RMW: u8 = 0xC8; // ty, AtomicRmwOp index, addr, value, offset
    pub const ATOMIC_CMPXCHG: u8 = 0xC9; // ty, addr, expected, replacement, offset

    // §12 fibers (stack switching).
    pub const CONT_NEW: u8 = 0xCA; // func (funcref idx), sp (data-stack base)
    pub const CONT_RESUME: u8 = 0xCB; // k, arg
    pub const SUSPEND: u8 = 0xCC; // value
    pub const THREAD_SPAWN: u8 = 0xCD; // func (funcidx), arg -> i32 handle
    pub const THREAD_JOIN: u8 = 0xCE; // handle -> i64 result
    pub const ATOMIC_WAIT: u8 = 0xCF; // ty, addr, expected, timeout -> i32 status
    pub const FTOI: u8 = 0xD0; // saturating trunc_sat: + FToI index (0..=7)
    pub const FTOI_END: u8 = 0xD7;
    pub const FTOI_TRAP: u8 = 0xD8; // trapping trunc: + FToI index (0..=7)
    pub const FTOI_TRAP_END: u8 = 0xDF;
    pub const ITOF: u8 = 0xE0; // + IToF index (0..=7)
    pub const ITOF_END: u8 = 0xE7;
    pub const ATOMIC_NOTIFY: u8 = 0xE8; // addr, count -> i32 woken
    pub const ATOMIC_FENCE: u8 = 0xE9; // order byte

    // §GC (GC.md) conservative root enumeration.
    pub const GC_ROOTS: u8 = 0xEA; // heap_lo, heap_hi, mask, buf, cap -> i64 count
    pub const VCPU_TLS_GET: u8 = 0xEB; // §12 per-vCPU TLS register: () -> i64
    pub const VCPU_TLS_SET: u8 = 0xEC; // §12 per-vCPU TLS register: val -> ()
    pub const DURABLE_SHADOW_BASE: u8 = 0xED; // durable-internal: () -> i64 (current ctx shadow base)
    pub const SETJMP: u8 = 0xEE; // <setjmp.h>: buf -> i32 (0, or the longjmp value on re-entry)
    pub const LONGJMP: u8 = 0xEF; // <setjmp.h>: buf, val -> () (noreturn)

    // §17 SIMD (D58). One prefix byte, then a sub-opcode (à la wasm's 0xFD) — keeps the
    // crowded primary opcode space free. Each `simd::*` sub-op's payload is documented inline.
    pub const SIMD: u8 = 0xFE;
    pub mod simd {
        pub const CONST: u8 = 0x00; // + 16 raw value bytes (LE)
        pub const LOAD: u8 = 0x01; // addr, offset (uleb), align (byte)
        pub const STORE: u8 = 0x02; // addr, value, offset, align
        pub const SPLAT: u8 = 0x03; // shape, a
        pub const EXTRACT_LANE: u8 = 0x04; // shape, lane (byte), signed (byte), a
        pub const REPLACE_LANE: u8 = 0x05; // shape, lane (byte), a, b
        pub const VINT_BIN: u8 = 0x06; // shape, op, a, b
        pub const VFLOAT_BIN: u8 = 0x07; // shape, op, a, b
        pub const VFLOAT_UN: u8 = 0x08; // shape, op, a
        pub const VBIT_BIN: u8 = 0x09; // op, a, b
        pub const NOT: u8 = 0x0A; // a
        pub const BITSELECT: u8 = 0x0B; // a, b, mask
        pub const SHUFFLE: u8 = 0x0C; // 16 lane bytes, a, b
        pub const SWIZZLE: u8 = 0x0D; // a, b
        pub const WIDTH_BYTES: u8 = 0x0E; // (no payload) -> i32
        pub const VINT_CMP: u8 = 0x0F; // shape, op, a, b
        pub const VFLOAT_CMP: u8 = 0x10; // shape, op, a, b
        pub const VSHIFT: u8 = 0x11; // shape, op, a (v128), amt (i32)
        pub const VINT_UN: u8 = 0x12; // shape, op, a
        pub const VANY_TRUE: u8 = 0x13; // a -> i32
        pub const VALL_TRUE: u8 = 0x14; // shape, a -> i32
        pub const VBITMASK: u8 = 0x15; // shape, a -> i32
        pub const VSAT_BIN: u8 = 0x16; // shape, op, a, b
        pub const VWIDEN: u8 = 0x17; // shape (result), op, a
        pub const VNARROW: u8 = 0x18; // shape (result), op, a, b
        pub const VCONVERT: u8 = 0x19; // op, a
        pub const VPMINMAX: u8 = 0x1A; // shape, op, a, b
        pub const VPOPCNT: u8 = 0x1B; // a (i8x16 implicit)
        pub const VAVGR: u8 = 0x1C; // shape, a, b
        pub const VDOT: u8 = 0x1D; // a, b (i16x8 -> i32x4 implicit)
        pub const VDOT_I8: u8 = 0x22; // a, b (i8x16 -> i16x8 implicit)
        pub const VEXTMUL: u8 = 0x1E; // shape (wide), op (VWidenOp), a, b
        pub const VEXTADD: u8 = 0x1F; // shape (wide), signed (u8), a
        pub const VQ15MULR: u8 = 0x20; // a, b (i16x8 implicit)
        pub const VFMA: u8 = 0x21; // shape, neg (u8), a, b, c
    }

    // Terminators (decoded in a separate context from instruction opcodes).
    pub const BR: u8 = 0x80;
    pub const BR_IF: u8 = 0x81;
    pub const BR_TABLE: u8 = 0x82;
    pub const RETURN: u8 = 0x83;
    pub const RETURN_CALL: u8 = 0x85; // uleb funcidx, arg idx-list
    pub const RETURN_CALL_INDIRECT: u8 = 0x86; // sig (params,results), idx, arg idx-list
    pub const UNREACHABLE: u8 = 0x8F;
}

const MAGIC: [u8; 4] = *b"SVM\x00";
// v3 adds the first-class **export section** (named function entry points: name + funcidx), the
// runtime-`Module` analogue of a link unit's exports — so an embedder can `call("main")` by name.
// The decoder accepts only the exact current `VERSION`, so the bump simply retires v2 readers; there
// is no in-place v2 blob to stay compatible with.
// v2 adds the §7 import section (name + op signature per import) and the `call.import` opcode, so a
// separately-compiled unit can be serialized with its symbols **still unresolved** — the precondition
// for host-assisted dynamic linking (DESIGN.md §22: the loader resolves a guest-shipped blob's imports
// against a symbol table, then re-verifies). v1 was always import-free (imports resolved pre-encode).
const VERSION: u8 = 3;

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
    /// A data segment's `readonly` flag byte was neither 0 nor 1.
    BadDataFlag(u8),
    /// An import name's length-prefixed bytes were not valid UTF-8.
    BadUtf8,
    /// Bytes remained after a complete module was decoded.
    TrailingBytes,
    /// A debug-info `TypeDef` kind byte was not one of the known variants.
    BadTypeDefKind(u8),
    /// A debug-info base-type encoding byte was out of range.
    BadEncoding(u8),
    /// A debug-info `VarLoc` discriminant byte was neither 0 (window) nor 1 (ssa).
    BadVarLoc(u8),
    /// A debug-info optional-field flag byte was neither 0 (none) nor 1 (some).
    BadOptionFlag(u8),
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
    // Data segments (§3a / D40): count, then each `readonly` flag, `offset`, and length-prefixed
    // bytes.
    write_uleb(&mut out, m.data.len() as u64);
    for d in &m.data {
        out.push(d.readonly as u8);
        write_uleb(&mut out, d.offset);
        write_uleb(&mut out, d.bytes.len() as u64);
        out.extend_from_slice(&d.bytes);
    }
    // §7 import section (v2): count, then each import's `name` and op `sig`. Usually empty — an
    // import-free module (every prior frontend's output, and any unit whose symbols were already
    // resolved) writes a single `0` here. A non-empty section is a unit shipped with **unresolved**
    // symbols, for a loader to bind by name (DESIGN.md §22 host-assisted resolve).
    write_uleb(&mut out, m.imports.len() as u64);
    for imp in &m.imports {
        write_str(&mut out, &imp.name);
        write_types(&mut out, &imp.sig.params);
        write_types(&mut out, &imp.sig.results);
    }
    // Export section (v3): count, then each export's `name` and target funcidx. Usually a handful
    // (the named entry points); empty for a bare kernel addressed only by index.
    write_uleb(&mut out, m.exports.len() as u64);
    for e in &m.exports {
        write_str(&mut out, &e.name);
        write_uleb(&mut out, e.func as u64);
    }
    write_uleb(&mut out, m.funcs.len() as u64);
    for f in &m.funcs {
        encode_func(&mut out, f);
    }
    // Optional strippable debug section (DEBUGGING.md §6/§2a). Written only when present, so a
    // module without debug info encodes byte-identically to before (snapshot digests, round-trip
    // fixtures): the decoder treats "no bytes after the funcs" as no debug info.
    if let Some(di) = &m.debug_info {
        encode_debug_info(&mut out, di);
    }
    out
}

/// Encode the §6 debug-info waist: the file table, source locations, the structured type table,
/// and the variable table. Mirrored by [`decode_debug_info`].
fn encode_debug_info(out: &mut Vec<u8>, di: &DebugInfo) {
    write_uleb(out, di.files.len() as u64);
    for f in &di.files {
        write_str(out, f);
    }
    write_uleb(out, di.locs.len() as u64);
    for l in &di.locs {
        for field in [l.func, l.block, l.inst, l.file, l.line, l.col] {
            write_uleb(out, field as u64);
        }
    }
    write_uleb(out, di.types.len() as u64);
    for t in &di.types {
        match t {
            TypeDef::Base {
                name,
                encoding,
                size,
            } => {
                out.push(0);
                write_str(out, name);
                out.push(match encoding {
                    Encoding::Signed => 0,
                    Encoding::Unsigned => 1,
                    Encoding::Float => 2,
                    Encoding::Bool => 3,
                });
                write_uleb(out, *size as u64);
            }
            TypeDef::Pointer {
                name,
                pointee,
                size,
            } => {
                out.push(1);
                write_str(out, name);
                write_uleb(out, *pointee as u64);
                write_uleb(out, *size as u64);
            }
            TypeDef::Array { name, elem, count } => {
                out.push(2);
                write_str(out, name);
                write_uleb(out, *elem as u64);
                write_uleb(out, *count as u64);
            }
            TypeDef::Aggregate { name, size, fields } => {
                out.push(3);
                write_str(out, name);
                write_uleb(out, *size as u64);
                write_uleb(out, fields.len() as u64);
                for f in fields {
                    write_str(out, &f.name);
                    write_uleb(out, f.offset as u64);
                    write_uleb(out, f.ty as u64);
                }
            }
            TypeDef::Opaque { name, size } => {
                out.push(4);
                write_str(out, name);
                write_uleb(out, *size as u64);
            }
        }
    }
    write_uleb(out, di.vars.len() as u64);
    for v in &di.vars {
        write_uleb(out, v.func as u64);
        write_str(out, &v.name);
        write_str(out, &v.ty);
        match &v.loc {
            VarLoc::Window { off } => {
                out.push(0);
                write_sleb(out, *off);
            }
            VarLoc::Ssa { value } => {
                out.push(1);
                write_uleb(out, *value as u64);
            }
            VarLoc::SsaList(locs) => {
                out.push(2);
                write_uleb(out, locs.len() as u64);
                for l in locs {
                    write_uleb(out, l.block as u64);
                    write_uleb(out, l.inst as u64);
                    write_uleb(out, l.value as u64);
                }
            }
            VarLoc::WindowVia { base, off } => {
                out.push(3);
                write_uleb(out, base.len() as u64);
                for l in base {
                    write_uleb(out, l.block as u64);
                    write_uleb(out, l.inst as u64);
                    write_uleb(out, l.value as u64);
                }
                write_sleb(out, *off);
            }
            VarLoc::Fixed { addr } => {
                out.push(4);
                write_uleb(out, *addr);
            }
        }
        match v.type_id {
            None => out.push(0),
            Some(t) => {
                out.push(1);
                write_uleb(out, t as u64);
            }
        }
        // Optional lexical scope `(start_line, end_line)` (§6 shadowing resolution).
        match v.scope {
            None => out.push(0),
            Some((s, e)) => {
                out.push(1);
                write_uleb(out, s as u64);
                write_uleb(out, e as u64);
            }
        }
    }
    // Opaque per-producer rich blobs (§6): count, then each `(producer, length-prefixed bytes)`.
    write_uleb(out, di.blobs.len() as u64);
    for b in &di.blobs {
        write_str(out, &b.producer);
        write_uleb(out, b.bytes.len() as u64);
        out.extend_from_slice(&b.bytes);
    }
    // Function names (§6, last so an older decoder stops cleanly at the blobs): `(func, name)`.
    write_uleb(out, di.func_names.len() as u64);
    for fname in &di.func_names {
        write_uleb(out, fname.func as u64);
        write_str(out, &fname.name);
    }
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
        // §7 named import (v2 wire form), mirroring `cap.call` but carrying an import *index*
        // (into the module's import section) instead of a bound `(type_id, op)`: the import stays
        // unresolved on the wire so a loader can bind it by name (DESIGN.md §22 host-assisted resolve).
        Inst::CallImport {
            import,
            sig,
            handle,
            args,
        } => {
            out.push(op::CALL_IMPORT);
            write_uleb(out, *import as u64);
            write_types(out, &sig.params);
            write_types(out, &sig.results);
            write_uleb(out, *handle as u64);
            write_idxs(out, args);
        }
        // §7 capability reflection intrinsics.
        Inst::CapSelfCount => out.push(op::CAP_SELF_COUNT),
        Inst::CapSelfGet { idx } => {
            out.push(op::CAP_SELF_GET);
            write_uleb(out, *idx as u64);
        }
        Inst::CapSelfResolve { name_ptr, name_len } => {
            out.push(op::CAP_SELF_RESOLVE);
            write_uleb(out, *name_ptr as u64);
            write_uleb(out, *name_len as u64);
        }
        Inst::CapSelfLabel {
            handle,
            buf_ptr,
            buf_cap,
        } => {
            out.push(op::CAP_SELF_LABEL);
            write_uleb(out, *handle as u64);
            write_uleb(out, *buf_ptr as u64);
            write_uleb(out, *buf_cap as u64);
        }
        Inst::VcpuTlsGet => out.push(op::VCPU_TLS_GET),
        Inst::DurableShadowBase => out.push(op::DURABLE_SHADOW_BASE),
        Inst::VcpuTlsSet { val } => {
            out.push(op::VCPU_TLS_SET);
            write_uleb(out, *val as u64);
        }
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
        Inst::IntUn { ty, op: o, a } => {
            out.push(un_base(*ty) + o.index());
            write_uleb(out, *a as u64);
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
        Inst::Fma { ty, a, b, c } => {
            out.push(op::FMA);
            out.push(match ty {
                FloatTy::F32 => 0,
                FloatTy::F64 => 1,
            });
            write_uleb(out, *a as u64);
            write_uleb(out, *b as u64);
            write_uleb(out, *c as u64);
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
        Inst::AtomicLoad {
            ty,
            addr,
            offset,
            order,
        } => {
            out.push(op::ATOMIC_LOAD);
            out.push(int_ty_byte(*ty));
            write_uleb(out, *addr as u64);
            write_uleb(out, *offset);
            out.push(order.index());
        }
        Inst::AtomicStore {
            ty,
            addr,
            value,
            offset,
            order,
        } => {
            out.push(op::ATOMIC_STORE);
            out.push(int_ty_byte(*ty));
            write_uleb(out, *addr as u64);
            write_uleb(out, *value as u64);
            write_uleb(out, *offset);
            out.push(order.index());
        }
        Inst::AtomicRmw {
            ty,
            op: rmw,
            addr,
            value,
            offset,
            order,
        } => {
            out.push(op::ATOMIC_RMW);
            out.push(int_ty_byte(*ty));
            out.push(rmw.index());
            write_uleb(out, *addr as u64);
            write_uleb(out, *value as u64);
            write_uleb(out, *offset);
            out.push(order.index());
        }
        Inst::AtomicCmpxchg {
            ty,
            addr,
            expected,
            replacement,
            offset,
            order,
        } => {
            out.push(op::ATOMIC_CMPXCHG);
            out.push(int_ty_byte(*ty));
            write_uleb(out, *addr as u64);
            write_uleb(out, *expected as u64);
            write_uleb(out, *replacement as u64);
            write_uleb(out, *offset);
            out.push(order.index());
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
        Inst::FToITrap { op: o, a } => {
            out.push(op::FTOI_TRAP + o.index());
            write_uleb(out, *a as u64);
        }
        Inst::PtrAdd { a, b } => {
            out.push(op::PTR_ADD);
            write_uleb(out, *a as u64);
            write_uleb(out, *b as u64);
        }
        Inst::PtrCast { to_int, a } => {
            out.push(if *to_int {
                op::PTR_TO_INT
            } else {
                op::PTR_FROM_INT
            });
            write_uleb(out, *a as u64);
        }
        Inst::CallIndirect { ty, idx, args } => {
            out.push(op::CALL_INDIRECT);
            write_types(out, &ty.params);
            write_types(out, &ty.results);
            write_uleb(out, *idx as u64);
            write_idxs(out, args);
        }
        Inst::CapCall {
            type_id,
            op,
            sig,
            handle,
            args,
        } => {
            out.push(op::CAP_CALL);
            write_uleb(out, *type_id as u64);
            write_uleb(out, *op as u64);
            write_types(out, &sig.params);
            write_types(out, &sig.results);
            write_uleb(out, *handle as u64);
            write_idxs(out, args);
        }
        Inst::ContNew { func, sp } => {
            out.push(op::CONT_NEW);
            write_uleb(out, *func as u64);
            write_uleb(out, *sp as u64);
        }
        Inst::ContResume { k, arg } => {
            out.push(op::CONT_RESUME);
            write_uleb(out, *k as u64);
            write_uleb(out, *arg as u64);
        }
        Inst::Suspend { value } => {
            out.push(op::SUSPEND);
            write_uleb(out, *value as u64);
        }
        Inst::SetJmp { buf } => {
            out.push(op::SETJMP);
            write_uleb(out, *buf as u64);
        }
        Inst::LongJmp { buf, val } => {
            out.push(op::LONGJMP);
            write_uleb(out, *buf as u64);
            write_uleb(out, *val as u64);
        }
        Inst::GcRoots {
            heap_lo,
            heap_hi,
            mask,
            buf,
            cap,
        } => {
            out.push(op::GC_ROOTS);
            write_uleb(out, *heap_lo as u64);
            write_uleb(out, *heap_hi as u64);
            write_uleb(out, *mask as u64);
            write_uleb(out, *buf as u64);
            write_uleb(out, *cap as u64);
        }
        Inst::ThreadSpawn { func, sp, arg } => {
            out.push(op::THREAD_SPAWN);
            write_uleb(out, *func as u64);
            write_uleb(out, *sp as u64);
            write_uleb(out, *arg as u64);
        }
        Inst::ThreadJoin { handle } => {
            out.push(op::THREAD_JOIN);
            write_uleb(out, *handle as u64);
        }
        Inst::MemoryWait {
            ty,
            addr,
            expected,
            timeout,
        } => {
            out.push(op::ATOMIC_WAIT);
            out.push(int_ty_byte(*ty));
            write_uleb(out, *addr as u64);
            write_uleb(out, *expected as u64);
            write_uleb(out, *timeout as u64);
        }
        Inst::MemoryNotify { addr, count } => {
            out.push(op::ATOMIC_NOTIFY);
            write_uleb(out, *addr as u64);
            write_uleb(out, *count as u64);
        }
        Inst::AtomicFence { order } => {
            out.push(op::ATOMIC_FENCE);
            out.push(order.index());
        }

        // ----- §17 SIMD (D58): prefix byte + sub-opcode -----
        Inst::ConstV128(bytes) => {
            out.push(op::SIMD);
            out.push(op::simd::CONST);
            out.extend_from_slice(bytes);
        }
        Inst::V128Load {
            addr,
            offset,
            align,
        } => {
            out.push(op::SIMD);
            out.push(op::simd::LOAD);
            write_uleb(out, *addr as u64);
            write_uleb(out, *offset);
            out.push(*align);
        }
        Inst::V128Store {
            addr,
            value,
            offset,
            align,
        } => {
            out.push(op::SIMD);
            out.push(op::simd::STORE);
            write_uleb(out, *addr as u64);
            write_uleb(out, *value as u64);
            write_uleb(out, *offset);
            out.push(*align);
        }
        Inst::Splat { shape, a } => {
            out.push(op::SIMD);
            out.push(op::simd::SPLAT);
            out.push(shape.index());
            write_uleb(out, *a as u64);
        }
        Inst::ExtractLane {
            shape,
            lane,
            signed,
            a,
        } => {
            out.push(op::SIMD);
            out.push(op::simd::EXTRACT_LANE);
            out.push(shape.index());
            out.push(*lane);
            out.push(*signed as u8);
            write_uleb(out, *a as u64);
        }
        Inst::ReplaceLane { shape, lane, a, b } => {
            out.push(op::SIMD);
            out.push(op::simd::REPLACE_LANE);
            out.push(shape.index());
            out.push(*lane);
            write_uleb(out, *a as u64);
            write_uleb(out, *b as u64);
        }
        Inst::VIntBin { shape, op: o, a, b } => {
            out.push(op::SIMD);
            out.push(op::simd::VINT_BIN);
            out.push(shape.index());
            out.push(o.index());
            write_uleb(out, *a as u64);
            write_uleb(out, *b as u64);
        }
        Inst::VIntCmp { shape, op: o, a, b } => {
            out.push(op::SIMD);
            out.push(op::simd::VINT_CMP);
            out.push(shape.index());
            out.push(o.index());
            write_uleb(out, *a as u64);
            write_uleb(out, *b as u64);
        }
        Inst::VFloatCmp { shape, op: o, a, b } => {
            out.push(op::SIMD);
            out.push(op::simd::VFLOAT_CMP);
            out.push(shape.index());
            out.push(o.index());
            write_uleb(out, *a as u64);
            write_uleb(out, *b as u64);
        }
        Inst::VShift {
            shape,
            op: o,
            a,
            amt,
        } => {
            out.push(op::SIMD);
            out.push(op::simd::VSHIFT);
            out.push(shape.index());
            out.push(o.index());
            write_uleb(out, *a as u64);
            write_uleb(out, *amt as u64);
        }
        Inst::VFloatBin { shape, op: o, a, b } => {
            out.push(op::SIMD);
            out.push(op::simd::VFLOAT_BIN);
            out.push(shape.index());
            out.push(o.index());
            write_uleb(out, *a as u64);
            write_uleb(out, *b as u64);
        }
        Inst::VFloatUn { shape, op: o, a } => {
            out.push(op::SIMD);
            out.push(op::simd::VFLOAT_UN);
            out.push(shape.index());
            out.push(o.index());
            write_uleb(out, *a as u64);
        }
        Inst::VIntUn { shape, op: o, a } => {
            out.push(op::SIMD);
            out.push(op::simd::VINT_UN);
            out.push(shape.index());
            out.push(o.index());
            write_uleb(out, *a as u64);
        }
        Inst::VSatBin { shape, op: o, a, b } => {
            out.push(op::SIMD);
            out.push(op::simd::VSAT_BIN);
            out.push(shape.index());
            out.push(o.index());
            write_uleb(out, *a as u64);
            write_uleb(out, *b as u64);
        }
        Inst::VWiden { shape, op: o, a } => {
            out.push(op::SIMD);
            out.push(op::simd::VWIDEN);
            out.push(shape.index());
            out.push(o.index());
            write_uleb(out, *a as u64);
        }
        Inst::VNarrow { shape, op: o, a, b } => {
            out.push(op::SIMD);
            out.push(op::simd::VNARROW);
            out.push(shape.index());
            out.push(o.index());
            write_uleb(out, *a as u64);
            write_uleb(out, *b as u64);
        }
        Inst::VConvert { op: o, a } => {
            out.push(op::SIMD);
            out.push(op::simd::VCONVERT);
            out.push(o.index());
            write_uleb(out, *a as u64);
        }
        Inst::VPMinMax { shape, op: o, a, b } => {
            out.push(op::SIMD);
            out.push(op::simd::VPMINMAX);
            out.push(shape.index());
            out.push(o.index());
            write_uleb(out, *a as u64);
            write_uleb(out, *b as u64);
        }
        Inst::VPopcnt { a } => {
            out.push(op::SIMD);
            out.push(op::simd::VPOPCNT);
            write_uleb(out, *a as u64);
        }
        Inst::VAvgr { shape, a, b } => {
            out.push(op::SIMD);
            out.push(op::simd::VAVGR);
            out.push(shape.index());
            write_uleb(out, *a as u64);
            write_uleb(out, *b as u64);
        }
        Inst::VDot { a, b } => {
            out.push(op::SIMD);
            out.push(op::simd::VDOT);
            write_uleb(out, *a as u64);
            write_uleb(out, *b as u64);
        }
        Inst::VDotI8 { a, b } => {
            out.push(op::SIMD);
            out.push(op::simd::VDOT_I8);
            write_uleb(out, *a as u64);
            write_uleb(out, *b as u64);
        }
        Inst::VExtMul { shape, op: o, a, b } => {
            out.push(op::SIMD);
            out.push(op::simd::VEXTMUL);
            out.push(shape.index());
            out.push(o.index());
            write_uleb(out, *a as u64);
            write_uleb(out, *b as u64);
        }
        Inst::VExtAddPairwise { shape, signed, a } => {
            out.push(op::SIMD);
            out.push(op::simd::VEXTADD);
            out.push(shape.index());
            out.push(*signed as u8);
            write_uleb(out, *a as u64);
        }
        Inst::VQ15MulrSat { a, b } => {
            out.push(op::SIMD);
            out.push(op::simd::VQ15MULR);
            write_uleb(out, *a as u64);
            write_uleb(out, *b as u64);
        }
        Inst::VFma {
            shape,
            neg,
            a,
            b,
            c,
        } => {
            out.push(op::SIMD);
            out.push(op::simd::VFMA);
            out.push(shape.index());
            out.push(*neg as u8);
            write_uleb(out, *a as u64);
            write_uleb(out, *b as u64);
            write_uleb(out, *c as u64);
        }
        Inst::VAnyTrue { a } => {
            out.push(op::SIMD);
            out.push(op::simd::VANY_TRUE);
            write_uleb(out, *a as u64);
        }
        Inst::VAllTrue { shape, a } => {
            out.push(op::SIMD);
            out.push(op::simd::VALL_TRUE);
            out.push(shape.index());
            write_uleb(out, *a as u64);
        }
        Inst::VBitmask { shape, a } => {
            out.push(op::SIMD);
            out.push(op::simd::VBITMASK);
            out.push(shape.index());
            write_uleb(out, *a as u64);
        }
        Inst::VBitBin { op: o, a, b } => {
            out.push(op::SIMD);
            out.push(op::simd::VBIT_BIN);
            out.push(o.index());
            write_uleb(out, *a as u64);
            write_uleb(out, *b as u64);
        }
        Inst::VNot { a } => {
            out.push(op::SIMD);
            out.push(op::simd::NOT);
            write_uleb(out, *a as u64);
        }
        Inst::Bitselect { a, b, mask } => {
            out.push(op::SIMD);
            out.push(op::simd::BITSELECT);
            write_uleb(out, *a as u64);
            write_uleb(out, *b as u64);
            write_uleb(out, *mask as u64);
        }
        Inst::Shuffle { lanes, a, b } => {
            out.push(op::SIMD);
            out.push(op::simd::SHUFFLE);
            out.extend_from_slice(lanes);
            write_uleb(out, *a as u64);
            write_uleb(out, *b as u64);
        }
        Inst::Swizzle { a, b } => {
            out.push(op::SIMD);
            out.push(op::simd::SWIZZLE);
            write_uleb(out, *a as u64);
            write_uleb(out, *b as u64);
        }
        Inst::SimdWidthBytes => {
            out.push(op::SIMD);
            out.push(op::simd::WIDTH_BYTES);
        }
    }
}

/// Decode the 16 value/lane bytes of a `v128.const` / `i8x16.shuffle`.
fn dec_byte16(c: &mut Cursor) -> Result<[u8; 16], DecodeError> {
    let s = c.take(16)?;
    let mut a = [0u8; 16];
    a.copy_from_slice(s);
    Ok(a)
}

/// Decode a [`VShape`] index byte.
fn dec_shape(c: &mut Cursor) -> Result<VShape, DecodeError> {
    let b = c.byte()?;
    VShape::from_index(b).ok_or(DecodeError::BadOpcode(b))
}

/// Decode one SIMD sub-opcode (the byte after the [`op::SIMD`] prefix).
fn decode_simd(c: &mut Cursor) -> Result<Inst, DecodeError> {
    let sub = c.byte()?;
    Ok(match sub {
        op::simd::CONST => Inst::ConstV128(dec_byte16(c)?),
        op::simd::LOAD => Inst::V128Load {
            addr: c.idx()?,
            offset: c.uleb()?,
            align: c.byte()?,
        },
        op::simd::STORE => Inst::V128Store {
            addr: c.idx()?,
            value: c.idx()?,
            offset: c.uleb()?,
            align: c.byte()?,
        },
        op::simd::SPLAT => Inst::Splat {
            shape: dec_shape(c)?,
            a: c.idx()?,
        },
        op::simd::EXTRACT_LANE => Inst::ExtractLane {
            shape: dec_shape(c)?,
            lane: c.byte()?,
            signed: c.byte()? != 0,
            a: c.idx()?,
        },
        op::simd::REPLACE_LANE => Inst::ReplaceLane {
            shape: dec_shape(c)?,
            lane: c.byte()?,
            a: c.idx()?,
            b: c.idx()?,
        },
        op::simd::VINT_BIN => {
            let shape = dec_shape(c)?;
            let ob = c.byte()?;
            Inst::VIntBin {
                shape,
                op: VIntBinOp::from_index(ob).ok_or(DecodeError::BadOpcode(ob))?,
                a: c.idx()?,
                b: c.idx()?,
            }
        }
        op::simd::VINT_CMP => {
            let shape = dec_shape(c)?;
            let ob = c.byte()?;
            Inst::VIntCmp {
                shape,
                op: VICmpOp::from_index(ob).ok_or(DecodeError::BadOpcode(ob))?,
                a: c.idx()?,
                b: c.idx()?,
            }
        }
        op::simd::VFLOAT_CMP => {
            let shape = dec_shape(c)?;
            let ob = c.byte()?;
            Inst::VFloatCmp {
                shape,
                op: VFCmpOp::from_index(ob).ok_or(DecodeError::BadOpcode(ob))?,
                a: c.idx()?,
                b: c.idx()?,
            }
        }
        op::simd::VSHIFT => {
            let shape = dec_shape(c)?;
            let ob = c.byte()?;
            Inst::VShift {
                shape,
                op: VShiftOp::from_index(ob).ok_or(DecodeError::BadOpcode(ob))?,
                a: c.idx()?,
                amt: c.idx()?,
            }
        }
        op::simd::VFLOAT_BIN => {
            let shape = dec_shape(c)?;
            let ob = c.byte()?;
            Inst::VFloatBin {
                shape,
                op: VFloatBinOp::from_index(ob).ok_or(DecodeError::BadOpcode(ob))?,
                a: c.idx()?,
                b: c.idx()?,
            }
        }
        op::simd::VFLOAT_UN => {
            let shape = dec_shape(c)?;
            let ob = c.byte()?;
            Inst::VFloatUn {
                shape,
                op: VFloatUnOp::from_index(ob).ok_or(DecodeError::BadOpcode(ob))?,
                a: c.idx()?,
            }
        }
        op::simd::VINT_UN => {
            let shape = dec_shape(c)?;
            let ob = c.byte()?;
            Inst::VIntUn {
                shape,
                op: VIntUnOp::from_index(ob).ok_or(DecodeError::BadOpcode(ob))?,
                a: c.idx()?,
            }
        }
        op::simd::VSAT_BIN => {
            let shape = dec_shape(c)?;
            let ob = c.byte()?;
            Inst::VSatBin {
                shape,
                op: VSatBinOp::from_index(ob).ok_or(DecodeError::BadOpcode(ob))?,
                a: c.idx()?,
                b: c.idx()?,
            }
        }
        op::simd::VWIDEN => {
            let shape = dec_shape(c)?;
            let ob = c.byte()?;
            Inst::VWiden {
                shape,
                op: VWidenOp::from_index(ob).ok_or(DecodeError::BadOpcode(ob))?,
                a: c.idx()?,
            }
        }
        op::simd::VNARROW => {
            let shape = dec_shape(c)?;
            let ob = c.byte()?;
            Inst::VNarrow {
                shape,
                op: VNarrowOp::from_index(ob).ok_or(DecodeError::BadOpcode(ob))?,
                a: c.idx()?,
                b: c.idx()?,
            }
        }
        op::simd::VCONVERT => {
            let ob = c.byte()?;
            Inst::VConvert {
                op: VCvtOp::from_index(ob).ok_or(DecodeError::BadOpcode(ob))?,
                a: c.idx()?,
            }
        }
        op::simd::VPMINMAX => {
            let shape = dec_shape(c)?;
            let ob = c.byte()?;
            Inst::VPMinMax {
                shape,
                op: VPMinMaxOp::from_index(ob).ok_or(DecodeError::BadOpcode(ob))?,
                a: c.idx()?,
                b: c.idx()?,
            }
        }
        op::simd::VPOPCNT => Inst::VPopcnt { a: c.idx()? },
        op::simd::VAVGR => Inst::VAvgr {
            shape: dec_shape(c)?,
            a: c.idx()?,
            b: c.idx()?,
        },
        op::simd::VDOT => Inst::VDot {
            a: c.idx()?,
            b: c.idx()?,
        },
        op::simd::VDOT_I8 => Inst::VDotI8 {
            a: c.idx()?,
            b: c.idx()?,
        },
        op::simd::VEXTMUL => {
            let shape = dec_shape(c)?;
            let ob = c.byte()?;
            Inst::VExtMul {
                shape,
                op: VWidenOp::from_index(ob).ok_or(DecodeError::BadOpcode(ob))?,
                a: c.idx()?,
                b: c.idx()?,
            }
        }
        op::simd::VEXTADD => {
            let shape = dec_shape(c)?;
            let signed = c.byte()? != 0;
            Inst::VExtAddPairwise {
                shape,
                signed,
                a: c.idx()?,
            }
        }
        op::simd::VQ15MULR => Inst::VQ15MulrSat {
            a: c.idx()?,
            b: c.idx()?,
        },
        op::simd::VFMA => {
            let shape = dec_shape(c)?;
            let neg = c.byte()? != 0;
            Inst::VFma {
                shape,
                neg,
                a: c.idx()?,
                b: c.idx()?,
                c: c.idx()?,
            }
        }
        op::simd::VANY_TRUE => Inst::VAnyTrue { a: c.idx()? },
        op::simd::VALL_TRUE => Inst::VAllTrue {
            shape: dec_shape(c)?,
            a: c.idx()?,
        },
        op::simd::VBITMASK => Inst::VBitmask {
            shape: dec_shape(c)?,
            a: c.idx()?,
        },
        op::simd::VBIT_BIN => {
            let ob = c.byte()?;
            Inst::VBitBin {
                op: VBitBinOp::from_index(ob).ok_or(DecodeError::BadOpcode(ob))?,
                a: c.idx()?,
                b: c.idx()?,
            }
        }
        op::simd::NOT => Inst::VNot { a: c.idx()? },
        op::simd::BITSELECT => Inst::Bitselect {
            a: c.idx()?,
            b: c.idx()?,
            mask: c.idx()?,
        },
        op::simd::SHUFFLE => Inst::Shuffle {
            lanes: dec_byte16(c)?,
            a: c.idx()?,
            b: c.idx()?,
        },
        op::simd::SWIZZLE => Inst::Swizzle {
            a: c.idx()?,
            b: c.idx()?,
        },
        op::simd::WIDTH_BYTES => Inst::SimdWidthBytes,
        other => return Err(DecodeError::BadOpcode(other)),
    })
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
        Terminator::ReturnCall { func, args } => {
            out.push(op::RETURN_CALL);
            write_uleb(out, *func as u64);
            write_idxs(out, args);
        }
        Terminator::ReturnCallIndirect { ty, idx, args } => {
            out.push(op::RETURN_CALL_INDIRECT);
            write_types(out, &ty.params);
            write_types(out, &ty.results);
            write_uleb(out, *idx as u64);
            write_idxs(out, args);
        }
        Terminator::Unreachable => out.push(op::UNREACHABLE),
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
fn un_base(ty: IntTy) -> u8 {
    match ty {
        IntTy::I32 => op::I32_UN,
        IntTy::I64 => op::I64_UN,
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

fn write_str(out: &mut Vec<u8>, s: &str) {
    write_uleb(out, s.len() as u64);
    out.extend_from_slice(s.as_bytes());
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
        ValType::V128 => op::T_V128,
        ValType::Ref => op::T_REF,
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
    // Data segments (§3a / D40), mirroring the encoder. Grow incrementally rather than
    // `with_capacity(ndata)` — `ndata` is attacker-influenced, and pre-reserving ~40 B/elem is a
    // ~40x allocation amplification (audit #7); every other decoder collection grows on demand.
    let ndata = c.count()?;
    let mut data = Vec::new();
    for _ in 0..ndata {
        let readonly = match c.byte()? {
            0 => false,
            1 => true,
            other => return Err(DecodeError::BadDataFlag(other)),
        };
        let offset = c.uleb()?;
        let len = c.count()?;
        let bytes = c.take(len)?.to_vec();
        data.push(Data {
            offset,
            readonly,
            bytes,
        });
    }
    // §7 import section (v2): mirrors the encoder. Grows on demand (the count is attacker-influenced).
    let nimports = c.count()?;
    let mut imports = Vec::new();
    for _ in 0..nimports {
        let name = c.str()?;
        let sig = FuncType {
            params: decode_types(&mut c)?,
            results: decode_types(&mut c)?,
        };
        imports.push(Import { name, sig });
    }
    // Export section (v3): mirrors the encoder. Grows on demand (the count is attacker-influenced).
    // Funcidx range + name uniqueness are the verifier's job, not the decoder's (it stays a pure,
    // fail-closed byte reader).
    let nexports = c.count()?;
    let mut exports = Vec::new();
    for _ in 0..nexports {
        let name = c.str()?;
        let func = c.uleb()? as FuncIdx;
        exports.push(Export { name, func });
    }
    let nfuncs = c.count()?;
    let mut funcs = Vec::new();
    for _ in 0..nfuncs {
        funcs.push(decode_func(&mut c)?);
    }
    // Optional strippable debug section (mirrors the encoder): present iff bytes remain after the
    // funcs. Strippable and untrusted-for-escape (§2a) — the verifier ignores it.
    let debug_info = if c.at_end() {
        None
    } else {
        Some(decode_debug_info(&mut c)?)
    };
    if !c.at_end() {
        return Err(DecodeError::TrailingBytes);
    }
    Ok(Module {
        funcs,
        memory,
        data,
        imports,
        exports,
        debug_info,
    })
}

/// Decode the §6 debug-info waist (untrusted input: counts are bounded, strings UTF-8-checked,
/// discriminants validated). Mirrors [`encode_debug_info`].
fn decode_debug_info(c: &mut Cursor) -> Result<DebugInfo, DecodeError> {
    let nfiles = c.count()?;
    let mut files = Vec::new();
    for _ in 0..nfiles {
        files.push(c.str()?);
    }
    let nlocs = c.count()?;
    let mut locs = Vec::new();
    for _ in 0..nlocs {
        locs.push(Loc {
            func: c.idx()?,
            block: c.idx()?,
            inst: c.idx()?,
            file: c.idx()?,
            line: c.idx()?,
            col: c.idx()?,
        });
    }
    let ntypes = c.count()?;
    let mut types = Vec::new();
    for _ in 0..ntypes {
        let t = match c.byte()? {
            0 => {
                let name = c.str()?;
                let encoding = match c.byte()? {
                    0 => Encoding::Signed,
                    1 => Encoding::Unsigned,
                    2 => Encoding::Float,
                    3 => Encoding::Bool,
                    b => return Err(DecodeError::BadEncoding(b)),
                };
                TypeDef::Base {
                    name,
                    encoding,
                    size: c.idx()?,
                }
            }
            1 => TypeDef::Pointer {
                name: c.str()?,
                pointee: c.idx()?,
                size: c.idx()?,
            },
            2 => TypeDef::Array {
                name: c.str()?,
                elem: c.idx()?,
                count: c.idx()?,
            },
            3 => {
                let name = c.str()?;
                let size = c.idx()?;
                let nfields = c.count()?;
                let mut fields = Vec::new();
                for _ in 0..nfields {
                    fields.push(Field {
                        name: c.str()?,
                        offset: c.idx()?,
                        ty: c.idx()?,
                    });
                }
                TypeDef::Aggregate { name, size, fields }
            }
            4 => TypeDef::Opaque {
                name: c.str()?,
                size: c.idx()?,
            },
            b => return Err(DecodeError::BadTypeDefKind(b)),
        };
        types.push(t);
    }
    let nvars = c.count()?;
    let mut vars = Vec::new();
    for _ in 0..nvars {
        let func = c.idx()?;
        let name = c.str()?;
        let ty = c.str()?;
        let loc = match c.byte()? {
            0 => VarLoc::Window { off: c.sleb()? },
            1 => VarLoc::Ssa { value: c.idx()? },
            2 => {
                let n = c.count()?;
                let mut locs = Vec::new();
                for _ in 0..n {
                    locs.push(SsaLoc {
                        block: c.idx()?,
                        inst: c.idx()?,
                        value: c.idx()?,
                    });
                }
                VarLoc::SsaList(locs)
            }
            3 => {
                let n = c.count()?;
                let mut base = Vec::new();
                for _ in 0..n {
                    base.push(SsaLoc {
                        block: c.idx()?,
                        inst: c.idx()?,
                        value: c.idx()?,
                    });
                }
                VarLoc::WindowVia {
                    base,
                    off: c.sleb()?,
                }
            }
            4 => VarLoc::Fixed { addr: c.uleb()? },
            b => return Err(DecodeError::BadVarLoc(b)),
        };
        let type_id = match c.byte()? {
            0 => None,
            1 => Some(c.idx()?),
            b => return Err(DecodeError::BadOptionFlag(b)),
        };
        let scope = match c.byte()? {
            0 => None,
            1 => Some((c.idx()?, c.idx()?)),
            b => return Err(DecodeError::BadOptionFlag(b)),
        };
        vars.push(VarInfo {
            func,
            name,
            ty,
            loc,
            type_id,
            scope,
        });
    }
    // Opaque per-producer rich blobs (§6): bounded count, then producer + length-prefixed bytes.
    let nblobs = c.count()?;
    let mut blobs = Vec::new();
    for _ in 0..nblobs {
        let producer = c.str()?;
        let len = c.count()?;
        let bytes = c.take(len)?.to_vec();
        blobs.push(ProducerBlob { producer, bytes });
    }
    // Function names (§6) — a trailing section: an artifact from before they existed ends right after
    // the blobs, so `at_end` ⇒ none (the field was appended last, after `blobs`, for this compat).
    let mut func_names = Vec::new();
    if !c.at_end() {
        let n = c.count()?;
        for _ in 0..n {
            func_names.push(FuncName {
                func: c.idx()?,
                name: c.str()?,
            });
        }
    }
    Ok(DebugInfo {
        files,
        locs,
        types,
        vars,
        blobs,
        func_names,
    })
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
        op::SIMD => decode_simd(c)?,
        op::CONST_I32 => Inst::ConstI32(c.sleb_i32()?),
        op::CONST_I64 => Inst::ConstI64(c.sleb()?),

        op::I32_UN..=op::I32_UN_END => int_un(IntTy::I32, b - op::I32_UN, c)?,
        op::I64_UN..=op::I64_UN_END => int_un(IntTy::I64, b - op::I64_UN, c)?,

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
        op::PTR_FROM_INT => Inst::PtrCast {
            to_int: false,
            a: c.idx()?,
        },
        op::PTR_TO_INT => Inst::PtrCast {
            to_int: true,
            a: c.idx()?,
        },
        op::PTR_ADD => Inst::PtrAdd {
            a: c.idx()?,
            b: c.idx()?,
        },
        op::FMA => {
            let ty = match c.byte()? {
                0 => FloatTy::F32,
                1 => FloatTy::F64,
                other => return Err(DecodeError::BadOpcode(other)),
            };
            Inst::Fma {
                ty,
                a: c.idx()?,
                b: c.idx()?,
                c: c.idx()?,
            }
        }
        op::FTOI_TRAP..=op::FTOI_TRAP_END => Inst::FToITrap {
            op: FToI::from_index(b - op::FTOI_TRAP).ok_or(DecodeError::BadOpcode(b))?,
            a: c.idx()?,
        },
        op::CALL_INDIRECT => Inst::CallIndirect {
            ty: FuncType {
                params: decode_types(c)?,
                results: decode_types(c)?,
            },
            idx: c.idx()?,
            args: decode_idxs(c)?,
        },
        op::CAP_CALL => Inst::CapCall {
            type_id: c.idx()?,
            op: c.idx()?,
            sig: FuncType {
                params: decode_types(c)?,
                results: decode_types(c)?,
            },
            handle: c.idx()?,
            args: decode_idxs(c)?,
        },
        op::CALL_IMPORT => Inst::CallImport {
            import: c.idx()?,
            sig: FuncType {
                params: decode_types(c)?,
                results: decode_types(c)?,
            },
            handle: c.idx()?,
            args: decode_idxs(c)?,
        },
        op::CAP_SELF_COUNT => Inst::CapSelfCount,
        op::CAP_SELF_GET => Inst::CapSelfGet { idx: c.idx()? },
        op::CAP_SELF_RESOLVE => Inst::CapSelfResolve {
            name_ptr: c.idx()?,
            name_len: c.idx()?,
        },
        op::CAP_SELF_LABEL => Inst::CapSelfLabel {
            handle: c.idx()?,
            buf_ptr: c.idx()?,
            buf_cap: c.idx()?,
        },
        op::VCPU_TLS_GET => Inst::VcpuTlsGet,
        op::VCPU_TLS_SET => Inst::VcpuTlsSet { val: c.idx()? },
        op::DURABLE_SHADOW_BASE => Inst::DurableShadowBase,

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

        op::ATOMIC_LOAD => Inst::AtomicLoad {
            ty: int_ty_from(c.byte()?, b)?,
            addr: c.idx()?,
            offset: c.uleb()?,
            order: ord_from(c.byte()?, b)?,
        },
        op::ATOMIC_STORE => Inst::AtomicStore {
            ty: int_ty_from(c.byte()?, b)?,
            addr: c.idx()?,
            value: c.idx()?,
            offset: c.uleb()?,
            order: ord_from(c.byte()?, b)?,
        },
        op::ATOMIC_RMW => Inst::AtomicRmw {
            ty: int_ty_from(c.byte()?, b)?,
            op: AtomicRmwOp::from_index(c.byte()?).ok_or(DecodeError::BadOpcode(b))?,
            addr: c.idx()?,
            value: c.idx()?,
            offset: c.uleb()?,
            order: ord_from(c.byte()?, b)?,
        },
        op::ATOMIC_CMPXCHG => Inst::AtomicCmpxchg {
            ty: int_ty_from(c.byte()?, b)?,
            addr: c.idx()?,
            expected: c.idx()?,
            replacement: c.idx()?,
            offset: c.uleb()?,
            order: ord_from(c.byte()?, b)?,
        },

        op::CONT_NEW => Inst::ContNew {
            func: c.idx()?,
            sp: c.idx()?,
        },
        op::CONT_RESUME => Inst::ContResume {
            k: c.idx()?,
            arg: c.idx()?,
        },
        op::SUSPEND => Inst::Suspend { value: c.idx()? },
        op::SETJMP => Inst::SetJmp { buf: c.idx()? },
        op::LONGJMP => Inst::LongJmp {
            buf: c.idx()?,
            val: c.idx()?,
        },
        op::GC_ROOTS => Inst::GcRoots {
            heap_lo: c.idx()?,
            heap_hi: c.idx()?,
            mask: c.idx()?,
            buf: c.idx()?,
            cap: c.idx()?,
        },

        op::THREAD_SPAWN => Inst::ThreadSpawn {
            func: c.idx()?,
            sp: c.idx()?,
            arg: c.idx()?,
        },
        op::THREAD_JOIN => Inst::ThreadJoin { handle: c.idx()? },
        op::ATOMIC_WAIT => Inst::MemoryWait {
            ty: int_ty_from(c.byte()?, b)?,
            addr: c.idx()?,
            expected: c.idx()?,
            timeout: c.idx()?,
        },
        op::ATOMIC_NOTIFY => Inst::MemoryNotify {
            addr: c.idx()?,
            count: c.idx()?,
        },
        op::ATOMIC_FENCE => Inst::AtomicFence {
            order: ord_from(c.byte()?, b)?,
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

fn int_un(ty: IntTy, index: u8, c: &mut Cursor) -> Result<Inst, DecodeError> {
    let op = IntUnOp::from_index(index).ok_or(DecodeError::BadOpcode(index))?;
    Ok(Inst::IntUn {
        ty,
        op,
        a: c.idx()?,
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
        op::RETURN_CALL => Terminator::ReturnCall {
            func: c.idx()?,
            args: decode_idxs(c)?,
        },
        op::RETURN_CALL_INDIRECT => Terminator::ReturnCallIndirect {
            ty: FuncType {
                params: decode_types(c)?,
                results: decode_types(c)?,
            },
            idx: c.idx()?,
            args: decode_idxs(c)?,
        },
        op::UNREACHABLE => Terminator::Unreachable,
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
        op::T_V128 => ValType::V128,
        op::T_REF => ValType::Ref,
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
            // At shift 63 only the sign bit (bit 63) still fits, so the final group's value bits must
            // be a pure sign extension: `0x00` (non-negative) or `0x7f` (negative). Any other value
            // has bits that do not fit `i64` — reject it as overflow (mirrors `uleb`'s `low > 1` check),
            // rather than silently dropping the over-wide bits.
            if shift == 63 && byte & 0x7f != 0x00 && byte & 0x7f != 0x7f {
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

    /// Read a length-prefixed UTF-8 string (an import name). The length is `count`-bounded
    /// (cannot exceed the remaining bytes), then the bytes must be valid UTF-8.
    fn str(&mut self) -> Result<String, DecodeError> {
        let n = self.count()?;
        let bytes = self.take(n)?;
        core::str::from_utf8(bytes)
            .map(str::to_owned)
            .map_err(|_| DecodeError::BadUtf8)
    }
}

#[cfg(test)]
mod debug_tests {
    use super::*;

    fn sample_debug() -> DebugInfo {
        DebugInfo {
            files: vec!["a.c".into(), "b.c".into()],
            locs: vec![
                Loc {
                    func: 0,
                    block: 1,
                    inst: 2,
                    file: 0,
                    line: 7,
                    col: 3,
                },
                Loc {
                    func: 0,
                    block: 1,
                    inst: 5,
                    file: 1,
                    line: 9,
                    col: 0,
                },
            ],
            // Every TypeDef variant + encoding.
            types: vec![
                TypeDef::Base {
                    name: "int".into(),
                    encoding: Encoding::Signed,
                    size: 4,
                },
                TypeDef::Base {
                    name: "double".into(),
                    encoding: Encoding::Float,
                    size: 8,
                },
                TypeDef::Pointer {
                    name: "int *".into(),
                    pointee: 0,
                    size: 8,
                },
                TypeDef::Array {
                    name: "int[4]".into(),
                    elem: 0,
                    count: 4,
                },
                TypeDef::Aggregate {
                    name: "struct Point".into(),
                    size: 8,
                    fields: vec![
                        Field {
                            name: "x".into(),
                            offset: 0,
                            ty: 0,
                        },
                        Field {
                            name: "y".into(),
                            offset: 4,
                            ty: 0,
                        },
                    ],
                },
                TypeDef::Opaque {
                    name: "void".into(),
                    size: 0,
                },
            ],
            // Window (incl. a negative offset → sleb) / Ssa locations, Some/None type ids.
            vars: vec![
                VarInfo {
                    func: 0,
                    name: "p".into(),
                    ty: "struct Point".into(),
                    loc: VarLoc::Window { off: 24 },
                    type_id: Some(4),
                    scope: Some((4, 9)),
                },
                VarInfo {
                    func: 0,
                    name: "i".into(),
                    ty: "int".into(),
                    loc: VarLoc::Ssa { value: 3 },
                    type_id: None,
                    scope: None,
                },
                VarInfo {
                    func: 0,
                    name: "neg".into(),
                    ty: "int".into(),
                    loc: VarLoc::Window { off: -8 },
                    type_id: Some(0),
                    scope: None,
                },
                // A location list (S2): the holding SSA value varies by pc.
                VarInfo {
                    func: 0,
                    name: "k".into(),
                    ty: "int".into(),
                    loc: VarLoc::SsaList(vec![
                        SsaLoc {
                            block: 0,
                            inst: 0,
                            value: 1,
                        },
                        SsaLoc {
                            block: 1,
                            inst: 2,
                            value: 4,
                        },
                    ]),
                    type_id: Some(0),
                    scope: None,
                },
                // A runtime-base window var (wasm/DWARF fbreg case): base loclist + a negative off.
                VarInfo {
                    func: 0,
                    name: "w".into(),
                    ty: "int".into(),
                    loc: VarLoc::WindowVia {
                        base: vec![
                            SsaLoc {
                                block: 0,
                                inst: 0,
                                value: 4,
                            },
                            SsaLoc {
                                block: 2,
                                inst: 1,
                                value: 4,
                            },
                        ],
                        off: -8,
                    },
                    type_id: Some(0),
                    scope: None,
                },
                // A module-scoped global at a fixed absolute window address (the GLOBAL_SCOPE
                // sentinel func + Fixed loc).
                VarInfo {
                    func: svm_ir::GLOBAL_SCOPE,
                    name: "counter".into(),
                    ty: "int".into(),
                    loc: VarLoc::Fixed { addr: 64 },
                    type_id: Some(0),
                    scope: None,
                },
            ],
            // An opaque per-producer rich blob (incl. non-UTF-8 / NUL bytes — verbatim DWARF).
            blobs: vec![ProducerBlob {
                producer: ".debug_info".into(),
                bytes: vec![0x00, 0x01, 0xff, 0x7f, 0x80, b'd', b'w'],
            }],
            func_names: vec![
                FuncName {
                    func: 0,
                    name: "compute".into(),
                },
                FuncName {
                    func: 2,
                    name: "main".into(),
                },
            ],
        }
    }

    fn module(debug_info: Option<DebugInfo>) -> Module {
        Module {
            funcs: vec![],
            memory: None,
            data: vec![],
            imports: vec![],
            exports: vec![],
            debug_info,
        }
    }

    #[test]
    fn debug_info_round_trips_through_binary() {
        let m = module(Some(sample_debug()));
        let back = decode_module(&encode_module(&m)).expect("decode");
        assert_eq!(
            back, m,
            "every files/locs/types/vars detail survives binary round-trip"
        );
    }

    #[test]
    fn no_debug_info_is_back_compatible_and_append_only() {
        // A module without debug info decodes to `None`, and its encoding is a strict prefix of the
        // same module *with* debug info — i.e. the section is appended after byte-identical output,
        // so existing import/debug-free blobs (and snapshot digests) are unchanged.
        let m_none = module(None);
        let bytes_none = encode_module(&m_none);
        assert_eq!(decode_module(&bytes_none).expect("decode"), m_none);

        let bytes_dbg = encode_module(&module(Some(sample_debug())));
        assert!(
            bytes_dbg.starts_with(&bytes_none),
            "debug section is appended after a byte-identical prefix"
        );
    }

    #[test]
    fn exports_round_trip_through_binary() {
        let mut m = module(None);
        m.funcs.push(Func {
            params: vec![],
            results: vec![],
            blocks: vec![Block {
                params: vec![],
                insts: vec![],
                term: Terminator::Return(vec![]),
            }],
        });
        m.exports = vec![
            Export {
                name: "main".to_string(),
                func: 0,
            },
            Export {
                name: "aux".to_string(),
                func: 0,
            },
        ];
        let decoded = decode_module(&encode_module(&m)).expect("decode");
        assert_eq!(decoded.exports, m.exports, "export section round-trips");
        assert_eq!(decoded, m);
    }

    #[test]
    fn rejects_a_truncated_debug_section() {
        // A truncated section must fail to decode, never panic (untrusted-input discipline): a
        // declared count/length runs past the bytes (a blob's length, a var/type count, …).
        let bytes = encode_module(&module(Some(sample_debug())));
        for cut in 1..=8 {
            let truncated = &bytes[..bytes.len() - cut];
            assert!(
                decode_module(truncated).is_err(),
                "truncating {cut} byte(s) must error, not panic"
            );
        }
    }
}

#[cfg(test)]
mod leb_tests {
    use super::*;

    /// `sleb` must round-trip every boundary value the encoder emits, including the 10-byte extremes.
    #[test]
    fn sleb_roundtrips_boundaries() {
        for v in [
            0i64,
            1,
            -1,
            63,
            -64,
            i32::MIN as i64,
            i32::MAX as i64,
            i64::MIN,
            i64::MAX,
        ] {
            let mut out = Vec::new();
            write_sleb(&mut out, v);
            let mut c = Cursor::new(&out);
            assert_eq!(c.sleb().unwrap(), v, "sleb round-trip for {v}");
        }
    }

    /// An over-wide final group (value bits beyond bit 63 that are not a pure sign extension) must be
    /// rejected as `LebOverflow`, not silently truncated — the L3 fidelity fix mirroring `uleb`.
    #[test]
    fn sleb_rejects_overwide_final_group() {
        // Nine 0x80 continuation bytes carry shift to 63; the 10th byte is the final group.
        let nine = [0x80u8; 9];
        // 0x00 / 0x7f are the only valid final groups (sign extension) — accepted.
        for last in [0x00u8, 0x7f] {
            let mut bytes = nine.to_vec();
            bytes.push(last);
            assert!(
                Cursor::new(&bytes).sleb().is_ok(),
                "valid final group {last:#x}"
            );
        }
        // Anything else has bits that don't fit i64 — rejected.
        for last in [0x01u8, 0x40, 0x3f, 0x7e] {
            let mut bytes = nine.to_vec();
            bytes.push(last);
            assert_eq!(
                Cursor::new(&bytes).sleb(),
                Err(DecodeError::LebOverflow),
                "over-wide final group {last:#x} must be rejected"
            );
        }
        // A continuation bit past the 10th byte is still overflow.
        let mut bytes = [0x80u8; 10].to_vec();
        bytes.push(0x00);
        assert_eq!(Cursor::new(&bytes).sleb(), Err(DecodeError::LebOverflow));
    }
}
