//! LLVM-bitcode → SVM-IR translator (the AOT LLVM on-ramp, D54). See `LLVM.md` for the
//! design, the decisions (binding, legalization), and the roadmap.
//!
//! **Trust:** this is an *untrusted frontend* (§2a). Everything it emits is re-checked by
//! `svm-verify`, so a translation bug is a clean error, never an escape. Correctness here is
//! a capability concern, not a safety one.
//!
//! **Pipeline (LLVM.md §4):** legalization is done *out of process* — `clang -O2 -emit-llvm`
//! runs `mem2reg`/SROA so scalars arrive in SSA registers (the §3a two-stack split for free)
//! and only address-taken `alloca`s remain. This crate ingests the legalized bitcode read-only
//! and walks it; it never runs an in-process pass manager.
//!
//! **Scope (Milestone 1, slices A–J):** multi-block scalar functions with stack memory, calls,
//! `switch`, globals, floats, indirect calls, struct aggregates, memory intrinsics, and by-value
//! aggregate args/returns.
//! - **A — control flow + scalar SSA.** The headline is the **SSA → block-argument conversion**:
//!   LLVM's dominance-based SSA (a value usable in any dominated block; φ-nodes merging across
//!   edges) becomes SVM's block-local form (§3a). Liveness makes every value live across a block's
//!   entry a parameter (φ-results included); each branch supplies the args — loops, joins, and
//!   critical edges all work without edge splitting. Integer arith/bitwise/shift/div-rem, `icmp`,
//!   `i1`/`i8`/`i16`/`i32`/`i64` `trunc`/`zext`/`sext`, `select`, `br`/`br_if`/`return`/`unreachable`.
//! - **B — the §3d data stack.** `alloca` → an `sp`-relative window frame slot, `load`/`store`
//!   (incl. narrow widths), `getelementptr` → address arithmetic. `undef`/`poison`/`null` → 0;
//!   `llvm.lifetime`/`dbg`/`assume` dropped. Pointers are `i64`.
//! - **C — calls + the threaded data-SP.** Every function takes a leading `sp` parameter (§3d),
//!   threaded as block-local index 0 of every block; a direct `call` passes the callee `sp +
//!   frame_size`, so activations get fresh frames and recursion is sound.
//! - **D — `switch`.** Lowered to a `br_table` biased by the minimum case value, gaps filled with
//!   the default edge (dense spans only; a too-sparse switch is `Unsupported`).
//! - **E — global variables.** Globals live low in the window as `data` segments (constants
//!   read-only, D40); a `@global` reference is its window address. The data stack starts just
//!   above them and grows up toward the window's guard region, so a stack overflow faults (§5)
//!   rather than corrupting globals. Int/array/string/zero initializers serialize to bytes.
//! - **F — floats.** `f32`/`f64` arithmetic/`fneg`/`fcmp`/`select`, the int↔float and f32↔f64
//!   conversions (`fptosi`/`sitofp`/`fpext`/`fptrunc`, float→int saturating per §3b), `bitcast`,
//!   and the common float math intrinsics (`fmuladd`/`fma` unfused, `sqrt`/`fabs`/`floor`/…) lowered
//!   inline. (Ordered/unordered fcmp collapse — the NaN corner is a documented fidelity gap.)
//! - **G — indirect calls.** Taking a function's address yields its §3c funcref index (widened to
//!   the `i64` pointer rep); an indirect `call` truncates the function-pointer value to the `i32`
//!   funcref and lowers to `call_indirect <sig>` (the runtime masks + type-id-checks it). The
//!   signature is the callee's function type plus the prepended data-SP, matching the IR signature.
//! - **H — aggregates (struct memory).** Struct layout (x86-64-SysV: natural field alignment +
//!   tail padding; named structs resolved); **struct GEP** (a constant field index → the field's
//!   byte offset); struct `alloca`s (struct-sized frame slots) and struct global initializers
//!   serialize with field padding. Covers structs accessed via pointers/locals/globals — *not* the
//!   by-value pass/return ABI (`sret`/`byval`), which is a follow-up.
//! - **I — memory intrinsics.** `llvm.memcpy`/`memmove`/`memset` (constant length) lower to inline
//!   chunked load/stores (widest-first 8/4/2/1, the `svm-wasm` plan); copies load-all-then-store-all
//!   (overlap-safe); `memset` replicates the fill byte across an `i64`. The data stack is page-aligned
//!   above the globals so a stack write never faults on a read-only global's page (D40).
//! - **J — by-value aggregates (`sret`/`byval`).** Works with **no dedicated code**: clang does the
//!   x86-64-SysV classification *in the IR* — a small struct is coerced to scalar register(s)
//!   (`{i32,i32}`→`i64`, `{int×3}`→`(i64,i32)`, SSE→`double`s), a large one passes via a `byval`/
//!   `sret` pointer (the caller `alloca`s + `memcpy`s + passes the pointer). So slices A–I (scalar
//!   params, memory, calls, struct GEP, memcpy) already cover it; this slice is the test lock-in.
//!
//! Out of the current subset (clean [`Error::Unsupported`]): variable-length `mem*`, libc/math
//! *function* calls (e.g. `sqrt` with errno), pointer-valued globals (relocations — incl.
//! function-pointer tables), SIMD vectors, `i33`.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use llvm_ir::instruction::Instruction;
use llvm_ir::terminator::Terminator as LTerm;
use llvm_ir::types::{FPType, Type, Typed, Types};
use llvm_ir::{constant::Constant, constant::Float, BasicBlock, Function, Module as LModule};
use llvm_ir::{FPPredicate, IntPredicate, Name, Operand};

use svm_ir::{
    BinOp, Block, CastOp, CmpOp, ConvOp, FBinOp, FCmpOp, FToI, FUnOp, FloatTy, Func, IToF, Inst,
    IntTy, Module, Terminator, ValIdx, ValType,
};

/// Why a translation could not be produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// A construct outside the frozen MVP subset. Fail-closed by design (LLVM.md §2/§8):
    /// we never emit IR we can't stand behind. Widen the subset, never silently mis-translate.
    Unsupported(String),
    /// libLLVM could not parse the bitcode (e.g. produced by an off-version LLVM — we pin 18).
    Parse(String),
}

/// Shorthand for the fail-closed chokepoint (the `svm-wasm` `unsup(...)` analog).
fn unsup<T>(what: impl Into<String>) -> Result<T, Error> {
    Err(Error::Unsupported(what.into()))
}

/// The translation result: the verifier-checkable module plus the **initial data-SP** the entry
/// must be invoked with (§3d). The data stack starts just above the globals and grows up toward
/// the window's guard region, so an overflow faults rather than corrupting globals.
#[derive(Debug)]
pub struct Translated {
    pub module: Module,
    /// The value to pass as the entry's first (`sp`) argument.
    pub entry_sp: u64,
}

/// Translate a legalized LLVM bitcode file (`*.bc`). The bitcode must come from the pinned LLVM
/// (18); off-version input is an [`Error::Parse`].
pub fn translate_bc_path(path: impl AsRef<Path>) -> Result<Translated, Error> {
    let m = LModule::from_bc_path(path).map_err(Error::Parse)?;
    translate(&m)
}

/// Translate an already-parsed `llvm-ir` module.
pub fn translate(m: &LModule) -> Result<Translated, Error> {
    // Pass 0: assign each *defined* function an IR index (its position among defined functions),
    // so a `call` can resolve its target by name. Declaration-only functions (extern/intrinsic
    // prototypes) have no body and are skipped — a call to one needs import support (a later slice).
    let defined: Vec<&Function> = m
        .functions
        .iter()
        .filter(|f| !f.basic_blocks.is_empty())
        .collect();
    let mut name2idx: HashMap<String, u32> = HashMap::new();
    for (i, f) in defined.iter().enumerate() {
        name2idx.insert(f.name.clone(), i as u32);
    }
    // Globals live low (from `DATA_BASE`); the data stack starts just above them.
    let (globals, data, globals_end) = globals_layout(m)?;
    // Page-align the data stack above the globals so it never shares a page with a *read-only*
    // global (D40 protects RO segments page-granularly — a stack write into a shared page would
    // fault). 16 KiB covers the largest common page size (macOS/aarch64). (A read-only and a
    // writable global sharing a page is a separate latent issue — page-isolating those is a follow-up.)
    let entry_sp = globals_end.div_ceil(STACK_PAGE) * STACK_PAGE;

    let mut funcs = Vec::with_capacity(defined.len());
    let mut any_frame = false; // does any function use the data stack (`alloca`)?
    for f in &defined {
        let (func, frame_size) = translate_func(f, &m.types, &name2idx, &globals)?;
        any_frame |= frame_size > 0;
        funcs.push(func);
    }
    // A window is declared if any function uses the data stack *or* the module has globals. Layout
    // (§3d): globals `[DATA_BASE, globals_end)` low, then the data stack from `entry_sp` growing up.
    // `mapped` covers the globals plus a stack reserve; the runtime leaves a faulting guard beyond
    // `mapped` (reserved > mapped), so a stack overflow faults (detect-and-kill, §5) instead of
    // corrupting the globals below it.
    let need_window = any_frame || !globals.is_empty();
    let memory = need_window.then(|| {
        let top = if any_frame {
            entry_sp + STACK_RESERVE
        } else {
            globals_end
        }
        .max(1);
        let log2 = (64 - (top - 1).leading_zeros()) as u8;
        svm_ir::Memory { size_log2: log2 }
    });
    Ok(Translated {
        module: Module {
            funcs,
            memory,
            data,
            // §7 named capability imports — the LLVM on-ramp emits none yet.
            imports: Vec::new(),
            // Debug info — the LLVM on-ramp will map `!DILocation`/`dbg.value` into the §6 waist
            // (DEBUGGING.md D-DBG-7); none yet.
            debug_info: None,
        },
        entry_sp,
    })
}

/// The low window offset where globals begin (kept off a null-like 0).
const DATA_BASE: u64 = 16;
/// The page granularity the data stack is aligned to above the globals (≥ the largest OS page so
/// a stack write never lands in a read-only global's protected page, D40).
const STACK_PAGE: u64 = 16384;
/// The data-stack reserve (bytes) above the entry SP before the guard region — a stack overflow
/// past this faults rather than escaping the window.
const STACK_RESERVE: u64 = 1 << 20;

/// The data-SP's synthetic value id — threaded as block-local index 0 of *every* block (§3d),
/// like chibicc's `v0`. It carries no LLVM name; it is supplied positionally.
const SP: ValueId = usize::MAX;

/// An LLVM value/global name as a `String` key (named or numbered).
fn name_str(n: &Name) -> String {
    match n {
        Name::Name(s) => s.to_string(),
        Name::Number(k) => k.to_string(),
    }
}

/// Serialize a constant initializer to its little-endian window bytes (the §3d/x86-64 layout).
/// Integers (incl. the `i8`s of a C string), floats, arrays, structs (with field padding), and
/// zero/null aggregates are covered; pointer-valued globals (relocations) are a later slice.
fn const_bytes(c: &Constant, types: &Types) -> Result<Vec<u8>, Error> {
    match c {
        Constant::Int { bits, value } if *bits <= 64 => {
            let n = (*bits as usize).div_ceil(8).max(1);
            Ok(value.to_le_bytes()[..n].to_vec())
        }
        Constant::Float(Float::Single(f)) => Ok(f.to_bits().to_le_bytes().to_vec()),
        Constant::Float(Float::Double(d)) => Ok(d.to_bits().to_le_bytes().to_vec()),
        Constant::Array { elements, .. } | Constant::Vector(elements) => {
            let mut out = Vec::new();
            for e in elements {
                out.extend(const_bytes(e.as_ref(), types)?);
            }
            Ok(out)
        }
        // A struct: place each field at its laid-out offset, zero-filling alignment padding.
        Constant::Struct {
            values, is_packed, ..
        } => {
            let fields: Vec<llvm_ir::TypeRef> = values.iter().map(|v| v.get_type(types)).collect();
            let (offsets, size, _) = struct_layout(&fields, *is_packed, types)?;
            let mut out = vec![0u8; size as usize];
            for (v, &off) in values.iter().zip(&offsets) {
                let b = const_bytes(v.as_ref(), types)?;
                out[off as usize..off as usize + b.len()].copy_from_slice(&b);
            }
            Ok(out)
        }
        Constant::AggregateZero(t) | Constant::Null(t) => {
            Ok(vec![0u8; type_size(t.as_ref(), types)? as usize])
        }
        other => unsup(format!("global initializer {other:?}")),
    }
}

/// The result of [`globals_layout`]: name → window address, the `data` segments to emit, and the
/// globals region's end offset (for window sizing).
type Globals = (HashMap<String, u64>, Vec<svm_ir::Data>, u64);

/// Lay out the module's global variables in the window's globals region (from [`GLOBALS_BASE`],
/// each natural-aligned), returning the name → window-address map, the `data` segments to emit
/// (constants read-only, §3a/D40; all-zero/BSS globals just reserve space in the zero-init
/// window), and the region's end (for window sizing).
fn globals_layout(m: &LModule) -> Result<Globals, Error> {
    let mut addr = HashMap::new();
    let mut segs = Vec::new();
    let mut off = DATA_BASE;
    for g in &m.global_vars {
        let (bytes, size) = match &g.initializer {
            Some(init) => {
                let b = const_bytes(init.as_ref(), &m.types)?;
                let n = b.len() as u64;
                (Some(b), n.max(1))
            }
            None => (None, type_size(g.ty.as_ref(), &m.types)?.max(1)),
        };
        let align = (g.alignment as u64).max(1);
        off = off.div_ceil(align) * align;
        addr.insert(name_str(&g.name), off);
        // Emit a segment only for non-zero initialized data (the window is zero-init, so BSS and
        // explicit zeros need none). A read-only segment is protected (D40), so a guest write faults.
        if let Some(b) = bytes {
            if g.is_constant || b.iter().any(|&x| x != 0) {
                segs.push(svm_ir::Data {
                    offset: off,
                    readonly: g.is_constant,
                    bytes: b,
                });
            }
        }
        off += size;
    }
    Ok((addr, segs, off))
}

/// Map an LLVM type to an SVM value type. Narrow integers collapse to `i32` (§3b: `i8`/`i16`
/// are memory widths only, not SSA value types); `i64` stays `i64`. Non-byte widths (`i33`,
/// `i128`), floats, pointers, and aggregates are outside the slice-A subset.
fn val_type(ty: &Type) -> Result<ValType, Error> {
    match ty {
        Type::IntegerType { bits } if *bits <= 32 => Ok(ValType::I32),
        Type::IntegerType { bits } if *bits == 64 => Ok(ValType::I64),
        Type::IntegerType { bits } => unsup(format!("integer width i{bits} (Milestone 1+)")),
        // Pointers are an erasable refinement of `i64` (§3a/§10) — a window offset.
        Type::PointerType { .. } => Ok(ValType::I64),
        Type::FPType(FPType::Single) => Ok(ValType::F32),
        Type::FPType(FPType::Double) => Ok(ValType::F64),
        other => unsup(format!("type {other} (Milestone 1+)")),
    }
}

/// The `FloatTy` (`f32`/`f64`) of a float-typed SVM value.
fn float_ty(v: ValType) -> Result<FloatTy, Error> {
    match v {
        ValType::F32 => Ok(FloatTy::F32),
        ValType::F64 => Ok(FloatTy::F64),
        other => unsup(format!("non-float type {}", other.as_str())),
    }
}

/// The saturating float→int conversion variant (§3b: `trunc_sat`, total — out-of-range saturates
/// rather than the C UB of `fptosi`).
fn ftoi_op(src: FloatTy, dst: IntTy, signed: bool) -> FToI {
    match (src, dst, signed) {
        (FloatTy::F32, IntTy::I32, true) => FToI::F32I32S,
        (FloatTy::F32, IntTy::I32, false) => FToI::F32I32U,
        (FloatTy::F32, IntTy::I64, true) => FToI::F32I64S,
        (FloatTy::F32, IntTy::I64, false) => FToI::F32I64U,
        (FloatTy::F64, IntTy::I32, true) => FToI::F64I32S,
        (FloatTy::F64, IntTy::I32, false) => FToI::F64I32U,
        (FloatTy::F64, IntTy::I64, true) => FToI::F64I64S,
        (FloatTy::F64, IntTy::I64, false) => FToI::F64I64U,
    }
}

/// The int→float conversion variant.
fn itof_op(src: IntTy, dst: FloatTy, signed: bool) -> IToF {
    match (src, dst, signed) {
        (IntTy::I32, FloatTy::F32, true) => IToF::I32F32S,
        (IntTy::I32, FloatTy::F32, false) => IToF::I32F32U,
        (IntTy::I64, FloatTy::F32, true) => IToF::I64F32S,
        (IntTy::I64, FloatTy::F32, false) => IToF::I64F32U,
        (IntTy::I32, FloatTy::F64, true) => IToF::I32F64S,
        (IntTy::I32, FloatTy::F64, false) => IToF::I32F64U,
        (IntTy::I64, FloatTy::F64, true) => IToF::I64F64S,
        (IntTy::I64, FloatTy::F64, false) => IToF::I64F64U,
    }
}

/// Map an LLVM float compare predicate to the SVM op. Ordered and unordered forms collapse to the
/// same op (the NaN-distinguishing `o`/`u` corner is a documented fidelity gap until needed);
/// `ord`/`uno`/`true`/`false` are `Unsupported`.
fn fcmp_op(p: FPPredicate) -> Result<FCmpOp, Error> {
    use FPPredicate as P;
    Ok(match p {
        P::OEQ | P::UEQ => FCmpOp::Eq,
        P::ONE | P::UNE => FCmpOp::Ne,
        P::OLT | P::ULT => FCmpOp::Lt,
        P::OLE | P::ULE => FCmpOp::Le,
        P::OGT | P::UGT => FCmpOp::Gt,
        P::OGE | P::UGE => FCmpOp::Ge,
        other => return unsup(format!("float compare predicate {other:?}")),
    })
}

/// The size in bytes of an LLVM type (the SysV/§3d layout for the subset we lower). Used to lay
/// out `alloca` frames and compute GEP strides. SIMD vectors and odd scalars are a clean
/// `Unsupported` until a later slice.
fn type_size(ty: &Type, types: &Types) -> Result<u64, Error> {
    match ty {
        Type::IntegerType { bits } => Ok((*bits as u64).div_ceil(8).max(1)),
        Type::PointerType { .. } => Ok(8),
        Type::FPType(FPType::Single) => Ok(4),
        Type::FPType(FPType::Double) => Ok(8),
        Type::ArrayType {
            element_type,
            num_elements,
        } => Ok(*num_elements as u64 * type_size(element_type.as_ref(), types)?),
        Type::StructType { .. } | Type::NamedStructType { .. } => {
            let (fields, packed) = resolve_struct(ty, types)?;
            Ok(struct_layout(&fields, packed, types)?.1)
        }
        other => unsup(format!("size of type {other} (Milestone 1+)")),
    }
}

/// The natural alignment (bytes) of an LLVM type — scalar align = size; array = element align;
/// struct = max field align (1 if packed).
fn type_align(ty: &Type, types: &Types) -> Result<u64, Error> {
    match ty {
        Type::IntegerType { .. } | Type::PointerType { .. } | Type::FPType(_) => {
            type_size(ty, types)
        }
        Type::ArrayType { element_type, .. } => type_align(element_type.as_ref(), types),
        Type::StructType { .. } | Type::NamedStructType { .. } => {
            let (fields, packed) = resolve_struct(ty, types)?;
            Ok(struct_layout(&fields, packed, types)?.2)
        }
        other => unsup(format!("align of type {other} (Milestone 1+)")),
    }
}

/// Resolve a struct type (literal or named) to its field types + packed flag.
fn resolve_struct(ty: &Type, types: &Types) -> Result<(Vec<llvm_ir::TypeRef>, bool), Error> {
    match ty {
        Type::StructType {
            element_types,
            is_packed,
        } => Ok((element_types.clone(), *is_packed)),
        Type::NamedStructType { name } => match types.named_struct_def(name) {
            Some(llvm_ir::types::NamedStructDef::Defined(t)) => resolve_struct(t.as_ref(), types),
            _ => unsup(format!("opaque/undefined struct `{name}`")),
        },
        other => unsup(format!("not a struct: {other}")),
    }
}

/// The x86-64-SysV/§3d struct layout: each field's byte offset, the total size, and the alignment.
/// Fields align naturally (offset rounded up to the field's alignment); the struct's size is padded
/// to its own alignment. A packed struct skips all padding.
fn struct_layout(
    fields: &[llvm_ir::TypeRef],
    packed: bool,
    types: &Types,
) -> Result<(Vec<u64>, u64, u64), Error> {
    let mut offsets = Vec::with_capacity(fields.len());
    let mut off = 0u64;
    let mut align = 1u64;
    for f in fields {
        let fsz = type_size(f.as_ref(), types)?;
        let fal = if packed {
            1
        } else {
            type_align(f.as_ref(), types)?
        };
        off = off.div_ceil(fal) * fal;
        offsets.push(off);
        off += fsz;
        align = align.max(fal);
    }
    if !packed {
        off = off.div_ceil(align) * align; // tail padding to the struct's alignment
    }
    Ok((offsets, off.max(1), align))
}

/// The integer bit width of an LLVM type, or `None` if it is not an integer.
fn int_bits(ty: &Type) -> Option<u32> {
    match ty {
        Type::IntegerType { bits } => Some(*bits),
        _ => None,
    }
}

/// The `IntTy` (`i32`/`i64`) a value of this SVM type is computed at.
fn int_ty(v: ValType) -> Result<IntTy, Error> {
    match v {
        ValType::I32 => Ok(IntTy::I32),
        ValType::I64 => Ok(IntTy::I64),
        other => unsup(format!("non-integer type {}", other.as_str())),
    }
}

/// A unique id for every SSA value in a function (parameters, then each block's φ-results and
/// instruction results, in scan order). The translation works in terms of these; SVM block-local
/// indices are derived per block.
type ValueId = usize;

/// Per-function scan tables: the value↔id maps and the block index map.
struct Scan {
    /// LLVM value name → its `ValueId`.
    name2id: HashMap<Name, ValueId>,
    /// `ValueId` → its SVM type.
    ty: Vec<ValType>,
    /// `ValueId` → the block it is defined in (parameters are defined in the entry block, 0).
    def_block: Vec<usize>,
    /// Block name → block index (entry is 0).
    block_idx: HashMap<Name, usize>,
    /// Block index → block name (for looking up φ incoming-by-predecessor).
    block_name: Vec<Name>,
}

fn translate_func(
    f: &Function,
    types: &Types,
    name2idx: &HashMap<String, u32>,
    globals: &HashMap<String, u64>,
) -> Result<(Func, u64), Error> {
    if f.is_var_arg {
        return unsup(format!("varargs function `{}`", f.name));
    }
    if f.basic_blocks.is_empty() {
        return unsup(format!("declaration-only function `{}`", f.name));
    }
    // The IR signature prepends the data-SP (§3d): `(sp:i64, c-params…) -> results`. The data-SP
    // is threaded as block-local index 0 of every block; a call passes `sp + frame_size`.
    let mut params: Vec<ValType> = vec![ValType::I64];
    for p in &f.parameters {
        params.push(val_type(&p.ty)?);
    }
    let results = match f.return_type.as_ref() {
        Type::VoidType => Vec::new(),
        t => vec![val_type(t)?],
    };

    let scan = scan_func(f, types)?;
    let live_in = liveness(f, &scan)?;
    let block_params = block_params(f, &scan, &live_in);
    let (frame, frame_size) = frame_layout(f, &scan, types)?;

    let mut blocks = Vec::with_capacity(f.basic_blocks.len());
    for (bi, bb) in f.basic_blocks.iter().enumerate() {
        blocks.push(translate_block(
            bb,
            bi,
            f,
            types,
            &scan,
            &block_params,
            &frame,
            frame_size,
            name2idx,
            globals,
        )?);
    }
    Ok((
        Func {
            params,
            results,
            blocks,
        },
        frame_size,
    ))
}

/// Lay out every `alloca`'s data-stack slot at a `sp`-relative offset (from 0, natural-aligned),
/// returning the `alloca`-id → offset map and the frame size (16-aligned, so a callee's frame —
/// at `sp + frame_size` — stays aligned). A dynamic (`num_elements` non-constant) `alloca` is a
/// clean `Unsupported` for now.
fn frame_layout(
    f: &Function,
    s: &Scan,
    types: &Types,
) -> Result<(HashMap<ValueId, u64>, u64), Error> {
    let mut frame = HashMap::new();
    let mut off = 0u64;
    for bb in &f.basic_blocks {
        for instr in &bb.instrs {
            if let Instruction::Alloca(a) = instr {
                let n = match &a.num_elements {
                    Operand::ConstantOperand(c) => match c.as_ref() {
                        Constant::Int { value, .. } => *value,
                        _ => return unsup("dynamic alloca (non-constant element count)"),
                    },
                    _ => return unsup("dynamic alloca (non-constant element count)"),
                };
                let size = type_size(a.allocated_type.as_ref(), types)?.saturating_mul(n);
                // Natural alignment: the larger of the type's alignment and the `alloca`'s declared
                // alignment; round the running offset up to it.
                let align = type_align(a.allocated_type.as_ref(), types)?
                    .max(a.alignment as u64)
                    .max(1);
                off = off.div_ceil(align) * align;
                if let Some(&vid) = s.name2id.get(&a.dest) {
                    frame.insert(vid, off);
                }
                off += size.max(1);
            }
        }
    }
    Ok((frame, off.div_ceil(16) * 16))
}

/// Pass 1a: assign a `ValueId` to every SSA value (parameters first, then per block the φ-results
/// and instruction results), recording each one's SVM type and defining block. Also validates that
/// every instruction is in the slice-A subset (so later passes can assume support).
fn scan_func(f: &Function, types: &Types) -> Result<Scan, Error> {
    let mut s = Scan {
        name2id: HashMap::new(),
        ty: Vec::new(),
        def_block: Vec::new(),
        block_idx: HashMap::new(),
        block_name: Vec::new(),
    };
    for (bi, bb) in f.basic_blocks.iter().enumerate() {
        s.block_idx.insert(bb.name.clone(), bi);
        s.block_name.push(bb.name.clone());
    }
    // Parameters are values defined at entry.
    for p in &f.parameters {
        let id = s.ty.len();
        s.name2id.insert(p.name.clone(), id);
        s.ty.push(val_type(&p.ty)?);
        s.def_block.push(0);
    }
    for (bi, bb) in f.basic_blocks.iter().enumerate() {
        if bi != 0 {
            // (entry φ is impossible — entry has no predecessors)
        }
        for instr in &bb.instrs {
            // Validate support + collect uses now so liveness can rely on it.
            let _ = local_uses(instr)?;
            if let Some(dest) = instr.try_get_result() {
                let id = s.ty.len();
                s.name2id.insert(dest.clone(), id);
                s.ty.push(val_type(instr.get_type(types).as_ref())?);
                s.def_block.push(bi);
            }
        }
        term_local_uses(&bb.term)?; // validate terminator support
    }
    Ok(s)
}

/// The local (non-constant) value operands an instruction *uses*, and — as a side effect — the
/// slice-A support check (an unsupported instruction is a fail-closed [`Error::Unsupported`]).
/// φ incoming values are **edge** uses (counted per-predecessor in liveness), so a `Phi` reports
/// no direct uses here.
fn local_uses(instr: &Instruction) -> Result<Vec<Name>, Error> {
    use Instruction as I;
    let locals = |ops: &[&Operand]| -> Vec<Name> {
        ops.iter()
            .filter_map(|o| match o {
                Operand::LocalOperand { name, .. } => Some(name.clone()),
                _ => None,
            })
            .collect()
    };
    let r = match instr {
        I::Add(x) => locals(&[&x.operand0, &x.operand1]),
        I::Sub(x) => locals(&[&x.operand0, &x.operand1]),
        I::Mul(x) => locals(&[&x.operand0, &x.operand1]),
        I::UDiv(x) => locals(&[&x.operand0, &x.operand1]),
        I::SDiv(x) => locals(&[&x.operand0, &x.operand1]),
        I::URem(x) => locals(&[&x.operand0, &x.operand1]),
        I::SRem(x) => locals(&[&x.operand0, &x.operand1]),
        I::And(x) => locals(&[&x.operand0, &x.operand1]),
        I::Or(x) => locals(&[&x.operand0, &x.operand1]),
        I::Xor(x) => locals(&[&x.operand0, &x.operand1]),
        I::Shl(x) => locals(&[&x.operand0, &x.operand1]),
        I::LShr(x) => locals(&[&x.operand0, &x.operand1]),
        I::AShr(x) => locals(&[&x.operand0, &x.operand1]),
        I::ICmp(x) => locals(&[&x.operand0, &x.operand1]),
        I::Select(x) => locals(&[&x.condition, &x.true_value, &x.false_value]),
        I::Trunc(x) => locals(&[&x.operand]),
        I::ZExt(x) => locals(&[&x.operand]),
        I::SExt(x) => locals(&[&x.operand]),
        // Floats.
        I::FAdd(x) => locals(&[&x.operand0, &x.operand1]),
        I::FSub(x) => locals(&[&x.operand0, &x.operand1]),
        I::FMul(x) => locals(&[&x.operand0, &x.operand1]),
        I::FDiv(x) => locals(&[&x.operand0, &x.operand1]),
        I::FCmp(x) => locals(&[&x.operand0, &x.operand1]),
        I::FNeg(x) => locals(&[&x.operand]),
        I::FPToSI(x) => locals(&[&x.operand]),
        I::FPToUI(x) => locals(&[&x.operand]),
        I::SIToFP(x) => locals(&[&x.operand]),
        I::UIToFP(x) => locals(&[&x.operand]),
        I::FPExt(x) => locals(&[&x.operand]),
        I::FPTrunc(x) => locals(&[&x.operand]),
        I::BitCast(x) => locals(&[&x.operand]),
        // Memory (§3d two-stack: address-taken locals live on the in-window data stack).
        I::Alloca(a) => locals(&[&a.num_elements]),
        I::Load(l) => locals(&[&l.address]),
        I::Store(st) => locals(&[&st.address, &st.value]),
        I::GetElementPtr(g) => {
            let mut v = locals(&[&g.address]);
            v.extend(g.indices.iter().filter_map(|o| match o {
                Operand::LocalOperand { name, .. } => Some(name.clone()),
                _ => None,
            }));
            v
        }
        // A droppable intrinsic (`llvm.lifetime`/`dbg`/`assume`) contributes no real uses — it is
        // a no-op. A real call uses its argument operands plus — for an indirect call — the
        // function-pointer callee; the data-SP it threads is the §3d positional parameter, not an
        // LLVM value, so it is not counted here.
        I::Call(c) if is_droppable_call(c) => Vec::new(),
        I::Call(c) => {
            let mut v: Vec<Name> = match c.function.as_ref().right() {
                Some(Operand::LocalOperand { name, .. }) => vec![name.clone()],
                _ => Vec::new(),
            };
            v.extend(c.arguments.iter().filter_map(|(o, _)| match o {
                Operand::LocalOperand { name, .. } => Some(name.clone()),
                _ => None,
            }));
            v
        }
        // A φ's operands are edge uses, handled in liveness via `PhiUses`.
        I::Phi(_) => Vec::new(),
        other => return unsup(format!("instruction {other:?}")),
    };
    Ok(r)
}

/// The name of a direct call's target (a `@global` function reference). An indirect call (the
/// callee is a computed value) or inline asm is a clean `Unsupported` for now.
/// The SVM signature of an indirect call's callee — the function type plus the prepended data-SP
/// param (§3d), so the runtime type-id check matches the callee's IR signature (§3c).
fn indirect_sig(c: &llvm_ir::instruction::Call) -> Result<svm_ir::FuncType, Error> {
    match c.function_ty.as_ref() {
        Type::FuncType {
            result_type,
            param_types,
            is_var_arg,
        } => {
            if *is_var_arg {
                return unsup("indirect varargs call");
            }
            let mut params = vec![ValType::I64]; // the prepended data-SP
            for p in param_types {
                params.push(val_type(p.as_ref())?);
            }
            let results = match result_type.as_ref() {
                Type::VoidType => Vec::new(),
                t => vec![val_type(t)?],
            };
            Ok(svm_ir::FuncType { params, results })
        }
        other => unsup(format!("indirect call through non-function type {other}")),
    }
}

/// The callee name of a direct call, or `None` for an indirect/inline-asm call.
fn callee_name(c: &llvm_ir::instruction::Call) -> Option<String> {
    match c.function.as_ref().right()? {
        Operand::ConstantOperand(cr) => match cr.as_ref() {
            Constant::GlobalReference {
                name: Name::Name(s),
                ..
            } => Some(s.to_string()),
            _ => None,
        },
        _ => None,
    }
}

/// The largest constant byte length we unroll a `memcpy`/`memset` into chunked load/stores; a
/// larger one would need a runtime loop (synthetic blocks), a later slice. clang's struct/array
/// bulk ops carry small constant sizes.
const MAX_MEM_UNROLL: u64 = 4096;

/// Split `len` bytes into `(offset, width)` chunks, widest first (8/4/2/1) — the same unroll plan
/// `svm-wasm` uses for `memory.copy`/`fill`.
fn mem_chunks(len: u64) -> Vec<(u64, u8)> {
    let mut out = Vec::new();
    let mut off = 0u64;
    let mut rem = len;
    for w in [8u64, 4, 2, 1] {
        while rem >= w {
            out.push((off, w as u8));
            off += w;
            rem -= w;
        }
    }
    out
}

fn load_w(w: u8) -> svm_ir::LoadOp {
    use svm_ir::LoadOp as L;
    match w {
        8 => L::I64,
        4 => L::I32,
        2 => L::I32_16U,
        _ => L::I32_8U,
    }
}

fn store_w(w: u8) -> svm_ir::StoreOp {
    use svm_ir::StoreOp as S;
    match w {
        8 => S::I64,
        4 => S::I32,
        2 => S::I32_16,
        _ => S::I32_8,
    }
}

/// The constant integer value of an operand, if it is one.
fn const_int(op: &Operand) -> Option<u64> {
    match op {
        Operand::ConstantOperand(c) => match c.as_ref() {
            Constant::Int { value, .. } => Some(*value),
            _ => None,
        },
        _ => None,
    }
}

/// Lower `llvm.memcpy`/`memmove`/`memset` (constant length) to inline chunked load/stores, the way
/// `svm-wasm` lowers `memory.copy`/`fill`. Copies **load all chunks then store all** (overlap-safe,
/// so `memmove` and `memcpy` share a path); `memset` replicates the fill byte across an `i64` and
/// stores it chunk-wide. Returns `Ok(true)` if it handled a (void) mem intrinsic, `Ok(false)`
/// otherwise. A variable or too-large length is a clean `Unsupported`.
fn lower_mem_intrinsic(ctx: &mut BlockCtx, c: &llvm_ir::instruction::Call) -> Result<bool, Error> {
    let Some(name) = callee_name(c) else {
        return Ok(false);
    };
    let is_copy = name.starts_with("llvm.memcpy") || name.starts_with("llvm.memmove");
    let is_set = name.starts_with("llvm.memset");
    if !is_copy && !is_set {
        return Ok(false);
    }
    let args: Vec<&Operand> = c.arguments.iter().map(|(a, _)| a).collect();
    let len = const_int(args[2])
        .ok_or_else(|| Error::Unsupported("variable-length mem intrinsic".into()))?;
    if len == 0 {
        return Ok(true);
    }
    if len > MAX_MEM_UNROLL {
        return unsup(format!(
            "mem intrinsic length {len} > {MAX_MEM_UNROLL} (needs a loop)"
        ));
    }
    let chunks = mem_chunks(len);
    if is_copy {
        let dst = ctx.operand(args[0])?;
        let src = ctx.operand(args[1])?;
        // Load every chunk first (overlap-safe), then store them all.
        let loaded: Vec<(u64, u8, ValIdx)> = chunks
            .iter()
            .map(|&(off, w)| {
                let v = ctx.push(Inst::Load {
                    op: load_w(w),
                    addr: src,
                    offset: off,
                    align: 0,
                });
                (off, w, v)
            })
            .collect();
        for (off, w, v) in loaded {
            ctx.push_effect(Inst::Store {
                op: store_w(w),
                addr: dst,
                value: v,
                offset: off,
                align: 0,
            });
        }
    } else {
        let dst = ctx.operand(args[0])?;
        let val = ctx.operand(args[1])?; // i8 fill, carried as i32
                                         // rep64 = (val & 0xFF) * 0x0101010101010101 — the fill byte replicated across 8 bytes.
        let mask = ctx.push(Inst::ConstI32(0xFF));
        let vb = ctx.push(Inst::IntBin {
            ty: IntTy::I32,
            op: BinOp::And,
            a: val,
            b: mask,
        });
        let vb64 = ctx.push(Inst::Convert {
            op: ConvOp::ExtendI32U,
            a: vb,
        });
        let magic = ctx.push(Inst::ConstI64(0x0101_0101_0101_0101u64 as i64));
        let rep64 = ctx.push(Inst::IntBin {
            ty: IntTy::I64,
            op: BinOp::Mul,
            a: vb64,
            b: magic,
        });
        let rep32 = ctx.push(Inst::Convert {
            op: ConvOp::WrapI64,
            a: rep64,
        });
        for &(off, w) in &chunks {
            let value = if w == 8 { rep64 } else { rep32 };
            ctx.push_effect(Inst::Store {
                op: store_w(w),
                addr: dst,
                value,
                offset: off,
                align: 0,
            });
        }
    }
    Ok(true)
}

/// Lower a float math intrinsic call to inline float ops, returning its result index. `fmuladd`/
/// `fma` lower to `fmul`+`fadd` (unfused — a defined IEEE approximation; both backends agree).
/// Returns `Ok(None)` if the call is not a recognized float intrinsic.
fn lower_float_intrinsic(
    ctx: &mut BlockCtx,
    c: &llvm_ir::instruction::Call,
    types: &Types,
) -> Result<Option<ValIdx>, Error> {
    let Some(name) = callee_name(c) else {
        return Ok(None);
    };
    // Strip the `.f32`/`.f64` overload suffix to match the base intrinsic.
    let base = name.rsplit_once('.').map_or(name.as_str(), |(b, _)| b);
    // Recognize the intrinsic *before* inspecting operand types — a non-float call (e.g. a normal
    // function) must fall through to the call path, not error on `float_ty`.
    let recognized = matches!(
        base,
        "llvm.sqrt"
            | "llvm.fabs"
            | "llvm.floor"
            | "llvm.ceil"
            | "llvm.trunc"
            | "llvm.rint"
            | "llvm.nearbyint"
            | "llvm.roundeven"
            | "llvm.minnum"
            | "llvm.minimum"
            | "llvm.maxnum"
            | "llvm.maximum"
            | "llvm.copysign"
            | "llvm.fmuladd"
            | "llvm.fma"
    );
    if !recognized {
        return Ok(None);
    }
    let args: Vec<&Operand> = c.arguments.iter().map(|(a, _)| a).collect();
    let ty = match args.first() {
        Some(a) => float_ty(val_type(a.get_type(types).as_ref())?)?,
        None => return Ok(None),
    };
    let un = |ctx: &mut BlockCtx, op: FUnOp| -> Result<ValIdx, Error> {
        let a = ctx.operand(args[0])?;
        Ok(ctx.push(Inst::FUn { ty, op, a }))
    };
    let bin2 = |ctx: &mut BlockCtx, op: FBinOp| -> Result<ValIdx, Error> {
        let a = ctx.operand(args[0])?;
        let b = ctx.operand(args[1])?;
        Ok(ctx.push(Inst::FBin { ty, op, a, b }))
    };
    let idx = match base {
        "llvm.sqrt" => un(ctx, FUnOp::Sqrt)?,
        "llvm.fabs" => un(ctx, FUnOp::Abs)?,
        "llvm.floor" => un(ctx, FUnOp::Floor)?,
        "llvm.ceil" => un(ctx, FUnOp::Ceil)?,
        "llvm.trunc" => un(ctx, FUnOp::Trunc)?,
        "llvm.rint" | "llvm.nearbyint" | "llvm.roundeven" => un(ctx, FUnOp::Nearest)?,
        "llvm.minnum" | "llvm.minimum" => bin2(ctx, FBinOp::Min)?,
        "llvm.maxnum" | "llvm.maximum" => bin2(ctx, FBinOp::Max)?,
        "llvm.copysign" => bin2(ctx, FBinOp::Copysign)?,
        // fmuladd(a,b,c) = a*b + c, lowered unfused.
        "llvm.fmuladd" | "llvm.fma" => {
            let a = ctx.operand(args[0])?;
            let b = ctx.operand(args[1])?;
            let prod = ctx.push(Inst::FBin {
                ty,
                op: FBinOp::Mul,
                a,
                b,
            });
            let cc = ctx.operand(args[2])?;
            ctx.push(Inst::FBin {
                ty,
                op: FBinOp::Add,
                a: prod,
                b: cc,
            })
        }
        _ => return Ok(None),
    };
    Ok(Some(idx))
}

/// Whether a `call` is a droppable intrinsic with no guest-visible effect for our subset —
/// `llvm.lifetime.*` (stack-slot liveness markers), `llvm.dbg.*` (debug info), `llvm.assume`.
/// These are lowered to nothing.
fn is_droppable_call(c: &llvm_ir::instruction::Call) -> bool {
    let Some(Operand::ConstantOperand(cr)) = c.function.as_ref().right() else {
        return false;
    };
    if let Constant::GlobalReference {
        name: Name::Name(s),
        ..
    } = cr.as_ref()
    {
        return s.starts_with("llvm.lifetime")
            || s.starts_with("llvm.dbg")
            || s.starts_with("llvm.assume")
            || s.starts_with("llvm.invariant");
    }
    false
}

/// The local value operands a terminator uses (the branch condition / returned value). Validates
/// terminator support. Branch *arguments* are synthesized from block parameters, not from here.
fn term_local_uses(term: &LTerm) -> Result<Vec<Name>, Error> {
    let one = |o: &Operand| match o {
        Operand::LocalOperand { name, .. } => vec![name.clone()],
        _ => Vec::new(),
    };
    match term {
        LTerm::Ret(r) => Ok(r.return_operand.as_ref().map(one).unwrap_or_default()),
        LTerm::Br(_) => Ok(Vec::new()),
        LTerm::CondBr(c) => Ok(one(&c.condition)),
        LTerm::Switch(sw) => Ok(one(&sw.operand)),
        LTerm::Unreachable(_) => Ok(Vec::new()),
        other => unsup(format!("terminator {other:?}")),
    }
}

/// Pass 1b: SSA liveness (backward fixpoint). Returns each block's **live-in** set — the values
/// defined elsewhere that are live at the block's entry (used here or threaded to a successor).
/// These become the block's threaded parameters (φ-results are added separately). φ semantics:
/// a φ in `S` taking `v` from predecessor `B` makes `v` live-*out* of `B` (an edge use), not
/// live-in of `S`.
fn liveness(f: &Function, s: &Scan) -> Result<Vec<HashSet<ValueId>>, Error> {
    let n = f.basic_blocks.len();
    // Per-block precomputed sets.
    let mut defs: Vec<HashSet<ValueId>> = vec![HashSet::new(); n];
    let mut uevar: Vec<HashSet<ValueId>> = vec![HashSet::new(); n]; // upward-exposed direct uses
    let mut succ: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut phi_defs: Vec<HashSet<ValueId>> = vec![HashSet::new(); n];
    // phi_uses[b] = values that some successor's φ pulls from predecessor `b`.
    let mut phi_uses: Vec<HashSet<ValueId>> = vec![HashSet::new(); n];

    let id = |name: &Name| -> Option<ValueId> { s.name2id.get(name).copied() };

    for (bi, bb) in f.basic_blocks.iter().enumerate() {
        for instr in &bb.instrs {
            if let Some(d) = instr.try_get_result() {
                if let Some(vid) = id(d) {
                    defs[bi].insert(vid);
                    if matches!(instr, Instruction::Phi(_)) {
                        phi_defs[bi].insert(vid);
                    }
                }
            }
            // A direct use of a value defined in another block is upward-exposed.
            for u in local_uses(instr)? {
                if let Some(vid) = id(&u) {
                    if s.def_block[vid] != bi {
                        uevar[bi].insert(vid);
                    }
                }
            }
        }
        for u in term_local_uses(&bb.term)? {
            if let Some(vid) = id(&u) {
                if s.def_block[vid] != bi {
                    uevar[bi].insert(vid);
                }
            }
        }
        for t in term_succs(&bb.term, s)? {
            succ[bi].push(t);
        }
    }
    // φ edge-uses: attribute each φ incoming to its named predecessor.
    for bb in &f.basic_blocks {
        for instr in &bb.instrs {
            if let Instruction::Phi(p) = instr {
                for (op, pred) in &p.incoming_values {
                    if let Operand::LocalOperand { name, .. } = op {
                        if let (Some(vid), Some(&pb)) = (id(name), s.block_idx.get(pred)) {
                            phi_uses[pb].insert(vid);
                        }
                    }
                }
            }
        }
    }

    let mut live_in: Vec<HashSet<ValueId>> = vec![HashSet::new(); n];
    let mut live_out: Vec<HashSet<ValueId>> = vec![HashSet::new(); n];
    let mut changed = true;
    while changed {
        changed = false;
        for bi in (0..n).rev() {
            // live_out(B) = ∪_succ [ (live_in(S) \ PhiDefs(S)) ∪ PhiUses(B,S-via-edge) ]
            let mut new_out: HashSet<ValueId> = phi_uses[bi].clone();
            for &sblk in &succ[bi] {
                for &v in &live_in[sblk] {
                    if !phi_defs[sblk].contains(&v) {
                        new_out.insert(v);
                    }
                }
            }
            // live_in(B) = UEVar(B) ∪ (live_out(B) \ Defs(B))
            let mut new_in = uevar[bi].clone();
            for &v in &new_out {
                if !defs[bi].contains(&v) {
                    new_in.insert(v);
                }
            }
            if new_out != live_out[bi] {
                live_out[bi] = new_out;
                changed = true;
            }
            if new_in != live_in[bi] {
                live_in[bi] = new_in;
                changed = true;
            }
        }
    }
    Ok(live_in)
}

/// The successor block indices of a terminator.
fn term_succs(term: &LTerm, s: &Scan) -> Result<Vec<usize>, Error> {
    let b = |name: &Name| -> Result<usize, Error> {
        s.block_idx
            .get(name)
            .copied()
            .ok_or_else(|| Error::Unsupported(format!("branch to unknown block {name:?}")))
    };
    match term {
        LTerm::Br(x) => Ok(vec![b(&x.dest)?]),
        LTerm::CondBr(x) => Ok(vec![b(&x.true_dest)?, b(&x.false_dest)?]),
        LTerm::Switch(sw) => {
            let mut v = vec![b(&sw.default_dest)?];
            for (_, dest) in &sw.dests {
                v.push(b(dest)?);
            }
            Ok(v)
        }
        LTerm::Ret(_) | LTerm::Unreachable(_) => Ok(Vec::new()),
        other => unsup(format!("terminator {other:?}")),
    }
}

/// Pass 1c: the ordered parameter value-ids of each block. Entry's parameters are the function's
/// parameters (§3b). Every other block's are its φ-results (in φ order) followed by its threaded
/// live-in values (sorted by id for a deterministic order shared by the block header and every
/// branch into it).
fn block_params(f: &Function, s: &Scan, live_in: &[HashSet<ValueId>]) -> Vec<Vec<ValueId>> {
    let mut out = Vec::with_capacity(f.basic_blocks.len());
    for (bi, bb) in f.basic_blocks.iter().enumerate() {
        if bi == 0 {
            // Entry: the data-SP then the function parameters (ids 0..nparams), matching the
            // prepended IR signature `(sp, c-params…)`.
            let mut params = vec![SP];
            params.extend(0..f.parameters.len());
            out.push(params);
            continue;
        }
        // Every non-entry block carries the data-SP as its first parameter (§3d), then its
        // φ-results and threaded live-ins.
        let mut params: Vec<ValueId> = vec![SP];
        let mut phi_set: HashSet<ValueId> = HashSet::new();
        for instr in &bb.instrs {
            if let Instruction::Phi(p) = instr {
                if let Some(&vid) = s.name2id.get(&p.dest) {
                    params.push(vid);
                    phi_set.insert(vid);
                }
            }
        }
        let mut threaded: Vec<ValueId> = live_in[bi]
            .iter()
            .copied()
            .filter(|v| !phi_set.contains(v))
            .collect();
        threaded.sort_unstable();
        params.extend(threaded);
        out.push(params);
    }
    out
}

/// A block under construction: the straight-line body, the value-id → block-local-index map
/// (seeded with the block's parameters), and the running block-local value counter.
struct BlockCtx<'a> {
    s: &'a Scan,
    /// `alloca` value-id → its `sp`-relative window offset (the data-stack frame layout).
    frame: &'a HashMap<ValueId, u64>,
    /// This function's 16-aligned frame size — a callee receives `sp + frame_size`.
    frame_size: u64,
    /// Defined LLVM function name → its IR function index (for resolving a direct `call`).
    name2idx: &'a HashMap<String, u32>,
    /// Global variable name → its window address (for resolving a `@global` reference).
    globals: &'a HashMap<String, u64>,
    insts: Vec<Inst>,
    idx_of: HashMap<ValueId, ValIdx>,
    next_val: ValIdx,
}

impl<'a> BlockCtx<'a> {
    fn push(&mut self, inst: Inst) -> ValIdx {
        self.insts.push(inst);
        let i = self.next_val;
        self.next_val += 1;
        i
    }

    /// The data-SP's block-local index (always parameter 0 of every block, §3d).
    fn sp(&self) -> Result<ValIdx, Error> {
        self.id(SP)
    }

    /// Append an instruction that produces **no** SSA value (e.g. `store`). It must not consume a
    /// block-local value index — the verifier/interpreter number only value-producing insts (§3a).
    fn push_effect(&mut self, inst: Inst) {
        self.insts.push(inst);
    }

    fn const_i64(&mut self, v: i64) -> ValIdx {
        self.push(Inst::ConstI64(v))
    }

    fn add_i64(&mut self, a: ValIdx, b: ValIdx) -> ValIdx {
        self.push(Inst::IntBin {
            ty: IntTy::I64,
            op: BinOp::Add,
            a,
            b,
        })
    }

    fn mul_i64(&mut self, a: ValIdx, b: ValIdx) -> ValIdx {
        self.push(Inst::IntBin {
            ty: IntTy::I64,
            op: BinOp::Mul,
            a,
            b,
        })
    }

    /// Resolve a value-id already available in this block (a parameter or an earlier result).
    fn id(&self, vid: ValueId) -> Result<ValIdx, Error> {
        self.idx_of
            .get(&vid)
            .copied()
            .ok_or_else(|| Error::Unsupported(format!("value {vid} not available in block")))
    }

    /// Resolve an operand to a block-local index, materializing a constant as a `const` inst
    /// (SVM has no constant pool — constants are instructions, §3b).
    fn operand(&mut self, op: &Operand) -> Result<ValIdx, Error> {
        match op {
            Operand::LocalOperand { name, .. } => {
                let vid = *self
                    .s
                    .name2id
                    .get(name)
                    .ok_or_else(|| Error::Unsupported(format!("unresolved local {name:?}")))?;
                self.id(vid)
            }
            Operand::ConstantOperand(c) => match c.as_ref() {
                Constant::Int { bits, value } if *bits <= 32 => {
                    Ok(self.push(Inst::ConstI32(*value as u32 as i32)))
                }
                Constant::Int { bits, value } if *bits == 64 => {
                    Ok(self.push(Inst::ConstI64(*value as i64)))
                }
                Constant::Float(Float::Single(f)) => Ok(self.push(Inst::ConstF32(f.to_bits()))),
                Constant::Float(Float::Double(d)) => Ok(self.push(Inst::ConstF64(d.to_bits()))),
                // `undef`/`poison`/`null` resolve to a defined zero of the type — the IR is total
                // (§3c), so no UB reaches it (the value is unused or its use is defined-on-zero).
                Constant::Undef(t) | Constant::Poison(t) | Constant::Null(t) => {
                    match val_type(t.as_ref())? {
                        ValType::I32 => Ok(self.push(Inst::ConstI32(0))),
                        ValType::I64 => Ok(self.push(Inst::ConstI64(0))),
                        other => unsup(format!("undef/poison/null of type {}", other.as_str())),
                    }
                }
                // A reference to a global variable is its window address (a constant `i64`). A
                // reference to a *function* is its §3c funcref index (the function-table index),
                // widened to the `i64` pointer representation (a function pointer is `ptr`/`i64`).
                Constant::GlobalReference { name, .. } => {
                    let n = name_str(name);
                    if let Some(&a) = self.globals.get(&n) {
                        Ok(self.push(Inst::ConstI64(a as i64)))
                    } else if let Some(&func) = self.name2idx.get(&n) {
                        let r = self.push(Inst::RefFunc { func });
                        Ok(self.push(Inst::Convert {
                            op: ConvOp::ExtendI32U,
                            a: r,
                        }))
                    } else {
                        unsup(format!("reference to `@{n}` (undefined/external global)"))
                    }
                }
                other => unsup(format!("constant operand {other:?}")),
            },
            Operand::MetadataOperand => unsup("metadata operand"),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn translate_block(
    bb: &BasicBlock,
    bi: usize,
    f: &Function,
    types: &Types,
    s: &Scan,
    block_params: &[Vec<ValueId>],
    frame: &HashMap<ValueId, u64>,
    frame_size: u64,
    name2idx: &HashMap<String, u32>,
    globals: &HashMap<String, u64>,
) -> Result<Block, Error> {
    let param_ids = &block_params[bi];
    // The data-SP (`SP` sentinel) types as `i64`; every other param reads its scanned type.
    let params: Vec<ValType> = param_ids
        .iter()
        .map(|&v| if v == SP { ValType::I64 } else { s.ty[v] })
        .collect();
    let mut ctx = BlockCtx {
        s,
        frame,
        frame_size,
        name2idx,
        globals,
        insts: Vec::new(),
        idx_of: HashMap::new(),
        next_val: 0,
    };
    for (pos, &vid) in param_ids.iter().enumerate() {
        ctx.idx_of.insert(vid, pos as ValIdx);
    }
    ctx.next_val = param_ids.len() as ValIdx;

    for instr in &bb.instrs {
        if matches!(instr, Instruction::Phi(_)) {
            continue; // φ-results are block parameters, supplied by predecessors
        }
        translate_inst(&mut ctx, instr, types)?;
    }
    let term = translate_term(&mut ctx, &bb.term, bi, f, s, block_params)?;
    Ok(Block {
        params,
        insts: ctx.insts,
        term,
    })
}

fn translate_inst(ctx: &mut BlockCtx, instr: &Instruction, types: &Types) -> Result<(), Error> {
    use Instruction as I;
    // The op's integer width, from operand0 (both operands share a type in LLVM binops).
    let bin_ty =
        |o: &Operand| -> Result<IntTy, Error> { int_ty(val_type(o.get_type(types).as_ref())?) };
    // The op's float width (f32/f64), likewise.
    let fty =
        |o: &Operand| -> Result<FloatTy, Error> { float_ty(val_type(o.get_type(types).as_ref())?) };

    // No-result instructions (effects only): handle and return early.
    if let I::Store(st) = instr {
        let addr = ctx.operand(&st.address)?;
        let value = ctx.operand(&st.value)?;
        let op = store_op(st.value.get_type(types).as_ref())?;
        ctx.push_effect(Inst::Store {
            op,
            addr,
            value,
            offset: 0,
            align: 0,
        });
        return Ok(());
    }
    if let I::Call(c) = instr {
        if is_droppable_call(c) {
            return Ok(()); // a no-op intrinsic (lifetime/dbg/assume)
        }
        // Float math intrinsics lower to inline float ops (not a call).
        if let Some(idx) = lower_float_intrinsic(ctx, c, types)? {
            if let Some(dest) = &c.dest {
                if let Some(&vid) = ctx.s.name2id.get(dest) {
                    ctx.idx_of.insert(vid, idx);
                }
            }
            return Ok(());
        }
        // `llvm.memcpy`/`memmove`/`memset` lower to inline chunked load/stores (constant length).
        if lower_mem_intrinsic(ctx, c)? {
            return Ok(()); // void — no SSA result
        }
        // Pass the callee its own data-stack frame at `sp + frame_size` (§3d), then the mapped
        // arguments. The IR signature is `(sp, c-args…)`, so the callee's frame never overlaps ours.
        let sp = ctx.sp()?;
        let fs = ctx.const_i64(ctx.frame_size as i64);
        let callee_sp = ctx.add_i64(sp, fs);
        let mut args = vec![callee_sp];
        for (a, _attrs) in &c.arguments {
            args.push(ctx.operand(a)?);
        }
        // A direct call (named, defined function) lowers to `call <idx>`; an indirect call (through
        // a function-pointer value) lowers to `call_indirect <sig>` (§3c: mask + type-id check).
        let inst = match callee_name(c) {
            Some(name) => {
                let func = *ctx.name2idx.get(&name).ok_or_else(|| {
                    Error::Unsupported(format!("call to external/undefined function `{name}`"))
                })?;
                Inst::Call { func, args }
            }
            None => {
                let op = c
                    .function
                    .as_ref()
                    .right()
                    .ok_or_else(|| Error::Unsupported("inline-asm call".into()))?;
                let fref64 = ctx.operand(op)?; // the function pointer (i64)
                let idx = ctx.push(Inst::Convert {
                    op: ConvOp::WrapI64,
                    a: fref64,
                }); // → i32 funcref index
                let ty = indirect_sig(c)?;
                Inst::CallIndirect { ty, idx, args }
            }
        };
        match &c.dest {
            Some(dest) => {
                let r = ctx.push(inst);
                if let Some(&vid) = ctx.s.name2id.get(dest) {
                    ctx.idx_of.insert(vid, r);
                }
            }
            None => ctx.push_effect(inst), // void call: no SSA result
        }
        return Ok(());
    }

    let (dest, idx) = match instr {
        I::Alloca(a) => {
            // The slot's `sp`-relative offset (laid out by `frame_layout`): address = `sp + off`.
            let vid = *ctx
                .s
                .name2id
                .get(&a.dest)
                .ok_or_else(|| Error::Unsupported("alloca without result".into()))?;
            let off = *ctx
                .frame
                .get(&vid)
                .ok_or_else(|| Error::Unsupported("alloca missing frame slot".into()))?;
            let sp = ctx.sp()?;
            let c = ctx.const_i64(off as i64);
            (&a.dest, ctx.add_i64(sp, c))
        }
        I::Load(l) => {
            let addr = ctx.operand(&l.address)?;
            let op = load_op(l.loaded_ty.as_ref())?;
            (
                &l.dest,
                ctx.push(Inst::Load {
                    op,
                    addr,
                    offset: 0,
                    align: 0,
                }),
            )
        }
        I::GetElementPtr(g) => {
            let addr = translate_gep(ctx, g, types)?;
            (&g.dest, addr)
        }
        I::Add(x) => bin(
            ctx,
            &x.dest,
            bin_ty(&x.operand0)?,
            BinOp::Add,
            &x.operand0,
            &x.operand1,
        )?,
        I::Sub(x) => bin(
            ctx,
            &x.dest,
            bin_ty(&x.operand0)?,
            BinOp::Sub,
            &x.operand0,
            &x.operand1,
        )?,
        I::Mul(x) => bin(
            ctx,
            &x.dest,
            bin_ty(&x.operand0)?,
            BinOp::Mul,
            &x.operand0,
            &x.operand1,
        )?,
        I::UDiv(x) => bin(
            ctx,
            &x.dest,
            bin_ty(&x.operand0)?,
            BinOp::DivU,
            &x.operand0,
            &x.operand1,
        )?,
        I::SDiv(x) => bin(
            ctx,
            &x.dest,
            bin_ty(&x.operand0)?,
            BinOp::DivS,
            &x.operand0,
            &x.operand1,
        )?,
        I::URem(x) => bin(
            ctx,
            &x.dest,
            bin_ty(&x.operand0)?,
            BinOp::RemU,
            &x.operand0,
            &x.operand1,
        )?,
        I::SRem(x) => bin(
            ctx,
            &x.dest,
            bin_ty(&x.operand0)?,
            BinOp::RemS,
            &x.operand0,
            &x.operand1,
        )?,
        I::And(x) => bin(
            ctx,
            &x.dest,
            bin_ty(&x.operand0)?,
            BinOp::And,
            &x.operand0,
            &x.operand1,
        )?,
        I::Or(x) => bin(
            ctx,
            &x.dest,
            bin_ty(&x.operand0)?,
            BinOp::Or,
            &x.operand0,
            &x.operand1,
        )?,
        I::Xor(x) => bin(
            ctx,
            &x.dest,
            bin_ty(&x.operand0)?,
            BinOp::Xor,
            &x.operand0,
            &x.operand1,
        )?,
        I::Shl(x) => bin(
            ctx,
            &x.dest,
            bin_ty(&x.operand0)?,
            BinOp::Shl,
            &x.operand0,
            &x.operand1,
        )?,
        I::LShr(x) => bin(
            ctx,
            &x.dest,
            bin_ty(&x.operand0)?,
            BinOp::ShrU,
            &x.operand0,
            &x.operand1,
        )?,
        I::AShr(x) => bin(
            ctx,
            &x.dest,
            bin_ty(&x.operand0)?,
            BinOp::ShrS,
            &x.operand0,
            &x.operand1,
        )?,
        I::ICmp(x) => {
            let ty = bin_ty(&x.operand0)?;
            let op = icmp_op(x.predicate);
            let a = ctx.operand(&x.operand0)?;
            let b = ctx.operand(&x.operand1)?;
            (&x.dest, ctx.push(Inst::IntCmp { ty, op, a, b }))
        }
        I::Select(x) => {
            let cond = ctx.operand(&x.condition)?;
            let a = ctx.operand(&x.true_value)?;
            let b = ctx.operand(&x.false_value)?;
            (&x.dest, ctx.push(Inst::Select { cond, a, b }))
        }
        I::Trunc(x) => {
            let from = src_bits(&x.operand, types)?;
            let to = int_bits(x.to_type.as_ref())
                .ok_or_else(|| Error::Unsupported("trunc to non-int".into()))?;
            let v = ctx.operand(&x.operand)?;
            (&x.dest, emit_trunc(ctx, v, from, to))
        }
        I::ZExt(x) => {
            let from = src_bits(&x.operand, types)?;
            let to = int_bits(x.to_type.as_ref())
                .ok_or_else(|| Error::Unsupported("zext to non-int".into()))?;
            let v = ctx.operand(&x.operand)?;
            (&x.dest, emit_ext(ctx, v, from, to, false))
        }
        I::SExt(x) => {
            let from = src_bits(&x.operand, types)?;
            let to = int_bits(x.to_type.as_ref())
                .ok_or_else(|| Error::Unsupported("sext to non-int".into()))?;
            let v = ctx.operand(&x.operand)?;
            (&x.dest, emit_ext(ctx, v, from, to, true))
        }
        // Floats (f32/f64) — IEEE 754, no traps (§3b).
        I::FAdd(x) => fbin(
            ctx,
            &x.dest,
            fty(&x.operand0)?,
            FBinOp::Add,
            &x.operand0,
            &x.operand1,
        )?,
        I::FSub(x) => fbin(
            ctx,
            &x.dest,
            fty(&x.operand0)?,
            FBinOp::Sub,
            &x.operand0,
            &x.operand1,
        )?,
        I::FMul(x) => fbin(
            ctx,
            &x.dest,
            fty(&x.operand0)?,
            FBinOp::Mul,
            &x.operand0,
            &x.operand1,
        )?,
        I::FDiv(x) => fbin(
            ctx,
            &x.dest,
            fty(&x.operand0)?,
            FBinOp::Div,
            &x.operand0,
            &x.operand1,
        )?,
        I::FNeg(x) => {
            let ty = fty(&x.operand)?;
            let a = ctx.operand(&x.operand)?;
            (
                &x.dest,
                ctx.push(Inst::FUn {
                    ty,
                    op: FUnOp::Neg,
                    a,
                }),
            )
        }
        I::FCmp(x) => {
            let ty = fty(&x.operand0)?;
            let op = fcmp_op(x.predicate)?;
            let a = ctx.operand(&x.operand0)?;
            let b = ctx.operand(&x.operand1)?;
            (&x.dest, ctx.push(Inst::FCmp { ty, op, a, b }))
        }
        I::FPToSI(x) => (&x.dest, ftoi(ctx, &x.operand, &x.to_type, types, true)?),
        I::FPToUI(x) => (&x.dest, ftoi(ctx, &x.operand, &x.to_type, types, false)?),
        I::SIToFP(x) => (&x.dest, itof(ctx, &x.operand, &x.to_type, types, true)?),
        I::UIToFP(x) => (&x.dest, itof(ctx, &x.operand, &x.to_type, types, false)?),
        I::FPExt(x) => {
            // f32 → f64.
            let a = ctx.operand(&x.operand)?;
            (
                &x.dest,
                ctx.push(Inst::Cast {
                    op: CastOp::Promote,
                    a,
                }),
            )
        }
        I::FPTrunc(x) => {
            // f64 → f32.
            let a = ctx.operand(&x.operand)?;
            (
                &x.dest,
                ctx.push(Inst::Cast {
                    op: CastOp::Demote,
                    a,
                }),
            )
        }
        I::BitCast(x) => {
            let from = val_type(x.operand.get_type(types).as_ref())?;
            let to = val_type(x.to_type.as_ref())?;
            let a = ctx.operand(&x.operand)?;
            let op = match (from, to) {
                (ValType::I32, ValType::F32) => CastOp::ReinterpI32F32,
                (ValType::F32, ValType::I32) => CastOp::ReinterpF32I32,
                (ValType::I64, ValType::F64) => CastOp::ReinterpI64F64,
                (ValType::F64, ValType::I64) => CastOp::ReinterpF64I64,
                (f, t) if f == t => return finish(ctx, &x.dest, a), // no-op bitcast
                (f, t) => return unsup(format!("bitcast {} → {}", f.as_str(), t.as_str())),
            };
            (&x.dest, ctx.push(Inst::Cast { op, a }))
        }
        other => return unsup(format!("instruction {other:?}")),
    };
    if let Some(&vid) = ctx.s.name2id.get(dest) {
        ctx.idx_of.insert(vid, idx);
    }
    Ok(())
}

/// Emit a binary integer op and return `(dest, result-index)`.
fn bin<'d>(
    ctx: &mut BlockCtx,
    dest: &'d Name,
    ty: IntTy,
    op: BinOp,
    a: &Operand,
    b: &Operand,
) -> Result<(&'d Name, ValIdx), Error> {
    let a = ctx.operand(a)?;
    let b = ctx.operand(b)?;
    Ok((dest, ctx.push(Inst::IntBin { ty, op, a, b })))
}

/// Emit a binary float op and return `(dest, result-index)`.
fn fbin<'d>(
    ctx: &mut BlockCtx,
    dest: &'d Name,
    ty: FloatTy,
    op: FBinOp,
    a: &Operand,
    b: &Operand,
) -> Result<(&'d Name, ValIdx), Error> {
    let a = ctx.operand(a)?;
    let b = ctx.operand(b)?;
    Ok((dest, ctx.push(Inst::FBin { ty, op, a, b })))
}

/// Emit a (saturating) float→int conversion, returning its result index.
fn ftoi(
    ctx: &mut BlockCtx,
    operand: &Operand,
    to_type: &llvm_ir::TypeRef,
    types: &Types,
    signed: bool,
) -> Result<ValIdx, Error> {
    let src = float_ty(val_type(operand.get_type(types).as_ref())?)?;
    let dst = int_ty(val_type(to_type.as_ref())?)?;
    let a = ctx.operand(operand)?;
    Ok(ctx.push(Inst::FToISat {
        op: ftoi_op(src, dst, signed),
        a,
    }))
}

/// Emit an int→float conversion, returning its result index.
fn itof(
    ctx: &mut BlockCtx,
    operand: &Operand,
    to_type: &llvm_ir::TypeRef,
    types: &Types,
    signed: bool,
) -> Result<ValIdx, Error> {
    let src = int_ty(val_type(operand.get_type(types).as_ref())?)?;
    let dst = float_ty(val_type(to_type.as_ref())?)?;
    let a = ctx.operand(operand)?;
    Ok(ctx.push(Inst::IToFConv {
        op: itof_op(src, dst, signed),
        a,
    }))
}

/// Record `dest`'s value as an existing index (an alias, e.g. a no-op bitcast) and return.
fn finish(ctx: &mut BlockCtx, dest: &Name, idx: ValIdx) -> Result<(), Error> {
    if let Some(&vid) = ctx.s.name2id.get(dest) {
        ctx.idx_of.insert(vid, idx);
    }
    Ok(())
}

/// The `LoadOp` (width + result container) for an LLVM loaded type. Narrow loads zero-extend
/// into the `i32` container; a following `sext`/`zext` (the §3b discipline) fixes signedness.
fn load_op(ty: &Type) -> Result<svm_ir::LoadOp, Error> {
    use svm_ir::LoadOp as L;
    match ty {
        Type::IntegerType { bits } if *bits <= 8 => Ok(L::I32_8U),
        Type::IntegerType { bits } if *bits <= 16 => Ok(L::I32_16U),
        Type::IntegerType { bits } if *bits <= 32 => Ok(L::I32),
        Type::IntegerType { bits } if *bits == 64 => Ok(L::I64),
        Type::PointerType { .. } => Ok(L::I64),
        Type::FPType(FPType::Single) => Ok(L::F32),
        Type::FPType(FPType::Double) => Ok(L::F64),
        other => unsup(format!("load of type {other} (Milestone 1+)")),
    }
}

/// The `StoreOp` (width) for an LLVM stored value type.
fn store_op(ty: &Type) -> Result<svm_ir::StoreOp, Error> {
    use svm_ir::StoreOp as S;
    match ty {
        Type::IntegerType { bits } if *bits <= 8 => Ok(S::I32_8),
        Type::IntegerType { bits } if *bits <= 16 => Ok(S::I32_16),
        Type::IntegerType { bits } if *bits <= 32 => Ok(S::I32),
        Type::IntegerType { bits } if *bits == 64 => Ok(S::I64),
        Type::PointerType { .. } => Ok(S::I64),
        Type::FPType(FPType::Single) => Ok(S::F32),
        Type::FPType(FPType::Double) => Ok(S::F64),
        other => unsup(format!("store of type {other} (Milestone 1+)")),
    }
}

/// Lower a `getelementptr` to an `i64` address: `base + Σ offset_k`. Index 0 strides by the pointee
/// size; each later index walks *into* the current type — an array/vector element (stride =
/// element size) or a **struct field** (a constant index → the field's byte offset). Constant
/// indices fold into one offset add; variable indices emit a `mul`+`add` (sign-extended to `i64`).
fn translate_gep(
    ctx: &mut BlockCtx,
    g: &llvm_ir::instruction::GetElementPtr,
    types: &Types,
) -> Result<ValIdx, Error> {
    let mut addr = ctx.operand(&g.address)?;
    let mut cur = g.source_element_type.clone();
    let mut const_off: i64 = 0;
    for (k, idx) in g.indices.iter().enumerate() {
        // A struct field index (k ≥ 1, current type is a struct): always a constant; add the
        // field's offset and descend into the field's type — no stride.
        if k > 0
            && matches!(
                cur.as_ref(),
                Type::StructType { .. } | Type::NamedStructType { .. }
            )
        {
            let (fields, packed) = resolve_struct(cur.as_ref(), types)?;
            let fidx = match idx {
                Operand::ConstantOperand(c) => match c.as_ref() {
                    Constant::Int { value, .. } => *value as usize,
                    _ => return unsup("struct GEP with non-constant field index"),
                },
                _ => return unsup("struct GEP with non-constant field index"),
            };
            let (offsets, _, _) = struct_layout(&fields, packed, types)?;
            const_off += *offsets
                .get(fidx)
                .ok_or_else(|| Error::Unsupported("struct field index out of range".into()))?
                as i64;
            cur = fields[fidx].clone();
            continue;
        }
        let stride = if k == 0 {
            type_size(cur.as_ref(), types)?
        } else {
            match cur.as_ref() {
                Type::ArrayType { element_type, .. } => {
                    let s = type_size(element_type.as_ref(), types)?;
                    cur = element_type.clone();
                    s
                }
                other => return unsup(format!("GEP into type {other} (Milestone 1+)")),
            }
        };
        // Constant index → fold into the running byte offset.
        if let Operand::ConstantOperand(c) = idx {
            if let Constant::Int { value, .. } = c.as_ref() {
                const_off += (*value as i64).wrapping_mul(stride as i64);
                continue;
            }
        }
        // Variable index → `addr += sext_i64(idx) * stride`.
        let bits = src_bits(idx, types)?;
        let iv = ctx.operand(idx)?;
        let iv64 = if bits >= 64 {
            iv
        } else {
            emit_ext(ctx, iv, bits, 64, true)
        };
        let sv = ctx.const_i64(stride as i64);
        let term = ctx.mul_i64(iv64, sv);
        addr = ctx.add_i64(addr, term);
    }
    if const_off != 0 {
        let c = ctx.const_i64(const_off);
        addr = ctx.add_i64(addr, c);
    }
    Ok(addr)
}

fn icmp_op(p: IntPredicate) -> CmpOp {
    match p {
        IntPredicate::EQ => CmpOp::Eq,
        IntPredicate::NE => CmpOp::Ne,
        IntPredicate::UGT => CmpOp::GtU,
        IntPredicate::UGE => CmpOp::GeU,
        IntPredicate::ULT => CmpOp::LtU,
        IntPredicate::ULE => CmpOp::LeU,
        IntPredicate::SGT => CmpOp::GtS,
        IntPredicate::SGE => CmpOp::GeS,
        IntPredicate::SLT => CmpOp::LtS,
        IntPredicate::SLE => CmpOp::LeS,
    }
}

fn src_bits(op: &Operand, types: &Types) -> Result<u32, Error> {
    int_bits(op.get_type(types).as_ref())
        .ok_or_else(|| Error::Unsupported("conversion of non-integer".into()))
}

/// Lower a `trunc from→to`. Narrow values are carried in their `i32`/`i64` container; truncation
/// drops the high bits, so we mask to `to` bits (within `i32`) or `wrap` (`i64`→`i32`).
fn emit_trunc(ctx: &mut BlockCtx, v: ValIdx, from: u32, to: u32) -> ValIdx {
    if from <= 32 {
        // i32 container → i32 container: mask to the low `to` bits.
        mask_to(ctx, v, to)
    } else if to <= 32 {
        let w = ctx.push(Inst::Convert {
            op: ConvOp::WrapI64,
            a: v,
        });
        mask_to(ctx, w, to)
    } else {
        v // i64 → i64 (no-op)
    }
}

/// Lower a `zext`/`sext from→to`. Produces a value whose low `to` bits are the (zero- or sign-)
/// extended result, in the destination container.
fn emit_ext(ctx: &mut BlockCtx, v: ValIdx, from: u32, to: u32, signed: bool) -> ValIdx {
    // First make a clean i32 holding the value extended from `from` bits (if `from < 32`).
    let i32v = if from >= 32 {
        v
    } else if signed {
        sext_in_i32(ctx, v, from)
    } else {
        mask_to(ctx, v, from)
    };
    if to <= 32 {
        i32v
    } else if signed {
        ctx.push(Inst::Convert {
            op: ConvOp::ExtendI32S,
            a: i32v,
        })
    } else {
        ctx.push(Inst::Convert {
            op: ConvOp::ExtendI32U,
            a: i32v,
        })
    }
}

/// Mask an `i32`-container value to its low `bits` (no-op for `bits >= 32`).
fn mask_to(ctx: &mut BlockCtx, v: ValIdx, bits: u32) -> ValIdx {
    if bits >= 32 {
        return v;
    }
    let m = ctx.push(Inst::ConstI32(((1u64 << bits) - 1) as i32));
    ctx.push(Inst::IntBin {
        ty: IntTy::I32,
        op: BinOp::And,
        a: v,
        b: m,
    })
}

/// Sign-extend the low `from` bits of an `i32`-container value to fill the `i32` (`shl` then
/// arithmetic `shr` by `32 - from`). Handles `i1` too; `extend8_s`/`extend16_s` would fold the
/// 8/16 cases, but Cranelift folds the shift pair, so one general path keeps the TCB small (§3b).
fn sext_in_i32(ctx: &mut BlockCtx, v: ValIdx, from: u32) -> ValIdx {
    debug_assert!(from < 32);
    let sh = ctx.push(Inst::ConstI32((32 - from) as i32));
    let up = ctx.push(Inst::IntBin {
        ty: IntTy::I32,
        op: BinOp::Shl,
        a: v,
        b: sh,
    });
    let sh2 = ctx.push(Inst::ConstI32((32 - from) as i32));
    ctx.push(Inst::IntBin {
        ty: IntTy::I32,
        op: BinOp::ShrS,
        a: up,
        b: sh2,
    })
}

fn translate_term(
    ctx: &mut BlockCtx,
    term: &LTerm,
    bi: usize,
    f: &Function,
    s: &Scan,
    block_params: &[Vec<ValueId>],
) -> Result<Terminator, Error> {
    match term {
        LTerm::Ret(r) => match &r.return_operand {
            None => Ok(Terminator::Return(Vec::new())),
            Some(op) => {
                let v = ctx.operand(op)?;
                Ok(Terminator::Return(vec![v]))
            }
        },
        LTerm::Br(x) => {
            let target = s.block_idx[&x.dest];
            let args = branch_args(ctx, bi, target, f, s, block_params)?;
            Ok(Terminator::Br {
                target: target as u32,
                args,
            })
        }
        LTerm::CondBr(x) => {
            let cond = ctx.operand(&x.condition)?;
            let then_blk = s.block_idx[&x.true_dest];
            let else_blk = s.block_idx[&x.false_dest];
            let then_args = branch_args(ctx, bi, then_blk, f, s, block_params)?;
            let else_args = branch_args(ctx, bi, else_blk, f, s, block_params)?;
            Ok(Terminator::BrIf {
                cond,
                then_blk: then_blk as u32,
                then_args,
                else_blk: else_blk as u32,
                else_args,
            })
        }
        LTerm::Switch(sw) => translate_switch(ctx, sw, bi, f, s, block_params),
        LTerm::Unreachable(_) => Ok(Terminator::Unreachable),
        other => unsup(format!("terminator {other:?}")),
    }
}

/// The largest `br_table` span we materialize for a `switch` (gaps fill with the default). A
/// sparser switch — clang usually lowers those to compare chains in the IR anyway — is a clean
/// `Unsupported` (a synthetic-block compare-chain lowering is a later option).
const MAX_SWITCH_SPAN: i64 = 4096;

/// Lower a `switch` to a `br_table` (§3b): bias the `i32` operand by the minimum case value, then
/// index a target vector spanning `[min, max]` with gaps filled by the default edge. Each edge
/// carries the destination's block arguments (computed once per distinct target). i64-operand or
/// too-sparse switches are `Unsupported`.
fn translate_switch(
    ctx: &mut BlockCtx,
    sw: &llvm_ir::terminator::Switch,
    bi: usize,
    f: &Function,
    s: &Scan,
    block_params: &[Vec<ValueId>],
) -> Result<Terminator, Error> {
    // The operand must be `i32` (the common C `switch(int)`); `br_table`'s index is `i32`.
    if operand_bits(&sw.operand)? > 32 {
        return unsup("switch on i64 (Milestone 1+)");
    }
    // Collect the (value, dest-block) cases.
    let mut cases: Vec<(i64, usize)> = Vec::with_capacity(sw.dests.len());
    for (v, dest) in &sw.dests {
        let val = match v.as_ref() {
            Constant::Int { value, .. } => *value as i32 as i64,
            other => return unsup(format!("switch case constant {other:?}")),
        };
        let blk = *s
            .block_idx
            .get(dest)
            .ok_or_else(|| Error::Unsupported(format!("switch to unknown block {dest:?}")))?;
        cases.push((val, blk));
    }
    let default_blk = *s
        .block_idx
        .get(&sw.default_dest)
        .ok_or_else(|| Error::Unsupported("switch default to unknown block".into()))?;
    if cases.is_empty() {
        // Degenerate: an unconditional branch to the default.
        let args = branch_args(ctx, bi, default_blk, f, s, block_params)?;
        return Ok(Terminator::Br {
            target: default_blk as u32,
            args,
        });
    }
    let min = cases.iter().map(|(v, _)| *v).min().unwrap();
    let max = cases.iter().map(|(v, _)| *v).max().unwrap();
    let span = max - min + 1;
    if span > MAX_SWITCH_SPAN {
        return unsup(format!("sparse switch (span {span} > {MAX_SWITCH_SPAN})"));
    }

    // Index = operand - min (so the table starts at 0). An out-of-range / unbiased value lands on
    // the default (a negative bias wraps to a large `u32`, ≥ len ⇒ default).
    let operand = ctx.operand(&sw.operand)?;
    let idx = if min == 0 {
        operand
    } else {
        let m = ctx.push(Inst::ConstI32(min as i32));
        ctx.push(Inst::IntBin {
            ty: IntTy::I32,
            op: BinOp::Sub,
            a: operand,
            b: m,
        })
    };

    // Block arguments per distinct target (computed once — `branch_args` materializes constants).
    let mut args_for: HashMap<usize, Vec<ValIdx>> = HashMap::new();
    let default_args = branch_args(ctx, bi, default_blk, f, s, block_params)?;
    args_for.insert(default_blk, default_args.clone());
    for &(_, blk) in &cases {
        if let std::collections::hash_map::Entry::Vacant(e) = args_for.entry(blk) {
            let a = branch_args(ctx, bi, blk, f, s, block_params)?;
            e.insert(a);
        }
    }

    // Build the dense target vector, gaps → default.
    let mut targets: Vec<svm_ir::Edge> =
        vec![(default_blk as u32, default_args.clone()); span as usize];
    for &(v, blk) in &cases {
        targets[(v - min) as usize] = (blk as u32, args_for[&blk].clone());
    }
    Ok(Terminator::BrTable {
        idx,
        targets,
        default: (default_blk as u32, default_args),
    })
}

/// The integer bit width of a switch operand (a local carries its type; a constant its width).
fn operand_bits(op: &Operand) -> Result<u32, Error> {
    match op {
        Operand::LocalOperand { ty, .. } => {
            int_bits(ty.as_ref()).ok_or_else(|| Error::Unsupported("switch on non-integer".into()))
        }
        Operand::ConstantOperand(c) => match c.as_ref() {
            Constant::Int { bits, .. } => Ok(*bits),
            other => unsup(format!("switch operand {other:?}")),
        },
        Operand::MetadataOperand => unsup("switch on metadata"),
    }
}

/// Build the argument list for a branch from `from` to `target`: for each of `target`'s
/// parameters (φ-results then threaded live-ins), supply — from the *source* block `from` —
/// the φ's incoming value for this predecessor, or the threaded value itself.
fn branch_args(
    ctx: &mut BlockCtx,
    from: usize,
    target: usize,
    f: &Function,
    s: &Scan,
    block_params: &[Vec<ValueId>],
) -> Result<Vec<ValIdx>, Error> {
    // Map each φ-result id in `target` to its incoming operand from predecessor `from`.
    let from_name = &s.block_name[from];
    let target_bb = &f.basic_blocks[target];
    let mut phi_incoming: HashMap<ValueId, &Operand> = HashMap::new();
    for instr in &target_bb.instrs {
        if let Instruction::Phi(p) = instr {
            if let Some(&vid) = s.name2id.get(&p.dest) {
                let inc = p
                    .incoming_values
                    .iter()
                    .find(|(_, pred)| pred == from_name)
                    .map(|(op, _)| op)
                    .ok_or_else(|| {
                        Error::Unsupported(format!(
                            "φ {:?} has no incoming for predecessor {from_name:?}",
                            p.dest
                        ))
                    })?;
                phi_incoming.insert(vid, inc);
            }
        }
    }
    let mut args = Vec::with_capacity(block_params[target].len());
    for &pv in &block_params[target] {
        if let Some(op) = phi_incoming.get(&pv) {
            args.push(ctx.operand(op)?);
        } else {
            // A threaded live-in: it is live-out of `from`, so available in this block.
            args.push(ctx.id(pv)?);
        }
    }
    Ok(args)
}
