//! **Core-wasm → SVM IR transpiler** (a frontend, not part of the escape-TCB — its output is
//! re-verified). Takes a wasm binary and lowers the subset of core wasm that overlaps our IR — the
//! numeric ops, locals, and structured control flow — into an [`svm_ir::Module`]. The point is
//! *apples-to-apples* benchmarking and a second, non-chibicc proof that the IR is a real target: take
//! any wasm, run it on SVM, compare to Wasmtime on the same bytes.
//!
//! **The interesting part is the stack → SSA reconstruction.** wasm is a stack machine over mutable
//! locals; our IR is SSA with no value crossing a block boundary except as a block parameter. So at
//! every control-flow target we thread the *entire live state* — all locals plus the surviving operand
//! stack — as block parameters, exactly the way the chibicc frontend threads the data-SP and promoted
//! locals. wasm's structured control flow + validation make the stack height/types statically known at
//! each point, so the carried-value layout is well-defined.
//!
//! Scope: i32/i64 const · arithmetic/bitwise/shift · comparisons · `eqz` · `clz`/`ctz`/`popcnt` ·
//! `extend{8,16,32}_s` · `wrap`/`extend_i32` · `local.{get,set,tee}` · `drop` · `select` · `nop` · the
//! full structured control set `block`/`loop`/`if`/`else`/`br`/`br_if`/`br_table`/`return`/
//! `unreachable` (with dead-code / else-resurrection bookkeeping) · **linear memory** load/store
//! (i32/i64, incl. narrow + `memory64`; the i32 address is zero-extended into our i64 window, which is
//! sized to the wasm memory's initial pages) · direct **`call`** (multi-function + recursion) ·
//! **floats** (f32/f64 const/arith/unary/compare, load/store, and every int↔float conversion incl.
//! `trunc`/`trunc_sat`/`convert`/`demote`/`promote`/`reinterpret`) · **globals** (`global.get`/`set`
//! lowered to a reserved window region) · active **data segments** (initialized linear memory) ·
//! **`call_indirect`** + tables/element segments (the table → an in-window array of funcref indices;
//! `call_indirect` loads the entry and feeds it to our `CallIndirect`'s §3c type-id check). Still a
//! clean [`Error::Unsupported`] (added in slices): `memory.{grow,size}`, passive data/elements,
//! imports, SIMD, reference types.

use svm_ir::{
    BinOp, Block, CastOp, CmpOp, ConvOp, Edge, FBinOp, FCmpOp, FToI, FUnOp, FloatTy, Func,
    FuncType, IToF, Inst, IntTy, IntUnOp, LoadOp, Module, StoreOp, Terminator, ValIdx, ValType,
};
use wasmparser::{BlockType, MemArg, Operator, Parser, Payload, ValType as W};

/// Why a wasm module couldn't be transpiled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The wasm binary was malformed (a `wasmparser` error).
    Parse(String),
    /// A wasm feature outside the shared subset (the message names it).
    Unsupported(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Parse(s) => write!(f, "wasm parse error: {s}"),
            Error::Unsupported(s) => write!(f, "unsupported wasm: {s}"),
        }
    }
}
impl std::error::Error for Error {}

impl From<wasmparser::BinaryReaderError> for Error {
    fn from(e: wasmparser::BinaryReaderError) -> Self {
        Error::Parse(e.to_string())
    }
}

fn unsup<T>(what: impl Into<String>) -> Result<T, Error> {
    Err(Error::Unsupported(what.into()))
}

/// The transpiled module plus the wasm `export name → function index` map (the IR carries no export
/// names, so the caller — e.g. a differential harness — needs this to pick the entry).
pub struct Transpiled {
    pub module: Module,
    pub exports: Vec<(String, u32)>,
}

/// Map a wasm value type to ours; reference/SIMD types are out of the shared subset.
fn val_type(w: W) -> Result<ValType, Error> {
    match w {
        W::I32 => Ok(ValType::I32),
        W::I64 => Ok(ValType::I64),
        W::F32 => Ok(ValType::F32),
        W::F64 => Ok(ValType::F64),
        W::V128 => unsup("v128 / SIMD"),
        W::Ref(_) => unsup("reference type"),
    }
}

/// Transpile a core-wasm binary into a verifier-checkable [`Module`].
pub fn transpile(wasm: &[u8]) -> Result<Transpiled, Error> {
    let mut types: Vec<(Vec<ValType>, Vec<ValType>)> = Vec::new();
    let mut func_type_idx: Vec<u32> = Vec::new();
    let mut bodies: Vec<wasmparser::FunctionBody> = Vec::new();
    let mut exports: Vec<(String, u32)> = Vec::new();
    let mut mem: Option<wasmparser::MemoryType> = None;
    let mut data: Vec<svm_ir::Data> = Vec::new();
    let mut globals: Vec<(ValType, Vec<u8>)> = Vec::new();
    let mut table_size: Option<u64> = None;
    let mut elements: Vec<(u64, Vec<u32>)> = Vec::new(); // (offset, func indices)

    for payload in Parser::new(0).parse_all(wasm) {
        match payload? {
            Payload::TypeSection(reader) => {
                for rec in reader {
                    for sub in rec?.into_types() {
                        let ft = sub.unwrap_func();
                        let params = ft
                            .params()
                            .iter()
                            .map(|t| val_type(*t))
                            .collect::<Result<_, _>>()?;
                        let results = ft
                            .results()
                            .iter()
                            .map(|t| val_type(*t))
                            .collect::<Result<_, _>>()?;
                        types.push((params, results));
                    }
                }
            }
            Payload::ImportSection(_) => return unsup("imports"),
            Payload::FunctionSection(reader) => {
                for idx in reader {
                    func_type_idx.push(idx?);
                }
            }
            Payload::MemorySection(reader) => {
                for mt in reader {
                    let mt = mt?;
                    if mt.shared {
                        return unsup("shared memory");
                    }
                    if mem.replace(mt).is_some() {
                        return unsup("multi-memory");
                    }
                }
            }
            Payload::GlobalSection(reader) => {
                for g in reader {
                    let g = g?;
                    let ty = val_type(g.ty.content_type)?;
                    globals.push((ty, const_bytes(g.init_expr, ty)?));
                }
            }
            Payload::TableSection(reader) => {
                for tb in reader {
                    let tb = tb?;
                    if table_size.replace(tb.ty.initial).is_some() {
                        return unsup("multiple tables");
                    }
                }
            }
            Payload::ElementSection(reader) => {
                for el in reader {
                    let el = el?;
                    match el.kind {
                        wasmparser::ElementKind::Active {
                            table_index,
                            offset_expr,
                        } => {
                            if table_index.unwrap_or(0) != 0 {
                                return unsup("multi-table element segment");
                            }
                            let off = const_offset(offset_expr)?;
                            match el.items {
                                wasmparser::ElementItems::Functions(fns) => {
                                    let fs: Vec<u32> = fns.into_iter().collect::<Result<_, _>>()?;
                                    elements.push((off, fs));
                                }
                                wasmparser::ElementItems::Expressions(..) => {
                                    return unsup("element segment with const-expr items")
                                }
                            }
                        }
                        _ => return unsup("passive/declared element segment"),
                    }
                }
            }
            Payload::ExportSection(reader) => {
                for e in reader {
                    let e = e?;
                    if matches!(e.kind, wasmparser::ExternalKind::Func) {
                        exports.push((e.name.to_string(), e.index));
                    }
                }
            }
            Payload::CodeSectionEntry(body) => bodies.push(body),
            Payload::DataSection(reader) => {
                for seg in reader {
                    let seg = seg?;
                    match seg.kind {
                        wasmparser::DataKind::Active {
                            memory_index,
                            offset_expr,
                        } => {
                            if memory_index != 0 {
                                return unsup("multi-memory data segment");
                            }
                            data.push(svm_ir::Data {
                                offset: const_offset(offset_expr)?,
                                readonly: false, // wasm linear memory is writable; RO data is a frontend choice
                                bytes: seg.data.to_vec(),
                            });
                        }
                        wasmparser::DataKind::Passive => return unsup("passive data segment"),
                    }
                }
            }
            _ => {} // version header, custom sections, datacount, ends, etc. — ignore
        }
    }

    if bodies.len() != func_type_idx.len() {
        return Err(Error::Parse("function/code section length mismatch".into()));
    }

    let mem64 = mem.as_ref().map(|m| m.memory64).unwrap_or(false);

    // wasm globals are module-level mutables our IR has no notion of, so we give them a reserved region
    // **above** the linear memory and lower `global.get`/`set` to load/store there (8-byte slots, the
    // standard "globals in memory" lowering). Initializers become `data` segments. A valid guest's
    // linear-memory accesses stay in `[0, mem_bytes)` and so never reach the globals; only an OOB access
    // would (which wasm traps and we don't — the documented confinement difference).
    let mem_bytes = mem
        .as_ref()
        .map(|m| (m.initial.max(1)).saturating_mul(1 << 16))
        .unwrap_or(0);
    let globals_base = mem_bytes.div_ceil(8) * 8; // 8-byte aligned, just past the linear memory
    let globals_types: Vec<ValType> = globals.iter().map(|(t, _)| *t).collect();
    for (g, (_, bytes)) in globals.iter().enumerate() {
        data.push(svm_ir::Data {
            offset: globals_base + g as u64 * 8,
            readonly: false,
            bytes: bytes.clone(),
        });
    }
    let globals_end = globals_base + globals.len() as u64 * 8;

    // The wasm function table → an in-window array of i32 function indices (each `funcref` is our
    // §3c funcref index = the function index). `call_indirect` loads the entry and feeds it to our
    // `CallIndirect`, whose `table_lookup` does the type-id check. Empty slots get a sentinel that
    // fails that check (≈ wasm's null-funcref trap). Element segments fill the live slots.
    let tsize = table_size.unwrap_or(0);
    let table_base = globals_end.div_ceil(4) * 4;
    if tsize > 0 {
        let mut bytes = vec![0xFFu8; tsize as usize * 4]; // sentinel = no/bad funcref
        for (off, fns) in &elements {
            for (k, &f) in fns.iter().enumerate() {
                let slot = (*off as usize + k) * 4;
                if slot + 4 <= bytes.len() {
                    bytes[slot..slot + 4].copy_from_slice(&f.to_le_bytes());
                }
            }
        }
        data.push(svm_ir::Data {
            offset: table_base,
            readonly: false,
            bytes,
        });
    }
    let table_end = table_base + tsize * 4;

    let func_sigs: Vec<(Vec<ValType>, Vec<ValType>)> = func_type_idx
        .iter()
        .map(|&ti| types[ti as usize].clone())
        .collect();
    let mut funcs = Vec::with_capacity(bodies.len());
    for (i, body) in bodies.into_iter().enumerate() {
        let ty = &types[func_type_idx[i] as usize];
        funcs.push(lower_func(
            &ty.0,
            &ty.1,
            &types,
            &func_sigs,
            &globals_types,
            globals_base,
            table_base,
            &body,
            mem64,
        )?);
    }

    // Our window is a power-of-two byte range (masking confines to it); size it to hold the linear
    // memory **and** the globals + function-table regions. (wasm bounds-checks-and-traps on
    // out-of-range access while we mask-confine to the ≥ power-of-two window — identical for in-bounds
    // accesses; `memory.grow` is a later slice.) Globals/table-only modules still need a window.
    let needed = table_end.max(globals_end).max(mem_bytes);
    let memory = if mem.is_some() || !globals.is_empty() || tsize > 0 {
        let size_log2 = needed.max(1).next_power_of_two().trailing_zeros().max(16) as u8;
        Some(svm_ir::Memory { size_log2 })
    } else {
        None
    };

    Ok(Transpiled {
        module: Module {
            funcs,
            memory,
            data,
        },
        exports,
    })
}

/// Evaluate a global's constant initializer to its little-endian bytes (4 or 8 wide).
fn const_bytes(expr: wasmparser::ConstExpr, ty: ValType) -> Result<Vec<u8>, Error> {
    let mut out = None;
    for op in expr.get_operators_reader() {
        match op? {
            Operator::I32Const { value } => out = Some((value as u32).to_le_bytes().to_vec()),
            Operator::I64Const { value } => out = Some((value as u64).to_le_bytes().to_vec()),
            Operator::F32Const { value } => out = Some(value.bits().to_le_bytes().to_vec()),
            Operator::F64Const { value } => out = Some(value.bits().to_le_bytes().to_vec()),
            Operator::End => {}
            other => return unsup(format!("global initializer {other:?}")),
        }
    }
    let _ = ty;
    out.ok_or_else(|| Error::Parse("empty global initializer".into()))
}

/// Evaluate an active data segment's offset (a constant expression — `i32.const`/`i64.const`; a
/// `global.get` initializer needs immutable imported globals, deferred).
fn const_offset(expr: wasmparser::ConstExpr) -> Result<u64, Error> {
    let mut off = None;
    for op in expr.get_operators_reader() {
        match op? {
            Operator::I32Const { value } => off = Some(value as u32 as u64),
            Operator::I64Const { value } => off = Some(value as u64),
            Operator::End => {}
            other => return unsup(format!("data offset expression {other:?}")),
        }
    }
    off.ok_or_else(|| Error::Parse("empty data offset expression".into()))
}

/// A block under construction: SSA values are block-local indices — params first (`0..params.len()`),
/// then each **value-producing** instruction's result. `next_val` tracks that index (a `store` is an
/// instruction but produces no value, so it must not consume an index). The terminator is filled when
/// the block ends.
struct BlockB {
    params: Vec<ValType>,
    insts: Vec<Inst>,
    next_val: ValIdx,
    term: Option<Terminator>,
}

/// Where a `br` to a control label goes, and what it carries (besides the always-threaded locals).
enum Tgt {
    /// The function's implicit outermost label: a branch returns the result values.
    Return,
    /// A forward `block`/`if` label → the merge IR block after it (carries the block's results). The
    /// block index itself lives in `Frame::end_merge`, realized lazily on the first exit.
    Merge,
    /// A backward `loop` label → the loop header IR block (carries the loop's params).
    Loop(usize),
}

/// One entry on the control stack — a wasm `block`/`loop`/`if` (or the function frame).
struct Frame {
    target: Tgt,
    /// Values a `br` to this label carries (results for block/if, params for loop, results for fn).
    br_arity: usize,
    /// Operand-stack height *below* the carried values when this frame was entered (the preserved
    /// base): `entry_height - n_params`. `br` keeps the top `br_arity` and unwinds to here.
    base: usize,
    /// Result types (what falls through the matching `end`), and the `end` merge block (lazy).
    results: Vec<ValType>,
    end_merge: Option<usize>,
    /// Present for a *live* `if` (not a dead placeholder): the else arm's block, the if's param types
    /// (for an `if` without `else`, where the inputs pass through as the results), and whether we have
    /// switched into the else arm yet.
    if_else: Option<IfElse>,
    /// `true` if this frame was pushed while control was unreachable (a placeholder that only needs to
    /// balance the matching `end`; never branched to from live code).
    dead: bool,
}

struct IfElse {
    else_block: usize,
    params: Vec<ValType>,
    in_else: bool,
}

struct Lower<'a> {
    blocks: Vec<BlockB>,
    cur: usize,
    /// Current SSA value of each local (param then declared), in `cur`'s value space.
    locals: Vec<ValIdx>,
    local_types: Vec<ValType>,
    /// Operand stack: (value, type).
    stack: Vec<(ValIdx, ValType)>,
    reachable: bool,
    control: Vec<Frame>,
    types: &'a [(Vec<ValType>, Vec<ValType>)],
    /// Per-function signatures by function index (for `call`). No imports, so wasm function index =
    /// our `Module` function index.
    func_sigs: &'a [(Vec<ValType>, Vec<ValType>)],
    /// Global types by index, and the window byte address of global 0 (each global an 8-byte slot).
    /// `global.get`/`set` lower to a load/store there.
    global_types: &'a [ValType],
    globals_base: u64,
    /// Window byte address of function-table slot 0 (each slot an i32 funcref index). `call_indirect`
    /// loads the slot and feeds it to our `CallIndirect`.
    table_base: u64,
    /// 64-bit linear memory (`memory64`): the address operand is already i64; otherwise it's an i32
    /// that must be zero-extended before our (i64-addressed) `load`/`store`.
    mem64: bool,
}

impl Lower<'_> {
    fn new_block(&mut self, params: Vec<ValType>) -> usize {
        let next_val = params.len() as ValIdx;
        self.blocks.push(BlockB {
            params,
            insts: Vec::new(),
            next_val,
            term: None,
        });
        self.blocks.len() - 1
    }

    /// Append a **value-producing** instruction and return its SSA value index.
    fn emit(&mut self, inst: Inst) -> ValIdx {
        let b = &mut self.blocks[self.cur];
        let idx = b.next_val;
        b.next_val += 1;
        b.insts.push(inst);
        idx
    }

    /// Append an instruction that produces **no** value (`store`/`atomic.store`): it does not consume
    /// a value index.
    fn emit_void(&mut self, inst: Inst) {
        self.blocks[self.cur].insts.push(inst);
    }

    /// Append a `call` producing `n` results (a multi-result call occupies `n` consecutive value
    /// indices — the callee's results are appended to the caller's value space in order).
    fn emit_call(&mut self, inst: Inst, n: usize) -> Vec<ValIdx> {
        let b = &mut self.blocks[self.cur];
        let start = b.next_val;
        b.next_val += n as ValIdx;
        b.insts.push(inst);
        (start..start + n as ValIdx).collect()
    }

    fn push(&mut self, v: ValIdx, t: ValType) {
        self.stack.push((v, t));
    }
    fn pop(&mut self) -> Result<(ValIdx, ValType), Error> {
        self.stack
            .pop()
            .ok_or_else(|| Error::Parse("operand stack underflow".into()))
    }

    /// The block-parameter signature for a target carrying `carried` stack types: every IR block
    /// threads all locals first, then the surviving stack.
    fn sig(&self, carried: &[ValType]) -> Vec<ValType> {
        let mut s = self.local_types.clone();
        s.extend_from_slice(carried);
        s
    }

    /// The arguments for a branch to a frame: all current locals, then the preserved base of the
    /// target and the top `arity` carried values (the middle is unwound away).
    fn branch_args(&self, base: usize, arity: usize) -> Vec<ValIdx> {
        let mut a = self.locals.clone();
        a.extend(self.stack[..base].iter().map(|(v, _)| *v));
        a.extend(
            self.stack[self.stack.len() - arity..]
                .iter()
                .map(|(v, _)| *v),
        );
        a
    }
    /// The stack *types* a branch to a frame carries (base ++ top `arity`).
    fn carried_types(&self, base: usize, arity: usize) -> Vec<ValType> {
        let mut t: Vec<ValType> = self.stack[..base].iter().map(|(_, t)| *t).collect();
        t.extend(
            self.stack[self.stack.len() - arity..]
                .iter()
                .map(|(_, t)| *t),
        );
        t
    }

    /// Make `blk` current and rebind locals + stack to its parameters. `stack_types` is the carried
    /// stack layout (after the locals) — the values become params `local_types.len()..`.
    fn enter(&mut self, blk: usize, stack_types: &[ValType]) {
        self.cur = blk;
        let nl = self.local_types.len();
        self.locals = (0..nl as ValIdx).collect();
        self.stack = stack_types
            .iter()
            .enumerate()
            .map(|(i, t)| ((nl + i) as ValIdx, *t))
            .collect();
        self.reachable = true;
    }

    fn set_term(&mut self, t: Terminator) {
        self.blocks[self.cur].term = Some(t);
        self.reachable = false;
    }

    /// The carried stack types a merge expects, read back from its params (locals stripped).
    fn merge_stack_types(&self, m: usize) -> Vec<ValType> {
        self.blocks[m].params[self.local_types.len()..].to_vec()
    }
}

/// Block-type → (param types, result types).
fn block_sig(
    bt: BlockType,
    types: &[(Vec<ValType>, Vec<ValType>)],
) -> Result<(Vec<ValType>, Vec<ValType>), Error> {
    match bt {
        BlockType::Empty => Ok((vec![], vec![])),
        BlockType::Type(t) => Ok((vec![], vec![val_type(t)?])),
        BlockType::FuncType(i) => {
            let (p, r) = &types[i as usize];
            Ok((p.clone(), r.clone()))
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn lower_func(
    params: &[ValType],
    results: &[ValType],
    types: &[(Vec<ValType>, Vec<ValType>)],
    func_sigs: &[(Vec<ValType>, Vec<ValType>)],
    global_types: &[ValType],
    globals_base: u64,
    table_base: u64,
    body: &wasmparser::FunctionBody,
    mem64: bool,
) -> Result<Func, Error> {
    // Locals = params (with their incoming param values) then declared locals (default 0).
    let mut local_types: Vec<ValType> = params.to_vec();
    for decl in body.get_locals_reader()? {
        let (count, t) = decl?;
        let t = val_type(t)?;
        for _ in 0..count {
            local_types.push(t);
        }
    }

    let entry = BlockB {
        params: params.to_vec(),
        insts: Vec::new(),
        next_val: params.len() as ValIdx,
        term: None,
    };
    let mut lo = Lower {
        blocks: vec![entry],
        cur: 0,
        locals: (0..params.len() as ValIdx).collect(),
        local_types: local_types.clone(),
        stack: Vec::new(),
        reachable: true,
        control: Vec::new(),
        types,
        func_sigs,
        global_types,
        globals_base,
        table_base,
        mem64,
    };
    // Initialize declared locals to zero (params already bound to block params), extending `locals`.
    for t in &local_types[params.len()..] {
        let v = match t {
            ValType::I32 => lo.emit(Inst::ConstI32(0)),
            ValType::I64 => lo.emit(Inst::ConstI64(0)),
            ValType::F32 => lo.emit(Inst::ConstF32(0)),
            ValType::F64 => lo.emit(Inst::ConstF64(0)),
        };
        lo.locals.push(v);
    }

    // The implicit function frame: a `br` to the outermost label (or the final `end`) returns.
    lo.control.push(Frame {
        target: Tgt::Return,
        br_arity: results.len(),
        base: 0,
        results: results.to_vec(),
        end_merge: None,
        if_else: None,
        dead: false,
    });

    for op in body.get_operators_reader()? {
        lower_op(&mut lo, op?, results)?;
    }

    let blocks = lo
        .blocks
        .into_iter()
        .map(|b| Block {
            params: b.params,
            insts: b.insts,
            // An un-terminated block is unreachable code wasm validation allows; make it explicit.
            term: b.term.unwrap_or(Terminator::Unreachable),
        })
        .collect();
    Ok(Func {
        params: params.to_vec(),
        results: results.to_vec(),
        blocks,
    })
}

fn int_bin(lo: &mut Lower, ty: IntTy, op: BinOp) -> Result<(), Error> {
    let (b, _) = lo.pop()?;
    let (a, _) = lo.pop()?;
    let v = lo.emit(Inst::IntBin { ty, op, a, b });
    lo.push(v, int_val(ty));
    Ok(())
}
fn int_cmp(lo: &mut Lower, ty: IntTy, op: CmpOp) -> Result<(), Error> {
    let (b, _) = lo.pop()?;
    let (a, _) = lo.pop()?;
    let v = lo.emit(Inst::IntCmp { ty, op, a, b });
    lo.push(v, ValType::I32); // comparisons yield i32
    Ok(())
}
fn int_un(lo: &mut Lower, ty: IntTy, op: IntUnOp) -> Result<(), Error> {
    let (a, _) = lo.pop()?;
    let v = lo.emit(Inst::IntUn { ty, op, a });
    lo.push(v, int_val(ty));
    Ok(())
}
fn convert(lo: &mut Lower, op: ConvOp, out: ValType) -> Result<(), Error> {
    let (a, _) = lo.pop()?;
    let v = lo.emit(Inst::Convert { op, a });
    lo.push(v, out);
    Ok(())
}
fn int_val(ty: IntTy) -> ValType {
    match ty {
        IntTy::I32 => ValType::I32,
        IntTy::I64 => ValType::I64,
    }
}
fn float_val(ty: FloatTy) -> ValType {
    match ty {
        FloatTy::F32 => ValType::F32,
        FloatTy::F64 => ValType::F64,
    }
}
fn fbin(lo: &mut Lower, ty: FloatTy, op: FBinOp) -> Result<(), Error> {
    let (b, _) = lo.pop()?;
    let (a, _) = lo.pop()?;
    let v = lo.emit(Inst::FBin { ty, op, a, b });
    lo.push(v, float_val(ty));
    Ok(())
}
fn fun(lo: &mut Lower, ty: FloatTy, op: FUnOp) -> Result<(), Error> {
    let (a, _) = lo.pop()?;
    let v = lo.emit(Inst::FUn { ty, op, a });
    lo.push(v, float_val(ty));
    Ok(())
}
fn fcmp(lo: &mut Lower, ty: FloatTy, op: FCmpOp) -> Result<(), Error> {
    let (b, _) = lo.pop()?;
    let (a, _) = lo.pop()?;
    let v = lo.emit(Inst::FCmp { ty, op, a, b });
    lo.push(v, ValType::I32);
    Ok(())
}
/// float → int. wasm `trunc_*` traps on out-of-range/NaN; `trunc_sat_*` saturates.
fn ftoi(lo: &mut Lower, op: FToI, sat: bool, out: ValType) -> Result<(), Error> {
    let (a, _) = lo.pop()?;
    let v = lo.emit(if sat {
        Inst::FToISat { op, a }
    } else {
        Inst::FToITrap { op, a }
    });
    lo.push(v, out);
    Ok(())
}
fn itof(lo: &mut Lower, op: IToF, out: ValType) -> Result<(), Error> {
    let (a, _) = lo.pop()?;
    let v = lo.emit(Inst::IToFConv { op, a });
    lo.push(v, out);
    Ok(())
}
fn fcast(lo: &mut Lower, op: CastOp, out: ValType) -> Result<(), Error> {
    let (a, _) = lo.pop()?;
    let v = lo.emit(Inst::Cast { op, a });
    lo.push(v, out);
    Ok(())
}

/// Pop the wasm address (an i32 for a 32-bit memory, zero-extended to our i64 address space; an i64 for
/// `memory64`).
fn pop_addr(lo: &mut Lower) -> Result<ValIdx, Error> {
    let (a, _) = lo.pop()?;
    Ok(if lo.mem64 {
        a
    } else {
        lo.emit(Inst::Convert {
            op: ConvOp::ExtendI32U,
            a,
        })
    })
}
fn mem_load(lo: &mut Lower, op: LoadOp, m: MemArg) -> Result<(), Error> {
    let addr = pop_addr(lo)?;
    let v = lo.emit(Inst::Load {
        op,
        addr,
        offset: m.offset,
        align: m.align,
    });
    lo.push(v, op.info().1); // `info().1` is the result value type (i32/i64)
    Ok(())
}
fn global_addr(lo: &Lower, g: u32) -> u64 {
    lo.globals_base + g as u64 * 8
}
/// The full-width load/store op for a value type (globals occupy whole 8-byte slots).
fn load_op(ty: ValType) -> LoadOp {
    match ty {
        ValType::I32 => LoadOp::I32,
        ValType::I64 => LoadOp::I64,
        ValType::F32 => LoadOp::F32,
        ValType::F64 => LoadOp::F64,
    }
}
fn store_op(ty: ValType) -> StoreOp {
    match ty {
        ValType::I32 => StoreOp::I32,
        ValType::I64 => StoreOp::I64,
        ValType::F32 => StoreOp::F32,
        ValType::F64 => StoreOp::F64,
    }
}

/// `call funcidx`: pop the callee's params (the last is on top), call it, push its results. No
/// imports, so the wasm function index is our `Module` function index directly.
fn call_op(lo: &mut Lower, func: u32) -> Result<(), Error> {
    let (params, results) = lo
        .func_sigs
        .get(func as usize)
        .ok_or_else(|| Error::Parse(format!("call to unknown function {func}")))?
        .clone();
    let mut args = Vec::with_capacity(params.len());
    for _ in 0..params.len() {
        args.push(lo.pop()?.0);
    }
    args.reverse(); // stack top is the last argument
    let res = lo.emit_call(Inst::Call { func, args }, results.len());
    for (v, t) in res.into_iter().zip(results.iter()) {
        lo.push(v, *t);
    }
    Ok(())
}

/// `call_indirect (type $t)`: the stack is `[args.., index]`. Pop the index, load the funcref (a
/// function index) from `table[index]` in the window, pop the args, and emit our `CallIndirect` (whose
/// `table_lookup` does the §3c type-id check against `$t`).
fn call_indirect_op(lo: &mut Lower, type_index: u32, table_index: u32) -> Result<(), Error> {
    if table_index != 0 {
        return unsup("call_indirect on a non-zero table");
    }
    let (params, results) = lo.types[type_index as usize].clone();
    // table[index] → function index, at window byte `table_base + index*4`.
    let (widx, _) = lo.pop()?;
    let idx64 = lo.emit(Inst::Convert {
        op: ConvOp::ExtendI32U,
        a: widx,
    });
    let four = lo.emit(Inst::ConstI64(4));
    let byte_off = lo.emit(Inst::IntBin {
        ty: IntTy::I64,
        op: BinOp::Mul,
        a: idx64,
        b: four,
    });
    let funcref = lo.emit(Inst::Load {
        op: LoadOp::I32,
        addr: byte_off,
        offset: lo.table_base,
        align: 2,
    });
    let mut args = Vec::with_capacity(params.len());
    for _ in 0..params.len() {
        args.push(lo.pop()?.0);
    }
    args.reverse();
    let ty = FuncType {
        params: params.clone(),
        results: results.clone(),
    };
    let res = lo.emit_call(
        Inst::CallIndirect {
            ty,
            idx: funcref,
            args,
        },
        results.len(),
    );
    for (v, t) in res.into_iter().zip(results.iter()) {
        lo.push(v, *t);
    }
    Ok(())
}

fn mem_store(lo: &mut Lower, op: StoreOp, m: MemArg) -> Result<(), Error> {
    let (value, _) = lo.pop()?;
    let addr = pop_addr(lo)?;
    lo.emit_void(Inst::Store {
        op,
        addr,
        value,
        offset: m.offset,
        align: m.align,
    });
    Ok(())
}

fn lower_op(lo: &mut Lower, op: Operator, fn_results: &[ValType]) -> Result<(), Error> {
    use Operator as O;
    // Dead code after a branch/return/unreachable: track structure (block depth) but emit nothing
    // until the matching `end` restores reachability.
    if !lo.reachable {
        return skip_unreachable(lo, op);
    }
    match op {
        O::Nop => {}
        O::Drop => {
            lo.pop()?;
        }
        O::Unreachable => lo.set_term(Terminator::Unreachable),
        O::I32Const { value } => {
            let v = lo.emit(Inst::ConstI32(value));
            lo.push(v, ValType::I32);
        }
        O::I64Const { value } => {
            let v = lo.emit(Inst::ConstI64(value));
            lo.push(v, ValType::I64);
        }
        O::LocalGet { local_index } => {
            let i = local_index as usize;
            lo.push(lo.locals[i], lo.local_types[i]);
        }
        O::LocalSet { local_index } => {
            let (v, _) = lo.pop()?;
            lo.locals[local_index as usize] = v;
        }
        O::LocalTee { local_index } => {
            let (v, _) = *lo
                .stack
                .last()
                .ok_or_else(|| Error::Parse("tee on empty stack".into()))?;
            lo.locals[local_index as usize] = v;
        }
        O::Select => {
            let (c, _) = lo.pop()?;
            let (b, _) = lo.pop()?;
            let (a, t) = lo.pop()?;
            let v = lo.emit(Inst::Select { cond: c, a, b });
            lo.push(v, t);
        }
        // ---- integer arithmetic / bitwise / shifts ----
        O::I32Add => int_bin(lo, IntTy::I32, BinOp::Add)?,
        O::I32Sub => int_bin(lo, IntTy::I32, BinOp::Sub)?,
        O::I32Mul => int_bin(lo, IntTy::I32, BinOp::Mul)?,
        O::I32DivS => int_bin(lo, IntTy::I32, BinOp::DivS)?,
        O::I32DivU => int_bin(lo, IntTy::I32, BinOp::DivU)?,
        O::I32RemS => int_bin(lo, IntTy::I32, BinOp::RemS)?,
        O::I32RemU => int_bin(lo, IntTy::I32, BinOp::RemU)?,
        O::I32And => int_bin(lo, IntTy::I32, BinOp::And)?,
        O::I32Or => int_bin(lo, IntTy::I32, BinOp::Or)?,
        O::I32Xor => int_bin(lo, IntTy::I32, BinOp::Xor)?,
        O::I32Shl => int_bin(lo, IntTy::I32, BinOp::Shl)?,
        O::I32ShrS => int_bin(lo, IntTy::I32, BinOp::ShrS)?,
        O::I32ShrU => int_bin(lo, IntTy::I32, BinOp::ShrU)?,
        O::I32Rotl => int_bin(lo, IntTy::I32, BinOp::Rotl)?,
        O::I32Rotr => int_bin(lo, IntTy::I32, BinOp::Rotr)?,
        O::I64Add => int_bin(lo, IntTy::I64, BinOp::Add)?,
        O::I64Sub => int_bin(lo, IntTy::I64, BinOp::Sub)?,
        O::I64Mul => int_bin(lo, IntTy::I64, BinOp::Mul)?,
        O::I64DivS => int_bin(lo, IntTy::I64, BinOp::DivS)?,
        O::I64DivU => int_bin(lo, IntTy::I64, BinOp::DivU)?,
        O::I64RemS => int_bin(lo, IntTy::I64, BinOp::RemS)?,
        O::I64RemU => int_bin(lo, IntTy::I64, BinOp::RemU)?,
        O::I64And => int_bin(lo, IntTy::I64, BinOp::And)?,
        O::I64Or => int_bin(lo, IntTy::I64, BinOp::Or)?,
        O::I64Xor => int_bin(lo, IntTy::I64, BinOp::Xor)?,
        O::I64Shl => int_bin(lo, IntTy::I64, BinOp::Shl)?,
        O::I64ShrS => int_bin(lo, IntTy::I64, BinOp::ShrS)?,
        O::I64ShrU => int_bin(lo, IntTy::I64, BinOp::ShrU)?,
        O::I64Rotl => int_bin(lo, IntTy::I64, BinOp::Rotl)?,
        O::I64Rotr => int_bin(lo, IntTy::I64, BinOp::Rotr)?,
        // ---- unary ----
        O::I32Clz => int_un(lo, IntTy::I32, IntUnOp::Clz)?,
        O::I32Ctz => int_un(lo, IntTy::I32, IntUnOp::Ctz)?,
        O::I32Popcnt => int_un(lo, IntTy::I32, IntUnOp::Popcnt)?,
        O::I64Clz => int_un(lo, IntTy::I64, IntUnOp::Clz)?,
        O::I64Ctz => int_un(lo, IntTy::I64, IntUnOp::Ctz)?,
        O::I64Popcnt => int_un(lo, IntTy::I64, IntUnOp::Popcnt)?,
        O::I32Extend8S => int_un(lo, IntTy::I32, IntUnOp::Extend8S)?,
        O::I32Extend16S => int_un(lo, IntTy::I32, IntUnOp::Extend16S)?,
        O::I64Extend8S => int_un(lo, IntTy::I64, IntUnOp::Extend8S)?,
        O::I64Extend16S => int_un(lo, IntTy::I64, IntUnOp::Extend16S)?,
        O::I64Extend32S => int_un(lo, IntTy::I64, IntUnOp::Extend32S)?,
        // ---- comparisons ----
        O::I32Eqz => {
            let (a, _) = lo.pop()?;
            let v = lo.emit(Inst::Eqz { ty: IntTy::I32, a });
            lo.push(v, ValType::I32);
        }
        O::I64Eqz => {
            let (a, _) = lo.pop()?;
            let v = lo.emit(Inst::Eqz { ty: IntTy::I64, a });
            lo.push(v, ValType::I32);
        }
        O::I32Eq => int_cmp(lo, IntTy::I32, CmpOp::Eq)?,
        O::I32Ne => int_cmp(lo, IntTy::I32, CmpOp::Ne)?,
        O::I32LtS => int_cmp(lo, IntTy::I32, CmpOp::LtS)?,
        O::I32LtU => int_cmp(lo, IntTy::I32, CmpOp::LtU)?,
        O::I32LeS => int_cmp(lo, IntTy::I32, CmpOp::LeS)?,
        O::I32LeU => int_cmp(lo, IntTy::I32, CmpOp::LeU)?,
        O::I32GtS => int_cmp(lo, IntTy::I32, CmpOp::GtS)?,
        O::I32GtU => int_cmp(lo, IntTy::I32, CmpOp::GtU)?,
        O::I32GeS => int_cmp(lo, IntTy::I32, CmpOp::GeS)?,
        O::I32GeU => int_cmp(lo, IntTy::I32, CmpOp::GeU)?,
        O::I64Eq => int_cmp(lo, IntTy::I64, CmpOp::Eq)?,
        O::I64Ne => int_cmp(lo, IntTy::I64, CmpOp::Ne)?,
        O::I64LtS => int_cmp(lo, IntTy::I64, CmpOp::LtS)?,
        O::I64LtU => int_cmp(lo, IntTy::I64, CmpOp::LtU)?,
        O::I64LeS => int_cmp(lo, IntTy::I64, CmpOp::LeS)?,
        O::I64LeU => int_cmp(lo, IntTy::I64, CmpOp::LeU)?,
        O::I64GtS => int_cmp(lo, IntTy::I64, CmpOp::GtS)?,
        O::I64GtU => int_cmp(lo, IntTy::I64, CmpOp::GtU)?,
        O::I64GeS => int_cmp(lo, IntTy::I64, CmpOp::GeS)?,
        O::I64GeU => int_cmp(lo, IntTy::I64, CmpOp::GeU)?,
        // ---- integer conversions ----
        O::I64ExtendI32S => convert(lo, ConvOp::ExtendI32S, ValType::I64)?,
        O::I64ExtendI32U => convert(lo, ConvOp::ExtendI32U, ValType::I64)?,
        O::I32WrapI64 => convert(lo, ConvOp::WrapI64, ValType::I32)?,
        // ---- linear memory load / store (i32/i64; floats are a later slice) ----
        O::I32Load { memarg } => mem_load(lo, LoadOp::I32, memarg)?,
        O::I64Load { memarg } => mem_load(lo, LoadOp::I64, memarg)?,
        O::I32Load8S { memarg } => mem_load(lo, LoadOp::I32_8S, memarg)?,
        O::I32Load8U { memarg } => mem_load(lo, LoadOp::I32_8U, memarg)?,
        O::I32Load16S { memarg } => mem_load(lo, LoadOp::I32_16S, memarg)?,
        O::I32Load16U { memarg } => mem_load(lo, LoadOp::I32_16U, memarg)?,
        O::I64Load8S { memarg } => mem_load(lo, LoadOp::I64_8S, memarg)?,
        O::I64Load8U { memarg } => mem_load(lo, LoadOp::I64_8U, memarg)?,
        O::I64Load16S { memarg } => mem_load(lo, LoadOp::I64_16S, memarg)?,
        O::I64Load16U { memarg } => mem_load(lo, LoadOp::I64_16U, memarg)?,
        O::I64Load32S { memarg } => mem_load(lo, LoadOp::I64_32S, memarg)?,
        O::I64Load32U { memarg } => mem_load(lo, LoadOp::I64_32U, memarg)?,
        O::I32Store { memarg } => mem_store(lo, StoreOp::I32, memarg)?,
        O::I64Store { memarg } => mem_store(lo, StoreOp::I64, memarg)?,
        O::I32Store8 { memarg } => mem_store(lo, StoreOp::I32_8, memarg)?,
        O::I32Store16 { memarg } => mem_store(lo, StoreOp::I32_16, memarg)?,
        O::I64Store8 { memarg } => mem_store(lo, StoreOp::I64_8, memarg)?,
        O::I64Store16 { memarg } => mem_store(lo, StoreOp::I64_16, memarg)?,
        O::I64Store32 { memarg } => mem_store(lo, StoreOp::I64_32, memarg)?,
        // ---- floats: const / arithmetic / unary / compare ----
        O::F32Const { value } => {
            let v = lo.emit(Inst::ConstF32(value.bits()));
            lo.push(v, ValType::F32);
        }
        O::F64Const { value } => {
            let v = lo.emit(Inst::ConstF64(value.bits()));
            lo.push(v, ValType::F64);
        }
        O::F32Add => fbin(lo, FloatTy::F32, FBinOp::Add)?,
        O::F32Sub => fbin(lo, FloatTy::F32, FBinOp::Sub)?,
        O::F32Mul => fbin(lo, FloatTy::F32, FBinOp::Mul)?,
        O::F32Div => fbin(lo, FloatTy::F32, FBinOp::Div)?,
        O::F32Min => fbin(lo, FloatTy::F32, FBinOp::Min)?,
        O::F32Max => fbin(lo, FloatTy::F32, FBinOp::Max)?,
        O::F32Copysign => fbin(lo, FloatTy::F32, FBinOp::Copysign)?,
        O::F64Add => fbin(lo, FloatTy::F64, FBinOp::Add)?,
        O::F64Sub => fbin(lo, FloatTy::F64, FBinOp::Sub)?,
        O::F64Mul => fbin(lo, FloatTy::F64, FBinOp::Mul)?,
        O::F64Div => fbin(lo, FloatTy::F64, FBinOp::Div)?,
        O::F64Min => fbin(lo, FloatTy::F64, FBinOp::Min)?,
        O::F64Max => fbin(lo, FloatTy::F64, FBinOp::Max)?,
        O::F64Copysign => fbin(lo, FloatTy::F64, FBinOp::Copysign)?,
        O::F32Abs => fun(lo, FloatTy::F32, FUnOp::Abs)?,
        O::F32Neg => fun(lo, FloatTy::F32, FUnOp::Neg)?,
        O::F32Sqrt => fun(lo, FloatTy::F32, FUnOp::Sqrt)?,
        O::F32Ceil => fun(lo, FloatTy::F32, FUnOp::Ceil)?,
        O::F32Floor => fun(lo, FloatTy::F32, FUnOp::Floor)?,
        O::F32Trunc => fun(lo, FloatTy::F32, FUnOp::Trunc)?,
        O::F32Nearest => fun(lo, FloatTy::F32, FUnOp::Nearest)?,
        O::F64Abs => fun(lo, FloatTy::F64, FUnOp::Abs)?,
        O::F64Neg => fun(lo, FloatTy::F64, FUnOp::Neg)?,
        O::F64Sqrt => fun(lo, FloatTy::F64, FUnOp::Sqrt)?,
        O::F64Ceil => fun(lo, FloatTy::F64, FUnOp::Ceil)?,
        O::F64Floor => fun(lo, FloatTy::F64, FUnOp::Floor)?,
        O::F64Trunc => fun(lo, FloatTy::F64, FUnOp::Trunc)?,
        O::F64Nearest => fun(lo, FloatTy::F64, FUnOp::Nearest)?,
        O::F32Eq => fcmp(lo, FloatTy::F32, FCmpOp::Eq)?,
        O::F32Ne => fcmp(lo, FloatTy::F32, FCmpOp::Ne)?,
        O::F32Lt => fcmp(lo, FloatTy::F32, FCmpOp::Lt)?,
        O::F32Le => fcmp(lo, FloatTy::F32, FCmpOp::Le)?,
        O::F32Gt => fcmp(lo, FloatTy::F32, FCmpOp::Gt)?,
        O::F32Ge => fcmp(lo, FloatTy::F32, FCmpOp::Ge)?,
        O::F64Eq => fcmp(lo, FloatTy::F64, FCmpOp::Eq)?,
        O::F64Ne => fcmp(lo, FloatTy::F64, FCmpOp::Ne)?,
        O::F64Lt => fcmp(lo, FloatTy::F64, FCmpOp::Lt)?,
        O::F64Le => fcmp(lo, FloatTy::F64, FCmpOp::Le)?,
        O::F64Gt => fcmp(lo, FloatTy::F64, FCmpOp::Gt)?,
        O::F64Ge => fcmp(lo, FloatTy::F64, FCmpOp::Ge)?,
        // ---- float load / store ----
        O::F32Load { memarg } => mem_load(lo, LoadOp::F32, memarg)?,
        O::F64Load { memarg } => mem_load(lo, LoadOp::F64, memarg)?,
        O::F32Store { memarg } => mem_store(lo, StoreOp::F32, memarg)?,
        O::F64Store { memarg } => mem_store(lo, StoreOp::F64, memarg)?,
        // ---- float ↔ int conversions (trunc traps; trunc_sat saturates) ----
        O::I32TruncF32S => ftoi(lo, FToI::F32I32S, false, ValType::I32)?,
        O::I32TruncF32U => ftoi(lo, FToI::F32I32U, false, ValType::I32)?,
        O::I32TruncF64S => ftoi(lo, FToI::F64I32S, false, ValType::I32)?,
        O::I32TruncF64U => ftoi(lo, FToI::F64I32U, false, ValType::I32)?,
        O::I64TruncF32S => ftoi(lo, FToI::F32I64S, false, ValType::I64)?,
        O::I64TruncF32U => ftoi(lo, FToI::F32I64U, false, ValType::I64)?,
        O::I64TruncF64S => ftoi(lo, FToI::F64I64S, false, ValType::I64)?,
        O::I64TruncF64U => ftoi(lo, FToI::F64I64U, false, ValType::I64)?,
        O::I32TruncSatF32S => ftoi(lo, FToI::F32I32S, true, ValType::I32)?,
        O::I32TruncSatF32U => ftoi(lo, FToI::F32I32U, true, ValType::I32)?,
        O::I32TruncSatF64S => ftoi(lo, FToI::F64I32S, true, ValType::I32)?,
        O::I32TruncSatF64U => ftoi(lo, FToI::F64I32U, true, ValType::I32)?,
        O::I64TruncSatF32S => ftoi(lo, FToI::F32I64S, true, ValType::I64)?,
        O::I64TruncSatF32U => ftoi(lo, FToI::F32I64U, true, ValType::I64)?,
        O::I64TruncSatF64S => ftoi(lo, FToI::F64I64S, true, ValType::I64)?,
        O::I64TruncSatF64U => ftoi(lo, FToI::F64I64U, true, ValType::I64)?,
        O::F32ConvertI32S => itof(lo, IToF::I32F32S, ValType::F32)?,
        O::F32ConvertI32U => itof(lo, IToF::I32F32U, ValType::F32)?,
        O::F32ConvertI64S => itof(lo, IToF::I64F32S, ValType::F32)?,
        O::F32ConvertI64U => itof(lo, IToF::I64F32U, ValType::F32)?,
        O::F64ConvertI32S => itof(lo, IToF::I32F64S, ValType::F64)?,
        O::F64ConvertI32U => itof(lo, IToF::I32F64U, ValType::F64)?,
        O::F64ConvertI64S => itof(lo, IToF::I64F64S, ValType::F64)?,
        O::F64ConvertI64U => itof(lo, IToF::I64F64U, ValType::F64)?,
        O::F32DemoteF64 => fcast(lo, CastOp::Demote, ValType::F32)?,
        O::F64PromoteF32 => fcast(lo, CastOp::Promote, ValType::F64)?,
        O::I32ReinterpretF32 => fcast(lo, CastOp::ReinterpF32I32, ValType::I32)?,
        O::F32ReinterpretI32 => fcast(lo, CastOp::ReinterpI32F32, ValType::F32)?,
        O::I64ReinterpretF64 => fcast(lo, CastOp::ReinterpF64I64, ValType::I64)?,
        O::F64ReinterpretI64 => fcast(lo, CastOp::ReinterpI64F64, ValType::F64)?,
        // ---- globals (lowered to load/store of a reserved window slot) ----
        O::GlobalGet { global_index } => {
            let ty = lo.global_types[global_index as usize];
            let a = lo.emit(Inst::ConstI64(global_addr(lo, global_index) as i64));
            let v = lo.emit(Inst::Load {
                op: load_op(ty),
                addr: a,
                offset: 0,
                align: 3,
            });
            lo.push(v, ty);
        }
        O::GlobalSet { global_index } => {
            let ty = lo.global_types[global_index as usize];
            let (value, _) = lo.pop()?;
            let a = lo.emit(Inst::ConstI64(global_addr(lo, global_index) as i64));
            lo.emit_void(Inst::Store {
                op: store_op(ty),
                addr: a,
                value,
                offset: 0,
                align: 3,
            });
        }
        // ---- calls ----
        O::Call { function_index } => call_op(lo, function_index)?,
        O::CallIndirect {
            type_index,
            table_index,
        } => call_indirect_op(lo, type_index, table_index)?,
        // ---- structured control flow ----
        O::Block { blockty } => {
            let (p, r) = block_sig(blockty, lo.types)?;
            lo.control.push(Frame {
                target: Tgt::Merge,
                br_arity: r.len(),
                base: lo.stack.len() - p.len(),
                results: r,
                end_merge: None,
                if_else: None,
                dead: false,
            });
        }
        O::Loop { blockty } => {
            let (p, r) = block_sig(blockty, lo.types)?;
            let base = lo.stack.len() - p.len();
            // The loop header carries locals + the entire entry stack (base ++ params).
            let carried = lo.carried_types(base, p.len());
            let hdr = lo.new_block(lo.sig(&carried));
            let args = lo.branch_args(base, p.len());
            lo.set_term(Terminator::Br {
                target: hdr as u32,
                args,
            });
            lo.enter(hdr, &carried);
            lo.control.push(Frame {
                target: Tgt::Loop(hdr),
                br_arity: p.len(),
                base,
                results: r,
                end_merge: None,
                if_else: None,
                dead: false,
            });
        }
        O::If { blockty } => if_op(lo, blockty)?,
        O::Else => else_op(lo)?,
        O::Br { relative_depth } => branch_to(lo, relative_depth as usize)?,
        O::BrIf { relative_depth } => {
            let (cond, _) = lo.pop()?;
            let d = relative_depth as usize;
            let fi = lo.control.len() - 1 - d;
            let (base, arity) = (lo.control[fi].base, lo.control[fi].br_arity);
            // Cond-true edge: the carried args + the resolved target.
            let then_blk = match lo.control[fi].target {
                Tgt::Return => return unsup("br_if targeting the function return"),
                _ => resolve_target(lo, d)?,
            };
            let then_args = lo.branch_args(base, arity);
            // Cond-false edge: continue in a fresh block carrying locals + the full current stack.
            let cont_types: Vec<ValType> = lo.stack.iter().map(|(_, t)| *t).collect();
            let cont = lo.new_block(lo.sig(&cont_types));
            let mut else_args = lo.locals.clone();
            else_args.extend(lo.stack.iter().map(|(v, _)| *v));
            lo.set_term(Terminator::BrIf {
                cond,
                then_blk: then_blk as u32,
                then_args,
                else_blk: cont as u32,
                else_args,
            });
            lo.enter(cont, &cont_types);
        }
        O::BrTable { targets } => {
            let (idx, _) = lo.pop()?;
            let mut edges: Vec<Edge> = Vec::new();
            for t in targets.targets() {
                edges.push(branch_edge(lo, t? as usize)?);
            }
            let default = branch_edge(lo, targets.default() as usize)?;
            lo.set_term(Terminator::BrTable {
                idx,
                targets: edges,
                default,
            });
        }
        O::Return => {
            let n = fn_results.len();
            let args: Vec<ValIdx> = lo.stack[lo.stack.len() - n..]
                .iter()
                .map(|(v, _)| *v)
                .collect();
            lo.set_term(Terminator::Return(args));
        }
        O::End => end_frame(lo)?,
        other => return unsup(format!("operator {other:?}")),
    }
    Ok(())
}

/// Resolve (creating if needed) the IR block a `br depth` targets, returning its index. Only valid for
/// `block`/`loop` targets — the function frame is handled separately (it returns).
fn resolve_target(lo: &mut Lower, depth: usize) -> Result<usize, Error> {
    let fi = lo.control.len() - 1 - depth;
    match lo.control[fi].target {
        Tgt::Loop(h) => Ok(h),
        Tgt::Merge => Ok(realize_merge(lo, fi)),
        Tgt::Return => unsup("internal: return target resolved as block"),
    }
}

/// Emit a `br depth` from the current (reachable) block.
fn branch_to(lo: &mut Lower, depth: usize) -> Result<(), Error> {
    let fi = lo.control.len() - 1 - depth;
    let (base, arity) = (lo.control[fi].base, lo.control[fi].br_arity);
    if let Tgt::Return = lo.control[fi].target {
        let args: Vec<ValIdx> = lo.stack[lo.stack.len() - arity..]
            .iter()
            .map(|(v, _)| *v)
            .collect();
        lo.set_term(Terminator::Return(args));
        return Ok(());
    }
    let args = lo.branch_args(base, arity);
    let blk = resolve_target(lo, depth)?;
    lo.set_term(Terminator::Br {
        target: blk as u32,
        args,
    });
    Ok(())
}

/// A `br_table` edge to `depth` (same carried-value layout as a `br`).
fn branch_edge(lo: &mut Lower, depth: usize) -> Result<Edge, Error> {
    let fi = lo.control.len() - 1 - depth;
    if let Tgt::Return = lo.control[fi].target {
        return unsup("br_table targeting the function return");
    }
    let (base, arity) = (lo.control[fi].base, lo.control[fi].br_arity);
    let args = lo.branch_args(base, arity);
    let blk = resolve_target(lo, depth)?;
    Ok((blk as u32, args)) // `Edge = (BlockIdx, Vec<ValIdx>)`
}

/// `if cond`: pop the condition and split into a then/else pair. Both arms start with the same state
/// (locals + the entry stack, the if's params on top), so they share the carried layout; a BrIf routes
/// to them. The merge after the if is created lazily on the first arm's exit (`else`/`end`/`br`).
fn if_op(lo: &mut Lower, blockty: BlockType) -> Result<(), Error> {
    let (p, r) = block_sig(blockty, lo.types)?;
    let (cond, _) = lo.pop()?;
    let base = lo.stack.len() - p.len();
    let carried: Vec<ValType> = lo.stack.iter().map(|(_, t)| *t).collect(); // base ++ params
    let then_blk = lo.new_block(lo.sig(&carried));
    let else_blk = lo.new_block(lo.sig(&carried));
    let mut args = lo.locals.clone();
    args.extend(lo.stack.iter().map(|(v, _)| *v));
    lo.set_term(Terminator::BrIf {
        cond,
        then_blk: then_blk as u32,
        then_args: args.clone(),
        else_blk: else_blk as u32,
        else_args: args,
    });
    lo.control.push(Frame {
        target: Tgt::Merge,
        br_arity: r.len(),
        base,
        results: r,
        end_merge: None,
        if_else: Some(IfElse {
            else_block: else_blk,
            params: p,
            in_else: false,
        }),
        dead: false,
    });
    lo.enter(then_blk, &carried);
    Ok(())
}

/// `else`: close the then arm (its fallthrough, if reachable, exits to the merge) and switch into the
/// else arm — which is reachable even if the then arm ended in a `br`. A no-op for a dead `if`.
fn else_op(lo: &mut Lower) -> Result<(), Error> {
    let i = lo.control.len() - 1;
    if lo.control[i].dead || lo.control[i].if_else.is_none() {
        return Ok(()); // the `else` of an unreachable `if`: nothing to switch into
    }
    let (base, arity) = (lo.control[i].base, lo.control[i].results.len());
    let merge = realize_merge(lo, i);
    if lo.reachable {
        let args = lo.branch_args(base, arity);
        lo.set_term(Terminator::Br {
            target: merge as u32,
            args,
        });
    }
    let else_blk = lo.control[i].if_else.as_ref().unwrap().else_block;
    let st = lo.merge_stack_types(else_blk); // base ++ params
    lo.enter(else_blk, &st);
    lo.control[i].if_else.as_mut().unwrap().in_else = true;
    Ok(())
}

/// Create a merge block carrying locals ++ the current preserved base ++ `results`.
fn make_merge(lo: &mut Lower, base: usize, results: &[ValType]) -> usize {
    let mut carried: Vec<ValType> = lo.stack[..base].iter().map(|(_, t)| *t).collect();
    carried.extend_from_slice(results);
    lo.new_block(lo.sig(&carried))
}

/// Realize (once) the merge block of the frame at index `i`, recording it as the frame's branch
/// target and `end` merge.
fn realize_merge(lo: &mut Lower, i: usize) -> usize {
    if let Some(m) = lo.control[i].end_merge {
        return m;
    }
    let (base, results) = (lo.control[i].base, lo.control[i].results.clone());
    let m = make_merge(lo, base, &results);
    lo.control[i].end_merge = Some(m);
    m
}

/// Handle `end`: close the current frame and continue in its merge (for the function frame, return).
fn end_frame(lo: &mut Lower) -> Result<(), Error> {
    let fr = lo.control.pop().expect("control underflow at end");
    if let Tgt::Return = fr.target {
        if lo.reachable {
            let n = fr.results.len();
            let args: Vec<ValIdx> = lo.stack[lo.stack.len() - n..]
                .iter()
                .map(|(v, _)| *v)
                .collect();
            lo.set_term(Terminator::Return(args));
        }
        return Ok(());
    }
    if fr.dead {
        // A placeholder from dead code: only balance. (A live `br` can't reach into a dead region.)
        if let Some(m) = fr.end_merge {
            let st = lo.merge_stack_types(m);
            lo.enter(m, &st);
        }
        return Ok(());
    }
    let (base, results) = (fr.base, fr.results.clone());
    if let Some(ie) = fr.if_else {
        // An `if`: both arms (or the then arm + an implicit pass-through else) exit to one merge.
        let merge = fr
            .end_merge
            .unwrap_or_else(|| make_merge(lo, base, &results));
        if !ie.in_else {
            // No `else`: current is the then arm; its fallthrough (if reachable) exits to merge, and
            // the implicit else forwards the if's inputs (params == results) through.
            if lo.reachable {
                let args = lo.branch_args(base, results.len());
                lo.set_term(Terminator::Br {
                    target: merge as u32,
                    args,
                });
            }
            let st = lo.merge_stack_types(ie.else_block); // base ++ params
            lo.enter(ie.else_block, &st);
            let args = lo.branch_args(base, ie.params.len());
            lo.set_term(Terminator::Br {
                target: merge as u32,
                args,
            });
        } else if lo.reachable {
            let args = lo.branch_args(base, results.len());
            lo.set_term(Terminator::Br {
                target: merge as u32,
                args,
            });
        }
        let st = lo.merge_stack_types(merge);
        lo.enter(merge, &st);
        return Ok(());
    }
    // block / loop frame.
    if lo.reachable {
        let m = fr
            .end_merge
            .unwrap_or_else(|| make_merge(lo, base, &results));
        let args = lo.branch_args(base, results.len());
        lo.set_term(Terminator::Br {
            target: m as u32,
            args,
        });
        let st = lo.merge_stack_types(m);
        lo.enter(m, &st);
    } else if let Some(m) = fr.end_merge {
        let st = lo.merge_stack_types(m);
        lo.enter(m, &st);
    }
    Ok(())
}

/// Track block structure through dead code (after a `br`/`return`/`unreachable`) without emitting.
/// wasm's polymorphic unreachable stack is approximated: control depth is tracked until a matching
/// `end`/`else` restores reachability (a live `if`'s else arm, or a live `br`'s merge).
fn skip_unreachable(lo: &mut Lower, op: Operator) -> Result<(), Error> {
    use Operator as O;
    match op {
        O::Block { .. } | O::Loop { .. } | O::If { .. } => {
            // A placeholder frame so the matching `end` balances; never branched to from live code.
            lo.control.push(Frame {
                target: Tgt::Merge,
                br_arity: 0,
                base: 0,
                results: vec![],
                end_merge: None,
                if_else: None,
                dead: true,
            });
            Ok(())
        }
        O::Else => else_op(lo), // a live `if`'s else arm resurrects even when the then arm went dead
        O::End => end_frame(lo),
        _ => Ok(()), // ignore every other op in dead code
    }
}
