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
//! (i32/i64, incl. narrow + `memory64`; the i32 address is zero-extended into our i64 window) ·
//! **`memory.size`/`memory.grow`** (pages; the window holds the full growable span, a runtime size
//! cell backs growth — see [`transpile`]) · **`memory.copy`/`memory.fill`** (a constant length is
//! unrolled into chunked load/stores; a runtime length lowers to a byte loop — `copy` is
//! direction-correct memmove) · direct **`call`** (multi-function + recursion) ·
//! **floats** (f32/f64 const/arith/unary/compare, load/store, and every int↔float conversion incl.
//! `trunc`/`trunc_sat`/`convert`/`demote`/`promote`/`reinterpret`) · **globals** (`global.get`/`set`
//! lowered to a reserved window region) · active **data segments** (initialized linear memory) ·
//! **`call_indirect`** + tables/element segments (the table → an in-window array of funcref indices;
//! `call_indirect` loads the entry and feeds it to our `CallIndirect`'s §3c type-id check) ·
//! **function imports** (a wasm `call` to an import → a `cap.call` on a threaded capability handle;
//! the host-ABI convention binds each import's `module`/`name` to a capability `type_id`/`op` — see
//! [`transpile`]) · **§17/D58 SIMD** (`v128` → the IR's first-class fixed-128 vector type: const,
//! masked load/store, splat, extract/replace_lane, integer-/float-lane arithmetic, bitwise +
//! `bitselect`, `shuffle`/`swizzle` — a real `clang -msimd128 -O2` saxpy transpiles to verified
//! SIMD IR, `tests/simd.rs`) · **§12 wasm threads**: the full-width (i32/i64) `*.atomic.*` ops map
//! 1:1 onto SVM's IR atomics (`tests/atomics.rs`), **shared** + **imported** memory are accepted,
//! and the **wasi-threads** ABI lowers to SVM's *native* `thread.spawn` — a `wasi:thread/spawn`
//! import becomes a real OS-thread vCPU over the shared window via a synthesized shim (concurrency in
//! the VM, DESIGN §1a; the same bytes `wasmtime-wasi-threads` runs — `tests/threads.rs`). Still a
//! clean [`Error::Unsupported`] (the niche features typical clang output doesn't emit): **narrow**
//! atomics (`*.atomic.rmw8`/`load16_u`/… — SVM atomics are 32/64-bit only); the `memory.init`/
//! `data.drop`/`table.*` bulk ops; passive data/element segments; imported table/global/tag; imports
//! across multiple capability interfaces (incl. wasi:thread/spawn *alongside* capability imports — the
//! per-thread handle stash); reference types; multi-memory/multi-table.

use svm_ir::{
    AtomicRmwOp, BinOp, Block, CastOp, CmpOp, ConvOp, Edge, FBinOp, FCmpOp, FToI, FUnOp, FloatTy,
    Func, FuncType, IToF, Inst, IntTy, IntUnOp, LoadOp, Module, Ordering, StoreOp, Terminator,
    VBitBinOp, VFloatBinOp, VFloatUnOp, VIntBinOp, VShape, ValIdx, ValType,
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

/// A wasm linear-memory page is 64 KiB.
const WASM_PAGE: u64 = 1 << 16;
/// The standard **wasi-threads** spawn import — `(import "wasi" "thread-spawn" (func (param i32)
/// (result i32)))` — which the host implements by starting a new thread over the shared memory and
/// calling the `wasi_thread_start` export. SVM lowers it to the native `thread.spawn` (§12) instead of
/// a `cap.call`: concurrency lives *in* the VM, not bolted onto the host (DESIGN §1a). Matches what
/// `wasmtime-wasi-threads` expects, so the same bytes run on both engines.
const WASI_THREAD_MODULE: &str = "wasi";
const WASI_THREAD_SPAWN_NAME: &str = "thread-spawn";
/// The export the host calls on a freshly spawned thread: `(func (param $tid i32) (param $start_arg
/// i32))`. The synthesized spawn shim adapts SVM's `(i64 sp, i64 arg)` thread-entry ABI to it.
const WASI_THREAD_START_EXPORT: &str = "wasi_thread_start";
/// `memory.grow` on an **unbounded** wasm memory may extend up to this many pages. The window must
/// hold the whole growable span (SVM masks accesses into the window rather than bounds-checking-and-
/// trapping — the documented confinement difference), so for unbounded memory this is a modest cap
/// (16 MiB) that keeps the eagerly-committed window small. A declared `maximum` is honored instead.
const DEFAULT_MAX_GROW_PAGES: u64 = 256;
/// Hard ceiling on the growable span regardless of a declared `maximum`, so a pathological `maximum`
/// can't blow up the committed window (256 MiB).
const MAX_GROW_PAGES: u64 = 4096;

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
        // §17/D58: wasm v128 maps directly to our fixed-128 vector type.
        W::V128 => Ok(ValType::V128),
        W::Ref(_) => unsup("reference type"),
    }
}

/// Transpile a core-wasm binary into a verifier-checkable [`Module`].
///
/// **Host-function imports (the host ABI).** A wasm `(import "<module>" "<name>" (func …))` binds to
/// an SVM capability by a naming convention: `module` is the decimal capability **`type_id`** and
/// `name` the decimal **`op`**. A wasm `call` to an import then lowers to `cap.call type_id op` on a
/// single capability **handle** the transpiler threads as the leading `i32` parameter of every
/// function (the data-SP trick). The embedder grants the matching capability and passes its handle as
/// the entry function's leading argument; the transpiler stays pure mechanism (it never interprets the
/// host semantics — `(type_id, op)` just select an interface/method). v1 threads **one** handle, so
/// every import must share one `type_id` (methods distinguished by `op`); a non-numeric name, a
/// table/memory/global import, or imports spanning multiple interfaces is a clean [`Error::Unsupported`]
/// (real WASI, whose imports are non-numeric, needs a dedicated shim). A no-import module is unchanged.
///
/// **Linear-memory growth (`memory.size` / `memory.grow`).** The linear memory sits at window offset 0
/// (wasm address `a` == window address `a`). When a module uses `memory.grow`, the window reserves the
/// memory's *full growable span* at the bottom — up to its declared `maximum`, or a modest default
/// ([`DEFAULT_MAX_GROW_PAGES`], bounded by [`MAX_GROW_PAGES`]) for unbounded memory — and places the
/// globals/table regions above it, so growth never collides with them. A runtime **size cell** (an
/// 8-byte window slot just above the linear memory, initialized to the initial page count) holds the
/// current size: `memory.size` loads it and `memory.grow` updates it branch-free (set to the new size
/// on success / unchanged on a past-cap failure, returning the old size or `-1`). Because SVM masks
/// accesses into the window rather than bounds-checking-and-trapping, a grown page is simply reachable;
/// the size cell only governs the `size`/`grow` *return values*. A module that never grows is
/// unchanged (no cell, the tight initial-sized window, `memory.size` a constant).
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
                                                         // Function imports, in import order: each binds to an SVM capability `(type_id, op)` by the naming
                                                         // convention (`module` = decimal type_id, `name` = decimal op) and lowers to a `cap.call`. The
                                                         // function index space puts imports first, so a wasm index `< imports.len()` is an import.
    let mut imports: Vec<(u32, u32, FuncType)> = Vec::new();
    // §12 wasm threads: the wasm function index of the `wasi:thread/spawn` import, if present. Its
    // `call` lowers to `thread.spawn` (not a cap.call); see the spawn-shim synthesis.
    let mut spawn_import: Option<u32> = None;
    // The `(start $f)` function index, if the module has a start section: it runs once at
    // instantiation, before any export. SVM has no instantiation hook (a run calls one entry over a
    // fresh window), so each export is wrapped to run `start` first — see the start-wrapper synthesis.
    let mut start: Option<u32> = None;

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
            Payload::ImportSection(reader) => {
                for imp in reader.into_imports() {
                    let imp = imp?;
                    let type_idx = match imp.ty {
                        wasmparser::TypeRef::Func(t) => t,
                        wasmparser::TypeRef::FuncExact(t) => t,
                        wasmparser::TypeRef::Table(_) => return unsup("imported table"),
                        // §12 wasm threads: an **imported** (shared) memory is the canonical
                        // wasi-threads shape — the host owns the one shared linear memory so it is
                        // shared across the per-thread instances. SVM treats it exactly like a defined
                        // memory (the window's linear region at offset 0); only the *declaration* site
                        // differs. (A non-memory import follows the function/host-ABI path below.)
                        wasmparser::TypeRef::Memory(mt) => {
                            if mem.replace(mt).is_some() {
                                return unsup("multi-memory");
                            }
                            continue;
                        }
                        wasmparser::TypeRef::Global(_) => return unsup("imported global"),
                        wasmparser::TypeRef::Tag(_) => return unsup("imported tag"),
                    };
                    let (p, r) = &types[type_idx as usize];
                    let sig = FuncType {
                        params: p.clone(),
                        results: r.clone(),
                    };
                    // §12 wasm threads (wasi-threads): the `wasi:thread/spawn` import is special — it
                    // lowers to the native `thread.spawn` (a new vCPU over the shared window), **not** a
                    // `cap.call`. It occupies a function-index slot like any import (so a placeholder
                    // keeps later indices aligned), but it is excluded from the capability-handle logic
                    // and the one-interface check below. See the spawn-shim synthesis after lowering.
                    if imp.module == WASI_THREAD_MODULE && imp.name == WASI_THREAD_SPAWN_NAME {
                        if spawn_import.is_some() {
                            return unsup("multiple wasi:thread/spawn imports");
                        }
                        spawn_import = Some(imports.len() as u32);
                        imports.push((u32::MAX, 0, sig)); // placeholder (type_id unused; never cap-called)
                        continue;
                    }
                    // The binding convention: `module` is the decimal capability type_id and `name`
                    // the decimal op. This keeps the transpiler pure mechanism — it never interprets
                    // the host semantics; the embedder grants the matching capability and the
                    // (type_id, op) just select an interface/method. A non-numeric name is a clear
                    // error rather than a silent mis-binding.
                    let type_id: u32 = imp.module.parse().map_err(|_| {
                        Error::Unsupported(format!(
                            "import module {:?} is not a decimal capability type_id (the host-ABI \
                             convention: module = type_id, name = op)",
                            imp.module
                        ))
                    })?;
                    let op: u32 = imp.name.parse().map_err(|_| {
                        Error::Unsupported(format!(
                            "import name {:?} is not a decimal capability op (the host-ABI \
                             convention: module = type_id, name = op)",
                            imp.name
                        ))
                    })?;
                    imports.push((type_id, op, sig));
                }
                // v1 threads a single capability handle, so every **capability** import must share one
                // type_id (one interface, methods distinguished by op); the `wasi:thread/spawn` import
                // (placeholder `type_id == u32::MAX`) is excluded — it is the native spawn, not a
                // cap.call. Distinct interfaces would need one handle each — a later slice.
                let mut cap_type_id: Option<u32> = None;
                for (t, _, _) in imports.iter().filter(|(t, _, _)| *t != u32::MAX) {
                    match cap_type_id {
                        None => cap_type_id = Some(*t),
                        Some(first) if *t != first => {
                            return unsup(
                                "imports spanning multiple capability interfaces (one handle is \
                                 threaded in v1 — give every import the same module/type_id)",
                            );
                        }
                        _ => {}
                    }
                }
            }
            Payload::FunctionSection(reader) => {
                for idx in reader {
                    func_type_idx.push(idx?);
                }
            }
            Payload::MemorySection(reader) => {
                for mt in reader {
                    let mt = mt?;
                    // wasm `shared` memory (the threads proposal) is accepted: SVM's window is already
                    // shared across vCPUs, so the layout is identical — only the `*.atomic.*` ops and
                    // the spawn convention differ (§12; the wasm→IR atomic mapping below). wasm requires
                    // a declared `maximum` on shared memory, which the grow path already honours.
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
                                    // Store IR function indices (imports first in the wasm index
                                    // space; the table holds defined functions). A funcref to an
                                    // import isn't representable as an IR funcref — reject it.
                                    let mut fs: Vec<u32> = Vec::new();
                                    for f in fns {
                                        let f = f?;
                                        match f.checked_sub(imports.len() as u32) {
                                            Some(ir) => fs.push(ir),
                                            None => {
                                                return unsup("funcref to an imported function")
                                            }
                                        }
                                    }
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
                        // wasm function indices put imports first; the IR module holds only defined
                        // functions (index `wasm_idx - n_imp`). A re-exported import has no IR function,
                        // so skip it (it isn't a runnable entry).
                        if let Some(ir_idx) = e.index.checked_sub(imports.len() as u32) {
                            exports.push((e.name.to_string(), ir_idx));
                        }
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
            Payload::StartSection { func, .. } => {
                // At most one start section per spec; record it (run via the per-export wrappers below).
                start = Some(func);
            }
            _ => {} // version header, custom sections, datacount, ends, etc. — ignore
        }
    }

    if bodies.len() != func_type_idx.len() {
        return Err(Error::Parse("function/code section length mismatch".into()));
    }

    let mem64 = mem.as_ref().map(|m| m.memory64).unwrap_or(false);

    // Does any function use `memory.grow`? Only then must the window reserve room for the linear memory
    // to expand (and carry a runtime size cell) — so a non-growing module (every existing kernel)
    // transpiles to byte-identical IR and the same window. (`memory.size` without growth is a constant.)
    let mut uses_grow = false;
    'scan: for body in &bodies {
        for op in body.get_operators_reader()? {
            if matches!(op?, Operator::MemoryGrow { .. }) {
                uses_grow = true;
                break 'scan;
            }
        }
    }

    // Linear-memory layout. The linear memory sits at window offset 0 (so wasm address `a` is window
    // address `a`); the page count it may occupy is its initial size, or — when `memory.grow` is used —
    // up to its declared `maximum` (a default cap for unbounded memory, bounded by `MAX_GROW_PAGES`).
    // The window must hold that whole span because SVM masks accesses into the window rather than
    // bounds-checking-and-trapping (so a grown page is reachable, an over-grow access just masks).
    let initial_pages = mem.as_ref().map(|m| m.initial).unwrap_or(0);
    let max_pages = if uses_grow {
        mem.as_ref()
            .and_then(|m| m.maximum)
            .unwrap_or(DEFAULT_MAX_GROW_PAGES)
            .clamp(initial_pages.max(1), MAX_GROW_PAGES.max(initial_pages))
    } else {
        initial_pages
    };
    let mem_span_pages = if mem.is_some() { max_pages.max(1) } else { 0 };
    let mem_bytes = mem_span_pages.saturating_mul(WASM_PAGE);

    // Runtime current-size cell (pages), an 8-byte slot just above the linear-memory span — present
    // only when `grow` is used; `memory.size`/`grow` load/store it, initialized to the initial page
    // count via a `data` segment. (Without growth there is no cell and `memory.size` is a constant.)
    let size_cell_off = mem_bytes;
    let after_mem = if uses_grow {
        data.push(svm_ir::Data {
            offset: size_cell_off,
            readonly: false,
            bytes: initial_pages.to_le_bytes().to_vec(),
        });
        size_cell_off + 8
    } else {
        mem_bytes
    };

    // wasm globals are module-level mutables our IR has no notion of, so we give them a reserved region
    // **above** the linear memory (and the size cell) and lower `global.get`/`set` to load/store there
    // (8-byte slots, the standard "globals in memory" lowering). Initializers become `data` segments. A
    // valid guest's linear-memory accesses stay in `[0, mem_bytes)` and so never reach the globals; only
    // an OOB access would (which wasm traps and we don't — the documented confinement difference).
    let globals_base = after_mem.div_ceil(8) * 8; // 8-byte aligned, just past the linear memory + cell
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
    // §12 wasm threads: a 4-byte reserved unique-tid counter just past the function table (a fresh
    // window reads 0, so the first spawned tid is 1). Only consumed when `wasi:thread/spawn` is used.
    let tid_slot = table_end.div_ceil(4) * 4;

    // §12 wasm threads validation + the spawn shim's IR index (it is appended right after the defined
    // functions, so its index is `bodies.len()`). The shim adapts SVM's thread-entry ABI to the
    // `wasi_thread_start` export.
    let spawn_shim = bodies.len() as u32;
    if spawn_import.is_some() {
        if imports.iter().any(|(t, _, _)| *t != u32::MAX) {
            return unsup(
                "wasi:thread/spawn alongside capability imports — needs the per-thread handle stash \
                 (threads-only modules are supported in this slice)",
            );
        }
        // The host calls `wasi_thread_start(tid, start_arg)` on each spawned thread; require the export
        // and that it is `(i32, i32) -> ()` (with no capability handle, since threads-only).
        let wts = exports
            .iter()
            .find(|(n, _)| n == WASI_THREAD_START_EXPORT)
            .map(|(_, i)| *i)
            .ok_or_else(|| {
                Error::Unsupported(format!(
                    "wasi:thread/spawn import without a `{WASI_THREAD_START_EXPORT}` export"
                ))
            })?;
        let (p, r) = &types[func_type_idx[wts as usize] as usize];
        if p.as_slice() != [ValType::I32, ValType::I32] || !r.is_empty() {
            return unsup("wasi_thread_start must have type (i32 tid, i32 start_arg) -> ()");
        }
    }
    let threads = ThreadCfg {
        spawn_import,
        spawn_shim,
        tid_slot,
    };

    let func_sigs: Vec<(Vec<ValType>, Vec<ValType>)> = func_type_idx
        .iter()
        .map(|&ti| types[ti as usize].clone())
        .collect();
    let mut funcs = Vec::with_capacity(bodies.len() + spawn_import.is_some() as usize);
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
            &imports,
            MemGrow {
                uses_grow,
                size_cell_off,
                max_pages,
                initial_pages,
            },
            threads,
        )?);
    }
    // Append the spawn shim (its `thread.spawn` target) for a threaded module.
    if spawn_import.is_some() {
        let wts = exports
            .iter()
            .find(|(n, _)| n == WASI_THREAD_START_EXPORT)
            .map(|(_, i)| *i)
            .expect("validated above");
        funcs.push(build_spawn_shim(wts));
    }

    // `(start $f)`: run the start function once before any export. SVM has no instantiation hook (a
    // run calls one entry over a fresh window, with data/element segments already materialized), so
    // each **exported** function is remapped to a synthesized wrapper that calls `start` then the real
    // export. Internal `call`s (by function index) still reach the real export directly, so `start`
    // runs exactly once, before the chosen entry — and a non-`(start)` module is unchanged.
    if let Some(start_wasm) = start {
        let n_imp = imports.len() as u32;
        let start_ir = start_wasm.checked_sub(n_imp).ok_or_else(|| {
            Error::Unsupported("start function is an import (must be a defined function)".into())
        })?;
        let (sp, sr) = func_sigs.get(start_ir as usize).ok_or_else(|| {
            Error::Parse(format!("start function index {start_wasm} out of range"))
        })?;
        if !sp.is_empty() || !sr.is_empty() {
            return unsup("start function must have type () -> ()");
        }
        let has_handle = imports.iter().any(|(t, _, _)| *t != u32::MAX);
        for (_, ir_idx) in exports.iter_mut() {
            if *ir_idx == start_ir {
                continue; // exporting the start function itself: don't double-run it
            }
            let params = funcs[*ir_idx as usize].params.clone();
            let results = funcs[*ir_idx as usize].results.clone();
            let wrap = build_start_wrapper(start_ir, *ir_idx, params, results, has_handle);
            *ir_idx = funcs.len() as u32;
            funcs.push(wrap);
        }
    }

    // Our window is a power-of-two byte range (masking confines to it); size it to hold the linear
    // memory (its full growable span) **and** the size cell + globals + function-table regions. (wasm
    // bounds-checks-and-traps on out-of-range access while we mask-confine to the ≥ power-of-two
    // window — identical for in-bounds accesses.) Globals/table-only modules still need a window.
    let needed = table_end
        .max(globals_end)
        .max(after_mem)
        .max(mem_bytes)
        .max(if spawn_import.is_some() {
            tid_slot + 4
        } else {
            0
        });
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
            // svm-wasm lowers capability imports to `cap.call` inline (numeric type_id/op
            // convention); it does not use §7 named imports.
            imports: Vec::new(),
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

/// `memory.size`/`memory.grow` lowering parameters. When `uses_grow` the linear memory may expand and
/// a runtime **size cell** (an 8-byte window slot at `size_cell_off`, holding the current page count)
/// backs both ops; otherwise `memory.size` is the constant `initial_pages` and `grow` never appears.
#[derive(Clone, Copy)]
struct MemGrow {
    uses_grow: bool,
    size_cell_off: u64,
    max_pages: u64,
    initial_pages: u64,
}

/// §12 wasm threads config (wasi-threads). Absent (`spawn_import == None`) for non-threaded modules,
/// in which case nothing here is referenced and the lowering is byte-identical to before.
#[derive(Clone, Copy)]
struct ThreadCfg {
    /// The wasm function index of the `wasi:thread/spawn` import — its `call` lowers to `thread.spawn`.
    spawn_import: Option<u32>,
    /// IR index of the synthesized spawn shim (`thread.spawn`'s target — a `(i64 sp, i64 arg) -> i64`
    /// adapter that unpacks `(tid, start_arg)` and calls `wasi_thread_start`).
    spawn_shim: u32,
    /// Window byte offset of the unique-TID counter: an i32 atomically `add`-incremented per spawn so
    /// each thread gets a unique positive tid (avoiding the spawn-handle circularity). Reads 0 in a
    /// fresh window, so the first tid is 1.
    tid_slot: u64,
}

struct Lower<'a> {
    blocks: Vec<BlockB>,
    cur: usize,
    /// Current SSA value of each local (param then declared), in `cur`'s value space.
    locals: Vec<ValIdx>,
    local_types: Vec<ValType>,
    /// Operand stack: (value, type).
    stack: Vec<(ValIdx, ValType)>,
    /// Constant SSA values in the **current block** (set when an `i32.const`/`i64.const` is emitted,
    /// cleared on block entry). Used to recognise the compile-time `len` of a `memory.copy`/`fill`
    /// (clang's bulk ops carry a constant size) so it can be unrolled into chunked load/stores.
    consts: std::collections::HashMap<ValIdx, i64>,
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
    /// `memory.size`/`memory.grow` lowering config (size-cell offset, page caps).
    mg: MemGrow,
    /// The single threaded **capability handle** (`i32`, the forgeable index a `cap.call` takes),
    /// present iff the module has function imports.
    /// Like the data-SP in the chibicc frontend, it is block param 0 of every block and is prepended
    /// to every branch's args, so every function can reach it and a wasm `call` to an import lowers to
    /// a `cap.call` on it. The embedder grants one capability and passes its handle as the entry's
    /// leading argument.
    handle: Option<ValIdx>,
    /// Per function-import (by import index): the `(type_id, op, signature)` its `call` lowers to as a
    /// `cap.call`. Empty when the module has no imports.
    imports: &'a [(u32, u32, FuncType)],
    /// Number of imported functions: a wasm function index `< n_imp` is an import (→ `cap.call`), else
    /// a defined function at IR index `idx - n_imp`.
    n_imp: usize,
    /// §12 wasm threads config — the `wasi:thread/spawn` lowering (the spawn import index, the shim,
    /// the unique-tid slot). `spawn_import` is `None` for non-threaded modules.
    threads: ThreadCfg,
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

    /// Width of the always-threaded prefix every block carries: the capability handle (if any),
    /// then all locals. The surviving operand stack follows.
    fn prefix_len(&self) -> usize {
        self.handle.is_some() as usize + self.local_types.len()
    }
    /// The prefix value list (handle ++ locals) every branch threads, in `cur`'s value space.
    fn prefix_vals(&self) -> Vec<ValIdx> {
        let mut v = Vec::with_capacity(self.prefix_len());
        if let Some(h) = self.handle {
            v.push(h);
        }
        v.extend_from_slice(&self.locals);
        v
    }
    /// The prefix types (the i32 capability handle ++ local types).
    fn prefix_types(&self) -> Vec<ValType> {
        let mut t = Vec::with_capacity(self.prefix_len());
        if self.handle.is_some() {
            t.push(ValType::I32);
        }
        t.extend_from_slice(&self.local_types);
        t
    }

    /// The block-parameter signature for a target carrying `carried` stack types: every IR block
    /// threads the handle + all locals first, then the surviving stack.
    fn sig(&self, carried: &[ValType]) -> Vec<ValType> {
        let mut s = self.prefix_types();
        s.extend_from_slice(carried);
        s
    }

    /// The arguments for a branch to a frame: the threaded prefix (handle + locals), then the
    /// preserved base of the target and the top `arity` carried values (the middle is unwound away).
    fn branch_args(&self, base: usize, arity: usize) -> Vec<ValIdx> {
        let mut a = self.prefix_vals();
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

    /// Make `blk` current and rebind the handle + locals + stack to its parameters. The prefix (handle
    /// then locals) occupies params `0..prefix_len()`; `stack_types` is the carried stack layout, whose
    /// values become the params after it.
    fn enter(&mut self, blk: usize, stack_types: &[ValType]) {
        self.cur = blk;
        let p = self.handle.is_some() as ValIdx;
        if self.handle.is_some() {
            self.handle = Some(0);
        }
        let nl = self.local_types.len() as ValIdx;
        self.locals = (p..p + nl).collect();
        self.stack = stack_types
            .iter()
            .enumerate()
            .map(|(i, t)| (p + nl + i as ValIdx, *t))
            .collect();
        self.consts.clear(); // SSA values are block-local; constants don't carry across blocks
        self.reachable = true;
    }

    /// The compile-time value of `idx` if it was produced by an `i32.const`/`i64.const` in the
    /// current block (used to recognise a `memory.copy`/`fill`'s constant length).
    fn const_of(&self, idx: ValIdx) -> Option<i64> {
        self.consts.get(&idx).copied()
    }

    // ---- synthesized-block helpers (a transpiler-emitted runtime loop, e.g. a dynamic bulk op) ----
    // A synthesized block's params are `prefix (handle + locals) ++ below ++ extra`, where `below` is
    // the operand stack carried through the loop and `extra` are loop-private values (addresses, the
    // length, the counter). This mirrors the normal block layout (`prefix ++ stack`) with the extra
    // loop-private values appended after the stack.

    /// Param types for a synthesized block: prefix ++ `below` ++ `extra`.
    fn synth_sig(&self, below: &[ValType], extra: &[ValType]) -> Vec<ValType> {
        let mut s = self.prefix_types();
        s.extend_from_slice(below);
        s.extend_from_slice(extra);
        s
    }
    /// Branch args to a synthesized block: prefix (of the *current* block) ++ `below_vals` ++ `extra`.
    fn synth_args(&self, below_vals: &[ValIdx], extra: &[ValIdx]) -> Vec<ValIdx> {
        let mut a = self.prefix_vals();
        a.extend_from_slice(below_vals);
        a.extend_from_slice(extra);
        a
    }
    /// Enter a synthesized block: rebind handle/locals to the prefix and the operand stack to `below`,
    /// and return the SSA values of the `n_extra` trailing loop-private params (in order).
    fn enter_synth(&mut self, blk: usize, below: &[ValType], n_extra: usize) -> Vec<ValIdx> {
        self.cur = blk;
        let p = self.handle.is_some() as ValIdx;
        if self.handle.is_some() {
            self.handle = Some(0);
        }
        let nl = self.local_types.len() as ValIdx;
        self.locals = (p..p + nl).collect();
        let stack_start = p + nl;
        self.stack = below
            .iter()
            .enumerate()
            .map(|(i, t)| (stack_start + i as ValIdx, *t))
            .collect();
        self.consts.clear();
        self.reachable = true;
        let extra_start = stack_start + below.len() as ValIdx;
        (extra_start..extra_start + n_extra as ValIdx).collect()
    }
    /// The current operand-stack values (the `below` carried by a synthesized loop).
    fn stack_vals(&self) -> Vec<ValIdx> {
        self.stack.iter().map(|(v, _)| *v).collect()
    }

    fn set_term(&mut self, t: Terminator) {
        self.blocks[self.cur].term = Some(t);
        self.reachable = false;
    }

    /// The carried stack types a merge expects, read back from its params (the handle + locals
    /// prefix stripped).
    fn merge_stack_types(&self, m: usize) -> Vec<ValType> {
        self.blocks[m].params[self.prefix_len()..].to_vec()
    }
}

/// Synthesize the §12 spawn shim: a `(i64 sp, i64 arg) -> (i64)` IR function (the `thread.spawn`
/// entry ABI) that unpacks `(tid, start_arg)` from its packed `arg` and calls the module's
/// `wasi_thread_start(tid, start_arg)` export (IR index `wts`), then returns 0. The data-SP `sp` is
/// unused (svm-wasm keeps the C stack in linear memory). `wasi_thread_start` is `(i32, i32) -> ()`
/// here — threads-only modules thread no capability handle, so it carries no leading handle param.
fn build_spawn_shim(wts: u32) -> Func {
    // values: v0=sp v1=arg | v2..  (a 0-result `call` appends no value)
    let insts = vec![
        Inst::ConstI64(32), // v2
        Inst::IntBin {
            ty: IntTy::I64,
            op: BinOp::ShrU,
            a: 1, // arg
            b: 2, // 32
        }, // v3 = arg >> 32
        Inst::Convert {
            op: ConvOp::WrapI64,
            a: 3,
        }, // v4 = tid (i32)
        Inst::Convert {
            op: ConvOp::WrapI64,
            a: 1,
        }, // v5 = start_arg (i32, low 32 of arg)
        Inst::Call {
            func: wts,
            args: vec![4, 5],
        }, // no result
        Inst::ConstI64(0),  // v6
    ];
    Func {
        params: vec![ValType::I64, ValType::I64],
        results: vec![ValType::I64],
        blocks: vec![Block {
            params: vec![ValType::I64, ValType::I64],
            insts,
            term: Terminator::Return(vec![6]),
        }],
    }
}

/// Synthesize a start wrapper for an exported function (`(start $f)` support): a function with the
/// **same IR signature** as `target` that first calls `start` (which is `() -> ()` in wasm, so
/// `(handle?) -> ()` here) and then `target` with all params, returning its results. The embedder
/// runs this in place of the bare export, so the start function runs once before the entry; internal
/// calls reach `target` directly and don't re-run it.
fn build_start_wrapper(
    start_ir: u32,
    target: u32,
    params: Vec<ValType>,
    results: Vec<ValType>,
    has_handle: bool,
) -> Func {
    let nparams = params.len() as ValIdx;
    let insts = vec![
        // call start() — thread the handle if the module has capability imports (start produces no
        // value, so it doesn't advance the value counter).
        Inst::Call {
            func: start_ir,
            args: if has_handle { vec![0] } else { vec![] },
        },
        // call the real export with every param in order (handle? ++ wasm params = values 0..nparams);
        // its results land at values nparams.. .
        Inst::Call {
            func: target,
            args: (0..nparams).collect(),
        },
    ];
    let ret: Vec<ValIdx> = (nparams..nparams + results.len() as ValIdx).collect();
    Func {
        params: params.clone(),
        results,
        blocks: vec![Block {
            params,
            insts,
            term: Terminator::Return(ret),
        }],
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
    imports: &[(u32, u32, FuncType)],
    mg: MemGrow,
    threads: ThreadCfg,
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

    // When the module has **capability** imports we thread one capability handle (i32) as the leading
    // param of every function/block (the data-SP trick): the IR signature is `(i32 handle,
    // wasm-params...) -> results` and param 0 is the handle. A module whose only import is
    // `wasi:thread/spawn` (placeholder `type_id == u32::MAX`, lowered to the native `thread.spawn`, not
    // a cap.call) needs no handle — so it is byte-identical to a no-import module. A no-import module
    // is likewise unchanged.
    let has_handle = imports.iter().any(|(t, _, _)| *t != u32::MAX);
    let n_imp = imports.len();
    let mut entry_params: Vec<ValType> = Vec::with_capacity(has_handle as usize + params.len());
    if has_handle {
        entry_params.push(ValType::I32);
    }
    entry_params.extend_from_slice(params);
    let nparams = params.len() as ValIdx;
    let base = has_handle as ValIdx; // value index of wasm param 0 (after the handle, if present)

    let entry = BlockB {
        params: entry_params.clone(),
        insts: Vec::new(),
        next_val: entry_params.len() as ValIdx,
        term: None,
    };
    let mut lo = Lower {
        blocks: vec![entry],
        cur: 0,
        locals: (base..base + nparams).collect(),
        local_types: local_types.clone(),
        stack: Vec::new(),
        consts: std::collections::HashMap::new(),
        reachable: true,
        control: Vec::new(),
        types,
        func_sigs,
        global_types,
        globals_base,
        table_base,
        mem64,
        mg,
        handle: has_handle.then_some(0),
        imports,
        n_imp,
        threads,
    };
    // Initialize declared locals to zero (params already bound to block params), extending `locals`.
    for t in &local_types[params.len()..] {
        let v = match t {
            ValType::I32 => lo.emit(Inst::ConstI32(0)),
            ValType::I64 => lo.emit(Inst::ConstI64(0)),
            ValType::F32 => lo.emit(Inst::ConstF32(0)),
            ValType::F64 => lo.emit(Inst::ConstF64(0)),
            ValType::V128 => lo.emit(Inst::ConstV128([0; 16])),
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
        params: entry_params,
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

// ---- §17 SIMD (D58): wasm v128 → IR v128 ----
fn v_splat(lo: &mut Lower, shape: VShape) -> Result<(), Error> {
    let (a, _) = lo.pop()?;
    let v = lo.emit(Inst::Splat { shape, a });
    lo.push(v, ValType::V128);
    Ok(())
}
fn v_extract(lo: &mut Lower, shape: VShape, lane: u8, signed: bool) -> Result<(), Error> {
    let (a, _) = lo.pop()?;
    let v = lo.emit(Inst::ExtractLane {
        shape,
        lane,
        signed,
        a,
    });
    lo.push(v, shape.lane_val());
    Ok(())
}
fn v_replace(lo: &mut Lower, shape: VShape, lane: u8) -> Result<(), Error> {
    let (b, _) = lo.pop()?;
    let (a, _) = lo.pop()?;
    let v = lo.emit(Inst::ReplaceLane { shape, lane, a, b });
    lo.push(v, ValType::V128);
    Ok(())
}
fn v_intbin(lo: &mut Lower, shape: VShape, op: VIntBinOp) -> Result<(), Error> {
    let (b, _) = lo.pop()?;
    let (a, _) = lo.pop()?;
    let v = lo.emit(Inst::VIntBin { shape, op, a, b });
    lo.push(v, ValType::V128);
    Ok(())
}
fn v_fbin(lo: &mut Lower, shape: VShape, op: VFloatBinOp) -> Result<(), Error> {
    let (b, _) = lo.pop()?;
    let (a, _) = lo.pop()?;
    let v = lo.emit(Inst::VFloatBin { shape, op, a, b });
    lo.push(v, ValType::V128);
    Ok(())
}
fn v_fun(lo: &mut Lower, shape: VShape, op: VFloatUnOp) -> Result<(), Error> {
    let (a, _) = lo.pop()?;
    let v = lo.emit(Inst::VFloatUn { shape, op, a });
    lo.push(v, ValType::V128);
    Ok(())
}
fn v_bitbin(lo: &mut Lower, op: VBitBinOp) -> Result<(), Error> {
    let (b, _) = lo.pop()?;
    let (a, _) = lo.pop()?;
    let v = lo.emit(Inst::VBitBin { op, a, b });
    lo.push(v, ValType::V128);
    Ok(())
}
fn v128_load(lo: &mut Lower, m: MemArg) -> Result<(), Error> {
    let addr = pop_addr(lo)?;
    let v = lo.emit(Inst::V128Load {
        addr,
        offset: m.offset,
        align: m.align,
    });
    lo.push(v, ValType::V128);
    Ok(())
}
fn v128_store(lo: &mut Lower, m: MemArg) -> Result<(), Error> {
    let (value, _) = lo.pop()?;
    let addr = pop_addr(lo)?;
    lo.emit_void(Inst::V128Store {
        addr,
        value,
        offset: m.offset,
        align: m.align,
    });
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
/// `v128` globals are out of MVP scope (a v128 needs 16 bytes, not the 8-byte slot); the
/// transpiler never lowers one, so the `V128` arms are unreachable placeholders that keep
/// these helpers total.
fn load_op(ty: ValType) -> LoadOp {
    match ty {
        ValType::I32 => LoadOp::I32,
        ValType::I64 | ValType::V128 => LoadOp::I64,
        ValType::F32 => LoadOp::F32,
        ValType::F64 => LoadOp::F64,
    }
}
fn store_op(ty: ValType) -> StoreOp {
    match ty {
        ValType::I32 => StoreOp::I32,
        ValType::I64 | ValType::V128 => StoreOp::I64,
        ValType::F32 => StoreOp::F32,
        ValType::F64 => StoreOp::F64,
    }
}

/// `call funcidx`: pop the callee's params (the last is on top), call it, push its results.
///
/// A wasm function index `< n_imp` is an **import**: lower to a `cap.call` on the threaded capability
/// handle (the import's `(type_id, op, sig)` from the convention). Otherwise it's a defined function
/// at IR index `func - n_imp`; prepend the handle (when threaded) to its args so the callee's leading
/// handle param is supplied.
fn call_op(lo: &mut Lower, func: u32) -> Result<(), Error> {
    // §12 wasm threads: a `call` to the `wasi:thread/spawn` import → the native `thread.spawn`.
    if lo.threads.spawn_import == Some(func) {
        return spawn_op(lo);
    }
    if (func as usize) < lo.n_imp {
        let (type_id, op, sig) = lo.imports[func as usize].clone();
        let mut args = Vec::with_capacity(sig.params.len());
        for _ in 0..sig.params.len() {
            args.push(lo.pop()?.0);
        }
        args.reverse(); // stack top is the last argument
        let handle = lo.handle.expect("import call requires a threaded handle");
        let results = sig.results.clone();
        let res = lo.emit_call(
            Inst::CapCall {
                type_id,
                op,
                sig,
                handle,
                args,
            },
            results.len(),
        );
        for (v, t) in res.into_iter().zip(results.iter()) {
            lo.push(v, *t);
        }
        return Ok(());
    }
    let ir_idx = func - lo.n_imp as u32;
    let (params, results) = lo
        .func_sigs
        .get(ir_idx as usize)
        .ok_or_else(|| Error::Parse(format!("call to unknown function {func}")))?
        .clone();
    let mut args = Vec::with_capacity(lo.handle.is_some() as usize + params.len());
    for _ in 0..params.len() {
        args.push(lo.pop()?.0);
    }
    args.reverse(); // stack top is the last argument
    if let Some(h) = lo.handle {
        args.insert(0, h); // the callee's leading handle param
    }
    let res = lo.emit_call(Inst::Call { func: ir_idx, args }, results.len());
    for (v, t) in res.into_iter().zip(results.iter()) {
        lo.push(v, *t);
    }
    Ok(())
}

/// `call $wasi_thread_spawn` (stack: `[start_arg: i32]`) → the native `thread.spawn` (§12): a new
/// vCPU over the shared window running the spawn shim. Returns the new thread's **tid** (a unique
/// positive i32), matching the wasi-threads ABI. The tid is allocated by an atomic increment of the
/// reserved `tid_slot` counter (avoiding the spawn-handle circularity), then packed with `start_arg`
/// into the single i64 `thread.spawn` carries: `arg = (tid << 32) | start_arg`. The shim
/// ([`build_spawn_shim`]) unpacks it and calls `wasi_thread_start(tid, start_arg)`.
fn spawn_op(lo: &mut Lower) -> Result<(), Error> {
    let (start_arg, _) = lo.pop()?; // i32
    let shim = lo.threads.spawn_shim;
    // tid = atomic_add(tid_slot, 1) + 1  ⇒ first spawn gets tid 1 (a fresh window reads 0).
    let slot = lo.emit(Inst::ConstI64(lo.threads.tid_slot as i64));
    let one = lo.emit(Inst::ConstI32(1));
    let old = lo.emit(Inst::AtomicRmw {
        ty: IntTy::I32,
        op: AtomicRmwOp::Add,
        addr: slot,
        value: one,
        offset: 0,
        order: Ordering::SeqCst,
    });
    let one2 = lo.emit(Inst::ConstI32(1));
    let tid = lo.emit(Inst::IntBin {
        ty: IntTy::I32,
        op: BinOp::Add,
        a: old,
        b: one2,
    });
    // arg = (zext(tid) << 32) | zext(start_arg)
    let tid64 = lo.emit(Inst::Convert {
        op: ConvOp::ExtendI32U,
        a: tid,
    });
    let shift = lo.emit(Inst::ConstI64(32));
    let hi = lo.emit(Inst::IntBin {
        ty: IntTy::I64,
        op: BinOp::Shl,
        a: tid64,
        b: shift,
    });
    let lo64 = lo.emit(Inst::Convert {
        op: ConvOp::ExtendI32U,
        a: start_arg,
    });
    let packed = lo.emit(Inst::IntBin {
        ty: IntTy::I64,
        op: BinOp::Or,
        a: hi,
        b: lo64,
    });
    // The shim ignores its data-SP (svm-wasm keeps the C stack in linear memory via `__stack_pointer`),
    // so any constant works.
    let sp0 = lo.emit(Inst::ConstI64(0));
    // `thread.spawn` yields an i32 join handle; wasi-libc manages join itself (futex on the thread
    // state), so we discard it and return the tid as the wasi:thread/spawn result.
    let _join = lo.emit(Inst::ThreadSpawn {
        func: shim,
        sp: sp0,
        arg: packed,
    });
    lo.push(tid, ValType::I32);
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
    let mut args = Vec::with_capacity(lo.handle.is_some() as usize + params.len());
    for _ in 0..params.len() {
        args.push(lo.pop()?.0);
    }
    args.reverse();
    // Every defined function carries a leading handle param when the module has imports, so the
    // indirect-call signature (used for the §3c runtime type-id check) and args must include it too.
    let mut ty_params = params.clone();
    if let Some(h) = lo.handle {
        args.insert(0, h);
        ty_params.insert(0, ValType::I32);
    }
    let ty = FuncType {
        params: ty_params,
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

/// `return_call $f` (the tail-call proposal): a **block-terminating** direct call — the callee
/// replaces the current frame and its results become this function's results. A defined `$f` lowers
/// to the IR `Terminator::ReturnCall` (a true tail call, no stack growth). A capability import can't
/// be a terminator, so a tail call to one degrades to `cap.call` + `return` (correct, not
/// tail-optimized); a tail call to `wasi:thread/spawn` is nonsensical and rejected.
fn return_call_op(lo: &mut Lower, func: u32, fn_results: &[ValType]) -> Result<(), Error> {
    if lo.threads.spawn_import == Some(func) {
        return unsup("return_call to wasi:thread/spawn");
    }
    if (func as usize) < lo.n_imp {
        // Tail call to a capability import: do the cap.call, then return its results.
        call_op(lo, func)?;
        let n = fn_results.len();
        let args: Vec<ValIdx> = lo.stack[lo.stack.len() - n..]
            .iter()
            .map(|(v, _)| *v)
            .collect();
        lo.set_term(Terminator::Return(args));
        return Ok(());
    }
    let ir_idx = func - lo.n_imp as u32;
    let (params, _) = lo
        .func_sigs
        .get(ir_idx as usize)
        .ok_or_else(|| Error::Parse(format!("return_call to unknown function {func}")))?
        .clone();
    let mut args = Vec::with_capacity(lo.handle.is_some() as usize + params.len());
    for _ in 0..params.len() {
        args.push(lo.pop()?.0);
    }
    args.reverse();
    if let Some(h) = lo.handle {
        args.insert(0, h); // the callee's leading handle param
    }
    lo.set_term(Terminator::ReturnCall { func: ir_idx, args });
    Ok(())
}

/// `return_call_indirect (type $t)`: the indirect tail call — like [`call_indirect_op`] but emits the
/// `Terminator::ReturnCallIndirect` (block-terminating, §3c masked + type-checked dispatch).
fn return_call_indirect_op(lo: &mut Lower, type_index: u32, table_index: u32) -> Result<(), Error> {
    if table_index != 0 {
        return unsup("return_call_indirect on a non-zero table");
    }
    let (params, results) = lo.types[type_index as usize].clone();
    // table[index] → function index, at window byte `table_base + index*4` (same as call_indirect).
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
    let mut args = Vec::with_capacity(lo.handle.is_some() as usize + params.len());
    for _ in 0..params.len() {
        args.push(lo.pop()?.0);
    }
    args.reverse();
    // The handle is a leading param of every defined function, so it rides both the args and the
    // §3c type-check signature (matching the targets that carry it — same as `call_indirect`).
    let mut ty_params = params;
    if let Some(h) = lo.handle {
        args.insert(0, h);
        ty_params.insert(0, ValType::I32);
    }
    let ty = FuncType {
        params: ty_params,
        results,
    };
    lo.set_term(Terminator::ReturnCallIndirect {
        ty,
        idx: funcref,
        args,
    });
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

// ---- §12 wasm threads: the full-width `*.atomic.*` ops → IR atomics ----
//
// The wasm threads atomics map 1:1 onto SVM's IR atomic surface (same widths, same
// trap-on-misalignment, all seq-cst). Only the **full-width** (i32/i64) ops are lowered here; the
// **narrow** forms (`*.atomic.load8_u`/`rmw8`/`rmw16`/…) have no IR analogue (SVM atomics are
// 32/64-bit only) and fall through to the `worker_op` catch-all as a clean `Unsupported`. The IR
// atomic load/store/rmw/cmpxchg carry an `offset` field (folded by the runtime, like a plain
// load/store); `wait`/`notify` do not, so [`pop_atomic_addr`] folds it into the address.

/// The IR value type for an atomic's integer width.
fn int_vt(ty: IntTy) -> ValType {
    match ty {
        IntTy::I32 => ValType::I32,
        IntTy::I64 => ValType::I64,
    }
}

/// Pop the wasm address and fold the memarg `offset` into it (for `wait`/`notify`, whose IR ops
/// have no `offset` field — unlike load/store/rmw/cmpxchg, which carry it).
fn pop_atomic_addr(lo: &mut Lower, offset: u64) -> Result<ValIdx, Error> {
    let addr = pop_addr(lo)?;
    if offset == 0 {
        return Ok(addr);
    }
    let off = lo.emit(Inst::ConstI64(offset as i64));
    Ok(lo.emit(Inst::IntBin {
        ty: IntTy::I64,
        op: BinOp::Add,
        a: addr,
        b: off,
    }))
}

fn atomic_load(lo: &mut Lower, ty: IntTy, m: MemArg) -> Result<(), Error> {
    let addr = pop_addr(lo)?;
    let v = lo.emit(Inst::AtomicLoad {
        ty,
        addr,
        offset: m.offset,
        order: Ordering::SeqCst,
    });
    lo.push(v, int_vt(ty));
    Ok(())
}

fn atomic_store(lo: &mut Lower, ty: IntTy, m: MemArg) -> Result<(), Error> {
    let (value, _) = lo.pop()?;
    let addr = pop_addr(lo)?;
    lo.emit_void(Inst::AtomicStore {
        ty,
        addr,
        value,
        offset: m.offset,
        order: Ordering::SeqCst,
    });
    Ok(())
}

fn atomic_rmw(lo: &mut Lower, ty: IntTy, op: AtomicRmwOp, m: MemArg) -> Result<(), Error> {
    let (value, _) = lo.pop()?;
    let addr = pop_addr(lo)?;
    let v = lo.emit(Inst::AtomicRmw {
        ty,
        op,
        addr,
        value,
        offset: m.offset,
        order: Ordering::SeqCst,
    });
    lo.push(v, int_vt(ty)); // yields the old value
    Ok(())
}

fn atomic_cmpxchg(lo: &mut Lower, ty: IntTy, m: MemArg) -> Result<(), Error> {
    let (replacement, _) = lo.pop()?;
    let (expected, _) = lo.pop()?;
    let addr = pop_addr(lo)?;
    let v = lo.emit(Inst::AtomicCmpxchg {
        ty,
        addr,
        expected,
        replacement,
        offset: m.offset,
        order: Ordering::SeqCst,
    });
    lo.push(v, int_vt(ty)); // yields the old value
    Ok(())
}

/// `memory.atomic.wait32`/`wait64`: pop `[addr, expected, timeout]`, futex-wait, push the i32 status
/// (0 woken / 1 not-equal / 2 timed-out — the IR `MemoryWait` contract).
fn atomic_wait(lo: &mut Lower, ty: IntTy, m: MemArg) -> Result<(), Error> {
    let (timeout, _) = lo.pop()?;
    let (expected, _) = lo.pop()?;
    let addr = pop_atomic_addr(lo, m.offset)?;
    let v = lo.emit(Inst::MemoryWait {
        ty,
        addr,
        expected,
        timeout,
    });
    lo.push(v, ValType::I32);
    Ok(())
}

/// `memory.atomic.notify`: pop `[addr, count]`, wake up to `count` waiters, push the i32 count woken.
fn atomic_notify(lo: &mut Lower, m: MemArg) -> Result<(), Error> {
    let (count, _) = lo.pop()?;
    let addr = pop_atomic_addr(lo, m.offset)?;
    let v = lo.emit(Inst::MemoryNotify { addr, count });
    lo.push(v, ValType::I32);
    Ok(())
}

/// The wasm memory index/size type: `i64` for `memory64`, else `i32`.
fn idx_ty(mem64: bool) -> ValType {
    if mem64 {
        ValType::I64
    } else {
        ValType::I32
    }
}

/// `memory.size`: the current size in pages. With growth it's a load of the runtime size cell; without
/// growth the size is constant (`initial_pages`), so no cell is needed.
fn mem_size_op(lo: &mut Lower) -> Result<(), Error> {
    let ty = idx_ty(lo.mem64);
    let v = if lo.mg.uses_grow {
        let a = lo.emit(Inst::ConstI64(lo.mg.size_cell_off as i64));
        let op = if lo.mem64 { LoadOp::I64 } else { LoadOp::I32 };
        lo.emit(Inst::Load {
            op,
            addr: a,
            offset: 0,
            align: 3,
        })
    } else if lo.mem64 {
        lo.emit(Inst::ConstI64(lo.mg.initial_pages as i64))
    } else {
        lo.emit(Inst::ConstI32(lo.mg.initial_pages as i32))
    };
    lo.push(v, ty);
    Ok(())
}

/// `memory.grow(delta)`: extend by `delta` pages, returning the previous size (or `-1` if it would
/// exceed `max_pages`). Lowered **branch-free**: page math in i64 (the grow delta is unsigned), then
/// the size cell is set to `new` on success / unchanged on failure and the result is `old`/`-1`, each
/// via `select`. Only emitted when `uses_grow`, so the size cell exists.
fn mem_grow_op(lo: &mut Lower) -> Result<(), Error> {
    let ty = idx_ty(lo.mem64);
    let (delta, _) = lo.pop()?;
    let (load_op, store_op) = if lo.mem64 {
        (LoadOp::I64, StoreOp::I64)
    } else {
        (LoadOp::I32, StoreOp::I32)
    };
    let cell = lo.emit(Inst::ConstI64(lo.mg.size_cell_off as i64));
    let old = lo.emit(Inst::Load {
        op: load_op,
        addr: cell,
        offset: 0,
        align: 3,
    });
    // Overflow-safe in i64 (a near-`u32::MAX` delta must not wrap past the cap into a "fits").
    let widen = |lo: &mut Lower, v| {
        if lo.mem64 {
            v
        } else {
            lo.emit(Inst::Convert {
                op: ConvOp::ExtendI32U,
                a: v,
            })
        }
    };
    let old64 = widen(lo, old);
    let delta64 = widen(lo, delta);
    let new64 = lo.emit(Inst::IntBin {
        ty: IntTy::I64,
        op: BinOp::Add,
        a: old64,
        b: delta64,
    });
    let maxc = lo.emit(Inst::ConstI64(lo.mg.max_pages as i64));
    let ok = lo.emit(Inst::IntCmp {
        ty: IntTy::I64,
        op: CmpOp::LeU,
        a: new64,
        b: maxc,
    });
    let new_idx = if lo.mem64 {
        new64
    } else {
        lo.emit(Inst::Convert {
            op: ConvOp::WrapI64,
            a: new64,
        })
    };
    // Store `new` on success, unchanged `old` on failure (a no-op write); reuse `cell` as the address.
    let stored = lo.emit(Inst::Select {
        cond: ok,
        a: new_idx,
        b: old,
    });
    lo.emit_void(Inst::Store {
        op: store_op,
        addr: cell,
        value: stored,
        offset: 0,
        align: 3,
    });
    let negone = if lo.mem64 {
        lo.emit(Inst::ConstI64(-1))
    } else {
        lo.emit(Inst::ConstI32(-1))
    };
    let result = lo.emit(Inst::Select {
        cond: ok,
        a: old,
        b: negone,
    });
    lo.push(result, ty);
    Ok(())
}

/// The largest constant byte length we unroll a `memory.copy`/`fill` into chunked load/stores. A
/// larger (or non-constant) length is a clean `Unsupported` — a dynamic-length runtime loop is a later
/// slice; clang's bulk ops carry small constant struct/array sizes.
const MAX_BULK_UNROLL: i64 = 1 << 16; // 64 KiB

/// Split `len` bytes into `(offset, width)` chunks, widest first (8/4/2/1) — the unroll plan a bulk op
/// lowers to (mirrors the chibicc frontend's `gen_memcpy`).
fn chunk_plan(len: u64) -> Vec<(u64, u64)> {
    let mut plan = Vec::new();
    let mut i = 0u64;
    while i < len {
        let rem = len - i;
        let w = if rem >= 8 {
            8
        } else if rem >= 4 {
            4
        } else if rem >= 2 {
            2
        } else {
            1
        };
        plan.push((i, w));
        i += w;
    }
    plan
}
fn load_w(w: u64) -> LoadOp {
    match w {
        8 => LoadOp::I64,
        4 => LoadOp::I32,
        2 => LoadOp::I32_16U,
        _ => LoadOp::I32_8U,
    }
}
fn store_w(w: u64) -> StoreOp {
    match w {
        8 => StoreOp::I64,
        4 => StoreOp::I32,
        2 => StoreOp::I32_16,
        _ => StoreOp::I32_8,
    }
}

/// The constant byte length of a bulk op, if it is a constant `≤ MAX_BULK_UNROLL` (then it's unrolled
/// into chunked load/stores); otherwise `None` ⇒ lower to a runtime byte loop.
fn const_bulk_len(lo: &Lower, len_v: ValIdx) -> Option<u64> {
    lo.const_of(len_v)
        .filter(|&n| (0..=MAX_BULK_UNROLL).contains(&n))
        .map(|n| n as u64)
}

/// Zero-extend a wasm memory length/index to the i64 window-address space (a no-op for `memory64`).
fn widen_to_i64(lo: &mut Lower, v: ValIdx) -> ValIdx {
    if lo.mem64 {
        v
    } else {
        lo.emit(Inst::Convert {
            op: ConvOp::ExtendI32U,
            a: v,
        })
    }
}

/// `memory.fill(dest, val, len)`: set `len` bytes at `dest` to byte `val`. A constant `len` is
/// unrolled into chunked stores of the fill byte broadcast to each chunk width; a runtime `len` lowers
/// to a byte loop.
fn mem_fill_op(lo: &mut Lower) -> Result<(), Error> {
    let (len_v, _) = lo.pop()?;
    let (val, _) = lo.pop()?; // the fill byte (low 8 bits of an i32)
    let dest = pop_addr(lo)?; // i64 window address
    match const_bulk_len(lo, len_v) {
        Some(n) => fill_unroll(lo, dest, val, n),
        None => fill_dynamic(lo, dest, val, len_v),
    }
}

/// Unrolled constant-length fill: store the fill byte (broadcast per chunk width) at each chunk.
fn fill_unroll(lo: &mut Lower, dest: ValIdx, val: ValIdx, n: u64) -> Result<(), Error> {
    if n == 0 {
        return Ok(());
    }
    // The fill byte broadcast to each width: vb·0x01… (so every byte of the chunk is the fill byte).
    let m255 = lo.emit(Inst::ConstI32(0xFF));
    let byte = lo.emit(Inst::IntBin {
        ty: IntTy::I32,
        op: BinOp::And,
        a: val,
        b: m255,
    });
    let mul_i32 = |lo: &mut Lower, k: i32| {
        let m = lo.emit(Inst::ConstI32(k));
        lo.emit(Inst::IntBin {
            ty: IntTy::I32,
            op: BinOp::Mul,
            a: byte,
            b: m,
        })
    };
    let b2 = mul_i32(lo, 0x0001_0101);
    let b4 = mul_i32(lo, 0x0101_0101);
    let b8 = {
        let b64 = lo.emit(Inst::Convert {
            op: ConvOp::ExtendI32U,
            a: byte,
        });
        let m = lo.emit(Inst::ConstI64(0x0101_0101_0101_0101));
        lo.emit(Inst::IntBin {
            ty: IntTy::I64,
            op: BinOp::Mul,
            a: b64,
            b: m,
        })
    };
    for (off, w) in chunk_plan(n) {
        let value = match w {
            8 => b8,
            4 => b4,
            2 => b2,
            _ => byte,
        };
        lo.emit_void(Inst::Store {
            op: store_w(w),
            addr: dest,
            value,
            offset: off,
            align: 0,
        });
    }
    Ok(())
}

/// Runtime-length fill as a forward byte loop: `for (i = 0; i < n; i++) store8(dest + i, val)`.
/// Synthesized as header/body/exit blocks threading the prefix + operand stack + the loop-private
/// `(dest, val, n, i)`.
fn fill_dynamic(lo: &mut Lower, dest: ValIdx, val: ValIdx, len: ValIdx) -> Result<(), Error> {
    let below_t: Vec<ValType> = lo.stack.iter().map(|(_, t)| *t).collect();
    let below_v = lo.stack_vals();
    let n = widen_to_i64(lo, len);
    let extra = [ValType::I64, ValType::I32, ValType::I64, ValType::I64]; // dest, val, n, i
    let hsig = lo.synth_sig(&below_t, &extra);
    let header = lo.new_block(hsig.clone());
    let body = lo.new_block(hsig);
    let exit_sig = lo.synth_sig(&below_t, &[]);
    let exit = lo.new_block(exit_sig);

    let zero = lo.emit(Inst::ConstI64(0));
    let args = lo.synth_args(&below_v, &[dest, val, n, zero]);
    lo.set_term(Terminator::Br {
        target: header as u32,
        args,
    });

    // header: while i < n → body, else → exit.
    let hx = lo.enter_synth(header, &below_t, 4);
    let (d, v, nn, i) = (hx[0], hx[1], hx[2], hx[3]);
    let cond = lo.emit(Inst::IntCmp {
        ty: IntTy::I64,
        op: CmpOp::LtU,
        a: i,
        b: nn,
    });
    let bv = lo.stack_vals();
    let then_args = lo.synth_args(&bv, &[d, v, nn, i]);
    let else_args = lo.synth_args(&bv, &[]);
    lo.set_term(Terminator::BrIf {
        cond,
        then_blk: body as u32,
        then_args,
        else_blk: exit as u32,
        else_args,
    });

    // body: store8(d + i, v); i += 1; back to header.
    let bx = lo.enter_synth(body, &below_t, 4);
    let (d, v, nn, i) = (bx[0], bx[1], bx[2], bx[3]);
    let addr = lo.emit(Inst::IntBin {
        ty: IntTy::I64,
        op: BinOp::Add,
        a: d,
        b: i,
    });
    lo.emit_void(Inst::Store {
        op: StoreOp::I32_8,
        addr,
        value: v,
        offset: 0,
        align: 0,
    });
    let one = lo.emit(Inst::ConstI64(1));
    let i1 = lo.emit(Inst::IntBin {
        ty: IntTy::I64,
        op: BinOp::Add,
        a: i,
        b: one,
    });
    let bv = lo.stack_vals();
    let back = lo.synth_args(&bv, &[d, v, nn, i1]);
    lo.set_term(Terminator::Br {
        target: header as u32,
        args: back,
    });

    lo.enter(exit, &below_t); // continue with the operand stack restored
    Ok(())
}

/// `memory.copy(dest, src, len)`: copy `len` bytes (overlap-safe, like memmove). A constant `len`
/// loads every chunk before storing any (overlap-safe); a runtime `len` lowers to a direction-correct
/// byte loop.
fn mem_copy_op(lo: &mut Lower) -> Result<(), Error> {
    let (len_v, _) = lo.pop()?;
    let src = pop_addr(lo)?;
    let dest = pop_addr(lo)?;
    match const_bulk_len(lo, len_v) {
        Some(n) => copy_unroll(lo, dest, src, n),
        None => copy_dynamic(lo, dest, src, len_v),
    }
}

/// Unrolled constant-length copy: load every chunk, then store every chunk (overlap-safe memmove).
fn copy_unroll(lo: &mut Lower, dest: ValIdx, src: ValIdx, n: u64) -> Result<(), Error> {
    if n == 0 {
        return Ok(());
    }
    let plan = chunk_plan(n);
    let loaded: Vec<ValIdx> = plan
        .iter()
        .map(|&(off, w)| {
            lo.emit(Inst::Load {
                op: load_w(w),
                addr: src,
                offset: off,
                align: 0,
            })
        })
        .collect();
    for (&(off, w), &value) in plan.iter().zip(&loaded) {
        lo.emit_void(Inst::Store {
            op: store_w(w),
            addr: dest,
            value,
            offset: off,
            align: 0,
        });
    }
    Ok(())
}

/// Runtime-length copy as a **memmove** byte loop: copy forward when `dest ≤ src`, backward when
/// `dest > src` (so overlapping ranges are correct). Synthesized as a direction branch into a
/// forward and a backward header/body, both exiting to one continuation block. All blocks thread the
/// prefix + operand stack + the loop-private `(dest, src, n, i)`.
fn copy_dynamic(lo: &mut Lower, dest: ValIdx, src: ValIdx, len: ValIdx) -> Result<(), Error> {
    let below_t: Vec<ValType> = lo.stack.iter().map(|(_, t)| *t).collect();
    let below_v = lo.stack_vals();
    let n = widen_to_i64(lo, len);
    let extra = [ValType::I64, ValType::I64, ValType::I64, ValType::I64]; // dest, src, n, i
    let lsig = lo.synth_sig(&below_t, &extra);
    let fwd_h = lo.new_block(lsig.clone());
    let fwd_b = lo.new_block(lsig.clone());
    let bwd_h = lo.new_block(lsig.clone());
    let bwd_b = lo.new_block(lsig);
    let exit = {
        let s = lo.synth_sig(&below_t, &[]);
        lo.new_block(s)
    };

    // Direction: backward (start i = n) when dest > src, else forward (start i = 0).
    let desc = lo.emit(Inst::IntCmp {
        ty: IntTy::I64,
        op: CmpOp::GtU,
        a: dest,
        b: src,
    });
    let zero = lo.emit(Inst::ConstI64(0));
    let fwd_args = lo.synth_args(&below_v, &[dest, src, n, zero]);
    let bwd_args = lo.synth_args(&below_v, &[dest, src, n, n]);
    lo.set_term(Terminator::BrIf {
        cond: desc,
        then_blk: bwd_h as u32,
        then_args: bwd_args,
        else_blk: fwd_h as u32,
        else_args: fwd_args,
    });

    // Forward: while i < n → copy [i], i++.
    let hx = lo.enter_synth(fwd_h, &below_t, 4);
    let (d, s, nn, i) = (hx[0], hx[1], hx[2], hx[3]);
    let cond = lo.emit(Inst::IntCmp {
        ty: IntTy::I64,
        op: CmpOp::LtU,
        a: i,
        b: nn,
    });
    let bv = lo.stack_vals();
    let ta = lo.synth_args(&bv, &[d, s, nn, i]);
    let ea = lo.synth_args(&bv, &[]);
    lo.set_term(Terminator::BrIf {
        cond,
        then_blk: fwd_b as u32,
        then_args: ta,
        else_blk: exit as u32,
        else_args: ea,
    });
    let bx = lo.enter_synth(fwd_b, &below_t, 4);
    let (d, s, nn, i) = (bx[0], bx[1], bx[2], bx[3]);
    copy_one(lo, d, s, i); // store8(d+i, load8(s+i))
    let one = lo.emit(Inst::ConstI64(1));
    let i1 = lo.emit(Inst::IntBin {
        ty: IntTy::I64,
        op: BinOp::Add,
        a: i,
        b: one,
    });
    let bv = lo.stack_vals();
    let back = lo.synth_args(&bv, &[d, s, nn, i1]);
    lo.set_term(Terminator::Br {
        target: fwd_h as u32,
        args: back,
    });

    // Backward: while i > 0 → i--, copy [i].
    let hx = lo.enter_synth(bwd_h, &below_t, 4);
    let (d, s, nn, i) = (hx[0], hx[1], hx[2], hx[3]);
    let z = lo.emit(Inst::ConstI64(0));
    let cond = lo.emit(Inst::IntCmp {
        ty: IntTy::I64,
        op: CmpOp::Ne,
        a: i,
        b: z,
    });
    let bv = lo.stack_vals();
    let ta = lo.synth_args(&bv, &[d, s, nn, i]);
    let ea = lo.synth_args(&bv, &[]);
    lo.set_term(Terminator::BrIf {
        cond,
        then_blk: bwd_b as u32,
        then_args: ta,
        else_blk: exit as u32,
        else_args: ea,
    });
    let bx = lo.enter_synth(bwd_b, &below_t, 4);
    let (d, s, nn, i) = (bx[0], bx[1], bx[2], bx[3]);
    let one = lo.emit(Inst::ConstI64(1));
    let j = lo.emit(Inst::IntBin {
        ty: IntTy::I64,
        op: BinOp::Sub,
        a: i,
        b: one,
    });
    copy_one(lo, d, s, j); // store8(d+j, load8(s+j))
    let bv = lo.stack_vals();
    let back = lo.synth_args(&bv, &[d, s, nn, j]);
    lo.set_term(Terminator::Br {
        target: bwd_h as u32,
        args: back,
    });

    lo.enter(exit, &below_t);
    Ok(())
}

/// Emit `store8(d + idx, load8(s + idx))` (one byte of a runtime-length copy).
fn copy_one(lo: &mut Lower, d: ValIdx, s: ValIdx, idx: ValIdx) {
    let sa = lo.emit(Inst::IntBin {
        ty: IntTy::I64,
        op: BinOp::Add,
        a: s,
        b: idx,
    });
    let byte = lo.emit(Inst::Load {
        op: LoadOp::I32_8U,
        addr: sa,
        offset: 0,
        align: 0,
    });
    let da = lo.emit(Inst::IntBin {
        ty: IntTy::I64,
        op: BinOp::Add,
        a: d,
        b: idx,
    });
    lo.emit_void(Inst::Store {
        op: StoreOp::I32_8,
        addr: da,
        value: byte,
        offset: 0,
        align: 0,
    });
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
            lo.consts.insert(v, value as i64);
            lo.push(v, ValType::I32);
        }
        O::I64Const { value } => {
            let v = lo.emit(Inst::ConstI64(value));
            lo.consts.insert(v, value);
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
        // ---- §12 wasm threads: the full-width (i32/i64) atomics (narrow forms hit the catch-all) ----
        O::I32AtomicLoad { memarg } => atomic_load(lo, IntTy::I32, memarg)?,
        O::I64AtomicLoad { memarg } => atomic_load(lo, IntTy::I64, memarg)?,
        O::I32AtomicStore { memarg } => atomic_store(lo, IntTy::I32, memarg)?,
        O::I64AtomicStore { memarg } => atomic_store(lo, IntTy::I64, memarg)?,
        O::I32AtomicRmwAdd { memarg } => atomic_rmw(lo, IntTy::I32, AtomicRmwOp::Add, memarg)?,
        O::I32AtomicRmwSub { memarg } => atomic_rmw(lo, IntTy::I32, AtomicRmwOp::Sub, memarg)?,
        O::I32AtomicRmwAnd { memarg } => atomic_rmw(lo, IntTy::I32, AtomicRmwOp::And, memarg)?,
        O::I32AtomicRmwOr { memarg } => atomic_rmw(lo, IntTy::I32, AtomicRmwOp::Or, memarg)?,
        O::I32AtomicRmwXor { memarg } => atomic_rmw(lo, IntTy::I32, AtomicRmwOp::Xor, memarg)?,
        O::I32AtomicRmwXchg { memarg } => atomic_rmw(lo, IntTy::I32, AtomicRmwOp::Xchg, memarg)?,
        O::I32AtomicRmwCmpxchg { memarg } => atomic_cmpxchg(lo, IntTy::I32, memarg)?,
        O::I64AtomicRmwAdd { memarg } => atomic_rmw(lo, IntTy::I64, AtomicRmwOp::Add, memarg)?,
        O::I64AtomicRmwSub { memarg } => atomic_rmw(lo, IntTy::I64, AtomicRmwOp::Sub, memarg)?,
        O::I64AtomicRmwAnd { memarg } => atomic_rmw(lo, IntTy::I64, AtomicRmwOp::And, memarg)?,
        O::I64AtomicRmwOr { memarg } => atomic_rmw(lo, IntTy::I64, AtomicRmwOp::Or, memarg)?,
        O::I64AtomicRmwXor { memarg } => atomic_rmw(lo, IntTy::I64, AtomicRmwOp::Xor, memarg)?,
        O::I64AtomicRmwXchg { memarg } => atomic_rmw(lo, IntTy::I64, AtomicRmwOp::Xchg, memarg)?,
        O::I64AtomicRmwCmpxchg { memarg } => atomic_cmpxchg(lo, IntTy::I64, memarg)?,
        O::MemoryAtomicWait32 { memarg } => atomic_wait(lo, IntTy::I32, memarg)?,
        O::MemoryAtomicWait64 { memarg } => atomic_wait(lo, IntTy::I64, memarg)?,
        O::MemoryAtomicNotify { memarg } => atomic_notify(lo, memarg)?,
        // A standalone seq-cst fence (`__atomic_thread_fence`): the IR fence is honoured by the interp
        // and lowered to a real hardware barrier by the JIT (all SVM atomics are already seq-cst).
        O::AtomicFence => lo.emit_void(Inst::AtomicFence {
            order: Ordering::SeqCst,
        }),
        // ---- memory.size / memory.grow (pages; the window holds the growable span) ----
        O::MemorySize { mem } => {
            if mem != 0 {
                return unsup("memory.size on a non-zero memory");
            }
            mem_size_op(lo)?;
        }
        O::MemoryGrow { mem } => {
            if mem != 0 {
                return unsup("memory.grow on a non-zero memory");
            }
            mem_grow_op(lo)?;
        }
        // ---- bulk memory: memory.fill / memory.copy (constant length ⇒ unrolled chunks) ----
        O::MemoryFill { mem } => {
            if mem != 0 {
                return unsup("memory.fill on a non-zero memory");
            }
            mem_fill_op(lo)?;
        }
        O::MemoryCopy { dst_mem, src_mem } => {
            if dst_mem != 0 || src_mem != 0 {
                return unsup("memory.copy on a non-zero memory");
            }
            mem_copy_op(lo)?;
        }
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
        // ---- tail calls (the tail-call proposal): a block-terminating call ----
        O::ReturnCall { function_index } => return_call_op(lo, function_index, fn_results)?,
        O::ReturnCallIndirect {
            type_index,
            table_index,
        } => return_call_indirect_op(lo, type_index, table_index)?,
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
            // Cond-false edge: continue in a fresh block carrying the prefix + the full current stack.
            let cont_types: Vec<ValType> = lo.stack.iter().map(|(_, t)| *t).collect();
            let cont = lo.new_block(lo.sig(&cont_types));
            let mut else_args = lo.prefix_vals();
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

        // ---- §17 SIMD (D58): the pragmatic v128 subset our IR supports ----
        O::V128Const { value } => {
            let v = lo.emit(Inst::ConstV128(*value.bytes()));
            lo.push(v, ValType::V128);
        }
        O::V128Load { memarg } => v128_load(lo, memarg)?,
        O::V128Store { memarg } => v128_store(lo, memarg)?,
        // splat
        O::I8x16Splat => v_splat(lo, VShape::I8x16)?,
        O::I16x8Splat => v_splat(lo, VShape::I16x8)?,
        O::I32x4Splat => v_splat(lo, VShape::I32x4)?,
        O::I64x2Splat => v_splat(lo, VShape::I64x2)?,
        O::F32x4Splat => v_splat(lo, VShape::F32x4)?,
        O::F64x2Splat => v_splat(lo, VShape::F64x2)?,
        // extract_lane (narrow int shapes carry sign)
        O::I8x16ExtractLaneS { lane } => v_extract(lo, VShape::I8x16, lane, true)?,
        O::I8x16ExtractLaneU { lane } => v_extract(lo, VShape::I8x16, lane, false)?,
        O::I16x8ExtractLaneS { lane } => v_extract(lo, VShape::I16x8, lane, true)?,
        O::I16x8ExtractLaneU { lane } => v_extract(lo, VShape::I16x8, lane, false)?,
        O::I32x4ExtractLane { lane } => v_extract(lo, VShape::I32x4, lane, false)?,
        O::I64x2ExtractLane { lane } => v_extract(lo, VShape::I64x2, lane, false)?,
        O::F32x4ExtractLane { lane } => v_extract(lo, VShape::F32x4, lane, false)?,
        O::F64x2ExtractLane { lane } => v_extract(lo, VShape::F64x2, lane, false)?,
        // replace_lane
        O::I8x16ReplaceLane { lane } => v_replace(lo, VShape::I8x16, lane)?,
        O::I16x8ReplaceLane { lane } => v_replace(lo, VShape::I16x8, lane)?,
        O::I32x4ReplaceLane { lane } => v_replace(lo, VShape::I32x4, lane)?,
        O::I64x2ReplaceLane { lane } => v_replace(lo, VShape::I64x2, lane)?,
        O::F32x4ReplaceLane { lane } => v_replace(lo, VShape::F32x4, lane)?,
        O::F64x2ReplaceLane { lane } => v_replace(lo, VShape::F64x2, lane)?,
        // integer lane add/sub/mul (i8x16.mul has no wasm op, so it never appears)
        O::I8x16Add => v_intbin(lo, VShape::I8x16, VIntBinOp::Add)?,
        O::I8x16Sub => v_intbin(lo, VShape::I8x16, VIntBinOp::Sub)?,
        O::I16x8Add => v_intbin(lo, VShape::I16x8, VIntBinOp::Add)?,
        O::I16x8Sub => v_intbin(lo, VShape::I16x8, VIntBinOp::Sub)?,
        O::I16x8Mul => v_intbin(lo, VShape::I16x8, VIntBinOp::Mul)?,
        O::I32x4Add => v_intbin(lo, VShape::I32x4, VIntBinOp::Add)?,
        O::I32x4Sub => v_intbin(lo, VShape::I32x4, VIntBinOp::Sub)?,
        O::I32x4Mul => v_intbin(lo, VShape::I32x4, VIntBinOp::Mul)?,
        O::I64x2Add => v_intbin(lo, VShape::I64x2, VIntBinOp::Add)?,
        O::I64x2Sub => v_intbin(lo, VShape::I64x2, VIntBinOp::Sub)?,
        O::I64x2Mul => v_intbin(lo, VShape::I64x2, VIntBinOp::Mul)?,
        // float lane arithmetic
        O::F32x4Add => v_fbin(lo, VShape::F32x4, VFloatBinOp::Add)?,
        O::F32x4Sub => v_fbin(lo, VShape::F32x4, VFloatBinOp::Sub)?,
        O::F32x4Mul => v_fbin(lo, VShape::F32x4, VFloatBinOp::Mul)?,
        O::F32x4Div => v_fbin(lo, VShape::F32x4, VFloatBinOp::Div)?,
        O::F32x4Min => v_fbin(lo, VShape::F32x4, VFloatBinOp::Min)?,
        O::F32x4Max => v_fbin(lo, VShape::F32x4, VFloatBinOp::Max)?,
        O::F64x2Add => v_fbin(lo, VShape::F64x2, VFloatBinOp::Add)?,
        O::F64x2Sub => v_fbin(lo, VShape::F64x2, VFloatBinOp::Sub)?,
        O::F64x2Mul => v_fbin(lo, VShape::F64x2, VFloatBinOp::Mul)?,
        O::F64x2Div => v_fbin(lo, VShape::F64x2, VFloatBinOp::Div)?,
        O::F64x2Min => v_fbin(lo, VShape::F64x2, VFloatBinOp::Min)?,
        O::F64x2Max => v_fbin(lo, VShape::F64x2, VFloatBinOp::Max)?,
        O::F32x4Abs => v_fun(lo, VShape::F32x4, VFloatUnOp::Abs)?,
        O::F32x4Neg => v_fun(lo, VShape::F32x4, VFloatUnOp::Neg)?,
        O::F32x4Sqrt => v_fun(lo, VShape::F32x4, VFloatUnOp::Sqrt)?,
        O::F64x2Abs => v_fun(lo, VShape::F64x2, VFloatUnOp::Abs)?,
        O::F64x2Neg => v_fun(lo, VShape::F64x2, VFloatUnOp::Neg)?,
        O::F64x2Sqrt => v_fun(lo, VShape::F64x2, VFloatUnOp::Sqrt)?,
        // whole-vector bitwise
        O::V128And => v_bitbin(lo, VBitBinOp::And)?,
        O::V128Or => v_bitbin(lo, VBitBinOp::Or)?,
        O::V128Xor => v_bitbin(lo, VBitBinOp::Xor)?,
        O::V128AndNot => v_bitbin(lo, VBitBinOp::AndNot)?,
        O::V128Not => {
            let (a, _) = lo.pop()?;
            let v = lo.emit(Inst::VNot { a });
            lo.push(v, ValType::V128);
        }
        O::V128Bitselect => {
            // wasm stack: a, b, mask (mask on top). IR `bitselect(a, b, mask)` = `(a&mask)|(b&!mask)`,
            // matching wasm's `v128.bitselect` (bit set in mask ⇒ take a).
            let (mask, _) = lo.pop()?;
            let (b, _) = lo.pop()?;
            let (a, _) = lo.pop()?;
            let v = lo.emit(Inst::Bitselect { a, b, mask });
            lo.push(v, ValType::V128);
        }
        O::I8x16Shuffle { lanes } => {
            let (b, _) = lo.pop()?;
            let (a, _) = lo.pop()?;
            let v = lo.emit(Inst::Shuffle { lanes, a, b });
            lo.push(v, ValType::V128);
        }
        O::I8x16Swizzle => {
            let (b, _) = lo.pop()?;
            let (a, _) = lo.pop()?;
            let v = lo.emit(Inst::Swizzle { a, b });
            lo.push(v, ValType::V128);
        }

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
    let mut args = lo.prefix_vals();
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
