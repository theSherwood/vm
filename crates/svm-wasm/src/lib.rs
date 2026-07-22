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
//! **function imports** (a wasm `call` to an import → a `call.import` on the module's manifest slot;
//! each import's `module`/`name` becomes a named manifest entry the host binds at instantiation — see
//! [`transpile`]) · **§17/D58 SIMD** (`v128` → the IR's first-class fixed-128 vector type: const,
//! masked load/store + the memory variants (splat-load/load-extend/load-zero/load+store-lane), splat,
//! extract/replace_lane, the full integer-/float-lane op set, bitwise + `bitselect`,
//! `shuffle`/`swizzle`, and **relaxed SIMD** — a real `clang -msimd128 -O2` saxpy transpiles to
//! verified SIMD IR, `tests/simd.rs`) · **§12 wasm threads**: the full-width (i32/i64) `*.atomic.*` ops map
//! 1:1 onto SVM's IR atomics (`tests/atomics.rs`), the **narrow** (8/16-bit) `*.atomic.*` forms
//! emulate via a 32-bit word-CAS loop (the i64 32-bit forms are word-sized natives), **shared** +
//! **imported** memory are accepted, and the **wasi-threads** ABI lowers to SVM's *native*
//! `thread.spawn` — a `wasi:thread/spawn` import becomes a real OS-thread vCPU over the shared window
//! via a synthesized shim (concurrency in the VM, DESIGN §1a; the same bytes `wasmtime-wasi-threads`
//! runs — `tests/threads.rs`). Capability imports dispatch through the instance's slot bindings on
//! every thread (so `wasi:thread/spawn` works *alongside* capability imports with no plumbing), and
//! **reference types** core
//! (`funcref`/`externref` as i32 values, `ref.null`/`is_null`/`func`, typed `select`,
//! `table.get`/`set`/`size`/`fill`). Still a clean [`Error::Unsupported`] (the niche features typical
//! clang output doesn't emit): `table.copy`/`init`/`grow` + passive *element* segments; imported
//! table/global/tag; multi-memory/multi-table. **Passive data segments + `memory.init`/`data.drop`**
//! are supported (a constant-offset init unrolls to const-stores of the segment's known bytes).

use std::collections::BTreeMap;
use svm_ir::{
    AtomicRmwOp, BinOp, Block, CastOp, CmpOp, ConvOp, DebugInfo, Edge, FBinOp, FCmpOp, FToI, FUnOp,
    Field, FloatTy, Func, FuncName, FuncType, IToF, Inst, IntTy, IntUnOp, LoadOp, Loc, Module,
    Ordering, SsaLoc, StoreOp, Terminator, TypeDef, VBitBinOp, VCvtOp, VFCmpOp, VFloatBinOp,
    VFloatUnOp, VICmpOp, VIntBinOp, VIntUnOp, VNarrowOp, VPMinMaxOp, VSatBinOp, VShape, VShiftOp,
    VWidenOp, ValIdx, ValType, VarInfo, VarLoc,
};
use wasmparser::{BlockType, MemArg, Operator, Parser, Payload, ValType as W};

/// DWARF `.debug_info` reader for source-variable ingest (DEBUGGING.md W4 — wasm producer). Public
/// so it is testable against a real fixture; the transpiler wiring lands in a follow-up slice.
pub mod dwarf_info;
pub mod dwarf_line;

/// Per-operator debug records from lowering: `(code-relative offset, block, inst index)` for the
/// first IR instruction each wasm operator emits (DEBUGGING.md §6/W4 — mapped onto DWARF line rows).
type OpLocs = Vec<(u32, u32, u32)>;

/// Per-wasm-local debug records from lowering: `(local index, block, inst, SSA value)` — each change
/// of a local's holding value, the `SsaList` a DWARF frame-pointer local needs (W4 variable ingest).
type LocalLocs = Vec<(u32, u32, u32, u32)>;

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
/// hold the whole growable span (it is eagerly committed, so a grown page is reachable; an access
/// past it traps under SVM's trap-confinement), so for unbounded memory this is a modest cap
/// (16 MiB) that keeps the committed window small. A declared `maximum` is honored instead.
const DEFAULT_MAX_GROW_PAGES: u64 = 256;
/// Hard ceiling on the growable span regardless of a declared `maximum`, so a pathological `maximum`
/// can't blow up the committed window (256 MiB).
const MAX_GROW_PAGES: u64 = 4096;

/// `table.grow` on an **unbounded** wasm table may extend up to this many slots. As with memory the
/// window must hold the whole growable span (eagerly committed; an over-span access traps under
/// trap-confinement), so for an unbounded table this is a modest cap (4 KiB of slots) keeping the
/// committed window small. A
/// declared table `maximum` is honored instead.
const DEFAULT_MAX_GROW_SLOTS: u64 = 1024;
/// Hard ceiling on the growable table span regardless of a declared `maximum` (256 KiB of slots).
const MAX_GROW_SLOTS: u64 = 65536;

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
        // §RT reference types: both `funcref` and `externref` are an opaque **i32 index** in SVM — a
        // `funcref` is the §3c function-table index (= the IR function index), an `externref` is a §7
        // capability handle (a host-table index). The wasm validator keeps the two from being mixed;
        // SVM only needs the bit representation, so both lower to `i32`.
        W::Ref(_) => Ok(ValType::I32),
    }
}

/// The null reference sentinel (both `funcref` and `externref`). Matches the `0xFFFF_FFFF` the
/// function table already uses for an empty slot, so a null `funcref` fed to `call_indirect` fails
/// the §3c index/type check (≈ wasm's null-funcref trap), and a null `externref` fed to `cap.call`
/// indexes no granted capability (a cap fault). `ref.is_null` is a compare against this.
const REF_NULL: i32 = -1;

/// Transpile a core-wasm binary into a verifier-checkable [`Module`].
///
/// **Host-function imports (the host ABI).** A wasm `(import "<module>" "<name>" (func …))` becomes
/// one entry in the module's import **manifest** (IMPORTS.md phase 3), named `"<module>.<name>"` —
/// for the numeric host-ABI convention (`module` = decimal capability `type_id`, `name` = decimal
/// `op`, e.g. `"13.0"`) and §7 named imports (e.g. `"wasi_snapshot_preview1.fd_write"`) alike. A
/// wasm `call` to import `i` lowers to `call.import i`; the host's instantiation policy maps each
/// name to a concrete capability and fills slot `i` before entry. Nothing is threaded through the
/// guest — functions carry exactly their wasm signatures, and the transpiler stays pure mechanism
/// (it never interprets host semantics — names just select an interface/method). A table/global/tag
/// import is still a clean [`Error::Unsupported`].
///
/// **Linear-memory growth (`memory.size` / `memory.grow`).** The linear memory sits at window offset 0
/// (wasm address `a` == window address `a`). When a module uses `memory.grow`, the window reserves the
/// memory's *full growable span* at the bottom — up to its declared `maximum`, or a modest default
/// ([`DEFAULT_MAX_GROW_PAGES`], bounded by [`MAX_GROW_PAGES`]) for unbounded memory — and places the
/// globals/table regions above it, so growth never collides with them. A runtime **size cell** (an
/// 8-byte window slot just above the linear memory, initialized to the initial page count) holds the
/// current size: `memory.size` loads it and `memory.grow` updates it branch-free (set to the new size
/// on success / unchanged on a past-cap failure, returning the old size or `-1`). The whole growable
/// span is eagerly committed (`mapped`) in the window, so a grown page is simply reachable and the size
/// cell only governs the `size`/`grow` *return values*; an access past the reserved span traps under
/// SVM's trap-confinement (matching wasm's out-of-bounds trap). A module that never grows is unchanged
/// (no cell, the tight initial-sized window, `memory.size` a constant).
pub fn transpile(wasm: &[u8]) -> Result<Transpiled, Error> {
    // Fail-closed on malformed / invalid wasm *before* any lowering. The lowering pass below indexes
    // attacker-controlled type/function/global/local/table/branch indices and derives operand-stack
    // heights straight from the byte stream; on hostile-but-decodable input those raw `[...]` accesses
    // and `len() - k` subtractions would panic, and an oversized locals/table declaration would
    // allocate unboundedly (a host-side OOM / `abort`). A full validation pass up front guarantees wasm
    // validity — in-range indices, well-typed and arity-correct operands, and wasmparser's
    // implementation limits (≤50 000 locals/function, ≤10 000 000 table entries, …) — so every such
    // hostile input becomes a clean `Error` here instead. The default feature set covers exactly the
    // proposals we lower (mutable-global, bulk-memory, SIMD, threads, multi-value, …); anything we do
    // not support is still rejected later by an `Unsupported` bail or, ultimately, the IR verifier.
    wasmparser::Validator::new().validate_all(wasm)?;

    let mut types: Vec<(Vec<ValType>, Vec<ValType>)> = Vec::new();
    let mut func_type_idx: Vec<u32> = Vec::new();
    let mut bodies: Vec<wasmparser::FunctionBody> = Vec::new();
    // Debug-info on-ramp (DEBUGGING.md §6/W4): the file offset where the code section's content
    // begins (operator offsets are file-absolute, DWARF addresses are code-relative — the delta),
    // and the raw embedded `.debug_line` DWARF, if any.
    let mut code_content_start = 0usize;
    let mut debug_line: Option<Vec<u8>> = None;
    // Every embedded DWARF section (`.debug_*`), passed through verbatim as §6 rich blobs so a
    // future DWARF re-emitter has the guest's full native debug info (nothing is lost on transpile).
    let mut debug_blobs: Vec<svm_ir::ProducerBlob> = Vec::new();
    let mut exports: Vec<(String, u32)> = Vec::new();
    let mut mem: Option<wasmparser::MemoryType> = None;
    let mut data: Vec<svm_ir::Data> = Vec::new();
    // Every data segment's bytes, in declaration order — what `memory.init`/`data.drop` reference by
    // index (both active and passive count). Active segments are *also* materialized into `data`
    // (placed at instantiation); passive ones live only here. The bytes are known at transpile time,
    // so a constant-offset `memory.init` is unrolled into stores of them (no runtime passive store).
    let mut data_segments: Vec<Vec<u8>> = Vec::new();
    let mut globals: Vec<(ValType, Vec<u8>)> = Vec::new();
    let mut table_size: Option<u64> = None;
    let mut table_max: Option<u64> = None;
    let mut elements: Vec<(u64, Vec<u32>)> = Vec::new(); // active segments: (offset, func indices)
                                                         // Every element segment's IR-function-index list, by segment index — the `table.init` sources
                                                         // (active and passive both count). A *declarative* segment is `table.init`-empty (dropped). Mirrors
                                                         // `data_segments` for `memory.init`.
    let mut element_segments: Vec<Vec<u32>> = Vec::new();
    // Function imports, in import order: `(manifest slot, sig)`. The function index space puts
    // imports first, so a wasm index `< imports.len()` is an import; a `call` to one lowers to
    // `call.import <slot>` (the spawn placeholder, which is not a capability, carries `u32::MAX`).
    let mut imports: Vec<(u32, FuncType)> = Vec::new();
    // The module-level import **manifest** (IMPORTS.md phase 3): every non-spawn function import is
    // one named, signature-carrying entry — `"<module>.<name>"` for both the numeric host-ABI
    // convention ("13.0") and §7 named imports ("wasi_snapshot_preview1.fd_write"). A `call` to the
    // import lowers to `call.import <slot>`; the host binds each slot at instantiation. Nothing is
    // threaded through the guest — no leading handle params, no spawn stash.
    let mut manifest: Vec<svm_ir::Import> = Vec::new();
    // §3.5: manifest signatures live in the type section; intern each import's sig once.
    let mut manifest_types: Vec<svm_ir::TypeEntry> = Vec::new();
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
                        // placeholder (never dispatched; no manifest slot)
                        imports.push((u32::MAX, sig));
                        continue;
                    }
                    // Every function import is one manifest slot, named `"<module>.<name>"` (IMPORTS.md
                    // phase 3). The numeric host-ABI convention ("13.0" — module = decimal type_id,
                    // name = decimal op) and §7 named imports ("wasi_snapshot_preview1.fd_write") are
                    // the same mechanism now: the transpiler stays pure — the host's instantiation
                    // policy maps each name to a concrete capability and fills slot `i`; a `call`
                    // lowers to `call.import <slot>` with nothing threaded through the guest.
                    let name = format!("{}.{}", imp.module, imp.name);
                    let idx = manifest.len() as u32;
                    let t = manifest_types
                        .iter()
                        .position(|e| matches!(e, svm_ir::TypeEntry::Func(f) if *f == sig))
                        .unwrap_or_else(|| {
                            manifest_types.push(svm_ir::TypeEntry::Func(sig.clone()));
                            manifest_types.len() - 1
                        }) as u32;
                    manifest.push(svm_ir::Import {
                        name,
                        shape: svm_ir::ImportShape::Func(t),
                        mode: svm_ir::ImportMode::Required,
                    });
                    imports.push((idx, sig));
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
                    table_max = tb.ty.maximum;
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
                            let fs = element_funcs(el.items, imports.len() as u32)?;
                            elements.push((off, fs.clone())); // materialized into the table below
                            element_segments.push(fs); // also a `table.init` source, by index
                        }
                        // Passive segments are `table.init` sources only (not auto-applied).
                        wasmparser::ElementKind::Passive => {
                            element_segments.push(element_funcs(el.items, imports.len() as u32)?);
                        }
                        // A *declarative* element segment (`elem declare`) only marks functions as
                        // referenceable by `ref.func` (the validation rule); no runtime effect, and it
                        // is `table.init`-empty (dropped at instantiation).
                        wasmparser::ElementKind::Declared => element_segments.push(Vec::new()),
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
            // The code section's content start — the base for converting file-absolute operator
            // offsets to the code-relative addresses DWARF line entries use.
            Payload::CodeSectionStart { range, .. } => code_content_start = range.start,
            Payload::CodeSectionEntry(body) => bodies.push(body),
            // Embedded DWARF: parse `.debug_line` for source locations, and pass every `.debug_*`
            // section through verbatim as a rich blob (§6 / D-DBG-7) for a future DWARF re-emitter.
            Payload::CustomSection(reader) if reader.name().starts_with(".debug") => {
                if reader.name() == ".debug_line" {
                    debug_line = Some(reader.data().to_vec());
                }
                debug_blobs.push(svm_ir::ProducerBlob {
                    producer: reader.name().to_string(),
                    bytes: reader.data().to_vec(),
                });
            }
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
                            data_segments.push(seg.data.to_vec());
                        }
                        // A passive segment isn't placed at instantiation; its bytes are kept for
                        // `memory.init` (lowered to const-stores of them — see `mem_init_op`).
                        wasmparser::DataKind::Passive => data_segments.push(seg.data.to_vec()),
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

    // Does any function use `memory.grow` / `table.grow`? Only then must the window reserve room for
    // the linear memory / function table to expand (and carry a runtime size cell) — so a non-growing
    // module (every existing kernel) transpiles to byte-identical IR and the same window. (`memory.size`
    // / `table.size` without growth is a constant.)
    let mut uses_grow = false;
    let mut uses_table_grow = false;
    for body in &bodies {
        for op in body.get_operators_reader()? {
            match op? {
                Operator::MemoryGrow { .. } => uses_grow = true,
                Operator::TableGrow { .. } => uses_table_grow = true,
                _ => {}
            }
        }
        if uses_grow && uses_table_grow {
            break;
        }
    }

    // Linear-memory layout. The linear memory sits at window offset 0 (so wasm address `a` is window
    // address `a`); the page count it may occupy is its initial size, or — when `memory.grow` is used —
    // up to its declared `maximum` (a default cap for unbounded memory, bounded by `MAX_GROW_PAGES`).
    // The window must hold that whole span (it is eagerly committed), so a grown page is reachable;
    // an access past the window traps under SVM's trap-confinement (matching wasm's OOB trap).
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
    // an OOB access would. Under trap-confinement an access past the whole window traps (as wasm's OOB
    // does), but one landing in this in-window globals/table region does not — the residual confinement
    // difference (wasm would trap it as past-linear-memory; co-locating the regions in one window keeps
    // it reachable). An exact-sized per-region window would remove even that, but is not needed here.
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
    // When `table.grow` is used the table region must span its growable maximum — the declared table
    // `maximum`, or a default cap for an unbounded table, bounded by `MAX_GROW_SLOTS` — so a grown slot
    // is reachable (eagerly committed, exactly the memory.grow span argument). A non-growing
    // module reserves only its initial slots and transpiles byte-identically to before.
    let table_max_slots = if uses_table_grow {
        table_max
            .unwrap_or(DEFAULT_MAX_GROW_SLOTS)
            .clamp(tsize.max(1), MAX_GROW_SLOTS.max(tsize))
    } else {
        tsize
    };
    let table_base = globals_end.div_ceil(4) * 4;
    let table_span_slots = if uses_table_grow {
        table_max_slots
    } else {
        tsize
    };
    if table_span_slots > 0 {
        let mut bytes = vec![0xFFu8; table_span_slots as usize * 4]; // sentinel = no/bad funcref
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
    let table_data_end = table_base + table_span_slots * 4;
    // Runtime current-size cell (slots), a 4-byte slot just past the table span — present only when
    // `table.grow` is used. `table.size`/`grow` load/store it, initialized to the initial slot count
    // via a `data` segment. (Without growth there is no cell and `table.size` is the constant `tsize`.)
    let table_size_cell_off = table_data_end;
    let table_end = if uses_table_grow {
        data.push(svm_ir::Data {
            offset: table_size_cell_off,
            readonly: false,
            bytes: (tsize as u32).to_le_bytes().to_vec(),
        });
        table_size_cell_off + 4
    } else {
        table_data_end
    };
    // §12 wasm threads: a 4-byte reserved unique-tid counter just past the function table (a fresh
    // window reads 0, so the first spawned tid is 1). Only consumed when `wasi:thread/spawn` is used.
    let tid_slot = table_end.div_ceil(4) * 4;

    // §12 wasm threads validation + the spawn shim's IR index (it is appended right after the defined
    // functions, so its index is `bodies.len()`). The shim adapts SVM's thread-entry ABI to the
    // `wasi_thread_start` export.
    let spawn_shim = bodies.len() as u32;
    if spawn_import.is_some() {
        // The host calls `wasi_thread_start(tid, start_arg)` on each spawned thread; require the export
        // and that its **wasm** type is `(i32, i32) -> ()`. (Capability imports dispatch through the
        // instance's slot bindings on every thread — nothing is threaded, so spawn needs no stash.)
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
    // Collect debug locations whenever any embedded DWARF is present (source lines *or* variables).
    let want_locs = !debug_blobs.is_empty();
    // Global `(code-relative offset, func, block, inst)` map for the DWARF→IR pc resolution below,
    // and per-(func, local) location records for `WindowVia` frame-pointer bases.
    let mut op_locs: Vec<(u32, u32, u32, u32)> = Vec::new();
    let mut local_locs: Vec<(u32, u32, u32, u32, u32)> = Vec::new(); // (func, local, block, inst, val)
    let mut funcs = Vec::with_capacity(bodies.len() + spawn_import.is_some() as usize);
    for (i, body) in bodies.into_iter().enumerate() {
        let ty = &types[func_type_idx[i] as usize];
        let func_idx = funcs.len() as u32; // defined funcs come first, so this is the IR index
        let (f, flocs, local_flocs) = lower_func(
            &ty.0,
            &ty.1,
            &types,
            &func_sigs,
            &globals_types,
            globals_base,
            table_base,
            tsize as u32,
            &body,
            mem64,
            &imports,
            &data_segments,
            &element_segments,
            MemGrow {
                uses_grow,
                size_cell_off,
                max_pages,
                initial_pages,
            },
            TableGrow {
                uses_grow: uses_table_grow,
                size_cell_off: table_size_cell_off,
                max_slots: table_max_slots,
            },
            threads,
            code_content_start,
            want_locs,
        )?;
        funcs.push(f);
        for (off, block, inst) in flocs {
            op_locs.push((off, func_idx, block, inst));
        }
        for (local, block, inst, val) in local_flocs {
            local_locs.push((func_idx, local, block, inst, val));
        }
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
        for (_, ir_idx) in exports.iter_mut() {
            if *ir_idx == start_ir {
                continue; // exporting the start function itself: don't double-run it
            }
            let params = funcs[*ir_idx as usize].params.clone();
            let results = funcs[*ir_idx as usize].results.clone();
            let wrap = build_start_wrapper(start_ir, *ir_idx, params, results);
            *ir_idx = funcs.len() as u32;
            funcs.push(wrap);
        }
    }

    // Our window is a power-of-two byte range (SVM's reserved domain rounds up to it); size it to hold
    // the linear memory (its full growable span) **and** the size cell + globals + function-table
    // regions. Both wasm and SVM bounds-check-and-trap on an out-of-range access; the residual
    // difference is only that an OOB linear-memory access landing in this window's globals/table region
    // stays reachable here (co-located) rather than trapping. Globals/table-only modules still need a window.
    let needed = table_end
        .max(globals_end)
        .max(after_mem)
        .max(mem_bytes)
        .max(if spawn_import.is_some() {
            // the reserved unique-tid counter
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
            // The import manifest (IMPORTS.md phase 3): one named slot per wasm function import,
            // for the numeric host-ABI convention and §7 named imports alike. The host binds each
            // slot at instantiation; the module bytes are never rewritten.
            imports: manifest,
            // wasm's exported functions map straight onto the first-class IR export table (name →
            // funcidx), so a transpiled module is name-addressable like its wasm source. (Also
            // returned out-of-band in `Transpiled::exports` for callers that want just the list.)
            exports: exports
                .iter()
                .map(|(name, func)| svm_ir::Export {
                    name: name.clone(),
                    func: *func,
                })
                .collect(),
            // wasm has no provider-side interface offers (IMPORTS.md §3.2) — nothing to transpile.
            impl_exports: vec![],
            types: manifest_types,
            // Debug info — map wasm's embedded DWARF `.debug_line` into the §6 waist (D-DBG-7) and
            // carry every `.debug_*` section through as a rich blob.
            debug_info: build_debug_info(debug_line.as_deref(), op_locs, local_locs, debug_blobs),
        },
        exports,
    })
}

/// Build the §6 debug-info waist from a wasm guest's embedded DWARF: map `.debug_line` rows onto IR
/// pcs, ingest `.debug_info` **source variables** (name, type, and a `WindowVia` location built from
/// the subprogram's frame-base wasm local), and carry every `.debug_*` section through as a rich
/// blob. Best-effort: returns `None` only if nothing was recovered. The verifier ignores it (§2a).
fn build_debug_info(
    debug_line: Option<&[u8]>,
    mut op_locs: Vec<(u32, u32, u32, u32)>,
    local_locs: Vec<(u32, u32, u32, u32, u32)>,
    blobs: Vec<svm_ir::ProducerBlob>,
) -> Option<DebugInfo> {
    op_locs.sort_by_key(|e| e.0);
    let mut files: Vec<String> = Vec::new();
    let mut locs: Vec<Loc> = Vec::new();
    // `(code address → source line)` rows, used to map a DWARF lexical block's PC range to a
    // source-line scope (§6 shadowing).
    let mut line_rows: Vec<(u64, u32)> = Vec::new();
    if let Some(prog) = debug_line.and_then(dwarf_line::parse) {
        line_rows = prog
            .rows
            .iter()
            .filter(|r| !r.end_sequence)
            .map(|r| (r.address, r.line))
            .collect();
        if prog.files.len() > 1 {
            // DWARF file indices are 1-based; flatten to a 0-based table for `Loc::file`.
            files = prog.files[1..].to_vec();
            for row in &prog.rows {
                if row.end_sequence || row.file == 0 || row.file as usize > files.len() {
                    continue;
                }
                // The line starts at the first recorded instruction at-or-after its address.
                let addr = row.address as u32;
                let idx = op_locs.partition_point(|e| e.0 < addr);
                let Some(&(_, func, block, inst)) = op_locs.get(idx) else {
                    continue;
                };
                // Coalesce: skip a row landing on the same pc as the previous one (the earlier/lower
                // address wins, matching `source_loc`'s nearest-preceding resolution).
                if locs
                    .last()
                    .is_some_and(|l| (l.func, l.block, l.inst) == (func, block, inst))
                {
                    continue;
                }
                locs.push(Loc {
                    func,
                    block,
                    inst,
                    file: (row.file - 1) as u32,
                    line: row.line,
                    col: row.col,
                });
            }
        }
    }
    // Source variables + function names from `.debug_info` (DEBUGGING.md W4/§6 — wasm ingest).
    let (types, vars, func_names) = ingest_variables(&blobs, &op_locs, &local_locs, &line_rows);

    if locs.is_empty() && vars.is_empty() && blobs.is_empty() {
        return None;
    }
    Some(DebugInfo {
        files,
        locs,
        types,
        vars,
        blobs,
        func_names,
    })
}

/// Ingest `.debug_info` source variables into `(types, vars)`: each DWARF variable's
/// `(frame_base_local + DW_OP_fbreg)` becomes a `VarLoc::WindowVia` whose base is that local's
/// recorded `SsaList`, and its `DW_TAG_base_type` becomes a structured `TypeRef`. A subprogram is
/// matched to its IR function by PC range (via `op_locs`). Empty if there is no parseable
/// `.debug_info`.
fn ingest_variables(
    blobs: &[svm_ir::ProducerBlob],
    op_locs: &[(u32, u32, u32, u32)],
    local_locs: &[(u32, u32, u32, u32, u32)],
    line_rows: &[(u64, u32)],
) -> (Vec<TypeDef>, Vec<VarInfo>, Vec<FuncName>) {
    let sec = |name: &str| {
        blobs
            .iter()
            .find(|b| b.producer == name)
            .map(|b| b.bytes.as_slice())
    };
    let (Some(info), Some(abbrev)) = (sec(".debug_info"), sec(".debug_abbrev")) else {
        return (Vec::new(), Vec::new(), Vec::new());
    };
    let Some(dw) = dwarf_info::parse(info, abbrev, sec(".debug_str").unwrap_or(&[])) else {
        return (Vec::new(), Vec::new(), Vec::new());
    };

    let mut types: Vec<TypeDef> = Vec::new();
    let mut type_ids: BTreeMap<u32, u32> = BTreeMap::new(); // DWARF type-DIE offset → svm TypeId
    let mut vars: Vec<VarInfo> = Vec::new();

    // §6 function names: each named `DW_TAG_subprogram` matched to its IR function by PC range. Not
    // gated on a frame base (unlike the var loop below) — a name is useful even without locals.
    let mut func_names: Vec<FuncName> = Vec::new();
    for sub in &dw.subs {
        if sub.name.is_empty() {
            continue;
        }
        if let Some(func) = func_for_pc_range(op_locs, sub.low_pc as u32, sub.high_pc as u32) {
            if !func_names.iter().any(|f: &FuncName| f.func == func) {
                func_names.push(FuncName {
                    func,
                    name: sub.name.clone(),
                });
            }
        }
    }

    for sub in &dw.subs {
        let Some(fb_local) = sub.frame_base_local else {
            continue; // only the `DW_OP_WASM_location <local>` frame base is supported
        };
        let Some(func) = func_for_pc_range(op_locs, sub.low_pc as u32, sub.high_pc as u32) else {
            continue;
        };
        // The frame-base local's location list (its SSA value per pc) becomes the `WindowVia` base.
        let base: Vec<SsaLoc> = local_locs
            .iter()
            .filter(|&&(f, l, _, _, _)| f == func && l == fb_local)
            .map(|&(_, _, block, inst, value)| SsaLoc { block, inst, value })
            .collect();
        if base.is_empty() {
            continue;
        }
        for v in &sub.vars {
            let type_id = intern_type(&dw, v.type_ref, &mut types, &mut type_ids);
            let ty = type_id
                .and_then(|id| types.get(id as usize))
                .map(type_render_name)
                .unwrap_or_else(|| "?".to_string());
            // §6 lexical scope: a var nested in a `DW_TAG_lexical_block` is in scope from its
            // declaration line to the block's last source line (mapped from the block's code range
            // via the line table). Directly in the subprogram ⇒ function-wide (`None`).
            let scope = v.scope_pc.and_then(|(lo, hi)| {
                let end = line_rows
                    .iter()
                    .filter(|&&(a, _)| a >= lo && a < hi)
                    .map(|&(_, l)| l)
                    .max()?;
                Some((v.decl_line, end))
            });
            vars.push(VarInfo {
                func,
                name: v.name.clone(),
                ty,
                loc: VarLoc::WindowVia {
                    base: base.clone(),
                    off: v.fbreg,
                },
                type_id,
                scope,
            });
        }
    }

    // Module-scoped globals (a CU-level `DW_TAG_variable` at a fixed `DW_OP_addr`): a wasm linear
    // address is the window address directly, so emit a `GLOBAL_SCOPE` `VarLoc::Fixed` var (visible
    // in every frame) — the §6 global primitive driven by the wasm DWARF producer.
    for g in &dw.globals {
        let type_id = intern_type(&dw, g.type_ref, &mut types, &mut type_ids);
        let ty = type_id
            .and_then(|id| types.get(id as usize))
            .map(type_render_name)
            .unwrap_or_else(|| "?".to_string());
        vars.push(VarInfo {
            func: svm_ir::GLOBAL_SCOPE,
            name: g.name.clone(),
            ty,
            loc: VarLoc::Fixed { addr: g.addr },
            type_id,
            scope: None,
        });
    }
    (types, vars, func_names)
}

/// The IR function whose code spans `[low, high)` (a DWARF subprogram's PC range), found via the
/// first recorded operator offset in that range. `op_locs` is sorted by offset.
fn func_for_pc_range(op_locs: &[(u32, u32, u32, u32)], low: u32, high: u32) -> Option<u32> {
    let idx = op_locs.partition_point(|e| e.0 < low);
    op_locs.get(idx).filter(|e| e.0 < high).map(|e| e.1)
}

/// Intern a DWARF type DIE (by offset) into the structured type table, recursively — base / pointer
/// / struct+union members / array — returning its `TypeId`. A `typedef` / cv-qualified type is
/// transparent (resolves to the underlying). Cycle-safe: reserves the id before recursing, so a
/// self-referential aggregate resolves to itself. `None` for an unmodeled / missing type.
fn intern_type(
    dw: &dwarf_info::DwarfInfo,
    type_ref: u32,
    types: &mut Vec<TypeDef>,
    type_ids: &mut BTreeMap<u32, u32>,
) -> Option<u32> {
    if let Some(&id) = type_ids.get(&type_ref) {
        return Some(id);
    }
    let dt = dw.types.get(&type_ref)?;
    // A transparent alias forwards straight to its underlying type.
    if let dwarf_info::DwarfType::Alias { underlying } = dt {
        let id = intern_type(dw, (*underlying)?, types, type_ids)?;
        type_ids.insert(type_ref, id);
        return Some(id);
    }
    // Reserve the id before recursing into members/pointees (cycle safety).
    let id = types.len() as u32;
    types.push(TypeDef::Opaque {
        name: String::new(),
        size: 0,
    });
    type_ids.insert(type_ref, id);
    let name_of =
        |types: &[TypeDef], i: Option<u32>| i.map(|i| type_render_name(&types[i as usize]));
    let resolved = match dt {
        dwarf_info::DwarfType::Base {
            name,
            encoding,
            size,
        } => TypeDef::Base {
            name: name.clone(),
            encoding: dwarf_encoding(*encoding),
            size: *size,
        },
        dwarf_info::DwarfType::Pointer { pointee, size } => {
            let pid = (*pointee).and_then(|p| intern_type(dw, p, types, type_ids));
            let pname = name_of(types, pid).unwrap_or_else(|| "void".to_string());
            TypeDef::Pointer {
                name: format!("{pname} *"),
                pointee: pid.unwrap_or(0),
                size: *size,
            }
        }
        dwarf_info::DwarfType::Aggregate {
            kw,
            name,
            size,
            members,
        } => {
            let fields = members
                .iter()
                .filter_map(|m| {
                    Some(Field {
                        name: m.name.clone(),
                        offset: m.offset,
                        ty: intern_type(dw, m.type_ref, types, type_ids)?,
                    })
                })
                .collect();
            let rname = if name.is_empty() {
                kw.to_string()
            } else {
                format!("{kw} {name}")
            };
            TypeDef::Aggregate {
                name: rname,
                size: *size,
                fields,
            }
        }
        dwarf_info::DwarfType::Array { elem, count } => {
            let eid = (*elem).and_then(|e| intern_type(dw, e, types, type_ids));
            let ename = name_of(types, eid).unwrap_or_else(|| "?".to_string());
            TypeDef::Array {
                name: format!("{ename}[{count}]"),
                elem: eid.unwrap_or(0),
                count: *count,
            }
        }
        dwarf_info::DwarfType::Alias { .. } => unreachable!("handled above"),
    };
    types[id as usize] = resolved;
    Some(id)
}

/// Map a DWARF `DW_AT_encoding` byte to the neutral [`svm_ir::Encoding`].
fn dwarf_encoding(e: u8) -> svm_ir::Encoding {
    match e {
        0x02 => svm_ir::Encoding::Bool,
        0x04 => svm_ir::Encoding::Float,
        0x07 | 0x08 => svm_ir::Encoding::Unsigned, // unsigned, unsigned_char
        _ => svm_ir::Encoding::Signed,             // signed, signed_char, address, …
    }
}

/// The render name of a structured type (each variant carries one).
fn type_render_name(t: &TypeDef) -> String {
    match t {
        TypeDef::Base { name, .. }
        | TypeDef::Pointer { name, .. }
        | TypeDef::Array { name, .. }
        | TypeDef::Aggregate { name, .. }
        | TypeDef::Opaque { name, .. } => name.clone(),
    }
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

/// An element segment's function list as **IR function indices** (the wasm function index space puts
/// imports first; the table holds only defined functions, so a funcref to an import has no IR index
/// and is rejected). `ref.func`-expression item lists aren't supported yet.
fn element_funcs(items: wasmparser::ElementItems, n_imp: u32) -> Result<Vec<u32>, Error> {
    match items {
        wasmparser::ElementItems::Functions(fns) => {
            let mut fs = Vec::new();
            for f in fns {
                let ir = f?
                    .checked_sub(n_imp)
                    .ok_or_else(|| Error::Unsupported("funcref to an imported function".into()))?;
                fs.push(ir);
            }
            Ok(fs)
        }
        wasmparser::ElementItems::Expressions(..) => unsup("element segment with const-expr items"),
    }
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

/// `table.size`/`table.grow` lowering parameters — the i32-slot twin of [`MemGrow`]. When `uses_grow`
/// the function table may expand and a runtime **size cell** (a 4-byte window slot at `size_cell_off`,
/// holding the current slot count) backs both ops; otherwise `table.size` is the constant initial size
/// and `grow` never appears.
#[derive(Clone, Copy)]
struct TableGrow {
    uses_grow: bool,
    size_cell_off: u64,
    max_slots: u64,
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
    /// loads the slot and feeds it to our `CallIndirect`; the §RT `table.*` ops read/write it.
    table_base: u64,
    /// The initial table size in slots — what `table.size` returns when the table isn't growable.
    table_size: u32,
    /// 64-bit linear memory (`memory64`): the address operand is already i64; otherwise it's an i32
    /// that must be zero-extended before our (i64-addressed) `load`/`store`.
    mem64: bool,
    /// `memory.size`/`memory.grow` lowering config (size-cell offset, page caps).
    mg: MemGrow,
    /// `table.size`/`table.grow` lowering config (size-cell offset, slot cap).
    tg: TableGrow,
    /// Per function-import (by import index): the `(manifest slot, signature)` its `call` lowers to
    /// as a `call.import` (IMPORTS.md phase 3 — the host binds the slot at instantiation; nothing is
    /// threaded through the guest). Empty when the module has no imports.
    imports: &'a [(u32, FuncType)],
    /// Number of imported functions: a wasm function index `< n_imp` is an import (→ `call.import`),
    /// else a defined function at IR index `idx - n_imp`.
    n_imp: usize,
    /// Every data segment's bytes by index (active + passive), for `memory.init`/`data.drop`.
    data_segments: &'a [Vec<u8>],
    /// Every element segment's IR-function-index list, by index — the `table.init` sources.
    element_segments: &'a [Vec<u32>],
    /// §12 wasm threads config — the `wasi:thread/spawn` lowering (the spawn import index, the shim,
    /// the unique-tid slot). `spawn_import` is `None` for non-threaded modules.
    threads: ThreadCfg,
    /// Debug-info collection (DEBUGGING.md §6/W4): when a `.debug_line` is present, each operator
    /// that emits an instruction records `(code-relative offset, block, inst index)` so the DWARF
    /// line rows can be mapped to IR pcs. The file offset where the code section content begins (to
    /// turn file-absolute operator offsets into code-relative ones); 0 ⇒ collection off.
    locs: Vec<(u32, u32, u32)>,
    /// Per-wasm-local location records `(local, block, inst, value)`: each change of a local's
    /// holding SSA value (block-entry re-threading + `local.set`/`tee`), the `SsaList` that supplies
    /// a `WindowVia` base for a DWARF frame-pointer local (DEBUGGING.md W4 — wasm variable ingest).
    local_locs: Vec<(u32, u32, u32, u32)>,
    code_content_start: usize,
    want_locs: bool,
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

    /// Width of the always-threaded prefix every block carries: all
    /// locals. The surviving operand stack follows.
    fn prefix_len(&self) -> usize {
        self.local_types.len()
    }
    /// The prefix value list (the locals) every branch threads, in `cur`'s value space.
    fn prefix_vals(&self) -> Vec<ValIdx> {
        self.locals.clone()
    }
    /// The prefix types (the local types).
    fn prefix_types(&self) -> Vec<ValType> {
        self.local_types.clone()
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

    /// Make `blk` current and rebind the locals + stack to its parameters. The prefix (the locals)
    /// occupies params `0..prefix_len()`; `stack_types` is the carried stack layout, whose values
    /// become the params after it.
    fn enter(&mut self, blk: usize, stack_types: &[ValType]) {
        self.cur = blk;
        let nl = self.local_types.len() as ValIdx;
        self.locals = (0..nl).collect();
        self.record_locals(); // each local re-enters as its block parameter (DEBUGGING.md W4)
        self.stack = stack_types
            .iter()
            .enumerate()
            .map(|(i, t)| (nl + i as ValIdx, *t))
            .collect();
        self.consts.clear(); // SSA values are block-local; constants don't carry across blocks
        self.reachable = true;
    }

    /// Set wasm local `i` to SSA value `v`, recording the change for the per-local location list.
    fn set_local(&mut self, i: usize, v: ValIdx) {
        self.locals[i] = v;
        if self.want_locs {
            let inst = self.blocks[self.cur].insts.len() as u32;
            self.local_locs.push((i as u32, self.cur as u32, inst, v));
        }
    }

    /// Record every local's current value at the start of the current block (inst 0) — the
    /// block-parameter re-threading that makes a local resolvable in each block.
    fn record_locals(&mut self) {
        if self.want_locs {
            for (i, &v) in self.locals.iter().enumerate() {
                self.local_locs.push((i as u32, self.cur as u32, 0, v));
            }
        }
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
        let nl = self.local_types.len() as ValIdx;
        self.locals = (0..nl).collect();
        self.record_locals();
        let stack_start = nl;
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
/// `wasi_thread_start` export (IR index `wts`), then returns 0. The data-SP `sp` is unused (svm-wasm
/// keeps the C stack in linear memory). Capability imports need no plumbing here: a spawned
/// thread's `call.import`s dispatch through the instance's slot bindings like any other (IMPORTS.md
/// phase 3 — the old per-spawn window handle stash is gone).
fn build_spawn_shim(wts: u32) -> Func {
    // values: v0=sp v1=arg | v2..  (a 0-result `call` appends no value)
    let mut insts: Vec<Inst> = Vec::new();
    let mut next: u32 = 2;
    // tid = arg >> 32 (i32); start_arg = arg's low 32 (i32).
    insts.push(Inst::ConstI64(32));
    let c32 = next;
    next += 1;
    insts.push(Inst::IntBin {
        ty: IntTy::I64,
        op: BinOp::ShrU,
        a: 1, // arg
        b: c32,
    });
    let arg_hi = next;
    next += 1;
    insts.push(Inst::Convert {
        op: ConvOp::WrapI64,
        a: arg_hi,
    });
    let tid = next;
    next += 1;
    insts.push(Inst::Convert {
        op: ConvOp::WrapI64,
        a: 1, // arg
    });
    let start_arg = next;
    next += 1;
    // call wasi_thread_start(tid, start_arg) — a 0-result call (appends no value).
    insts.push(Inst::Call {
        func: wts,
        args: vec![tid, start_arg],
    });
    insts.push(Inst::ConstI64(0));
    let ret = next;
    Func {
        params: vec![ValType::I64, ValType::I64],
        results: vec![ValType::I64],
        blocks: vec![Block {
            params: vec![ValType::I64, ValType::I64],
            insts,
            term: Terminator::Return(vec![ret]),
        }],
    }
}

/// Synthesize a start wrapper for an exported function (`(start $f)` support): a function with the
/// **same IR signature** as `target` that first calls `start` (`() -> ()`) and then `target` with
/// all params, returning its results. The embedder runs this in place of the bare export, so the
/// start function runs once before the entry; internal calls reach `target` directly and don't
/// re-run it.
fn build_start_wrapper(
    start_ir: u32,
    target: u32,
    params: Vec<ValType>,
    results: Vec<ValType>,
) -> Func {
    let nparams = params.len() as ValIdx;
    let insts = vec![
        // call start() (produces no value, so it doesn't advance the value counter).
        Inst::Call {
            func: start_ir,
            args: Vec::new(),
        },
        // call the real export with every param in order (values 0..nparams); its results land at
        // values nparams.. .
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
    table_size: u32,
    body: &wasmparser::FunctionBody,
    mem64: bool,
    imports: &[(u32, FuncType)],
    data_segments: &[Vec<u8>],
    element_segments: &[Vec<u32>],
    mg: MemGrow,
    tg: TableGrow,
    threads: ThreadCfg,
    code_content_start: usize,
    want_locs: bool,
) -> Result<(Func, OpLocs, LocalLocs), Error> {
    // Locals = params (with their incoming param values) then declared locals (default 0).
    let mut local_types: Vec<ValType> = params.to_vec();
    for decl in body.get_locals_reader()? {
        let (count, t) = decl?;
        let t = val_type(t)?;
        for _ in 0..count {
            local_types.push(t);
        }
    }

    // IMPORTS.md phase 3: capability imports dispatch through the module's manifest slots
    // (`call.import`), so functions carry exactly their wasm signature — no leading handle params.
    let n_imp = imports.len();
    let entry_params: Vec<ValType> = params.to_vec();
    let nparams = params.len() as ValIdx;

    let entry = BlockB {
        params: entry_params.clone(),
        insts: Vec::new(),
        next_val: entry_params.len() as ValIdx,
        term: None,
    };
    let mut lo = Lower {
        blocks: vec![entry],
        cur: 0,
        locals: (0..nparams).collect(),
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
        table_size,
        mem64,
        mg,
        tg,
        imports,
        n_imp,
        data_segments,
        element_segments,
        threads,
        locs: Vec::new(),
        local_locs: Vec::new(),
        code_content_start,
        want_locs,
    };
    // Initialize declared locals to zero (params already bound to block params), extending `locals`.
    for t in &local_types[params.len()..] {
        let v = match t {
            ValType::I32 => lo.emit(Inst::ConstI32(0)),
            ValType::I64 => lo.emit(Inst::ConstI64(0)),
            ValType::F32 => lo.emit(Inst::ConstF32(0)),
            ValType::F64 => lo.emit(Inst::ConstF64(0)),
            ValType::V128 => lo.emit(Inst::ConstV128([0; 16])),
            // WASM never declares a `ref` local (it's an svm-only GC reservation); treat as i64.
            ValType::Ref => lo.emit(Inst::ConstI64(0)),
            // `cap` is svm-only (§3.5); wasm never declares one. i32-width zero.
            ValType::Cap => lo.emit(Inst::ConstI32(0)),
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

    lo.record_locals(); // the entry block's initial local values (params + zero-inits)
    for item in body.get_operators_reader()?.into_iter_with_offsets() {
        let (op, off) = item?;
        // Record where this operator's first IR instruction lands, for the DWARF→IR pc mapping.
        // (Control ops emit a terminator, not an `insts` entry, so they add nothing here.)
        let (blk, n) = (lo.cur, lo.blocks[lo.cur].insts.len());
        lower_op(&mut lo, op, results)?;
        if lo.want_locs && lo.blocks[blk].insts.len() > n {
            if let Some(rel) = off.checked_sub(lo.code_content_start) {
                lo.locs.push((rel as u32, blk as u32, n as u32));
            }
        }
    }

    let locs = std::mem::take(&mut lo.locs);
    let local_locs = std::mem::take(&mut lo.local_locs);
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
    Ok((
        Func {
            params: entry_params,
            results: results.to_vec(),
            blocks,
        },
        locs,
        local_locs,
    ))
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
fn v_icmp(lo: &mut Lower, shape: VShape, op: VICmpOp) -> Result<(), Error> {
    let (b, _) = lo.pop()?;
    let (a, _) = lo.pop()?;
    let v = lo.emit(Inst::VIntCmp { shape, op, a, b });
    lo.push(v, ValType::V128);
    Ok(())
}
fn v_fcmp(lo: &mut Lower, shape: VShape, op: VFCmpOp) -> Result<(), Error> {
    let (b, _) = lo.pop()?;
    let (a, _) = lo.pop()?;
    let v = lo.emit(Inst::VFloatCmp { shape, op, a, b });
    lo.push(v, ValType::V128);
    Ok(())
}
fn v_shift(lo: &mut Lower, shape: VShape, op: VShiftOp) -> Result<(), Error> {
    // Stack: [vector, amount] — pop the i32 amount, then the v128.
    let (amt, _) = lo.pop()?;
    let (a, _) = lo.pop()?;
    let v = lo.emit(Inst::VShift { shape, op, a, amt });
    lo.push(v, ValType::V128);
    Ok(())
}
fn v_intun(lo: &mut Lower, shape: VShape, op: VIntUnOp) -> Result<(), Error> {
    let (a, _) = lo.pop()?;
    let v = lo.emit(Inst::VIntUn { shape, op, a });
    lo.push(v, ValType::V128);
    Ok(())
}
fn v_satbin(lo: &mut Lower, shape: VShape, op: VSatBinOp) -> Result<(), Error> {
    let (b, _) = lo.pop()?;
    let (a, _) = lo.pop()?;
    let v = lo.emit(Inst::VSatBin { shape, op, a, b });
    lo.push(v, ValType::V128);
    Ok(())
}
fn v_avgr(lo: &mut Lower, shape: VShape) -> Result<(), Error> {
    let (b, _) = lo.pop()?;
    let (a, _) = lo.pop()?;
    let v = lo.emit(Inst::VAvgr { shape, a, b });
    lo.push(v, ValType::V128);
    Ok(())
}
fn v_dot(lo: &mut Lower) -> Result<(), Error> {
    let (b, _) = lo.pop()?;
    let (a, _) = lo.pop()?;
    let v = lo.emit(Inst::VDot { a, b });
    lo.push(v, ValType::V128);
    Ok(())
}
/// `i16x8.relaxed_dot_i8x16_i7x16_s` — the deterministic signed-i8 dot (`Inst::VDotI8`).
fn v_dot_i8(lo: &mut Lower) -> Result<(), Error> {
    let (b, _) = lo.pop()?;
    let (a, _) = lo.pop()?;
    let v = lo.emit(Inst::VDotI8 { a, b });
    lo.push(v, ValType::V128);
    Ok(())
}
/// `i32x4.relaxed_dot_i8x16_i7x16_add_s` = `extadd_pairwise_s(dot_i8(a, b)) + c` — composed from the
/// i8 dot, the existing widening pairwise add, and an i32x4 add (the deterministic recipe).
fn v_dot_i8_add(lo: &mut Lower) -> Result<(), Error> {
    let (c, _) = lo.pop()?;
    let (b, _) = lo.pop()?;
    let (a, _) = lo.pop()?;
    let dot = lo.emit(Inst::VDotI8 { a, b });
    let wide = lo.emit(Inst::VExtAddPairwise {
        shape: VShape::I32x4,
        signed: true,
        a: dot,
    });
    let v = lo.emit(Inst::VIntBin {
        shape: VShape::I32x4,
        op: VIntBinOp::Add,
        a: wide,
        b: c,
    });
    lo.push(v, ValType::V128);
    Ok(())
}
fn v_extmul(lo: &mut Lower, shape: VShape, op: VWidenOp) -> Result<(), Error> {
    let (b, _) = lo.pop()?;
    let (a, _) = lo.pop()?;
    let v = lo.emit(Inst::VExtMul { shape, op, a, b });
    lo.push(v, ValType::V128);
    Ok(())
}
fn v_extadd(lo: &mut Lower, shape: VShape, signed: bool) -> Result<(), Error> {
    let (a, _) = lo.pop()?;
    let v = lo.emit(Inst::VExtAddPairwise { shape, signed, a });
    lo.push(v, ValType::V128);
    Ok(())
}
fn v_q15mulr(lo: &mut Lower) -> Result<(), Error> {
    let (b, _) = lo.pop()?;
    let (a, _) = lo.pop()?;
    let v = lo.emit(Inst::VQ15MulrSat { a, b });
    lo.push(v, ValType::V128);
    Ok(())
}
/// `<f>.relaxed_madd`/`relaxed_nmadd` (a ternary `[a, b, c]` op): a fused `±a·b + c`.
fn v_fma(lo: &mut Lower, shape: VShape, neg: bool) -> Result<(), Error> {
    let (c, _) = lo.pop()?;
    let (b, _) = lo.pop()?;
    let (a, _) = lo.pop()?;
    let v = lo.emit(Inst::VFma {
        shape,
        neg,
        a,
        b,
        c,
    });
    lo.push(v, ValType::V128);
    Ok(())
}
fn v_widen(lo: &mut Lower, shape: VShape, op: VWidenOp) -> Result<(), Error> {
    let (a, _) = lo.pop()?;
    let v = lo.emit(Inst::VWiden { shape, op, a });
    lo.push(v, ValType::V128);
    Ok(())
}
fn v_narrow(lo: &mut Lower, shape: VShape, op: VNarrowOp) -> Result<(), Error> {
    let (b, _) = lo.pop()?;
    let (a, _) = lo.pop()?;
    let v = lo.emit(Inst::VNarrow { shape, op, a, b });
    lo.push(v, ValType::V128);
    Ok(())
}
fn v_convert(lo: &mut Lower, op: VCvtOp) -> Result<(), Error> {
    let (a, _) = lo.pop()?;
    let v = lo.emit(Inst::VConvert { op, a });
    lo.push(v, ValType::V128);
    Ok(())
}
fn v_anytrue(lo: &mut Lower) -> Result<(), Error> {
    let (a, _) = lo.pop()?;
    let v = lo.emit(Inst::VAnyTrue { a });
    lo.push(v, ValType::I32);
    Ok(())
}
fn v_popcnt(lo: &mut Lower) -> Result<(), Error> {
    let (a, _) = lo.pop()?;
    let v = lo.emit(Inst::VPopcnt { a });
    lo.push(v, ValType::V128);
    Ok(())
}
fn v_alltrue(lo: &mut Lower, shape: VShape) -> Result<(), Error> {
    let (a, _) = lo.pop()?;
    let v = lo.emit(Inst::VAllTrue { shape, a });
    lo.push(v, ValType::I32);
    Ok(())
}
fn v_bitmask(lo: &mut Lower, shape: VShape) -> Result<(), Error> {
    let (a, _) = lo.pop()?;
    let v = lo.emit(Inst::VBitmask { shape, a });
    lo.push(v, ValType::I32);
    Ok(())
}
fn v_fbin(lo: &mut Lower, shape: VShape, op: VFloatBinOp) -> Result<(), Error> {
    let (b, _) = lo.pop()?;
    let (a, _) = lo.pop()?;
    let v = lo.emit(Inst::VFloatBin { shape, op, a, b });
    lo.push(v, ValType::V128);
    Ok(())
}
fn v_pminmax(lo: &mut Lower, shape: VShape, op: VPMinMaxOp) -> Result<(), Error> {
    let (b, _) = lo.pop()?;
    let (a, _) = lo.pop()?;
    let v = lo.emit(Inst::VPMinMax { shape, op, a, b });
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

// ---- SIMD memory variants (splat-load / load-extend / load-zero / load+store-lane) ----
// None need new IR: each composes a scalar `Load`/`Store` with `Splat`/`ReplaceLane`/`ExtractLane`/
// `VWiden` (the lane immediates are wasm-validated `< lanes`). Synthesized scalar accesses carry
// `align: 0` (advisory only; SVM does not trap on misalignment for non-atomic accesses).

/// `v128.loadN_splat`: load a scalar of width N and broadcast it across every lane of `shape`.
fn v_load_splat(lo: &mut Lower, shape: VShape, op: LoadOp, m: MemArg) -> Result<(), Error> {
    let addr = pop_addr(lo)?;
    let s = lo.emit(Inst::Load {
        op,
        addr,
        offset: m.offset,
        align: 0,
    });
    let v = lo.emit(Inst::Splat { shape, a: s });
    lo.push(v, ValType::V128);
    Ok(())
}

/// `v128.loadN_zero`: load a scalar into lane 0 of a zero vector (`load32_zero`/`load64_zero`).
fn v_load_zero(lo: &mut Lower, shape: VShape, op: LoadOp, m: MemArg) -> Result<(), Error> {
    let addr = pop_addr(lo)?;
    let s = lo.emit(Inst::Load {
        op,
        addr,
        offset: m.offset,
        align: 0,
    });
    let zero = lo.emit(Inst::ConstV128([0u8; 16]));
    let v = lo.emit(Inst::ReplaceLane {
        shape,
        lane: 0,
        a: zero,
        b: s,
    });
    lo.push(v, ValType::V128);
    Ok(())
}

/// `v128.loadAxB_{s,u}`: load 8 bytes into the low half of a v128, then widen-extend the low lanes to
/// the result `shape` (`op` picks sign/zero extension). The high 8 bytes are zeroed (unused by widen).
fn v_load_extend(lo: &mut Lower, shape: VShape, op: VWidenOp, m: MemArg) -> Result<(), Error> {
    let addr = pop_addr(lo)?;
    let bytes = lo.emit(Inst::Load {
        op: LoadOp::I64,
        addr,
        offset: m.offset,
        align: 0,
    });
    let zero = lo.emit(Inst::ConstV128([0u8; 16]));
    let packed = lo.emit(Inst::ReplaceLane {
        shape: VShape::I64x2,
        lane: 0,
        a: zero,
        b: bytes,
    });
    let v = lo.emit(Inst::VWiden {
        shape,
        op,
        a: packed,
    });
    lo.push(v, ValType::V128);
    Ok(())
}

/// `v128.loadN_lane`: load a scalar of width N and splice it into lane `lane` of the vector operand.
fn v_load_lane(
    lo: &mut Lower,
    shape: VShape,
    op: LoadOp,
    lane: u8,
    m: MemArg,
) -> Result<(), Error> {
    let (vec, _) = lo.pop()?; // the v128 (stack top), then the address below it
    let addr = pop_addr(lo)?;
    let s = lo.emit(Inst::Load {
        op,
        addr,
        offset: m.offset,
        align: 0,
    });
    let v = lo.emit(Inst::ReplaceLane {
        shape,
        lane,
        a: vec,
        b: s,
    });
    lo.push(v, ValType::V128);
    Ok(())
}

/// `v128.storeN_lane`: extract lane `lane` of the vector operand and store it as a width-N scalar.
fn v_store_lane(
    lo: &mut Lower,
    shape: VShape,
    op: StoreOp,
    lane: u8,
    m: MemArg,
) -> Result<(), Error> {
    let (vec, _) = lo.pop()?;
    let addr = pop_addr(lo)?;
    let s = lo.emit(Inst::ExtractLane {
        shape,
        lane,
        signed: false,
        a: vec,
    });
    lo.emit_void(Inst::Store {
        op,
        addr,
        value: s,
        offset: m.offset,
        align: 0,
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
        ValType::I64 | ValType::V128 | ValType::Ref => LoadOp::I64,
        ValType::Cap => LoadOp::I32,
        ValType::F32 => LoadOp::F32,
        ValType::F64 => LoadOp::F64,
    }
}
fn store_op(ty: ValType) -> StoreOp {
    match ty {
        ValType::I32 => StoreOp::I32,
        ValType::I64 | ValType::V128 | ValType::Ref => StoreOp::I64,
        ValType::Cap => StoreOp::I32,
        ValType::F32 => StoreOp::F32,
        ValType::F64 => StoreOp::F64,
    }
}

/// `call funcidx`: pop the callee's params (the last is on top), call it, push its results.
///
/// A wasm function index `< n_imp` is an **import**: lower to `call.import <slot>` on the module's
/// manifest (IMPORTS.md phase 3) — the host binds the slot at instantiation, nothing is threaded.
/// Otherwise it's a defined function at IR index `func - n_imp`.
fn call_op(lo: &mut Lower, func: u32) -> Result<(), Error> {
    // §12 wasm threads: a `call` to the `wasi:thread/spawn` import → the native `thread.spawn`.
    if lo.threads.spawn_import == Some(func) {
        return spawn_op(lo);
    }
    if (func as usize) < lo.n_imp {
        let (slot, sig) = lo.imports[func as usize].clone();
        let mut args = Vec::with_capacity(sig.params.len());
        for _ in 0..sig.params.len() {
            args.push(lo.pop()?.0);
        }
        args.reverse(); // stack top is the last argument
        let results = sig.results.clone();
        let inst = Inst::CallImport {
            import: slot,
            op: 0,
            sig,
            args,
        };
        let res = lo.emit_call(inst, results.len());
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
    let mut args = Vec::with_capacity(params.len());
    for _ in 0..params.len() {
        args.push(lo.pop()?.0);
    }
    args.reverse(); // stack top is the last argument
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
    // Capability imports need no per-spawn plumbing (IMPORTS.md phase 3): the spawned thread's
    // `call.import`s dispatch through the instance's slot bindings, which are host state shared
    // across vCPUs — the old handle stash is gone.
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
    let mut args = Vec::with_capacity(params.len());
    for _ in 0..params.len() {
        args.push(lo.pop()?.0);
    }
    args.reverse();
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
    let mut args = Vec::with_capacity(params.len());
    for _ in 0..params.len() {
        args.push(lo.pop()?.0);
    }
    args.reverse();
    let ty = FuncType { params, results };
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

// ---- narrow (sub-word) atomics: emulate via a 32-bit word CAS loop ----
//
// SVM IR atomics are 32/64-bit only, but wasm has 8/16-bit (and i64's 32-bit) atomic
// load/store/rmw/cmpxchg. A *naturally aligned* narrow access lies entirely within one 32-bit word
// (wasm requires natural alignment for atomics), so we operate on the **containing word** atomically:
//   - load: atomic word-load, then shift+mask the sub-word out (zero-extended).
//   - store/rmw/cmpxchg: a compare-and-swap loop on the word that splices the new sub-word in,
//     retrying until the word CAS lands (a concurrent change to an adjacent byte just retries).
// The i64 **32-bit** forms are word-sized (not sub-word): a native i32 atomic op, zero-extended.
//
// Note: like SVM's §1a OOB-masking (confine, don't trap), a *misaligned* narrow atomic is not
// trapped — a valid wasm module never emits one (the validator + toolchain guarantee alignment), so
// the word-CAS is always exact for the programs we accept.

/// The low-`w`-byte mask as an i32 (`w` ∈ {1,2}): `0xFF` / `0xFFFF`.
fn sub_mask(w: u32) -> i32 {
    ((1u64 << (w * 8)) - 1) as i32
}

/// Wrap an `i64` operand down to `i32` for the sub-word word-math (the sub-word is ≤16 bits, so the
/// low 32 bits suffice); an `i32` operand passes through.
fn wrap_i32(lo: &mut Lower, src: IntTy, v: ValIdx) -> ValIdx {
    match src {
        IntTy::I32 => v,
        IntTy::I64 => lo.emit(Inst::Convert {
            op: ConvOp::WrapI64,
            a: v,
        }),
    }
}

/// Zero-extend a sub-word `i32` result back to the op's destination width (`i64` ops wrap operands to
/// i32 for the word-math, then zero-extend the result).
fn zext_result(lo: &mut Lower, dst: IntTy, v: ValIdx) -> ValIdx {
    match dst {
        IntTy::I32 => v,
        IntTy::I64 => lo.emit(Inst::Convert {
            op: ConvOp::ExtendI32U,
            a: v,
        }),
    }
}

/// Pop the address (folding the memarg offset) and split it into `(word_addr = addr & ~3, shift =
/// (addr & 3) * 8)` — the inputs every sub-word atomic needs. `shift` is an i32 bit-count (0/8/16/24).
fn narrow_word(lo: &mut Lower, offset: u64) -> Result<(ValIdx, ValIdx), Error> {
    let addr = pop_atomic_addr(lo, offset)?; // i64, offset folded
    let three = lo.emit(Inst::ConstI64(3));
    let byte = lo.emit(Inst::IntBin {
        ty: IntTy::I64,
        op: BinOp::And,
        a: addr,
        b: three,
    });
    let neg4 = lo.emit(Inst::ConstI64(!3i64));
    let word_addr = lo.emit(Inst::IntBin {
        ty: IntTy::I64,
        op: BinOp::And,
        a: addr,
        b: neg4,
    });
    let byte32 = lo.emit(Inst::Convert {
        op: ConvOp::WrapI64,
        a: byte,
    });
    let eight = lo.emit(Inst::ConstI32(8));
    let shift = lo.emit(Inst::IntBin {
        ty: IntTy::I32,
        op: BinOp::Mul,
        a: byte32,
        b: eight,
    });
    Ok((word_addr, shift))
}

/// `<i>.atomic.load{8,16}_u` (sub-word) / `i64.atomic.load32_u` (word-sized): atomic load, then for a
/// sub-word width extract+zero-extend the lane.
fn narrow_atomic_load(lo: &mut Lower, dst: IntTy, w: u32, m: MemArg) -> Result<(), Error> {
    if w == 4 {
        // i64.atomic.load32_u: word-sized — native i32 atomic load, zero-extended to i64.
        let addr = pop_addr(lo)?;
        let word = lo.emit(Inst::AtomicLoad {
            ty: IntTy::I32,
            addr,
            offset: m.offset,
            order: Ordering::SeqCst,
        });
        let v = zext_result(lo, dst, word);
        lo.push(v, int_vt(dst));
        return Ok(());
    }
    let (word_addr, shift) = narrow_word(lo, m.offset)?;
    let word = lo.emit(Inst::AtomicLoad {
        ty: IntTy::I32,
        addr: word_addr,
        offset: 0,
        order: Ordering::SeqCst,
    });
    let shifted = lo.emit(Inst::IntBin {
        ty: IntTy::I32,
        op: BinOp::ShrU,
        a: word,
        b: shift,
    });
    let submask = lo.emit(Inst::ConstI32(sub_mask(w)));
    let sub = lo.emit(Inst::IntBin {
        ty: IntTy::I32,
        op: BinOp::And,
        a: shifted,
        b: submask,
    });
    let v = zext_result(lo, dst, sub);
    lo.push(v, int_vt(dst));
    Ok(())
}

/// Which sub-word write a CAS loop performs.
enum SubAtomic {
    Store,
    Rmw(AtomicRmwOp),
    Cmpxchg,
}

/// `<i>.atomic.{store,rmwN.*,cmpxchg}{8,16}` (sub-word, `w` ∈ {1,2}): a 32-bit word CAS loop that
/// splices the new sub-word into the containing word, retrying until the word CAS lands. Store pushes
/// nothing; rmw/cmpxchg push the (zero-extended) **old** sub-word.
fn narrow_sub_word(
    lo: &mut Lower,
    dst: IntTy,
    w: u32,
    kind: SubAtomic,
    m: MemArg,
) -> Result<(), Error> {
    let cmpxchg = matches!(kind, SubAtomic::Cmpxchg);
    let want_result = !matches!(kind, SubAtomic::Store);
    // Operands (wrapped to i32 for the word-math). cmpxchg: [expected, replacement]; else: [value]
    // (carried in `a`; `b` is unused but threaded uniformly to keep the loop layout fixed).
    let (a, b) = if cmpxchg {
        let (rep, _) = lo.pop()?;
        let (exp, _) = lo.pop()?;
        (wrap_i32(lo, dst, exp), wrap_i32(lo, dst, rep))
    } else {
        let (val, _) = lo.pop()?;
        let v = wrap_i32(lo, dst, val);
        (v, v)
    };
    let (word_addr, shift) = narrow_word(lo, m.offset)?;
    let old0 = lo.emit(Inst::AtomicLoad {
        ty: IntTy::I32,
        addr: word_addr,
        offset: 0,
        order: Ordering::SeqCst,
    });

    // The loop carries: prefix ++ operand-stack ++ [word_addr i64, shift i32, a i32, b i32, old i32].
    let below_t: Vec<ValType> = lo.stack.iter().map(|(_, t)| *t).collect();
    let below_v = lo.stack_vals();
    let extra_t = [
        ValType::I64,
        ValType::I32,
        ValType::I32,
        ValType::I32,
        ValType::I32,
    ];
    let body_sig = lo.synth_sig(&below_t, &extra_t);
    let body = lo.new_block(body_sig);
    let exit_extra: &[ValType] = if want_result { &[ValType::I32] } else { &[] };
    let exit_sig = lo.synth_sig(&below_t, exit_extra);
    let exit = lo.new_block(exit_sig);

    let args = lo.synth_args(&below_v, &[word_addr, shift, a, b, old0]);
    lo.set_term(Terminator::Br {
        target: body as u32,
        args,
    });

    // body: splice, CAS, retry on failure.
    let bx = lo.enter_synth(body, &below_t, 5);
    let (wa, sh, a, b, old_word) = (bx[0], bx[1], bx[2], bx[3], bx[4]);
    let submask = lo.emit(Inst::ConstI32(sub_mask(w)));
    let mask = lo.emit(Inst::IntBin {
        ty: IntTy::I32,
        op: BinOp::Shl,
        a: submask,
        b: sh,
    });
    let all_ones = lo.emit(Inst::ConstI32(-1));
    let notmask = lo.emit(Inst::IntBin {
        ty: IntTy::I32,
        op: BinOp::Xor,
        a: mask,
        b: all_ones,
    });
    // old_sub = (old_word >> shift) & submask
    let shifted = lo.emit(Inst::IntBin {
        ty: IntTy::I32,
        op: BinOp::ShrU,
        a: old_word,
        b: sh,
    });
    let old_sub = lo.emit(Inst::IntBin {
        ty: IntTy::I32,
        op: BinOp::And,
        a: shifted,
        b: submask,
    });
    // new_sub by kind.
    let new_sub = match &kind {
        SubAtomic::Store => a,
        SubAtomic::Rmw(AtomicRmwOp::Xchg) => a,
        SubAtomic::Rmw(op) => {
            let bin = match op {
                AtomicRmwOp::Add => BinOp::Add,
                AtomicRmwOp::Sub => BinOp::Sub,
                AtomicRmwOp::And => BinOp::And,
                AtomicRmwOp::Or => BinOp::Or,
                AtomicRmwOp::Xor => BinOp::Xor,
                AtomicRmwOp::Xchg => unreachable!("handled above"),
            };
            lo.emit(Inst::IntBin {
                ty: IntTy::I32,
                op: bin,
                a: old_sub,
                b: a,
            })
        }
        SubAtomic::Cmpxchg => {
            // (old_sub == expected) ? replacement : old_sub
            let eq = lo.emit(Inst::IntCmp {
                ty: IntTy::I32,
                op: CmpOp::Eq,
                a: old_sub,
                b: a,
            });
            lo.emit(Inst::Select {
                cond: eq,
                a: b,
                b: old_sub,
            })
        }
    };
    // new_word = (old_word & ~mask) | ((new_sub << shift) & mask)
    let cleared = lo.emit(Inst::IntBin {
        ty: IntTy::I32,
        op: BinOp::And,
        a: old_word,
        b: notmask,
    });
    let placed_raw = lo.emit(Inst::IntBin {
        ty: IntTy::I32,
        op: BinOp::Shl,
        a: new_sub,
        b: sh,
    });
    let placed = lo.emit(Inst::IntBin {
        ty: IntTy::I32,
        op: BinOp::And,
        a: placed_raw,
        b: mask,
    });
    let new_word = lo.emit(Inst::IntBin {
        ty: IntTy::I32,
        op: BinOp::Or,
        a: cleared,
        b: placed,
    });
    let prev = lo.emit(Inst::AtomicCmpxchg {
        ty: IntTy::I32,
        addr: wa,
        expected: old_word,
        replacement: new_word,
        offset: 0,
        order: Ordering::SeqCst,
    });
    let success = lo.emit(Inst::IntCmp {
        ty: IntTy::I32,
        op: CmpOp::Eq,
        a: prev,
        b: old_word,
    });
    let bv = lo.stack_vals();
    let then_args = if want_result {
        lo.synth_args(&bv, &[old_sub])
    } else {
        lo.synth_args(&bv, &[])
    };
    let else_args = lo.synth_args(&bv, &[wa, sh, a, b, prev]);
    lo.set_term(Terminator::BrIf {
        cond: success,
        then_blk: exit as u32,
        then_args,
        else_blk: body as u32,
        else_args,
    });

    // exit: restore the operand stack; push the old sub-word for rmw/cmpxchg.
    if want_result {
        let ex = lo.enter_synth(exit, &below_t, 1);
        let v = zext_result(lo, dst, ex[0]);
        lo.push(v, int_vt(dst));
    } else {
        lo.enter(exit, &below_t);
    }
    Ok(())
}

/// `<i>.atomic.store{8,16,32}`: sub-word (8/16) via the CAS loop; the i64 32-bit form is word-sized
/// (native i32 atomic store of the wrapped low word).
fn narrow_atomic_store(lo: &mut Lower, dst: IntTy, w: u32, m: MemArg) -> Result<(), Error> {
    if w == 4 {
        let (value, _) = lo.pop()?;
        let value = wrap_i32(lo, dst, value);
        let addr = pop_addr(lo)?;
        lo.emit_void(Inst::AtomicStore {
            ty: IntTy::I32,
            addr,
            value,
            offset: m.offset,
            order: Ordering::SeqCst,
        });
        return Ok(());
    }
    narrow_sub_word(lo, dst, w, SubAtomic::Store, m)
}

/// `<i>.atomic.rmw{8,16,32}.<op>`: sub-word (8/16) via the CAS loop; the i64 32-bit form is a native
/// i32 atomic rmw, zero-extended.
fn narrow_atomic_rmw(
    lo: &mut Lower,
    dst: IntTy,
    w: u32,
    op: AtomicRmwOp,
    m: MemArg,
) -> Result<(), Error> {
    if w == 4 {
        let (value, _) = lo.pop()?;
        let value = wrap_i32(lo, dst, value);
        let addr = pop_addr(lo)?;
        let old = lo.emit(Inst::AtomicRmw {
            ty: IntTy::I32,
            op,
            addr,
            value,
            offset: m.offset,
            order: Ordering::SeqCst,
        });
        let v = zext_result(lo, dst, old);
        lo.push(v, int_vt(dst));
        return Ok(());
    }
    narrow_sub_word(lo, dst, w, SubAtomic::Rmw(op), m)
}

/// `<i>.atomic.rmw{8,16,32}.cmpxchg`: sub-word (8/16) via the CAS loop; the i64 32-bit form is a
/// native i32 atomic cmpxchg, zero-extended.
fn narrow_atomic_cmpxchg(lo: &mut Lower, dst: IntTy, w: u32, m: MemArg) -> Result<(), Error> {
    if w == 4 {
        let (replacement, _) = lo.pop()?;
        let (expected, _) = lo.pop()?;
        let replacement = wrap_i32(lo, dst, replacement);
        let expected = wrap_i32(lo, dst, expected);
        let addr = pop_addr(lo)?;
        let old = lo.emit(Inst::AtomicCmpxchg {
            ty: IntTy::I32,
            addr,
            expected,
            replacement,
            offset: m.offset,
            order: Ordering::SeqCst,
        });
        let v = zext_result(lo, dst, old);
        lo.push(v, int_vt(dst));
        return Ok(());
    }
    narrow_sub_word(lo, dst, w, SubAtomic::Cmpxchg, m)
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

/// `memory.fill(dest, val, len)`: set `len` bytes at `dest` to byte `val`. Lowers to the D62
/// [`Inst::MemFill`] — one whole-span confinement then a `memset` (the JIT emits the platform libcall),
/// the same fast path the LLVM frontend's `llvm.memset` takes. No 64 KiB constant cap and no runtime
/// byte loop; `MemFill` keeps only the low byte of the `i32` `val`, matching wasm's fill-byte semantics.
fn mem_fill_op(lo: &mut Lower) -> Result<(), Error> {
    let (len_v, _) = lo.pop()?;
    let (val, _) = lo.pop()?; // the fill byte (low 8 bits of an i32)
    let dest = pop_addr(lo)?; // i64 window address
    let len = widen_to_i64(lo, len_v);
    lo.emit_void(Inst::MemFill {
        dst: dest,
        val,
        len,
    });
    Ok(())
}

/// `memory.init(data_index, dest, src, len)`: copy `len` bytes from data segment `data_index` —
/// whose bytes are known at transpile time — into the window at `dest`. The source range `src`/`len`
/// must be **constant** (the toolchain's `__wasm_init_memory` uses `src = 0`, `len = seg_len`), so the
/// exact bytes are known and unrolled into chunked const-stores at `dest` (a possibly-runtime
/// address); a non-constant `src`/`len` is fail-closed (`Unsupported`) — there is no runtime
/// passive-data store to read from. A static source out-of-bounds (`src + len > seg.len()`, which the
/// toolchain never emits) is a clean transpile error.
fn mem_init_op(lo: &mut Lower, data_index: u32) -> Result<(), Error> {
    let (len_v, _) = lo.pop()?;
    let (src_v, _) = lo.pop()?;
    let dest = pop_addr(lo)?;
    let (Some(src), Some(len)) = (const_bulk_len(lo, src_v), const_bulk_len(lo, len_v)) else {
        return unsup(
            "memory.init with a non-constant src/len (only the toolchain's constant-offset \
             initialization is supported; there is no runtime passive-data store)",
        );
    };
    let seg = lo.data_segments.get(data_index as usize).ok_or_else(|| {
        Error::Parse(format!(
            "memory.init references unknown data segment {data_index}"
        ))
    })?;
    let (src, len) = (src as usize, len as usize);
    let bytes = match src.checked_add(len).filter(|&e| e <= seg.len()) {
        Some(end) => seg[src..end].to_vec(), // clone to release the borrow on `lo`
        None => {
            return Err(Error::Parse(format!(
                "memory.init source [{src}, {src}+{len}) is out of segment {data_index}'s {} bytes",
                seg.len()
            )))
        }
    };
    init_unroll(lo, dest, &bytes);
    Ok(())
}

/// Unroll a known byte string into chunked const-stores at `dest` (the inverse of `copy_unroll`:
/// the source is compile-time bytes, not loads). Mirrors the active-data placement, but at runtime.
fn init_unroll(lo: &mut Lower, dest: ValIdx, bytes: &[u8]) {
    for (off, w) in chunk_plan(bytes.len() as u64) {
        let b = &bytes[off as usize..off as usize + w as usize];
        let value = match w {
            8 => lo.emit(Inst::ConstI64(
                u64::from_le_bytes(b.try_into().unwrap()) as i64
            )),
            4 => lo.emit(Inst::ConstI32(
                u32::from_le_bytes(b.try_into().unwrap()) as i32
            )),
            2 => lo.emit(Inst::ConstI32(
                u16::from_le_bytes(b.try_into().unwrap()) as i32
            )),
            _ => lo.emit(Inst::ConstI32(b[0] as i32)),
        };
        lo.emit_void(Inst::Store {
            op: store_w(w),
            addr: dest,
            value,
            offset: off,
            align: 0,
        });
    }
}

/// `memory.copy(dest, src, len)`: copy `len` bytes, overlap-safe (wasm `memory.copy` is memmove
/// semantics). Lowers to the D62 [`Inst::MemMove`] — one whole-span confinement then a `memmove` (the
/// JIT emits the platform libcall), matching the LLVM frontend's `llvm.memmove` fast path. No 64 KiB
/// constant cap and no runtime byte loop. (`table.copy` lowers to the same op — the table is window
/// memory.)
fn mem_copy_op(lo: &mut Lower) -> Result<(), Error> {
    let (len_v, _) = lo.pop()?;
    let src = pop_addr(lo)?;
    let dest = pop_addr(lo)?;
    let len = widen_to_i64(lo, len_v);
    lo.emit_void(Inst::MemMove {
        dst: dest,
        src,
        len,
    });
    Ok(())
}

// ---- §RT table access ops — the table is i32-granular window memory at `table_base`, so these are
// the i32-slot twins of the memory ops. OOB indices mask into the window (the §1a model, like memory).

/// The window byte offset of table slot `index` (an i32 on the stack): `index*4`. The caller passes
/// `lo.table_base` as the Load/Store `offset`.
fn table_elem_off(lo: &mut Lower, index: ValIdx) -> ValIdx {
    let idx64 = lo.emit(Inst::Convert {
        op: ConvOp::ExtendI32U,
        a: index,
    });
    let four = lo.emit(Inst::ConstI64(4));
    lo.emit(Inst::IntBin {
        ty: IntTy::I64,
        op: BinOp::Mul,
        a: idx64,
        b: four,
    })
}

/// `table.get $t`: load the i32 ref at `table_base + index*4`.
fn table_get_op(lo: &mut Lower) -> Result<(), Error> {
    let (index, _) = lo.pop()?;
    let off = table_elem_off(lo, index);
    let base = lo.table_base;
    let v = lo.emit(Inst::Load {
        op: LoadOp::I32,
        addr: off,
        offset: base,
        align: 2,
    });
    lo.push(v, ValType::I32);
    Ok(())
}

/// `table.set $t`: store an i32 ref at `table_base + index*4`. Stack: `[index, value]`.
fn table_set_op(lo: &mut Lower) -> Result<(), Error> {
    let (value, _) = lo.pop()?;
    let (index, _) = lo.pop()?;
    let off = table_elem_off(lo, index);
    let base = lo.table_base;
    lo.emit_void(Inst::Store {
        op: StoreOp::I32,
        addr: off,
        value,
        offset: base,
        align: 2,
    });
    Ok(())
}

/// `table.size $t`: the current size in slots. With growth it's a load of the runtime size cell;
/// without growth the size is constant (the declared `table_size`), so no cell is needed.
fn table_size_op(lo: &mut Lower) -> Result<(), Error> {
    let v = if lo.tg.uses_grow {
        let a = lo.emit(Inst::ConstI64(lo.tg.size_cell_off as i64));
        lo.emit(Inst::Load {
            op: LoadOp::I32,
            addr: a,
            offset: 0,
            align: 2,
        })
    } else {
        lo.emit(Inst::ConstI32(lo.table_size as i32))
    };
    lo.push(v, ValType::I32);
    Ok(())
}

/// `table.fill $t`: `for (i=0; i<n; i++) store_i32(table_base + (dest+i)*4, val)` — the i32-slot twin
/// of `memory.fill`, as a synthesized header/body/exit loop. Stack: `[dest, val, count]`.
fn table_fill_op(lo: &mut Lower) -> Result<(), Error> {
    let (count, _) = lo.pop()?;
    let (val, _) = lo.pop()?;
    let (dest, _) = lo.pop()?;
    let dest64 = lo.emit(Inst::Convert {
        op: ConvOp::ExtendI32U,
        a: dest,
    });
    let n = lo.emit(Inst::Convert {
        op: ConvOp::ExtendI32U,
        a: count,
    });
    table_fill_loop(lo, dest64, val, n);
    Ok(())
}

/// Synthesize the table fill loop `for (i=0; i<n; i++) store_i32(table_base + (dest+i)*4, val)`, with
/// `dest`/`n` as i64 slot indices and `val` an i32 ref. Shared by `table.fill` and `table.grow` (which
/// fills the newly-grown slots). The current operand stack is carried through the loop's block params.
fn table_fill_loop(lo: &mut Lower, dest64: ValIdx, val: ValIdx, n: ValIdx) {
    let base = lo.table_base;

    let below_t: Vec<ValType> = lo.stack.iter().map(|(_, t)| *t).collect();
    let below_v = lo.stack_vals();
    let extra = [ValType::I64, ValType::I32, ValType::I64, ValType::I64]; // dest, val, n, i
    let hsig = lo.synth_sig(&below_t, &extra);
    let header = lo.new_block(hsig.clone());
    let body = lo.new_block(hsig);
    let exit_sig = lo.synth_sig(&below_t, &[]);
    let exit = lo.new_block(exit_sig);

    let zero = lo.emit(Inst::ConstI64(0));
    let args = lo.synth_args(&below_v, &[dest64, val, n, zero]);
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

    // body: store_i32(base + (d+i)*4, v); i += 1; back to header.
    let bx = lo.enter_synth(body, &below_t, 4);
    let (d, v, nn, i) = (bx[0], bx[1], bx[2], bx[3]);
    let slot = lo.emit(Inst::IntBin {
        ty: IntTy::I64,
        op: BinOp::Add,
        a: d,
        b: i,
    });
    let four = lo.emit(Inst::ConstI64(4));
    let off = lo.emit(Inst::IntBin {
        ty: IntTy::I64,
        op: BinOp::Mul,
        a: slot,
        b: four,
    });
    lo.emit_void(Inst::Store {
        op: StoreOp::I32,
        addr: off,
        value: v,
        offset: base,
        align: 2,
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

    lo.enter(exit, &below_t);
}

/// The absolute window byte address of table slot `index` (an i32 on the stack): `table_base +
/// index*4`, as an i64 (the `dst`/`src` form `Inst::MemMove` consumes).
fn table_byte_addr(lo: &mut Lower, index: ValIdx, base: u64) -> ValIdx {
    let idx64 = lo.emit(Inst::Convert {
        op: ConvOp::ExtendI32U,
        a: index,
    });
    let four = lo.emit(Inst::ConstI64(4));
    let off = lo.emit(Inst::IntBin {
        ty: IntTy::I64,
        op: BinOp::Mul,
        a: idx64,
        b: four,
    });
    let base_c = lo.emit(Inst::ConstI64(base as i64));
    lo.emit(Inst::IntBin {
        ty: IntTy::I64,
        op: BinOp::Add,
        a: off,
        b: base_c,
    })
}

/// `table.copy`: copy `count` slots — `count*4` bytes — between table regions. Lowers to the D62
/// [`Inst::MemMove`] (overlap-safe, one whole-span confinement + a `memmove` libcall in the JIT), the
/// same fast path `memory.copy` takes — the table is just i32-granular window memory. Stack:
/// `[dest, src, count]`. (Single-table for now.)
fn table_copy_op(lo: &mut Lower) -> Result<(), Error> {
    let (count, _) = lo.pop()?;
    let (src_idx, _) = lo.pop()?;
    let (dest_idx, _) = lo.pop()?;
    let base = lo.table_base;
    let dest = table_byte_addr(lo, dest_idx, base);
    let src = table_byte_addr(lo, src_idx, base);
    // `count` is always an i32 slot count (both mem32 and mem64); the byte length is `count*4` as i64,
    // the width `MemMove` expects.
    let count64 = lo.emit(Inst::Convert {
        op: ConvOp::ExtendI32U,
        a: count,
    });
    let four = lo.emit(Inst::ConstI64(4));
    let byte_len = lo.emit(Inst::IntBin {
        ty: IntTy::I64,
        op: BinOp::Mul,
        a: count64,
        b: four,
    });
    lo.emit_void(Inst::MemMove {
        dst: dest,
        src,
        len: byte_len,
    });
    Ok(())
}

/// `table.init(elem_index, dest, src, count)`: copy `count` funcref indices from element segment
/// `elem_index` — known at transpile time — into the table at `dest` (a possibly-runtime slot index).
/// `src`/`count` must be **constant** (the same restriction as `memory.init`; there is no runtime
/// passive-element store), so the exact indices are unrolled into const i32-stores.
fn table_init_op(lo: &mut Lower, elem_index: u32) -> Result<(), Error> {
    let (count_v, _) = lo.pop()?;
    let (src_v, _) = lo.pop()?;
    let (dest_idx, _) = lo.pop()?;
    let (Some(src), Some(count)) = (const_bulk_len(lo, src_v), const_bulk_len(lo, count_v)) else {
        return unsup(
            "table.init with a non-constant src/count (only constant-range init is supported)",
        );
    };
    let seg = lo
        .element_segments
        .get(elem_index as usize)
        .ok_or_else(|| {
            Error::Parse(format!(
                "table.init references unknown element segment {elem_index}"
            ))
        })?;
    let (src, count) = (src as usize, count as usize);
    let end = src
        .checked_add(count)
        .filter(|&e| e <= seg.len())
        .ok_or_else(|| {
            Error::Parse(format!(
                "table.init source [{src}, {src}+{count}) is out of segment {elem_index}'s {} entries",
                seg.len()
            ))
        })?;
    let funcs = seg[src..end].to_vec(); // clone to release the borrow on `lo`
    let base = lo.table_base;
    let dest = table_byte_addr(lo, dest_idx, base); // absolute address of slot `dest_idx`
    for (k, &f) in funcs.iter().enumerate() {
        let value = lo.emit(Inst::ConstI32(f as i32));
        lo.emit_void(Inst::Store {
            op: StoreOp::I32,
            addr: dest,
            value,
            offset: (k * 4) as u64,
            align: 2,
        });
    }
    Ok(())
}

/// `table.grow(init, delta)`: extend the table by `delta` slots — filling the new slots with the `init`
/// ref — and return the previous size (or `-1` if it would exceed `max_slots`). The i32-slot twin of
/// `memory.grow`: branch-free size-cell update (slot math in i64, the delta being unsigned) via
/// `select`, then the newly-grown slots `[old, new)` are filled with a `table_fill_loop` (empty on
/// failure, since `new == old`). Only emitted when `uses_grow`, so the size cell exists.
fn table_grow_op(lo: &mut Lower) -> Result<(), Error> {
    let (delta, _) = lo.pop()?;
    let (init, _) = lo.pop()?;
    let cell = lo.emit(Inst::ConstI64(lo.tg.size_cell_off as i64));
    let old = lo.emit(Inst::Load {
        op: LoadOp::I32,
        addr: cell,
        offset: 0,
        align: 2,
    });
    // Overflow-safe in i64 (a near-`u32::MAX` delta must not wrap past the cap into a "fits").
    let zext = |lo: &mut Lower, v| {
        lo.emit(Inst::Convert {
            op: ConvOp::ExtendI32U,
            a: v,
        })
    };
    let old64 = zext(lo, old);
    let delta64 = zext(lo, delta);
    let new64 = lo.emit(Inst::IntBin {
        ty: IntTy::I64,
        op: BinOp::Add,
        a: old64,
        b: delta64,
    });
    let maxc = lo.emit(Inst::ConstI64(lo.tg.max_slots as i64));
    let ok = lo.emit(Inst::IntCmp {
        ty: IntTy::I64,
        op: CmpOp::LeU,
        a: new64,
        b: maxc,
    });
    let new32 = lo.emit(Inst::Convert {
        op: ConvOp::WrapI64,
        a: new64,
    });
    // Store `new` on success, unchanged `old` on failure (a no-op write); reuse `cell` as the address.
    let stored = lo.emit(Inst::Select {
        cond: ok,
        a: new32,
        b: old,
    });
    lo.emit_void(Inst::Store {
        op: StoreOp::I32,
        addr: cell,
        value: stored,
        offset: 0,
        align: 2,
    });
    let negone = lo.emit(Inst::ConstI32(-1));
    let result = lo.emit(Inst::Select {
        cond: ok,
        a: old,
        b: negone,
    });
    // Push the result *before* the fill loop so it rides the loop's carried operand stack (a value
    // defined here is otherwise unreachable in the loop's exit block). Then fill `[old, stored)` with
    // `init` — `stored - old` slots, which is `delta` on success and `0` (an empty loop) on failure.
    lo.push(result, ValType::I32);
    let stored64 = zext(lo, stored);
    let count = lo.emit(Inst::IntBin {
        ty: IntTy::I64,
        op: BinOp::Sub,
        a: stored64,
        b: old64,
    });
    table_fill_loop(lo, old64, init, count);
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
            lo.set_local(local_index as usize, v);
        }
        O::LocalTee { local_index } => {
            let (v, _) = *lo
                .stack
                .last()
                .ok_or_else(|| Error::Parse("tee on empty stack".into()))?;
            lo.set_local(local_index as usize, v);
        }
        O::Select => {
            let (c, _) = lo.pop()?;
            let (b, _) = lo.pop()?;
            let (a, t) = lo.pop()?;
            let v = lo.emit(Inst::Select { cond: c, a, b });
            lo.push(v, t);
        }
        // §RT typed select `select (result t)` — the reference-types-era form that names its type.
        // Operationally identical to plain `select`; the named type is just what the operands carry.
        O::TypedSelect { ty } => {
            let t = val_type(ty)?;
            let (c, _) = lo.pop()?;
            let (b, _) = lo.pop()?;
            let (a, _) = lo.pop()?;
            let v = lo.emit(Inst::Select { cond: c, a, b });
            lo.push(v, t);
        }
        // ---- §RT reference instructions (both ref types are an i32 index in SVM) ----
        O::RefNull { .. } => {
            let v = lo.emit(Inst::ConstI32(REF_NULL));
            lo.consts.insert(v, REF_NULL as i64);
            lo.push(v, ValType::I32);
        }
        O::RefIsNull => {
            let (a, _) = lo.pop()?;
            let null = lo.emit(Inst::ConstI32(REF_NULL));
            let v = lo.emit(Inst::IntCmp {
                ty: IntTy::I32,
                op: CmpOp::Eq,
                a,
                b: null,
            });
            lo.push(v, ValType::I32);
        }
        // `ref.func $f` → the function's §3c index (the IR function index). A funcref to an *import*
        // has no IR index (imports lower to `cap.call`, not a defined function), so it is rejected —
        // the same limit as a funcref element segment.
        O::RefFunc { function_index } => {
            let ir = function_index
                .checked_sub(lo.n_imp as u32)
                .ok_or_else(|| Error::Unsupported("ref.func to an imported function".into()))?;
            let v = lo.emit(Inst::ConstI32(ir as i32));
            lo.consts.insert(v, ir as i64);
            lo.push(v, ValType::I32);
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
        // ---- narrow (8/16-bit, and i64's 32-bit) atomics: word-CAS emulation ----
        O::I32AtomicLoad8U { memarg } => narrow_atomic_load(lo, IntTy::I32, 1, memarg)?,
        O::I32AtomicLoad16U { memarg } => narrow_atomic_load(lo, IntTy::I32, 2, memarg)?,
        O::I64AtomicLoad8U { memarg } => narrow_atomic_load(lo, IntTy::I64, 1, memarg)?,
        O::I64AtomicLoad16U { memarg } => narrow_atomic_load(lo, IntTy::I64, 2, memarg)?,
        O::I64AtomicLoad32U { memarg } => narrow_atomic_load(lo, IntTy::I64, 4, memarg)?,
        O::I32AtomicStore8 { memarg } => narrow_atomic_store(lo, IntTy::I32, 1, memarg)?,
        O::I32AtomicStore16 { memarg } => narrow_atomic_store(lo, IntTy::I32, 2, memarg)?,
        O::I64AtomicStore8 { memarg } => narrow_atomic_store(lo, IntTy::I64, 1, memarg)?,
        O::I64AtomicStore16 { memarg } => narrow_atomic_store(lo, IntTy::I64, 2, memarg)?,
        O::I64AtomicStore32 { memarg } => narrow_atomic_store(lo, IntTy::I64, 4, memarg)?,
        O::I32AtomicRmw8AddU { memarg } => {
            narrow_atomic_rmw(lo, IntTy::I32, 1, AtomicRmwOp::Add, memarg)?
        }
        O::I32AtomicRmw16AddU { memarg } => {
            narrow_atomic_rmw(lo, IntTy::I32, 2, AtomicRmwOp::Add, memarg)?
        }
        O::I64AtomicRmw8AddU { memarg } => {
            narrow_atomic_rmw(lo, IntTy::I64, 1, AtomicRmwOp::Add, memarg)?
        }
        O::I64AtomicRmw16AddU { memarg } => {
            narrow_atomic_rmw(lo, IntTy::I64, 2, AtomicRmwOp::Add, memarg)?
        }
        O::I64AtomicRmw32AddU { memarg } => {
            narrow_atomic_rmw(lo, IntTy::I64, 4, AtomicRmwOp::Add, memarg)?
        }
        O::I32AtomicRmw8SubU { memarg } => {
            narrow_atomic_rmw(lo, IntTy::I32, 1, AtomicRmwOp::Sub, memarg)?
        }
        O::I32AtomicRmw16SubU { memarg } => {
            narrow_atomic_rmw(lo, IntTy::I32, 2, AtomicRmwOp::Sub, memarg)?
        }
        O::I64AtomicRmw8SubU { memarg } => {
            narrow_atomic_rmw(lo, IntTy::I64, 1, AtomicRmwOp::Sub, memarg)?
        }
        O::I64AtomicRmw16SubU { memarg } => {
            narrow_atomic_rmw(lo, IntTy::I64, 2, AtomicRmwOp::Sub, memarg)?
        }
        O::I64AtomicRmw32SubU { memarg } => {
            narrow_atomic_rmw(lo, IntTy::I64, 4, AtomicRmwOp::Sub, memarg)?
        }
        O::I32AtomicRmw8AndU { memarg } => {
            narrow_atomic_rmw(lo, IntTy::I32, 1, AtomicRmwOp::And, memarg)?
        }
        O::I32AtomicRmw16AndU { memarg } => {
            narrow_atomic_rmw(lo, IntTy::I32, 2, AtomicRmwOp::And, memarg)?
        }
        O::I64AtomicRmw8AndU { memarg } => {
            narrow_atomic_rmw(lo, IntTy::I64, 1, AtomicRmwOp::And, memarg)?
        }
        O::I64AtomicRmw16AndU { memarg } => {
            narrow_atomic_rmw(lo, IntTy::I64, 2, AtomicRmwOp::And, memarg)?
        }
        O::I64AtomicRmw32AndU { memarg } => {
            narrow_atomic_rmw(lo, IntTy::I64, 4, AtomicRmwOp::And, memarg)?
        }
        O::I32AtomicRmw8OrU { memarg } => {
            narrow_atomic_rmw(lo, IntTy::I32, 1, AtomicRmwOp::Or, memarg)?
        }
        O::I32AtomicRmw16OrU { memarg } => {
            narrow_atomic_rmw(lo, IntTy::I32, 2, AtomicRmwOp::Or, memarg)?
        }
        O::I64AtomicRmw8OrU { memarg } => {
            narrow_atomic_rmw(lo, IntTy::I64, 1, AtomicRmwOp::Or, memarg)?
        }
        O::I64AtomicRmw16OrU { memarg } => {
            narrow_atomic_rmw(lo, IntTy::I64, 2, AtomicRmwOp::Or, memarg)?
        }
        O::I64AtomicRmw32OrU { memarg } => {
            narrow_atomic_rmw(lo, IntTy::I64, 4, AtomicRmwOp::Or, memarg)?
        }
        O::I32AtomicRmw8XorU { memarg } => {
            narrow_atomic_rmw(lo, IntTy::I32, 1, AtomicRmwOp::Xor, memarg)?
        }
        O::I32AtomicRmw16XorU { memarg } => {
            narrow_atomic_rmw(lo, IntTy::I32, 2, AtomicRmwOp::Xor, memarg)?
        }
        O::I64AtomicRmw8XorU { memarg } => {
            narrow_atomic_rmw(lo, IntTy::I64, 1, AtomicRmwOp::Xor, memarg)?
        }
        O::I64AtomicRmw16XorU { memarg } => {
            narrow_atomic_rmw(lo, IntTy::I64, 2, AtomicRmwOp::Xor, memarg)?
        }
        O::I64AtomicRmw32XorU { memarg } => {
            narrow_atomic_rmw(lo, IntTy::I64, 4, AtomicRmwOp::Xor, memarg)?
        }
        O::I32AtomicRmw8XchgU { memarg } => {
            narrow_atomic_rmw(lo, IntTy::I32, 1, AtomicRmwOp::Xchg, memarg)?
        }
        O::I32AtomicRmw16XchgU { memarg } => {
            narrow_atomic_rmw(lo, IntTy::I32, 2, AtomicRmwOp::Xchg, memarg)?
        }
        O::I64AtomicRmw8XchgU { memarg } => {
            narrow_atomic_rmw(lo, IntTy::I64, 1, AtomicRmwOp::Xchg, memarg)?
        }
        O::I64AtomicRmw16XchgU { memarg } => {
            narrow_atomic_rmw(lo, IntTy::I64, 2, AtomicRmwOp::Xchg, memarg)?
        }
        O::I64AtomicRmw32XchgU { memarg } => {
            narrow_atomic_rmw(lo, IntTy::I64, 4, AtomicRmwOp::Xchg, memarg)?
        }
        O::I32AtomicRmw8CmpxchgU { memarg } => narrow_atomic_cmpxchg(lo, IntTy::I32, 1, memarg)?,
        O::I32AtomicRmw16CmpxchgU { memarg } => narrow_atomic_cmpxchg(lo, IntTy::I32, 2, memarg)?,
        O::I64AtomicRmw8CmpxchgU { memarg } => narrow_atomic_cmpxchg(lo, IntTy::I64, 1, memarg)?,
        O::I64AtomicRmw16CmpxchgU { memarg } => narrow_atomic_cmpxchg(lo, IntTy::I64, 2, memarg)?,
        O::I64AtomicRmw32CmpxchgU { memarg } => narrow_atomic_cmpxchg(lo, IntTy::I64, 4, memarg)?,
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
        O::MemoryInit { data_index, mem } => {
            if mem != 0 {
                return unsup("memory.init on a non-zero memory");
            }
            mem_init_op(lo, data_index)?;
        }
        O::DataDrop { data_index } => {
            if (data_index as usize) >= lo.data_segments.len() {
                return unsup("data.drop of an unknown data segment");
            }
            // No-op: a passive segment's bytes are inlined at each `memory.init` site (so there is
            // nothing to free), and the toolchain's `__wasm_init_memory` drops only *after* its inits.
            // (A `memory.init` of an already-dropped segment would diverge — it would still copy — but
            // toolchain output never re-inits after a drop; the §1a "not the spec suite" stance.)
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
        // ---- §RT table access ops (table 0 only; multiple tables are a follow-up) ----
        O::TableGet { table } => {
            if table != 0 {
                return unsup("table.get on a non-zero table");
            }
            table_get_op(lo)?;
        }
        O::TableSet { table } => {
            if table != 0 {
                return unsup("table.set on a non-zero table");
            }
            table_set_op(lo)?;
        }
        O::TableSize { table } => {
            if table != 0 {
                return unsup("table.size on a non-zero table");
            }
            table_size_op(lo)?;
        }
        O::TableFill { table } => {
            if table != 0 {
                return unsup("table.fill on a non-zero table");
            }
            table_fill_op(lo)?;
        }
        O::TableGrow { table } => {
            if table != 0 {
                return unsup("table.grow on a non-zero table");
            }
            table_grow_op(lo)?;
        }
        O::TableCopy {
            dst_table,
            src_table,
        } => {
            if dst_table != 0 || src_table != 0 {
                return unsup("table.copy on a non-zero table");
            }
            table_copy_op(lo)?;
        }
        O::TableInit { elem_index, table } => {
            if table != 0 {
                return unsup("table.init on a non-zero table");
            }
            table_init_op(lo, elem_index)?;
        }
        // The element segment's indices are inlined at each `table.init` site, so a drop is a no-op
        // (like `data.drop`).
        O::ElemDrop { .. } => {}
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
        // SIMD memory variants — splat-load, load-extend, load-zero, load/store-lane.
        O::V128Load8Splat { memarg } => v_load_splat(lo, VShape::I8x16, LoadOp::I32_8U, memarg)?,
        O::V128Load16Splat { memarg } => v_load_splat(lo, VShape::I16x8, LoadOp::I32_16U, memarg)?,
        O::V128Load32Splat { memarg } => v_load_splat(lo, VShape::I32x4, LoadOp::I32, memarg)?,
        O::V128Load64Splat { memarg } => v_load_splat(lo, VShape::I64x2, LoadOp::I64, memarg)?,
        O::V128Load32Zero { memarg } => v_load_zero(lo, VShape::I32x4, LoadOp::I32, memarg)?,
        O::V128Load64Zero { memarg } => v_load_zero(lo, VShape::I64x2, LoadOp::I64, memarg)?,
        O::V128Load8x8S { memarg } => v_load_extend(lo, VShape::I16x8, VWidenOp::LowS, memarg)?,
        O::V128Load8x8U { memarg } => v_load_extend(lo, VShape::I16x8, VWidenOp::LowU, memarg)?,
        O::V128Load16x4S { memarg } => v_load_extend(lo, VShape::I32x4, VWidenOp::LowS, memarg)?,
        O::V128Load16x4U { memarg } => v_load_extend(lo, VShape::I32x4, VWidenOp::LowU, memarg)?,
        O::V128Load32x2S { memarg } => v_load_extend(lo, VShape::I64x2, VWidenOp::LowS, memarg)?,
        O::V128Load32x2U { memarg } => v_load_extend(lo, VShape::I64x2, VWidenOp::LowU, memarg)?,
        O::V128Load8Lane { memarg, lane } => {
            v_load_lane(lo, VShape::I8x16, LoadOp::I32_8U, lane, memarg)?
        }
        O::V128Load16Lane { memarg, lane } => {
            v_load_lane(lo, VShape::I16x8, LoadOp::I32_16U, lane, memarg)?
        }
        O::V128Load32Lane { memarg, lane } => {
            v_load_lane(lo, VShape::I32x4, LoadOp::I32, lane, memarg)?
        }
        O::V128Load64Lane { memarg, lane } => {
            v_load_lane(lo, VShape::I64x2, LoadOp::I64, lane, memarg)?
        }
        O::V128Store8Lane { memarg, lane } => {
            v_store_lane(lo, VShape::I8x16, StoreOp::I32_8, lane, memarg)?
        }
        O::V128Store16Lane { memarg, lane } => {
            v_store_lane(lo, VShape::I16x8, StoreOp::I32_16, lane, memarg)?
        }
        O::V128Store32Lane { memarg, lane } => {
            v_store_lane(lo, VShape::I32x4, StoreOp::I32, lane, memarg)?
        }
        O::V128Store64Lane { memarg, lane } => {
            v_store_lane(lo, VShape::I64x2, StoreOp::I64, lane, memarg)?
        }
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
        // integer lane comparisons → a per-lane all-ones/all-zeros mask. `i64x2` has signed-only
        // ordering in the wasm spec (no unsigned lt/gt/le/ge); `eq`/`ne` exist for every shape.
        O::I8x16Eq => v_icmp(lo, VShape::I8x16, VICmpOp::Eq)?,
        O::I8x16Ne => v_icmp(lo, VShape::I8x16, VICmpOp::Ne)?,
        O::I8x16LtS => v_icmp(lo, VShape::I8x16, VICmpOp::LtS)?,
        O::I8x16LtU => v_icmp(lo, VShape::I8x16, VICmpOp::LtU)?,
        O::I8x16GtS => v_icmp(lo, VShape::I8x16, VICmpOp::GtS)?,
        O::I8x16GtU => v_icmp(lo, VShape::I8x16, VICmpOp::GtU)?,
        O::I8x16LeS => v_icmp(lo, VShape::I8x16, VICmpOp::LeS)?,
        O::I8x16LeU => v_icmp(lo, VShape::I8x16, VICmpOp::LeU)?,
        O::I8x16GeS => v_icmp(lo, VShape::I8x16, VICmpOp::GeS)?,
        O::I8x16GeU => v_icmp(lo, VShape::I8x16, VICmpOp::GeU)?,
        O::I16x8Eq => v_icmp(lo, VShape::I16x8, VICmpOp::Eq)?,
        O::I16x8Ne => v_icmp(lo, VShape::I16x8, VICmpOp::Ne)?,
        O::I16x8LtS => v_icmp(lo, VShape::I16x8, VICmpOp::LtS)?,
        O::I16x8LtU => v_icmp(lo, VShape::I16x8, VICmpOp::LtU)?,
        O::I16x8GtS => v_icmp(lo, VShape::I16x8, VICmpOp::GtS)?,
        O::I16x8GtU => v_icmp(lo, VShape::I16x8, VICmpOp::GtU)?,
        O::I16x8LeS => v_icmp(lo, VShape::I16x8, VICmpOp::LeS)?,
        O::I16x8LeU => v_icmp(lo, VShape::I16x8, VICmpOp::LeU)?,
        O::I16x8GeS => v_icmp(lo, VShape::I16x8, VICmpOp::GeS)?,
        O::I16x8GeU => v_icmp(lo, VShape::I16x8, VICmpOp::GeU)?,
        O::I32x4Eq => v_icmp(lo, VShape::I32x4, VICmpOp::Eq)?,
        O::I32x4Ne => v_icmp(lo, VShape::I32x4, VICmpOp::Ne)?,
        O::I32x4LtS => v_icmp(lo, VShape::I32x4, VICmpOp::LtS)?,
        O::I32x4LtU => v_icmp(lo, VShape::I32x4, VICmpOp::LtU)?,
        O::I32x4GtS => v_icmp(lo, VShape::I32x4, VICmpOp::GtS)?,
        O::I32x4GtU => v_icmp(lo, VShape::I32x4, VICmpOp::GtU)?,
        O::I32x4LeS => v_icmp(lo, VShape::I32x4, VICmpOp::LeS)?,
        O::I32x4LeU => v_icmp(lo, VShape::I32x4, VICmpOp::LeU)?,
        O::I32x4GeS => v_icmp(lo, VShape::I32x4, VICmpOp::GeS)?,
        O::I32x4GeU => v_icmp(lo, VShape::I32x4, VICmpOp::GeU)?,
        O::I64x2Eq => v_icmp(lo, VShape::I64x2, VICmpOp::Eq)?,
        O::I64x2Ne => v_icmp(lo, VShape::I64x2, VICmpOp::Ne)?,
        O::I64x2LtS => v_icmp(lo, VShape::I64x2, VICmpOp::LtS)?,
        O::I64x2GtS => v_icmp(lo, VShape::I64x2, VICmpOp::GtS)?,
        O::I64x2LeS => v_icmp(lo, VShape::I64x2, VICmpOp::LeS)?,
        O::I64x2GeS => v_icmp(lo, VShape::I64x2, VICmpOp::GeS)?,
        // integer lane min/max (signed + unsigned); `i64x2` has none in the wasm spec.
        O::I8x16MinS => v_intbin(lo, VShape::I8x16, VIntBinOp::MinS)?,
        O::I8x16MinU => v_intbin(lo, VShape::I8x16, VIntBinOp::MinU)?,
        O::I8x16MaxS => v_intbin(lo, VShape::I8x16, VIntBinOp::MaxS)?,
        O::I8x16MaxU => v_intbin(lo, VShape::I8x16, VIntBinOp::MaxU)?,
        O::I16x8MinS => v_intbin(lo, VShape::I16x8, VIntBinOp::MinS)?,
        O::I16x8MinU => v_intbin(lo, VShape::I16x8, VIntBinOp::MinU)?,
        O::I16x8MaxS => v_intbin(lo, VShape::I16x8, VIntBinOp::MaxS)?,
        O::I16x8MaxU => v_intbin(lo, VShape::I16x8, VIntBinOp::MaxU)?,
        O::I32x4MinS => v_intbin(lo, VShape::I32x4, VIntBinOp::MinS)?,
        O::I32x4MinU => v_intbin(lo, VShape::I32x4, VIntBinOp::MinU)?,
        O::I32x4MaxS => v_intbin(lo, VShape::I32x4, VIntBinOp::MaxS)?,
        O::I32x4MaxU => v_intbin(lo, VShape::I32x4, VIntBinOp::MaxU)?,
        // integer lane shifts (one scalar i32 amount, taken mod the lane bit-width)
        O::I8x16Shl => v_shift(lo, VShape::I8x16, VShiftOp::Shl)?,
        O::I8x16ShrS => v_shift(lo, VShape::I8x16, VShiftOp::ShrS)?,
        O::I8x16ShrU => v_shift(lo, VShape::I8x16, VShiftOp::ShrU)?,
        O::I16x8Shl => v_shift(lo, VShape::I16x8, VShiftOp::Shl)?,
        O::I16x8ShrS => v_shift(lo, VShape::I16x8, VShiftOp::ShrS)?,
        O::I16x8ShrU => v_shift(lo, VShape::I16x8, VShiftOp::ShrU)?,
        O::I32x4Shl => v_shift(lo, VShape::I32x4, VShiftOp::Shl)?,
        O::I32x4ShrS => v_shift(lo, VShape::I32x4, VShiftOp::ShrS)?,
        O::I32x4ShrU => v_shift(lo, VShape::I32x4, VShiftOp::ShrU)?,
        O::I64x2Shl => v_shift(lo, VShape::I64x2, VShiftOp::Shl)?,
        O::I64x2ShrS => v_shift(lo, VShape::I64x2, VShiftOp::ShrS)?,
        O::I64x2ShrU => v_shift(lo, VShape::I64x2, VShiftOp::ShrU)?,
        // saturating add/sub (i8x16/i16x8 only)
        O::I8x16AddSatS => v_satbin(lo, VShape::I8x16, VSatBinOp::AddS)?,
        O::I8x16AddSatU => v_satbin(lo, VShape::I8x16, VSatBinOp::AddU)?,
        O::I8x16SubSatS => v_satbin(lo, VShape::I8x16, VSatBinOp::SubS)?,
        O::I8x16SubSatU => v_satbin(lo, VShape::I8x16, VSatBinOp::SubU)?,
        O::I16x8AddSatS => v_satbin(lo, VShape::I16x8, VSatBinOp::AddS)?,
        O::I16x8AddSatU => v_satbin(lo, VShape::I16x8, VSatBinOp::AddU)?,
        O::I16x8SubSatS => v_satbin(lo, VShape::I16x8, VSatBinOp::SubS)?,
        O::I16x8SubSatU => v_satbin(lo, VShape::I16x8, VSatBinOp::SubU)?,
        O::I8x16AvgrU => v_avgr(lo, VShape::I8x16)?,
        O::I16x8AvgrU => v_avgr(lo, VShape::I16x8)?,
        O::I32x4DotI16x8S => v_dot(lo)?,
        O::I16x8ExtMulLowI8x16S => v_extmul(lo, VShape::I16x8, VWidenOp::LowS)?,
        O::I16x8ExtMulHighI8x16S => v_extmul(lo, VShape::I16x8, VWidenOp::HighS)?,
        O::I16x8ExtMulLowI8x16U => v_extmul(lo, VShape::I16x8, VWidenOp::LowU)?,
        O::I16x8ExtMulHighI8x16U => v_extmul(lo, VShape::I16x8, VWidenOp::HighU)?,
        O::I32x4ExtMulLowI16x8S => v_extmul(lo, VShape::I32x4, VWidenOp::LowS)?,
        O::I32x4ExtMulHighI16x8S => v_extmul(lo, VShape::I32x4, VWidenOp::HighS)?,
        O::I32x4ExtMulLowI16x8U => v_extmul(lo, VShape::I32x4, VWidenOp::LowU)?,
        O::I32x4ExtMulHighI16x8U => v_extmul(lo, VShape::I32x4, VWidenOp::HighU)?,
        O::I64x2ExtMulLowI32x4S => v_extmul(lo, VShape::I64x2, VWidenOp::LowS)?,
        O::I64x2ExtMulHighI32x4S => v_extmul(lo, VShape::I64x2, VWidenOp::HighS)?,
        O::I64x2ExtMulLowI32x4U => v_extmul(lo, VShape::I64x2, VWidenOp::LowU)?,
        O::I64x2ExtMulHighI32x4U => v_extmul(lo, VShape::I64x2, VWidenOp::HighU)?,
        O::I16x8ExtAddPairwiseI8x16S => v_extadd(lo, VShape::I16x8, true)?,
        O::I16x8ExtAddPairwiseI8x16U => v_extadd(lo, VShape::I16x8, false)?,
        O::I32x4ExtAddPairwiseI16x8S => v_extadd(lo, VShape::I32x4, true)?,
        O::I32x4ExtAddPairwiseI16x8U => v_extadd(lo, VShape::I32x4, false)?,
        O::I16x8Q15MulrSatS => v_q15mulr(lo)?,
        // int↔float / float↔float conversions
        O::F32x4ConvertI32x4S => v_convert(lo, VCvtOp::F32x4ConvertI32x4S)?,
        O::F32x4ConvertI32x4U => v_convert(lo, VCvtOp::F32x4ConvertI32x4U)?,
        O::I32x4TruncSatF32x4S => v_convert(lo, VCvtOp::I32x4TruncSatF32x4S)?,
        O::I32x4TruncSatF32x4U => v_convert(lo, VCvtOp::I32x4TruncSatF32x4U)?,
        O::F32x4DemoteF64x2Zero => v_convert(lo, VCvtOp::F32x4DemoteF64x2Zero)?,
        O::F64x2PromoteLowF32x4 => v_convert(lo, VCvtOp::F64x2PromoteLowF32x4)?,
        O::F64x2ConvertLowI32x4S => v_convert(lo, VCvtOp::F64x2ConvertLowI32x4S)?,
        O::F64x2ConvertLowI32x4U => v_convert(lo, VCvtOp::F64x2ConvertLowI32x4U)?,
        O::I32x4TruncSatF64x2SZero => v_convert(lo, VCvtOp::I32x4TruncSatF64x2SZero)?,
        O::I32x4TruncSatF64x2UZero => v_convert(lo, VCvtOp::I32x4TruncSatF64x2UZero)?,
        // lane narrowing (saturating): result shape is the narrower one
        O::I8x16NarrowI16x8S => v_narrow(lo, VShape::I8x16, VNarrowOp::S)?,
        O::I8x16NarrowI16x8U => v_narrow(lo, VShape::I8x16, VNarrowOp::U)?,
        O::I16x8NarrowI32x4S => v_narrow(lo, VShape::I16x8, VNarrowOp::S)?,
        O::I16x8NarrowI32x4U => v_narrow(lo, VShape::I16x8, VNarrowOp::U)?,
        // lane widening (extend): result shape is the wider one
        O::I16x8ExtendLowI8x16S => v_widen(lo, VShape::I16x8, VWidenOp::LowS)?,
        O::I16x8ExtendHighI8x16S => v_widen(lo, VShape::I16x8, VWidenOp::HighS)?,
        O::I16x8ExtendLowI8x16U => v_widen(lo, VShape::I16x8, VWidenOp::LowU)?,
        O::I16x8ExtendHighI8x16U => v_widen(lo, VShape::I16x8, VWidenOp::HighU)?,
        O::I32x4ExtendLowI16x8S => v_widen(lo, VShape::I32x4, VWidenOp::LowS)?,
        O::I32x4ExtendHighI16x8S => v_widen(lo, VShape::I32x4, VWidenOp::HighS)?,
        O::I32x4ExtendLowI16x8U => v_widen(lo, VShape::I32x4, VWidenOp::LowU)?,
        O::I32x4ExtendHighI16x8U => v_widen(lo, VShape::I32x4, VWidenOp::HighU)?,
        O::I64x2ExtendLowI32x4S => v_widen(lo, VShape::I64x2, VWidenOp::LowS)?,
        O::I64x2ExtendHighI32x4S => v_widen(lo, VShape::I64x2, VWidenOp::HighS)?,
        O::I64x2ExtendLowI32x4U => v_widen(lo, VShape::I64x2, VWidenOp::LowU)?,
        O::I64x2ExtendHighI32x4U => v_widen(lo, VShape::I64x2, VWidenOp::HighU)?,
        // integer lane abs/neg
        O::I8x16Abs => v_intun(lo, VShape::I8x16, VIntUnOp::Abs)?,
        O::I8x16Neg => v_intun(lo, VShape::I8x16, VIntUnOp::Neg)?,
        O::I8x16Popcnt => v_popcnt(lo)?,
        O::I16x8Abs => v_intun(lo, VShape::I16x8, VIntUnOp::Abs)?,
        O::I16x8Neg => v_intun(lo, VShape::I16x8, VIntUnOp::Neg)?,
        O::I32x4Abs => v_intun(lo, VShape::I32x4, VIntUnOp::Abs)?,
        O::I32x4Neg => v_intun(lo, VShape::I32x4, VIntUnOp::Neg)?,
        O::I64x2Abs => v_intun(lo, VShape::I64x2, VIntUnOp::Abs)?,
        O::I64x2Neg => v_intun(lo, VShape::I64x2, VIntUnOp::Neg)?,
        // boolean reductions (v128 → i32): any_true (whole-vector), all_true / bitmask (per shape)
        O::V128AnyTrue => v_anytrue(lo)?,
        O::I8x16AllTrue => v_alltrue(lo, VShape::I8x16)?,
        O::I16x8AllTrue => v_alltrue(lo, VShape::I16x8)?,
        O::I32x4AllTrue => v_alltrue(lo, VShape::I32x4)?,
        O::I64x2AllTrue => v_alltrue(lo, VShape::I64x2)?,
        O::I8x16Bitmask => v_bitmask(lo, VShape::I8x16)?,
        O::I16x8Bitmask => v_bitmask(lo, VShape::I16x8)?,
        O::I32x4Bitmask => v_bitmask(lo, VShape::I32x4)?,
        O::I64x2Bitmask => v_bitmask(lo, VShape::I64x2)?,
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
        O::F32x4PMin => v_pminmax(lo, VShape::F32x4, VPMinMaxOp::Pmin)?,
        O::F32x4PMax => v_pminmax(lo, VShape::F32x4, VPMinMaxOp::Pmax)?,
        O::F64x2PMin => v_pminmax(lo, VShape::F64x2, VPMinMaxOp::Pmin)?,
        O::F64x2PMax => v_pminmax(lo, VShape::F64x2, VPMinMaxOp::Pmax)?,
        O::F32x4Abs => v_fun(lo, VShape::F32x4, VFloatUnOp::Abs)?,
        O::F32x4Neg => v_fun(lo, VShape::F32x4, VFloatUnOp::Neg)?,
        O::F32x4Sqrt => v_fun(lo, VShape::F32x4, VFloatUnOp::Sqrt)?,
        O::F64x2Abs => v_fun(lo, VShape::F64x2, VFloatUnOp::Abs)?,
        O::F64x2Neg => v_fun(lo, VShape::F64x2, VFloatUnOp::Neg)?,
        O::F64x2Sqrt => v_fun(lo, VShape::F64x2, VFloatUnOp::Sqrt)?,
        // SIMD float rounding (the rounding `FUnOp`s applied lane-wise).
        O::F32x4Ceil => v_fun(lo, VShape::F32x4, VFloatUnOp::Ceil)?,
        O::F32x4Floor => v_fun(lo, VShape::F32x4, VFloatUnOp::Floor)?,
        O::F32x4Trunc => v_fun(lo, VShape::F32x4, VFloatUnOp::Trunc)?,
        O::F32x4Nearest => v_fun(lo, VShape::F32x4, VFloatUnOp::Nearest)?,
        O::F64x2Ceil => v_fun(lo, VShape::F64x2, VFloatUnOp::Ceil)?,
        O::F64x2Floor => v_fun(lo, VShape::F64x2, VFloatUnOp::Floor)?,
        O::F64x2Trunc => v_fun(lo, VShape::F64x2, VFloatUnOp::Trunc)?,
        O::F64x2Nearest => v_fun(lo, VShape::F64x2, VFloatUnOp::Nearest)?,
        // float lane comparisons → a per-lane all-ones/all-zeros mask (ordered; `ne` unordered).
        O::F32x4Eq => v_fcmp(lo, VShape::F32x4, VFCmpOp::Eq)?,
        O::F32x4Ne => v_fcmp(lo, VShape::F32x4, VFCmpOp::Ne)?,
        O::F32x4Lt => v_fcmp(lo, VShape::F32x4, VFCmpOp::Lt)?,
        O::F32x4Gt => v_fcmp(lo, VShape::F32x4, VFCmpOp::Gt)?,
        O::F32x4Le => v_fcmp(lo, VShape::F32x4, VFCmpOp::Le)?,
        O::F32x4Ge => v_fcmp(lo, VShape::F32x4, VFCmpOp::Ge)?,
        O::F64x2Eq => v_fcmp(lo, VShape::F64x2, VFCmpOp::Eq)?,
        O::F64x2Ne => v_fcmp(lo, VShape::F64x2, VFCmpOp::Ne)?,
        O::F64x2Lt => v_fcmp(lo, VShape::F64x2, VFCmpOp::Lt)?,
        O::F64x2Gt => v_fcmp(lo, VShape::F64x2, VFCmpOp::Gt)?,
        O::F64x2Le => v_fcmp(lo, VShape::F64x2, VFCmpOp::Le)?,
        O::F64x2Ge => v_fcmp(lo, VShape::F64x2, VFCmpOp::Ge)?,
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

        // ---- relaxed SIMD: each op lowers to one spec-allowed deterministic behavior, computed
        // identically in both backends (so the interp↔JIT differential holds). `relaxed_madd`/`nmadd`
        // get a genuine fused FMA; the rest alias to the deterministic op SVM already has. The two
        // `relaxed_dot_i8x16_i7x16` ops (pmaddubsw-shaped) are not yet lowered → clean `Unsupported`.
        O::F32x4RelaxedMadd => v_fma(lo, VShape::F32x4, false)?,
        O::F32x4RelaxedNmadd => v_fma(lo, VShape::F32x4, true)?,
        O::F64x2RelaxedMadd => v_fma(lo, VShape::F64x2, false)?,
        O::F64x2RelaxedNmadd => v_fma(lo, VShape::F64x2, true)?,
        O::F32x4RelaxedMin => v_fbin(lo, VShape::F32x4, VFloatBinOp::Min)?,
        O::F32x4RelaxedMax => v_fbin(lo, VShape::F32x4, VFloatBinOp::Max)?,
        O::F64x2RelaxedMin => v_fbin(lo, VShape::F64x2, VFloatBinOp::Min)?,
        O::F64x2RelaxedMax => v_fbin(lo, VShape::F64x2, VFloatBinOp::Max)?,
        O::I32x4RelaxedTruncF32x4S => v_convert(lo, VCvtOp::I32x4TruncSatF32x4S)?,
        O::I32x4RelaxedTruncF32x4U => v_convert(lo, VCvtOp::I32x4TruncSatF32x4U)?,
        O::I32x4RelaxedTruncF64x2SZero => v_convert(lo, VCvtOp::I32x4TruncSatF64x2SZero)?,
        O::I32x4RelaxedTruncF64x2UZero => v_convert(lo, VCvtOp::I32x4TruncSatF64x2UZero)?,
        O::I16x8RelaxedQ15mulrS => v_q15mulr(lo)?,
        // relaxed_laneselect(a, b, mask) = mask ? a : b for a valid (all-0/all-1) mask — exactly
        // `bitselect`, the deterministic behavior the proposal permits.
        O::I8x16RelaxedLaneselect
        | O::I16x8RelaxedLaneselect
        | O::I32x4RelaxedLaneselect
        | O::I64x2RelaxedLaneselect => {
            let (mask, _) = lo.pop()?;
            let (b, _) = lo.pop()?;
            let (a, _) = lo.pop()?;
            let v = lo.emit(Inst::Bitselect { a, b, mask });
            lo.push(v, ValType::V128);
        }
        O::I8x16RelaxedSwizzle => {
            let (b, _) = lo.pop()?;
            let (a, _) = lo.pop()?;
            let v = lo.emit(Inst::Swizzle { a, b });
            lo.push(v, ValType::V128);
        }
        O::I16x8RelaxedDotI8x16I7x16S => v_dot_i8(lo)?,
        O::I32x4RelaxedDotI8x16I7x16AddS => v_dot_i8_add(lo)?,

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
