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
//! **Scope (Milestone 1, slices A–M):** multi-block scalar functions with stack memory, calls,
//! `switch`, globals, floats, indirect calls, struct aggregates, memory intrinsics, by-value
//! aggregates, pointer-valued global relocations, libm math calls, and integer min/max+bit intrinsics.
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
//! - **K — relocations (pointer-valued globals).** A global initializer holding a function pointer,
//!   `&other_global`, or arithmetic over those resolves via a constexpr evaluator (`GlobalReference`
//!   → address/funcref, `ptrtoint`/`sub`/`add`/`trunc`). The globals layout is two-phase (assign all
//!   addresses, then serialize — forward references resolve). Covers function-pointer tables and
//!   struct/array pointer members.
//! - **L — libm math calls.** A call to an *external* `sqrt`/`fabs`/`floor`/`ceil`/`trunc`/`rint`/
//!   `copysign`/`fmin`/`fmax` (and `…f` f32 variants) lowers to the matching SVM float op inline —
//!   unless the guest defines its own. (`round` and transcendentals have no SVM op → still a call.)
//! - **M — integer min/max + bit intrinsics.** `llvm.smax`/`smin`/`umax`/`umin` → `icmp`+`select`;
//!   `llvm.ctlz`/`cttz`/`ctpop` → `clz`/`ctz`/`popcnt`; `llvm.abs` → `select(x<0,-x,x)`.
//! - **N — the powerbox on-ramp (libc → capabilities, "Lane C").** A program that does I/O gets a
//!   synthesized **powerbox entry** (`_start`, function 0): it takes the granted `(stdout, stdin,
//!   exit)` handles (§3e), stashes them in the reserved low window (page-isolated from the globals),
//!   then calls `main`. An external libc call bound to a host capability (`write`/`read` → `Stream`,
//!   `exit` → `Exit`) lowers to an `Inst::CallImport` the embedder resolves at load (§7); the handle
//!   is reloaded from the stash (the POSIX `fd` is dropped — the handle selects the endpoint). Runs
//!   end-to-end through the reference powerbox with stdout + exit code matching the native build.
//! - **O — the stdio output surface.** The non-varargs libc output family funnels to `Stream.write`
//!   on stdout: `puts` (the literal's bytes + a newline, length from the string global — no runtime
//!   strlen), `putchar`/`putc`/`fputc` (one byte staged through the stash scratch), `fwrite`/`fputs`
//!   (a `size×nmemb` slice / a string), and `fflush` (a no-op — the `Stream` is unbuffered). The
//!   libc `FILE*` stream argument is ignored (the handle is the endpoint). `clang -O2` also lowers
//!   `printf("…\n")` → `puts` and `printf("%c",c)` → `putc`, so format-free `printf` rides this path.
//! - **P — funnel shifts + runtime mem-loop helpers (first real corpus demo).** `llvm.fshl`/`fshr`
//!   lower to `rotl`/`rotr` for the rotate idiom (identical operands — SHA-256's `ROTRIGHT`). A
//!   variable-length (or oversized-constant) `memset`/`memcpy` calls a **synthesized runtime loop
//!   helper** (`__svm_memset`/`__svm_memcpy`, a real counted byte loop — the first multi-block helper)
//!   instead of an inline unroll. Together these make B-Con's **SHA-256** run byte-identical to
//!   native `clang` (`demo_sha256_vs_native`).
//! - **Q — more corpus demos + the gaps they revealed.** `ptrtoint`/`inttoptr` (a width adjust —
//!   pointers are `i64`), `freeze` (identity — the IR is total), and **constexpr GEP** (an interior
//!   pointer into a constant aggregate, `&".."[k]`/`&g.f`, folded to base+offset). Plus a layout fix:
//!   **read-only globals are page-isolated from writable ones** (a `const` next to a mutable `static`
//!   would otherwise fault writes on the shared D40-protected page). Lands **xxHash**, **stb_perlin**,
//!   and **tiny-regex-c** byte-identical to native.
//! - **R — `llvm.load.relative` (clang's relative lookup table).** A `switch` returning constants
//!   compiles to a table of 32-bit `&target − &table` offsets; `load.relative(P, off)` →
//!   `P + sext_i32(*(i32*)(P+off))`. The table initializer (`trunc(sub(ptrtoint…))`) already folds via
//!   the constexpr evaluator. Lands **jsmn** (a zero-alloc JSON parser) byte-identical to native.
//! - **S — `malloc`/heap (the §1a sparse address space).** `malloc`/`calloc` lower to a synthesized
//!   **bump allocator** (`__svm_malloc`) that grows the heap into the window's reserved tail by
//!   `vm_map`-committing pages on demand via the `Memory` capability (a 4th powerbox handle); `free`
//!   is a no-op and the heap never reuses, so freshly-committed (zeroed) pages make `calloc` ≡
//!   `malloc`. Lands **heapgrow** (a guest growing past ~16× its initial window) byte-identical to native.
//! - **T — multi-value struct returns.** A small by-value struct returned in registers (clang coerces
//!   it to e.g. `{ i64, i64 }` / `{ i64, ptr }`) maps to an SVM **multi-result** function (§3a):
//!   `insertvalue`/`extractvalue`/`ret` and multi-result `call`s track the aggregate field-wise in a
//!   block-local side-table (assumed not to cross blocks — clang's register-coercion pattern). Plus
//!   `llvm.experimental.noalias.scope.decl` dropped (an alias hint).
//!
//! Out of the current subset (clean [`Error::Unsupported`]): variable-length `memmove` (overlap),
//! general (non-rotate) funnel shifts, varargs `printf`/`fprintf` (formatting), `realloc`,
//! transcendental math, `puts`/`fputs` of a *non-literal* string (runtime strlen), **SIMD vectors**
//! (`<N x T>` — e.g. clay's `<2 x float>` 2D points), `i33`.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use llvm_ir::instruction::Instruction;
use llvm_ir::terminator::Terminator as LTerm;
use llvm_ir::types::{FPType, Type, Typed, Types};
use llvm_ir::{constant::Constant, constant::Float, BasicBlock, Function, Module as LModule};
use llvm_ir::{FPPredicate, IntPredicate, Name, Operand};

use svm_ir::{
    BinOp, Block, CastOp, CmpOp, ConvOp, FBinOp, FCmpOp, FToI, FUnOp, FloatTy, Func, IToF, Inst,
    IntTy, IntUnOp, Module, Terminator, ValIdx, ValType,
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
    let mut defined_names: HashMap<String, u32> = HashMap::new();
    for (i, f) in defined.iter().enumerate() {
        defined_names.insert(f.name.clone(), i as u32);
    }
    // §7 capability imports: external libc calls bound to host capabilities (`write`/`read`/`exit`).
    // A program that uses any of them — or that allocates (`malloc`, which needs the `Memory`
    // capability) — gets a synthesized powerbox entry (`_start`, function 0); the other functions
    // then sit at index `base..` so `main` can be called from `_start`.
    let (mut imports, mut caps) = collect_cap_imports(m, &defined_names);
    let has_main = defined_names.contains_key("main");
    let need_malloc = needs_malloc(m, &defined_names) && has_main;
    let synth = (!imports.is_empty() || need_malloc) && has_main;
    // The allocator grows the heap via `Memory.map`; register that import (the bump allocator emits a
    // `CallImport "vm_map"`, resolved like any other §7 import at load).
    if need_malloc {
        caps.entry("vm_map".to_string()).or_insert_with(|| {
            let i = imports.len() as u32;
            imports.push(svm_ir::Import {
                name: "vm_map".to_string(),
                sig: import_sig("vm_map"),
            });
            i
        });
    }
    let base: u32 = synth as u32;
    let mut name2idx: HashMap<String, u32> = HashMap::new();
    for (i, f) in defined.iter().enumerate() {
        name2idx.insert(f.name.clone(), i as u32 + base);
    }
    // Globals live low (from `DATA_BASE`); the data stack starts just above them. For a powerbox
    // program the writable **handle stash** occupies the reserved low scratch (`[0, DATA_BASE)`), so
    // start the globals one page up (`STACK_PAGE`): a *read-only* global (D40, protected
    // page-granularly) must never share a page with the stash, or `_start`'s handle stores would
    // fault on the read-only page (the same page-isolation the data stack already gets above globals).
    let globals_base = if synth { STACK_PAGE } else { DATA_BASE };
    let (globals, data, globals_end, cstrs) = globals_layout(m, &name2idx, globals_base)?;
    // Page-align the data stack above the globals so it never shares a page with a *read-only*
    // global (D40 protects RO segments page-granularly — a stack write into a shared page would
    // fault). 16 KiB covers the largest common page size (macOS/aarch64). (A read-only and a
    // writable global sharing a page is a separate latent issue — page-isolating those is a follow-up.)
    let entry_sp = globals_end.div_ceil(STACK_PAGE) * STACK_PAGE;

    // Synthesized helpers (mem-loop `memset`/`memcpy`, the `malloc` allocator) sit after the defined
    // functions and `_start` (index 0 when `synth`), at `base + defined.len()` onward — their indices
    // are fixed before translating call sites. The allocator references the `vm_map` import index.
    let (need_memset, need_memcpy) = needs_mem_helpers(m);
    let helper_base = base + defined.len() as u32;
    let helpers = Helpers {
        memset: need_memset.then_some(helper_base),
        memcpy: need_memcpy.then_some(helper_base + need_memset as u32),
        malloc: need_malloc.then_some(helper_base + need_memset as u32 + need_memcpy as u32),
    };

    let mut funcs = Vec::with_capacity(defined.len() + synth as usize);
    let mut any_frame = false; // does any function use the data stack (`alloca`)?
    for f in &defined {
        let (func, frame_size) =
            translate_func(f, &m.types, &name2idx, &globals, &caps, &cstrs, &helpers)?;
        any_frame |= frame_size > 0;
        funcs.push(func);
    }

    // The window: globals low, then the data stack from `entry_sp` growing up; `mapped` covers the
    // globals plus a stack reserve, with a faulting guard beyond (reserved > mapped, §5). Declared if
    // any function uses the data stack, the module has globals, or it uses the powerbox (the handle
    // stash / heap state live in the reserved low window).
    let need_window = any_frame || !globals.is_empty() || synth;
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
    // The guest heap begins at the window's mapped boundary (the first reserved page) and grows up
    // into the reserved tail as the allocator `vm_map`-commits it (§1a sparse address space).
    let heap_base = need_malloc
        .then(|| memory.map(|mc| 1u64 << mc.size_log2))
        .flatten();

    // Prepend the synthesized powerbox entry (`_start`) at function 0: it receives the granted
    // capability handles, stashes them (and seeds the heap), then calls `main(entry_sp)`.
    if synth {
        let main_idx = name2idx["main"];
        let main_results = funcs[(main_idx - base) as usize].results.clone();
        funcs.insert(0, synth_start(main_idx, &main_results, entry_sp, heap_base));
    }
    // Append the synthesized helpers in index order (memset, memcpy, malloc) — matching `helper_base`.
    if need_memset {
        funcs.push(synth_memset());
    }
    if need_memcpy {
        funcs.push(synth_memcpy());
    }
    if need_malloc {
        funcs.push(synth_malloc(caps["vm_map"]));
    }
    Ok(Translated {
        module: Module {
            funcs,
            memory,
            data,
            // §7 named capability imports (`write`/`read`/`exit` …) the host resolves at load
            // (`resolve_capability_imports`); empty for a pure-compute (kernel) module.
            imports,
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

/// Evaluate an integer/pointer **constexpr** to its window value (a relocation): a global's
/// address, a function's funcref index, or arithmetic over those (`sub`/`add`/`ptrtoint`/`trunc`…).
/// This is what lets a global hold `&other_global`, a function pointer, or clang's relative-offset
/// table (`@.str − @table`). GEP-constexprs (`&arr[k]`) are a later addition.
fn const_eval(
    c: &Constant,
    globals: &HashMap<String, u64>,
    funcs: &HashMap<String, u32>,
    types: &Types,
) -> Result<i64, Error> {
    use Constant as K;
    let bin = |a: &Constant, b: &Constant| -> Result<(i64, i64), Error> {
        Ok((
            const_eval(a, globals, funcs, types)?,
            const_eval(b, globals, funcs, types)?,
        ))
    };
    match c {
        K::Int { value, .. } => Ok(*value as i64),
        K::Null(_) => Ok(0),
        K::GlobalReference { name, .. } => {
            let n = name_str(name);
            if let Some(&a) = globals.get(&n) {
                Ok(a as i64) // a data global's window address
            } else if let Some(&f) = funcs.get(&n) {
                Ok(f as i64) // a function's §3c funcref index
            } else {
                unsup(format!("constexpr reference to `@{n}`"))
            }
        }
        // Pointer/width casts pass the value through; the byte width is the *consumer*'s job.
        K::PtrToInt(x) => const_eval(x.operand.as_ref(), globals, funcs, types),
        K::IntToPtr(x) => const_eval(x.operand.as_ref(), globals, funcs, types),
        K::BitCast(x) => const_eval(x.operand.as_ref(), globals, funcs, types),
        K::Trunc(x) => const_eval(x.operand.as_ref(), globals, funcs, types),
        K::Add(x) => bin(x.operand0.as_ref(), x.operand1.as_ref()).map(|(a, b)| a.wrapping_add(b)),
        K::Sub(x) => bin(x.operand0.as_ref(), x.operand1.as_ref()).map(|(a, b)| a.wrapping_sub(b)),
        K::Mul(x) => bin(x.operand0.as_ref(), x.operand1.as_ref()).map(|(a, b)| a.wrapping_mul(b)),
        // An interior pointer into a constant aggregate (`&arr[k]`, `&s.f`, a string-literal tail
        // `&".."[k]`) — base address plus the type-walked constant byte offset (§3b, like `getelementptr`).
        K::GetElementPtr(g) => {
            let base = const_eval(g.address.as_ref(), globals, funcs, types)?;
            Ok(base.wrapping_add(const_gep_offset(g, types)?))
        }
        other => unsup(format!("constexpr initializer {other:?}")),
    }
}

/// The constant byte offset of a **constexpr** `getelementptr` (all indices constant), walking the
/// pointee type (carried by the base `GlobalReference`) exactly as [`translate_gep`] does for the
/// instruction form: index 0 strides by the whole pointee, later indices descend array elements /
/// struct fields.
fn const_gep_offset(g: &llvm_ir::constant::GetElementPtr, types: &Types) -> Result<i64, Error> {
    // The pointee type the GEP indexes from — a `GlobalReference` carries it directly.
    let mut cur = match g.address.as_ref() {
        Constant::GlobalReference { ty, .. } => ty.clone(),
        other => return unsup(format!("constexpr GEP base {other:?}")),
    };
    let idx_val = |c: &Constant| -> Result<i64, Error> {
        match c {
            Constant::Int { value, .. } => Ok(*value as i64),
            _ => unsup("constexpr GEP with non-constant index"),
        }
    };
    let mut off: i64 = 0;
    for (k, idx) in g.indices.iter().enumerate() {
        let iv = idx_val(idx.as_ref())?;
        if k > 0
            && matches!(
                cur.as_ref(),
                Type::StructType { .. } | Type::NamedStructType { .. }
            )
        {
            let (fields, packed) = resolve_struct(cur.as_ref(), types)?;
            let (offsets, _, _) = struct_layout(&fields, packed, types)?;
            off += *offsets
                .get(iv as usize)
                .ok_or_else(|| Error::Unsupported("constexpr GEP field out of range".into()))?
                as i64;
            cur = fields[iv as usize].clone();
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
                other => return unsup(format!("constexpr GEP into type {other}")),
            }
        };
        off += iv.wrapping_mul(stride as i64);
    }
    Ok(off)
}

/// The serialized byte length of a constant initializer — identical to `const_bytes(…).len()`, but
/// computed *without* resolving relocations (a pointer is 8 bytes whatever it points to). Used in
/// the globals layout's phase A (assign addresses) before phase B can serialize the actual bytes.
fn const_size(c: &Constant, types: &Types) -> Result<u64, Error> {
    match c {
        Constant::Int { bits, .. } if *bits <= 64 => Ok((*bits as u64).div_ceil(8).max(1)),
        Constant::Float(Float::Single(_)) => Ok(4),
        Constant::Float(Float::Double(_)) => Ok(8),
        Constant::Array { elements, .. } | Constant::Vector(elements) => {
            let mut n = 0;
            for e in elements {
                n += const_size(e.as_ref(), types)?;
            }
            Ok(n)
        }
        Constant::Struct {
            values, is_packed, ..
        } => {
            let fields: Vec<llvm_ir::TypeRef> = values.iter().map(|v| v.get_type(types)).collect();
            Ok(struct_layout(&fields, *is_packed, types)?.1)
        }
        Constant::AggregateZero(t) | Constant::Undef(t) | Constant::Poison(t) => {
            type_size(t.as_ref(), types)
        }
        // A pointer / constexpr scalar leaf — its width is its type's size (8 for a pointer).
        other => type_size(other.get_type(types).as_ref(), types),
    }
}

/// Serialize a constant initializer to its little-endian window bytes (the §3d/x86-64 layout).
/// Aggregates recurse structurally (arrays/structs with field padding); a scalar leaf that is a
/// pointer or constexpr is resolved via [`const_eval`] (relocations) and emitted at its type width.
fn const_bytes(
    c: &Constant,
    types: &Types,
    globals: &HashMap<String, u64>,
    funcs: &HashMap<String, u32>,
) -> Result<Vec<u8>, Error> {
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
                out.extend(const_bytes(e.as_ref(), types, globals, funcs)?);
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
                let b = const_bytes(v.as_ref(), types, globals, funcs)?;
                out[off as usize..off as usize + b.len()].copy_from_slice(&b);
            }
            Ok(out)
        }
        Constant::AggregateZero(t) | Constant::Undef(t) | Constant::Poison(t) => {
            Ok(vec![0u8; type_size(t.as_ref(), types)? as usize])
        }
        // A pointer / constexpr scalar leaf (a relocation): resolve its value, emit at type width.
        other => {
            let width = type_size(other.get_type(types).as_ref(), types)?;
            if width > 8 {
                return unsup(format!(
                    "constexpr initializer wider than 8 bytes ({width})"
                ));
            }
            let v = const_eval(other, globals, funcs, types)?;
            Ok(v.to_le_bytes()[..width as usize].to_vec())
        }
    }
}

/// The result of [`globals_layout`]: name → window address, the `data` segments to emit, and the
/// globals region's end offset (for window sizing).
/// `globals_layout`'s output: name → window-address, the `data` segments to emit, the region's end
/// offset, and name → C-string length (bytes before the first NUL) for the string-literal globals a
/// `puts`/`fputs` argument points at (so the on-ramp knows the write length without a runtime strlen).
type Globals = (
    HashMap<String, u64>,
    Vec<svm_ir::Data>,
    u64,
    HashMap<String, u64>,
);

/// Lay out the module's global variables in the window's globals region (from `base`, each
/// natural-aligned), returning the name → window-address map, the `data` segments to emit
/// (constants read-only, §3a/D40; all-zero/BSS globals just reserve space in the zero-init
/// window), the region's end (for window sizing), and the string-literal lengths (for `puts`).
fn globals_layout(
    m: &LModule,
    name2idx: &HashMap<String, u32>,
    base: u64,
) -> Result<Globals, Error> {
    // Phase A: assign every global a window address (from its declared type size), so a relocation
    // in any initializer can resolve a forward/backward reference to another global in phase B.
    //
    // **Read-only globals are page-isolated from writable ones** (D40): a constant segment is
    // protected page-granularly, so if a `const` global shared a page with a writable/BSS global
    // (e.g. clang's `static char buf[]` next to a string literal), a legitimate write to the
    // writable one would fault on the read-only page. So lay writable globals first, page-align, then
    // the read-only globals — the read-only region begins on a fresh page and the writable region's
    // last page carries no constant. (The data stack is already page-aligned above all of them.)
    let mut addr = HashMap::new();
    let mut off = base;
    let mut placed: Vec<(usize, u64)> = Vec::with_capacity(m.global_vars.len());
    let mut place =
        |off: &mut u64, addr: &mut HashMap<String, u64>, want_const: bool| -> Result<(), Error> {
            for (gi, g) in m.global_vars.iter().enumerate() {
                if g.is_constant != want_const {
                    continue;
                }
                // Size from the initializer's serialized length (matches phase B exactly); BSS/extern
                // globals have no initializer, so fall back to the declared type size.
                let size = match &g.initializer {
                    Some(init) => const_size(init.as_ref(), &m.types)?,
                    None => type_size(g.ty.as_ref(), &m.types)?,
                }
                .max(1);
                let align = (g.alignment as u64).max(1);
                *off = off.div_ceil(align) * align;
                addr.insert(name_str(&g.name), *off);
                placed.push((gi, *off));
                *off += size;
            }
            Ok(())
        };
    place(&mut off, &mut addr, false)?; // writable + BSS globals
    let any_const = m.global_vars.iter().any(|g| g.is_constant);
    if any_const && off > base {
        off = off.div_ceil(STACK_PAGE) * STACK_PAGE; // page-isolate the read-only region
    }
    place(&mut off, &mut addr, true)?; // read-only (constant) globals
                                       // Phase B: serialize each initialized global (now able to resolve relocations via `addr`).
    let mut segs = Vec::new();
    let mut cstrs = HashMap::new();
    for (gi, at) in placed {
        let g = &m.global_vars[gi];
        let Some(init) = &g.initializer else { continue }; // BSS / extern → zero-init window
        let bytes = const_bytes(init.as_ref(), &m.types, &addr, name2idx)?;
        // Record the C-string length (up to the first NUL) so `puts`/`fputs` on this literal can
        // write the right slice without a runtime strlen.
        let slen = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len()) as u64;
        cstrs.insert(name_str(&g.name), slen);
        // Emit a segment only for non-zero initialized data (the window is zero-init). A read-only
        // segment is protected (D40), so a guest write to it faults.
        if g.is_constant || bytes.iter().any(|&x| x != 0) {
            segs.push(svm_ir::Data {
                offset: at,
                readonly: g.is_constant,
                bytes,
            });
        }
    }
    Ok((addr, segs, off, cstrs))
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

/// The scalar **fields of a small by-value struct** (clang's multi-register return/arg coercion,
/// e.g. `{ i64, ptr }`), each mapped to its SVM value type — the components of an SVM **multi-result**
/// (§3a). Only scalar fields are supported (a nested aggregate is `Unsupported`). `None` if `ty` is
/// not a struct (the caller handles the scalar/void cases).
fn struct_field_vtypes(ty: &Type, types: &Types) -> Option<Result<Vec<ValType>, Error>> {
    match ty {
        Type::StructType { element_types, .. } => {
            Some(element_types.iter().map(|t| val_type(t.as_ref())).collect())
        }
        Type::NamedStructType { name } => match types.named_struct_def(name) {
            Some(llvm_ir::types::NamedStructDef::Defined(t)) => {
                struct_field_vtypes(t.as_ref(), types)
            }
            _ => None,
        },
        _ => None,
    }
}

/// The SVM result list for an LLVM return type: `[]` for `void`, the flattened scalar fields for a
/// small by-value struct (a multi-result function, §3a), or a single scalar otherwise.
fn result_types(ty: &Type, types: &Types) -> Result<Vec<ValType>, Error> {
    match ty {
        Type::VoidType => Ok(Vec::new()),
        _ => match struct_field_vtypes(ty, types) {
            Some(fields) => fields,
            None => Ok(vec![val_type(ty)?]),
        },
    }
}

/// A typed zero constant (the placeholder for an as-yet-unset aggregate field, before `insertvalue`
/// overwrites it).
fn zero_inst(t: ValType) -> Inst {
    match t {
        ValType::I32 => Inst::ConstI32(0),
        ValType::F32 => Inst::ConstF32(0),
        ValType::F64 => Inst::ConstF64(0),
        _ => Inst::ConstI64(0), // I64 / pointer (and the unreachable V128)
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
    caps: &HashMap<String, u32>,
    cstrs: &HashMap<String, u64>,
    helpers: &Helpers,
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
    // A small by-value struct return flattens to a multi-result signature (§3a).
    let results = result_types(f.return_type.as_ref(), types)?;

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
            caps,
            cstrs,
            helpers,
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
                let ty = instr.get_type(types);
                let vt = match val_type(ty.as_ref()) {
                    Ok(t) => t,
                    // A small by-value struct (a call/`insertvalue` result) is tracked field-wise via
                    // the aggregate side-table, never used as a scalar — record a placeholder type.
                    Err(_) if struct_field_vtypes(ty.as_ref(), types).is_some() => ValType::I64,
                    Err(e) => return Err(e),
                };
                s.ty.push(vt);
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
        I::PtrToInt(x) => locals(&[&x.operand]),
        I::IntToPtr(x) => locals(&[&x.operand]),
        I::Freeze(x) => locals(&[&x.operand]),
        // Aggregate build/destructure (a small by-value struct, register-coerced): the aggregate
        // operand + (for insert) the inserted element.
        I::InsertValue(x) => locals(&[&x.aggregate, &x.element]),
        I::ExtractValue(x) => locals(&[&x.aggregate]),
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
fn indirect_sig(c: &llvm_ir::instruction::Call, types: &Types) -> Result<svm_ir::FuncType, Error> {
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
            let results = result_types(result_type.as_ref(), types)?;
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

// --- §7 capability imports / the powerbox on-ramp (LLVM.md §9 "Lane C") --------------------------
//
// A C program that does I/O calls libc (`write`/`read`/`exit`); clang leaves those as
// declaration-only externals. The on-ramp binds each to a **host capability** (§7 named import): a
// call lowers to an `Inst::CallImport "<name>"` the embedder resolves at load (`default_cap_resolver`
// → `(type_id, op)`). The capability **handle** is not a C argument — it is granted to the powerbox
// entry and threaded to every call site through the *handle stash*, the reserved low window
// (`[0, DATA_BASE)`): `_start` stores the granted handles there and each call site reloads the one
// it needs (so a handle reaches arbitrary call depth without a viral extra parameter). This keeps
// the translator pure mechanism — it never interprets host semantics, just defers the bind (§2a).

/// Window offsets of the **powerbox handle stash + allocator state** (the reserved low scratch on
/// page 0, which is writable — for a powerbox program the globals start a page up). `_start` stores
/// the granted handles here; each call site reloads what it needs. The heap allocator (slice S) keeps
/// its bump pointer + committed boundary here too.
const STASH_STDOUT: u64 = 0;
const STASH_STDIN: u64 = 4;
const STASH_EXIT: u64 = 8;
/// The `Memory` capability handle (`i32`) — present only when the program uses `malloc` (then `_start`
/// takes a 4th granted handle). The bump allocator reloads it to `vm_map`-commit reserved pages.
const STASH_MEMORY: u64 = 12;
/// The guest heap's bump pointer and committed boundary (`i64` each) — the allocator's only state.
const HEAP_BRK: u64 = 16;
const HEAP_TOP: u64 = 24;
/// A 1-byte writable scratch used by `putc`/`puts` to stage a single byte (a char, a newline) the
/// `Stream` capability writes (its ABI is a `(buf, len)` window slice, so a scalar char must transit
/// memory). Reused per call — single-threaded, fully produced-then-consumed within one lowering.
const STASH_SCRATCH: u64 = 32;
/// The `prot` bits a guest passes to `Memory.map` for a read-write commit (`PROT_READ|PROT_WRITE`).
const PROT_RW: i32 = 3;

/// Which stash slot a capability call reads its handle from.
#[derive(Clone, Copy)]
enum HandleSlot {
    Stdout,
    Stdin,
    Exit,
}

/// A libc/POSIX function the on-ramp binds to a host capability: the import `name` the host resolves
/// (via `default_cap_resolver`), the op `sig` (the **capability ABI**, not the C prototype), the
/// stash slot its handle comes from, and how many leading C args to drop (the POSIX `fd`, which the
/// capability handle subsumes — the endpoint is selected by the handle, not the fd).
struct CapSpec {
    name: &'static str,
    sig: svm_ir::FuncType,
    handle: HandleSlot,
    drop_args: usize,
}

/// The reference powerbox binding for a libc/POSIX function name, or `None` if it is not a bound
/// capability (so it stays an ordinary direct call / a fail-closed `Unsupported`). The op signatures
/// match `svm-run`'s `default_cap_resolver`: `write`/`read` are `Stream` (`(i64 buf, i64 len) ->
/// (i64)`, the `fd` dropped — the handle is the endpoint), `exit` is `Exit` (`(i32) -> ()`).
fn cap_spec(name: &str) -> Option<CapSpec> {
    use ValType::{I32, I64};
    let ft = |params: Vec<ValType>, results: Vec<ValType>| svm_ir::FuncType { params, results };
    Some(match name {
        "write" => CapSpec {
            name: "write",
            sig: ft(vec![I64, I64], vec![I64]),
            handle: HandleSlot::Stdout,
            drop_args: 1,
        },
        "read" => CapSpec {
            name: "read",
            sig: ft(vec![I64, I64], vec![I64]),
            handle: HandleSlot::Stdin,
            drop_args: 1,
        },
        "exit" | "_exit" | "_Exit" => CapSpec {
            name: "exit",
            sig: ft(vec![I32], vec![]),
            handle: HandleSlot::Exit,
            drop_args: 0,
        },
        _ => return None,
    })
}

/// The §7 import a libc I/O function ultimately needs, or `None` if it is not a bound I/O function.
/// The *stdio* wrappers (`puts`/`putc`/`putchar`/`fputc`/`fwrite`/`fputs`) all funnel to the same
/// `Stream.write` capability — they differ only in how the on-ramp marshals their args (a single
/// char, a NUL-terminated string + newline, a `size×nmemb` slice). `fflush` is recognized by the
/// lowering but needs *no* import (an unbuffered `Stream` makes it a no-op), so it is not listed here.
fn cap_import_name(name: &str) -> Option<&'static str> {
    Some(match name {
        "write" | "puts" | "putc" | "putchar" | "fputc" | "fwrite" | "fputs" => "write",
        "read" => "read",
        "exit" | "_exit" | "_Exit" => "exit",
        _ => return None,
    })
}

/// The capability op signature for an import name (`default_cap_resolver`'s ABI): `Stream`
/// (`write`/`read`) is `(i64 buf, i64 len) -> (i64)`, `Exit` is `(i32) -> ()`.
fn import_sig(import: &str) -> svm_ir::FuncType {
    use ValType::{I32, I64};
    let ft = |params: Vec<ValType>, results: Vec<ValType>| svm_ir::FuncType { params, results };
    match import {
        "exit" => ft(vec![I32], vec![]),
        // `Memory.map(offset, len, prot)` (§3e op 0) — the allocator's page-commit primitive.
        "vm_map" => ft(vec![I64, I64, I32], vec![I64]),
        _ => ft(vec![I64, I64], vec![I64]), // write / read (Stream)
    }
}

/// Scan the module for calls to external (not guest-defined) functions bound to a host capability,
/// building the module's §7 import table (deduplicated) and an `import-name → import index` map the
/// call lowering uses. Several libc names can funnel to one import (e.g. `write`/`puts`/`fwrite` all
/// need `Stream.write`), so the table is keyed by the *import* name, not the C name. A name the guest
/// *defines* is never treated as a capability (it shadows the libc symbol), mirroring the libm rule.
fn collect_cap_imports(
    m: &LModule,
    defined: &HashMap<String, u32>,
) -> (Vec<svm_ir::Import>, HashMap<String, u32>) {
    let mut imports: Vec<svm_ir::Import> = Vec::new();
    let mut import_of: HashMap<String, u32> = HashMap::new();
    for f in &m.functions {
        for bb in &f.basic_blocks {
            for instr in &bb.instrs {
                let Instruction::Call(c) = instr else {
                    continue;
                };
                let Some(name) = callee_name(c) else { continue };
                if defined.contains_key(&name) {
                    continue;
                }
                if let Some(import) = cap_import_name(&name) {
                    import_of.entry(import.to_string()).or_insert_with(|| {
                        let i = imports.len() as u32;
                        imports.push(svm_ir::Import {
                            name: import.to_string(),
                            sig: import_sig(import),
                        });
                        i
                    });
                }
            }
        }
    }
    (imports, import_of)
}

/// Synthesize the **powerbox entry** (`_start`, function 0) for a program that uses host
/// capabilities. It takes the granted handles `(stdout, stdin, exit)` as `i32` params (the §3e
/// powerbox shape `is_powerbox_entry` recognizes — no threaded data-SP, since it is the root),
/// stores them into the handle stash so every capability call site can reload its handle, then calls
/// the C `main(sp)` at the page-aligned data-stack base and returns its exit code.
fn synth_start(
    main_idx: u32,
    main_results: &[ValType],
    entry_sp: u64,
    heap_base: Option<u64>,
) -> Func {
    use svm_ir::StoreOp;
    let mut insts: Vec<Inst> = Vec::new();
    // params: v0 = stdout, v1 = stdin, v2 = exit, [v3 = memory] (i32 capability handles). A program
    // that uses `malloc` takes the 4th `Memory` handle (the powerbox grants it for a 4-param entry).
    let mut handles = vec![(STASH_STDOUT, 0), (STASH_STDIN, 1), (STASH_EXIT, 2)];
    let mut params = vec![ValType::I32, ValType::I32, ValType::I32];
    if heap_base.is_some() {
        handles.push((STASH_MEMORY, 3));
        params.push(ValType::I32);
    }
    let mut next: ValIdx = params.len() as ValIdx;
    for (off, handle) in handles {
        insts.push(Inst::ConstI64(off as i64));
        let addr = next;
        next += 1;
        insts.push(Inst::Store {
            op: StoreOp::I32,
            addr,
            value: handle,
            offset: 0,
            align: 0,
        });
    }
    // Initialize the heap: the bump pointer and the committed boundary both start at `heap_base` (the
    // window's mapped boundary — the first reserved page); the allocator `vm_map`-commits upward.
    if let Some(hb) = heap_base {
        for off in [HEAP_BRK, HEAP_TOP] {
            insts.push(Inst::ConstI64(off as i64));
            let addr = next;
            next += 1;
            insts.push(Inst::ConstI64(hb as i64));
            let val = next;
            next += 1;
            insts.push(Inst::Store {
                op: StoreOp::I64,
                addr,
                value: val,
                offset: 0,
                align: 0,
            });
        }
    }
    // sp = entry_sp (constant); call main(sp). `main` carries the threaded data-SP as param 0.
    insts.push(Inst::ConstI64(entry_sp as i64));
    let sp = next;
    next += 1;
    insts.push(Inst::Call {
        func: main_idx,
        args: vec![sp],
    });
    let term = if main_results.is_empty() {
        Terminator::Return(vec![])
    } else {
        Terminator::Return(vec![next]) // main's single result, appended by the call
    };
    Func {
        results: main_results.to_vec(),
        blocks: vec![Block {
            params: params.clone(),
            insts,
            term,
        }],
        params,
    }
}

/// Synthesize `__svm_malloc(size:i64) -> i64`: an on-demand **bump allocator** that grows the guest
/// heap into the window's reserved tail by `vm_map`-committing pages as needed (§3e/§4 — the §1a
/// "grow past the initial window" capability). State is two `i64`s in the low scratch: `HEAP_BRK` (the
/// next free address) and `HEAP_TOP` (the committed boundary). `free` is a no-op and the heap never
/// reuses, so every result is freshly `vm_map`-zeroed memory (hence `calloc` ≡ `malloc`).
///
/// ```text
///   block0(size):                              ; align the request to 16, compute the new break
///     brk = load.i64 [HEAP_BRK]
///     new = (brk + size + 15) & ~15
///     top = load.i64 [HEAP_TOP]
///     grow? = new >u top   → grow(brk,new,top) : commit(brk,new)
///   grow(brk,new,top):                         ; commit [top, page_up(new)) via the Memory cap
///     limit = (new + (PAGE-1)) & ~(PAGE-1)
///     vm_map(mem_handle, top, limit - top, RW)
///     store.i64 [HEAP_TOP] = limit
///     → commit(brk,new)
///   commit(brk,new):                           ; publish the new break, return the old one
///     store.i64 [HEAP_BRK] = new
///     return brk
/// ```
fn synth_malloc(vm_map_import: u32) -> Func {
    use svm_ir::{LoadOp, StoreOp};
    let i64add = |a: ValIdx, b: ValIdx| Inst::IntBin {
        ty: IntTy::I64,
        op: BinOp::Add,
        a,
        b,
    };
    let i64and = |a: ValIdx, b: ValIdx| Inst::IntBin {
        ty: IntTy::I64,
        op: BinOp::And,
        a,
        b,
    };

    let load_i64 = |addr: ValIdx| Inst::Load {
        op: LoadOp::I64,
        addr,
        offset: 0,
        align: 0,
    };
    let store_i64 = |addr: ValIdx, value: ValIdx| Inst::Store {
        op: StoreOp::I64,
        addr,
        value,
        offset: 0,
        align: 0,
    };
    // block0(size=0): brk = *HEAP_BRK; new = align16(brk + size); top = *HEAP_TOP; branch on new>top.
    let b0 = Block {
        params: vec![ValType::I64], // size = v0
        insts: vec![
            Inst::ConstI64(HEAP_BRK as i64), // v1
            load_i64(1),                     // v2 = brk
            i64add(2, 0),                    // v3 = brk + size
            Inst::ConstI64(15),              // v4
            i64add(3, 4),                    // v5 = brk+size+15
            Inst::ConstI64(!15i64),          // v6 = ~15
            i64and(5, 6),                    // v7 = new (aligned)
            Inst::ConstI64(HEAP_TOP as i64), // v8
            load_i64(8),                     // v9 = top
            Inst::IntCmp {
                ty: IntTy::I64,
                op: CmpOp::GtU,
                a: 7,
                b: 9,
            }, // v10 = new > top
        ],
        term: Terminator::BrIf {
            cond: 10,
            then_blk: 1, // grow(brk=v2, new=v7, top=v9)
            then_args: vec![2, 7, 9],
            else_blk: 2, // commit(brk=v2, new=v7)
            else_args: vec![2, 7],
        },
    };

    // grow(brk=0, new=1, top=2): commit [top, page_up(new)) via vm_map, update HEAP_TOP.
    let page = STACK_PAGE as i64; // commit in ≥-OS-page units (16 KiB covers any real page)
    let g = Block {
        params: vec![ValType::I64, ValType::I64, ValType::I64], // brk, new, top
        insts: vec![
            Inst::ConstI64(page - 1),            // v3
            i64add(1, 3),                        // v4 = new + (PAGE-1)
            Inst::ConstI64(!(page - 1)),         // v5 = ~(PAGE-1)
            i64and(4, 5),                        // v6 = limit (page-aligned)
            Inst::ConstI64(STASH_MEMORY as i64), // v7
            Inst::Load {
                op: LoadOp::I32,
                addr: 7,
                offset: 0,
                align: 0,
            }, // v8 = mem handle
            Inst::IntBin {
                ty: IntTy::I64,
                op: BinOp::Sub,
                a: 6,
                b: 2,
            }, // v9 = limit - top (len)
            Inst::ConstI32(PROT_RW),             // v10 = prot
            Inst::CallImport {
                import: vm_map_import,
                sig: import_sig("vm_map"),
                handle: 8,
                args: vec![2, 9, 10],
            }, // v11 = map result (ignored)
            Inst::ConstI64(HEAP_TOP as i64),     // v12
            store_i64(12, 6),                    // *HEAP_TOP = limit
        ],
        term: Terminator::Br {
            target: 2,
            args: vec![0, 1],
        }, // commit(brk, new)
    };

    // commit(brk=0, new=1): publish the new break, return the old break.
    let c = Block {
        params: vec![ValType::I64, ValType::I64], // brk, new
        insts: vec![
            Inst::ConstI64(HEAP_BRK as i64), // v2
            store_i64(2, 1),                 // *HEAP_BRK = new
        ],
        term: Terminator::Return(vec![0]), // brk
    };

    Func {
        params: vec![ValType::I64],
        results: vec![ValType::I64],
        blocks: vec![b0, g, c],
    }
}

/// Indices of the synthesized **runtime mem-loop helpers**. A variable-length (or oversized-constant)
/// `llvm.memset`/`memcpy` calls one of these instead of an inline unroll — the first use of the
/// synthesized-multi-block-helper machinery (a real CFG with a counted loop, like a tiny libc). The
/// helpers take no data-SP (they touch only the passed window addresses). `None` when not needed.
#[derive(Clone, Copy, Default)]
struct Helpers {
    /// `__svm_memset(dst:i64, byte:i32, len:i64)` — fill `len` bytes at `dst` with `byte`'s low byte.
    memset: Option<u32>,
    /// `__svm_memcpy(dst:i64, src:i64, len:i64)` — copy `len` bytes `src`→`dst` (forward; no overlap).
    memcpy: Option<u32>,
    /// `__svm_malloc(size:i64) -> i64` — the `vm_map`-growing bump allocator (`malloc`/`calloc`).
    malloc: Option<u32>,
}

/// Does the module call any heap allocator function (`malloc`/`calloc`) not defined by the guest?
/// (`free` is lowered to a no-op and needs no allocator; `realloc` is still `Unsupported`.)
fn needs_malloc(m: &LModule, defined: &HashMap<String, u32>) -> bool {
    m.functions.iter().flat_map(|f| &f.basic_blocks).any(|bb| {
        bb.instrs.iter().any(|i| {
            matches!(i, Instruction::Call(c)
                if callee_name(c).is_some_and(|n| matches!(n.as_str(), "malloc" | "calloc")
                    && !defined.contains_key(&n)))
        })
    })
}

/// Does any mem intrinsic need the runtime loop helper — a **non-constant** length, or a constant
/// one too large to unroll inline? Returns `(needs_memset, needs_memcpy)`. (`memmove` with a
/// variable length is left `Unsupported` — overlap direction needs a runtime branch, a later slice.)
fn needs_mem_helpers(m: &LModule) -> (bool, bool) {
    let (mut set, mut cpy) = (false, false);
    for f in &m.functions {
        for bb in &f.basic_blocks {
            for instr in &bb.instrs {
                let Instruction::Call(c) = instr else {
                    continue;
                };
                let Some(name) = callee_name(c) else { continue };
                let big = |c: &llvm_ir::instruction::Call| {
                    c.arguments
                        .get(2)
                        .is_some_and(|(a, _)| const_int(a).is_none_or(|n| n > MAX_MEM_UNROLL))
                };
                if name.starts_with("llvm.memset") && big(c) {
                    set = true;
                } else if name.starts_with("llvm.memcpy") && big(c) {
                    cpy = true;
                }
            }
        }
    }
    (set, cpy)
}

/// Synthesize `__svm_memset(dst:i64, byte:i32, len:i64)`: a counted byte loop
/// `for (i=0; i<len; i++) dst[i] = byte`. Four blocks — entry, the `i<len` test, the body, and the
/// return — threading `(dst, byte, len, i)` as block params (the SSA → block-arg form, hand-built).
fn synth_memset() -> Func {
    use svm_ir::StoreOp;
    let params = vec![ValType::I64, ValType::I32, ValType::I64];
    // block0(dst=0, byte=1, len=2): i = 0; br loop(dst, byte, len, i)
    let entry = Block {
        params: params.clone(),
        insts: vec![Inst::ConstI64(0)],
        term: Terminator::Br {
            target: 1,
            args: vec![0, 1, 2, 3],
        },
    };
    // loop(dst=0, byte=1, len=2, i=3): cond = i <u len; br_if cond body(..) done()
    let loop_params = vec![ValType::I64, ValType::I32, ValType::I64, ValType::I64];
    let test = Block {
        params: loop_params.clone(),
        insts: vec![Inst::IntCmp {
            ty: IntTy::I64,
            op: CmpOp::LtU,
            a: 3,
            b: 2,
        }],
        term: Terminator::BrIf {
            cond: 4,
            then_blk: 2,
            then_args: vec![0, 1, 2, 3],
            else_blk: 3,
            else_args: vec![],
        },
    };
    // body(dst=0, byte=1, len=2, i=3): dst[i] = byte; br loop(dst, byte, len, i+1)
    let body = Block {
        params: loop_params,
        insts: vec![
            Inst::IntBin {
                ty: IntTy::I64,
                op: BinOp::Add,
                a: 0,
                b: 3,
            }, // v4 = dst + i
            Inst::Store {
                op: StoreOp::I32_8,
                addr: 4,
                value: 1,
                offset: 0,
                align: 0,
            },
            Inst::ConstI64(1), // v5
            Inst::IntBin {
                ty: IntTy::I64,
                op: BinOp::Add,
                a: 3,
                b: 5,
            }, // v6 = i + 1
        ],
        term: Terminator::Br {
            target: 1,
            args: vec![0, 1, 2, 6],
        },
    };
    let done = Block {
        params: vec![],
        insts: vec![],
        term: Terminator::Return(vec![]),
    };
    Func {
        params,
        results: vec![],
        blocks: vec![entry, test, body, done],
    }
}

/// Synthesize `__svm_memcpy(dst:i64, src:i64, len:i64)`: a counted byte loop
/// `for (i=0; i<len; i++) dst[i] = src[i]` (forward — caller guarantees no overlap, as `memcpy` does).
fn synth_memcpy() -> Func {
    use svm_ir::{LoadOp, StoreOp};
    let params = vec![ValType::I64, ValType::I64, ValType::I64];
    // block0(dst=0, src=1, len=2): i = 0; br loop(dst, src, len, i)
    let entry = Block {
        params: params.clone(),
        insts: vec![Inst::ConstI64(0)],
        term: Terminator::Br {
            target: 1,
            args: vec![0, 1, 2, 3],
        },
    };
    let loop_params = vec![ValType::I64, ValType::I64, ValType::I64, ValType::I64];
    // loop(dst=0, src=1, len=2, i=3): cond = i <u len; br_if cond body done
    let test = Block {
        params: loop_params.clone(),
        insts: vec![Inst::IntCmp {
            ty: IntTy::I64,
            op: CmpOp::LtU,
            a: 3,
            b: 2,
        }],
        term: Terminator::BrIf {
            cond: 4,
            then_blk: 2,
            then_args: vec![0, 1, 2, 3],
            else_blk: 3,
            else_args: vec![],
        },
    };
    // body(dst=0, src=1, len=2, i=3): dst[i] = src[i]; br loop(dst, src, len, i+1)
    let body = Block {
        params: loop_params,
        insts: vec![
            Inst::IntBin {
                ty: IntTy::I64,
                op: BinOp::Add,
                a: 1,
                b: 3,
            }, // v4 = src + i
            Inst::Load {
                op: LoadOp::I32_8U,
                addr: 4,
                offset: 0,
                align: 0,
            }, // v5 = src[i]
            Inst::IntBin {
                ty: IntTy::I64,
                op: BinOp::Add,
                a: 0,
                b: 3,
            }, // v6 = dst + i
            Inst::Store {
                op: StoreOp::I32_8,
                addr: 6,
                value: 5,
                offset: 0,
                align: 0,
            },
            Inst::ConstI64(1), // v7
            Inst::IntBin {
                ty: IntTy::I64,
                op: BinOp::Add,
                a: 3,
                b: 7,
            }, // v8 = i + 1
        ],
        term: Terminator::Br {
            target: 1,
            args: vec![0, 1, 2, 8],
        },
    };
    let done = Block {
        params: vec![],
        insts: vec![],
        term: Terminator::Return(vec![]),
    };
    Func {
        params,
        results: vec![],
        blocks: vec![entry, test, body, done],
    }
}

/// The global-variable name a pointer operand points at, if it is a direct `@global` reference (the
/// shape clang passes a string literal to `puts`/`fputs`). A computed pointer returns `None`.
fn global_name_of(op: &Operand) -> Option<String> {
    match op {
        Operand::ConstantOperand(c) => match c.as_ref() {
            Constant::GlobalReference { name, .. } => Some(name_str(name)),
            _ => None,
        },
        _ => None,
    }
}

/// Lower a call to an external libc function bound to a host capability (§7): the primitive
/// `write`/`read`/`exit`, or a stdio **output wrapper** that funnels to `Stream.write` on stdout —
/// `puts` (string + newline), `putchar`/`putc`/`fputc` (one byte via the stash scratch),
/// `fwrite`/`fputs` (a `size×nmemb` slice / a string), and `fflush` (a no-op — the `Stream` is
/// unbuffered). The libc `FILE*`/`fd` stream argument is ignored: the powerbox handle (always
/// stdout here) selects the endpoint. Returns `Ok(false)` if `name` is not a bound function (so it
/// stays an ordinary call). Result-fidelity notes (the *stdout bytes* are exact regardless): `putc`
/// yields the written char, `fwrite` the item count `nmemb`, `puts`/`fputs`/`fflush` a `0` success.
fn lower_io_call(
    ctx: &mut BlockCtx,
    c: &llvm_ir::instruction::Call,
    name: &str,
) -> Result<bool, Error> {
    // The primitive capability mapping (write/read/exit): drop the dropped args, map the rest.
    if let Some(spec) = cap_spec(name) {
        let import = ctx.import_of(spec.name)?;
        let off = match spec.handle {
            HandleSlot::Stdout => STASH_STDOUT,
            HandleSlot::Stdin => STASH_STDIN,
            HandleSlot::Exit => STASH_EXIT,
        };
        let handle = ctx.stash_load(off);
        let mut args = Vec::new();
        for (a, _attrs) in c.arguments.iter().skip(spec.drop_args) {
            args.push(ctx.operand(a)?);
        }
        let inst = Inst::CallImport {
            import,
            sig: spec.sig,
            handle,
            args,
        };
        // A non-void result (write/read) is a value to bind; `exit` is void (`push_effect`).
        match &c.dest {
            Some(_) => {
                let r = ctx.push(inst);
                ctx.bind_dest(&c.dest, r);
            }
            None => ctx.push_effect(inst),
        }
        return Ok(true);
    }

    match name {
        // `puts(s)` / `fputs(s, stream)`: write the literal's bytes; `puts` then writes a newline.
        // The length comes from the string-literal global (no runtime strlen); a non-literal pointer
        // is a clean `Unsupported` (a runtime strlen loop is a later slice).
        "puts" | "fputs" => {
            let gname = global_name_of(&c.arguments[0].0)
                .ok_or_else(|| Error::Unsupported(format!("`{name}` of a non-literal string")))?;
            let &addr = ctx.globals.get(&gname).ok_or_else(|| {
                Error::Unsupported(format!("`{name}` of unknown global `@{gname}`"))
            })?;
            let &len = ctx.cstrs.get(&gname).ok_or_else(|| {
                Error::Unsupported(format!("`{name}` of non-string global `@{gname}`"))
            })?;
            let buf = ctx.const_i64(addr as i64);
            let n = ctx.const_i64(len as i64);
            ctx.emit_write(buf, n)?;
            if name == "puts" {
                // puts appends a newline (this is why `printf("…\n")` lowers to `puts`).
                let nl = ctx.push(Inst::ConstI32(b'\n' as i32));
                let scratch = ctx.scratch_byte(nl);
                let one = ctx.const_i64(1);
                ctx.emit_write(scratch, one)?;
            }
            // Both return a non-negative success — 0 suffices.
            let r = ctx.push(Inst::ConstI32(0));
            ctx.bind_dest(&c.dest, r);
            Ok(true)
        }
        // `putchar(c)` / `putc(c, stream)` / `fputc(c, stream)`: write the low byte of `c`.
        "putchar" | "putc" | "fputc" => {
            let ch = ctx.operand(&c.arguments[0].0)?;
            let scratch = ctx.scratch_byte(ch);
            let one = ctx.const_i64(1);
            ctx.emit_write(scratch, one)?;
            // Returns the written char (the low byte). Re-materialize from the input value.
            ctx.bind_dest(&c.dest, ch);
            Ok(true)
        }
        // `fwrite(buf, size, nmemb, stream)`: write `size*nmemb` bytes; return `nmemb` (items).
        "fwrite" => {
            let buf = ctx.operand(&c.arguments[0].0)?;
            let size = ctx.operand(&c.arguments[1].0)?;
            let nmemb = ctx.operand(&c.arguments[2].0)?;
            let len = ctx.mul_i64(size, nmemb);
            ctx.emit_write(buf, len)?;
            ctx.bind_dest(&c.dest, nmemb);
            Ok(true)
        }
        // `fflush(stream)`: the `Stream` capability is unbuffered, so a flush is a no-op returning 0.
        "fflush" | "fflush_unlocked" => {
            if c.dest.is_some() {
                let r = ctx.push(Inst::ConstI32(0));
                ctx.bind_dest(&c.dest, r);
            }
            Ok(true)
        }
        // `malloc(size)` / `calloc(n, size)`: the synthesized `vm_map`-growing bump allocator. The heap
        // never reuses and freshly-committed pages are zeroed, so returned memory is zero — hence
        // `calloc` is just `malloc` of `n*size` with no explicit clear.
        "malloc" | "calloc" => {
            let Some(f) = ctx.helpers.malloc else {
                return Ok(false); // no allocator synthesized (e.g. no powerbox entry) → fail-closed
            };
            let size = if name == "calloc" {
                let n = ctx.operand(&c.arguments[0].0)?;
                let sz = ctx.operand(&c.arguments[1].0)?;
                ctx.mul_i64(n, sz)
            } else {
                ctx.operand(&c.arguments[0].0)?
            };
            let r = ctx.push(Inst::Call {
                func: f,
                args: vec![size],
            });
            ctx.bind_dest(&c.dest, r);
            Ok(true)
        }
        // `free(ptr)`: the bump allocator never reclaims, so this is a no-op.
        "free" => Ok(true),
        _ => Ok(false),
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

/// Lower an integer min/max or bit intrinsic to inline ops: `llvm.smax`/`smin`/`umax`/`umin` →
/// `icmp`+`select`; `llvm.ctlz`/`cttz`/`ctpop` → the `clz`/`ctz`/`popcnt` unary op (the trailing
/// `is_*_poison` `i1` arg is ignored — SVM defines the zero case); `llvm.abs` → `select(x<0,-x,x)`.
/// Returns `Ok(None)` for any other call.
fn lower_int_intrinsic(
    ctx: &mut BlockCtx,
    c: &llvm_ir::instruction::Call,
    types: &Types,
) -> Result<Option<ValIdx>, Error> {
    let Some(name) = callee_name(c) else {
        return Ok(None);
    };
    let base = name.rsplit_once('.').map_or(name.as_str(), |(b, _)| b); // drop the `.iN` suffix
    if !matches!(
        base,
        "llvm.smax"
            | "llvm.smin"
            | "llvm.umax"
            | "llvm.umin"
            | "llvm.ctlz"
            | "llvm.cttz"
            | "llvm.ctpop"
            | "llvm.abs"
            | "llvm.fshl"
            | "llvm.fshr"
    ) {
        return Ok(None);
    }
    let args: Vec<&Operand> = c.arguments.iter().map(|(a, _)| a).collect();
    let ty = int_ty(val_type(args[0].get_type(types).as_ref())?)?;
    let cmp_select = |ctx: &mut BlockCtx, op: CmpOp| -> Result<ValIdx, Error> {
        let a = ctx.operand(args[0])?;
        let b = ctx.operand(args[1])?;
        let cond = ctx.push(Inst::IntCmp { ty, op, a, b });
        Ok(ctx.push(Inst::Select { cond, a, b }))
    };
    let unop = |ctx: &mut BlockCtx, op: IntUnOp| -> Result<ValIdx, Error> {
        let a = ctx.operand(args[0])?;
        Ok(ctx.push(Inst::IntUn { ty, op, a }))
    };
    let idx = match base {
        "llvm.smax" => cmp_select(ctx, CmpOp::GtS)?,
        "llvm.smin" => cmp_select(ctx, CmpOp::LtS)?,
        "llvm.umax" => cmp_select(ctx, CmpOp::GtU)?,
        "llvm.umin" => cmp_select(ctx, CmpOp::LtU)?,
        "llvm.ctlz" => unop(ctx, IntUnOp::Clz)?,
        "llvm.cttz" => unop(ctx, IntUnOp::Ctz)?,
        "llvm.ctpop" => unop(ctx, IntUnOp::Popcnt)?,
        // abs(x) = select(x < 0, 0 - x, x).
        "llvm.abs" => {
            let x = ctx.operand(args[0])?;
            let zero = ctx.push(if ty == IntTy::I64 {
                Inst::ConstI64(0)
            } else {
                Inst::ConstI32(0)
            });
            let cond = ctx.push(Inst::IntCmp {
                ty,
                op: CmpOp::LtS,
                a: x,
                b: zero,
            });
            let neg = ctx.push(Inst::IntBin {
                ty,
                op: BinOp::Sub,
                a: zero,
                b: x,
            });
            ctx.push(Inst::Select { cond, a: neg, b: x })
        }
        // `llvm.fshl(a, b, s)` / `fshr`: funnel shift. The **rotate idiom** (the two value operands
        // identical — what clang emits for `(x<<n)|(x>>(w-n))`, e.g. SHA-256's `ROTRIGHT`) lowers to
        // `rotl`/`rotr`, which mask the count mod width and so have no shift-by-`w` edge case. A true
        // funnel shift (distinct operands) needs a width-edge-safe `select` sequence — deferred.
        "llvm.fshl" | "llvm.fshr" => {
            if args[0] != args[1] {
                return unsup(format!("general funnel shift `{name}` (non-rotate)"));
            }
            let a = ctx.operand(args[0])?;
            let amt = ctx.operand(args[2])?;
            let op = if base == "llvm.fshl" {
                BinOp::Rotl
            } else {
                BinOp::Rotr
            };
            ctx.push(Inst::IntBin { ty, op, a, b: amt })
        }
        _ => unreachable!(),
    };
    Ok(Some(idx))
}

/// Lower a call to an external **libm math** function that has a direct SVM float op (`sqrt`,
/// `fabs`, `floor`, `ceil`, `trunc`, `rint`/`nearbyint`, `copysign`, `fmin`, `fmax` and their `…f`
/// f32 variants) to that op inline. Skipped if the guest *defines* a function of that name (then
/// it's an ordinary direct call). `round` (half-away-from-zero) and transcendentals (`sin`/`exp`/…)
/// have no SVM op, so they fall through to the call path (currently `Unsupported`).
fn lower_libm_call(
    ctx: &mut BlockCtx,
    c: &llvm_ir::instruction::Call,
    types: &Types,
) -> Result<Option<ValIdx>, Error> {
    let Some(name) = callee_name(c) else {
        return Ok(None);
    };
    if ctx.name2idx.contains_key(&name) {
        return Ok(None); // a guest-defined function — not the libm intrinsic
    }
    let base = name.strip_suffix('f').unwrap_or(&name); // the f32 variant drops a trailing `f`
    let args: Vec<&Operand> = c.arguments.iter().map(|(a, _)| a).collect();
    let ty = match args.first() {
        Some(a) => match val_type(a.get_type(types).as_ref()) {
            Ok(ValType::F32) | Ok(ValType::F64) => float_ty(val_type(a.get_type(types).as_ref())?)?,
            _ => return Ok(None),
        },
        None => return Ok(None),
    };
    let un = |ctx: &mut BlockCtx, op: FUnOp| -> Result<ValIdx, Error> {
        let a = ctx.operand(args[0])?;
        Ok(ctx.push(Inst::FUn { ty, op, a }))
    };
    let bin = |ctx: &mut BlockCtx, op: FBinOp| -> Result<ValIdx, Error> {
        let a = ctx.operand(args[0])?;
        let b = ctx.operand(args[1])?;
        Ok(ctx.push(Inst::FBin { ty, op, a, b }))
    };
    let idx = match base {
        "sqrt" => un(ctx, FUnOp::Sqrt)?,
        "fabs" => un(ctx, FUnOp::Abs)?,
        "floor" => un(ctx, FUnOp::Floor)?,
        "ceil" => un(ctx, FUnOp::Ceil)?,
        "trunc" => un(ctx, FUnOp::Trunc)?,
        "rint" | "nearbyint" => un(ctx, FUnOp::Nearest)?,
        "copysign" => bin(ctx, FBinOp::Copysign)?,
        "fmin" => bin(ctx, FBinOp::Min)?,
        "fmax" => bin(ctx, FBinOp::Max)?,
        _ => return Ok(None),
    };
    Ok(Some(idx))
}

/// Lower `llvm.memcpy`/`memmove`/`memset` (constant length) to inline chunked load/stores, the way
/// `svm-wasm` lowers `memory.copy`/`fill`. Copies **load all chunks then store all** (overlap-safe,
/// so `memmove` and `memcpy` share a path); `memset` replicates the fill byte across an `i64` and
/// stores it chunk-wide. Returns `Ok(true)` if it handled a (void) mem intrinsic, `Ok(false)`
/// otherwise. A variable or too-large length is a clean `Unsupported`.
/// Lower `llvm.load.relative.iN(ptr P, iN offset)` — clang's **relative lookup table** (used for a
/// `switch` returning string/function constants): the table at `P` holds 32-bit signed offsets
/// `&target − &P`, so the absolute target is `P + sext_i32(*(i32*)(P + offset))`. The table itself
/// (`trunc(sub(ptrtoint …))` initializers) is serialized by [`const_eval`]. Returns the result index.
fn lower_load_relative(
    ctx: &mut BlockCtx,
    c: &llvm_ir::instruction::Call,
) -> Result<Option<ValIdx>, Error> {
    let Some(name) = callee_name(c) else {
        return Ok(None);
    };
    if !name.starts_with("llvm.load.relative") {
        return Ok(None);
    }
    let p = ctx.operand(&c.arguments[0].0)?;
    let off = ctx.operand(&c.arguments[1].0)?;
    let ea = ctx.add_i64(p, off); // address of the i32 table entry
    let raw = ctx.push(Inst::Load {
        op: svm_ir::LoadOp::I32,
        addr: ea,
        offset: 0,
        align: 0,
    });
    let delta = ctx.push(Inst::Convert {
        op: ConvOp::ExtendI32S,
        a: raw,
    }); // sign-extend the relative offset
    Ok(Some(ctx.add_i64(p, delta)))
}

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
    // A non-constant length — or a constant too large to unroll inline — calls the synthesized
    // runtime loop helper (`__svm_memset`/`__svm_memcpy`). A variable-length `memmove` (overlap)
    // has no helper yet (the copy direction needs a runtime branch) — a clean `Unsupported`.
    let len = match const_int(args[2]) {
        Some(n) if n <= MAX_MEM_UNROLL => n,
        _ => {
            let len = ctx.operand(args[2])?;
            if is_set {
                let f = ctx.helpers.memset.expect("memset helper synthesized");
                let dst = ctx.operand(args[0])?;
                let byte = ctx.operand(args[1])?;
                ctx.push_effect(Inst::Call {
                    func: f,
                    args: vec![dst, byte, len],
                });
            } else if name.starts_with("llvm.memcpy") {
                let f = ctx.helpers.memcpy.expect("memcpy helper synthesized");
                let dst = ctx.operand(args[0])?;
                let src = ctx.operand(args[1])?;
                ctx.push_effect(Inst::Call {
                    func: f,
                    args: vec![dst, src, len],
                });
            } else {
                return unsup(format!(
                    "variable-length `{name}` (memmove overlap needs a runtime branch)"
                ));
            }
            return Ok(true);
        }
    };
    if len == 0 {
        return Ok(true);
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
            || s.starts_with("llvm.invariant")
            // Alias-analysis metadata hints (no runtime effect) — e.g. clang's `restrict` scopes.
            || s.starts_with("llvm.experimental.noalias.scope.decl");
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
    /// Import name (`write`/`read`/`exit`) → its §7 import index (for lowering a libc call to
    /// `CallImport`); several libc names can share one import (e.g. `puts`/`fwrite` → `write`).
    caps: &'a HashMap<String, u32>,
    /// String-literal global name → its C-string length (for `puts`/`fputs` write lengths).
    cstrs: &'a HashMap<String, u64>,
    /// Synthesized mem-loop helper indices (for a variable-length `memset`/`memcpy`).
    helpers: Helpers,
    /// The module's type table — for resolving a constexpr-GEP operand's strides in [`operand`].
    types: &'a Types,
    insts: Vec<Inst>,
    idx_of: HashMap<ValueId, ValIdx>,
    /// Aggregate SSA values (a small by-value struct), tracked field-wise: value-id → its scalar
    /// fields' block-local indices. Built by a multi-result `call`/`insertvalue`, read by
    /// `extractvalue`/`ret` (§3a multi-result). Assumed not to cross block boundaries (clang's
    /// register-coercion pattern produces and consumes them in one block).
    agg: HashMap<ValueId, Vec<ValIdx>>,
    next_val: ValIdx,
}

impl<'a> BlockCtx<'a> {
    fn push(&mut self, inst: Inst) -> ValIdx {
        self.insts.push(inst);
        let i = self.next_val;
        self.next_val += 1;
        i
    }

    /// Append an instruction producing **`n` results** (a multi-result `call`, §3a) and return their
    /// `n` consecutive block-local indices.
    fn push_multi(&mut self, inst: Inst, n: usize) -> Vec<ValIdx> {
        self.insts.push(inst);
        let start = self.next_val;
        self.next_val += n as ValIdx;
        (start..self.next_val).collect()
    }

    /// The field indices of an aggregate-typed operand (a value built by a multi-result `call` or
    /// `insertvalue`), or `None` if `op` is not a tracked aggregate value.
    fn agg_of(&self, op: &Operand) -> Option<Vec<ValIdx>> {
        if let Operand::LocalOperand { name, .. } = op {
            let vid = *self.s.name2id.get(name)?;
            return self.agg.get(&vid).cloned();
        }
        None
    }

    /// The data-SP's block-local index (always parameter 0 of every block, §3d).
    fn sp(&self) -> Result<ValIdx, Error> {
        self.id(SP)
    }

    /// Load a powerbox capability handle (`i32`) from its stash slot in the reserved low window.
    fn stash_load(&mut self, off: u64) -> ValIdx {
        let addr = self.const_i64(off as i64);
        self.push(Inst::Load {
            op: svm_ir::LoadOp::I32,
            addr,
            offset: 0,
            align: 0,
        })
    }

    /// The §7 import index for an import name (registered by `collect_cap_imports`).
    fn import_of(&self, name: &str) -> Result<u32, Error> {
        self.caps
            .get(name)
            .copied()
            .ok_or_else(|| Error::Unsupported(format!("capability `{name}` import not registered")))
    }

    /// Emit a `Stream.write(buf, len)` on the stdout handle (a `CallImport`); returns the result
    /// (bytes written). Used by `write` and every stdio output wrapper.
    fn emit_write(&mut self, buf: ValIdx, len: ValIdx) -> Result<ValIdx, Error> {
        let import = self.import_of("write")?;
        let handle = self.stash_load(STASH_STDOUT);
        Ok(self.push(Inst::CallImport {
            import,
            sig: import_sig("write"),
            handle,
            args: vec![buf, len],
        }))
    }

    /// Store a single byte `value` (`i32`-typed) into the 1-byte stash scratch and return its window
    /// address — the staging point a `Stream.write(scratch, 1)` then sends (for `putc`/the newline).
    fn scratch_byte(&mut self, value: ValIdx) -> ValIdx {
        let addr = self.const_i64(STASH_SCRATCH as i64);
        self.push_effect(Inst::Store {
            op: svm_ir::StoreOp::I32_8,
            addr,
            value,
            offset: 0,
            align: 0,
        });
        addr
    }

    /// Append an instruction that produces **no** SSA value (e.g. `store`). It must not consume a
    /// block-local value index — the verifier/interpreter number only value-producing insts (§3a).
    fn push_effect(&mut self, inst: Inst) {
        self.insts.push(inst);
    }

    /// Bind a call's LLVM result name to a block-local value (its translated result).
    fn bind_dest(&mut self, dest: &Option<Name>, r: ValIdx) {
        if let Some(d) = dest {
            if let Some(&vid) = self.s.name2id.get(d) {
                self.idx_of.insert(vid, r);
            }
        }
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
                // A constexpr interior pointer (`&".str"[k]`, `&g.field`) — fold to its constant
                // window address (base global + type-walked offset), like the `getelementptr` instr.
                Constant::GetElementPtr(_) => {
                    let v = const_eval(c.as_ref(), self.globals, self.name2idx, self.types)?;
                    Ok(self.push(Inst::ConstI64(v)))
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
    caps: &HashMap<String, u32>,
    cstrs: &HashMap<String, u64>,
    helpers: &Helpers,
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
        caps,
        cstrs,
        helpers: *helpers,
        types,
        insts: Vec::new(),
        idx_of: HashMap::new(),
        agg: HashMap::new(),
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
        // A call to an external libm-math function with a direct SVM op (`sqrt`/`floor`/…) lowers
        // inline (only when the guest hasn't defined its own function of that name).
        if let Some(idx) = lower_libm_call(ctx, c, types)? {
            if let Some(dest) = &c.dest {
                if let Some(&vid) = ctx.s.name2id.get(dest) {
                    ctx.idx_of.insert(vid, idx);
                }
            }
            return Ok(());
        }
        // Integer min/max + bit intrinsics (`llvm.smax`/`umin`/`ctlz`/`ctpop`/`abs`/…) lower inline.
        if let Some(idx) = lower_int_intrinsic(ctx, c, types)? {
            if let Some(dest) = &c.dest {
                if let Some(&vid) = ctx.s.name2id.get(dest) {
                    ctx.idx_of.insert(vid, idx);
                }
            }
            return Ok(());
        }
        // `llvm.load.relative` (clang's relative lookup table) → load the 32-bit relative offset and
        // add it back to the table base.
        if let Some(idx) = lower_load_relative(ctx, c)? {
            if let Some(dest) = &c.dest {
                if let Some(&vid) = ctx.s.name2id.get(dest) {
                    ctx.idx_of.insert(vid, idx);
                }
            }
            return Ok(());
        }
        // A call to an external libc function bound to a host capability (§7): the primitive
        // `write`/`read`/`exit`, or a stdio output wrapper (`puts`/`putc`/`fwrite`/…). All lower to
        // `Stream`/`Exit` capability calls (`CallImport`) the embedder resolves at load. A
        // guest-*defined* function of the same name is never a capability — it falls through to the
        // direct-call path below.
        if let Some(name) = callee_name(c) {
            if !ctx.name2idx.contains_key(&name) && lower_io_call(ctx, c, &name)? {
                return Ok(());
            }
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
                let ty = indirect_sig(c, types)?;
                Inst::CallIndirect { ty, idx, args }
            }
        };
        // The result: a small by-value struct return is a **multi-result** (§3a) recorded field-wise
        // in the aggregate table; a scalar is one value; `void` is none.
        let result_ty = match c.function_ty.as_ref() {
            Type::FuncType { result_type, .. } => result_type.clone(),
            other => return unsup(format!("call through non-function type {other}")),
        };
        let agg_fields = match struct_field_vtypes(result_ty.as_ref(), types) {
            Some(r) => Some(r?),
            None => None,
        };
        match (&c.dest, agg_fields) {
            (Some(dest), Some(fields)) => {
                let rs = ctx.push_multi(inst, fields.len());
                if let Some(&vid) = ctx.s.name2id.get(dest) {
                    ctx.agg.insert(vid, rs);
                }
            }
            (Some(dest), None) => {
                let r = ctx.push(inst);
                if let Some(&vid) = ctx.s.name2id.get(dest) {
                    ctx.idx_of.insert(vid, r);
                }
            }
            (None, _) => ctx.push_effect(inst), // void call: no SSA result
        }
        return Ok(());
    }
    // `insertvalue` builds a small by-value struct field-wise (no scalar result) — record/update its
    // field list in the aggregate side-table. The source is a prior aggregate value or a
    // poison/undef/zero constant (start from zeroed fields). Single-level only (clang's coercion).
    if let I::InsertValue(iv) = instr {
        if iv.indices.len() != 1 {
            return unsup("nested insertvalue");
        }
        let i = iv.indices[0] as usize;
        let mut fields = match ctx.agg_of(&iv.aggregate) {
            Some(f) => f,
            None => {
                let aty = iv.aggregate.get_type(types);
                let ftys = struct_field_vtypes(aty.as_ref(), types)
                    .ok_or_else(|| Error::Unsupported("insertvalue into non-struct".into()))??;
                ftys.into_iter().map(|t| ctx.push(zero_inst(t))).collect()
            }
        };
        let v = ctx.operand(&iv.element)?;
        *fields
            .get_mut(i)
            .ok_or_else(|| Error::Unsupported("insertvalue index out of range".into()))? = v;
        if let Some(&vid) = ctx.s.name2id.get(&iv.dest) {
            ctx.agg.insert(vid, fields);
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
        // Pointers are an `i64` window offset in our model (§3a/§10), so `ptr`↔`int` is a width
        // adjust, never a reinterpret: `ptrtoint` truncates the `i64` pointer to the target width
        // (identity at `i64`); `inttoptr` zero-extends a narrow integer up to the `i64` pointer.
        I::PtrToInt(x) => {
            let to = int_bits(x.to_type.as_ref())
                .ok_or_else(|| Error::Unsupported("ptrtoint to non-int".into()))?;
            let v = ctx.operand(&x.operand)?;
            (&x.dest, emit_trunc(ctx, v, 64, to))
        }
        I::IntToPtr(x) => {
            let from = src_bits(&x.operand, types)?;
            let v = ctx.operand(&x.operand)?;
            let r = if from >= 64 {
                v // already i64 — identity
            } else {
                emit_ext(ctx, v, from, 64, false)
            };
            (&x.dest, r)
        }
        // `freeze` pins a would-be poison/undef to a fixed value. Our IR is total — `undef`/`poison`
        // already resolve to a defined 0 (§3c) and no poison propagates — so `freeze` is an identity.
        I::Freeze(x) => {
            let a = ctx.operand(&x.operand)?;
            (&x.dest, a)
        }
        // `extractvalue` reads a field of a small by-value struct — alias the field's value (§3a).
        I::ExtractValue(ev) => {
            if ev.indices.len() != 1 {
                return unsup("nested extractvalue");
            }
            let fields = ctx
                .agg_of(&ev.aggregate)
                .ok_or_else(|| Error::Unsupported("extractvalue of non-aggregate value".into()))?;
            let v = *fields
                .get(ev.indices[0] as usize)
                .ok_or_else(|| Error::Unsupported("extractvalue index out of range".into()))?;
            (&ev.dest, v)
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
            // A small by-value struct return yields its fields (§3a multi-result); a scalar, one value.
            Some(op) => match ctx.agg_of(op) {
                Some(fields) => Ok(Terminator::Return(fields)),
                None => Ok(Terminator::Return(vec![ctx.operand(op)?])),
            },
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
