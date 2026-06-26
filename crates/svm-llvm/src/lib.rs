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
//!   and the common float math intrinsics (`fma` → the shared fused-FMA op, `fmuladd` unfused, `sqrt`/`fabs`/`floor`/…) lowered
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
//!   lower to `rotl`/`rotr` for the rotate idiom (identical operands — SHA-256's `ROTRIGHT`); the
//!   general (distinct-operand) case with a **constant** amount on an i32/i64 lowers to
//!   `(a << s) | (b >>u (w - s))` (Embench `aha-mont64`'s double-word `modul64` shift). A
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
//! - **U — narrow-signed `icmp` fix (lands tinfl).** A **narrow** (`i8`/`i16`) operand of a signed
//!   `icmp` is sign-extended to `i32` first (§3b: a narrow value sits in an `i32` container with
//!   unspecified high bits — e.g. a zero-extended `i16` load of a *signed* value, where `< 0` would
//!   otherwise always be false). This corrects tinfl's Huffman slow-path (`mz_int16` table entry
//!   `< 0`), which had produced a corrupt back-reference pointer.
//! - **V — 2-lane vectors (`<2 x float>`/`<2 x i32>`), lands clay → the full corpus.** A 2-lane
//!   32-bit vector (clang's `Clay_Vector2`/2D-point coercion) is **scalarized to a packed `i64`**
//!   (lane 0 = bits 0–31, lane 1 = 32–63 — its little-endian image), so it flows through
//!   `phi`/`call`/`ret`/`load`/`store`/block-params as an ordinary `i64`. Only the vector *ops*
//!   unpack/repack lanes: `extractelement`/`insertelement`, lane-wise `fadd`/`fsub`/`fmul`/`fdiv`,
//!   `shufflevector` (constant mask), and vector constants; a `bitcast` between 2-lane vectors is a
//!   no-op (same packed `i64`). Lands **clay** (UI layout) byte-identical to native — the **8th of 8
//!   corpus demos**, meeting the D54 "matches native clang" exit criterion.
//!
//! Beyond the corpus (general-C breadth, demo-driven — see `LLVM.md`):
//! - **W — varargs `printf` (a guest-side format engine).** A `printf(fmt, …)` with a **constant**
//!   format string is parsed at translate time: literal runs are written straight from the format
//!   global, and each conversion lowers to the synthesized `__svm_utoa` (int→ASCII) + width/zero-pad
//!   (a buffer pre-fill) → `Stream.write`. Unsigned `%u`/`%x`, `%c`, `%%`, field width and the `0`
//!   flag, and length modifiers (the LLVM arg carries the real width). All formatting runs **in the
//!   guest**; only the bytes cross the boundary. Lands the **`hexdump`** demo byte-identical to native.
//! - **X — `realloc` + signed `printf` (`%d`).** `malloc` now writes a 16-byte **size header** before
//!   the data (keeping it 16-aligned), so `realloc(p, n)` recovers the old size, `malloc`s `n`, copies
//!   `min(old, n)` bytes (`__svm_realloc` → `__svm_malloc` + `__svm_memcpy`; `realloc(NULL,…)` ≡
//!   `malloc`). `printf` gains signed `%d`/`%i` (sign computed, magnitude via `__svm_utoa`, `-`
//!   prepended) with plain/space-padded fields. Lands the **`sortvec`** demo (a `realloc`-doubling
//!   vector + insertion sort) byte-identical to native.
//! - **Y — 128-bit SIMD (`<4 x float>` → native `v128`).** A 4-lane 32-bit vector maps to SVM's §17
//!   `v128`: `load`/`store` → `v128.load`/`store`; `fadd`/`fmul`/… → `f32x4` `VFloatBin`;
//!   `extractelement`/`insertelement` → extract/replace lane; `shufflevector` → an `i8x16.shuffle`
//!   byte mask (the all-equal mask is a splat); vector constants → `ConstV128`; `llvm.fmuladd.v4f32`
//!   → `f32x4` mul+add (unfused; `llvm.fma.v4f32` → the shared `Inst::VFma`). (2-lane vectors stay scalarized to `i64` — they're 8 bytes.) Lands
//!   the **`mat4`** demo (a 4×4 × vec4 transform) byte-identical to native.
//! - **Z — `llvm.bswap`.** Byte-reverse synthesized inline (no SVM op): each source byte `i` →
//!   destination byte `nbytes-1-i` via shift/mask/or (`i16`/`i32`/`i64`). Lands the **`crc32`** demo
//!   (CRC-32 + a big-endian `u32` reader) byte-identical to native.
//! - **AA — overlap-safe `memmove` (a direction-aware runtime helper).** A variable-length (or
//!   oversized-constant) `llvm.memmove` calls the synthesized **`__svm_memmove`** — a counted byte
//!   copy that runs *forward* when `dst <= src` and *backward* otherwise (the direction `memcpy`
//!   can't do), so overlapping shifts in either direction are correct. (Constant small `memmove`
//!   still inlines load-all-then-store-all, already overlap-safe.) Lands the **`lineedit`** demo (a
//!   line editor doing overlapping left/right shifts) byte-identical to native.
//! - **AB — transcendental libm, bundled as guest code.** Math beyond the SVM float ops (`sin`/
//!   `cos`/`exp`/`pow`/…) is supplied *by the program* as ordinary guest C (polynomial
//!   approximations) — no new lowering, and no host math capability (the on-ramp keeps math in the
//!   sandbox). This is the key to a clean differential: native `cc` compiles the same guest `libm`,
//!   so every value is bit-identical (the only machine ops in play — `sqrt`/`floor` (slices F/L),
//!   `fmuladd` (unfused), `fma` (fused, shared `Inst::VFma`/`Fma`), `+−*∕` — are IEEE on both sides). Lands the **`raytrace`** demo (an ASCII
//!   sphere raytracer: `sqrt` intersection + guest `g_sin`/`g_exp` shading) byte-identical to native.
//!
//! Out of the current subset (clean [`Error::Unsupported`]): `printf` float conversions
//! (`%f`/`%e`/`%g` — need exact-decimal/bignum formatting), `*` (dynamic width/precision), and
//! non-constant formats; general (non-rotate) funnel shifts with a *non-constant* amount (the
//! constant-amount i32/i64 case is lowered), `llvm.bitreverse`, transcendental math
//! as *external* libm calls (the program must supply it as guest code — see slice AB), other SIMD
//! (`<2 x double>`, `<8 x i16>`, dynamic lanes), and `i33`.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use llvm_ir::debugloc::{DebugLoc, HasDebugLoc};
use llvm_ir::instruction::Instruction;
use llvm_ir::terminator::Terminator as LTerm;
use llvm_ir::types::{FPType, Type, Typed, Types};
use llvm_ir::{constant::Constant, constant::Float, BasicBlock, Function, Module as LModule};
use llvm_ir::{FPPredicate, IntPredicate, Name, Operand};

use svm_ir::{
    AtomicRmwOp, BinOp, Block, CastOp, CmpOp, ConvOp, DebugInfo, FBinOp, FCmpOp, FToI, FUnOp,
    FloatTy, Func, FuncName, IToF, Inst, IntTy, IntUnOp, LoadOp, Loc, Module, Ordering, SsaLoc,
    StoreOp, Terminator, TypeDef, ValIdx, ValType, VarInfo, VarLoc,
};

pub mod blockaddr;
pub mod di;
/// The in-house textual-`.ll` reader (LLVM.md §8 Q1a) — replacing the `llvm-ir`/libLLVM binding.
/// Built behind a differential parity check before it becomes the default; see `tests/translate.rs`.
pub mod ll;
pub mod wideint;

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
    /// Exported function symbols: each *defined* function's name paired with its final index in
    /// `module.funcs` (`base + i`, accounting for a synthesized `_start` prologue). This feeds a
    /// `svm_ir::LinkUnit.exports` so a separate program module can resolve a `call.import` of these
    /// names through `svm_ir::link` — the separate-artifact path (compile a runtime once, link many
    /// programs against it). Synthesized helpers (`_start`, `memset`, `malloc`, …) carry no source
    /// name and are not exported.
    pub exports: Vec<(String, u32)>,
}

/// Translate a legalized LLVM bitcode file (`*.bc`). The bitcode must come from the pinned LLVM
/// (18); off-version input is an [`Error::Parse`].
///
/// A `-g` build additionally feeds the §6 debug-info waist's *variable/type half*: the structured
/// `!DILocalVariable`/DI-type graph (which the `llvm-ir` AST doesn't expose) is read by a direct
/// `llvm-sys` walk ([`di`]) over the same file and correlated to the IR ([`di::read_debug`]).
pub fn translate_bc_path(path: impl AsRef<Path>) -> Result<Translated, Error> {
    let path = path.as_ref();
    // I14 fail-closed guard: `llvm-ir` 0.11.3 truncates a `bits > 64` integer constant to its low
    // word, so a wide/negative `i128` literal would silently miscompile. We can't recover the high
    // word from the truncated AST, so reject such a module up front (clean `Unsupported`, never a
    // miscompile). Constants that fit in `[0, 2^64)` round-trip exactly and are unaffected.
    if let Some(c) = path.to_str().and_then(wideint::out_of_range_constant) {
        return unsup(format!(
            "wide integer constant `{c}`: a ≥2⁶⁴ / negative i128 literal is not supported \
             (`llvm-ir` truncates it; fail-closed to avoid a miscompile — ISSUES.md I14)"
        ));
    }
    let m = LModule::from_bc_path(path).map_err(Error::Parse)?;
    let di = path.to_str().and_then(di::read_debug);
    // Computed-`goto` support needs the `blockaddress` operands `llvm-ir` erases (see [`blockaddr`]).
    let ba = path.to_str().and_then(blockaddr::read_block_addrs);
    translate_impl(&m, di.as_ref(), ba.as_ref())
}

/// Translate an already-parsed `llvm-ir` module. The neutral core's source-line half is populated
/// from `!DILocation`; the variable/type half requires the bitcode path (see [`translate_bc_path`]).
pub fn translate(m: &LModule) -> Result<Translated, Error> {
    translate_impl(m, None, None)
}

fn translate_impl(
    m: &LModule,
    di: Option<&di::LlvmDebug>,
    ba: Option<&blockaddr::BlockAddrs>,
) -> Result<Translated, Error> {
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
    // Direct `<svm.h>` Memory-capability builtins (`__vm_map`/`unmap`/`protect`/`page_size`): register
    // their §7 imports and remember the program needs the `Memory` handle granted, even with no `malloc`.
    let uses_vm_memory = register_vm_memory_imports(m, &defined_names, &mut imports, &mut caps);
    // §9/§12 async-ring builtins (`__vm_io_submit_async`/`__vm_io_reap`): register their `IoRing`
    // imports; `__vm_blocking_handle` only reads the stashed `Blocking` handle (no import). Together
    // they raise the powerbox arity to grant the `IoRing`/`Blocking` handles.
    let uses_vm_io = register_vm_io_imports(m, &defined_names, &mut imports, &mut caps);
    let uses_blocking = calls_external(m, &defined_names, "__vm_blocking_handle") && has_main;
    // §22 guest-driven-JIT builtins (`__vm_jit_*`): register their `Jit` imports; the program then
    // needs the full 8-handle powerbox (the `Jit` handle is the last `VM_CAP_*` index).
    let uses_vm_jit = register_vm_jit_imports(m, &defined_names, &mut imports, &mut caps);
    // §13/§14 SharedRegion builtins (`__vm_region_*`): register their imports; `__vm_region_create`
    // mints from the `AddressSpace` handle, so the program needs it granted (slot 4).
    let uses_vm_region = register_vm_region_imports(m, &defined_names, &mut imports, &mut caps);
    // `realloc` is a synthesized helper built on `malloc` + `memcpy`, so it forces both on.
    let need_realloc = calls_external(m, &defined_names, "realloc") && has_main;
    let need_malloc = (needs_malloc(m, &defined_names) || need_realloc) && has_main;
    // The `Memory` handle (4th powerbox grant) is needed by the allocator *and* the direct Memory
    // builtins; the heap is seeded only for `malloc`.
    let need_memory_cap = (need_malloc || uses_vm_memory) && has_main;
    // `printf` is lowered inline (a guest-side format engine → `Stream.write`); it pulls in the
    // `__svm_utoa` helper and (via `cap_import_name`) the `write` import, so it also forces a powerbox.
    let need_printf = calls_external(m, &defined_names, "printf") && has_main;
    // `snprintf(buf, size, fmt, …)` reuses the entire `printf` format engine ([`lower_format`]) with
    // output redirected into `buf` (the [`FmtSink`] path of `emit_write`) — so it needs the same
    // synthesized helpers: `utoa` (`%d`), the bignum `dtoa` family (`%f`/`%g`/`%e`), `strlen` (`%s`),
    // and `memcpy` (the per-segment buffer copy). Unlike `printf` it writes no stdout, so it needs no
    // `write` import / powerbox `main`.
    let need_snprintf = calls_external(m, &defined_names, "snprintf");
    // A direct `strlen` call routes to the same synthesized `__svm_strlen` byte loop that `printf %s`
    // uses. Unlike `printf` it needs no powerbox/`main` (it only reads guest memory), so a `run`-only
    // module — e.g. an Embench kernel compiled without `main` — can call it; hence *not* `&& has_main`.
    let need_strlen = need_printf || need_snprintf || calls_external(m, &defined_names, "strlen");
    // `strcmp` plus its C-locale alias `strcoll` share one synthesized byte-compare helper; `strchr`
    // its own byte scan (the §varargs/libc batch for real-program targets like Lua).
    let need_strcmp =
        calls_external(m, &defined_names, "strcmp") || calls_external(m, &defined_names, "strcoll");
    let need_strchr = calls_external(m, &defined_names, "strchr");
    let need_strcpy = calls_external(m, &defined_names, "strcpy");
    let need_strspn = calls_external(m, &defined_names, "strspn");
    let need_strpbrk = calls_external(m, &defined_names, "strpbrk");
    let need_ldexp =
        calls_external(m, &defined_names, "ldexp") || calls_external(m, &defined_names, "scalbn");
    let need_pow = calls_external(m, &defined_names, "pow");
    let need_fmod = calls_external(m, &defined_names, "fmod");
    let need_frexp = calls_external(m, &defined_names, "frexp");
    let need_strtod = calls_external(m, &defined_names, "strtod");
    let need_localeconv = calls_external(m, &defined_names, "localeconv");
    let need_errno = calls_external(m, &defined_names, "__errno_location");
    let need_time = calls_external(m, &defined_names, "time");
    // `getenv` is a synthesized helper that scans the §3e blob's env strings directly. It needs no
    // capability or import of its own, but it *does* need the powerbox window (the blob lives in the
    // reserved low scratch), so it forces a `_start` below.
    let need_getenv = calls_external(m, &defined_names, "getenv") && has_main;
    // `%f` formatting (`__svm_dtoa_fixed`) rides on `printf`: it writes via the same `write` import and
    // stashed stdout handle. Synthesized for any `printf` program (dead if no `%f` appears — scanning
    // the formats to tighten this is a later refinement).
    let need_dtoa = need_printf || need_snprintf;
    // C++ exception handling (Itanium ABI on-ramp). `need_eh` reserves the EH state region + drives
    // the `invoke`/`landingpad`/`resume`/`__cxa_*` lowering; the typeinfo-id table assigns each
    // `@_ZTI*` referenced by a throw or `llvm.eh.typeid.for` a distinct nonzero id so the thrown-type
    // selector and the catch-clause `eh.typeid.for` agree. Needs the powerbox window (so `&& has_main`).
    let (uses_eh, eh_typeids, eh_thrown) = scan_eh(m);
    // Precompute the subtype match table so `catch (Base&)` matches a thrown `Derived` (§ polymorphic
    // catch). Empty when EH is unused.
    let eh_subtype_ids = build_eh_subtypes(m, &eh_typeids, &eh_thrown);
    let need_eh = uses_eh && has_main;
    let need_narrow_atomic = uses_narrow_atomic(m);
    // The powerbox is granted a **contiguous prefix** of the `VM_CAP_*` handles (the runner grants by
    // declared arity), sized to the highest capability index the program uses: exit(2) always,
    // memory(3) for `malloc`/Memory builtins, addrspace(4) for the SharedRegion builtins, ioring(5)
    // for the async ring, blocking(6) for `__vm_blocking_handle`, jit(7) for the guest-driven-JIT
    // builtins.
    let max_cap_index = if uses_vm_jit {
        7
    } else if uses_blocking {
        6
    } else if uses_vm_io {
        5
    } else if uses_vm_region {
        4
    } else if need_memory_cap {
        3
    } else {
        2
    };
    let n_handles = max_cap_index + 1;
    // A powerbox entry is synthesized when the program needs the handle stash: it uses a named import,
    // `malloc`, or a stash-only builtin (`__vm_blocking_handle`, which adds no import of its own).
    // C++ static init: a program with `@llvm.global_ctors` needs a `_start` that runs the ctors before
    // `main` (the on-ramp otherwise jumps straight to `main`), so it forces a powerbox entry too.
    let has_global_ctors = m
        .global_vars
        .iter()
        .any(|g| name_str(&g.name) == "llvm.global_ctors");
    // `snprintf` writes the format scratch (`FMT_BUF`, via `utoa`/`dtoa`) — page 0 of the **writable**
    // low scratch. It must force the powerbox layout so the globals start one page up (`STACK_PAGE`,
    // below): otherwise a read-only global (the constant format string) shares page 0 with `FMT_BUF`
    // and D40 page-granular protection makes `utoa`'s scratch writes fault. (`printf` already forces
    // this via its `write` import; `snprintf` has no import of its own, hence the explicit term.)
    let needs_powerbox_entry = !imports.is_empty()
        || need_malloc
        || uses_blocking
        || has_global_ctors
        || need_getenv
        || need_snprintf;
    let synth = needs_powerbox_entry && has_main;
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
    // LLVM **function aliases** (`@x = alias <ty>, ptr @y`): identical-code folding — both LLVM's
    // own pass and Rust's cross-crate dedup — collapses functions with byte-identical bodies into one
    // definition plus an alias to it (e.g. svm-ir's `VIntUnOp::index` and `VPMinMaxOp::index`, both
    // 2-variant enum→byte, become one body + an alias). An alias has no body, so a `call`/`ref.func`
    // to it would otherwise look like an undefined external. Register each alias whose aliasee is a
    // defined function under that function's index, so it resolves like the function it names. A
    // fixpoint loop covers alias→alias chains; aliases to data globals are skipped (aliasee ∉
    // `name2idx`, which holds only functions).
    loop {
        let mut progressed = false;
        for ga in &m.global_aliases {
            let Name::Name(alias_name) = &ga.name else {
                continue;
            };
            let alias_name = alias_name.to_string();
            if name2idx.contains_key(&alias_name) {
                continue;
            }
            if let Some(target) = alias_target_name(ga.aliasee.as_ref()) {
                if let Some(&idx) = name2idx.get(&target) {
                    name2idx.insert(alias_name, idx);
                    progressed = true;
                }
            }
        }
        if !progressed {
            break;
        }
    }
    // Globals live low (from `DATA_BASE`); the data stack starts just above them. For a powerbox
    // program the writable **handle stash + allocator/format state** occupies the reserved low scratch
    // (page 0, below `STACK_PAGE`), so start the globals one page up (`STACK_PAGE`): a *read-only*
    // global (D40, protected page-granularly) must never share a page with the stash, or `_start`'s
    // handle stores would fault on the read-only page (the same page-isolation the data stack gets).
    let globals_base = if synth { STACK_PAGE } else { DATA_BASE };
    let (globals, mut data, mut globals_end, cstrs, gbytes) =
        globals_layout(m, &name2idx, globals_base, ba)?;
    // Synthesize the glibc ctype tables (flags + lower/upper case maps) as **read-only data in the
    // module image** when the program calls the ctype locators (`isalpha`/`isspace`/`tolower`/… lower to
    // `__ctype_b_loc`/`__ctype_tolower_loc`/`__ctype_toupper_loc`, e.g. Embench `slre`). Placed after the
    // globals (and below the data stack) so a `run`-only module needs no `_start` to initialize them.
    let ctype = build_ctype_data(m, &defined_names, &mut data, &mut globals_end);
    // Page-align the data stack above the globals so it never shares a page with a *read-only*
    // global (D40 protects RO segments page-granularly — a stack write into a shared page would
    // fault). 16 KiB covers the largest common page size (macOS/aarch64). (A read-only and a
    // writable global sharing a page is a separate latent issue — page-isolating those is a follow-up.)
    let entry_sp = globals_end.div_ceil(STACK_PAGE) * STACK_PAGE;

    // Synthesized helpers (mem-loop `memset`/`memcpy`, the `malloc` allocator) sit after the defined
    // functions and `_start` (index 0 when `synth`), at `base + defined.len()` onward — their indices
    // are fixed before translating call sites. The allocator references the `vm_map` import index.
    let (need_memset, need_memcpy0, need_memmove) = needs_mem_helpers(m);
    let need_memcpy = need_memcpy0 || need_realloc || need_snprintf; // `realloc`/`snprintf` copy via `__svm_memcpy`
                                                                     // `memcmp`/`bcmp` (Rust slice equality + `BTreeMap` key ordering) → the synthesized `__svm_memcmp`.
                                                                     // A pure address helper (no capability), so unlike `malloc` it needs no powerbox/`has_main`.
    let need_memcmp =
        calls_external(m, &defined_names, "memcmp") || calls_external(m, &defined_names, "bcmp");
    // `memchr(s, c, n)` (string/buffer scans — e.g. Embench `slre`) → the synthesized `__svm_memchr`
    // byte loop. Like `memcmp`, a pure address helper (no powerbox/`has_main`).
    let need_memchr = calls_external(m, &defined_names, "memchr");
    // `__cxa_end_catch` destroys the caught exception object via `__svm_eh_destroy` (gated on EH being
    // active, so the helper can address the EH region and the dtor funcref resolves through the table).
    let need_eh_destroy = need_eh && calls_external(m, &defined_names, "__cxa_end_catch");
    // `__svm_eh_unwind` backs every throw/rethrow/resume — present whenever EH is active.
    let need_eh_unwind = need_eh;
    // i128 `udiv`/`sdiv`/`urem`/`srem` lower to the synthesized 128÷128 long-division helper.
    let need_idiv128 = uses_i128_divrem(m);
    // Helper indices are assigned in a fixed order after the defined functions (and `_start`):
    // memset, memcpy, malloc, utoa, realloc, memmove — each present only if needed. The append
    // order below must match.
    let mut next_helper = base + defined.len() as u32;
    let mut take = |needed: bool| {
        needed.then(|| {
            let i = next_helper;
            next_helper += 1;
            i
        })
    };
    // The float scratch sits just above the data-stack reserve (computed here so it can ride in
    // `Helpers` to the printf lowering, and drive the window sizing + helper append below).
    let float_scratch_base = need_dtoa.then_some(entry_sp + STACK_RESERVE);
    // The C++ EH region sits just above the float scratch (or directly above the stack reserve when
    // there is no float scratch), reserved only when `need_eh`. Rides in `Helpers` to the
    // `invoke`/`landingpad`/`resume`/`__cxa_*` lowerings.
    let eh_base = need_eh.then(|| {
        let mut b = entry_sp + STACK_RESERVE;
        if need_dtoa {
            b += FLOAT_SCRATCH_SIZE;
        }
        b
    });
    let helpers = Helpers {
        memset: take(need_memset),
        memcpy: take(need_memcpy),
        malloc: take(need_malloc),
        utoa: take(need_printf || need_snprintf),
        // `%s` needs a runtime strlen (synthesized alongside `utoa` for any `printf`); a direct
        // `strlen` call also routes here — `need_strlen` covers both (see above).
        strlen: take(need_strlen),
        realloc: take(need_realloc),
        memmove: take(need_memmove),
        getenv: take(need_getenv),
        big_zero: take(need_dtoa),
        big_copy: take(need_dtoa),
        big_cmp: take(need_dtoa),
        big_sub: take(need_dtoa),
        big_mul: take(need_dtoa),
        big_shl: take(need_dtoa),
        big_shr1: take(need_dtoa),
        big_inc: take(need_dtoa),
        big_divmod: take(need_dtoa),
        big_iszero: take(need_dtoa),
        dtoa_digits: take(need_dtoa),
        dtoa_sci: take(need_dtoa),
        dtoa_gen: take(need_dtoa),
        dtoa_fix: take(need_dtoa),
        atomic_rmw_narrow: take(need_narrow_atomic),
        atomic_cas_narrow: take(need_narrow_atomic),
        memcmp: take(need_memcmp),
        memchr: take(need_memchr),
        eh_destroy: take(need_eh_destroy),
        eh_unwind: take(need_eh_unwind),
        udivmod128: take(need_idiv128),
        // The libc string batch — appended last; the matching `funcs.push` order below mirrors this.
        strcmp: take(need_strcmp),
        strchr: take(need_strchr),
        strcpy: take(need_strcpy),
        strspn: take(need_strspn),
        strpbrk: take(need_strpbrk),
        ldexp: take(need_ldexp),
        pow_stub: take(need_pow),
        fmod_stub: take(need_fmod),
        frexp: take(need_frexp),
        strtod_stub: take(need_strtod),
        localeconv_stub: take(need_localeconv),
        errno_stub: take(need_errno),
        time_zero: take(need_time),
        float_scratch: float_scratch_base,
        ctype_b_loc: ctype.b_loc,
        ctype_tolower_loc: ctype.tolower_loc,
        ctype_toupper_loc: ctype.toupper_loc,
        eh_base,
        eh_typeids,
        eh_subtype_ids,
    };

    let mut funcs = Vec::with_capacity(defined.len() + synth as usize);
    let mut any_frame = false; // does any function use the data stack (`alloca`)?
                               // §6 debug-info: each defined function's final index is `base + i` (the synth `_start`, inserted
                               // at 0 below, shifts them up by `base`; the appended helpers carry no source). Source positions
                               // come from each LLVM instruction's `!DILocation`; the structured type graph (when a `-g`
                               // bitcode path was walked by the `di` reader) seeds the shared `types` table up front, its
                               // `TypeId`s referenced unchanged by the per-function variables.
    let mut dbg = DebugAcc::default();
    if let Some(di) = di {
        dbg.types = di.types.clone();
    }
    for (i, f) in defined.iter().enumerate() {
        let (func, frame_size) = translate_func(
            f,
            base + i as u32,
            i as u32,
            &m.types,
            &name2idx,
            &globals,
            &caps,
            &cstrs,
            &gbytes,
            &helpers,
            di,
            ba,
            &mut dbg,
        )?;
        any_frame |= frame_size > 0;
        funcs.push(func);
    }

    // §6 module-scoped globals: a source global at a fixed window address. The `di` reader gives the
    // LLVM symbol; `globals_layout` placed it at `globals[symbol]`, so emit a `GLOBAL_SCOPE`
    // `VarLoc::Fixed` var (visible in every frame). A global with no laid-out address is skipped.
    if let Some(di) = di {
        for gv in &di.globals {
            if let Some(&addr) = globals.get(&gv.symbol) {
                dbg.vars.push(VarInfo {
                    func: svm_ir::GLOBAL_SCOPE,
                    name: gv.name.clone(),
                    ty: gv.ty.clone(),
                    loc: VarLoc::Fixed { addr },
                    type_id: gv.type_id,
                    scope: None,
                });
            }
        }
    }

    // Does the program's `main` take `argc`/`argv` (so the synthesized entry is `synth_start_argv`)?
    // Its IR arity is the threaded SP + the C params: 1 ⇒ `main(void)`, 3 ⇒ `main(int, char**)`,
    // 4 ⇒ `main(int, char**, char** envp)`; a 2-param `main` is a fail-closed error. Computed here (not
    // just at the `synth_start` call) because the argv `_start` *uses* the data stack — it parks
    // `argv[]` (and `envp[]`) at the entry SP and relocates `main`'s frame a page above — so the
    // window must reserve stack space.
    let main_arity = if synth {
        funcs[(name2idx["main"] - base) as usize].params.len()
    } else {
        1
    };
    if !matches!(main_arity, 1 | 3 | 4) {
        return Err(Error::Unsupported(format!(
            "main with {} parameter(s): only main(void), main(int, char**), and \
             main(int, char**, char** envp) are supported",
            main_arity - 1
        )));
    }
    // `argc`/`argv` for a 3- *or* 4-param `main`; the 4-param form additionally gets `envp` (the
    // §3e blob's env strings parsed into a second `char**` array parked above `argv[]`).
    let wants_argv = main_arity == 3 || main_arity == 4;
    let wants_envp = main_arity == 4;

    // The window: globals low, then the data stack from `entry_sp` growing up; `mapped` covers the
    // globals plus a stack reserve, with a faulting guard beyond (reserved > mapped, §5). Declared if
    // any function uses the data stack, the module has globals, or it uses the powerbox (the handle
    // stash / heap state live in the reserved low window).
    let need_window = any_frame || !globals.is_empty() || synth || ctype.any() || eh_base.is_some();
    let memory = need_window.then(|| {
        // Reserve stack when any function (or the argv `_start`) uses the data stack.
        let mut top = if any_frame || wants_argv {
            entry_sp + STACK_RESERVE
        } else {
            globals_end
        }
        .max(1);
        if let Some(fsb) = float_scratch_base {
            top = top.max(fsb + FLOAT_SCRATCH_SIZE);
        }
        if let Some(eb) = eh_base {
            top = top.max(eb + EH_REGION_SIZE);
        }
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
        // The C++ global constructors (`@llvm.global_ctors`) `_start` runs before `main` (their
        // funcrefs resolve through `name2idx`, now built).
        let ctors = collect_global_ctors(m, &name2idx)?;
        // Both entries share `StartBuilder`; `synth_start_argv` adds the §3e args-buffer → `argv[]`
        // parsing for a `main(int, char**)`, `synth_start` is the plain `main(void)` entry.
        let build_start: StartBuilder = if wants_argv {
            synth_start_argv
        } else {
            synth_start
        };
        let start = build_start(
            main_idx,
            &main_results,
            entry_sp,
            n_handles,
            heap_base,
            &ctors,
            wants_envp,
        );
        funcs.insert(0, start);
    }
    // Append the synthesized helpers in index order (memset, memcpy, malloc, utoa, realloc) —
    // matching the indices assigned above.
    if need_memset {
        funcs.push(synth_memset());
    }
    if need_memcpy {
        funcs.push(synth_memcpy());
    }
    if need_malloc {
        funcs.push(synth_malloc(caps["vm_map"]));
    }
    if need_printf || need_snprintf {
        funcs.push(synth_utoa());
    }
    if need_strlen {
        funcs.push(synth_strlen());
    }
    if need_realloc {
        funcs.push(synth_realloc(
            helpers.malloc.expect("realloc needs malloc"),
            helpers.memcpy.expect("realloc needs memcpy"),
        ));
    }
    if need_memmove {
        funcs.push(synth_memmove());
    }
    if need_getenv {
        funcs.push(synth_getenv());
    }
    if need_dtoa {
        // The bignum float family, appended in `Helpers` order; the formatters are built against the
        // primitives' / `dtoa_digits`'s indices (all interdependent).
        funcs.push(synth_big_zero());
        funcs.push(synth_big_copy());
        funcs.push(synth_big_cmp());
        funcs.push(synth_big_sub());
        funcs.push(synth_big_mul_small());
        funcs.push(synth_big_shl_bits());
        funcs.push(synth_big_shr1());
        funcs.push(synth_big_inc());
        funcs.push(synth_big_divmod10());
        funcs.push(synth_big_is_zero());
        funcs.push(synth_dtoa_digits(
            helpers.big_zero.unwrap(),
            helpers.big_copy.unwrap(),
            helpers.big_cmp.unwrap(),
            helpers.big_sub.unwrap(),
            helpers.big_mul.unwrap(),
            helpers.big_shl.unwrap(),
        ));
        funcs.push(synth_dtoa_sci(helpers.dtoa_digits.unwrap()));
        funcs.push(synth_dtoa_gen(helpers.dtoa_digits.unwrap()));
        funcs.push(synth_dtoa_fix_big(
            helpers.big_zero.unwrap(),
            helpers.big_mul.unwrap(),
            helpers.big_shl.unwrap(),
            helpers.big_shr1.unwrap(),
            helpers.big_inc.unwrap(),
            helpers.big_divmod.unwrap(),
            helpers.big_iszero.unwrap(),
        ));
    }
    // Narrow-atomic CAS-loop helpers (appended after the bignum family — matches the `take` order so
    // the recorded indices line up). Self-contained: they touch only the passed window addresses.
    if need_narrow_atomic {
        funcs.push(synth_atomic_rmw_narrow());
        funcs.push(synth_atomic_cas_narrow());
    }
    // `__svm_memcmp` (after the atomics — matches the `take` order). Self-contained: it touches only
    // the two passed window addresses.
    if need_memcmp {
        funcs.push(synth_memcmp());
    }
    if need_memchr {
        funcs.push(synth_memchr());
    }
    // `__svm_eh_destroy` (after `__svm_memchr` — matches the `take` order). Self-contained: it reads
    // only its arguments and indirect-calls the passed destructor funcref.
    if need_eh_destroy {
        funcs.push(synth_eh_destroy());
    }
    // `__svm_eh_unwind` (after `__svm_eh_destroy` — matches the `take` order). Self-contained: it
    // touches only the EH region addressed off the passed `base` and the handler checkpoints.
    if need_eh_unwind {
        funcs.push(synth_eh_unwind());
    }
    // `__svm_udivmod128` (after `__svm_eh_unwind` — matches the `take` order). Self-contained: it
    // touches no memory, only its four i64 operands.
    if need_idiv128 {
        funcs.push(synth_udivmod128());
    }
    // The libc string batch — appended last, mirroring the `take()` order in `Helpers` above.
    if need_strcmp {
        funcs.push(synth_strcmp());
    }
    if need_strchr {
        funcs.push(synth_strchr());
    }
    if need_strcpy {
        funcs.push(synth_strcpy());
    }
    if need_strspn {
        funcs.push(synth_strspn());
    }
    if need_strpbrk {
        funcs.push(synth_strpbrk());
    }
    if need_ldexp {
        funcs.push(synth_ldexp());
    }
    if need_pow {
        funcs.push(synth_trap_stub(
            vec![ValType::F64, ValType::F64],
            vec![ValType::F64],
        ));
    }
    if need_fmod {
        // Exact-synthesizable but a large loop-nest CFG — stubbed pending its own slice (see `Helpers`).
        funcs.push(synth_trap_stub(
            vec![ValType::F64, ValType::F64],
            vec![ValType::F64],
        ));
    }
    if need_frexp {
        funcs.push(synth_frexp());
    }
    if need_strtod {
        // strtod(const char*, char**) -> double.
        funcs.push(synth_trap_stub(
            vec![ValType::I64, ValType::I64],
            vec![ValType::F64],
        ));
    }
    if need_localeconv {
        // localeconv(void) -> struct lconv*.
        funcs.push(synth_trap_stub(vec![], vec![ValType::I64]));
    }
    if need_errno {
        // __errno_location(void) -> int*.
        funcs.push(synth_trap_stub(vec![], vec![ValType::I64]));
    }
    if need_time {
        // time(time_t*) -> time_t — returns 0 (seed value is result-irrelevant; see synth_const_i64).
        funcs.push(synth_const_i64(vec![ValType::I64], 0));
    }
    Ok(Translated {
        module: Module {
            funcs,
            memory,
            data,
            // §7 named capability imports (`write`/`read`/`exit` …) the host resolves at load
            // (`resolve_capability_imports`); empty for a pure-compute (kernel) module.
            imports,
            // First-class function exports: each defined function's name → its final `module.funcs`
            // index, so a C-compiled module is name-addressable (`call("main")`) like the wasm path.
            // Mirrors the out-of-band `Translated::exports` (the `.syms` sidecar source).
            exports: defined
                .iter()
                .enumerate()
                .map(|(i, f)| svm_ir::Export {
                    name: f.name.clone(),
                    func: base + i as u32,
                })
                .collect(),
            // §6 debug-info waist: the source-line half, mapped from each LLVM `!DILocation` (the
            // variable/type half is blocked on the `llvm-ir` metadata reader — see `DebugAcc`).
            // `None` for a non-`-g` build (no instruction carried a location).
            debug_info: dbg.finish(),
        },
        entry_sp,
        // Each defined function's name → its final `module.funcs` index (`base + i`), the same
        // mapping `name2idx` holds, emitted in defined order for determinism.
        exports: defined
            .iter()
            .enumerate()
            .map(|(i, f)| (f.name.clone(), base + i as u32))
            .collect(),
    })
}

/// The low window offset where globals begin (kept off a null-like 0).
const DATA_BASE: u64 = 16;
/// The page granularity the data stack is aligned to above the globals (≥ the largest OS page so
/// a stack write never lands in a read-only global's protected page, D40). For a powerbox program
/// this is also the globals base, so `[0, STACK_PAGE)` is the reserved low scratch — the handle
/// stash, allocator/format state, and the §3e args buffer all live there. The powerbox layout is a
/// public ABI ([`svm_ir::POWERBOX_STACK_PAGE`]), shared with the frontend-independent
/// [`svm_ir::synth_powerbox_start`] so the two `_start` synthesizers stay byte-identical.
const STACK_PAGE: u64 = svm_ir::POWERBOX_STACK_PAGE;
// The §3e powerbox args buffer (`svm_ir::POWERBOX_ARGS_BASE..POWERBOX_ARGS_END`) must sit *above*
// the frontend's format/scratch region and *below* the globals base, so it never overlaps either.
const _: () = assert!(svm_ir::POWERBOX_ARGS_BASE >= FMT_BUF_END);
const _: () = assert!(svm_ir::POWERBOX_ARGS_END == STACK_PAGE);
/// The data-stack reserve (bytes) above the entry SP before the guard region — a stack overflow
/// past this faults rather than escaping the window ([`svm_ir::POWERBOX_STACK_RESERVE`]).
const STACK_RESERVE: u64 = svm_ir::POWERBOX_STACK_RESERVE;

/// The data-SP's synthetic value id — threaded as block-local index 0 of *every* block (§3d),
/// like chibicc's `v0`. It carries no LLVM name; it is supplied positionally.
const SP: ValueId = usize::MAX;

/// A sentinel `frame` key (never a real SSA value) holding the `sp`-relative offset of this
/// function's **outgoing-varargs marshaling scratch** (§varargs). A call to a `(...)` function
/// stores its variadic arguments into 8-byte slots starting here, then hands the callee a pointer
/// to it via the callee's reserved frame slot (offset 0). Present in `frame` only when the function
/// makes at least one direct varargs call. See `frame_layout` / the varargs call-site lowering.
const VARARG_SCRATCH: ValueId = usize::MAX - 1;

/// Accumulates the §6 debug-info **neutral core** as functions are lowered — the LLVM on-ramp as a
/// third independent producer feeding the frontend-neutral waist (DEBUGGING.md §6 / D-DBG-7).
///
/// Source positions come straight from each LLVM instruction's `!DILocation` (`HasDebugLoc`), keyed
/// onto the SVM `(func, block, inst)` pc it lowered to. This is the **source-line half** of the
/// waist — the analog of the wasm producer's `.debug_line` ingest (W4 slice 15), demonstrating the
/// waist is genuinely frontend-neutral across a third frontend.
///
/// The **variable / type half** (`llvm.dbg.value` → `VarLoc` location lists, the DI type graph →
/// `TypeDef`) is *not* reachable here: the pinned `llvm-ir` reader (0.11.3) leaves the structured
/// metadata graph unimplemented (`Metadata::from_llvm_ref` is `unimplemented!`, `MetadataOperand`
/// is payloadless), so the `DILocalVariable`/`DIType` nodes never reach this crate. Recovering them
/// needs a metadata-capable reader (a direct `llvm-sys` DI walk) — its own effort (see `LLVM.md`).
#[derive(Default)]
struct DebugAcc {
    files: Vec<String>,
    file_idx: HashMap<String, u32>,
    locs: Vec<Loc>,
    /// The structured `TypeDef` graph (from the `di` reader; empty without `-g` variables).
    types: Vec<TypeDef>,
    /// Source variables (the `dbg.declare` → `Window` half; empty without `-g` / on the `translate`
    /// entry, which has no bitcode path to walk).
    vars: Vec<VarInfo>,
    /// Source function names (`DISubprogram` `DW_AT_name` → IR function index): the §6 function-name
    /// table, so an LLVM-frontend backtrace reads `compute` instead of `fn{N}`.
    func_names: Vec<FuncName>,
}

impl DebugAcc {
    /// Intern a `DebugLoc`'s file path (directory + filename joined, mirroring `DebugLoc`'s own
    /// display join) into the `files` table, returning its index.
    fn intern_file(&mut self, dl: &DebugLoc) -> u32 {
        let path = match &dl.directory {
            Some(d) if !d.is_empty() && !dl.filename.starts_with('/') => {
                format!("{d}/{}", dl.filename)
            }
            _ => dl.filename.clone(),
        };
        if let Some(&i) = self.file_idx.get(&path) {
            return i;
        }
        let i = self.files.len() as u32;
        self.file_idx.insert(path.clone(), i);
        self.files.push(path);
        i
    }

    /// Record that the SVM ops at `func`/`block`/`inst in [start, end)` came from `dl`'s source
    /// position (one `Loc` per op — the per-op granularity the `Loc` table models).
    fn map_range(&mut self, func: u32, block: u32, start: usize, end: usize, dl: &DebugLoc) {
        let file = self.intern_file(dl);
        for inst in start..end {
            self.locs.push(Loc {
                func,
                block,
                inst: inst as u32,
                file,
                line: dl.line,
                col: dl.col.unwrap_or(0),
            });
        }
    }

    /// The populated waist, or `None` when nothing was recorded (a non-`-g` build) — so a debug-free
    /// module stays `debug_info: None`, byte-identical to before.
    fn finish(self) -> Option<DebugInfo> {
        if self.locs.is_empty() && self.vars.is_empty() && self.func_names.is_empty() {
            None
        } else {
            Some(DebugInfo {
                files: self.files,
                locs: self.locs,
                types: self.types,
                vars: self.vars,
                func_names: self.func_names,
                ..Default::default()
            })
        }
    }
}

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
        // A `blockaddress` is pointer-width (its `get_type` is the unsized `label` type, so it must be
        // special-cased here rather than falling through to `type_size`).
        Constant::BlockAddress => Ok(8),
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
    ba: &mut impl Iterator<Item = u32>,
) -> Result<Vec<u8>, Error> {
    match c {
        Constant::Int { bits, value } if *bits <= 64 => {
            let n = (*bits as usize).div_ceil(8).max(1);
            Ok(value.to_le_bytes()[..n].to_vec())
        }
        Constant::Float(Float::Single(f)) => Ok(f.to_bits().to_le_bytes().to_vec()),
        Constant::Float(Float::Double(d)) => Ok(d.to_bits().to_le_bytes().to_vec()),
        // A `blockaddress(@f, %bb)` (a computed-`goto` label table entry): emit the recovered block
        // index (8 LE bytes — pointer width) the `indirectbr` consumes as a `br_table` index. The
        // labels arrive in this same DFS order from [`blockaddr`]; an empty feed (no recovery, e.g. the
        // path-less `translate` entry) is a clean fail-closed `Unsupported`.
        Constant::BlockAddress => {
            let label = ba.next().ok_or_else(|| {
                Error::Unsupported(
                    "blockaddress without a recovered label (needs the .bc path)".into(),
                )
            })?;
            Ok((label as u64).to_le_bytes().to_vec())
        }
        Constant::Array { elements, .. } | Constant::Vector(elements) => {
            let mut out = Vec::new();
            for e in elements {
                out.extend(const_bytes(e.as_ref(), types, globals, funcs, ba)?);
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
                let b = const_bytes(v.as_ref(), types, globals, funcs, ba)?;
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
    HashMap<String, Vec<u8>>,
);

/// Lay out the module's global variables in the window's globals region (from `base`, each
/// natural-aligned), returning the name → window-address map, the `data` segments to emit
/// (constants read-only, §3a/D40; all-zero/BSS globals just reserve space in the zero-init
/// window), the region's end (for window sizing), and the string-literal lengths (for `puts`).
fn globals_layout(
    m: &LModule,
    name2idx: &HashMap<String, u32>,
    base: u64,
    ba: Option<&blockaddr::BlockAddrs>,
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
                // LLVM-reserved globals (`llvm.global_ctors`/`global_dtors`/`used`/`compiler.used`)
                // are metadata, never real window data — they are handled out of band (the ctors run
                // in `_start`, the rest are dropped), so never lay them out / serialize them.
                if name_str(&g.name).starts_with("llvm.") {
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
    let mut gbytes = HashMap::new();
    for (gi, at) in placed {
        let g = &m.global_vars[gi];
        let Some(init) = &g.initializer else { continue }; // BSS / extern → zero-init window
                                                           // The `blockaddress` labels this global's initializer holds (in DFS order — see [`blockaddr`]);
                                                           // the serializer pops them as it reaches each `Constant::BlockAddress` leaf.
        let empty: Vec<u32> = Vec::new();
        let labels = ba
            .and_then(|b| b.per_global.get(&name_str(&g.name)))
            .unwrap_or(&empty);
        let mut feed = labels.iter().copied();
        let bytes = const_bytes(init.as_ref(), &m.types, &addr, name2idx, &mut feed)?;
        // Record the C-string length (up to the first NUL) so `puts`/`fputs` on this literal can
        // write the right slice without a runtime strlen.
        let slen = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len()) as u64;
        cstrs.insert(name_str(&g.name), slen);
        // Keep a constant global's bytes so `printf` can parse a constant format string at translate
        // time (the format engine reads `@.str`'s content here, not at runtime).
        if g.is_constant {
            gbytes.insert(name_str(&g.name), bytes.clone());
        }
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
    Ok((addr, segs, off, cstrs, gbytes))
}

/// The pointer-cell window addresses the synthesized glibc ctype locators return (`None` = the program
/// doesn't call that locator). See [`build_ctype_data`].
#[derive(Default, Clone, Copy)]
struct CtypeAddrs {
    b_loc: Option<u64>,
    tolower_loc: Option<u64>,
    toupper_loc: Option<u64>,
}

impl CtypeAddrs {
    /// Did any ctype table get synthesized? (If so the module gained read-only data and therefore needs
    /// a window declared, even if the program itself laid out no globals.)
    fn any(self) -> bool {
        self.b_loc.is_some() || self.tolower_loc.is_some() || self.toupper_loc.is_some()
    }
}

/// The C-locale `<ctype.h>` flag word for byte `c`, in **glibc's little-endian bit layout** — the exact
/// values `clang` masks against (`_ISdigit=0x0800`, `_ISxdigit=0x1000`, `_ISspace=0x2000`, …).
fn ctype_flags(c: u8) -> u16 {
    let (upper, lower, digit) = (
        c.is_ascii_uppercase(),
        c.is_ascii_lowercase(),
        c.is_ascii_digit(),
    );
    let alpha = upper || lower;
    let graph = c.is_ascii_graphic();
    let space = matches!(c, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r');
    let mut f = 0u16;
    if upper {
        f |= 0x0100;
    }
    if lower {
        f |= 0x0200;
    }
    if alpha {
        f |= 0x0400;
    }
    if digit {
        f |= 0x0800;
    }
    if c.is_ascii_hexdigit() {
        f |= 0x1000;
    }
    if space {
        f |= 0x2000;
    }
    if graph || c == b' ' {
        f |= 0x4000; // print = graph ∪ space-char ' '
    }
    if graph {
        f |= 0x8000;
    }
    if matches!(c, b' ' | b'\t') {
        f |= 0x0001; // blank
    }
    if c.is_ascii_control() {
        f |= 0x0002; // cntrl
    }
    if c.is_ascii_punctuation() {
        f |= 0x0004; // punct
    }
    if alpha || digit {
        f |= 0x0008; // alnum
    }
    f
}

/// C-locale `tolower`/`toupper` of an `i32` table index in glibc's `-128..=255` range — identity outside
/// `0..=255` and for non-cased bytes.
fn ctype_case_map(v: i32, to_lower: bool) -> i32 {
    if !(0..=255).contains(&v) {
        return v;
    }
    let c = v as u8;
    if to_lower && c.is_ascii_uppercase() {
        v + 32
    } else if !to_lower && c.is_ascii_lowercase() {
        v - 32
    } else {
        v
    }
}

/// Synthesize the C-locale **ctype tables** as read-only data in the module image, for programs that call
/// the glibc ctype locators (`<ctype.h>`'s `isalpha`/`isspace`/`tolower`/… lower to
/// `(*__ctype_b_loc())[c] & _ISxxx` / `(*__ctype_tolower_loc())[c]`). Each table spans glibc's index range
/// `-128..=255` (384 entries) and the locator returns a pointer **into** it at index 0, so a signed *or*
/// unsigned `char` indexes it safely. Flags are `u16` (glibc LE bit layout); case maps are `i32`. The
/// tables and their indirection pointer cells are appended after the globals as read-only segments (no
/// runtime init — works for a `run`-only module), and `*end` is advanced past them.
fn build_ctype_data(
    m: &LModule,
    defined: &HashMap<String, u32>,
    data: &mut Vec<svm_ir::Data>,
    end: &mut u64,
) -> CtypeAddrs {
    let need_b = calls_external(m, defined, "__ctype_b_loc");
    let need_tl = calls_external(m, defined, "__ctype_tolower_loc");
    let need_tu = calls_external(m, defined, "__ctype_toupper_loc");
    if !(need_b || need_tl || need_tu) {
        return CtypeAddrs::default();
    }
    // Read-only region — page-isolate from any preceding writable global (D40).
    *end = end.div_ceil(STACK_PAGE) * STACK_PAGE;

    // Append a read-only `elem`-byte-per-entry table + its indirection pointer cell (which holds
    // `table + 128*elem`, the index-0 base a signed/unsigned char offsets from); return the cell address.
    fn emit_ctype(bytes: Vec<u8>, elem: u64, data: &mut Vec<svm_ir::Data>, end: &mut u64) -> u64 {
        let table = end.div_ceil(elem) * elem;
        data.push(svm_ir::Data {
            offset: table,
            readonly: true,
            bytes,
        });
        *end = table + 384 * elem;
        let cell = end.div_ceil(8) * 8;
        data.push(svm_ir::Data {
            offset: cell,
            readonly: true,
            bytes: (table + 128 * elem).to_le_bytes().to_vec(),
        });
        *end = cell + 8;
        cell
    }

    let mut addrs = CtypeAddrs::default();
    if need_b {
        let mut b = Vec::with_capacity(384 * 2);
        for j in 0..384i32 {
            let v = j - 128;
            let f = if (0..=255).contains(&v) {
                ctype_flags(v as u8)
            } else {
                0
            };
            b.extend_from_slice(&f.to_le_bytes());
        }
        addrs.b_loc = Some(emit_ctype(b, 2, data, end));
    }
    let case_table = |to_lower: bool, data: &mut Vec<svm_ir::Data>, end: &mut u64| -> u64 {
        let mut b = Vec::with_capacity(384 * 4);
        for j in 0..384i32 {
            b.extend_from_slice(&ctype_case_map(j - 128, to_lower).to_le_bytes());
        }
        emit_ctype(b, 4, data, end)
    };
    if need_tl {
        addrs.tolower_loc = Some(case_table(true, data, end));
    }
    if need_tu {
        addrs.toupper_loc = Some(case_table(false, data, end));
    }
    addrs
}

/// Map an LLVM type to an SVM value type. Narrow integers collapse to `i32` (§3b: `i8`/`i16`
/// are memory widths only, not SSA value types); `i64` stays `i64`. A non-power-of-two width in
/// `33..=64` (LLVM's `-O2` SCEV often produces `i33` etc. closing a loop into a polynomial) is held
/// in an `i64`, kept canonical by masking after the de-normalizing ops (`bin`). `i128`+ is rejected.
fn val_type(ty: &Type) -> Result<ValType, Error> {
    match ty {
        Type::IntegerType { bits } if *bits <= 32 => Ok(ValType::I32),
        Type::IntegerType { bits } if *bits <= 64 => Ok(ValType::I64),
        Type::IntegerType { bits } => unsup(format!("integer width i{bits} (i128+ unsupported)")),
        // Pointers are an erasable refinement of `i64` (§3a/§10) — a window offset.
        Type::PointerType { .. } => Ok(ValType::I64),
        Type::FPType(FPType::Single) => Ok(ValType::F32),
        Type::FPType(FPType::Double) => Ok(ValType::F64),
        // A `<2 x float>` (clang's `Clay_Vector2`-style coercion) is **scalarized to a packed `i64`**
        // (lane 0 = bits 0–31, lane 1 = bits 32–63 — its little-endian memory image). It then flows
        // through `phi`/`call`/`ret`/block-params as an ordinary `i64`; only the vector *ops*
        // (`extractelement`/`insertelement`/`fadd`…) unpack/repack the lanes. SIMD-proper is §17 V128.
        _ if is_vec2(ty) => Ok(ValType::I64),
        // Any 128-bit vector (`i8x16`/`i16x8`/`i32x4`/`i64x2`/`f32x4`/`f64x2`) is a native v128 (§17).
        _ if vec128_shape(ty).is_some() => Ok(ValType::V128),
        other => unsup(format!("type {other} (Milestone 1+)")),
    }
}

/// The lane type of a **2-lane 32-bit vector** (`<2 x float>` or `<2 x i32>`) — the only vectors the
/// on-ramp scalarizes (packed into an `i64`, lane 0 low). `None` for any other vector.
fn vec2_lane_ty(ty: &Type) -> Option<ValType> {
    match ty {
        Type::VectorType {
            element_type,
            num_elements: 2,
            scalable: false,
        } => match element_type.as_ref() {
            Type::FPType(FPType::Single) => Some(ValType::F32),
            Type::IntegerType { bits: 32 } => Some(ValType::I32),
            _ => None,
        },
        _ => None,
    }
}

/// Is `ty` a scalarizable 2-lane 32-bit vector (`<2 x float>`/`<2 x i32>`)?
fn is_vec2(ty: &Type) -> bool {
    vec2_lane_ty(ty).is_some()
}

/// Is `ty` specifically `<2 x float>` (the vector that takes the lane-wise float-arith path)?
fn is_vec2f(ty: &Type) -> bool {
    vec2_lane_ty(ty) == Some(ValType::F32)
}

/// The 128-bit `VShape` whose lane *type* matches a vector's element type (`i8`→`i8x16`,
/// `i16`→`i16x8`, `i32`→`i32x4`, `i64`→`i64x2`, `float`→`f32x4`, `double`→`f64x2`), **independent of
/// the element count**. `None` for non-vectors, scalable vectors, or an unsupported lane type. This
/// is the lane shape both a single-`v128` vector and a legalized chunk of a wider vector pack into.
fn vec_lane_shape(ty: &Type) -> Option<svm_ir::VShape> {
    use svm_ir::VShape;
    let Type::VectorType {
        element_type,
        scalable: false,
        ..
    } = ty
    else {
        return None;
    };
    Some(match element_type.as_ref() {
        Type::IntegerType { bits: 8 } => VShape::I8x16,
        Type::IntegerType { bits: 16 } => VShape::I16x8,
        Type::IntegerType { bits: 32 } => VShape::I32x4,
        Type::IntegerType { bits: 64 } => VShape::I64x2,
        // A pointer lane is an `i64` window offset (§3a/§10), so a pointer vector packs exactly like an
        // `i64` vector — `<2 x ptr>` ≡ `<2 x i64>` (an `i64x2` v128). Lets a verbatim pointer-pair
        // load/store (SLP-vectorized struct/list copy, e.g. Embench `sglib-combined`) ride the v128 path.
        Type::PointerType { .. } => VShape::I64x2,
        Type::FPType(FPType::Single) => VShape::F32x4,
        Type::FPType(FPType::Double) => VShape::F64x2,
        _ => return None,
    })
}

/// The `VShape` of a **128-bit-wide LLVM vector** — any of the six legal-width shapes
/// `<16 x i8>`/`<8 x i16>`/`<4 x i32>`/`<2 x i64>`/`<4 x float>`/`<2 x double>`, each mapping to a
/// native `v128` (§17/D58). `None` for any vector that is not exactly 16 bytes (a 2-lane 32-bit
/// vector is 8 bytes and takes the packed-`i64` path; wider-than-128 vectors are split into chunks
/// by the legalization pass). The lane shape is carried per-op, so this is the single source of
/// truth for "this LLVM vector is one `v128`".
fn vec128_shape(ty: &Type) -> Option<svm_ir::VShape> {
    let shape = vec_lane_shape(ty)?;
    match ty {
        Type::VectorType { num_elements, .. } if *num_elements == shape.lanes() as usize => {
            Some(shape)
        }
        _ => None,
    }
}

/// How an LLVM vector that is **not a single `v128`** legalizes to fixed-128 chunks (I2 fix-sketch
/// step 1): the lane [`svm_ir::VShape`] its lanes pack into, the number of full 16-byte `v128`
/// chunks, and the count of leftover scalar **tail** lanes — so
/// `total_lanes = full_chunks * shape.lanes() + tail_lanes`. A wider-than-128 vector has
/// `full_chunks ≥ 1`; a sub-128 one (e.g. `<8 x i8>`) has `full_chunks == 0` and is fully
/// scalarized into the tail (a 16-byte `v128.load` would overrun its memory image, so its lanes are
/// per-element loads/ops). `None` for a single `v128` (use [`vec128_shape`]), a non-vector, a
/// scalable vector, an unsupported lane type, or the empty vector.
#[derive(Clone, Copy)]
struct WideLayout {
    shape: svm_ir::VShape,
    full_chunks: usize,
    tail_lanes: usize,
}

impl WideLayout {
    /// Total lane count (`full_chunks * lanes_per_chunk + tail_lanes`).
    fn total_lanes(self) -> usize {
        self.full_chunks * self.shape.lanes() as usize + self.tail_lanes
    }
    /// Number of legalized parts (chunk `v128`s + tail lane scalars) a value of this layout holds.
    fn nparts(self) -> usize {
        self.full_chunks + self.tail_lanes
    }
    /// The byte size of the vector's memory image.
    fn byte_size(self) -> u64 {
        self.total_lanes() as u64 * self.shape.lane_bytes() as u64
    }
}

fn wide_vec_layout(ty: &Type) -> Option<WideLayout> {
    if vec128_shape(ty).is_some() || is_vec2(ty) {
        // A single `v128` keeps its native path; a `<2 x {i32,float}>` keeps its packed-`i64` path.
        return None;
    }
    let shape = vec_lane_shape(ty)?;
    let Type::VectorType {
        num_elements: n, ..
    } = ty
    else {
        return None;
    };
    if *n == 0 {
        return None;
    }
    let lpc = shape.lanes() as usize;
    Some(WideLayout {
        shape,
        full_chunks: n / lpc,
        tail_lanes: n % lpc,
    })
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

/// Emit a float compare, returning its `i32` (0/1) result. The ordered/unordered NaN predicates have
/// no single svm-ir op, so they expand: `uno` (either operand NaN) = `(a != a) | (b != b)` and `ord`
/// (neither NaN) = `(a == a) & (b == b)` — `x != x` is true iff `x` is NaN. `true`/`false` fold to a
/// constant. Everything else is the direct [`fcmp_op`] mapping. Rust's float code (`is_nan`, `min`/
/// `max`, partial compares) emits these.
fn emit_fcmp(
    ctx: &mut BlockCtx,
    ty: FloatTy,
    p: FPPredicate,
    a: ValIdx,
    b: ValIdx,
) -> Result<ValIdx, Error> {
    use FPPredicate as P;
    Ok(match p {
        P::UNO => {
            let na = ctx.push(Inst::FCmp {
                ty,
                op: FCmpOp::Ne,
                a,
                b: a,
            });
            let nb = ctx.push(Inst::FCmp {
                ty,
                op: FCmpOp::Ne,
                a: b,
                b,
            });
            ctx.push(Inst::IntBin {
                ty: IntTy::I32,
                op: BinOp::Or,
                a: na,
                b: nb,
            })
        }
        P::ORD => {
            let aa = ctx.push(Inst::FCmp {
                ty,
                op: FCmpOp::Eq,
                a,
                b: a,
            });
            let bb = ctx.push(Inst::FCmp {
                ty,
                op: FCmpOp::Eq,
                a: b,
                b,
            });
            ctx.push(Inst::IntBin {
                ty: IntTy::I32,
                op: BinOp::And,
                a: aa,
                b: bb,
            })
        }
        P::True => ctx.push(Inst::ConstI32(1)),
        P::False => ctx.push(Inst::ConstI32(0)),
        _ => {
            let op = fcmp_op(p)?;
            ctx.push(Inst::FCmp { ty, op, a, b })
        }
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
        _ if is_vec2(ty) => Ok(8), // `<2 x float>`/`<2 x i32>` — two packed 32-bit lanes
        _ if vec128_shape(ty).is_some() => Ok(16), // any 128-bit vector — a v128
        // A wider/sub-128 vector: its full lane-packed byte image (legalized to v128 chunks + tail).
        _ if wide_vec_layout(ty).is_some() => Ok(wide_vec_layout(ty).unwrap().byte_size()),
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
        _ if is_vec2(ty) => Ok(8), // a 2-lane 32-bit vector is 8-aligned
        _ if vec128_shape(ty).is_some() => Ok(16), // a v128 is 16-aligned
        // A wider/sub-128 vector — 16-aligned when it has full chunks, else its (smaller) byte size.
        _ if wide_vec_layout(ty).is_some() => {
            let l = wide_vec_layout(ty).unwrap();
            Ok(if l.full_chunks > 0 {
                16
            } else {
                l.byte_size().max(1)
            })
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
    // An **empty struct** is size 0 — LLVM lays `type {}` out as zero bytes regardless of source
    // language (it is the layout every `getelementptr` offset is computed against). A Rust **ZST**
    // field — `PhantomData`, `alloc::alloc::Global` (the `Vec`/`RawVec` allocator marker), a unit
    // struct — must therefore contribute 0, not 1: clamping to 1 here inflated every `RawVec` by a
    // byte (→ 24-byte `Vec`s padded to 32, `len` shifted 16→24), desyncing field offsets from the
    // GEPs and corrupting every `Vec::len()`/element access through such a struct.
    Ok((offsets, off, align))
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
    /// `ValueId` → its SVM type. A wide-vector value (tracked in `wide`) carries a `V128` placeholder
    /// here — its real representation is its chunk/tail parts, never a single scalar.
    ty: Vec<ValType>,
    /// `ValueId` → its legalization layout, for values whose LLVM type is a wider-than-128 or
    /// sub-128 vector (I2 step 1). These flow as several `v128` chunks + scalar tail lanes, not one
    /// SSA value; the layout drives the chunk/tail fan-out at block boundaries (`block_params`,
    /// `branch_args`) and the per-chunk lowering (`lower_wide`).
    wide: HashMap<ValueId, WideLayout>,
    /// `ValueId` → the scalar field types of a **small by-value struct** (`{i64, ptr}`, `{i64, i64}`,
    /// …). Like `wide`, these flow as several SSA values (one per field), not one scalar — recorded only
    /// for **flat** structs (scalar fields). Drives the per-field fan-out at block boundaries so a
    /// struct can cross a block edge (a `phi` of a struct, e.g. wikisort's `MakeRange` result); within a
    /// block they are the same field-wise `agg` values built by `insertvalue`/multi-result `call`.
    agg_layout: HashMap<ValueId, Vec<ValType>>,
    /// `ValueId` → the block it is defined in (parameters are defined in the entry block, 0).
    def_block: Vec<usize>,
    /// Block name → block index (entry is 0).
    block_idx: HashMap<Name, usize>,
    /// Block index → block name (for looking up φ incoming-by-predecessor).
    block_name: Vec<Name>,
}

#[allow(clippy::too_many_arguments)]
fn translate_func(
    f: &Function,
    func_idx: u32,
    bc_func_idx: u32,
    types: &Types,
    name2idx: &HashMap<String, u32>,
    globals: &HashMap<String, u64>,
    caps: &HashMap<String, u32>,
    cstrs: &HashMap<String, u64>,
    gbytes: &HashMap<String, Vec<u8>>,
    helpers: &Helpers,
    di: Option<&di::LlvmDebug>,
    ba: Option<&blockaddr::BlockAddrs>,
    dbg: &mut DebugAcc,
) -> Result<(Func, u64), Error> {
    // A `(...)`-defined function (`f.is_var_arg`) lowers like any other: its IR signature is
    // `(sp, fixed-params…)` — the variadic arguments are not IR parameters but are read by `va_start`
    // from the caller-deposited overflow area (§varargs). `frame_layout` reserves the incoming-pointer
    // slot at frame offset 0; the `llvm.va_start` lowering reads it.
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

    // §6 debug-info variables. The `di` reader gives this function's source locals; resolve each to
    // an IR location:
    //   - `dbg.declare` (`-O0`): an alloca *ordinal* → the alloca's data-stack offset (same textual
    //     order `frame_layout` lays out) → `VarLoc::Window`.
    //   - `dbg.value` bound to argument `k` (`-O2`/`-Og`): the argument is ValueId `k`, threaded as
    //     a block parameter wherever it is live; its block-local value index is its position in that
    //     block's param list, so one `SsaLoc` per such block (effective from block entry) gives an
    //     `SsaList` covering the argument's whole live range.
    // §6 function name: the `DISubprogram` source name → this IR function index.
    if let Some(src) = di.and_then(|d| d.func_names.get(&f.name)) {
        dbg.func_names.push(FuncName {
            func: func_idx,
            name: src.clone(),
        });
    }
    if let Some(vars) = di.and_then(|d| d.vars.get(&f.name)) {
        let alloca_offsets = alloca_order_offsets(f, &scan, &frame);
        for v in vars {
            let loc = match v.loc {
                di::DiLoc::Window { alloca_ordinal } => match alloca_offsets.get(alloca_ordinal) {
                    Some(&off) => VarLoc::Window { off: off as i64 },
                    None => continue,
                },
                di::DiLoc::Arg { index } => {
                    let mut locs = Vec::new();
                    for (bi, params) in block_params.iter().enumerate() {
                        if let Some(pos) = params.iter().position(|&p| p == index as ValueId) {
                            locs.push(SsaLoc {
                                block: bi as u32,
                                inst: 0,
                                value: pos as u32,
                            });
                        }
                    }
                    if locs.is_empty() {
                        continue;
                    }
                    VarLoc::SsaList(locs)
                }
            };
            dbg.vars.push(VarInfo {
                func: func_idx,
                name: v.name.clone(),
                ty: v.ty.clone(),
                loc,
                type_id: v.type_id,
                scope: v.scope,
            });
        }
    }

    let mut blocks = Vec::with_capacity(f.basic_blocks.len());
    // Synthetic blocks appended *after* all real blocks (so real block indices are unchanged): a
    // sparse `switch` lowers to a comparison chain whose extra blocks land here (see `translate_switch`).
    let mut aux_blocks: Vec<Block> = Vec::new();
    for (bi, bb) in f.basic_blocks.iter().enumerate() {
        blocks.push(translate_block(
            bb,
            bi,
            func_idx,
            bc_func_idx,
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
            gbytes,
            helpers,
            ba,
            dbg,
            &mut aux_blocks,
        )?);
    }
    blocks.extend(aux_blocks);
    Ok((
        Func {
            params,
            results,
            blocks,
        },
        frame_size,
    ))
}

/// The data-stack offset of each `alloca` **in textual order** (ordinal → offset). This is the
/// correlation target for the `di` reader's `alloca_ordinal`: both walk the function's allocas in
/// the same block/instruction order, so the Nth alloca here is the Nth there. An alloca with no
/// frame slot (shouldn't happen at `-O0`) contributes a `0` placeholder to keep ordinals aligned.
fn alloca_order_offsets(f: &Function, s: &Scan, frame: &HashMap<ValueId, u64>) -> Vec<u64> {
    let mut offs = Vec::new();
    for bb in &f.basic_blocks {
        for instr in &bb.instrs {
            if let Instruction::Alloca(a) = instr {
                let off = s
                    .name2id
                    .get(&a.dest)
                    .and_then(|vid| frame.get(vid))
                    .copied()
                    .unwrap_or(0);
                offs.push(off);
            }
        }
    }
    offs
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
    // A `(...)`-defined function reserves the first 8 bytes of its frame (offset 0) for the
    // **incoming varargs pointer** the caller deposits at `callee_sp + 0` before the call; `va_start`
    // reads it from `sp + 0`. Shifting allocas past it keeps that slot dedicated (§varargs).
    if f.is_var_arg {
        off = 8;
    }
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
    // Reserve the **outgoing-varargs marshaling scratch**: the widest variadic-argument list across
    // all direct `(...)` call sites, one 8-byte slot per variadic argument (§varargs). Recorded under
    // the `VARARG_SCRATCH` sentinel key so the call-site lowering can find its `sp`-relative base.
    let mut any_vararg_call = false;
    let mut max_vararg_slots = 0u64;
    for bb in &f.basic_blocks {
        for instr in &bb.instrs {
            if let Instruction::Call(c) = instr {
                if let Some(extra) = vararg_call_extra(c) {
                    any_vararg_call = true;
                    max_vararg_slots = max_vararg_slots.max(extra as u64);
                }
            }
        }
    }
    if any_vararg_call {
        // Reserve at least one slot so a varargs call with *zero* variadic arguments (e.g. a `(...)`
        // function invoked with only its fixed parameters) still has a valid scratch base to hand the
        // callee — the marshaling stores nothing but still deposits the area pointer.
        off = off.div_ceil(8) * 8;
        frame.insert(VARARG_SCRATCH, off);
        off += max_vararg_slots.max(1) * 8;
    }
    Ok((frame, off.div_ceil(16) * 16))
}

/// The count of **variadic** (beyond-the-fixed) arguments of a direct call to a `(...)` function,
/// or `None` if `c` is not a direct varargs call. The fixed-parameter count comes from the call's
/// declared function type (`param_types`); the variadic arguments are `arguments[fixed..]`.
fn vararg_call_extra(c: &llvm_ir::instruction::Call) -> Option<usize> {
    callee_name(c)?; // indirect varargs calls are rejected separately
    if let Type::FuncType {
        param_types,
        is_var_arg: true,
        ..
    } = c.function_ty.as_ref()
    {
        return Some(c.arguments.len().saturating_sub(param_types.len()));
    }
    None
}

/// Pass 1a: assign a `ValueId` to every SSA value (parameters first, then per block the φ-results
/// and instruction results), recording each one's SVM type and defining block. Also validates that
/// every instruction is in the slice-A subset (so later passes can assume support).
fn scan_func(f: &Function, types: &Types) -> Result<Scan, Error> {
    let mut s = Scan {
        name2id: HashMap::new(),
        ty: Vec::new(),
        wide: HashMap::new(),
        agg_layout: HashMap::new(),
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
                    // A wider-than-128 / sub-128 vector result is legalized to `v128` chunks + a
                    // scalar tail (I2 step 1) — record its layout and a `V128` placeholder type.
                    Err(_) if wide_vec_layout(ty.as_ref()).is_some() => {
                        s.wide.insert(id, wide_vec_layout(ty.as_ref()).unwrap());
                        ValType::V128
                    }
                    // A small by-value struct (a call/`insertvalue` result) is tracked field-wise via
                    // the aggregate side-table, never used as a scalar — record a placeholder type.
                    Err(_) if struct_field_vtypes(ty.as_ref(), types).is_some() => {
                        // A **flat** struct (scalar fields) records its field types so it can cross a
                        // block edge (a struct φ — wikisort's `MakeRange` result): the per-field fan-out
                        // mirrors `wide`. A *nested* struct keeps only the placeholder (block-local; it
                        // fails-closed if it ever reaches a scalar or cross-block use).
                        if let Some(Ok(ftys)) = struct_field_vtypes(ty.as_ref(), types) {
                            s.agg_layout.insert(id, ftys);
                        }
                        ValType::I64
                    }
                    // An `<N x i1>` boolean mask (vector `icmp`/`fcmp`) is held lane-wise as `N` `i32`
                    // `0`/`1` scalars in the `agg` side-table — exactly like a flat `N`-field struct, so
                    // recording an `[i32; N]` `agg_layout` lets the mask **cross block edges** via the
                    // per-field fan-out in `block_params`/`branch_args` (clang's auto-vectorizer can
                    // produce a mask in one block and `extractelement` its lanes in successors — e.g.
                    // Lua's GC fuses two byte-tests into a `<2 x i8>` compare). Placeholder scalar `i32`.
                    Err(_) if i1_vector_lanes(ty.as_ref()).is_some() => {
                        let n = i1_vector_lanes(ty.as_ref()).unwrap();
                        s.agg_layout.insert(id, vec![ValType::I32; n]);
                        ValType::I32
                    }
                    // An `i128` result is held as a 2×`i64` aggregate pair `(lo, hi)` (the unified
                    // `agg`-pair representation, shared with `load i128` / `icmp i128` / the tier-3
                    // `lower_i128` ops). Recording an `[i64, i64]` `agg_layout` — exactly like a flat
                    // 2-field struct — lets the value **cross block edges**: the per-field fan-out in
                    // `block_params`/`branch_args` threads its `(lo, hi)` as two block params (an i128
                    // loop-carried φ / live-across value), not just same-block.
                    Err(_) if matches!(ty.as_ref(), Type::IntegerType { bits: 128 }) => {
                        s.agg_layout.insert(id, vec![ValType::I64, ValType::I64]);
                        ValType::I64
                    }
                    Err(e) => return Err(e),
                };
                s.ty.push(vt);
                s.def_block.push(bi);
            }
        }
        term_local_uses(&bb.term)?; // validate terminator support
                                    // An `invoke`'s result is defined by the *terminator* (not an instruction), so assign it an
                                    // id + type here — the callee's (non-void) return type. The normal-edge successor then
                                    // threads it like any cross-block value (it is recorded as a def of this block in liveness).
        if let LTerm::Invoke(inv) = &bb.term {
            if let Type::FuncType { result_type, .. } = inv.function_ty.as_ref() {
                if !matches!(result_type.as_ref(), Type::VoidType) {
                    let id = s.ty.len();
                    s.name2id.insert(inv.result.clone(), id);
                    s.ty.push(val_type(result_type)?);
                    s.def_block.push(bi);
                }
            }
        }
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
        // `<2 x float>` lane access (the vector itself is a scalarized packed `i64`).
        I::InsertElement(x) => locals(&[&x.vector, &x.element]),
        I::ExtractElement(x) => locals(&[&x.vector]),
        I::ShuffleVector(x) => locals(&[&x.operand0, &x.operand1]),
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
        // Atomics (§12): same operand uses as the corresponding load/store, plus the rmw value /
        // cmpxchg expected+replacement.
        I::AtomicRMW(r) => locals(&[&r.address, &r.value]),
        I::CmpXchg(cx) => locals(&[&cx.address, &cx.expected, &cx.replacement]),
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
        // A `landingpad` reads only the reserved EH slots (the current exception/selector), no LLVM
        // value operands — its `{ptr,i32}` result is bound field-wise in `translate_inst`.
        I::LandingPad(_) => Vec::new(),
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
/// The function symbol a global alias's `aliasee` names, unwrapping any `bitcast` constant-expr LLVM
/// places around it. `None` if it is not a (wrapped) global reference (e.g. an alias to a data global,
/// or a GEP expression — neither resolves to a function index).
fn alias_target_name(c: &Constant) -> Option<String> {
    match c {
        Constant::GlobalReference {
            name: Name::Name(s),
            ..
        } => Some(s.to_string()),
        Constant::BitCast(bc) => alias_target_name(bc.operand.as_ref()),
        _ => None,
    }
}

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
// entry and threaded to every call site through the *handle stash*, the reserved handle region
// (`[0, HANDLE_REGION_END)`): `_start` stores the granted handles there and each call site reloads the
// one it needs (so a handle reaches arbitrary call depth without a viral extra parameter). This keeps
// the translator pure mechanism — it never interprets host semantics, just defers the bind (§2a).

/// Window offsets of the **powerbox handle stash + allocator state** (the reserved low scratch on
/// page 0, which is writable — for a powerbox program the globals start a page up). `_start` stores
/// the granted handles here; each call site reloads what it needs. The heap allocator (slice S) keeps
/// its bump pointer + committed boundary here too.
///
/// **Layout (locked to the full §3e powerbox).** The handle region is `[0, HANDLE_REGION_END)` =
/// `[0, 32)` — eight `i32` slots, one per `VM_CAP_*` index (`<svm.h>`): stdout, stdin, exit, memory,
/// addrspace, ioring, blocking, jit. `_start` stashes a *prefix* of these (today 3 or 4 — stdout,
/// stdin, exit, `[memory]`); offsets `16/20/24/28` are **reserved** for the AddressSpace/IoRing/
/// Blocking/Jit tail so granting it later (the P2 async-I/O work) needs **no stash relocation**. The
/// allocator/scratch/format state lives strictly **above** the handle region, so it never collides
/// with a newly-granted handle (the bug this layout forecloses).
// The handle stash is the public powerbox layout ([`svm_ir::POWERBOX_STASH_BASE`] + `i*4`), shared
// with the frontend-independent [`svm_ir::synth_powerbox_start`]; the per-`VM_CAP_*` slots derive
// from it so this and the public synthesizer can never drift.
const STASH_STDOUT: u64 = svm_ir::POWERBOX_STASH_BASE;
const STASH_STDIN: u64 = STASH_STDOUT + 4;
const STASH_EXIT: u64 = STASH_STDOUT + 8;
/// The `Memory` capability handle (`i32`) — present when the program uses `malloc` *or* a direct
/// `<svm.h>` Memory builtin (then `_start` takes a 4th granted handle). The bump allocator + the
/// `__vm_map`/… builtins reload it to drive `Memory` capability calls.
const STASH_MEMORY: u64 = STASH_STDOUT + 12;
/// The `AddressSpace` handle (slot 4) — granted when the program mints a §13/§14 `SharedRegion`
/// (`__vm_region_create` calls `AddressSpace.create_region`). The region handle it returns is then the
/// capability for `__vm_region_map`/`unmap`/`page_size` (not a stash slot — those take it as an arg).
const STASH_ADDRSPACE: u64 = STASH_STDOUT + 16;
/// The `IoRing` (slot 5) and `Blocking` (slot 6) handles — granted when the program uses the §9/§12
/// async-ring builtins (`__vm_io_submit_async`/`__vm_io_reap` drive `IoRing`; `__vm_blocking_handle`
/// returns the `Blocking` handle a guest names in an SQE).
const STASH_IORING: u64 = STASH_STDOUT + 20;
const STASH_BLOCKING: u64 = STASH_STDOUT + 24;
/// The `Jit` handle (slot 7) — granted when the program uses the §22 guest-driven-JIT builtins
/// (`__vm_jit_compile`/`invoke2`/`release`/`install`/`uninstall`/`compile_linked`): a guest submits
/// serialized SVM IR built in its own window and the host verifies + Cranelift-compiles it into THIS
/// domain. Slot 4 (`AddressSpace`) stays reserved (offset 16) for the §13/§14 region builtins.
const STASH_JIT: u64 = STASH_STDOUT + 28;
/// End of the reserved 8-handle region (`[0, 32)`, one `i32` slot per `VM_CAP_*` index). The
/// allocator/scratch/format state begins here, so it is collision-proof against the full handle set.
/// This is exactly the public heap base ([`svm_ir::POWERBOX_HEAP_BRK`]).
const HANDLE_REGION_END: u64 = svm_ir::POWERBOX_HEAP_BRK;
/// The guest heap's bump pointer and committed boundary (`i64` each) — the allocator's only state,
/// placed just above the 8-handle region ([`svm_ir::POWERBOX_HEAP_BRK`]/[`svm_ir::POWERBOX_HEAP_TOP`]).
const HEAP_BRK: u64 = HANDLE_REGION_END; // 32
const HEAP_TOP: u64 = HEAP_BRK + 8; // 40
                                    // Pin the C `_start` layout to the public powerbox ABI so this and `svm_ir::synth_powerbox_start`
                                    // can never silently diverge (the dedup hinge: one source of truth in `svm-ir`).
const _: () = assert!(HEAP_TOP == svm_ir::POWERBOX_HEAP_TOP);
const _: () = assert!(STASH_STDOUT == svm_ir::POWERBOX_STASH_BASE);
/// A 1-byte writable scratch used by `putc`/`puts` to stage a single byte (a char, a newline) the
/// `Stream` capability writes (its ABI is a `(buf, len)` window slice, so a scalar char must transit
/// memory). Reused per call — single-threaded, fully produced-then-consumed within one lowering.
const STASH_SCRATCH: u64 = HEAP_TOP + 8; // 48
/// The `prot` bits a guest passes to `Memory.map` for a read-write commit (`PROT_READ|PROT_WRITE`).
const PROT_RW: i32 = 3;
/// A scratch buffer (`[FMT_BUF, FMT_BUF_END)`, on the writable page 0) where `printf` formats one
/// integer conversion: `__svm_utoa` writes the digits backward from `FMT_BUF_END`, and width padding
/// pre-fills the low end. 64 bytes covers a 64-bit value in any base plus a generous field width.
const FMT_BUF: u64 = 64;
const FMT_BUF_END: u64 = 128;
/// Scratch for the `%f` helper (`__svm_dtoa_fixed`): the 128-bit working limbs, the extracted decimal
/// digits, the assembled content, a small stashed-context block, and the padded output field. Placed
/// just above the data-stack reserve in the window (a fixed, reusable region — no per-call alloc),
/// reserved only when `need_dtoa`. Sized for the bignum formatters (three 40-limb big integers, the
/// digit buffer, content, padded field, and scalar locals — see the `FMT_*_O` offsets).
const FLOAT_SCRATCH_SIZE: u64 = 2304;

// ── C++ exception-handling (Itanium ABI on-ramp) reserved scratch ──────────────────────────────
// EH state rides in a fixed window region just above the float scratch (`eh_base`, reserved only
// when `need_eh`). The window is freshly mapped and zero-filled, so the handler stack pointer and
// the current-exception slots start at 0 with no explicit init. Offsets are relative to `eh_base`.
/// Handler-stack pointer (`i32`): the count of active `try` handlers, indexing the buf slots below.
const EH_HSP_O: u64 = 0;
/// Current in-flight exception object pointer (`i64`), set by `__cxa_throw`, read by `landingpad`.
const EH_EXN_O: u64 = 8;
/// Current exception type selector (`i32`): the typeinfo id of the in-flight exception (see
/// `Helpers::eh_typeids`), set by `__cxa_throw`, read by `landingpad` / compared by `eh.typeid.for`.
const EH_SEL_O: u64 = 16;
/// The in-flight exception object's destructor (`i64` funcref, `0` for a trivially-destructible type):
/// the third `__cxa_throw` argument, stashed so `__cxa_end_catch` can run it on the exception object
/// when the handler completes (the `__svm_eh_destroy` helper guards the null case). `__cxa_rethrow`
/// preserves it (it re-raises the same object).
const EH_DTOR_O: u64 = 24;
/// Start of the handler buf slots — one `EH_SLOT`-byte slot per active handler. The slots only need
/// distinct guest addresses (the `SetJmp`/`LongJmp` ops key a host-side checkpoint by buf address),
/// so the bytes themselves are never read as a `jmp_buf`.
const EH_BUFS_O: u64 = 32;
/// Bytes per handler buf slot (only its address matters; sized for alignment headroom).
const EH_SLOT: u64 = 32;
/// Maximum nesting depth of active `try` handlers (region is sized for this many buf slots).
const EH_NHANDLERS: u64 = 64;
/// A fixed scratch slot `__cxa_allocate_exception` hands back to hold the thrown object (the boxed
/// `int` / `const char*`). One in-flight exception at a time (non-reentrant, no nested allocate
/// before the matching throw lands) — sufficient for the on-ramp's single-threaded EH.
const EH_EXNOBJ_O: u64 = EH_BUFS_O + EH_SLOT * EH_NHANDLERS;
/// Bytes of the exception-object scratch (covers a pointer / small scalar payload).
const EH_EXNOBJ_SIZE: u64 = 64;
/// Total bytes of the reserved EH region.
const EH_REGION_SIZE: u64 = EH_EXNOBJ_O + EH_EXNOBJ_SIZE;

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
        "write" | "puts" | "putc" | "putchar" | "fputc" | "fwrite" | "fputs" | "printf" => "write",
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
        // `Memory.unmap(offset, len)` (op 1) / `protect(offset, len, prot)` (op 2) /
        // `page_size()` (op 3) — the rest of the §3e/§4 Memory surface a guest reaches from `<svm.h>`.
        "vm_unmap" => ft(vec![I64, I64], vec![I64]),
        "vm_protect" => ft(vec![I64, I64, I32], vec![I64]),
        "vm_page_size" => ft(vec![], vec![I64]),
        // §9/§12 async I/O ring (`IoRing`): submit a batch of deferred ops onto the host offload pool
        // (op 1) / reap ready completions (op 2). `(sq, n, counter)` / `(cq, max)`, returning a count.
        "vm_io_submit_async" => ft(vec![I64, I64, I64], vec![I64]),
        "vm_io_reap" => ft(vec![I64, I64], vec![I64]),
        // §22 guest-driven JIT (`Jit`): submit serialized IR → code handle (op 0) / call a compiled
        // `(i64,i64)->(i64)` unit (op 1) / release (op 2) / install into the call_indirect table (op 3)
        // / uninstall a slot (op 4) / compile against a guest symbol table (op 5). All return an `i64`.
        "vm_jit_compile" => ft(vec![I64, I64], vec![I64]),
        "vm_jit_invoke2" => ft(vec![I64, I64, I64], vec![I64]),
        "vm_jit_release" | "vm_jit_install" | "vm_jit_uninstall" => ft(vec![I64], vec![I64]),
        "vm_jit_compile_linked" => ft(vec![I64, I64, I64, I64], vec![I64]),
        // §13/§14 SharedRegion: mint a region from `AddressSpace` (`create`, op 5 on the AddressSpace
        // handle) → a region handle; `map`/`unmap`/`page_size` (ops 0/1/3 on that *region* handle) then
        // alias its bytes into the window (the magic ring buffer / zero-copy child data plane).
        "vm_region_create" => ft(vec![I64], vec![I64]),
        "vm_region_map" => ft(vec![I64, I64, I64, I32], vec![I64]),
        "vm_region_unmap" => ft(vec![I64, I64], vec![I64]),
        "vm_region_page_size" => ft(vec![], vec![I64]),
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

/// The §7 import a `<svm.h>` **Memory-capability** builtin needs (`__vm_map`/`unmap`/`protect`/
/// `page_size` → `vm_map`/`vm_unmap`/`vm_protect`/`vm_page_size`), or `None` if `name` is not one of
/// them. These reach `Memory` (the 4th powerbox handle, slot 12) — the same cap the bump allocator
/// uses, exposed directly so a guest manages window pages itself (the §1a sparse-address-space path).
fn vm_memory_builtin_import(name: &str) -> Option<&'static str> {
    Some(match name {
        "__vm_map" => "vm_map",
        "__vm_unmap" => "vm_unmap",
        "__vm_protect" => "vm_protect",
        "__vm_page_size" => "vm_page_size",
        _ => return None,
    })
}

/// Scan for direct `<svm.h>` Memory-capability builtin calls (`__vm_map`/…), registering each one's
/// §7 import into the table (deduplicated, like [`collect_cap_imports`]) so the call lowering's
/// `import_of` resolves it. Returns whether any were used — the signal that `_start` must be granted
/// the `Memory` handle even if the program never calls `malloc`. A guest-*defined* function of the
/// same name shadows the builtin (mirrors the libc/libm rule).
fn register_vm_memory_imports(
    m: &LModule,
    defined: &HashMap<String, u32>,
    imports: &mut Vec<svm_ir::Import>,
    caps: &mut HashMap<String, u32>,
) -> bool {
    let mut used = false;
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
                if let Some(import) = vm_memory_builtin_import(&name) {
                    used = true;
                    caps.entry(import.to_string()).or_insert_with(|| {
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
    used
}

/// The §7 import a §9/§12 async-ring builtin needs (`__vm_io_submit_async` → `vm_io_submit_async`,
/// `__vm_io_reap` → `vm_io_reap`), or `None`. Both reach the `IoRing` (slot 5) handle.
fn vm_io_builtin_import(name: &str) -> Option<&'static str> {
    Some(match name {
        "__vm_io_submit_async" => "vm_io_submit_async",
        "__vm_io_reap" => "vm_io_reap",
        _ => return None,
    })
}

/// Scan for the async-ring builtins (`__vm_io_submit_async`/`__vm_io_reap`), registering each one's
/// §7 import. Returns whether any were used — the signal that `_start` must be granted up through the
/// `IoRing` handle. A guest-*defined* function of the same name shadows the builtin.
fn register_vm_io_imports(
    m: &LModule,
    defined: &HashMap<String, u32>,
    imports: &mut Vec<svm_ir::Import>,
    caps: &mut HashMap<String, u32>,
) -> bool {
    let mut used = false;
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
                if let Some(import) = vm_io_builtin_import(&name) {
                    used = true;
                    caps.entry(import.to_string()).or_insert_with(|| {
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
    used
}

/// The §7 import a §22 guest-driven-JIT builtin needs (`__vm_jit_compile` → `vm_jit_compile`, …), or
/// `None`. All reach the `Jit` (slot 7) handle; the host verifies + Cranelift-compiles the submitted IR.
fn vm_jit_builtin_import(name: &str) -> Option<&'static str> {
    Some(match name {
        "__vm_jit_compile" => "vm_jit_compile",
        "__vm_jit_invoke2" => "vm_jit_invoke2",
        "__vm_jit_release" => "vm_jit_release",
        "__vm_jit_install" => "vm_jit_install",
        "__vm_jit_uninstall" => "vm_jit_uninstall",
        "__vm_jit_compile_linked" => "vm_jit_compile_linked",
        _ => return None,
    })
}

/// Scan for the guest-driven-JIT builtins, registering each one's §7 import. Returns whether any were
/// used — the signal that `_start` must be granted up through the `Jit` handle (the full 8-handle
/// powerbox). A guest-*defined* function of the same name shadows the builtin.
fn register_vm_jit_imports(
    m: &LModule,
    defined: &HashMap<String, u32>,
    imports: &mut Vec<svm_ir::Import>,
    caps: &mut HashMap<String, u32>,
) -> bool {
    let mut used = false;
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
                if let Some(import) = vm_jit_builtin_import(&name) {
                    used = true;
                    caps.entry(import.to_string()).or_insert_with(|| {
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
    used
}

/// The §7 import a §13/§14 SharedRegion builtin needs (`__vm_region_create` → `vm_region_create`, …),
/// or `None`. `create` reaches the `AddressSpace` (slot 4) handle; `map`/`unmap`/`page_size` reach the
/// region handle `create` returned (passed as the call's first arg, not a stash slot).
fn vm_region_builtin_import(name: &str) -> Option<&'static str> {
    Some(match name {
        "__vm_region_create" => "vm_region_create",
        "__vm_region_map" => "vm_region_map",
        "__vm_region_unmap" => "vm_region_unmap",
        "__vm_region_page_size" => "vm_region_page_size",
        _ => return None,
    })
}

/// Scan for the SharedRegion builtins, registering each one's §7 import. Returns whether any were used
/// — the signal that `_start` must be granted the `AddressSpace` handle (`__vm_region_create` mints
/// from it). A guest-*defined* function of the same name shadows the builtin.
fn register_vm_region_imports(
    m: &LModule,
    defined: &HashMap<String, u32>,
    imports: &mut Vec<svm_ir::Import>,
    caps: &mut HashMap<String, u32>,
) -> bool {
    let mut used = false;
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
                if let Some(import) = vm_region_builtin_import(&name) {
                    used = true;
                    caps.entry(import.to_string()).or_insert_with(|| {
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
    used
}

/// Collect the program's **global constructors** (`@llvm.global_ctors`, the C++ static-init / `__attribute__((constructor))` runners clang emits) as IR function indices in **priority order** (low
/// runs first). Each entry is `{ i32 priority, ptr ctor, ptr data }`; `data` is ignored (always null
/// for the cases we accept). `_start` calls these — each a `(i64 sp) -> ()` — before `main`, so static
/// init runs exactly as it does natively (the on-ramp otherwise jumps straight to `main`). An empty /
/// absent list is the common C case (no static init). A non-null `data` or an indirect ctor operand is
/// a clean `Unsupported`.
fn collect_global_ctors(m: &LModule, name2idx: &HashMap<String, u32>) -> Result<Vec<u32>, Error> {
    let Some(g) = m
        .global_vars
        .iter()
        .find(|g| name_str(&g.name) == "llvm.global_ctors")
    else {
        return Ok(Vec::new());
    };
    let Some(init) = &g.initializer else {
        return Ok(Vec::new());
    };
    let elements = match init.as_ref() {
        Constant::Array { elements, .. } => elements,
        Constant::AggregateZero(_) => return Ok(Vec::new()),
        other => return unsup(format!("llvm.global_ctors initializer: {other:?}")),
    };
    let mut entries: Vec<(u64, u32)> = Vec::new();
    for e in elements {
        let Constant::Struct { values, .. } = e.as_ref() else {
            return unsup("llvm.global_ctors element is not a struct");
        };
        let priority = match values.first().map(|v| v.as_ref()) {
            Some(Constant::Int { value, .. }) => *value,
            _ => 0,
        };
        match values.get(1).map(|v| v.as_ref()) {
            Some(Constant::GlobalReference { name, .. }) => {
                let fname = name_str(name);
                let idx = *name2idx.get(&fname).ok_or_else(|| {
                    Error::Unsupported(format!("global ctor `{fname}` is not a defined function"))
                })?;
                entries.push((priority, idx));
            }
            // A null ctor slot (clang sometimes pads) — nothing to run.
            Some(Constant::Null(_)) => {}
            other => return unsup(format!("llvm.global_ctors ctor operand: {other:?}")),
        }
    }
    entries.sort_by_key(|&(p, _)| p); // ascending priority: lower runs first (C++ [basic.start])
    Ok(entries.into_iter().map(|(_, idx)| idx).collect())
}

/// The shared signature of the two synthesized powerbox entries, [`synth_start`] (plain
/// `main(void)`) and [`synth_start_argv`] (`main(int, char**)`): `(main_idx, main_results, entry_sp,
/// n_handles, heap_base, ctors) -> Func`. A `type` alias so the dispatch can pick one by function
/// pointer without tripping `clippy::type_complexity`.
type StartBuilder = fn(u32, &[ValType], u64, usize, Option<u64>, &[u32], bool) -> Func;

/// Synthesize the **powerbox entry** (`_start`, function 0) for a program that uses host
/// capabilities. It takes the `n_handles` granted handles as `i32` params (the §3e powerbox shape
/// `is_powerbox_entry` recognizes — no threaded data-SP, since it is the root), in the `VM_CAP_*`
/// order (stdout, stdin, exit, memory, addrspace, ioring, blocking, jit), stores each into its stash
/// slot (offset `i*4`) so every capability call site can reload its handle, then calls the C
/// `main(sp)` at the page-aligned data-stack base and returns its exit code. The runner grants
/// exactly `n_handles` (a contiguous prefix, by declared arity — `run_powerbox`).
fn synth_start(
    main_idx: u32,
    main_results: &[ValType],
    entry_sp: u64,
    n_handles: usize,
    heap_base: Option<u64>,
    ctors: &[u32],
    _wants_envp: bool,
) -> Func {
    use svm_ir::StoreOp;
    let mut insts: Vec<Inst> = Vec::new();
    // params v0..v(n-1) = the granted handles. Each slot offset is just `i*4` (the `STASH_*` layout),
    // so stashing is a uniform loop: store param `i` at byte offset `i*4`. A program is granted a
    // prefix sized to the highest capability index it uses (e.g. 4 with `malloc`/Memory, 7 with the
    // async-ring builtins through `Blocking`).
    let params = vec![ValType::I32; n_handles];
    let mut next: ValIdx = n_handles as ValIdx;
    for i in 0..n_handles {
        insts.push(Inst::ConstI64((i as i64) * 4));
        let addr = next;
        next += 1;
        insts.push(Inst::Store {
            op: StoreOp::I32,
            addr,
            value: i as ValIdx,
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
    // sp = entry_sp (constant). The data-SP `main` (and each ctor) carries as param 0.
    insts.push(Inst::ConstI64(entry_sp as i64));
    let sp = next;
    next += 1;
    // Run the C++ global constructors (priority order) before `main` — each is `(i64 sp) -> ()`, so
    // it appends no value (sequential calls, each takes its own frame above `sp`). Static init then
    // happens exactly as native, before the program proper.
    for &ctor in ctors {
        insts.push(Inst::Call {
            func: ctor,
            args: vec![sp],
        });
    }
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

/// The `_start` for a `main(int argc, char** argv)` (and, when `wants_envp`, `main(…, char** envp)`):
/// same handle-stash + heap-seed prologue as [`synth_start`], then it parses the §3e args buffer
/// (`svm_ir::POWERBOX_ARGS_BASE`, seeded by the host) into a C `argv[]`. It reads `argc`, walks the
/// `argc` packed NUL-terminated strings (each `argv[i]` points *into* the buffer — no copy), and
/// writes the pointer array (plus the required `argv[argc] == NULL`) at the entry data-stack base.
/// For a 4-param `main` it then parses the `envc` env strings (packed right after the argv strings)
/// into a second NULL-terminated `envp[]` parked just above `argv[]`. Finally it calls
/// `main(main_sp, argc, argv[, envp])` with `main`'s frame parked one page above the array(s). This is
/// the only place the C `char**` convention exists — the powerbox ABI itself only delivers the
/// neutral byte blob.
fn synth_start_argv(
    main_idx: u32,
    main_results: &[ValType],
    entry_sp: u64,
    n_handles: usize,
    heap_base: Option<u64>,
    ctors: &[u32],
    wants_envp: bool,
) -> Func {
    use svm_ir::{LoadOp, StoreOp};
    let args_base = svm_ir::POWERBOX_ARGS_BASE as i64;
    let page = STACK_PAGE as i64;
    let add = |a: ValIdx, b: ValIdx| Inst::IntBin {
        ty: IntTy::I64,
        op: BinOp::Add,
        a,
        b,
    };
    let mul = |a: ValIdx, b: ValIdx| Inst::IntBin {
        ty: IntTy::I64,
        op: BinOp::Mul,
        a,
        b,
    };
    let and = |a: ValIdx, b: ValIdx| Inst::IntBin {
        ty: IntTy::I64,
        op: BinOp::And,
        a,
        b,
    };
    let store64 = |addr: ValIdx, value: ValIdx| Inst::Store {
        op: StoreOp::I64,
        addr,
        value,
        offset: 0,
        align: 0,
    };

    // ---- block 0: entry — stash handles, seed the heap, load argc, jump into the argv loop. ----
    let params = vec![ValType::I32; n_handles];
    let mut insts: Vec<Inst> = Vec::new();
    let mut next: ValIdx = n_handles as ValIdx;
    for i in 0..n_handles {
        insts.push(Inst::ConstI64((i as i64) * 4));
        let addr = next;
        next += 1;
        insts.push(Inst::Store {
            op: StoreOp::I32,
            addr,
            value: i as ValIdx,
            offset: 0,
            align: 0,
        });
    }
    if let Some(hb) = heap_base {
        for off in [HEAP_BRK, HEAP_TOP] {
            insts.push(Inst::ConstI64(off as i64));
            let addr = next;
            next += 1;
            insts.push(Inst::ConstI64(hb as i64));
            let val = next;
            next += 1;
            insts.push(store64(addr, val));
        }
    }
    insts.push(Inst::ConstI64(args_base));
    let v_aa = next;
    next += 1;
    insts.push(Inst::Load {
        op: LoadOp::I32,
        addr: v_aa,
        offset: 0,
        align: 0,
    }); // argc (u32)
    let v_argc32 = next;
    next += 1;
    insts.push(Inst::Convert {
        op: ConvOp::ExtendI32U,
        a: v_argc32,
    });
    let v_argc64 = next;
    next += 1;
    insts.push(Inst::ConstI64(args_base + 8)); // first string (past {argc,envc})
    let v_p0 = next;
    next += 1;
    insts.push(Inst::ConstI64(0)); // i = 0
    let v_i0 = next;
    let b0 = Block {
        params,
        insts,
        term: Terminator::Br {
            target: 1,
            args: vec![v_argc64, v_i0, v_p0],
        },
    };

    // ---- block 1: loop_head(argc=0, i=1, p=2) — continue while i <u argc, else finish. ----
    let b1 = Block {
        params: vec![ValType::I64, ValType::I64, ValType::I64],
        insts: vec![Inst::IntCmp {
            ty: IntTy::I64,
            op: CmpOp::LtU,
            a: 1,
            b: 0,
        }], // v3 = i < argc
        term: Terminator::BrIf {
            cond: 3,
            then_blk: 2,
            then_args: vec![0, 1, 2],
            else_blk: 5,
            // The env strings begin where the argv walk left off, so when we need `envp` carry `p`
            // (= &first env string) into block 5; the argv-only path drops it.
            else_args: if wants_envp { vec![0, 2] } else { vec![0] },
        },
    };

    // ---- block 2: body(argc=0, i=1, p=2) — argv[i] = p, then scan p to its NUL. ----
    let b2 = Block {
        params: vec![ValType::I64, ValType::I64, ValType::I64],
        insts: vec![
            Inst::ConstI64(entry_sp as i64), // v3 = argv base
            Inst::ConstI64(8),               // v4
            mul(1, 4),                       // v5 = i*8
            add(3, 5),                       // v6 = &argv[i]
            store64(6, 2),                   // argv[i] = p
        ],
        term: Terminator::Br {
            target: 3,
            args: vec![0, 1, 2, 2], // scan(argc, i, p, q=p)
        },
    };

    // ---- block 3: scan(argc=0, i=1, p=2, q=3) — advance q to the byte past the NUL. ----
    let b3 = Block {
        params: vec![ValType::I64, ValType::I64, ValType::I64, ValType::I64],
        insts: vec![
            Inst::Load {
                op: LoadOp::I32_8U,
                addr: 3,
                offset: 0,
                align: 0,
            }, // v4 = *q
            Inst::ConstI64(1), // v5
            add(3, 5),         // v6 = q+1
            Inst::ConstI32(0), // v7
            Inst::IntCmp {
                ty: IntTy::I32,
                op: CmpOp::Eq,
                a: 4,
                b: 7,
            }, // v8 = (*q == 0)
        ],
        term: Terminator::BrIf {
            cond: 8,
            then_blk: 4,
            then_args: vec![0, 1, 6], // next(argc, i, p_next = q+1)
            else_blk: 3,
            else_args: vec![0, 1, 2, 6], // scan(argc, i, p, q+1)
        },
    };

    // ---- block 4: next(argc=0, i=1, p_next=2) — i++ and loop. ----
    let b4 = Block {
        params: vec![ValType::I64, ValType::I64, ValType::I64],
        insts: vec![
            Inst::ConstI64(1), // v3
            add(1, 3),         // v4 = i+1
        ],
        term: Terminator::Br {
            target: 1,
            args: vec![0, 4, 2], // loop_head(argc, i+1, p_next)
        },
    };

    // ---- block 5 (argv-only `main(int, char**)`): done(argc) — argv[argc] = NULL, compute main_sp
    // one page above the array, run ctors, call main(main_sp, argc, argv). ----
    if !wants_envp {
        let mut d: Vec<Inst> = vec![
            Inst::ConstI64(entry_sp as i64), // v1 = argv base
            Inst::ConstI64(8),               // v2
            mul(0, 2),                       // v3 = argc*8
            add(1, 3),                       // v4 = &argv[argc]
            Inst::ConstI64(0),               // v5
            store64(4, 5),                   // argv[argc] = NULL
            Inst::ConstI64(1),               // v6
            add(0, 6),                       // v7 = argc+1
            mul(7, 2),                       // v8 = (argc+1)*8
            add(1, 8),                       // v9 = entry_sp + array bytes
            Inst::ConstI64(page - 1),        // v10
            add(9, 10),                      // v11
            Inst::ConstI64(!(page - 1)),     // v12
            and(11, 12),                     // v13 = main_sp (page-aligned)
            Inst::Convert {
                op: ConvOp::WrapI64,
                a: 0,
            }, // v14 = argc (i32)
        ];
        // ctors run on the real frame (`main_sp`); each is `(i64 sp) -> ()`, appending no value.
        for &ctor in ctors {
            d.push(Inst::Call {
                func: ctor,
                args: vec![13],
            });
        }
        // main(main_sp, argc, argv) — argv is the pointer array at the entry SP (v1).
        d.push(Inst::Call {
            func: main_idx,
            args: vec![13, 14, 1],
        });
        let term = if main_results.is_empty() {
            Terminator::Return(vec![])
        } else {
            Terminator::Return(vec![15]) // main's result (params 1 + 14 value insts before the call)
        };
        let b5 = Block {
            params: vec![ValType::I64],
            insts: d,
            term,
        };
        return Func {
            results: main_results.to_vec(),
            blocks: vec![b0, b1, b2, b3, b4, b5],
            params: vec![ValType::I32; n_handles],
        };
    }

    // ===== `main(int, char**, char** envp)`: parse the blob's `envc` env strings into a second =====
    // pointer array parked just above `argv[]`, then call `main(main_sp, argc, argv, envp)`. The env
    // loop (blocks 6..9) mirrors the argv loop; `envc` is the second u32 at `args_base + 4`, and the
    // env strings immediately follow the argv strings, so block 1 handed us `p` already pointing at
    // the first one.
    let load_i32 = |addr: ValIdx| Inst::Load {
        op: LoadOp::I32,
        addr,
        offset: 0,
        align: 0,
    };

    // ---- block 5: argv_done(argc=0, p=1) — terminate argv[], compute envp_base, load envc, loop. ----
    let b5 = Block {
        params: vec![ValType::I64, ValType::I64], // argc, p (= &first env string)
        insts: vec![
            Inst::ConstI64(entry_sp as i64), // v2 = argv base
            Inst::ConstI64(8),               // v3
            mul(0, 3),                       // v4 = argc*8
            add(2, 4),                       // v5 = &argv[argc]
            Inst::ConstI64(0),               // v6
            store64(5, 6),                   // argv[argc] = NULL
            Inst::ConstI64(1),               // v7
            add(0, 7),                       // v8 = argc+1
            mul(8, 3),                       // v9 = (argc+1)*8
            add(2, 9),                       // v10 = envp_base (just above argv[])
            Inst::ConstI64(args_base + 4),   // v11
            load_i32(11),                    // v12 = envc (u32)
            Inst::Convert {
                op: ConvOp::ExtendI32U,
                a: 12,
            }, // v13 = envc (i64)
            Inst::ConstI64(0),               // v14 = j = 0
        ],
        term: Terminator::Br {
            target: 6,
            args: vec![13, 14, 1, 10], // env_head(envc, j=0, q=p, envp_base)
        },
    };

    // ---- block 6: env_head(envc=0, j=1, q=2, envp_base=3) — continue while j <u envc, else done. ----
    let b6 = Block {
        params: vec![ValType::I64, ValType::I64, ValType::I64, ValType::I64],
        insts: vec![Inst::IntCmp {
            ty: IntTy::I64,
            op: CmpOp::LtU,
            a: 1,
            b: 0,
        }], // v4 = j < envc
        term: Terminator::BrIf {
            cond: 4,
            then_blk: 7,
            then_args: vec![0, 1, 2, 3],
            else_blk: 10,
            else_args: vec![0, 3], // env_done(envc, envp_base)
        },
    };

    // ---- block 7: env_body(envc=0, j=1, q=2, envp_base=3) — envp[j] = q, then scan q. ----
    let b7 = Block {
        params: vec![ValType::I64, ValType::I64, ValType::I64, ValType::I64],
        insts: vec![
            Inst::ConstI64(8), // v4
            mul(1, 4),         // v5 = j*8
            add(3, 5),         // v6 = &envp[j]
            store64(6, 2),     // envp[j] = q
        ],
        term: Terminator::Br {
            target: 8,
            args: vec![0, 1, 2, 3, 2], // env_scan(envc, j, q, envp_base, r=q)
        },
    };

    // ---- block 8: env_scan(envc=0, j=1, q=2, envp_base=3, r=4) — advance r to the byte past NUL. ----
    let b8 = Block {
        params: vec![
            ValType::I64,
            ValType::I64,
            ValType::I64,
            ValType::I64,
            ValType::I64,
        ],
        insts: vec![
            Inst::Load {
                op: LoadOp::I32_8U,
                addr: 4,
                offset: 0,
                align: 0,
            }, // v5 = *r
            Inst::ConstI64(1), // v6
            add(4, 6),         // v7 = r+1
            Inst::ConstI32(0), // v8
            Inst::IntCmp {
                ty: IntTy::I32,
                op: CmpOp::Eq,
                a: 5,
                b: 8,
            }, // v9 = (*r == 0)
        ],
        term: Terminator::BrIf {
            cond: 9,
            then_blk: 9,
            then_args: vec![0, 1, 7, 3], // env_next(envc, j, q_next = r+1, envp_base)
            else_blk: 8,
            else_args: vec![0, 1, 2, 3, 7], // env_scan(envc, j, q, envp_base, r+1)
        },
    };

    // ---- block 9: env_next(envc=0, j=1, q_next=2, envp_base=3) — j++ and loop. ----
    let b9 = Block {
        params: vec![ValType::I64, ValType::I64, ValType::I64, ValType::I64],
        insts: vec![
            Inst::ConstI64(1), // v4
            add(1, 4),         // v5 = j+1
        ],
        term: Terminator::Br {
            target: 6,
            args: vec![0, 5, 2, 3], // env_head(envc, j+1, q_next, envp_base)
        },
    };

    // ---- block 10: env_done(envc=0, envp_base=1) — envp[envc] = NULL, compute main_sp above both
    // arrays, run ctors, call main(main_sp, argc, argv, envp). ----
    let mut e: Vec<Inst> = vec![
        Inst::ConstI64(8),               // v2
        mul(0, 2),                       // v3 = envc*8
        add(1, 3),                       // v4 = &envp[envc]
        Inst::ConstI64(0),               // v5
        store64(4, 5),                   // envp[envc] = NULL
        Inst::ConstI64(1),               // v6
        add(0, 6),                       // v7 = envc+1
        mul(7, 2),                       // v8 = (envc+1)*8
        add(1, 8),                       // v9 = top of the envp array
        Inst::ConstI64(page - 1),        // v10
        add(9, 10),                      // v11
        Inst::ConstI64(!(page - 1)),     // v12
        and(11, 12),                     // v13 = main_sp (page-aligned above both arrays)
        Inst::ConstI64(args_base),       // v14
        load_i32(14),                    // v15 = argc (i32)
        Inst::ConstI64(entry_sp as i64), // v16 = argv base
    ];
    for &ctor in ctors {
        e.push(Inst::Call {
            func: ctor,
            args: vec![13],
        });
    }
    // main(main_sp, argc, argv, envp): argv is at the entry SP (v16), envp just above it (v1).
    e.push(Inst::Call {
        func: main_idx,
        args: vec![13, 15, 16, 1],
    });
    let term = if main_results.is_empty() {
        Terminator::Return(vec![])
    } else {
        Terminator::Return(vec![17]) // main's result (params 2 + 15 value insts before the call)
    };
    let b10 = Block {
        params: vec![ValType::I64, ValType::I64],
        insts: e,
        term,
    };

    Func {
        results: main_results.to_vec(),
        blocks: vec![b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10],
        params: vec![ValType::I32; n_handles],
    }
}

/// Synthesize `__svm_malloc(size:i64) -> i64`: an on-demand **bump allocator** that grows the guest
/// heap into the window's reserved tail by `vm_map`-committing pages as needed (§3e/§4 — the §1a
/// "grow past the initial window" capability). State is two `i64`s in the low scratch: `HEAP_BRK` (the
/// next free address) and `HEAP_TOP` (the committed boundary). `free` is a no-op and the heap never
/// reuses, so every result is freshly `vm_map`-zeroed memory (hence `calloc` ≡ `malloc`).
///
/// ```text
///   block0(size):                              ; data=brk+16; new=align16(data+size)
///     brk = load.i64 [HEAP_BRK]; top = load.i64 [HEAP_TOP]
///     grow? = new >u top   → grow(brk,size,new,top) : commit(brk,size,new)
///   grow(brk,size,new,top):                    ; commit [top, page_up(new)) via the Memory cap
///     vm_map(mem_handle, top, page_up(new) - top, RW); store.i64 [HEAP_TOP] = page_up(new)
///     → commit(brk,size,new)
///   commit(brk,size,new):                       ; now the page is mapped: write the header + publish
///     store.i64 [brk] = size                    ; 16-byte size header (for realloc)
///     store.i64 [HEAP_BRK] = new
///     return brk + 16                           ; the data pointer
/// ```
/// The header is written in `commit` (not `block0`) because on the first `malloc` `brk` is an
/// *uncommitted* reserved page — only `grow` (or the prior commit) maps it.
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
    // block0(size=0): brk = *HEAP_BRK; new = align16(brk+16+size); branch on new > *HEAP_TOP. No
    // heap write here — `brk` may be an uncommitted page until `grow` maps it.
    let b0 = Block {
        params: vec![ValType::I64], // size = v0
        insts: vec![
            Inst::ConstI64(HEAP_BRK as i64), // v1
            load_i64(1),                     // v2 = brk
            Inst::ConstI64(16),              // v3
            i64add(2, 3),                    // v4 = brk + 16
            i64add(4, 0),                    // v5 = brk+16+size
            Inst::ConstI64(15),              // v6
            i64add(5, 6),                    // v7
            Inst::ConstI64(!15i64),          // v8 = ~15
            i64and(7, 8),                    // v9 = new (aligned)
            Inst::ConstI64(HEAP_TOP as i64), // v10
            load_i64(10),                    // v11 = top
            Inst::IntCmp {
                ty: IntTy::I64,
                op: CmpOp::GtU,
                a: 9,
                b: 11,
            }, // v12 = new > top
        ],
        term: Terminator::BrIf {
            cond: 12,
            then_blk: 1, // grow(brk=v2, size=v0, new=v9, top=v11)
            then_args: vec![2, 0, 9, 11],
            else_blk: 2, // commit(brk=v2, size=v0, new=v9)
            else_args: vec![2, 0, 9],
        },
    };

    // grow(brk=0, size=1, new=2, top=3): commit [top, page_up(new)) via vm_map, update HEAP_TOP.
    let page = STACK_PAGE as i64; // commit in ≥-OS-page units (16 KiB covers any real page)
    let g = Block {
        params: vec![ValType::I64, ValType::I64, ValType::I64, ValType::I64], // brk, size, new, top
        insts: vec![
            Inst::ConstI64(page - 1),            // v4
            i64add(2, 4),                        // v5 = new + (PAGE-1)
            Inst::ConstI64(!(page - 1)),         // v6 = ~(PAGE-1)
            i64and(5, 6),                        // v7 = limit (page-aligned)
            Inst::ConstI64(STASH_MEMORY as i64), // v8
            Inst::Load {
                op: LoadOp::I32,
                addr: 8,
                offset: 0,
                align: 0,
            }, // v9 = mem handle
            Inst::IntBin {
                ty: IntTy::I64,
                op: BinOp::Sub,
                a: 7,
                b: 3,
            }, // v10 = limit - top (len)
            Inst::ConstI32(PROT_RW),             // v11 = prot
            Inst::CallImport {
                import: vm_map_import,
                sig: import_sig("vm_map"),
                handle: 9,
                args: vec![3, 10, 11],
            }, // v12 = map result (ignored)
            Inst::ConstI64(HEAP_TOP as i64),     // v13
            store_i64(13, 7),                    // *HEAP_TOP = limit
        ],
        term: Terminator::Br {
            target: 2,
            args: vec![0, 1, 2], // commit(brk, size, new)
        },
    };

    // commit(brk=0, size=1, new=2): the page is now mapped — write the size header at brk, publish
    // the new break, and return the data pointer brk+16.
    let c = Block {
        params: vec![ValType::I64, ValType::I64, ValType::I64], // brk, size, new
        insts: vec![
            store_i64(0, 1),                 // *brk = size (header) — no value
            Inst::ConstI64(HEAP_BRK as i64), // v3
            store_i64(3, 2),                 // *HEAP_BRK = new — no value
            Inst::ConstI64(16),              // v4
            i64add(0, 4),                    // v5 = brk + 16 (data)
        ],
        term: Terminator::Return(vec![5]), // data
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
#[derive(Clone, Default)]
struct Helpers {
    /// `__svm_memset(dst:i64, byte:i32, len:i64)` — fill `len` bytes at `dst` with `byte`'s low byte.
    memset: Option<u32>,
    /// `__svm_memcpy(dst:i64, src:i64, len:i64)` — copy `len` bytes `src`→`dst` (forward; no overlap).
    memcpy: Option<u32>,
    /// `__svm_malloc(size:i64) -> i64` — the `vm_map`-growing bump allocator (`malloc`/`calloc`).
    malloc: Option<u32>,
    /// `__svm_utoa(value:i64, base:i64, bufend:i64) -> i64` — unsigned→ASCII for `printf`.
    utoa: Option<u32>,
    /// `__svm_strlen(p:i64) -> i64` — the NUL-terminated byte length, for `printf` `%s`.
    strlen: Option<u32>,
    /// `__svm_strcmp(a:i64, b:i64) -> i32` — NUL-terminated lexicographic byte compare (unsigned-char
    /// difference at the first mismatch). Backs `strcmp` and, in the C locale, `strcoll`.
    strcmp: Option<u32>,
    /// `__svm_strchr(s:i64, c:i32) -> i64` — first `(unsigned char)c` in `s`, or NULL (`c==0` → the
    /// terminating NUL). Backs `strchr`.
    strchr: Option<u32>,
    /// `__svm_strcpy(dst:i64, src:i64) -> i64` — copy `src` (incl. NUL) to `dst`, return `dst`.
    strcpy: Option<u32>,
    /// `__svm_strspn(s:i64, set:i64) -> i64` — length of the initial run of `s` within `set`.
    strspn: Option<u32>,
    /// `__svm_strpbrk(s:i64, set:i64) -> i64` — first byte of `s` that is in `set`, or NULL.
    strpbrk: Option<u32>,
    /// `__svm_ldexp(x:f64, n:i32) -> f64` — `x · 2^n` (the `scalbn` algorithm), bit-exact to libc.
    ldexp: Option<u32>,
    /// Fail-closed trap stub for `pow` (and the other libm transcendentals): bit-exact vs native
    /// requires matching a specific host libm, so it stays a `synth_trap_stub` pending that decision.
    pow_stub: Option<u32>,
    /// Fail-closed trap stub for `fmod`. Exact-synthesizable (the remainder is always representable, so
    /// glibc/musl's bit-twiddling algorithm is bit-exact with no libm dependency) — a sizeable
    /// loop-nest CFG slated as the next exact slice; stubbed until then.
    fmod_stub: Option<u32>,
    /// `__svm_frexp(x:f64, eptr:i64) -> f64` — split `x` into mantissa∈[0.5,1) and exponent; writes the
    /// exponent to `*eptr` (an `int`). Pure bit ops, bit-exact to glibc `frexp` (incl. zero/subnormal/
    /// inf/nan: `*eptr=0`, returns `x+x`).
    frexp: Option<u32>,
    /// More fail-closed stubs: `strtod` (string→double — not exercised by an integer-only path),
    /// `localeconv`/`__errno_location` (locale + errno). Translate, trap if run. (`snprintf` is real —
    /// it reuses the `printf` engine, see [`lower_snprintf`].)
    strtod_stub: Option<u32>,
    localeconv_stub: Option<u32>,
    errno_stub: Option<u32>,
    /// `time(t)` — the RNG seed source in `makeseed`; executed during state creation but the seed does
    /// not affect a deterministic script's result, so a constant `0` suffices (a real `Clock` cap later).
    time_zero: Option<u32>,
    /// `__svm_realloc(p:i64, n:i64) -> i64` — `realloc` over the header-bearing bump allocator.
    realloc: Option<u32>,
    /// `__svm_memmove(dst:i64, src:i64, len:i64)` — overlap-safe (direction-aware) byte copy.
    memmove: Option<u32>,
    /// `__svm_memcmp(a:i64, b:i64, len:i64) -> i32` — compare `len` bytes as unsigned; `0` if equal,
    /// else the signed first-mismatch difference (`a[i] - b[i]`). Backs `memcmp` *and* `bcmp` (Rust's
    /// `[u8]`/slice equality and `BTreeMap` key ordering emit these).
    memcmp: Option<u32>,
    /// `__svm_memchr(s:i64, c:i32, n:i64) -> i64` — first occurrence of `(unsigned char)c` in the first
    /// `n` bytes at `s`, or NULL. Backs `memchr` (string/buffer scans).
    memchr: Option<u32>,
    /// `__svm_eh_destroy(sp:i64, exn:i64, dtor:i64)` — run an exception object's destructor for
    /// `__cxa_end_catch` (indirect-call `dtor` on `exn` when non-null). Present only when the module
    /// catches a C++ exception (calls `__cxa_end_catch`).
    eh_destroy: Option<u32>,
    /// `__svm_eh_unwind(base:i64)` — the throw/rethrow/resume tail: pop the innermost handler and
    /// long-jump into its checkpoint, or trap (`std::terminate`) when the handler stack is empty (an
    /// uncaught exception). Present whenever the module uses C++ EH.
    eh_unwind: Option<u32>,
    /// `__svm_getenv(name:i64) -> i64` — scan the §3e env strings for `name=`, returning the value
    /// pointer (just past the `=`) or NULL. Reads the blob in the reserved low scratch directly.
    getenv: Option<u32>,
    /// `__svm_udivmod128(nlo, nhi, dlo, dhi) -> (qlo, qhi, rlo, rhi)` — unsigned 128÷128 quotient +
    /// remainder (I14). Present only when the module has an i128 `udiv`/`sdiv`/`urem`/`srem`; signed
    /// div/rem reuse it (the lowering abs-es the operands and re-signs the result).
    udivmod128: Option<u32>,
    /// The bignum float-formatter helper family (all of `%f`/`%e`/`%g`): the ten big-integer
    /// primitives, the `dtoa_digits` engine, and the three formatters (`dtoa_sci`/`dtoa_gen`/
    /// `dtoa_fix`). Indices are interdependent (`dtoa_digits` calls the primitives; the formatters
    /// call `dtoa_digits` / the primitives), so they are appended as one block in this order.
    big_zero: Option<u32>,
    big_copy: Option<u32>,
    big_cmp: Option<u32>,
    big_sub: Option<u32>,
    big_mul: Option<u32>,
    big_shl: Option<u32>,
    big_shr1: Option<u32>,
    big_inc: Option<u32>,
    big_divmod: Option<u32>,
    big_iszero: Option<u32>,
    dtoa_digits: Option<u32>,
    dtoa_sci: Option<u32>,
    /// `__svm_dtoa_gen(bits, prec, width, flags, scratch) -> i64` — the `%g`/`%G` formatter.
    dtoa_gen: Option<u32>,
    /// `__svm_dtoa_fix_big(bits, prec, width, flags, scratch) -> i64` — the `%f`/`%F` formatter.
    dtoa_fix: Option<u32>,
    /// Narrow (i8/i16) atomic CAS-loop helpers: `__svm_atomic_rmw_narrow` and
    /// `__svm_atomic_cas_narrow`. Present only when the module uses a narrow atomic rmw/store/cmpxchg.
    atomic_rmw_narrow: Option<u32>,
    atomic_cas_narrow: Option<u32>,
    /// Base window address of the reserved float scratch (`= float_scratch_base`), so the printf
    /// lowering can `emit_write` the field a bignum formatter fills at `+FMT_OUT_O`.
    float_scratch: Option<u64>,
    /// Window address of the **pointer cell** each glibc ctype locator returns: `__ctype_b_loc()` →
    /// `&(ptr to the u16 flags table, base at index 0)`, `__ctype_tolower_loc()` / `__ctype_toupper_loc()`
    /// → `&(ptr to the i32 case-map table)`. The tables + cells are synthesized as read-only data in the
    /// module image (no runtime init — works for a `run`-only module), so the locator lowers to a const.
    ctype_b_loc: Option<u64>,
    ctype_tolower_loc: Option<u64>,
    ctype_toupper_loc: Option<u64>,
    /// Base window address of the reserved C++ exception-handling region (the handler stack + the
    /// current-exception slots + the per-handler buf slots), reserved only when `need_eh`. The
    /// `invoke`/`landingpad`/`resume`/`__cxa_*` lowerings address their state at `+EH_*_O` off this.
    eh_base: Option<u64>,
    /// Module-global typeinfo-id table: each `@_ZTI*` typeinfo global referenced by a `__cxa_throw`
    /// or `llvm.eh.typeid.for` is assigned a small distinct nonzero id (1-based, in first-seen
    /// order). The throw stores the thrown type's id into the selector slot; `eh.typeid.for` returns
    /// the same id, so clang's emitted catch dispatch compares them and selects the right handler.
    eh_typeids: HashMap<String, u32>,
    /// Subtype match table for polymorphic catch: typeinfo name → the thrown-type ids a `catch` of
    /// that type matches (itself + every thrown type derived from it; see [`build_eh_subtypes`]). The
    /// `llvm.eh.typeid.for` lowering uses it to make `icmp eq sel, typeid.for(C)` mean "thrown is-a C".
    eh_subtype_ids: HashMap<String, Vec<u32>>,
}

/// Does the module call an external (not guest-defined) function with name `n`?
/// Does the module use a **narrow** (i8/i16) atomic that needs the CAS-loop helpers — an `atomicrmw`,
/// a `cmpxchg`, or an atomic `store` on an i8/i16? (A narrow atomic *load* is emulated inline, so it
/// doesn't pull in a helper.) Wide (i32/i64) atomics lower directly and need no helper.
fn uses_narrow_atomic(m: &LModule) -> bool {
    let narrow = |o: &Operand| matches!(o.get_type(&m.types).as_ref(), Type::IntegerType { bits } if *bits == 8 || *bits == 16);
    m.functions.iter().flat_map(|f| &f.basic_blocks).any(|bb| {
        bb.instrs.iter().any(|i| match i {
            Instruction::AtomicRMW(r) => narrow(&r.value),
            Instruction::CmpXchg(cx) => narrow(&cx.expected),
            Instruction::Store(st) => st.atomicity.is_some() && narrow(&st.value),
            _ => false,
        })
    })
}

/// Does the module have an i128 `udiv`/`sdiv`/`urem`/`srem`? (clang keeps these as IR ops at `-O2`;
/// the on-ramp lowers them to the synthesized 128÷128 long-division helper, `__svm_udivmod128`.)
fn uses_i128_divrem(m: &LModule) -> bool {
    let is_i128 = |o: &Operand| {
        matches!(
            o.get_type(&m.types).as_ref(),
            Type::IntegerType { bits: 128 }
        )
    };
    m.functions.iter().flat_map(|f| &f.basic_blocks).any(|bb| {
        bb.instrs.iter().any(|i| match i {
            Instruction::UDiv(x) => is_i128(&x.operand0),
            Instruction::SDiv(x) => is_i128(&x.operand0),
            Instruction::URem(x) => is_i128(&x.operand0),
            Instruction::SRem(x) => is_i128(&x.operand0),
            _ => false,
        })
    })
}

fn calls_external(m: &LModule, defined: &HashMap<String, u32>, want: &str) -> bool {
    m.functions.iter().flat_map(|f| &f.basic_blocks).any(|bb| {
        bb.instrs.iter().any(|i| {
            matches!(i, Instruction::Call(c)
                if callee_name(c).is_some_and(|n| n == want && !defined.contains_key(&n)))
        })
    })
}

/// Scan the module for C++ exception-handling constructs. Returns `(uses_eh, typeinfo_ids)`:
/// `uses_eh` is set by any `invoke`/`resume` terminator, any `landingpad`, or any call into the
/// Itanium `__cxa_*` / `llvm.eh.typeid.for` runtime. The id table interns each typeinfo global —
/// the `@_ZTI*` operand of `__cxa_throw` (arg 1) or `llvm.eh.typeid.for` (arg 0) — to a distinct
/// 1-based id in first-seen order, so a thrown type's stored selector matches the catch clause's
/// `eh.typeid.for` exactly when (and only when) the types are the same.
fn scan_eh(m: &LModule) -> (bool, HashMap<String, u32>, Vec<String>) {
    let mut uses = false;
    let mut ids: HashMap<String, u32> = HashMap::new();
    // The typeinfos that are actually *thrown* (`__cxa_throw` arg1) — the set of types that can reach
    // a landing pad's selector. Drives the subtype table (`build_eh_subtypes`): a `catch (B&)` matches
    // a thrown `D` exactly when some thrown `D` has `B` in its base chain.
    let mut thrown: Vec<String> = Vec::new();
    let intern = |op: &Operand, ids: &mut HashMap<String, u32>| {
        if let Some(n) = global_name_of(op) {
            let next = ids.len() as u32 + 1;
            ids.entry(n).or_insert(next);
        }
    };
    for f in &m.functions {
        for bb in &f.basic_blocks {
            if matches!(bb.term, LTerm::Invoke(_) | LTerm::Resume(_)) {
                uses = true;
            }
            for i in &bb.instrs {
                match i {
                    Instruction::LandingPad(_) => uses = true,
                    Instruction::Call(c) => match callee_name(c).as_deref() {
                        Some("__cxa_throw") => {
                            uses = true;
                            if let Some((op, _)) = c.arguments.get(1) {
                                intern(op, &mut ids);
                                if let Some(n) = global_name_of(op) {
                                    thrown.push(n);
                                }
                            }
                        }
                        Some("llvm.eh.typeid.for") => {
                            uses = true;
                            if let Some((op, _)) = c.arguments.first() {
                                intern(op, &mut ids);
                            }
                        }
                        Some(
                            "__cxa_begin_catch"
                            | "__cxa_get_exception_ptr"
                            | "__cxa_end_catch"
                            | "__cxa_allocate_exception"
                            | "__cxa_rethrow"
                            | "_Unwind_Resume",
                        ) => uses = true,
                        _ => {}
                    },
                    _ => {}
                }
            }
        }
    }
    (uses, ids, thrown)
}

/// Recursively collect the `@_ZTI*` base-class typeinfos referenced by a typeinfo global's
/// initializer. A `__class_type_info` (a root, no bases) yields none; a `__si_class_type_info` yields
/// its one base (initializer field 2); a `__vmi_class_type_info` yields each base (nested in the
/// base-info array). Every `_ZTI`-prefixed reference inside a typeinfo initializer *is* a direct base
/// — field 0 is the `_ZTV` abi vtable and field 1 the `_ZTS` type-name string, neither prefixed
/// `_ZTI` — so a prefix filter over a full walk recovers exactly the bases.
fn collect_ti_bases(c: &Constant, out: &mut Vec<String>) {
    match c {
        Constant::GlobalReference { name, .. } => {
            let n = name_str(name);
            if n.starts_with("_ZTI") {
                out.push(n);
            }
        }
        Constant::Struct { values, .. } => values.iter().for_each(|v| collect_ti_bases(v, out)),
        Constant::Array { elements, .. } | Constant::Vector(elements) => {
            elements.iter().for_each(|v| collect_ti_bases(v, out))
        }
        Constant::BitCast(bc) => collect_ti_bases(bc.operand.as_ref(), out),
        _ => {}
    }
}

/// Build the C++ exception subtype table: for each typeinfo, the set of *thrown* type ids that a
/// `catch` of that type matches — its own id plus every thrown type derived from it. `catch (B&)`
/// matches a thrown `D` when `B` is a base of `D` (the Itanium personality's `__do_catch` walk); we
/// precompute that walk here over the typeinfo base-chains so `llvm.eh.typeid.for` can decide it at
/// runtime with a flat id compare. A type whose typeinfo is external (no initializer — e.g. a `std`
/// type) contributes only its own id: its base chain is invisible to a single TU.
fn build_eh_subtypes(
    m: &LModule,
    ids: &HashMap<String, u32>,
    thrown: &[String],
) -> HashMap<String, Vec<u32>> {
    // Direct bases per typeinfo name (only those defined — with an initializer — in this module).
    let mut bases: HashMap<String, Vec<String>> = HashMap::new();
    for g in &m.global_vars {
        let n = name_str(&g.name);
        if !n.starts_with("_ZTI") {
            continue;
        }
        if let Some(init) = &g.initializer {
            let mut b = Vec::new();
            collect_ti_bases(init.as_ref(), &mut b);
            bases.insert(n, b);
        }
    }
    // For each thrown type, record its id against every ancestor (transitive closure up the base
    // chain, including itself) — that ancestor's `catch` clause must match this thrown type.
    let mut out: HashMap<String, Vec<u32>> = HashMap::new();
    for t in thrown {
        let Some(&tid) = ids.get(t) else { continue };
        let mut seen: HashSet<String> = HashSet::new();
        let mut stack = vec![t.clone()];
        while let Some(x) = stack.pop() {
            if !seen.insert(x.clone()) {
                continue;
            }
            out.entry(x.clone()).or_default().push(tid);
            if let Some(bs) = bases.get(&x) {
                stack.extend(bs.iter().cloned());
            }
        }
    }
    for v in out.values_mut() {
        v.sort_unstable();
        v.dedup();
    }
    out
}

/// Does the module call the heap allocator (`malloc`/`calloc`)? (`free` is a no-op; `realloc` is
/// still `Unsupported`.)
fn needs_malloc(m: &LModule, defined: &HashMap<String, u32>) -> bool {
    calls_external(m, defined, "malloc") || calls_external(m, defined, "calloc")
}

/// Does any mem intrinsic need a runtime helper — a **non-constant** length, or a constant one too
/// large to unroll inline? Returns `(needs_memset, needs_memcpy, needs_memmove)`.
fn needs_mem_helpers(m: &LModule) -> (bool, bool, bool) {
    let (mut set, mut cpy, mut mov) = (false, false, false);
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
                } else if name.starts_with("llvm.memmove") && big(c) {
                    mov = true;
                }
            }
        }
    }
    (set, cpy, mov)
}

/// Synthesize `__svm_realloc(p:i64, n:i64) -> i64`: `realloc` over the header-bearing bump allocator.
/// `realloc(NULL, n)` ≡ `malloc(n)`; otherwise allocate `n`, copy `min(old, n)` bytes (the old size
/// is the 16-byte header at `p-16`), and return the new pointer (the old block is leaked — `free` is
/// a no-op). The copy never overlaps: the fresh block sits above the old one by construction.
fn synth_realloc(malloc_idx: u32, memcpy_idx: u32) -> Func {
    use svm_ir::LoadOp;
    // block0(p=0, n=1): p == 0 ? malloc(n) : copy from the old block.
    let b0 = Block {
        params: vec![ValType::I64, ValType::I64],
        insts: vec![
            Inst::ConstI64(0), // v2
            Inst::IntCmp {
                ty: IntTy::I64,
                op: CmpOp::Eq,
                a: 0,
                b: 2,
            }, // v3 = p == 0
        ],
        term: Terminator::BrIf {
            cond: 3,
            then_blk: 1, // null_case(n)
            then_args: vec![1],
            else_blk: 2, // have(p, n)
            else_args: vec![0, 1],
        },
    };
    // null_case(n=0): return malloc(n).
    let null_case = Block {
        params: vec![ValType::I64],
        insts: vec![Inst::Call {
            func: malloc_idx,
            args: vec![0],
        }], // v1 = q
        term: Terminator::Return(vec![1]),
    };
    // have(p=0, n=1): old = *(p-16); q = malloc(n); memcpy(q, p, min(old, n)); return q.
    let have = Block {
        params: vec![ValType::I64, ValType::I64],
        insts: vec![
            Inst::ConstI64(16), // v2
            Inst::IntBin {
                ty: IntTy::I64,
                op: BinOp::Sub,
                a: 0,
                b: 2,
            }, // v3 = p - 16 (header)
            Inst::Load {
                op: LoadOp::I64,
                addr: 3,
                offset: 0,
                align: 0,
            }, // v4 = old size
            Inst::Call {
                func: malloc_idx,
                args: vec![1],
            }, // v5 = q
            Inst::IntCmp {
                ty: IntTy::I64,
                op: CmpOp::LtU,
                a: 4,
                b: 1,
            }, // v6 = old < n
            Inst::Select {
                cond: 6,
                a: 4,
                b: 1,
            }, // v7 = min(old, n)
            Inst::Call {
                func: memcpy_idx,
                args: vec![5, 0, 7],
            }, // memcpy(q, p, min) — void
        ],
        term: Terminator::Return(vec![5]), // q
    };
    Func {
        params: vec![ValType::I64, ValType::I64],
        results: vec![ValType::I64],
        blocks: vec![b0, null_case, have],
    }
}

/// Synthesize `__svm_eh_destroy(sp:i64, exn:i64, dtor:i64)`: run an exception object's destructor on
/// behalf of `__cxa_end_catch`. `dtor` is a `void(T*)` funcref (the third `__cxa_throw` argument) or
/// `0` for a trivially-destructible type; when non-null it is indirect-called on the object `exn`,
/// passing the helper's own data-SP as the destructor's frame base (the helper holds no frame, so the
/// destructor sits directly above `__cxa_end_catch`'s). The null case returns immediately. This is
/// the guard `__cxa_end_catch` cannot express inline — it lowers in effect position, where it has no
/// way to branch — so the conditional indirect call is pushed into this helper instead.
fn synth_eh_destroy() -> Func {
    let params = vec![ValType::I64, ValType::I64, ValType::I64];
    // block0(sp=0, exn=1, dtor=2): have = dtor != 0; br_if have call(sp,exn,dtor) done()
    let entry = Block {
        params: params.clone(),
        insts: vec![
            Inst::ConstI64(0), // v3
            Inst::IntCmp {
                ty: IntTy::I64,
                op: CmpOp::Ne,
                a: 2,
                b: 3,
            }, // v4 = dtor != 0
        ],
        term: Terminator::BrIf {
            cond: 4,
            then_blk: 1,
            then_args: vec![0, 1, 2],
            else_blk: 2,
            else_args: vec![],
        },
    };
    // call(sp=0, exn=1, dtor=2): the destructor's funcref index is the low 32 bits of `dtor`; call it
    // `dtor(sp, exn)` (void — the `(data-SP, this)` calling convention), then return.
    let call = Block {
        params: params.clone(),
        insts: vec![
            Inst::Convert {
                op: ConvOp::WrapI64,
                a: 2,
            }, // v3 = funcref idx
            Inst::CallIndirect {
                ty: svm_ir::FuncType {
                    params: vec![ValType::I64, ValType::I64],
                    results: vec![],
                },
                idx: 3,
                args: vec![0, 1], // (dtor_sp = sp, this = exn)
            }, // void
        ],
        term: Terminator::Return(vec![]),
    };
    let done = Block {
        params: vec![],
        insts: vec![],
        term: Terminator::Return(vec![]),
    };
    Func {
        params,
        results: vec![],
        blocks: vec![entry, call, done],
    }
}

/// Synthesize `__svm_eh_unwind(base:i64)`: the throw/rethrow/resume tail. Pop the innermost active
/// handler (`HSP-1`) and long-jump into its `invoke`-installed checkpoint — *unless* the handler
/// stack is empty (`HSP == 0`: no enclosing `try` anywhere up the call chain, i.e. an uncaught
/// exception), in which case trap (`std::terminate`). The empty-stack branch is why this is a helper:
/// `__cxa_throw`/`__cxa_rethrow`/`_Unwind_Resume` lower in effect position and cannot branch inline,
/// and a bare `HSP-1` would underflow to a bogus slot and long-jump into garbage.
fn synth_eh_unwind() -> Func {
    // block0(base=0): k = *(base+EH_HSP_O); if k == 0 -> terminate() else unwind(hsp_addr, k, base)
    let entry = Block {
        params: vec![ValType::I64],
        insts: vec![
            Inst::ConstI64(EH_HSP_O as i64), // v1
            Inst::IntBin {
                ty: IntTy::I64,
                op: BinOp::Add,
                a: 0,
                b: 1,
            }, // v2 = hsp_addr
            Inst::Load {
                op: LoadOp::I64,
                addr: 2,
                offset: 0,
                align: 8,
            }, // v3 = k (HSP)
            Inst::ConstI64(0),               // v4
            Inst::IntCmp {
                ty: IntTy::I64,
                op: CmpOp::Eq,
                a: 3,
                b: 4,
            }, // v5 = k == 0
        ],
        term: Terminator::BrIf {
            cond: 5,
            then_blk: 1, // terminate()
            then_args: vec![],
            else_blk: 2, // unwind(hsp_addr, k, base)
            else_args: vec![2, 3, 0],
        },
    };
    // terminate(): an uncaught exception — trap (the on-ramp's `unreachable` is the clean fault).
    let terminate = Block {
        params: vec![],
        insts: vec![],
        term: Terminator::Unreachable,
    };
    // unwind(hsp_addr=0, k=1, base=2): *(hsp_addr) = k-1; longjmp(base+EH_BUFS_O + (k-1)*EH_SLOT, 1).
    let unwind = Block {
        params: vec![ValType::I64, ValType::I64, ValType::I64],
        insts: vec![
            Inst::ConstI64(1), // v3
            Inst::IntBin {
                ty: IntTy::I64,
                op: BinOp::Sub,
                a: 1,
                b: 3,
            }, // v4 = target = k - 1
            Inst::Store {
                op: StoreOp::I64,
                addr: 0,
                value: 4,
                offset: 0,
                align: 8,
            }, // *(hsp_addr) = target
            Inst::ConstI64(EH_BUFS_O as i64), // v5
            Inst::IntBin {
                ty: IntTy::I64,
                op: BinOp::Add,
                a: 2,
                b: 5,
            }, // v6 = bufs_base = base + EH_BUFS_O
            Inst::ConstI64(EH_SLOT as i64), // v7
            Inst::IntBin {
                ty: IntTy::I64,
                op: BinOp::Mul,
                a: 4,
                b: 7,
            }, // v8 = off = target * EH_SLOT
            Inst::IntBin {
                ty: IntTy::I64,
                op: BinOp::Add,
                a: 6,
                b: 8,
            }, // v9 = buf = bufs_base + off
            Inst::ConstI32(1), // v10
            Inst::LongJmp { buf: 9, val: 10 },
        ],
        term: Terminator::Unreachable,
    };
    Func {
        params: vec![ValType::I64],
        results: vec![],
        blocks: vec![entry, terminate, unwind],
    }
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

/// Synthesize `__svm_memcmp(a:i64, b:i64, len:i64) -> i32`: compare `len` bytes as **unsigned**.
/// Returns `0` if all equal, else `a[i] - b[i]` at the first mismatch — each byte is loaded
/// zero-extended (0..=255), so the `i32` difference carries `memcmp`'s sign. Five blocks: entry seeds
/// `i=0`; the loop tests `i <u len` (fell through ⇒ all equal ⇒ `0`); the body loads both bytes and
/// either returns the difference or steps `i`. Backs `memcmp` *and* `bcmp` (a `bcmp` caller only tests
/// `!= 0`, which this preserves).
fn synth_memcmp() -> Func {
    use svm_ir::LoadOp;
    let params = vec![ValType::I64, ValType::I64, ValType::I64]; // a, b, len
                                                                 // block0(a=0, b=1, len=2): i = 0; br loop(a, b, len, i)
    let entry = Block {
        params: params.clone(),
        insts: vec![Inst::ConstI64(0)], // v3 = i
        term: Terminator::Br {
            target: 1,
            args: vec![0, 1, 2, 3],
        },
    };
    let loop_params = vec![ValType::I64, ValType::I64, ValType::I64, ValType::I64]; // a, b, len, i
                                                                                    // loop(a=0, b=1, len=2, i=3): cond = i <u len; br_if cond → body, else → equal
    let test = Block {
        params: loop_params.clone(),
        insts: vec![Inst::IntCmp {
            ty: IntTy::I64,
            op: CmpOp::LtU,
            a: 3,
            b: 2,
        }], // v4
        term: Terminator::BrIf {
            cond: 4,
            then_blk: 2, // body
            then_args: vec![0, 1, 2, 3],
            else_blk: 4, // equal
            else_args: vec![],
        },
    };
    // body(a=0, b=1, len=2, i=3): av = a[i]; bv = b[i]; if av != bv → diff(av,bv) else loop(.., i+1)
    let body = Block {
        params: loop_params,
        insts: vec![
            Inst::IntBin {
                ty: IntTy::I64,
                op: BinOp::Add,
                a: 0,
                b: 3,
            }, // v4 = a + i
            Inst::Load {
                op: LoadOp::I32_8U,
                addr: 4,
                offset: 0,
                align: 0,
            }, // v5 = av
            Inst::IntBin {
                ty: IntTy::I64,
                op: BinOp::Add,
                a: 1,
                b: 3,
            }, // v6 = b + i
            Inst::Load {
                op: LoadOp::I32_8U,
                addr: 6,
                offset: 0,
                align: 0,
            }, // v7 = bv
            Inst::IntCmp {
                ty: IntTy::I32,
                op: CmpOp::Ne,
                a: 5,
                b: 7,
            }, // v8 = av != bv
            Inst::ConstI64(1), // v9
            Inst::IntBin {
                ty: IntTy::I64,
                op: BinOp::Add,
                a: 3,
                b: 9,
            }, // v10 = i + 1
        ],
        term: Terminator::BrIf {
            cond: 8,
            then_blk: 3, // diff(av, bv)
            then_args: vec![5, 7],
            else_blk: 1, // loop(a, b, len, i+1)
            else_args: vec![0, 1, 2, 10],
        },
    };
    // diff(av=0, bv=1): return av - bv (signed i32 difference of two unsigned bytes).
    let diff = Block {
        params: vec![ValType::I32, ValType::I32],
        insts: vec![Inst::IntBin {
            ty: IntTy::I32,
            op: BinOp::Sub,
            a: 0,
            b: 1,
        }], // v2
        term: Terminator::Return(vec![2]),
    };
    // equal(): all bytes matched → 0.
    let equal = Block {
        params: vec![],
        insts: vec![Inst::ConstI32(0)], // v0
        term: Terminator::Return(vec![0]),
    };
    Func {
        params,
        results: vec![ValType::I32],
        blocks: vec![entry, test, body, diff, equal],
    }
}

/// Synthesize `__svm_memchr(s:i64, c:i32, n:i64) -> i64`: scan the first `n` bytes at `s` for the byte
/// `(unsigned char)c`, returning a pointer to it or NULL. A counted forward byte loop.
///
/// ```text
///   entry(s, c, n):       end = s + n          → loop(s, end, c)
///   loop(cur, end, c):    cur == end ? notfound : body
///   body(cur, end, c):    *cur == (c & 0xff) ? found(cur) : loop(cur+1, end, c)
///   found(p):             return p
///   notfound:             return 0
/// ```
fn synth_memchr() -> Func {
    use svm_ir::LoadOp;
    let params = vec![ValType::I64, ValType::I32, ValType::I64]; // s, c, n (matches the call ABI)
    let entry = Block {
        params: params.clone(),
        insts: vec![Inst::IntBin {
            ty: IntTy::I64,
            op: BinOp::Add,
            a: 0,
            b: 2,
        }], // v3 = end = s + n
        term: Terminator::Br {
            target: 1,
            args: vec![0, 3, 1], // loop(cur=s, end, c)
        },
    };
    let loop_params = vec![ValType::I64, ValType::I64, ValType::I32]; // cur, end, c
    let test = Block {
        params: loop_params.clone(),
        insts: vec![Inst::IntCmp {
            ty: IntTy::I64,
            op: CmpOp::Eq,
            a: 0,
            b: 1,
        }], // v3 = cur == end
        term: Terminator::BrIf {
            cond: 3,
            then_blk: 4, // notfound
            then_args: vec![],
            else_blk: 2, // body
            else_args: vec![0, 1, 2],
        },
    };
    let body = Block {
        params: loop_params,
        insts: vec![
            Inst::Load {
                op: LoadOp::I32_8U,
                addr: 0,
                offset: 0,
                align: 0,
            }, // v3 = *cur
            Inst::ConstI32(0xff), // v4
            Inst::IntBin {
                ty: IntTy::I32,
                op: BinOp::And,
                a: 2,
                b: 4,
            }, // v5 = c & 0xff
            Inst::IntCmp {
                ty: IntTy::I32,
                op: CmpOp::Eq,
                a: 3,
                b: 5,
            }, // v6 = *cur == (c & 0xff)
            Inst::ConstI64(1),    // v7
            Inst::IntBin {
                ty: IntTy::I64,
                op: BinOp::Add,
                a: 0,
                b: 7,
            }, // v8 = cur + 1
        ],
        term: Terminator::BrIf {
            cond: 6,
            then_blk: 3, // found(cur)
            then_args: vec![0],
            else_blk: 1, // loop(cur+1, end, c)
            else_args: vec![8, 1, 2],
        },
    };
    let found = Block {
        params: vec![ValType::I64], // p
        insts: vec![],
        term: Terminator::Return(vec![0]),
    };
    let notfound = Block {
        params: vec![],
        insts: vec![Inst::ConstI64(0)], // v0
        term: Terminator::Return(vec![0]),
    };
    Func {
        params,
        results: vec![ValType::I64],
        blocks: vec![entry, test, body, found, notfound],
    }
}

/// Synthesize `__svm_memmove(dst:i64, src:i64, len:i64)`: an **overlap-safe** byte copy — forward
/// when `dst <= src`, backward otherwise (the direction `memcpy` can't do). 8 blocks: the direction
/// branch, then a forward and a backward counted byte loop sharing the `done` return.
fn synth_memmove() -> Func {
    use svm_ir::{LoadOp, StoreOp};
    let p3 = || vec![ValType::I64, ValType::I64, ValType::I64]; // dst, src, len
    let p4 = || vec![ValType::I64, ValType::I64, ValType::I64, ValType::I64]; // + i
    let add = |a: ValIdx, b: ValIdx| Inst::IntBin {
        ty: IntTy::I64,
        op: BinOp::Add,
        a,
        b,
    };
    let load8 = |addr: ValIdx| Inst::Load {
        op: LoadOp::I32_8U,
        addr,
        offset: 0,
        align: 0,
    };
    let store8 = |addr: ValIdx, value: ValIdx| Inst::Store {
        op: StoreOp::I32_8,
        addr,
        value,
        offset: 0,
        align: 0,
    };
    // block0(dst=0,src=1,len=2): dst <=u src ? forward : backward.
    let b0 = Block {
        params: p3(),
        insts: vec![Inst::IntCmp {
            ty: IntTy::I64,
            op: CmpOp::LeU,
            a: 0,
            b: 1,
        }], // v3
        term: Terminator::BrIf {
            cond: 3,
            then_blk: 1, // fwd
            then_args: vec![0, 1, 2],
            else_blk: 4, // bwd
            else_args: vec![0, 1, 2],
        },
    };
    // fwd(dst,src,len): i = 0; → floop.
    let fwd = Block {
        params: p3(),
        insts: vec![Inst::ConstI64(0)], // v3 = i
        term: Terminator::Br {
            target: 2,
            args: vec![0, 1, 2, 3],
        },
    };
    // floop(dst,src,len,i): i <u len ? fbody : done.
    let floop = Block {
        params: p4(),
        insts: vec![Inst::IntCmp {
            ty: IntTy::I64,
            op: CmpOp::LtU,
            a: 3,
            b: 2,
        }], // v4
        term: Terminator::BrIf {
            cond: 4,
            then_blk: 3,
            then_args: vec![0, 1, 2, 3],
            else_blk: 7,
            else_args: vec![],
        },
    };
    // fbody(dst,src,len,i): dst[i] = src[i]; i++ → floop.
    let fbody = Block {
        params: p4(),
        insts: vec![
            add(1, 3),         // v4 = src + i
            load8(4),          // v5 = src[i]
            add(0, 3),         // v6 = dst + i
            store8(6, 5),      // dst[i] = src[i]
            Inst::ConstI64(1), // v7
            add(3, 7),         // v8 = i + 1
        ],
        term: Terminator::Br {
            target: 2,
            args: vec![0, 1, 2, 8],
        },
    };
    // bwd(dst,src,len): i = len; → bloop.
    let bwd = Block {
        params: p3(),
        insts: vec![],
        term: Terminator::Br {
            target: 5,
            args: vec![0, 1, 2, 2], // i = len
        },
    };
    // bloop(dst,src,len,i): i >u 0 ? bbody : done.
    let bloop = Block {
        params: p4(),
        insts: vec![
            Inst::ConstI64(0), // v4
            Inst::IntCmp {
                ty: IntTy::I64,
                op: CmpOp::GtU,
                a: 3,
                b: 4,
            }, // v5 = i > 0
        ],
        term: Terminator::BrIf {
            cond: 5,
            then_blk: 6,
            then_args: vec![0, 1, 2, 3],
            else_blk: 7,
            else_args: vec![],
        },
    };
    // bbody(dst,src,len,i): i--; dst[i] = src[i]; → bloop.
    let bbody = Block {
        params: p4(),
        insts: vec![
            Inst::ConstI64(1), // v4
            Inst::IntBin {
                ty: IntTy::I64,
                op: BinOp::Sub,
                a: 3,
                b: 4,
            }, // v5 = i - 1
            add(1, 5),         // v6 = src + (i-1)
            load8(6),          // v7 = src[i-1]
            add(0, 5),         // v8 = dst + (i-1)
            store8(8, 7),      // dst[i-1] = src[i-1]
        ],
        term: Terminator::Br {
            target: 5,
            args: vec![0, 1, 2, 5], // loop with i-1
        },
    };
    let done = Block {
        params: vec![],
        insts: vec![],
        term: Terminator::Return(vec![]),
    };
    Func {
        params: p3(),
        results: vec![],
        blocks: vec![b0, fwd, floop, fbody, bwd, bloop, bbody, done],
    }
}

/// Synthesize `__svm_utoa(value:i64, base:i64, bufend:i64) -> i64` — write the **unsigned** `value`
/// in `base` (10 or 16, lowercase) as ASCII *backward* from `bufend` and return the start pointer
/// (so the digit count is `bufend - start`). A counted divide loop, like a tiny libc `utoa`; the
/// `printf` lowering handles sign, width, and padding around it. `value == 0` writes a single `'0'`.
///
/// ```text
///   block0(value, base, bufend):           → loop(value, base, bufend)
///   loop(value, base, p):                  ; d = value%base; *--p = digit(d); value/=base
///     digit = d + '0' + (d>=10 ? 39 : 0)   ; 0-9 → '0'-'9', 10-15 → 'a'-'f'
///     value != 0 ? loop(value/base, base, p-1) : done(p-1)
///   done(start):                           return start
/// ```
fn synth_utoa() -> Func {
    use svm_ir::StoreOp;
    let params = vec![ValType::I64, ValType::I64, ValType::I64]; // value, base, bufend
    let entry = Block {
        params: params.clone(),
        insts: vec![],
        term: Terminator::Br {
            target: 1,
            args: vec![0, 1, 2],
        },
    };
    // loop(value=0, base=1, p=2)
    let i64bin = |op: BinOp, a: ValIdx, b: ValIdx| Inst::IntBin {
        ty: IntTy::I64,
        op,
        a,
        b,
    };
    let lp = Block {
        params: vec![ValType::I64, ValType::I64, ValType::I64],
        insts: vec![
            i64bin(BinOp::RemU, 0, 1), // v3 = value % base
            Inst::Convert {
                op: ConvOp::WrapI64,
                a: 3,
            }, // v4 = d (i32)
            Inst::ConstI32(10),        // v5
            Inst::IntCmp {
                ty: IntTy::I32,
                op: CmpOp::GeU,
                a: 4,
                b: 5,
            }, // v6 = d>=10
            Inst::ConstI32(39),        // v7
            Inst::ConstI32(0),         // v8
            Inst::Select {
                cond: 6,
                a: 7,
                b: 8,
            }, // v9 = d>=10 ? 39 : 0
            Inst::ConstI32(48),        // v10 = '0'
            Inst::IntBin {
                ty: IntTy::I32,
                op: BinOp::Add,
                a: 4,
                b: 10,
            }, // v11 = d+'0'
            Inst::IntBin {
                ty: IntTy::I32,
                op: BinOp::Add,
                a: 11,
                b: 9,
            }, // v12 = digit char
            Inst::ConstI64(1),         // v13
            i64bin(BinOp::Sub, 2, 13), // v14 = p - 1
            Inst::Store {
                op: StoreOp::I32_8,
                addr: 14,
                value: 12,
                offset: 0,
                align: 0,
            }, // *--p = ch
            i64bin(BinOp::DivU, 0, 1), // v15 = value / base
            Inst::ConstI64(0),         // v16
            Inst::IntCmp {
                ty: IntTy::I64,
                op: CmpOp::Ne,
                a: 15,
                b: 16,
            }, // v17 = value' != 0
        ],
        term: Terminator::BrIf {
            cond: 17,
            then_blk: 1,
            then_args: vec![15, 1, 14], // loop(value/base, base, p-1)
            else_blk: 2,
            else_args: vec![14], // done(p-1)
        },
    };
    let done = Block {
        params: vec![ValType::I64], // start
        insts: vec![],
        term: Terminator::Return(vec![0]),
    };
    Func {
        params,
        results: vec![ValType::I64],
        blocks: vec![entry, lp, done],
    }
}

/// Synthesize `__svm_udivmod128(n_lo, n_hi, d_lo, d_hi) -> (q_lo, q_hi, r_lo, r_hi)` — **unsigned
/// 128÷128 division and remainder** in one pass (I14: `udiv`/`sdiv`/`urem`/`srem i128`; the on-ramp
/// sees these as IR ops at `-O2` — the `__udivti3`-family libcall is a *backend* lowering it never
/// gets). Classic binary long division over the `(lo, hi)` i64 pair: shift the 256-bit `[R:Q]`
/// register left one bit per step (quotient in the low 128, remainder in the high 128, dividend
/// seeded into `Q`); when the running remainder `R ≥ D`, subtract `D` and set the low quotient bit.
/// After 128 steps `Q` is the quotient and `R` the remainder. Division by zero **traps** (matching
/// the scalar `i64` divide), via an `i64 / 0` in the guard block.
///
/// ```text
///   entry(nlo,nhi,dlo,dhi):  (dlo|dhi)==0 ? trap() : loop(0, nlo, nhi, 0, 0, dlo, dhi)
///   trap():                  1 / 0   ; DivByZero, never returns
///   loop(i, qlo,qhi, rlo,rhi, dlo,dhi):
///     [R:Q] <<= 1 (256-bit)                                  ; bit i of the dividend enters R
///     ge = R >=u D                                           ; 128-bit unsigned compare
///     R = ge ? R - D : R ;  qlo |= ge                        ; conditional subtract + set bit
///     i+1 < 128 ? loop(i+1, …) : done(qlo,qhi, rlo,rhi)
///   done(qlo,qhi,rlo,rhi):   return (qlo,qhi,rlo,rhi)
/// ```
fn synth_udivmod128() -> Func {
    use BinOp::{Add, And, Or, Shl, ShrU, Sub};
    use CmpOp::{Eq, GeU, GtU, LtU};
    let i64t = ValType::I64;
    const LOOP: u32 = 2;
    const DONE: u32 = 3;

    // entry(nlo=0, nhi=1, dlo=2, dhi=3): trap on a zero divisor, else seed the loop.
    let entry = {
        let mut b = Bdr::new(4);
        let dor = b.bin(Or, 2, 3); // dlo | dhi
        let dz = b.cmpi(Eq, dor, 0); // divisor == 0?
        let zero = b.k(0);
        // ge/lt etc. read `dlo`/`dhi` (2,3) again in the loop; pass them through.
        b.block(
            vec![i64t; 4],
            Terminator::BrIf {
                cond: dz,
                then_blk: 1,
                then_args: vec![],
                else_blk: LOOP,
                // loop(i=0, qlo=nlo, qhi=nhi, rlo=0, rhi=0, dlo, dhi)
                else_args: vec![zero, 0, 1, zero, zero, 2, 3],
            },
        )
    };
    // trap(): an `i64 / 0` raises DivByZero (i128 divide-by-zero, like the scalar path). Unreachable
    // return keeps the block well-formed.
    let trap = {
        let mut b = Bdr::new(0);
        let one = b.k(1);
        let z = b.k(0);
        let t = b.bin(svm_ir::BinOp::DivU, one, z); // traps
        b.block(vec![], Terminator::Return(vec![t, t, t, t]))
    };
    // loop(i=0, qlo=1, qhi=2, rlo=3, rhi=4, dlo=5, dhi=6).
    let lp = {
        let mut b = Bdr::new(7);
        // 256-bit left shift of [rhi:rlo:qhi:qlo] by one — each word's incoming bit0 is the prior
        // (lower) word's outgoing top bit.
        let car_rhi = b.bini(ShrU, 3, 63); // top bit of rlo → rhi bit0
        let car_rlo = b.bini(ShrU, 2, 63); // top bit of qhi → rlo bit0
        let car_qhi = b.bini(ShrU, 1, 63); // top bit of qlo → qhi bit0
        let rhi_s = b.bini(Shl, 4, 1);
        let nr_hi = b.bin(Or, rhi_s, car_rhi);
        let rlo_s = b.bini(Shl, 3, 1);
        let nr_lo = b.bin(Or, rlo_s, car_rlo);
        let qhi_s = b.bini(Shl, 2, 1);
        let nq_hi = b.bin(Or, qhi_s, car_qhi);
        let nq_lo = b.bini(Shl, 1, 1);
        // ge = nr >=u d : (nr_hi >u d_hi) | (nr_hi == d_hi & nr_lo >=u d_lo)
        let gt_hi = b.cmp(GtU, nr_hi, 6);
        let eq_hi = b.cmp(Eq, nr_hi, 6);
        let ge_lo = b.cmp(GeU, nr_lo, 5);
        let and1 = b.push(Inst::IntBin {
            ty: IntTy::I32,
            op: And,
            a: eq_hi,
            b: ge_lo,
        });
        let ge = b.push(Inst::IntBin {
            ty: IntTy::I32,
            op: Or,
            a: gt_hi,
            b: and1,
        });
        // nr - d (128-bit), used only when ge.
        let diff_lo = b.bin(Sub, nr_lo, 5);
        let borrow = b.cmp(LtU, nr_lo, 5);
        let borrow64 = b.ext(borrow);
        let thi = b.bin(Sub, nr_hi, 6);
        let diff_hi = b.bin(Sub, thi, borrow64);
        let r_lo2 = b.sel(ge, diff_lo, nr_lo);
        let r_hi2 = b.sel(ge, diff_hi, nr_hi);
        let q_lo_or1 = b.bini(Or, nq_lo, 1);
        let q_lo2 = b.sel(ge, q_lo_or1, nq_lo);
        // i + 1 < 128 ? loop : done
        let i1 = b.bini(Add, 0, 1);
        let cont = b.cmpi(LtU, i1, 128);
        b.block(
            vec![i64t; 7],
            Terminator::BrIf {
                cond: cont,
                then_blk: LOOP,
                then_args: vec![i1, q_lo2, nq_hi, r_lo2, r_hi2, 5, 6],
                else_blk: DONE,
                else_args: vec![q_lo2, nq_hi, r_lo2, r_hi2],
            },
        )
    };
    // done(qlo, qhi, rlo, rhi): return the pair-of-pairs.
    let done = Block {
        params: vec![i64t; 4],
        insts: vec![],
        term: Terminator::Return(vec![0, 1, 2, 3]),
    };
    Func {
        params: vec![i64t; 4],
        results: vec![i64t; 4],
        blocks: vec![entry, trap, lp, done],
    }
}

/// Synthesize `__svm_strlen(p:i64) -> i64` — the NUL-terminated byte length, for `printf` `%s`. A
/// counted scan: walk forward from `p` until the zero byte, returning `cur - p`.
///
/// ```text
///   entry(p):            → loop(p, p)            ; cur=p, start=p
///   loop(cur, start):    b = *cur (i8)
///     b != 0 ? loop(cur+1, start) : done(cur-start)
///   done(len):           return len
/// ```
fn synth_strlen() -> Func {
    use svm_ir::LoadOp;
    let params = vec![ValType::I64]; // p
    let entry = Block {
        params: params.clone(),
        insts: vec![],
        term: Terminator::Br {
            target: 1,
            args: vec![0, 0], // loop(p, p)
        },
    };
    let lp = Block {
        params: vec![ValType::I64, ValType::I64], // cur, start
        insts: vec![
            Inst::Load {
                op: LoadOp::I32_8U,
                addr: 0,
                offset: 0,
                align: 0,
            }, // v2 = *cur
            Inst::ConstI32(0), // v3
            Inst::IntCmp {
                ty: IntTy::I32,
                op: CmpOp::Ne,
                a: 2,
                b: 3,
            }, // v4 = *cur != 0
            Inst::ConstI64(1), // v5
            Inst::IntBin {
                ty: IntTy::I64,
                op: BinOp::Add,
                a: 0,
                b: 5,
            }, // v6 = cur + 1
            Inst::IntBin {
                ty: IntTy::I64,
                op: BinOp::Sub,
                a: 0,
                b: 1,
            }, // v7 = cur - start
        ],
        term: Terminator::BrIf {
            cond: 4,
            then_blk: 1,
            then_args: vec![6, 1], // loop(cur+1, start)
            else_blk: 2,
            else_args: vec![7], // done(cur-start)
        },
    };
    let done = Block {
        params: vec![ValType::I64], // len
        insts: vec![],
        term: Terminator::Return(vec![0]),
    };
    Func {
        params,
        results: vec![ValType::I64],
        blocks: vec![entry, lp, done],
    }
}

/// Synthesize `__svm_strcmp(a:i64, b:i64) -> i32` — the lexicographic NUL-terminated byte compare.
/// Returns `0` when the strings are equal, else the signed difference of the first mismatching bytes
/// as **unsigned `char`s** (`(unsigned char)a[i] - (unsigned char)b[i]`, matching glibc). Backs
/// `strcmp` and (in the C locale) `strcoll`.
fn synth_strcmp() -> Func {
    use svm_ir::LoadOp;
    let params = vec![ValType::I64, ValType::I64]; // a, b
    let entry = Block {
        params: params.clone(),
        insts: vec![],
        term: Terminator::Br {
            target: 1,
            args: vec![0, 1], // loop(a, b)
        },
    };
    let lp = Block {
        params: vec![ValType::I64, ValType::I64], // pa, pb
        insts: vec![
            Inst::Load {
                op: LoadOp::I32_8U,
                addr: 0,
                offset: 0,
                align: 0,
            }, // v2 = ca = *pa
            Inst::Load {
                op: LoadOp::I32_8U,
                addr: 1,
                offset: 0,
                align: 0,
            }, // v3 = cb = *pb
            Inst::IntBin {
                ty: IntTy::I32,
                op: BinOp::Sub,
                a: 2,
                b: 3,
            }, // v4 = ca - cb
            Inst::IntCmp {
                ty: IntTy::I32,
                op: CmpOp::Eq,
                a: 2,
                b: 3,
            }, // v5 = ca == cb
            Inst::ConstI32(0), // v6
            Inst::IntCmp {
                ty: IntTy::I32,
                op: CmpOp::Ne,
                a: 2,
                b: 6,
            }, // v7 = ca != 0
            Inst::IntBin {
                ty: IntTy::I32,
                op: BinOp::And,
                a: 5,
                b: 7,
            }, // v8 = equal-and-not-end → keep going
            Inst::ConstI64(1), // v9
            Inst::IntBin {
                ty: IntTy::I64,
                op: BinOp::Add,
                a: 0,
                b: 9,
            }, // v10 = pa + 1
            Inst::IntBin {
                ty: IntTy::I64,
                op: BinOp::Add,
                a: 1,
                b: 9,
            }, // v11 = pb + 1
        ],
        term: Terminator::BrIf {
            cond: 8,
            then_blk: 1,
            then_args: vec![10, 11], // loop(pa+1, pb+1)
            else_blk: 2,
            else_args: vec![4], // done(ca - cb)
        },
    };
    let done = Block {
        params: vec![ValType::I32], // diff
        insts: vec![],
        term: Terminator::Return(vec![0]),
    };
    Func {
        params,
        results: vec![ValType::I32],
        blocks: vec![entry, lp, done],
    }
}

/// Synthesize `__svm_strchr(s:i64, c:i32) -> i64` — the first occurrence of `(unsigned char)c` in the
/// NUL-terminated string `s`, or NULL. When `c == 0` this returns a pointer to the terminating NUL (C
/// semantics): the byte test fires on the NUL itself.
fn synth_strchr() -> Func {
    use svm_ir::LoadOp;
    let params = vec![ValType::I64, ValType::I32]; // s, c
    let entry = Block {
        params: params.clone(),
        insts: vec![
            Inst::ConstI32(255), // v2
            Inst::IntBin {
                ty: IntTy::I32,
                op: BinOp::And,
                a: 1,
                b: 2,
            }, // v3 = cc = c & 0xff
        ],
        term: Terminator::Br {
            target: 1,
            args: vec![0, 3], // loop(s, cc)
        },
    };
    let lp = Block {
        params: vec![ValType::I64, ValType::I32], // p, cc
        insts: vec![
            Inst::Load {
                op: LoadOp::I32_8U,
                addr: 0,
                offset: 0,
                align: 0,
            }, // v2 = ch = *p
            Inst::IntCmp {
                ty: IntTy::I32,
                op: CmpOp::Eq,
                a: 2,
                b: 1,
            }, // v3 = ch == cc
            Inst::ConstI32(0), // v4
            Inst::IntCmp {
                ty: IntTy::I32,
                op: CmpOp::Eq,
                a: 2,
                b: 4,
            }, // v5 = ch == 0
            Inst::IntBin {
                ty: IntTy::I32,
                op: BinOp::Or,
                a: 3,
                b: 5,
            }, // v6 = done = hit || end
            Inst::ConstI64(1), // v7
            Inst::IntBin {
                ty: IntTy::I64,
                op: BinOp::Add,
                a: 0,
                b: 7,
            }, // v8 = p + 1
        ],
        term: Terminator::BrIf {
            cond: 6,
            then_blk: 2,
            then_args: vec![0, 1, 2], // finish(p, cc, ch)
            else_blk: 1,
            else_args: vec![8, 1], // loop(p+1, cc)
        },
    };
    let finish = Block {
        params: vec![ValType::I64, ValType::I32, ValType::I32], // p, cc, ch
        insts: vec![
            Inst::IntCmp {
                ty: IntTy::I32,
                op: CmpOp::Eq,
                a: 2,
                b: 1,
            }, // v3 = hit = ch == cc
            Inst::ConstI64(0), // v4 = NULL
            Inst::Select {
                cond: 3,
                a: 0,
                b: 4,
            }, // v5 = hit ? p : NULL
        ],
        term: Terminator::Return(vec![5]),
    };
    Func {
        params,
        results: vec![ValType::I64],
        blocks: vec![entry, lp, finish],
    }
}

/// Synthesize `__svm_strcpy(dst:i64, src:i64) -> i64` — copy the NUL-terminated string `src` into
/// `dst` (including the terminator) and return the original `dst`. No overlap handling (C `strcpy`).
fn synth_strcpy() -> Func {
    use svm_ir::{LoadOp, StoreOp};
    let params = vec![ValType::I64, ValType::I64]; // dst, src
    let entry = Block {
        params: params.clone(),
        insts: vec![],
        term: Terminator::Br {
            target: 1,
            args: vec![0, 1, 0], // loop(dst, src, orig=dst)
        },
    };
    let lp = Block {
        params: vec![ValType::I64, ValType::I64, ValType::I64], // d, s, orig
        insts: vec![
            Inst::Load {
                op: LoadOp::I32_8U,
                addr: 1,
                offset: 0,
                align: 0,
            }, // v3 = c = *s
            Inst::Store {
                op: StoreOp::I32_8,
                addr: 0,
                value: 3,
                offset: 0,
                align: 0,
            }, // *d = c   (void — no value index)
            Inst::ConstI32(0), // v4
            Inst::IntCmp {
                ty: IntTy::I32,
                op: CmpOp::Eq,
                a: 3,
                b: 4,
            }, // v5 = c == 0
            Inst::ConstI64(1), // v6
            Inst::IntBin {
                ty: IntTy::I64,
                op: BinOp::Add,
                a: 0,
                b: 6,
            }, // v7 = d + 1
            Inst::IntBin {
                ty: IntTy::I64,
                op: BinOp::Add,
                a: 1,
                b: 6,
            }, // v8 = s + 1
        ],
        term: Terminator::BrIf {
            cond: 5,
            then_blk: 2,
            then_args: vec![2], // done(orig)
            else_blk: 1,
            else_args: vec![7, 8, 2], // loop(d+1, s+1, orig)
        },
    };
    let done = Block {
        params: vec![ValType::I64], // orig
        insts: vec![],
        term: Terminator::Return(vec![0]),
    };
    Func {
        params,
        results: vec![ValType::I64],
        blocks: vec![entry, lp, done],
    }
}

/// Synthesize `__svm_strspn(s:i64, set:i64) -> i64` — the length of the initial segment of `s`
/// consisting entirely of bytes in the NUL-terminated `set`. An inner loop scans `set` for each
/// byte of `s`; the first `s` byte not found in `set` ends the span.
fn synth_strspn() -> Func {
    use svm_ir::LoadOp;
    let l8 = |addr: u32| Inst::Load {
        op: LoadOp::I32_8U,
        addr,
        offset: 0,
        align: 0,
    };
    let params = vec![ValType::I64, ValType::I64]; // s, set
                                                   // block0 entry(s=0, set=1): outer(p=s, set, s0=s)
    let entry = Block {
        params: params.clone(),
        insts: vec![],
        term: Terminator::Br {
            target: 1,
            args: vec![0, 1, 0],
        },
    };
    // block1 outer(p=0, set=1, s0=2)
    let outer = Block {
        params: vec![ValType::I64, ValType::I64, ValType::I64],
        insts: vec![
            l8(0),             // v3 = cp = *p
            Inst::ConstI32(0), // v4
            Inst::IntCmp {
                ty: IntTy::I32,
                op: CmpOp::Eq,
                a: 3,
                b: 4,
            }, // v5 = cp == 0
        ],
        term: Terminator::BrIf {
            cond: 5,
            then_blk: 4,
            then_args: vec![0, 2], // done(p, s0)
            else_blk: 2,
            else_args: vec![1, 3, 0, 1, 2], // inner(q=set, cp, p, set, s0)
        },
    };
    // block2 inner(q=0, cp=1, p=2, set=3, s0=4)
    let inner = Block {
        params: vec![
            ValType::I64,
            ValType::I32,
            ValType::I64,
            ValType::I64,
            ValType::I64,
        ],
        insts: vec![
            l8(0),             // v5 = cq = *q
            Inst::ConstI32(0), // v6
            Inst::IntCmp {
                ty: IntTy::I32,
                op: CmpOp::Eq,
                a: 5,
                b: 6,
            }, // v7 = cq == 0
            Inst::IntCmp {
                ty: IntTy::I32,
                op: CmpOp::Eq,
                a: 5,
                b: 1,
            }, // v8 = cq == cp
            Inst::IntBin {
                ty: IntTy::I32,
                op: BinOp::Or,
                a: 7,
                b: 8,
            }, // v9 = stop
            Inst::ConstI64(1), // v10
            Inst::IntBin {
                ty: IntTy::I64,
                op: BinOp::Add,
                a: 0,
                b: 10,
            }, // v11 = q + 1
        ],
        term: Terminator::BrIf {
            cond: 9,
            then_blk: 3,
            then_args: vec![5, 1, 2, 3, 4], // inner_done(cq, cp, p, set, s0)
            else_blk: 2,
            else_args: vec![11, 1, 2, 3, 4], // inner(q+1, cp, p, set, s0)
        },
    };
    // block3 inner_done(cq=0, cp=1, p=2, set=3, s0=4)
    let inner_done = Block {
        params: vec![
            ValType::I32,
            ValType::I32,
            ValType::I64,
            ValType::I64,
            ValType::I64,
        ],
        insts: vec![
            Inst::IntCmp {
                ty: IntTy::I32,
                op: CmpOp::Eq,
                a: 0,
                b: 1,
            }, // v5 = match = cq == cp
            Inst::ConstI64(1), // v6
            Inst::IntBin {
                ty: IntTy::I64,
                op: BinOp::Add,
                a: 2,
                b: 6,
            }, // v7 = p + 1
        ],
        term: Terminator::BrIf {
            cond: 5,
            then_blk: 1,
            then_args: vec![7, 3, 4], // outer(p+1, set, s0)  — cp was in set
            else_blk: 4,
            else_args: vec![2, 4], // done(p, s0)  — cp not in set
        },
    };
    // block4 done(p=0, s0=1)
    let done = Block {
        params: vec![ValType::I64, ValType::I64],
        insts: vec![Inst::IntBin {
            ty: IntTy::I64,
            op: BinOp::Sub,
            a: 0,
            b: 1,
        }], // v2 = p - s0
        term: Terminator::Return(vec![2]),
    };
    Func {
        params,
        results: vec![ValType::I64],
        blocks: vec![entry, outer, inner, inner_done, done],
    }
}

/// Synthesize `__svm_strpbrk(s:i64, set:i64) -> i64` — a pointer to the first byte of `s` that is in
/// the NUL-terminated `set`, or NULL if none. An inner loop scans `set` for each byte of `s`.
fn synth_strpbrk() -> Func {
    use svm_ir::LoadOp;
    let l8 = |addr: u32| Inst::Load {
        op: LoadOp::I32_8U,
        addr,
        offset: 0,
        align: 0,
    };
    let params = vec![ValType::I64, ValType::I64]; // s, set
                                                   // block0 entry(s=0, set=1): outer(p=s, set)
    let entry = Block {
        params: params.clone(),
        insts: vec![],
        term: Terminator::Br {
            target: 1,
            args: vec![0, 1],
        },
    };
    // block1 outer(p=0, set=1)
    let outer = Block {
        params: vec![ValType::I64, ValType::I64],
        insts: vec![
            l8(0),             // v2 = cp = *p
            Inst::ConstI32(0), // v3
            Inst::IntCmp {
                ty: IntTy::I32,
                op: CmpOp::Eq,
                a: 2,
                b: 3,
            }, // v4 = cp == 0
        ],
        term: Terminator::BrIf {
            cond: 4,
            then_blk: 5,
            then_args: vec![], // retnull
            else_blk: 2,
            else_args: vec![1, 2, 0, 1], // inner(q=set, cp, p, set)
        },
    };
    // block2 inner(q=0, cp=1, p=2, set=3)
    let inner = Block {
        params: vec![ValType::I64, ValType::I32, ValType::I64, ValType::I64],
        insts: vec![
            l8(0),             // v4 = cq = *q
            Inst::ConstI32(0), // v5
            Inst::IntCmp {
                ty: IntTy::I32,
                op: CmpOp::Eq,
                a: 4,
                b: 5,
            }, // v6 = qz = cq == 0
            Inst::IntCmp {
                ty: IntTy::I32,
                op: CmpOp::Eq,
                a: 4,
                b: 1,
            }, // v7 = match = cq == cp
            Inst::IntBin {
                ty: IntTy::I32,
                op: BinOp::Or,
                a: 6,
                b: 7,
            }, // v8 = stop
            Inst::ConstI64(1), // v9
            Inst::IntBin {
                ty: IntTy::I64,
                op: BinOp::Add,
                a: 0,
                b: 9,
            }, // v10 = q + 1
        ],
        term: Terminator::BrIf {
            cond: 8,
            then_blk: 3,
            then_args: vec![7, 2, 3], // inner_done(match, p, set)
            else_blk: 2,
            else_args: vec![10, 1, 2, 3], // inner(q+1, cp, p, set)
        },
    };
    // block3 inner_done(match=0, p=1, set=2)
    let inner_done = Block {
        params: vec![ValType::I32, ValType::I64, ValType::I64],
        insts: vec![
            Inst::ConstI64(1), // v3
            Inst::IntBin {
                ty: IntTy::I64,
                op: BinOp::Add,
                a: 1,
                b: 3,
            }, // v4 = p + 1
        ],
        term: Terminator::BrIf {
            cond: 0,
            then_blk: 4,
            then_args: vec![1], // found(p)
            else_blk: 1,
            else_args: vec![4, 2], // outer(p+1, set)
        },
    };
    // block4 found(p=0)
    let found = Block {
        params: vec![ValType::I64],
        insts: vec![],
        term: Terminator::Return(vec![0]),
    };
    // block5 retnull()
    let retnull = Block {
        params: vec![],
        insts: vec![Inst::ConstI64(0)], // v0
        term: Terminator::Return(vec![0]),
    };
    Func {
        params,
        results: vec![ValType::I64],
        blocks: vec![entry, outer, inner, inner_done, found, retnull],
    }
}

/// Synthesize a **fail-closed trap stub** `(params) -> results` whose body is a single `Unreachable`
/// block — for a libm/libc function the on-ramp does not yet implement exactly. The stub *translates*
/// (so a module that merely references the function on an **unexecuted** path still lowers — e.g. Lua's
/// `^`/`pow` when a script does no float exponentiation) but **traps if ever called**. Bit-exact
/// transcendentals (`pow`/`fmod`/…) require matching a specific host libm — an architecture decision
/// (host-libm delegation) tracked separately; until then this keeps the module honest and translating.
fn synth_trap_stub(params: Vec<ValType>, results: Vec<ValType>) -> Func {
    Func {
        params: params.clone(),
        results,
        blocks: vec![Block {
            params,
            insts: vec![],
            term: Terminator::Unreachable,
        }],
    }
}

/// Synthesize `(params) -> i64` that ignores its arguments and returns the constant `val` — for a
/// libc function whose *value* does not affect a deterministic result (e.g. `time()` as the `makeseed`
/// RNG source: the seed only perturbs hash iteration order, not computed results).
fn synth_const_i64(params: Vec<ValType>, val: i64) -> Func {
    let nparams = params.len() as u32;
    Func {
        params: params.clone(),
        results: vec![ValType::I64],
        blocks: vec![Block {
            params,
            insts: vec![Inst::ConstI64(val)], // value index `nparams`
            term: Terminator::Return(vec![nparams]),
        }],
    }
}

/// Synthesize `__svm_ldexp(x:f64, n:i32) -> f64` — `x * 2^n`, the musl `scalbn` algorithm: scale by
/// `2^±1023`/`2^∓969` at most twice to bring an extreme `n` into `[-1022, 1023]`, then multiply by a
/// `2^n` built directly from the exponent field. Bit-exact to libc (the IEEE multiplies carry the
/// rounding, incl. overflow→±inf and gradual underflow→denormal/0). `ldexp` ≡ `scalbn` for `double`.
fn synth_ldexp() -> Func {
    // 2^1023 and 2^-969 (= 2^-1022 · 2^53) as raw double bit patterns (exponent field only).
    const TWO_P1023: i64 = 0x7FE0000000000000u64 as i64;
    const TWO_M969: i64 = 0x0360000000000000;
    let reinterp = |a: u32| Inst::Cast {
        op: CastOp::ReinterpI64F64,
        a,
    };
    let fmul = |a: u32, b: u32| Inst::FBin {
        ty: FloatTy::F64,
        op: FBinOp::Mul,
        a,
        b,
    };
    let params = vec![ValType::F64, ValType::I32]; // x, n
                                                   // block0 entry(x=0, n=1): n>1023 → hi1; else chk_lo
    let entry = Block {
        params: params.clone(),
        insts: vec![
            Inst::ConstI32(1023), // v2
            Inst::IntCmp {
                ty: IntTy::I32,
                op: CmpOp::GtS,
                a: 1,
                b: 2,
            }, // v3 = n > 1023
        ],
        term: Terminator::BrIf {
            cond: 3,
            then_blk: 2,
            then_args: vec![0, 1],
            else_blk: 1,
            else_args: vec![0, 1],
        },
    };
    // block1 chk_lo(x=0, n=1): n < -1022 → lo1; else finish
    let chk_lo = Block {
        params: vec![ValType::F64, ValType::I32],
        insts: vec![
            Inst::ConstI32(-1022), // v2
            Inst::IntCmp {
                ty: IntTy::I32,
                op: CmpOp::LtS,
                a: 1,
                b: 2,
            }, // v3 = n < -1022
        ],
        term: Terminator::BrIf {
            cond: 3,
            then_blk: 4,
            then_args: vec![0, 1],
            else_blk: 6,
            else_args: vec![0, 1],
        },
    };
    // block2 hi1(x=0, n=1): y = x·2^1023; m = n-1023; m>1023 → hi2 else finish
    let hi1 = Block {
        params: vec![ValType::F64, ValType::I32],
        insts: vec![
            Inst::ConstI64(TWO_P1023), // v2
            reinterp(2),               // v3 = 2^1023
            fmul(0, 3),                // v4 = y
            Inst::ConstI32(1023),      // v5
            Inst::IntBin {
                ty: IntTy::I32,
                op: BinOp::Sub,
                a: 1,
                b: 5,
            }, // v6 = m
            Inst::ConstI32(1023),      // v7
            Inst::IntCmp {
                ty: IntTy::I32,
                op: CmpOp::GtS,
                a: 6,
                b: 7,
            }, // v8 = m > 1023
        ],
        term: Terminator::BrIf {
            cond: 8,
            then_blk: 3,
            then_args: vec![4, 6],
            else_blk: 6,
            else_args: vec![4, 6],
        },
    };
    // block3 hi2(y=0, m=1): y·2^1023; clamp m-1023 to 1023; → finish
    let hi2 = Block {
        params: vec![ValType::F64, ValType::I32],
        insts: vec![
            Inst::ConstI64(TWO_P1023), // v2
            reinterp(2),               // v3
            fmul(0, 3),                // v4 = y2
            Inst::ConstI32(1023),      // v5
            Inst::IntBin {
                ty: IntTy::I32,
                op: BinOp::Sub,
                a: 1,
                b: 5,
            }, // v6 = m-1023
            Inst::ConstI32(1023),      // v7
            Inst::IntCmp {
                ty: IntTy::I32,
                op: CmpOp::GtS,
                a: 6,
                b: 7,
            }, // v8
            Inst::Select {
                cond: 8,
                a: 7,
                b: 6,
            }, // v9 = min(m-1023, 1023)
        ],
        term: Terminator::Br {
            target: 6,
            args: vec![4, 9],
        },
    };
    // block4 lo1(x=0, n=1): y = x·2^-969; m = n+969; m<-1022 → lo2 else finish
    let lo1 = Block {
        params: vec![ValType::F64, ValType::I32],
        insts: vec![
            Inst::ConstI64(TWO_M969), // v2
            reinterp(2),              // v3 = 2^-969
            fmul(0, 3),               // v4 = y
            Inst::ConstI32(969),      // v5
            Inst::IntBin {
                ty: IntTy::I32,
                op: BinOp::Add,
                a: 1,
                b: 5,
            }, // v6 = m
            Inst::ConstI32(-1022),    // v7
            Inst::IntCmp {
                ty: IntTy::I32,
                op: CmpOp::LtS,
                a: 6,
                b: 7,
            }, // v8 = m < -1022
        ],
        term: Terminator::BrIf {
            cond: 8,
            then_blk: 5,
            then_args: vec![4, 6],
            else_blk: 6,
            else_args: vec![4, 6],
        },
    };
    // block5 lo2(y=0, m=1): y·2^-969; clamp m+969 to -1022; → finish
    let lo2 = Block {
        params: vec![ValType::F64, ValType::I32],
        insts: vec![
            Inst::ConstI64(TWO_M969), // v2
            reinterp(2),              // v3
            fmul(0, 3),               // v4 = y2
            Inst::ConstI32(969),      // v5
            Inst::IntBin {
                ty: IntTy::I32,
                op: BinOp::Add,
                a: 1,
                b: 5,
            }, // v6 = m+969
            Inst::ConstI32(-1022),    // v7
            Inst::IntCmp {
                ty: IntTy::I32,
                op: CmpOp::LtS,
                a: 6,
                b: 7,
            }, // v8
            Inst::Select {
                cond: 8,
                a: 7,
                b: 6,
            }, // v9 = max(m+969, -1022)
        ],
        term: Terminator::Br {
            target: 6,
            args: vec![4, 9],
        },
    };
    // block6 finish(y=0, m=1): return y · 2^m, 2^m built from the (0x3ff+m) exponent field
    let finish = Block {
        params: vec![ValType::F64, ValType::I32],
        insts: vec![
            Inst::ConstI32(1023), // v2
            Inst::IntBin {
                ty: IntTy::I32,
                op: BinOp::Add,
                a: 1,
                b: 2,
            }, // v3 = 0x3ff + m
            Inst::Convert {
                op: ConvOp::ExtendI32U,
                a: 3,
            }, // v4 = (i64)
            Inst::ConstI64(52),   // v5
            Inst::IntBin {
                ty: IntTy::I64,
                op: BinOp::Shl,
                a: 4,
                b: 5,
            }, // v6 = bits
            reinterp(6),          // v7 = 2^m
            fmul(0, 7),           // v8 = y · 2^m
        ],
        term: Terminator::Return(vec![8]),
    };
    Func {
        params,
        results: vec![ValType::F64],
        blocks: vec![entry, chk_lo, hi1, hi2, lo1, lo2, finish],
    }
}

/// Synthesize `__svm_frexp(x:f64, eptr:i64) -> f64` — split `x = m·2^e` with `m ∈ [0.5, 1)`, writing
/// `e` (an `int`) to `*eptr`. Bit-exact to glibc `__frexp`:
/// ```c
/// int ex = 0x7ff & (ix >> 52); int e = 0;
/// if (ex != 0x7ff && x != 0.0) {        // finite, nonzero
///     e = ex - 1022;
///     if (ex == 0) { x *= 0x1p54; ix = bits(x); ex = 0x7ff & (ix>>52); e = ex - 1022 - 54; }
///     ix = (ix & 0x800fffffffffffff) | 0x3fe0000000000000; x = bits_to_f64(ix);
/// } else x += x;                        // zero/inf/nan: e stays 0, x+x signals on sNaN
/// *eptr = e; return x;
/// ```
/// Five blocks: classify → (finite | special); finite → (subnormal-normalize | mantissa-pack).
fn synth_frexp() -> Func {
    use svm_ir::StoreOp;
    let reinterp_to_f = |a: u32| Inst::Cast {
        op: CastOp::ReinterpI64F64,
        a,
    };
    let reinterp_to_i = |a: u32| Inst::Cast {
        op: CastOp::ReinterpF64I64,
        a,
    };
    let shru = |a: u32, b: u32| Inst::IntBin {
        ty: IntTy::I64,
        op: BinOp::ShrU,
        a,
        b,
    };
    let and = |a: u32, b: u32| Inst::IntBin {
        ty: IntTy::I64,
        op: BinOp::And,
        a,
        b,
    };
    let sub = |a: u32, b: u32| Inst::IntBin {
        ty: IntTy::I64,
        op: BinOp::Sub,
        a,
        b,
    };

    // block0 entry(x=v0:f64, eptr=v1:i64): classify finite-and-nonzero vs special (zero/inf/nan).
    let entry = Block {
        params: vec![ValType::F64, ValType::I64],
        insts: vec![
            reinterp_to_i(0),      // v2 = ix (raw bits of x)
            Inst::ConstI64(52),    // v3
            shru(2, 3),            // v4 = ix >> 52
            Inst::ConstI64(0x7ff), // v5
            and(4, 5),             // v6 = ex (biased exponent)
            Inst::ConstI64(0x7ff), // v7
            Inst::IntCmp {
                ty: IntTy::I64,
                op: CmpOp::Ne,
                a: 6,
                b: 7,
            }, // v8 = (ex != 0x7ff)  → i32 0/1
            Inst::ConstI64(0x7fff_ffff_ffff_ffff), // v9 = abs mask
            and(2, 9),             // v10 = |ix| (sign cleared)
            Inst::ConstI64(0),     // v11
            Inst::IntCmp {
                ty: IntTy::I64,
                op: CmpOp::Ne,
                a: 10,
                b: 11,
            }, // v12 = (x != 0)  → i32 0/1
            Inst::IntBin {
                ty: IntTy::I32,
                op: BinOp::And,
                a: 8,
                b: 12,
            }, // v13 = finite && nonzero
        ],
        term: Terminator::BrIf {
            cond: 13,
            then_blk: 1, // finite(x, ix, ex, eptr)
            then_args: vec![0, 2, 6, 1],
            else_blk: 2, // special(x, eptr)
            else_args: vec![0, 1],
        },
    };

    // block1 finite(x=v0:f64, ix=v1:i64, ex=v2:i64, eptr=v3:i64): subnormal → normalize; else pack.
    let finite = Block {
        params: vec![ValType::F64, ValType::I64, ValType::I64, ValType::I64],
        insts: vec![
            Inst::ConstI64(0), // v4
            Inst::IntCmp {
                ty: IntTy::I64,
                op: CmpOp::Eq,
                a: 2,
                b: 4,
            }, // v5 = (ex == 0) subnormal?  → i32
            Inst::ConstI64(1022), // v6
            sub(2, 6),         // v7 = e = ex - 1022
        ],
        term: Terminator::BrIf {
            cond: 5,
            then_blk: 3, // sub(x, eptr)
            then_args: vec![0, 3],
            else_blk: 4, // pack(ix, e, eptr)
            else_args: vec![1, 7, 3],
        },
    };

    // block2 special(x=v0:f64, eptr=v1:i64): *eptr = 0; return x + x (quiets/signals NaN, ±0/±inf pass).
    let special = Block {
        params: vec![ValType::F64, ValType::I64],
        insts: vec![
            Inst::ConstI32(0), // v2
            Inst::Store {
                op: StoreOp::I32,
                addr: 1,
                value: 2,
                offset: 0,
                align: 0,
            }, // *eptr = 0 (no value)
            Inst::FBin {
                ty: FloatTy::F64,
                op: FBinOp::Add,
                a: 0,
                b: 0,
            }, // v3 = x + x
        ],
        term: Terminator::Return(vec![3]),
    };

    // block3 sub(x=v0:f64, eptr=v1:i64): scale by 2^54 into the normal range, recompute ex and e.
    let sub_blk = Block {
        params: vec![ValType::F64, ValType::I64],
        insts: vec![
            Inst::ConstI64(0x4350_0000_0000_0000), // v2 = bits(2^54)
            reinterp_to_f(2),                      // v3 = 2^54
            Inst::FBin {
                ty: FloatTy::F64,
                op: FBinOp::Mul,
                a: 0,
                b: 3,
            }, // v4 = x · 2^54
            reinterp_to_i(4),                      // v5 = ix2
            Inst::ConstI64(52),                    // v6
            shru(5, 6),                            // v7 = ix2 >> 52
            Inst::ConstI64(0x7ff),                 // v8
            and(7, 8),                             // v9 = ex2
            Inst::ConstI64(1076),                  // v10 = 1022 + 54
            sub(9, 10),                            // v11 = e2 = ex2 - 1076
        ],
        term: Terminator::Br {
            target: 4, // pack(ix2, e2, eptr)
            args: vec![5, 11, 1],
        },
    };

    // block4 pack(ix=v0:i64, e=v1:i64, eptr=v2:i64): set the biased exponent to 0x3fe (m ∈ [0.5,1)),
    // store e (as int), return the reconstructed mantissa.
    let pack = Block {
        params: vec![ValType::I64, ValType::I64, ValType::I64],
        insts: vec![
            Inst::ConstI64(0x800f_ffff_ffff_ffffu64 as i64), // v3 = sign|mantissa mask
            and(0, 3),                                       // v4 = ix & mask
            Inst::ConstI64(0x3fe0_0000_0000_0000),           // v5 = exponent field for 0.5
            Inst::IntBin {
                ty: IntTy::I64,
                op: BinOp::Or,
                a: 4,
                b: 5,
            }, // v6 = packed bits
            reinterp_to_f(6),                                // v7 = mantissa f64
            Inst::Convert {
                op: ConvOp::WrapI64,
                a: 1,
            }, // v8 = (int)e
            Inst::Store {
                op: StoreOp::I32,
                addr: 2,
                value: 8,
                offset: 0,
                align: 0,
            }, // *eptr = e (no value)
        ],
        term: Terminator::Return(vec![7]),
    };

    Func {
        params: vec![ValType::F64, ValType::I64],
        results: vec![ValType::F64],
        blocks: vec![entry, finite, special, sub_blk, pack],
    }
}

/// Synthesize `__svm_getenv(name:i64) -> i64` (the C `getenv`). Scans the §3e args blob
/// (`POWERBOX_ARGS_BASE`: `{argc:u32, envc:u32}` then the packed argv + env strings) for the first env
/// entry whose key equals the NUL-terminated `name` followed by `=`, returning a pointer to the value
/// (just past the `=`) or NULL. Reads the blob directly from the reserved low scratch — no `environ`
/// global, no coupling to `_start` — so it works at any `main` arity (and returns NULL when the host
/// seeded no env, since the window then reads `argc==envc==0`). Three phases: skip the `argc` argv
/// strings to reach the env section, then for each of the `envc` env strings compare it against `name`
/// char by char; a full `name` match landing on `=` yields the value pointer.
fn synth_getenv() -> Func {
    use svm_ir::LoadOp;
    let args_base = svm_ir::POWERBOX_ARGS_BASE as i64;
    let add = |a: ValIdx, b: ValIdx| Inst::IntBin {
        ty: IntTy::I64,
        op: BinOp::Add,
        a,
        b,
    };
    let load8 = |addr: ValIdx| Inst::Load {
        op: LoadOp::I32_8U,
        addr,
        offset: 0,
        align: 0,
    };
    let load32 = |addr: ValIdx| Inst::Load {
        op: LoadOp::I32,
        addr,
        offset: 0,
        align: 0,
    };
    let eq32 = |a: ValIdx, b: ValIdx| Inst::IntCmp {
        ty: IntTy::I32,
        op: CmpOp::Eq,
        a,
        b,
    };
    let ltu64 = |a: ValIdx, b: ValIdx| Inst::IntCmp {
        ty: IntTy::I64,
        op: CmpOp::LtU,
        a,
        b,
    };

    // block 0: entry(name=0) — load argc, jump into the argv-skip loop.
    let b0 = Block {
        params: vec![ValType::I64], // name
        insts: vec![
            Inst::ConstI64(args_base), // v1
            load32(1),                 // v2 = argc (u32)
            Inst::Convert {
                op: ConvOp::ExtendI32U,
                a: 2,
            }, // v3 = argc (i64)
            Inst::ConstI64(args_base + 8), // v4 = p0 (first string)
            Inst::ConstI64(0),         // v5 = i = 0
        ],
        term: Terminator::Br {
            target: 1,
            args: vec![0, 3, 5, 4], // skip_head(name, argc, i, p)
        },
    };

    // block 1: skip_head(name=0, argc=1, i=2, p=3) — while i <u argc skip a string, else env_setup.
    let b1 = Block {
        params: vec![ValType::I64, ValType::I64, ValType::I64, ValType::I64],
        insts: vec![ltu64(2, 1)], // v4 = i < argc
        term: Terminator::BrIf {
            cond: 4,
            then_blk: 2,
            then_args: vec![0, 1, 2, 3],
            else_blk: 4,
            else_args: vec![0, 3], // env_setup(name, env_start = p)
        },
    };

    // block 2: skip_scan(name=0, argc=1, i=2, p=3) — advance p past this string's NUL.
    let b2 = Block {
        params: vec![ValType::I64, ValType::I64, ValType::I64, ValType::I64],
        insts: vec![
            load8(3),          // v4 = *p
            Inst::ConstI64(1), // v5
            add(3, 5),         // v6 = p+1
            Inst::ConstI32(0), // v7
            eq32(4, 7),        // v8 = (*p == 0)
        ],
        term: Terminator::BrIf {
            cond: 8,
            then_blk: 3,
            then_args: vec![0, 1, 2, 6], // skip_next(name, argc, i, p_next)
            else_blk: 2,
            else_args: vec![0, 1, 2, 6], // skip_scan(name, argc, i, p+1)
        },
    };

    // block 3: skip_next(name=0, argc=1, i=2, p=3) — i++ and loop.
    let b3 = Block {
        params: vec![ValType::I64, ValType::I64, ValType::I64, ValType::I64],
        insts: vec![Inst::ConstI64(1), add(2, 4)], // v4=1, v5 = i+1
        term: Terminator::Br {
            target: 1,
            args: vec![0, 1, 5, 3],
        },
    };

    // block 4: env_setup(name=0, p=1) — load envc, enter the env-match loop.
    let b4 = Block {
        params: vec![ValType::I64, ValType::I64],
        insts: vec![
            Inst::ConstI64(args_base + 4), // v2
            load32(2),                     // v3 = envc (u32)
            Inst::Convert {
                op: ConvOp::ExtendI32U,
                a: 3,
            }, // v4 = envc (i64)
            Inst::ConstI64(0),             // v5 = j = 0
        ],
        term: Terminator::Br {
            target: 5,
            args: vec![0, 4, 5, 1], // env_head(name, envc, j, p)
        },
    };

    // block 5: env_head(name=0, envc=1, j=2, p=3) — while j <u envc try a match, else return NULL.
    let b5 = Block {
        params: vec![ValType::I64, ValType::I64, ValType::I64, ValType::I64],
        insts: vec![ltu64(2, 1)], // v4 = j < envc
        term: Terminator::BrIf {
            cond: 4,
            then_blk: 6,
            then_args: vec![0, 1, 2, 3, 0, 3], // cmp(name, envc, j, p, a=name, b=p)
            else_blk: 13,
            else_args: vec![],
        },
    };

    // block 6: cmp(name=0, envc=1, j=2, p=3, a=4, b=5) — *a==0 ⇒ key ended, check '='; else compare.
    let b6 = Block {
        params: vec![ValType::I64; 6],
        insts: vec![
            load8(4),          // v6 = *a (name char)
            Inst::ConstI32(0), // v7
            eq32(6, 7),        // v8 = (*a == 0)
        ],
        term: Terminator::BrIf {
            cond: 8,
            then_blk: 9,
            then_args: vec![0, 1, 2, 3, 5], // check_sep(name, envc, j, p, b)
            else_blk: 7,
            else_args: vec![0, 1, 2, 3, 4, 5, 6], // cmp2(.., a, b, ca=*a)
        },
    };

    // block 7: cmp2(name=0, envc=1, j=2, p=3, a=4, b=5, ca=6) — *a==*b ? advance : skip this entry.
    // `ca` (the just-read name byte) is an i32, the rest are pointers/counts.
    let b7 = Block {
        params: vec![
            ValType::I64,
            ValType::I64,
            ValType::I64,
            ValType::I64,
            ValType::I64,
            ValType::I64,
            ValType::I32,
        ],
        insts: vec![
            load8(5),   // v7 = *b
            eq32(6, 7), // v8 = (ca == *b)
        ],
        term: Terminator::BrIf {
            cond: 8,
            then_blk: 8,
            then_args: vec![0, 1, 2, 3, 4, 5], // cmp_adv(name, envc, j, p, a, b)
            else_blk: 11,
            else_args: vec![0, 1, 2, 3, 3], // env_scan(name, envc, j, p, r = p)
        },
    };

    // block 8: cmp_adv(name=0, envc=1, j=2, p=3, a=4, b=5) — a++, b++ and re-compare.
    let b8 = Block {
        params: vec![ValType::I64; 6],
        insts: vec![
            Inst::ConstI64(1), // v6
            add(4, 6),         // v7 = a+1
            add(5, 6),         // v8 = b+1
        ],
        term: Terminator::Br {
            target: 6,
            args: vec![0, 1, 2, 3, 7, 8],
        },
    };

    // block 9: check_sep(name=0, envc=1, j=2, p=3, b=4) — key matched; is *b the '=' separator?
    let b9 = Block {
        params: vec![ValType::I64; 5],
        insts: vec![
            load8(4),                    // v5 = *b
            Inst::ConstI32(b'=' as i32), // v6
            eq32(5, 6),                  // v7 = (*b == '=')
        ],
        term: Terminator::BrIf {
            cond: 7,
            then_blk: 10,
            then_args: vec![4], // found(b)
            else_blk: 11,
            else_args: vec![0, 1, 2, 3, 3], // env_scan(..) — `name` was only a key prefix
        },
    };

    // block 10: found(b=0) — the value is the byte just past the '='.
    let b10 = Block {
        params: vec![ValType::I64],
        insts: vec![Inst::ConstI64(1), add(0, 1)], // v1=1, v2 = b+1
        term: Terminator::Return(vec![2]),
    };

    // block 11: env_scan(name=0, envc=1, j=2, p=3, r=4) — advance r past this entry's NUL.
    let b11 = Block {
        params: vec![ValType::I64; 5],
        insts: vec![
            load8(4),          // v5 = *r
            Inst::ConstI64(1), // v6
            add(4, 6),         // v7 = r+1
            Inst::ConstI32(0), // v8
            eq32(5, 8),        // v9 = (*r == 0)
        ],
        term: Terminator::BrIf {
            cond: 9,
            then_blk: 12,
            then_args: vec![0, 1, 2, 7], // env_next(name, envc, j, p_next = r+1)
            else_blk: 11,
            else_args: vec![0, 1, 2, 3, 7], // env_scan(.., r+1)
        },
    };

    // block 12: env_next(name=0, envc=1, j=2, p_next=3) — j++ and loop.
    let b12 = Block {
        params: vec![ValType::I64, ValType::I64, ValType::I64, ValType::I64],
        insts: vec![Inst::ConstI64(1), add(2, 4)], // v4=1, v5 = j+1
        term: Terminator::Br {
            target: 5,
            args: vec![0, 1, 5, 3],
        },
    };

    // block 13: not_found() — return NULL.
    let b13 = Block {
        params: vec![],
        insts: vec![Inst::ConstI64(0)], // v0
        term: Terminator::Return(vec![0]),
    };

    Func {
        params: vec![ValType::I64],
        results: vec![ValType::I64],
        blocks: vec![b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13],
    }
}

/// A tiny auto-numbering block builder for synthesizing CFGs by hand. Value indices in svm-ir are
/// **block-local** (params first, then each value-producing inst in order, §3a), so `Bdr` tracks the
/// running count: `push` returns the new value's index and bumps the counter, `eff` appends a
/// 0-result inst (store/fence) without consuming an index. The convenience methods cover the shapes
/// `__svm_dtoa_fixed` leans on (consts, i64 arith/compare, width-typed loads/stores, select).
struct Bdr {
    insts: Vec<Inst>,
    n: u32,
}
impl Bdr {
    fn new(nparams: u32) -> Self {
        Bdr {
            insts: Vec::new(),
            n: nparams,
        }
    }
    fn push(&mut self, i: Inst) -> u32 {
        let id = self.n;
        self.insts.push(i);
        self.n += 1;
        id
    }
    fn eff(&mut self, i: Inst) {
        self.insts.push(i);
    }
    fn k(&mut self, v: i64) -> u32 {
        self.push(Inst::ConstI64(v))
    }
    fn bin(&mut self, op: BinOp, a: ValIdx, b: ValIdx) -> u32 {
        self.push(Inst::IntBin {
            ty: IntTy::I64,
            op,
            a,
            b,
        })
    }
    fn cmp(&mut self, op: CmpOp, a: ValIdx, b: ValIdx) -> u32 {
        self.push(Inst::IntCmp {
            ty: IntTy::I64,
            op,
            a,
            b,
        })
    }
    /// `bin` with an immediate second operand — materializes the constant first (so a `b.k(..)` need
    /// not be nested inside another `b.method(..)` call, which would double-borrow the builder).
    fn bini(&mut self, op: BinOp, a: ValIdx, imm: i64) -> u32 {
        let k = self.k(imm);
        self.bin(op, a, k)
    }
    /// Compare `a` against an immediate — materializes the constant first, so callers never pass a
    /// bare literal as a value index (in a no-param block index `0` is the first inst, not zero).
    fn cmpi(&mut self, op: CmpOp, a: ValIdx, imm: i64) -> u32 {
        let c = self.k(imm);
        self.cmp(op, a, c)
    }
    fn sel(&mut self, cond: ValIdx, a: ValIdx, b: ValIdx) -> u32 {
        self.push(Inst::Select { cond, a, b })
    }
    fn load64(&mut self, addr: ValIdx) -> u32 {
        self.push(Inst::Load {
            op: svm_ir::LoadOp::I64,
            addr,
            offset: 0,
            align: 0,
        })
    }
    /// Zero-extended u32 load (a 128-bit limb) into an i64.
    fn load32u(&mut self, addr: ValIdx) -> u32 {
        let t = self.push(Inst::Load {
            op: svm_ir::LoadOp::I32,
            addr,
            offset: 0,
            align: 0,
        });
        self.push(Inst::Convert {
            op: ConvOp::ExtendI32U,
            a: t,
        })
    }
    /// Zero-extended u8 load into an i64.
    fn load8u(&mut self, addr: ValIdx) -> u32 {
        let t = self.push(Inst::Load {
            op: svm_ir::LoadOp::I32_8U,
            addr,
            offset: 0,
            align: 0,
        });
        self.push(Inst::Convert {
            op: ConvOp::ExtendI32U,
            a: t,
        })
    }
    fn store64(&mut self, addr: ValIdx, value: ValIdx) {
        self.eff(Inst::Store {
            op: svm_ir::StoreOp::I64,
            addr,
            value,
            offset: 0,
            align: 0,
        });
    }
    fn store32(&mut self, addr: ValIdx, value: ValIdx) {
        let w = self.push(Inst::Convert {
            op: ConvOp::WrapI64,
            a: value,
        });
        self.eff(Inst::Store {
            op: svm_ir::StoreOp::I32,
            addr,
            value: w,
            offset: 0,
            align: 0,
        });
    }
    fn store8(&mut self, addr: ValIdx, value: ValIdx) {
        let w = self.push(Inst::Convert {
            op: ConvOp::WrapI64,
            a: value,
        });
        self.eff(Inst::Store {
            op: svm_ir::StoreOp::I32_8,
            addr,
            value: w,
            offset: 0,
            align: 0,
        });
    }
    /// Zero-extend an `i32` (e.g. a compare result) to `i64`, so it can feed the I64-typed `bin`.
    fn ext(&mut self, a: ValIdx) -> u32 {
        self.push(Inst::Convert {
            op: ConvOp::ExtendI32U,
            a,
        })
    }
    /// Truncate an i64 to i32 (for an `i32.atomic.*` operand / the enclosing-word value).
    fn wrap32(&mut self, a: ValIdx) -> u32 {
        self.push(Inst::Convert {
            op: ConvOp::WrapI64,
            a,
        })
    }
    /// Seq-cst `i32.atomic.load` of the enclosing word (yields i32).
    fn atomic_load32(&mut self, addr: ValIdx) -> u32 {
        self.push(Inst::AtomicLoad {
            ty: IntTy::I32,
            addr,
            offset: 0,
            order: Ordering::SeqCst,
        })
    }
    /// Seq-cst `i32.atomic.cmpxchg` of the enclosing word; yields the old (i32) word value.
    fn atomic_cas32(&mut self, addr: ValIdx, expected: ValIdx, replacement: ValIdx) -> u32 {
        self.push(Inst::AtomicCmpxchg {
            ty: IntTy::I32,
            addr,
            expected,
            replacement,
            offset: 0,
            order: Ordering::SeqCst,
        })
    }
    /// Call a synth helper that returns one i64 (`big_cmp`); yields its result value.
    fn call1(&mut self, func: u32, args: Vec<ValIdx>) -> u32 {
        self.push(Inst::Call { func, args })
    }
    /// Call a synth helper that returns nothing (`big_zero`/`sub`/`mul_small`/`shl_bits`/`copy`).
    fn call0(&mut self, func: u32, args: Vec<ValIdx>) {
        self.eff(Inst::Call { func, args });
    }
    fn block(self, params: Vec<ValType>, term: Terminator) -> Block {
        Block {
            params,
            insts: self.insts,
            term,
        }
    }
}

/// Fixed limb count for the bignum float formatter (`%f`/`%e`/`%g`). A double's exact value `f·2^e`
/// needs a denominator up to `2^1074` (≈ 34 u32 limbs) plus decimal scaling headroom; 40 limbs
/// (1280 bits) covers every finite double with margin. Each big integer is a little-endian array of
/// `BIG_NLIMBS` u32 limbs at a byte address passed to these helpers.
const BIG_NLIMBS: i64 = 40;

/// `__svm_big_is_zero(a) -> i64` — 1 if every limb of the big integer at `a` is zero, else 0.
#[allow(dead_code)]
fn synth_big_is_zero() -> Func {
    use BinOp::{Add, Shl};
    use CmpOp::{GeS, Ne};
    let i64t = ValType::I64;
    const LOOP: u32 = 1;
    const BODY: u32 = 2;
    const RET1: u32 = 3;
    const RET0: u32 = 4;
    // 0: entry(a) → LOOP(a, 0)
    let b0 = {
        let mut b = Bdr::new(1);
        let z = b.k(0);
        b.block(
            vec![i64t],
            Terminator::Br {
                target: LOOP,
                args: vec![0, z],
            },
        )
    };
    // 1: LOOP(a, i) — i ≥ N ⇒ all-zero, else test limb i
    let b1 = {
        let mut b = Bdr::new(2);
        let done = b.cmpi(GeS, 1, BIG_NLIMBS);
        b.block(
            vec![i64t, i64t],
            Terminator::BrIf {
                cond: done,
                then_blk: RET1,
                then_args: vec![],
                else_blk: BODY,
                else_args: vec![0, 1],
            },
        )
    };
    // 2: BODY(a, i) — limb != 0 ⇒ not zero, else advance
    let b2 = {
        let mut b = Bdr::new(2);
        let c2 = b.k(2);
        let off = b.bin(Shl, 1, c2);
        let addr = b.bin(Add, 0, off);
        let la = b.load32u(addr);
        let nz = b.cmpi(Ne, la, 0);
        let c1 = b.k(1);
        let ni = b.bin(Add, 1, c1);
        b.block(
            vec![i64t, i64t],
            Terminator::BrIf {
                cond: nz,
                then_blk: RET0,
                then_args: vec![],
                else_blk: LOOP,
                else_args: vec![0, ni],
            },
        )
    };
    let b3 = {
        let mut b = Bdr::new(0);
        let one = b.k(1);
        b.block(vec![], Terminator::Return(vec![one]))
    };
    let b4 = {
        let mut b = Bdr::new(0);
        let zero = b.k(0);
        b.block(vec![], Terminator::Return(vec![zero]))
    };
    Func {
        params: vec![i64t],
        results: vec![i64t],
        blocks: vec![b0, b1, b2, b3, b4],
    }
}

/// `__svm_big_cmp(a, b) -> i64` — `-1`/`0`/`1` for `a < b` / `a == b` / `a > b` (unsigned, high limb
/// first).
#[allow(dead_code)]
fn synth_big_cmp() -> Func {
    use BinOp::{Add, Shl, Sub};
    use CmpOp::{GtU, LtS, LtU};
    let i64t = ValType::I64;
    const LOOP: u32 = 1;
    const BODY: u32 = 2;
    const BODY2: u32 = 3;
    const RETM1: u32 = 4;
    const RETP1: u32 = 5;
    const RET0: u32 = 6;
    // 0: entry(a, b) → LOOP(a, b, N-1)
    let b0 = {
        let mut b = Bdr::new(2);
        let top = b.k(BIG_NLIMBS - 1);
        b.block(
            vec![i64t, i64t],
            Terminator::Br {
                target: LOOP,
                args: vec![0, 1, top],
            },
        )
    };
    // 1: LOOP(a, b, i) — i < 0 ⇒ equal, else compare limb i
    let b1 = {
        let mut b = Bdr::new(3);
        let done = b.cmpi(LtS, 2, 0);
        b.block(
            vec![i64t, i64t, i64t],
            Terminator::BrIf {
                cond: done,
                then_blk: RET0,
                then_args: vec![],
                else_blk: BODY,
                else_args: vec![0, 1, 2],
            },
        )
    };
    // 2: BODY(a, b, i) — la < lb ⇒ -1, else check greater
    let b2 = {
        let mut b = Bdr::new(3);
        let c2 = b.k(2);
        let off = b.bin(Shl, 2, c2);
        let aa = b.bin(Add, 0, off);
        let la = b.load32u(aa);
        let ba = b.bin(Add, 1, off);
        let lb = b.load32u(ba);
        let lt = b.cmp(LtU, la, lb);
        b.block(
            vec![i64t, i64t, i64t],
            Terminator::BrIf {
                cond: lt,
                then_blk: RETM1,
                then_args: vec![],
                else_blk: BODY2,
                else_args: vec![0, 1, 2, la, lb],
            },
        )
    };
    // 3: BODY2(a, b, i, la, lb) — la > lb ⇒ 1, else next limb
    let b3 = {
        let mut b = Bdr::new(5);
        let gt = b.cmp(GtU, 3, 4);
        let c1 = b.k(1);
        let ni = b.bin(Sub, 2, c1);
        b.block(
            vec![i64t, i64t, i64t, i64t, i64t],
            Terminator::BrIf {
                cond: gt,
                then_blk: RETP1,
                then_args: vec![],
                else_blk: LOOP,
                else_args: vec![0, 1, ni],
            },
        )
    };
    let bm1 = {
        let mut b = Bdr::new(0);
        let v = b.k(-1);
        b.block(vec![], Terminator::Return(vec![v]))
    };
    let bp1 = {
        let mut b = Bdr::new(0);
        let v = b.k(1);
        b.block(vec![], Terminator::Return(vec![v]))
    };
    let b0e = {
        let mut b = Bdr::new(0);
        let v = b.k(0);
        b.block(vec![], Terminator::Return(vec![v]))
    };
    Func {
        params: vec![i64t, i64t],
        results: vec![i64t],
        blocks: vec![b0, b1, b2, b3, bm1, bp1, b0e],
    }
}

/// `__svm_big_sub(a, b)` — `a -= b` in place (assumes `a ≥ b`); borrow-propagating, low limb first.
#[allow(dead_code)]
fn synth_big_sub() -> Func {
    use BinOp::{Add, And, Shl, ShrS, Sub};
    use CmpOp::GeS;
    let i64t = ValType::I64;
    let mask32: i64 = 0xFFFF_FFFF;
    const LOOP: u32 = 1;
    const BODY: u32 = 2;
    const RET: u32 = 3;
    // 0: entry(a, b) → LOOP(a, b, 0, 0)
    let b0 = {
        let mut b = Bdr::new(2);
        let z0 = b.k(0);
        let z1 = b.k(0);
        b.block(
            vec![i64t, i64t],
            Terminator::Br {
                target: LOOP,
                args: vec![0, 1, z0, z1],
            },
        )
    };
    // 1: LOOP(a, b, i, borrow)
    let b1 = {
        let mut b = Bdr::new(4);
        let done = b.cmpi(GeS, 2, BIG_NLIMBS);
        b.block(
            vec![i64t, i64t, i64t, i64t],
            Terminator::BrIf {
                cond: done,
                then_blk: RET,
                then_args: vec![],
                else_blk: BODY,
                else_args: vec![0, 1, 2, 3],
            },
        )
    };
    // 2: BODY(a, b, i, borrow) — t = la - lb - borrow; store low 32; borrow' = (t < 0)
    let b2 = {
        let mut b = Bdr::new(4);
        let c2 = b.k(2);
        let off = b.bin(Shl, 2, c2);
        let aa = b.bin(Add, 0, off);
        let la = b.load32u(aa);
        let ba = b.bin(Add, 1, off);
        let lb = b.load32u(ba);
        let d = b.bin(Sub, la, lb);
        let t = b.bin(Sub, d, 3);
        let mask = b.k(mask32);
        let new = b.bin(And, t, mask);
        b.store32(aa, new);
        let c63 = b.k(63);
        let sgn = b.bin(ShrS, t, c63);
        let one = b.k(1);
        let nb = b.bin(And, sgn, one);
        let c1 = b.k(1);
        let ni = b.bin(Add, 2, c1);
        b.block(
            vec![i64t, i64t, i64t, i64t],
            Terminator::Br {
                target: LOOP,
                args: vec![0, 1, ni, nb],
            },
        )
    };
    let b3 = {
        let b = Bdr::new(0);
        b.block(vec![], Terminator::Return(vec![]))
    };
    Func {
        params: vec![i64t, i64t],
        results: vec![],
        blocks: vec![b0, b1, b2, b3],
    }
}

/// `__svm_big_mul_small(a, c)` — `a *= c` in place (small `c`; result must fit `BIG_NLIMBS` limbs);
/// carry-propagating, low limb first.
#[allow(dead_code)]
fn synth_big_mul_small() -> Func {
    use BinOp::{Add, And, Mul, Shl, ShrU};
    use CmpOp::GeS;
    let i64t = ValType::I64;
    let mask32: i64 = 0xFFFF_FFFF;
    const LOOP: u32 = 1;
    const BODY: u32 = 2;
    const RET: u32 = 3;
    // 0: entry(a, c) → LOOP(a, c, 0, 0)
    let b0 = {
        let mut b = Bdr::new(2);
        let i0 = b.k(0);
        let cy0 = b.k(0);
        b.block(
            vec![i64t, i64t],
            Terminator::Br {
                target: LOOP,
                args: vec![0, 1, i0, cy0],
            },
        )
    };
    // 1: LOOP(a, c, i, carry)
    let b1 = {
        let mut b = Bdr::new(4);
        let done = b.cmpi(GeS, 2, BIG_NLIMBS);
        b.block(
            vec![i64t, i64t, i64t, i64t],
            Terminator::BrIf {
                cond: done,
                then_blk: RET,
                then_args: vec![],
                else_blk: BODY,
                else_args: vec![0, 1, 2, 3],
            },
        )
    };
    // 2: BODY(a, c, i, carry) — t = la*c + carry; store low 32; carry' = t >> 32
    let b2 = {
        let mut b = Bdr::new(4);
        let c2 = b.k(2);
        let off = b.bin(Shl, 2, c2);
        let aa = b.bin(Add, 0, off);
        let la = b.load32u(aa);
        let prod = b.bin(Mul, la, 1);
        let t = b.bin(Add, prod, 3);
        let mask = b.k(mask32);
        let new = b.bin(And, t, mask);
        b.store32(aa, new);
        let c32 = b.k(32);
        let ncy = b.bin(ShrU, t, c32);
        let c1 = b.k(1);
        let ni = b.bin(Add, 2, c1);
        b.block(
            vec![i64t, i64t, i64t, i64t],
            Terminator::Br {
                target: LOOP,
                args: vec![0, 1, ni, ncy],
            },
        )
    };
    let b3 = {
        let b = Bdr::new(0);
        b.block(vec![], Terminator::Return(vec![]))
    };
    Func {
        params: vec![i64t, i64t],
        results: vec![],
        blocks: vec![b0, b1, b2, b3],
    }
}

/// `__svm_big_shl_bits(a, n)` — `a <<= n` bits in place. Processes high limb first (so the in-place
/// reads see pre-shift limbs); a shift past the top simply zero-fills.
#[allow(dead_code)]
fn synth_big_shl_bits() -> Func {
    use BinOp::{Add, And, Or, Shl, ShrU, Sub};
    use CmpOp::LtS;
    let i64t = ValType::I64;
    let mask32: i64 = 0xFFFF_FFFF;
    const LOOP: u32 = 1;
    const BODY: u32 = 2;
    const RET: u32 = 3;
    // 0: entry(a, n) — word = n>>5, bit = n&31 → LOOP(a, word, bit, N-1)
    let b0 = {
        let mut b = Bdr::new(2);
        let c5 = b.k(5);
        let word = b.bin(ShrU, 1, c5);
        let c31 = b.k(31);
        let bit = b.bin(And, 1, c31);
        let top = b.k(BIG_NLIMBS - 1);
        b.block(
            vec![i64t, i64t],
            Terminator::Br {
                target: LOOP,
                args: vec![0, word, bit, top],
            },
        )
    };
    // 1: LOOP(a, word, bit, i) — i < 0 ⇒ done
    let b1 = {
        let mut b = Bdr::new(4);
        let done = b.cmpi(LtS, 3, 0);
        b.block(
            vec![i64t, i64t, i64t, i64t],
            Terminator::BrIf {
                cond: done,
                then_blk: RET,
                then_args: vec![],
                else_blk: BODY,
                else_args: vec![0, 1, 2, 3],
            },
        )
    };
    // 2: BODY(a, word, bit, i) — res = (a[i-word] << bit) | (a[i-word-1] >> (32-bit))
    let b2 = {
        let mut b = Bdr::new(4); // a=0, word=1, bit=2, i=3
        let c2 = b.k(2);
        // src_hi = i - word, src_lo = src_hi - 1
        let shi = b.bin(Sub, 3, 1);
        let one = b.k(1);
        let slo = b.bin(Sub, shi, one);
        // hi limb (0 when src_hi < 0): clamp address to a, select 0
        let zero = b.k(0);
        let hi_ok = b.cmp(CmpOp::GeS, shi, zero);
        let shc = b.sel(hi_ok, shi, zero);
        let offh = b.bin(Shl, shc, c2);
        let addh = b.bin(Add, 0, offh);
        let vhr = b.load32u(addh);
        let vhi = b.sel(hi_ok, vhr, zero);
        // lo limb (0 when src_lo < 0)
        let zero2 = b.k(0);
        let lo_ok = b.cmp(CmpOp::GeS, slo, zero2);
        let slc = b.sel(lo_ok, slo, zero2);
        let offl = b.bin(Shl, slc, c2);
        let addl = b.bin(Add, 0, offl);
        let vlr = b.load32u(addl);
        let vlo = b.sel(lo_ok, vlr, zero2);
        // res = ((vhi << bit) | (vlo >> (32 - bit))) & mask  (bit==0 ⇒ vlo>>32 == 0 ⇒ res == vhi)
        let up = b.bin(Shl, vhi, 2);
        let c32 = b.k(32);
        let rsh = b.bin(Sub, c32, 2);
        let dn = b.bin(ShrU, vlo, rsh);
        let orr = b.bin(Or, up, dn);
        let mask = b.k(mask32);
        let res = b.bin(And, orr, mask);
        let offi = b.bin(Shl, 3, c2);
        let addi = b.bin(Add, 0, offi);
        b.store32(addi, res);
        let c1 = b.k(1);
        let ni = b.bin(Sub, 3, c1);
        b.block(
            vec![i64t, i64t, i64t, i64t],
            Terminator::Br {
                target: LOOP,
                args: vec![0, 1, 2, ni],
            },
        )
    };
    let b3 = {
        let b = Bdr::new(0);
        b.block(vec![], Terminator::Return(vec![]))
    };
    Func {
        params: vec![i64t, i64t],
        results: vec![],
        blocks: vec![b0, b1, b2, b3],
    }
}

/// `__svm_big_zero(a)` — set every limb of the big integer at `a` to 0.
#[allow(dead_code)]
fn synth_big_zero() -> Func {
    use BinOp::{Add, Shl};
    use CmpOp::GeS;
    let i64t = ValType::I64;
    const LOOP: u32 = 1;
    const BODY: u32 = 2;
    const RET: u32 = 3;
    let b0 = {
        let mut b = Bdr::new(1);
        let z = b.k(0);
        b.block(
            vec![i64t],
            Terminator::Br {
                target: LOOP,
                args: vec![0, z],
            },
        )
    };
    let b1 = {
        let mut b = Bdr::new(2);
        let done = b.cmpi(GeS, 1, BIG_NLIMBS);
        b.block(
            vec![i64t, i64t],
            Terminator::BrIf {
                cond: done,
                then_blk: RET,
                then_args: vec![],
                else_blk: BODY,
                else_args: vec![0, 1],
            },
        )
    };
    let b2 = {
        let mut b = Bdr::new(2);
        let c2 = b.k(2);
        let off = b.bin(Shl, 1, c2);
        let addr = b.bin(Add, 0, off);
        let z = b.k(0);
        b.store32(addr, z);
        let c1 = b.k(1);
        let ni = b.bin(Add, 1, c1);
        b.block(
            vec![i64t, i64t],
            Terminator::Br {
                target: LOOP,
                args: vec![0, ni],
            },
        )
    };
    let b3 = {
        let b = Bdr::new(0);
        b.block(vec![], Terminator::Return(vec![]))
    };
    Func {
        params: vec![i64t],
        results: vec![],
        blocks: vec![b0, b1, b2, b3],
    }
}

/// `__svm_big_copy(dst, src)` — copy all `BIG_NLIMBS` limbs from `src` to `dst`.
#[allow(dead_code)]
fn synth_big_copy() -> Func {
    use BinOp::{Add, Shl};
    use CmpOp::GeS;
    let i64t = ValType::I64;
    const LOOP: u32 = 1;
    const BODY: u32 = 2;
    const RET: u32 = 3;
    let b0 = {
        let mut b = Bdr::new(2);
        let z = b.k(0);
        b.block(
            vec![i64t, i64t],
            Terminator::Br {
                target: LOOP,
                args: vec![0, 1, z],
            },
        )
    };
    let b1 = {
        let mut b = Bdr::new(3);
        let done = b.cmpi(GeS, 2, BIG_NLIMBS);
        b.block(
            vec![i64t, i64t, i64t],
            Terminator::BrIf {
                cond: done,
                then_blk: RET,
                then_args: vec![],
                else_blk: BODY,
                else_args: vec![0, 1, 2],
            },
        )
    };
    let b2 = {
        let mut b = Bdr::new(3); // dst=0, src=1, i=2
        let c2 = b.k(2);
        let off = b.bin(Shl, 2, c2);
        let sa = b.bin(Add, 1, off);
        let v = b.load32u(sa);
        let da = b.bin(Add, 0, off);
        b.store32(da, v);
        let c1 = b.k(1);
        let ni = b.bin(Add, 2, c1);
        b.block(
            vec![i64t, i64t, i64t],
            Terminator::Br {
                target: LOOP,
                args: vec![0, 1, ni],
            },
        )
    };
    let b3 = {
        let b = Bdr::new(0);
        b.block(vec![], Terminator::Return(vec![]))
    };
    Func {
        params: vec![i64t, i64t],
        results: vec![],
        blocks: vec![b0, b1, b2, b3],
    }
}

/// `__svm_dtoa_digits(bits, nsig, scratch) -> E` — the exact Dragon4-style digit generator: produce
/// `nsig` correctly-rounded (half-to-even) significant decimal digits of the finite double `bits`,
/// writing them (one digit value `0..9` per byte) to `scratch+DD_DBUF`, and return the decimal
/// exponent `E` such that `value = d0.d1…d(nsig-1) × 10^E`. Zero yields all-zero digits and `E=0`;
/// the caller handles inf/nan. Big integers `num`/`den`/`tmp` live in `scratch` (40-limb each); the
/// algorithm builds `value = num/den` exactly, scales by `10^E_est` (estimate, then exact ±1 fixup)
/// so `num/den ∈ [1,10)`, then emits digits by `d = #(num ≥ den) subtractions; num ×= 10`.
///
/// Composes the `synth_big_*` primitives by their **func indices** (so it is independently
/// unit-testable: build a module of `[dtoa_digits, big_zero, big_copy, big_cmp, big_sub,
/// big_mul_small, big_shl_bits]` and inspect `scratch` after `run_capture`).
#[allow(dead_code)]
fn synth_dtoa_digits(zero: u32, copy: u32, cmp: u32, sub: u32, mul: u32, shl: u32) -> Func {
    use BinOp::{Add, And, Mul, Or, ShrS, ShrU, Sub};
    use CmpOp::{Eq, GeS, LtS, Ne};
    let i64t = ValType::I64;
    // `scratch` layout (byte offsets): three 40-limb (160-byte) big integers, the digit buffer, then
    // the two stashed scalars `nsig`/`E` (so blocks thread only `scratch` + loop vars).
    let num_o = 0i64;
    let den_o = 160i64;
    let tmp_o = 320i64;
    // The two scalars sit just above the three 40-limb (160-byte) big integers, so the digit buffer
    // that follows can grow as large as high-precision / large-magnitude formats need.
    let nsig_o = 480i64;
    let e_o = 488i64;
    let dbuf_o = 496i64;

    const ZEROCASE: u32 = 1;
    const ZCASE_LOOP: u32 = 2;
    const BUILD: u32 = 3;
    const SHLNUM: u32 = 4;
    const SHLDEN: u32 = 5;
    const SCALE: u32 = 6;
    const SCALE_DEN: u32 = 7;
    const SCALE_NUM: u32 = 8;
    const FIXUP_HI: u32 = 9;
    const FIXUP_LO: u32 = 10;
    const DIGIT_INNER: u32 = 11;
    const DIGIT_STORE: u32 = 12;
    const ROUND: u32 = 13;
    const ROUND_CARRY: u32 = 14;
    const CARRY_OUT: u32 = 15;
    const ZERO_TAIL: u32 = 16;
    const RET: u32 = 17;

    // 0: ENTRY(bits, nsig, scratch) — decode IEEE-754; stash nsig; split zero vs finite-nonzero.
    let b_entry = {
        let mut b = Bdr::new(3); // bits=0, nsig=1, scratch=2
        let kn = b.k(nsig_o);
        let nsa = b.bin(Add, 2, kn);
        b.store64(nsa, 1);
        let c52 = b.k(52);
        let e0 = b.bin(ShrU, 0, c52);
        let m7ff = b.k(0x7FF);
        let exp = b.bin(And, e0, m7ff);
        let mmask = b.k((1i64 << 52) - 1);
        let mant = b.bin(And, 0, mmask);
        let iszexp = b.cmpi(Eq, exp, 0);
        let imp = b.k(1i64 << 52);
        let zc = b.k(0);
        let fimp = b.sel(iszexp, zc, imp);
        let f = b.bin(Or, mant, fimp);
        let cm1074 = b.k(-1074);
        let c1075 = b.k(1075);
        let enorm = b.bin(Sub, exp, c1075);
        let ebin = b.sel(iszexp, cm1074, enorm);
        let iszero = b.cmpi(Eq, f, 0);
        b.block(
            vec![i64t, i64t, i64t],
            Terminator::BrIf {
                cond: iszero,
                then_blk: ZEROCASE,
                then_args: vec![2],
                else_blk: BUILD,
                else_args: vec![2, f, ebin],
            },
        )
    };

    // 1: ZEROCASE(scratch) — E = 0, digits all zero.
    let b_zerocase = {
        let mut b = Bdr::new(1);
        let ke = b.k(e_o);
        let ea = b.bin(Add, 0, ke);
        let z = b.k(0);
        b.store64(ea, z);
        let z2 = b.k(0);
        b.block(
            vec![i64t],
            Terminator::Br {
                target: ZCASE_LOOP,
                args: vec![0, z2],
            },
        )
    };

    // 2: ZCASE_LOOP(scratch, i) — dbuf[i]=0 for i in 0..nsig.
    let b_zcase_loop = {
        let mut b = Bdr::new(2);
        let kn = b.k(nsig_o);
        let nsa = b.bin(Add, 0, kn);
        let nsig = b.load64(nsa);
        let done = b.cmp(GeS, 1, nsig);
        b.block(
            vec![i64t, i64t],
            Terminator::BrIf {
                cond: done,
                then_blk: RET,
                then_args: vec![0],
                else_blk: 18, // ZCASE_BODY
                else_args: vec![0, 1],
            },
        )
    };

    // 3: BUILD(scratch, f, ebin) — zero the bigints, seed num=f / den=1, stash E_est, then shift.
    let b_build = {
        let mut b = Bdr::new(3); // scratch=0, f=1, ebin=2
        let kn = b.k(num_o);
        let num = b.bin(Add, 0, kn);
        let kd = b.k(den_o);
        let den = b.bin(Add, 0, kd);
        let kt = b.k(tmp_o);
        let tmp = b.bin(Add, 0, kt);
        b.call0(zero, vec![num]);
        b.call0(zero, vec![den]);
        b.call0(zero, vec![tmp]);
        // num[0..1] = f (low/high 32); den[0] = 1
        b.store32(num, 1);
        let c4 = b.k(4);
        let num4 = b.bin(Add, num, c4);
        let c32 = b.k(32);
        let fhi = b.bin(ShrU, 1, c32);
        b.store32(num4, fhi);
        let one = b.k(1);
        b.store32(den, one);
        // E_est = ((ebin + 52) * 1233) >> 12   (≈ floor(log10(2^(ebin+52))))
        let c52 = b.k(52);
        let s1 = b.bin(Add, 2, c52);
        let c1233 = b.k(1233);
        let s2 = b.bin(Mul, s1, c1233);
        let c12 = b.k(12);
        let eest = b.bin(ShrS, s2, c12);
        let ke = b.k(e_o);
        let ea = b.bin(Add, 0, ke);
        b.store64(ea, eest);
        let eneg = b.cmpi(LtS, 2, 0);
        b.block(
            vec![i64t, i64t, i64t],
            Terminator::BrIf {
                cond: eneg,
                then_blk: SHLDEN,
                then_args: vec![0, 2],
                else_blk: SHLNUM,
                else_args: vec![0, 2],
            },
        )
    };

    // 4: SHLNUM(scratch, ebin) — num <<= ebin (ebin ≥ 0).
    let b_shlnum = {
        let mut b = Bdr::new(2);
        let kn = b.k(num_o);
        let num = b.bin(Add, 0, kn);
        b.call0(shl, vec![num, 1]);
        b.block(
            vec![i64t, i64t],
            Terminator::Br {
                target: SCALE,
                args: vec![0],
            },
        )
    };

    // 5: SHLDEN(scratch, ebin) — den <<= -ebin (ebin < 0).
    let b_shlden = {
        let mut b = Bdr::new(2);
        let kd = b.k(den_o);
        let den = b.bin(Add, 0, kd);
        let z = b.k(0);
        let neg = b.bin(Sub, z, 1);
        b.call0(shl, vec![den, neg]);
        b.block(
            vec![i64t, i64t],
            Terminator::Br {
                target: SCALE,
                args: vec![0],
            },
        )
    };

    // 6: SCALE(scratch) — multiply den by 10^E (E≥0) or num by 10^(-E) (E<0).
    let b_scale = {
        let mut b = Bdr::new(1);
        let ke = b.k(e_o);
        let ea = b.bin(Add, 0, ke);
        let e = b.load64(ea);
        let eneg = b.cmpi(LtS, e, 0);
        let z = b.k(0);
        let nege = b.bin(Sub, z, e);
        b.block(
            vec![i64t],
            Terminator::BrIf {
                cond: eneg,
                then_blk: SCALE_NUM,
                then_args: vec![0, nege],
                else_blk: SCALE_DEN,
                else_args: vec![0, e],
            },
        )
    };

    // 7: SCALE_DEN(scratch, cnt) — loop test; the den *= 10 effect lives in SCALE_DEN_BODY.
    let b_scale_den = {
        let mut b = Bdr::new(2);
        let done = b.cmpi(LtS, 1, 1); // cnt < 1 ⇒ cnt <= 0
        b.block(
            vec![i64t, i64t],
            Terminator::BrIf {
                cond: done,
                then_blk: FIXUP_HI,
                then_args: vec![0],
                else_blk: 19, // SCALE_DEN_BODY
                else_args: vec![0, 1],
            },
        )
    };

    // 8: SCALE_NUM(scratch, cnt) — num *= 10, cnt times.
    let b_scale_num = {
        let mut b = Bdr::new(2);
        let done = b.cmpi(LtS, 1, 1);
        b.block(
            vec![i64t, i64t],
            Terminator::BrIf {
                cond: done,
                then_blk: FIXUP_HI,
                then_args: vec![0],
                else_blk: 20, // SCALE_NUM_BODY
                else_args: vec![0, 1],
            },
        )
    };

    // 9: FIXUP_HI(scratch) — while num ≥ 10·den: den *= 10, E++.
    let b_fixup_hi = {
        let mut b = Bdr::new(1);
        let kt = b.k(tmp_o);
        let tmp = b.bin(Add, 0, kt);
        let kd = b.k(den_o);
        let den = b.bin(Add, 0, kd);
        let kn = b.k(num_o);
        let num = b.bin(Add, 0, kn);
        b.call0(copy, vec![tmp, den]); // tmp = den
        let c10 = b.k(10);
        b.call0(mul, vec![tmp, c10]); // tmp = 10·den
        let c = b.call1(cmp, vec![num, tmp]); // num vs 10·den
        let lt = b.cmpi(LtS, c, 0);
        b.block(
            vec![i64t],
            Terminator::BrIf {
                cond: lt,
                then_blk: FIXUP_LO,
                then_args: vec![0],
                else_blk: 21, // FIXUP_HI_BODY
                else_args: vec![0],
            },
        )
    };

    // 10: FIXUP_LO(scratch) — while num < den: num *= 10, E--.
    let b_fixup_lo = {
        let mut b = Bdr::new(1);
        let kd = b.k(den_o);
        let den = b.bin(Add, 0, kd);
        let kn = b.k(num_o);
        let num = b.bin(Add, 0, kn);
        let c = b.call1(cmp, vec![num, den]);
        let ge = b.cmpi(GeS, c, 0);
        let z = b.k(0);
        b.block(
            vec![i64t],
            Terminator::BrIf {
                cond: ge,
                then_blk: DIGIT_INNER,
                then_args: vec![0, z, z], // j = 0, d = 0
                else_blk: 22,             // FIXUP_LO_BODY
                else_args: vec![0],
            },
        )
    };

    // 11: DIGIT_INNER(scratch, j, d) — count d = #(num ≥ den) subtractions for digit j.
    let b_digit_inner = {
        let mut b = Bdr::new(3); // scratch=0, j=1, d=2
        let kd = b.k(den_o);
        let den = b.bin(Add, 0, kd);
        let kn = b.k(num_o);
        let num = b.bin(Add, 0, kn);
        let c = b.call1(cmp, vec![num, den]);
        let lt = b.cmpi(LtS, c, 0);
        b.block(
            vec![i64t, i64t, i64t],
            Terminator::BrIf {
                cond: lt,
                then_blk: DIGIT_STORE,
                then_args: vec![0, 1, 2],
                else_blk: 23, // DIGIT_SUB
                else_args: vec![0, 1, 2],
            },
        )
    };

    // 12: DIGIT_STORE(scratch, j, d) — dbuf[j]=d; if last digit ⇒ ROUND, else num*=10 and next j.
    let b_digit_store = {
        let mut b = Bdr::new(3);
        let kb = b.k(dbuf_o);
        let dba = b.bin(Add, 0, kb);
        let da = b.bin(Add, dba, 1);
        b.store8(da, 2);
        let c1 = b.k(1);
        let j2 = b.bin(Add, 1, c1);
        let kn = b.k(nsig_o);
        let nsa = b.bin(Add, 0, kn);
        let nsig = b.load64(nsa);
        let last = b.cmp(GeS, j2, nsig);
        b.block(
            vec![i64t, i64t, i64t],
            Terminator::BrIf {
                cond: last,
                then_blk: ROUND,
                then_args: vec![0],
                else_blk: 24, // DIGIT_NEXT
                else_args: vec![0, j2],
            },
        )
    };

    // 13: ROUND(scratch) — round half-to-even using 2·remainder vs den; carry if rounding up.
    let b_round = {
        let mut b = Bdr::new(1);
        let kn = b.k(num_o);
        let num = b.bin(Add, 0, kn);
        let c10 = b.k(2);
        b.call0(mul, vec![num, c10]); // num = 2·remainder
        let kd = b.k(den_o);
        let den = b.bin(Add, 0, kd);
        let c = b.call1(cmp, vec![num, den]);
        let gt = b.cmpi(CmpOp::GtS, c, 0);
        let gt64 = b.ext(gt);
        let eq = b.cmpi(Eq, c, 0);
        let eq64 = b.ext(eq);
        // last digit parity (tie ⇒ round to even). `odd` is an i64 0/1 already (And of i64s).
        let knsig = b.k(nsig_o);
        let nsa = b.bin(Add, 0, knsig);
        let nsig = b.load64(nsa);
        let c1 = b.k(1);
        let lastidx = b.bin(Sub, nsig, c1);
        let kb = b.k(dbuf_o);
        let dba = b.bin(Add, 0, kb);
        let la = b.bin(Add, dba, lastidx);
        let ld = b.load8u(la);
        let odd = b.bin(And, ld, c1);
        let tie = b.bin(And, eq64, odd); // tie (2·rem == den) and last digit odd ⇒ round to even
        let up = b.bin(Or, gt64, tie);
        let upb = b.cmpi(Ne, up, 0);
        b.block(
            vec![i64t],
            Terminator::BrIf {
                cond: upb,
                then_blk: ROUND_CARRY,
                then_args: vec![0, lastidx],
                else_blk: RET,
                else_args: vec![0],
            },
        )
    };

    // 14: ROUND_CARRY(scratch, k) — dbuf[k]++ with carry; past 0 ⇒ CARRY_OUT.
    let b_round_carry = {
        let mut b = Bdr::new(2); // scratch=0, k=1
        let neg = b.cmpi(LtS, 1, 0);
        b.block(
            vec![i64t, i64t],
            Terminator::BrIf {
                cond: neg,
                then_blk: CARRY_OUT,
                then_args: vec![0],
                else_blk: 25, // ROUND_CARRY_BODY
                else_args: vec![0, 1],
            },
        )
    };

    // 15: CARRY_OUT(scratch) — all 9s rolled over: dbuf="1"+zeros, E++.
    let b_carry_out = {
        let mut b = Bdr::new(1);
        let kb = b.k(dbuf_o);
        let dba = b.bin(Add, 0, kb);
        let one = b.k(1);
        b.store8(dba, one);
        let ke = b.k(e_o);
        let ea = b.bin(Add, 0, ke);
        let e = b.load64(ea);
        let c1 = b.k(1);
        let e2 = b.bin(Add, e, c1);
        b.store64(ea, e2);
        let one2 = b.k(1);
        b.block(
            vec![i64t],
            Terminator::Br {
                target: ZERO_TAIL,
                args: vec![0, one2],
            },
        )
    };

    // 16: ZERO_TAIL(scratch, i) — dbuf[i]=0 for i in 1..nsig (after a carry-out).
    let b_zero_tail = {
        let mut b = Bdr::new(2);
        let kn = b.k(nsig_o);
        let nsa = b.bin(Add, 0, kn);
        let nsig = b.load64(nsa);
        let done = b.cmp(GeS, 1, nsig);
        b.block(
            vec![i64t, i64t],
            Terminator::BrIf {
                cond: done,
                then_blk: RET,
                then_args: vec![0],
                else_blk: 26, // ZERO_TAIL_BODY
                else_args: vec![0, 1],
            },
        )
    };

    // 17: RET(scratch) — return the stashed E.
    let b_ret = {
        let mut b = Bdr::new(1);
        let ke = b.k(e_o);
        let ea = b.bin(Add, 0, ke);
        let e = b.load64(ea);
        b.block(vec![i64t], Terminator::Return(vec![e]))
    };

    // 18: ZCASE_BODY(scratch, i) — dbuf[i]=0; i++.
    let b_zcase_body = {
        let mut b = Bdr::new(2);
        let kb = b.k(dbuf_o);
        let dba = b.bin(Add, 0, kb);
        let a = b.bin(Add, dba, 1);
        let z = b.k(0);
        b.store8(a, z);
        let c1 = b.k(1);
        let ni = b.bin(Add, 1, c1);
        b.block(
            vec![i64t, i64t],
            Terminator::Br {
                target: ZCASE_LOOP,
                args: vec![0, ni],
            },
        )
    };

    // 19: SCALE_DEN_BODY(scratch, cnt) — den *= 10; cnt--.
    let b_scale_den_body = {
        let mut b = Bdr::new(2);
        let kd = b.k(den_o);
        let den = b.bin(Add, 0, kd);
        let c10 = b.k(10);
        b.call0(mul, vec![den, c10]);
        let c1 = b.k(1);
        let nc = b.bin(Sub, 1, c1);
        b.block(
            vec![i64t, i64t],
            Terminator::Br {
                target: SCALE_DEN,
                args: vec![0, nc],
            },
        )
    };

    // 20: SCALE_NUM_BODY(scratch, cnt) — num *= 10; cnt--.
    let b_scale_num_body = {
        let mut b = Bdr::new(2);
        let kn = b.k(num_o);
        let num = b.bin(Add, 0, kn);
        let c10 = b.k(10);
        b.call0(mul, vec![num, c10]);
        let c1 = b.k(1);
        let nc = b.bin(Sub, 1, c1);
        b.block(
            vec![i64t, i64t],
            Terminator::Br {
                target: SCALE_NUM,
                args: vec![0, nc],
            },
        )
    };

    // 21: FIXUP_HI_BODY(scratch) — den *= 10 (via tmp copy), E++.
    let b_fixup_hi_body = {
        let mut b = Bdr::new(1);
        let kd = b.k(den_o);
        let den = b.bin(Add, 0, kd);
        let c10 = b.k(10);
        b.call0(mul, vec![den, c10]);
        let ke = b.k(e_o);
        let ea = b.bin(Add, 0, ke);
        let e = b.load64(ea);
        let c1 = b.k(1);
        let e2 = b.bin(Add, e, c1);
        b.store64(ea, e2);
        b.block(
            vec![i64t],
            Terminator::Br {
                target: FIXUP_HI,
                args: vec![0],
            },
        )
    };

    // 22: FIXUP_LO_BODY(scratch) — num *= 10, E--.
    let b_fixup_lo_body = {
        let mut b = Bdr::new(1);
        let kn = b.k(num_o);
        let num = b.bin(Add, 0, kn);
        let c10 = b.k(10);
        b.call0(mul, vec![num, c10]);
        let ke = b.k(e_o);
        let ea = b.bin(Add, 0, ke);
        let e = b.load64(ea);
        let c1 = b.k(1);
        let e2 = b.bin(Sub, e, c1);
        b.store64(ea, e2);
        b.block(
            vec![i64t],
            Terminator::Br {
                target: FIXUP_LO,
                args: vec![0],
            },
        )
    };

    // 23: DIGIT_SUB(scratch, j, d) — num -= den; d++.
    let b_digit_sub = {
        let mut b = Bdr::new(3);
        let kd = b.k(den_o);
        let den = b.bin(Add, 0, kd);
        let kn = b.k(num_o);
        let num = b.bin(Add, 0, kn);
        b.call0(sub, vec![num, den]);
        let c1 = b.k(1);
        let d2 = b.bin(Add, 2, c1);
        b.block(
            vec![i64t, i64t, i64t],
            Terminator::Br {
                target: DIGIT_INNER,
                args: vec![0, 1, d2],
            },
        )
    };

    // 24: DIGIT_NEXT(scratch, j2) — num *= 10; start digit j2.
    let b_digit_next = {
        let mut b = Bdr::new(2);
        let kn = b.k(num_o);
        let num = b.bin(Add, 0, kn);
        let c10 = b.k(10);
        b.call0(mul, vec![num, c10]);
        let z = b.k(0);
        b.block(
            vec![i64t, i64t],
            Terminator::Br {
                target: DIGIT_INNER,
                args: vec![0, 1, z],
            },
        )
    };

    // 25: ROUND_CARRY_BODY(scratch, k) — dbuf[k]+1; <10 ⇒ done, else dbuf[k]=0 and carry to k-1.
    let b_round_carry_body = {
        let mut b = Bdr::new(2); // scratch=0, k=1
        let kb = b.k(dbuf_o);
        let dba = b.bin(Add, 0, kb);
        let a = b.bin(Add, dba, 1);
        let d = b.load8u(a);
        let c1 = b.k(1);
        let d2 = b.bin(Add, d, c1);
        let lt10 = b.cmpi(LtS, d2, 10);
        // Store either d2 (no carry) or 0 (carry); pick via select, then branch.
        let z = b.k(0);
        let stored = b.sel(lt10, d2, z);
        b.store8(a, stored);
        let km1 = b.bin(Sub, 1, c1);
        b.block(
            vec![i64t, i64t],
            Terminator::BrIf {
                cond: lt10,
                then_blk: RET,
                then_args: vec![0],
                else_blk: ROUND_CARRY,
                else_args: vec![0, km1],
            },
        )
    };

    // 26: ZERO_TAIL_BODY(scratch, i) — dbuf[i]=0; i++.
    let b_zero_tail_body = {
        let mut b = Bdr::new(2);
        let kb = b.k(dbuf_o);
        let dba = b.bin(Add, 0, kb);
        let a = b.bin(Add, dba, 1);
        let z = b.k(0);
        b.store8(a, z);
        let c1 = b.k(1);
        let ni = b.bin(Add, 1, c1);
        b.block(
            vec![i64t, i64t],
            Terminator::Br {
                target: ZERO_TAIL,
                args: vec![0, ni],
            },
        )
    };

    Func {
        params: vec![i64t, i64t, i64t],
        results: vec![i64t],
        blocks: vec![
            b_entry,
            b_zerocase,
            b_zcase_loop,
            b_build,
            b_shlnum,
            b_shlden,
            b_scale,
            b_scale_den,
            b_scale_num,
            b_fixup_hi,
            b_fixup_lo,
            b_digit_inner,
            b_digit_store,
            b_round,
            b_round_carry,
            b_carry_out,
            b_zero_tail,
            b_ret,
            b_zcase_body,
            b_scale_den_body,
            b_scale_num_body,
            b_fixup_hi_body,
            b_fixup_lo_body,
            b_digit_sub,
            b_digit_next,
            b_round_carry_body,
            b_zero_tail_body,
        ],
    }
}

/// `__svm_big_shr1(a) -> i64` — shift the big integer at `a` right by 1 bit, returning the bit shifted
/// out (its old bit 0). Low limb first (so each limb reads its not-yet-shifted higher neighbor).
#[allow(dead_code)]
fn synth_big_shr1() -> Func {
    use BinOp::{Add, And, Or, Shl, ShrU};
    use CmpOp::GeS;
    let i64t = ValType::I64;
    let mask32: i64 = 0xFFFF_FFFF;
    const LOOP: u32 = 1;
    const BODY: u32 = 2;
    const RET: u32 = 3;
    // 0: entry(a) — out = a[0] & 1 → LOOP(a, 0, out)
    let b0 = {
        let mut b = Bdr::new(1);
        let v0 = b.load32u(0);
        let out = b.bini(And, v0, 1);
        let z = b.k(0);
        b.block(
            vec![i64t],
            Terminator::Br {
                target: LOOP,
                args: vec![0, z, out],
            },
        )
    };
    // 1: LOOP(a, i, out)
    let b1 = {
        let mut b = Bdr::new(3);
        let done = b.cmpi(GeS, 1, BIG_NLIMBS);
        b.block(
            vec![i64t, i64t, i64t],
            Terminator::BrIf {
                cond: done,
                then_blk: RET,
                then_args: vec![2],
                else_blk: BODY,
                else_args: vec![0, 1, 2],
            },
        )
    };
    // 2: BODY(a, i, out) — a[i] = (a[i]>>1) | (a[i+1]<<31)
    let b2 = {
        let mut b = Bdr::new(3); // a=0, i=1, out=2
        let off = b.bini(Shl, 1, 2);
        let addr = b.bin(Add, 0, off);
        let cur = b.load32u(addr);
        // next limb (0 if i+1 == N)
        let ip1 = b.bini(Add, 1, 1);
        let last = b.cmpi(GeS, ip1, BIG_NLIMBS);
        let zero = b.k(0);
        let noff = b.bini(Shl, ip1, 2);
        let naddr = b.bin(Add, 0, noff);
        let nraw = b.load32u(naddr);
        let nxt = b.sel(last, zero, nraw);
        let rsh = b.bini(ShrU, cur, 1);
        let lsh = b.bini(Shl, nxt, 31);
        let orr = b.bin(Or, rsh, lsh);
        let mask = b.k(mask32);
        let new = b.bin(And, orr, mask);
        b.store32(addr, new);
        let ni = b.bini(Add, 1, 1);
        b.block(
            vec![i64t, i64t, i64t],
            Terminator::Br {
                target: LOOP,
                args: vec![0, ni, 2],
            },
        )
    };
    let b3 = {
        let b = Bdr::new(1);
        b.block(vec![i64t], Terminator::Return(vec![0]))
    };
    Func {
        params: vec![i64t],
        results: vec![i64t],
        blocks: vec![b0, b1, b2, b3],
    }
}

/// `__svm_big_divmod10(a) -> i64` — divide the big integer at `a` by 10 in place, returning the
/// remainder (`0..9`). High limb first.
#[allow(dead_code)]
fn synth_big_divmod10() -> Func {
    use BinOp::{Add, DivU, Or, RemU, Shl, Sub};
    use CmpOp::LtS;
    let i64t = ValType::I64;
    const LOOP: u32 = 1;
    const BODY: u32 = 2;
    const RET: u32 = 3;
    // 0: entry(a) → LOOP(a, N-1, rem=0)
    let b0 = {
        let mut b = Bdr::new(1);
        let top = b.k(BIG_NLIMBS - 1);
        let z = b.k(0);
        b.block(
            vec![i64t],
            Terminator::Br {
                target: LOOP,
                args: vec![0, top, z],
            },
        )
    };
    // 1: LOOP(a, i, rem)
    let b1 = {
        let mut b = Bdr::new(3);
        let done = b.cmpi(LtS, 1, 0);
        b.block(
            vec![i64t, i64t, i64t],
            Terminator::BrIf {
                cond: done,
                then_blk: RET,
                then_args: vec![2],
                else_blk: BODY,
                else_args: vec![0, 1, 2],
            },
        )
    };
    // 2: BODY(a, i, rem) — cur = (rem<<32)|a[i]; a[i]=cur/10; rem=cur%10
    let b2 = {
        let mut b = Bdr::new(3); // a=0, i=1, rem=2
        let off = b.bini(Shl, 1, 2);
        let addr = b.bin(Add, 0, off);
        let v = b.load32u(addr);
        let hi = b.bini(Shl, 2, 32);
        let cur = b.bin(Or, hi, v);
        let q = b.bini(DivU, cur, 10);
        let r = b.bini(RemU, cur, 10);
        b.store32(addr, q);
        let ni = b.bini(Sub, 1, 1);
        b.block(
            vec![i64t, i64t, i64t],
            Terminator::Br {
                target: LOOP,
                args: vec![0, ni, r],
            },
        )
    };
    let b3 = {
        let b = Bdr::new(1);
        b.block(vec![i64t], Terminator::Return(vec![0]))
    };
    Func {
        params: vec![i64t],
        results: vec![i64t],
        blocks: vec![b0, b1, b2, b3],
    }
}

/// `__svm_big_inc(a)` — add 1 to the big integer at `a` in place (carry-propagating).
#[allow(dead_code)]
fn synth_big_inc() -> Func {
    use BinOp::{Add, And, ShrU};
    use CmpOp::{Eq, GeS};
    let i64t = ValType::I64;
    let mask32: i64 = 0xFFFF_FFFF;
    const LOOP: u32 = 1;
    const BODY: u32 = 2;
    const RET: u32 = 3;
    // 0: entry(a) → LOOP(a, 0, carry=1)
    let b0 = {
        let mut b = Bdr::new(1);
        let one = b.k(1);
        let z = b.k(0);
        b.block(
            vec![i64t],
            Terminator::Br {
                target: LOOP,
                args: vec![0, z, one],
            },
        )
    };
    // 1: LOOP(a, i, carry) — stop at end or when carry is 0
    let b1 = {
        let mut b = Bdr::new(3);
        let end = b.cmpi(GeS, 1, BIG_NLIMBS);
        let noc = b.cmpi(Eq, 2, 0);
        let ende = b.ext(end);
        let noce = b.ext(noc);
        let stop = b.bin(BinOp::Or, ende, noce);
        let stopb = b.cmpi(CmpOp::Ne, stop, 0);
        b.block(
            vec![i64t, i64t, i64t],
            Terminator::BrIf {
                cond: stopb,
                then_blk: RET,
                then_args: vec![],
                else_blk: BODY,
                else_args: vec![0, 1, 2],
            },
        )
    };
    // 2: BODY(a, i, carry) — t = a[i]+carry; a[i]=t&mask; carry'=t>>32
    let b2 = {
        let mut b = Bdr::new(3);
        let off = b.bini(BinOp::Shl, 1, 2);
        let addr = b.bin(Add, 0, off);
        let v = b.load32u(addr);
        let t = b.bin(Add, v, 2);
        let mask = b.k(mask32);
        let new = b.bin(And, t, mask);
        b.store32(addr, new);
        let c32 = b.k(32);
        let nc = b.bin(ShrU, t, c32);
        let ni = b.bini(Add, 1, 1);
        b.block(
            vec![i64t, i64t, i64t],
            Terminator::Br {
                target: LOOP,
                args: vec![0, ni, nc],
            },
        )
    };
    let b3 = {
        let b = Bdr::new(0);
        b.block(vec![], Terminator::Return(vec![]))
    };
    Func {
        params: vec![i64t],
        results: vec![],
        blocks: vec![b0, b1, b2, b3],
    }
}

// Scratch byte offsets shared by the bignum formatters (`%e`/`%g`/big `%f`): the `dtoa_digits` region
// [0,496) (three 40-limb big integers + the nsig/E scalars), then the digit buffer, the assembled
// content, the padded output field, and the formatter's scalar locals.
const FMT_DBUF_O: i64 = 496; // up to ~528 decimal digit values
const FMT_CBUF_O: i64 = 1024; // assembled "[sign]d.dddde±dd" content
const FMT_OUT_O: i64 = 1536; // padded field (what the lowering writes)
const FMT_WIDTH_O: i64 = 2048;
const FMT_FLAGS_O: i64 = 2056;
const FMT_SIGN_O: i64 = 2064;
const FMT_CLEN_O: i64 = 2072; // content cursor → content length
const FMT_E_O: i64 = 2080;
const FMT_PREC_O: i64 = 2088;
const FMT_TOTAL_O: i64 = 2096; // padded field length
const FMT_LEAD_O: i64 = 2104; // leading-pad count (right-justify)
const FMT_P_O: i64 = 2112; // %g significant-digit count P
const FMT_SIGEND_O: i64 = 2120; // %g: content cursor just past the last significant fraction digit

/// Emit the sign byte for a float field: `'-'` if the sign bit is set, else `'+'`/`' '` for the
/// `+`/space flags, else `0` (none). `sign`/`flags` are loaded from the scratch locals at `scratch`.
fn fmt_sign_byte(b: &mut Bdr, scratch: ValIdx) -> ValIdx {
    use BinOp::{Add, And};
    use CmpOp::Ne;
    let ksa = b.k(FMT_SIGN_O);
    let sa = b.bin(Add, scratch, ksa);
    let sign = b.load64(sa);
    let kfa = b.k(FMT_FLAGS_O);
    let fa = b.bin(Add, scratch, kfa);
    let flags = b.load64(fa);
    let isneg = b.cmpi(Ne, sign, 0);
    let c2 = b.k(2);
    let plusf = b.bin(And, flags, c2);
    let hasplus = b.cmpi(Ne, plusf, 0);
    let c4 = b.k(4);
    let spacef = b.bin(And, flags, c4);
    let hasspace = b.cmpi(Ne, spacef, 0);
    let space = b.k(b' ' as i64);
    let zero = b.k(0);
    let sc1 = b.sel(hasspace, space, zero);
    let plus = b.k(b'+' as i64);
    let sc2 = b.sel(hasplus, plus, sc1);
    let minus = b.k(b'-' as i64);
    b.sel(isneg, minus, sc2)
}

/// `__svm_dtoa_sci(bits, prec, width, flags, scratch) -> i64` — format the double `bits` in `%e`
/// scientific notation (`[sign]d.ddde±dd`, `prec` fraction digits, ≥2 exponent digits) into the
/// scratch output field, returning its length; the caller writes `scratch+FMT_OUT_O .. +len`. Uses
/// the exact `dtoa_digits` engine (so correctly rounded across the whole double range). `flags` adds
/// bit3 = uppercase (`%E` ⇒ `E`/`INF`/`NAN`). Composes `dtoa_digits` by func index `dd`.
#[allow(dead_code)]
fn synth_dtoa_sci(dd: u32) -> Func {
    use BinOp::{Add, And, DivU, RemU, ShrU, Sub};
    use CmpOp::{Eq, GtS, LtS, Ne};
    let i64t = ValType::I64;

    const SPECIAL: u32 = 1;
    const FINITE: u32 = 2;
    const ASSEMBLE: u32 = 3;
    const DOT: u32 = 4;
    const FRAC_LOOP: u32 = 5;
    const FRAC_BODY: u32 = 6;
    const EXPSTART: u32 = 7;
    const EXPDIGITS: u32 = 8;
    const PAD_START: u32 = 9;
    const PAD_FILL_TEST: u32 = 10;
    const PAD_FILL_BODY: u32 = 11;
    const PAD_COPY_TEST: u32 = 12;
    const PAD_COPY_BODY: u32 = 13;
    const RET: u32 = 14;

    // 0: ENTRY(bits, prec, width, flags, scratch) — stash params; split non-finite vs finite.
    let b_entry = {
        let mut b = Bdr::new(5); // bits=0, prec=1, width=2, flags=3, scratch=4
        let kp = b.k(FMT_PREC_O);
        let pa = b.bin(Add, 4, kp);
        b.store64(pa, 1);
        let kw = b.k(FMT_WIDTH_O);
        let wa = b.bin(Add, 4, kw);
        b.store64(wa, 2);
        let kf = b.k(FMT_FLAGS_O);
        let fa = b.bin(Add, 4, kf);
        b.store64(fa, 3);
        let c63 = b.k(63);
        let sgn = b.bin(ShrU, 0, c63);
        let ks = b.k(FMT_SIGN_O);
        let sa = b.bin(Add, 4, ks);
        b.store64(sa, sgn);
        let c52 = b.k(52);
        let e0 = b.bin(ShrU, 0, c52);
        let m7ff = b.k(0x7FF);
        let exp = b.bin(And, e0, m7ff);
        let isspec = b.cmpi(Eq, exp, 0x7FF);
        b.block(
            vec![i64t, i64t, i64t, i64t, i64t],
            Terminator::BrIf {
                cond: isspec,
                then_blk: SPECIAL,
                then_args: vec![4, 0],
                else_blk: FINITE,
                else_args: vec![4, 0, 1],
            },
        )
    };

    // 1: SPECIAL(scratch, bits) — "[sign]inf" (mant==0) or "nan", uppercased for %E.
    let b_special = {
        let mut b = Bdr::new(2); // scratch=0, bits=1
        let sc = fmt_sign_byte(&mut b, 0);
        let kc = b.k(FMT_CBUF_O);
        let cb = b.bin(Add, 0, kc);
        b.store8(cb, sc);
        let hassign = b.cmpi(Ne, sc, 0);
        let one = b.k(1);
        let zero = b.k(0);
        let count = b.sel(hassign, one, zero);
        // uppercase? (flags bit3)
        let kf = b.k(FMT_FLAGS_O);
        let fa = b.bin(Add, 0, kf);
        let flags = b.load64(fa);
        let c8 = b.k(8);
        let upf = b.bin(And, flags, c8);
        let upper = b.cmpi(Ne, upf, 0);
        // letters: inf vs nan, lower vs upper
        let mmask = b.k((1i64 << 52) - 1);
        let mant = b.bin(And, 1, mmask);
        let isinf = b.cmpi(Eq, mant, 0);
        // ch0: i/I vs n/N ; ch1: n/N vs a/A ; ch2: f/F vs n/N
        let li = b.k(b'i' as i64);
        let ui = b.k(b'I' as i64);
        let i_c = b.sel(upper, ui, li);
        let ln = b.k(b'n' as i64);
        let un = b.k(b'N' as i64);
        let n_c = b.sel(upper, un, ln);
        let la = b.k(b'a' as i64);
        let ua = b.k(b'A' as i64);
        let a_c = b.sel(upper, ua, la);
        let lf = b.k(b'f' as i64);
        let uf = b.k(b'F' as i64);
        let f_c = b.sel(upper, uf, lf);
        let ch0 = b.sel(isinf, i_c, n_c);
        let ch1 = b.sel(isinf, n_c, a_c);
        let ch2 = b.sel(isinf, f_c, n_c);
        let base = b.k(FMT_CBUF_O);
        let cbase = b.bin(Add, 0, base);
        let a0 = b.bin(Add, cbase, count);
        b.store8(a0, ch0);
        let one1 = b.k(1);
        let p1 = b.bin(Add, count, one1);
        let a1 = b.bin(Add, cbase, p1);
        b.store8(a1, ch1);
        let two = b.k(2);
        let p2 = b.bin(Add, count, two);
        let a2 = b.bin(Add, cbase, p2);
        b.store8(a2, ch2);
        let three = b.k(3);
        let clen = b.bin(Add, count, three);
        let kcl = b.k(FMT_CLEN_O);
        let cla = b.bin(Add, 0, kcl);
        b.store64(cla, clen);
        b.block(
            vec![i64t, i64t],
            Terminator::Br {
                target: PAD_START,
                args: vec![0],
            },
        )
    };

    // 2: FINITE(scratch, bits, prec) — nsig = prec+1; run the digit engine; stash E.
    let b_finite = {
        let mut b = Bdr::new(3); // scratch=0, bits=1, prec=2
        let one = b.k(1);
        let nsig = b.bin(Add, 2, one);
        let e = b.call1(dd, vec![1, nsig, 0]); // dtoa_digits(bits, nsig, scratch)
        let ke = b.k(FMT_E_O);
        let ea = b.bin(Add, 0, ke);
        b.store64(ea, e);
        b.block(
            vec![i64t, i64t, i64t],
            Terminator::Br {
                target: ASSEMBLE,
                args: vec![0],
            },
        )
    };

    // 3: ASSEMBLE(scratch) — [sign] d0 ; then '.' + fraction iff prec>0.
    let b_assemble = {
        let mut b = Bdr::new(1);
        let sc = fmt_sign_byte(&mut b, 0);
        let kc = b.k(FMT_CBUF_O);
        let cb = b.bin(Add, 0, kc);
        b.store8(cb, sc);
        let hassign = b.cmpi(Ne, sc, 0);
        let one = b.k(1);
        let zero = b.k(0);
        let cur0 = b.sel(hassign, one, zero);
        // d0 = '0' + dbuf[0]
        let kd = b.k(FMT_DBUF_O);
        let dba = b.bin(Add, 0, kd);
        let d0 = b.load8u(dba);
        let z0 = b.k(b'0' as i64);
        let ch = b.bin(Add, d0, z0);
        let a0 = b.bin(Add, cb, cur0);
        b.store8(a0, ch);
        let one2 = b.k(1);
        let cur1 = b.bin(Add, cur0, one2);
        let kcl = b.k(FMT_CLEN_O);
        let cla = b.bin(Add, 0, kcl);
        b.store64(cla, cur1);
        let kp = b.k(FMT_PREC_O);
        let pa = b.bin(Add, 0, kp);
        let prec = b.load64(pa);
        let pos = b.cmpi(GtS, prec, 0);
        b.block(
            vec![i64t],
            Terminator::BrIf {
                cond: pos,
                then_blk: DOT,
                then_args: vec![0],
                else_blk: EXPSTART,
                else_args: vec![0],
            },
        )
    };

    // 4: DOT(scratch) — write '.', enter the fraction loop at j=1.
    let b_dot = {
        let mut b = Bdr::new(1);
        let kcl = b.k(FMT_CLEN_O);
        let cla = b.bin(Add, 0, kcl);
        let cur = b.load64(cla);
        let kc = b.k(FMT_CBUF_O);
        let cb = b.bin(Add, 0, kc);
        let a = b.bin(Add, cb, cur);
        let dot = b.k(b'.' as i64);
        b.store8(a, dot);
        let one = b.k(1);
        let cur2 = b.bin(Add, cur, one);
        b.store64(cla, cur2);
        let j = b.k(1);
        b.block(
            vec![i64t],
            Terminator::Br {
                target: FRAC_LOOP,
                args: vec![0, j],
            },
        )
    };

    // 5: FRAC_LOOP(scratch, j) — emit dbuf[1..=prec].
    let b_frac_loop = {
        let mut b = Bdr::new(2);
        let kp = b.k(FMT_PREC_O);
        let pa = b.bin(Add, 0, kp);
        let prec = b.load64(pa);
        let done = b.cmp(GtS, 1, prec); // j > prec
        b.block(
            vec![i64t, i64t],
            Terminator::BrIf {
                cond: done,
                then_blk: EXPSTART,
                then_args: vec![0],
                else_blk: FRAC_BODY,
                else_args: vec![0, 1],
            },
        )
    };

    // 6: FRAC_BODY(scratch, j) — cbuf[cur++] = '0' + dbuf[j].
    let b_frac_body = {
        let mut b = Bdr::new(2);
        let kd = b.k(FMT_DBUF_O);
        let dba = b.bin(Add, 0, kd);
        let da = b.bin(Add, dba, 1);
        let d = b.load8u(da);
        let z = b.k(b'0' as i64);
        let ch = b.bin(Add, d, z);
        let kcl = b.k(FMT_CLEN_O);
        let cla = b.bin(Add, 0, kcl);
        let cur = b.load64(cla);
        let kc = b.k(FMT_CBUF_O);
        let cb = b.bin(Add, 0, kc);
        let a = b.bin(Add, cb, cur);
        b.store8(a, ch);
        let one = b.k(1);
        let cur2 = b.bin(Add, cur, one);
        b.store64(cla, cur2);
        let j2 = b.bin(Add, 1, one);
        b.block(
            vec![i64t, i64t],
            Terminator::Br {
                target: FRAC_LOOP,
                args: vec![0, j2],
            },
        )
    };

    // 7: EXPSTART(scratch) — write 'e'/'E', the exponent sign, and set up |E|.
    let b_expstart = {
        let mut b = Bdr::new(1);
        let kcl = b.k(FMT_CLEN_O);
        let cla = b.bin(Add, 0, kcl);
        let cur = b.load64(cla);
        let kc = b.k(FMT_CBUF_O);
        let cb = b.bin(Add, 0, kc);
        let a = b.bin(Add, cb, cur);
        // 'e' or 'E'
        let kf = b.k(FMT_FLAGS_O);
        let fa = b.bin(Add, 0, kf);
        let flags = b.load64(fa);
        let c8 = b.k(8);
        let upf = b.bin(And, flags, c8);
        let upper = b.cmpi(Ne, upf, 0);
        let le = b.k(b'e' as i64);
        let ue = b.k(b'E' as i64);
        let echar = b.sel(upper, ue, le);
        b.store8(a, echar);
        let one = b.k(1);
        let cur2 = b.bin(Add, cur, one);
        // exponent sign
        let ke = b.k(FMT_E_O);
        let ea = b.bin(Add, 0, ke);
        let e = b.load64(ea);
        let eneg = b.cmpi(LtS, e, 0);
        let plus = b.k(b'+' as i64);
        let minus = b.k(b'-' as i64);
        let esign = b.sel(eneg, minus, plus);
        let a2 = b.bin(Add, cb, cur2);
        b.store8(a2, esign);
        let cur3 = b.bin(Add, cur2, one);
        b.store64(cla, cur3);
        let zero = b.k(0);
        let nege = b.bin(Sub, zero, e);
        let abs_e = b.sel(eneg, nege, e);
        b.block(
            vec![i64t],
            Terminator::Br {
                target: EXPDIGITS,
                args: vec![0, abs_e],
            },
        )
    };

    // 8: EXPDIGITS(scratch, absE) — |E| as ≥2 digits (3 if ≥100), then pad.
    let b_expdigits = {
        let mut b = Bdr::new(2); // scratch=0, absE=1
        let c100 = b.k(100);
        let h = b.bin(DivU, 1, c100);
        let c10 = b.k(10);
        let tens0 = b.bin(DivU, 1, c10);
        let t = b.bin(RemU, tens0, c10);
        let o = b.bin(RemU, 1, c10);
        let has3 = b.cmpi(GtS, h, 0);
        let kcl = b.k(FMT_CLEN_O);
        let cla = b.bin(Add, 0, kcl);
        let cur = b.load64(cla);
        let kc = b.k(FMT_CBUF_O);
        let cb = b.bin(Add, 0, kc);
        let z = b.k(b'0' as i64);
        // hundreds digit (overwritten by tens when has3 is false, since cur doesn't advance)
        let hch = b.bin(Add, h, z);
        let ah = b.bin(Add, cb, cur);
        b.store8(ah, hch);
        let one = b.k(1);
        let curp = b.bin(Add, cur, one);
        let cur_t = b.sel(has3, curp, cur);
        let tch = b.bin(Add, t, z);
        let at = b.bin(Add, cb, cur_t);
        b.store8(at, tch);
        let cur_o = b.bin(Add, cur_t, one);
        let och = b.bin(Add, o, z);
        let ao = b.bin(Add, cb, cur_o);
        b.store8(ao, och);
        let cur_end = b.bin(Add, cur_o, one);
        b.store64(cla, cur_end);
        b.block(
            vec![i64t, i64t],
            Terminator::Br {
                target: PAD_START,
                args: vec![0],
            },
        )
    };

    // 9: PAD_START(scratch) — total = max(clen,width); lead = left?0:pad; fill the field.
    let b_pad_start = {
        let mut b = Bdr::new(1);
        let kcl = b.k(FMT_CLEN_O);
        let cla = b.bin(Add, 0, kcl);
        let clen = b.load64(cla);
        let kw = b.k(FMT_WIDTH_O);
        let wa = b.bin(Add, 0, kw);
        let width = b.load64(wa);
        let wgt = b.cmp(GtS, width, clen);
        let total = b.sel(wgt, width, clen);
        let pad = b.bin(Sub, total, clen);
        let kf = b.k(FMT_FLAGS_O);
        let fa = b.bin(Add, 0, kf);
        let flags = b.load64(fa);
        let c1 = b.k(1);
        let leftf = b.bin(And, flags, c1);
        let isleft = b.cmpi(Ne, leftf, 0);
        let zero = b.k(0);
        let lead = b.sel(isleft, zero, pad);
        let kt = b.k(FMT_TOTAL_O);
        let ta = b.bin(Add, 0, kt);
        b.store64(ta, total);
        let kl = b.k(FMT_LEAD_O);
        let lla = b.bin(Add, 0, kl);
        b.store64(lla, lead);
        let z = b.k(0);
        b.block(
            vec![i64t],
            Terminator::Br {
                target: PAD_FILL_TEST,
                args: vec![0, z],
            },
        )
    };

    // 10: PAD_FILL_TEST(scratch, j) — fill out[0..total] with spaces.
    let b_pad_fill_test = {
        let mut b = Bdr::new(2);
        let kt = b.k(FMT_TOTAL_O);
        let ta = b.bin(Add, 0, kt);
        let total = b.load64(ta);
        let go = b.cmp(LtS, 1, total);
        let z = b.k(0);
        b.block(
            vec![i64t, i64t],
            Terminator::BrIf {
                cond: go,
                then_blk: PAD_FILL_BODY,
                then_args: vec![0, 1],
                else_blk: PAD_COPY_TEST,
                else_args: vec![0, z],
            },
        )
    };

    // 11: PAD_FILL_BODY(scratch, j) — out[j] = ' '.
    let b_pad_fill_body = {
        let mut b = Bdr::new(2);
        let ko = b.k(FMT_OUT_O);
        let ob = b.bin(Add, 0, ko);
        let a = b.bin(Add, ob, 1);
        let sp = b.k(b' ' as i64);
        b.store8(a, sp);
        let one = b.k(1);
        let nj = b.bin(Add, 1, one);
        b.block(
            vec![i64t, i64t],
            Terminator::Br {
                target: PAD_FILL_TEST,
                args: vec![0, nj],
            },
        )
    };

    // 12: PAD_COPY_TEST(scratch, k) — copy content[0..clen] into out[lead..].
    let b_pad_copy_test = {
        let mut b = Bdr::new(2);
        let kcl = b.k(FMT_CLEN_O);
        let cla = b.bin(Add, 0, kcl);
        let clen = b.load64(cla);
        let go = b.cmp(LtS, 1, clen);
        b.block(
            vec![i64t, i64t],
            Terminator::BrIf {
                cond: go,
                then_blk: PAD_COPY_BODY,
                then_args: vec![0, 1],
                else_blk: RET,
                else_args: vec![0],
            },
        )
    };

    // 13: PAD_COPY_BODY(scratch, k) — out[lead+k] = content[k].
    let b_pad_copy_body = {
        let mut b = Bdr::new(2);
        let kc = b.k(FMT_CBUF_O);
        let cb = b.bin(Add, 0, kc);
        let ca = b.bin(Add, cb, 1);
        let ch = b.load8u(ca);
        let kl = b.k(FMT_LEAD_O);
        let lla = b.bin(Add, 0, kl);
        let lead = b.load64(lla);
        let ko = b.k(FMT_OUT_O);
        let ob = b.bin(Add, 0, ko);
        let off = b.bin(Add, lead, 1);
        let oa = b.bin(Add, ob, off);
        b.store8(oa, ch);
        let one = b.k(1);
        let nk = b.bin(Add, 1, one);
        b.block(
            vec![i64t, i64t],
            Terminator::Br {
                target: PAD_COPY_TEST,
                args: vec![0, nk],
            },
        )
    };

    // 14: RET(scratch) — return the padded field length.
    let b_ret = {
        let mut b = Bdr::new(1);
        let kt = b.k(FMT_TOTAL_O);
        let ta = b.bin(Add, 0, kt);
        let total = b.load64(ta);
        b.block(vec![i64t], Terminator::Return(vec![total]))
    };

    Func {
        params: vec![i64t, i64t, i64t, i64t, i64t],
        results: vec![i64t],
        blocks: vec![
            b_entry,
            b_special,
            b_finite,
            b_assemble,
            b_dot,
            b_frac_loop,
            b_frac_body,
            b_expstart,
            b_expdigits,
            b_pad_start,
            b_pad_fill_test,
            b_pad_fill_body,
            b_pad_copy_test,
            b_pad_copy_body,
            b_ret,
        ],
    }
}

/// `__svm_dtoa_gen(bits, prec, width, flags, scratch) -> i64` — format the double `bits` in `%g`
/// notation into the scratch output field, returning its length. `%g` rounds to `P` significant
/// digits (`P = max(prec,1)` — the `dtoa_digits` engine's native mode), then chooses `%e` layout when
/// the decimal exponent `E < -4 || E >= P`, else `%f` layout, and strips trailing zeros from the
/// fraction (and a bare `.`) unless the `#` flag (bit4) is set. `flags` bit3 = uppercase (`%G`).
#[allow(dead_code)]
fn synth_dtoa_gen(dd: u32) -> Func {
    use BinOp::{Add, And, DivU, RemU, ShrU, Sub};
    use CmpOp::{Eq, GeS, GtS, LtS, Ne};
    let i64t = ValType::I64;

    const SPECIAL: u32 = 1;
    const FINITE: u32 = 2;
    const EM_D0: u32 = 3;
    const EM_FRAC_TEST: u32 = 4;
    const EM_FRAC_BODY: u32 = 5;
    const EM_STRIP: u32 = 6;
    const EXPSTART: u32 = 7;
    const EXPDIGITS: u32 = 8;
    const FM_INT: u32 = 9;
    const FM_INT_TEST: u32 = 10;
    const FM_INT_BODY: u32 = 11;
    const FM_DOT: u32 = 12;
    const FM_FRAC_TEST: u32 = 13;
    const FM_FRAC_BODY: u32 = 14;
    const FM_STRIP: u32 = 15;
    const PAD_START: u32 = 16;
    const PAD_FILL_TEST: u32 = 17;
    const PAD_FILL_BODY: u32 = 18;
    const PAD_COPY_TEST: u32 = 19;
    const PAD_COPY_BODY: u32 = 20;
    const RET: u32 = 21;

    // Append byte `ch` at the content cursor (FMT_CLEN_O), advancing it; if `ch != '0'+0` is a
    // significant fraction digit the caller separately bumps FMT_SIGEND_O. Returns nothing.
    let emit = |b: &mut Bdr, scratch: ValIdx, ch: ValIdx| {
        let kcl = b.k(FMT_CLEN_O);
        let cla = b.bin(Add, scratch, kcl);
        let cur = b.load64(cla);
        let kc = b.k(FMT_CBUF_O);
        let cb = b.bin(Add, scratch, kc);
        let a = b.bin(Add, cb, cur);
        b.store8(a, ch);
        let one = b.k(1);
        let cur2 = b.bin(Add, cur, one);
        b.store64(cla, cur2);
    };

    // 0: ENTRY(bits, prec, width, flags, scratch) — stash params (P = max(prec,1)); split special.
    let b_entry = {
        let mut b = Bdr::new(5);
        let one = b.k(1);
        let pgt = b.cmpi(GtS, 1, 1); // prec > 1 ? (prec >= 1 already after sel below)
        let pp = b.sel(pgt, 1, one); // P = max(prec, 1)
        let kp = b.k(FMT_P_O);
        let pa = b.bin(Add, 4, kp);
        b.store64(pa, pp);
        let kw = b.k(FMT_WIDTH_O);
        let wa = b.bin(Add, 4, kw);
        b.store64(wa, 2);
        let kf = b.k(FMT_FLAGS_O);
        let fa = b.bin(Add, 4, kf);
        b.store64(fa, 3);
        let c63 = b.k(63);
        let sgn = b.bin(ShrU, 0, c63);
        let ks = b.k(FMT_SIGN_O);
        let sa = b.bin(Add, 4, ks);
        b.store64(sa, sgn);
        let c52 = b.k(52);
        let e0 = b.bin(ShrU, 0, c52);
        let m7ff = b.k(0x7FF);
        let exp = b.bin(And, e0, m7ff);
        let isspec = b.cmpi(Eq, exp, 0x7FF);
        b.block(
            vec![i64t, i64t, i64t, i64t, i64t],
            Terminator::BrIf {
                cond: isspec,
                then_blk: SPECIAL,
                then_args: vec![4, 0],
                else_blk: FINITE,
                else_args: vec![4, 0],
            },
        )
    };

    // 1: SPECIAL(scratch, bits) — "[sign]inf"/"nan" (uppercased for %G), then pad.
    let b_special = {
        let mut b = Bdr::new(2);
        let sc = fmt_sign_byte(&mut b, 0);
        let kc = b.k(FMT_CBUF_O);
        let cb = b.bin(Add, 0, kc);
        b.store8(cb, sc);
        let hassign = b.cmpi(Ne, sc, 0);
        let one = b.k(1);
        let zero = b.k(0);
        let count = b.sel(hassign, one, zero);
        let kf = b.k(FMT_FLAGS_O);
        let fa = b.bin(Add, 0, kf);
        let flags = b.load64(fa);
        let c8 = b.k(8);
        let upf = b.bin(And, flags, c8);
        let upper = b.cmpi(Ne, upf, 0);
        let mmask = b.k((1i64 << 52) - 1);
        let mant = b.bin(And, 1, mmask);
        let isinf = b.cmpi(Eq, mant, 0);
        let li = b.k(b'i' as i64);
        let ui = b.k(b'I' as i64);
        let i_c = b.sel(upper, ui, li);
        let ln = b.k(b'n' as i64);
        let un = b.k(b'N' as i64);
        let n_c = b.sel(upper, un, ln);
        let la = b.k(b'a' as i64);
        let ua = b.k(b'A' as i64);
        let a_c = b.sel(upper, ua, la);
        let lf = b.k(b'f' as i64);
        let uf = b.k(b'F' as i64);
        let f_c = b.sel(upper, uf, lf);
        let ch0 = b.sel(isinf, i_c, n_c);
        let ch1 = b.sel(isinf, n_c, a_c);
        let ch2 = b.sel(isinf, f_c, n_c);
        let cbase = b.bini(Add, 0, FMT_CBUF_O);
        let a0 = b.bin(Add, cbase, count);
        b.store8(a0, ch0);
        let p1 = b.bini(Add, count, 1);
        let a1 = b.bin(Add, cbase, p1);
        b.store8(a1, ch1);
        let p2 = b.bini(Add, count, 2);
        let a2 = b.bin(Add, cbase, p2);
        b.store8(a2, ch2);
        let clen = b.bini(Add, count, 3);
        let cla = b.bini(Add, 0, FMT_CLEN_O);
        b.store64(cla, clen);
        b.block(
            vec![i64t, i64t],
            Terminator::Br {
                target: PAD_START,
                args: vec![0],
            },
        )
    };

    // 2: FINITE(scratch, bits) — P sig digits + E from the engine; seed sign/cursor; pick e vs f.
    let b_finite = {
        let mut b = Bdr::new(2); // scratch=0, bits=1
        let pa = b.bini(Add, 0, FMT_P_O);
        let pp = b.load64(pa);
        let e = b.call1(dd, vec![1, pp, 0]);
        let ea = b.bini(Add, 0, FMT_E_O);
        b.store64(ea, e);
        // sign byte + initial cursor
        let sc = fmt_sign_byte(&mut b, 0);
        let cb = b.bini(Add, 0, FMT_CBUF_O);
        b.store8(cb, sc);
        let hassign = b.cmpi(Ne, sc, 0);
        let one1 = b.k(1);
        let zero1 = b.k(0);
        let cur0 = b.sel(hassign, one1, zero1);
        let cla = b.bini(Add, 0, FMT_CLEN_O);
        b.store64(cla, cur0);
        // use_exp = E < -4 || E >= P
        let lt = b.cmpi(LtS, e, -4);
        let ge = b.cmp(GeS, e, pp);
        let lt64 = b.ext(lt);
        let ge64 = b.ext(ge);
        let ue = b.bin(BinOp::Or, lt64, ge64);
        let useexp = b.cmpi(Ne, ue, 0);
        b.block(
            vec![i64t, i64t],
            Terminator::BrIf {
                cond: useexp,
                then_blk: EM_D0,
                then_args: vec![0],
                else_blk: FM_INT,
                else_args: vec![0],
            },
        )
    };

    // 3: EM_D0(scratch) — e-mode: d0, '.', seed sigend (= position before '.').
    let b_em_d0 = {
        let mut b = Bdr::new(1);
        let dba = b.bini(Add, 0, FMT_DBUF_O);
        let d0 = b.load8u(dba);
        let z = b.k(b'0' as i64);
        let ch = b.bin(Add, d0, z);
        emit(&mut b, 0, ch);
        // sigend = cur now (just past d0, before '.')
        let cla = b.bini(Add, 0, FMT_CLEN_O);
        let cur = b.load64(cla);
        let sea = b.bini(Add, 0, FMT_SIGEND_O);
        b.store64(sea, cur);
        let dot = b.k(b'.' as i64);
        emit(&mut b, 0, dot);
        let one = b.k(1);
        b.block(
            vec![i64t],
            Terminator::Br {
                target: EM_FRAC_TEST,
                args: vec![0, one],
            },
        )
    };

    // 5: EM_FRAC_TEST(scratch, k) — emit dbuf[1..P-1].
    let b_em_frac_test = {
        let mut b = Bdr::new(2);
        let pa = b.bini(Add, 0, FMT_P_O);
        let pp = b.load64(pa);
        let done = b.cmp(GeS, 1, pp); // k >= P
        b.block(
            vec![i64t, i64t],
            Terminator::BrIf {
                cond: done,
                then_blk: EM_STRIP,
                then_args: vec![0],
                else_blk: EM_FRAC_BODY,
                else_args: vec![0, 1],
            },
        )
    };

    // 6: EM_FRAC_BODY(scratch, k) — emit '0'+dbuf[k]; if nonzero, sigend = cur.
    let b_em_frac_body = {
        let mut b = Bdr::new(2);
        let dba = b.bini(Add, 0, FMT_DBUF_O);
        let da = b.bin(Add, dba, 1);
        let d = b.load8u(da);
        let z = b.k(b'0' as i64);
        let ch = b.bin(Add, d, z);
        emit(&mut b, 0, ch);
        let nz = b.cmpi(Ne, d, 0);
        let cla = b.bini(Add, 0, FMT_CLEN_O);
        let cur = b.load64(cla);
        let sea = b.bini(Add, 0, FMT_SIGEND_O);
        let oldse = b.load64(sea);
        let newse = b.sel(nz, cur, oldse);
        b.store64(sea, newse);
        let k2 = b.bini(Add, 1, 1);
        b.block(
            vec![i64t, i64t],
            Terminator::Br {
                target: EM_FRAC_TEST,
                args: vec![0, k2],
            },
        )
    };

    // 7: EM_STRIP(scratch) — drop trailing zeros (cursor = alt ? cur : sigend), then exponent.
    let b_em_strip = {
        let mut b = Bdr::new(1);
        let fa = b.bini(Add, 0, FMT_FLAGS_O);
        let flags = b.load64(fa);
        let altf = b.bini(And, flags, 16);
        let isalt = b.cmpi(Ne, altf, 0);
        let cla = b.bini(Add, 0, FMT_CLEN_O);
        let cur = b.load64(cla);
        let sea = b.bini(Add, 0, FMT_SIGEND_O);
        let se = b.load64(sea);
        let newcur = b.sel(isalt, cur, se);
        b.store64(cla, newcur);
        b.block(
            vec![i64t],
            Terminator::Br {
                target: EXPSTART,
                args: vec![0],
            },
        )
    };

    // 8: EXPSTART(scratch) — 'e'/'E', exponent sign, set up |E|.
    let b_expstart = {
        let mut b = Bdr::new(1);
        let fa = b.bini(Add, 0, FMT_FLAGS_O);
        let flags = b.load64(fa);
        let upf = b.bini(And, flags, 8);
        let upper = b.cmpi(Ne, upf, 0);
        let le = b.k(b'e' as i64);
        let ue = b.k(b'E' as i64);
        let echar = b.sel(upper, ue, le);
        emit(&mut b, 0, echar);
        let ea = b.bini(Add, 0, FMT_E_O);
        let e = b.load64(ea);
        let eneg = b.cmpi(LtS, e, 0);
        let minus = b.k(b'-' as i64);
        let plus = b.k(b'+' as i64);
        let esign = b.sel(eneg, minus, plus);
        emit(&mut b, 0, esign);
        let zero = b.k(0);
        let nege = b.bin(Sub, zero, e);
        let abse = b.sel(eneg, nege, e);
        b.block(
            vec![i64t],
            Terminator::Br {
                target: EXPDIGITS,
                args: vec![0, abse],
            },
        )
    };

    // 9: EXPDIGITS(scratch, absE) — ≥2 digits (3 if ≥100), then pad.
    let b_expdigits = {
        let mut b = Bdr::new(2);
        let h = b.bini(DivU, 1, 100);
        let tens0 = b.bini(DivU, 1, 10);
        let t = b.bini(RemU, tens0, 10);
        let o = b.bini(RemU, 1, 10);
        let has3 = b.cmpi(GtS, h, 0);
        let cla = b.bini(Add, 0, FMT_CLEN_O);
        let cur = b.load64(cla);
        let cb = b.bini(Add, 0, FMT_CBUF_O);
        let z = b.k(b'0' as i64);
        let hch = b.bin(Add, h, z);
        let ah = b.bin(Add, cb, cur);
        b.store8(ah, hch);
        let curp = b.bini(Add, cur, 1);
        let cur_t = b.sel(has3, curp, cur);
        let tch = b.bin(Add, t, z);
        let at = b.bin(Add, cb, cur_t);
        b.store8(at, tch);
        let cur_o = b.bini(Add, cur_t, 1);
        let och = b.bin(Add, o, z);
        let ao = b.bin(Add, cb, cur_o);
        b.store8(ao, och);
        let cur_end = b.bini(Add, cur_o, 1);
        b.store64(cla, cur_end);
        b.block(
            vec![i64t, i64t],
            Terminator::Br {
                target: PAD_START,
                args: vec![0],
            },
        )
    };

    // 10: FM_INT(scratch) — f-mode integer part: d[0..E] if E≥0, else '0'.
    let b_fm_int = {
        let mut b = Bdr::new(1);
        let ea = b.bini(Add, 0, FMT_E_O);
        let e = b.load64(ea);
        let eneg = b.cmpi(LtS, e, 0);
        let z = b.k(0);
        b.block(
            vec![i64t],
            Terminator::BrIf {
                cond: eneg,
                then_blk: FM_DOT, // integer "0" handled in FM_DOT's pre-step
                then_args: vec![0],
                else_blk: FM_INT_TEST,
                else_args: vec![0, z], // j = 0
            },
        )
    };

    // 11: FM_INT_TEST(scratch, j) — emit d[j] for j in 0..=E.
    let b_fm_int_test = {
        let mut b = Bdr::new(2);
        let ea = b.bini(Add, 0, FMT_E_O);
        let e = b.load64(ea);
        let done = b.cmp(GtS, 1, e); // j > E
        b.block(
            vec![i64t, i64t],
            Terminator::BrIf {
                cond: done,
                then_blk: FM_DOT,
                then_args: vec![0],
                else_blk: FM_INT_BODY,
                else_args: vec![0, 1],
            },
        )
    };

    // 12: FM_INT_BODY(scratch, j) — emit '0'+dbuf[j].
    let b_fm_int_body = {
        let mut b = Bdr::new(2);
        let dba = b.bini(Add, 0, FMT_DBUF_O);
        let da = b.bin(Add, dba, 1);
        let d = b.load8u(da);
        let ch = b.bini(Add, d, b'0' as i64);
        emit(&mut b, 0, ch);
        let j2 = b.bini(Add, 1, 1);
        b.block(
            vec![i64t, i64t],
            Terminator::Br {
                target: FM_INT_TEST,
                args: vec![0, j2],
            },
        )
    };

    // 13: FM_DOT(scratch) — write integer '0' when E<0, then '.', seed sigend.
    let b_fm_dot = {
        let mut b = Bdr::new(1);
        // If nothing was written yet (E<0), the cursor is still at the sign end ⇒ emit a leading '0'.
        let ea = b.bini(Add, 0, FMT_E_O);
        let e = b.load64(ea);
        let eneg = b.cmpi(LtS, e, 0);
        let dba = b.bini(Add, 0, FMT_CBUF_O);
        let cla = b.bini(Add, 0, FMT_CLEN_O);
        let cur = b.load64(cla);
        // store '0' at cur (only consumed when E<0); advance cursor by eneg?1:0
        let zc = b.k(b'0' as i64);
        let a = b.bin(Add, dba, cur);
        b.store8(a, zc);
        let curp = b.bini(Add, cur, 1);
        let cur2 = b.sel(eneg, curp, cur);
        b.store64(cla, cur2);
        // '.' and sigend = position before '.'
        let sea = b.bini(Add, 0, FMT_SIGEND_O);
        b.store64(sea, cur2);
        let dot = b.k(b'.' as i64);
        emit(&mut b, 0, dot);
        let z = b.k(0);
        b.block(
            vec![i64t],
            Terminator::Br {
                target: FM_FRAC_TEST,
                args: vec![0, z],
            },
        )
    };

    // 14: FM_FRAC_TEST(scratch, k) — L = P-1-E fraction places.
    let b_fm_frac_test = {
        let mut b = Bdr::new(2);
        let pa = b.bini(Add, 0, FMT_P_O);
        let pp = b.load64(pa);
        let ea = b.bini(Add, 0, FMT_E_O);
        let e = b.load64(ea);
        let pm1 = b.bini(Sub, pp, 1);
        let lcount = b.bin(Sub, pm1, e); // P-1-E
        let done = b.cmp(GeS, 1, lcount); // k >= L
        b.block(
            vec![i64t, i64t],
            Terminator::BrIf {
                cond: done,
                then_blk: FM_STRIP,
                then_args: vec![0],
                else_blk: FM_FRAC_BODY,
                else_args: vec![0, 1],
            },
        )
    };

    // 15: FM_FRAC_BODY(scratch, k) — idx=E+1+k; digit = (idx≥0) ? dbuf[idx] : 0; track sigend.
    let b_fm_frac_body = {
        let mut b = Bdr::new(2);
        let ea = b.bini(Add, 0, FMT_E_O);
        let e = b.load64(ea);
        let idx = b.bin(Add, e, 1); // E + k ... then +1 below
        let idx1 = b.bini(Add, idx, 1); // E + 1 + k
        let valid = b.cmpi(GeS, idx1, 0);
        // load dbuf[idx1] when valid (clamp address to dbuf when not, value discarded)
        let zero = b.k(0);
        let safeidx = b.sel(valid, idx1, zero);
        let dba = b.bini(Add, 0, FMT_DBUF_O);
        let da = b.bin(Add, dba, safeidx);
        let raw = b.load8u(da);
        let d = b.sel(valid, raw, zero);
        let ch = b.bini(Add, d, b'0' as i64);
        emit(&mut b, 0, ch);
        let nz = b.cmpi(Ne, d, 0);
        let cla = b.bini(Add, 0, FMT_CLEN_O);
        let cur = b.load64(cla);
        let sea = b.bini(Add, 0, FMT_SIGEND_O);
        let oldse = b.load64(sea);
        let newse = b.sel(nz, cur, oldse);
        b.store64(sea, newse);
        let k2 = b.bini(Add, 1, 1);
        b.block(
            vec![i64t, i64t],
            Terminator::Br {
                target: FM_FRAC_TEST,
                args: vec![0, k2],
            },
        )
    };

    // 16: FM_STRIP(scratch) — drop trailing zeros, then pad.
    let b_fm_strip = {
        let mut b = Bdr::new(1);
        let fa = b.bini(Add, 0, FMT_FLAGS_O);
        let flags = b.load64(fa);
        let altf = b.bini(And, flags, 16);
        let isalt = b.cmpi(Ne, altf, 0);
        let cla = b.bini(Add, 0, FMT_CLEN_O);
        let cur = b.load64(cla);
        let sea = b.bini(Add, 0, FMT_SIGEND_O);
        let se = b.load64(sea);
        let newcur = b.sel(isalt, cur, se);
        b.store64(cla, newcur);
        b.block(
            vec![i64t],
            Terminator::Br {
                target: PAD_START,
                args: vec![0],
            },
        )
    };

    // 17: PAD_START(scratch) — total = max(clen,width); lead = left?0:pad; fill.
    let b_pad_start = {
        let mut b = Bdr::new(1);
        let cla = b.bini(Add, 0, FMT_CLEN_O);
        let clen = b.load64(cla);
        let wa = b.bini(Add, 0, FMT_WIDTH_O);
        let width = b.load64(wa);
        let wgt = b.cmp(GtS, width, clen);
        let total = b.sel(wgt, width, clen);
        let pad = b.bin(Sub, total, clen);
        let fa = b.bini(Add, 0, FMT_FLAGS_O);
        let flags = b.load64(fa);
        let leftf = b.bini(And, flags, 1);
        let isleft = b.cmpi(Ne, leftf, 0);
        let zero = b.k(0);
        let lead = b.sel(isleft, zero, pad);
        let ta = b.bini(Add, 0, FMT_TOTAL_O);
        b.store64(ta, total);
        let lla = b.bini(Add, 0, FMT_LEAD_O);
        b.store64(lla, lead);
        let z = b.k(0);
        b.block(
            vec![i64t],
            Terminator::Br {
                target: PAD_FILL_TEST,
                args: vec![0, z],
            },
        )
    };

    // 18: PAD_FILL_TEST(scratch, j).
    let b_pad_fill_test = {
        let mut b = Bdr::new(2);
        let ta = b.bini(Add, 0, FMT_TOTAL_O);
        let total = b.load64(ta);
        let go = b.cmp(LtS, 1, total);
        let z = b.k(0);
        b.block(
            vec![i64t, i64t],
            Terminator::BrIf {
                cond: go,
                then_blk: PAD_FILL_BODY,
                then_args: vec![0, 1],
                else_blk: PAD_COPY_TEST,
                else_args: vec![0, z],
            },
        )
    };

    // 19: PAD_FILL_BODY(scratch, j).
    let b_pad_fill_body = {
        let mut b = Bdr::new(2);
        let ob = b.bini(Add, 0, FMT_OUT_O);
        let a = b.bin(Add, ob, 1);
        let sp = b.k(b' ' as i64);
        b.store8(a, sp);
        let nj = b.bini(Add, 1, 1);
        b.block(
            vec![i64t, i64t],
            Terminator::Br {
                target: PAD_FILL_TEST,
                args: vec![0, nj],
            },
        )
    };

    // 20: PAD_COPY_TEST(scratch, k).
    let b_pad_copy_test = {
        let mut b = Bdr::new(2);
        let cla = b.bini(Add, 0, FMT_CLEN_O);
        let clen = b.load64(cla);
        let go = b.cmp(LtS, 1, clen);
        b.block(
            vec![i64t, i64t],
            Terminator::BrIf {
                cond: go,
                then_blk: PAD_COPY_BODY,
                then_args: vec![0, 1],
                else_blk: RET,
                else_args: vec![0],
            },
        )
    };

    // 21: PAD_COPY_BODY(scratch, k).
    let b_pad_copy_body = {
        let mut b = Bdr::new(2);
        let cb = b.bini(Add, 0, FMT_CBUF_O);
        let ca = b.bin(Add, cb, 1);
        let ch = b.load8u(ca);
        let lla = b.bini(Add, 0, FMT_LEAD_O);
        let lead = b.load64(lla);
        let ob = b.bini(Add, 0, FMT_OUT_O);
        let off = b.bin(Add, lead, 1);
        let oa = b.bin(Add, ob, off);
        b.store8(oa, ch);
        let nk = b.bini(Add, 1, 1);
        b.block(
            vec![i64t, i64t],
            Terminator::Br {
                target: PAD_COPY_TEST,
                args: vec![0, nk],
            },
        )
    };

    // 22: RET(scratch) — return the padded field length.
    let b_ret = {
        let mut b = Bdr::new(1);
        let ta = b.bini(Add, 0, FMT_TOTAL_O);
        let total = b.load64(ta);
        b.block(vec![i64t], Terminator::Return(vec![total]))
    };

    Func {
        params: vec![i64t, i64t, i64t, i64t, i64t],
        results: vec![i64t],
        blocks: vec![
            b_entry,
            b_special,
            b_finite,
            b_em_d0,
            b_em_frac_test,
            b_em_frac_body,
            b_em_strip,
            b_expstart,
            b_expdigits,
            b_fm_int,
            b_fm_int_test,
            b_fm_int_body,
            b_fm_dot,
            b_fm_frac_test,
            b_fm_frac_body,
            b_fm_strip,
            b_pad_start,
            b_pad_fill_test,
            b_pad_fill_body,
            b_pad_copy_test,
            b_pad_copy_body,
            b_ret,
        ],
    }
}

/// `__svm_dtoa_fix_big(bits, prec, width, flags, scratch) -> i64` — fixed-notation `%f` via exact
/// big-integer arithmetic (no magnitude ceiling), returning the padded field length. Computes the
/// integer `N = round(value·10^prec)` the same way the 128-bit helper does but with a 40-limb big
/// integer: `A = f·5^prec`, then `A << (e+prec)` (exact) or a round-half-to-even right shift, then
/// `÷10` digit extraction and the `[-]ddd.fff` assembly. Correct for all magnitudes, tiny values, and
/// ties. Composes the `big_*` primitives by func index. `flags` bit3 = uppercase (`%F` ⇒ `INF`/`NAN`).
#[allow(dead_code)]
fn synth_dtoa_fix_big(
    zero: u32,
    mul: u32,
    shl: u32,
    shr1: u32,
    inc: u32,
    divmod: u32,
    iszero: u32,
) -> Func {
    use BinOp::{Add, And, Or, ShrU, Sub};
    use CmpOp::{Eq, GeS, GtS, LeS, LtS, Ne};
    let i64t = ValType::I64;
    let a_o = 0i64; // the big integer A / N
    let dcnt_o = FMT_SIGEND_O; // digit count D
    let s_o = FMT_E_O; // shift exponent s = e+prec

    const SPECIAL: u32 = 1;
    const FINITE: u32 = 2;
    const MUL5_TEST: u32 = 3;
    const MUL5_BODY: u32 = 4;
    const SHIFT: u32 = 5;
    const SHL_DO: u32 = 6;
    const SHR_SETUP: u32 = 7;
    const SHR_LOOP: u32 = 8;
    const SHR_BODY: u32 = 9;
    const SHR_ROUND: u32 = 10;
    const SHR_INC: u32 = 11;
    const DEC_INIT: u32 = 12;
    const DEC_BODY: u32 = 13;
    const DEC_TEST: u32 = 14;
    const ASM_SIGN: u32 = 15;
    const ASM_INT: u32 = 16;
    const ASM_INT_ZERO: u32 = 17;
    const ASM_INT_TEST: u32 = 18;
    const ASM_INT_BODY: u32 = 19;
    const ASM_DOT: u32 = 20;
    const ASM_FRAC_TEST: u32 = 21;
    const ASM_FRAC_BODY: u32 = 22;
    const PAD_START: u32 = 23;
    const PAD_FILL_TEST: u32 = 24;
    const PAD_FILL_BODY: u32 = 25;
    const PAD_COPY_TEST: u32 = 26;
    const PAD_COPY_BODY: u32 = 27;
    const RET: u32 = 28;

    // append `ch` at cursor (FMT_CLEN_O), advancing it.
    let emit = |b: &mut Bdr, scratch: ValIdx, ch: ValIdx| {
        let kcl = b.k(FMT_CLEN_O);
        let cla = b.bin(Add, scratch, kcl);
        let cur = b.load64(cla);
        let kc = b.k(FMT_CBUF_O);
        let cb = b.bin(Add, scratch, kc);
        let a = b.bin(Add, cb, cur);
        b.store8(a, ch);
        let cur2 = b.bini(Add, cur, 1);
        b.store64(cla, cur2);
    };

    // 0: ENTRY(bits, prec, width, flags, scratch) — decompose; stash; split special/finite.
    let b_entry = {
        let mut b = Bdr::new(5); // bits=0, prec=1, width=2, flags=3, scratch=4
        let pa = b.bini(Add, 4, FMT_PREC_O);
        b.store64(pa, 1);
        let wa = b.bini(Add, 4, FMT_WIDTH_O);
        b.store64(wa, 2);
        let fa = b.bini(Add, 4, FMT_FLAGS_O);
        b.store64(fa, 3);
        let sgn = b.bini(ShrU, 0, 63);
        let sa = b.bini(Add, 4, FMT_SIGN_O);
        b.store64(sa, sgn);
        let e0 = b.bini(ShrU, 0, 52);
        let exp = b.bini(And, e0, 0x7FF);
        let mant = b.bini(And, 0, (1i64 << 52) - 1);
        let iszexp = b.cmpi(Eq, exp, 0);
        let imp = b.k(1i64 << 52);
        let zc = b.k(0);
        let fimp = b.sel(iszexp, zc, imp);
        let f = b.bin(Or, mant, fimp);
        let cm1074 = b.k(-1074);
        let enorm = b.bini(Sub, exp, 1075);
        let ebin = b.sel(iszexp, cm1074, enorm);
        let prec = b.load64(pa); // prec just stashed at FMT_PREC_O
        let s = b.bin(Add, ebin, prec); // e + prec
        let sa2 = b.bini(Add, 4, s_o);
        b.store64(sa2, s);
        let isspec = b.cmpi(Eq, exp, 0x7FF);
        b.block(
            vec![i64t, i64t, i64t, i64t, i64t],
            Terminator::BrIf {
                cond: isspec,
                then_blk: SPECIAL,
                then_args: vec![4, 0],
                else_blk: FINITE,
                else_args: vec![4, f],
            },
        )
    };

    // 1: SPECIAL(scratch, bits) — "[sign]inf"/"nan".
    let b_special = {
        let mut b = Bdr::new(2);
        let sc = fmt_sign_byte(&mut b, 0);
        let cb = b.bini(Add, 0, FMT_CBUF_O);
        b.store8(cb, sc);
        let hassign = b.cmpi(Ne, sc, 0);
        let one = b.k(1);
        let zero = b.k(0);
        let count = b.sel(hassign, one, zero);
        let fa = b.bini(Add, 0, FMT_FLAGS_O);
        let flags = b.load64(fa);
        let upf = b.bini(And, flags, 8);
        let upper = b.cmpi(Ne, upf, 0);
        let mant = b.bini(And, 1, (1i64 << 52) - 1);
        let isinf = b.cmpi(Eq, mant, 0);
        let li = b.k(b'i' as i64);
        let ui = b.k(b'I' as i64);
        let i_c = b.sel(upper, ui, li);
        let ln = b.k(b'n' as i64);
        let un = b.k(b'N' as i64);
        let n_c = b.sel(upper, un, ln);
        let la = b.k(b'a' as i64);
        let ua = b.k(b'A' as i64);
        let a_c = b.sel(upper, ua, la);
        let lf = b.k(b'f' as i64);
        let uf = b.k(b'F' as i64);
        let f_c = b.sel(upper, uf, lf);
        let ch0 = b.sel(isinf, i_c, n_c);
        let ch1 = b.sel(isinf, n_c, a_c);
        let ch2 = b.sel(isinf, f_c, n_c);
        let cbase = b.bini(Add, 0, FMT_CBUF_O);
        let a0 = b.bin(Add, cbase, count);
        b.store8(a0, ch0);
        let p1 = b.bini(Add, count, 1);
        let a1 = b.bin(Add, cbase, p1);
        b.store8(a1, ch1);
        let p2 = b.bini(Add, count, 2);
        let a2 = b.bin(Add, cbase, p2);
        b.store8(a2, ch2);
        let clen = b.bini(Add, count, 3);
        let cla = b.bini(Add, 0, FMT_CLEN_O);
        b.store64(cla, clen);
        b.block(
            vec![i64t, i64t],
            Terminator::Br {
                target: PAD_START,
                args: vec![0],
            },
        )
    };

    // 2: FINITE(scratch, f) — A = f; enter ×5^prec.
    let b_finite = {
        let mut b = Bdr::new(2); // scratch=0, f=1
        let a = b.bini(Add, 0, a_o);
        b.call0(zero, vec![a]);
        b.store32(a, 1); // A[0] = f low 32
        let a4 = b.bini(Add, a, 4);
        let fhi = b.bini(ShrU, 1, 32);
        b.store32(a4, fhi); // A[1] = f high
        let pa = b.bini(Add, 0, FMT_PREC_O);
        let prec = b.load64(pa);
        b.block(
            vec![i64t, i64t],
            Terminator::Br {
                target: MUL5_TEST,
                args: vec![0, prec],
            },
        )
    };

    // 3: MUL5_TEST(scratch, count).
    let b_mul5_test = {
        let mut b = Bdr::new(2);
        let done = b.cmpi(LeS, 1, 0);
        b.block(
            vec![i64t, i64t],
            Terminator::BrIf {
                cond: done,
                then_blk: SHIFT,
                then_args: vec![0],
                else_blk: MUL5_BODY,
                else_args: vec![0, 1],
            },
        )
    };

    // 4: MUL5_BODY(scratch, count) — A *= 5.
    let b_mul5_body = {
        let mut b = Bdr::new(2);
        let a = b.bini(Add, 0, a_o);
        let c5 = b.k(5);
        b.call0(mul, vec![a, c5]);
        let nc = b.bini(Sub, 1, 1);
        b.block(
            vec![i64t, i64t],
            Terminator::Br {
                target: MUL5_TEST,
                args: vec![0, nc],
            },
        )
    };

    // 5: SHIFT(scratch) — s ≥ 0 ⇒ left shift, else round-shift right.
    let b_shift = {
        let mut b = Bdr::new(1);
        let sa = b.bini(Add, 0, s_o);
        let s = b.load64(sa);
        let neg = b.cmpi(LtS, s, 0);
        b.block(
            vec![i64t],
            Terminator::BrIf {
                cond: neg,
                then_blk: SHR_SETUP,
                then_args: vec![0],
                else_blk: SHL_DO,
                else_args: vec![0],
            },
        )
    };

    // 6: SHL_DO(scratch) — A <<= s.
    let b_shl_do = {
        let mut b = Bdr::new(1);
        let a = b.bini(Add, 0, a_o);
        let sa = b.bini(Add, 0, s_o);
        let s = b.load64(sa);
        b.call0(shl, vec![a, s]);
        b.block(
            vec![i64t],
            Terminator::Br {
                target: DEC_INIT,
                args: vec![0],
            },
        )
    };

    // 7: SHR_SETUP(scratch) — shift = -s; enter the ≫1 round loop.
    let b_shr_setup = {
        let mut b = Bdr::new(1);
        let sa = b.bini(Add, 0, s_o);
        let s = b.load64(sa);
        let zero = b.k(0);
        let shift = b.bin(Sub, zero, s);
        let z1 = b.k(0);
        let z2 = b.k(0);
        b.block(
            vec![i64t],
            Terminator::Br {
                target: SHR_LOOP,
                args: vec![0, shift, z1, z2],
            },
        )
    };

    // 8: SHR_LOOP(scratch, count, sticky, lastout).
    let b_shr_loop = {
        let mut b = Bdr::new(4);
        let done = b.cmpi(LeS, 1, 0);
        b.block(
            vec![i64t, i64t, i64t, i64t],
            Terminator::BrIf {
                cond: done,
                then_blk: SHR_ROUND,
                then_args: vec![0, 2, 3],
                else_blk: SHR_BODY,
                else_args: vec![0, 1, 2, 3],
            },
        )
    };

    // 9: SHR_BODY(scratch, count, sticky, lastout) — out = A≫1; sticky |= lastout.
    let b_shr_body = {
        let mut b = Bdr::new(4);
        let a = b.bini(Add, 0, a_o);
        let out = b.call1(shr1, vec![a]);
        let nsticky = b.bin(Or, 2, 3);
        let nc = b.bini(Sub, 1, 1);
        b.block(
            vec![i64t, i64t, i64t, i64t],
            Terminator::Br {
                target: SHR_LOOP,
                args: vec![0, nc, nsticky, out],
            },
        )
    };

    // 10: SHR_ROUND(scratch, sticky, lastout) — round half-to-even.
    let b_shr_round = {
        let mut b = Bdr::new(3); // scratch=0, sticky=1, lastout=2
        let a = b.bini(Add, 0, a_o);
        let v0 = b.load32u(a);
        let lsb = b.bini(And, v0, 1);
        let tie = b.bin(Or, 1, lsb); // sticky | lsb
        let up = b.bin(And, 2, tie); // lastout & (sticky | lsb)
        let upb = b.cmpi(Ne, up, 0);
        b.block(
            vec![i64t, i64t, i64t],
            Terminator::BrIf {
                cond: upb,
                then_blk: SHR_INC,
                then_args: vec![0],
                else_blk: DEC_INIT,
                else_args: vec![0],
            },
        )
    };

    // 11: SHR_INC(scratch) — N += 1.
    let b_shr_inc = {
        let mut b = Bdr::new(1);
        let a = b.bini(Add, 0, a_o);
        b.call0(inc, vec![a]);
        b.block(
            vec![i64t],
            Terminator::Br {
                target: DEC_INIT,
                args: vec![0],
            },
        )
    };

    // 12: DEC_INIT(scratch) — D = 0 (do-while ⇒ ≥1 digit).
    let b_dec_init = {
        let mut b = Bdr::new(1);
        let z = b.k(0);
        let da = b.bini(Add, 0, dcnt_o);
        b.store64(da, z);
        b.block(
            vec![i64t],
            Terminator::Br {
                target: DEC_BODY,
                args: vec![0],
            },
        )
    };

    // 13: DEC_BODY(scratch) — digit = N % 10; N /= 10; dbuf[D++] = digit.
    let b_dec_body = {
        let mut b = Bdr::new(1);
        let a = b.bini(Add, 0, a_o);
        let digit = b.call1(divmod, vec![a]);
        let da = b.bini(Add, 0, dcnt_o);
        let d = b.load64(da);
        let dbase = b.bini(Add, 0, FMT_DBUF_O);
        let addr = b.bin(Add, dbase, d);
        b.store8(addr, digit);
        let nd = b.bini(Add, d, 1);
        b.store64(da, nd);
        b.block(
            vec![i64t],
            Terminator::Br {
                target: DEC_TEST,
                args: vec![0],
            },
        )
    };

    // 14: DEC_TEST(scratch) — all zero ⇒ assemble.
    let b_dec_test = {
        let mut b = Bdr::new(1);
        let a = b.bini(Add, 0, a_o);
        let z = b.call1(iszero, vec![a]);
        let zb = b.cmpi(Ne, z, 0);
        b.block(
            vec![i64t],
            Terminator::BrIf {
                cond: zb,
                then_blk: ASM_SIGN,
                then_args: vec![0],
                else_blk: DEC_BODY,
                else_args: vec![0],
            },
        )
    };

    // 15: ASM_SIGN(scratch) — sign byte + cursor seed.
    let b_asm_sign = {
        let mut b = Bdr::new(1);
        let sc = fmt_sign_byte(&mut b, 0);
        let cb = b.bini(Add, 0, FMT_CBUF_O);
        b.store8(cb, sc);
        let hassign = b.cmpi(Ne, sc, 0);
        let one = b.k(1);
        let zero = b.k(0);
        let count = b.sel(hassign, one, zero);
        let cla = b.bini(Add, 0, FMT_CLEN_O);
        b.store64(cla, count);
        b.block(
            vec![i64t],
            Terminator::Br {
                target: ASM_INT,
                args: vec![0],
            },
        )
    };

    // 16: ASM_INT(scratch) — integer = D-prec digits, else "0".
    let b_asm_int = {
        let mut b = Bdr::new(1);
        let da = b.bini(Add, 0, dcnt_o);
        let d = b.load64(da);
        let pa = b.bini(Add, 0, FMT_PREC_O);
        let prec = b.load64(pa);
        let has = b.cmp(GtS, d, prec);
        let istart = b.bini(Sub, d, 1);
        b.block(
            vec![i64t],
            Terminator::BrIf {
                cond: has,
                then_blk: ASM_INT_TEST,
                then_args: vec![0, istart],
                else_blk: ASM_INT_ZERO,
                else_args: vec![0],
            },
        )
    };

    // 17: ASM_INT_ZERO(scratch) — emit "0".
    let b_asm_int_zero = {
        let mut b = Bdr::new(1);
        let z = b.k(b'0' as i64);
        emit(&mut b, 0, z);
        b.block(
            vec![i64t],
            Terminator::Br {
                target: ASM_DOT,
                args: vec![0],
            },
        )
    };

    // 18: ASM_INT_TEST(scratch, i) — i ≥ prec ⇒ emit dbuf[i].
    let b_asm_int_test = {
        let mut b = Bdr::new(2);
        let pa = b.bini(Add, 0, FMT_PREC_O);
        let prec = b.load64(pa);
        let go = b.cmp(GeS, 1, prec);
        b.block(
            vec![i64t, i64t],
            Terminator::BrIf {
                cond: go,
                then_blk: ASM_INT_BODY,
                then_args: vec![0, 1],
                else_blk: ASM_DOT,
                else_args: vec![0],
            },
        )
    };

    // 19: ASM_INT_BODY(scratch, i) — emit '0'+dbuf[i]; i--.
    let b_asm_int_body = {
        let mut b = Bdr::new(2);
        let dbase = b.bini(Add, 0, FMT_DBUF_O);
        let da = b.bin(Add, dbase, 1);
        let dig = b.load8u(da);
        let ch = b.bini(Add, dig, b'0' as i64);
        emit(&mut b, 0, ch);
        let ni = b.bini(Sub, 1, 1);
        b.block(
            vec![i64t, i64t],
            Terminator::Br {
                target: ASM_INT_TEST,
                args: vec![0, ni],
            },
        )
    };

    // 20: ASM_DOT(scratch) — '.' iff prec>0, then fraction.
    let b_asm_dot = {
        let mut b = Bdr::new(1);
        let pa = b.bini(Add, 0, FMT_PREC_O);
        let prec = b.load64(pa);
        let cla = b.bini(Add, 0, FMT_CLEN_O);
        let cur = b.load64(cla);
        let cb = b.bini(Add, 0, FMT_CBUF_O);
        let a = b.bin(Add, cb, cur);
        let dot = b.k(b'.' as i64);
        b.store8(a, dot);
        let pos = b.cmpi(GtS, prec, 0);
        let curp = b.bini(Add, cur, 1);
        let cur2 = b.sel(pos, curp, cur);
        b.store64(cla, cur2);
        let istart = b.bini(Sub, prec, 1);
        b.block(
            vec![i64t],
            Terminator::BrIf {
                cond: pos,
                then_blk: ASM_FRAC_TEST,
                then_args: vec![0, istart],
                else_blk: PAD_START,
                else_args: vec![0],
            },
        )
    };

    // 21: ASM_FRAC_TEST(scratch, i) — i ≥ 0 ⇒ emit fraction digit.
    let b_asm_frac_test = {
        let mut b = Bdr::new(2);
        let go = b.cmpi(GeS, 1, 0);
        b.block(
            vec![i64t, i64t],
            Terminator::BrIf {
                cond: go,
                then_blk: ASM_FRAC_BODY,
                then_args: vec![0, 1],
                else_blk: PAD_START,
                else_args: vec![0],
            },
        )
    };

    // 22: ASM_FRAC_BODY(scratch, i) — digit = (i<D)?dbuf[i]:0.
    let b_asm_frac_body = {
        let mut b = Bdr::new(2);
        let da = b.bini(Add, 0, dcnt_o);
        let d = b.load64(da);
        let iltd = b.cmp(LtS, 1, d);
        let dbase = b.bini(Add, 0, FMT_DBUF_O);
        let addr = b.bin(Add, dbase, 1);
        let raw = b.load8u(addr);
        let zero = b.k(0);
        let dig = b.sel(iltd, raw, zero);
        let ch = b.bini(Add, dig, b'0' as i64);
        emit(&mut b, 0, ch);
        let ni = b.bini(Sub, 1, 1);
        b.block(
            vec![i64t, i64t],
            Terminator::Br {
                target: ASM_FRAC_TEST,
                args: vec![0, ni],
            },
        )
    };

    // 23: PAD_START(scratch).
    let b_pad_start = {
        let mut b = Bdr::new(1);
        let cla = b.bini(Add, 0, FMT_CLEN_O);
        let clen = b.load64(cla);
        let wa = b.bini(Add, 0, FMT_WIDTH_O);
        let width = b.load64(wa);
        let wgt = b.cmp(GtS, width, clen);
        let total = b.sel(wgt, width, clen);
        let pad = b.bin(Sub, total, clen);
        let fa = b.bini(Add, 0, FMT_FLAGS_O);
        let flags = b.load64(fa);
        let leftf = b.bini(And, flags, 1);
        let isleft = b.cmpi(Ne, leftf, 0);
        let zero = b.k(0);
        let lead = b.sel(isleft, zero, pad);
        let ta = b.bini(Add, 0, FMT_TOTAL_O);
        b.store64(ta, total);
        let lla = b.bini(Add, 0, FMT_LEAD_O);
        b.store64(lla, lead);
        let z = b.k(0);
        b.block(
            vec![i64t],
            Terminator::Br {
                target: PAD_FILL_TEST,
                args: vec![0, z],
            },
        )
    };

    // 24: PAD_FILL_TEST(scratch, j).
    let b_pad_fill_test = {
        let mut b = Bdr::new(2);
        let ta = b.bini(Add, 0, FMT_TOTAL_O);
        let total = b.load64(ta);
        let go = b.cmp(LtS, 1, total);
        let z = b.k(0);
        b.block(
            vec![i64t, i64t],
            Terminator::BrIf {
                cond: go,
                then_blk: PAD_FILL_BODY,
                then_args: vec![0, 1],
                else_blk: PAD_COPY_TEST,
                else_args: vec![0, z],
            },
        )
    };

    // 25: PAD_FILL_BODY(scratch, j).
    let b_pad_fill_body = {
        let mut b = Bdr::new(2);
        let ob = b.bini(Add, 0, FMT_OUT_O);
        let a = b.bin(Add, ob, 1);
        let sp = b.k(b' ' as i64);
        b.store8(a, sp);
        let nj = b.bini(Add, 1, 1);
        b.block(
            vec![i64t, i64t],
            Terminator::Br {
                target: PAD_FILL_TEST,
                args: vec![0, nj],
            },
        )
    };

    // 26: PAD_COPY_TEST(scratch, k).
    let b_pad_copy_test = {
        let mut b = Bdr::new(2);
        let cla = b.bini(Add, 0, FMT_CLEN_O);
        let clen = b.load64(cla);
        let go = b.cmp(LtS, 1, clen);
        b.block(
            vec![i64t, i64t],
            Terminator::BrIf {
                cond: go,
                then_blk: PAD_COPY_BODY,
                then_args: vec![0, 1],
                else_blk: RET,
                else_args: vec![0],
            },
        )
    };

    // 27: PAD_COPY_BODY(scratch, k).
    let b_pad_copy_body = {
        let mut b = Bdr::new(2);
        let cb = b.bini(Add, 0, FMT_CBUF_O);
        let ca = b.bin(Add, cb, 1);
        let ch = b.load8u(ca);
        let lla = b.bini(Add, 0, FMT_LEAD_O);
        let lead = b.load64(lla);
        let ob = b.bini(Add, 0, FMT_OUT_O);
        let off = b.bin(Add, lead, 1);
        let oa = b.bin(Add, ob, off);
        b.store8(oa, ch);
        let nk = b.bini(Add, 1, 1);
        b.block(
            vec![i64t, i64t],
            Terminator::Br {
                target: PAD_COPY_TEST,
                args: vec![0, nk],
            },
        )
    };

    // 28: RET(scratch) — return padded field length.
    let b_ret = {
        let mut b = Bdr::new(1);
        let ta = b.bini(Add, 0, FMT_TOTAL_O);
        let total = b.load64(ta);
        b.block(vec![i64t], Terminator::Return(vec![total]))
    };

    Func {
        params: vec![i64t, i64t, i64t, i64t, i64t],
        results: vec![i64t],
        blocks: vec![
            b_entry,
            b_special,
            b_finite,
            b_mul5_test,
            b_mul5_body,
            b_shift,
            b_shl_do,
            b_shr_setup,
            b_shr_loop,
            b_shr_body,
            b_shr_round,
            b_shr_inc,
            b_dec_init,
            b_dec_body,
            b_dec_test,
            b_asm_sign,
            b_asm_int,
            b_asm_int_zero,
            b_asm_int_test,
            b_asm_int_body,
            b_asm_dot,
            b_asm_frac_test,
            b_asm_frac_body,
            b_pad_start,
            b_pad_fill_test,
            b_pad_fill_body,
            b_pad_copy_test,
            b_pad_copy_body,
            b_ret,
        ],
    }
}

/// `__svm_atomic_rmw_narrow(word_addr, shift, width_mask, value, opcode) -> i64` — emulate a narrow
/// (i8/i16) `atomicrmw` via a seq-cst 32-bit CAS loop over the enclosing aligned word. `shift` is the
/// field's bit offset within the word, `width_mask` its unshifted mask (0xFF / 0xFFFF). Returns the
/// **old** field value (zero-extended). `opcode`: 0=xchg 1=add 2=sub 3=and 4=or 5=xor.
fn synth_atomic_rmw_narrow() -> Func {
    use BinOp::{Add, And, Or, Shl, ShrU, Sub, Xor};
    use CmpOp::Eq;
    let i64t = ValType::I64;
    const LOOP: u32 = 1;
    const RET: u32 = 2;
    // 0: ENTRY → LOOP(word_addr, shift, width_mask, value, opcode)
    let b0 = {
        let b = Bdr::new(5);
        b.block(
            vec![i64t, i64t, i64t, i64t, i64t],
            Terminator::Br {
                target: LOOP,
                args: vec![0, 1, 2, 3, 4],
            },
        )
    };
    // 1: LOOP — load word, splice the new field, CAS; retry on contention.
    let b1 = {
        let mut b = Bdr::new(5); // word_addr=0, shift=1, width_mask=2, value=3, opcode=4
        let w32 = b.atomic_load32(0);
        let w = b.ext(w32);
        let sh = b.bin(ShrU, w, 1);
        let old = b.bin(And, sh, 2);
        // new field = op(old, value), masked to the field width
        let addv = b.bin(Add, old, 3);
        let subv = b.bin(Sub, old, 3);
        let andv = b.bin(And, old, 3);
        let orv = b.bin(Or, old, 3);
        let xorv = b.bin(Xor, old, 3);
        let is0 = b.cmpi(Eq, 4, 0);
        let is1 = b.cmpi(Eq, 4, 1);
        let is2 = b.cmpi(Eq, 4, 2);
        let is3 = b.cmpi(Eq, 4, 3);
        let is4 = b.cmpi(Eq, 4, 4);
        let t = b.sel(is4, orv, xorv);
        let t = b.sel(is3, andv, t);
        let t = b.sel(is2, subv, t);
        let t = b.sel(is1, addv, t);
        let nf = b.sel(is0, 3, t); // opcode 0 = xchg ⇒ the raw value
        let newfield = b.bin(And, nf, 2);
        let fieldmask = b.bin(Shl, 2, 1);
        let notmask = b.bini(BinOp::Xor, fieldmask, -1);
        let clr = b.bin(And, w, notmask);
        let shifted = b.bin(Shl, newfield, 1);
        let neww = b.bin(Or, clr, shifted);
        let neww32 = b.wrap32(neww);
        let got32 = b.atomic_cas32(0, w32, neww32);
        let gotw = b.ext(got32);
        let ok = b.cmp(Eq, gotw, w);
        b.block(
            vec![i64t, i64t, i64t, i64t, i64t],
            Terminator::BrIf {
                cond: ok,
                then_blk: RET,
                then_args: vec![old],
                else_blk: LOOP,
                else_args: vec![0, 1, 2, 3, 4],
            },
        )
    };
    let b2 = {
        let b = Bdr::new(1);
        b.block(vec![i64t], Terminator::Return(vec![0]))
    };
    Func {
        params: vec![i64t, i64t, i64t, i64t, i64t],
        results: vec![i64t],
        blocks: vec![b0, b1, b2],
    }
}

/// `__svm_atomic_cas_narrow(word_addr, shift, width_mask, expected, replacement) -> i64` — emulate a
/// narrow (i8/i16) `cmpxchg` via a seq-cst 32-bit CAS loop. `expected`/`replacement` are pre-masked
/// by the caller. Returns the **old** field value; the caller derives success from `old == expected`.
fn synth_atomic_cas_narrow() -> Func {
    use BinOp::{And, Or, Shl, ShrU};
    use CmpOp::{Eq, Ne};
    let i64t = ValType::I64;
    const LOOP: u32 = 1;
    const TRY: u32 = 2;
    const RET: u32 = 3;
    // 0: ENTRY → LOOP
    let b0 = {
        let b = Bdr::new(5);
        b.block(
            vec![i64t, i64t, i64t, i64t, i64t],
            Terminator::Br {
                target: LOOP,
                args: vec![0, 1, 2, 3, 4],
            },
        )
    };
    // 1: LOOP(word_addr, shift, width_mask, expected, replacement) — load + mismatch check.
    let b1 = {
        let mut b = Bdr::new(5);
        let w32 = b.atomic_load32(0);
        let w = b.ext(w32);
        let sh = b.bin(ShrU, w, 1);
        let old = b.bin(And, sh, 2);
        let ne = b.cmp(Ne, old, 3); // old != expected ⇒ cmpxchg fails, return old
        b.block(
            vec![i64t, i64t, i64t, i64t, i64t],
            Terminator::BrIf {
                cond: ne,
                then_blk: RET,
                then_args: vec![old],
                else_blk: TRY,
                else_args: vec![0, 1, 2, 3, 4, w32, w],
            },
        )
    };
    // 2: TRY(word_addr, shift, width_mask, expected, replacement, w32:i32, w:i64) — splice, CAS.
    let i32t = ValType::I32;
    let b2 = {
        let mut b = Bdr::new(7);
        let newfield = b.bin(And, 4, 2); // replacement & width_mask
        let fieldmask = b.bin(Shl, 2, 1);
        let notmask = b.bini(BinOp::Xor, fieldmask, -1);
        let clr = b.bin(And, 6, notmask); // w & ~fieldmask
        let shifted = b.bin(Shl, newfield, 1);
        let neww = b.bin(Or, clr, shifted);
        let neww32 = b.wrap32(neww);
        let got32 = b.atomic_cas32(0, 5, neww32); // expected = w32 (param 5)
        let gotw = b.ext(got32);
        let ok = b.cmp(Eq, gotw, 6); // gotw == w
        let sh = b.bin(ShrU, 6, 1);
        let old = b.bin(And, sh, 2); // == expected (we got here only on a match)
        b.block(
            vec![i64t, i64t, i64t, i64t, i64t, i32t, i64t],
            Terminator::BrIf {
                cond: ok,
                then_blk: RET,
                then_args: vec![old],
                else_blk: LOOP,
                else_args: vec![0, 1, 2, 3, 4],
            },
        )
    };
    let b3 = {
        let b = Bdr::new(1);
        b.block(vec![i64t], Terminator::Return(vec![0]))
    };
    Func {
        params: vec![i64t, i64t, i64t, i64t, i64t],
        results: vec![i64t],
        blocks: vec![b0, b1, b2, b3],
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
/// A parsed `printf` format-string segment (the constant format is parsed at translate time, §ramp
/// Lane C). Integer conversions (`%d`/`%i`/`%u`/`%x`) with the `-`/`+`/` `/`0`/`#` flags + field
/// width + min-digit precision, `%c`, `%s` (runtime strlen, with width + truncating precision), and
/// `%%` are handled; float conversions (`%f`/`%e`/`%g`), `*` (dynamic width/precision), and
/// non-constant format strings are fail-closed `Unsupported` (float needs exact-decimal/bignum).
/// Which floating-point notation a `%f`/`%e`/`%g` conversion uses.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum FloatKind {
    /// `%f` — fixed notation `[-]ddd.ffff`.
    Fixed,
    /// `%e` — scientific notation `[-]d.ddde±dd`.
    Sci,
    /// `%g` — the shorter of `%e`/`%f` with trailing zeros stripped.
    Gen,
}

enum FmtSeg {
    /// A verbatim run — bytes `[off, off+len)` of the format global, written directly.
    Lit { off: usize, len: usize },
    /// An integer conversion: `base` (10/16, lowercase), `signed` (`%d`) vs unsigned (`%u`/`%x`),
    /// field `width`, optional `prec` (minimum digit count — zero-extended), and the layout `flags`.
    Int {
        base: u64,
        signed: bool,
        width: u32,
        prec: Option<u32>,
        flags: FmtFlags,
    },
    /// `%c` — the argument's low byte.
    Char,
    /// `%s` — a NUL-terminated string argument (runtime `strlen`), right-justified in field `width`
    /// (space-padded), optionally truncated to `prec` bytes.
    Str { width: u32, prec: Option<u32> },
    /// A floating-point `double` conversion. `kind` selects fixed (`%f`), scientific (`%e`), or
    /// general (`%g`) notation; `upper` is the uppercase spelling (`%F`/`%E`/`%G` ⇒ `E`/`INF`/`NAN`).
    /// `prec` is the parsed precision (C default 6); the caller space-pads to field `width` per
    /// `flags`. Fixed uses the synthesized 128-bit `__svm_dtoa_fixed`; scientific/general use the
    /// exact bignum `__svm_dtoa_*` engine.
    Float {
        width: u32,
        prec: u32,
        flags: FmtFlags,
        kind: FloatKind,
        upper: bool,
    },
    /// `%%` — a literal percent.
    Percent,
}

/// `printf` integer-conversion layout flags (`0`/`-`/`+`/space/`#`), already normalized for the
/// standard precedence: `-` (left-justify) suppresses `0` (zero-pad), and `+` (force sign) suppresses
/// ` ` (space-before-positive).
#[derive(Clone, Copy, Default)]
struct FmtFlags {
    /// `0` — pad with leading zeros (between the sign/prefix and the digits).
    zero: bool,
    /// `-` — left-justify in the field (trailing spaces).
    left: bool,
    /// `+` — always emit a sign for a signed conversion (`+` for non-negatives).
    plus: bool,
    /// ` ` — emit a leading space before a non-negative signed conversion.
    space: bool,
    /// `#` — alternate form; for `%x` prefixes a non-zero value with `0x`.
    alt: bool,
}

/// Parse a (NUL-free) `printf` format string into segments. Fail-closed on anything not yet
/// supported, so an unhandled directive is a clean `Unsupported`, never a silent mis-format.
fn parse_format(fmt: &[u8]) -> Result<Vec<FmtSeg>, Error> {
    let mut segs = Vec::new();
    let mut i = 0;
    let mut lit_start = 0;
    while i < fmt.len() {
        if fmt[i] != b'%' {
            i += 1;
            continue;
        }
        if i > lit_start {
            segs.push(FmtSeg::Lit {
                off: lit_start,
                len: i - lit_start,
            });
        }
        i += 1; // past '%'
        let bad = |w: &str| Error::Unsupported(format!("printf: {w}"));
        if i >= fmt.len() {
            return Err(bad("trailing '%'"));
        }
        if fmt[i] == b'%' {
            segs.push(FmtSeg::Percent);
            i += 1;
            lit_start = i;
            continue;
        }
        // Flags (any order, repeatable): `0` zero-pad, `-` left-justify, `+` force sign, ` ` space
        // before positive, `#` alternate form.
        let mut flags = FmtFlags::default();
        while i < fmt.len() {
            match fmt[i] {
                b'0' => flags.zero = true,
                b'-' => flags.left = true,
                b'+' => flags.plus = true,
                b' ' => flags.space = true,
                b'#' => flags.alt = true,
                _ => break,
            }
            i += 1;
        }
        // Standard precedence: `-` overrides `0`; `+` overrides ` `.
        if flags.left {
            flags.zero = false;
        }
        if flags.plus {
            flags.space = false;
        }
        // Field width (decimal). `*` (dynamic) and `.` (precision) are deferred.
        let mut width = 0u32;
        while i < fmt.len() && fmt[i].is_ascii_digit() {
            width = width * 10 + u32::from(fmt[i] - b'0');
            i += 1;
        }
        if width as u64 + 2 >= FMT_BUF_END - FMT_BUF {
            return Err(bad("field width too large"));
        }
        if i < fmt.len() && fmt[i] == b'*' {
            return Err(bad("dynamic width (`*`)"));
        }
        // Optional precision (`.N`; bare `.` means `.0`). `*` precision is deferred.
        let mut prec: Option<u32> = None;
        if i < fmt.len() && fmt[i] == b'.' {
            i += 1;
            if i < fmt.len() && fmt[i] == b'*' {
                return Err(bad("dynamic precision (`.*`)"));
            }
            let mut p = 0u32;
            while i < fmt.len() && fmt[i].is_ascii_digit() {
                p = p * 10 + u32::from(fmt[i] - b'0');
                i += 1;
            }
            prec = Some(p);
        }
        // Length modifiers are informational here — the LLVM arg already carries the real width.
        while i < fmt.len() && matches!(fmt[i], b'l' | b'h' | b'z' | b'j' | b't' | b'L') {
            i += 1;
        }
        let conv = *fmt.get(i).ok_or_else(|| bad("trailing conversion"))?;
        i += 1;
        // `#` (alternate) is only meaningful for `%x` here (the `0x` prefix); on `%d`/`%i`/`%u` it is
        // a no-op, so drop it to keep the emitter's prefix logic hex-only.
        // An integer min-digits precision is bounded so the zero-extension fits the scratch buffer.
        if let Some(p) = prec {
            if p as u64 + 2 >= FMT_BUF_END - FMT_BUF {
                return Err(bad("precision too large"));
            }
        }
        // `#` (alternate) is only meaningful for `%x` here (the `0x` prefix); on `%d`/`%i`/`%u` it is
        // a no-op, so drop it to keep the emitter's prefix logic hex-only. A precision also disables
        // the `0` flag (C standard) for integers — handled in `emit_printf_int_field`.
        let int = |base: u64, signed| {
            let mut f = flags;
            if base != 16 {
                f.alt = false;
            }
            FmtSeg::Int {
                base,
                signed,
                width,
                prec,
                flags: f,
            }
        };
        segs.push(match conv {
            b'd' | b'i' => int(10, true),
            b'u' => int(10, false),
            b'x' => int(16, false),
            // `%p`: a pointer as `0x`-prefixed lowercase hex of the address (glibc form) — i.e. `%#x`
            // of the `i64` pointer. (glibc prints `(nil)` for a null pointer; that edge prints `0x0`
            // here — a documented minor divergence. Lua uses `%p` to tag objects in error/debug text.)
            b'p' => {
                let mut f = flags;
                f.alt = true;
                FmtSeg::Int {
                    base: 16,
                    signed: false,
                    width,
                    prec,
                    flags: f,
                }
            }
            b'c' => FmtSeg::Char,
            b's' => FmtSeg::Str { width, prec },
            // Fixed-notation float (`%f`). Exact (correctly-rounded) decimal conversion via the
            // synthesized `__svm_dtoa_fixed` (fixed 128-bit integer arithmetic — no host float
            // formatting, so interp≡JIT≡native). `prec` defaults to 6 (C); capped at 31 so the
            // `m·5^prec` intermediate fits 128 bits (`5^31·2^53 < 2^128`), and `width` is bounded to
            // the helper's scratch field. A value so large that `round(|v|·10^prec)` exceeds 128 bits
            // traps deterministically (never a silent mis-format; bignum lifts both limits later). The
            // `0`/`#` flags are not yet handled for floats (only sign / space-pad / left-justify).
            // Fixed-notation float (`%f`/`%F`). Exact decimal via the bignum `__svm_dtoa_fix_big`
            // (Dragon-style big-integer `N = round(|v|·10^prec)`), so correctly-rounded with no
            // magnitude ceiling — large values that overflowed the old 128-bit path now format.
            b'f' | b'F' => {
                let p = prec.unwrap_or(6);
                if p > 510 {
                    return Err(bad("float precision > 510 (digit-buffer cap)"));
                }
                if width > 200 {
                    return Err(bad("float field width > 200 (scratch cap)"));
                }
                if flags.zero || flags.alt {
                    return Err(bad("float `0`/`#` flag (later slice)"));
                }
                FmtSeg::Float {
                    width,
                    prec: p,
                    flags,
                    kind: FloatKind::Fixed,
                    upper: conv == b'F',
                }
            }
            // Scientific notation (`%e`/`%E`). Exact decimal via the bignum `__svm_dtoa_sci` (Dragon4
            // digit engine), so correctly-rounded across the whole double range — no magnitude cap.
            b'e' | b'E' => {
                let p = prec.unwrap_or(6);
                if p > 510 {
                    return Err(bad("float precision > 510 (digit-buffer cap)"));
                }
                if width > 200 {
                    return Err(bad("float field width > 200 (scratch cap)"));
                }
                if flags.zero || flags.alt {
                    return Err(bad("float `0`/`#` flag (later slice)"));
                }
                FmtSeg::Float {
                    width,
                    prec: p,
                    flags,
                    kind: FloatKind::Sci,
                    upper: conv == b'E',
                }
            }
            // General notation (`%g`/`%G`): shorter of `%e`/`%f`, trailing zeros stripped. Exact via
            // the bignum `__svm_dtoa_gen`. `prec` is significant digits (C default 6; 0 ⇒ 1).
            b'g' | b'G' => {
                let p = prec.unwrap_or(6);
                if p > 510 {
                    return Err(bad("float precision > 510 (digit-buffer cap)"));
                }
                if width > 200 {
                    return Err(bad("float field width > 200 (scratch cap)"));
                }
                if flags.zero || flags.alt {
                    return Err(bad("float `0`/`#` flag (later slice)"));
                }
                FmtSeg::Float {
                    width,
                    prec: p,
                    flags,
                    kind: FloatKind::Gen,
                    upper: conv == b'G',
                }
            }
            other => return Err(bad(&format!("conversion %{}", other as char))),
        });
        lit_start = i;
    }
    if fmt.len() > lit_start {
        segs.push(FmtSeg::Lit {
            off: lit_start,
            len: fmt.len() - lit_start,
        });
    }
    Ok(segs)
}

/// The `i`-th call argument operand, bounds-checked (a fail-closed error rather than a panic when a
/// declaration has fewer args than the builtin expects).
fn vm_arg(c: &llvm_ir::instruction::Call, i: usize) -> Result<&Operand, Error> {
    c.arguments
        .get(i)
        .map(|(o, _)| o)
        .ok_or_else(|| Error::Unsupported("`__vm_*` builtin: too few arguments".into()))
}

/// Lower a `<svm.h>` low-level builtin (`crates/.../include/svm.h`, the chibicc oracle in
/// `frontend/chibicc/codegen_ir.c`) to the matching SVM IR op or `Memory` capability call. Returns
/// `Ok(true)` if `name` is one of these intrinsics (and it was lowered), `Ok(false)` otherwise so the
/// caller falls through to the ordinary direct/indirect call path. The caller gates this on `name`
/// being **external** (not guest-defined), so a guest function of the same name shadows the builtin
/// (mirrors the libc/libm rule).
///
/// Coverage (the P0+P1+Memory surface): the §3e/§4 **Memory** capability (`__vm_map`/`unmap`/
/// `protect`/`page_size`); §12 **fibers** (`__vm_fiber_new`/`resume`/`suspend` → `cont.new`/`resume`/
/// `suspend`); §GC conservative **roots** (`__vm_gc_roots` → `gc.roots`); §12 **threads**
/// (`__vm_thread_spawn`/`join`) and **atomics** (`__vm_atomic_*` → the `iN.atomic.*` ops); the §12
/// **futex** (`__vm_wait32`/`__vm_notify`); and §7 capability **reflection** (`__vm_cap` reads the
/// handle stash; `__vm_cap_count`/`__vm_cap_at` → `cap.self.count`/`cap.self.get`). Each mirrors the
/// chibicc lowering exactly, so a program built through either frontend produces equivalent IR.
fn lower_vm_builtin(
    ctx: &mut BlockCtx,
    c: &llvm_ir::instruction::Call,
    name: &str,
) -> Result<bool, Error> {
    use svm_ir::{AtomicRmwOp, LoadOp, Ordering, StoreOp};
    // All §12 atomics are sequentially consistent (the op makes the JIT emit a hardware atomic).
    let sc = Ordering::SeqCst;
    match name {
        // ---- §3e/§4 Memory capability: `cap.call` on the stashed Memory handle (slot 12) ----
        "__vm_map" | "__vm_unmap" | "__vm_protect" => {
            let import = vm_memory_builtin_import(name).expect("memory builtin");
            let imp = ctx.import_of(import)?;
            let off = ctx.operand_i64(vm_arg(c, 0)?)?;
            let len = ctx.operand_i64(vm_arg(c, 1)?)?;
            let mut args = vec![off, len];
            if name != "__vm_unmap" {
                args.push(ctx.operand_i32(vm_arg(c, 2)?)?); // prot
            }
            let handle = ctx.stash_load(STASH_MEMORY);
            let r = ctx.push(Inst::CallImport {
                import: imp,
                sig: import_sig(import),
                handle,
                args,
            });
            ctx.bind_dest(&c.dest, r);
            Ok(true)
        }
        "__vm_page_size" => {
            let imp = ctx.import_of("vm_page_size")?;
            let handle = ctx.stash_load(STASH_MEMORY);
            let r = ctx.push(Inst::CallImport {
                import: imp,
                sig: import_sig("vm_page_size"),
                handle,
                args: vec![],
            });
            ctx.bind_dest(&c.dest, r);
            Ok(true)
        }
        // ---- §9/§12 async I/O ring: `cap.call` on the stashed IoRing handle (slot 5) ----
        "__vm_io_submit_async" | "__vm_io_reap" => {
            let import = vm_io_builtin_import(name).expect("io builtin");
            let imp = ctx.import_of(import)?;
            let mut args = vec![
                ctx.operand_i64(vm_arg(c, 0)?)?,
                ctx.operand_i64(vm_arg(c, 1)?)?,
            ];
            if name == "__vm_io_submit_async" {
                args.push(ctx.operand_i64(vm_arg(c, 2)?)?); // the completion counter pointer
            }
            let handle = ctx.stash_load(STASH_IORING);
            let r = ctx.push(Inst::CallImport {
                import: imp,
                sig: import_sig(import),
                handle,
                args,
            });
            ctx.bind_dest(&c.dest, r);
            Ok(true)
        }
        // `__vm_blocking_handle()` returns the stashed Blocking handle (slot 6) — the `i32` a guest
        // names in an SQE's `handle` field when building a `Blocking.work` request. Just a stash read.
        "__vm_blocking_handle" => {
            let r = ctx.stash_load(STASH_BLOCKING);
            ctx.bind_dest(&c.dest, r);
            Ok(true)
        }
        // ---- §22 guest-driven JIT: `cap.call` on the stashed Jit handle (slot 7) ----
        // Each builtin marshals its `i64` args (a blob/symtab pointer+len, a code/slot handle, two
        // invoke args) and lowers to `CallImport` on the `Jit` handle. The host verifies the submitted
        // IR and compiles it into THIS domain (verification, not isolation, is the boundary — §2a).
        "__vm_jit_compile"
        | "__vm_jit_invoke2"
        | "__vm_jit_release"
        | "__vm_jit_install"
        | "__vm_jit_uninstall"
        | "__vm_jit_compile_linked" => {
            let import = vm_jit_builtin_import(name).expect("jit builtin");
            let imp = ctx.import_of(import)?;
            let argc = import_sig(import).params.len();
            let mut args = Vec::with_capacity(argc);
            for i in 0..argc {
                args.push(ctx.operand_i64(vm_arg(c, i)?)?);
            }
            let handle = ctx.stash_load(STASH_JIT);
            let r = ctx.push(Inst::CallImport {
                import: imp,
                sig: import_sig(import),
                handle,
                args,
            });
            ctx.bind_dest(&c.dest, r);
            Ok(true)
        }
        // ---- §13/§14 SharedRegion: mint from AddressSpace, then alias via the region handle ----
        // `create(len)` calls `AddressSpace.create_region` on the stashed AddressSpace handle (slot 4)
        // and returns a region handle. `map`/`unmap`/`page_size` take that region handle as their first
        // C arg (`int region`) and `cap.call` *it* — the handle is the capability, not a stash slot.
        "__vm_region_create" => {
            let imp = ctx.import_of("vm_region_create")?;
            let len = ctx.operand_i64(vm_arg(c, 0)?)?;
            let handle = ctx.stash_load(STASH_ADDRSPACE);
            let r = ctx.push(Inst::CallImport {
                import: imp,
                sig: import_sig("vm_region_create"),
                handle,
                args: vec![len],
            });
            ctx.bind_dest(&c.dest, r);
            Ok(true)
        }
        "__vm_region_map" | "__vm_region_unmap" | "__vm_region_page_size" => {
            let import = vm_region_builtin_import(name).expect("region builtin");
            let imp = ctx.import_of(import)?;
            let handle = ctx.operand_i32(vm_arg(c, 0)?)?; // the region handle (arg 0)
            let args = match name {
                "__vm_region_map" => vec![
                    ctx.operand_i64(vm_arg(c, 1)?)?, // win_off
                    ctx.operand_i64(vm_arg(c, 2)?)?, // region_off
                    ctx.operand_i64(vm_arg(c, 3)?)?, // len
                    ctx.operand_i32(vm_arg(c, 4)?)?, // prot
                ],
                "__vm_region_unmap" => vec![
                    ctx.operand_i64(vm_arg(c, 1)?)?, // win_off
                    ctx.operand_i64(vm_arg(c, 2)?)?, // len
                ],
                _ => vec![], // page_size
            };
            let r = ctx.push(Inst::CallImport {
                import: imp,
                sig: import_sig(import),
                handle,
                args,
            });
            ctx.bind_dest(&c.dest, r);
            Ok(true)
        }
        // ---- §12 fibers (stack switching) ----
        "__vm_fiber_new" => {
            // arg0 is a function pointer (an `i64` funcref); `cont.new` wants the `i32` funcref.
            let fn64 = ctx.operand(vm_arg(c, 0)?)?;
            let func = ctx.push(Inst::Convert {
                op: ConvOp::WrapI64,
                a: fn64,
            });
            let sp = ctx.operand_i64(vm_arg(c, 1)?)?; // the fiber's own data-stack base
            let r = ctx.push(Inst::ContNew { func, sp });
            ctx.bind_dest(&c.dest, r); // i64 fiber handle (16-bit slot + 48-bit generation)
            Ok(true)
        }
        "__vm_fiber_resume" => {
            let k = ctx.operand_i64(vm_arg(c, 0)?)?; // i64 fiber handle
            let arg = ctx.operand_i64(vm_arg(c, 1)?)?;
            let done = ctx.operand_i64(vm_arg(c, 2)?)?; // `int *done`
            let rs = ctx.push_multi(Inst::ContResume { k, arg }, 2); // (status, value)
            ctx.push_effect(Inst::Store {
                op: StoreOp::I32,
                addr: done,
                value: rs[0],
                offset: 0,
                align: 0,
            }); // *done = status (0 suspended / 1 returned)
            ctx.bind_dest(&c.dest, rs[1]); // the yielded/returned i64
            Ok(true)
        }
        "__vm_fiber_suspend" => {
            let value = ctx.operand_i64(vm_arg(c, 0)?)?;
            let r = ctx.push(Inst::Suspend { value });
            ctx.bind_dest(&c.dest, r); // the next resume's arg
            Ok(true)
        }
        // ---- §GC conservative root enumeration (`gc.roots`) ----
        "__vm_gc_roots" => {
            let heap_lo = ctx.operand_i64(vm_arg(c, 0)?)?;
            let heap_hi = ctx.operand_i64(vm_arg(c, 1)?)?;
            // §GC tagged-pointer payload mask: each scanned word is AND-ed with this before the range
            // test (and emitted), so a tag in the high byte is stripped to the bare offset. The VM
            // constrains it to top-byte-strip only (no host-address leak); `~0UL` is the untagged case.
            let mask = ctx.operand_i64(vm_arg(c, 2)?)?;
            let buf = ctx.operand_i64(vm_arg(c, 3)?)?;
            let cap = ctx.operand_i64(vm_arg(c, 4)?)?;
            let r = ctx.push(Inst::GcRoots {
                heap_lo,
                heap_hi,
                mask,
                buf,
                cap,
            });
            ctx.bind_dest(&c.dest, r); // total candidate count (i64)
            Ok(true)
        }
        // ---- §12 threads ----
        "__vm_thread_spawn" => {
            let func = ctx.direct_func_idx(vm_arg(c, 0)?)?; // a static funcidx
            let sp = ctx.operand_i64(vm_arg(c, 1)?)?; // the thread's data-stack base
            let arg = ctx.operand_i64(vm_arg(c, 2)?)?;
            let r = ctx.push(Inst::ThreadSpawn { func, sp, arg });
            ctx.bind_dest(&c.dest, r); // i32 thread handle
            Ok(true)
        }
        "__vm_thread_join" => {
            let handle = ctx.operand_i32(vm_arg(c, 0)?)?;
            let r = ctx.push(Inst::ThreadJoin { handle });
            ctx.bind_dest(&c.dest, r); // i64 result
            Ok(true)
        }
        // ---- §12 atomics (linear-memory) ----
        "__vm_atomic_add" | "__vm_atomic_add32" => {
            let ty = if name.ends_with("32") {
                IntTy::I32
            } else {
                IntTy::I64
            };
            let addr = ctx.operand_i64(vm_arg(c, 0)?)?;
            let value = if ty == IntTy::I64 {
                ctx.operand_i64(vm_arg(c, 1)?)?
            } else {
                ctx.operand_i32(vm_arg(c, 1)?)?
            };
            let r = ctx.push(Inst::AtomicRmw {
                ty,
                op: AtomicRmwOp::Add,
                addr,
                value,
                offset: 0,
                order: sc,
            });
            ctx.bind_dest(&c.dest, r); // the old value
            Ok(true)
        }
        "__vm_atomic_load" | "__vm_atomic_load32" => {
            let ty = if name.ends_with("32") {
                IntTy::I32
            } else {
                IntTy::I64
            };
            let addr = ctx.operand_i64(vm_arg(c, 0)?)?;
            let r = ctx.push(Inst::AtomicLoad {
                ty,
                addr,
                offset: 0,
                order: sc,
            });
            ctx.bind_dest(&c.dest, r);
            Ok(true)
        }
        "__vm_atomic_store" | "__vm_atomic_store32" => {
            let ty = if name.ends_with("32") {
                IntTy::I32
            } else {
                IntTy::I64
            };
            let addr = ctx.operand_i64(vm_arg(c, 0)?)?;
            let value = if ty == IntTy::I64 {
                ctx.operand_i64(vm_arg(c, 1)?)?
            } else {
                ctx.operand_i32(vm_arg(c, 1)?)?
            };
            ctx.push_effect(Inst::AtomicStore {
                ty,
                addr,
                value,
                offset: 0,
                order: sc,
            });
            Ok(true) // void
        }
        "__vm_atomic_cas32" => {
            let addr = ctx.operand_i64(vm_arg(c, 0)?)?;
            let expected = ctx.operand_i32(vm_arg(c, 1)?)?;
            let replacement = ctx.operand_i32(vm_arg(c, 2)?)?;
            let r = ctx.push(Inst::AtomicCmpxchg {
                ty: IntTy::I32,
                addr,
                expected,
                replacement,
                offset: 0,
                order: sc,
            });
            ctx.bind_dest(&c.dest, r); // the old value (i32)
            Ok(true)
        }
        // ---- §12 futex ----
        "__vm_wait32" => {
            let addr = ctx.operand_i64(vm_arg(c, 0)?)?;
            let expected = ctx.operand_i32(vm_arg(c, 1)?)?;
            let timeout = ctx.operand_i64(vm_arg(c, 2)?)?; // ns
            let r = ctx.push(Inst::MemoryWait {
                ty: IntTy::I32,
                addr,
                expected,
                timeout,
            });
            ctx.bind_dest(&c.dest, r); // 0 woken / 1 not-equal / 2 timed-out
            Ok(true)
        }
        "__vm_notify" => {
            let addr = ctx.operand_i64(vm_arg(c, 0)?)?;
            let count = ctx.operand_i32(vm_arg(c, 1)?)?;
            let r = ctx.push(Inst::MemoryNotify { addr, count });
            ctx.bind_dest(&c.dest, r); // number woken
            Ok(true)
        }
        // ---- §7 capability reflection ----
        "__vm_cap" => {
            // The i-th stashed powerbox handle: an `i32.load` at byte offset `i*4` in the reserved
            // low window (the handle stash), exactly where `_start` stored the granted handles.
            let i = ctx.operand_i64(vm_arg(c, 0)?)?;
            let four = ctx.const_i64(4);
            let off = ctx.mul_i64(i, four);
            let r = ctx.push(Inst::Load {
                op: LoadOp::I32,
                addr: off,
                offset: 0,
                align: 0,
            });
            ctx.bind_dest(&c.dest, r);
            Ok(true)
        }
        "__vm_cap_count" => {
            let r = ctx.push(Inst::CapSelfCount);
            ctx.bind_dest(&c.dest, r);
            Ok(true)
        }
        "__vm_cap_at" => {
            let idx = ctx.operand_i32(vm_arg(c, 0)?)?;
            let type_id_out = ctx.operand_i64(vm_arg(c, 1)?)?; // `int *type_id_out`
            let rs = ctx.push_multi(Inst::CapSelfGet { idx }, 2); // (handle, type_id)
            ctx.push_effect(Inst::Store {
                op: StoreOp::I32,
                addr: type_id_out,
                value: rs[1],
                offset: 0,
                align: 0,
            }); // *type_id_out = type_id
            ctx.bind_dest(&c.dest, rs[0]); // the capability handle
            Ok(true)
        }
        // §12 per-vCPU TLS register: `__vm_vcpu_tls_get()` reads the current vCPU's word (seeded to the
        // dense vCPU id, so it doubles as a vCPU id); `__vm_vcpu_tls_set(x)` overwrites it (e.g. a
        // pointer to the guest's per-CPU block, for full __thread-style TLS).
        "__vm_vcpu_tls_get" => {
            let r = ctx.push(Inst::VcpuTlsGet);
            ctx.bind_dest(&c.dest, r); // i64 TLS word
            Ok(true)
        }
        "__vm_vcpu_tls_set" => {
            let val = ctx.operand_i64(vm_arg(c, 0)?)?;
            ctx.push_effect(Inst::VcpuTlsSet { val });
            Ok(true)
        }
        _ => Ok(false),
    }
}

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
        // `memcmp(a,b,n)` / `bcmp(a,b,n)`: the synthesized `__svm_memcmp` (counted unsigned byte
        // compare). `bcmp` shares it — callers only test `!= 0`, which the `0`-iff-equal result keeps.
        "memcmp" | "bcmp" => {
            let Some(f) = ctx.helpers.memcmp else {
                return Ok(false);
            };
            let a = ctx.operand(&c.arguments[0].0)?;
            let b = ctx.operand(&c.arguments[1].0)?;
            let n = ctx.operand(&c.arguments[2].0)?;
            let r = ctx.push(Inst::Call {
                func: f,
                args: vec![a, b, n],
            });
            ctx.bind_dest(&c.dest, r);
            Ok(true)
        }
        // `realloc(ptr, n)`: the synthesized `__svm_realloc` (malloc + header-sized copy).
        "realloc" => {
            let Some(f) = ctx.helpers.realloc else {
                return Ok(false);
            };
            let p = ctx.operand(&c.arguments[0].0)?;
            let n = ctx.operand(&c.arguments[1].0)?;
            let r = ctx.push(Inst::Call {
                func: f,
                args: vec![p, n],
            });
            ctx.bind_dest(&c.dest, r);
            Ok(true)
        }
        // `printf(fmt, …)`: a **guest-side format engine**. The constant format string is parsed at
        // translate time; literal runs are written straight from the format global, and each
        // conversion lowers to `__svm_utoa` + width/zero-padding (a buffer pre-fill) → `Stream.write`.
        // Returns 0 (the char count is rarely used). Non-constant formats / unsupported directives are
        // fail-closed `Unsupported` (see `parse_format`).
        "printf" => {
            lower_printf(ctx, c)?;
            let r = ctx.push(Inst::ConstI32(0));
            ctx.bind_dest(&c.dest, r);
            Ok(true)
        }
        // `strlen(s)`: the synthesized `__svm_strlen` NUL-scan loop (also used by `printf %s`).
        "strlen" => {
            let Some(f) = ctx.helpers.strlen else {
                return Ok(false); // no strlen helper synthesized → fail-closed
            };
            let p = ctx.operand(&c.arguments[0].0)?;
            let r = ctx.push(Inst::Call {
                func: f,
                args: vec![p],
            });
            ctx.bind_dest(&c.dest, r);
            Ok(true)
        }
        // `strcmp(a, b)` / `strcoll(a, b)`: the synthesized `__svm_strcmp` byte compare. `strcoll` is
        // locale-sensitive in general, but the on-ramp runs in the C locale, where it is `strcmp`.
        "strcmp" | "strcoll" => {
            let Some(f) = ctx.helpers.strcmp else {
                return Ok(false); // helper not synthesized → fail-closed
            };
            let a = ctx.operand(&c.arguments[0].0)?;
            let b = ctx.operand(&c.arguments[1].0)?;
            let r = ctx.push(Inst::Call {
                func: f,
                args: vec![a, b],
            });
            ctx.bind_dest(&c.dest, r);
            Ok(true)
        }
        // `strchr(s, c)`: the synthesized `__svm_strchr` byte scan → pointer or NULL.
        "strchr" => {
            let Some(f) = ctx.helpers.strchr else {
                return Ok(false); // helper not synthesized → fail-closed
            };
            let s = ctx.operand(&c.arguments[0].0)?;
            let ch = ctx.operand(&c.arguments[1].0)?;
            let r = ctx.push(Inst::Call {
                func: f,
                args: vec![s, ch],
            });
            ctx.bind_dest(&c.dest, r);
            Ok(true)
        }
        // `strcpy(dst, src)`: the synthesized `__svm_strcpy` copy loop → `dst`.
        "strcpy" => {
            let Some(f) = ctx.helpers.strcpy else {
                return Ok(false);
            };
            let d = ctx.operand(&c.arguments[0].0)?;
            let s = ctx.operand(&c.arguments[1].0)?;
            let r = ctx.push(Inst::Call {
                func: f,
                args: vec![d, s],
            });
            ctx.bind_dest(&c.dest, r);
            Ok(true)
        }
        // `strspn(s, set)` / `strpbrk(s, set)`: the synthesized nested-scan helpers.
        "strspn" => {
            let Some(f) = ctx.helpers.strspn else {
                return Ok(false);
            };
            let s = ctx.operand(&c.arguments[0].0)?;
            let set = ctx.operand(&c.arguments[1].0)?;
            let r = ctx.push(Inst::Call {
                func: f,
                args: vec![s, set],
            });
            ctx.bind_dest(&c.dest, r);
            Ok(true)
        }
        "strpbrk" => {
            let Some(f) = ctx.helpers.strpbrk else {
                return Ok(false);
            };
            let s = ctx.operand(&c.arguments[0].0)?;
            let set = ctx.operand(&c.arguments[1].0)?;
            let r = ctx.push(Inst::Call {
                func: f,
                args: vec![s, set],
            });
            ctx.bind_dest(&c.dest, r);
            Ok(true)
        }
        // `ldexp(x, n)` / `scalbn(x, n)`: the synthesized `__svm_ldexp` (`x · 2^n`, bit-exact to libc).
        "ldexp" | "scalbn" => {
            let Some(f) = ctx.helpers.ldexp else {
                return Ok(false);
            };
            let x = ctx.operand(&c.arguments[0].0)?;
            let n = ctx.operand(&c.arguments[1].0)?;
            let r = ctx.push(Inst::Call {
                func: f,
                args: vec![x, n],
            });
            ctx.bind_dest(&c.dest, r);
            Ok(true)
        }
        // `pow` (and the other transcendentals): fail-closed trap stub — bit-exact vs native needs a
        // matching host libm (the host-libm decision, LLVM.md). `frexp`/`fmod` are real below.
        "pow" => {
            let Some(f) = ctx.helpers.pow_stub else {
                return Ok(false);
            };
            let x = ctx.operand(&c.arguments[0].0)?;
            let y = ctx.operand(&c.arguments[1].0)?;
            let r = ctx.push(Inst::Call {
                func: f,
                args: vec![x, y],
            });
            ctx.bind_dest(&c.dest, r);
            Ok(true)
        }
        // `fmod(x, y)`: fail-closed trap stub for now (exact-synthesizable — its own slice, see `Helpers`).
        "fmod" => {
            let Some(f) = ctx.helpers.fmod_stub else {
                return Ok(false);
            };
            let x = ctx.operand(&c.arguments[0].0)?;
            let y = ctx.operand(&c.arguments[1].0)?;
            let r = ctx.push(Inst::Call {
                func: f,
                args: vec![x, y],
            });
            ctx.bind_dest(&c.dest, r);
            Ok(true)
        }
        // `frexp(x, eptr)`: the synthesized `__svm_frexp` — mantissa/exponent split, writes `*eptr`.
        "frexp" => {
            let Some(f) = ctx.helpers.frexp else {
                return Ok(false);
            };
            let x = ctx.operand(&c.arguments[0].0)?;
            let e = ctx.operand(&c.arguments[1].0)?;
            let r = ctx.push(Inst::Call {
                func: f,
                args: vec![x, e],
            });
            ctx.bind_dest(&c.dest, r);
            Ok(true)
        }
        // `strtod(nptr, endptr)`: fail-closed trap stub (string→double — not on an integer path).
        "strtod" => {
            let Some(f) = ctx.helpers.strtod_stub else {
                return Ok(false);
            };
            let s = ctx.operand(&c.arguments[0].0)?;
            let e = ctx.operand(&c.arguments[1].0)?;
            let r = ctx.push(Inst::Call {
                func: f,
                args: vec![s, e],
            });
            ctx.bind_dest(&c.dest, r);
            Ok(true)
        }
        // `snprintf(buf, size, fmt, …)`: a varargs call — caught here (before the general varargs
        // marshaling) and lowered through the shared `printf` format engine with output redirected
        // into `buf` (§printf / [`lower_snprintf`]).
        "snprintf" => {
            lower_snprintf(ctx, c)?;
            Ok(true)
        }
        // `localeconv()` / `__errno_location()`: fail-closed trap stubs (locale + errno, no-arg → ptr).
        "localeconv" => {
            let Some(f) = ctx.helpers.localeconv_stub else {
                return Ok(false);
            };
            let r = ctx.push(Inst::Call {
                func: f,
                args: vec![],
            });
            ctx.bind_dest(&c.dest, r);
            Ok(true)
        }
        "__errno_location" => {
            let Some(f) = ctx.helpers.errno_stub else {
                return Ok(false);
            };
            let r = ctx.push(Inst::Call {
                func: f,
                args: vec![],
            });
            ctx.bind_dest(&c.dest, r);
            Ok(true)
        }
        // `time(t)`: returns 0 (the `makeseed` RNG source — value-irrelevant for a deterministic run).
        "time" => {
            let Some(f) = ctx.helpers.time_zero else {
                return Ok(false);
            };
            let t = ctx.operand(&c.arguments[0].0)?;
            let r = ctx.push(Inst::Call {
                func: f,
                args: vec![t],
            });
            ctx.bind_dest(&c.dest, r);
            Ok(true)
        }
        // `memchr(s, c, n)`: the synthesized `__svm_memchr` byte scan → pointer or NULL.
        "memchr" => {
            let Some(f) = ctx.helpers.memchr else {
                return Ok(false); // helper not synthesized → fail-closed
            };
            let s = ctx.operand(&c.arguments[0].0)?;
            let ch = ctx.operand(&c.arguments[1].0)?;
            let n = ctx.operand(&c.arguments[2].0)?;
            let r = ctx.push(Inst::Call {
                func: f,
                args: vec![s, ch, n],
            });
            ctx.bind_dest(&c.dest, r);
            Ok(true)
        }
        // `<ctype.h>` locators: `isalpha`/`isspace`/`tolower`/… lower to a load through the pointer cell
        // these return (`(*__ctype_b_loc())[c] & _ISxxx`, `(*__ctype_tolower_loc())[c]`). The tables +
        // cells are synthesized as read-only data (`build_ctype_data`), so the call is just a const of the
        // cell's window address. No args.
        "__ctype_b_loc" | "__ctype_tolower_loc" | "__ctype_toupper_loc" => {
            let cell = match name {
                "__ctype_b_loc" => ctx.helpers.ctype_b_loc,
                "__ctype_tolower_loc" => ctx.helpers.ctype_tolower_loc,
                _ => ctx.helpers.ctype_toupper_loc,
            };
            let Some(addr) = cell else {
                return Ok(false); // table not synthesized → fail-closed
            };
            let r = ctx.push(Inst::ConstI64(addr as i64));
            ctx.bind_dest(&c.dest, r);
            Ok(true)
        }
        // `getenv(name)`: the synthesized `__svm_getenv` scans the §3e env strings for `name=`.
        "getenv" => {
            let Some(f) = ctx.helpers.getenv else {
                return Ok(false); // no powerbox entry → fail-closed
            };
            let name = ctx.operand(&c.arguments[0].0)?;
            let r = ctx.push(Inst::Call {
                func: f,
                args: vec![name],
            });
            ctx.bind_dest(&c.dest, r);
            Ok(true)
        }
        _ => Ok(false),
    }
}

/// Lower a `printf(fmt, …)` call (the constant format engine — see the `"printf"` arm). Emits the
/// `Stream.write`s for the literals and conversions in order, consuming the variadic args.
fn lower_printf(ctx: &mut BlockCtx, c: &llvm_ir::instruction::Call) -> Result<(), Error> {
    // `printf(fmt, …)`: format to stdout. `fmt` is arg 0; the conversion arguments start at arg 1.
    lower_format(ctx, c, 0, 1)
}

/// The shared `printf`-family format engine: parse the **constant** format at `c.arguments[fmt_idx]`
/// and emit each segment, taking conversion arguments from `c.arguments[arg_base..]`. Output goes to
/// stdout, or — when `ctx.fmt_sink` is set (`snprintf`) — into a destination buffer (the redirected
/// [`BlockCtx::emit_write`]). Used by both `printf` (no sink) and `snprintf` (sink set by the caller).
fn lower_format(
    ctx: &mut BlockCtx,
    c: &llvm_ir::instruction::Call,
    fmt_idx: usize,
    arg_base: usize,
) -> Result<(), Error> {
    let gname = global_name_of(&c.arguments[fmt_idx].0)
        .ok_or_else(|| Error::Unsupported("printf: non-constant format string".into()))?;
    let fmt_addr = *ctx
        .globals
        .get(&gname)
        .ok_or_else(|| Error::Unsupported("printf: format not in window".into()))?;
    let bytes = ctx
        .gbytes
        .get(&gname)
        .ok_or_else(|| Error::Unsupported("printf: format not a constant string".into()))?
        .clone();
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    let segs = parse_format(&bytes[..end])?;

    let utoa = ctx
        .helpers
        .utoa
        .ok_or_else(|| Error::Unsupported("printf: utoa helper missing".into()))?;
    let mut argi = arg_base; // conversion arguments follow the format string
    for seg in segs {
        match seg {
            FmtSeg::Lit { off, len } => {
                let a = ctx.const_i64((fmt_addr + off as u64) as i64);
                let n = ctx.const_i64(len as i64);
                ctx.emit_write(a, n)?;
            }
            FmtSeg::Percent => {
                let pct = ctx.push(Inst::ConstI32(b'%' as i32));
                let scratch = ctx.scratch_byte(pct);
                let one = ctx.const_i64(1);
                ctx.emit_write(scratch, one)?;
            }
            FmtSeg::Char => {
                let arg = &c.arguments.get(argi).ok_or_else(|| {
                    Error::Unsupported("printf: more conversions than args".into())
                })?;
                argi += 1;
                let v = ctx.operand(&arg.0)?;
                let scratch = ctx.scratch_byte(v);
                let one = ctx.const_i64(1);
                ctx.emit_write(scratch, one)?;
            }
            FmtSeg::Str { width, prec } => {
                let arg = c.arguments.get(argi).ok_or_else(|| {
                    Error::Unsupported("printf: more conversions than args".into())
                })?;
                argi += 1;
                let ptr = ctx.operand_i64(&arg.0)?;
                let strlen = ctx
                    .helpers
                    .strlen
                    .ok_or_else(|| Error::Unsupported("printf: strlen helper missing".into()))?;
                let mut len = ctx.push(Inst::Call {
                    func: strlen,
                    args: vec![ptr],
                });
                // Precision truncates: write at most `prec` bytes (`len = min(strlen, prec)`).
                if let Some(p) = prec {
                    let pv = ctx.const_i64(p as i64);
                    let lt = ctx.push(Inst::IntCmp {
                        ty: IntTy::I64,
                        op: CmpOp::LtU,
                        a: len,
                        b: pv,
                    });
                    len = ctx.push(Inst::Select {
                        cond: lt,
                        a: len,
                        b: pv,
                    });
                }
                // Right-justify in `width`: emit `max(0, width - len)` leading spaces from a
                // pre-filled scratch run, then the string bytes straight from its pointer.
                if width > 0 {
                    let space = ctx.push(Inst::ConstI32(b' ' as i32));
                    for k in 0..width as u64 {
                        let a = ctx.const_i64((FMT_BUF + k) as i64);
                        ctx.push_effect(Inst::Store {
                            op: svm_ir::StoreOp::I32_8,
                            addr: a,
                            value: space,
                            offset: 0,
                            align: 0,
                        });
                    }
                    let wv = ctx.const_i64(width as i64);
                    let gt = ctx.push(Inst::IntCmp {
                        ty: IntTy::I64,
                        op: CmpOp::GtU,
                        a: wv,
                        b: len,
                    });
                    let diff = ctx.push(Inst::IntBin {
                        ty: IntTy::I64,
                        op: BinOp::Sub,
                        a: wv,
                        b: len,
                    });
                    let zero = ctx.const_i64(0);
                    let padlen = ctx.push(Inst::Select {
                        cond: gt,
                        a: diff,
                        b: zero,
                    });
                    let padbuf = ctx.const_i64(FMT_BUF as i64);
                    ctx.emit_write(padbuf, padlen)?;
                }
                ctx.emit_write(ptr, len)?;
            }
            FmtSeg::Int {
                base,
                signed,
                width,
                prec,
                flags,
            } => {
                let arg = c.arguments.get(argi).ok_or_else(|| {
                    Error::Unsupported("printf: more conversions than args".into())
                })?;
                argi += 1;
                let av = ctx.operand(&arg.0)?;
                let is64 = matches!(val_type(arg.0.get_type(ctx.types).as_ref())?, ValType::I64);
                // Compute the unsigned magnitude `mag` to format and, for `%d`, an `i64` 0/1 `neg`.
                let (mag, neg) = if signed {
                    let sval = if is64 {
                        av
                    } else {
                        ctx.push(Inst::Convert {
                            op: ConvOp::ExtendI32S,
                            a: av,
                        })
                    };
                    let zero64 = ctx.const_i64(0);
                    let negi = ctx.push(Inst::IntCmp {
                        ty: IntTy::I64,
                        op: CmpOp::LtS,
                        a: sval,
                        b: zero64,
                    });
                    let nsval = ctx.push(Inst::IntBin {
                        ty: IntTy::I64,
                        op: BinOp::Sub,
                        a: zero64,
                        b: sval,
                    });
                    let mag = ctx.push(Inst::Select {
                        cond: negi,
                        a: nsval,
                        b: sval,
                    });
                    let neg = ctx.push(Inst::Convert {
                        op: ConvOp::ExtendI32U,
                        a: negi,
                    });
                    (mag, Some(neg))
                } else {
                    let mag = if is64 {
                        av
                    } else {
                        ctx.push(Inst::Convert {
                            op: ConvOp::ExtendI32U,
                            a: av,
                        })
                    };
                    (mag, None)
                };
                emit_printf_int_field(ctx, mag, base, neg, width, prec, flags, utoa)?;
            }
            FmtSeg::Float {
                width,
                prec,
                flags,
                kind,
                upper,
            } => {
                let arg = c.arguments.get(argi).ok_or_else(|| {
                    Error::Unsupported("printf: more conversions than args".into())
                })?;
                argi += 1;
                // The variadic argument is a `double` (C promotes `float` → `double`); pass its
                // IEEE-754 bit pattern so the helper decodes it with pure integer arithmetic.
                match val_type(arg.0.get_type(ctx.types).as_ref())? {
                    ValType::F64 => {}
                    other => {
                        return Err(Error::Unsupported(format!(
                            "printf float expects a double, got {}",
                            other.as_str()
                        )))
                    }
                }
                let dval = ctx.operand(&arg.0)?;
                let bits = ctx.push(Inst::Cast {
                    op: CastOp::ReinterpF64I64,
                    a: dval,
                });
                let precv = ctx.const_i64(prec as i64);
                let widthv = ctx.const_i64(width as i64);
                // flags: bit0 left-justify, bit1 `+`, bit2 space, bit3 uppercase (all compile-time).
                let fbits = (flags.left as i64)
                    | ((flags.plus as i64) << 1)
                    | ((flags.space as i64) << 2)
                    | ((upper as i64) << 3);
                let flagsv = ctx.const_i64(fbits);
                // Every float kind uses the exact bignum engine: the formatter fills the scratch
                // output field and returns its length; we write `scratch+FMT_OUT_O .. +len`.
                let formatter = match kind {
                    FloatKind::Fixed => ctx.helpers.dtoa_fix,
                    FloatKind::Sci => ctx.helpers.dtoa_sci,
                    FloatKind::Gen => ctx.helpers.dtoa_gen,
                }
                .ok_or_else(|| Error::Unsupported("printf: bignum float helper missing".into()))?;
                let scratch = ctx
                    .helpers
                    .float_scratch
                    .ok_or_else(|| Error::Unsupported("printf: float scratch missing".into()))?;
                let scratchv = ctx.const_i64(scratch as i64);
                let len = ctx.push(Inst::Call {
                    func: formatter,
                    args: vec![bits, precv, widthv, flagsv, scratchv],
                });
                let outp = ctx.const_i64((scratch + FMT_OUT_O as u64) as i64);
                ctx.emit_write(outp, len)?;
            }
        }
    }
    Ok(())
}

/// `snprintf(buf, size, fmt, …)` — reuse the `printf` format engine ([`lower_format`]) with output
/// redirected into `buf` via the [`FmtSink`] path of [`BlockCtx::emit_write`], then NUL-terminate
/// within `size` and return the would-be length (C semantics — the count that *would* have been
/// written, excluding the NUL). `buf`/`size` are runtime values; `fmt` is the constant string at
/// argument 2 (e.g. Lua's number formats `%lld`/`%.14g` and the `%d`/`%s` error-message conversions).
/// A `size` of 0 is not faithfully supported (Lua always passes an adequate buffer); every other size
/// is bounded so `buf[..size]` is never overrun.
fn lower_snprintf(ctx: &mut BlockCtx, c: &llvm_ir::instruction::Call) -> Result<(), Error> {
    let dest = ctx.operand_i64(&c.arguments[0].0)?;
    let size = ctx.operand_i64(&c.arguments[1].0)?;
    let zero = ctx.const_i64(0);
    ctx.fmt_sink = Some(FmtSink {
        dest,
        size,
        offset: zero,
    });
    let res = lower_format(ctx, c, 2, 3);
    let sink = ctx.fmt_sink.take(); // clear the sink even if formatting failed
    res?;
    let offset = sink.expect("snprintf sink present").offset; // would-be length
                                                              // NUL-terminate at min(offset, max(0, size - 1)).
    let one = ctx.const_i64(1);
    let size_m1 = ctx.push(Inst::IntBin {
        ty: IntTy::I64,
        op: BinOp::Sub,
        a: size,
        b: one,
    });
    let zero2 = ctx.const_i64(0);
    let neg = ctx.push(Inst::IntCmp {
        ty: IntTy::I64,
        op: CmpOp::LtS,
        a: size_m1,
        b: zero2,
    });
    let cap = ctx.push(Inst::Select {
        cond: neg,
        a: zero2,
        b: size_m1,
    }); // max(0, size-1)
    let off_lt = ctx.push(Inst::IntCmp {
        ty: IntTy::I64,
        op: CmpOp::LtU,
        a: offset,
        b: cap,
    });
    let nul_pos = ctx.push(Inst::Select {
        cond: off_lt,
        a: offset,
        b: cap,
    });
    let nul_addr = ctx.add_i64(dest, nul_pos);
    let zbyte = ctx.push(Inst::ConstI32(0));
    ctx.push_effect(Inst::Store {
        op: svm_ir::StoreOp::I32_8,
        addr: nul_addr,
        value: zbyte,
        offset: 0,
        align: 0,
    });
    // Return value: the would-be length, as `int` (low 32 bits).
    let ret = ctx.push(Inst::Convert {
        op: ConvOp::WrapI64,
        a: offset,
    });
    ctx.bind_dest(&c.dest, ret);
    Ok(())
}

fn pf_sub(ctx: &mut BlockCtx, a: ValIdx, b: ValIdx) -> ValIdx {
    ctx.push(Inst::IntBin {
        ty: IntTy::I64,
        op: BinOp::Sub,
        a,
        b,
    })
}

fn pf_store8(ctx: &mut BlockCtx, addr: ValIdx, value: ValIdx) {
    ctx.push_effect(Inst::Store {
        op: svm_ir::StoreOp::I32_8,
        addr,
        value,
        offset: 0,
        align: 0,
    });
}

/// Pre-fill the low pad run (`[FMT_BUF, FMT_BUF+width)`) with the pad char (`'0'` only for a non-left
/// zero-pad, else `' '`). Called *before* the digits are formatted into the high end of the buffer, so
/// any overlap is harmlessly overwritten by the digits; the bytes actually read for padding are always
/// strictly below the content, so a single fill serves any runtime pad length.
fn pf_prefill_pad(ctx: &mut BlockCtx, width: u32, flags: FmtFlags) {
    if width == 0 {
        return;
    }
    let pad_char = if flags.zero { b'0' } else { b' ' };
    let padch = ctx.push(Inst::ConstI32(pad_char as i32));
    for k in 0..width as u64 {
        let a = ctx.const_i64((FMT_BUF + k) as i64);
        pf_store8(ctx, a, padch);
    }
}

/// Store the sign char just below `start` and return its runtime length (`0`/`1`). `neg` is the
/// `i64` `0/1` "is negative" flag; `+`/space force a sign on non-negatives.
fn pf_sign_prefix(ctx: &mut BlockCtx, start: ValIdx, neg: ValIdx, flags: FmtFlags) -> ValIdx {
    let one = ctx.const_i64(1);
    let (sign, plen) = if flags.plus || flags.space {
        let pos = ctx.push(Inst::ConstI32(if flags.plus { b'+' } else { b' ' } as i32));
        let dash = ctx.push(Inst::ConstI32(b'-' as i32));
        let zero64 = ctx.const_i64(0);
        let negb = ctx.push(Inst::IntCmp {
            ty: IntTy::I64,
            op: CmpOp::Ne,
            a: neg,
            b: zero64,
        });
        let s = ctx.push(Inst::Select {
            cond: negb,
            a: dash,
            b: pos,
        });
        (s, one)
    } else {
        let dash = ctx.push(Inst::ConstI32(b'-' as i32));
        (dash, neg) // present only when negative
    };
    let sm1 = pf_sub(ctx, start, one);
    pf_store8(ctx, sm1, sign);
    plen
}

/// The shared field-layout tail: given the content `[content_start, FMT_BUF_END)` (digits at
/// `[start, FMT_BUF_END)`, an optional sign/`0x` prefix of `prefixlen` just below), apply the field
/// `width` and the justify/pad flags (the pad run was pre-filled by [`pf_prefill_pad`]). Emits the
/// `Stream.write`s in output order. The layout is compile-time; only digit/pad lengths are runtime.
fn pf_field_layout(
    ctx: &mut BlockCtx,
    start: ValIdx,
    content_start: ValIdx,
    prefixlen: ValIdx,
    width: u32,
    flags: FmtFlags,
) -> Result<(), Error> {
    let bufend = ctx.const_i64(FMT_BUF_END as i64);
    let contentlen = pf_sub(ctx, bufend, content_start);
    if width == 0 {
        ctx.emit_write(content_start, contentlen)?;
        return Ok(());
    }
    let wv = ctx.const_i64(width as i64);
    let gt = ctx.push(Inst::IntCmp {
        ty: IntTy::I64,
        op: CmpOp::GtU,
        a: wv,
        b: contentlen,
    });
    let diff = pf_sub(ctx, wv, contentlen);
    let zero = ctx.const_i64(0);
    let pad = ctx.push(Inst::Select {
        cond: gt,
        a: diff,
        b: zero,
    });
    let padbuf = ctx.const_i64(FMT_BUF as i64);
    if flags.left {
        // Left-justify: content, then trailing spaces.
        ctx.emit_write(content_start, contentlen)?;
        ctx.emit_write(padbuf, pad)?;
    } else if flags.zero {
        // Zero-pad: prefix, then zeros, then digits (so the sign/`0x` precedes the zeros).
        let ndigits = pf_sub(ctx, bufend, start);
        ctx.emit_write(content_start, prefixlen)?;
        ctx.emit_write(padbuf, pad)?;
        ctx.emit_write(start, ndigits)?;
    } else {
        // Right-justify with spaces: leading spaces, then content.
        ctx.emit_write(padbuf, pad)?;
        ctx.emit_write(content_start, contentlen)?;
    }
    Ok(())
}

/// Emit the `Stream.write`s for one `printf` integer conversion with full flag/width/precision
/// handling, to match C `printf` byte-for-byte. `mag` is the unsigned magnitude (i64); `neg` is the
/// runtime `0/1` "is negative" flag for a signed conversion (`None` for unsigned). `__svm_utoa`
/// formats the digits backward into the high end of the scratch buffer; the sign / `0x` prefix and
/// field padding are applied around them.
///
/// A precision is a *minimum digit count*: the digit region is zero-extended to `prec` digits (by
/// pre-filling `'0'`s that `utoa` overwrites from the right), and a precision disables the `0` flag
/// (C standard — field padding then uses spaces). `%.0d`/`%.0x` of `0` prints **no** digits.
#[allow(clippy::too_many_arguments)]
fn emit_printf_int_field(
    ctx: &mut BlockCtx,
    mag: ValIdx,
    base: u64,
    neg: Option<ValIdx>,
    width: u32,
    prec: Option<u32>,
    flags: FmtFlags,
    utoa: u32,
) -> Result<(), Error> {
    // A precision overrides the `0` flag: the field is space-padded around the (already zero-extended)
    // digits.
    let layout_flags = if prec.is_some() {
        FmtFlags {
            zero: false,
            ..flags
        }
    } else {
        flags
    };
    pf_prefill_pad(ctx, width, layout_flags);

    let bufend_off = FMT_BUF_END;
    // Zero-extension to `prec` digits: pre-fill `[bufend-prec, bufend)` with `'0'` *before* `utoa`,
    // which overwrites the rightmost `ndigits` with the real digits — leaving `max(prec, ndigits)`.
    if let Some(p) = prec {
        if p > 0 {
            let zc = ctx.push(Inst::ConstI32(b'0' as i32));
            for k in 0..p as u64 {
                let a = ctx.const_i64((bufend_off - p as u64 + k) as i64);
                pf_store8(ctx, a, zc);
            }
        }
    }
    let basec = ctx.const_i64(base as i64);
    let bufend = ctx.const_i64(bufend_off as i64);
    let start = ctx.push(Inst::Call {
        func: utoa,
        args: vec![mag, basec, bufend],
    }); // real digits at [start, bufend)

    // The effective leftmost digit, after precision zero-extension / `%.0` suppression.
    let eff_start = match prec {
        None => start,
        Some(p) if p > 0 => {
            // min(start, bufend - p): use all real digits if more than `p`, else the padded region.
            let limit = ctx.const_i64((bufend_off - p as u64) as i64);
            let lt = ctx.push(Inst::IntCmp {
                ty: IntTy::I64,
                op: CmpOp::LtU,
                a: start,
                b: limit,
            });
            ctx.push(Inst::Select {
                cond: lt,
                a: start,
                b: limit,
            })
        }
        Some(_) => {
            // `%.0`: value `0` prints nothing; any other value keeps its digits.
            let z = ctx.const_i64(0);
            let is0 = ctx.push(Inst::IntCmp {
                ty: IntTy::I64,
                op: CmpOp::Eq,
                a: mag,
                b: z,
            });
            ctx.push(Inst::Select {
                cond: is0,
                a: bufend,
                b: start,
            })
        }
    };

    // Prefix (sign for signed, `0x` for `#x`): store the bytes just below `eff_start`; `prefixlen`
    // (0/1/2, runtime) selects how many are included in the field.
    let prefixlen = match neg {
        Some(negv) => pf_sign_prefix(ctx, eff_start, negv, flags),
        None if flags.alt => {
            // `%#x`: a `0x` prefix, but only for a non-zero value.
            let z = ctx.const_i64(0);
            let nz = ctx.push(Inst::IntCmp {
                ty: IntTy::I64,
                op: CmpOp::Ne,
                a: mag,
                b: z,
            });
            let x = ctx.push(Inst::ConstI32(b'x' as i32));
            let zero_ch = ctx.push(Inst::ConstI32(b'0' as i32));
            let one = ctx.const_i64(1);
            let sm1 = pf_sub(ctx, eff_start, one);
            pf_store8(ctx, sm1, x);
            let two = ctx.const_i64(2);
            let sm2 = pf_sub(ctx, eff_start, two);
            pf_store8(ctx, sm2, zero_ch);
            ctx.push(Inst::Select {
                cond: nz,
                a: two,
                b: z,
            })
        }
        None => ctx.const_i64(0),
    };
    let content_start = pf_sub(ctx, eff_start, prefixlen);
    pf_field_layout(
        ctx,
        eff_start,
        content_start,
        prefixlen,
        width,
        layout_flags,
    )
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

/// The lane value of a **constant integer splat** vector — `<i32 1, i32 1, i32 1, i32 1>`, which is how
/// clang encodes a vector shift by a *uniform* amount (the common auto-vectorized `x >> k` shape). The
/// SVM `VShift` shifts every lane by one scalar amount, so a splat maps directly. `None` for a
/// non-constant or non-uniform amount (those stay fail-closed — a per-lane vector shift is rare).
fn const_splat_int(op: &Operand) -> Option<u64> {
    let Operand::ConstantOperand(c) = op else {
        return None;
    };
    let Constant::Vector(elems) = c.as_ref() else {
        return None;
    };
    let mut lanes = elems.iter().map(|e| match e.as_ref() {
        Constant::Int { value, .. } => Some(*value),
        _ => None,
    });
    let first = lanes.next()??;
    for v in lanes {
        if v? != first {
            return None;
        }
    }
    Some(first)
}

/// Lower a **saturating float→int** intrinsic (`llvm.fptosi.sat.<int>.<float>` /
/// `llvm.fptoui.sat.…`, which Rust's `as` casts from float emit) to svm-IR's `FToISat` — exactly the
/// saturating semantics (NaN→0, out-of-range→clamped) the on-ramp already gives the plain `fptosi`
/// instruction (§3b). The src/dst widths are parsed from the mangled name. `Ok(None)` for other calls;
/// `Unsupported` for a vector form (`v4i32`/…), a later slice.
fn lower_fp_sat_intrinsic(
    ctx: &mut BlockCtx,
    c: &llvm_ir::instruction::Call,
) -> Result<Option<ValIdx>, Error> {
    let Some(name) = callee_name(c) else {
        return Ok(None);
    };
    let signed = if name.starts_with("llvm.fptosi.sat.") {
        true
    } else if name.starts_with("llvm.fptoui.sat.") {
        false
    } else {
        return Ok(None);
    };
    // `llvm.fpto{si,ui}.sat.<int>.<float>` — the last two dot-components are the dst int / src float.
    let mut parts = name.rsplit('.');
    let fstr = parts.next().unwrap_or("");
    let istr = parts.next().unwrap_or("");
    let dst = match istr {
        "i32" => IntTy::I32,
        "i64" => IntTy::I64,
        _ => return unsup(format!("`{name}` (only i32/i64 result)")),
    };
    let src = match fstr {
        "f32" => FloatTy::F32,
        "f64" => FloatTy::F64,
        _ => return unsup(format!("`{name}` (only f32/f64 source)")),
    };
    let a = ctx.operand(&c.arguments[0].0)?;
    Ok(Some(ctx.push(Inst::FToISat {
        op: ftoi_op(src, dst, signed),
        a,
    })))
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
            | "llvm.bswap"
            | "llvm.bitreverse"
            | "llvm.uadd.sat"
            | "llvm.usub.sat"
            | "llvm.sadd.sat"
            | "llvm.ssub.sat"
    ) {
        return Ok(None);
    }
    let args: Vec<&Operand> = c.arguments.iter().map(|(a, _)| a).collect();
    // A 128-bit integer-vector min/max (auto-vectorized) lowers to the lane-wise `VIntBin` (§17) in
    // the operand's shape; the other bit intrinsics have no vector form here. (Float vector min/max
    // go through the float path.)
    if let Some(shape) = vec128_shape(args[0].get_type(types).as_ref()) {
        // A vector funnel shift (`llvm.fshl`/`fshr`) in the **rotate idiom** (`a == b`). svm-ir's
        // `VShift` takes only a *scalar* amount, but the auto-vectorizer emits per-lane-varying
        // constant amounts (e.g. xxHash's `<1,7,12,18>`), so scalarize: rotate each lane by its own
        // amount, then repack into the `v128` (the lane `Rotl`/`Rotr` masks the count mod width, so
        // there is no shift-by-width edge case, mirroring the scalar rotate path).
        if matches!(base, "llvm.fshl" | "llvm.fshr") {
            if shape.is_float() {
                return unsup("vector funnel shift on a float shape");
            }
            if args[0] != args[1] {
                return unsup(format!("general vector funnel shift `{name}` (non-rotate)"));
            }
            let lane_ty = int_ty(shape.lane_val())?;
            let data = vec_explode(ctx, args[0], types, false)?;
            let amts = vec_explode(ctx, args[2], types, false)?;
            let op = if base == "llvm.fshl" {
                BinOp::Rotl
            } else {
                BinOp::Rotr
            };
            let mut out = Vec::with_capacity(data.len());
            for (&d, &s) in data.iter().zip(amts.iter()) {
                out.push(ctx.push(Inst::IntBin {
                    ty: lane_ty,
                    op,
                    a: d,
                    b: s,
                }));
            }
            return Ok(Some(build_v128_from_lanes(ctx, shape, &out)));
        }
        // Per-lane popcount — wasm has it only for `i8x16` (the sole vector `ctpop` width).
        if base == "llvm.ctpop" {
            if shape != svm_ir::VShape::I8x16 {
                return unsup(format!("vector ctpop on {shape:?} (only i8x16)"));
            }
            let a = ctx.operand(args[0])?;
            return Ok(Some(ctx.push(Inst::VPopcnt { a })));
        }
        let op = match base {
            "llvm.smax" => svm_ir::VIntBinOp::MaxS,
            "llvm.smin" => svm_ir::VIntBinOp::MinS,
            "llvm.umax" => svm_ir::VIntBinOp::MaxU,
            "llvm.umin" => svm_ir::VIntBinOp::MinU,
            other => return unsup(format!("vector `{other}` (only min/max or ctpop)")),
        };
        let a = ctx.operand(args[0])?;
        let b = ctx.operand(args[1])?;
        return Ok(Some(ctx.push(Inst::VIntBin { shape, op, a, b })));
    }
    let ty = int_ty(val_type(args[0].get_type(types).as_ref())?)?;
    // A **narrow** (< i32) operand sits in an i32 container whose high bits are unspecified — e.g. a
    // non-canonical `add i8 x, -1` (the narrow `bin` path doesn't mask power-of-two widths). min/max
    // selects via an i32 compare, so — exactly like `ICmp` (§3b narrow-int hazard) — the operands must
    // first be canonically extended: sign- for signed min/max, zero- for unsigned. Without this,
    // `umin.i8(add i8 x,-1, y)` compares dirty containers and picks the wrong operand (a silent
    // miscompile — found via Embench `qrduino`). The `Select` then runs on the canonical values, so the
    // chosen result is canonical too.
    let narrow = int_bits(args[0].get_type(types).as_ref()).filter(|&w| w < 32);
    let cmp_select = |ctx: &mut BlockCtx, op: CmpOp, signed: bool| -> Result<ValIdx, Error> {
        let mut a = ctx.operand(args[0])?;
        let mut b = ctx.operand(args[1])?;
        if let Some(w) = narrow {
            a = emit_ext(ctx, a, w, 32, signed);
            b = emit_ext(ctx, b, w, 32, signed);
        }
        let cond = ctx.push(Inst::IntCmp { ty, op, a, b });
        Ok(ctx.push(Inst::Select { cond, a, b }))
    };
    let unop = |ctx: &mut BlockCtx, op: IntUnOp| -> Result<ValIdx, Error> {
        let a = ctx.operand(args[0])?;
        Ok(ctx.push(Inst::IntUn { ty, op, a }))
    };
    let idx = match base {
        "llvm.smax" => cmp_select(ctx, CmpOp::GtS, true)?,
        "llvm.smin" => cmp_select(ctx, CmpOp::LtS, true)?,
        "llvm.umax" => cmp_select(ctx, CmpOp::GtU, false)?,
        "llvm.umin" => cmp_select(ctx, CmpOp::LtU, false)?,
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
        // `rotl`/`rotr`, which mask the count mod width and accept a runtime amount — but only at a
        // *full* width (`i32`/`i64`), where the svm-ir rotate's width equals the i32/i64 container; a
        // narrow rotate would wrongly rotate the whole 32-bit container, so it falls through to the
        // general path. A true funnel shift (distinct operands) **or** a narrow rotate, with a
        // **constant** amount, lowers to `(a << s) | (b >>u (w - s))` (`s = amt mod w`; `s == 0` ⇒ the
        // no-shift operand) — both shift counts are then in `1..w`, no shift-by-`w` edge case — and the
        // result is masked back to `w` (a narrow value lives in a wider container). Found via Embench
        // `aha-mont64`'s `modul64` (`fshl.i64(hi, lo, 1)`) and `picojpeg` (`fshl.i16`). A non-constant
        // amount (needs a width-edge-safe `select`) or a `33..63` width stays fail-closed.
        "llvm.fshl" | "llvm.fshr" => {
            let is_fshl = base == "llvm.fshl";
            let w = src_bits(args[0], types)?;
            if args[0] == args[1] && (w == 32 || w == 64) {
                let a = ctx.operand(args[0])?;
                let amt = ctx.operand(args[2])?;
                let op = if is_fshl { BinOp::Rotl } else { BinOp::Rotr };
                ctx.push(Inst::IntBin { ty, op, a, b: amt })
            } else {
                if w > 32 && w != 64 {
                    return unsup(format!(
                        "funnel shift `{name}` on i{w} (only i8..=i32 or i64)"
                    ));
                }
                let Some(c) = const_int(args[2]) else {
                    return unsup(format!(
                        "general funnel shift `{name}` (non-constant amount)"
                    ));
                };
                let s = (c % w as u64) as i64;
                let a = ctx.operand(args[0])?;
                let b = ctx.operand(args[1])?;
                let r = if s == 0 {
                    // `fshl(a,b,0) = a`, `fshr(a,b,0) = b` (the concatenation shifted by nothing).
                    if is_fshl {
                        a
                    } else {
                        b
                    }
                } else {
                    // fshl: a's bits go up by `s`, b supplies the low `w-s`; fshr is the mirror.
                    let (lsh, rsh) = if is_fshl {
                        (s, w as i64 - s)
                    } else {
                        (w as i64 - s, s)
                    };
                    let kk = |ctx: &mut BlockCtx, n: i64| {
                        if ty == IntTy::I64 {
                            ctx.push(Inst::ConstI64(n))
                        } else {
                            ctx.push(Inst::ConstI32(n as i32))
                        }
                    };
                    let lc = kk(ctx, lsh);
                    let rc = kk(ctx, rsh);
                    let hi = ctx.push(Inst::IntBin {
                        ty,
                        op: BinOp::Shl,
                        a,
                        b: lc,
                    });
                    let lo = ctx.push(Inst::IntBin {
                        ty,
                        op: BinOp::ShrU,
                        a: b,
                        b: rc,
                    });
                    ctx.push(Inst::IntBin {
                        ty,
                        op: BinOp::Or,
                        a: hi,
                        b: lo,
                    })
                };
                // A narrow funnel/rotate result lives in a wider container; mask off bits ≥ `w` so the
                // value stays canonical (`mask_to` is a no-op at `w == 32`; `w == 64` fills its i64).
                if w <= 32 {
                    mask_to(ctx, r, w)
                } else {
                    r
                }
            }
        }
        // `llvm.bswap` — reverse the value's bytes inline (no SVM op): each source byte `i` is
        // shifted to destination byte `nbytes-1-i`.
        "llvm.bswap" => {
            let bits = src_bits(args[0], types)?;
            let v = ctx.operand(args[0])?;
            emit_bswap(ctx, v, ty, (bits / 8).max(1) as u64)
        }
        // `llvm.bitreverse` — reverse the value's bits inline (no SVM op), via the log-N swap network.
        "llvm.bitreverse" => {
            let bits = src_bits(args[0], types)?;
            let v = ctx.operand(args[0])?;
            emit_bitreverse(ctx, v, ty, bits)?
        }
        // Saturating add/sub (`llvm.{u,s}{add,sub}.sat`): the wrapping op, then clamp on over/underflow
        // via `select`. Rust's `saturating_add`/`sub` (and slice/capacity math) emit these. Only the
        // native widths (i32/i64) — a narrow saturating width would need width-specific clamp bounds.
        "llvm.uadd.sat" | "llvm.usub.sat" | "llvm.sadd.sat" | "llvm.ssub.sat" => {
            let bits = src_bits(args[0], types)?;
            if bits != 32 && bits != 64 {
                return unsup(format!("`{name}` (only i32/i64 saturating)"));
            }
            let a = ctx.operand(args[0])?;
            let b = ctx.operand(args[1])?;
            let k = |ctx: &mut BlockCtx, v: i64| {
                ctx.push(if ty == IntTy::I64 {
                    Inst::ConstI64(v)
                } else {
                    Inst::ConstI32(v as i32)
                })
            };
            let bin = |ctx: &mut BlockCtx, op: BinOp, a: ValIdx, b: ValIdx| {
                ctx.push(Inst::IntBin { ty, op, a, b })
            };
            match base {
                // unsigned add: carry (s <u a) ⇒ all-ones (UMAX).
                "llvm.uadd.sat" => {
                    let s = bin(ctx, BinOp::Add, a, b);
                    let carry = ctx.push(Inst::IntCmp {
                        ty,
                        op: CmpOp::LtU,
                        a: s,
                        b: a,
                    });
                    let umax = k(ctx, -1);
                    ctx.push(Inst::Select {
                        cond: carry,
                        a: umax,
                        b: s,
                    })
                }
                // unsigned sub: borrow (a <u b) ⇒ 0.
                "llvm.usub.sat" => {
                    let under = ctx.push(Inst::IntCmp {
                        ty,
                        op: CmpOp::LtU,
                        a,
                        b,
                    });
                    let diff = bin(ctx, BinOp::Sub, a, b);
                    let zero = k(ctx, 0);
                    ctx.push(Inst::Select {
                        cond: under,
                        a: zero,
                        b: diff,
                    })
                }
                // signed add/sub: overflow when the sign rule is violated ⇒ clamp to SMIN/SMAX by the
                // sign of `a`. `sadd`: overflow iff `(a^s)&(b^s) < 0`; `ssub`: iff `(a^b)&(a^d) < 0`.
                "llvm.sadd.sat" | "llvm.ssub.sat" => {
                    let is_add = base == "llvm.sadd.sat";
                    let r = bin(ctx, if is_add { BinOp::Add } else { BinOp::Sub }, a, b);
                    let (l, rr) = if is_add {
                        (bin(ctx, BinOp::Xor, a, r), bin(ctx, BinOp::Xor, b, r))
                    } else {
                        (bin(ctx, BinOp::Xor, a, b), bin(ctx, BinOp::Xor, a, r))
                    };
                    let ov_bits = bin(ctx, BinOp::And, l, rr);
                    let zero = k(ctx, 0);
                    let ov = ctx.push(Inst::IntCmp {
                        ty,
                        op: CmpOp::LtS,
                        a: ov_bits,
                        b: zero,
                    });
                    let (smax, smin) = if ty == IntTy::I64 {
                        (k(ctx, i64::MAX), k(ctx, i64::MIN))
                    } else {
                        (k(ctx, i32::MAX as i64), k(ctx, i32::MIN as i64))
                    };
                    let neg = ctx.push(Inst::IntCmp {
                        ty,
                        op: CmpOp::LtS,
                        a,
                        b: zero,
                    });
                    let sat = ctx.push(Inst::Select {
                        cond: neg,
                        a: smin,
                        b: smax,
                    });
                    ctx.push(Inst::Select {
                        cond: ov,
                        a: sat,
                        b: r,
                    })
                }
                _ => unreachable!(),
            }
        }
        _ => unreachable!(),
    };
    Ok(Some(idx))
}

/// Lower `llvm.{u,s}{add,sub,mul}.with.overflow.iN` → the wrapping op plus a computed overflow flag,
/// recorded as a **2-field aggregate** `{result, overflow}` (consumed by `extractvalue 0`/`1`). Rust's
/// checked capacity/index arithmetic (`Vec`/`String`/`Layout::array`) emits these; the overflow flag
/// feeds a branch to `handle_error`/`panicking` (which traps), so the result must be exact and the flag
/// correct in the no-overflow case (it is — the formulas are exact). Returns `true` if handled.
fn lower_overflow_intrinsic(
    ctx: &mut BlockCtx,
    c: &llvm_ir::instruction::Call,
    types: &Types,
) -> Result<bool, Error> {
    let Some(name) = callee_name(c) else {
        return Ok(false);
    };
    let base = name.rsplit_once('.').map_or(name.as_str(), |(b, _)| b);
    // (arith op, signed?) — `None` if not an overflow intrinsic.
    let spec = match base {
        "llvm.uadd.with.overflow" => (BinOp::Add, false),
        "llvm.usub.with.overflow" => (BinOp::Sub, false),
        "llvm.umul.with.overflow" => (BinOp::Mul, false),
        "llvm.sadd.with.overflow" => (BinOp::Add, true),
        "llvm.ssub.with.overflow" => (BinOp::Sub, true),
        "llvm.smul.with.overflow" => (BinOp::Mul, true),
        _ => return Ok(false),
    };
    let (arith, signed) = spec;
    let args: Vec<&Operand> = c.arguments.iter().map(|(a, _)| a).collect();
    let ty = int_ty(val_type(args[0].get_type(types).as_ref())?)?;
    let a = ctx.operand(args[0])?;
    let b = ctx.operand(args[1])?;
    let k = |ctx: &mut BlockCtx, v: i64| {
        ctx.push(if ty == IntTy::I64 {
            Inst::ConstI64(v)
        } else {
            Inst::ConstI32(v as i32)
        })
    };
    let bin = |ctx: &mut BlockCtx, op: BinOp, x: ValIdx, y: ValIdx| {
        ctx.push(Inst::IntBin { ty, op, a: x, b: y })
    };
    let cmp = |ctx: &mut BlockCtx, op: CmpOp, x: ValIdx, y: ValIdx| {
        ctx.push(Inst::IntCmp { ty, op, a: x, b: y })
    };
    let r = bin(ctx, arith, a, b);
    let overflow = match (arith, signed) {
        // unsigned add: wrapped sum is below an operand.
        (BinOp::Add, false) => cmp(ctx, CmpOp::LtU, r, a),
        // unsigned sub: borrow ⇔ a <u b.
        (BinOp::Sub, false) => cmp(ctx, CmpOp::LtU, a, b),
        // signed add: operands agree in sign but the result disagrees ⇔ `((a^r)&(b^r)) < 0`.
        (BinOp::Add, true) => {
            let ar = bin(ctx, BinOp::Xor, a, r);
            let br = bin(ctx, BinOp::Xor, b, r);
            let m = bin(ctx, BinOp::And, ar, br);
            let zero = k(ctx, 0);
            cmp(ctx, CmpOp::LtS, m, zero)
        }
        // signed sub: `((a^b)&(a^r)) < 0`.
        (BinOp::Sub, true) => {
            let ab = bin(ctx, BinOp::Xor, a, b);
            let ar = bin(ctx, BinOp::Xor, a, r);
            let m = bin(ctx, BinOp::And, ab, ar);
            let zero = k(ctx, 0);
            cmp(ctx, CmpOp::LtS, m, zero)
        }
        // mul: `a != 0 && (r / a) != b`, dividing by a zero-guarded `a` so the check never traps.
        (BinOp::Mul, _) => {
            let zero = k(ctx, 0);
            let one = k(ctx, 1);
            let a_nz = cmp(ctx, CmpOp::Ne, a, zero);
            let safe_a = ctx.push(Inst::Select {
                cond: a_nz,
                a,
                b: one,
            });
            let q = bin(
                ctx,
                if signed { BinOp::DivS } else { BinOp::DivU },
                r,
                safe_a,
            );
            let q_ne = cmp(ctx, CmpOp::Ne, q, b);
            // `a_nz & q_ne` (both `i32` 0/1) — a plain bitwise AND of the boolean flags.
            ctx.push(Inst::IntBin {
                ty: IntTy::I32,
                op: BinOp::And,
                a: a_nz,
                b: q_ne,
            })
        }
        _ => unreachable!(),
    };
    if let Some(dest) = &c.dest {
        if let Some(&vid) = ctx.s.name2id.get(dest) {
            ctx.agg.insert(vid, vec![r, overflow]);
        }
    }
    Ok(true)
}

/// Lower `llvm.vector.reduce.{add,mul,and,or,xor,smax,smin,umax,umin}.v4i32` — the horizontal reduction
/// `-O2` auto-vectorization emits to close a reduction loop (the `i32x4` accumulator → a scalar). No
/// SVM reduce op, so it is unrolled: extract the 4 lanes and fold them with the scalar op (`add`/`mul`/
/// `and`/`or`/`xor` via `IntBin`; `min`/`max` via `cmp`+`select`). Returns the scalar `i32`, or `None`
/// if not a (supported) reduce. (Only `i32x4` for now; wider/float reductions are a later slice.)
fn lower_vector_reduce(
    ctx: &mut BlockCtx,
    c: &llvm_ir::instruction::Call,
    types: &Types,
) -> Result<Option<ValIdx>, Error> {
    let Some(name) = callee_name(c) else {
        return Ok(None);
    };
    let Some(rest) = name.strip_prefix("llvm.vector.reduce.") else {
        return Ok(None);
    };
    let kind = rest.split('.').next().unwrap_or(""); // "add"/"mul"/… (before the `.vNiM` suffix)
    let vec_op = &c.arguments[0].0;
    let vty = vec_op.get_type(types);
    let Some(shape) = vec128_shape(vty.as_ref()) else {
        return unsup(format!("vector.reduce on non-128-bit vector ({kind})"));
    };
    if shape.is_float() {
        return unsup(format!("float vector.reduce.{kind} (later slice)"));
    }
    // The fold runs in the lane scalar type: `i32` for i8/i16/i32 lanes (narrow lanes widen, §3b —
    // the result is consumed truncated to the lane width, and modular add/mul stays exact), `i64`
    // for i64 lanes. A signed min/max needs sign-extended narrow lanes so the `i32` compare orders
    // them correctly; every other reduce is bit-identical under zero-extension.
    let lane_ty = int_ty(shape.lane_val())?;
    let signed = matches!(kind, "smax" | "smin");
    let v = ctx.operand(vec_op)?;
    let lanes: Vec<ValIdx> = (0..shape.lanes())
        .map(|l| {
            ctx.push(Inst::ExtractLane {
                shape,
                lane: l,
                signed,
                a: v,
            })
        })
        .collect();
    let fold_bin = |ctx: &mut BlockCtx, op: BinOp| {
        let mut acc = lanes[0];
        for &l in &lanes[1..] {
            acc = ctx.push(Inst::IntBin {
                ty: lane_ty,
                op,
                a: acc,
                b: l,
            });
        }
        acc
    };
    let fold_minmax = |ctx: &mut BlockCtx, cmp: CmpOp| {
        let mut acc = lanes[0];
        for &l in &lanes[1..] {
            let cond = ctx.push(Inst::IntCmp {
                ty: lane_ty,
                op: cmp,
                a: acc,
                b: l,
            });
            acc = ctx.push(Inst::Select { cond, a: acc, b: l });
        }
        acc
    };
    let r = match kind {
        "add" => fold_bin(ctx, BinOp::Add),
        "mul" => fold_bin(ctx, BinOp::Mul),
        "and" => fold_bin(ctx, BinOp::And),
        "or" => fold_bin(ctx, BinOp::Or),
        "xor" => fold_bin(ctx, BinOp::Xor),
        "smax" => fold_minmax(ctx, CmpOp::GtS),
        "smin" => fold_minmax(ctx, CmpOp::LtS),
        "umax" => fold_minmax(ctx, CmpOp::GtU),
        "umin" => fold_minmax(ctx, CmpOp::LtU),
        other => return unsup(format!("vector.reduce.{other}")),
    };
    Ok(Some(r))
}

/// Emit an inline byte-reverse of `v` (`ty`-wide, `nbytes` bytes): OR together, for each source byte
/// `i`, `((v >> 8*i) & 0xff) << 8*(nbytes-1-i)`. Lowers `llvm.bswap.{i16,i32,i64}`.
fn emit_bswap(ctx: &mut BlockCtx, v: ValIdx, ty: IntTy, nbytes: u64) -> ValIdx {
    let kof = |ctx: &mut BlockCtx, k: i64| {
        ctx.push(if ty == IntTy::I64 {
            Inst::ConstI64(k)
        } else {
            Inst::ConstI32(k as i32)
        })
    };
    let ff = kof(ctx, 0xff);
    let mut acc: Option<ValIdx> = None;
    for i in 0..nbytes {
        let shifted = if i == 0 {
            v
        } else {
            let s = kof(ctx, (8 * i) as i64);
            ctx.push(Inst::IntBin {
                ty,
                op: BinOp::ShrU,
                a: v,
                b: s,
            })
        };
        let byte = ctx.push(Inst::IntBin {
            ty,
            op: BinOp::And,
            a: shifted,
            b: ff,
        });
        let dst_sh = 8 * (nbytes - 1 - i);
        let placed = if dst_sh == 0 {
            byte
        } else {
            let d = kof(ctx, dst_sh as i64);
            ctx.push(Inst::IntBin {
                ty,
                op: BinOp::Shl,
                a: byte,
                b: d,
            })
        };
        acc = Some(match acc {
            None => placed,
            Some(a) => ctx.push(Inst::IntBin {
                ty,
                op: BinOp::Or,
                a,
                b: placed,
            }),
        });
    }
    acc.unwrap_or(v)
}

/// Reverse the low `bits` bits of `v` (a power-of-2 width: 8/16/32/64) with the classic log-N swap
/// network: `((v & m_s) << s) | ((v >> s) & m_s)` for s = 1,2,…,bits/2, where `m_s` selects the
/// even-indexed `s`-bit groups (`s` ones, `s` zeros, repeating). The on-ramp keeps narrow integers
/// zero-extended in their container (§3b), so a 16-bit value reverses cleanly into the low 16 bits.
/// Lowers `llvm.bitreverse.{i8,i16,i32,i64}`; a non-power-of-2 width is `Unsupported` (fail-closed).
fn emit_bitreverse(ctx: &mut BlockCtx, v: ValIdx, ty: IntTy, bits: u32) -> Result<ValIdx, Error> {
    if !matches!(bits, 8 | 16 | 32 | 64) {
        return unsup(format!("llvm.bitreverse.i{bits} (non-power-of-2 width)"));
    }
    let kof = |ctx: &mut BlockCtx, k: u64| {
        ctx.push(if ty == IntTy::I64 {
            Inst::ConstI64(k as i64)
        } else {
            Inst::ConstI32(k as i32)
        })
    };
    let mut cur = v;
    let mut s = 1u32;
    while s < bits {
        // `m_s`: `s` ones then `s` zeros, repeating across the low `bits` bits.
        let mut mask = 0u64;
        let mut i = 0u32;
        while i < bits {
            for b in i..(i + s).min(bits) {
                mask |= 1u64 << b;
            }
            i += 2 * s;
        }
        let m = kof(ctx, mask);
        let sc = kof(ctx, s as u64);
        // (v & m_s) << s  |  (v >> s) & m_s
        let lo = ctx.push(Inst::IntBin {
            ty,
            op: BinOp::And,
            a: cur,
            b: m,
        });
        let lo = ctx.push(Inst::IntBin {
            ty,
            op: BinOp::Shl,
            a: lo,
            b: sc,
        });
        let hi = ctx.push(Inst::IntBin {
            ty,
            op: BinOp::ShrU,
            a: cur,
            b: sc,
        });
        let hi = ctx.push(Inst::IntBin {
            ty,
            op: BinOp::And,
            a: hi,
            b: m,
        });
        cur = ctx.push(Inst::IntBin {
            ty,
            op: BinOp::Or,
            a: lo,
            b: hi,
        });
        s *= 2;
    }
    Ok(cur)
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
    // runtime loop helper (`__svm_memset`/`__svm_memcpy`/`__svm_memmove`). Variable-length `memmove`
    // routes to the overlap-safe (direction-aware) `__svm_memmove`.
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
                let f = ctx.helpers.memmove.expect("memmove helper synthesized");
                let dst = ctx.operand(args[0])?;
                let src = ctx.operand(args[1])?;
                ctx.push_effect(Inst::Call {
                    func: f,
                    args: vec![dst, src, len],
                });
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

/// Lower a float math intrinsic call to inline float ops, returning its result index. `llvm.fma`
/// (IEEE-required fused) lowers to the shared fused-FMA op (`Inst::Fma`/`VFma`, the same op the wasm
/// `relaxed_madd` emits — interp `mul_add` == JIT `fma`, and bit-equal to native libm `fma()`);
/// `llvm.fmuladd` (contractible) stays unfused `fmul`+`fadd`, bit-equal to baseline native (no HW FMA).
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
    // A 128-bit float vector (`<4 x float>`/`<2 x double>`) → native `v128` lane-wise ops (§17) in
    // the operand's shape. `llvm.fma` → the shared fused `Inst::VFma`; `llvm.fmuladd` → unfused mul+add.
    if let Some(shape) = args
        .first()
        .and_then(|a| vec128_shape(a.get_type(types).as_ref()))
        .filter(|s| s.is_float())
    {
        use svm_ir::{VFloatBinOp as VB, VFloatUnOp as VU};
        let un = |ctx: &mut BlockCtx, op: VU| -> Result<ValIdx, Error> {
            let a = ctx.operand(args[0])?;
            Ok(ctx.push(Inst::VFloatUn { shape, op, a }))
        };
        let bin = |ctx: &mut BlockCtx, op: VB| -> Result<ValIdx, Error> {
            let a = ctx.operand(args[0])?;
            let b = ctx.operand(args[1])?;
            Ok(ctx.push(Inst::VFloatBin { shape, op, a, b }))
        };
        let idx = match base {
            "llvm.sqrt" => un(ctx, VU::Sqrt)?,
            "llvm.fabs" => un(ctx, VU::Abs)?,
            "llvm.minnum" | "llvm.minimum" => bin(ctx, VB::Min)?,
            "llvm.maxnum" | "llvm.maximum" => bin(ctx, VB::Max)?,
            // `llvm.fma` is IEEE-required fused → the shared fused-FMA primitive (`Inst::VFma`; the
            // same op the wasm frontend's `relaxed_madd` emits, interp `mul_add` == JIT `fma`). This
            // also matches native's libm `fma()`. `llvm.fmuladd` is *contractible*: on a baseline
            // target (no hardware FMA) native lowers it to mul+add, so we keep it unfused to stay
            // bit-equal to the native oracle.
            "llvm.fma" => {
                let a = ctx.operand(args[0])?;
                let bb = ctx.operand(args[1])?;
                let cc = ctx.operand(args[2])?;
                ctx.push(Inst::VFma {
                    shape,
                    neg: false,
                    a,
                    b: bb,
                    c: cc,
                })
            }
            "llvm.fmuladd" => {
                let prod = bin(ctx, VB::Mul)?;
                let cc = ctx.operand(args[2])?;
                ctx.push(Inst::VFloatBin {
                    shape,
                    op: VB::Add,
                    a: prod,
                    b: cc,
                })
            }
            _ => return unsup(format!("vector float intrinsic `{base}`")),
        };
        return Ok(Some(idx));
    }
    // A `<2 x float>` float intrinsic (the auto-vectorizer's `fmuladd.v2f32` etc.) has no native
    // svm-ir op — scalarize: explode the two `f32` lanes, apply the scalar float op per lane, repack
    // the packed-`i64` vec2 (`vec_pack`). The single packed result is returned for `c.dest` to bind.
    if is_vec2f(args[0].get_type(types).as_ref()) {
        let lane_ty = FloatTy::F32;
        let la = vec_explode(ctx, args[0], types, false)?;
        let lane = |ctx: &mut BlockCtx, op: FUnOp, k: usize| {
            ctx.push(Inst::FUn {
                ty: lane_ty,
                op,
                a: la[k],
            })
        };
        let (r0, r1) = match base {
            "llvm.sqrt" => (lane(ctx, FUnOp::Sqrt, 0), lane(ctx, FUnOp::Sqrt, 1)),
            "llvm.fabs" => (lane(ctx, FUnOp::Abs, 0), lane(ctx, FUnOp::Abs, 1)),
            "llvm.floor" => (lane(ctx, FUnOp::Floor, 0), lane(ctx, FUnOp::Floor, 1)),
            "llvm.ceil" => (lane(ctx, FUnOp::Ceil, 0), lane(ctx, FUnOp::Ceil, 1)),
            "llvm.trunc" => (lane(ctx, FUnOp::Trunc, 0), lane(ctx, FUnOp::Trunc, 1)),
            "llvm.rint" | "llvm.nearbyint" | "llvm.roundeven" => {
                (lane(ctx, FUnOp::Nearest, 0), lane(ctx, FUnOp::Nearest, 1))
            }
            "llvm.minnum" | "llvm.minimum" | "llvm.maxnum" | "llvm.maximum" | "llvm.copysign" => {
                let lb = vec_explode(ctx, args[1], types, false)?;
                let op = match base {
                    "llvm.minnum" | "llvm.minimum" => FBinOp::Min,
                    "llvm.maxnum" | "llvm.maximum" => FBinOp::Max,
                    _ => FBinOp::Copysign,
                };
                let bin = |ctx: &mut BlockCtx, k: usize| {
                    ctx.push(Inst::FBin {
                        ty: lane_ty,
                        op,
                        a: la[k],
                        b: lb[k],
                    })
                };
                (bin(ctx, 0), bin(ctx, 1))
            }
            // `llvm.fma` → fused per lane (`Inst::Fma`, matches native libm); `llvm.fmuladd` → unfused
            // mul+add per lane (matches baseline native, no hardware FMA). See the vec128 path.
            "llvm.fma" => {
                let lb = vec_explode(ctx, args[1], types, false)?;
                let lc = vec_explode(ctx, args[2], types, false)?;
                let fma = |ctx: &mut BlockCtx, k: usize| {
                    ctx.push(Inst::Fma {
                        ty: lane_ty,
                        a: la[k],
                        b: lb[k],
                        c: lc[k],
                    })
                };
                (fma(ctx, 0), fma(ctx, 1))
            }
            "llvm.fmuladd" => {
                let lb = vec_explode(ctx, args[1], types, false)?;
                let lc = vec_explode(ctx, args[2], types, false)?;
                let madd = |ctx: &mut BlockCtx, k: usize| {
                    let prod = ctx.push(Inst::FBin {
                        ty: lane_ty,
                        op: FBinOp::Mul,
                        a: la[k],
                        b: lb[k],
                    });
                    ctx.push(Inst::FBin {
                        ty: lane_ty,
                        op: FBinOp::Add,
                        a: prod,
                        b: lc[k],
                    })
                };
                (madd(ctx, 0), madd(ctx, 1))
            }
            _ => return unsup(format!("vec2 float intrinsic `{base}`")),
        };
        return Ok(Some(ctx.vec_pack(r0, r1, ValType::F32)));
    }
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
        // `llvm.fma` → the shared fused-FMA primitive (`Inst::Fma`, matches native libm `fma()`).
        "llvm.fma" => {
            let a = ctx.operand(args[0])?;
            let b = ctx.operand(args[1])?;
            let c = ctx.operand(args[2])?;
            ctx.push(Inst::Fma { ty, a, b, c })
        }
        // `llvm.fmuladd` is contractible → unfused mul+add, bit-equal to baseline native (no HW FMA).
        "llvm.fmuladd" => {
            let a = ctx.operand(args[0])?;
            let b = ctx.operand(args[1])?;
            let prod = ctx.push(Inst::FBin {
                ty,
                op: FBinOp::Mul,
                a,
                b,
            });
            let c = ctx.operand(args[2])?;
            ctx.push(Inst::FBin {
                ty,
                op: FBinOp::Add,
                a: prod,
                b: c,
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
            // `llvm.va_end` only marks the end of a `va_list` traversal — no runtime state to tear
            // down in our overflow-only varargs ABI (§varargs); `va_start`/`va_copy` are lowered.
            || s.starts_with("llvm.va_end")
            // Alias-analysis metadata hints (no runtime effect) — e.g. clang's `restrict` scopes.
            || s.starts_with("llvm.experimental.noalias.scope.decl");
    }
    false
}

/// Lower a varargs `llvm.va_start` / `llvm.va_copy` for the **overflow-only varargs ABI** (§varargs).
/// Returns `Ok(true)` if `name` named a handled varargs intrinsic (`va_end` is dropped earlier in
/// `is_droppable_call`).
///
/// `va_start(list)` initializes the System V AMD64 `__va_list_tag` at `list` so that clang's already-
/// lowered `va_arg` *always* takes the memory (`overflow_arg_area`) branch: `gp_offset = 48` and
/// `fp_offset = 176` both sit at/over their register-save thresholds, so the register path is dead and
/// no `reg_save_area` need be synthesized. `overflow_arg_area` is the caller-deposited pointer read
/// from this frame's reserved slot (`sp + 0`); `reg_save_area` is left null. Tag layout (x86-64):
/// `i32 gp_offset @0`, `i32 fp_offset @4`, `ptr overflow_arg_area @8`, `ptr reg_save_area @16` — the
/// two `i32` offsets are written as one packed `i64` at `@0`.
///
/// `va_copy(dst, src)` byte-copies the 24-byte tag.
fn lower_va_intrinsic(
    ctx: &mut BlockCtx,
    c: &llvm_ir::instruction::Call,
    name: &str,
) -> Result<bool, Error> {
    if name.starts_with("llvm.va_start") {
        let list = ctx.operand(&c.arguments[0].0)?;
        let sp = ctx.sp()?;
        // overflow_arg_area = *(sp + 0): the area pointer the caller deposited at our frame base.
        let area = ctx.push(Inst::Load {
            op: svm_ir::LoadOp::I64,
            addr: sp,
            offset: 0,
            align: 0,
        });
        // gp_offset (48) | fp_offset (176) << 32 — both past their thresholds → memory branch only.
        let off_word = ctx.const_i64(48 | (176i64 << 32));
        ctx.push_effect(Inst::Store {
            op: svm_ir::StoreOp::I64,
            addr: list,
            value: off_word,
            offset: 0,
            align: 0,
        });
        ctx.push_effect(Inst::Store {
            op: svm_ir::StoreOp::I64,
            addr: list,
            value: area,
            offset: 8,
            align: 0,
        });
        let zero = ctx.const_i64(0);
        ctx.push_effect(Inst::Store {
            op: svm_ir::StoreOp::I64,
            addr: list,
            value: zero,
            offset: 16,
            align: 0,
        });
        return Ok(true);
    }
    if name.starts_with("llvm.va_copy") {
        let dst = ctx.operand(&c.arguments[0].0)?;
        let src = ctx.operand(&c.arguments[1].0)?;
        for off in [0u64, 8, 16] {
            let w = ctx.push(Inst::Load {
                op: svm_ir::LoadOp::I64,
                addr: src,
                offset: off,
                align: 0,
            });
            ctx.push_effect(Inst::Store {
                op: svm_ir::StoreOp::I64,
                addr: dst,
                value: w,
                offset: off,
                align: 0,
            });
        }
        return Ok(true);
    }
    Ok(false)
}

/// Is this a call to a Rust **panic/abort lang item**? Under `-C panic=abort` the panic entry points
/// (`core::panicking::*` — `panic`, `panic_fmt`, `panic_const_*`, `panic_bounds_check` — plus the
/// `unwrap`/`expect`/slice-index failure helpers) are `-> !` and abort the process, and they are
/// **external** (precompiled libcore, never in the bitcode). A real Rust program is littered with
/// these on its non-elidable panic paths (div-by-zero, bounds, overflow), so a call to one lowers to
/// a **trap** (the SVM abort, §3b/§5): the on-ramp drops the call and relies on the `unreachable` that
/// LLVM always places after a `noreturn` call (already lowered to a trap). Gated by the caller on the
/// name being an *undefined external* (a guest-defined function of a matching name is a real call).
fn is_rust_abort_call(name: &str) -> bool {
    name.contains("panicking")
        || name.contains("unwrap_failed")
        || name.contains("expect_failed")
        // The whole `core::slice::index` panic family — `slice_index_order_fail`,
        // `slice_{start,end}_index_len_fail`, `slice_index_len_fail` — all `-> !`. Matching the
        // narrower `slice_index` substring missed `slice_end_index_len_fail` (BTreeMap, slicing).
        || (name.contains("slice") && name.contains("_fail"))
        || name.contains("panic_cannot_unwind")
        // `alloc`'s out-of-memory / capacity-overflow aborts (`alloc::raw_vec::handle_error`,
        // `alloc::alloc::handle_alloc_error`) — also `-> !` external lang items under `panic=abort`.
        || name.contains("handle_error")
        || name.contains("alloc_error")
        // `RefCell` borrow-conflict aborts (`core::cell::panic_already_borrowed` /
        // `…_already_mutably_borrowed`) — `-> !` cold lang items the specializer's `RefCell<OutlineState>`
        // borrows pull in. Like the slice/alloc family: external, never in the bitcode, lower to a trap.
        || name.contains("panic_already")
}

/// Lower a `<setjmp.h>` non-local jump to the `SetJmp`/`LongJmp` core ops. `setjmp`/`_setjmp`/
/// `sigsetjmp` take the `jmp_buf` pointer as their first argument (`sigsetjmp`'s `savesigs` is
/// ignored) → `Inst::SetJmp { buf }`, yielding the `i32` result (0 on the direct call, the long-jump
/// value on re-entry); `longjmp`/`siglongjmp` take `(jmp_buf, i32 val)` → `Inst::LongJmp { buf, val }`
/// (no result — LLVM follows it with `unreachable`). Returns `Ok(true)` if it handled the call. Gated
/// external by the caller, so a guest definition of the same name shadows it.
fn lower_setjmp_call(
    ctx: &mut BlockCtx,
    c: &llvm_ir::instruction::Call,
    name: &str,
) -> Result<bool, Error> {
    let arg = |ctx: &mut BlockCtx, i: usize| -> Result<ValIdx, Error> {
        let op = c
            .arguments
            .get(i)
            .map(|(o, _)| o)
            .ok_or_else(|| Error::Unsupported(format!("{name} missing argument {i}")))?;
        ctx.operand(op)
    };
    match name {
        "setjmp" | "_setjmp" | "sigsetjmp" => {
            let buf = arg(ctx, 0)?;
            let r = ctx.push(Inst::SetJmp { buf });
            if let Some(dest) = &c.dest {
                if let Some(&vid) = ctx.s.name2id.get(dest) {
                    ctx.idx_of.insert(vid, r);
                }
            }
            Ok(true)
        }
        "longjmp" | "siglongjmp" => {
            let buf = arg(ctx, 0)?;
            let val = arg(ctx, 1)?;
            ctx.push(Inst::LongJmp { buf, val });
            Ok(true)
        }
        _ => Ok(false),
    }
}

/// The **noreturn EH unwinders** — `__cxa_throw` / `__cxa_rethrow` / `_Unwind_Resume` — shared by
/// the `call` form ([`lower_eh_call`]) and the `invoke` form ([`lower_invoke`]). clang emits them as
/// an `invoke` (rather than a `call`) whenever the throw/rethrow sits inside an active cleanup scope
/// (e.g. a `catch` body, whose unwind edge runs `__cxa_end_catch`); in our setjmp/longjmp model they
/// never return and long-jump straight into the enclosing handler, so *both* invoke successors are
/// dead. `__cxa_throw` first stores the object ptr + the thrown type's selector; the bare re-unwinds
/// reuse the already-stored `cur_exn`/`cur_sel`. Generic over the per-arg attribute so it accepts a
/// `Call`'s and an `Invoke`'s `arguments` alike. `Ok(false)` ⇒ not an unwinder (fall through).
fn lower_eh_unwinder<A>(
    ctx: &mut BlockCtx,
    name: &str,
    arguments: &[(Operand, A)],
) -> Result<bool, Error> {
    // `eh_base()` is consulted only once the name is a known unwinder — a non-EH call/invoke must
    // fall through to `Ok(false)` without requiring a reserved EH region.
    match name {
        "__cxa_throw" => {
            let base = ctx.eh_base()?;
            let exn = ctx.operand_i64(&arguments[0].0)?;
            let exn_addr = ctx.const_i64((base + EH_EXN_O) as i64);
            ctx.push_effect(Inst::Store {
                op: StoreOp::I64,
                addr: exn_addr,
                value: exn,
                offset: 0,
                align: 8,
            });
            let tid = ctx.eh_typeid(&arguments[1].0)?;
            let sel = ctx.push(Inst::ConstI32(tid as i32));
            let sel_addr = ctx.const_i64((base + EH_SEL_O) as i64);
            ctx.push_effect(Inst::Store {
                op: StoreOp::I32,
                addr: sel_addr,
                value: sel,
                offset: 0,
                align: 4,
            });
            // The third argument is the object's destructor (a `void(T*)` funcref, or null for a
            // trivially-destructible type). Stash it so `__cxa_end_catch` can run it on the exception
            // object when the catching handler completes (`__svm_eh_destroy` guards the null case).
            let dtor = ctx.operand_i64(&arguments[2].0)?;
            let dtor_addr = ctx.const_i64((base + EH_DTOR_O) as i64);
            ctx.push_effect(Inst::Store {
                op: StoreOp::I64,
                addr: dtor_addr,
                value: dtor,
                offset: 0,
                align: 8,
            });
            ctx.eh_unwind(base)?;
            Ok(true)
        }
        "__cxa_rethrow" | "_Unwind_Resume" => {
            let base = ctx.eh_base()?;
            ctx.eh_unwind(base)?;
            Ok(true)
        }
        _ => Ok(false),
    }
}

/// Lower the Itanium C++ exception-handling runtime (`__cxa_*`) and `llvm.eh.typeid.for`. Exception
/// state lives in the reserved EH region ([`BlockCtx::eh_base`]):
/// - `__cxa_allocate_exception(size)` → the fixed object scratch slot (`EH_EXNOBJ_O`).
/// - `__cxa_throw(obj, ti, dtor)` → store the object ptr + the thrown type's selector, then unwind
///   ([`BlockCtx::eh_unwind`]) into the enclosing `invoke`-installed handler (noreturn; the IR's
///   trailing `unreachable` is dead). When the throw sits in a cleanup scope clang emits it as an
///   `invoke` instead — [`lower_invoke`] routes that form through the shared [`lower_eh_unwinder`].
/// - `__cxa_rethrow()` / `_Unwind_Resume` → re-unwind with the current (already-stored) selector.
/// - `__cxa_begin_catch(p)` → the caught object pointer (identity for a primitive/pointer catch).
/// - `__cxa_end_catch()` → a no-op (no object destruction in the supported scope).
/// - `llvm.eh.typeid.for(ti)` → the operand type's compile-time id (the selector the catch compares).
fn lower_eh_call(
    ctx: &mut BlockCtx,
    c: &llvm_ir::instruction::Call,
    name: &str,
) -> Result<bool, Error> {
    // The noreturn unwinders (`__cxa_throw` / `__cxa_rethrow` / `_Unwind_Resume`) are shared with the
    // `invoke` form; the IR's trailing `unreachable` is the dead terminator after the long-jump.
    if lower_eh_unwinder(ctx, name, &c.arguments)? {
        return Ok(true);
    }
    match name {
        "__cxa_allocate_exception" => {
            let base = ctx.eh_base()?;
            let p = ctx.const_i64((base + EH_EXNOBJ_O) as i64);
            ctx.bind_dest(&c.dest, p);
            Ok(true)
        }
        // `__cxa_begin_catch(p)` officially enters the handler; `__cxa_get_exception_ptr(p)` fetches the
        // object pointer *before* a catch-by-value copy-construct (which may itself throw). Both return
        // the exception object pointer — identity in our model (it already points into the EH region).
        "__cxa_begin_catch" | "__cxa_get_exception_ptr" => {
            let p = ctx.operand_i64(&c.arguments[0].0)?;
            ctx.bind_dest(&c.dest, p);
            Ok(true)
        }
        "__cxa_end_catch" => {
            // The handler is complete: destroy the caught exception object by running its registered
            // destructor (stashed at `EH_DTOR_O` by `__cxa_throw`; `0` for a trivially-destructible
            // type). The null guard + indirect call live in the `__svm_eh_destroy` helper because this
            // lowering runs in effect position and cannot branch. `need_eh_destroy` guarantees the
            // helper exists whenever a module reaches here; fall back to a no-op if somehow absent.
            let Some(helper) = ctx.helpers.eh_destroy else {
                return Ok(true);
            };
            let base = ctx.eh_base()?;
            let exn_addr = ctx.const_i64((base + EH_EXN_O) as i64);
            let exn = ctx.push(Inst::Load {
                op: LoadOp::I64,
                addr: exn_addr,
                offset: 0,
                align: 8,
            });
            let dtor_addr = ctx.const_i64((base + EH_DTOR_O) as i64);
            let dtor = ctx.push(Inst::Load {
                op: LoadOp::I64,
                addr: dtor_addr,
                offset: 0,
                align: 8,
            });
            let sp = ctx.sp()?;
            let fs = ctx.const_i64(ctx.frame_size as i64);
            let callee_sp = ctx.add_i64(sp, fs);
            ctx.push_effect(Inst::Call {
                func: helper,
                args: vec![callee_sp, exn, dtor],
            });
            // Clear the dtor slot so a later `__cxa_end_catch` without an intervening throw cannot
            // re-destroy the (now-dead) object; the next `__cxa_throw` re-stores it.
            let zero = ctx.const_i64(0);
            ctx.push_effect(Inst::Store {
                op: StoreOp::I64,
                addr: dtor_addr,
                value: zero,
                offset: 0,
                align: 8,
            });
            Ok(true)
        }
        "llvm.eh.typeid.for" => {
            // Subtype-aware selector match (the Itanium personality's `__do_catch`). clang compares the
            // landing-pad selector — the *thrown* type's id — against this value with `icmp eq` to pick
            // a catch clause. Return the live selector when the thrown type *is-a* the clause type (so
            // the compare is true), else a sentinel (`-1`) that no real id equals. The match set — the
            // thrown ids the clause catches — is precomputed from the typeinfo base-chains, so this is
            // what lets `catch (Base&)` catch a thrown `Derived` (and an exact match still works: a
            // type is in its own match set). A clause type that is never thrown has an empty match set
            // and folds to the sentinel — it simply never matches.
            let name = global_name_of(&c.arguments[0].0).ok_or_else(|| {
                Error::Unsupported("eh.typeid.for operand is not a global".into())
            })?;
            let base = ctx.eh_base()?;
            let sel_addr = ctx.const_i64((base + EH_SEL_O) as i64);
            let sel = ctx.push(Inst::Load {
                op: LoadOp::I32,
                addr: sel_addr,
                offset: 0,
                align: 4,
            });
            let mut result = ctx.push(Inst::ConstI32(-1)); // sentinel: never equals a (1-based) id
            if let Some(matchset) = ctx.helpers.eh_subtype_ids.get(&name).cloned() {
                for t in matchset {
                    let tc = ctx.push(Inst::ConstI32(t as i32));
                    let eq = ctx.push(Inst::IntCmp {
                        ty: IntTy::I32,
                        op: CmpOp::Eq,
                        a: sel,
                        b: tc,
                    });
                    result = ctx.push(Inst::Select {
                        cond: eq,
                        a: sel,
                        b: result,
                    });
                }
            }
            ctx.bind_dest(&c.dest, result);
            Ok(true)
        }
        _ => Ok(false),
    }
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
        // `indirectbr`'s address operand is a use (the loaded label → the `br_table` index).
        LTerm::IndirectBr(ib) => Ok(one(&ib.operand)),
        LTerm::Unreachable(_) => Ok(Vec::new()),
        // An `invoke` uses its argument operands (and, for an indirect callee, the function pointer);
        // the result is bound in the synthetic call block. Its two edges' block-args come from block
        // parameters, like any branch.
        LTerm::Invoke(inv) => {
            let mut v = inv.function.as_ref().right().map(one).unwrap_or_default();
            v.extend(inv.arguments.iter().flat_map(|(o, _)| one(o)));
            Ok(v)
        }
        // `resume` re-throws the current (slot-held) exception — no scalar operand use.
        LTerm::Resume(_) => Ok(Vec::new()),
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
        // An `invoke`'s (non-void) result is defined by this block (the synthetic call block on the
        // normal edge), so record it as a def here — it threads out to the normal successor.
        if let LTerm::Invoke(inv) = &bb.term {
            if let Some(vid) = id(&inv.result) {
                defs[bi].insert(vid);
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
        // `indirectbr` enumerates its full destination set (`possible_dests`) — every one is a
        // successor, so liveness threads each target's live-ins out of this block (the `br_table`
        // edges then supply them).
        LTerm::IndirectBr(ib) => ib.possible_dests.iter().map(b).collect(),
        LTerm::Ret(_) | LTerm::Unreachable(_) => Ok(Vec::new()),
        // An `invoke`'s two edges (normal return / unwind to the landing pad) are both successors —
        // liveness threads each target's live-ins out of this block.
        LTerm::Invoke(inv) => Ok(vec![b(&inv.return_label)?, b(&inv.exception_label)?]),
        // `resume` leaves the function (re-throws to the runtime), so it has no successors.
        LTerm::Resume(_) => Ok(Vec::new()),
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
    /// Constant global name → its bytes (for parsing a `printf` constant format string at translate
    /// time — the format engine reads `@.str`'s content here, never at runtime).
    gbytes: &'a HashMap<String, Vec<u8>>,
    /// Synthesized mem-loop helper indices (for a variable-length `memset`/`memcpy`).
    helpers: Helpers,
    /// The module's type table — for resolving a constexpr-GEP operand's strides in [`operand`].
    types: &'a Types,
    /// This function's **defined**-function index (the `blockaddr.rs` `func_idx` key space — i.e.
    /// position among defined functions, *before* any synthesized `_start` shift). Used only to look
    /// up an operand-position `blockaddress` in `blockaddrs.phi`.
    func_idx: u32,
    /// Recovered `blockaddress` labels ([`blockaddr`]) — the `phi` map resolves an operand-position
    /// (φ-threaded) blockaddress to its target block index.
    blockaddrs: Option<&'a blockaddr::BlockAddrs>,
    insts: Vec<Inst>,
    idx_of: HashMap<ValueId, ValIdx>,
    /// Aggregate SSA values (a small by-value struct), tracked field-wise: value-id → its scalar
    /// fields' block-local indices. Built by a multi-result `call`/`insertvalue`, read by
    /// `extractvalue`/`ret` (§3a multi-result). Assumed not to cross block boundaries (clang's
    /// register-coercion pattern produces and consumes them in one block).
    agg: HashMap<ValueId, Vec<ValIdx>>,
    /// Wide-vector SSA values (I2 step 1): value-id → its legalized parts, ordered
    /// `[chunk_0 … chunk_{C-1}, tail_0 … tail_{T-1}]` (`C` full `v128`s then `T` lane scalars, per
    /// the value's [`WideLayout`] in `s.wide`). The vector analog of `agg` — one LLVM value is
    /// several IR values — but, unlike `agg`, these *do* cross block boundaries (a vectorized loop's
    /// wide accumulator), fanned out into per-part block params by `block_params`/`branch_args`.
    wide_vals: HashMap<ValueId, Vec<ValIdx>>,
    next_val: ValIdx,
    /// Set true only while lowering a block's final instruction when it is a tail-position call
    /// (`tail`/`musttail` + the block's `ret` returns exactly its result). The direct/indirect call
    /// lowering then emits a `ReturnCall`/`ReturnCallIndirect` *terminator* into `pending_tail`
    /// instead of a body `call` + result bind.
    tail_return: bool,
    /// A `ReturnCall(Indirect)` terminator produced by the tail-position call above; consumed by
    /// `translate_block` in place of lowering the `ret`.
    pending_tail: Option<Terminator>,
    /// When set (during `snprintf` lowering), the shared format engine's [`BlockCtx::emit_write`]
    /// copies each formatted run into this destination buffer (bounded by `size`, advancing `offset`)
    /// instead of writing it to `Stream`/stdout — so `snprintf` reuses the entire `printf` formatter.
    fmt_sink: Option<FmtSink>,
}

/// The `snprintf` destination for the redirected [`BlockCtx::emit_write`] sink (§printf): the buffer
/// base, its `size` bound, and the running write `offset` (a runtime SSA value threaded across the
/// per-segment writes; its final value is `snprintf`'s return — the would-be length).
#[derive(Clone, Copy)]
struct FmtSink {
    dest: ValIdx,
    size: ValIdx,
    offset: ValIdx,
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

    /// Resolve a struct-typed operand (field types `ftys`) to its per-field block-local indices, for
    /// supplying a struct φ across a block edge ([`branch_args`]). A local reads its recorded `agg`
    /// fields; a constant struct is materialized field-wise — `zeroinitializer`/`undef`/`poison` to
    /// per-field zeros, an explicit `{…}` literal to its per-field constants.
    fn agg_operand(&mut self, op: &Operand, ftys: &[ValType]) -> Result<Vec<ValIdx>, Error> {
        match op {
            Operand::LocalOperand { name, .. } => {
                let vid = *self
                    .s
                    .name2id
                    .get(name)
                    .ok_or_else(|| Error::Unsupported(format!("unresolved local {name:?}")))?;
                self.agg.get(&vid).cloned().ok_or_else(|| {
                    Error::Unsupported(format!("struct value {vid} not available in block"))
                })
            }
            Operand::ConstantOperand(c) => match c.as_ref() {
                Constant::AggregateZero(_) | Constant::Undef(_) | Constant::Poison(_) => {
                    Ok(ftys.iter().map(|&t| self.push(zero_inst(t))).collect())
                }
                Constant::Struct { values, .. } if values.len() == ftys.len() => values
                    .iter()
                    .map(|v| self.operand(&Operand::ConstantOperand(v.clone())))
                    .collect(),
                // A constant i128 φ incoming (e.g. the entry edge of `phi i128 [0, entry], [next, loop]`):
                // materialize its `(lo, hi)` pair, mirroring `i128_parts`. `llvm-ir` 0.11.3 holds the
                // value in a `u64`, so the high word is always 0 here (a ≥2⁶⁴ / negative i128 constant
                // fails the bitcode parse upstream — see ISSUES.md I14 — so it never reaches us).
                Constant::Int { bits: 128, value } if ftys.len() == 2 => {
                    let lo = self.const_i64(*value as i64);
                    let hi = self.const_i64(0);
                    Ok(vec![lo, hi])
                }
                // A constant `<N x i1>` **mask** φ incoming (the mask analog of the cases above): its
                // per-lane `i1`s materialize as `i32` `0`/`1` consts, one per `agg` field.
                Constant::Vector(elems) if elems.len() == ftys.len() => Ok(elems
                    .iter()
                    .map(|e| {
                        let bit = matches!(e.as_ref(), Constant::Int { value: 1, .. }) as i32;
                        self.push(Inst::ConstI32(bit))
                    })
                    .collect()),
                other => unsup(format!("aggregate φ constant incoming {other:?}")),
            },
            Operand::MetadataOperand => unsup("metadata struct operand"),
        }
    }

    /// The data-SP's block-local index (always parameter 0 of every block, §3d).
    fn sp(&self) -> Result<ValIdx, Error> {
        self.id(SP)
    }

    /// Extract lane `i` (0 or 1) of a scalarized 2-lane vector (a packed `i64`): the low 32 bits for
    /// lane 0, the high 32 for lane 1. An `f32` lane reinterprets the `i32` bits as float; an `i32`
    /// lane returns them directly.
    fn vec_lane(&mut self, v: ValIdx, i: u64, lane_ty: ValType) -> ValIdx {
        let bits32 = if i == 0 {
            self.push(Inst::Convert {
                op: ConvOp::WrapI64,
                a: v,
            })
        } else {
            let c32 = self.const_i64(32);
            let hi = self.push(Inst::IntBin {
                ty: IntTy::I64,
                op: BinOp::ShrU,
                a: v,
                b: c32,
            });
            self.push(Inst::Convert {
                op: ConvOp::WrapI64,
                a: hi,
            })
        };
        match lane_ty {
            ValType::F32 => self.push(Inst::Cast {
                op: CastOp::ReinterpI32F32,
                a: bits32,
            }),
            _ => bits32, // i32 lane
        }
    }

    /// Pack two lanes into a scalarized 2-lane vector (`lane0 | lane1 << 32`, an `i64`); `f32` lanes
    /// are first reinterpreted to their `i32` bits.
    fn vec_pack(&mut self, lane0: ValIdx, lane1: ValIdx, lane_ty: ValType) -> ValIdx {
        let bits = |ctx: &mut Self, lane: ValIdx| match lane_ty {
            ValType::F32 => ctx.push(Inst::Cast {
                op: CastOp::ReinterpF32I32,
                a: lane,
            }),
            _ => lane,
        };
        let i0 = bits(self, lane0);
        let i1 = bits(self, lane1);
        let e0 = self.push(Inst::Convert {
            op: ConvOp::ExtendI32U,
            a: i0,
        });
        let e1 = self.push(Inst::Convert {
            op: ConvOp::ExtendI32U,
            a: i1,
        });
        let c32 = self.const_i64(32);
        let hi = self.push(Inst::IntBin {
            ty: IntTy::I64,
            op: BinOp::Shl,
            a: e1,
            b: c32,
        });
        self.push(Inst::IntBin {
            ty: IntTy::I64,
            op: BinOp::Or,
            a: e0,
            b: hi,
        })
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
        // `snprintf` redirect (§printf): copy this formatted run into the destination buffer instead
        // of writing it to stdout. Bounded so `dest[..size]` (with room for the trailing NUL) is never
        // overrun; `offset` advances by the FULL `len` (C `snprintf` returns the would-be length).
        if let Some(sink) = self.fmt_sink {
            let memcpy = self
                .helpers
                .memcpy
                .ok_or_else(|| Error::Unsupported("snprintf: memcpy helper missing".into()))?;
            let one = self.const_i64(1);
            let cap = self.push(Inst::IntBin {
                ty: IntTy::I64,
                op: BinOp::Sub,
                a: sink.size,
                b: one,
            }); // size - 1 (last index is reserved for the NUL)
            let off_lt_cap = self.push(Inst::IntCmp {
                ty: IntTy::I64,
                op: CmpOp::LtS,
                a: sink.offset,
                b: cap,
            });
            let room_raw = self.push(Inst::IntBin {
                ty: IntTy::I64,
                op: BinOp::Sub,
                a: cap,
                b: sink.offset,
            });
            let zero = self.const_i64(0);
            let room = self.push(Inst::Select {
                cond: off_lt_cap,
                a: room_raw,
                b: zero,
            }); // max(0, (size-1) - offset)
            let len_lt_room = self.push(Inst::IntCmp {
                ty: IntTy::I64,
                op: CmpOp::LtU,
                a: len,
                b: room,
            });
            let ncopy = self.push(Inst::Select {
                cond: len_lt_room,
                a: len,
                b: room,
            }); // min(len, room)
            let dst = self.add_i64(sink.dest, sink.offset);
            self.push_effect(Inst::Call {
                func: memcpy,
                args: vec![dst, buf, ncopy],
            });
            let new_off = self.push(Inst::IntBin {
                ty: IntTy::I64,
                op: BinOp::Add,
                a: sink.offset,
                b: len,
            });
            self.fmt_sink = Some(FmtSink {
                offset: new_off,
                ..sink
            });
            return Ok(new_off);
        }
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

    /// Bind a wide-vector dest name to its legalized parts (chunks ++ tail).
    fn bind_wide(&mut self, dest: &Name, parts: Vec<ValIdx>) {
        if let Some(&vid) = self.s.name2id.get(dest) {
            self.wide_vals.insert(vid, parts);
        }
    }

    /// Resolve a wide-vector operand to its legalized parts. A local reads its recorded parts; a
    /// constant wide vector is materialized (per-chunk `ConstV128` + tail lane consts).
    fn wide_operand(&mut self, op: &Operand, layout: WideLayout) -> Result<Vec<ValIdx>, Error> {
        match op {
            Operand::LocalOperand { name, .. } => {
                let vid = *self
                    .s
                    .name2id
                    .get(name)
                    .ok_or_else(|| Error::Unsupported(format!("unresolved local {name:?}")))?;
                self.wide_vals.get(&vid).cloned().ok_or_else(|| {
                    Error::Unsupported(format!("wide vector value {vid} not available in block"))
                })
            }
            Operand::ConstantOperand(c) => self.wide_const(c.as_ref(), layout),
            Operand::MetadataOperand => unsup("metadata operand"),
        }
    }

    /// Materialize a constant wide vector into its legalized parts: the lanes' little-endian byte
    /// image split into `full_chunks` `ConstV128`s, then `tail_lanes` scalar lane consts.
    fn wide_const(&mut self, c: &Constant, layout: WideLayout) -> Result<Vec<ValIdx>, Error> {
        let lb = layout.shape.lane_bytes() as usize;
        let n = layout.total_lanes();
        let lane_words: Vec<u64> = match c {
            Constant::AggregateZero(_)
            | Constant::Undef(_)
            | Constant::Poison(_)
            | Constant::Null(_) => vec![0u64; n],
            Constant::Vector(elems) => (0..n)
                .map(|k| match elems.get(k).map(|e| e.as_ref()) {
                    Some(Constant::Float(Float::Single(f))) => f.to_bits() as u64,
                    Some(Constant::Float(Float::Double(f))) => f.to_bits(),
                    Some(Constant::Int { value, .. }) => *value,
                    _ => 0, // undef/poison lane → 0
                })
                .collect(),
            other => return unsup(format!("wide vector constant {other:?}")),
        };
        let mut img = vec![0u8; n * lb];
        for (k, w) in lane_words.iter().enumerate() {
            img[k * lb..k * lb + lb].copy_from_slice(&w.to_le_bytes()[..lb]);
        }
        let mut parts = Vec::with_capacity(layout.nparts());
        for ci in 0..layout.full_chunks {
            let mut b = [0u8; 16];
            b.copy_from_slice(&img[ci * 16..ci * 16 + 16]);
            parts.push(self.push(Inst::ConstV128(b)));
        }
        let tail_start = layout.full_chunks * layout.shape.lanes() as usize;
        for t in 0..layout.tail_lanes {
            let inst = tail_const_inst(layout.shape, lane_words[tail_start + t]);
            parts.push(self.push(inst));
        }
        Ok(parts)
    }

    /// Emit the loads for a wide vector at `addr`: `full_chunks` 16-byte `V128Load`s at offsets
    /// `0,16,…`, then `tail_lanes` width-tagged scalar loads of the leftover lanes.
    fn wide_load(&mut self, addr: ValIdx, layout: WideLayout) -> Vec<ValIdx> {
        let mut parts = Vec::with_capacity(layout.nparts());
        for ci in 0..layout.full_chunks {
            parts.push(self.push(Inst::V128Load {
                addr,
                offset: (ci * 16) as u64,
                align: 0,
            }));
        }
        let lb = layout.shape.lane_bytes() as u64;
        let base = (layout.full_chunks * 16) as u64;
        for t in 0..layout.tail_lanes {
            parts.push(self.push(Inst::Load {
                op: lane_load_op(layout.shape),
                addr,
                offset: base + t as u64 * lb,
                align: 0,
            }));
        }
        parts
    }

    /// Emit the stores of a wide vector's `parts` to `addr`: `full_chunks` `V128Store`s at offsets
    /// `0,16,…`, then `tail_lanes` width-tagged scalar stores of the leftover lanes.
    fn wide_store(&mut self, addr: ValIdx, parts: &[ValIdx], layout: WideLayout) {
        for (ci, &value) in parts[..layout.full_chunks].iter().enumerate() {
            self.push_effect(Inst::V128Store {
                addr,
                value,
                offset: (ci * 16) as u64,
                align: 0,
            });
        }
        let lb = layout.shape.lane_bytes() as u64;
        let base = (layout.full_chunks * 16) as u64;
        for t in 0..layout.tail_lanes {
            self.push_effect(Inst::Store {
                op: lane_store_op(layout.shape),
                addr,
                value: parts[layout.full_chunks + t],
                offset: base + t as u64 * lb,
                align: 0,
            });
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

    /// The reserved C++ EH region's base window address (`Helpers::eh_base`), or `Unsupported` if an
    /// EH op was reached without the region having been reserved (a gating bug — should never fire).
    fn eh_base(&self) -> Result<u64, Error> {
        self.helpers
            .eh_base
            .ok_or_else(|| Error::Unsupported("C++ EH op without a reserved EH region".into()))
    }

    /// The compile-time typeinfo id of a typeinfo-global operand (the `@_ZTI*` of a `__cxa_throw` /
    /// `llvm.eh.typeid.for`), interned by [`scan_eh`]. A `catch(...)` clause's `null` operand, or any
    /// typeinfo not seen by the scan, is a clean `Unsupported`.
    fn eh_typeid(&self, op: &Operand) -> Result<u32, Error> {
        let name = global_name_of(op)
            .ok_or_else(|| Error::Unsupported("EH typeinfo operand is not a global".into()))?;
        self.helpers
            .eh_typeids
            .get(&name)
            .copied()
            .ok_or_else(|| Error::Unsupported(format!("unregistered EH typeinfo `@{name}`")))
    }

    /// The throw/rethrow/resume tail: pop the innermost active handler (`HSP-1`) and long-jump into
    /// its `invoke`-installed checkpoint, making that `SetJmp` return nonzero so control resumes at
    /// the landing pad. Emits `LongJmp` (a noreturn control op) as an effect; the caller supplies the
    /// trailing `unreachable`/dead terminator.
    fn eh_unwind(&mut self, base: u64) -> Result<(), Error> {
        // Hand off to `__svm_eh_unwind(base)`: it pops the innermost handler and long-jumps into its
        // checkpoint, or traps (`std::terminate`) when no handler remains (an uncaught exception). The
        // empty-stack branch lives in the helper because this lowers in effect position — it cannot
        // branch inline, and a bare `HSP-1` would underflow to a bogus slot. The caller still emits the
        // trailing `unreachable` (the LLVM `unreachable` after the throw); the helper never returns.
        let helper = self
            .helpers
            .eh_unwind
            .ok_or_else(|| Error::Unsupported("EH unwinder helper not synthesized".into()))?;
        let base_c = self.const_i64(base as i64);
        self.push_effect(Inst::Call {
            func: helper,
            args: vec![base_c],
        });
        Ok(())
    }

    /// An operand widened to the host word `i64` (the §7/§3e capability-call ABI): a pointer or
    /// `i64` is already there; a narrow `i32` is zero-extended (addresses/lengths/indices are
    /// non-negative window quantities). Float/vector operands are a clean `Unsupported`.
    fn operand_i64(&mut self, op: &Operand) -> Result<ValIdx, Error> {
        let v = self.operand(op)?;
        match val_type(op.get_type(self.types).as_ref())? {
            ValType::I64 => Ok(v),
            ValType::I32 => Ok(self.push(Inst::Convert {
                op: ConvOp::ExtendI32U,
                a: v,
            })),
            other => unsup(format!(
                "expected an integer/pointer argument, got {}",
                other.as_str()
            )),
        }
    }

    /// An operand narrowed to `i32` (a capability handle, a 32-bit atomic word, a fiber/thread
    /// handle): already `i32`, or an `i64` truncated. Float/vector operands are `Unsupported`.
    fn operand_i32(&mut self, op: &Operand) -> Result<ValIdx, Error> {
        let v = self.operand(op)?;
        match val_type(op.get_type(self.types).as_ref())? {
            ValType::I32 => Ok(v),
            ValType::I64 => Ok(self.push(Inst::Convert {
                op: ConvOp::WrapI64,
                a: v,
            })),
            other => unsup(format!(
                "expected an integer argument, got {}",
                other.as_str()
            )),
        }
    }

    /// Resolve an operand that must be a **direct function designator** (a `@func` reference) to its
    /// IR function index — the static `func` immediate `thread.spawn` requires (§12). A computed
    /// function pointer is a clean `Unsupported` (mirrors the chibicc `__vm_thread_spawn` rule).
    fn direct_func_idx(&self, op: &Operand) -> Result<u32, Error> {
        if let Operand::ConstantOperand(c) = op {
            if let Constant::GlobalReference { name, .. } = c.as_ref() {
                let n = name_str(name);
                if let Some(&f) = self.name2idx.get(&n) {
                    return Ok(f);
                }
            }
        }
        unsup("`__vm_thread_spawn` requires a direct function name as its first argument")
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
    /// A φ-incoming operand, resolving an **operand-position `blockaddress`** (clang jump-threaded one
    /// through this φ) to its recovered target block index (the same integer the `br_table` consumes).
    /// `target`/`phi_ord`/`inc_idx` locate the φ-incoming in [`blockaddr::BlockAddrs::phi`]. Any other
    /// operand defers to [`Self::operand`].
    fn phi_operand(
        &mut self,
        op: &Operand,
        target: u32,
        phi_ord: u32,
        inc_idx: u32,
    ) -> Result<ValIdx, Error> {
        if let Operand::ConstantOperand(c) = op {
            if matches!(c.as_ref(), Constant::BlockAddress) {
                let key = (self.func_idx, target, phi_ord, inc_idx);
                let label = self
                    .blockaddrs
                    .and_then(|b| b.phi.get(&key).copied())
                    .ok_or_else(|| {
                        Error::Unsupported(
                            "operand-position blockaddress without a recovered label".into(),
                        )
                    })?;
                return Ok(self.push(Inst::ConstI64(label as i64)));
            }
        }
        self.operand(op)
    }

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
                // `i64` and the `iN` (33..63) widths share the `i64` container; an `iN` constant is
                // canonicalized to its low `N` bits (its in-container representation, see `val_type`).
                Constant::Int { bits, value } if *bits <= 64 => {
                    let v = if *bits == 64 {
                        *value
                    } else {
                        *value & ((1u64 << *bits) - 1)
                    };
                    Ok(self.push(Inst::ConstI64(v as i64)))
                }
                Constant::Float(Float::Single(f)) => Ok(self.push(Inst::ConstF32(f.to_bits()))),
                Constant::Float(Float::Double(d)) => Ok(self.push(Inst::ConstF64(d.to_bits()))),
                // `undef`/`poison`/`null` resolve to a defined zero of the type — the IR is total
                // (§3c), so no UB reaches it (the value is unused or its use is defined-on-zero).
                Constant::Undef(t) | Constant::Poison(t) | Constant::Null(t) => {
                    match val_type(t.as_ref())? {
                        ValType::I32 => Ok(self.push(Inst::ConstI32(0))),
                        ValType::I64 => Ok(self.push(Inst::ConstI64(0))),
                        ValType::F32 => Ok(self.push(Inst::ConstF32(0))),
                        ValType::F64 => Ok(self.push(Inst::ConstF64(0))),
                        ValType::V128 => Ok(self.push(Inst::ConstV128([0u8; 16]))),
                        other => unsup(format!("undef/poison/null of type {}", other.as_str())),
                    }
                }
                // A `zeroinitializer` of a 2-lane vector (scalarized to `i64`) is the zero word; any
                // 128-bit one is a zero `v128`.
                Constant::AggregateZero(t) if is_vec2(t.as_ref()) => {
                    Ok(self.push(Inst::ConstI64(0)))
                }
                Constant::AggregateZero(t) if vec128_shape(t.as_ref()).is_some() => {
                    Ok(self.push(Inst::ConstV128([0u8; 16])))
                }
                // A `<2 x float>`/`<2 x i32>` literal — pack the two lanes' 32-bit values into the
                // `i64` (lane 0 low).
                Constant::Vector(elems) if is_vec2(c.get_type(self.types).as_ref()) => {
                    let lane = |k: usize| -> u32 {
                        match elems.get(k).map(|e| e.as_ref()) {
                            Some(Constant::Float(Float::Single(f))) => f.to_bits(),
                            Some(Constant::Int { value, .. }) => *value as u32,
                            _ => 0, // undef/poison lane → 0
                        }
                    };
                    let packed = lane(0) as u64 | ((lane(1) as u64) << 32);
                    Ok(self.push(Inst::ConstI64(packed as i64)))
                }
                // Any 128-bit vector literal — its 16-byte little-endian memory image (lane 0 first),
                // written `shape.lane_bytes()` bytes per element.
                Constant::Vector(elems)
                    if vec128_shape(c.get_type(self.types).as_ref()).is_some() =>
                {
                    let shape = vec128_shape(c.get_type(self.types).as_ref()).unwrap();
                    let lb = shape.lane_bytes() as usize;
                    let mut bytes = [0u8; 16];
                    for (k, e) in elems.iter().take(shape.lanes() as usize).enumerate() {
                        // Each lane's little-endian bits: float lanes use their IEEE image, int
                        // lanes their low `lane_bytes` bytes. An undef/poison lane is 0.
                        let w: u64 = match e.as_ref() {
                            Constant::Float(Float::Single(f)) => f.to_bits() as u64,
                            Constant::Float(Float::Double(f)) => f.to_bits(),
                            Constant::Int { value, .. } => *value,
                            _ => 0,
                        };
                        bytes[k * lb..k * lb + lb].copy_from_slice(&w.to_le_bytes()[..lb]);
                    }
                    Ok(self.push(Inst::ConstV128(bytes)))
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
                // A constexpr interior pointer, or a pointer materialized from an integer
                // (`inttoptr`/`ptrtoint` over constants — e.g. Rust's `NonNull::dangling()` for an
                // empty `Vec`, an `inttoptr(align)`), folds to its constant `i64` window value.
                Constant::GetElementPtr(_) | Constant::IntToPtr(_) | Constant::PtrToInt(_) => {
                    let v = const_eval(c.as_ref(), self.globals, self.name2idx, self.types)?;
                    Ok(self.push(Inst::ConstI64(v)))
                }
                other => unsup(format!("constant operand {other:?}")),
            },
            Operand::MetadataOperand => unsup("metadata operand"),
        }
    }
}

/// The index of `bb`'s final instruction if it is a **tail-position call** — a `tail`/`musttail`
/// `call` whose result the block's `ret` returns directly (or a void tail call before `ret void`).
/// Such a call lowers to a `return_call` terminator (constant native-stack space for unbounded
/// tail/mutual recursion). A call that turns out to route to a capability import/builtin is filtered
/// at the user-call emit site, so a false positive here is harmless — it simply isn't converted.
fn tail_call_index(bb: &BasicBlock) -> Option<usize> {
    let LTerm::Ret(r) = &bb.term else { return None };
    let idx = bb.instrs.len().checked_sub(1)?;
    let Instruction::Call(c) = &bb.instrs[idx] else {
        return None;
    };
    if !c.is_tail_call {
        return None;
    }
    match (&c.dest, &r.return_operand) {
        (None, None) => Some(idx), // void tail call + `ret void`
        (Some(d), Some(Operand::LocalOperand { name, .. })) if name == d => Some(idx),
        _ => None,
    }
}

#[allow(clippy::too_many_arguments)]
fn translate_block(
    bb: &BasicBlock,
    bi: usize,
    func_idx: u32,
    bc_func_idx: u32,
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
    gbytes: &HashMap<String, Vec<u8>>,
    helpers: &Helpers,
    ba: Option<&blockaddr::BlockAddrs>,
    dbg: &mut DebugAcc,
    aux_blocks: &mut Vec<Block>,
) -> Result<Block, Error> {
    let param_ids = &block_params[bi];
    // Materialize the block parameters. A scalar value (incl. the data-SP, which types as `i64`) is
    // one slot; a **wide-vector** value fans out into `K = full_chunks + tail_lanes` consecutive
    // slots — `C` `v128` chunks then `T` lane scalars — matching the order its parts are supplied in
    // `branch_args` and held in `wide_vals` (I2 step 1 cross-block).
    let mut params: Vec<ValType> = Vec::with_capacity(param_ids.len());
    let mut scalar_seed: Vec<(ValueId, ValIdx)> = Vec::new();
    let mut wide_seed: Vec<(ValueId, Vec<ValIdx>)> = Vec::new();
    // A **struct** param (a φ of a small by-value struct) fans out into one slot per field, seeded into
    // the block-local `agg` table — the aggregate analog of `wide_seed`.
    let mut agg_seed: Vec<(ValueId, Vec<ValIdx>)> = Vec::new();
    for &vid in param_ids {
        let start = params.len() as ValIdx;
        if let Some(&layout) = s.wide.get(&vid) {
            for _ in 0..layout.full_chunks {
                params.push(ValType::V128);
            }
            let lane = layout.shape.lane_val();
            for _ in 0..layout.tail_lanes {
                params.push(lane);
            }
            wide_seed.push((vid, (start..params.len() as ValIdx).collect()));
        } else if let Some(ftys) = s.agg_layout.get(&vid) {
            params.extend_from_slice(ftys);
            agg_seed.push((vid, (start..params.len() as ValIdx).collect()));
        } else {
            params.push(if vid == SP { ValType::I64 } else { s.ty[vid] });
            scalar_seed.push((vid, start));
        }
    }
    let mut ctx = BlockCtx {
        s,
        frame,
        frame_size,
        name2idx,
        globals,
        caps,
        cstrs,
        gbytes,
        helpers: helpers.clone(),
        types,
        func_idx: bc_func_idx,
        blockaddrs: ba,
        insts: Vec::new(),
        idx_of: HashMap::new(),
        agg: HashMap::new(),
        wide_vals: HashMap::new(),
        next_val: 0,
        tail_return: false,
        pending_tail: None,
        fmt_sink: None,
    };
    for (vid, pos) in scalar_seed {
        ctx.idx_of.insert(vid, pos);
    }
    for (vid, parts) in wide_seed {
        ctx.wide_vals.insert(vid, parts);
    }
    for (vid, parts) in agg_seed {
        ctx.agg.insert(vid, parts);
    }
    ctx.next_val = params.len() as ValIdx;

    // A tail-position call in this block (final instruction is a `tail`/`musttail` call whose result
    // the `ret` returns) lowers to a `return_call` terminator instead of a `call` + `ret`.
    let tail_idx = tail_call_index(bb);
    for (idx, instr) in bb.instrs.iter().enumerate() {
        if matches!(instr, Instruction::Phi(_)) {
            continue; // φ-results are block parameters, supplied by predecessors
        }
        // §6 source map: the SVM ops this LLVM instruction lowers to inherit its `!DILocation`.
        let start = ctx.insts.len();
        ctx.tail_return = tail_idx == Some(idx);
        translate_inst(&mut ctx, instr, types)?;
        ctx.tail_return = false;
        if let Some(dl) = instr.get_debug_loc() {
            dbg.map_range(func_idx, bi as u32, start, ctx.insts.len(), dl);
        }
    }
    // If the tail call produced a `return_call` terminator, use it; otherwise lower the real
    // terminator. (A `tail`-marked call that routes to an import/builtin leaves `pending_tail` unset
    // and falls through to the ordinary `ret` here.)
    let term = match ctx.pending_tail.take() {
        Some(t) => t,
        None => translate_term(&mut ctx, &bb.term, bi, f, s, block_params, aux_blocks)?,
    };
    Ok(Block {
        params,
        insts: ctx.insts,
        term,
    })
}

/// The width class of an atomic operation's value type. `i32`/`i64`/pointer map to a native
/// `IntTy`; `i8`/`i16` are **narrow** — svm-ir has no sub-word atomic (it keeps only `i32`/`i64`),
/// so they emulate via a 32-bit CAS loop over the enclosing aligned word (DESIGN §3b note 2).
enum AtomWidth {
    Wide(IntTy),
    Narrow(u8), // byte count: 1 (i8) or 2 (i16)
}

fn atom_width(ty: &Type) -> Result<AtomWidth, Error> {
    match ty {
        Type::IntegerType { bits } => match bits {
            8 => Ok(AtomWidth::Narrow(1)),
            16 => Ok(AtomWidth::Narrow(2)),
            32 => Ok(AtomWidth::Wide(IntTy::I32)),
            64 => Ok(AtomWidth::Wide(IntTy::I64)),
            b => unsup(format!("atomic on i{b} (only i8/i16/i32/i64)")),
        },
        Type::PointerType { .. } => Ok(AtomWidth::Wide(IntTy::I64)),
        other => unsup(format!("atomic on non-integer type {other}")),
    }
}

/// Map an LLVM `atomicrmw` binop to the svm-ir [`AtomicRmwOp`], if svm-ir has it natively. `nand`,
/// the min/max family, and the float ops have no native op — `None` ⇒ fail-closed for now (a later
/// slice can CAS-loop-emulate them, like the narrow path).
fn rmw_op(op: llvm_ir::instruction::RMWBinOp) -> Option<AtomicRmwOp> {
    use llvm_ir::instruction::RMWBinOp as L;
    Some(match op {
        L::Xchg => AtomicRmwOp::Xchg,
        L::Add => AtomicRmwOp::Add,
        L::Sub => AtomicRmwOp::Sub,
        L::And => AtomicRmwOp::And,
        L::Or => AtomicRmwOp::Or,
        L::Xor => AtomicRmwOp::Xor,
        _ => return None,
    })
}

// Narrow-atomic RMW op codes passed to the `__svm_atomic_rmw_narrow` helper.
const NARROW_RMW_XCHG: i64 = 0;
const NARROW_RMW_ADD: i64 = 1;
const NARROW_RMW_SUB: i64 = 2;
const NARROW_RMW_AND: i64 = 3;
const NARROW_RMW_OR: i64 = 4;
const NARROW_RMW_XOR: i64 = 5;

/// LLVM `atomicrmw` binop → narrow-helper opcode (the subset svm-ir/the helper supports).
fn narrow_rmw_opcode(op: llvm_ir::instruction::RMWBinOp) -> Option<i64> {
    use llvm_ir::instruction::RMWBinOp as L;
    Some(match op {
        L::Xchg => NARROW_RMW_XCHG,
        L::Add => NARROW_RMW_ADD,
        L::Sub => NARROW_RMW_SUB,
        L::And => NARROW_RMW_AND,
        L::Or => NARROW_RMW_OR,
        L::Xor => NARROW_RMW_XOR,
        _ => return None,
    })
}

// Narrow (i8/i16) atomic lowering — emulated via a seq-cst 32-bit CAS loop over the enclosing
// aligned word (DESIGN §3b note 2): keep the IR i32/i64-only and splice the field in `__svm_atomic_*`
// helpers. `narrow_word_and_shift` gives the aligned word address and the field's bit offset.

/// `(addr & ~3, (addr & 3) * 8)` — the enclosing aligned 32-bit word and the field's bit offset
/// within it. A naturally-aligned i8/i16 lies wholly inside its enclosing 4-byte word.
fn narrow_word_and_shift(ctx: &mut BlockCtx, addr: ValIdx) -> (ValIdx, ValIdx) {
    let not3 = ctx.const_i64(!3i64);
    let word = ctx.push(Inst::IntBin {
        ty: IntTy::I64,
        op: BinOp::And,
        a: addr,
        b: not3,
    });
    let k3 = ctx.const_i64(3);
    let low = ctx.push(Inst::IntBin {
        ty: IntTy::I64,
        op: BinOp::And,
        a: addr,
        b: k3,
    });
    let k8 = ctx.const_i64(8);
    let shift = ctx.push(Inst::IntBin {
        ty: IntTy::I64,
        op: BinOp::Mul,
        a: low,
        b: k8,
    });
    (word, shift)
}

fn narrow_mask(w: u8) -> i64 {
    if w == 1 {
        0xFF
    } else {
        0xFFFF
    }
}

/// Narrow atomic load: an atomic load of the enclosing word, then extract the field. A single aligned
/// word read is atomic for the byte/halfword within it — no CAS needed.
fn lower_narrow_atomic_load(ctx: &mut BlockCtx, addr: ValIdx, w: u8) -> Result<ValIdx, Error> {
    let (word, shift) = narrow_word_and_shift(ctx, addr);
    let mask = ctx.const_i64(narrow_mask(w));
    let w32 = ctx.push(Inst::AtomicLoad {
        ty: IntTy::I32,
        addr: word,
        offset: 0,
        order: Ordering::SeqCst,
    });
    let w64 = ctx.push(Inst::Convert {
        op: ConvOp::ExtendI32U,
        a: w32,
    });
    let sh = ctx.push(Inst::IntBin {
        ty: IntTy::I64,
        op: BinOp::ShrU,
        a: w64,
        b: shift,
    });
    let field = ctx.push(Inst::IntBin {
        ty: IntTy::I64,
        op: BinOp::And,
        a: sh,
        b: mask,
    });
    Ok(ctx.push(Inst::Convert {
        op: ConvOp::WrapI64,
        a: field,
    }))
}

/// Narrow `atomicrmw` (and atomic store, as an `xchg`): call the CAS-loop helper, return the old
/// field (i32 container).
fn lower_narrow_atomic_rmw(
    ctx: &mut BlockCtx,
    addr: ValIdx,
    value: ValIdx,
    w: u8,
    opcode: i64,
) -> Result<ValIdx, Error> {
    let h = ctx
        .helpers
        .atomic_rmw_narrow
        .ok_or_else(|| Error::Unsupported("narrow atomic rmw helper missing".into()))?;
    let (word, shift) = narrow_word_and_shift(ctx, addr);
    let mask = ctx.const_i64(narrow_mask(w));
    let value64 = ctx.push(Inst::Convert {
        op: ConvOp::ExtendI32U,
        a: value,
    });
    let opc = ctx.const_i64(opcode);
    let old64 = ctx.push(Inst::Call {
        func: h,
        args: vec![word, shift, mask, value64, opc],
    });
    Ok(ctx.push(Inst::Convert {
        op: ConvOp::WrapI64,
        a: old64,
    }))
}

/// Narrow `cmpxchg`: call the CAS-loop helper with a pre-masked `expected`/`replacement`; return the
/// old field and the masked `expected` (the caller compares them for the success flag).
fn lower_narrow_atomic_cas(
    ctx: &mut BlockCtx,
    addr: ValIdx,
    expected: ValIdx,
    replacement: ValIdx,
    w: u8,
) -> Result<(ValIdx, ValIdx), Error> {
    let h = ctx
        .helpers
        .atomic_cas_narrow
        .ok_or_else(|| Error::Unsupported("narrow atomic cas helper missing".into()))?;
    let (word, shift) = narrow_word_and_shift(ctx, addr);
    let mask = ctx.const_i64(narrow_mask(w));
    let ext = |ctx: &mut BlockCtx, a| {
        ctx.push(Inst::Convert {
            op: ConvOp::ExtendI32U,
            a,
        })
    };
    let and_mask = |ctx: &mut BlockCtx, a| {
        ctx.push(Inst::IntBin {
            ty: IntTy::I64,
            op: BinOp::And,
            a,
            b: mask,
        })
    };
    let exp64 = ext(ctx, expected);
    let masked_exp64 = and_mask(ctx, exp64);
    let masked_exp32 = ctx.push(Inst::Convert {
        op: ConvOp::WrapI64,
        a: masked_exp64,
    });
    let repl64 = ext(ctx, replacement);
    let old64 = ctx.push(Inst::Call {
        func: h,
        args: vec![word, shift, mask, masked_exp64, repl64],
    });
    let old32 = ctx.push(Inst::Convert {
        op: ConvOp::WrapI64,
        a: old64,
    });
    Ok((old32, masked_exp32))
}

/// I14 tier 3 — read an i128 operand as its `(lo, hi)` i64 parts. A local i128 is an `agg` pair
/// `[lo, hi]` (built by `lower_i128`, `load i128`, …); a constant i128 below 2⁶⁴ materializes as
/// `(value, 0)`. A cross-block i128 (no `agg` entry here) or a wide/negative i128 constant fails
/// closed — rare, and never a miscompile.
fn i128_parts(ctx: &mut BlockCtx, op: &Operand) -> Result<(ValIdx, ValIdx), Error> {
    if let Some(parts) = ctx.agg_of(op) {
        if parts.len() == 2 {
            return Ok((parts[0], parts[1]));
        }
    }
    if let Operand::ConstantOperand(c) = op {
        if let Constant::Int { bits: 128, value } = c.as_ref() {
            let lo = ctx.const_i64(*value as i64);
            let hi = ctx.const_i64(0);
            return Ok((lo, hi));
        }
    }
    unsup("i128 operand not available in this block (cross-block / unsupported i128 constant)")
}

/// Bind a destination i128 value to its `(lo, hi)` parts (the unified `agg`-pair representation, shared
/// with `load i128` / `icmp i128`).
fn set_i128(ctx: &mut BlockCtx, dest: &Name, lo: ValIdx, hi: ValIdx) {
    if let Some(&vid) = ctx.s.name2id.get(dest) {
        ctx.agg.insert(vid, vec![lo, hi]);
    }
}

/// Emit an `i64` binary op.
fn i64bin(ctx: &mut BlockCtx, op: BinOp, a: ValIdx, b: ValIdx) -> ValIdx {
    ctx.push(Inst::IntBin {
        ty: IntTy::I64,
        op,
        a,
        b,
    })
}

/// Emit an `i64` comparison, zero-extended from its `i32` boolean result to an `i64` `0`/`1` — the form
/// the carry/borrow chains add into a high word.
fn i64cmp_ext(ctx: &mut BlockCtx, op: CmpOp, a: ValIdx, b: ValIdx) -> ValIdx {
    let c = ctx.push(Inst::IntCmp {
        ty: IntTy::I64,
        op,
        a,
        b,
    });
    emit_ext(ctx, c, 32, 64, false)
}

/// An `i64` comparison (raw `i32` `0`/`1` result).
fn icmp64(ctx: &mut BlockCtx, op: CmpOp, a: ValIdx, b: ValIdx) -> ValIdx {
    ctx.push(Inst::IntCmp {
        ty: IntTy::I64,
        op,
        a,
        b,
    })
}

/// An `i32` binary op (the boolean-combining width).
fn bin32(ctx: &mut BlockCtx, op: BinOp, a: ValIdx, b: ValIdx) -> ValIdx {
    ctx.push(Inst::IntBin {
        ty: IntTy::I32,
        op,
        a,
        b,
    })
}

/// Lower an i128 comparison (any predicate) on `(lo, hi)` pairs to an `i32` boolean. `eq`/`ne` is the
/// AND of the word equalities; an ordering is `ahi <strict> bhi | (ahi == bhi & alo <op_u> blo)` — the
/// high word carries the predicate's signedness, the low word is always unsigned.
fn i128_icmp(
    ctx: &mut BlockCtx,
    pred: IntPredicate,
    alo: ValIdx,
    ahi: ValIdx,
    blo: ValIdx,
    bhi: ValIdx,
) -> ValIdx {
    use IntPredicate as P;
    if matches!(pred, P::EQ | P::NE) {
        let lo = icmp64(ctx, CmpOp::Eq, alo, blo);
        let hi = icmp64(ctx, CmpOp::Eq, ahi, bhi);
        let eq = bin32(ctx, BinOp::And, lo, hi);
        if pred == P::EQ {
            return eq;
        }
        let one = ctx.push(Inst::ConstI32(1));
        return bin32(ctx, BinOp::Xor, eq, one);
    }
    let (hi_op, lo_op) = match pred {
        P::ULT => (CmpOp::LtU, CmpOp::LtU),
        P::SLT => (CmpOp::LtS, CmpOp::LtU),
        P::ULE => (CmpOp::LtU, CmpOp::LeU),
        P::SLE => (CmpOp::LtS, CmpOp::LeU),
        P::UGT => (CmpOp::GtU, CmpOp::GtU),
        P::SGT => (CmpOp::GtS, CmpOp::GtU),
        P::UGE => (CmpOp::GtU, CmpOp::GeU),
        P::SGE => (CmpOp::GtS, CmpOp::GeU),
        P::EQ | P::NE => unreachable!("handled above"),
    };
    let hi_strict = icmp64(ctx, hi_op, ahi, bhi);
    let hi_eq = icmp64(ctx, CmpOp::Eq, ahi, bhi);
    let lo_cmp = icmp64(ctx, lo_op, alo, blo);
    let lo_and = bin32(ctx, BinOp::And, hi_eq, lo_cmp);
    bin32(ctx, BinOp::Or, hi_strict, lo_and)
}

/// Emit an inline unsigned 64×64→64 **high-half multiply** (`umulhi`) via the 32×32 schoolbook
/// expansion: with `a = ah·2³² + al`, `b = bh·2³² + bl`, the high word of `a·b` is
/// `ah·bh + (al·bh)>>32 + (ah·bl)>>32 + cross>>32` where `cross = (al·bl)>>32 + (al·bh & lo) + (ah·bl & lo)`.
/// The engine has no scalar high-multiply primitive, so the i128 mulhi idiom lowers to this. Returns the
/// block-local index holding the high 64 bits of `a*b`.
fn emit_umulhi(ctx: &mut BlockCtx, a: ValIdx, b: ValIdx) -> ValIdx {
    let mask = ctx.const_i64(0xFFFF_FFFF);
    let sh = ctx.const_i64(32);
    fn ibin(ctx: &mut BlockCtx, op: BinOp, a: ValIdx, b: ValIdx) -> ValIdx {
        ctx.push(Inst::IntBin {
            ty: IntTy::I64,
            op,
            a,
            b,
        })
    }
    let al = ibin(ctx, BinOp::And, a, mask);
    let ah = ibin(ctx, BinOp::ShrU, a, sh);
    let bl = ibin(ctx, BinOp::And, b, mask);
    let bh = ibin(ctx, BinOp::ShrU, b, sh);
    let ll = ibin(ctx, BinOp::Mul, al, bl);
    let lh = ibin(ctx, BinOp::Mul, al, bh);
    let hl = ibin(ctx, BinOp::Mul, ah, bl);
    let hh = ibin(ctx, BinOp::Mul, ah, bh);
    // cross = (ll >> 32) + (lh & lo) + (hl & lo) — bounded by 3·(2³²−1) < 2³⁴, so it cannot overflow i64.
    let ll_hi = ibin(ctx, BinOp::ShrU, ll, sh);
    let lh_lo = ibin(ctx, BinOp::And, lh, mask);
    let hl_lo = ibin(ctx, BinOp::And, hl, mask);
    let c1 = ibin(ctx, BinOp::Add, ll_hi, lh_lo);
    let cross = ibin(ctx, BinOp::Add, c1, hl_lo);
    // hi = hh + (lh >> 32) + (hl >> 32) + (cross >> 32)
    let lh_hi = ibin(ctx, BinOp::ShrU, lh, sh);
    let hl_hi = ibin(ctx, BinOp::ShrU, hl, sh);
    let cross_hi = ibin(ctx, BinOp::ShrU, cross, sh);
    let s1 = ibin(ctx, BinOp::Add, hh, lh_hi);
    let s2 = ibin(ctx, BinOp::Add, s1, hl_hi);
    ibin(ctx, BinOp::Add, s2, cross_hi)
}

/// I14 tier 2 — recognize the i128 **widening-multiply** idiom and lower it without ever materializing a
/// 128-bit value. `-O2` clang emits `zext i64 → mul i128 → lshr 64 → trunc i64` for `(u128)a*b >> 64`
/// (a 64×64→128 mulhi; the low half is a plain `mul i64`). Each i128 SSA value is tracked symbolically
/// ([`I128Sym`]); a concrete i64 op is emitted only at the `trunc` — the source for a `zext`'s low half,
/// `mul` for a product's low half, an inline [`emit_umulhi`] for its high half. Any i128 use outside this
/// idiom fails closed (`Unsupported`) — never a miscompile. `Ok(true)` ⇒ handled an i128 idiom op.
/// Which double-word shift to lower.
#[derive(Clone, Copy)]
enum I128ShiftKind {
    Shl,
    LShr,
    AShr,
}

/// Lower a bitwise i128 op lane-wise on the `(lo, hi)` pair.
fn i128_bitwise(
    ctx: &mut BlockCtx,
    op0: &Operand,
    op1: &Operand,
    dest: &Name,
    op: BinOp,
) -> Result<bool, Error> {
    let (alo, ahi) = i128_parts(ctx, op0)?;
    let (blo, bhi) = i128_parts(ctx, op1)?;
    let lo = i64bin(ctx, op, alo, blo);
    let hi = i64bin(ctx, op, ahi, bhi);
    set_i128(ctx, dest, lo, hi);
    Ok(true)
}

/// Lower a **double-word** i128 shift by a runtime amount `n` (LLVM guarantees `n < 128`). Branchless,
/// via the engine's `Select`: the within-word part `m = n & 63`, the cross-word carry (guarded for
/// `m == 0`, where `64 - m == 64` would be a no-op shift, not a full one), and an `n >= 64` select that
/// moves a whole word. `AShr` additionally fills with the sign word.
fn i128_shift(
    ctx: &mut BlockCtx,
    op0: &Operand,
    op1: &Operand,
    dest: &Name,
    kind: I128ShiftKind,
) -> Result<bool, Error> {
    let (lo, hi) = i128_parts(ctx, op0)?;
    let (n, _n_hi) = i128_parts(ctx, op1)?; // shift count = low word (n < 128)
    let c0 = ctx.const_i64(0);
    let c63 = ctx.const_i64(63);
    let c64 = ctx.const_i64(64);
    let m = i64bin(ctx, BinOp::And, n, c63); // n mod 64
    let inv = i64bin(ctx, BinOp::Sub, c64, m); // 64 - m (== 64 when m == 0)
    let ge64 = ctx.push(Inst::IntCmp {
        ty: IntTy::I64,
        op: CmpOp::GeU,
        a: n,
        b: c64,
    });
    let m_is0 = ctx.push(Inst::IntCmp {
        ty: IntTy::I64,
        op: CmpOp::Eq,
        a: m,
        b: c0,
    });
    let sel = |ctx: &mut BlockCtx, cond: ValIdx, a: ValIdx, b: ValIdx| {
        ctx.push(Inst::Select { cond, a, b })
    };
    let (res_lo, res_hi) = match kind {
        // shl: low bits flow up into hi; n >= 64 moves lo into hi and zeroes lo.
        I128ShiftKind::Shl => {
            let lo_m = i64bin(ctx, BinOp::Shl, lo, m);
            let hi_m = i64bin(ctx, BinOp::Shl, hi, m);
            let carry_raw = i64bin(ctx, BinOp::ShrU, lo, inv);
            let carry = sel(ctx, m_is0, c0, carry_raw);
            let hi_lt64 = i64bin(ctx, BinOp::Or, hi_m, carry);
            let res_lo = sel(ctx, ge64, c0, lo_m);
            let res_hi = sel(ctx, ge64, lo_m, hi_lt64); // n>=64: hi = lo << (n-64) = lo << m
            (res_lo, res_hi)
        }
        // lshr: high bits flow down into lo; n >= 64 moves hi into lo and zeroes hi.
        I128ShiftKind::LShr => {
            let lo_m = i64bin(ctx, BinOp::ShrU, lo, m);
            let hi_m = i64bin(ctx, BinOp::ShrU, hi, m);
            let carry_raw = i64bin(ctx, BinOp::Shl, hi, inv);
            let carry = sel(ctx, m_is0, c0, carry_raw);
            let lo_lt64 = i64bin(ctx, BinOp::Or, lo_m, carry);
            let res_hi = sel(ctx, ge64, c0, hi_m);
            let res_lo = sel(ctx, ge64, hi_m, lo_lt64); // n>=64: lo = hi >> (n-64) = hi >> m
            (res_lo, res_hi)
        }
        // ashr: like lshr but hi shifts arithmetically and the fill is the sign word.
        I128ShiftKind::AShr => {
            let sign = i64bin(ctx, BinOp::ShrS, hi, c63);
            let lo_m = i64bin(ctx, BinOp::ShrU, lo, m);
            let hi_m = i64bin(ctx, BinOp::ShrS, hi, m);
            let carry_raw = i64bin(ctx, BinOp::Shl, hi, inv);
            let carry = sel(ctx, m_is0, c0, carry_raw);
            let lo_lt64 = i64bin(ctx, BinOp::Or, lo_m, carry);
            let res_hi = sel(ctx, ge64, sign, hi_m);
            let res_lo = sel(ctx, ge64, hi_m, lo_lt64); // n>=64: lo = hi >>s (n-64) = hi >>s m
            (res_lo, res_hi)
        }
    };
    set_i128(ctx, dest, res_lo, res_hi);
    Ok(true)
}

/// I14 tier 3 — general i128 legalization: every i128 SSA value is a materialized `(lo, hi)` i64 pair
/// (the unified `agg`-pair representation, shared with `load i128` / `icmp i128`), and each i128 op
/// lowers to 64-bit ops over the parts. Covers `zext`/`sext`/`trunc`, `and`/`or`/`xor`, `add`/`sub`
/// (carry/borrow chains), `mul` (the schoolbook 64×64 expansion), and double-word `shl`/`lshr`/`ashr`.
/// Cross-block i128 values (no `agg` entry in the new block), i128 call args/results, and wide/negative
/// i128 constants fail closed — never a miscompile. `Ok(true)` ⇒ handled an i128 op.
fn lower_i128(ctx: &mut BlockCtx, instr: &Instruction, types: &Types) -> Result<bool, Error> {
    use Instruction as I;
    let is_i128 = |o: &Operand| int_bits(o.get_type(types).as_ref()) == Some(128);
    match instr {
        // zext iN X to i128 (N ≤ 64) → (zext(X, N→64), 0)
        I::ZExt(x) if int_bits(x.to_type.as_ref()) == Some(128) => {
            let from = int_bits(x.operand.get_type(types).as_ref())
                .filter(|&w| w <= 64)
                .ok_or_else(|| Error::Unsupported("i128 zext from a non-integer source".into()))?;
            let src = ctx.operand(&x.operand)?;
            let lo = if from == 64 {
                src
            } else {
                emit_ext(ctx, src, from, 64, false)
            };
            let hi = ctx.const_i64(0);
            set_i128(ctx, &x.dest, lo, hi);
            Ok(true)
        }
        // sext iN X to i128 (N ≤ 64) → (sext(X, N→64), sign >>s 63)
        I::SExt(x) if int_bits(x.to_type.as_ref()) == Some(128) => {
            let from = int_bits(x.operand.get_type(types).as_ref())
                .filter(|&w| w <= 64)
                .ok_or_else(|| Error::Unsupported("i128 sext from a non-integer source".into()))?;
            let src = ctx.operand(&x.operand)?;
            let lo = if from == 64 {
                src
            } else {
                emit_ext(ctx, src, from, 64, true)
            };
            let c63 = ctx.const_i64(63);
            let hi = i64bin(ctx, BinOp::ShrS, lo, c63);
            set_i128(ctx, &x.dest, lo, hi);
            Ok(true)
        }
        // trunc i128 V to iN (N ≤ 64) → the low word, masked when N < 64.
        I::Trunc(x) if is_i128(&x.operand) => {
            let to = int_bits(x.to_type.as_ref())
                .ok_or_else(|| Error::Unsupported("i128 trunc to non-int".into()))?;
            let (lo, _hi) = i128_parts(ctx, &x.operand)?;
            let v = if to >= 64 {
                lo
            } else {
                emit_trunc(ctx, lo, 64, to)
            };
            finish(ctx, &x.dest, v)?;
            Ok(true)
        }
        // `freeze i128` — identity on the `(lo, hi)` pair (our IR is total: `undef`/`poison` already
        // resolve to a defined 0 and no poison propagates, §3c). clang emits it on `udiv`/`urem`
        // operands at `-O2`; without this arm the generic scalar `freeze` mishandles the i128 pair.
        I::Freeze(x) if is_i128(&x.operand) => {
            let (lo, hi) = i128_parts(ctx, &x.operand)?;
            set_i128(ctx, &x.dest, lo, hi);
            Ok(true)
        }
        I::And(x) if is_i128(&x.operand0) => {
            i128_bitwise(ctx, &x.operand0, &x.operand1, &x.dest, BinOp::And)
        }
        I::Or(x) if is_i128(&x.operand0) => {
            i128_bitwise(ctx, &x.operand0, &x.operand1, &x.dest, BinOp::Or)
        }
        I::Xor(x) if is_i128(&x.operand0) => {
            i128_bitwise(ctx, &x.operand0, &x.operand1, &x.dest, BinOp::Xor)
        }
        // add: lo = alo + blo; hi = ahi + bhi + carry, where carry = (lo <u alo).
        I::Add(x) if is_i128(&x.operand0) => {
            let (alo, ahi) = i128_parts(ctx, &x.operand0)?;
            let (blo, bhi) = i128_parts(ctx, &x.operand1)?;
            let lo = i64bin(ctx, BinOp::Add, alo, blo);
            let carry = i64cmp_ext(ctx, CmpOp::LtU, lo, alo);
            let hi0 = i64bin(ctx, BinOp::Add, ahi, bhi);
            let hi = i64bin(ctx, BinOp::Add, hi0, carry);
            set_i128(ctx, &x.dest, lo, hi);
            Ok(true)
        }
        // sub: lo = alo - blo; hi = ahi - bhi - borrow, where borrow = (alo <u blo).
        I::Sub(x) if is_i128(&x.operand0) => {
            let (alo, ahi) = i128_parts(ctx, &x.operand0)?;
            let (blo, bhi) = i128_parts(ctx, &x.operand1)?;
            let lo = i64bin(ctx, BinOp::Sub, alo, blo);
            let borrow = i64cmp_ext(ctx, CmpOp::LtU, alo, blo);
            let hi0 = i64bin(ctx, BinOp::Sub, ahi, bhi);
            let hi = i64bin(ctx, BinOp::Sub, hi0, borrow);
            set_i128(ctx, &x.dest, lo, hi);
            Ok(true)
        }
        // mul: lo = alo·blo; hi = umulhi(alo,blo) + alo·bhi + ahi·blo (mod 2⁶⁴).
        I::Mul(x) if is_i128(&x.operand0) => {
            let (alo, ahi) = i128_parts(ctx, &x.operand0)?;
            let (blo, bhi) = i128_parts(ctx, &x.operand1)?;
            let lo = i64bin(ctx, BinOp::Mul, alo, blo);
            let mh = emit_umulhi(ctx, alo, blo);
            let ab = i64bin(ctx, BinOp::Mul, alo, bhi);
            let cd = i64bin(ctx, BinOp::Mul, ahi, blo);
            let t = i64bin(ctx, BinOp::Add, mh, ab);
            let hi = i64bin(ctx, BinOp::Add, t, cd);
            set_i128(ctx, &x.dest, lo, hi);
            Ok(true)
        }
        I::Shl(x) if is_i128(&x.operand0) => {
            i128_shift(ctx, &x.operand0, &x.operand1, &x.dest, I128ShiftKind::Shl)
        }
        I::LShr(x) if is_i128(&x.operand0) => {
            i128_shift(ctx, &x.operand0, &x.operand1, &x.dest, I128ShiftKind::LShr)
        }
        I::AShr(x) if is_i128(&x.operand0) => {
            i128_shift(ctx, &x.operand0, &x.operand1, &x.dest, I128ShiftKind::AShr)
        }
        // div/rem: the synthesized 128÷128 long-division helper (`__svm_udivmod128`) returns both
        // quotient and remainder; signed forms abs the operands and re-sign the result.
        I::UDiv(x) if is_i128(&x.operand0) => {
            i128_divrem(ctx, &x.operand0, &x.operand1, &x.dest, false, false)
        }
        I::SDiv(x) if is_i128(&x.operand0) => {
            i128_divrem(ctx, &x.operand0, &x.operand1, &x.dest, true, false)
        }
        I::URem(x) if is_i128(&x.operand0) => {
            i128_divrem(ctx, &x.operand0, &x.operand1, &x.dest, false, true)
        }
        I::SRem(x) if is_i128(&x.operand0) => {
            i128_divrem(ctx, &x.operand0, &x.operand1, &x.dest, true, true)
        }
        _ => Ok(false),
    }
}

/// Conditionally two's-complement negate an i128 `(lo, hi)` pair: returns `cond ? -value : value`.
/// `-value = (0 - lo, 0 - hi - borrow)` with `borrow = (lo != 0)`, selected per word on `cond` (an
/// `i32` boolean). Used for i128 signed div/rem (abs the operands, re-sign the result).
fn i128_select_neg(ctx: &mut BlockCtx, lo: ValIdx, hi: ValIdx, cond: ValIdx) -> (ValIdx, ValIdx) {
    let zero = ctx.const_i64(0);
    let nlo = i64bin(ctx, BinOp::Sub, zero, lo);
    let borrow = i64cmp_ext(ctx, CmpOp::LtU, zero, lo); // (0 <u lo) == (lo != 0)
    let nhi0 = i64bin(ctx, BinOp::Sub, zero, hi);
    let nhi = i64bin(ctx, BinOp::Sub, nhi0, borrow);
    let rlo = ctx.push(Inst::Select {
        cond,
        a: nlo,
        b: lo,
    });
    let rhi = ctx.push(Inst::Select {
        cond,
        a: nhi,
        b: hi,
    });
    (rlo, rhi)
}

/// Lower an i128 `udiv`/`sdiv`/`urem`/`srem` via the synthesized 128÷128 helper. The helper is
/// unsigned, so a signed op abs-es both operands first and re-signs the result: a quotient is
/// negative iff the operand signs differ; a remainder takes the **dividend's** sign (C99 truncation
/// toward zero). `Ok(true)` — an i128 op was handled.
fn i128_divrem(
    ctx: &mut BlockCtx,
    op0: &Operand,
    op1: &Operand,
    dest: &Name,
    signed: bool,
    rem: bool,
) -> Result<bool, Error> {
    let udivmod = ctx.helpers.udivmod128.ok_or(Error::Unsupported(
        "i128 div/rem helper not registered (uses_i128_divrem missed it)".into(),
    ))?;
    let (alo, ahi) = i128_parts(ctx, op0)?;
    let (blo, bhi) = i128_parts(ctx, op1)?;
    // For signed, the sign of an i128 is the sign of its high word (icmp slt hi, 0).
    let (sign_a, sign_b) = if signed {
        let za = ctx.const_i64(0);
        let sa = ctx.push(Inst::IntCmp {
            ty: IntTy::I64,
            op: CmpOp::LtS,
            a: ahi,
            b: za,
        });
        let sb = ctx.push(Inst::IntCmp {
            ty: IntTy::I64,
            op: CmpOp::LtS,
            a: bhi,
            b: za,
        });
        (sa, sb)
    } else {
        (0, 0)
    };
    let (nlo, nhi, dlo, dhi) = if signed {
        let (nlo, nhi) = i128_select_neg(ctx, alo, ahi, sign_a);
        let (dlo, dhi) = i128_select_neg(ctx, blo, bhi, sign_b);
        (nlo, nhi, dlo, dhi)
    } else {
        (alo, ahi, blo, bhi)
    };
    let parts = ctx.push_multi(
        Inst::Call {
            func: udivmod,
            args: vec![nlo, nhi, dlo, dhi],
        },
        4,
    );
    // parts = [q_lo, q_hi, r_lo, r_hi]; pick the quotient or the remainder.
    let (lo, hi) = if rem {
        (parts[2], parts[3])
    } else {
        (parts[0], parts[1])
    };
    let (lo, hi) = if signed {
        // quotient negative iff operand signs differ; remainder takes the dividend's sign.
        let neg = if rem {
            sign_a
        } else {
            ctx.push(Inst::IntBin {
                ty: IntTy::I32,
                op: BinOp::Xor,
                a: sign_a,
                b: sign_b,
            })
        };
        i128_select_neg(ctx, lo, hi, neg)
    } else {
        (lo, hi)
    };
    set_i128(ctx, dest, lo, hi);
    Ok(true)
}

fn translate_inst(ctx: &mut BlockCtx, instr: &Instruction, types: &Types) -> Result<(), Error> {
    use Instruction as I;
    // The op's integer width, from operand0 (both operands share a type in LLVM binops).
    // The op's integer width. A `<4 x i32>` operand (auto-vectorized) returns a harmless `I32` — `bin`
    // detects the vector and lowers lane-wise, ignoring this; computing it before `bin` must not fail.
    let bin_ty = |o: &Operand| -> Result<IntTy, Error> {
        match val_type(o.get_type(types).as_ref())? {
            ValType::I64 => Ok(IntTy::I64),
            _ => Ok(IntTy::I32),
        }
    };
    // The op's float width (f32/f64), likewise.
    let fty =
        |o: &Operand| -> Result<FloatTy, Error> { float_ty(val_type(o.get_type(types).as_ref())?) };

    // Wider-than-128 / sub-128 vector ops legalize to fixed-128 `v128` chunks + a scalar tail (I2
    // step 1), handled entirely here. `Ok(false)` means no wide vector is involved — fall through.
    if lower_wide(ctx, instr, types)? {
        return Ok(());
    }

    // `<N x i1>` boolean masks (vector `icmp`/`fcmp`/`select`/movemask) have no first-class svm-ir
    // type; they are held lane-wise and scalarized here. `Ok(false)` ⇒ not a mask op — fall through.
    if lower_mask(ctx, instr, types)? {
        return Ok(());
    }

    // A `landingpad` binds the `{ptr,i32}` exception aggregate from the reserved EH slots: field 0
    // the in-flight exception object pointer (`cur_exn`), field 1 the type selector (`cur_sel`) the
    // unwinding throw stored. clang's `extractvalue 0/1` then reads them back through the `agg` table
    // (the same field-wise model as a multi-result call), and the catch dispatch compares the
    // selector against each clause's `llvm.eh.typeid.for`.
    if let I::LandingPad(lp) = instr {
        let base = ctx.eh_base()?;
        let exn_addr = ctx.const_i64((base + EH_EXN_O) as i64);
        let exn = ctx.push(Inst::Load {
            op: LoadOp::I64,
            addr: exn_addr,
            offset: 0,
            align: 8,
        });
        let sel_addr = ctx.const_i64((base + EH_SEL_O) as i64);
        let sel = ctx.push(Inst::Load {
            op: LoadOp::I32,
            addr: sel_addr,
            offset: 0,
            align: 4,
        });
        if let Some(&vid) = ctx.s.name2id.get(&lp.dest) {
            ctx.agg.insert(vid, vec![exn, sel]);
        }
        return Ok(());
    }

    // I14 tier 3: general i128 ops lower to 64-bit ops over a materialized `(lo, hi)` pair here.
    // `Ok(false)` ⇒ not an i128 op (fall through); an unsupported i128 shape fails closed.
    if lower_i128(ctx, instr, types)? {
        return Ok(());
    }

    // No-result instructions (effects only): handle and return early.
    if let I::Store(st) = instr {
        // A `store atomic` (seq-cst): a native `iN.atomic.store` for i32/i64, or the narrow path for
        // i8/i16 (an `xchg` CAS loop over the enclosing word, discarding the old value).
        if st.atomicity.is_some() {
            let addr = ctx.operand(&st.address)?;
            let value = ctx.operand(&st.value)?;
            match atom_width(st.value.get_type(types).as_ref())? {
                AtomWidth::Wide(ty) => ctx.push_effect(Inst::AtomicStore {
                    ty,
                    addr,
                    value,
                    offset: 0,
                    order: Ordering::SeqCst,
                }),
                AtomWidth::Narrow(w) => {
                    lower_narrow_atomic_rmw(ctx, addr, value, w, NARROW_RMW_XCHG)?;
                }
            }
            return Ok(());
        }
        let addr = ctx.operand(&st.address)?;
        let value = ctx.operand(&st.value)?;
        // A 128-bit vector store is a 16-byte `v128.store`; everything else is a width-tagged `store`.
        if vec128_shape(st.value.get_type(types).as_ref()).is_some() {
            ctx.push_effect(Inst::V128Store {
                addr,
                value,
                offset: 0,
                align: 0,
            });
        } else if let Some(bits) = nonstd_int_bits(st.value.get_type(types).as_ref()) {
            // Non-power-of-two integer store (e.g. an `i56` niche field): write exactly its bytes.
            store_nonstd_int(ctx, addr, value, bits);
        } else {
            let op = store_op(st.value.get_type(types).as_ref())?;
            ctx.push_effect(Inst::Store {
                op,
                addr,
                value,
                offset: 0,
                align: 0,
            });
        }
        return Ok(());
    }
    // `cmpxchg` (seq-cst) yields a `{ iN old, i1 success }` pair (§3a multi-result). The CAS gives the
    // old value; success is `old == expected`. i32/i64 → `iN.atomic.cmpxchg`; i8/i16 → the narrow CAS.
    if let I::CmpXchg(cx) = instr {
        let addr = ctx.operand(&cx.address)?;
        let expected = ctx.operand(&cx.expected)?;
        let replacement = ctx.operand(&cx.replacement)?;
        let (old, exp_cmp, cmp_ty) = match atom_width(cx.expected.get_type(types).as_ref())? {
            AtomWidth::Wide(ty) => {
                let old = ctx.push(Inst::AtomicCmpxchg {
                    ty,
                    addr,
                    expected,
                    replacement,
                    offset: 0,
                    order: Ordering::SeqCst,
                });
                (old, expected, ty)
            }
            // The narrow helper returns the old field and the masked `expected`, both `i32`-width.
            AtomWidth::Narrow(w) => {
                let (old, masked_exp) =
                    lower_narrow_atomic_cas(ctx, addr, expected, replacement, w)?;
                (old, masked_exp, IntTy::I32)
            }
        };
        let success = ctx.push(Inst::IntCmp {
            ty: cmp_ty,
            op: CmpOp::Eq,
            a: old,
            b: exp_cmp,
        });
        if let Some(&vid) = ctx.s.name2id.get(&cx.dest) {
            ctx.agg.insert(vid, vec![old, success]);
        }
        return Ok(());
    }
    if let I::Call(c) = instr {
        if is_droppable_call(c) {
            return Ok(()); // a no-op intrinsic (lifetime/dbg/assume)
        }
        // `llvm.va_start`/`llvm.va_copy` set up the `__va_list_tag` for the overflow-only varargs ABI.
        if let Some(name) = callee_name(c) {
            if lower_va_intrinsic(ctx, c, &name)? {
                return Ok(());
            }
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
        // Saturating float→int intrinsics (`llvm.fptosi.sat`/`fptoui.sat`, Rust's float `as` casts).
        if let Some(idx) = lower_fp_sat_intrinsic(ctx, c)? {
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
        // `llvm.{u,s}{add,sub,mul}.with.overflow.iN` → the op + an overflow flag, as a 2-field
        // aggregate `{result, overflow}` (Rust's checked capacity/index arithmetic). Records the
        // aggregate itself, so nothing to bind here.
        if lower_overflow_intrinsic(ctx, c, types)? {
            return Ok(());
        }
        // `llvm.vector.reduce.*` (the horizontal reduce auto-vectorization emits) → an unrolled
        // lane fold to a scalar.
        if let Some(idx) = lower_vector_reduce(ctx, c, types)? {
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
        // A `<svm.h>` low-level builtin (`__vm_map`/`__vm_fiber_*`/`__vm_atomic_*`/`__vm_cap`/…) lowers
        // to the matching SVM IR op or `Memory` capability call. Gated on the name being external (a
        // guest-defined function of the same name shadows it), like the libc/libm rules below.
        if let Some(name) = callee_name(c) {
            if !ctx.name2idx.contains_key(&name) && lower_vm_builtin(ctx, c, &name)? {
                return Ok(());
            }
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
        // A `<setjmp.h>` non-local jump (`setjmp`/`longjmp`, and the `_setjmp`/`sig*` variants) lowers
        // to the `SetJmp`/`LongJmp` core ops. Gated external (a guest-defined function shadows it).
        if let Some(name) = callee_name(c) {
            if !ctx.name2idx.contains_key(&name) && lower_setjmp_call(ctx, c, &name)? {
                return Ok(());
            }
            // The C++ EH runtime (`__cxa_*` / `llvm.eh.typeid.for`) — see `lower_eh_call`. `__cxa_throw`
            // is a plain `call` (a thrower with no local handler), lowered inline to a store + unwind.
            if !ctx.name2idx.contains_key(&name) && lower_eh_call(ctx, c, &name)? {
                return Ok(());
            }
        }
        // A call to a Rust panic/abort lang item (`-C panic=abort`) — or the C library `abort()` —
        // lowers to a trap: drop the call, since it is `noreturn` and LLVM always follows it with
        // `unreachable`, which the on-ramp traps on (§3b/§5). This is what lets real Rust (non-elidable
        // panic paths) and C programs (`abort`/`assert` failure paths, e.g. Lua's `lua_assert`)
        // translate. Gated external so a guest-defined `abort` shadows it.
        if let Some(name) = callee_name(c) {
            if !ctx.name2idx.contains_key(&name) && (is_rust_abort_call(&name) || name == "abort") {
                return Ok(());
            }
        }
        // Pass the callee its own data-stack frame at `sp + frame_size` (§3d), then the mapped
        // arguments. The IR signature is `(sp, c-args…)`, so the callee's frame never overlaps ours.
        let sp = ctx.sp()?;
        let fs = ctx.const_i64(ctx.frame_size as i64);
        let callee_sp = ctx.add_i64(sp, fs);
        let mut args = vec![callee_sp];
        // A direct call to a `(...)` function (§varargs): only the fixed parameters are IR arguments;
        // the variadic arguments are marshaled into this frame's scratch (one 8-byte slot each, the
        // overflow-area layout clang's lowered `va_arg` reads), and a pointer to that scratch is
        // deposited at the callee's reserved frame slot (`callee_sp + 0`) for its `va_start`.
        let fixed = match c.function_ty.as_ref() {
            Type::FuncType {
                param_types,
                is_var_arg: true,
                ..
            } if callee_name(c).is_some() => Some(param_types.len()),
            _ => None,
        };
        if let Some(fixed) = fixed {
            let scratch_off = *ctx.frame.get(&VARARG_SCRATCH).ok_or_else(|| {
                Error::Unsupported("varargs call without reserved scratch".into())
            })?;
            let area = {
                let k = ctx.const_i64(scratch_off as i64);
                ctx.add_i64(sp, k)
            };
            for (i, (a, _)) in c.arguments.iter().enumerate().skip(fixed) {
                let aty = a.get_type(types);
                // Each variadic argument occupies one 8-byte overflow slot; a 16-byte `v128` (or any
                // aggregate, which `store_op` rejects below) would need a wider slot + stride.
                if matches!(val_type(aty.as_ref()), Ok(ValType::V128)) {
                    return unsup("varargs argument wider than 8 bytes");
                }
                let op = store_op(aty.as_ref())?;
                let value = ctx.operand(a)?;
                ctx.push_effect(Inst::Store {
                    op,
                    addr: area,
                    value,
                    offset: (i - fixed) as u64 * 8,
                    align: 0,
                });
            }
            ctx.push_effect(Inst::Store {
                op: svm_ir::StoreOp::I64,
                addr: callee_sp,
                value: area,
                offset: 0,
                align: 0,
            });
            for (a, _attrs) in c.arguments.iter().take(fixed) {
                args.push(ctx.operand(a)?);
            }
        } else {
            for (a, _attrs) in &c.arguments {
                args.push(ctx.operand(a)?);
            }
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
        // A tail-position call (`tail`/`musttail`, the block's `ret` returns exactly this result)
        // becomes a `return_call` terminator: the IR frame is replaced rather than nested, so an
        // unbounded tail-recursion / mutual-recursion chain runs in constant native-stack space.
        // The callee still gets `sp + frame_size` (frame strictly above ours), so this is correct
        // even if a pointer into our frame escaped — and is data-stack-constant for the common
        // leaf-frame (`frame_size == 0`) case. (Reclaiming a non-empty caller frame would need
        // `musttail` detection, which the LLVM-C binding doesn't expose.)
        if ctx.tail_return {
            let term = match inst {
                Inst::Call { func, args } => Terminator::ReturnCall { func, args },
                Inst::CallIndirect { ty, idx, args } => {
                    Terminator::ReturnCallIndirect { ty, idx, args }
                }
                _ => unreachable!("tail call lowered to a non-call inst"),
            };
            ctx.pending_tail = Some(term);
            return Ok(());
        }
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
    // i128 is held as a **pair of i64 halves** (`[lo, hi]`) in the aggregate side-table — svm-IR has no
    // 128-bit type. This covers what `-O2` actually produces: clang coalesces a 16-byte struct/array
    // equality (e.g. comparing `Known`'s `[u8;16]` payload) into a `load i128` + `icmp eq/ne i128`.
    // Only those two ops are supported; any other i128 use stays a clean `Unsupported`.
    if let I::Load(l) = instr {
        if matches!(l.loaded_ty.as_ref(), Type::IntegerType { bits: 128 }) {
            let addr = ctx.operand(&l.address)?;
            let lo = ctx.push(Inst::Load {
                op: svm_ir::LoadOp::I64,
                addr,
                offset: 0,
                align: 0,
            });
            let c8 = ctx.const_i64(8);
            let hi_addr = ctx.add_i64(addr, c8);
            let hi = ctx.push(Inst::Load {
                op: svm_ir::LoadOp::I64,
                addr: hi_addr,
                offset: 0,
                align: 0,
            });
            if let Some(&vid) = ctx.s.name2id.get(&l.dest) {
                ctx.agg.insert(vid, vec![lo, hi]);
            }
            return Ok(());
        }
    }
    if let I::ICmp(x) = instr {
        if matches!(
            x.operand0.get_type(types).as_ref(),
            Type::IntegerType { bits: 128 }
        ) {
            let (alo, ahi) = i128_parts(ctx, &x.operand0)?;
            let (blo, bhi) = i128_parts(ctx, &x.operand1)?;
            let r = i128_icmp(ctx, x.predicate, alo, ahi, blo, bhi);
            if let Some(&vid) = ctx.s.name2id.get(&x.dest) {
                ctx.idx_of.insert(vid, r);
            }
            return Ok(());
        }
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
            // A `load atomic` (seq-cst): a native `iN.atomic.load` for i32/i64; for i8/i16, an
            // atomic load of the enclosing aligned 32-bit word then extract the field — atomic for
            // the narrow value (no CAS needed: a single aligned word read sees a consistent byte).
            if l.atomicity.is_some() {
                let v = match atom_width(l.loaded_ty.as_ref())? {
                    AtomWidth::Wide(ty) => ctx.push(Inst::AtomicLoad {
                        ty,
                        addr,
                        offset: 0,
                        order: Ordering::SeqCst,
                    }),
                    AtomWidth::Narrow(w) => lower_narrow_atomic_load(ctx, addr, w)?,
                };
                (&l.dest, v)
            } else if vec128_shape(l.loaded_ty.as_ref()).is_some() {
                (
                    &l.dest,
                    ctx.push(Inst::V128Load {
                        addr,
                        offset: 0,
                        align: 0,
                    }),
                )
            } else if let Some(bits) = nonstd_int_bits(l.loaded_ty.as_ref()) {
                // Non-power-of-two integer load (e.g. an `i56` niche-discriminant field): svm-IR has no
                // `iN` load, so read the enclosing `i64` and mask to N bits. The window-bounded read may
                // touch up to 3 padding/adjacent bytes, discarded by the mask (the narrow-int path
                // over-reads the same way); a read straddling the window end traps, which is safe.
                let raw = ctx.push(Inst::Load {
                    op: svm_ir::LoadOp::I64,
                    addr,
                    offset: 0,
                    align: 0,
                });
                let m = ctx.const_i64(((1u64 << bits) - 1) as i64);
                let v = ctx.push(Inst::IntBin {
                    ty: IntTy::I64,
                    op: BinOp::And,
                    a: raw,
                    b: m,
                });
                (&l.dest, v)
            } else {
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
        }
        // `atomicrmw <op>` (seq-cst), yielding the old value. i32/i64 with a natively-supported op
        // → `iN.atomic.rmw`; i8/i16 (or any width via the op map) → the narrow CAS loop. Unsupported
        // ops (nand/min/max/float) fail-closed for now.
        I::AtomicRMW(rmw) => {
            let addr = ctx.operand(&rmw.address)?;
            let value = ctx.operand(&rmw.value)?;
            let v = match atom_width(rmw.value.get_type(types).as_ref())? {
                AtomWidth::Wide(ty) => {
                    let op = rmw_op(rmw.operation).ok_or_else(|| {
                        Error::Unsupported(format!("atomicrmw {:?}", rmw.operation))
                    })?;
                    ctx.push(Inst::AtomicRmw {
                        ty,
                        op,
                        addr,
                        value,
                        offset: 0,
                        order: Ordering::SeqCst,
                    })
                }
                AtomWidth::Narrow(w) => {
                    let op = narrow_rmw_opcode(rmw.operation).ok_or_else(|| {
                        Error::Unsupported(format!("narrow atomicrmw {:?}", rmw.operation))
                    })?;
                    lower_narrow_atomic_rmw(ctx, addr, value, w, op)?
                }
            };
            (&rmw.dest, v)
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
            types,
        )?,
        I::Sub(x) => bin(
            ctx,
            &x.dest,
            bin_ty(&x.operand0)?,
            BinOp::Sub,
            &x.operand0,
            &x.operand1,
            types,
        )?,
        I::Mul(x) => bin(
            ctx,
            &x.dest,
            bin_ty(&x.operand0)?,
            BinOp::Mul,
            &x.operand0,
            &x.operand1,
            types,
        )?,
        I::UDiv(x) => bin(
            ctx,
            &x.dest,
            bin_ty(&x.operand0)?,
            BinOp::DivU,
            &x.operand0,
            &x.operand1,
            types,
        )?,
        I::SDiv(x) => bin(
            ctx,
            &x.dest,
            bin_ty(&x.operand0)?,
            BinOp::DivS,
            &x.operand0,
            &x.operand1,
            types,
        )?,
        I::URem(x) => bin(
            ctx,
            &x.dest,
            bin_ty(&x.operand0)?,
            BinOp::RemU,
            &x.operand0,
            &x.operand1,
            types,
        )?,
        I::SRem(x) => bin(
            ctx,
            &x.dest,
            bin_ty(&x.operand0)?,
            BinOp::RemS,
            &x.operand0,
            &x.operand1,
            types,
        )?,
        I::And(x) => bin(
            ctx,
            &x.dest,
            bin_ty(&x.operand0)?,
            BinOp::And,
            &x.operand0,
            &x.operand1,
            types,
        )?,
        I::Or(x) => bin(
            ctx,
            &x.dest,
            bin_ty(&x.operand0)?,
            BinOp::Or,
            &x.operand0,
            &x.operand1,
            types,
        )?,
        I::Xor(x) => bin(
            ctx,
            &x.dest,
            bin_ty(&x.operand0)?,
            BinOp::Xor,
            &x.operand0,
            &x.operand1,
            types,
        )?,
        I::Shl(x) => bin(
            ctx,
            &x.dest,
            bin_ty(&x.operand0)?,
            BinOp::Shl,
            &x.operand0,
            &x.operand1,
            types,
        )?,
        I::LShr(x) => bin(
            ctx,
            &x.dest,
            bin_ty(&x.operand0)?,
            BinOp::ShrU,
            &x.operand0,
            &x.operand1,
            types,
        )?,
        I::AShr(x) => bin(
            ctx,
            &x.dest,
            bin_ty(&x.operand0)?,
            BinOp::ShrS,
            &x.operand0,
            &x.operand1,
            types,
        )?,
        I::ICmp(x) => {
            let ty = bin_ty(&x.operand0)?;
            let op = icmp_op(x.predicate);
            let mut a = ctx.operand(&x.operand0)?;
            let mut b = ctx.operand(&x.operand1)?;
            // A **narrow** (< i32) operand sits in an i32 container whose high bits are unspecified
            // (e.g. a zero-extended `i16` load of a *signed* value) — the i32 compare needs it
            // canonically extended: sign-extended for a signed predicate, zero-extended otherwise.
            // Without this, `icmp slt i16 (zext-loaded), 0` is always false (§3b narrow-int hazard).
            // (i32/i64/pointer operands are already full-width.)
            if let Some(w) = int_bits(x.operand0.get_type(types).as_ref()) {
                if w < 32 {
                    let signed = matches!(
                        x.predicate,
                        IntPredicate::SLT
                            | IntPredicate::SLE
                            | IntPredicate::SGT
                            | IntPredicate::SGE
                    );
                    a = emit_ext(ctx, a, w, 32, signed);
                    b = emit_ext(ctx, b, w, 32, signed);
                }
            }
            (&x.dest, ctx.push(Inst::IntCmp { ty, op, a, b }))
        }
        I::Select(x) => {
            let cond = ctx.operand(&x.condition)?;
            let a = ctx.operand(&x.true_value)?;
            let b = ctx.operand(&x.false_value)?;
            (&x.dest, ctx.push(Inst::Select { cond, a, b }))
        }
        I::Trunc(x) => {
            // A lane-wise vector `trunc` scalarizes through the unified converter (svm-ir has no
            // vector-convert op); a scalar `trunc` keeps the direct width-adjust path.
            if vec_lane_shape(x.operand.get_type(types).as_ref()).is_some() {
                return lower_vec_int_convert(
                    ctx,
                    &x.dest,
                    &x.operand,
                    x.to_type.as_ref(),
                    VConv::Trunc,
                    types,
                );
            }
            let from = src_bits(&x.operand, types)?;
            let to = int_bits(x.to_type.as_ref())
                .ok_or_else(|| Error::Unsupported("trunc to non-int".into()))?;
            let v = ctx.operand(&x.operand)?;
            (&x.dest, emit_trunc(ctx, v, from, to))
        }
        I::ZExt(x) => {
            let st = x.operand.get_type(types);
            if vec_lane_shape(st.as_ref()).is_some() || i1_vector_lanes(st.as_ref()).is_some() {
                return lower_vec_int_convert(
                    ctx,
                    &x.dest,
                    &x.operand,
                    x.to_type.as_ref(),
                    VConv::ZExt,
                    types,
                );
            }
            let from = src_bits(&x.operand, types)?;
            let to = int_bits(x.to_type.as_ref())
                .ok_or_else(|| Error::Unsupported("zext to non-int".into()))?;
            let v = ctx.operand(&x.operand)?;
            (&x.dest, emit_ext(ctx, v, from, to, false))
        }
        I::SExt(x) => {
            let st = x.operand.get_type(types);
            if vec_lane_shape(st.as_ref()).is_some() || i1_vector_lanes(st.as_ref()).is_some() {
                return lower_vec_int_convert(
                    ctx,
                    &x.dest,
                    &x.operand,
                    x.to_type.as_ref(),
                    VConv::SExt,
                    types,
                );
            }
            let from = src_bits(&x.operand, types)?;
            let to = int_bits(x.to_type.as_ref())
                .ok_or_else(|| Error::Unsupported("sext to non-int".into()))?;
            let v = ctx.operand(&x.operand)?;
            (&x.dest, emit_ext(ctx, v, from, to, true))
        }
        // Floats (f32/f64) — IEEE 754, no traps (§3b). A `<2 x float>` operand goes lane-wise
        // (`fp_binop` unpacks the packed-i64 lanes, applies the op per lane, repacks).
        I::FAdd(x) => fp_binop(ctx, &x.dest, FBinOp::Add, &x.operand0, &x.operand1, types)?,
        I::FSub(x) => fp_binop(ctx, &x.dest, FBinOp::Sub, &x.operand0, &x.operand1, types)?,
        I::FMul(x) => fp_binop(ctx, &x.dest, FBinOp::Mul, &x.operand0, &x.operand1, types)?,
        I::FDiv(x) => fp_binop(ctx, &x.dest, FBinOp::Div, &x.operand0, &x.operand1, types)?,
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
            let a = ctx.operand(&x.operand0)?;
            let b = ctx.operand(&x.operand1)?;
            (&x.dest, emit_fcmp(ctx, ty, x.predicate, a, b)?)
        }
        // A lane-wise vector int↔float / float↔float conversion scalarizes through the unified
        // float-vector converter; a scalar one keeps the direct path.
        I::FPToSI(x) if vec_lane_shape(x.operand.get_type(types).as_ref()).is_some() => {
            return lower_vec_fp_convert(
                ctx,
                &x.dest,
                &x.operand,
                x.to_type.as_ref(),
                FpConv::FToSI,
                types,
            )
        }
        I::FPToUI(x) if vec_lane_shape(x.operand.get_type(types).as_ref()).is_some() => {
            return lower_vec_fp_convert(
                ctx,
                &x.dest,
                &x.operand,
                x.to_type.as_ref(),
                FpConv::FToUI,
                types,
            )
        }
        I::SIToFP(x) if vec_lane_shape(x.operand.get_type(types).as_ref()).is_some() => {
            return lower_vec_fp_convert(
                ctx,
                &x.dest,
                &x.operand,
                x.to_type.as_ref(),
                FpConv::SIToF,
                types,
            )
        }
        I::UIToFP(x) if vec_lane_shape(x.operand.get_type(types).as_ref()).is_some() => {
            return lower_vec_fp_convert(
                ctx,
                &x.dest,
                &x.operand,
                x.to_type.as_ref(),
                FpConv::UIToF,
                types,
            )
        }
        I::FPExt(x) if vec_lane_shape(x.operand.get_type(types).as_ref()).is_some() => {
            return lower_vec_fp_convert(
                ctx,
                &x.dest,
                &x.operand,
                x.to_type.as_ref(),
                FpConv::FpExt,
                types,
            )
        }
        I::FPTrunc(x) if vec_lane_shape(x.operand.get_type(types).as_ref()).is_some() => {
            return lower_vec_fp_convert(
                ctx,
                &x.dest,
                &x.operand,
                x.to_type.as_ref(),
                FpConv::FpTrunc,
                types,
            )
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
        // `<2 x float>` lane ops (the vector is a packed `i64`). Indices are constant (0/1); a
        // dynamic lane is `Unsupported`.
        I::ExtractElement(ee) => {
            let vty = ee.vector.get_type(types);
            let i = const_int(&ee.index)
                .ok_or_else(|| Error::Unsupported("extractelement: dynamic lane".into()))?;
            let v = ctx.operand(&ee.vector)?;
            if let Some(shape) = vec128_shape(vty.as_ref()) {
                let idx = ctx.push(Inst::ExtractLane {
                    shape,
                    lane: i as u8,
                    signed: false,
                    a: v,
                });
                (&ee.dest, idx)
            } else {
                let lane_ty = vec2_lane_ty(vty.as_ref())
                    .filter(|_| i < 2)
                    .ok_or_else(|| {
                        Error::Unsupported("extractelement: unsupported vector".into())
                    })?;
                (&ee.dest, ctx.vec_lane(v, i, lane_ty))
            }
        }
        I::InsertElement(ie) => {
            let vty = ie.vector.get_type(types);
            let i = const_int(&ie.index)
                .ok_or_else(|| Error::Unsupported("insertelement: dynamic lane".into()))?;
            let v = ctx.operand(&ie.vector)?;
            let x = ctx.operand(&ie.element)?;
            if let Some(shape) = vec128_shape(vty.as_ref()) {
                let idx = ctx.push(Inst::ReplaceLane {
                    shape,
                    lane: i as u8,
                    a: v,
                    b: x,
                });
                (&ie.dest, idx)
            } else {
                let lane_ty = vec2_lane_ty(vty.as_ref())
                    .filter(|_| i < 2)
                    .ok_or_else(|| {
                        Error::Unsupported("insertelement: unsupported vector".into())
                    })?;
                let other = ctx.vec_lane(v, 1 - i, lane_ty); // the lane we keep
                let packed = if i == 0 {
                    ctx.vec_pack(x, other, lane_ty)
                } else {
                    ctx.vec_pack(other, x, lane_ty)
                };
                (&ie.dest, packed)
            }
        }
        // `shufflevector` with a constant mask. Each result lane picks a lane from the `a ++ b`
        // concatenation. `<4 x …>` → a byte-level `i8x16.shuffle` (4 bytes per 32-bit lane; the
        // common all-equal mask is a broadcast/splat); `<2 x …>` → pick + repack the scalarized lanes.
        I::ShuffleVector(sv) => {
            let vty = sv.operand0.get_type(types);
            let mask: Vec<u64> = match sv.mask.as_ref() {
                Constant::Vector(m) => m
                    .iter()
                    .map(|e| match e.as_ref() {
                        Constant::Int { value, .. } => *value,
                        _ => 0, // undef mask lane
                    })
                    .collect(),
                _ => return unsup("shufflevector: non-constant mask"),
            };
            let v0 = ctx.operand(&sv.operand0)?;
            let v1 = ctx.operand(&sv.operand1)?;
            // The v128 fast-path is a same-width permute (result lane count == input lane count) →
            // one `i8x16.shuffle`. A width-changing shuffle (clang's auto-vectorizer emits these to
            // widen/narrow/concat, e.g. `<16 x i8>` → `<8 x i8>`) falls through to the generic
            // scalarize path below, which gathers lanes per the mask into the result's own shape.
            if let Some(shape) =
                vec128_shape(vty.as_ref()).filter(|s| mask.len() == s.lanes() as usize)
            {
                let n = shape.lanes() as u64; // lanes per input vector
                let lb = shape.lane_bytes() as u64; // bytes per lane
                                                    // `lb`-byte lanes; source lane `src` (0..2n over the `a ++ b` concat) → concat byte
                                                    // base (src<n ? lb*src : 16 + lb*(src-n)). An undef/oob mask lane reads lane 0 of `a`.
                let mut lanes = [0u8; 16];
                for (k, &src) in mask.iter().enumerate() {
                    let base = if src < n {
                        lb * src
                    } else if src < 2 * n {
                        16 + lb * (src - n)
                    } else {
                        0
                    } as u8;
                    for b in 0..lb as u8 {
                        lanes[lb as usize * k + b as usize] = base + b;
                    }
                }
                (
                    &sv.dest,
                    ctx.push(Inst::Shuffle {
                        lanes,
                        a: v0,
                        b: v1,
                    }),
                )
            } else if let Some(lane_ty) = vec2_lane_ty(vty.as_ref()).filter(|_| mask.len() == 2) {
                // vec2 → vec2: pick + repack the two scalarized lanes.
                let pick = |ctx: &mut BlockCtx, m: u64| {
                    if m < 2 {
                        ctx.vec_lane(v0, m, lane_ty)
                    } else {
                        ctx.vec_lane(v1, m.saturating_sub(2).min(1), lane_ty)
                    }
                };
                let l0 = pick(ctx, mask[0]);
                let l1 = pick(ctx, mask[1]);
                (&sv.dest, ctx.vec_pack(l0, l1, lane_ty))
            } else {
                // A shuffle whose operands and result use **different representations** (e.g. two
                // `<2 x float>`s concatenated into a `<4 x float>` `v128`). Scalarize generically:
                // explode both operands' lanes, gather per the mask, repack into the result's shape.
                let Some(elem) = vec_lane_shape(vty.as_ref()) else {
                    return unsup("shufflevector: unsupported vector");
                };
                let a = vec_explode(ctx, &sv.operand0, types, false)?;
                let b = vec_explode(ctx, &sv.operand1, types, false)?;
                let n = a.len();
                let mut res = Vec::with_capacity(mask.len());
                for &mr in &mask {
                    let idx = mr as usize;
                    res.push(if idx < n {
                        a[idx]
                    } else if idx < 2 * n {
                        b[idx - n]
                    } else {
                        a[0]
                    });
                }
                return bind_shuffle_result(ctx, &sv.dest, &res, elem);
            }
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

/// The width-tagged `LoadOp` for a single tail lane of a legalized wide vector (narrow integer
/// lanes zero-extend into the `i32` container, matching the chunk lanes' `ExtractLane` convention).
fn lane_load_op(shape: svm_ir::VShape) -> svm_ir::LoadOp {
    use svm_ir::{LoadOp as L, VShape};
    match shape {
        VShape::I8x16 => L::I32_8U,
        VShape::I16x8 => L::I32_16U,
        VShape::I32x4 => L::I32,
        VShape::I64x2 => L::I64,
        VShape::F32x4 => L::F32,
        VShape::F64x2 => L::F64,
    }
}

/// The width-tagged `StoreOp` for a single tail lane of a legalized wide vector.
fn lane_store_op(shape: svm_ir::VShape) -> svm_ir::StoreOp {
    use svm_ir::{StoreOp as S, VShape};
    match shape {
        VShape::I8x16 => S::I32_8,
        VShape::I16x8 => S::I32_16,
        VShape::I32x4 => S::I32,
        VShape::I64x2 => S::I64,
        VShape::F32x4 => S::F32,
        VShape::F64x2 => S::F64,
    }
}

/// A constant for one tail lane (its lane-scalar type), from the lane's little-endian word.
fn tail_const_inst(shape: svm_ir::VShape, w: u64) -> Inst {
    match shape.lane_val() {
        ValType::I64 => Inst::ConstI64(w as i64),
        ValType::F32 => Inst::ConstF32(w as u32),
        ValType::F64 => Inst::ConstF64(w),
        _ => Inst::ConstI32(w as u32 as i32), // i8/i16/i32 lane → i32 container
    }
}

/// The chunk op for a wide integer lane binop: lane arithmetic (`VIntBin`) or whole-vector bitwise
/// (`VBitBin`); the matching scalar `BinOp` is applied to tail lanes.
enum WIntChunk {
    Arith(svm_ir::VIntBinOp),
    Bit(svm_ir::VBitBinOp),
}

/// Lower a wide **integer** lane binop per chunk + per tail lane, binding `dest` to the result
/// parts. Returns `Ok(false)` if the operands aren't a wide vector (take the normal path).
fn wide_int_binop(
    ctx: &mut BlockCtx,
    types: &Types,
    dest: &Name,
    a: &Operand,
    b: &Operand,
    chunk: WIntChunk,
    tail_op: BinOp,
) -> Result<bool, Error> {
    let Some(layout) = wide_vec_layout(a.get_type(types).as_ref()) else {
        return Ok(false);
    };
    let pa = ctx.wide_operand(a, layout)?;
    let pb = ctx.wide_operand(b, layout)?;
    let tail_ty = int_ty(layout.shape.lane_val())?;
    let mut out = Vec::with_capacity(layout.nparts());
    for i in 0..layout.full_chunks {
        out.push(ctx.push(match chunk {
            WIntChunk::Arith(op) => Inst::VIntBin {
                shape: layout.shape,
                op,
                a: pa[i],
                b: pb[i],
            },
            WIntChunk::Bit(op) => Inst::VBitBin {
                op,
                a: pa[i],
                b: pb[i],
            },
        }));
    }
    for i in layout.full_chunks..layout.nparts() {
        out.push(ctx.push(Inst::IntBin {
            ty: tail_ty,
            op: tail_op,
            a: pa[i],
            b: pb[i],
        }));
    }
    ctx.bind_wide(dest, out);
    Ok(true)
}

/// Lower a wide **lane-wise shift** by a uniform amount (the wide counterpart of the 128-bit `VShift`
/// path): each `v128` chunk shifts via `VShift` (one scalar amount for all its lanes), each scalar tail
/// lane via `IntBin`. The amount must be a constant splat — `const_splat_int` — since `VShift` takes a
/// single scalar count; a per-lane-varying amount stays fail-closed (rare in auto-vectorized code).
fn wide_int_shift(
    ctx: &mut BlockCtx,
    types: &Types,
    dest: &Name,
    a: &Operand,
    b: &Operand,
    op: svm_ir::VShiftOp,
    tail_op: BinOp,
) -> Result<bool, Error> {
    let Some(layout) = wide_vec_layout(a.get_type(types).as_ref()) else {
        return Ok(false);
    };
    let Some(amt) = const_splat_int(b) else {
        return unsup(format!(
            "wide vector shift {op:?} with non-constant-splat amount"
        ));
    };
    let pa = ctx.wide_operand(a, layout)?;
    let tail_ty = int_ty(layout.shape.lane_val())?;
    // `VShift`'s amount is a scalar `i32` (one count for every lane of every chunk); the scalar tail
    // shifts by the same count in the lane's own integer type.
    let amt_i32 = ctx.push(Inst::ConstI32(amt as i32));
    let tail_amt = match tail_ty {
        IntTy::I64 => ctx.push(Inst::ConstI64(amt as i64)),
        _ => amt_i32,
    };
    let mut out = Vec::with_capacity(layout.nparts());
    for &chunk in pa.iter().take(layout.full_chunks) {
        out.push(ctx.push(Inst::VShift {
            shape: layout.shape,
            op,
            a: chunk,
            amt: amt_i32,
        }));
    }
    for &lane in pa.iter().take(layout.nparts()).skip(layout.full_chunks) {
        out.push(ctx.push(Inst::IntBin {
            ty: tail_ty,
            op: tail_op,
            a: lane,
            b: tail_amt,
        }));
    }
    ctx.bind_wide(dest, out);
    Ok(true)
}

/// Lower a wide **float** lane binop (`VFloatBin` per chunk + scalar `FBin` per tail lane).
fn wide_float_binop(
    ctx: &mut BlockCtx,
    types: &Types,
    dest: &Name,
    a: &Operand,
    b: &Operand,
    op: FBinOp,
) -> Result<bool, Error> {
    let Some(layout) = wide_vec_layout(a.get_type(types).as_ref()) else {
        return Ok(false);
    };
    let pa = ctx.wide_operand(a, layout)?;
    let pb = ctx.wide_operand(b, layout)?;
    let vop = vfbin_op(op)?;
    let tail_ty = float_ty(layout.shape.lane_val())?;
    let mut out = Vec::with_capacity(layout.nparts());
    for i in 0..layout.full_chunks {
        out.push(ctx.push(Inst::VFloatBin {
            shape: layout.shape,
            op: vop,
            a: pa[i],
            b: pb[i],
        }));
    }
    for i in layout.full_chunks..layout.nparts() {
        out.push(ctx.push(Inst::FBin {
            ty: tail_ty,
            op,
            a: pa[i],
            b: pb[i],
        }));
    }
    ctx.bind_wide(dest, out);
    Ok(true)
}

/// Lower a wide integer lane **min/max** (`llvm.{s,u}{min,max}.vNiM`) per chunk (`VIntBin`). Tail
/// lanes would need per-lane signed/unsigned compares (a rare odd-width corner) — `Unsupported`.
fn wide_minmax(
    ctx: &mut BlockCtx,
    layout: WideLayout,
    op: svm_ir::VIntBinOp,
    pa: &[ValIdx],
    pb: &[ValIdx],
) -> Result<Vec<ValIdx>, Error> {
    if layout.tail_lanes != 0 {
        return unsup("wide vector min/max with a scalar tail (odd lane count)");
    }
    Ok((0..layout.full_chunks)
        .map(|i| {
            ctx.push(Inst::VIntBin {
                shape: layout.shape,
                op,
                a: pa[i],
                b: pb[i],
            })
        })
        .collect())
}

/// Lower a wide horizontal `llvm.vector.reduce.{add,mul,and,or,xor,smax,smin,umax,umin}` to a single
/// scalar: extract every lane (from the `v128` chunks + tail lanes) and fold it, like the 128-bit
/// reduce. Min/max with a non-empty tail is `Unsupported` (the tail lanes would need explicit
/// sign/zero extension).
fn wide_reduce(
    ctx: &mut BlockCtx,
    layout: WideLayout,
    kind: &str,
    parts: &[ValIdx],
) -> Result<ValIdx, Error> {
    let signed = matches!(kind, "smax" | "smin");
    let is_minmax = matches!(kind, "smax" | "smin" | "umax" | "umin");
    if is_minmax && layout.tail_lanes != 0 {
        return unsup(format!("wide vector.reduce.{kind} with a scalar tail"));
    }
    let lpc = layout.shape.lanes();
    let mut lanes: Vec<ValIdx> = Vec::with_capacity(layout.total_lanes());
    for &chunk in &parts[..layout.full_chunks] {
        for l in 0..lpc {
            lanes.push(ctx.push(Inst::ExtractLane {
                shape: layout.shape,
                lane: l,
                signed,
                a: chunk,
            }));
        }
    }
    // Tail lanes are already lane scalars (add/mul/and/or/xor only — guarded above for min/max).
    lanes.extend_from_slice(&parts[layout.full_chunks..]);
    let lane_ty = int_ty(layout.shape.lane_val())?;
    let mut acc = lanes[0];
    for &l in &lanes[1..] {
        acc = match kind {
            "add" => ctx.push(Inst::IntBin {
                ty: lane_ty,
                op: BinOp::Add,
                a: acc,
                b: l,
            }),
            "mul" => ctx.push(Inst::IntBin {
                ty: lane_ty,
                op: BinOp::Mul,
                a: acc,
                b: l,
            }),
            "and" => ctx.push(Inst::IntBin {
                ty: lane_ty,
                op: BinOp::And,
                a: acc,
                b: l,
            }),
            "or" => ctx.push(Inst::IntBin {
                ty: lane_ty,
                op: BinOp::Or,
                a: acc,
                b: l,
            }),
            "xor" => ctx.push(Inst::IntBin {
                ty: lane_ty,
                op: BinOp::Xor,
                a: acc,
                b: l,
            }),
            "smax" | "umax" | "smin" | "umin" => {
                let cmp = match kind {
                    "smax" => CmpOp::GtS,
                    "umax" => CmpOp::GtU,
                    "smin" => CmpOp::LtS,
                    _ => CmpOp::LtU,
                };
                let cond = ctx.push(Inst::IntCmp {
                    ty: lane_ty,
                    op: cmp,
                    a: acc,
                    b: l,
                });
                ctx.push(Inst::Select { cond, a: acc, b: l })
            }
            other => return unsup(format!("wide vector.reduce.{other}")),
        };
    }
    Ok(acc)
}

/// Legalize a wide (>128-bit) or sub-128 vector instruction into fixed-128 `v128` chunks plus a
/// scalar tail (I2 fix-sketch step 1). Returns `Ok(true)` if it fully handled `instr` (the caller
/// returns early); `Ok(false)` if `instr` involves no wide vector, so the normal lowering runs.
/// Any wide-vector form this does not explicitly handle stays fail-closed: the normal path's operand
/// resolver finds no scalar binding for a wide value and returns a clean `Unsupported` (§2a).
fn lower_wide(ctx: &mut BlockCtx, instr: &Instruction, types: &Types) -> Result<bool, Error> {
    use svm_ir::{VBitBinOp as Bit, VIntBinOp as VI};
    use Instruction as I;
    match instr {
        I::Store(st) => {
            let Some(layout) = wide_vec_layout(st.value.get_type(types).as_ref()) else {
                return Ok(false);
            };
            let addr = ctx.operand(&st.address)?;
            let parts = ctx.wide_operand(&st.value, layout)?;
            ctx.wide_store(addr, &parts, layout);
            Ok(true)
        }
        I::Load(l) => {
            let Some(layout) = wide_vec_layout(l.loaded_ty.as_ref()) else {
                return Ok(false);
            };
            let addr = ctx.operand(&l.address)?;
            let parts = ctx.wide_load(addr, layout);
            ctx.bind_wide(&l.dest, parts);
            Ok(true)
        }
        I::Add(x) => wide_int_binop(
            ctx,
            types,
            &x.dest,
            &x.operand0,
            &x.operand1,
            WIntChunk::Arith(VI::Add),
            BinOp::Add,
        ),
        I::Sub(x) => wide_int_binop(
            ctx,
            types,
            &x.dest,
            &x.operand0,
            &x.operand1,
            WIntChunk::Arith(VI::Sub),
            BinOp::Sub,
        ),
        I::Mul(x) => wide_int_binop(
            ctx,
            types,
            &x.dest,
            &x.operand0,
            &x.operand1,
            WIntChunk::Arith(VI::Mul),
            BinOp::Mul,
        ),
        I::And(x) => wide_int_binop(
            ctx,
            types,
            &x.dest,
            &x.operand0,
            &x.operand1,
            WIntChunk::Bit(Bit::And),
            BinOp::And,
        ),
        I::Or(x) => wide_int_binop(
            ctx,
            types,
            &x.dest,
            &x.operand0,
            &x.operand1,
            WIntChunk::Bit(Bit::Or),
            BinOp::Or,
        ),
        I::Xor(x) => wide_int_binop(
            ctx,
            types,
            &x.dest,
            &x.operand0,
            &x.operand1,
            WIntChunk::Bit(Bit::Xor),
            BinOp::Xor,
        ),
        // Wide lane-wise shift by a uniform amount → per-chunk `VShift` + scalar-tail shift (the wide
        // counterpart of the 128-bit shift path; a non-constant-splat amount stays fail-closed). I11:
        // an auto-vectorized widening multiply (`short` DSP fixed-point, e.g. Embench `edn`) emits a
        // `<8 x i32>` shift the 128-bit path couldn't reach.
        I::Shl(x) => wide_int_shift(
            ctx,
            types,
            &x.dest,
            &x.operand0,
            &x.operand1,
            svm_ir::VShiftOp::Shl,
            BinOp::Shl,
        ),
        I::LShr(x) => wide_int_shift(
            ctx,
            types,
            &x.dest,
            &x.operand0,
            &x.operand1,
            svm_ir::VShiftOp::ShrU,
            BinOp::ShrU,
        ),
        I::AShr(x) => wide_int_shift(
            ctx,
            types,
            &x.dest,
            &x.operand0,
            &x.operand1,
            svm_ir::VShiftOp::ShrS,
            BinOp::ShrS,
        ),
        I::FAdd(x) => wide_float_binop(ctx, types, &x.dest, &x.operand0, &x.operand1, FBinOp::Add),
        I::FSub(x) => wide_float_binop(ctx, types, &x.dest, &x.operand0, &x.operand1, FBinOp::Sub),
        I::FMul(x) => wide_float_binop(ctx, types, &x.dest, &x.operand0, &x.operand1, FBinOp::Mul),
        I::FDiv(x) => wide_float_binop(ctx, types, &x.dest, &x.operand0, &x.operand1, FBinOp::Div),
        I::ExtractElement(ee) => {
            let Some(layout) = wide_vec_layout(ee.vector.get_type(types).as_ref()) else {
                return Ok(false);
            };
            let i = const_int(&ee.index)
                .ok_or_else(|| Error::Unsupported("extractelement: dynamic lane".into()))?
                as usize;
            if i >= layout.total_lanes() {
                return unsup("extractelement: lane out of range");
            }
            let parts = ctx.wide_operand(&ee.vector, layout)?;
            let lpc = layout.shape.lanes() as usize;
            let chunked = layout.full_chunks * lpc;
            let idx = if i < chunked {
                ctx.push(Inst::ExtractLane {
                    shape: layout.shape,
                    lane: (i % lpc) as u8,
                    signed: false,
                    a: parts[i / lpc],
                })
            } else {
                parts[layout.full_chunks + (i - chunked)]
            };
            finish(ctx, &ee.dest, idx)?;
            Ok(true)
        }
        I::InsertElement(ie) => {
            let Some(layout) = wide_vec_layout(ie.vector.get_type(types).as_ref()) else {
                return Ok(false);
            };
            let i = const_int(&ie.index)
                .ok_or_else(|| Error::Unsupported("insertelement: dynamic lane".into()))?
                as usize;
            if i >= layout.total_lanes() {
                return unsup("insertelement: lane out of range");
            }
            let mut parts = ctx.wide_operand(&ie.vector, layout)?;
            let x = ctx.operand(&ie.element)?;
            let lpc = layout.shape.lanes() as usize;
            let chunked = layout.full_chunks * lpc;
            if i < chunked {
                parts[i / lpc] = ctx.push(Inst::ReplaceLane {
                    shape: layout.shape,
                    lane: (i % lpc) as u8,
                    a: parts[i / lpc],
                    b: x,
                });
            } else {
                parts[layout.full_chunks + (i - chunked)] = x;
            }
            ctx.bind_wide(&ie.dest, parts);
            Ok(true)
        }
        I::ShuffleVector(sv) => {
            let Some(src) = wide_vec_layout(sv.operand0.get_type(types).as_ref()) else {
                return Ok(false);
            };
            // A **general constant-mask** wide shuffle: each result lane picks a source lane from the
            // `operand0 ++ operand1` concatenation (a `<8 x i8>` byte-reverse `<7,6,…,0>`, a broadcast
            // `zeroinitializer`, …). Scalarize: explode both operands' lanes, gather per the mask,
            // repack into the result's representation. svm-ir's in-block `Inst::Shuffle` covers the
            // single-`v128` case; this is its legalized analog for wider/sub-128 (chunk+tail) vectors.
            let total = src.total_lanes();
            let mask: Vec<i64> = match sv.mask.as_ref() {
                Constant::AggregateZero(_) => vec![0; total],
                Constant::Vector(m) => m
                    .iter()
                    .map(|e| match e.as_ref() {
                        Constant::Int { value, .. } => *value as i64,
                        _ => 0, // undef/poison mask lane → source lane 0
                    })
                    .collect(),
                _ => return unsup("wide shufflevector: non-constant mask"),
            };
            let a_parts = ctx.wide_operand(&sv.operand0, src)?;
            let a_lanes = wide_explode_lanes(ctx, &a_parts, src);
            let b_parts = ctx.wide_operand(&sv.operand1, src)?;
            let b_lanes = wide_explode_lanes(ctx, &b_parts, src);
            let mut res = Vec::with_capacity(mask.len());
            for &mr in &mask {
                let idx = if mr < 0 { 0 } else { mr as usize };
                // `idx` in `0..total` selects `operand0`, `total..2*total` selects `operand1`; an
                // out-of-range (undef) lane reads `operand0` lane 0 (a defined value, §3c).
                let pick = if idx < total {
                    a_lanes[idx]
                } else if idx < 2 * total {
                    b_lanes[idx - total]
                } else {
                    a_lanes[0]
                };
                res.push(pick);
            }
            bind_lanes_as_vector(ctx, &sv.dest, &res, src.shape);
            Ok(true)
        }
        I::Call(c) => {
            let Some(name) = callee_name(c) else {
                return Ok(false);
            };
            // A wide horizontal reduce (`llvm.vector.reduce.*`) folds to one scalar.
            if let Some(rest) = name.strip_prefix("llvm.vector.reduce.") {
                let vec_op = &c.arguments[0].0;
                let Some(layout) = wide_vec_layout(vec_op.get_type(types).as_ref()) else {
                    return Ok(false);
                };
                if layout.shape.is_float() {
                    return unsup("wide float vector.reduce (later slice)");
                }
                let kind = rest.split('.').next().unwrap_or("");
                let parts = ctx.wide_operand(vec_op, layout)?;
                let r = wide_reduce(ctx, layout, kind, &parts)?;
                if let Some(dest) = &c.dest {
                    finish(ctx, dest, r)?;
                }
                return Ok(true);
            }
            // A wide lane-wise min/max (`llvm.{s,u}{min,max}.vNiM`).
            let base = name.rsplit_once('.').map_or(name.as_str(), |(b, _)| b);
            let vop = match base {
                "llvm.smax" => VI::MaxS,
                "llvm.smin" => VI::MinS,
                "llvm.umax" => VI::MaxU,
                "llvm.umin" => VI::MinU,
                _ => return Ok(false),
            };
            let a0 = &c.arguments[0].0;
            let Some(layout) = wide_vec_layout(a0.get_type(types).as_ref()) else {
                return Ok(false);
            };
            let pa = ctx.wide_operand(a0, layout)?;
            let pb = ctx.wide_operand(&c.arguments[1].0, layout)?;
            let out = wide_minmax(ctx, layout, vop, &pa, &pb)?;
            if let Some(dest) = &c.dest {
                ctx.bind_wide(dest, out);
            }
            Ok(true)
        }
        _ => Ok(false),
    }
}

/// `Some(N)` if `ty` is an `<N x i1>` boolean-mask vector (the type a vector `icmp`/`fcmp` produces).
fn i1_vector_lanes(ty: &Type) -> Option<usize> {
    match ty {
        Type::VectorType {
            element_type,
            num_elements,
            scalable: false,
        } => {
            matches!(element_type.as_ref(), Type::IntegerType { bits: 1 }).then_some(*num_elements)
        }
        _ => None,
    }
}

/// Resolve a `<N x i1>` mask operand to its `N` scalarized lane values (each `0`/`1` in an `i32`). A
/// local reads its recorded `mask_lanes`; a constant mask materializes per-lane `0`/`1` consts.
fn mask_operand(ctx: &mut BlockCtx, op: &Operand, n: usize) -> Result<Vec<ValIdx>, Error> {
    match op {
        Operand::LocalOperand { name, .. } => {
            let vid = *ctx
                .s
                .name2id
                .get(name)
                .ok_or_else(|| Error::Unsupported(format!("unresolved local {name:?}")))?;
            ctx.agg.get(&vid).cloned().ok_or_else(|| {
                Error::Unsupported(format!("mask value {vid} not available in block"))
            })
        }
        // `poison`/`undef` (the base of an `insertelement` mask build) and `zeroinitializer` are all
        // false; an explicit constant mask is its per-lane `i1` values.
        Operand::ConstantOperand(c) => {
            let bit = |e: &Constant| -> i32 { matches!(e, Constant::Int { value: 1, .. }) as i32 };
            let lanes: Vec<i32> = match c.as_ref() {
                Constant::Vector(elems) => elems.iter().map(|e| bit(e.as_ref())).collect(),
                _ => vec![0; n], // AggregateZero / Undef / Poison / Null
            };
            Ok(lanes
                .into_iter()
                .take(n)
                .chain(std::iter::repeat(0))
                .take(n)
                .map(|b| ctx.push(Inst::ConstI32(b)))
                .collect())
        }
        Operand::MetadataOperand => unsup("metadata mask operand"),
    }
}

/// Lower the **`<N x i1>` boolean-mask** instructions a vectorized program produces. svm-ir has no
/// first-class `<N x i1>` type, so a mask is held lane-wise (`mask_lanes`: `N` scalar `0`/`1`s) and
/// every producer/consumer is scalarized: vector `icmp`/`fcmp` → per-lane scalar compare; `select`
/// (mask condition) → per-lane scalar `select` over the exploded data; `extractelement` → the lane;
/// `insertelement`/`shufflevector` → build/permute the lanes; `bitcast … to iN` (the SIMD movemask)
/// → OR the lanes into a bitmap; `freeze` → identity. Returns `true` if handled (dispatched after
/// [`lower_wide`], before the scalar match). Each form re-verifies under `svm-verify`.
fn lower_mask(ctx: &mut BlockCtx, instr: &Instruction, types: &Types) -> Result<bool, Error> {
    use Instruction as I;
    match instr {
        // Vector integer compare → `N` scalar `IntCmp`s (narrow lanes extended per the predicate's
        // signedness so the `i32` compare orders them correctly, §3b).
        I::ICmp(x) if vec_lane_shape(x.operand0.get_type(types).as_ref()).is_some() => {
            let Some(shape) = vec_lane_shape(x.operand0.get_type(types).as_ref()) else {
                return Ok(false);
            };
            let op = icmp_op(x.predicate);
            let signed = matches!(
                x.predicate,
                IntPredicate::SLT | IntPredicate::SLE | IntPredicate::SGT | IntPredicate::SGE
            );
            let lane_ty = int_ty(shape.lane_val())?;
            let a = vec_explode(ctx, &x.operand0, types, signed)?;
            let b = vec_explode(ctx, &x.operand1, types, signed)?;
            let mut m = Vec::with_capacity(a.len());
            for (&av, &bv) in a.iter().zip(b.iter()) {
                m.push(ctx.push(Inst::IntCmp {
                    ty: lane_ty,
                    op,
                    a: av,
                    b: bv,
                }));
            }
            bind_mask(ctx, &x.dest, m);
            Ok(true)
        }
        // Vector float compare → `N` scalar `FCmp`s.
        I::FCmp(x) if vec_lane_shape(x.operand0.get_type(types).as_ref()).is_some() => {
            let Some(shape) = vec_lane_shape(x.operand0.get_type(types).as_ref()) else {
                return Ok(false);
            };
            let lane_ty = float_ty(shape.lane_val())?;
            let a = vec_explode(ctx, &x.operand0, types, false)?;
            let b = vec_explode(ctx, &x.operand1, types, false)?;
            let mut m = Vec::with_capacity(a.len());
            for (&av, &bv) in a.iter().zip(b.iter()) {
                m.push(emit_fcmp(ctx, lane_ty, x.predicate, av, bv)?);
            }
            bind_mask(ctx, &x.dest, m);
            Ok(true)
        }
        // `select <N x i1> mask, a, b` → a per-lane scalar `select` over the exploded data, repacked.
        I::Select(x) if i1_vector_lanes(x.condition.get_type(types).as_ref()).is_some() => {
            let n = i1_vector_lanes(x.condition.get_type(types).as_ref()).unwrap();
            let mask = mask_operand(ctx, &x.condition, n)?;
            let a = vec_explode(ctx, &x.true_value, types, false)?;
            let b = vec_explode(ctx, &x.false_value, types, false)?;
            let mut out = Vec::with_capacity(n);
            for k in 0..n {
                out.push(ctx.push(Inst::Select {
                    cond: mask[k],
                    a: a[k],
                    b: b[k],
                }));
            }
            let res_ty = x.true_value.get_type(types);
            vec_implode(ctx, &x.dest, &out, res_ty.as_ref(), types)?;
            Ok(true)
        }
        // `extractelement <N x i1> mask, k` → lane `k` (a clean `0`/`1`).
        I::ExtractElement(ee) if i1_vector_lanes(ee.vector.get_type(types).as_ref()).is_some() => {
            let n = i1_vector_lanes(ee.vector.get_type(types).as_ref()).unwrap();
            let k = const_int(&ee.index)
                .ok_or_else(|| Error::Unsupported("mask extractelement: dynamic lane".into()))?
                as usize;
            if k >= n {
                return unsup("mask extractelement: lane out of range");
            }
            let mask = mask_operand(ctx, &ee.vector, n)?;
            finish(ctx, &ee.dest, mask[k])?;
            Ok(true)
        }
        // `insertelement <N x i1> base, i1 x, k` → the base lanes with lane `k` replaced by `x`.
        I::InsertElement(ie) if i1_vector_lanes(ie.vector.get_type(types).as_ref()).is_some() => {
            let n = i1_vector_lanes(ie.vector.get_type(types).as_ref()).unwrap();
            let k = const_int(&ie.index)
                .ok_or_else(|| Error::Unsupported("mask insertelement: dynamic lane".into()))?
                as usize;
            if k >= n {
                return unsup("mask insertelement: lane out of range");
            }
            let mut lanes = mask_operand(ctx, &ie.vector, n)?;
            lanes[k] = ctx.operand(&ie.element)?;
            bind_mask(ctx, &ie.dest, lanes);
            Ok(true)
        }
        // `shufflevector` of masks: gather result lanes from the `a ++ b` concat per the constant mask
        // (the splat `zeroinitializer` form broadcasts lane 0 — what an `insertelement`+`shuffle`
        // mask-splat emits).
        I::ShuffleVector(sv) if i1_vector_lanes(sv.operand0.get_type(types).as_ref()).is_some() => {
            let n = i1_vector_lanes(sv.operand0.get_type(types).as_ref()).unwrap();
            let a = mask_operand(ctx, &sv.operand0, n)?;
            let b = mask_operand(ctx, &sv.operand1, n)?;
            let mask: Vec<i64> = match sv.mask.as_ref() {
                Constant::AggregateZero(_) => vec![0; n],
                Constant::Vector(m) => m
                    .iter()
                    .map(|e| match e.as_ref() {
                        Constant::Int { value, .. } => *value as i64,
                        _ => 0,
                    })
                    .collect(),
                _ => return unsup("mask shufflevector: non-constant mask"),
            };
            let mut out = Vec::with_capacity(mask.len());
            for &mr in &mask {
                let idx = if mr < 0 { 0 } else { mr as usize };
                out.push(if idx < n {
                    a[idx]
                } else if idx < 2 * n {
                    b[idx - n]
                } else {
                    a[0]
                });
            }
            bind_mask(ctx, &sv.dest, out);
            Ok(true)
        }
        // `bitcast <N x i1> to iN` — the SIMD **movemask**: gather lane `i` into bit `i` of an integer.
        I::BitCast(x) if i1_vector_lanes(x.operand.get_type(types).as_ref()).is_some() => {
            let n = i1_vector_lanes(x.operand.get_type(types).as_ref()).unwrap();
            let lanes = mask_operand(ctx, &x.operand, n)?;
            let to_bits = int_bits(x.to_type.as_ref())
                .ok_or_else(|| Error::Unsupported("mask bitcast to non-int".into()))?;
            // Lanes are `0`/`1` in `i32`; OR each into its bit position. `iN` with `N ≤ 32` builds in
            // an `i32` container, wider in `i64` (extend each lane first).
            let wide = to_bits > 32;
            let lane_ty = if wide { IntTy::I64 } else { IntTy::I32 };
            let mut acc = ctx.push(if wide {
                Inst::ConstI64(0)
            } else {
                Inst::ConstI32(0)
            });
            for (i, &l) in lanes.iter().enumerate() {
                let lw = if wide {
                    ctx.push(Inst::Convert {
                        op: ConvOp::ExtendI32U,
                        a: l,
                    })
                } else {
                    l
                };
                let shifted = if i == 0 {
                    lw
                } else {
                    let sh = ctx.push(if wide {
                        Inst::ConstI64(i as i64)
                    } else {
                        Inst::ConstI32(i as i32)
                    });
                    ctx.push(Inst::IntBin {
                        ty: lane_ty,
                        op: BinOp::Shl,
                        a: lw,
                        b: sh,
                    })
                };
                acc = ctx.push(Inst::IntBin {
                    ty: lane_ty,
                    op: BinOp::Or,
                    a: acc,
                    b: shifted,
                });
            }
            finish(ctx, &x.dest, acc)?;
            Ok(true)
        }
        // `freeze` of a mask is the identity (the lanes are already defined `0`/`1`, §3c).
        I::Freeze(f) if i1_vector_lanes(f.operand.get_type(types).as_ref()).is_some() => {
            let n = i1_vector_lanes(f.operand.get_type(types).as_ref()).unwrap();
            let lanes = mask_operand(ctx, &f.operand, n)?;
            bind_mask(ctx, &f.dest, lanes);
            Ok(true)
        }
        _ => Ok(false),
    }
}

/// Record `dest`'s scalarized mask lanes. A mask's `N` lanes live in the `agg` table (an `[i32; N]`
/// `agg_layout` is recorded in `scan_func`), so the value crosses block edges via the per-field
/// fan-out in `block_params`/`branch_args` — the `<N x i1>` analog of [`BlockCtx::bind_wide`].
fn bind_mask(ctx: &mut BlockCtx, dest: &Name, lanes: Vec<ValIdx>) {
    if let Some(&vid) = ctx.s.name2id.get(dest) {
        ctx.agg.insert(vid, lanes);
    }
}

/// Emit a binary integer op and return `(dest, result-index)`.
fn bin<'d>(
    ctx: &mut BlockCtx,
    dest: &'d Name,
    ty: IntTy,
    op: BinOp,
    a: &Operand,
    b: &Operand,
    types: &Types,
) -> Result<(&'d Name, ValIdx), Error> {
    // A 128-bit integer vector operand (auto-vectorized integer loop) lowers lane-wise to a `v128`
    // op (§17): the arithmetic ops are `VIntBin` (in the operand's lane shape), the bitwise ops are
    // whole-vector `VBitBin`. (Float vector binops go through `fp_binop`, not here, so a vector
    // operand here is integer-lane.)
    if let Some(shape) = vec128_shape(a.get_type(types).as_ref()) {
        let av = ctx.operand(a)?;
        let bv = ctx.operand(b)?;
        let inst = match op {
            BinOp::Add => Inst::VIntBin {
                shape,
                op: svm_ir::VIntBinOp::Add,
                a: av,
                b: bv,
            },
            BinOp::Sub => Inst::VIntBin {
                shape,
                op: svm_ir::VIntBinOp::Sub,
                a: av,
                b: bv,
            },
            BinOp::Mul => Inst::VIntBin {
                shape,
                op: svm_ir::VIntBinOp::Mul,
                a: av,
                b: bv,
            },
            BinOp::And => Inst::VBitBin {
                op: svm_ir::VBitBinOp::And,
                a: av,
                b: bv,
            },
            BinOp::Or => Inst::VBitBin {
                op: svm_ir::VBitBinOp::Or,
                a: av,
                b: bv,
            },
            BinOp::Xor => Inst::VBitBin {
                op: svm_ir::VBitBinOp::Xor,
                a: av,
                b: bv,
            },
            // Lane-wise shift by a uniform amount → `VShift` (§17). clang emits the amount as a
            // constant splat; a non-uniform (per-lane) amount stays fail-closed (rare in auto-vec code).
            BinOp::Shl | BinOp::ShrU | BinOp::ShrS => {
                let Some(amt) = const_splat_int(b) else {
                    return unsup(format!(
                        "vector shift {op:?} with non-constant-splat amount"
                    ));
                };
                let vop = match op {
                    BinOp::Shl => svm_ir::VShiftOp::Shl,
                    BinOp::ShrU => svm_ir::VShiftOp::ShrU,
                    _ => svm_ir::VShiftOp::ShrS,
                };
                let amtv = ctx.push(Inst::ConstI32(amt as i32));
                Inst::VShift {
                    shape,
                    op: vop,
                    a: av,
                    amt: amtv,
                }
            }
            other => {
                return unsup(format!(
                    "vector integer op {other:?} (only add/sub/mul/and/or/xor/shl/lshr/ashr)"
                ))
            }
        };
        return Ok((dest, ctx.push(inst)));
    }
    // A **2-lane 32-bit integer vector** (`<2 x i32>`, e.g. the deinterleaved widening multiply in
    // Embench `edn`'s `fir_no_red_ld`) is carried as a *packed* `i64` (lane 0 low, lane 1 high). A plain
    // `i64` op on that image is **not** lane-wise: `mul` mixes the lanes (the low product's carry and
    // the lane0×lane1 cross term land in lane 1), and `add`/`sub`/`shl`/`lshr`/`ashr` carry/shift across
    // the 32-bit lane boundary. So operate on the two `i32` lanes independently and repack. (The bitwise
    // `and`/`or`/`xor` would be lane-safe even packed, but lane-wise is uniformly correct.) This is the
    // ISSUES.md I13 root fix — previously a silent miscompile, narrowly fail-closed via a φ guard.
    if vec2_lane_ty(a.get_type(types).as_ref()) == Some(ValType::I32) {
        let la = vec_explode(ctx, a, types, false)?;
        let lb = vec_explode(ctx, b, types, false)?;
        let r0 = ctx.push(Inst::IntBin {
            ty: IntTy::I32,
            op,
            a: la[0],
            b: lb[0],
        });
        let r1 = ctx.push(Inst::IntBin {
            ty: IntTy::I32,
            op,
            a: la[1],
            b: lb[1],
        });
        let packed = ctx.vec_pack(r0, r1, ValType::I32);
        return Ok((dest, packed));
    }
    let width = int_bits(a.get_type(types).as_ref());
    let a = ctx.operand(a)?;
    let b = ctx.operand(b)?;
    let r = ctx.push(Inst::IntBin { ty, op, a, b });
    // Keep a **narrow** `iN` value **canonical**: the de-normalizing ops (`add`/`sub`/`mul`/`shl`) can
    // set bits `≥ N`, so mask the result back to its low `N` bits. Downstream `lshr`/`trunc`/unsigned-
    // compare/min-max then see clean bits (§3b widen-and-mask); `and`/`or`/`xor`/`lshr`/`div`/`rem` of
    // canonical inputs stay canonical (no extra mask). This covers **both** sub-32 widths held in an
    // `i32` (`i8`/`i16`) *and* the `33..63` widths held in an `i64`. Previously only `33..63` was masked;
    // a non-canonical `i8`/`i16` then silently miscompiled any width-sensitive consumer that read the
    // dirty container without re-masking (e.g. `umin.i8` of an `add i8 x,-1` — found via Embench
    // `qrduino`). 32-/64-bit ops are exact in their container, so nothing to do.
    let r = match width {
        Some(w) if w < 64 && matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Shl) => {
            if w <= 32 {
                mask_to(ctx, r, w)
            } else {
                mask_to_i64(ctx, r, w)
            }
        }
        _ => r,
    };
    Ok((dest, r))
}

/// Mask an `i64`-container value to its low `n` bits (`n` in `33..=63`) — the canonical form of an
/// `iN` value (`val_type`). `n == 64` is the identity (no mask).
fn mask_to_i64(ctx: &mut BlockCtx, v: ValIdx, n: u32) -> ValIdx {
    if n >= 64 {
        return v;
    }
    let m = ctx.const_i64(((1u64 << n) - 1) as i64);
    ctx.push(Inst::IntBin {
        ty: IntTy::I64,
        op: BinOp::And,
        a: v,
        b: m,
    })
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

/// Map a scalar float binop to its lane-wise `v128` form.
fn vfbin_op(op: FBinOp) -> Result<svm_ir::VFloatBinOp, Error> {
    use svm_ir::VFloatBinOp as V;
    Ok(match op {
        FBinOp::Add => V::Add,
        FBinOp::Sub => V::Sub,
        FBinOp::Mul => V::Mul,
        FBinOp::Div => V::Div,
        FBinOp::Min => V::Min,
        FBinOp::Max => V::Max,
        FBinOp::Copysign => return unsup("vector copysign"),
    })
}

/// A float binary op that may be scalar (`f32`/`f64`), a `<2 x float>` (scalarized to a packed
/// `i64`, applied per lane), or a `<4 x float>` (a native `v128` lane-wise `VFloatBin`, §17).
fn fp_binop<'d>(
    ctx: &mut BlockCtx,
    dest: &'d Name,
    op: FBinOp,
    o0: &'d Operand,
    o1: &'d Operand,
    types: &Types,
) -> Result<(&'d Name, ValIdx), Error> {
    if let Some(shape) = vec128_shape(o0.get_type(types).as_ref()).filter(|s| s.is_float()) {
        let a = ctx.operand(o0)?;
        let b = ctx.operand(o1)?;
        return Ok((
            dest,
            ctx.push(Inst::VFloatBin {
                shape,
                op: vfbin_op(op)?,
                a,
                b,
            }),
        ));
    }
    if is_vec2f(o0.get_type(types).as_ref()) {
        let a = ctx.operand(o0)?;
        let b = ctx.operand(o1)?;
        let lanes: Vec<ValIdx> = (0..2)
            .map(|i| {
                let la = ctx.vec_lane(a, i, ValType::F32);
                let lb = ctx.vec_lane(b, i, ValType::F32);
                ctx.push(Inst::FBin {
                    ty: FloatTy::F32,
                    op,
                    a: la,
                    b: lb,
                })
            })
            .collect();
        return Ok((dest, ctx.vec_pack(lanes[0], lanes[1], ValType::F32)));
    }
    let ty = float_ty(val_type(o0.get_type(types).as_ref())?)?;
    fbin(ctx, dest, ty, op, o0, o1)
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

/// The bit width of a **non-power-of-two integer** (33..=63 bits, e.g. an `i56` niche-discriminant
/// field) that the on-ramp legalizes via the enclosing `i64`. `None` for the natively-handled widths
/// (≤32 → `i32` container, 64 → `i64`) and non-integers. svm-IR has only `i32`/`i64` memory ops, so a
/// load of such a width reads the enclosing `i64` and masks to the width; a store splits into exact
/// byte-width writes (so it never clobbers an adjacent field).
fn nonstd_int_bits(ty: &Type) -> Option<u32> {
    match ty {
        Type::IntegerType { bits } if *bits > 32 && *bits < 64 => Some(*bits),
        _ => None,
    }
}

/// Store a non-power-of-two integer (`bits` in 33..=63, held in an `i64`) **byte-exactly** — exactly
/// `ceil(bits/8)` bytes, so it never clobbers an adjacent field (unlike an over-write). The low 32
/// bits go as an `i32`; the remaining 1–3 bytes as `i16`/`i8` chunks shifted down from the high half.
/// (`bits` ≥ 57 ⇒ 8 bytes ⇒ a plain `i64` store.)
fn store_nonstd_int(ctx: &mut BlockCtx, addr: ValIdx, value: ValIdx, bits: u32) {
    use svm_ir::StoreOp;
    let byte_len = bits.div_ceil(8); // 5..=8 for 33..=63
    if byte_len >= 8 {
        ctx.push_effect(Inst::Store {
            op: StoreOp::I64,
            addr,
            value,
            offset: 0,
            align: 0,
        });
        return;
    }
    // Low 32 bits at +0.
    let lo = ctx.push(Inst::Convert {
        op: ConvOp::WrapI64,
        a: value,
    });
    ctx.push_effect(Inst::Store {
        op: StoreOp::I32,
        addr,
        value: lo,
        offset: 0,
        align: 0,
    });
    // High bytes (bits 32+) shifted down, written exactly.
    let c32 = ctx.const_i64(32);
    let hi = ctx.push(Inst::IntBin {
        ty: IntTy::I64,
        op: BinOp::ShrU,
        a: value,
        b: c32,
    });
    let hi32 = ctx.push(Inst::Convert {
        op: ConvOp::WrapI64,
        a: hi,
    });
    let c4 = ctx.const_i64(4);
    let addr4 = ctx.add_i64(addr, c4);
    let rem = byte_len - 4; // 1, 2, or 3
    let wide_op = if rem == 1 {
        StoreOp::I32_8
    } else {
        StoreOp::I32_16
    };
    ctx.push_effect(Inst::Store {
        op: wide_op,
        addr: addr4,
        value: hi32,
        offset: 0,
        align: 0,
    });
    if rem == 3 {
        // The 7th byte (bits 48..56) at +6.
        let c48 = ctx.const_i64(48);
        let hi2 = ctx.push(Inst::IntBin {
            ty: IntTy::I64,
            op: BinOp::ShrU,
            a: value,
            b: c48,
        });
        let hi2_32 = ctx.push(Inst::Convert {
            op: ConvOp::WrapI64,
            a: hi2,
        });
        let c6 = ctx.const_i64(6);
        let addr6 = ctx.add_i64(addr, c6);
        ctx.push_effect(Inst::Store {
            op: StoreOp::I32_8,
            addr: addr6,
            value: hi2_32,
            offset: 0,
            align: 0,
        });
    }
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
        _ if is_vec2(ty) => Ok(L::I64), // `<2 x {float,i32}>` ≡ its packed-i64 image
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
        _ if is_vec2(ty) => Ok(S::I64), // `<2 x {float,i32}>` ≡ its packed-i64 image
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
    // A non-power-of-two source wider than 32 bits (e.g. an `i56` niche field) sits in an `i64`;
    // extend it **in i64** (the i32 helpers below don't apply). The widening target is always i64.
    if (33..64).contains(&from) {
        return if signed {
            let sh = ctx.const_i64((64 - from) as i64);
            let l = ctx.push(Inst::IntBin {
                ty: IntTy::I64,
                op: BinOp::Shl,
                a: v,
                b: sh,
            });
            ctx.push(Inst::IntBin {
                ty: IntTy::I64,
                op: BinOp::ShrS,
                a: l,
                b: sh,
            })
        } else {
            let m = ctx.const_i64(((1u64 << from) - 1) as i64);
            ctx.push(Inst::IntBin {
                ty: IntTy::I64,
                op: BinOp::And,
                a: v,
                b: m,
            })
        };
    }
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

/// Which integer-widening/narrowing conversion a vector `zext`/`sext`/`trunc` applies per lane.
#[derive(Clone, Copy)]
enum VConv {
    ZExt,
    SExt,
    Trunc,
}

/// The integer lane width of a vector whose elements are integers (`<N x iB>` → `B`). `None` if
/// `ty` is not a vector or its lanes are not integers (a float-lane vector conversion stays
/// `Unsupported` for now — a later slice).
fn vec_int_lane_bits(ty: &Type) -> Option<u32> {
    match ty {
        Type::VectorType { element_type, .. } => match element_type.as_ref() {
            Type::IntegerType { bits } => Some(*bits),
            _ => None,
        },
        _ => None,
    }
}

/// Explode a vector operand into its `N` per-lane scalars, each the lane's value in its natural
/// `i32`/`i64`/`f32`/`f64` container (integer lanes zero-extended — a per-lane conversion
/// re-canonicalizes from the known source width; float lanes are the lane's value). Covers all three
/// vector representations: a `<2 x {i32,float}>` packed in an `i64` (lane 0 = low half), a single
/// `v128` (one `ExtractLane` per lane), and a legalized wide value (`ExtractLane` per chunk lane, then
/// the already-scalar tail lanes).
fn vec_explode(
    ctx: &mut BlockCtx,
    op: &Operand,
    types: &Types,
    signed: bool,
) -> Result<Vec<ValIdx>, Error> {
    let ty = op.get_type(types);
    let ty = ty.as_ref();
    if let Some(lane) = vec2_lane_ty(ty) {
        let v = ctx.operand(op)?; // the packed i64
        let lo = ctx.push(Inst::Convert {
            op: ConvOp::WrapI64,
            a: v,
        });
        let sh = ctx.const_i64(32);
        let hi64 = ctx.push(Inst::IntBin {
            ty: IntTy::I64,
            op: BinOp::ShrU,
            a: v,
            b: sh,
        });
        let hi = ctx.push(Inst::Convert {
            op: ConvOp::WrapI64,
            a: hi64,
        });
        // A `<2 x float>`'s lanes are the two 32-bit halves reinterpreted as `f32`.
        if lane == ValType::F32 {
            let f0 = ctx.push(Inst::Cast {
                op: CastOp::ReinterpI32F32,
                a: lo,
            });
            let f1 = ctx.push(Inst::Cast {
                op: CastOp::ReinterpI32F32,
                a: hi,
            });
            return Ok(vec![f0, f1]);
        }
        return Ok(vec![lo, hi]);
    }
    if let Some(shape) = vec128_shape(ty) {
        let v = ctx.operand(op)?;
        return Ok((0..shape.lanes())
            .map(|l| {
                ctx.push(Inst::ExtractLane {
                    shape,
                    lane: l,
                    signed,
                    a: v,
                })
            })
            .collect());
    }
    if let Some(layout) = wide_vec_layout(ty) {
        let parts = ctx.wide_operand(op, layout)?;
        let lpc = layout.shape.lanes() as usize;
        let mut out = Vec::with_capacity(layout.total_lanes());
        for &chunk in parts.iter().take(layout.full_chunks) {
            for l in 0..lpc {
                out.push(ctx.push(Inst::ExtractLane {
                    shape: layout.shape,
                    lane: l as u8,
                    signed,
                    a: chunk,
                }));
            }
        }
        for t in 0..layout.tail_lanes {
            out.push(parts[layout.full_chunks + t]);
        }
        return Ok(out);
    }
    unsup("vector explode: unsupported source vector shape")
}

/// Explode a legalized **wide** value (its `parts` = chunk `v128`s + tail scalars, per `layout`) into
/// its `total_lanes` per-lane scalars — `ExtractLane` for each chunk lane, then the already-scalar
/// tail lanes. Shape-generic (integer or float lanes); the inverse is [`bind_lanes_as_vector`].
fn wide_explode_lanes(ctx: &mut BlockCtx, parts: &[ValIdx], layout: WideLayout) -> Vec<ValIdx> {
    let lpc = layout.shape.lanes() as usize;
    let mut out = Vec::with_capacity(layout.total_lanes());
    for &chunk in parts.iter().take(layout.full_chunks) {
        for l in 0..lpc {
            out.push(ctx.push(Inst::ExtractLane {
                shape: layout.shape,
                lane: l as u8,
                signed: false,
                a: chunk,
            }));
        }
    }
    for t in 0..layout.tail_lanes {
        out.push(parts[layout.full_chunks + t]);
    }
    out
}

/// Bind `dest` to a vector built from `lanes` (of lane type `shape`), choosing the representation by
/// the lane count: exactly one `v128`'s worth → a single `v128` ([`finish`]); otherwise a legalized
/// wide value (`full_chunks` `v128`s + a scalar tail, [`BlockCtx::bind_wide`]). Used to land a wide
/// shuffle's gathered result lanes.
fn bind_lanes_as_vector(ctx: &mut BlockCtx, dest: &Name, lanes: &[ValIdx], shape: svm_ir::VShape) {
    let lpc = shape.lanes() as usize;
    if lanes.len() == lpc {
        let v = build_v128_from_lanes(ctx, shape, lanes);
        let _ = finish(ctx, dest, v);
        return;
    }
    let full_chunks = lanes.len() / lpc;
    let mut parts = Vec::with_capacity(full_chunks + lanes.len() % lpc);
    for ci in 0..full_chunks {
        parts.push(build_v128_from_lanes(
            ctx,
            shape,
            &lanes[ci * lpc..ci * lpc + lpc],
        ));
    }
    for &l in &lanes[full_chunks * lpc..] {
        parts.push(l);
    }
    ctx.bind_wide(dest, parts);
}

/// Bind `dest` to a shuffle's gathered `lanes` (of element shape `elem`), choosing the result
/// representation by lane count: a single `v128` (one `v128`'s worth of lanes), a packed-`i64` vec2
/// (2 lanes of a 32-bit element), or a legalized wide value otherwise.
fn bind_shuffle_result(
    ctx: &mut BlockCtx,
    dest: &Name,
    lanes: &[ValIdx],
    elem: svm_ir::VShape,
) -> Result<(), Error> {
    let m = lanes.len();
    if m == elem.lanes() as usize {
        let v = build_v128_from_lanes(ctx, elem, lanes);
        return finish(ctx, dest, v);
    }
    if m == 2 && elem.lane_bytes() == 4 {
        let packed = ctx.vec_pack(lanes[0], lanes[1], elem.lane_val());
        return finish(ctx, dest, packed);
    }
    bind_lanes_as_vector(ctx, dest, lanes, elem);
    Ok(())
}

/// Build a single `v128` of `shape` from its `lane scalars` (`Splat` lane 0, then `ReplaceLane` the
/// rest). `lanes.len()` must equal `shape.lanes()`.
fn build_v128_from_lanes(ctx: &mut BlockCtx, shape: svm_ir::VShape, lanes: &[ValIdx]) -> ValIdx {
    let mut v = ctx.push(Inst::Splat { shape, a: lanes[0] });
    for (i, &l) in lanes.iter().enumerate().skip(1) {
        v = ctx.push(Inst::ReplaceLane {
            shape,
            lane: i as u8,
            a: v,
            b: l,
        });
    }
    v
}

/// Repack `N` lane scalars into the destination vector `to_type`, binding `dest` in whichever
/// representation that type uses (a `<2 x {i32,float}>` packed `i64`, a single `v128`, or a
/// legalized wide value). The inverse of [`vec_explode`].
fn vec_implode(
    ctx: &mut BlockCtx,
    dest: &Name,
    lanes: &[ValIdx],
    to_type: &Type,
    _types: &Types,
) -> Result<(), Error> {
    if let Some(lane) = vec2_lane_ty(to_type) {
        // Each lane's 32-bit image: an `i32` directly, or a `<2 x float>` lane reinterpreted from `f32`.
        let as_i32 = |ctx: &mut BlockCtx, l: ValIdx| {
            if lane == ValType::F32 {
                ctx.push(Inst::Cast {
                    op: CastOp::ReinterpF32I32,
                    a: l,
                })
            } else {
                l
            }
        };
        let l0 = as_i32(ctx, lanes[0]);
        let l1 = as_i32(ctx, lanes[1]);
        // Pack lane 0 into the low 32 bits, lane 1 into the high 32 bits of an `i64`.
        let lo = ctx.push(Inst::Convert {
            op: ConvOp::ExtendI32U,
            a: l0,
        });
        let hi = ctx.push(Inst::Convert {
            op: ConvOp::ExtendI32U,
            a: l1,
        });
        let sh = ctx.const_i64(32);
        let hishift = ctx.push(Inst::IntBin {
            ty: IntTy::I64,
            op: BinOp::Shl,
            a: hi,
            b: sh,
        });
        let packed = ctx.push(Inst::IntBin {
            ty: IntTy::I64,
            op: BinOp::Or,
            a: lo,
            b: hishift,
        });
        return finish(ctx, dest, packed);
    }
    if let Some(shape) = vec128_shape(to_type) {
        let v = build_v128_from_lanes(ctx, shape, lanes);
        return finish(ctx, dest, v);
    }
    if let Some(layout) = wide_vec_layout(to_type) {
        let lpc = layout.shape.lanes() as usize;
        let mut parts = Vec::with_capacity(layout.nparts());
        for ci in 0..layout.full_chunks {
            parts.push(build_v128_from_lanes(
                ctx,
                layout.shape,
                &lanes[ci * lpc..ci * lpc + lpc],
            ));
        }
        let tail_start = layout.full_chunks * lpc;
        for t in 0..layout.tail_lanes {
            parts.push(lanes[tail_start + t]);
        }
        ctx.bind_wide(dest, parts);
        return Ok(());
    }
    unsup("vector convert: unsupported destination vector shape")
}

/// Lower a lane-wise **integer** vector conversion (`zext`/`sext`/`trunc`, `<N x iA> → <N x iB>`) —
/// the auto-vectorizer's widen/narrow. svm-ir has no vector-convert op, so we scalarize: explode the
/// source to `N` lane scalars, convert each in its `i32`/`i64` container via the same `emit_ext`/
/// `emit_trunc` used for scalars, then repack into the destination representation. The result re-
/// verifies under `svm-verify` exactly as the hand-written scalar form would.
fn lower_vec_int_convert(
    ctx: &mut BlockCtx,
    dest: &Name,
    operand: &Operand,
    to_type: &Type,
    kind: VConv,
    types: &Types,
) -> Result<(), Error> {
    let from_ty = operand.get_type(types);
    // A `<N x i1>` **mask** source is held lane-wise (`mask_lanes`: `0`/`1` scalars), not a packed
    // vector, so it explodes through `mask_operand` rather than `vec_explode`. `zext` widens each lane
    // `0`/`1`; `sext` widens to `0`/`-1` (all-ones). Found via Embench `picojpeg`'s
    // `sext <8 x i1> to <8 x i8>` (a `select` mask materialized to a byte vector).
    if let Some(n) = i1_vector_lanes(from_ty.as_ref()) {
        let to_bits = vec_int_lane_bits(to_type).ok_or_else(|| {
            Error::Unsupported("vector mask conversion to non-integer lanes".into())
        })?;
        let lanes_in = mask_operand(ctx, operand, n)?;
        let mut out = Vec::with_capacity(n);
        for v in lanes_in {
            out.push(match kind {
                VConv::ZExt => emit_ext(ctx, v, 1, to_bits, false),
                VConv::SExt => emit_ext(ctx, v, 1, to_bits, true),
                VConv::Trunc => return unsup("vector trunc from an i1 mask"),
            });
        }
        return vec_implode(ctx, dest, &out, to_type, types);
    }
    let from_bits = vec_int_lane_bits(from_ty.as_ref())
        .ok_or_else(|| Error::Unsupported("vector conversion of non-integer lanes".into()))?;
    let to_bits = vec_int_lane_bits(to_type)
        .ok_or_else(|| Error::Unsupported("vector conversion to non-integer lanes".into()))?;
    let lanes_in = vec_explode(ctx, operand, types, false)?;
    let mut out = Vec::with_capacity(lanes_in.len());
    for v in lanes_in {
        out.push(match kind {
            VConv::ZExt => emit_ext(ctx, v, from_bits, to_bits, false),
            VConv::SExt => emit_ext(ctx, v, from_bits, to_bits, true),
            VConv::Trunc => emit_trunc(ctx, v, from_bits, to_bits),
        });
    }
    vec_implode(ctx, dest, &out, to_type, types)
}

/// Which int↔float / float↔float conversion a vector `fptosi`/`fptoui`/`sitofp`/`uitofp`/`fpext`/
/// `fptrunc` applies per lane.
#[derive(Clone, Copy)]
enum FpConv {
    FToSI,
    FToUI,
    SIToF,
    UIToF,
    FpExt,
    FpTrunc,
}

/// Lower a lane-wise **int↔float / float↔float** vector conversion (the float analog of
/// [`lower_vec_int_convert`]) — scalarize: explode the source lanes, apply the scalar `FToISat`/
/// `IToFConv`/`Cast` per lane, repack. Lands the auto-vectorizer's `<2 x float>`↔`<2 x i32>`
/// (perlin's gradient math) and the 128-bit float-vector convdersions. The lane types come from the
/// source / destination vector shapes.
fn lower_vec_fp_convert(
    ctx: &mut BlockCtx,
    dest: &Name,
    operand: &Operand,
    to_type: &Type,
    kind: FpConv,
    types: &Types,
) -> Result<(), Error> {
    let from_ty = operand.get_type(types);
    let from_shape = vec_lane_shape(from_ty.as_ref())
        .ok_or_else(|| Error::Unsupported("vector fp-convert: bad source shape".into()))?;
    let to_shape = vec_lane_shape(to_type)
        .ok_or_else(|| Error::Unsupported("vector fp-convert: bad dest shape".into()))?;
    let lanes = vec_explode(ctx, operand, types, false)?;
    let mut out = Vec::with_capacity(lanes.len());
    for v in lanes {
        let inst = match kind {
            FpConv::FToSI => Inst::FToISat {
                op: ftoi_op(
                    float_ty(from_shape.lane_val())?,
                    int_ty(to_shape.lane_val())?,
                    true,
                ),
                a: v,
            },
            FpConv::FToUI => Inst::FToISat {
                op: ftoi_op(
                    float_ty(from_shape.lane_val())?,
                    int_ty(to_shape.lane_val())?,
                    false,
                ),
                a: v,
            },
            FpConv::SIToF => Inst::IToFConv {
                op: itof_op(
                    int_ty(from_shape.lane_val())?,
                    float_ty(to_shape.lane_val())?,
                    true,
                ),
                a: v,
            },
            FpConv::UIToF => Inst::IToFConv {
                op: itof_op(
                    int_ty(from_shape.lane_val())?,
                    float_ty(to_shape.lane_val())?,
                    false,
                ),
                a: v,
            },
            FpConv::FpExt => Inst::Cast {
                op: CastOp::Promote,
                a: v,
            },
            FpConv::FpTrunc => Inst::Cast {
                op: CastOp::Demote,
                a: v,
            },
        };
        out.push(ctx.push(inst));
    }
    vec_implode(ctx, dest, &out, to_type, types)
}

fn translate_term(
    ctx: &mut BlockCtx,
    term: &LTerm,
    bi: usize,
    f: &Function,
    s: &Scan,
    block_params: &[Vec<ValueId>],
    aux_blocks: &mut Vec<Block>,
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
        LTerm::Switch(sw) => translate_switch(ctx, sw, bi, f, s, block_params, aux_blocks),
        LTerm::IndirectBr(ib) => translate_indirectbr(ctx, ib, bi, f, s, block_params),
        LTerm::Unreachable(_) => Ok(Terminator::Unreachable),
        LTerm::Invoke(inv) => lower_invoke(ctx, inv, bi, f, s, block_params, aux_blocks),
        // `resume` re-throws the current (slot-held) exception into the next outer handler — the same
        // unwind as a rethrow. `eh_unwind` ends in `LongJmp` (noreturn), so the terminator is dead.
        LTerm::Resume(_) => {
            let base = ctx.eh_base()?;
            ctx.eh_unwind(base)?;
            Ok(Terminator::Unreachable)
        }
        other => unsup(format!("terminator {other:?}")),
    }
}

/// Lower an `invoke F(args) to %ok unwind %lpad` (the `try`-body call) onto the setjmp/longjmp core.
///
/// This block installs a handler and tests its `setjmp`; the call runs in a synthetic block reached
/// only on the first (zero) return:
/// ```text
///   B:      d   = load HSP;  r = setjmp(BUFS + d*SLOT)
///           br (r == 0) -> Bcall(T)            else -> %lpad(lpad-args)
///   Bcall:  store HSP = d+1;  result = call F(sp+frame, args…);  store HSP = d
///           br -> %ok(ok-args, result substituted in)
/// ```
/// A `__cxa_throw` deeper in `F` long-jumps back to this `setjmp`, which then returns nonzero and
/// routes to `%lpad`. `HSP` is the handler-stack depth: pushed around the call, popped on normal
/// return, and left at `d` by the throw's unwind — so `%lpad` always sees the outer depth.
///
/// `T` (the synthetic block's threaded params) is the data-SP, the call-argument values, and every
/// value `%ok` needs *except* the call result (defined inside `Bcall`). Only a direct call to a
/// defined function with a scalar/void result is supported (a struct return / indirect invoke is a
/// clean `Unsupported`).
fn lower_invoke(
    ctx: &mut BlockCtx,
    inv: &llvm_ir::terminator::Invoke,
    bi: usize,
    f: &Function,
    s: &Scan,
    block_params: &[Vec<ValueId>],
    aux_blocks: &mut Vec<Block>,
) -> Result<Terminator, Error> {
    use std::collections::hash_map::Entry;
    let base = ctx.eh_base()?;

    // Resolve the callee — a direct call to a defined function only.
    let op = inv
        .function
        .as_ref()
        .right()
        .ok_or_else(|| Error::Unsupported("invoke of inline asm".into()))?;
    let cname =
        global_name_of(op).ok_or_else(|| Error::Unsupported("invoke of indirect callee".into()))?;

    // A noreturn EH unwinder (`__cxa_rethrow`/`_Unwind_Resume`/`__cxa_throw`) appears as an `invoke`
    // when it sits in a cleanup scope (the unwind edge runs `__cxa_end_catch`). It is not a defined
    // function we can `setjmp` around — it long-jumps straight into the enclosing handler — so emit
    // the unwind inline and terminate `unreachable`: both the normal and unwind successors are dead
    // (the skipped `__cxa_end_catch` cleanup is a no-op in the supported scope).
    if lower_eh_unwinder(ctx, &cname, &inv.arguments)? {
        return Ok(Terminator::Unreachable);
    }

    let func = *ctx
        .name2idx
        .get(&cname)
        .ok_or_else(|| Error::Unsupported(format!("invoke of external/undefined `{cname}`")))?;

    // The callee's parameter + result types (for typing the threaded args and the result).
    let (param_tys, result_ty) = match inv.function_ty.as_ref() {
        Type::FuncType {
            result_type,
            param_types,
            ..
        } => (param_types.clone(), result_type.clone()),
        other => return unsup(format!("invoke through non-function type {other}")),
    };
    let n_results = match result_ty.as_ref() {
        Type::VoidType => 0usize,
        t if struct_field_vtypes(t, ctx.types).is_some() => {
            return unsup("invoke returning a by-value struct");
        }
        _ => 1,
    };

    // Values computed in *this* block (B-local): the data-SP and each call argument.
    let sp = ctx.sp()?;
    let mut arg_vals: Vec<ValIdx> = Vec::with_capacity(inv.arguments.len());
    for (a, _) in &inv.arguments {
        arg_vals.push(ctx.operand(a)?);
    }

    let ok_blk = s.block_idx[&inv.return_label];
    let lpad_blk = s.block_idx[&inv.exception_label];

    // The unwind edge is taken directly from B — resolve its args in B's context.
    let lpad_args = branch_args(ctx, bi, lpad_blk, f, s, block_params)?;

    // The normal edge feeds `%ok`, whose args may include the invoke result — directly (a live-in) or
    // as a φ incoming (when `return_label` is itself a φ block). The result is defined only in `Bcall`,
    // so resolve `%ok`'s args in B's context with the result temporarily bound to a fresh dummy const;
    // every `ok_args` slot that comes back equal to that dummy's id is a result slot (handles a result
    // used zero, one, or several times), filled with the real result inside `Bcall` and kept out of `T`.
    let result_vid = ctx.s.name2id.get(&inv.result).copied();
    let saved = if n_results == 1 {
        result_vid.map(|rv| {
            let dummy = ctx.const_i64(0);
            (rv, dummy, ctx.idx_of.insert(rv, dummy))
        })
    } else {
        None
    };
    let ok_args = branch_args(ctx, bi, ok_blk, f, s, block_params)?;
    if let Some((rv, _, prev)) = saved {
        match prev {
            Some(old) => {
                ctx.idx_of.insert(rv, old);
            }
            None => {
                ctx.idx_of.remove(&rv);
            }
        }
    }
    let dummy_idx = saved.map(|(_, d, _)| d);
    let is_result = |a: ValIdx| Some(a) == dummy_idx;

    // Build the threaded set `T` (deduped, ordered): SP, then each call arg, then every `%ok` arg
    // except the result slots. A value's index in `T` is its parameter index in `Bcall`.
    let mut t_vals: Vec<ValIdx> = Vec::new();
    let mut pos: HashMap<ValIdx, usize> = HashMap::new();
    let add = |t: &mut Vec<ValIdx>, p: &mut HashMap<ValIdx, usize>, v: ValIdx| {
        if let Entry::Vacant(e) = p.entry(v) {
            e.insert(t.len());
            t.push(v);
        }
    };
    add(&mut t_vals, &mut pos, sp);
    for &a in &arg_vals {
        add(&mut t_vals, &mut pos, a);
    }
    for &a in &ok_args {
        if !is_result(a) {
            add(&mut t_vals, &mut pos, a);
        }
    }

    // Type each `T` member: SP is `i64`, a call arg by the callee's parameter type, an `%ok` arg by
    // the block parameter it feeds (`branch_args` / `block_param_types` agree in order).
    let mut t_types: Vec<ValType> = vec![ValType::I64; t_vals.len()];
    t_types[pos[&sp]] = ValType::I64;
    for (k, &a) in arg_vals.iter().enumerate() {
        t_types[pos[&a]] = val_type(param_tys[k].as_ref())?;
    }
    let ok_param_types = block_param_types(&block_params[ok_blk], s);
    for (i, &a) in ok_args.iter().enumerate() {
        if !is_result(a) {
            t_types[pos[&a]] = ok_param_types[i];
        }
    }

    // ── Build the synthetic call block `Bcall` (block-local indices: params `0..P`, then pushed
    // result-producing insts; a `store` produces no value and does not advance the index). ──
    let p = t_vals.len() as ValIdx;
    let sp_param = pos[&sp] as ValIdx;
    let mut insts: Vec<Inst> = Vec::new();
    // callee SP = sp + frame_size.
    insts.push(Inst::ConstI64(ctx.frame_size as i64)); // p
    insts.push(Inst::IntBin {
        ty: IntTy::I64,
        op: BinOp::Add,
        a: sp_param,
        b: p,
    }); // p+1  (callee_sp)
    let callee_sp = p + 1;
    // push HSP: d = load HSP; store HSP = d + 1.
    insts.push(Inst::ConstI64((base + EH_HSP_O) as i64)); // p+2 (hsp_addr)
    let hsp_addr = p + 2;
    insts.push(Inst::Load {
        op: LoadOp::I64,
        addr: hsp_addr,
        offset: 0,
        align: 8,
    }); // p+3 (d)
    let d = p + 3;
    insts.push(Inst::ConstI64(1)); // p+4 (one)
    let one = p + 4;
    insts.push(Inst::IntBin {
        ty: IntTy::I64,
        op: BinOp::Add,
        a: d,
        b: one,
    }); // p+5 (d+1)
    let dp = p + 5;
    insts.push(Inst::Store {
        op: StoreOp::I64,
        addr: hsp_addr,
        value: dp,
        offset: 0,
        align: 8,
    }); // no result
        // the call: args = [callee_sp, arg params…].
    let mut cargs: Vec<ValIdx> = vec![callee_sp];
    for &a in &arg_vals {
        cargs.push(pos[&a] as ValIdx);
    }
    insts.push(Inst::Call { func, args: cargs }); // p+6 (result, if any)
    let result = p + 6;
    // pop HSP back to `d` (normal return).
    insts.push(Inst::Store {
        op: StoreOp::I64,
        addr: hsp_addr,
        value: d,
        offset: 0,
        align: 8,
    }); // no result
        // Br -> %ok, remapping each arg: a result slot to the local call `result`, every other to its
        // `Bcall` parameter index.
    let ok_remapped: Vec<ValIdx> = ok_args
        .iter()
        .map(|&a| {
            if is_result(a) {
                result
            } else {
                pos[&a] as ValIdx
            }
        })
        .collect();
    let bcall_idx = (f.basic_blocks.len() + aux_blocks.len()) as u32;
    aux_blocks.push(Block {
        params: t_types,
        insts,
        term: Terminator::Br {
            target: ok_blk as u32,
            args: ok_remapped,
        },
    });

    // ── This block's terminator: install the handler and branch on the `setjmp` return. ──
    let d_addr = ctx.const_i64((base + EH_HSP_O) as i64);
    let depth = ctx.push(Inst::Load {
        op: LoadOp::I64,
        addr: d_addr,
        offset: 0,
        align: 8,
    });
    let bufs = ctx.const_i64((base + EH_BUFS_O) as i64);
    let slot = ctx.const_i64(EH_SLOT as i64);
    let off = ctx.mul_i64(depth, slot);
    let buf = ctx.add_i64(bufs, off);
    let r = ctx.push(Inst::SetJmp { buf });
    let zero = ctx.push(Inst::ConstI32(0));
    let cond = ctx.push(Inst::IntCmp {
        ty: IntTy::I32,
        op: CmpOp::Eq,
        a: r,
        b: zero,
    });
    Ok(Terminator::BrIf {
        cond,
        then_blk: bcall_idx,
        then_args: t_vals,
        else_blk: lpad_blk as u32,
        else_args: lpad_args,
    })
}

/// Lower an `indirectbr` (computed `goto`, `goto *p`) to a `br_table` (the §computed-goto half). The
/// address operand is a `blockaddress` value the guest loaded from its dispatch table — i.e. a **block
/// index** (see [`blockaddr`]; the matching label was baked into the global by `const_bytes`). So the
/// table is indexed directly by that block index over `[0, nblocks)`: each listed destination routes
/// to its own block; out-of-list slots (and any out-of-range / UB address) fall to the default — the
/// first listed destination. LLVM guarantees the address is one of `possible_dests`, so the default is
/// unreachable on well-defined input; on UB it stays in-sandbox (a defined branch to a real block,
/// §3b totality — no escape, no stuck state).
fn translate_indirectbr(
    ctx: &mut BlockCtx,
    ib: &llvm_ir::terminator::IndirectBr,
    bi: usize,
    f: &Function,
    s: &Scan,
    block_params: &[Vec<ValueId>],
) -> Result<Terminator, Error> {
    // The address is a pointer (the loaded `blockaddress`); narrow it to the `i32` `br_table` index.
    let operand = ctx.operand(&ib.operand)?;
    let idx = ctx.push(Inst::Convert {
        op: ConvOp::WrapI64,
        a: operand,
    });

    let mut dests: Vec<usize> = Vec::with_capacity(ib.possible_dests.len());
    for n in &ib.possible_dests {
        let blk = *s
            .block_idx
            .get(n)
            .ok_or_else(|| Error::Unsupported(format!("indirectbr to unknown block {n:?}")))?;
        dests.push(blk);
    }
    if dests.is_empty() {
        return unsup("indirectbr with no destinations");
    }

    // Branch arguments per distinct destination (each target's φ-results + threaded live-ins).
    let mut args_for: HashMap<usize, Vec<ValIdx>> = HashMap::new();
    for &d in &dests {
        if let std::collections::hash_map::Entry::Vacant(e) = args_for.entry(d) {
            e.insert(branch_args(ctx, bi, d, f, s, block_params)?);
        }
    }

    // The table is indexed by block index, so it spans every block; unlisted indices → the default.
    let nblocks = f.basic_blocks.len();
    let default_blk = dests[0];
    let default_edge: svm_ir::Edge = (default_blk as u32, args_for[&default_blk].clone());
    let mut targets: Vec<svm_ir::Edge> = vec![default_edge.clone(); nblocks];
    for &d in &dests {
        targets[d] = (d as u32, args_for[&d].clone());
    }
    Ok(Terminator::BrTable {
        idx,
        targets,
        default: default_edge,
    })
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
    aux_blocks: &mut Vec<Block>,
) -> Result<Terminator, Error> {
    // `br_table`'s index is `i32`; an `i64` operand (e.g. a Rust enum discriminant) is handled by
    // folding its high 32 bits into the index below (an out-of-`[0,2^32)` value forces the default).
    let width = operand_bits(&sw.operand)?;
    if width > 64 {
        return unsup(format!("switch on i{width} (i128+ unsupported)"));
    }
    // Collect the (value, dest-block) cases (full `i64` for an `i64` switch; sign-fit for `i32`).
    let mut cases: Vec<(i64, usize)> = Vec::with_capacity(sw.dests.len());
    for (v, dest) in &sw.dests {
        let val = match v.as_ref() {
            Constant::Int { value, .. } if width <= 32 => *value as i32 as i64,
            Constant::Int { value, .. } => *value as i64,
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
    // Compute the span in `i128` so a switch whose `i64` cases straddle more than the `i64` range
    // (a sparse `match` on a wide value) reports `Unsupported` instead of overflow-panicking. Once
    // bounded by `MAX_SWITCH_SPAN` it fits `i64`/`usize` for the table below.
    let span_wide = max as i128 - min as i128 + 1;
    if span_wide > MAX_SWITCH_SPAN as i128 {
        // Too sparse for a dense `br_table` (e.g. a niche-optimized enum discriminant with
        // `i64::MIN`-ish sentinels): lower to an equality compare chain of synthetic blocks instead.
        return lower_sparse_switch(
            ctx,
            sw,
            bi,
            f,
            s,
            block_params,
            aux_blocks,
            &cases,
            default_blk,
        );
    }
    let span = span_wide as i64;

    // Index = operand - min (so the table starts at 0). An out-of-range / unbiased value lands on
    // the default (a too-large biased value, ≥ len ⇒ default).
    let operand = ctx.operand(&sw.operand)?;
    let idx = if width <= 32 {
        if min == 0 {
            operand
        } else {
            let m = ctx.push(Inst::ConstI32(min as i32));
            ctx.push(Inst::IntBin {
                ty: IntTy::I32,
                op: BinOp::Sub,
                a: operand,
                b: m,
            })
        }
    } else {
        // `i64` operand: bias by `min` (`i64`), then fold the high 32 bits in — if `diff` doesn't fit
        // in `[0, 2^32)` (high bits set, incl. a negative `diff`) force the index to `0xFFFFFFFF` so it
        // exceeds the table and hits the default. Sound for any `i64` (the low-32 `br_table` alone
        // would alias far-apart values onto a case).
        let diff = if min == 0 {
            operand
        } else {
            let m = ctx.push(Inst::ConstI64(min));
            ctx.push(Inst::IntBin {
                ty: IntTy::I64,
                op: BinOp::Sub,
                a: operand,
                b: m,
            })
        };
        let c32 = ctx.push(Inst::ConstI64(32));
        let hi = ctx.push(Inst::IntBin {
            ty: IntTy::I64,
            op: BinOp::ShrU,
            a: diff,
            b: c32,
        });
        let hi32 = ctx.push(Inst::Convert {
            op: ConvOp::WrapI64,
            a: hi,
        });
        let zero = ctx.push(Inst::ConstI32(0));
        let oor = ctx.push(Inst::IntCmp {
            ty: IntTy::I32,
            op: CmpOp::Ne,
            a: hi32,
            b: zero,
        });
        // mask = 0 - oor → 0 (in range) or 0xFFFFFFFF (out of range).
        let mask = ctx.push(Inst::IntBin {
            ty: IntTy::I32,
            op: BinOp::Sub,
            a: zero,
            b: oor,
        });
        let lo = ctx.push(Inst::Convert {
            op: ConvOp::WrapI64,
            a: diff,
        });
        ctx.push(Inst::IntBin {
            ty: IntTy::I32,
            op: BinOp::Or,
            a: lo,
            b: mask,
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

/// The svm-IR types of a block's parameters, in the fan-out order `branch_args` supplies them: the
/// data-SP and scalars one slot each (`i64` for the SP), a wide-vector value its `C` `v128` chunks then
/// `T` lane scalars. Mirrors `translate_block`'s param materialization — used to type the synthetic
/// blocks of a sparse-switch compare chain so their params line up with the edge args.
fn block_param_types(param_ids: &[ValueId], s: &Scan) -> Vec<ValType> {
    let mut types = Vec::new();
    for &vid in param_ids {
        if let Some(&layout) = s.wide.get(&vid) {
            for _ in 0..layout.full_chunks {
                types.push(ValType::V128);
            }
            let lane = layout.shape.lane_val();
            for _ in 0..layout.tail_lanes {
                types.push(lane);
            }
        } else {
            types.push(if vid == SP { ValType::I64 } else { s.ty[vid] });
        }
    }
    types
}

/// Lower a too-sparse `switch` to an **equality compare chain** (§3b). The switch block tests the
/// operand against the first case (`br_if x == v0 → t0, else → c1`); each synthetic chain block tests
/// the next case; the last falls through to the default. svm-IR has no first-class sparse jump, and
/// Rust's niche-optimized enums (discriminants at `i64::MIN`-ish sentinels) produce exactly these
/// astronomically-sparse switches. The chain blocks are appended to `aux_blocks` — *after* all real
/// blocks, so existing block indices are unchanged — and thread, as block parameters, everything a
/// downstream edge consumes: the data-SP (every block's param 0, §3d), the compared operand, and every
/// case/default target's branch args. Those args are computed **once here**, in the switch block's
/// context, because φ-operand / live-in resolution needs the real predecessor (`bi`), not the synthetic
/// blocks; they are then threaded to wherever the chain consumes them.
#[allow(clippy::too_many_arguments)]
fn lower_sparse_switch(
    ctx: &mut BlockCtx,
    sw: &llvm_ir::terminator::Switch,
    bi: usize,
    f: &Function,
    s: &Scan,
    block_params: &[Vec<ValueId>],
    aux_blocks: &mut Vec<Block>,
    cases: &[(i64, usize)],
    default_blk: usize,
) -> Result<Terminator, Error> {
    let width = operand_bits(&sw.operand)?;
    let cmp_ty = if width <= 32 { IntTy::I32 } else { IntTy::I64 };
    let operand_ty = if width <= 32 {
        ValType::I32
    } else {
        ValType::I64
    };
    let mk_const = |v: i64| {
        if width <= 32 {
            Inst::ConstI32(v as i32)
        } else {
            Inst::ConstI64(v)
        }
    };
    let operand = ctx.operand(&sw.operand)?;

    // Compute every target's branch args in *this* block's context (φ/live-in resolution needs the
    // real predecessor `bi`). The returned ids are this-block-local.
    let default_args = branch_args(ctx, bi, default_blk, f, s, block_params)?;
    let mut case_args: Vec<Vec<ValIdx>> = Vec::with_capacity(cases.len());
    for &(_, blk) in cases {
        case_args.push(branch_args(ctx, bi, blk, f, s, block_params)?);
    }

    // The threaded value set `T`, in order: the data-SP (param 0 of every block), the operand, then
    // every distinct value any edge consumes. A chain block's params are exactly `T` (this order), so a
    // value's chain-block param index is its position in `T`.
    let sp = ctx.sp()?;
    let mut t_vals: Vec<ValIdx> = Vec::new();
    let mut pos: HashMap<ValIdx, usize> = HashMap::new();
    for &v in std::iter::once(&sp)
        .chain(std::iter::once(&operand))
        .chain(default_args.iter())
        .chain(case_args.iter().flatten())
    {
        if let std::collections::hash_map::Entry::Vacant(e) = pos.entry(v) {
            e.insert(t_vals.len());
            t_vals.push(v);
        }
    }

    // Type each threaded value by position: SP is `i64`, the operand its compare type, and every other
    // value by the target parameter it feeds (`branch_args` and `block_param_types` agree in order).
    let mut t_types: Vec<ValType> = vec![ValType::I32; t_vals.len()];
    t_types[pos[&sp]] = ValType::I64;
    t_types[pos[&operand]] = operand_ty;
    for (args, target) in std::iter::once((&default_args, default_blk))
        .chain(case_args.iter().zip(cases.iter().map(|&(_, b)| b)))
    {
        for (a, ty) in args.iter().zip(block_param_types(&block_params[target], s)) {
            t_types[pos[a]] = ty;
        }
    }

    // Identity edge: pass all of `T` through unchanged. Remap a this-block arg list to chain-block
    // param indices (every arg is in `T`).
    let pass_through: Vec<ValIdx> = (0..t_vals.len() as ValIdx).collect();
    let remap =
        |args: &[ValIdx]| -> Vec<ValIdx> { args.iter().map(|a| pos[a] as ValIdx).collect() };

    // Synthetic blocks land after all real blocks and any chain from an earlier switch in this fn.
    let base = (f.basic_blocks.len() + aux_blocks.len()) as u32;
    let n = cases.len();
    // Chain block for case `k` (1..n) has index `base + (k - 1)`; case 0 is the switch block itself.
    let chain_blk = |k: usize| base + (k as u32 - 1);
    let x_param = pos[&operand] as ValIdx;

    for k in 1..n {
        let (v, target) = cases[k];
        // Block-local ids: params `0..t_vals.len()`, then the two pushed insts.
        let cst = t_vals.len() as ValIdx;
        let cond = cst + 1;
        let insts = vec![
            mk_const(v),
            Inst::IntCmp {
                ty: cmp_ty,
                op: CmpOp::Eq,
                a: x_param,
                b: cst,
            },
        ];
        // Match → target; else → next chain block (pass `T` on) or, for the last, the default.
        let (else_blk, else_args) = if k + 1 < n {
            (chain_blk(k + 1), pass_through.clone())
        } else {
            (default_blk as u32, remap(&default_args))
        };
        aux_blocks.push(Block {
            params: t_types.clone(),
            insts,
            term: Terminator::BrIf {
                cond,
                then_blk: target as u32,
                then_args: remap(&case_args[k]),
                else_blk,
                else_args,
            },
        });
    }

    // The switch block's own terminator: test case 0, else enter the chain (threading `T`). A single
    // case can't exceed `MAX_SWITCH_SPAN`, so `n > 1` here, but handle `n == 1` for totality.
    let (v0, target0) = cases[0];
    let cst = ctx.push(mk_const(v0));
    let cond = ctx.push(Inst::IntCmp {
        ty: cmp_ty,
        op: CmpOp::Eq,
        a: operand,
        b: cst,
    });
    let (else_blk, else_args) = if n > 1 {
        (chain_blk(1), t_vals.clone())
    } else {
        (default_blk as u32, default_args.clone())
    };
    Ok(Terminator::BrIf {
        cond,
        then_blk: target0 as u32,
        then_args: case_args[0].clone(),
        else_blk,
        else_args,
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
    // Map each φ-result id in `target` to its incoming operand from predecessor `from`, plus the φ's
    // position (`phi_ord` within the block, `incoming_idx` within the φ) — the key into the recovered
    // operand-position `blockaddress` map (an incoming may be a φ-threaded `blockaddress`).
    let from_name = &s.block_name[from];
    let target_bb = &f.basic_blocks[target];
    let mut phi_incoming: HashMap<ValueId, (&Operand, u32, u32)> = HashMap::new();
    let mut phi_ord = 0u32;
    for instr in &target_bb.instrs {
        if let Instruction::Phi(p) = instr {
            if let Some(&vid) = s.name2id.get(&p.dest) {
                let inc_idx = p
                    .incoming_values
                    .iter()
                    .position(|(_, pred)| pred == from_name)
                    .ok_or_else(|| {
                        Error::Unsupported(format!(
                            "φ {:?} has no incoming for predecessor {from_name:?}",
                            p.dest
                        ))
                    })?;
                phi_incoming.insert(
                    vid,
                    (&p.incoming_values[inc_idx].0, phi_ord, inc_idx as u32),
                );
            }
            phi_ord += 1;
        }
    }
    let mut args = Vec::with_capacity(block_params[target].len());
    for &pv in &block_params[target] {
        // A wide-vector param takes `K` args — its incoming value's chunk/tail parts, in the same
        // order the target block's param fan-out expects (I2 step 1 cross-block).
        if let Some(&layout) = s.wide.get(&pv) {
            let parts = if let Some(&(op, _, _)) = phi_incoming.get(&pv) {
                ctx.wide_operand(op, layout)?
            } else {
                // A threaded wide live-in: its parts are live-out of `from`, available here.
                ctx.wide_vals.get(&pv).cloned().ok_or_else(|| {
                    Error::Unsupported(format!("wide value {pv} not available across edge"))
                })?
            };
            args.extend(parts);
        } else if let Some(ftys) = s.agg_layout.get(&pv) {
            // A struct param takes one arg per field — the incoming struct's fields, in the same order
            // the target's per-field fan-out expects (the aggregate analog of the wide branch above).
            let parts = if let Some(&(op, _, _)) = phi_incoming.get(&pv) {
                ctx.agg_operand(op, ftys)?
            } else {
                // A threaded struct live-in: its fields are live-out of `from`, available here.
                ctx.agg.get(&pv).cloned().ok_or_else(|| {
                    Error::Unsupported(format!("struct value {pv} not available across edge"))
                })?
            };
            args.extend(parts);
        } else if let Some(&(op, phi_ord, inc_idx)) = phi_incoming.get(&pv) {
            args.push(ctx.phi_operand(op, target as u32, phi_ord, inc_idx)?);
        } else {
            // A threaded live-in: it is live-out of `from`, so available in this block.
            args.push(ctx.id(pv)?);
        }
    }
    Ok(args)
}

#[cfg(test)]
mod bigint_tests {
    //! Direct unit tests for the synthesized big-integer primitives (`synth_big_*`), the foundation
    //! of the bignum `%e`/`%g` formatter. Each runs the helper in isolation via `svm_interp::run_capture`
    //! with the operands placed in the initial window, then inspects the resulting limbs — so a bug in
    //! one primitive is caught here, not only end-to-end.
    use super::*;
    use svm_interp::Value;

    const WIN_LOG2: u8 = 16; // 64 KiB scratch window

    fn run(func: Func, args: &[i64], mem: &[(usize, &[u32])]) -> (Vec<Value>, Vec<u8>) {
        let mut init = vec![0u8; 1 << WIN_LOG2];
        for (off, limbs) in mem {
            for (k, &l) in limbs.iter().enumerate() {
                init[off + 4 * k..off + 4 * k + 4].copy_from_slice(&l.to_le_bytes());
            }
        }
        let m = Module {
            funcs: vec![func],
            memory: Some(svm_ir::Memory {
                size_log2: WIN_LOG2,
            }),
            ..Default::default()
        };
        let mut fuel = 1_000_000_000u64;
        let argv: Vec<Value> = args.iter().map(|&x| Value::I64(x)).collect();
        let (res, win) = svm_interp::run_capture(&m, 0, &argv, &mut fuel, &init);
        (res.expect("bigint helper trapped"), win)
    }

    fn run_funcs(funcs: Vec<Func>, args: &[i64], mem: &[(usize, &[u32])]) -> (Vec<Value>, Vec<u8>) {
        let mut init = vec![0u8; 1 << WIN_LOG2];
        for (off, ls) in mem {
            for (k, &l) in ls.iter().enumerate() {
                init[off + 4 * k..off + 4 * k + 4].copy_from_slice(&l.to_le_bytes());
            }
        }
        let m = Module {
            funcs,
            memory: Some(svm_ir::Memory {
                size_log2: WIN_LOG2,
            }),
            ..Default::default()
        };
        let mut fuel = 5_000_000_000u64;
        let argv: Vec<Value> = args.iter().map(|&x| Value::I64(x)).collect();
        let (res, win) = svm_interp::run_capture(&m, 0, &argv, &mut fuel, &init);
        (res.expect("dtoa helper trapped"), win)
    }

    // The full helper set with `dtoa_digits` at index 0 (its primitive indices match the vec order).
    fn dtoa_funcs() -> Vec<Func> {
        vec![
            synth_dtoa_digits(1, 2, 3, 4, 5, 6),
            synth_big_zero(),
            synth_big_copy(),
            synth_big_cmp(),
            synth_big_sub(),
            synth_big_mul_small(),
            synth_big_shl_bits(),
        ]
    }

    // Reference digits + exponent via Rust's own correctly-rounded formatter (`{:e}`).
    fn expected(v: f64, nsig: usize) -> (Vec<u8>, i64) {
        let s = format!("{:.*e}", nsig - 1, v);
        let (m, e) = s.split_once('e').unwrap();
        let exp: i64 = e.parse().unwrap();
        let digits: Vec<u8> = m
            .bytes()
            .filter(|c| c.is_ascii_digit())
            .map(|c| c - b'0')
            .collect();
        assert_eq!(digits.len(), nsig, "ref `{s}` digit count");
        (digits, exp)
    }

    // The %e/%E helper set with dtoa_sci at index 0.
    fn sci_funcs() -> Vec<Func> {
        vec![
            synth_dtoa_sci(1),
            synth_dtoa_digits(2, 3, 4, 5, 6, 7),
            synth_big_zero(),
            synth_big_copy(),
            synth_big_cmp(),
            synth_big_sub(),
            synth_big_mul_small(),
            synth_big_shl_bits(),
        ]
    }

    // The %g/%G helper set with dtoa_gen at index 0.
    fn gen_funcs() -> Vec<Func> {
        vec![
            synth_dtoa_gen(1),
            synth_dtoa_digits(2, 3, 4, 5, 6, 7),
            synth_big_zero(),
            synth_big_copy(),
            synth_big_cmp(),
            synth_big_sub(),
            synth_big_mul_small(),
            synth_big_shl_bits(),
        ]
    }

    // The expected C `%g` rendering (default flags: strip trailing zeros, choose %e vs %f by exp).
    fn c_g(v: f64, prec: usize, upper: bool) -> String {
        let p = prec.max(1);
        let (d, e) = expected(v.abs(), p);
        let mut s = String::new();
        if v.is_sign_negative() {
            s.push('-');
        }
        let use_exp = e < -4 || e >= p as i64;
        if use_exp {
            let mut frac: Vec<u8> = d[1..p].to_vec();
            while frac.last() == Some(&0) {
                frac.pop();
            }
            s.push((b'0' + d[0]) as char);
            if !frac.is_empty() {
                s.push('.');
                for &x in &frac {
                    s.push((b'0' + x) as char);
                }
            }
            s.push(if upper { 'E' } else { 'e' });
            s.push(if e < 0 { '-' } else { '+' });
            let ae = e.unsigned_abs();
            s.push_str(&if ae >= 100 {
                format!("{ae:03}")
            } else {
                format!("{ae:02}")
            });
        } else {
            if e >= 0 {
                for &dig in &d[..=e as usize] {
                    s.push((b'0' + dig) as char);
                }
            } else {
                s.push('0');
            }
            let l = (p as i64 - 1 - e).max(0) as usize;
            let mut frac: Vec<u8> = (0..l)
                .map(|k| {
                    let idx = e + 1 + k as i64;
                    if idx >= 0 && (idx as usize) < p {
                        d[idx as usize]
                    } else {
                        0
                    }
                })
                .collect();
            while frac.last() == Some(&0) {
                frac.pop();
            }
            if !frac.is_empty() {
                s.push('.');
                for &x in &frac {
                    s.push((b'0' + x) as char);
                }
            }
        }
        s
    }

    // The expected C `%e` rendering (sign from the sign bit, ≥2 exponent digits).
    fn c_e(v: f64, prec: usize, upper: bool) -> String {
        let (d, e) = expected(v.abs(), prec + 1);
        let mut s = String::new();
        if v.is_sign_negative() {
            s.push('-');
        }
        s.push((b'0' + d[0]) as char);
        if prec > 0 {
            s.push('.');
            for &dig in &d[1..=prec] {
                s.push((b'0' + dig) as char);
            }
        }
        s.push(if upper { 'E' } else { 'e' });
        s.push(if e < 0 { '-' } else { '+' });
        let ae = e.unsigned_abs();
        s.push_str(&if ae >= 100 {
            format!("{ae:03}")
        } else {
            format!("{ae:02}")
        });
        s
    }

    fn limbs(win: &[u8], off: usize, n: usize) -> Vec<u32> {
        (0..n)
            .map(|k| u32::from_le_bytes(win[off + 4 * k..off + 4 * k + 4].try_into().unwrap()))
            .collect()
    }

    fn ret_i64(res: &[Value]) -> i64 {
        match res.first() {
            Some(Value::I64(v)) => *v,
            other => panic!("expected i64 result, got {other:?}"),
        }
    }

    #[test]
    fn big_is_zero_works() {
        let (r, _) = run(synth_big_is_zero(), &[256], &[]);
        assert_eq!(ret_i64(&r), 1, "all-zero ⇒ 1");
        let (r, _) = run(synth_big_is_zero(), &[256], &[(256, &[0, 0, 0, 0, 0, 7])]);
        assert_eq!(ret_i64(&r), 0, "limb 5 set ⇒ 0");
        let (r, _) = run(synth_big_is_zero(), &[256], &[(256, &[1])]);
        assert_eq!(ret_i64(&r), 0, "limb 0 set ⇒ 0");
    }

    #[test]
    fn big_cmp_works() {
        let a = 256usize;
        let b = 512usize;
        // low-limb difference
        let (r, _) = run(
            synth_big_cmp(),
            &[a as i64, b as i64],
            &[(a, &[1]), (b, &[2])],
        );
        assert_eq!(ret_i64(&r), -1);
        let (r, _) = run(
            synth_big_cmp(),
            &[a as i64, b as i64],
            &[(a, &[2]), (b, &[1])],
        );
        assert_eq!(ret_i64(&r), 1);
        let (r, _) = run(
            synth_big_cmp(),
            &[a as i64, b as i64],
            &[(a, &[9, 9, 9]), (b, &[9, 9, 9])],
        );
        assert_eq!(ret_i64(&r), 0);
        // a high limb dominates a larger low limb
        let (r, _) = run(
            synth_big_cmp(),
            &[a as i64, b as i64],
            &[(a, &[0, 0, 5]), (b, &[9, 9, 4])],
        );
        assert_eq!(ret_i64(&r), 1);
    }

    #[test]
    fn big_sub_works() {
        let a = 256usize;
        let b = 512usize;
        let (_, w) = run(
            synth_big_sub(),
            &[a as i64, b as i64],
            &[(a, &[5]), (b, &[3])],
        );
        assert_eq!(limbs(&w, a, 2), vec![2, 0]);
        // borrow across a limb: 2^32 - 1
        let (_, w) = run(
            synth_big_sub(),
            &[a as i64, b as i64],
            &[(a, &[0, 1]), (b, &[1])],
        );
        assert_eq!(limbs(&w, a, 2), vec![0xFFFF_FFFF, 0]);
        // multi-limb borrow chain: 2^64 - 1
        let (_, w) = run(
            synth_big_sub(),
            &[a as i64, b as i64],
            &[(a, &[0, 0, 1]), (b, &[1])],
        );
        assert_eq!(limbs(&w, a, 3), vec![0xFFFF_FFFF, 0xFFFF_FFFF, 0]);
    }

    #[test]
    fn big_mul_small_works() {
        let a = 256usize;
        let (_, w) = run(
            synth_big_mul_small(),
            &[a as i64, 2],
            &[(a, &[0xFFFF_FFFF])],
        );
        assert_eq!(limbs(&w, a, 2), vec![0xFFFF_FFFE, 1]); // 2·(2^32-1) = 2^33-2
        let (_, w) = run(
            synth_big_mul_small(),
            &[a as i64, 10],
            &[(a, &[123_456_789])],
        );
        assert_eq!(limbs(&w, a, 2), vec![1_234_567_890, 0]);
        // carry chain across two limbs: (2^64-1)·10
        let prod = (u64::MAX as u128) * 10;
        let (_, w) = run(
            synth_big_mul_small(),
            &[a as i64, 10],
            &[(a, &[0xFFFF_FFFF, 0xFFFF_FFFF])],
        );
        let got = limbs(&w, a, 3);
        let exp = [prod as u32, (prod >> 32) as u32, (prod >> 64) as u32];
        assert_eq!(got, exp);
    }

    #[test]
    fn big_shr1_works() {
        let a = 256usize;
        let (r, w) = run(synth_big_shr1(), &[a as i64], &[(a, &[0b1011])]);
        assert_eq!(ret_i64(&r), 1); // out bit = old bit 0
        assert_eq!(limbs(&w, a, 1), vec![0b101]);
        // borrow across a limb boundary: bit 0 of limb1 falls into limb0's top
        let (r, w) = run(synth_big_shr1(), &[a as i64], &[(a, &[0, 1])]);
        assert_eq!(ret_i64(&r), 0);
        assert_eq!(limbs(&w, a, 2), vec![0x8000_0000, 0]);
    }

    #[test]
    fn big_divmod10_works() {
        let a = 256usize;
        let (r, w) = run(synth_big_divmod10(), &[a as i64], &[(a, &[123])]);
        assert_eq!(ret_i64(&r), 3);
        assert_eq!(limbs(&w, a, 1), vec![12]);
        // multi-limb: (3*2^32 + 7) / 10
        let val: u128 = 3 * (1u128 << 32) + 7;
        let (r, w) = run(synth_big_divmod10(), &[a as i64], &[(a, &[7, 3])]);
        assert_eq!(ret_i64(&r) as u128, val % 10);
        let q = val / 10;
        assert_eq!(limbs(&w, a, 2), vec![q as u32, (q >> 32) as u32]);
    }

    #[test]
    fn big_inc_works() {
        let a = 256usize;
        let (_, w) = run(synth_big_inc(), &[a as i64], &[(a, &[41])]);
        assert_eq!(limbs(&w, a, 1), vec![42]);
        // carry across two limbs: (2^64 - 1) + 1 = 2^64
        let (_, w) = run(
            synth_big_inc(),
            &[a as i64],
            &[(a, &[0xFFFF_FFFF, 0xFFFF_FFFF])],
        );
        assert_eq!(limbs(&w, a, 3), vec![0, 0, 1]);
    }

    #[test]
    fn dtoa_digits_works() {
        let scratch = 1024usize;
        let dbuf = scratch + 496;
        let cases: &[(f64, usize)] = &[
            (3.25, 3),
            (1.0, 3),
            (2.0 / 3.0, 5),
            (0.1, 17),
            (9.999, 3), // round carries into a new leading digit (→ 1.00e1)
            (123456.789, 9),
            (1e30, 5),                   // large positive exponent (left-shift build path)
            (1e-30, 5),                  // negative exponent (denominator scaling)
            (2.5, 1),                    // round-half-to-even tie → 2
            (3.5, 1),                    // tie → 4
            (5e-324, 3),                 // smallest subnormal (E_est far off ⇒ many fixup steps)
            (1.7976931348623157e308, 5), // largest finite double
        ];
        for &(v, nsig) in cases {
            let bits = v.to_bits() as i64;
            let (res, win) = run_funcs(dtoa_funcs(), &[bits, nsig as i64, scratch as i64], &[]);
            let e = ret_i64(&res);
            let got: Vec<u8> = (0..nsig).map(|j| win[dbuf + j]).collect();
            let (exp_d, exp_e) = expected(v, nsig);
            assert_eq!((got, e), (exp_d, exp_e), "v={v} nsig={nsig}");
        }
        // zero ⇒ all-zero digits, E = 0
        let (res, win) = run_funcs(dtoa_funcs(), &[0i64, 4, scratch as i64], &[]);
        assert_eq!(ret_i64(&res), 0);
        assert_eq!(
            (0..4).map(|j| win[dbuf + j]).collect::<Vec<_>>(),
            vec![0, 0, 0, 0]
        );
    }

    #[test]
    fn dtoa_sci_works() {
        let scratch = 4096usize;
        let out = scratch + 1536;
        // (value, prec); no flags, no width
        let cases: &[(f64, usize)] = &[
            (3.25, 2),
            (0.0, 6),
            (2.0 / 3.0, 6),
            (1e30, 3),
            (1e-30, 3),
            (123456.789, 4),
            (9.999999, 2), // rounds up, carries → 1.00e+01
            (5e-324, 2),
            (1.7976931348623157e308, 4),
            (-2.5, 3),
            (-0.0, 2),
            (1.0, 0), // prec 0 ⇒ no decimal point
        ];
        for &(v, prec) in cases {
            let bits = v.to_bits() as i64;
            let (res, win) =
                run_funcs(sci_funcs(), &[bits, prec as i64, 0, 0, scratch as i64], &[]);
            let len = ret_i64(&res) as usize;
            let got = String::from_utf8(win[out..out + len].to_vec()).unwrap();
            assert_eq!(got, c_e(v, prec, false), "v={v} prec={prec}");
        }
        // %E (uppercase) + inf/nan + flags + width
        let bits = 6.022e23f64.to_bits() as i64;
        let (res, win) = run_funcs(sci_funcs(), &[bits, 3, 0, 8, scratch as i64], &[]); // flags bit3 = upper
        let len = ret_i64(&res) as usize;
        assert_eq!(
            String::from_utf8(win[out..out + len].to_vec()).unwrap(),
            c_e(6.022e23, 3, true)
        );
        // +/space flags
        let (res, win) = run_funcs(
            sci_funcs(),
            &[1.5f64.to_bits() as i64, 2, 0, 2, scratch as i64],
            &[],
        );
        let len = ret_i64(&res) as usize;
        assert_eq!(
            String::from_utf8(win[out..out + len].to_vec()).unwrap(),
            "+1.50e+00"
        );
        // width 14, right-justified
        let (res, win) = run_funcs(
            sci_funcs(),
            &[3.25f64.to_bits() as i64, 2, 14, 0, scratch as i64],
            &[],
        );
        let len = ret_i64(&res) as usize;
        assert_eq!(
            String::from_utf8(win[out..out + len].to_vec()).unwrap(),
            "      3.25e+00"
        );
        // inf / nan, lower and upper
        let (res, win) = run_funcs(
            sci_funcs(),
            &[f64::INFINITY.to_bits() as i64, 6, 0, 0, scratch as i64],
            &[],
        );
        let len = ret_i64(&res) as usize;
        assert_eq!(
            String::from_utf8(win[out..out + len].to_vec()).unwrap(),
            "inf"
        );
        let (res, win) = run_funcs(
            sci_funcs(),
            &[f64::NAN.to_bits() as i64, 6, 0, 8, scratch as i64],
            &[],
        );
        let len = ret_i64(&res) as usize;
        assert_eq!(
            String::from_utf8(win[out..out + len].to_vec()).unwrap(),
            "NAN"
        );
        let (res, win) = run_funcs(
            sci_funcs(),
            &[
                (f64::NEG_INFINITY).to_bits() as i64,
                6,
                0,
                0,
                scratch as i64,
            ],
            &[],
        );
        let len = ret_i64(&res) as usize;
        assert_eq!(
            String::from_utf8(win[out..out + len].to_vec()).unwrap(),
            "-inf"
        );
    }

    #[test]
    fn dtoa_fix_big_works() {
        let scratch = 4096usize;
        let out = scratch + 1536;
        let fns = || {
            vec![
                synth_dtoa_fix_big(1, 2, 3, 4, 5, 6, 7),
                synth_big_zero(),
                synth_big_mul_small(),
                synth_big_shl_bits(),
                synth_big_shr1(),
                synth_big_inc(),
                synth_big_divmod10(),
                synth_big_is_zero(),
            ]
        };
        let cases: &[(f64, usize)] = &[
            (3.375, 2),
            (0.0, 6),
            (0.1, 3),
            (2.5, 0), // tie ⇒ even ⇒ 2
            (3.5, 0), // tie ⇒ even ⇒ 4
            (100.0, 2),
            (12345.6789, 3),
            (0.5, 10),
            (2.0 / 3.0, 4),
            (-2.75, 6),
            (-0.0, 2),
            (9007199254740992.0, 1), // 2^53
            (1e30, 2),               // 33-digit integer (over the 128-bit ceiling)
            (1e300, 0),              // huge — the case the 128-bit path traps on
            (1e-300, 3),             // tiny ⇒ 0.000
            (0.0000006, 6),          // tiny round-up edge ⇒ 0.000001
            (0.0000004, 6),          // ⇒ 0.000000
            (1234567890123456.0, 4),
        ];
        for &(v, prec) in cases {
            let bits = v.to_bits() as i64;
            let (res, win) = run_funcs(fns(), &[bits, prec as i64, 0, 0, scratch as i64], &[]);
            let len = ret_i64(&res) as usize;
            let got = String::from_utf8(win[out..out + len].to_vec()).unwrap();
            assert_eq!(got, format!("{v:.prec$}"), "v={v} prec={prec}");
        }
        // sign flags / width / inf-nan
        let (res, win) = run_funcs(
            fns(),
            &[1.5f64.to_bits() as i64, 2, 0, 2, scratch as i64],
            &[],
        );
        let len = ret_i64(&res) as usize;
        assert_eq!(
            String::from_utf8(win[out..out + len].to_vec()).unwrap(),
            "+1.50"
        );
        let (res, win) = run_funcs(
            fns(),
            &[2.5f64.to_bits() as i64, 1, 10, 0, scratch as i64],
            &[],
        );
        let len = ret_i64(&res) as usize;
        assert_eq!(
            String::from_utf8(win[out..out + len].to_vec()).unwrap(),
            "       2.5"
        );
        let (res, win) = run_funcs(
            fns(),
            &[f64::INFINITY.to_bits() as i64, 6, 0, 8, scratch as i64],
            &[],
        );
        let len = ret_i64(&res) as usize;
        assert_eq!(
            String::from_utf8(win[out..out + len].to_vec()).unwrap(),
            "INF"
        );
    }

    #[test]
    fn dtoa_gen_works() {
        let scratch = 4096usize;
        let out = scratch + 1536;
        let cases: &[(f64, usize)] = &[
            (3.375, 6),
            (100000.0, 6),  // E=5 < P ⇒ f-mode "100000"
            (1000000.0, 6), // E=6 ≥ P ⇒ e-mode "1e+06"
            (0.0001, 6),    // E=-4 ⇒ f-mode
            (0.00001, 6),   // E=-5 ⇒ e-mode
            (0.1, 6),
            (123456789.0, 6),
            (2.0 / 3.0, 6),
            (1.5, 6),
            (0.0, 6),
            (1e300, 6),
            (999999.9, 6), // rounds to 1e6 ⇒ e-mode (carry across the e/f boundary)
            (1.0, 1),
            (-2.5, 3),
            (-0.0, 4),
            (42.0, 6), // integer-valued ⇒ "42"
            (3.0, 1),
        ];
        for &(v, prec) in cases {
            let bits = v.to_bits() as i64;
            let (res, win) =
                run_funcs(gen_funcs(), &[bits, prec as i64, 0, 0, scratch as i64], &[]);
            let len = ret_i64(&res) as usize;
            let got = String::from_utf8(win[out..out + len].to_vec()).unwrap();
            assert_eq!(got, c_g(v, prec, false), "v={v} prec={prec}");
        }
        // %G uppercase + width + inf/nan
        let (res, win) = run_funcs(
            gen_funcs(),
            &[1e-20f64.to_bits() as i64, 4, 0, 8, scratch as i64],
            &[],
        );
        let len = ret_i64(&res) as usize;
        assert_eq!(
            String::from_utf8(win[out..out + len].to_vec()).unwrap(),
            c_g(1e-20, 4, true)
        );
        let (res, win) = run_funcs(
            gen_funcs(),
            &[f64::INFINITY.to_bits() as i64, 6, 0, 0, scratch as i64],
            &[],
        );
        let len = ret_i64(&res) as usize;
        assert_eq!(
            String::from_utf8(win[out..out + len].to_vec()).unwrap(),
            "inf"
        );
    }

    #[test]
    fn big_zero_works() {
        let a = 256usize;
        let (_, w) = run(
            synth_big_zero(),
            &[a as i64],
            &[(a, &[1, 2, 3, 4, 5, 6, 7, 8])],
        );
        assert_eq!(limbs(&w, a, 8), vec![0; 8]);
    }

    #[test]
    fn big_copy_works() {
        let a = 256usize;
        let b = 512usize;
        let src = [9u32, 8, 7, 0, 0, 42];
        let (_, w) = run(synth_big_copy(), &[a as i64, b as i64], &[(b, &src)]);
        assert_eq!(limbs(&w, a, 6), src.to_vec());
        // copying overwrites prior dst content fully (all 40 limbs)
        let (_, w) = run(
            synth_big_copy(),
            &[a as i64, b as i64],
            &[(a, &[1, 1, 1, 1, 1, 1]), (b, &[5])],
        );
        assert_eq!(limbs(&w, a, 6), vec![5, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn big_shl_bits_works() {
        let a = 256usize;
        let (_, w) = run(synth_big_shl_bits(), &[a as i64, 1], &[(a, &[1])]);
        assert_eq!(limbs(&w, a, 2), vec![2, 0]);
        let (_, w) = run(synth_big_shl_bits(), &[a as i64, 32], &[(a, &[1])]);
        assert_eq!(limbs(&w, a, 2), vec![0, 1]);
        let (_, w) = run(synth_big_shl_bits(), &[a as i64, 33], &[(a, &[1])]);
        assert_eq!(limbs(&w, a, 2), vec![0, 2]);
        // carry out of bit 31 into the next limb
        let (_, w) = run(synth_big_shl_bits(), &[a as i64, 1], &[(a, &[0x8000_0000])]);
        assert_eq!(limbs(&w, a, 2), vec![0, 1]);
        // word + bit shift: 3 << 40 = 0x300 in limb 1
        let (_, w) = run(synth_big_shl_bits(), &[a as i64, 40], &[(a, &[3])]);
        assert_eq!(limbs(&w, a, 3), vec![0, 0x300, 0]);
        // a 53-bit significand shifted by a large exponent (the %f/%e build path), checked vs u128
        let f: u128 = 0x1F_FFFF_FFFF_FFFF; // 2^53 - 1
        let n = 70u32;
        let exp = f << n;
        let lo = [f as u32, (f >> 32) as u32];
        let (_, w) = run(synth_big_shl_bits(), &[a as i64, n as i64], &[(a, &lo)]);
        let got = limbs(&w, a, 5);
        let expv: Vec<u32> = (0..5)
            .map(|k| {
                let sh = 32 * k;
                if sh >= 128 {
                    0
                } else {
                    (exp >> sh) as u32
                }
            })
            .collect();
        assert_eq!(got, expv);
    }
}
