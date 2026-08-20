//! Core IR: block-local typed SSA over a CFG of basic blocks.
//!
//! See `DESIGN.md` §3/§3a/§3b. Key disciplines encoded here:
//! - Values are **block-local**: within a block, indices run `0..k` for the block
//!   parameters, then one more per instruction result. Operands reference *earlier*
//!   same-block indices only. Cross-block dataflow is *only* via block parameters,
//!   so dominance analysis is impossible to need (verifier is a linear pass).
//! - Every block ends in exactly one terminator.
//!
//! Phase-1 integer core: `i32`/`i64` constants, the full integer arithmetic /
//! bitwise / shift / comparison set, `i32`↔`i64` conversions, `select`, and the
//! `br`/`br_if`/`br_table`/`return` terminators. Float, memory, calls, and
//! capabilities come in later batches per §3b.
#![forbid(unsafe_code)]
#![cfg_attr(not(test), no_std)]

extern crate alloc;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec; // the `vec!` macro
use alloc::vec::Vec;

/// Block-local value index (parameters first, then instruction results in order).
pub type ValIdx = u32;
/// Index of a block within a function (`0` = entry).
pub type BlockIdx = u32;
/// Index of a function within a module.
pub type FuncIdx = u32;

/// Reserved pseudo-`type_id` for §7 capability **reflection** (`cap.self.*`). It is not a real
/// capability (no handle ever carries it): both backends lower `cap.self.count`/`cap.self.get` to a
/// host `cap.call` with this `type_id` (op 0 = count, op 1 = get), which the host's dispatch services
/// directly — read-only over the calling domain's own table — instead of resolving a handle. Sharing
/// one host entry point keeps the interpreter and JIT in lockstep. (Equivalent to issuing the
/// intrinsic, since reflection is ambient/authority-neutral; `u32::MAX` collides with no interface.)
pub const CAP_SELF_TYPE_ID: u32 = u32::MAX;

/// Reserved pseudo-`type_id` for **executable named imports** (IMPORTS.md phase 1). A verified
/// [`Inst::CallImport`] dispatches as a host `cap.call` with this `type_id` and the **import index**
/// as the `op`; the host translates it through the domain's instantiation-time import-binding table
/// (import `i` → the bound `(type_id, op)` + granted handle — the powerbox-prefix slot) and
/// re-dispatches. Like [`CAP_SELF_TYPE_ID`], sharing one host entry point keeps the interpreter,
/// bytecode engine, and JIT in lockstep over one implementation, and the module bytes are **never
/// rewritten** (the `resolve_imports` lowering becomes a linker-only concern). Not a real capability:
/// no handle ever carries it, and a table entry can never bind to it.
pub const CAP_IMPORT_TYPE_ID: u32 = u32::MAX - 2;

/// Reserved pseudo-`type_id` for **`import.attach`** (IMPORTS.md phase 2): rebinding a
/// [`ImportMode::Rebindable`] import slot to a capability the domain already holds. Dispatched like
/// [`CAP_IMPORT_TYPE_ID`] — the `op` carries the import index, the single argument the handle value
/// to attach — and serviced host-side against the instantiation-time binding table: the handle must
/// resolve **live** under the slot's declared interface `type_id` (mask + type + generation, §3c),
/// then the slot's bound handle is swapped. Authority-neutral: it aliases a held capability into a
/// named slot (no new grant-graph edge, the D37 argument). Returns `0` ok / `-errno`.
pub const CAP_IMPORT_ATTACH_TYPE_ID: u32 = u32::MAX - 3;

/// Reserved pseudo-`type_id` for **dynamic-mode dispatch by type-section reference**
/// ([`Inst::CallImportDyn`], IMPORTS.md §3.5). The `op` packs `(type_idx | op << 16)`; the handle
/// operand is the live handle value. Serviced host-side: the domain's registered self-module
/// interface shape at `type_idx` is interned to its structural `type_id` and the dispatch
/// re-enters with `(that id, op, handle)` — the ordinary §3c use-site check does the rest. One
/// shared entry keeps the three backends in lockstep, like [`CAP_IMPORT_TYPE_ID`].
pub const CAP_DYN_TYPE_ID: u32 = u32::MAX - 4;

/// Built-in capability **interface ids** — the `type_id` a `cap.call` names (§3c/§3e). One
/// definition, in the wire-IR crate, so every backend (the interpreter's dispatch, both JITs'
/// gate predicates) agrees without copying raw integers; `svm_interp::cap_id` re-exports this
/// (moved here from svm-interp, 2026-08-18, #902). Doc links to host-crate types are rendered as
/// plain code spans since this crate has no `Host`.
pub mod cap_id {
    /// `Stream` — byte stream: op 0 `read`, op 1 `write`, op 2 `close` (§3e D43).
    pub const STREAM: u32 = 0;
    /// `Exit` — lifecycle: op 0 `exit(code)` (noreturn).
    pub const EXIT: u32 = 1;
    /// `Clock` — op 0 `now(clock_id) -> i64` nanoseconds.
    pub const CLOCK: u32 = 2;
    // 3: retired `Memory` (CONSOLIDATION §4, 2026-08-06). It was `AddressSpace` over the whole
    // window minus `sub` — the degenerate form. Whole-window memory authority is now an
    // `ADDRESS_SPACE` grant with the whole-window range (`Host::grant_memory`); the id stays
    // reserved so an old artifact or guest naming it fails closed rather than aliasing a new kind.
    /// `SharedRegion` — a host-backed memory object aliased into the window (§13). op 0
    /// `map(window_offset, region_offset, len, prot)` aliases the region's pages into the window
    /// (the same backing may be mapped at *multiple* window offsets → zero-overhead aliasing, the
    /// magic-ring-buffer primitive); op 1 `unmap(window_offset, len)` drops the alias; op 2
    /// `len() -> i64` reports the region size; op 3 `page_size() -> i64`. Granting the handle is how
    /// two domains come to share memory; `create`/`grant` (guest-minted regions, cross-domain) are a
    /// §14 follow-up — today regions are host-granted. A backing may be a fresh OS
    /// shared object (`memfd`) **or a real host file** (`svm-run`'s `FileBacking`, minted by an
    /// mmap-capable fs cap): mapping the latter aliases the file into the window zero-copy — the
    /// file-backed-mmap bridge (MMAP_CAPABILITY.md §4b).
    pub const SHARED_REGION: u32 = 4;
    /// `AddressSpace` — the §14 memory-management capability, **attenuable to a power-of-two
    /// window sub-range** `[base, base+size)`. Every op is confined to the
    /// holder's sub-range (offsets are sub-range-relative, shifted by `base`): op 0 `map(off,len,prot)`,
    /// 1 `unmap(off,len)`, 2 `protect(off,len,prot)`, 3 `page_size() -> i64`, and 4
    /// **`sub(off, size_log2) -> handle`** — the **attenuation** primitive: mint a child `AddressSpace`
    /// over the power-of-two-aligned sub-range `[base+off, base+off + 2^size_log2)`, which must lie
    /// within the holder's range (a parent can only sub-allocate what it holds, §14). This is the
    /// memory half of the `Instantiator`: a guest carves a child's window from its own.
    pub const ADDRESS_SPACE: u32 = 5;
    /// `Instantiator` — the §14 nesting primitive: spawn a **child domain** confined to a
    /// power-of-two sub-window `[base, base+size)` of the holder's window (VM-in-VM). op 0
    /// `instantiate(entry, off, size_log2, fuel) -> child_handle` enqueues a child vCPU running the
    /// same module's `entry` (which returns one `i64` and takes one or two — its starter caps)
    /// confined to `[base+off, base+off+2^size_log2)` with an **attenuated** powerbox over the child's
    /// own window: an `Instantiator` (so it can recurse — confinement composes to any depth) and an
    /// `AddressSpace` (so it can manage its own pages), passed as the entry's arguments. A fuel quota
    /// caps it; returns immediately (non-blocking). op 1 `join(child_handle) -> result` parks **only
    /// the calling fiber** until that child finishes, then yields its result (siblings keep running —
    /// the child rides the same §12 executor). Holding the handle is the authority to nest (D19: a
    /// child can only get what the parent sub-allocates).
    pub const INSTANTIATOR: u32 = 6;
    /// `Module` — a host-granted, host-**verified** module a guest may instantiate (§14). The handle
    /// confers only the authority to pass it to the `Instantiator`'s module ops (5/6/7 —
    /// `instantiate_module` / `spawn_coroutine_module` / `spawn_demand_coroutine_module`), which
    /// spawn a child domain running *that* module's code confined to a carve of the holder's window
    /// — the "plugin-in-plugin" story: a guest can only instantiate modules it was given (no ambient
    /// authority). It has no directly callable ops (`cap.call` on it is an inert `CapFault`).
    pub const MODULE: u32 = 8;
    // id 9 was `IoRing` (the submit/complete ring), retired 2026-08-07 by the §12
    // parking-on-blocking re-measure: batching measured 13× negative, overlap subsumed by
    // parked blocking dispatches (DESIGN.md §12). The id stays reserved — never reuse it.
    /// §12 `Blocking` — a *mock* synchronous-only / blocking host capability (DNS-/FS-blocking-shaped)
    /// whose op 0 `work(arg) -> mix(arg)` is **window-independent and `&mut Host`-free**, so a
    /// punting dispatch hands it to the offload pool instead of the guest's vCPU thread. Op 0 is
    /// also a perfectly ordinary synchronous `cap.call` (it then blocks the caller — the degenerate path).
    ///
    /// **Test-only since CONSOLIDATION §5a:** no product powerbox grants it — it is the
    /// offload-pool / §12 parking exerciser (an offloadable dispatch that always punts when it
    /// would genuinely block). A harness that needs it calls `Host::grant_blocking`
    /// (and registers the `"blocking"` name if its guest resolves by name).
    pub const BLOCKING: u32 = 10;
    /// `Jit` — the guest-driven JIT capability (DESIGN.md §22): submit serialized IR at runtime to
    /// be validated (decode + verify + the memory-match precondition, via the host-injected
    /// `JitValidator`) and compiled into the **same domain** (same window, same powerbox —
    /// a module is not an isolation unit, DESIGN §8). op 0 `compile(ptr, len) -> code_handle | -errno`
    /// (fail-closed: nothing is installed on any validation failure); op 1
    /// `invoke(code_handle, args…) -> results` runs the compiled unit's entry (`funcs[0]`) over the
    /// caller's **live window** — serviced by the eval loop on the interpreter (it must run guest
    /// code, which the generic dispatch can't) and by the embedder's cap thunk on the JIT (it calls
    /// the unit's native trampoline); traps in invoked code are **terminal for the domain**; op 2
    /// `release(code_handle) -> 0 | -errno` revokes the handle (no code reclaim yet — DESIGN.md §22
    /// "Code reclaim"); op 3 `install(code_handle) -> slot_index | -errno` (Model B2) installs the
    /// unit into the `call_indirect` table's next reserved slot so old code (or another unit) can
    /// dispatch it at native speed (old→new), `-ENOSPC` if the table is full; op 4
    /// `uninstall(slot) -> 0 | -errno` clears an installed slot so the index is reusable and a
    /// stale `call_indirect` of it traps (slot reclaim — the code memory itself is not freed).
    pub const JIT: u32 = 11;
    /// `CompiledCode` — a unit minted by `Jit.compile`. Like `Module`, it has no directly callable
    /// ops (`cap.call` on it is an inert `CapFault`); it confers only the authority to be named in
    /// `Jit.invoke`/`release` on the domain handle that compiled it.
    pub const JIT_CODE: u32 = 12;
    /// `HostProc` — an **embedder-registered** capability (§7 "host-defined capabilities"): the host
    /// installs a handler closure with `grant_host_proc` and the guest reaches it like
    /// any capability (`cap.call HOST_PROC op …`). The interface's *semantics* live entirely in the
    /// embedder's closure (e.g. a WASI shim), **outside** this crate's TCB match — so a host
    /// can add capabilities without touching the VM. The handler reads/writes the guest window
    /// through the same masked `GuestMem` the built-in ops use (authority-TCB, not escape-TCB).
    pub const HOST_PROC: u32 = 13;
    /// §15 / PROCESS.md §5 `Budget` — a passable, **splittable** resource-quota vector (fuel / mem /
    /// spawn), §15's "every meterable resource is a capability with a quota" promoted to an object.
    /// op 0 `split(fuel, mem, spawn) -> sub_handle | -errno`: mint a child `Budget` holding those
    /// amounts, **deducted** from the holder's remaining — attenuation (a child can never exceed the
    /// parent, D19); a field of `-1` means "all remaining"; asking for more than remains is `-EINVAL`.
    /// op 1 `read(field) -> remaining | -EINVAL`: report one field's remaining quota (`0` fuel, `1`
    /// mem, `2` spawn) — the §15 monitoring readout. Charging a domain's consumption against its budget
    /// (the `create(module, window, budget)` accounting) is the follow-up; this is the passable object
    /// + attenuation the rest builds on.
    pub const BUDGET: u32 = 14;
    /// PROCESS.md §5 **window minter** — the authority to mint **detached** windows: a child
    /// spawned through it (`Instantiator.instantiate_detached`, op 15) gets a fresh platform
    /// window *outside* the parent's — no ancestor below the minter holds read authority, and
    /// the child attests `window_exposed = false` (the jacl distrust-spawner trust anchor).
    /// The capability carries a **byte quota**, deducted at each mint (host-enforced); an
    /// ordinary granted authority (D46 `Resolver`-shaped: you can mint detached windows only
    /// if someone granted you that), embedder-granted at the root.
    pub const WINDOW_MINTER: u32 = 15;
    /// Base of the **guest-interface id space** (IMPORTS.md §3.2): ids for wired interface offers
    /// are interned per-`Host` from this base upward (`intern_interface` — the id ≡
    /// the structural op-signature list, the D59 rule applied to capability interfaces). Far above
    /// the fixed built-ins and far below the reserved `u32::MAX`-family dispatch sentinels.
    pub const GUEST_IMPL_BASE: u32 = 0x1000_0000;
}

/// Guest-facing **errno values** — the negative `-errno` a fallible capability op returns on its
/// own error path (INVARIANTS #5: "errors are values"). **One definition** here, in the wire-IR
/// crate every host crate already depends on, so the values and their **sign** cannot drift between
/// the servicers (svm-fs, svm-exec, svm-posix), the runtime (svm-interp), the embedder (svm-run),
/// and the JIT thunks (svm-jit). The convention is **negative**: this is what crosses the ABI, since
/// the runtime sign-tests a result as `handle | -errno` (a negative i64 is an error, a non-negative
/// one a value — the invariant-11 packing). Standard Linux numeric values, negated. Moved here from
/// the six crates that each defined their own copy, 2026-08-18 (#905). The guest-side mirrors in
/// `svm-llvm/rust-svm/*-imp.rs` are a separate compilation universe (guest code cannot link this
/// crate) and keep their own copies, citing this module as the authority.
pub mod errno {
    /// Operation not permitted.
    pub const EPERM: i64 = -1;
    /// No such file or directory.
    pub const ENOENT: i64 = -2;
    /// No such process (a pid the table does not know).
    pub const ESRCH: i64 = -3;
    /// Interrupted by a delivered signal (#796 — a blocking op woken by a signal).
    pub const EINTR: i64 = -4;
    /// Bad file descriptor / handle.
    pub const EBADF: i64 = -9;
    /// No child processes (`wait`/`reap` for a pid that is not a live child).
    pub const ECHILD: i64 = -10;
    /// Try again — would block a cooperative guest (empty socket/pipe read, etc.).
    pub const EAGAIN: i64 = -11;
    /// Out of memory — a resource quota was exhausted (e.g. the JIT compile budget).
    pub const ENOMEM: i64 = -12;
    /// Permission denied.
    pub const EACCES: i64 = -13;
    /// Bad address — a buffer not fully within the guest window.
    pub const EFAULT: i64 = -14;
    /// File exists (`mkdir`/`rename` onto an existing path).
    pub const EEXIST: i64 = -17;
    /// Not a directory (`opendir`/dir op on a regular file).
    pub const ENOTDIR: i64 = -20;
    /// Invalid argument.
    pub const EINVAL: i64 = -22;
    /// Too many open files — the handle table is full (§3c).
    pub const EMFILE: i64 = -24;
    /// Inappropriate ioctl for device — a `tc*` op on a non-terminal fd (#798).
    pub const ENOTTY: i64 = -25;
    /// No space left — no free table slot (e.g. the JIT install table is full).
    pub const ENOSPC: i64 = -28;
    /// Illegal seek (`lseek` on a pipe/stdio fd).
    pub const ESPIPE: i64 = -29;
    /// Broken pipe — write to a pipe whose read end is fully closed (FORK.md §8.6).
    pub const EPIPE: i64 = -32;
    /// Result too large for the caller's buffer (`getcwd`).
    pub const ERANGE: i64 = -34;
    /// Function not implemented — no embedder-wired delegate (fail closed).
    pub const ENOSYS: i64 = -38;
    /// Directory not empty (`rmdir` on a non-empty directory).
    pub const ENOTEMPTY: i64 = -39;
    /// Socket operation on a non-socket fd.
    pub const ENOTSOCK: i64 = -88;
    /// Address already in use (loopback port another listener holds).
    pub const EADDRINUSE: i64 = -98;
    /// Connection refused (no listener, or no delegate beyond loopback).
    pub const ECONNREFUSED: i64 = -111;
}

/// The **durable (freeze/thaw) ABI layout** (DURABILITY.md §12) — the byte offsets and state-word
/// values the transform emits and every backend (svm-durable, svm-interp, svm-jit) plus the durable
/// runtime must agree on. Hoisted here (svm-ir is the common dependency of all three) so the layout
/// has **one definition** instead of three hand-synced copies gated by a partial pin test (#915).
pub mod durable_abi {
    /// Window byte offset of the `i32` **freeze state** word (`NORMAL`/`UNWINDING`/`REWINDING`/
    /// `ARMED`). A freeze is stop-the-world, so this single global word is the broadcast every poll
    /// reads (per-context thaw uses [`STATE_IN_REGION_OFF`] instead).
    pub const STATE_OFF: u64 = 0;
    /// Window byte offset of the `i64` shadow-stack pointer (itself a window byte offset).
    pub const SHADOW_SP_OFF: u64 = 8;
    /// Byte length of the in-region shadow-SP word (before the per-context thaw state word).
    pub const SHADOW_SP_WORD_LEN: u64 = 8;
    /// Window byte offset of the `i64` **fiber-safepoint arm countdown** — safepoints
    /// (`cont.resume`/`suspend`) still to pass before an `ARMED` run promotes to `UNWINDING`. Inert
    /// unless armed; lives in the reserve's `[16, 64)` gap, so an unarmed run is byte-identical.
    pub const ARM_COUNTDOWN_OFF: u64 = 16;
    /// Window byte offset of the `i64` **back-edge arm countdown** — loop back-edges still to pass
    /// before an `ARMED` run promotes to `UNWINDING` (the Phase-4 Slice A back-edge-poll trigger,
    /// separate from [`ARM_COUNTDOWN_OFF`]). Reserve's `[24, 64)` gap.
    pub const ARM_BACKEDGE_OFF: u64 = 24;
    /// Window byte offset of the `i8` **freeze-on-quiesce** flag: non-zero arms the runtime to freeze
    /// when the run would otherwise block only on `svc.wait`-parked consumers (an idle server no
    /// countdown can reach). Reserve's `[32, 64)` gap.
    pub const ARM_QUIESCE_OFF: u64 = 32;
    /// §12.8 concurrent-thaw: byte offset of a context's **per-context thaw state** word
    /// (`REWINDING`/`NORMAL`) within its region — just past the [`SHADOW_SP_WORD_LEN`]-byte in-region
    /// SP word. Each frozen vCPU rewinds against its own word, so thaw can run them as concurrent
    /// threads; the stop-the-world freeze state stays at the global [`STATE_OFF`].
    pub const STATE_IN_REGION_OFF: u64 = SHADOW_SP_WORD_LEN;
    /// §12.8: bytes reserved at a context region's base before its shadow frames — the SP word plus
    /// the thaw state word at [`STATE_IN_REGION_OFF`], padded to 8 to keep frames 8-aligned.
    pub const REGION_HEADER_LEN: u64 = 16;
    /// Window byte offset where the shadow stack begins (grows upward, bounded by [`DURABLE_RESERVE`]).
    pub const SHADOW_BASE: u64 = 64;
    /// Per-context shadow-region stride: context `i` owns `[SHADOW_BASE + i*SHADOW_STRIDE, +stride)`.
    pub const SHADOW_STRIDE: u64 = 1 << 12;
    /// Size of the reserved low region (one 64 KiB wasm page): `[0, DURABLE_RESERVE)` holds the state
    /// word, shadow-SP, and shadow stack; the guest's memory is `[DURABLE_RESERVE, window)`.
    pub const DURABLE_RESERVE: u64 = 1 << 16;

    /// Freeze/thaw **state-word values** ([`STATE_OFF`] / [`STATE_IN_REGION_OFF`]).
    pub const STATE_NORMAL: i32 = 0;
    /// A stop-the-world freeze is in progress (unwinding shadow frames).
    pub const STATE_UNWINDING: i32 = 1;
    /// A thaw is in progress (rewinding into shadow frames).
    pub const STATE_REWINDING: i32 = 2;
    /// The run is armed to begin a freeze once a countdown / quiesce trigger fires.
    pub const STATE_ARMED: i32 = 3;

    /// `svc.poll` / `svc.wait` op indices on the durable service interface (§13.4) — the two the
    /// quiesce-freeze arming (`ARM_QUIESCE_OFF`) keys on.
    pub const SVC_POLL_OP: u32 = 9;
    pub const SVC_WAIT_OP: u32 = 10;
}

/// The **op-17 spawn config record** (CONSOLIDATION.md §3/§3c/§3d): the fixed 56-byte little-endian
/// layout that every `instantiate_rec` driver decodes — the tree-walker's op-17 arm, the bytecode
/// tier's `Op::InstantiateRec`, and the Cranelift `instantiate_rec` thunk. One record subsumes every
/// §14 spawn shape as data (module / entry / carve / pager / budget / quota / named-grant list).
/// Hoisted here (svm-ir is the common dependency of all three) so the byte layout has **one**
/// definition instead of three hand-decoded copies (#911). Only the *field extraction* is shared:
/// each tier keeps its own pager / budget / grant handling and error order — they diverge by design
/// (see the call sites), so this decodes the fields and validates nothing beyond the version word.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SpawnRec {
    /// Child entry function index (record offset 4).
    pub entry: u32,
    /// Carve window offset within the parent's address space (offset 8).
    pub off: u64,
    /// Carve size as a power-of-two shift count; the driver validates it against `0..64` (offset 16).
    pub size_log2: i64,
    /// Pager impl-export index, or `u32::MAX` for none (offset 20).
    pub pager: u32,
    /// `Module` handle, or `-1` for self (offset 24).
    pub modh: i32,
    /// `Budget` handle, or `0` for none — mutually exclusive with a nonzero `quota` (offset 28).
    pub budget: i32,
    /// Raw fuel quota (offset 32).
    pub quota: i64,
    /// Named-grant list pointer (offset 40).
    pub grants_ptr: u64,
    /// Named-grant list count (offset 48).
    pub grants_n: u64,
}

impl SpawnRec {
    /// Decode the 56-byte record. `None` when the `version` word (offset 0) is nonzero — every
    /// driver fails that closed. Pure layout decode: no handle/window/geometry validation, all of
    /// which stays tier-local.
    pub fn parse(rec: &[u8; 56]) -> Option<SpawnRec> {
        let u32_at = |o: usize| u32::from_le_bytes([rec[o], rec[o + 1], rec[o + 2], rec[o + 3]]);
        let u64_at = |o: usize| u64::from_le_bytes(rec[o..o + 8].try_into().unwrap());
        if u32_at(0) != 0 {
            return None; // version — fail closed
        }
        Some(SpawnRec {
            entry: u32_at(4),
            off: u64_at(8),
            size_log2: u32_at(16) as i64,
            pager: u32_at(20),
            modh: u32_at(24) as i32,
            budget: u32_at(28) as i32,
            quota: u64_at(32) as i64,
            grants_ptr: u64_at(40),
            grants_n: u64_at(48),
        })
    }
}

/// SSA value types. `i8`/`i16` are memory access *widths*, not value types (§3a).
/// `v128` is the fixed-128 SIMD vector (§17/D58): a first-class value carrying 16
/// raw bytes whose lane interpretation is per-op, never per-value.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum ValType {
    I32,
    I64,
    F32,
    F64,
    V128,
    /// An opaque 64-bit **reference** (§GC.md §6 forward-compat reservation). Reserved now so
    /// future *precise* GC (stack maps + value-location metadata) can name pointer-typed slots
    /// without a format break. Today it is a pure reservation: no instruction produces a `ref`
    /// literal, and wherever a `ref` value does flow (a `ref`-typed param/result/block-arg) it is
    /// indistinguishable from an `i64` — it lowers as `i64` in the JIT and as the opaque
    /// `Value::Ref` in the interp. Conservative GC needs none of this; it scans raw words.
    Ref,
    /// A **capability handle** slot (IMPORTS.md §3.5, wire v7 reservation). In guest code it is
    /// indistinguishable from an `i32` (the packed `(generation, slot)` handle value) and lowers
    /// exactly as `i32` everywhere. The marker exists for *boundaries*: a `cap`-declared
    /// parameter or result tells the host to translate the handle between domains (resolve in
    /// the sender's table, re-grant into the receiver's) — the guest↔guest half of "objects are
    /// arguments". Translation machinery is the recorded follow-up; today this is a pure
    /// reservation, like [`ValType::Ref`].
    Cap,
}

impl ValType {
    /// Stable text token (the text form is 1:1 with the binary, §3a).
    pub fn as_str(self) -> &'static str {
        match self {
            ValType::I32 => "i32",
            ValType::I64 => "i64",
            ValType::F32 => "f32",
            ValType::F64 => "f64",
            ValType::V128 => "v128",
            ValType::Ref => "ref",
            ValType::Cap => "cap",
        }
    }

    /// Parse a type token, if recognized.
    #[allow(clippy::should_implement_trait)] // `Option` return, not `FromStr`'s `Result`
    pub fn from_str(s: &str) -> Option<ValType> {
        Some(match s {
            "i32" => ValType::I32,
            "i64" => ValType::I64,
            "f32" => ValType::F32,
            "f64" => ValType::F64,
            "v128" => ValType::V128,
            "ref" => ValType::Ref,
            "cap" => ValType::Cap,
            _ => return None,
        })
    }
}

/// A `v128` **lane shape** (§17/D58): how a 16-byte vector is split into typed lanes
/// for one op. The shape is carried by the op, never by the `v128` value itself — the
/// same bytes are reinterpreted per instruction, exactly like hardware SIMD.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VShape {
    I8x16,
    I16x8,
    I32x4,
    I64x2,
    F32x4,
    F64x2,
}

impl VShape {
    pub const ALL: [VShape; 6] = [
        VShape::I8x16,
        VShape::I16x8,
        VShape::I32x4,
        VShape::I64x2,
        VShape::F32x4,
        VShape::F64x2,
    ];
    /// Number of lanes.
    pub fn lanes(self) -> u8 {
        match self {
            VShape::I8x16 => 16,
            VShape::I16x8 => 8,
            VShape::I32x4 | VShape::F32x4 => 4,
            VShape::I64x2 | VShape::F64x2 => 2,
        }
    }
    /// Lane width in bytes.
    pub fn lane_bytes(self) -> u32 {
        match self {
            VShape::I8x16 => 1,
            VShape::I16x8 => 2,
            VShape::I32x4 | VShape::F32x4 => 4,
            VShape::I64x2 | VShape::F64x2 => 8,
        }
    }
    /// Whether the lanes are floating-point.
    pub fn is_float(self) -> bool {
        matches!(self, VShape::F32x4 | VShape::F64x2)
    }
    /// The **scalar** value type a lane extracts to / splats from / replaces with.
    /// Narrow integer lanes (`i8`/`i16`) widen to `i32` (the lane scalar is an `i32`),
    /// matching the wasm/hardware convention.
    pub fn lane_val(self) -> ValType {
        match self {
            VShape::I8x16 | VShape::I16x8 | VShape::I32x4 => ValType::I32,
            VShape::I64x2 => ValType::I64,
            VShape::F32x4 => ValType::F32,
            VShape::F64x2 => ValType::F64,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            VShape::I8x16 => "i8x16",
            VShape::I16x8 => "i16x8",
            VShape::I32x4 => "i32x4",
            VShape::I64x2 => "i64x2",
            VShape::F32x4 => "f32x4",
            VShape::F64x2 => "f64x2",
        }
    }
    pub fn index(self) -> u8 {
        Self::ALL.iter().position(|&o| o == self).unwrap() as u8
    }
    pub fn from_index(i: u8) -> Option<VShape> {
        Self::ALL.get(i as usize).copied()
    }
    pub fn from_name(s: &str) -> Option<VShape> {
        Self::ALL.iter().copied().find(|o| o.name() == s)
    }

    /// The integer shape with **half** the lane width (and twice the lanes): `i16x8`→`i8x16`,
    /// `i32x4`→`i16x8`, `i64x2`→`i32x4`. `None` for `i8x16` and the float shapes. The source of a
    /// widen / the result of a narrow.
    pub fn narrower(self) -> Option<VShape> {
        match self {
            VShape::I16x8 => Some(VShape::I8x16),
            VShape::I32x4 => Some(VShape::I16x8),
            VShape::I64x2 => Some(VShape::I32x4),
            _ => None,
        }
    }

    /// The integer shape with **double** the lane width: `i8x16`→`i16x8`, `i16x8`→`i32x4`,
    /// `i32x4`→`i64x2`. `None` for `i64x2` and the float shapes. The result of a widen / the source
    /// of a narrow.
    pub fn wider(self) -> Option<VShape> {
        match self {
            VShape::I8x16 => Some(VShape::I16x8),
            VShape::I16x8 => Some(VShape::I32x4),
            VShape::I32x4 => Some(VShape::I64x2),
            _ => None,
        }
    }
}

/// Lane-wise binary integer ops on a `v128` (§17). Defined for every integer [`VShape`]
/// (the JIT may lower a shape to several instructions — e.g. `i64x2.mul` — but the lane
/// semantics are always total). Wrapping arithmetic; shifts take the scalar amount mod
/// the lane bit-width.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VIntBinOp {
    Add,
    Sub,
    Mul,
    MinS,
    MinU,
    MaxS,
    MaxU,
}

impl VIntBinOp {
    pub const ALL: [VIntBinOp; 7] = [
        VIntBinOp::Add,
        VIntBinOp::Sub,
        VIntBinOp::Mul,
        VIntBinOp::MinS,
        VIntBinOp::MinU,
        VIntBinOp::MaxS,
        VIntBinOp::MaxU,
    ];
    pub fn name(self) -> &'static str {
        match self {
            VIntBinOp::Add => "add",
            VIntBinOp::Sub => "sub",
            VIntBinOp::Mul => "mul",
            VIntBinOp::MinS => "min_s",
            VIntBinOp::MinU => "min_u",
            VIntBinOp::MaxS => "max_s",
            VIntBinOp::MaxU => "max_u",
        }
    }
    pub fn index(self) -> u8 {
        Self::ALL.iter().position(|&o| o == self).unwrap() as u8
    }
    pub fn from_index(i: u8) -> Option<VIntBinOp> {
        Self::ALL.get(i as usize).copied()
    }
    pub fn from_name(s: &str) -> Option<VIntBinOp> {
        Self::ALL.iter().copied().find(|o| o.name() == s)
    }
}

/// Lane-wise integer **comparison** ops on a `v128` (§17): each lane yields an all-ones (true) or
/// all-zeros (false) mask of the lane width, so the result is a `v128`. `s`/`u` select signed vs
/// unsigned lane ordering (`Eq`/`Ne` are sign-agnostic). Defined for every integer [`VShape`] — the
/// wasm spec omits unsigned `i64x2` compares, but the op set is total and the transpiler only emits
/// the shapes wasm defines.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VICmpOp {
    Eq,
    Ne,
    LtS,
    LtU,
    GtS,
    GtU,
    LeS,
    LeU,
    GeS,
    GeU,
}

impl VICmpOp {
    pub const ALL: [VICmpOp; 10] = [
        VICmpOp::Eq,
        VICmpOp::Ne,
        VICmpOp::LtS,
        VICmpOp::LtU,
        VICmpOp::GtS,
        VICmpOp::GtU,
        VICmpOp::LeS,
        VICmpOp::LeU,
        VICmpOp::GeS,
        VICmpOp::GeU,
    ];
    pub fn name(self) -> &'static str {
        match self {
            VICmpOp::Eq => "eq",
            VICmpOp::Ne => "ne",
            VICmpOp::LtS => "lt_s",
            VICmpOp::LtU => "lt_u",
            VICmpOp::GtS => "gt_s",
            VICmpOp::GtU => "gt_u",
            VICmpOp::LeS => "le_s",
            VICmpOp::LeU => "le_u",
            VICmpOp::GeS => "ge_s",
            VICmpOp::GeU => "ge_u",
        }
    }
    pub fn index(self) -> u8 {
        Self::ALL.iter().position(|&o| o == self).unwrap() as u8
    }
    pub fn from_index(i: u8) -> Option<VICmpOp> {
        Self::ALL.get(i as usize).copied()
    }
    pub fn from_name(s: &str) -> Option<VICmpOp> {
        Self::ALL.iter().copied().find(|o| o.name() == s)
    }
}

/// Lane-wise **float** comparison ops on a `v128` (§17): each lane yields an all-ones (true) or
/// all-zeros (false) mask of the lane width → result `v128`. Defined for the float [`VShape`]s
/// (`f32x4`/`f64x2`). `eq`/`lt`/`gt`/`le`/`ge` are **ordered** (a NaN operand ⇒ false); `ne` is the
/// **unordered** negation (a NaN operand ⇒ true) — exactly the wasm (and Rust `==`/`!=`/`<`/…) rule.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VFCmpOp {
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
}

impl VFCmpOp {
    pub const ALL: [VFCmpOp; 6] = [
        VFCmpOp::Eq,
        VFCmpOp::Ne,
        VFCmpOp::Lt,
        VFCmpOp::Gt,
        VFCmpOp::Le,
        VFCmpOp::Ge,
    ];
    pub fn name(self) -> &'static str {
        match self {
            VFCmpOp::Eq => "eq",
            VFCmpOp::Ne => "ne",
            VFCmpOp::Lt => "lt",
            VFCmpOp::Gt => "gt",
            VFCmpOp::Le => "le",
            VFCmpOp::Ge => "ge",
        }
    }
    pub fn index(self) -> u8 {
        Self::ALL.iter().position(|&o| o == self).unwrap() as u8
    }
    pub fn from_index(i: u8) -> Option<VFCmpOp> {
        Self::ALL.get(i as usize).copied()
    }
    pub fn from_name(s: &str) -> Option<VFCmpOp> {
        Self::ALL.iter().copied().find(|o| o.name() == s)
    }
}

/// Lane-wise integer **shift** ops on a `v128` (§17): every lane is shifted by the **same** scalar
/// `i32` amount, taken **modulo the lane bit-width** (the wasm rule). `ShrS` is arithmetic
/// (sign-replicating); `Shl`/`ShrU` are logical. Defined for every integer [`VShape`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VShiftOp {
    Shl,
    ShrS,
    ShrU,
}

impl VShiftOp {
    pub const ALL: [VShiftOp; 3] = [VShiftOp::Shl, VShiftOp::ShrS, VShiftOp::ShrU];
    pub fn name(self) -> &'static str {
        match self {
            VShiftOp::Shl => "shl",
            VShiftOp::ShrS => "shr_s",
            VShiftOp::ShrU => "shr_u",
        }
    }
    pub fn index(self) -> u8 {
        Self::ALL.iter().position(|&o| o == self).unwrap() as u8
    }
    pub fn from_index(i: u8) -> Option<VShiftOp> {
        Self::ALL.get(i as usize).copied()
    }
    pub fn from_name(s: &str) -> Option<VShiftOp> {
        Self::ALL.iter().copied().find(|o| o.name() == s)
    }
}

/// Lane-wise **unary** integer ops on a `v128` (§17): `Abs` (`|x|`, two's-complement, so
/// `abs(INT_MIN) == INT_MIN`, the wasm/hardware wrap) and `Neg` (`0 - x`, wrapping). `a`/result are
/// `v128`. Defined for every integer [`VShape`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VIntUnOp {
    Abs,
    Neg,
}

impl VIntUnOp {
    pub const ALL: [VIntUnOp; 2] = [VIntUnOp::Abs, VIntUnOp::Neg];
    pub fn name(self) -> &'static str {
        match self {
            VIntUnOp::Abs => "abs",
            VIntUnOp::Neg => "neg",
        }
    }
    pub fn index(self) -> u8 {
        Self::ALL.iter().position(|&o| o == self).unwrap() as u8
    }
    pub fn from_index(i: u8) -> Option<VIntUnOp> {
        Self::ALL.get(i as usize).copied()
    }
    pub fn from_name(s: &str) -> Option<VIntUnOp> {
        Self::ALL.iter().copied().find(|o| o.name() == s)
    }
}

/// Lane-wise **saturating** add/sub on a `v128` (§17): a lane that would overflow clamps to the
/// lane's signed/unsigned min or max instead of wrapping. Defined **only for `i8x16`/`i16x8`** (the
/// wasm spec has no wider saturating add/sub) — the verifier rejects any other [`VShape`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VSatBinOp {
    AddS,
    AddU,
    SubS,
    SubU,
}

impl VSatBinOp {
    pub const ALL: [VSatBinOp; 4] = [
        VSatBinOp::AddS,
        VSatBinOp::AddU,
        VSatBinOp::SubS,
        VSatBinOp::SubU,
    ];
    pub fn name(self) -> &'static str {
        match self {
            VSatBinOp::AddS => "add_sat_s",
            VSatBinOp::AddU => "add_sat_u",
            VSatBinOp::SubS => "sub_sat_s",
            VSatBinOp::SubU => "sub_sat_u",
        }
    }
    pub fn index(self) -> u8 {
        Self::ALL.iter().position(|&o| o == self).unwrap() as u8
    }
    pub fn from_index(i: u8) -> Option<VSatBinOp> {
        Self::ALL.get(i as usize).copied()
    }
    pub fn from_name(s: &str) -> Option<VSatBinOp> {
        Self::ALL.iter().copied().find(|o| o.name() == s)
    }
}

/// Lane **widening** (`extend`): take the low or high half of the source lanes and sign/zero-extend
/// each to twice the width. The result [`VShape`] is the wider one; the source is its [`VShape::narrower`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VWidenOp {
    LowS,
    LowU,
    HighS,
    HighU,
}

impl VWidenOp {
    pub const ALL: [VWidenOp; 4] = [
        VWidenOp::LowS,
        VWidenOp::LowU,
        VWidenOp::HighS,
        VWidenOp::HighU,
    ];
    pub fn name(self) -> &'static str {
        match self {
            VWidenOp::LowS => "extend_low_s",
            VWidenOp::LowU => "extend_low_u",
            VWidenOp::HighS => "extend_high_s",
            VWidenOp::HighU => "extend_high_u",
        }
    }
    /// `(low_half, signed)`.
    pub fn parts(self) -> (bool, bool) {
        match self {
            VWidenOp::LowS => (true, true),
            VWidenOp::LowU => (true, false),
            VWidenOp::HighS => (false, true),
            VWidenOp::HighU => (false, false),
        }
    }
    pub fn index(self) -> u8 {
        Self::ALL.iter().position(|&o| o == self).unwrap() as u8
    }
    pub fn from_index(i: u8) -> Option<VWidenOp> {
        Self::ALL.get(i as usize).copied()
    }
    pub fn from_name(s: &str) -> Option<VWidenOp> {
        Self::ALL.iter().copied().find(|o| o.name() == s)
    }
}

/// Lane **narrowing**: take two source vectors (each the wider shape), saturate every lane to the
/// narrow width, and concatenate (`a`'s lanes then `b`'s). `S`/`U` pick the *saturation* range; the
/// source is always read as **signed** (the wasm rule). `i8x16`/`i16x8` results only.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VNarrowOp {
    S,
    U,
}

impl VNarrowOp {
    pub const ALL: [VNarrowOp; 2] = [VNarrowOp::S, VNarrowOp::U];
    pub fn name(self) -> &'static str {
        match self {
            VNarrowOp::S => "narrow_s",
            VNarrowOp::U => "narrow_u",
        }
    }
    pub fn index(self) -> u8 {
        Self::ALL.iter().position(|&o| o == self).unwrap() as u8
    }
    pub fn from_index(i: u8) -> Option<VNarrowOp> {
        Self::ALL.get(i as usize).copied()
    }
    pub fn from_name(s: &str) -> Option<VNarrowOp> {
        Self::ALL.iter().copied().find(|o| o.name() == s)
    }
}

/// Lane **int↔float / float↔float conversions** (§17). Each is a whole-instruction mnemonic (the
/// source and result lane shapes differ, so unlike the lane-op families these don't share a
/// `shape.suffix` form). `a`/result are `v128`. `trunc_sat` is the non-trapping float→int (NaN→0,
/// clamp to the integer range); `demote`/`promote` change float width (low 2 lanes, high zeroed).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VCvtOp {
    /// `f32x4.convert_i32x4_s`: each `i32` lane → `f32`.
    F32x4ConvertI32x4S,
    /// `f32x4.convert_i32x4_u`: each `u32` lane → `f32`.
    F32x4ConvertI32x4U,
    /// `i32x4.trunc_sat_f32x4_s`: each `f32` lane → saturating `i32`.
    I32x4TruncSatF32x4S,
    /// `i32x4.trunc_sat_f32x4_u`: each `f32` lane → saturating `u32`.
    I32x4TruncSatF32x4U,
    /// `f32x4.demote_f64x2_zero`: the two `f64` lanes → `f32` (lanes 0/1); lanes 2/3 = 0.
    F32x4DemoteF64x2Zero,
    /// `f64x2.promote_low_f32x4`: the low two `f32` lanes → `f64`.
    F64x2PromoteLowF32x4,
    /// `f64x2.convert_low_i32x4_s`: the low two `i32` lanes → `f64` (lanes 0/1).
    F64x2ConvertLowI32x4S,
    /// `f64x2.convert_low_i32x4_u`: the low two `u32` lanes → `f64` (lanes 0/1).
    F64x2ConvertLowI32x4U,
    /// `i32x4.trunc_sat_f64x2_s_zero`: the two `f64` lanes → saturating `i32` (lanes 0/1); 2/3 = 0.
    I32x4TruncSatF64x2SZero,
    /// `i32x4.trunc_sat_f64x2_u_zero`: the two `f64` lanes → saturating `u32` (lanes 0/1); 2/3 = 0.
    I32x4TruncSatF64x2UZero,
}

impl VCvtOp {
    pub const ALL: [VCvtOp; 10] = [
        VCvtOp::F32x4ConvertI32x4S,
        VCvtOp::F32x4ConvertI32x4U,
        VCvtOp::I32x4TruncSatF32x4S,
        VCvtOp::I32x4TruncSatF32x4U,
        VCvtOp::F32x4DemoteF64x2Zero,
        VCvtOp::F64x2PromoteLowF32x4,
        VCvtOp::F64x2ConvertLowI32x4S,
        VCvtOp::F64x2ConvertLowI32x4U,
        VCvtOp::I32x4TruncSatF64x2SZero,
        VCvtOp::I32x4TruncSatF64x2UZero,
    ];
    pub fn name(self) -> &'static str {
        match self {
            VCvtOp::F32x4ConvertI32x4S => "f32x4.convert_i32x4_s",
            VCvtOp::F32x4ConvertI32x4U => "f32x4.convert_i32x4_u",
            VCvtOp::I32x4TruncSatF32x4S => "i32x4.trunc_sat_f32x4_s",
            VCvtOp::I32x4TruncSatF32x4U => "i32x4.trunc_sat_f32x4_u",
            VCvtOp::F32x4DemoteF64x2Zero => "f32x4.demote_f64x2_zero",
            VCvtOp::F64x2PromoteLowF32x4 => "f64x2.promote_low_f32x4",
            VCvtOp::F64x2ConvertLowI32x4S => "f64x2.convert_low_i32x4_s",
            VCvtOp::F64x2ConvertLowI32x4U => "f64x2.convert_low_i32x4_u",
            VCvtOp::I32x4TruncSatF64x2SZero => "i32x4.trunc_sat_f64x2_s_zero",
            VCvtOp::I32x4TruncSatF64x2UZero => "i32x4.trunc_sat_f64x2_u_zero",
        }
    }
    pub fn index(self) -> u8 {
        Self::ALL.iter().position(|&o| o == self).unwrap() as u8
    }
    pub fn from_index(i: u8) -> Option<VCvtOp> {
        Self::ALL.get(i as usize).copied()
    }
    pub fn from_name(s: &str) -> Option<VCvtOp> {
        Self::ALL.iter().copied().find(|o| o.name() == s)
    }
}

/// Lane-wise **pseudo** min/max on a float `v128` (§17). Unlike the IEEE [`VFloatBinOp::Min`]/`Max`,
/// these are the wasm `pmin`/`pmax`: a plain compare-and-select — `pmin(a,b) = b < a ? b : a`,
/// `pmax(a,b) = a < b ? b : a` — so a NaN operand (and `±0`) follow the select, not IEEE rules.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VPMinMaxOp {
    Pmin,
    Pmax,
}

impl VPMinMaxOp {
    pub const ALL: [VPMinMaxOp; 2] = [VPMinMaxOp::Pmin, VPMinMaxOp::Pmax];
    pub fn name(self) -> &'static str {
        match self {
            VPMinMaxOp::Pmin => "pmin",
            VPMinMaxOp::Pmax => "pmax",
        }
    }
    pub fn index(self) -> u8 {
        Self::ALL.iter().position(|&o| o == self).unwrap() as u8
    }
    pub fn from_index(i: u8) -> Option<VPMinMaxOp> {
        Self::ALL.get(i as usize).copied()
    }
    pub fn from_name(s: &str) -> Option<VPMinMaxOp> {
        Self::ALL.iter().copied().find(|o| o.name() == s)
    }
}

/// Lane-wise binary float ops on a `v128` (§17, IEEE 754, no traps). `Min`/`Max` are the
/// IEEE `minimum`/`maximum` (NaN-propagating, `-0 < +0`) matching the scalar [`FBinOp`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VFloatBinOp {
    Add,
    Sub,
    Mul,
    Div,
    Min,
    Max,
}

impl VFloatBinOp {
    pub const ALL: [VFloatBinOp; 6] = [
        VFloatBinOp::Add,
        VFloatBinOp::Sub,
        VFloatBinOp::Mul,
        VFloatBinOp::Div,
        VFloatBinOp::Min,
        VFloatBinOp::Max,
    ];
    pub fn name(self) -> &'static str {
        match self {
            VFloatBinOp::Add => "add",
            VFloatBinOp::Sub => "sub",
            VFloatBinOp::Mul => "mul",
            VFloatBinOp::Div => "div",
            VFloatBinOp::Min => "min",
            VFloatBinOp::Max => "max",
        }
    }
    pub fn index(self) -> u8 {
        Self::ALL.iter().position(|&o| o == self).unwrap() as u8
    }
    pub fn from_index(i: u8) -> Option<VFloatBinOp> {
        Self::ALL.get(i as usize).copied()
    }
    pub fn from_name(s: &str) -> Option<VFloatBinOp> {
        Self::ALL.iter().copied().find(|o| o.name() == s)
    }
}

/// Lane-wise unary float ops on a `v128` (§17, IEEE 754, no traps).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VFloatUnOp {
    Abs,
    Neg,
    Sqrt,
    Ceil,
    Floor,
    Trunc,
    Nearest,
}

impl VFloatUnOp {
    // Appended (not reordered) so the binary `index` of Abs/Neg/Sqrt stays stable.
    pub const ALL: [VFloatUnOp; 7] = [
        VFloatUnOp::Abs,
        VFloatUnOp::Neg,
        VFloatUnOp::Sqrt,
        VFloatUnOp::Ceil,
        VFloatUnOp::Floor,
        VFloatUnOp::Trunc,
        VFloatUnOp::Nearest,
    ];
    pub fn name(self) -> &'static str {
        match self {
            VFloatUnOp::Abs => "abs",
            VFloatUnOp::Neg => "neg",
            VFloatUnOp::Sqrt => "sqrt",
            VFloatUnOp::Ceil => "ceil",
            VFloatUnOp::Floor => "floor",
            VFloatUnOp::Trunc => "trunc",
            VFloatUnOp::Nearest => "nearest",
        }
    }
    pub fn index(self) -> u8 {
        Self::ALL.iter().position(|&o| o == self).unwrap() as u8
    }
    pub fn from_index(i: u8) -> Option<VFloatUnOp> {
        Self::ALL.get(i as usize).copied()
    }
    pub fn from_name(s: &str) -> Option<VFloatUnOp> {
        Self::ALL.iter().copied().find(|o| o.name() == s)
    }
}

/// Whole-vector bitwise binary ops on a `v128` (§17). Shape-agnostic — they operate on
/// all 128 bits regardless of lane interpretation. `AndNot` is `a & !b`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VBitBinOp {
    And,
    Or,
    Xor,
    AndNot,
}

impl VBitBinOp {
    pub const ALL: [VBitBinOp; 4] = [
        VBitBinOp::And,
        VBitBinOp::Or,
        VBitBinOp::Xor,
        VBitBinOp::AndNot,
    ];
    pub fn name(self) -> &'static str {
        match self {
            VBitBinOp::And => "and",
            VBitBinOp::Or => "or",
            VBitBinOp::Xor => "xor",
            VBitBinOp::AndNot => "andnot",
        }
    }
    pub fn index(self) -> u8 {
        Self::ALL.iter().position(|&o| o == self).unwrap() as u8
    }
    pub fn from_index(i: u8) -> Option<VBitBinOp> {
        Self::ALL.get(i as usize).copied()
    }
    pub fn from_name(s: &str) -> Option<VBitBinOp> {
        Self::ALL.iter().copied().find(|o| o.name() == s)
    }
}

/// The integer width an op operates at. Maps to the `i32`/`i64` text prefix.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum IntTy {
    I32,
    I64,
}

impl IntTy {
    pub fn val(self) -> ValType {
        match self {
            IntTy::I32 => ValType::I32,
            IntTy::I64 => ValType::I64,
        }
    }
    pub fn prefix(self) -> &'static str {
        match self {
            IntTy::I32 => "i32",
            IntTy::I64 => "i64",
        }
    }
}

/// Binary integer ops (same type in, same type out). Wrapping arithmetic; `div`/`rem`
/// trap on `/0` and on `INT_MIN/-1` (signed); shifts take the amount mod bitwidth.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    DivS,
    DivU,
    RemS,
    RemU,
    And,
    Or,
    Xor,
    Shl,
    ShrS,
    ShrU,
    Rotl,
    Rotr,
}

impl BinOp {
    pub const ALL: [BinOp; 15] = [
        BinOp::Add,
        BinOp::Sub,
        BinOp::Mul,
        BinOp::DivS,
        BinOp::DivU,
        BinOp::RemS,
        BinOp::RemU,
        BinOp::And,
        BinOp::Or,
        BinOp::Xor,
        BinOp::Shl,
        BinOp::ShrS,
        BinOp::ShrU,
        BinOp::Rotl,
        BinOp::Rotr,
    ];

    pub fn name(self) -> &'static str {
        match self {
            BinOp::Add => "add",
            BinOp::Sub => "sub",
            BinOp::Mul => "mul",
            BinOp::DivS => "div_s",
            BinOp::DivU => "div_u",
            BinOp::RemS => "rem_s",
            BinOp::RemU => "rem_u",
            BinOp::And => "and",
            BinOp::Or => "or",
            BinOp::Xor => "xor",
            BinOp::Shl => "shl",
            BinOp::ShrS => "shr_s",
            BinOp::ShrU => "shr_u",
            BinOp::Rotl => "rotl",
            BinOp::Rotr => "rotr",
        }
    }

    pub fn index(self) -> u8 {
        Self::ALL.iter().position(|&o| o == self).unwrap() as u8
    }
    pub fn from_index(i: u8) -> Option<BinOp> {
        Self::ALL.get(i as usize).copied()
    }
    pub fn from_name(s: &str) -> Option<BinOp> {
        Self::ALL.iter().copied().find(|o| o.name() == s)
    }
}

/// Integer comparisons (same type in, `i32` 0/1 out).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CmpOp {
    Eq,
    Ne,
    LtS,
    LtU,
    LeS,
    LeU,
    GtS,
    GtU,
    GeS,
    GeU,
}

impl CmpOp {
    pub const ALL: [CmpOp; 10] = [
        CmpOp::Eq,
        CmpOp::Ne,
        CmpOp::LtS,
        CmpOp::LtU,
        CmpOp::LeS,
        CmpOp::LeU,
        CmpOp::GtS,
        CmpOp::GtU,
        CmpOp::GeS,
        CmpOp::GeU,
    ];

    pub fn name(self) -> &'static str {
        match self {
            CmpOp::Eq => "eq",
            CmpOp::Ne => "ne",
            CmpOp::LtS => "lt_s",
            CmpOp::LtU => "lt_u",
            CmpOp::LeS => "le_s",
            CmpOp::LeU => "le_u",
            CmpOp::GtS => "gt_s",
            CmpOp::GtU => "gt_u",
            CmpOp::GeS => "ge_s",
            CmpOp::GeU => "ge_u",
        }
    }

    pub fn index(self) -> u8 {
        Self::ALL.iter().position(|&o| o == self).unwrap() as u8
    }
    pub fn from_index(i: u8) -> Option<CmpOp> {
        Self::ALL.get(i as usize).copied()
    }
    pub fn from_name(s: &str) -> Option<CmpOp> {
        Self::ALL.iter().copied().find(|o| o.name() == s)
    }
}

/// Unary integer ops (same type in and out). `clz`/`ctz`/`popcnt` are bit counts;
/// `extendN_s` sign-extends the low N bits. (`extend32_s` on `i32` is the identity.)
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum IntUnOp {
    Clz,
    Ctz,
    Popcnt,
    Extend8S,
    Extend16S,
    Extend32S,
}

impl IntUnOp {
    pub const ALL: [IntUnOp; 6] = [
        IntUnOp::Clz,
        IntUnOp::Ctz,
        IntUnOp::Popcnt,
        IntUnOp::Extend8S,
        IntUnOp::Extend16S,
        IntUnOp::Extend32S,
    ];
    pub fn name(self) -> &'static str {
        match self {
            IntUnOp::Clz => "clz",
            IntUnOp::Ctz => "ctz",
            IntUnOp::Popcnt => "popcnt",
            IntUnOp::Extend8S => "extend8_s",
            IntUnOp::Extend16S => "extend16_s",
            IntUnOp::Extend32S => "extend32_s",
        }
    }
    pub fn index(self) -> u8 {
        Self::ALL.iter().position(|&o| o == self).unwrap() as u8
    }
    pub fn from_index(i: u8) -> Option<IntUnOp> {
        Self::ALL.get(i as usize).copied()
    }
    pub fn from_name(s: &str) -> Option<IntUnOp> {
        Self::ALL.iter().copied().find(|o| o.name() == s)
    }
}

/// Width-changing integer conversions between `i32` and `i64`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ConvOp {
    /// `i64.extend_i32_s`: sign-extend `i32` → `i64`.
    ExtendI32S,
    /// `i64.extend_i32_u`: zero-extend `i32` → `i64`.
    ExtendI32U,
    /// `i32.wrap_i64`: truncate `i64` → `i32`.
    WrapI64,
}

impl ConvOp {
    /// `(text name, source type, result type)`.
    pub fn sig(self) -> (&'static str, ValType, ValType) {
        match self {
            ConvOp::ExtendI32S => ("i64.extend_i32_s", ValType::I32, ValType::I64),
            ConvOp::ExtendI32U => ("i64.extend_i32_u", ValType::I32, ValType::I64),
            ConvOp::WrapI64 => ("i32.wrap_i64", ValType::I64, ValType::I32),
        }
    }
    pub fn from_name(s: &str) -> Option<ConvOp> {
        [ConvOp::ExtendI32S, ConvOp::ExtendI32U, ConvOp::WrapI64]
            .into_iter()
            .find(|o| o.sig().0 == s)
    }
}

/// Float width. Maps to the `f32`/`f64` text prefix.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FloatTy {
    F32,
    F64,
}

impl FloatTy {
    pub fn val(self) -> ValType {
        match self {
            FloatTy::F32 => ValType::F32,
            FloatTy::F64 => ValType::F64,
        }
    }
    pub fn prefix(self) -> &'static str {
        match self {
            FloatTy::F32 => "f32",
            FloatTy::F64 => "f64",
        }
    }
}

/// Binary float ops (IEEE 754, no traps; same type in and out).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FBinOp {
    Add,
    Sub,
    Mul,
    Div,
    Min,
    Max,
    Copysign,
}

impl FBinOp {
    pub const ALL: [FBinOp; 7] = [
        FBinOp::Add,
        FBinOp::Sub,
        FBinOp::Mul,
        FBinOp::Div,
        FBinOp::Min,
        FBinOp::Max,
        FBinOp::Copysign,
    ];
    pub fn name(self) -> &'static str {
        match self {
            FBinOp::Add => "add",
            FBinOp::Sub => "sub",
            FBinOp::Mul => "mul",
            FBinOp::Div => "div",
            FBinOp::Min => "min",
            FBinOp::Max => "max",
            FBinOp::Copysign => "copysign",
        }
    }
    pub fn index(self) -> u8 {
        Self::ALL.iter().position(|&o| o == self).unwrap() as u8
    }
    pub fn from_index(i: u8) -> Option<FBinOp> {
        Self::ALL.get(i as usize).copied()
    }
    pub fn from_name(s: &str) -> Option<FBinOp> {
        Self::ALL.iter().copied().find(|o| o.name() == s)
    }
}

/// Unary float ops (IEEE 754, no traps).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FUnOp {
    Abs,
    Neg,
    Sqrt,
    Ceil,
    Floor,
    Trunc,
    Nearest,
}

impl FUnOp {
    pub const ALL: [FUnOp; 7] = [
        FUnOp::Abs,
        FUnOp::Neg,
        FUnOp::Sqrt,
        FUnOp::Ceil,
        FUnOp::Floor,
        FUnOp::Trunc,
        FUnOp::Nearest,
    ];
    pub fn name(self) -> &'static str {
        match self {
            FUnOp::Abs => "abs",
            FUnOp::Neg => "neg",
            FUnOp::Sqrt => "sqrt",
            FUnOp::Ceil => "ceil",
            FUnOp::Floor => "floor",
            FUnOp::Trunc => "trunc",
            FUnOp::Nearest => "nearest",
        }
    }
    pub fn index(self) -> u8 {
        Self::ALL.iter().position(|&o| o == self).unwrap() as u8
    }
    pub fn from_index(i: u8) -> Option<FUnOp> {
        Self::ALL.get(i as usize).copied()
    }
    pub fn from_name(s: &str) -> Option<FUnOp> {
        Self::ALL.iter().copied().find(|o| o.name() == s)
    }
}

/// Float comparisons (same type in, `i32` 0/1 out).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FCmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl FCmpOp {
    pub const ALL: [FCmpOp; 6] = [
        FCmpOp::Eq,
        FCmpOp::Ne,
        FCmpOp::Lt,
        FCmpOp::Le,
        FCmpOp::Gt,
        FCmpOp::Ge,
    ];
    pub fn name(self) -> &'static str {
        match self {
            FCmpOp::Eq => "eq",
            FCmpOp::Ne => "ne",
            FCmpOp::Lt => "lt",
            FCmpOp::Le => "le",
            FCmpOp::Gt => "gt",
            FCmpOp::Ge => "ge",
        }
    }
    pub fn index(self) -> u8 {
        Self::ALL.iter().position(|&o| o == self).unwrap() as u8
    }
    pub fn from_index(i: u8) -> Option<FCmpOp> {
        Self::ALL.get(i as usize).copied()
    }
    pub fn from_name(s: &str) -> Option<FCmpOp> {
        Self::ALL.iter().copied().find(|o| o.name() == s)
    }
}

/// Saturating float→int conversions (`trunc_sat`): NaN→0, out-of-range saturates.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FToI {
    F32I32S,
    F32I32U,
    F32I64S,
    F32I64U,
    F64I32S,
    F64I32U,
    F64I64S,
    F64I64U,
}

impl FToI {
    pub const ALL: [FToI; 8] = [
        FToI::F32I32S,
        FToI::F32I32U,
        FToI::F32I64S,
        FToI::F32I64U,
        FToI::F64I32S,
        FToI::F64I32U,
        FToI::F64I64S,
        FToI::F64I64U,
    ];
    /// `(from float, to int, signed)`.
    pub fn parts(self) -> (FloatTy, IntTy, bool) {
        match self {
            FToI::F32I32S => (FloatTy::F32, IntTy::I32, true),
            FToI::F32I32U => (FloatTy::F32, IntTy::I32, false),
            FToI::F32I64S => (FloatTy::F32, IntTy::I64, true),
            FToI::F32I64U => (FloatTy::F32, IntTy::I64, false),
            FToI::F64I32S => (FloatTy::F64, IntTy::I32, true),
            FToI::F64I32U => (FloatTy::F64, IntTy::I32, false),
            FToI::F64I64S => (FloatTy::F64, IntTy::I64, true),
            FToI::F64I64U => (FloatTy::F64, IntTy::I64, false),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            FToI::F32I32S => "i32.trunc_sat_f32_s",
            FToI::F32I32U => "i32.trunc_sat_f32_u",
            FToI::F32I64S => "i64.trunc_sat_f32_s",
            FToI::F32I64U => "i64.trunc_sat_f32_u",
            FToI::F64I32S => "i32.trunc_sat_f64_s",
            FToI::F64I32U => "i32.trunc_sat_f64_u",
            FToI::F64I64S => "i64.trunc_sat_f64_s",
            FToI::F64I64U => "i64.trunc_sat_f64_u",
        }
    }
    pub fn index(self) -> u8 {
        Self::ALL.iter().position(|&o| o == self).unwrap() as u8
    }
    pub fn from_index(i: u8) -> Option<FToI> {
        Self::ALL.get(i as usize).copied()
    }
    pub fn from_name(s: &str) -> Option<FToI> {
        Self::ALL.iter().copied().find(|o| o.name() == s)
    }
    /// The **trapping** spelling (`trunc`, no `_sat`) of the same conversion — NaN
    /// and out-of-range inputs trap instead of saturating.
    pub fn trap_name(self) -> &'static str {
        match self {
            FToI::F32I32S => "i32.trunc_f32_s",
            FToI::F32I32U => "i32.trunc_f32_u",
            FToI::F32I64S => "i64.trunc_f32_s",
            FToI::F32I64U => "i64.trunc_f32_u",
            FToI::F64I32S => "i32.trunc_f64_s",
            FToI::F64I32U => "i32.trunc_f64_u",
            FToI::F64I64S => "i64.trunc_f64_s",
            FToI::F64I64U => "i64.trunc_f64_u",
        }
    }
    pub fn from_trap_name(s: &str) -> Option<FToI> {
        Self::ALL.iter().copied().find(|o| o.trap_name() == s)
    }
}

/// Int→float conversions (`convert`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum IToF {
    I32F32S,
    I32F32U,
    I64F32S,
    I64F32U,
    I32F64S,
    I32F64U,
    I64F64S,
    I64F64U,
}

impl IToF {
    pub const ALL: [IToF; 8] = [
        IToF::I32F32S,
        IToF::I32F32U,
        IToF::I64F32S,
        IToF::I64F32U,
        IToF::I32F64S,
        IToF::I32F64U,
        IToF::I64F64S,
        IToF::I64F64U,
    ];
    /// `(from int, to float, signed)`.
    pub fn parts(self) -> (IntTy, FloatTy, bool) {
        match self {
            IToF::I32F32S => (IntTy::I32, FloatTy::F32, true),
            IToF::I32F32U => (IntTy::I32, FloatTy::F32, false),
            IToF::I64F32S => (IntTy::I64, FloatTy::F32, true),
            IToF::I64F32U => (IntTy::I64, FloatTy::F32, false),
            IToF::I32F64S => (IntTy::I32, FloatTy::F64, true),
            IToF::I32F64U => (IntTy::I32, FloatTy::F64, false),
            IToF::I64F64S => (IntTy::I64, FloatTy::F64, true),
            IToF::I64F64U => (IntTy::I64, FloatTy::F64, false),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            IToF::I32F32S => "f32.convert_i32_s",
            IToF::I32F32U => "f32.convert_i32_u",
            IToF::I64F32S => "f32.convert_i64_s",
            IToF::I64F32U => "f32.convert_i64_u",
            IToF::I32F64S => "f64.convert_i32_s",
            IToF::I32F64U => "f64.convert_i32_u",
            IToF::I64F64S => "f64.convert_i64_s",
            IToF::I64F64U => "f64.convert_i64_u",
        }
    }
    pub fn index(self) -> u8 {
        Self::ALL.iter().position(|&o| o == self).unwrap() as u8
    }
    pub fn from_index(i: u8) -> Option<IToF> {
        Self::ALL.get(i as usize).copied()
    }
    pub fn from_name(s: &str) -> Option<IToF> {
        Self::ALL.iter().copied().find(|o| o.name() == s)
    }
}

/// Float width-change (`demote`/`promote`) and bit-`reinterpret` casts.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CastOp {
    Demote,  // f64 -> f32
    Promote, // f32 -> f64
    ReinterpI32F32,
    ReinterpF32I32,
    ReinterpI64F64,
    ReinterpF64I64,
}

impl CastOp {
    pub const ALL: [CastOp; 6] = [
        CastOp::Demote,
        CastOp::Promote,
        CastOp::ReinterpI32F32,
        CastOp::ReinterpF32I32,
        CastOp::ReinterpI64F64,
        CastOp::ReinterpF64I64,
    ];
    /// `(text name, source type, result type)`.
    pub fn sig(self) -> (&'static str, ValType, ValType) {
        match self {
            CastOp::Demote => ("f32.demote_f64", ValType::F64, ValType::F32),
            CastOp::Promote => ("f64.promote_f32", ValType::F32, ValType::F64),
            CastOp::ReinterpI32F32 => ("f32.reinterpret_i32", ValType::I32, ValType::F32),
            CastOp::ReinterpF32I32 => ("i32.reinterpret_f32", ValType::F32, ValType::I32),
            CastOp::ReinterpI64F64 => ("f64.reinterpret_i64", ValType::I64, ValType::F64),
            CastOp::ReinterpF64I64 => ("i64.reinterpret_f64", ValType::F64, ValType::I64),
        }
    }
    pub fn index(self) -> u8 {
        Self::ALL.iter().position(|&o| o == self).unwrap() as u8
    }
    pub fn from_index(i: u8) -> Option<CastOp> {
        Self::ALL.get(i as usize).copied()
    }
    pub fn from_name(s: &str) -> Option<CastOp> {
        Self::ALL.iter().copied().find(|o| o.sig().0 == s)
    }
}

/// Memory load ops. Each reads `width` little-endian bytes at the confined effective
/// address and produces `result`; narrow integer loads sign- or zero-extend per
/// `signed` into the (i32/i64) result type.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LoadOp {
    I32,
    I64,
    F32,
    F64,
    I32_8S,
    I32_8U,
    I32_16S,
    I32_16U,
    I64_8S,
    I64_8U,
    I64_16S,
    I64_16U,
    I64_32S,
    I64_32U,
}

impl LoadOp {
    pub const ALL: [LoadOp; 14] = [
        LoadOp::I32,
        LoadOp::I64,
        LoadOp::F32,
        LoadOp::F64,
        LoadOp::I32_8S,
        LoadOp::I32_8U,
        LoadOp::I32_16S,
        LoadOp::I32_16U,
        LoadOp::I64_8S,
        LoadOp::I64_8U,
        LoadOp::I64_16S,
        LoadOp::I64_16U,
        LoadOp::I64_32S,
        LoadOp::I64_32U,
    ];
    /// `(text name, result type, access width in bytes, sign-extended)`.
    pub fn info(self) -> (&'static str, ValType, u32, bool) {
        match self {
            LoadOp::I32 => ("i32.load", ValType::I32, 4, false),
            LoadOp::I64 => ("i64.load", ValType::I64, 8, false),
            LoadOp::F32 => ("f32.load", ValType::F32, 4, false),
            LoadOp::F64 => ("f64.load", ValType::F64, 8, false),
            LoadOp::I32_8S => ("i32.load8_s", ValType::I32, 1, true),
            LoadOp::I32_8U => ("i32.load8_u", ValType::I32, 1, false),
            LoadOp::I32_16S => ("i32.load16_s", ValType::I32, 2, true),
            LoadOp::I32_16U => ("i32.load16_u", ValType::I32, 2, false),
            LoadOp::I64_8S => ("i64.load8_s", ValType::I64, 1, true),
            LoadOp::I64_8U => ("i64.load8_u", ValType::I64, 1, false),
            LoadOp::I64_16S => ("i64.load16_s", ValType::I64, 2, true),
            LoadOp::I64_16U => ("i64.load16_u", ValType::I64, 2, false),
            LoadOp::I64_32S => ("i64.load32_s", ValType::I64, 4, true),
            LoadOp::I64_32U => ("i64.load32_u", ValType::I64, 4, false),
        }
    }
    pub fn index(self) -> u8 {
        Self::ALL.iter().position(|&o| o == self).unwrap() as u8
    }
    pub fn from_index(i: u8) -> Option<LoadOp> {
        Self::ALL.get(i as usize).copied()
    }
    pub fn from_name(s: &str) -> Option<LoadOp> {
        Self::ALL.iter().copied().find(|o| o.info().0 == s)
    }
}

/// Memory store ops. Each writes the low `width` little-endian bytes of the value.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StoreOp {
    I32,
    I64,
    F32,
    F64,
    I32_8,
    I32_16,
    I64_8,
    I64_16,
    I64_32,
}

impl StoreOp {
    pub const ALL: [StoreOp; 9] = [
        StoreOp::I32,
        StoreOp::I64,
        StoreOp::F32,
        StoreOp::F64,
        StoreOp::I32_8,
        StoreOp::I32_16,
        StoreOp::I64_8,
        StoreOp::I64_16,
        StoreOp::I64_32,
    ];
    /// `(text name, value type, access width in bytes)`.
    pub fn info(self) -> (&'static str, ValType, u32) {
        match self {
            StoreOp::I32 => ("i32.store", ValType::I32, 4),
            StoreOp::I64 => ("i64.store", ValType::I64, 8),
            StoreOp::F32 => ("f32.store", ValType::F32, 4),
            StoreOp::F64 => ("f64.store", ValType::F64, 8),
            StoreOp::I32_8 => ("i32.store8", ValType::I32, 1),
            StoreOp::I32_16 => ("i32.store16", ValType::I32, 2),
            StoreOp::I64_8 => ("i64.store8", ValType::I64, 1),
            StoreOp::I64_16 => ("i64.store16", ValType::I64, 2),
            StoreOp::I64_32 => ("i64.store32", ValType::I64, 4),
        }
    }
    pub fn index(self) -> u8 {
        Self::ALL.iter().position(|&o| o == self).unwrap() as u8
    }
    pub fn from_index(i: u8) -> Option<StoreOp> {
        Self::ALL.get(i as usize).copied()
    }
    pub fn from_name(s: &str) -> Option<StoreOp> {
        Self::ALL.iter().copied().find(|o| o.info().0 == s)
    }
}

/// §12 atomic read-modify-write operation. Each atomically loads the operand, applies the op with
/// the argument, stores the result, and yields the **old** value (`Xchg` just swaps the argument in).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AtomicRmwOp {
    Add,
    Sub,
    And,
    Or,
    Xor,
    Xchg,
}

impl AtomicRmwOp {
    pub const ALL: [AtomicRmwOp; 6] = [
        AtomicRmwOp::Add,
        AtomicRmwOp::Sub,
        AtomicRmwOp::And,
        AtomicRmwOp::Or,
        AtomicRmwOp::Xor,
        AtomicRmwOp::Xchg,
    ];
    /// The text suffix in `<ty>.atomic.rmw.<suffix>`.
    pub fn name(self) -> &'static str {
        match self {
            AtomicRmwOp::Add => "add",
            AtomicRmwOp::Sub => "sub",
            AtomicRmwOp::And => "and",
            AtomicRmwOp::Or => "or",
            AtomicRmwOp::Xor => "xor",
            AtomicRmwOp::Xchg => "xchg",
        }
    }
    pub fn index(self) -> u8 {
        Self::ALL.iter().position(|&o| o == self).unwrap() as u8
    }
    pub fn from_index(i: u8) -> Option<AtomicRmwOp> {
        Self::ALL.get(i as usize).copied()
    }
    pub fn from_name(s: &str) -> Option<AtomicRmwOp> {
        Self::ALL.iter().copied().find(|o| o.name() == s)
    }
}

/// C11/§12 memory ordering for atomic ops and fences. The IR carries the full lattice so a frontend
/// can express it and the verifier can reject impossible op/ordering pairs; **both backends currently
/// execute every atomic sequentially-consistent** (a sound strengthening — Cranelift atomics are
/// seq-cst only, and it keeps the interpreter↔JIT oracle exact). Honoring weaker orderings in
/// execution awaits a backend that can, and the concurrent-oracle story (§18).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Ordering {
    Relaxed,
    Acquire,
    Release,
    AcqRel,
    SeqCst,
}

impl Ordering {
    pub const ALL: [Ordering; 5] = [
        Ordering::Relaxed,
        Ordering::Acquire,
        Ordering::Release,
        Ordering::AcqRel,
        Ordering::SeqCst,
    ];
    /// The text suffix; the default [`Ordering::SeqCst`] is rendered by omitting the suffix entirely
    /// (so existing `.atomic.` text round-trips unchanged).
    pub fn name(self) -> &'static str {
        match self {
            Ordering::Relaxed => "relaxed",
            Ordering::Acquire => "acquire",
            Ordering::Release => "release",
            Ordering::AcqRel => "acqrel",
            Ordering::SeqCst => "seqcst",
        }
    }
    pub fn index(self) -> u8 {
        Self::ALL.iter().position(|&o| o == self).unwrap() as u8
    }
    pub fn from_index(i: u8) -> Option<Ordering> {
        Self::ALL.get(i as usize).copied()
    }
    pub fn from_name(s: &str) -> Option<Ordering> {
        Self::ALL.iter().copied().find(|o| o.name() == s)
    }
}

/// Non-terminator instructions. Each produces exactly one result — appended at the
/// next block-local value index — **except `Store`, which produces no value** (see
/// [`Inst::produces_value`]).
#[derive(Clone, PartialEq, Debug)]
pub enum Inst {
    ConstI32(i32),
    ConstI64(i64),
    /// Binary integer op; operands and result are `ty`.
    IntBin {
        ty: IntTy,
        op: BinOp,
        a: ValIdx,
        b: ValIdx,
    },
    /// Integer compare; operands are `ty`, result is `i32` 0/1.
    IntCmp {
        ty: IntTy,
        op: CmpOp,
        a: ValIdx,
        b: ValIdx,
    },
    /// Unary integer op; operand and result are `ty`.
    IntUn {
        ty: IntTy,
        op: IntUnOp,
        a: ValIdx,
    },
    /// `T.eqz`: 1 if the operand is zero else 0; result `i32`.
    Eqz {
        ty: IntTy,
        a: ValIdx,
    },
    /// Width conversion (see `ConvOp`).
    Convert {
        op: ConvOp,
        a: ValIdx,
    },
    /// Branchless choice: `cond` is `i32`; `a`/`b` share a type `T`; result `T`.
    Select {
        cond: ValIdx,
        a: ValIdx,
        b: ValIdx,
    },
    /// `f32`/`f64` constants, stored as raw bits for exact (NaN-safe) round-tripping.
    ConstF32(u32),
    ConstF64(u64),
    /// Binary float op; operands and result are `ty`.
    FBin {
        ty: FloatTy,
        op: FBinOp,
        a: ValIdx,
        b: ValIdx,
    },
    /// Unary float op; operand and result are `ty`.
    FUn {
        ty: FloatTy,
        op: FUnOp,
        a: ValIdx,
    },
    /// Scalar **fused multiply-add** `a·b + c` with a single rounding (IEEE-754 FMA) — the scalar
    /// sibling of [`Inst::VFma`], emitted for `llvm.fma`/`fmuladd`. Cranelift `fma` / Rust
    /// `f*::mul_add` are both correctly-rounded, so interp and JIT agree. Operands/result are `ty`.
    Fma {
        ty: FloatTy,
        a: ValIdx,
        b: ValIdx,
        c: ValIdx,
    },
    /// Float compare; operands are `ty`, result is `i32` 0/1.
    FCmp {
        ty: FloatTy,
        op: FCmpOp,
        a: ValIdx,
        b: ValIdx,
    },
    /// Saturating float→int conversion.
    FToISat {
        op: FToI,
        a: ValIdx,
    },
    /// Trapping float→int conversion: NaN or out-of-range input traps (vs the
    /// saturating [`Inst::FToISat`] default).
    FToITrap {
        op: FToI,
        a: ValIdx,
    },
    /// Int→float conversion.
    IToFConv {
        op: IToF,
        a: ValIdx,
    },
    /// `demote`/`promote`/`reinterpret` cast.
    Cast {
        op: CastOp,
        a: ValIdx,
    },
    /// Load `op`'s width from the confined effective address `addr + offset`.
    /// Unaligned access is allowed; confinement masking is implicit.
    Load {
        op: LoadOp,
        addr: ValIdx,
        offset: u64,
    },
    /// Store `value` (`op`'s width) at the confined effective address. Produces no
    /// SSA result.
    Store {
        op: StoreOp,
        addr: ValIdx,
        value: ValIdx,
        offset: u64,
    },
    /// Bulk copy `len` bytes from the confined span `[src, src+len)` to `[dst, dst+len)`
    /// (**non-overlapping**; lowered from `llvm.memcpy`). Both spans are confined **as a whole**
    /// to `[0, reserved)` — a *single* range check instead of one per byte — then a bulk copy
    /// (JIT: the platform `memcpy` libcall; interp: a checked span copy). The security hinge for
    /// memory-copy-dense kernels (D62): identical confinement to per-byte `Store` (the whole span
    /// is proven in-bounds before any access), checked once. All operands `i64`; no SSA result.
    MemCopy {
        dst: ValIdx,
        src: ValIdx,
        len: ValIdx,
    },
    /// Overlap-safe bulk copy (lowered from `llvm.memmove`): the source span is fully read before
    /// the destination is written, so overlapping `[src, src+len)`/`[dst, dst+len)` are correct.
    /// Same whole-span confinement as [`Inst::MemCopy`]. All operands `i64`; no SSA result.
    MemMove {
        dst: ValIdx,
        src: ValIdx,
        len: ValIdx,
    },
    /// Fill the confined span `[dst, dst+len)` with `val`'s low byte (lowered from `llvm.memset`).
    /// `val` is an `i32` (the fill byte, carried zero/sign-extended). Same whole-span confinement
    /// as [`Inst::MemCopy`]. No SSA result.
    MemFill {
        dst: ValIdx,
        val: ValIdx,
        len: ValIdx,
    },
    /// §12 atomic load — a naturally-aligned read of `ty` from the confined effective address
    /// `addr + offset`; a misaligned effective address **traps**. Single-threaded value semantics
    /// equal a plain [`Inst::Load`]; the distinct op makes the JIT emit a hardware atomic
    /// (sequentially consistent), so it stays correct once threads exist (§12).
    AtomicLoad {
        ty: IntTy,
        addr: ValIdx,
        offset: u64,
    },
    /// §12 atomic store — a naturally-aligned write of `value` (`ty`) to `addr + offset`; a
    /// misaligned effective address **traps**. Produces no SSA result (like [`Inst::Store`]).
    AtomicStore {
        ty: IntTy,
        addr: ValIdx,
        value: ValIdx,
        offset: u64,
    },
    /// §12 atomic read-modify-write: atomically apply `op` with `value` to `*(addr+offset)`
    /// (`ty`-wide, naturally aligned ⇒ else **traps**) and yield the **old** value.
    AtomicRmw {
        ty: IntTy,
        op: AtomicRmwOp,
        addr: ValIdx,
        value: ValIdx,
        offset: u64,
    },
    /// §12 atomic compare-exchange: if `*(addr+offset) == expected`, store `replacement`; always
    /// yield the **old** value (`ty`-wide, naturally aligned ⇒ else **traps**).
    AtomicCmpxchg {
        ty: IntTy,
        addr: ValIdx,
        expected: ValIdx,
        replacement: ValIdx,
        offset: u64,
    },
    /// Direct call to a function by index (fully static; the verifier checks the
    /// index and argument types). Appends the callee's result values — **0, 1, or
    /// many** — at the next block-local indices.
    Call {
        func: FuncIdx,
        args: Vec<ValIdx>,
    },
    /// `ref.func`: materialize a function reference — just the function index as an
    /// `i32` (a `funcref` is a forgeable integer, §3c). The verifier checks the index
    /// is in range; the *value* is plain data.
    RefFunc {
        func: FuncIdx,
    },
    /// Indirect call through the function table (§3c): mask `idx` into the table,
    /// runtime-check the selected function's signature against `ty`, then call.
    /// `idx` is an `i32` table index; results are `ty.results`.
    CallIndirect {
        ty: FuncType,
        idx: ValIdx,
        args: Vec<ValIdx>,
    },
    /// Capability call (§3c): invoke operation `op` of the interface identified by
    /// `type_id` on the capability named by `handle` — a forgeable `i32` index into
    /// the **host-owned** handle table. At this use site the index is masked into the
    /// table and the entry's `type_id`/generation are re-checked, so a forged index is
    /// **inert**: it traps (wrong type / dead generation) or selects one of this
    /// domain's own granted `type_id` capabilities — never host memory or arbitrary
    /// code (§3c). `sig` is the operation's static signature; its results are appended.
    ///
    /// Phase-1 simplification: `type_id`/`op`/`sig` are inlined immediates (mirroring
    /// `call_indirect`'s inlined `FuncType`). A module-level interface/type section —
    /// which would let the verifier also bound `op` and cross-check `sig` against the
    /// canonical interface — is deferred to §13 linking. Safety does **not** depend on
    /// it: the host-owned table's use-site checks carry it, and the host handler
    /// treats all guest inputs as hostile (§2a authority-TCB).
    CapCall {
        type_id: u32,
        op: u32,
        sig: FuncType,
        handle: ValIdx,
        args: Vec<ValIdx>,
    },
    /// Call a host capability through an **import slot** — the §7 late-binding import,
    /// reshaped by IMPORTS.md §3.5 (wire v7). `import` indexes [`Module::imports`]; `op` is
    /// the **consumer-local op index** into the import's declared interface (always `0` for a
    /// flat [`ImportShape::Func`] import) — dispatch remaps it to the provider's op through
    /// the slot's bind-time remap; `args` are the op arguments; `sig` is a self-describing
    /// copy of the op's signature (mirroring `cap.call`/`call_indirect`, so result counting
    /// needs no module context — the verifier checks it equals the type-section resolution).
    /// It deliberately carries **no** `type_id` and **no** handle operand (retired at v8):
    /// bound at instantiation through the domain's import-binding table
    /// ([`CAP_IMPORT_TYPE_ID`]) — the module bytes are never rewritten. Link-form symbolic
    /// calls are a different instruction ([`Inst::CallSym`]).
    CallImport {
        import: u32,
        op: u32,
        sig: FuncType,
        args: Vec<ValIdx>,
    },
    /// §7/§22 **link-form symbolic call** — the loader ABI's placeholder, never executable.
    /// `import` names an entry of the unit's import section (its symbol list);
    /// [`resolve_imports_with`] rewrites every `CallSym` 1:1 by what the name resolves to —
    /// [`Resolved::Func`] → a direct [`Inst::Call`] (`handle` unused), [`Resolved::Slot`] →
    /// [`Inst::CallIndirect`] (the `ConstI32` defining `handle` is patched to the table slot
    /// and `handle` becomes the index), [`Resolved::Cap`] → [`Inst::CapCall`] (`handle` is the
    /// live runtime handle value — link-form cap calls are per-call-site-handle *dynamic*
    /// dispatch; a manifest slot could never carry them). A `CallSym` that survives to
    /// verification is an unresolved symbol and is rejected unconditionally — resolve is
    /// source-to-source *before* `verify_module`, so a mis-link fails closed.
    ///
    /// CONSOLIDATION §7: the always-present `handle` operand (meaningful only for the
    /// `Resolved::Cap` outcome; dead weight for `Func`/`Slot`) is the last special case in the
    /// call encodings — the v7 `ns` field's sibling. It leaves the instruction and the wire at
    /// the **next wire rev** (whenever one happens for its own reasons); no rev is spent on it.
    CallSym {
        import: u32,
        sig: FuncType,
        handle: ValIdx,
        args: Vec<ValIdx>,
    },
    /// A **link-form data-symbol address** — the data-side analogue of [`Inst::CallSym`] (D-LINK):
    /// the window address of the exported data symbol `name`, plus `addend`, materialized as a
    /// value. Emitted by a separately-compiled unit that references a global it does **not** define
    /// (a cross-unit `extern`); [`link`] rewrites it **1:1** into a [`Inst::ConstI64`] holding the
    /// resolved address once it has placed the defining unit's data. Fail-closed: an unresolved name
    /// fails the link ([`LinkError::Unresolved`]), and a `DataSym` that *survives* into a runnable
    /// module is a verify error (an object is not executable — the same guarantee as a surviving
    /// `CallSym`). The name rides in the instruction, so instruction insertion/reordering never
    /// desyncs it — there is no position-keyed relocation table. Result is one `i64`. The name is a
    /// `Vec<u8>` (not `String`) so `Inst: Clone` monomorphizes to a `Copy`-element `Vec` clone — like
    /// `CallSym`'s `args` — and never pulls in the concrete `alloc` `String::clone`, keeping every
    /// `Inst`-manipulating crate (e.g. `svm-peval`) translatable by the strict LLVM on-ramp (the
    /// data-oriented IR invariant that keeps the partial evaluator self-hostable).
    DataSym {
        name: alloc::vec::Vec<u8>,
        addend: i64,
    },
    /// A **link-form own-data address**: this unit's assigned data base plus `offset` (a unit-local
    /// data offset), materialized as a value — the self-relative counterpart of [`Inst::DataSym`],
    /// for a reference to the unit's *own* global whose final window placement the frontend did not
    /// know at emit time (it is the linker that assigns each unit's data region). [`link`] rewrites
    /// it to [`Inst::ConstI64`]; surviving into a runnable module is a verify error. Result is `i64`.
    DataSelf {
        offset: u64,
    },
    /// A **link-form data-stack base**: the address just above *all* of the linked program's data
    /// (the post-link top-of-data, [`powerbox_entry_sp`]-aligned), materialized as a value. A
    /// separately-compiled entry unit cannot bake its data-stack pointer as a constant — its own
    /// `data_end` is only this unit's top, but the linker stacks every unit's data into one window,
    /// so the true stack base is not known until link time. A frontend emits `data.top` where a
    /// whole-program build would emit `i64.const data_end` (the `_start` data-SP, and the argv/scratch
    /// scratch it builds there). [`link`] rewrites it to [`Inst::ConstI64`] once the window layout is
    /// fixed and grows the merged window to reserve the data stack above it; surviving into a runnable
    /// module is a verify error. Result is `i64`. (Order-independent — the value is the whole
    /// program's, not any one unit's — so the entry unit still links wherever `_start` needs it.)
    DataTop,
    /// **Dynamic-mode** interface dispatch by type-section reference (IMPORTS.md §3.5): drive
    /// the capability behind a runtime `handle` value as interface `ty` (an index into
    /// [`Module::types`] naming a [`TypeEntry::Interface`]), op `op`. The use-site check is
    /// exact-id: the entry's `type_id` must equal the intern of `types[ty]` (resolved once at
    /// instantiation), plus the §3c generation check. This closes the recorded encoding gap —
    /// `cap.call` needs a compile-time `type_id` immediate a guest cannot know for a wired
    /// offer; a type-section reference is resolvable at instantiation. `sig` is the
    /// self-describing copy; the verifier checks it equals `types[ty]`'s op-`op` signature.
    /// Costs the manifest-complete bit, like every dynamic-mode site.
    CallImportDyn {
        ty: u32,
        op: u32,
        sig: FuncType,
        handle: ValIdx,
        args: Vec<ValIdx>,
    },
    /// `export.handle` (IMPORTS.md §3.5): reify **this module's own** impl export `export`
    /// (an index into [`Module::impl_exports`]) as an ordinary capability handle — the only
    /// guest-reachable source of offer wiring rights (offer exposure is consent-based: bytes
    /// are ambient, instances are consensual). All offers a domain reifies share its one
    /// service state; re-reifying the same export returns a handle to the same backing.
    /// Result is `i32` (the packed handle), or `-errno` on an out-of-range index.
    ExportHandle {
        export: u32,
    },
    /// `import.attach` (IMPORTS.md phase 2): (re)bind **rebindable** import slot `import` to the
    /// capability behind `handle` — an `i32` handle value the domain already holds (typically
    /// discovered via `cap.self.get`/`resolve`). The handle must resolve live under the slot's
    /// declared interface `type_id` (the §3c mask + type + generation check); on success the slot's
    /// binding swaps and subsequent `call.import <import>` dispatch through it. Authority-neutral
    /// (aliases a held capability into a named slot — no new grant-graph edge). The verifier checks
    /// `import` is in range and names a [`ImportMode::Rebindable`] declaration — attaching to a
    /// `Required` slot is a verify error, keeping required bindings immutable-per-instance. Result
    /// is `i32`: `0` ok, `-errno` (wrong-type / dead handle) — a guest may probe and fall back.
    ImportAttach {
        import: u32,
        handle: ValIdx,
    },
    // §7 capability **reflection** — `cap.self.count` / `.get` / `.resolve` / `.label` / `.attest`
    // are no longer first-class IR ops. Every backend lowered them to `cap.call CAP_SELF_TYPE_ID op N`
    // (op 0/1/2/3/4), so the wire rev retired the typed fronts to that generic `CapCall` form (svm-llvm
    // builds it, svm-text spells the `cap.self.*` sugar over it, and the runtime CAP_SELF handler
    // dispatches on `op`). `cap.self.type_id`/`covers` stay typed ops below: they carry a type-section
    // index, not expressible as a plain `cap.call` immediate.
    /// `cap.self.type_id <ty>` (IMPORTS.md §3.5): intern **this module's** type-section entry
    /// `ty` (a [`TypeEntry::Interface`]) in the domain's host and return the runtime `type_id`
    /// as `i32`. Authority-neutral pure reflection — the shape is already the module's own
    /// declaration. Enables shape-indexed discovery: iterate `cap.self.get`, compare ids.
    /// Out-of-range / non-interface `ty` is a verify error, not a runtime probe.
    CapSelfTypeId {
        ty: u32,
    },
    /// `cap.self.covers v<h>, <ty>` (IMPORTS.md §3.5): does the live capability behind `handle`
    /// **cover** this module's interface `types[ty]` (every required op present by name with an
    /// equal signature)? Result `i32`: `1` covers, `0` does not, `-errno` for a dead/forged
    /// handle. Authority-neutral subset discovery — the probe form of coverage binding (a
    /// failed `import.attach` is the probe without it).
    CapSelfCovers {
        handle: ValIdx,
        ty: u32,
    },
    /// §12 per-vCPU **thread-local register** read (`vcpu.tls.get`): the `i64` TLS word of the vCPU
    /// **currently executing** this op. svm carries one i64 of per-vCPU state; it is read *at the
    /// execution point*, so after a fiber migrates between vCPUs (D57: any vCPU may resume any
    /// resumable fiber) `get` returns the *new* vCPU's word — the correct per-CPU value, which the
    /// guest cannot otherwise name (the `thread.spawn` handle is the parent's view, not "which vCPU am
    /// I on now"). Seeded at vCPU creation to a **dense id** (root = 0, children sequential in spawn
    /// order), so before any `set` it doubles as a `vcpu.id`; the guest may overwrite it (e.g. a
    /// pointer to its per-CPU block) for full thread-local storage. Authority-neutral, ambient (the
    /// `cap.self`/`gc.roots` family). Result is `i64`. (Determinism: program *output* must not depend
    /// on *which* vCPU runs you, only on per-CPU state being self-consistent — GC.md §3.2.)
    VcpuTlsGet,
    /// §12 per-vCPU **thread-local register** write (`vcpu.tls.set`): set the executing vCPU's `i64`
    /// TLS word to `val`. No result (like `store`). See [`Inst::VcpuTlsGet`].
    VcpuTlsSet {
        val: ValIdx,
    },
    /// **Durable-runtime-internal** (DURABILITY.md §12.8, Phase 4 Slice A.5): read the **current
    /// durable context's shadow region base** — a window byte offset. Emitted only by the durable
    /// transform (`svm-durable`) to address that context's *own* per-context shadow-SP word, so
    /// concurrent vCPUs each spill against their own region with no shared word. Like
    /// [`Inst::VcpuTlsGet`] it is a per-OS-thread runtime register read (no window/trap context, cannot
    /// fault), but **runtime-private**: the runtime seeds it per dispatch / per child and there is no
    /// guest write op, so a guest cannot clobber it (unlike the guest-overwritable `vcpu.tls`). Result
    /// is `i64`.
    DurableShadowBase,
    /// §12 fiber create (`cont.new`): allocate a new suspended fiber that will run the
    /// function referenced by `func` on the data stack based at `sp`. `func` is an `i32`
    /// funcref, resolved through the function table with signature `(i64 sp, i64 arg) ->
    /// i64` at first resume (a bad ref traps there, like [`Inst::CallIndirect`]); `sp`
    /// (`i64`) is the fiber's own data-stack base — a fiber owns a **stack pair** (§3d): its
    /// in-window data stack (based here) plus the out-of-band control stack the runtime
    /// allocates. Yields an `i32` **fiber handle**: a forgeable index into the runtime-owned
    /// fiber table, masked + generation-checked at use like a capability handle (§3c), so a
    /// forged handle is inert (it traps or selects one of this domain's own fibers, never
    /// host state). The fiber does not run yet; the first resume calls `func(sp, arg)`.
    ContNew {
        func: ValIdx,
        sp: ValIdx,
    },
    /// §12 fiber resume (`cont.resume` / `cont.resume.block`): switch to fiber `k` (an `i32`
    /// handle), delivering `arg` (`i64`) — the argument to the fiber's function on the first
    /// resume, or the result of the fiber's `suspend` on later resumes. Runs the fiber until it
    /// suspends or returns, then yields `(status: i32, value: i64)`: `status` 0 = **suspended**
    /// (the fiber stays resumable), 1 = **returned** (the fiber is done; resuming it again
    /// traps). A **call-clobbering** control op — like a call it switches stacks, but it
    /// does not end the block.
    ///
    /// The `block: true` form (`cont.resume.block`, ISSUES.md I48) is an advisory scheduling
    /// hint — returning `FIBER_PARKED (3)` is always conforming, so a guest still loops for
    /// completion; it issues this form only when it has nothing else to run, to avoid
    /// busy-polling a lone parked fiber. After the parity commits the cooperative bytecode
    /// driver genuinely idles the resumer (`TaskState::BlockedOnFiber`, zero fuel), the
    /// wasm-JIT inherits that via its `DriveMode::InterpDriven` fold, and the Cranelift JIT
    /// parks the resumer's OS thread on `Domain.futex_cv` via the `fiber_resume_block` thunk;
    /// only the OS-thread-parallel bytecode drivers (`drive_parallel`, single-vCPU
    /// `Vcpu::run`) take the advisory `FIBER_PARKED` downgrade. `block: false` never idles.
    /// Advisory only — no new semantics, invariant 9 preserved.
    ContResume {
        k: ValIdx,
        arg: ValIdx,
        block: bool,
    },
    /// §12 fiber suspend (`suspend`): from within a running fiber, suspend back to the
    /// resumer delivering `value` (`i64`); evaluates to the `i64` `arg` of the next resume.
    /// Suspending when no fiber is running (the root computation) **traps**. Like
    /// [`Inst::ContResume`] this is a call-clobbering control op.
    Suspend {
        value: ValIdx,
    },
    /// `setjmp` (the `<setjmp.h>` non-local-jump save). Captures the **current frame's resume point**
    /// — a checkpoint of (this call-stack depth, the data-stack pointer, this frame's continuation just
    /// after the `setjmp`) — into a runtime-owned checkpoint table, writing an opaque token into the
    /// guest `jmp_buf` at byte offset `buf` (`i64`). Evaluates to `i32` **0** on the direct call; a
    /// later [`Inst::LongJmp`] re-enters the frame *here* (returns "twice") with the long-jump value.
    /// Like `cont.*` it is a **call-clobbering** control op (the live state is captured), but it does
    /// not switch stacks and falls through normally on the direct call. The `jmp_buf` token is
    /// **backend-internal** (the interpreter stores a checkpoint index) and opaque to the guest, so
    /// observable behavior matches across engines though the bytes differ; it is transient (not
    /// snapshot-portable). Lowers from the recognized external `setjmp`/`_setjmp`/`sigsetjmp` call.
    SetJmp {
        buf: ValIdx,
    },
    /// `longjmp` (the `<setjmp.h>` non-local jump). Reads the checkpoint token from the guest `jmp_buf`
    /// at byte offset `buf` (`i64`), **unwinds** the call stack back to the captured [`Inst::SetJmp`]
    /// frame (the intervening frames discarded with no per-frame work — C has no cleanups), restores the
    /// data-stack pointer, and re-enters at the `setjmp` continuation, making *that* `setjmp` evaluate
    /// to `val` (`i32`; a `0` `val` becomes `1`, per C). **Never returns** to the next instruction (a
    /// `noreturn` control op; the trailing `unreachable` is dead). A stale/forged token, or a checkpoint
    /// whose frame has already returned, **traps** (in-sandbox; §3b totality). Lowers from the
    /// recognized external `longjmp`/`siglongjmp` call.
    LongJmp {
        buf: ValIdx,
        val: ValIdx,
    },
    /// §GC (`GC.md`) **conservative root enumeration** (`gc.roots`): scan every fiber of the
    /// domain — parked fibers, resume-chain ancestors, and the calling computation's own live
    /// frames (the op is **call-clobbering**, so the caller's roots are already spilled to its
    /// control stack, exactly like `cont.resume`/`suspend`) — for candidate pointer words that
    /// — after masking each scanned word `w` with `mask` (`m = w & mask`) — fall in the half-open
    /// guest-window range `[heap_lo, heap_hi)` (`i64` window offsets). The **masked** value `m` is
    /// what's tested and emitted, letting a guest with **tagged** pointers (e.g. a tag in the high
    /// byte, `(tag << 56) | offset`) recover the bare offset; `mask = !0` reproduces the untagged
    /// behavior. Writes up to `cap` **distinct (deduplicated)** `i64`-width candidate words,
    /// ascending, into guest memory at byte offset `buf`, and yields the **total** number found
    /// (`i64`); if that exceeds `cap` the guest retries with a larger buffer (only the first `cap`
    /// are written). An **ambient introspection op** — authority-neutral like `cap.self` reflection:
    /// every candidate is an in-window word the guest's own heap already encodes, while
    /// out-of-window words (host return addresses, frame pointers, host pointers) are filtered
    /// *inside* the VM and never cross the boundary, so no host layout leaks (GC.md §3, §6).
    /// **Security — `mask` may only clear the top byte** (the low 56 bits must be all-ones:
    /// `mask | 0xFF00_0000_0000_0000 == !0`): a mask that cleared lower bits could fold a host
    /// pointer (canonical, `< 2^56`) down into the guest window and leak host-address bits past the
    /// range filter (ASLR). The constraint keeps any host word large → excluded; it's enforced
    /// statically by the verifier (constant masks) and defensively at runtime on both backends.
    /// Implemented on **both backends**: the interpreter scans its reified `Value` frames; the JIT
    /// conservatively walks the live native control stacks of its fibers (parked fibers' saved
    /// extents `[ctx, top)`, the running resume chain, and the root computation's frames). The two
    /// over-approximate differently (a sound superset of the live roots, not a matching set —
    /// GC.md §3.2). Where the stack-switch substrate is absent the JIT bails `Unsupported` and the
    /// interpreter covers it.
    GcRoots {
        heap_lo: ValIdx,
        heap_hi: ValIdx,
        /// `i64` payload mask AND-ed with each scanned word before the range test (and emitted).
        /// Constrained to top-byte-strip only (`mask | 0xFF00_0000_0000_0000 == !0`); see above.
        mask: ValIdx,
        buf: ValIdx,
        cap: ValIdx,
    },
    /// §12 thread spawn (`thread.spawn`): start a new vCPU — **one real OS thread** (1:1; the VM
    /// provides the thread + futex as *primitives*, not a scheduler — any M:N model is built by the
    /// guest runtime over `thread.spawn` + `cont.*`, D22) — running `funcs[func]` on the data stack
    /// based at `sp` (the §3d two-stack split — every vCPU owns its own in-window data stack, exactly
    /// like a fiber) with `arg`, over the **same** guest memory (anonymous `Region` bytes and §13
    /// aliases are shared; post-spawn mapping changes are thread-local for now). `func` must have the
    /// fixed thread-entry type `(i64 sp, i64 arg) -> i64` (verifier-checked) — the same signature as a
    /// fiber, so a frontend function works as-is.
    /// Yields an `i32` **thread handle**: a forgeable index into the runtime-owned thread table,
    /// masked + generation-checked at [`Inst::ThreadJoin`] like a fiber/capability handle (§3c), so a
    /// forged handle is inert (it traps).
    ThreadSpawn {
        func: FuncIdx,
        sp: ValIdx,
        arg: ValIdx,
    },
    /// §12 thread join (`thread.join`): block until the vCPU named by `handle` (an `i32` thread
    /// handle) finishes and yield its `i64` result. A forged / out-of-range / already-joined handle
    /// is inert (**traps**); if the joined vCPU itself trapped, that trap propagates here.
    ThreadJoin {
        handle: ValIdx,
    },
    /// §12 futex wait (`<ty>.atomic.wait`): if the `ty`-wide value at the confined, naturally-aligned
    /// address `addr` still equals `expected`, block this vCPU until a [`Inst::MemoryNotify`] on the
    /// same address wakes it or `timeout` nanoseconds (`i64`) elapse. Yields an `i32` status: `0` =
    /// woken by a notify, `1` = the value did not equal `expected` (no wait), `2` = timed out. A
    /// misaligned address **traps** (like the other atomics).
    MemoryWait {
        ty: IntTy,
        addr: ValIdx,
        expected: ValIdx,
        timeout: ValIdx,
    },
    /// §12 futex notify (`atomic.notify`): wake up to `count` vCPUs waiting on the confined address
    /// `addr`. The count is the **unsigned** "wake up to N" bound (wasm's `memory.atomic.notify` count
    /// is u32; the wake-all idiom is `-1` = `u32::MAX`), so the runtime reinterprets the `i32` bits as
    /// u32 and caps the result at the real waiter count. Yields an `i32`: the number woken. Accesses no
    /// memory, so it never faults on protection — only the address is confined.
    MemoryNotify {
        addr: ValIdx,
        count: ValIdx,
    },
    /// §12 standalone memory fence (`atomic.fence <order>`): orders this vCPU's accesses without
    /// touching memory. Produces no SSA result. Honored by the interpreter; the JIT does not yet
    /// lower it (interp-only, like fibers).
    AtomicFence {
        order: Ordering,
    },

    // ----- §17 SIMD: fixed-128 `v128` (D58) -----
    /// `v128.const`: materialize a 16-byte vector constant (little-endian byte order).
    ConstV128([u8; 16]),
    /// `v128.load`: read 16 little-endian bytes from the confined effective address
    /// `addr + offset` into a `v128`. The single widened (16-byte) masked access — the
    /// only escape-TCB delta SIMD adds (§17/D58); confinement masking is implicit, as for
    /// [`Inst::Load`].
    V128Load {
        addr: ValIdx,
        offset: u64,
    },
    /// `v128.store`: write the 16 little-endian bytes of `value` at the confined effective
    /// address. Produces no SSA result (like [`Inst::Store`]).
    V128Store {
        addr: ValIdx,
        value: ValIdx,
        offset: u64,
    },
    /// `<shape>.splat`: broadcast a scalar (the shape's [`VShape::lane_val`] type) into
    /// every lane, producing a `v128`.
    Splat {
        shape: VShape,
        a: ValIdx,
    },
    /// `<shape>.extract_lane <lane>`: read lane `lane` of `a` as the shape's scalar type.
    /// For narrow integer shapes (`i8x16`/`i16x8`) `signed` selects sign- vs zero-extension
    /// into the `i32` result; it is ignored for the other shapes.
    ExtractLane {
        shape: VShape,
        lane: u8,
        signed: bool,
        a: ValIdx,
    },
    /// `<shape>.replace_lane <lane>`: `a` with lane `lane` set to scalar `b` (the shape's
    /// [`VShape::lane_val`] type); result `v128`.
    ReplaceLane {
        shape: VShape,
        lane: u8,
        a: ValIdx,
        b: ValIdx,
    },
    /// Lane-wise binary integer op (see [`VIntBinOp`]); `a`/`b`/result are `v128`.
    VIntBin {
        shape: VShape,
        op: VIntBinOp,
        a: ValIdx,
        b: ValIdx,
    },
    /// Lane-wise integer comparison (see [`VICmpOp`]); `a`/`b`/result are `v128` (per-lane all-ones
    /// or all-zeros mask of the lane width).
    VIntCmp {
        shape: VShape,
        op: VICmpOp,
        a: ValIdx,
        b: ValIdx,
    },
    /// Lane-wise float comparison (see [`VFCmpOp`]); `a`/`b`/result are `v128` (per-lane all-ones or
    /// all-zeros mask of the lane width).
    VFloatCmp {
        shape: VShape,
        op: VFCmpOp,
        a: ValIdx,
        b: ValIdx,
    },
    /// Lane-wise integer shift by a scalar amount (see [`VShiftOp`]): `a`/result are `v128`, `amt`
    /// is an `i32` (taken modulo the lane bit-width).
    VShift {
        shape: VShape,
        op: VShiftOp,
        a: ValIdx,
        amt: ValIdx,
    },
    /// Lane-wise unary integer op (see [`VIntUnOp`]); `a`/result are `v128`.
    VIntUn {
        shape: VShape,
        op: VIntUnOp,
        a: ValIdx,
    },
    /// Lane-wise saturating add/sub (see [`VSatBinOp`]); `a`/`b`/result are `v128`. `i8x16`/`i16x8`
    /// only (verifier-enforced).
    VSatBin {
        shape: VShape,
        op: VSatBinOp,
        a: ValIdx,
        b: ValIdx,
    },
    /// Lane **widen** (`extend`, see [`VWidenOp`]); `shape` is the **result** (wider) shape, the
    /// source is its [`VShape::narrower`]. `a`/result are `v128`.
    VWiden {
        shape: VShape,
        op: VWidenOp,
        a: ValIdx,
    },
    /// Lane **narrow** (see [`VNarrowOp`]); `shape` is the **result** (narrow) shape, the source is
    /// its [`VShape::wider`]. `a`/`b`/result are `v128`. `i8x16`/`i16x8` only (verifier-enforced).
    VNarrow {
        shape: VShape,
        op: VNarrowOp,
        a: ValIdx,
        b: ValIdx,
    },
    /// Lane int↔float / float↔float conversion (see [`VCvtOp`]); `a`/result are `v128`.
    VConvert {
        op: VCvtOp,
        a: ValIdx,
    },
    /// Lane-wise float pseudo-min/max (see [`VPMinMaxOp`]); `a`/`b`/result are `v128`. Float shapes.
    VPMinMax {
        shape: VShape,
        op: VPMinMaxOp,
        a: ValIdx,
        b: ValIdx,
    },
    /// `i8x16.popcnt`: per-byte population count. `a`/result are `v128`. Shape is always `i8x16`
    /// (the only shape wasm defines), so no shape field — the verifier needs no lane rule.
    VPopcnt {
        a: ValIdx,
    },
    /// Lane-wise unsigned rounding average `(a + b + 1) >> 1` (computed wide, no overflow).
    /// `a`/`b`/result are `v128`. `i8x16`/`i16x8` only (verifier-enforced — the only shapes wasm
    /// defines `avgr_u` for), so like [`Inst::VSatBin`] there is no JIT bail list.
    VAvgr {
        shape: VShape,
        a: ValIdx,
        b: ValIdx,
    },
    /// `i32x4.dot_i16x8_s`: signed dot product of adjacent `i16` pairs into `i32` lanes —
    /// `result[i] = a[2i]·b[2i] + a[2i+1]·b[2i+1]`. Source `i16x8`, result `i32x4` (the only dot
    /// wasm defines), so no shape field. `a`/`b`/result are `v128`.
    VDot {
        a: ValIdx,
        b: ValIdx,
    },
    /// Signed `i8` dot product of adjacent pairs into `i16` lanes — `result[j] = a[2j]·b[2j] +
    /// a[2j+1]·b[2j+1]` (wrapping at `i16`), both operands read as signed `i8`. Source `i8x16`,
    /// result `i16x8`. The **deterministic** lowering of the relaxed `relaxed_dot_i8x16_i7x16_s`
    /// (the spec-allowed signed-×-signed behavior, not the x86 `pmaddubsw` unsigned-×-signed one).
    VDotI8 {
        a: ValIdx,
        b: ValIdx,
    },
    /// Extended (widening) multiply: widen the low/high half of both `i8x16`/`i16x8`/`i32x4`
    /// operands (sign- or zero-, per [`VWidenOp`]) to the next wider shape, then multiply lane-wise.
    /// `shape` is the **wide result** (`i16x8`/`i32x4`/`i64x2`); `a`/`b`/result are `v128`.
    VExtMul {
        shape: VShape,
        op: VWidenOp,
        a: ValIdx,
        b: ValIdx,
    },
    /// Extended pairwise add: widen every lane of an `i8x16`/`i16x8` source (sign- or zero-, per
    /// `signed`) and sum adjacent pairs into the next wider shape — `out[i] = w(a[2i]) + w(a[2i+1])`.
    /// `shape` is the **wide result** (`i16x8`/`i32x4`); `a`/result are `v128`.
    VExtAddPairwise {
        shape: VShape,
        signed: bool,
        a: ValIdx,
    },
    /// `i16x8.q15mulr_sat_s`: signed Q15 fixed-point multiply with rounding and saturation —
    /// `out[i] = sat_i16((a[i]·b[i] + 0x4000) >> 15)`. Fixed `i16x8` (the only shape wasm defines),
    /// so no shape field. `a`/`b`/result are `v128`.
    VQ15MulrSat {
        a: ValIdx,
        b: ValIdx,
    },
    /// Lane-wise **fused multiply-add** (the relaxed-SIMD `relaxed_madd`/`relaxed_nmadd`): each lane
    /// is `a·b + c` (`neg == false`) or `−a·b + c` (`neg == true`), computed with a **single rounding**
    /// (IEEE-754 FMA). Float shapes (`f32x4`/`f64x2`); `a`/`b`/`c`/result are `v128`. SVM picks the
    /// fused behavior (one of the two the relaxed proposal permits) consistently in both backends —
    /// Cranelift `fma` and Rust `f*::mul_add` are both correctly-rounded, so the differential holds.
    VFma {
        shape: VShape,
        neg: bool,
        a: ValIdx,
        b: ValIdx,
        c: ValIdx,
    },
    /// `v128.any_true`: `i32` `1` if **any** bit of the 128-bit vector is set, else `0`
    /// (shape-agnostic). `a` is `v128`, result `i32`.
    VAnyTrue {
        a: ValIdx,
    },
    /// `<shape>.all_true`: `i32` `1` if **every** lane (of `shape`) is non-zero, else `0`. `a` is
    /// `v128`, result `i32`.
    VAllTrue {
        shape: VShape,
        a: ValIdx,
    },
    /// `<shape>.bitmask`: gather the **high (sign) bit** of each lane into the low bits of an `i32`
    /// (lane `i` → bit `i`). `a` is `v128`, result `i32`.
    VBitmask {
        shape: VShape,
        a: ValIdx,
    },
    /// Lane-wise binary float op (see [`VFloatBinOp`]); `a`/`b`/result are `v128`.
    VFloatBin {
        shape: VShape,
        op: VFloatBinOp,
        a: ValIdx,
        b: ValIdx,
    },
    /// Lane-wise unary float op (see [`VFloatUnOp`]); `a`/result are `v128`.
    VFloatUn {
        shape: VShape,
        op: VFloatUnOp,
        a: ValIdx,
    },
    /// Whole-vector bitwise binary op (see [`VBitBinOp`]); `a`/`b`/result are `v128`.
    VBitBin {
        op: VBitBinOp,
        a: ValIdx,
        b: ValIdx,
    },
    /// `v128.not`: bitwise complement of all 128 bits.
    VNot {
        a: ValIdx,
    },
    /// `v128.bitselect`: per-bit `(a & mask) | (b & !mask)`. All three operands `v128`.
    Bitselect {
        a: ValIdx,
        b: ValIdx,
        mask: ValIdx,
    },
    /// `i8x16.shuffle`: a constant byte shuffle. Each `lanes[i]` (0..32) selects byte `i`
    /// of the result from the 32-byte concatenation `a ++ b` (indices 0..16 = `a`, 16..32
    /// = `b`). Out-of-range indices (≥32) are verifier-rejected.
    Shuffle {
        lanes: [u8; 16],
        a: ValIdx,
        b: ValIdx,
    },
    /// `i8x16.swizzle`: dynamic byte select — result byte `i` is `a[b[i]]` when `b[i] < 16`,
    /// else `0`. Both operands and result `v128`.
    Swizzle {
        a: ValIdx,
        b: ValIdx,
    },
}

/// What an instruction can do **besides** producing its SSA result(s) — the single source of truth
/// the optimizer (`svm-opt`) consults for legality (see `OPT.md`, "the equivalence contract"). The
/// four axes are orthogonal; a **pure** value op (register-to-register, always the same result, no
/// fault) has all four `false`. Built by [`Inst::effects`], whose match is exhaustive on purpose (no
/// wildcard): a new `Inst` variant must fail to compile until it is classified there, so the optimizer
/// can never silently treat an unknown effect as pure.
///
/// Conservative by construction — when in doubt an axis is set (a spurious effect only forgoes an
/// optimization; a missed one would miscompile). `reads_mem`/`writes_mem` mean **guest linear
/// memory**; every other kind of runtime state (the handle table via `cap.self`, the per-vCPU TLS
/// word, fiber/thread tables, fences) is folded into `side_effect`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Effects {
    /// May raise a deterministic trap: an out-of-window / misaligned access, div/rem by zero or
    /// signed `INT_MIN/-1`, a trapping float→int, an out-of-range table/handle index, or a
    /// noreturn-class fault. A trapping op is observable — it may not be deleted, and two trapping
    /// ops may not be reordered past one another.
    pub can_trap: bool,
    /// Reads guest linear memory.
    pub reads_mem: bool,
    /// Writes guest linear memory (or other in-window state the "same final memory window" invariant
    /// pins).
    pub writes_mem: bool,
    /// Any **other** effect that bars removal and reordering: a call or host `cap.call` (arbitrary
    /// host-visible effect), a stack switch / non-falling-through control transfer
    /// (`cont.*`/`suspend`/`setjmp`/`longjmp`), a thread spawn/join or futex wait/notify, a fence, or
    /// a read/write of mutable runtime state outside linear memory (`vcpu.tls`, `cap.self`,
    /// `gc.roots`, the durable shadow base).
    pub side_effect: bool,
}

impl Effects {
    /// A **pure** value op: no trap, no memory access, no other effect. Freely removable when dead,
    /// reorderable, and CSE-able.
    pub fn is_pure(&self) -> bool {
        !self.can_trap && !self.reads_mem && !self.writes_mem && !self.side_effect
    }
    /// Safe to delete when **all** of its results are unused: it cannot trap, cannot write memory, and
    /// has no other side effect. A dead *read* is removable (reading has no observable effect), so
    /// `reads_mem` alone does not block removal.
    pub fn removable_if_dead(&self) -> bool {
        !self.can_trap && !self.writes_mem && !self.side_effect
    }
}

impl Inst {
    /// This instruction's [`Effects`] — the optimizer's legality oracle. Exhaustive by design (no
    /// wildcard arm) so adding an `Inst` variant forces a classification decision here.
    pub fn effects(&self) -> Effects {
        // Terse constructor for the non-pure arms: (can_trap, reads_mem, writes_mem, side_effect).
        let fx = |can_trap, reads_mem, writes_mem, side_effect| Effects {
            can_trap,
            reads_mem,
            writes_mem,
            side_effect,
        };
        match self {
            // ---- Pure value ops: consts, integer/float/SIMD arithmetic, compares, conversions,
            // lane ops, pointer casts. Always the same result, no fault, no memory, no effect. ----
            Inst::ConstI32(_)
            | Inst::ConstI64(_)
            | Inst::ConstF32(_)
            | Inst::ConstF64(_)
            | Inst::ConstV128(_)
            | Inst::IntCmp { .. }
            | Inst::IntUn { .. }
            | Inst::Eqz { .. }
            | Inst::Convert { .. }
            | Inst::Select { .. }
            | Inst::FBin { .. }
            | Inst::FUn { .. }
            | Inst::FCmp { .. }
            // saturating float→int does not trap (the trapping variant is `FToITrap`, below)
            | Inst::FToISat { .. }
            | Inst::IToFConv { .. }
            | Inst::Cast { .. }
            | Inst::Fma { .. }
            | Inst::RefFunc { .. }
            // Link-form address materialization: pure (a to-be-`ConstI64`), no fault/mem/effect.
            // CSE/DCE on them is sound (same name+addend ⇒ same address); the linker rewrites them
            // to real consts before any backend runs.
            | Inst::DataSym { .. }
            | Inst::DataSelf { .. }
            | Inst::DataTop
            | Inst::Splat { .. }
            | Inst::ExtractLane { .. }
            | Inst::ReplaceLane { .. }
            | Inst::VIntBin { .. }
            | Inst::VIntCmp { .. }
            | Inst::VFloatCmp { .. }
            | Inst::VShift { .. }
            | Inst::VIntUn { .. }
            | Inst::VSatBin { .. }
            | Inst::VWiden { .. }
            | Inst::VNarrow { .. }
            | Inst::VConvert { .. }
            | Inst::VPMinMax { .. }
            | Inst::VPopcnt { .. }
            | Inst::VAvgr { .. }
            | Inst::VDot { .. }
            | Inst::VDotI8 { .. }
            | Inst::VFma { .. }
            | Inst::VExtMul { .. }
            | Inst::VExtAddPairwise { .. }
            | Inst::VQ15MulrSat { .. }
            | Inst::VAnyTrue { .. }
            | Inst::VAllTrue { .. }
            | Inst::VBitmask { .. }
            | Inst::VFloatBin { .. }
            | Inst::VFloatUn { .. }
            | Inst::VBitBin { .. }
            | Inst::VNot { .. }
            | Inst::Bitselect { .. }
            | Inst::Shuffle { .. }
            | Inst::Swizzle { .. } => Effects::default(),

            // `div`/`rem` trap on a zero (or signed-overflow) divisor; the rest of `IntBin` is pure.
            Inst::IntBin { op, .. } => fx(
                matches!(op, BinOp::DivS | BinOp::DivU | BinOp::RemS | BinOp::RemU),
                false,
                false,
                false,
            ),
            // Trapping float→int: NaN / out-of-range input traps. Pure otherwise.
            Inst::FToITrap { .. } => fx(true, false, false, false),

            // ---- Guest linear-memory access. Every one can trap (out-of-window / misaligned). ----
            Inst::Load { .. } | Inst::V128Load { .. } => fx(true, true, false, false),
            Inst::Store { .. } | Inst::V128Store { .. } | Inst::MemFill { .. } => {
                fx(true, false, true, false)
            }
            Inst::MemCopy { .. } | Inst::MemMove { .. } => fx(true, true, true, false),

            // ---- Atomics: memory access **plus** a synchronization barrier (`side_effect`), so they
            // are never removed, duplicated, or reordered across. ----
            Inst::AtomicLoad { .. } => fx(true, true, false, true),
            Inst::AtomicStore { .. } => fx(true, false, true, true),
            Inst::AtomicRmw { .. } | Inst::AtomicCmpxchg { .. } => fx(true, true, true, true),
            Inst::AtomicFence { .. } => fx(false, false, false, true),
            // futex wait reads the compared word and blocks; notify touches no memory and cannot fault.
            Inst::MemoryWait { .. } => fx(true, true, false, true),
            Inst::MemoryNotify { .. } => fx(false, false, false, true),

            // ---- Calls: a guest callee or host handler may read, write, trap, or do anything — the
            // conservative full clobber. (`CallImport` is resolved to `CapCall` before a backend, but
            // is classified for completeness.) ----
            Inst::Call { .. }
            | Inst::CallIndirect { .. }
            | Inst::CallImport { .. }
            | Inst::CallImportDyn { .. }
            | Inst::CallSym { .. }
            | Inst::CapCall { .. } => fx(true, true, true, true),
            // `export.handle` mints/aliases host capability state (a grant into the own table).
            Inst::ExportHandle { .. } => fx(false, false, false, true),
            // `import.attach` mutates host binding state (no guest-memory access, but ordering
            // against every `call.import` matters) — conservative full clobber, like the calls.
            Inst::ImportAttach { .. } => fx(true, true, true, true),

            // ---- Fibers, threads, and non-local control transfer. ----
            Inst::ContNew { .. } => fx(false, false, false, true), // allocates a fiber; runs nothing yet
            Inst::ContResume { .. } | Inst::Suspend { .. } => {
                fx(true, true, true, true) // stack switch → clobber
            }
            Inst::SetJmp { .. } => fx(true, false, true, true), // writes an opaque token into the guest jmp_buf
            Inst::LongJmp { .. } => fx(true, true, false, true), // reads the jmp_buf; noreturn unwind
            Inst::ThreadSpawn { .. } => fx(false, true, true, true), // shares memory with the new vCPU
            Inst::ThreadJoin { .. } => fx(true, true, true, true), // blocks; joined writes + trap propagate

            // §GC root enumeration: scans window words, writes the candidate buffer, can fault on an
            // out-of-window buffer.
            Inst::GcRoots { .. } => fx(true, true, true, true),

            // ---- Ambient runtime-state intrinsics (`vcpu.tls`, durable shadow base). Authority-neutral
            // and read-only over guest memory, but they touch mutable runtime state (the per-vCPU TLS
            // word), so they carry `side_effect` — never CSE'd across a clobber, never removed. The
            // `cap.self.count`/`get`/`resolve`/`label`/`attest` reflection ops are now `cap.call
            // CAP_SELF` and take the generic `CapCall` effects. ----
            Inst::VcpuTlsGet | Inst::VcpuTlsSet { .. } | Inst::DurableShadowBase => {
                fx(false, false, false, true)
            }
            // §3.5 reflection: `type_id` interns into the host table (mutable runtime state);
            // `covers` probes a handle (dead handle → -errno, no trap).
            Inst::CapSelfTypeId { .. } | Inst::CapSelfCovers { .. } => fx(false, false, false, true),
        }
    }

    /// Apply `f` to **every value operand** of this instruction, in place. Exhaustive on purpose
    /// (no wildcard arm): adding an `Inst` variant that carries a [`ValIdx`] must fail to compile
    /// here rather than silently skip an operand and miscompile after a renumbering pass. `FuncIdx`
    /// immediates (`RefFunc`/`ThreadSpawn::func`) are *not* value operands and are left alone.
    ///
    /// This lives beside [`Inst::effects`] so svm-ir is the single source of truth for both "what
    /// effect does this op have" and "what are its operands"; the optimizer's `map_operands` /
    /// `each_operand` are thin adapters over it (#913).
    pub fn for_each_operand_mut(&mut self, f: &mut impl FnMut(&mut ValIdx)) {
        match self {
            // No value operands.
            Inst::ConstI32(_)
            | Inst::ConstI64(_)
            | Inst::ConstF32(_)
            | Inst::ConstF64(_)
            | Inst::ConstV128(_)
            // Link-form data addresses carry only immediates (a name+addend, or an offset) — no value
            // operands to renumber.
            | Inst::DataSym { .. }
            | Inst::DataSelf { .. }
            | Inst::DataTop
            | Inst::RefFunc { .. }
            | Inst::CapSelfTypeId { .. }
            | Inst::ExportHandle { .. }
            | Inst::VcpuTlsGet
            | Inst::DurableShadowBase
            | Inst::AtomicFence { .. } => {}

            // Exactly one operand, named `a`.
            Inst::IntUn { a, .. }
            | Inst::Eqz { a, .. }
            | Inst::Convert { a, .. }
            | Inst::FUn { a, .. }
            | Inst::FToISat { a, .. }
            | Inst::FToITrap { a, .. }
            | Inst::IToFConv { a, .. }
            | Inst::Cast { a, .. }
            | Inst::Load { addr: a, .. }
            | Inst::AtomicLoad { addr: a, .. }
            | Inst::V128Load { addr: a, .. }
            | Inst::VcpuTlsSet { val: a }
            | Inst::Suspend { value: a }
            | Inst::SetJmp { buf: a }
            | Inst::ThreadJoin { handle: a }
            | Inst::Splat { a, .. }
            | Inst::ExtractLane { a, .. }
            | Inst::VIntUn { a, .. }
            | Inst::VWiden { a, .. }
            | Inst::VConvert { a, .. }
            | Inst::VPopcnt { a, .. }
            | Inst::VExtAddPairwise { a, .. }
            | Inst::VAnyTrue { a, .. }
            | Inst::VAllTrue { a, .. }
            | Inst::VBitmask { a, .. }
            | Inst::VFloatUn { a, .. }
            | Inst::VNot { a, .. } => {
                f(a);
            }

            // Exactly two operands, named `a` and `b`.
            Inst::IntBin { a, b, .. }
            | Inst::IntCmp { a, b, .. }
            | Inst::FBin { a, b, .. }
            | Inst::FCmp { a, b, .. }
            | Inst::Store {
                addr: a, value: b, ..
            }
            | Inst::AtomicStore {
                addr: a, value: b, ..
            }
            | Inst::V128Store {
                addr: a, value: b, ..
            }
            | Inst::AtomicRmw {
                addr: a, value: b, ..
            }
            | Inst::MemoryNotify {
                addr: a, count: b, ..
            }
            | Inst::ContNew { func: a, sp: b }
            | Inst::ContResume { k: a, arg: b, .. }
            | Inst::LongJmp { buf: a, val: b }
            | Inst::ThreadSpawn { sp: a, arg: b, .. }
            | Inst::ReplaceLane { a, b, .. }
            | Inst::VIntBin { a, b, .. }
            | Inst::VIntCmp { a, b, .. }
            | Inst::VFloatCmp { a, b, .. }
            | Inst::VShift { a, amt: b, .. }
            | Inst::VSatBin { a, b, .. }
            | Inst::VNarrow { a, b, .. }
            | Inst::VPMinMax { a, b, .. }
            | Inst::VAvgr { a, b, .. }
            | Inst::VDot { a, b }
            | Inst::VDotI8 { a, b }
            | Inst::VExtMul { a, b, .. }
            | Inst::VQ15MulrSat { a, b }
            | Inst::VFloatBin { a, b, .. }
            | Inst::VBitBin { a, b, .. }
            | Inst::Shuffle { a, b, .. }
            | Inst::Swizzle { a, b } => {
                f(a);
                f(b);
            }

            // Three operands.
            Inst::Select { cond, a, b } => {
                f(cond);
                f(a);
                f(b);
            }
            // Bulk-memory ops (D62): dst, src/val, len — all value operands.
            Inst::MemCopy { dst, src, len } | Inst::MemMove { dst, src, len } => {
                f(dst);
                f(src);
                f(len);
            }
            Inst::MemFill { dst, val, len } => {
                f(dst);
                f(val);
                f(len);
            }
            Inst::Bitselect { a, b, mask } => {
                f(a);
                f(b);
                f(mask);
            }
            // Scalar / vector fused multiply-add: `a·b + c`.
            Inst::Fma { a, b, c, .. } | Inst::VFma { a, b, c, .. } => {
                f(a);
                f(b);
                f(c);
            }
            Inst::AtomicCmpxchg {
                addr,
                expected,
                replacement,
                ..
            } => {
                f(addr);
                f(expected);
                f(replacement);
            }
            Inst::MemoryWait {
                addr,
                expected,
                timeout,
                ..
            } => {
                f(addr);
                f(expected);
                f(timeout);
            }
            Inst::GcRoots {
                heap_lo,
                heap_hi,
                mask,
                buf,
                cap,
            } => {
                f(heap_lo);
                f(heap_hi);
                f(mask);
                f(buf);
                f(cap);
            }

            // Variable-length operand lists.
            Inst::Call { args, .. } => {
                for v in args.iter_mut() {
                    f(v);
                }
            }
            Inst::CallIndirect { idx, args, .. } => {
                f(idx);
                for v in args.iter_mut() {
                    f(v);
                }
            }
            Inst::CapSelfCovers { handle, .. } => f(handle),
            Inst::CapCall { handle, args, .. }
            | Inst::CallImportDyn { handle, args, .. }
            | Inst::CallSym { handle, args, .. } => {
                f(handle);
                for v in args.iter_mut() {
                    f(v);
                }
            }
            Inst::CallImport { args, .. } => {
                for v in args.iter_mut() {
                    f(v);
                }
            }
            // Phase-2 attach (IMPORTS.md): the handle is its one value operand (`import` is an
            // immediate index into the manifest, not a value).
            Inst::ImportAttach { handle, .. } => f(handle),
        }
    }

    /// How many values this instruction appends at the next block-local indices.
    ///
    /// Most instructions append exactly one; `Store` appends none; a `Call` appends
    /// its callee's result count, so it needs the per-function result arities
    /// (indexed by [`FuncIdx`]) to answer; `CallIndirect` carries its own signature.
    pub fn result_count(&self, fn_results: &[usize]) -> usize {
        match self {
            Inst::Store { .. }
            | Inst::MemCopy { .. }
            | Inst::MemMove { .. }
            | Inst::MemFill { .. }
            | Inst::AtomicStore { .. }
            | Inst::AtomicFence { .. }
            | Inst::VcpuTlsSet { .. }
            | Inst::LongJmp { .. }
            | Inst::V128Store { .. } => 0,
            // `vcpu.tls.get` appends one `i64`; `durable.shadow_base` likewise (a window byte offset).
            Inst::VcpuTlsGet | Inst::DurableShadowBase => 1,
            // `cont.resume` (in both its blocking and non-blocking forms) is the multi-result non-call op: `(status, value)`.
            Inst::ContResume { .. } => 2,
            // `cap.self.type_id`/`covers` append one `i32`. (The `cap.self.count`/`get`/`resolve`/
            // `label`/`attest` reflection ops are now `cap.call CAP_SELF` — counted by their `sig`.)
            Inst::CapSelfTypeId { .. } | Inst::CapSelfCovers { .. } | Inst::ExportHandle { .. } => {
                1
            }
            Inst::Call { func, .. } => fn_results.get(*func as usize).copied().unwrap_or(0),
            Inst::CallIndirect { ty, .. } => ty.results.len(),
            Inst::CapCall { sig, .. } => sig.results.len(),
            Inst::CallImport { sig, .. } => sig.results.len(),
            Inst::CallImportDyn { sig, .. } => sig.results.len(),
            Inst::CallSym { sig, .. } => sig.results.len(),
            _ => 1,
        }
    }
}

/// A function signature — the immediate carried by `call_indirect` and (later) the
/// function-table type ids. Equality is structural (the runtime "type_id" check).
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct FuncType {
    pub params: Vec<ValType>,
    pub results: Vec<ValType>,
}

/// One branch edge: a target block plus the argument values for its parameters.
pub type Edge = (BlockIdx, Vec<ValIdx>);

/// Block terminators. Exactly one per block; only at the block end.
#[derive(Clone, PartialEq, Debug)]
pub enum Terminator {
    /// Unconditional branch with block arguments.
    Br { target: BlockIdx, args: Vec<ValIdx> },
    /// Two-target conditional branch (no implicit fallthrough, §3b). `cond` is `i32`.
    BrIf {
        cond: ValIdx,
        then_blk: BlockIdx,
        then_args: Vec<ValIdx>,
        else_blk: BlockIdx,
        else_args: Vec<ValIdx>,
    },
    /// Indexed multi-way branch. `idx` (`i32`) selects `targets[idx]`, or `default`
    /// when out of range. Each edge carries its own block arguments.
    BrTable {
        idx: ValIdx,
        targets: Vec<Edge>,
        default: Edge,
    },
    /// Return values matching the function's result signature.
    Return(Vec<ValIdx>),
    /// Tail call (`return_call`): replace the current frame with a direct callee
    /// whose results are this function's results. Args match the callee's params.
    ReturnCall { func: FuncIdx, args: Vec<ValIdx> },
    /// Indirect tail call (`return_call_indirect`): like [`Terminator::ReturnCall`]
    /// but dispatched through the function table (masked + signature-checked, §3c).
    ReturnCallIndirect {
        ty: FuncType,
        idx: ValIdx,
        args: Vec<ValIdx>,
    },
    /// Abort: control must not reach here. Delivers a trap to the host (§3b/§5).
    /// Covers both `unreachable` and language-level `trap`/`assert` failure.
    Unreachable,
}

impl Terminator {
    /// Apply `f` to every **value operand** of this terminator, in place — the branch condition /
    /// table index, all edge block-arguments, and return / tail-call arguments. Block-index
    /// *targets* are **not** value operands and are left untouched (the optimizer remaps those
    /// separately). Exhaustive on purpose (no wildcard arm), the sibling of
    /// [`Inst::for_each_operand_mut`], so svm-ir owns terminator operand traversal too (#913).
    pub fn for_each_operand_mut(&mut self, f: &mut impl FnMut(&mut ValIdx)) {
        match self {
            Terminator::Br { args, .. } => {
                for v in args.iter_mut() {
                    f(v);
                }
            }
            Terminator::BrIf {
                cond,
                then_args,
                else_args,
                ..
            } => {
                f(cond);
                for v in then_args.iter_mut().chain(else_args.iter_mut()) {
                    f(v);
                }
            }
            Terminator::BrTable {
                idx,
                targets,
                default,
            } => {
                f(idx);
                for (_, args) in targets.iter_mut() {
                    for v in args.iter_mut() {
                        f(v);
                    }
                }
                for v in default.1.iter_mut() {
                    f(v);
                }
            }
            Terminator::Return(vals) => {
                for v in vals.iter_mut() {
                    f(v);
                }
            }
            Terminator::ReturnCall { args, .. } => {
                for v in args.iter_mut() {
                    f(v);
                }
            }
            Terminator::ReturnCallIndirect { idx, args, .. } => {
                f(idx);
                for v in args.iter_mut() {
                    f(v);
                }
            }
            Terminator::Unreachable => {}
        }
    }
}

/// A basic block: a typed parameter list, a straight-line body, one terminator.
#[derive(Clone, PartialEq, Debug)]
pub struct Block {
    pub params: Vec<ValType>,
    pub insts: Vec<Inst>,
    pub term: Terminator,
}

/// A function: signature plus its blocks (`blocks[0]` is the entry block, whose
/// parameter types must equal the function's parameter types — §3b).
#[derive(Clone, PartialEq, Debug)]
pub struct Func {
    pub params: Vec<ValType>,
    pub results: Vec<ValType>,
    pub blocks: Vec<Block>,
}

impl Func {
    /// Whether this function uses any §12 fiber/thread/futex op (`cont.*`, `thread.*`,
    /// `atomic.wait`/`notify`). The single source of truth for backends that must agree on
    /// rejecting concurrency in a context that cannot host it — e.g. a §14 JIT child (no
    /// per-child runtimes). A guest-submitted `Jit`-capability unit is now a *finer* gate — it
    /// admits fibers and rejects only threads/futex ([`uses_threads`](Func::uses_threads) `||`
    /// [`uses_futex`](Func::uses_futex)), DESIGN.md §22 "Concurrency" (renegotiated 2026-07-30) —
    /// so this whole-set predicate is no longer what that path uses.
    pub fn uses_concurrency(&self) -> bool {
        self.blocks.iter().any(|b| {
            b.insts.iter().any(|i| {
                matches!(
                    i,
                    Inst::ContNew { .. }
                        | Inst::ContResume { .. }
                        | Inst::Suspend { .. }
                        | Inst::ThreadSpawn { .. }
                        | Inst::ThreadJoin { .. }
                        | Inst::MemoryWait { .. }
                        | Inst::MemoryNotify { .. }
                )
            })
        })
    }

    /// Whether this function contains a **fiber** op (`cont.new`/`cont.resume`/`suspend`) — the
    /// scheduling primitives a guest-submitted `Jit` unit **may** host: they switch stacks *within*
    /// the domain, on the caller's thread, so a unit that runs its own scheduler to completion never
    /// parks across the synchronous `cap.call` it runs inside (DESIGN.md §22 "Concurrency"). Split
    /// out of [`uses_concurrency`](Func::uses_concurrency) so the submitted-unit gate can admit
    /// fibers while still rejecting threads/futex.
    pub fn uses_fibers(&self) -> bool {
        self.blocks.iter().any(|b| {
            b.insts.iter().any(|i| {
                matches!(
                    i,
                    Inst::ContNew { .. } | Inst::ContResume { .. } | Inst::Suspend { .. }
                )
            })
        })
    }

    /// Whether this function contains a **vCPU/thread** op (`thread.spawn`/`thread.join`) — real OS
    /// threads (§12 vCPUs). A submitted `Jit` unit **may** host these once **installed** into a
    /// thread-hosting domain (CONSOLIDATION.md §11 — installed code runs in the caller's frames on the
    /// scheduler seam); an **invoked** unit still may not (its sealed `run_invoke` is seam-free, so a
    /// thread op CapFaults). Split out for the submitted-unit gate (`define_extra`'s null-thunk check /
    /// the invoke-dispatch seam gate).
    pub fn uses_threads(&self) -> bool {
        self.blocks.iter().any(|b| {
            b.insts
                .iter()
                .any(|i| matches!(i, Inst::ThreadSpawn { .. } | Inst::ThreadJoin { .. }))
        })
    }

    /// Whether this function contains a fiber or thread **scheduling** op (`cont.*`, `suspend`,
    /// `thread.spawn`/`join`) — [`uses_concurrency`](Func::uses_concurrency) minus the futex ops
    /// (`atomic.wait`/`notify`). A §14 JIT child gets no per-child fiber/thread runtime (rejected),
    /// but *can* wait/notify against its parent domain's shared futex — the granted-children
    /// pipeline rendezvous — so the child compile distinguishes the two.
    pub fn uses_fibers_or_threads(&self) -> bool {
        self.blocks.iter().any(|b| {
            b.insts.iter().any(|i| {
                matches!(
                    i,
                    Inst::ContNew { .. }
                        | Inst::ContResume { .. }
                        | Inst::Suspend { .. }
                        | Inst::ThreadSpawn { .. }
                        | Inst::ThreadJoin { .. }
                )
            })
        })
    }

    /// Whether this function contains a futex op (`atomic.wait`/`notify`). Split out of
    /// [`uses_concurrency`](Func::uses_concurrency) for the §14 JIT child compile: waits/notifies
    /// are allowed when the child shares its parent domain's futex, rejected otherwise.
    pub fn uses_futex(&self) -> bool {
        self.blocks.iter().any(|b| {
            b.insts
                .iter()
                .any(|i| matches!(i, Inst::MemoryWait { .. } | Inst::MemoryNotify { .. }))
        })
    }

    /// Whether this function contains any `setjmp`/`longjmp` op ([`Inst::SetJmp`]/[`Inst::LongJmp`]).
    /// Used to reject a §14 JIT child that uses `setjmp` (no per-child `setjmp` runtime yet — like
    /// `uses_concurrency` for fibers/threads).
    pub fn uses_setjmp(&self) -> bool {
        self.blocks.iter().any(|b| {
            b.insts
                .iter()
                .any(|i| matches!(i, Inst::SetJmp { .. } | Inst::LongJmp { .. }))
        })
    }

    /// Whether this function contains a [`Inst::GcRoots`] op. It walks the fiber runtime's live
    /// stacks, so a module holding one needs that runtime stood up even if it never explicitly
    /// creates a fiber — the JIT's fiber-runtime gate unions this with [`uses_fibers`](Func::uses_fibers).
    pub fn uses_gc_roots(&self) -> bool {
        self.blocks
            .iter()
            .any(|b| b.insts.iter().any(|i| matches!(i, Inst::GcRoots { .. })))
    }
}

/// A linear-memory window declaration (§4). The window is `1 << size_log2` bytes —
/// a power of two, so confinement is a single `addr & (size − 1)` mask. The window
/// is a reserved virtual range; guest pointers are offsets into it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Memory {
    pub size_log2: u8,
}

impl Memory {
    /// Window size in bytes (`1 << size_log2`). `size_log2` is verified `< 64`.
    pub fn size(self) -> u64 {
        1u64 << self.size_log2
    }
    /// The confinement mask (`size − 1`).
    pub fn mask(self) -> u64 {
        self.size() - 1
    }
}

/// The reference host's **default reservation policy** (§4): the size (`log2`) of the reserved
/// virtual range a window's `mapped` bytes live inside. DESIGN §4 makes this host-configurable
/// ("e.g. 2^40"); this is the default the reference `run`/`compile_and_run` entries apply when a
/// caller doesn't pass one. It is *policy*, not verified semantics — both backends share this one
/// constant so they stay in differential lockstep, and the masking unit (`svm-mask`) never
/// hard-codes a reservation. The reserved range is `PROT_NONE` + lazily paged, so a large value
/// costs virtual address space, not committed memory.
pub const DEFAULT_RESERVED_LOG2: u8 = 40;

/// The §3e powerbox **args-buffer** window offset: where a host seeds the program-arguments blob so
/// the frontend's `_start` can hand `argc`/`argv` to a C `main(int, char**)`. This is the
/// "borrowed buffer at a known window offset" of DESIGN §3e / D44, realized as a *fixed* offset (not
/// an extra entry parameter) so the powerbox entry signature stays the language-neutral handle
/// vector — the C-specific `argv[]` marshalling lives entirely in the on-ramp's `_start`.
///
/// Layout at `[POWERBOX_ARGS_BASE, POWERBOX_ARGS_END)`:
/// `{ argc: u32-LE, envc: u32-LE }` then `argc` + `envc` NUL-terminated UTF-8 strings, packed
/// (the argv strings first, then the envp strings). A guest that never reads it (e.g. `main(void)`)
/// is unaffected. The region sits below the frontend's globals base, so it never overlaps a data
/// segment; a host must reject a blob that would reach `POWERBOX_ARGS_END`.
pub const POWERBOX_ARGS_BASE: u64 = 128;
/// The end of the powerbox args-buffer region (exclusive) — the frontend's globals/data-stack base
/// for a powerbox program (`svm-llvm`'s `STACK_PAGE`). The args blob must fit in
/// `[POWERBOX_ARGS_BASE, POWERBOX_ARGS_END)` so it never collides with a data segment.
pub const POWERBOX_ARGS_END: u64 = 16384;

/// The canonical **names** of the fixed §3e powerbox capabilities, in `VM_CAP_*` / grant order (the
/// same order/names `svm_run` grants + registers). A powerbox guest's manifest imports resolve
/// against this vocabulary (and `cap.self.resolve` re-finds them by name); `[..n]` is the prefix a
/// program granted `n` capabilities uses. (`"stderr"` is appended last — a second write-only
/// `Stream`, distinct from `"stdout"` so `eprintln!`/fd 2 capture separately; grant order keeps the
/// earlier indices stable.) (`"blocking"` left this vocabulary with CONSOLIDATION §5a
/// — the mock `Blocking` cap is test-only wiring now; a test harness that grants it registers the
/// name itself, and an unregistered resolve fails closed. `"ioring"` left with the §12
/// parking-on-blocking re-measure, 2026-08-07 — blocking host calls park instead; the retired
/// ring's iface id 9 stays reserved.)
pub const POWERBOX_CAP_NAMES: [&str; 7] = [
    "stdout",
    "stdin",
    "exit",
    "memory",
    "addrspace",
    "jit",
    "stderr",
];
/// The guest heap's bump-pointer word (`i64`) at window offset 32 (page 0 is reserved scratch).
/// Seeded by a frontend's `_start` (to the window's mapped boundary) when the program allocates.
pub const POWERBOX_HEAP_BRK: u64 = 32;
/// The guest heap's committed-boundary word (`i64`), just above [`POWERBOX_HEAP_BRK`]. The allocator
/// `Memory.map`-commits upward from here into the reserved tail (§1a sparse address space).
pub const POWERBOX_HEAP_TOP: u64 = 40;
/// The powerbox globals / data-stack base (= [`POWERBOX_ARGS_END`]): page 0 is the writable
/// stash + heap state + format scratch + args buffer, so a frontend's globals and the data stack
/// live at/above this page — a read-only global never shares page 0 with the writable stash, and
/// `_start`'s handle stores never fault on a read-only page (D40 page isolation).
pub const POWERBOX_STACK_PAGE: u64 = POWERBOX_ARGS_END; // 16384

/// The **powerbox entry shape** (IMPORTS.md phase 4): a paramless func 0 exported as `_start`. An
/// import-bearing module must carry this shape for its manifest slots to bind at instantiation (the
/// runtime never rewrites); the `_start` export is also the powerbox-entry *marker*. The one
/// definition every host shares — `svm-run`'s front door, the browser on-ramp, and the DAP backend
/// all key off the identical predicate so a module accepted as an entry by one host is by all
/// (guest-ABI shape must not drift per host — #912).
pub fn is_named_powerbox_entry(module: &Module) -> bool {
    module.funcs.first().is_some_and(|f| f.params.is_empty())
        && module
            .exports
            .iter()
            .any(|e| e.name == "_start" && e.func == 0)
}

/// Encode the powerbox **args buffer** layout (§3e / D44): `argc` and `envc` as little-endian `u32`,
/// then each arg string followed by each env string, every string NUL-terminated. The exact bytes a
/// frontend's `_start` parses into `argc`/`argv`; the one definition of the wire shape every host
/// emits (an argv-only host passes `env = &[]`). Pure layout — callers own their own validation
/// (embedded-NUL rejection, the `[POWERBOX_ARGS_BASE, POWERBOX_ARGS_END)` bound) since host policies
/// differ. (#912)
pub fn write_args_blob(args: &[&[u8]], env: &[&[u8]]) -> Vec<u8> {
    let mut blob = Vec::new();
    blob.extend_from_slice(&(args.len() as u32).to_le_bytes());
    blob.extend_from_slice(&(env.len() as u32).to_le_bytes());
    for s in args.iter().chain(env.iter()) {
        blob.extend_from_slice(s);
        blob.push(0);
    }
    blob
}

/// **Trap-on-NULL** (#964): the byte extent of the reserved NULL region for a guard-marked powerbox
/// module — `[0, POWERBOX_NULL_GUARD)` holds *nothing*, so a NULL dereference (any offset below the
/// guard) traps like native platforms instead of silently touching low scratch. 16 KiB = the **max
/// host page** (macOS), which keeps three enforcement mechanisms exactly aligned: the interpreter's
/// host-granular page map, native `mprotect`, and the wasm tier's baked compare (a smaller guard
/// diverges on 16 KiB-page hosts — proven by the #965 macOS CI catch). A marked module's writable
/// low scratch (handle stash, heap words, format buffer, args blob) relocates to
/// `[POWERBOX_NULL_GUARD, POWERBOX_NULL_GUARD + POWERBOX_ARGS_END)` — the legacy layout shifted up
/// by exactly one guard — and its globals/stack base sits at or above `2 * POWERBOX_NULL_GUARD`.
pub const POWERBOX_NULL_GUARD: u64 = 16384;

/// The **guard marker**: a function export with this name (aliasing `_start`'s funcidx) declares
/// that the module was built with the guarded layout — its low scratch lives above
/// [`POWERBOX_NULL_GUARD`], so a host may seed `[0, POWERBOX_NULL_GUARD)` unmapped and must place
/// the args blob at the shifted base. A module **without** the marker uses the legacy layout
/// (scratch from 0, args at [`POWERBOX_ARGS_BASE`]) and is never guarded — old artifacts keep
/// running unchanged. The marker is **semantics, not observability**: hosts resolve it, so it is
/// never stripped/demoted (`svmb-strip` keeps it like `_start`).
pub const NULL_GUARD_EXPORT: &str = "__null_guard";

/// The NULL-guard extent of `m`, from its [`NULL_GUARD_EXPORT`] marker: `Some(POWERBOX_NULL_GUARD)`
/// for a guard-marked module (seed `[0, guard)` unmapped; args at `guard + POWERBOX_ARGS_BASE`),
/// `None` for a legacy module (no guard; args at [`POWERBOX_ARGS_BASE`]).
pub fn module_null_guard(m: &Module) -> Option<u64> {
    m.resolve_export(NULL_GUARD_EXPORT)
        .map(|_| POWERBOX_NULL_GUARD)
}

/// Where a host seeds the args blob for `m` (and where its `_start` reads it): the guarded base for
/// a [`NULL_GUARD_EXPORT`]-marked module, the legacy [`POWERBOX_ARGS_BASE`] otherwise.
pub fn module_args_base(m: &Module) -> u64 {
    module_null_guard(m).map_or(POWERBOX_ARGS_BASE, |g| g + POWERBOX_ARGS_BASE)
}

/// The exclusive end of `m`'s args region — the bound a host must reject a blob at (the guarded
/// layout's region is the legacy one shifted up by the guard, same span).
pub fn module_args_end(m: &Module) -> u64 {
    module_null_guard(m).map_or(POWERBOX_ARGS_END, |g| g + POWERBOX_ARGS_END)
}
/// The alignment of the powerbox **data-stack base** ([`powerbox_entry_sp`]). It must be **≥ the
/// largest host page any artifact may run on** — 64 KiB (the wasm linear-memory page) — because the
/// D40 read-only const-segment protection is applied at *host-page* granularity: the runtime rounds a
/// read-only segment's end **up to the host page**, so a stack base that shared that page with the
/// read-only region would have its first store faulted. A guest large enough that its data end lands
/// in the same 64 KiB page as its read-only globals (Doom) hit exactly this on a 64 KiB host while a
/// finer host page (4 KiB native) masked it. Aligning the stack base to 64 KiB guarantees it begins on
/// a fresh host page **above** the read-only region on every host — the `_start` layout (`svm-llvm`'s
/// `stack_page`, 64 KiB for wasm builds) already does this; this keeps the reactor's per-frame `tick`
/// `sp` consistent with it. Over-alignment is free (a few KiB of window) and only ever adds clearance.
pub const POWERBOX_STACK_ALIGN: u64 = 65536;
/// The data-stack reserve a frontend's `_start` layout leaves above the globals when sizing the
/// window (`svm-llvm`'s `STACK_RESERVE`): a faulting guard region lies beyond the mapped window (§5).
pub const POWERBOX_STACK_RESERVE: u64 = 1 << 20;
/// Hard anti-bomb ceiling on the fibers (`cont.new`) a single run may create (§12/§15). Bounds the
/// fiber table so a fiber-bomb yields a clean `FiberFault` instead of unbounded host allocation. A
/// [`Quota`] can only *tighten* below this, never raise it. `1 << 24` (~16.7M) — the ceiling equals the
/// cross-backend fiber-handle index width (`FIBER_GEN_SHIFT`), raised from `1 << 16` once the arena
/// stack backend removed the `vm.max_map_count` VMA wall that used to bind concurrency lower.
pub const MAX_FIBERS: usize = 1 << 24;

/// Hard anti-bomb ceiling on the vCPUs (`thread.spawn`) a single run may create (§12/§15) — a clean
/// `ThreadFault` past it. The interpreter bounds *concurrently-live* vCPUs; the JIT's table is
/// cumulative, so there it bounds *total* spawns (stricter, but containment holds either way).
pub const MAX_VCPUS: usize = 1 << 16;

/// §15 **spawn quota** — host-configurable ceilings on how many fibers (`cont.new`) / vCPUs
/// (`thread.spawn`) a run may create, *below* the fixed [`MAX_FIBERS`]/[`MAX_VCPUS`] anti-bomb
/// ceilings. The **single** quota type shared by both runtimes (re-exported as `svm_interp::Quota` and
/// `svm_jit::Quota`), so a powerbox embedder sets it once and it binds the tree-walker, bytecode
/// engine, and JIT identically (no facade conversion). The embedder sets it on the `Host`
/// (`Host::set_quota`, which [`Quota::clamped`]s it); a guest that exceeds it traps cleanly
/// (`FiberFault`/`ThreadFault`) — DoS *containment* policy (§15/D48), not just the host-OOM backstop.
/// [`Default`] is the hard ceilings, so an unconfigured run is unchanged.
///
/// `max_vcpus` semantics differ slightly by backend (documented at the ceilings): the interpreter
/// counts concurrent liveness, the JIT counts cumulative spawns. The *type* is one; the runtimes apply
/// it per their model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Quota {
    /// Max fibers a **run** (domain) may create (`cont.new`, counting the root computation as 1);
    /// clamped to [`MAX_FIBERS`]. Per-run, not per-vCPU (the fiber table is the run-shared registry).
    pub max_fibers: usize,
    /// Max vCPUs a run may create (`thread.spawn`); clamped to [`MAX_VCPUS`].
    pub max_vcpus: usize,
}

impl Default for Quota {
    fn default() -> Quota {
        Quota {
            max_fibers: MAX_FIBERS,
            max_vcpus: MAX_VCPUS,
        }
    }
}

impl Quota {
    /// Clamp each limit to its hard anti-bomb ceiling (a quota can only *tighten*, never raise the
    /// ceiling), and to ≥ 1 (the root vCPU/computation always exists).
    pub fn clamped(self) -> Quota {
        Quota {
            max_fibers: self.max_fibers.clamp(1, MAX_FIBERS),
            max_vcpus: self.max_vcpus.clamp(1, MAX_VCPUS),
        }
    }
}

/// The powerbox data-stack base for `module`: the page-aligned offset just above its globals/data
/// segments (and never below [`POWERBOX_STACK_PAGE`]) — the `sp` a frontend's `_start` passes to
/// the entry. Exposed so an embedder driving exports directly (the reactor / `Session` model) can
/// synthesize the same `sp` per call that `_start` would.
pub fn powerbox_entry_sp(module: &Module) -> u64 {
    let data_end = module
        .data
        .iter()
        .map(|d| d.offset + d.bytes.len() as u64)
        .max()
        .unwrap_or(0);
    // Align to [`POWERBOX_STACK_ALIGN`] (64 KiB, the max host page), not merely `POWERBOX_STACK_PAGE`
    // (16 KiB): the D40 read-only protection rounds a const segment's end up to the *host* page, so the
    // stack base must start on a fresh 64 KiB page above the read-only region or its first store faults
    // on a 64 KiB host (the Doom-in-the-browser bug — a finer native page masked it).
    data_end
        .max(POWERBOX_STACK_ALIGN)
        .div_ceil(POWERBOX_STACK_ALIGN)
        * POWERBOX_STACK_ALIGN
}

/// Wrap `entry` in a synthesized **paramless powerbox `_start`** (function 0, `export "_start" 0`
/// — the §3e manifest-module entry shape) so a frontend that emits SVM-IR directly (e.g. the
/// [`link_with_manifest`] output of a separately-compiled runtime + program) becomes a runnable
/// powerbox module without reimplementing the linker's func-index reshuffle. Capabilities are
/// **manifest slots the host binds at instantiation** (IMPORTS.md phases 3–4): the synthesized
/// entry has no capability prologue and the module's import manifest travels through untouched —
/// nothing here is the retired resolve-and-rewrite bootstrap.
///
/// The entry must take a single `i64` (the data-stack pointer) and return 0 or 1 values; its
/// result becomes `_start`'s result. `_start` optionally seeds the guest heap words
/// ([`POWERBOX_HEAP_BRK`]/[`POWERBOX_HEAP_TOP`], to the window's mapped boundary) when
/// `seed_heap`, then calls the entry with `sp` = [`powerbox_entry_sp`]. The declared memory grows
/// (never shrinks) to cover the data stack reserve. Every existing funcidx — in code, exports,
/// impl-export ops, and debug info — shifts up by one as `_start` becomes function 0. (Funcref
/// *values* already flowing through data or patched constants are the caller's to fix; synthesize
/// before any [`Resolved::Slot`]-style patching.)
pub fn synth_manifest_start(
    mut module: Module,
    entry: FuncIdx,
    seed_heap: bool,
) -> Result<Module, String> {
    let ef = module.funcs.get(entry as usize).ok_or_else(|| {
        format!(
            "entry funcidx {entry} out of range ({} funcs)",
            module.funcs.len()
        )
    })?;
    if ef.params.as_slice() != [ValType::I64] {
        return Err(format!(
            "powerbox entry must take a single i64 (the data-stack pointer), got params {:?}",
            ef.params
        ));
    }
    if ef.results.len() > 1 {
        return Err(format!(
            "powerbox entry must return 0 or 1 value, got {:?}",
            ef.results
        ));
    }
    if module.exports.iter().any(|e| e.name == "_start") {
        return Err("module already exports `_start` — it already has an entry bootstrap".into());
    }
    let results = ef.results.clone();

    // Globals/data live at/above STACK_PAGE; the data stack starts page-aligned above the highest
    // segment. The window must cover it plus the data-stack reserve — grow the declared memory to
    // fit (never shrink); beyond the mapped window is the faulting guard region (§5).
    let entry_sp = powerbox_entry_sp(&module);
    let top = entry_sp + POWERBOX_STACK_RESERVE;
    let need_log2 = (64 - (top - 1).leading_zeros()) as u8;
    let size_log2 = module
        .memory
        .map_or(need_log2, |m| m.size_log2.max(need_log2));
    module.memory = Some(Memory { size_log2 });
    // The guest heap (when the program allocates) begins at the window's mapped boundary and grows
    // up into the reserved tail via `Memory.map`.
    let heap_base = seed_heap.then(|| 1u64 << size_log2);

    // Every existing funcidx (code, exports, impl-export ops, debug info) shifts up by one — the
    // prepended `_start` becomes function 0.
    offset_func_indices(&mut module, 1);
    let mut insts: Vec<Inst> = Vec::new();
    let mut next: ValIdx = 0;
    if let Some(hb) = heap_base {
        for off in [POWERBOX_HEAP_BRK, POWERBOX_HEAP_TOP] {
            insts.push(Inst::ConstI64(off as i64));
            let addr = next;
            next += 1;
            insts.push(Inst::ConstI64(hb as i64));
            let value = next;
            next += 1;
            insts.push(Inst::Store {
                op: StoreOp::I64,
                addr,
                value,
                offset: 0,
            });
        }
    }
    insts.push(Inst::ConstI64(entry_sp as i64));
    let sp = next;
    next += 1;
    insts.push(Inst::Call {
        func: entry + 1,
        args: vec![sp],
    });
    let term = if results.is_empty() {
        Terminator::Return(vec![])
    } else {
        Terminator::Return(vec![next]) // the entry's single result, appended by the call
    };
    module.funcs.insert(
        0,
        Func {
            params: Vec::new(),
            results,
            blocks: vec![Block {
                params: Vec::new(),
                insts,
                term,
            }],
        },
    );
    // Expose the bootstrap as a named export so an embedder reaches it by name (`call("_start")`),
    // not by a magic funcidx. The frontend's own entry export (e.g. "main") survives, shifted.
    module.exports.push(Export {
        name: "_start".to_string(),
        func: 0,
    });
    Ok(module)
}

/// A module: a flat list of functions plus an optional linear-memory window.
#[derive(Clone, PartialEq, Debug, Default)]
pub struct Module {
    pub funcs: Vec<Func>,
    pub memory: Option<Memory>,
    /// Initialized data segments placed in the window at instantiation (§3a): each writes
    /// `bytes` at `offset`, and a `readonly` segment is then mapped read-only (D40 — a write
    /// to it faults, §4/§5). Like an ELF loader laying out `.data`/`.rodata`; replaces the
    /// frontend's per-byte `_start` init stores.
    pub data: Vec<Data>,
    /// The import **manifest** (§7 "Host-defined capabilities & discoverability" / IMPORTS.md):
    /// the named capability slots this module expects the host to bind. Each is a `name` + op
    /// `sig` + [`ImportMode`]; a [`Inst::CallImport`] references one by index and dispatches
    /// through the domain's instantiation-time binding table ([`CAP_IMPORT_TYPE_ID`]) — the
    /// module bytes are never rewritten. The linker ([`resolve_imports_with`]) is the one pass
    /// that lowers `CallImport`s away, resolving names as link-time symbols. Empty for modules
    /// that inline their capability calls (`cap.call` on a live handle — dynamic mode).
    pub imports: Vec<Import>,
    /// Named function **exports** (name → funcidx): the host-addressable entry points, the
    /// runtime-`Module` analogue of [`LinkUnit::exports`]. Populated by [`link`] from each unit's
    /// exports, or declared directly by a frontend (`export "name" <funcidx>` in the text IR). Lets
    /// an embedder reach a function by name ([`resolve_export`]) instead of tracking funcidxs. The
    /// verifier checks each `func` is in range and names are unique; both backends ignore the table
    /// (they execute a funcidx). Empty for a module with no named entry points.
    pub exports: Vec<Export>,
    /// Named **data exports** (name → window offset): the data-symbol counterpart of
    /// [`Module::exports`], the runtime-`Module` analogue of [`LinkUnit::data_exports`]. A
    /// separately-compiled unit publishes its external-linkage globals here so another unit's
    /// [`Inst::DataSym`] / `data.ptr` can bind to them at [`link`] time. Declared in the text IR as
    /// `export <k> data "<name>" <offset>`. Backends ignore it (a data symbol resolves to a plain
    /// address before anything runs); empty for a module with no exported data. Names share the
    /// export namespace with [`Module::exports`].
    pub data_exports: Vec<DataExport>,
    /// **Data-image pointer relocations** (D-LINK, the data→data case): pointers baked into a
    /// global's own initializer whose target address the frontend cannot know at emit time because
    /// the linker assigns each unit's data base — `int *p = &g;`, `static char *kw[] = {…}`,
    /// chibicc's `Type *ty_int = &(Type){…}`. Each [`DataPtr`] names a byte offset in this module's
    /// data image and the data address to write there; [`link`] resolves and writes it once the
    /// window layout is fixed, then clears the list. The code→data case rides the instruction stream
    /// ([`Inst::DataSelf`]/[`Inst::DataSym`]) — a data pointer has no instruction to carry it, so it
    /// rides a data-offset-keyed slot instead. Unlike the retired `(func,block,inst)` `DataReloc`,
    /// the key is a **data** offset, and data images are never instrumented or reordered by a pass
    /// (only instruction streams are), so it cannot desync. Empty for a runnable module (a survivor
    /// is a verify error, since nothing would patch the placeholder bytes before they are read).
    pub data_ptrs: Vec<DataPtr>,
    /// **Data-image funcref relocations** (D-LINK, the data→code case): a **function index** baked
    /// into a global's initializer whose value the frontend cannot know at emit time because the
    /// linker assigns each unit's function base — a `proctype` gvar with a static proc initializer
    /// (`var oomHandler = continueAfterOutOfMem`, `var gExitFlush = nimNoopFlush`). Each
    /// [`DataFuncref`] names a byte offset in this module's data image and the exported function it
    /// refers to; [`link`] resolves the name to the merged funcidx and writes it as a 4-byte
    /// little-endian `i32` — the value `ref.func` would yield — then clears the list. The funcref
    /// twin of [`data_ptrs`](Module::data_ptrs): `ref.func` rides the instruction stream, but a
    /// funcref stored in static data has no instruction to carry it. Empty for a runnable module.
    pub data_funcrefs: Vec<DataFuncref>,
    /// Provider-side interface **offers** (IMPORTS.md §3.2): interfaces this module implements,
    /// one function per op ([`ImplExport`]). Declaring one confers nothing — the host wires an
    /// offer into an importer's slot, checking signatures structurally, fail-closed. Names share
    /// a namespace with [`Module::exports`]; backends ignore the table. Empty for the common
    /// consumer-only module.
    pub impl_exports: Vec<ImplExport>,
    /// The module-level **type section** (IMPORTS.md OQ3, wire v6): the one place shapes are
    /// declared. A [`TypeEntry::Func`] entry is a function signature; a [`TypeEntry::Interface`]
    /// entry is a *tuple of indices to `Func` entries* — a capability interface — so each
    /// signature is written once and shared. Identity is structural per D59 (the host interns
    /// shapes to runtime `type_id`s at wiring; two modules declaring the same shape mean the
    /// same interface). Referenced today by [`ImplExport::iface`]; interface-grouped imports
    /// and type-referencing call sites (§2.1) are the recorded next consumers. Declarations
    /// only — no code, no funcidxs; backends ignore the section.
    pub types: Vec<TypeEntry>,
    /// **Debug info — the frontend-neutral waist** (`DEBUGGING.md` §6 / D-DBG-7). Strippable
    /// tooling, **untrusted for escape** (§2a): the verifier never reads it and neither backend's
    /// safety depends on it; `None` ⇒ no debug info, zero cost. Populated by a frontend *during
    /// lowering* (only it knows which source produced which op); consumed host-side by the
    /// interpreter debugger and (later) DWARF/DAP. Slice 1 carries the neutral core (source
    /// locations + variables); the per-producer rich blob is a later field.
    pub debug_info: Option<DebugInfo>,
}

impl Module {
    /// Resolve a named [export](Module::exports) to its function index, or `None` if no export
    /// carries `name`. The verifier guarantees export names are unique, so the first match is the
    /// only match.
    pub fn resolve_export(&self, name: &str) -> Option<FuncIdx> {
        self.exports.iter().find(|e| e.name == name).map(|e| e.func)
    }

    /// Resolve a named [impl export](Module::impl_exports) (an interface offer, IMPORTS.md §3.2),
    /// or `None` if no offer carries `name`. Verifier-unique, so the first match is the only match.
    pub fn resolve_impl_export(&self, name: &str) -> Option<&ImplExport> {
        self.impl_exports.iter().find(|e| e.name == name)
    }

    /// Resolve type-section entry `idx` as an interface: the ordered op signatures it names,
    /// or `None` if `idx` is out of range, not a [`TypeDef::Interface`], or any element fails
    /// to name a [`TypeDef::Func`] entry. The verifiers' and hosts' one lookup.
    pub fn interface_ops(&self, idx: u32) -> Option<Vec<&FuncType>> {
        match self.types.get(idx as usize)? {
            TypeEntry::Interface(elems) => elems
                .iter()
                .map(|e| match self.types.get(e.ty as usize)? {
                    TypeEntry::Func(ft) => Some(ft),
                    TypeEntry::Interface(_) => None,
                })
                .collect(),
            TypeEntry::Func(_) => None,
        }
    }

    /// Resolve interface entry `idx` to its **named** op list `(name, signature)`, or `None`
    /// if `idx` is out of range, names a `Func`, or any element reference is not a `Func`
    /// entry. The coverage-binding view (IMPORTS.md §3.5).
    pub fn interface_named_ops(&self, idx: u32) -> Option<Vec<(&str, &FuncType)>> {
        match self.types.get(idx as usize)? {
            TypeEntry::Interface(elems) => elems
                .iter()
                .map(|e| match self.types.get(e.ty as usize)? {
                    TypeEntry::Func(ft) => Some((e.name.as_str(), ft)),
                    TypeEntry::Interface(_) => None,
                })
                .collect(),
            TypeEntry::Func(_) => None,
        }
    }

    /// The signature of op `op` of import `i`, resolved through the type section: a
    /// [`ImportShape::Func`] import has exactly op `0`; a grouped import resolves the
    /// interface element. `None` for out-of-range indices or non-well-formed references.
    pub fn import_op_sig(&self, i: u32, op: u32) -> Option<&FuncType> {
        match self.imports.get(i as usize)?.shape {
            ImportShape::Func(t) => match (op, self.types.get(t as usize)?) {
                (0, TypeEntry::Func(ft)) => Some(ft),
                _ => None,
            },
            ImportShape::Interface(t) => {
                let ops = self.interface_named_ops(t)?;
                ops.get(op as usize).map(|&(_, ft)| ft)
            }
        }
    }

    /// The full requirement set of import `i` as named ops: a flat import is the singleton
    /// `[(name, sig)]` (its op name is the import's second-level name); a grouped import is
    /// its interface's named op list.
    pub fn import_named_ops(&self, i: u32) -> Option<Vec<(&str, &FuncType)>> {
        let im = self.imports.get(i as usize)?;
        match im.shape {
            ImportShape::Func(t) => match self.types.get(t as usize)? {
                TypeEntry::Func(ft) => Some(vec![(im.name.as_str(), ft)]),
                TypeEntry::Interface(_) => None,
            },
            ImportShape::Interface(t) => self.interface_named_ops(t),
        }
    }

    /// Intern `ft` into the type section (linear-scan dedup — sections are small), returning
    /// its index. The builder-side helper that keeps each signature written once.
    pub fn intern_func_type(&mut self, ft: FuncType) -> u32 {
        if let Some(i) = self
            .types
            .iter()
            .position(|t| matches!(t, TypeEntry::Func(f) if *f == ft))
        {
            return i as u32;
        }
        self.types.push(TypeEntry::Func(ft));
        (self.types.len() - 1) as u32
    }

    /// Push a flat func import (interning its signature), returning the import index. The
    /// mechanical migration path from the v6 inline-signature shape.
    pub fn add_func_import(
        &mut self,
        name: impl Into<String>,
        sig: FuncType,
        mode: ImportMode,
    ) -> u32 {
        let t = self.intern_func_type(sig);
        self.imports.push(Import {
            name: name.into(),
            shape: ImportShape::Func(t),
            mode,
        });
        (self.imports.len() - 1) as u32
    }
}

/// The neutral core of the debug-info waist (`DEBUGGING.md` §6): everything the interpreter
/// stepper and backtraces need, in a form **every** frontend can populate (chibicc tokens, LLVM
/// `!DILocation`/`dbg.value`, wasm DWARF). Positions key on `(func, block, inst)` — module 0, the
/// guest's own program (installed §22 units have no source). Format-specific richness (full DWARF
/// DIEs / LLVM DI) is a later opaque per-producer blob the middle never parses.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct DebugInfo {
    /// Source file paths, referenced by index from [`Loc::file`].
    pub files: Vec<String>,
    /// Source location of individual ops. An op with no entry inherits nothing (unmapped).
    pub locs: Vec<Loc>,
    /// Structured source types, referenced by index ([`TypeId`]) from [`VarInfo::type_id`] and
    /// [`Field::ty`]. The §6 `TypeRef` enriched with the field offsets / element strides that
    /// aggregate inspection needs (struct/array expansion, `a.b` / `arr[i]`). Optional: a module
    /// can carry `vars` with only render-name `ty` strings and no `types` table at all.
    pub types: Vec<TypeDef>,
    /// Source variables and where their value lives (the §6 neutral `VarLoc` = S2).
    pub vars: Vec<VarInfo>,
    /// Opaque per-producer debug blobs (the §6 / D-DBG-7 "rich blob"): a frontend's native debug
    /// info (DWARF sections, LLVM DI metadata) carried through the IR verbatim. The middle never
    /// parses it — only a future DWARF/DI re-emitter (W5) does — and the verifier ignores it (§2a,
    /// strippable / untrusted-for-escape). Empty for the common case.
    pub blobs: Vec<ProducerBlob>,
    /// Source **function names** (the §6 waist's function-name table): a sparse `func → name`, so a
    /// backtrace / DWARF subprogram / kill message reads `compute` instead of `fn3`. Module 0 only
    /// (installed §22 units have no source). Empty ⇒ no names — consumers fall back to the
    /// synthesized `fn{N}`. Frontend-emitted under `-g`; strippable / untrusted-for-escape (§2a).
    pub func_names: Vec<FuncName>,
}

/// One source function name (DEBUGGING.md §6): the symbolic `name` of function index `func`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FuncName {
    pub func: u32,
    pub name: String,
}

/// An opaque per-producer debug blob (DEBUGGING.md §6 rich blob). `producer` tags the format so a
/// consumer can dispatch (e.g. `".debug_info"`, `".debug_str"`, `"llvm-di"`); `bytes` is verbatim.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ProducerBlob {
    pub producer: String,
    pub bytes: Vec<u8>,
}

/// An index into [`DebugInfo::types`].
pub type TypeId = u32;

/// How to interpret the bytes of a [`TypeDef::Base`] scalar (the §6 neutral encoding).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Encoding {
    Signed,
    Unsigned,
    Float,
    Bool,
}

/// A structured source type (`DEBUGGING.md` §6 `TypeRef`, enriched for aggregate inspection). The
/// `name` each variant carries is the neutral render name; aggregates additionally carry the
/// layout (field offsets, element count) a debugger needs to expand them. The middle never reads
/// this — it is host-side tooling, strippable and untrusted-for-escape (§2a).
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum TypeDef {
    /// A scalar primitive: `int`, `char`, `_Bool`, `double`, …
    Base {
        name: String,
        encoding: Encoding,
        size: u32,
    },
    /// `T *` — `pointee` indexes the pointed-to type; `size` is the pointer width.
    Pointer {
        name: String,
        pointee: TypeId,
        size: u32,
    },
    /// `T[count]` — `elem` indexes the element type; the stride is the element's size.
    Array {
        name: String,
        elem: TypeId,
        count: u32,
    },
    /// A `struct` (distinct field offsets) or `union` (overlapping offsets); `size` is `sizeof`.
    Aggregate {
        name: String,
        size: u32,
        fields: Vec<Field>,
    },
    /// A type whose structure isn't carried (function, VLA, …): render name + `sizeof` only.
    Opaque { name: String, size: u32 },
}

impl TypeDef {
    /// The `sizeof` of this type (the array stride for [`TypeDef::Array`] is `elem`'s size, which
    /// the consumer resolves through the table).
    pub fn size(&self) -> u32 {
        match self {
            TypeDef::Base { size, .. }
            | TypeDef::Pointer { size, .. }
            | TypeDef::Opaque { size, .. }
            | TypeDef::Aggregate { size, .. } => *size,
            // An array's size isn't stored; it's `count * elem.size`, resolved by the consumer.
            TypeDef::Array { .. } => 0,
        }
    }
}

/// One member of a [`TypeDef::Aggregate`]: its source name, byte `offset` from the aggregate's
/// base, and the [`TypeId`] of its type.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Field {
    pub name: String,
    pub offset: u32,
    pub ty: TypeId,
}

/// A source location for one op (`DEBUGGING.md` §6 neutral core). `col == 0` means "no column"
/// (wasm DWARF often omits columns).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Loc {
    pub func: u32,
    pub block: u32,
    pub inst: u32,
    pub file: u32,
    pub line: u32,
    pub col: u32,
}

/// A source variable and its value location (`DEBUGGING.md` §6 / S2). Carries a neutral render
/// name (`ty`) and, when known, a [`TypeId`] into [`DebugInfo::types`] for its structured layout
/// (`type_id`). A **module-scoped global** (source-level `static`/global variable) uses `func ==
/// `[`GLOBAL_SCOPE`] — visible in every frame — and a [`VarLoc::Fixed`] absolute window address.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct VarInfo {
    pub func: u32,
    pub name: String,
    /// Neutral render name (e.g. `"int"`, `"struct"`). Kept for the scalar common case and
    /// as the always-present human label; aggregate *layout* lives in [`VarInfo::type_id`].
    pub ty: String,
    pub loc: VarLoc,
    /// The structured type ([`DebugInfo::types`] index) when this var's layout is carried — set for
    /// aggregates (and, when emitted, scalars). `None` ⇒ render via `ty` only (no expansion).
    pub type_id: Option<TypeId>,
    /// The variable's **lexical scope** as an inclusive source-line range `(start_line, end_line)`,
    /// for resolving C shadowing (an inner-block redeclaration, or a local shadowing a global):
    /// among same-name variables, the consumer picks the one whose scope covers the stopped source
    /// line, innermost (largest `start_line`) winning. `None` ⇒ function-wide (the outermost scope;
    /// the back-compatible default). Line-based rather than `IrPc`-based because a frontend knows
    /// source spans at parse time and the consumer already maps a stopped pc to a source line.
    pub scope: Option<(u32, u32)>,
}

/// Where a source variable's value lives at runtime (the S2 value-location model, IR form). The
/// `Machine` (Cranelift register/stack) variant for debugging JIT-optimized code is a later field.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum VarLoc {
    /// An address-taken / aggregate / narrow local: window data-stack slot at `data-SP + off`.
    Window { off: i64 },
    /// A promoted scalar with a single function-wide SSA value index (resolved directly from the
    /// frame's values by the interpreter — no debug-build mode needed). Valid when the holding value
    /// never changes (e.g. a parameter, or chibicc `-Og` window-free scalars).
    Ssa { value: u32 },
    /// A promoted scalar whose holding SSA value **varies through the function** — the DWARF
    /// location-list case (S2). Each [`SsaLoc`] says "from this `(block, inst)` onward (within the
    /// block) the var is held by this block-local value index"; resolution is nearest-preceding
    /// within the stopped block (a block with no covering entry ⇒ the var is not live there). This
    /// is what lets promoted scalars — and wasm/LLVM SSA-valued locals, which change per block — be
    /// inspected without a window slot.
    SsaList(Vec<SsaLoc>),
    /// A variable in **window memory at a runtime base address + offset**: the base is an SSA value
    /// that varies per pc (a location list, `base`), so `read = window[resolve(base) + off ..]`.
    /// This is the wasm/DWARF case — clang describes a C local as `DW_OP_fbreg <off>` relative to a
    /// frame base held in a wasm local (an SSA value here), not as a fixed `data-SP + off` window
    /// slot. ([`Window`] is the special case where the base is always frame value 0, the data-SP.)
    ///
    /// [`Window`]: VarLoc::Window
    WindowVia { base: Vec<SsaLoc>, off: i64 },
    /// A variable at a **fixed absolute window address** — a module-scoped global (source-level
    /// `static`/global): `read = window[addr ..]`, frame-independent. Paired with `func ==
    /// `[`GLOBAL_SCOPE`] so it is visible in every frame. (Unlike [`Window`], which is `data-SP +
    /// off`, this is an absolute address — globals live low in the window, below the data stack.)
    Fixed { addr: u64 },
}

/// The sentinel [`VarInfo::func`] of a **module-scoped global** (no owning function): it resolves in
/// every frame, not just one function's. Real function indices are `0..funcs.len()`, so `u32::MAX`
/// never collides.
pub const GLOBAL_SCOPE: u32 = u32::MAX;

/// One entry of a [`VarLoc::SsaList`] location list: within block `block`, from instruction `inst`
/// onward (until a later entry in the same block), the variable is held by block-local SSA value
/// index `value`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SsaLoc {
    pub block: u32,
    pub inst: u32,
    pub value: u32,
}

/// A named capability import (§7, reshaped by IMPORTS.md §3.5; single-string names at v8).
/// The `name` is **one string the core only compares for equality** — namespacing is a dotted
/// convention inside it (`posix.fs`, `app.log`; `svm.` reserved for platform interfaces), and
/// prefix-based policy is wirer code, never a core concept. Resolver vocabulary, never
/// identity. The signature lives in the module's type section: [`ImportShape::Func`] is a flat
/// one-op import referencing a [`TypeEntry::Func`]; [`ImportShape::Interface`] is a **grouped**
/// import — one slot binding a whole interface, referencing a [`TypeEntry::Interface`] that
/// states the *requirement set* the binding must cover (coverage binding: name-keyed,
/// signature-equal, extra provider ops ignored). Declared in [`Module::imports`] and referenced
/// by index from [`Inst::CallImport`]. Listing a module's imports is the up-front, fail-closed
/// "what capabilities does this need?" check (a missing binding never silently no-ops).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Import {
    pub name: String,
    /// What this import requires, as a type-section reference.
    pub shape: ImportShape,
    /// How the slot binds (IMPORTS.md phase 2): [`ImportMode::Required`] is bound fail-closed at
    /// instantiation and immutable for the instance's lifetime; [`ImportMode::Rebindable`] is
    /// declared and typed but may start empty and be (re)bound at runtime via
    /// [`Inst::ImportAttach`]. Calling through an empty slot traps (`CapFault`).
    pub mode: ImportMode,
}

/// An import's requirement, by type-section index (IMPORTS.md §3.5). `Func(t)` must name a
/// [`TypeEntry::Func`] (a flat one-op import — the singleton requirement set); `Interface(t)`
/// must name a [`TypeEntry::Interface`] (a grouped import — the whole requirement set behind
/// one slot). The verifier checks the reference kind; binding checks coverage.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ImportShape {
    Func(u32),
    Interface(u32),
}

/// A declared import's binding mode (IMPORTS.md §2.1). `Required` is the wasm-like default: a
/// missing binding refuses instantiation, and the binding is immutable-per-instance (always legal
/// to devirtualize). `Rebindable` supports the reflect-then-attach discovery pattern: the slot is
/// declared with its interface up front, filled or re-filled at runtime with a held capability.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum ImportMode {
    #[default]
    Required,
    Rebindable,
}

/// A named function **export**: a `name` the host (or a linker) addresses a function by, mapping to
/// its index in [`Module::funcs`]. The runtime-`Module` analogue of [`LinkUnit::exports`] — wasm-like
/// name-addressable entry points, so an embedder can `call("main")` without tracking funcidxs. The
/// verifier checks `func` is in range and names are unique; backends ignore exports (they run a
/// funcidx). Empty for a module with no named entry points (e.g. a bare kernel run by index).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Export {
    pub name: String,
    pub func: FuncIdx,
}

/// A named **data export**: a `name` a linker binds a cross-unit data reference to, mapping to a
/// byte `offset` in this unit's (un-relocated) data window — the data-symbol counterpart of
/// [`Export`], and the in-`Module` form of a [`LinkUnit::data_exports`] entry. Declared in the text
/// IR as `export <k> data "<name>" <offset>`. [`link`] adds each unit's base to `offset` to place
/// the symbol in the merged window, then resolves every [`Inst::DataSym`] / `data.ptr` naming it.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct DataExport {
    pub name: String,
    pub offset: u64,
}

/// A **data-image pointer relocation** ([`Module::data_ptrs`], D-LINK): write the 8-byte
/// little-endian window address of `target` into this unit's data image at byte offset `at`. Models
/// a pointer stored inside a global's initializer (`int *p = &g;`) — the data→data counterpart of
/// the code→data [`Inst::DataSelf`]/[`Inst::DataSym`] forms, for the case where the pointer lives in
/// static data rather than in an instruction. The frontend emits placeholder bytes (any 8 bytes) in
/// a `data` segment covering `[at, at+8)` and one of these to fix them up; [`link`] overwrites the
/// full 8 bytes with the resolved absolute address (the addend rides `target`, not the placeholder).
/// Pointers are 8 bytes because window addresses are `i64` (the width `DataSym`/`DataSelf` yield).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct DataPtr {
    /// Byte offset within this unit's (un-relocated) data image where the 8-byte pointer sits.
    pub at: u64,
    /// The data address the pointer resolves to, once the linker has placed the units' data.
    pub target: DataPtrTarget,
}

/// A **data-image funcref relocation** ([`Module::data_funcrefs`], D-LINK, data→code): write the
/// 4-byte little-endian merged **function index** of the exported func `name` into this unit's data
/// image at byte offset `at`. Models a function pointer stored in a global's static initializer
/// (`void (*f)() = g;` — nimony's `var oomHandler = continueAfterOutOfMem`). The value written is the
/// one `ref.func name` would yield after the merge (module-0 funcref = its funcidx, §22), so loading
/// the slot and `call_indirect`-ing through it dispatches to `name`. The frontend emits placeholder
/// bytes in a `data` segment covering `[at, at+4)` and one of these to fix them up; [`link`]
/// overwrites the 4 bytes and fails closed ([`LinkError::Unresolved`]) if no unit exports `name`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct DataFuncref {
    /// Byte offset within this unit's (un-relocated) data image where the 4-byte funcidx sits.
    pub at: u64,
    /// The exported function whose merged index is written — resolved via the link func-symbol table.
    pub name: String,
}

/// What a [`DataPtr`] points at — the data-image twin of the code→data link forms.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum DataPtrTarget {
    /// This unit's own data at `offset` → `dbase + offset` (self-relative; the twin of
    /// [`Inst::DataSelf`]). Any `&g + k` arithmetic is folded into `offset` by the frontend.
    SelfOff(u64),
    /// An exported data symbol `name` plus `addend` → `addr(name) + addend` (cross-unit; the twin of
    /// [`Inst::DataSym`]). Fail-closed at link if no unit exports `name` ([`LinkError::Unresolved`]).
    Sym { name: String, addend: i64 },
}

/// One entry in the module-level type section ([`Module::types`]): every declared shape is
/// either a function signature or an interface built from them. One index space, so imports,
/// call sites, and impl exports can all reference the same entries (D59: identity is the
/// shape, so an index is pure interning — never a nominal type). Distinct from the
/// debug-info [`TypeDef`] (DEBUGGING.md's source-type table), which is strippable tooling.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum TypeEntry {
    /// A function signature.
    Func(FuncType),
    /// A capability interface: an ordered tuple of **named** ops, each referencing a
    /// [`TypeEntry::Func`] entry for its signature. Op names are required (wire v7) and are the
    /// binding-time contract (coverage matching is name-keyed); they are **excluded from the
    /// structural intern key** — runtime `type_id` identity stays shape-only (D59). Interfaces
    /// never nest.
    Interface(Vec<IfaceOp>),
}

/// One named op of a [`TypeEntry::Interface`]: `name` is the coverage-matching key (required,
/// non-identity); `ty` indexes the [`TypeEntry::Func`] carrying the op's signature.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct IfaceOp {
    pub name: String,
    pub ty: u32,
}

/// A provider-side interface **offer** (IMPORTS.md §3.2): this module declares it *implements*
/// the interface named by `iface`, one function per operation — `ops[i]` is the funcidx
/// implementing op `i`.
///
/// Declaring an offer confers nothing: authority moves only when a wiring party (the host, or a
/// parent holding both ends) connects the offer to an importer's slot, minting a table entry that
/// trampolines into these functions in *this* module's domain — the signature check at wiring is
/// structural and fail-closed. The verifier checks each funcidx is in range, `ops` is non-empty,
/// and the name is unique across both export namespaces; backends ignore the table (dispatch goes
/// through the host-owned handle table, never through the offer).
///
/// (Amendment to the §3.2 sketch, which drew a single `(op, args…)` dispatch func: guest functions
/// have fixed signatures, so one-func-per-op is what keeps the check exact — no padded marshaling
/// convention, and the trampoline invokes the op's function directly with the call's arguments.)
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ImplExport {
    pub name: String,
    /// The declared interface — an index into [`Module::types`] that must name a
    /// [`TypeEntry::Interface`]. The verifier checks the offer *implements* it exactly:
    /// `ops.len()` equals the interface's op count and each `funcs[ops[i]]`'s declared type
    /// equals the interface's op-`i` signature — so "implemented the wrong interface" is a
    /// verify error, not a wiring surprise.
    pub interface: u32,
    pub ops: Vec<FuncIdx>,
    /// CALLS.md increment 7 (7.4) — the provider's own **concurrency-policy declaration**
    /// (§10.1: policy is a declaration by the provider, never an inference): `true` ⇒ when this
    /// offer is wired as an instance, handlers admit **concurrently** with no gate (the
    /// `Threaded` policy) and the module synchronizes its own state with atomics/futexes (§12).
    /// `false` (default, text omits it) ⇒ `single`: run-to-park atomicity, admissions serialized.
    /// Text: the `threaded` keyword between the offer name and its interface index. Like the rest
    /// of this declaration it confers nothing by itself — the wiring party mints the entry.
    pub threaded: bool,
}

/// A capability binding resolved from an import name at link time (§7 / DESIGN.md §22): the
/// concrete interface `type_id` and operation `op` a name bound to. Returned by the resolver
/// passed to [`resolve_imports_with`] (as [`Resolved::Cap`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ResolvedCap {
    pub type_id: u32,
    pub op: u32,
}

/// What an import **name** binds to when [`resolve_imports_with`] lowers it — **link-time symbol
/// resolution** (IMPORTS.md §2.5: the linker legitimately produces new module bytes; the runtime
/// never rewrites — a manifest module's imports bind to slots at instantiation instead). The §7
/// capability case (`Cap`) is the host-ABI binding a guest loader's symbol table can deliver;
/// `Func` is the **compile-time (static) linking** case — the name resolved to a concrete
/// function index, so the call lowers to a direct [`Inst::Call`]. (A data-symbol binding —
/// lowering to a constant window offset — is a natural follow-up.)
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Resolved {
    /// A host capability: lower to a `cap.call` on the import's handle operand (§7).
    Cap(ResolvedCap),
    /// Another function in the **same** linked module (by index): lower to a direct `call`. The
    /// static-linking case — a symbol resolved to a function merged into this module at link time.
    Func(FuncIdx),
    /// A function reached through the shared `call_indirect` **table slot** (the *dynamic*-linking
    /// case): lower to `call_indirect <slot>`, so a separately-compiled unit can call a function it
    /// doesn't share an index space with (e.g. a plugin calling the host program it was loaded into).
    /// The import's handle operand must be a `ConstI32` placeholder — it is patched to `slot` and
    /// reused as the `call_indirect` index (a 1:1 rewrite, no value renumbering).
    Slot(u32),
}

impl From<ResolvedCap> for Resolved {
    fn from(c: ResolvedCap) -> Self {
        Resolved::Cap(c)
    }
}

/// Why [`resolve_imports_with`] failed (fail-closed: a missing/garbled import never silently
/// becomes a no-op or a wrong call).
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum ImportError {
    /// The resolver returned no binding for this import name.
    Unresolved(String),
    /// A `CallImport` referenced an import index past the module's [`Module::imports`].
    BadImportIndex(u32),
    /// A [`Resolved::Slot`] binding had a handle operand that is **not** a `ConstI32` placeholder
    /// (the frontend must emit one for these imports, since resolution patches it to the
    /// `call_indirect` slot index).
    SlotHandleNotConst,
}

/// The **link-time** §7 symbol-resolution pass (IMPORTS.md §2.5: this survives in the linker
/// only — [`link`], `compile_linked` — which legitimately produces new module bytes;
/// instantiation never rewrites). A name may bind to a host capability ([`Resolved::Cap`] →
/// `cap.call` on the placeholder's handle operand — per-call-site-handle *dynamic* dispatch),
/// another function in the linked module ([`Resolved::Func`] → a direct `call`), or a
/// `call_indirect` table slot ([`Resolved::Slot`]). `Func` is the compile-time (static)
/// linking step the in-window loader builds on. Each [`Inst::CallSym`] rewrites **1:1** (no
/// value renumbering) — a `Func` binding drops the unused handle operand. Fails closed on an
/// unresolved name, so a missing symbol surfaces at link, never as a silent miscompile. The
/// result is symbol-free (verifier/both backends accept it; the linked module is re-verified —
/// and a surviving `CallSym` is itself a verify error, so nothing unresolved slips through).
pub fn resolve_imports_with(
    module: &Module,
    mut resolve: impl FnMut(&str) -> Option<Resolved>,
) -> Result<Module, ImportError> {
    // Resolve each declared import once, up front (so a name binds consistently and a
    // missing one fails before any rewriting).
    let bound: Vec<Resolved> = module
        .imports
        .iter()
        .map(|imp| resolve(&imp.name).ok_or_else(|| ImportError::Unresolved(imp.name.clone())))
        .collect::<Result<_, _>>()?;
    let mut out = module.clone();
    let fn_results: Vec<usize> = out.funcs.iter().map(|f| f.results.len()).collect();
    for f in &mut out.funcs {
        for b in &mut f.blocks {
            // Map each value index to its defining instruction (block params → `None`) — a `Slot`
            // import patches the `ConstI32` that defines its handle operand, so we may need to reach
            // a *different* instruction than the `CallSym` we're rewriting.
            let mut def_of: Vec<Option<usize>> = vec![None; b.params.len()];
            for (p, inst) in b.insts.iter().enumerate() {
                for _ in 0..inst.result_count(&fn_results) {
                    def_of.push(Some(p));
                }
            }
            for i in 0..b.insts.len() {
                let (import, handle) = match &b.insts[i] {
                    Inst::CallSym { import, handle, .. } => (*import, *handle),
                    _ => continue,
                };
                let bind = *bound
                    .get(import as usize)
                    .ok_or(ImportError::BadImportIndex(import))?;
                // Pull the call's pieces out of the placeholder so we can rebuild it.
                let (sig, args) = match &mut b.insts[i] {
                    Inst::CallSym { sig, args, .. } => {
                        (core::mem::take(sig), core::mem::take(args))
                    }
                    _ => unreachable!(),
                };
                b.insts[i] = match bind {
                    Resolved::Cap(cap) => Inst::CapCall {
                        type_id: cap.type_id,
                        op: cap.op,
                        sig,
                        handle,
                        args,
                    },
                    // Static-link a function symbol → a direct call (handle unused; sig re-checked).
                    Resolved::Func(func) => Inst::Call { func, args },
                    // Dynamic-link a function symbol → a `call_indirect` through the table slot: patch
                    // the handle's `ConstI32` placeholder to `slot` and reuse it as the index.
                    Resolved::Slot(slot) => {
                        patch_placeholder(&mut b.insts, &def_of, handle, slot as i32)?;
                        Inst::CallIndirect {
                            ty: sig,
                            idx: handle,
                            args,
                        }
                    }
                };
            }
        }
    }
    out.imports.clear();
    Ok(out)
}

/// Patch the `ConstI32` placeholder defining value `handle` to `value` — the [`Resolved::Slot`]
/// rewrite. Fails closed if the operand's defining instruction is not a `ConstI32` in the same
/// block (a block param, or hoisted).
fn patch_placeholder(
    insts: &mut [Inst],
    def_of: &[Option<usize>],
    handle: u32,
    value: i32,
) -> Result<(), ImportError> {
    let def = def_of
        .get(handle as usize)
        .copied()
        .flatten()
        .ok_or(ImportError::SlotHandleNotConst)?;
    match &mut insts[def] {
        Inst::ConstI32(c) => {
            *c = value;
            Ok(())
        }
        _ => Err(ImportError::SlotHandleNotConst),
    }
}

/// One unit to statically link: a module plus the symbols it **exports**. Function symbols
/// (`exports`) and the unit's named `Module::imports` (`call.sym`) resolve against the other units
/// by [`link`]; `data_exports` name window offsets within the unit's (un-relocated) data, which the
/// unit's [`Inst::DataSym`] / `data.ptr` references bind to. A unit's own data addresses ride in the
/// instruction stream ([`Inst::DataSelf`] / [`Inst::DataSym`]) and in `data.ptr` slots, not in a
/// side table — the linker rewrites them in place once it has placed the unit's data.
#[derive(Clone, Debug, Default)]
pub struct LinkUnit {
    pub module: Module,
    /// Function symbols this unit provides: `name → local function index`.
    pub exports: Vec<(String, FuncIdx)>,
    /// Data symbols this unit provides: `name → byte offset within the unit's (un-relocated) data`.
    pub data_exports: Vec<(String, u64)>,
}

/// Why [`link`] failed (fail-closed; the linked module is also re-verified before it runs).
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum LinkError {
    /// Two units export the same symbol.
    DuplicateSymbol(String),
    /// An export names a function index past its unit's `funcs`.
    BadExport { symbol: String, index: FuncIdx },
    /// A unit imports a symbol no unit exports.
    Unresolved(String),
    /// A `data.ptr` slot (`at`) does not lie within any of the unit's data segments — the frontend
    /// must emit an 8-byte placeholder in a `data` segment covering `[at, at+8)`. A malformed unit.
    BadDataPtr { at: u64 },
    /// A `CallImport` referenced an out-of-range import (a malformed unit).
    BadImportIndex(u32),
    /// Retained-manifest linking ([`link_with_manifest`]) found units disagreeing on a shared
    /// import's structural shape or mode, a malformed shape reference, or a grouped
    /// (interface-shaped) call site whose name resolved to a single exported function. The
    /// message names the import and the disagreement.
    ImportShapeMismatch(String),
    /// An `import.attach` targeted an import whose name resolved to an exported **function** —
    /// a statically-linked call has no slot to rebind (the unit meant a runtime-bound name).
    AttachResolved(String),
}

/// **Statically link** units into one module — the compile-time loader (dynamic-linking milestones
/// 1–2). Concatenate the units' functions into one list, **reindexing** each unit's internal function
/// references by its base offset; place each unit's **data** in a non-overlapping window region and
/// rewrite its link-form data addresses ([`Inst::DataSelf`]/[`Inst::DataSym`], `data.ptr`) to
/// concrete `ConstI64`s so they follow the data; build function + data symbol tables from all
/// exports; and resolve every unit's named imports — a `call` symbol to a **direct call**, a data
/// symbol to a **constant address** — against them. The result is one import-free, relocated module —
/// re-verify it before running, since a unit is untrusted like any frontend output (a cross-unit
/// signature mismatch is caught there).
pub fn link(units: &[LinkUnit]) -> Result<Module, LinkError> {
    link_impl(units, false)
}

/// [`link`], for **separate-artifact frontends** (a runtime library carrying live §7 capability
/// imports, linked against program units — the jacl shape): cross-unit symbols resolve exactly as
/// [`link`] does, but an import **no unit exports is retained** in the merged module's manifest
/// instead of failing the link — the host binds it at instantiation, per DESIGN §7 ("the same
/// named-import mechanism generalizes to cross-unit linking"). Same-named retained imports dedup
/// to one slot when their structural shape and mode agree (D59 — meaning, not type indices);
/// disagreement fails closed ([`LinkError::ImportShapeMismatch`]). Slot references
/// (`call.import`, `call.sym`, `import.attach`) reindex into the merged manifest; a `call.sym`
/// to a retained name stays symbolic (executable slot dispatch at wire v8). Fail-closed moves to
/// instantiation for retained names (an unregistered name still cannot run); data symbols are
/// unaffected (an unresolved one is still [`LinkError::Unresolved`]). Feed the result to
/// [`synth_manifest_start`] for a runnable powerbox module, and re-verify like any linked output.
pub fn link_with_manifest(units: &[LinkUnit]) -> Result<Module, LinkError> {
    link_impl(units, true)
}

fn link_impl(units: &[LinkUnit], retain: bool) -> Result<Module, LinkError> {
    // Function and data layout: each unit's functions occupy `[fbase, fbase + n_funcs)` in the merged
    // list, and its data occupies the window region `[dbase, dbase + data_span)`. `data_span` is the
    // high-water mark of the unit's own (un-relocated) data.
    //
    // `dbase` is aligned to the **max host page** ([`POWERBOX_STACK_ALIGN`], 64 KiB), not merely 16
    // bytes. The confinement runtime applies the D40 read-only-segment protection at *host-page*
    // granularity — it rounds a read-only segment's end **up to the host page** and marks the page
    // `PROT_READ`. A frontend lays each unit out so its own read-only and writable data never share a
    // host page (rodata last, page-aligned), but stacking units 16-byte-tight reintroduces the hazard
    // *across* units: one unit's read-only tail lands on the same host page as the next unit's writable
    // head, and that writable data's first store faults. Page-aligning every unit's base keeps each
    // unit on its own host pages (no cross-unit sharing) **and** preserves each unit's internal
    // offset-mod-page coloring (the shift is a page multiple), so the intra-unit separation survives
    // the relocation. The padding is a few reserved (`PROT_NONE`, lazily paged) KiB per unit — free in
    // the sparse window (§1a), and load-bearing for confinement, not cosmetic.
    let page_align = |x: u64| (x + (POWERBOX_STACK_ALIGN - 1)) & !(POWERBOX_STACK_ALIGN - 1);
    let mut fbases = Vec::with_capacity(units.len());
    let mut dbases = Vec::with_capacity(units.len());
    let (mut ftotal, mut dtotal): (u32, u64) = (0, 0);
    for u in units {
        fbases.push(ftotal);
        ftotal += u.module.funcs.len() as u32;
        let dbase = page_align(dtotal);
        dbases.push(dbase);
        let span = u
            .module
            .data
            .iter()
            .map(|d| d.offset + d.bytes.len() as u64)
            .max()
            .unwrap_or(0);
        dtotal = dbase + span;
    }
    // Symbol tables: exported name → global function index, and exported data name → window address.
    let mut funcs_tab: alloc::collections::BTreeMap<String, FuncIdx> =
        alloc::collections::BTreeMap::new();
    let mut data_tab: alloc::collections::BTreeMap<String, u64> =
        alloc::collections::BTreeMap::new();
    // The merged module's first-class export table — every unit's function exports, in declaration
    // order (deterministic, unlike a by-name map walk), at their reindexed global funcidxs.
    let mut exports: Vec<Export> = Vec::new();
    // The merged module's first-class data-export table — every unit's data exports at their
    // reindexed (base-added) window offsets, in declaration order. Symmetric with `exports`.
    let mut data_exports: Vec<DataExport> = Vec::new();
    for (u, (&fbase, &dbase)) in units.iter().zip(fbases.iter().zip(&dbases)) {
        for (name, local) in &u.exports {
            if *local as usize >= u.module.funcs.len() {
                return Err(LinkError::BadExport {
                    symbol: name.clone(),
                    index: *local,
                });
            }
            if funcs_tab.insert(name.clone(), fbase + local).is_some()
                || data_tab.contains_key(name)
            {
                return Err(LinkError::DuplicateSymbol(name.clone()));
            }
            exports.push(Export {
                name: name.clone(),
                func: fbase + local,
            });
        }
        for (name, local_off) in &u.data_exports {
            let addr = dbase + local_off;
            if data_tab.insert(name.clone(), addr).is_some() || funcs_tab.contains_key(name) {
                return Err(LinkError::DuplicateSymbol(name.clone()));
            }
            data_exports.push(DataExport {
                name: name.clone(),
                offset: addr,
            });
        }
    }
    // Per-unit type-section bases (prefix sums), so instruction-level type references
    // (`call.import` dynamic mode, `cap.self.type_id`/`covers`) reindex alongside funcidxs.
    let tbases: Vec<u32> = units
        .iter()
        .scan(0u32, |acc, u| {
            let b = *acc;
            *acc += u.module.types.len() as u32;
            Some(b)
        })
        .collect();
    // Partition every unit's imports (DESIGN §7: "the same named-import mechanism generalizes to
    // cross-unit linking"): a name some unit exports is link-time resolved to a direct call; any
    // other name is a host-bound slot — retained in the merged manifest when `retain`
    // ([`link_with_manifest`]), fail-closed otherwise ([`link`]). Retained same-named imports
    // dedup to one merged slot iff their **resolved** shapes and modes agree (D59: structural
    // identity — compare meaning, never raw type indices).
    let mut merged_imports: Vec<Import> = Vec::new();
    let mut merged_shapes: Vec<ResolvedShape> = Vec::new();
    let mut merged_by_name: alloc::collections::BTreeMap<String, u32> =
        alloc::collections::BTreeMap::new();
    let mut disps: Vec<Vec<ImportDisp>> = Vec::with_capacity(units.len());
    for (u, &tbase) in units.iter().zip(&tbases) {
        let mut d = Vec::with_capacity(u.module.imports.len());
        for imp in &u.module.imports {
            if let Some(&f) = funcs_tab.get(&imp.name) {
                d.push(ImportDisp::Func(f));
                continue;
            }
            if !retain {
                return Err(LinkError::Unresolved(imp.name.clone()));
            }
            let shape = resolved_import_shape(&u.module, imp.shape).ok_or_else(|| {
                LinkError::ImportShapeMismatch(format!(
                    "import `{}` has a malformed type-section reference",
                    imp.name
                ))
            })?;
            if let Some(&j) = merged_by_name.get(&imp.name) {
                let prev = &merged_imports[j as usize];
                if prev.mode != imp.mode || merged_shapes[j as usize] != shape {
                    return Err(LinkError::ImportShapeMismatch(format!(
                        "units disagree on import `{}` (structural shape or mode)",
                        imp.name
                    )));
                }
                d.push(ImportDisp::Keep(j));
            } else {
                let j = merged_imports.len() as u32;
                merged_imports.push(Import {
                    name: imp.name.clone(),
                    // The shape's type reference follows its unit's entries into the merged
                    // type section (concatenated at `tbase`, below).
                    shape: match imp.shape {
                        ImportShape::Func(t) => ImportShape::Func(tbase + t),
                        ImportShape::Interface(t) => ImportShape::Interface(tbase + t),
                    },
                    mode: imp.mode,
                });
                merged_shapes.push(shape);
                merged_by_name.insert(imp.name.clone(), j);
                d.push(ImportDisp::Keep(j));
            }
        }
        disps.push(d);
    }
    // Per unit: place its data, apply its relocations, reindex its functions, resolve its imports.
    // The data-stack base for a `data.top` (the `_start` data-SP of a linked powerbox program): the
    // window address just above *all* placed data, aligned exactly as [`powerbox_entry_sp`] does for
    // the on-ramp entry. It is the whole program's, not any one unit's, so every unit resolves the
    // same value regardless of link order — the entry unit's `_start` can still be function 0.
    let entry_sp = dtotal
        .max(POWERBOX_STACK_ALIGN)
        .div_ceil(POWERBOX_STACK_ALIGN)
        * POWERBOX_STACK_ALIGN;
    let mut has_data_top = false;
    let mut funcs: Vec<Func> = Vec::with_capacity(ftotal as usize);
    let mut data: Vec<Data> = Vec::new();
    for ((u, ((&fbase, &dbase), &tbase)), disp) in units
        .iter()
        .zip(fbases.iter().zip(&dbases).zip(&tbases))
        .zip(&disps)
    {
        let mut m = u.module.clone();
        offset_type_indices(&mut m, tbase);
        // Patch this unit's data-image pointers (`data.ptr`, the data→data case) while segment
        // offsets are still unit-local: overwrite the 8 placeholder bytes at each slot with the
        // resolved absolute window address. Must precede the segment shift below so `at` and the
        // covering segment share one coordinate frame; the address written is already absolute
        // (`dbase`-relative for `self`, the symbol's window address for `sym`).
        apply_unit_data_ptrs(&mut m, dbase, &data_tab)?;
        // Patch this unit's data-image funcrefs (`data.funcref`, the data→code case): overwrite the
        // 4 placeholder bytes at each slot with the resolved merged funcidx. Like `data.ptr`, this
        // runs while segment offsets are still unit-local (`at` and its covering segment share a
        // frame). The funcidx is already global (`funcs_tab` holds `fbase + local`), so it needs no
        // later shift by this unit's `offset_func_indices`.
        apply_unit_data_funcrefs(&mut m, &funcs_tab)?;
        // Relocate this unit's data segments into its assigned window region…
        for d in &mut m.data {
            d.offset += dbase;
        }
        // …and rewrite its link-form data addresses to concrete `ConstI64`s, now that the window
        // layout is fixed: `data.self <off>` → `dbase + off` (own data), `data.sym "name" +addend`
        // → `addr(name) + addend` (a cross-unit symbol, fail-closed if unexported). This is the
        // data twin of the `call.sym → call` rewrite below — a 1:1, position-independent edit.
        has_data_top |= resolve_unit_data_addrs(&mut m, dbase, entry_sp, &data_tab)?;
        offset_func_indices(&mut m, fbase);
        rewrite_unit_imports(&mut m, disp)?;
        funcs.extend(m.funcs);
        data.extend(m.data);
    }
    // Merge the units' impl surfaces (offers + the type section, IMPORTS.md §3.2/OQ3): type
    // entries concatenate with a per-unit index offset (declarations only — identity is
    // structural, so cross-unit duplicates are harmless; the host intern canonicalizes at
    // wiring), interface entries reindex their element references, and each offer reindexes
    // its interface reference and its op funcidxs. Offer names share the export namespace,
    // so a collision is the same `DuplicateSymbol` a function export would raise.
    let mut merged_types: Vec<TypeEntry> = Vec::new();
    let mut merged_impls: Vec<ImplExport> = Vec::new();
    for (u, &fbase) in units.iter().zip(&fbases) {
        let tbase = merged_types.len() as u32;
        merged_types.extend(u.module.types.iter().map(|t| {
            match t {
                TypeEntry::Func(ft) => TypeEntry::Func(ft.clone()),
                TypeEntry::Interface(elems) => TypeEntry::Interface(
                    elems
                        .iter()
                        .map(|e| IfaceOp {
                            name: e.name.clone(),
                            ty: tbase + e.ty,
                        })
                        .collect(),
                ),
            }
        }));
        for e in &u.module.impl_exports {
            if funcs_tab.contains_key(&e.name)
                || data_tab.contains_key(&e.name)
                || merged_impls.iter().any(|o| o.name == e.name)
            {
                return Err(LinkError::DuplicateSymbol(e.name.clone()));
            }
            merged_impls.push(ImplExport {
                name: e.name.clone(),
                interface: tbase + e.interface,
                ops: e.ops.iter().map(|&f| fbase + f).collect(),
                threaded: e.threaded,
            });
        }
    }
    Ok(Module {
        funcs,
        // The merged window (one shared linear memory) must (a) be at least as large as any unit
        // declared, (b) cover every relocated data segment — the units' data is **stacked** into
        // non-overlapping windows, so the top (`dtotal`) can exceed any single unit's 64 KiB — and
        // (c) when the program has a `data.top` data stack, reserve [`POWERBOX_STACK_RESERVE`] above
        // `entry_sp` for it (as [`synth_manifest_start`] does for the on-ramp entry), since the
        // entry unit sized its own window for a stack that has since been pushed up by the other
        // units' data. Grow to the smallest power-of-two window that holds the max of these — never
        // shrinking a unit's request. No unit declaring memory ⇒ no data segments, so `None` stays
        // `None`.
        memory: units
            .iter()
            .filter_map(|u| u.module.memory)
            .map(|m| m.size_log2)
            .max()
            .map(|declared| {
                let cover = if has_data_top {
                    entry_sp + POWERBOX_STACK_RESERVE
                } else {
                    dtotal
                };
                let need = if cover <= 1 {
                    0
                } else {
                    (64 - (cover - 1).leading_zeros()) as u8
                };
                Memory {
                    size_log2: declared.max(need),
                }
            }),
        data,
        // Empty for [`link`]; the deduped host-bound slots for [`link_with_manifest`].
        imports: merged_imports,
        exports,
        // Every unit's data symbols at their merged window offsets (symmetric with `exports`).
        data_exports,
        // Every unit's `data.ptr` slots were resolved and cleared per-unit above; the linked
        // module is runnable and carries none.
        data_ptrs: Vec::new(),
        data_funcrefs: Vec::new(),
        // Interfaces + impl exports (offers, §3.2) merge across units below — see
        // `merge_impl_surfaces`; a link unit's own module may carry both.
        impl_exports: merged_impls,
        types: merged_types,
        // Merging per-unit debug info (with the reindexed function indices) is a follow-up.
        debug_info: None,
    })
}

/// How one unit-local import is handled by [`link`]/[`link_with_manifest`]: statically resolved
/// to a merged function index, or kept as a host-bound slot at its merged-manifest index.
enum ImportDisp {
    Func(FuncIdx),
    Keep(u32),
}

/// An [`ImportShape`] with its type reference **resolved** — what the shape *means*, so cross-unit
/// dedup compares structure (D59 identity), never unit-local type indices. `None`-producing
/// references (out of range, wrong entry kind) fail the link closed before any merge.
#[derive(PartialEq)]
enum ResolvedShape {
    Func(FuncType),
    Interface(Vec<(String, FuncType)>),
}

fn resolved_import_shape(m: &Module, shape: ImportShape) -> Option<ResolvedShape> {
    match shape {
        ImportShape::Func(t) => match m.types.get(t as usize)? {
            TypeEntry::Func(ft) => Some(ResolvedShape::Func(ft.clone())),
            TypeEntry::Interface(_) => None,
        },
        ImportShape::Interface(t) => Some(ResolvedShape::Interface(
            m.interface_named_ops(t)?
                .into_iter()
                .map(|(n, ft)| (n.to_string(), ft.clone()))
                .collect(),
        )),
    }
}

/// Rewrite one unit's import references per its dispositions — the [`link_impl`] counterpart of
/// [`resolve_imports_with`]'s `Func` case, extended with slot **retention**. A link-resolved name
/// lowers 1:1 to a direct [`Inst::Call`] (`call.sym`'s unused handle operand is dropped, no value
/// renumbering — the same rewrite [`resolve_imports_with`] does; a manifest-form `call.import`
/// must be flat, op 0). A retained name's references reindex into the merged manifest: `call.sym`
/// stays symbolic (executable slot dispatch at wire v8), `call.import`/`import.attach` follow
/// their slot. Fail-closed on an out-of-range index; `import.attach` to a link-resolved function
/// is [`LinkError::AttachResolved`] (there is no slot to rebind). Clears the unit manifest — the
/// retained entries live in the merged module's.
fn rewrite_unit_imports(m: &mut Module, disps: &[ImportDisp]) -> Result<(), LinkError> {
    let names: Vec<String> = m.imports.iter().map(|i| i.name.clone()).collect();
    for f in &mut m.funcs {
        for b in &mut f.blocks {
            for inst in &mut b.insts {
                let slot = match inst {
                    Inst::CallSym { import, .. }
                    | Inst::CallImport { import, .. }
                    | Inst::ImportAttach { import, .. } => *import,
                    _ => continue,
                };
                let disp = disps
                    .get(slot as usize)
                    .ok_or(LinkError::BadImportIndex(slot))?;
                match (disp, &mut *inst) {
                    (ImportDisp::Func(func), Inst::CallSym { args, .. }) => {
                        let args = core::mem::take(args);
                        *inst = Inst::Call { func: *func, args };
                    }
                    (ImportDisp::Func(func), Inst::CallImport { op, args, .. }) => {
                        if *op != 0 {
                            return Err(LinkError::ImportShapeMismatch(format!(
                                "grouped `call.import` op {} of `{}` cannot resolve to a single \
                                 exported function",
                                op, names[slot as usize]
                            )));
                        }
                        let args = core::mem::take(args);
                        *inst = Inst::Call { func: *func, args };
                    }
                    (ImportDisp::Func(_), Inst::ImportAttach { .. }) => {
                        return Err(LinkError::AttachResolved(names[slot as usize].clone()));
                    }
                    (
                        ImportDisp::Keep(j),
                        Inst::CallSym { import, .. }
                        | Inst::CallImport { import, .. }
                        | Inst::ImportAttach { import, .. },
                    ) => *import = *j,
                    _ => unreachable!("slot extraction and rewrite match the same instructions"),
                }
            }
        }
    }
    m.imports.clear();
    Ok(())
}

/// Resolve a unit's **link-form data addresses** to concrete `ConstI64`s, now that the linker has
/// fixed the window layout (the data twin of [`resolve_imports_with`]'s `call.sym → call` rewrite).
/// [`Inst::DataSelf`] `{offset}` → `ConstI64(dbase + offset)` (own data). [`Inst::DataSym`]
/// `{name, addend}` → `ConstI64(addr(name) + addend)`, where `addr` comes from the merged
/// `data_tab`; an unexported name is fail-closed ([`LinkError::Unresolved`], like a missing function
/// symbol). Rewrites **1:1** in place (the result value index is unchanged), so no value
/// renumbering — the same discipline the call-symbol rewrite follows.
/// `data.top` resolves to `entry_sp` (the whole program's post-link data-stack base, the same for
/// every unit). Returns `true` if any `data.top` was rewritten, so [`link`] knows to reserve the
/// data stack above it in the merged window.
fn resolve_unit_data_addrs(
    m: &mut Module,
    dbase: u64,
    entry_sp: u64,
    data_tab: &alloc::collections::BTreeMap<String, u64>,
) -> Result<bool, LinkError> {
    let mut saw_data_top = false;
    for f in &mut m.funcs {
        for b in &mut f.blocks {
            for inst in &mut b.insts {
                let addr = match inst {
                    Inst::DataSelf { offset } => dbase.wrapping_add(*offset),
                    Inst::DataTop => {
                        saw_data_top = true;
                        entry_sp
                    }
                    Inst::DataSym { name, addend } => {
                        // The name is stored as raw bytes (Copy-clone friendly); resolve it against
                        // the string-keyed symbol table. Non-UTF-8 or unexported ⇒ fail closed.
                        let key = core::str::from_utf8(name).map_err(|_| {
                            LinkError::Unresolved(String::from_utf8_lossy(name).into_owned())
                        })?;
                        let base = *data_tab
                            .get(key)
                            .ok_or_else(|| LinkError::Unresolved(key.to_string()))?;
                        base.wrapping_add(*addend as u64)
                    }
                    _ => continue,
                };
                *inst = Inst::ConstI64(addr as i64);
            }
        }
    }
    Ok(saw_data_top)
}

/// Apply a unit's **data-image pointer relocations** ([`Module::data_ptrs`], the data→data case):
/// for each [`DataPtr`], resolve its `target` to an absolute window address — `SelfOff(off)` →
/// `dbase + off` (own data), `Sym { name, addend }` → `addr(name) + addend` (fail-closed on an
/// unexported name) — and overwrite the 8 little-endian bytes at slot `at` in the covering data
/// segment. Called with segment offsets still **unit-local**, so `at` indexes directly into the
/// segment whose `[offset, offset+len)` contains `[at, at+8)`; a slot with no covering segment (or
/// one whose tail would run past the segment end) is fail-closed ([`LinkError::BadDataPtr`]). Clears
/// `data_ptrs` on success — the linked module carries none, mirroring the `data.sym`/`data.self`
/// rewrite that leaves no link form behind.
fn apply_unit_data_ptrs(
    m: &mut Module,
    dbase: u64,
    data_tab: &alloc::collections::BTreeMap<String, u64>,
) -> Result<(), LinkError> {
    for p in &m.data_ptrs {
        let addr: u64 = match &p.target {
            DataPtrTarget::SelfOff(off) => dbase.wrapping_add(*off),
            DataPtrTarget::Sym { name, addend } => {
                let base = *data_tab
                    .get(name.as_str())
                    .ok_or_else(|| LinkError::Unresolved(name.clone()))?;
                base.wrapping_add(*addend as u64)
            }
        };
        // Find the segment covering `[at, at+8)` and overwrite those 8 bytes (little-endian).
        // Untrusted unit: use checked arithmetic so a bogus offset fails closed, never panics.
        let end =
            p.at.checked_add(8)
                .ok_or(LinkError::BadDataPtr { at: p.at })?;
        let seg = m.data.iter_mut().find(|d| {
            d.offset <= p.at
                && d.offset
                    .checked_add(d.bytes.len() as u64)
                    .is_some_and(|seg_end| end <= seg_end)
        });
        let seg = seg.ok_or(LinkError::BadDataPtr { at: p.at })?;
        let lo = (p.at - seg.offset) as usize;
        seg.bytes[lo..lo + 8].copy_from_slice(&addr.to_le_bytes());
    }
    m.data_ptrs.clear();
    Ok(())
}

/// Apply a unit's **data-image funcref relocations** ([`Module::data_funcrefs`], the data→code
/// case): for each [`DataFuncref`], resolve `name` to its merged funcidx via `funcs_tab` (fail-closed
/// [`LinkError::Unresolved`] if unexported) and write it as a 4-byte little-endian `i32` into the
/// covering data segment — the value `ref.func name` yields. A slot not covered by a segment (or one
/// whose 4 bytes run past a segment end) fails closed ([`LinkError::BadDataPtr`]). Clears
/// `data_funcrefs` on success — the linked module carries none, mirroring `data_ptrs`.
fn apply_unit_data_funcrefs(
    m: &mut Module,
    funcs_tab: &alloc::collections::BTreeMap<String, FuncIdx>,
) -> Result<(), LinkError> {
    for r in &m.data_funcrefs {
        let func = *funcs_tab
            .get(r.name.as_str())
            .ok_or_else(|| LinkError::Unresolved(r.name.clone()))?;
        let end =
            r.at.checked_add(4)
                .ok_or(LinkError::BadDataPtr { at: r.at })?;
        let seg = m
            .data
            .iter_mut()
            .find(|d| {
                d.offset <= r.at
                    && d.offset
                        .checked_add(d.bytes.len() as u64)
                        .is_some_and(|seg_end| end <= seg_end)
            })
            .ok_or(LinkError::BadDataPtr { at: r.at })?;
        let lo = (r.at - seg.offset) as usize;
        seg.bytes[lo..lo + 4].copy_from_slice(&func.to_le_bytes());
    }
    m.data_funcrefs.clear();
    Ok(())
}

/// Add `offset` to every **static function index** in `m` (the merged-module reindex): `call`,
/// `ref.func`, `thread.spawn`, and the `return_call` terminator. `call_indirect`/`cont.*` dispatch on
/// runtime funcref *values*, not static indices, so they are untouched. `call.import` carries an
/// import index (not a function index) and is likewise untouched — it is resolved separately.
/// Add `offset` to every **type-section index** carried by an instruction (`call.import`
/// dynamic mode, `cap.self.type_id`, `cap.self.covers`) — the merged-module type reindex.
/// Section-level references (`ImportShape`, `ImplExport::interface`, interface elements) are
/// remapped where their sections merge, not here.
fn offset_type_indices(m: &mut Module, offset: u32) {
    if offset == 0 {
        return;
    }
    for f in &mut m.funcs {
        for b in &mut f.blocks {
            for inst in &mut b.insts {
                match inst {
                    Inst::CallImportDyn { ty, .. }
                    | Inst::CapSelfTypeId { ty }
                    | Inst::CapSelfCovers { ty, .. } => *ty += offset,
                    _ => {}
                }
            }
        }
    }
}

fn offset_func_indices(m: &mut Module, offset: u32) {
    if offset == 0 {
        return;
    }
    for f in &mut m.funcs {
        for b in &mut f.blocks {
            for inst in &mut b.insts {
                match inst {
                    Inst::Call { func, .. }
                    | Inst::RefFunc { func }
                    | Inst::ThreadSpawn { func, .. } => *func += offset,
                    _ => {}
                }
            }
            if let Terminator::ReturnCall { func, .. } = &mut b.term {
                *func += offset;
            }
        }
    }
    // Named exports, impl-export ops, and function-indexed debug info point at funcidxs too, so
    // they shift with the functions. ([`link`] merges impl surfaces and debug info from the
    // *original* unit modules, so this only serves whole-module shifts like
    // [`synth_manifest_start`]'s prepend — never double-applies there.)
    for e in &mut m.exports {
        e.func += offset;
    }
    for e in &mut m.impl_exports {
        for f in &mut e.ops {
            *f += offset;
        }
    }
    if let Some(di) = &mut m.debug_info {
        for l in &mut di.locs {
            l.func += offset;
        }
        for v in &mut di.vars {
            if v.func != GLOBAL_SCOPE {
                v.func += offset;
            }
        }
        for n in &mut di.func_names {
            n.func += offset;
        }
    }
}

/// An initialized data segment (§3a / D40). Placed in the window `[offset, offset+bytes.len())`
/// at instantiation; `readonly` ones are protected after the copy so guest writes fault.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Data {
    pub offset: u64,
    pub readonly: bool,
    pub bytes: Vec<u8>,
}

/// **Compile-time confinement bound analysis** (§1a "guard-when-bounded", D36–D38/D63) — the shared,
/// single-definition proof both native and wasm JITs use to decide when a memory access is *provably*
/// in-window and its bounds check is therefore redundant. Lifted here (from the native JIT's private
/// copy) so there is **one** audited copy of the veto predicate rather than a per-backend copy that
/// could diverge (INVARIANTS #9); kept dependency-free and structural so it stays auditable.
///
/// **Soundness contract (escape-critical in the native JIT):** [`ub_of`] never *under*-estimates the
/// real maximum (unknown ⇒ [`UB_TOP`]), and [`in_window`] is checked/saturating so an overflow can
/// only make it return `false` (fall back to the runtime check), never `true`. The native JIT drops
/// the confinement mask on an `in_window` proof, so a *wrong* proof there is a confinement escape
/// (caught by the escape-oracle differential); the wasm JIT keeps its `& MASK` clamp regardless and
/// only elides the trap branch, so there a wrong proof is at worst a trap-parity divergence (caught
/// by the interpreter differential), never an escape.
pub mod bounds {
    use super::{BinOp, ConvOp, Inst, ValIdx};

    /// The "unknown / no useful bound" element of the upper-bound lattice (`u64::MAX`): any value
    /// annotated `UB_TOP` forces the runtime check (never elided).
    pub const UB_TOP: u64 = u64::MAX;

    /// Look up the tracked upper bound of SSA value `i`, defaulting to [`UB_TOP`] when out of range
    /// (e.g. a block parameter, whose bound does not cross the block boundary).
    #[inline]
    pub fn ub_at(ubs: &[u64], i: ValIdx) -> u64 {
        ubs.get(i as usize).copied().unwrap_or(UB_TOP)
    }

    /// A **sound, conservative upper bound** on an SSA value's unsigned (`u64`) magnitude, used only
    /// to decide mask/check elision. Every rule must never under-estimate the real maximum; anything
    /// not modelled returns [`UB_TOP`]. Lower bounds are irrelevant (a `u64` is `≥ 0`), so only the
    /// upper bound is tracked. `ubs` is indexed like the value map (block params = [`UB_TOP`]); the
    /// caller must keep `ubs` in lockstep with the SSA value numbering (a misalignment could
    /// mis-elide).
    pub fn ub_of(inst: &Inst, ubs: &[u64]) -> u64 {
        let ub = |i: ValIdx| ub_at(ubs, i);
        match inst {
            Inst::ConstI64(c) => *c as u64,
            Inst::ConstI32(c) => *c as u32 as u64,
            Inst::IntBin { op, a, b, .. } => {
                let (x, y) = (ub(*a), ub(*b));
                match op {
                    // a & b ≤ min(a, b); a|b, a^b, a+b ≤ a + b; a*b ≤ a * b (wrap ⇒ Top).
                    BinOp::And => x.min(y),
                    BinOp::Add | BinOp::Or | BinOp::Xor => x.checked_add(y).unwrap_or(UB_TOP),
                    BinOp::Mul => x.checked_mul(y).unwrap_or(UB_TOP),
                    _ => UB_TOP,
                }
            }
            // Zero-extend: the i64 value is the (≤ u32::MAX) source, no wider.
            Inst::Convert {
                op: ConvOp::ExtendI32U,
                a,
            } => ub(*a).min(0xFFFF_FFFF),
            Inst::Convert {
                op: ConvOp::WrapI64,
                ..
            } => 0xFFFF_FFFF,
            _ => UB_TOP,
        }
    }

    /// True iff every access `[addr+offset, addr+offset+width)` is provably within `[0, size)` given
    /// `addr ≤ addr_ub` — i.e. the runtime confinement check is redundant and may be elided.
    /// Saturating/checked throughout so an overflow can only make this *false* (fall back to the
    /// check), never escape.
    #[inline]
    pub fn in_window(addr_ub: u64, offset: u64, width: u32, size: u64) -> bool {
        match addr_ub
            .checked_add(offset)
            .and_then(|s| s.checked_add(width as u64))
        {
            Some(top) => top <= size,
            None => false,
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn in_window_is_fail_closed_on_overflow() {
            // An addr upper bound near u64::MAX must never prove in-window (overflow ⇒ false).
            assert!(!in_window(UB_TOP, 0, 1, u64::MAX));
            assert!(!in_window(u64::MAX - 4, 8, 4, u64::MAX));
            // Exactly touching the top byte is in-window; one past is not.
            assert!(in_window(0, 0, 8, 8));
            assert!(!in_window(1, 0, 8, 8));
            assert!(in_window(100, 4, 4, 108));
            assert!(!in_window(100, 4, 4, 107));
        }

        #[test]
        fn ub_of_masks_and_extends_bound() {
            // (v0 & 0xFFFF) is bounded by 0xFFFF regardless of v0's (unknown) bound.
            let insts = [
                Inst::ConstI64(0xFFFF), // v-? const K
            ];
            // Model: ubs = [UB_TOP (v0 block param), K's bound]
            let ubs = [UB_TOP, ub_of(&insts[0], &[UB_TOP])];
            // v2 = v0 & K  → min(UB_TOP, 0xFFFF) = 0xFFFF
            let band = Inst::IntBin {
                ty: crate::IntTy::I64,
                op: BinOp::And,
                a: 0,
                b: 1,
            };
            assert_eq!(ub_of(&band, &ubs), 0xFFFF);
            // ExtendI32U caps at u32::MAX.
            let ext = Inst::Convert {
                op: ConvOp::ExtendI32U,
                a: 0,
            };
            assert_eq!(ub_of(&ext, &[UB_TOP]), 0xFFFF_FFFF);
            // An unmodelled op is Top (fail-closed).
            let cmp = Inst::IntCmp {
                ty: crate::IntTy::I64,
                op: crate::CmpOp::Eq,
                a: 0,
                b: 1,
            };
            assert_eq!(ub_of(&cmp, &ubs), UB_TOP);
        }
    }
}

#[cfg(test)]
mod import_tests {
    use super::*;

    // Build a one-function module whose body issues two link-form CallSyms ("write", "exit").
    fn module_with_imports() -> Module {
        let sig_write = FuncType {
            params: vec![ValType::I64, ValType::I64],
            results: vec![ValType::I64],
        };
        let sig_exit = FuncType {
            params: vec![ValType::I32],
            results: vec![],
        };
        let block = Block {
            params: vec![ValType::I32], // v0 = a capability handle
            insts: vec![
                Inst::ConstI64(0), // v1 = buf
                Inst::ConstI64(3), // v2 = len
                Inst::CallSym {
                    // v3 = write(handle=v0, v1, v2)
                    import: 0,
                    sig: sig_write.clone(),
                    handle: 0,
                    args: vec![1, 2],
                },
                Inst::ConstI32(0), // v4 = exit code
                Inst::CallSym {
                    // exit(handle=v0, v4)
                    import: 1,
                    sig: sig_exit.clone(),
                    handle: 0,
                    args: vec![4],
                },
            ],
            term: Terminator::Unreachable,
        };
        let mut m = Module {
            data_ptrs: Vec::new(),
            data_funcrefs: Vec::new(),
            types: vec![],
            funcs: vec![Func {
                params: vec![ValType::I32],
                results: vec![],
                blocks: vec![block],
            }],
            memory: None,
            data: vec![],
            imports: vec![],
            exports: vec![],
            data_exports: vec![],
            impl_exports: vec![],
            debug_info: None,
        };
        m.add_func_import("write", sig_write, ImportMode::Required);
        m.add_func_import("exit", sig_exit, ImportMode::Required);
        m
    }

    // The host policy under test: "write" → (Stream=0, op 1), "exit" → (Exit=1, op 0).
    fn policy(name: &str) -> Option<ResolvedCap> {
        match name {
            "write" => Some(ResolvedCap { type_id: 0, op: 1 }),
            "exit" => Some(ResolvedCap { type_id: 1, op: 0 }),
            _ => None,
        }
    }

    #[test]
    fn resolves_callsyms_to_capcalls() {
        let m = module_with_imports();
        let r = resolve_imports_with(&m, |n| policy(n).map(Resolved::Cap)).expect("resolve");
        // Import section is gone; the module is now backend-ready.
        assert!(r.imports.is_empty());
        let insts = &r.funcs[0].blocks[0].insts;
        // No CallSym survives.
        assert!(
            !insts.iter().any(|i| matches!(i, Inst::CallSym { .. })),
            "all symbols must be lowered"
        );
        // "write" became cap.call 0 1 on handle v0 with args [1,2].
        match &insts[2] {
            Inst::CapCall {
                type_id,
                op,
                handle,
                args,
                sig,
            } => {
                assert_eq!((*type_id, *op, *handle), (0, 1, 0));
                assert_eq!(args, &vec![1, 2]);
                assert_eq!(sig.results.len(), 1);
            }
            other => panic!("expected CapCall, got {other:?}"),
        }
        // "exit" became cap.call 1 0.
        match &insts[4] {
            Inst::CapCall {
                type_id, op, args, ..
            } => {
                assert_eq!((*type_id, *op), (1, 0));
                assert_eq!(args, &vec![4]);
            }
            other => panic!("expected CapCall, got {other:?}"),
        }
    }

    #[test]
    fn unresolved_import_fails_closed() {
        let m = module_with_imports();
        // A policy that knows "write" but not "exit" must error, not silently drop it.
        let err = resolve_imports_with(&m, |n| {
            (n == "write").then_some(Resolved::Cap(ResolvedCap { type_id: 0, op: 1 }))
        })
        .expect_err("must fail closed");
        assert_eq!(err, ImportError::Unresolved("exit".into()));
    }

    #[test]
    fn module_without_imports_is_unchanged() {
        let mut m = module_with_imports();
        // Replace the import calls with a plain return so there's nothing to resolve.
        m.imports.clear();
        m.funcs[0].blocks[0].insts.clear();
        m.funcs[0].blocks[0].term = Terminator::Return(vec![]);
        let r = resolve_imports_with(&m, |n| policy(n).map(Resolved::Cap)).expect("resolve");
        assert_eq!(r, m, "a no-import module round-trips identically");
    }
}

#[cfg(test)]
mod link_layout_tests {
    use super::*;

    /// A minimal link unit: one linear-memory window and a single data segment (no funcs/exports),
    /// enough to exercise the linker's data-stacking layout.
    fn data_unit(seg_offset: u64, len: usize, readonly: bool) -> LinkUnit {
        LinkUnit {
            module: Module {
                data_ptrs: Vec::new(),
                data_funcrefs: Vec::new(),
                types: vec![],
                funcs: vec![],
                memory: Some(Memory { size_log2: 16 }),
                data: vec![Data {
                    offset: seg_offset,
                    readonly,
                    bytes: vec![0u8; len],
                }],
                imports: vec![],
                exports: vec![],
                data_exports: vec![],
                impl_exports: vec![],
                debug_info: None,
            },
            exports: vec![],
            data_exports: vec![],
        }
    }

    /// The confinement property the linker must preserve when it stacks units: the runtime applies the
    /// read-only-segment protection at **host-page** granularity (it rounds a read-only segment up to
    /// the host page and marks the whole page `PROT_READ`), so **no host page may hold bytes from both
    /// a read-only and a writable segment** — otherwise the writable data's first store faults. A
    /// frontend guarantees this within one unit; the linker must not break it *across* units by packing
    /// one unit's writable head onto the same page as the previous unit's read-only tail. Regression
    /// guard for the emit-object self-host libc, whose writable arena landed on the entry unit's
    /// read-only symbol page under the old 16-byte-tight stacking (task #20).
    #[test]
    fn stacked_units_never_share_a_host_page_between_ro_and_rw() {
        // Unit A ends in read-only data whose tail spills into a second 4 KiB page; unit B begins with
        // writable data at a low in-unit offset. 16-byte-tight, B's store page would coincide with A's
        // read-only page; page-aligned unit bases keep them apart.
        let a = data_unit(0, 4096 + 32, true); // read-only
        let b = data_unit(32, 64, false); // writable, low offset
        let linked = link_with_manifest(&[a, b]).expect("link ro-provider + rw-consumer");

        // No 64 KiB host page (the max host page, [`POWERBOX_STACK_ALIGN`]) mixes ro and rw bytes.
        let page = POWERBOX_STACK_ALIGN;
        let mut ro_pages = alloc::collections::BTreeSet::new();
        let mut rw_pages = alloc::collections::BTreeSet::new();
        for d in &linked.data {
            if d.bytes.is_empty() {
                continue;
            }
            let first = d.offset / page;
            let last = (d.offset + d.bytes.len() as u64 - 1) / page;
            for p in first..=last {
                if d.readonly {
                    ro_pages.insert(p);
                } else {
                    rw_pages.insert(p);
                }
            }
        }
        let shared: Vec<_> = ro_pages.intersection(&rw_pages).collect();
        assert!(
            shared.is_empty(),
            "no host page may mix read-only and writable data; shared pages: {shared:?}"
        );

        // Concretely: unit B was relocated onto its own page-aligned base (the mechanism), so its
        // writable segment keeps its in-unit offset (32) modulo the page and sits above unit A.
        let rw = linked
            .data
            .iter()
            .find(|d| !d.readonly)
            .expect("writable segment present");
        assert_eq!(
            rw.offset % page,
            32,
            "unit base page-aligned; seg keeps in-unit offset 32"
        );
        assert!(
            rw.offset >= page,
            "unit B relocated to a fresh host page above unit A"
        );
    }
}

#[cfg(test)]
mod effects_tests {
    use super::*;

    fn sig() -> FuncType {
        FuncType {
            params: vec![],
            results: vec![ValType::I64],
        }
    }

    #[test]
    fn pure_ops_have_no_effects_and_are_removable() {
        for inst in [
            Inst::ConstI32(0),
            Inst::ConstI64(0),
            Inst::ConstF64(0),
            Inst::ConstV128([0; 16]),
            Inst::IntBin {
                ty: IntTy::I32,
                op: BinOp::Add,
                a: 0,
                b: 1,
            },
            Inst::IntCmp {
                ty: IntTy::I32,
                op: CmpOp::Eq,
                a: 0,
                b: 1,
            },
            Inst::Select {
                cond: 0,
                a: 1,
                b: 2,
            },
            Inst::FToISat {
                op: FToI::F64I32S,
                a: 0,
            },
            Inst::RefFunc { func: 0 },
            Inst::Splat {
                shape: VShape::I32x4,
                a: 0,
            },
        ] {
            let e = inst.effects();
            assert!(e.is_pure(), "{inst:?} should be pure: {e:?}");
            assert!(e.removable_if_dead(), "{inst:?} should be removable");
        }
    }

    #[test]
    fn div_rem_trap_but_add_does_not() {
        for op in [BinOp::DivS, BinOp::DivU, BinOp::RemS, BinOp::RemU] {
            let e = Inst::IntBin {
                ty: IntTy::I32,
                op,
                a: 0,
                b: 1,
            }
            .effects();
            assert!(e.can_trap, "{op:?} traps");
            assert!(!e.removable_if_dead(), "{op:?} not removable (can trap)");
        }
        let add = Inst::IntBin {
            ty: IntTy::I32,
            op: BinOp::Add,
            a: 0,
            b: 1,
        }
        .effects();
        assert!(!add.can_trap && add.is_pure());
    }

    #[test]
    fn loads_read_and_trap_stores_write_and_trap() {
        let load = Inst::Load {
            op: LoadOp::I64,
            addr: 0,
            offset: 0,
        }
        .effects();
        assert!(load.can_trap && load.reads_mem && !load.writes_mem && !load.side_effect);
        assert!(!load.removable_if_dead(), "a load can trap → kept");

        let store = Inst::Store {
            op: StoreOp::I64,
            addr: 0,
            value: 1,
            offset: 0,
        }
        .effects();
        assert!(store.can_trap && store.writes_mem && !store.reads_mem);
        assert!(!store.removable_if_dead(), "a store writes memory → kept");
    }

    #[test]
    fn atomics_are_barriers() {
        let al = Inst::AtomicLoad {
            ty: IntTy::I32,
            addr: 0,
            offset: 0,
        }
        .effects();
        assert!(al.reads_mem && al.side_effect && !al.removable_if_dead());
        let rmw = Inst::AtomicRmw {
            ty: IntTy::I32,
            op: AtomicRmwOp::Add,
            addr: 0,
            value: 1,
            offset: 0,
        }
        .effects();
        assert!(rmw.reads_mem && rmw.writes_mem && rmw.side_effect && rmw.can_trap);
        let fence = Inst::AtomicFence {
            order: Ordering::SeqCst,
        }
        .effects();
        assert!(fence.side_effect && !fence.reads_mem && !fence.writes_mem && !fence.can_trap);
    }

    #[test]
    fn calls_and_control_ops_are_full_clobbers() {
        for inst in [
            Inst::Call {
                func: 0,
                args: vec![],
            },
            Inst::CallIndirect {
                ty: sig(),
                idx: 0,
                args: vec![],
            },
            Inst::CapCall {
                type_id: 0,
                op: 0,
                sig: sig(),
                handle: 0,
                args: vec![],
            },
            Inst::ContResume {
                k: 0,
                arg: 1,
                block: false,
            },
            Inst::ThreadJoin { handle: 0 },
            Inst::GcRoots {
                heap_lo: 0,
                heap_hi: 1,
                mask: 2,
                buf: 3,
                cap: 4,
            },
        ] {
            let e = inst.effects();
            assert!(e.side_effect, "{inst:?} must be a barrier: {e:?}");
            assert!(!e.removable_if_dead(), "{inst:?} must be kept");
            assert!(!e.is_pure());
        }
    }

    #[test]
    fn ambient_intrinsics_carry_a_side_effect_but_no_guest_memory() {
        // `vcpu.tls`/durable-shadow read or write *runtime* state, not guest memory. (The `cap.self.*`
        // reflection ops are now `cap.call CAP_SELF` and take the generic full-clobber `CapCall`
        // effects.)
        let tls = Inst::VcpuTlsGet.effects();
        assert!(tls.side_effect && !tls.reads_mem && !tls.writes_mem && !tls.can_trap);
        assert!(!tls.removable_if_dead());
        let shadow = Inst::DurableShadowBase.effects();
        assert!(shadow.side_effect && !shadow.can_trap);
    }

    #[test]
    fn pure_implies_removable_for_every_representative() {
        // The load-bearing invariant every pass leans on: a pure op is always removable-if-dead.
        for inst in [
            Inst::ConstI32(1),
            Inst::IntUn {
                ty: IntTy::I64,
                op: IntUnOp::Clz,
                a: 0,
            },
            Inst::Cast {
                op: CastOp::ReinterpF64I64,
                a: 0,
            },
            Inst::VNot { a: 0 },
        ] {
            let e = inst.effects();
            assert_eq!(
                e.is_pure(),
                e.is_pure() && e.removable_if_dead(),
                "pure ⇒ removable for {inst:?}"
            );
        }
    }
}
