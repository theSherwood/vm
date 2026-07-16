#![forbid(unsafe_code)]
#![cfg_attr(not(test), no_std)]
//! `svm-peval` — the partial-evaluation / Futamura on-ramp (see `DESIGN.md` §20c).
//!
//! The **specializer** ([`specialize`]) — the first Futamura projection: turn an interpreter + a
//! fixed program (in readonly "constant memory") into a compiled residual, with the dispatch loop,
//! opcode decode, and program-walking folded away. `spec(interp, program)(input) ≡
//! interp(program, input)`. See [`mod@specialize`].
//!
//! The **generic IR→IR optimizer** it builds on now lives in the [`svm_opt`] crate: the specializer
//! reuses that crate's constant-fold machinery ([`svm_opt::Known`] + the `fold_*` helpers) and its
//! CFG cleanup ([`optimize_module`], re-exported here so existing consumers keep their `svm_peval::`
//! path). The split keeps this crate about specialization while the optimizer grows independently
//! (see `OPT.md`).
//!
//! **Untrusted for escape (§2a / §20a posture).** Like the LLVM on-ramp, this pass is *not* in the
//! escape-TCB: its output is re-verified with `svm_verify::verify_module` before it runs, so a bug
//! here is a clean verify error, never an escape.
//!
//! **`no_std` + `alloc`.** This crate compiles `no_std` (gated on `not(test)`; its own test harness
//! gets `std`) so it can itself be translated to svm-IR through the Rust on-ramp and run *inside* svm
//! (DESIGN.md §20c). The `libm-floats` feature is forwarded to [`svm_opt`] (the in-svm build drops it,
//! since libm's `fma` brings untranslatable x86 inline-asm + i128).

extern crate alloc;

mod specialize;
pub use specialize::{
    specialize, specialize_with, specialize_with_config, SpecArg, SpecConfig, SpecError,
};

// The generic optimizer + remap helpers moved to `svm-opt` (see `OPT.md` Phase 0). Re-exported so
// existing consumers (`svm-run`, the `svm-llvm` demos, this crate's own tests) keep working through
// the `svm_peval::` path.
pub use svm_opt::{
    is_removable_if_dead, map_operands, map_term_operands, optimize_func, optimize_func_with,
    optimize_module, optimize_module_with, OptConfig,
};

// The constant-fold machinery the specializer reuses. `pub(crate)` re-exports keep the specializer's
// `crate::fold_*` / `crate::Known` paths resolving without widening this crate's public surface.
pub(crate) use svm_opt::{
    fold_cast, fold_fbin, fold_fcmp, fold_fma, fold_ftoi_sat, fold_ftoi_trap, fold_fun,
    fold_int_bin, fold_int_cmp, fold_int_un, fold_itof, fold_simd, Known,
};
