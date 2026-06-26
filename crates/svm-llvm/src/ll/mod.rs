//! A dependency-free, in-house reader for **textual LLVM IR** (`.ll`), replacing the `llvm-ir` +
//! libLLVM binding (LLVM.md §8 Q1a). Two problems with that binding drove this: it is **lossy**
//! (collapses integer constants to a `u64`, truncating `i128` literals — ISSUES.md I14) and
//! **version-locked** (`llvm-ir` tops out at LLVM 19, while `rustc`/`clang` march on). Textual IR
//! carries full-width constants and is far more version-stable than the bitcode format / C API, and
//! reading it needs no libLLVM link.
//!
//! Shape: [`ast`] mirrors the slice of `llvm-ir`'s data model the translator consumes (same variant
//! and field names, same `get_type`/`try_get_result`/… methods), so `lib.rs`'s ~17k-line
//! pattern-match-and-emit walk is unchanged — only its `use llvm_ir::…` becomes `use crate::ll::…`.
//! [`lex`] tokenizes the `.ll` text; [`parse`] is a recursive-descent parser producing [`ast::Module`].
//! Anything outside the subset the on-ramp translates fails closed (a clean parse/Unsupported error,
//! re-verified downstream — §2a), exactly as the bitcode path does today.
//!
//! Migration status: built behind a differential parity check against `llvm-ir` (same source → both
//! readers → identical svm-ir) before it becomes the default; see `tests/translate.rs`.

pub mod ast;
pub mod lex;
pub mod parse;

pub use ast::Module;
