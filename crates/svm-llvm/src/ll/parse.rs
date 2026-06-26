//! Recursive-descent parser for textual LLVM IR (`.ll`) → [`ast::Module`](super::ast::Module).
//! (PR1, in progress — built incrementally simplest-first and gated by a differential parity check
//! against `llvm-ir` in `tests/translate.rs`.)

use super::ast::Module;

/// A parse error with a byte offset into the source.
#[derive(Debug)]
pub struct ParseError {
    pub offset: usize,
    pub msg: String,
}

/// Parse `.ll` source text into a [`Module`]. (PR1 stub — implementation lands next.)
pub fn parse_module(_src: &str) -> Result<Module, ParseError> {
    Err(ParseError {
        offset: 0,
        msg: "ll::parse not yet implemented".into(),
    })
}
