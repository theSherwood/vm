//! Tokenizer for textual LLVM IR (`.ll`). Splits the source into [`Token`]s the [`parse`](super::parse)
//! recursive-descent reader consumes. (PR1, in progress — the token set is the seed; it grows with the
//! parser under the parity harness.)

/// A lexical token of `.ll` source.
#[derive(Clone, Debug, PartialEq)]
pub enum Token {
    /// `%name` / `%3` / `%"quoted"` — a local value or block label (sigil stripped).
    Local(String),
    /// `@name` / `@"quoted"` — a global value/function.
    Global(String),
    /// `!name` / `!3` — a metadata reference (sigil stripped).
    Meta(String),
    /// An integer literal, kept as text so a full-width `i128` value never truncates (I14).
    Int(String),
    /// A floating-point literal (decimal or LLVM `0x…` hex form), kept as text.
    Float(String),
    /// A `"…"` string literal (quotes stripped, escapes decoded).
    Str(String),
    /// A bareword: a keyword (`define`, `ret`, …), a type (`i32`, `ptr`, `float`), or an attribute.
    Word(String),
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Lt,
    Gt,
    Comma,
    Equals,
    Star,
    /// `...` (varargs marker).
    Ellipsis,
}

/// A lex error with a byte offset into the source (for diagnostics).
#[derive(Debug)]
pub struct LexError {
    pub offset: usize,
    pub msg: String,
}

/// Tokenize `.ll` source. (PR1 stub — implementation lands next.)
pub fn lex(_src: &str) -> Result<Vec<Token>, LexError> {
    Err(LexError {
        offset: 0,
        msg: "ll::lex not yet implemented".into(),
    })
}
