//! Tokenizer for textual LLVM IR (`.ll`). Splits the source into [`Token`]s the [`parse`](super::parse)
//! recursive-descent reader consumes.
//!
//! The lexer is **context-free and lossless**: it never decodes a string's escapes (a `c"…"` array can
//! hold arbitrary bytes, so decoding is the *parser*'s job once it knows whether the text is a UTF-8
//! name or a raw byte array — see [`unescape`]), and it keeps integer/float literals as their source
//! text so a full-width `i128` value never truncates on the way in (LLVM.md §8 Q1a / ISSUES.md I14).

use std::str;

/// A lexical token of `.ll` source.
#[derive(Clone, Debug, PartialEq)]
pub enum Token {
    /// `%name` / `%3` / `%"quoted"` — a local value or block label (sigil stripped; quotes stripped,
    /// escapes left encoded).
    Local(String),
    /// `@name` / `@"quoted"` — a global value/function (sigil + quotes stripped, escapes encoded).
    Global(String),
    /// `!name` / `!3` / `!"str"` — a metadata reference (sigil stripped). The bare `!` sigil that opens
    /// an inline node (`!{…}`) lexes as `Meta("")` followed by `{`.
    Meta(String),
    /// An integer literal (with optional leading `-`), kept as text so a full-width `i128` value never
    /// truncates (I14). The parser converts to a `u128` two's-complement pattern given the bit width.
    Int(String),
    /// A floating-point literal — decimal (`1.5`, `-1.5e10`) or LLVM's `0x…` hex bit-pattern form
    /// (incl. the `0xK`/`0xL`/`0xM`/`0xR`/`0xH` wide-FP prefixes) — kept as text.
    Float(String),
    /// A `"…"` string literal with the quotes stripped but **escapes left encoded** (`\XX` / `\\`).
    /// Decode with [`unescape`] in the parser, where the target (UTF-8 name vs. raw byte array) is known.
    Str(String),
    /// A bareword: a keyword (`define`, `ret`, …), a type (`i32`, `ptr`, `float`), an attribute, or an
    /// attribute-group ref (`#0`).
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
    /// `:` — terminates a basic-block label (`entry:`, `5:`) and separates `key: value` in a
    /// specialized metadata node (`!DILocation(line: 1, …)`).
    Colon,
    /// `|` — the flag-set separator in debug-info metadata (`DISPFlagLocalToUnit | DISPFlagDefinition`).
    Pipe,
    /// `...` (varargs marker).
    Ellipsis,
}

/// A lex error with a byte offset into the source (for diagnostics).
#[derive(Debug)]
pub struct LexError {
    pub offset: usize,
    pub msg: String,
}

/// First char of an unquoted LLVM identifier / bareword: `[A-Za-z$._]` (digits and `-` may follow but
/// not lead an unquoted name; numbered names like `%3` are read explicitly after a sigil).
fn is_ident_start(c: u8) -> bool {
    c.is_ascii_alphabetic() || matches!(c, b'$' | b'.' | b'_')
}

/// A subsequent char of an unquoted LLVM identifier: `[-A-Za-z0-9$._]`.
fn is_ident_char(c: u8) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, b'-' | b'$' | b'.' | b'_')
}

/// Tokenize `.ll` source into a flat [`Token`] stream (comments and whitespace dropped).
pub fn lex(src: &str) -> Result<Vec<Token>, LexError> {
    let b = src.as_bytes();
    let n = b.len();
    let mut i = 0usize;
    let mut out = Vec::new();

    while i < n {
        let c = b[i];
        match c {
            b' ' | b'\t' | b'\r' | b'\n' => i += 1,
            // Line comment to end of line.
            b';' => {
                while i < n && b[i] != b'\n' {
                    i += 1;
                }
            }
            b'(' => {
                out.push(Token::LParen);
                i += 1;
            }
            b')' => {
                out.push(Token::RParen);
                i += 1;
            }
            b'{' => {
                out.push(Token::LBrace);
                i += 1;
            }
            b'}' => {
                out.push(Token::RBrace);
                i += 1;
            }
            b'[' => {
                out.push(Token::LBracket);
                i += 1;
            }
            b']' => {
                out.push(Token::RBracket);
                i += 1;
            }
            b'<' => {
                out.push(Token::Lt);
                i += 1;
            }
            b'>' => {
                out.push(Token::Gt);
                i += 1;
            }
            b',' => {
                out.push(Token::Comma);
                i += 1;
            }
            b'=' => {
                out.push(Token::Equals);
                i += 1;
            }
            b'*' => {
                out.push(Token::Star);
                i += 1;
            }
            b':' => {
                out.push(Token::Colon);
                i += 1;
            }
            b'|' => {
                out.push(Token::Pipe);
                i += 1;
            }
            // `...` varargs marker (a lone/double `.` outside a name is not valid IR).
            b'.' if i + 2 < n && b[i + 1] == b'.' && b[i + 2] == b'.' => {
                out.push(Token::Ellipsis);
                i += 3;
            }
            b'%' => {
                let (name, ni) = lex_name(b, i + 1, i)?;
                out.push(Token::Local(name));
                i = ni;
            }
            b'@' => {
                let (name, ni) = lex_name(b, i + 1, i)?;
                out.push(Token::Global(name));
                i = ni;
            }
            // Metadata: `!name` / `!3` / `!"str"`, or the bare `!` that opens an inline `!{…}` node.
            b'!' => {
                let j = i + 1;
                if j < n && (is_ident_start(b[j]) || b[j].is_ascii_digit() || b[j] == b'"') {
                    let (name, ni) = lex_name(b, j, i)?;
                    out.push(Token::Meta(name));
                    i = ni;
                } else {
                    out.push(Token::Meta(String::new()));
                    i += 1;
                }
            }
            // Attribute-group ref / id (`#0`) — kept as a bareword so the parser can skip it.
            b'#' => {
                let start = i;
                i += 1;
                while i < n && is_ident_char(b[i]) {
                    i += 1;
                }
                out.push(Token::Word(slice_str(b, start, i)?));
            }
            b'"' => {
                let (s, ni) = lex_quoted(b, i + 1, i)?;
                out.push(Token::Str(s));
                i = ni;
            }
            // A number: optional `-`, then decimal (`1`, `1.5`, `1e9`) or `0x…` hex bit-pattern.
            b'-' | b'0'..=b'9' => {
                let (tok, ni) = lex_number(b, i)?;
                out.push(tok);
                i = ni;
            }
            _ if is_ident_start(c) => {
                let start = i;
                while i < n && is_ident_char(b[i]) {
                    i += 1;
                }
                out.push(Token::Word(slice_str(b, start, i)?));
            }
            _ => {
                return Err(LexError {
                    offset: i,
                    msg: format!("unexpected character {:?}", c as char),
                })
            }
        }
    }

    Ok(out)
}

/// Read the name after a `%`/`@`/`!` sigil at `j`: either a `"quoted"` name (returned with quotes
/// stripped, escapes left encoded) or a run of identifier chars (`foo`, `.str`, `3`). `sigil` is the
/// sigil's offset, used for error reporting.
fn lex_name(b: &[u8], j: usize, sigil: usize) -> Result<(String, usize), LexError> {
    if j < b.len() && b[j] == b'"' {
        return lex_quoted(b, j + 1, sigil);
    }
    let start = j;
    let mut k = j;
    while k < b.len() && is_ident_char(b[k]) {
        k += 1;
    }
    if k == start {
        return Err(LexError {
            offset: sigil,
            msg: "expected a name after sigil".into(),
        });
    }
    Ok((slice_str(b, start, k)?, k))
}

/// Read a `"…"` body whose opening quote was at `open-1`; `j` points just past it. Returns the raw
/// inner text (escapes left encoded) and the index just past the closing quote. A literal `"` never
/// occurs inside (LLVM escapes it as `\22`), so the next `"` always terminates.
fn lex_quoted(b: &[u8], j: usize, open: usize) -> Result<(String, usize), LexError> {
    let start = j;
    let mut k = j;
    while k < b.len() && b[k] != b'"' {
        k += 1;
    }
    if k >= b.len() {
        return Err(LexError {
            offset: open,
            msg: "unterminated string literal".into(),
        });
    }
    Ok((slice_str(b, start, k)?, k + 1))
}

/// Lex a numeric literal starting at `i`. Distinguishes int from float: anything with a `.`, an `e`/`E`
/// exponent, or a `0x` prefix is a [`Token::Float`]; otherwise a [`Token::Int`]. Both keep source text.
fn lex_number(b: &[u8], i: usize) -> Result<(Token, usize), LexError> {
    let n = b.len();
    let start = i;
    let mut k = i;
    if b[k] == b'-' {
        k += 1;
    }
    // `0x…` hex bit-pattern form (always floating-point in `.ll`; ints are printed decimal). The
    // optional wide-FP prefix letter (K/L/M/R/H) precedes the hex digits.
    if k + 1 < n && b[k] == b'0' && (b[k + 1] == b'x' || b[k + 1] == b'X') {
        k += 2;
        if k < n && matches!(b[k], b'K' | b'L' | b'M' | b'R' | b'H') {
            k += 1;
        }
        while k < n && b[k].is_ascii_hexdigit() {
            k += 1;
        }
        return Ok((Token::Float(slice_str(b, start, k)?), k));
    }
    let mut is_float = false;
    while k < n && b[k].is_ascii_digit() {
        k += 1;
    }
    if k < n && b[k] == b'.' {
        is_float = true;
        k += 1;
        while k < n && b[k].is_ascii_digit() {
            k += 1;
        }
    }
    if k < n && (b[k] == b'e' || b[k] == b'E') {
        is_float = true;
        k += 1;
        if k < n && (b[k] == b'+' || b[k] == b'-') {
            k += 1;
        }
        while k < n && b[k].is_ascii_digit() {
            k += 1;
        }
    }
    let text = slice_str(b, start, k)?;
    Ok((
        if is_float {
            Token::Float(text)
        } else {
            Token::Int(text)
        },
        k,
    ))
}

/// `b[start..end]` as an owned `String`, erroring (rather than panicking) on non-UTF-8 source.
fn slice_str(b: &[u8], start: usize, end: usize) -> Result<String, LexError> {
    str::from_utf8(&b[start..end])
        .map(|s| s.to_string())
        .map_err(|_| LexError {
            offset: start,
            msg: "non-UTF-8 bytes in token".into(),
        })
}

/// Decode the `\XX` (two hex digits) and `\\` escapes in a [`Token::Str`]/quoted-name body into the raw
/// bytes it denotes. LLVM's only string escapes are `\XX` and `\\`; a lone `\` not forming either is
/// passed through verbatim (defensive — valid IR never produces one).
pub fn unescape(s: &str) -> Vec<u8> {
    let b = s.as_bytes();
    let n = b.len();
    let mut out = Vec::with_capacity(n);
    let mut i = 0;
    while i < n {
        if b[i] == b'\\' && i + 1 < n {
            if b[i + 1] == b'\\' {
                out.push(b'\\');
                i += 2;
                continue;
            }
            if i + 2 < n {
                let hi = (b[i + 1] as char).to_digit(16);
                let lo = (b[i + 2] as char).to_digit(16);
                if let (Some(hi), Some(lo)) = (hi, lo) {
                    out.push((hi * 16 + lo) as u8);
                    i += 3;
                    continue;
                }
            }
        }
        out.push(b[i]);
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lex_ok(src: &str) -> Vec<Token> {
        lex(src).unwrap_or_else(|e| panic!("lex error at {}: {}", e.offset, e.msg))
    }

    #[test]
    fn punctuation_and_words() {
        use Token::*;
        let toks = lex_ok("define i32 @f() {\nret i32 0\n}");
        assert_eq!(
            toks,
            vec![
                Word("define".into()),
                Word("i32".into()),
                Global("f".into()),
                LParen,
                RParen,
                LBrace,
                Word("ret".into()),
                Word("i32".into()),
                Int("0".into()),
                RBrace,
            ]
        );
    }

    #[test]
    fn comments_and_whitespace_are_dropped() {
        let toks = lex_ok("  ; a comment %not @real\n  ret ; trailing\n");
        assert_eq!(toks, vec![Token::Word("ret".into())]);
    }

    #[test]
    fn sigil_names_numbered_and_quoted() {
        use Token::*;
        let toks = lex_ok(r#"%a %3 @"q name" !dbg !5 !"strmeta""#);
        assert_eq!(
            toks,
            vec![
                Local("a".into()),
                Local("3".into()),
                Global("q name".into()),
                Meta("dbg".into()),
                Meta("5".into()),
                Meta("strmeta".into()),
            ]
        );
    }

    #[test]
    fn inline_metadata_node_sigil() {
        use Token::*;
        // `!{!0, !1}` — the bare `!` lexes as Meta("") then the brace/elements follow.
        assert_eq!(
            lex_ok("!{!0, !1}"),
            vec![
                Meta("".into()),
                LBrace,
                Meta("0".into()),
                Comma,
                Meta("1".into()),
                RBrace,
            ]
        );
    }

    #[test]
    fn integers_keep_full_width_text() {
        use Token::*;
        // A 128-bit literal that would truncate as a u64 (the I14 bug) survives as text.
        let big = "170141183460469231731687303715884105727"; // i128::MAX
        let toks = lex_ok(&format!("i128 {big}, i32 -7"));
        assert_eq!(
            toks,
            vec![
                Word("i128".into()),
                Int(big.into()),
                Comma,
                Word("i32".into()),
                Int("-7".into()),
            ]
        );
    }

    #[test]
    fn floats_decimal_and_hex() {
        use Token::*;
        assert_eq!(
            lex_ok("double 1.5, double -2.5e-3, double 0x7FF8000000000000, x86_fp80 0xK4000C000000000000000"),
            vec![
                Word("double".into()),
                Float("1.5".into()),
                Comma,
                Word("double".into()),
                Float("-2.5e-3".into()),
                Comma,
                Word("double".into()),
                Float("0x7FF8000000000000".into()),
                Comma,
                Word("x86_fp80".into()),
                Float("0xK4000C000000000000000".into()),
            ]
        );
    }

    #[test]
    fn array_string_constant_and_unescape() {
        use Token::*;
        // `c"hi\00"` lexes as Word("c") + Str (escapes encoded); unescape yields the raw bytes.
        let toks = lex_ok(r#"[3 x i8] c"hi\00""#);
        assert_eq!(
            toks,
            vec![
                LBracket,
                Int("3".into()),
                Word("x".into()),
                Word("i8".into()),
                RBracket,
                Word("c".into()),
                Str(r"hi\00".into()),
            ]
        );
        assert_eq!(unescape(r"hi\00"), vec![b'h', b'i', 0]);
        assert_eq!(unescape(r"a\5Cb"), vec![b'a', b'\\', b'b']);
        assert_eq!(unescape(r"x\\y"), vec![b'x', b'\\', b'y']);
    }

    #[test]
    fn labels_and_varargs_and_vectors() {
        use Token::*;
        assert_eq!(
            lex_ok("entry:\n5:\ndeclare i32 @p(ptr, ...)\n<4 x float>"),
            vec![
                Word("entry".into()),
                Colon,
                Int("5".into()),
                Colon,
                Word("declare".into()),
                Word("i32".into()),
                Global("p".into()),
                LParen,
                Word("ptr".into()),
                Comma,
                Ellipsis,
                RParen,
                Lt,
                Int("4".into()),
                Word("x".into()),
                Word("float".into()),
                Gt,
            ]
        );
    }

    #[test]
    fn attribute_group_ref() {
        use Token::*;
        assert_eq!(
            lex_ok("define void @f() #0 {"),
            vec![
                Word("define".into()),
                Word("void".into()),
                Global("f".into()),
                LParen,
                RParen,
                Word("#0".into()),
                LBrace,
            ]
        );
    }

    #[test]
    fn debug_metadata_flag_set() {
        use Token::*;
        // The `spFlags: A | B` flag-set form from `-g` `!DISubprogram` nodes.
        assert_eq!(
            lex_ok("spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2"),
            vec![
                Word("spFlags".into()),
                Colon,
                Word("DISPFlagLocalToUnit".into()),
                Pipe,
                Word("DISPFlagDefinition".into()),
                Comma,
                Word("unit".into()),
                Colon,
                Meta("2".into()),
            ]
        );
    }

    #[test]
    fn unterminated_string_is_an_error() {
        let err = lex(r#"@g = c"oops"#).unwrap_err();
        assert!(err.msg.contains("unterminated"));
    }
}
