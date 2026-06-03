//! Text format for the IR — a CLIF/LLVM-flavored form 1:1 with the binary
//! (`DESIGN.md` §3a, "text format first"). This is the human/agent debugging
//! interface and the source for hand-written test corpora.
//!
//! It is a *dev tool*, not escape-TCB: the binary decoder (`svm-encode`) is the
//! untrusted-input path. The parser still returns `Result` and never panics, but it
//! need not be exhaustively hardened. The printer normalizes value/block names to
//! `vN`/`blockN`, so a parse→print→parse round-trip is identity at the IR level.
//!
//! Example:
//! ```text
//! func (i32) -> (i32) {
//! block0(v0: i32):
//!   v1 = i32.const 10
//!   v2 = i32.add v0 v1
//!   return v2
//! }
//! ```
#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::fmt::Write as _;

use svm_ir::{Block, Func, Inst, Module, Terminator, ValType};

/// Parse error with a human-readable message (dev tool; not safety-load-bearing).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ParseError(pub String);

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "parse error: {}", self.0)
    }
}

fn err<T>(msg: impl Into<String>) -> Result<T, ParseError> {
    Err(ParseError(msg.into()))
}

// ----------------------------------------------------------------------------
// Printing
// ----------------------------------------------------------------------------

/// Render a module to canonical text.
pub fn print_module(m: &Module) -> String {
    let mut s = String::new();
    for (i, f) in m.funcs.iter().enumerate() {
        if i > 0 {
            s.push('\n');
        }
        print_func(&mut s, f);
    }
    s
}

fn print_func(s: &mut String, f: &Func) {
    let _ = writeln!(
        s,
        "func ({}) -> ({}) {{",
        types(&f.params),
        types(&f.results)
    );
    for (bi, b) in f.blocks.iter().enumerate() {
        // Block header with named, typed parameters.
        let base = 0u32; // params are indices 0..k within the block
        let params: Vec<String> = b
            .params
            .iter()
            .enumerate()
            .map(|(i, t)| format!("v{}: {}", base + i as u32, t.as_str()))
            .collect();
        let _ = writeln!(s, "block{}({}):", bi, params.join(", "));

        let mut next = b.params.len() as u32; // next value index in this block
        for inst in &b.insts {
            let _ = writeln!(s, "  v{} = {}", next, print_inst(inst));
            next += 1;
        }
        let _ = writeln!(s, "  {}", print_term(&b.term));
    }
    s.push_str("}\n");
}

fn print_inst(inst: &Inst) -> String {
    match inst {
        Inst::I32Const(c) => format!("i32.const {c}"),
        Inst::I64Const(c) => format!("i64.const {c}"),
        Inst::I32Add(a, b) => format!("i32.add v{a} v{b}"),
        Inst::I64Add(a, b) => format!("i64.add v{a} v{b}"),
    }
}

fn print_term(t: &Terminator) -> String {
    match t {
        Terminator::Br { target, args } => format!("br block{target}{}", arglist(args)),
        Terminator::BrIf {
            cond,
            then_blk,
            then_args,
            else_blk,
            else_args,
        } => format!(
            "br_if v{cond} block{then_blk}{} block{else_blk}{}",
            arglist(then_args),
            arglist(else_args)
        ),
        Terminator::Return(vals) => {
            if vals.is_empty() {
                "return".to_string()
            } else {
                let vs: Vec<String> = vals.iter().map(|v| format!("v{v}")).collect();
                format!("return {}", vs.join(", "))
            }
        }
    }
}

fn arglist(args: &[u32]) -> String {
    let vs: Vec<String> = args.iter().map(|v| format!("v{v}")).collect();
    format!("({})", vs.join(", "))
}

fn types(ts: &[ValType]) -> String {
    let v: Vec<&str> = ts.iter().map(|t| t.as_str()).collect();
    v.join(", ")
}

// ----------------------------------------------------------------------------
// Tokenizing
// ----------------------------------------------------------------------------

#[derive(Clone, PartialEq, Debug)]
enum Tok {
    LParen,
    RParen,
    LBrace,
    RBrace,
    Colon,
    Comma,
    Equals,
    Arrow,
    Ident(String),
    Int(i64),
}

fn tokenize(src: &str) -> Result<Vec<Tok>, ParseError> {
    let bytes = src.as_bytes();
    let mut i = 0;
    let mut toks = Vec::new();
    while i < bytes.len() {
        let c = bytes[i];
        match c {
            b' ' | b'\t' | b'\r' | b'\n' => i += 1,
            b';' => {
                // line comment to end of line
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                }
            }
            b'(' => {
                toks.push(Tok::LParen);
                i += 1;
            }
            b')' => {
                toks.push(Tok::RParen);
                i += 1;
            }
            b'{' => {
                toks.push(Tok::LBrace);
                i += 1;
            }
            b'}' => {
                toks.push(Tok::RBrace);
                i += 1;
            }
            b':' => {
                toks.push(Tok::Colon);
                i += 1;
            }
            b',' => {
                toks.push(Tok::Comma);
                i += 1;
            }
            b'=' => {
                toks.push(Tok::Equals);
                i += 1;
            }
            b'-' => {
                if i + 1 < bytes.len() && bytes[i + 1] == b'>' {
                    toks.push(Tok::Arrow);
                    i += 2;
                } else {
                    // negative integer
                    let (tok, ni) = lex_int(bytes, i)?;
                    toks.push(tok);
                    i = ni;
                }
            }
            b'0'..=b'9' => {
                let (tok, ni) = lex_int(bytes, i)?;
                toks.push(tok);
                i = ni;
            }
            _ if is_ident_start(c) => {
                let start = i;
                while i < bytes.len() && is_ident_char(bytes[i]) {
                    i += 1;
                }
                let s = std::str::from_utf8(&bytes[start..i])
                    .map_err(|_| ParseError("non-utf8 identifier".into()))?;
                toks.push(Tok::Ident(s.to_string()));
            }
            _ => return err(format!("unexpected character {:?}", c as char)),
        }
    }
    Ok(toks)
}

fn lex_int(bytes: &[u8], start: usize) -> Result<(Tok, usize), ParseError> {
    let mut i = start;
    if bytes[i] == b'-' {
        i += 1;
    }
    let digits_start = i;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    if i == digits_start {
        return err("expected digits in integer");
    }
    let s = std::str::from_utf8(&bytes[start..i]).unwrap_or("");
    let v: i64 = s
        .parse()
        .map_err(|_| ParseError(format!("integer out of range: {s}")))?;
    Ok((Tok::Int(v), i))
}

fn is_ident_start(c: u8) -> bool {
    c.is_ascii_alphabetic() || c == b'_' || c == b'%'
}
fn is_ident_char(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_' || c == b'.'
}

// ----------------------------------------------------------------------------
// Parsing
// ----------------------------------------------------------------------------

/// Parse a module from text.
pub fn parse_module(src: &str) -> Result<Module, ParseError> {
    let toks = tokenize(src)?;
    let mut p = Parser {
        toks: &toks,
        pos: 0,
    };
    let mut funcs = Vec::new();
    while !p.at_end() {
        funcs.push(p.parse_func()?);
    }
    Ok(Module { funcs })
}

struct Parser<'a> {
    toks: &'a [Tok],
    pos: usize,
}

/// Intermediate block whose terminator still refers to block labels by name.
struct PBlock {
    label: String,
    params: Vec<ValType>,
    insts: Vec<Inst>,
    term: PTerm,
}

enum PTerm {
    Br(String, Vec<u32>),
    BrIf {
        cond: u32,
        then: (String, Vec<u32>),
        els: (String, Vec<u32>),
    },
    Return(Vec<u32>),
}

impl<'a> Parser<'a> {
    fn at_end(&self) -> bool {
        self.pos >= self.toks.len()
    }

    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }

    fn next(&mut self) -> Result<&Tok, ParseError> {
        let t = self
            .toks
            .get(self.pos)
            .ok_or(ParseError("unexpected end of input".into()))?;
        self.pos += 1;
        Ok(t)
    }

    fn expect(&mut self, want: &Tok) -> Result<(), ParseError> {
        let got = self.next()?;
        if got == want {
            Ok(())
        } else {
            err(format!("expected {want:?}, found {got:?}"))
        }
    }

    fn ident(&mut self) -> Result<String, ParseError> {
        match self.next()? {
            Tok::Ident(s) => Ok(s.clone()),
            other => err(format!("expected identifier, found {other:?}")),
        }
    }

    fn parse_func(&mut self) -> Result<Func, ParseError> {
        // `func (types) -> (types) { blocks }`
        let kw = self.ident()?;
        if kw != "func" {
            return err(format!("expected `func`, found `{kw}`"));
        }
        let params = self.parse_type_list()?;
        self.expect(&Tok::Arrow)?;
        let results = self.parse_type_list()?;
        self.expect(&Tok::LBrace)?;

        let mut pblocks = Vec::new();
        while self.peek() != Some(&Tok::RBrace) {
            if self.at_end() {
                return err("unterminated function body");
            }
            pblocks.push(self.parse_block()?);
        }
        self.expect(&Tok::RBrace)?;

        // Resolve block labels to indices.
        let mut labels: HashMap<String, u32> = HashMap::new();
        for (i, b) in pblocks.iter().enumerate() {
            if labels.insert(b.label.clone(), i as u32).is_some() {
                return err(format!("duplicate block label `{}`", b.label));
            }
        }
        let resolve = |name: &str| -> Result<u32, ParseError> {
            labels
                .get(name)
                .copied()
                .ok_or_else(|| ParseError(format!("unknown block label `{name}`")))
        };

        let mut blocks = Vec::new();
        for b in pblocks {
            let term = match b.term {
                PTerm::Br(t, args) => Terminator::Br {
                    target: resolve(&t)?,
                    args,
                },
                PTerm::BrIf { cond, then, els } => Terminator::BrIf {
                    cond,
                    then_blk: resolve(&then.0)?,
                    then_args: then.1,
                    else_blk: resolve(&els.0)?,
                    else_args: els.1,
                },
                PTerm::Return(v) => Terminator::Return(v),
            };
            blocks.push(Block {
                params: b.params,
                insts: b.insts,
                term,
            });
        }

        Ok(Func {
            params,
            results,
            blocks,
        })
    }

    fn parse_type_list(&mut self) -> Result<Vec<ValType>, ParseError> {
        self.expect(&Tok::LParen)?;
        let mut ts = Vec::new();
        while self.peek() != Some(&Tok::RParen) {
            if !ts.is_empty() {
                self.expect(&Tok::Comma)?;
            }
            ts.push(self.parse_type()?);
        }
        self.expect(&Tok::RParen)?;
        Ok(ts)
    }

    fn parse_type(&mut self) -> Result<ValType, ParseError> {
        let s = self.ident()?;
        ValType::from_str(&s).ok_or_else(|| ParseError(format!("unknown type `{s}`")))
    }

    fn parse_block(&mut self) -> Result<PBlock, ParseError> {
        // `label(name: type, ...):` then instruction lines, then a terminator.
        let label = self.ident()?;
        // Per-block value-name table; parameters take indices 0..k.
        let mut names: HashMap<String, u32> = HashMap::new();
        let mut params = Vec::new();

        self.expect(&Tok::LParen)?;
        while self.peek() != Some(&Tok::RParen) {
            if !params.is_empty() {
                self.expect(&Tok::Comma)?;
            }
            let n = self.ident()?;
            self.expect(&Tok::Colon)?;
            let t = self.parse_type()?;
            let idx = params.len() as u32;
            if names.insert(n.clone(), idx).is_some() {
                return err(format!("duplicate value name `{n}`"));
            }
            params.push(t);
        }
        self.expect(&Tok::RParen)?;
        self.expect(&Tok::Colon)?;

        let mut insts = Vec::new();
        let mut next_idx = params.len() as u32;

        // Parse instruction lines until we hit a terminator keyword.
        loop {
            let kw = match self.peek() {
                Some(Tok::Ident(s)) => s.clone(),
                _ => return err("expected instruction or terminator"),
            };
            match kw.as_str() {
                "br" | "br_if" | "return" => {
                    let term = self.parse_term(&names)?;
                    return Ok(PBlock {
                        label,
                        params,
                        insts,
                        term,
                    });
                }
                _ => {
                    // `name = opcode operands`
                    let name = self.ident()?;
                    self.expect(&Tok::Equals)?;
                    let inst = self.parse_inst(&names)?;
                    if names.insert(name.clone(), next_idx).is_some() {
                        return err(format!("duplicate value name `{name}`"));
                    }
                    next_idx += 1;
                    insts.push(inst);
                }
            }
        }
    }

    fn parse_inst(&mut self, names: &HashMap<String, u32>) -> Result<Inst, ParseError> {
        let op = self.ident()?;
        Ok(match op.as_str() {
            "i32.const" => Inst::I32Const(self.parse_i32()?),
            "i64.const" => Inst::I64Const(self.parse_int()?),
            "i32.add" => {
                let a = self.value(names)?;
                let b = self.value(names)?;
                Inst::I32Add(a, b)
            }
            "i64.add" => {
                let a = self.value(names)?;
                let b = self.value(names)?;
                Inst::I64Add(a, b)
            }
            other => return err(format!("unknown opcode `{other}`")),
        })
    }

    fn parse_term(&mut self, names: &HashMap<String, u32>) -> Result<PTerm, ParseError> {
        let kw = self.ident()?;
        Ok(match kw.as_str() {
            "br" => {
                let (label, args) = self.parse_target(names)?;
                PTerm::Br(label, args)
            }
            "br_if" => {
                let cond = self.value(names)?;
                let then = self.parse_target(names)?;
                let els = self.parse_target(names)?;
                PTerm::BrIf { cond, then, els }
            }
            "return" => {
                let mut vals = Vec::new();
                // Optional comma-separated value list until the next block/`}`.
                while matches!(self.peek(), Some(Tok::Ident(_))) {
                    if !vals.is_empty() {
                        // tolerate optional commas
                        if self.peek() == Some(&Tok::Comma) {
                            self.pos += 1;
                        }
                    }
                    // Stop if this ident is actually a block label followed by `(`
                    // — but returns end a block, so any following ident starts a new
                    // block header; we must not consume it. Disambiguate: a return
                    // operand is a known value name; a new block label is not.
                    if let Some(Tok::Ident(s)) = self.peek() {
                        if !names.contains_key(s) {
                            break;
                        }
                    }
                    vals.push(self.value(names)?);
                    if self.peek() == Some(&Tok::Comma) {
                        self.pos += 1;
                    }
                }
                PTerm::Return(vals)
            }
            other => return err(format!("unknown terminator `{other}`")),
        })
    }

    /// Parse `label(arg, arg, ...)`.
    fn parse_target(
        &mut self,
        names: &HashMap<String, u32>,
    ) -> Result<(String, Vec<u32>), ParseError> {
        let label = self.ident()?;
        let mut args = Vec::new();
        self.expect(&Tok::LParen)?;
        while self.peek() != Some(&Tok::RParen) {
            if !args.is_empty() {
                self.expect(&Tok::Comma)?;
            }
            args.push(self.value(names)?);
        }
        self.expect(&Tok::RParen)?;
        Ok((label, args))
    }

    fn value(&mut self, names: &HashMap<String, u32>) -> Result<u32, ParseError> {
        let n = self.ident()?;
        names
            .get(&n)
            .copied()
            .ok_or_else(|| ParseError(format!("unknown value `{n}`")))
    }

    fn parse_int(&mut self) -> Result<i64, ParseError> {
        match self.next()? {
            Tok::Int(v) => Ok(*v),
            other => err(format!("expected integer, found {other:?}")),
        }
    }

    fn parse_i32(&mut self) -> Result<i32, ParseError> {
        let v = self.parse_int()?;
        i32::try_from(v).map_err(|_| ParseError(format!("i32 constant out of range: {v}")))
    }
}
