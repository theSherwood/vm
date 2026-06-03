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

use svm_ir::{BinOp, Block, CmpOp, ConvOp, Func, Inst, IntTy, Module, Terminator, ValType};

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
        // Block header with named, typed parameters (params are indices 0..k).
        let params: Vec<String> = b
            .params
            .iter()
            .enumerate()
            .map(|(i, t)| format!("v{}: {}", i, t.as_str()))
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
        Inst::ConstI32(c) => format!("i32.const {c}"),
        Inst::ConstI64(c) => format!("i64.const {c}"),
        Inst::IntBin { ty, op, a, b } => format!("{}.{} v{a} v{b}", ty.prefix(), op.name()),
        Inst::IntCmp { ty, op, a, b } => format!("{}.{} v{a} v{b}", ty.prefix(), op.name()),
        Inst::Eqz { ty, a } => format!("{}.eqz v{a}", ty.prefix()),
        Inst::Convert { op, a } => format!("{} v{a}", op.sig().0),
        Inst::Select { cond, a, b } => format!("select v{cond} v{a} v{b}"),
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
        Terminator::BrTable {
            idx,
            targets,
            default,
        } => {
            let ts: Vec<String> = targets
                .iter()
                .map(|(t, args)| format!("block{t}{}", arglist(args)))
                .collect();
            format!(
                "br_table v{idx} [{}] block{}{}",
                ts.join(", "),
                default.0,
                arglist(&default.1)
            )
        }
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
    LBracket,
    RBracket,
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
            b'(' => push(&mut toks, Tok::LParen, &mut i),
            b')' => push(&mut toks, Tok::RParen, &mut i),
            b'{' => push(&mut toks, Tok::LBrace, &mut i),
            b'}' => push(&mut toks, Tok::RBrace, &mut i),
            b'[' => push(&mut toks, Tok::LBracket, &mut i),
            b']' => push(&mut toks, Tok::RBracket, &mut i),
            b':' => push(&mut toks, Tok::Colon, &mut i),
            b',' => push(&mut toks, Tok::Comma, &mut i),
            b'=' => push(&mut toks, Tok::Equals, &mut i),
            b'-' => {
                if i + 1 < bytes.len() && bytes[i + 1] == b'>' {
                    toks.push(Tok::Arrow);
                    i += 2;
                } else {
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

fn push(toks: &mut Vec<Tok>, t: Tok, i: &mut usize) {
    toks.push(t);
    *i += 1;
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

/// A branch edge whose target is still a label name.
type PEdge = (String, Vec<u32>);

/// Intermediate block whose terminator still refers to block labels by name.
struct PBlock {
    label: String,
    params: Vec<ValType>,
    insts: Vec<Inst>,
    term: PTerm,
}

enum PTerm {
    Br(PEdge),
    BrIf {
        cond: u32,
        then: PEdge,
        els: PEdge,
    },
    BrTable {
        idx: u32,
        targets: Vec<PEdge>,
        default: PEdge,
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
        let edge = |e: PEdge| -> Result<(u32, Vec<u32>), ParseError> {
            let t = labels
                .get(&e.0)
                .copied()
                .ok_or_else(|| ParseError(format!("unknown block label `{}`", e.0)))?;
            Ok((t, e.1))
        };

        let mut blocks = Vec::new();
        for b in pblocks {
            let term = match b.term {
                PTerm::Br(e) => {
                    let (target, args) = edge(e)?;
                    Terminator::Br { target, args }
                }
                PTerm::BrIf { cond, then, els } => {
                    let (then_blk, then_args) = edge(then)?;
                    let (else_blk, else_args) = edge(els)?;
                    Terminator::BrIf {
                        cond,
                        then_blk,
                        then_args,
                        else_blk,
                        else_args,
                    }
                }
                PTerm::BrTable {
                    idx,
                    targets,
                    default,
                } => Terminator::BrTable {
                    idx,
                    targets: targets
                        .into_iter()
                        .map(edge)
                        .collect::<Result<Vec<_>, _>>()?,
                    default: edge(default)?,
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
                "br" | "br_if" | "br_table" | "return" => {
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

        if op == "select" {
            let cond = self.value(names)?;
            let a = self.value(names)?;
            let b = self.value(names)?;
            return Ok(Inst::Select { cond, a, b });
        }
        if let Some(cv) = ConvOp::from_name(&op) {
            let a = self.value(names)?;
            return Ok(Inst::Convert { op: cv, a });
        }

        let (prefix, suffix) = op
            .split_once('.')
            .ok_or_else(|| ParseError(format!("unknown opcode `{op}`")))?;
        let ty = match prefix {
            "i32" => IntTy::I32,
            "i64" => IntTy::I64,
            _ => return err(format!("unknown opcode `{op}`")),
        };

        if suffix == "const" {
            let v = self.parse_int()?;
            return Ok(match ty {
                IntTy::I32 => Inst::ConstI32(
                    i32::try_from(v)
                        .map_err(|_| ParseError(format!("i32 const out of range: {v}")))?,
                ),
                IntTy::I64 => Inst::ConstI64(v),
            });
        }
        if suffix == "eqz" {
            let a = self.value(names)?;
            return Ok(Inst::Eqz { ty, a });
        }
        if let Some(o) = BinOp::from_name(suffix) {
            let a = self.value(names)?;
            let b = self.value(names)?;
            return Ok(Inst::IntBin { ty, op: o, a, b });
        }
        if let Some(o) = CmpOp::from_name(suffix) {
            let a = self.value(names)?;
            let b = self.value(names)?;
            return Ok(Inst::IntCmp { ty, op: o, a, b });
        }
        err(format!("unknown opcode `{op}`"))
    }

    fn parse_term(&mut self, names: &HashMap<String, u32>) -> Result<PTerm, ParseError> {
        let kw = self.ident()?;
        Ok(match kw.as_str() {
            "br" => PTerm::Br(self.parse_edge(names)?),
            "br_if" => {
                let cond = self.value(names)?;
                let then = self.parse_edge(names)?;
                let els = self.parse_edge(names)?;
                PTerm::BrIf { cond, then, els }
            }
            "br_table" => {
                let idx = self.value(names)?;
                self.expect(&Tok::LBracket)?;
                let mut targets = Vec::new();
                while self.peek() != Some(&Tok::RBracket) {
                    if !targets.is_empty() {
                        self.expect(&Tok::Comma)?;
                    }
                    targets.push(self.parse_edge(names)?);
                }
                self.expect(&Tok::RBracket)?;
                let default = self.parse_edge(names)?;
                PTerm::BrTable {
                    idx,
                    targets,
                    default,
                }
            }
            "return" => {
                let mut vals = Vec::new();
                // Comma-separated value list; a return ends the block, so any ident
                // that is not a known value name starts the next block — stop there.
                while let Some(Tok::Ident(s)) = self.peek() {
                    if !names.contains_key(s) {
                        break;
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
    fn parse_edge(&mut self, names: &HashMap<String, u32>) -> Result<PEdge, ParseError> {
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
}
