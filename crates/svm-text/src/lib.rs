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

use svm_ir::{
    AtomicRmwOp, BinOp, Block, CastOp, CmpOp, ConvOp, Data, FBinOp, FCmpOp, FToI, FUnOp, FloatTy,
    Func, FuncType, IToF, Import, Inst, IntTy, IntUnOp, LoadOp, Memory, Module, Ordering, StoreOp,
    Terminator, VBitBinOp, VFloatBinOp, VFloatUnOp, VIntBinOp, VShape, ValType,
};

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
    for d in &m.data {
        let _ = writeln!(
            s,
            "data {}{} \"{}\"",
            if d.readonly { "ro " } else { "" },
            d.offset,
            escape_bytes(&d.bytes)
        );
    }
    if let Some(mem) = &m.memory {
        let _ = writeln!(s, "memory {}", mem.size_log2);
        s.push('\n');
    }
    // §7 named capability imports, one per line, in declaration order (the index a
    // `call.import` references): `import <idx> "<name>" (params) -> (results)`.
    if !m.imports.is_empty() {
        for (i, imp) in m.imports.iter().enumerate() {
            let _ = writeln!(
                s,
                "import {i} \"{}\" ({}) -> ({})",
                imp.name,
                types(&imp.sig.params),
                types(&imp.sig.results)
            );
        }
        s.push('\n');
    }
    let fn_results: Vec<usize> = m.funcs.iter().map(|f| f.results.len()).collect();
    for (i, f) in m.funcs.iter().enumerate() {
        if i > 0 {
            s.push('\n');
        }
        print_func(&mut s, f, &fn_results);
    }
    s
}

/// Escape data-segment bytes for the text form: printable ASCII verbatim (except `\` and `"`),
/// everything else as `\xHH`. Round-trips through [`lex_string`].
fn escape_bytes(bytes: &[u8]) -> String {
    let mut s = String::new();
    for &b in bytes {
        match b {
            b'\\' => s.push_str("\\\\"),
            b'"' => s.push_str("\\\""),
            0x20..=0x7e => s.push(b as char),
            _ => {
                let _ = write!(s, "\\x{b:02x}");
            }
        }
    }
    s
}

fn print_func(s: &mut String, f: &Func, fn_results: &[usize]) {
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
            let n = inst.result_count(fn_results);
            if n == 0 {
                // No-result instruction (`store`, void `call`): no `vN =` binding.
                let _ = writeln!(s, "  {}", print_inst(inst));
            } else {
                let lhs: Vec<String> = (0..n).map(|k| format!("v{}", next + k as u32)).collect();
                let _ = writeln!(s, "  {} = {}", lhs.join(", "), print_inst(inst));
                next += n as u32;
            }
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
        Inst::IntUn { ty, op, a } => format!("{}.{} v{a}", ty.prefix(), op.name()),
        Inst::IntCmp { ty, op, a, b } => format!("{}.{} v{a} v{b}", ty.prefix(), op.name()),
        Inst::Eqz { ty, a } => format!("{}.eqz v{a}", ty.prefix()),
        Inst::Convert { op, a } => format!("{} v{a}", op.sig().0),
        Inst::Select { cond, a, b } => format!("select v{cond} v{a} v{b}"),
        // `{:?}` gives the shortest round-tripping form with a decimal point
        // (e.g. `2.0`, `1.5`, `-3.25`), so it re-tokenizes as a number.
        Inst::ConstF32(bits) => format!("f32.const {:?}", f32::from_bits(*bits)),
        Inst::ConstF64(bits) => format!("f64.const {:?}", f64::from_bits(*bits)),
        Inst::FBin { ty, op, a, b } => format!("{}.{} v{a} v{b}", ty.prefix(), op.name()),
        Inst::FUn { ty, op, a } => format!("{}.{} v{a}", ty.prefix(), op.name()),
        Inst::FCmp { ty, op, a, b } => format!("{}.{} v{a} v{b}", ty.prefix(), op.name()),
        Inst::FToISat { op, a } => format!("{} v{a}", op.name()),
        Inst::FToITrap { op, a } => format!("{} v{a}", op.trap_name()),
        Inst::IToFConv { op, a } => format!("{} v{a}", op.name()),
        Inst::PtrAdd { a, b } => format!("ptr.add v{a} v{b}"),
        Inst::PtrCast { to_int, a } => {
            format!("ptr.{} v{a}", if *to_int { "to_int" } else { "from_int" })
        }
        Inst::Cast { op, a } => format!("{} v{a}", op.sig().0),
        Inst::Load {
            op,
            addr,
            offset,
            align,
        } => format!("{} v{addr}{}", op.info().0, memarg(*offset, *align)),
        Inst::Store {
            op,
            addr,
            value,
            offset,
            align,
        } => format!(
            "{} v{addr} v{value}{}",
            op.info().0,
            memarg(*offset, *align)
        ),
        // §12 atomics: `<ty>.atomic.<op>[.<order>]`, naturally aligned (no `align=`, only `offset=`).
        // The default `seqcst` ordering is omitted, so seq-cst atomics round-trip unchanged.
        Inst::AtomicLoad {
            ty,
            addr,
            offset,
            order,
        } => format!(
            "{}.atomic.load{} v{addr}{}",
            ty.prefix(),
            ord_suffix(*order),
            memarg(*offset, 0)
        ),
        Inst::AtomicStore {
            ty,
            addr,
            value,
            offset,
            order,
        } => format!(
            "{}.atomic.store{} v{addr} v{value}{}",
            ty.prefix(),
            ord_suffix(*order),
            memarg(*offset, 0)
        ),
        Inst::AtomicRmw {
            ty,
            op,
            addr,
            value,
            offset,
            order,
        } => format!(
            "{}.atomic.rmw.{}{} v{addr} v{value}{}",
            ty.prefix(),
            op.name(),
            ord_suffix(*order),
            memarg(*offset, 0)
        ),
        Inst::AtomicCmpxchg {
            ty,
            addr,
            expected,
            replacement,
            offset,
            order,
        } => format!(
            "{}.atomic.cmpxchg{} v{addr} v{expected} v{replacement}{}",
            ty.prefix(),
            ord_suffix(*order),
            memarg(*offset, 0)
        ),
        Inst::Call { func, args } => format!("call {func}{}", arglist(args)),
        Inst::RefFunc { func } => format!("ref.func {func}"),
        Inst::CallIndirect { ty, idx, args } => format!(
            "call_indirect ({}) -> ({}) v{idx}{}",
            types(&ty.params),
            types(&ty.results),
            arglist(args)
        ),
        Inst::CapCall {
            type_id,
            op,
            sig,
            handle,
            args,
        } => format!(
            "cap.call {type_id} {op} ({}) -> ({}) v{handle}{}",
            types(&sig.params),
            types(&sig.results),
            arglist(args)
        ),
        // §7 named import call: `call.import <idx> v<handle> (args)`. The op signature is
        // recovered from the module's `import <idx>` declaration on re-parse, so it is not
        // re-printed here (the import index is the link).
        Inst::CallImport {
            import,
            handle,
            args,
            ..
        } => format!("call.import {import} v{handle}{}", arglist(args)),
        // §12 fibers (stack switching).
        Inst::ContNew { func, sp } => format!("cont.new v{func} v{sp}"),
        Inst::ContResume { k, arg } => format!("cont.resume v{k} v{arg}"),
        Inst::Suspend { value } => format!("suspend v{value}"),
        // §12 real threads (OS-thread vCPUs over shared memory).
        Inst::ThreadSpawn { func, sp, arg } => format!("thread.spawn {func} v{sp} v{arg}"),
        Inst::ThreadJoin { handle } => format!("thread.join v{handle}"),
        // §12 futex wait/notify.
        Inst::MemoryWait {
            ty,
            addr,
            expected,
            timeout,
        } => format!("{}.atomic.wait v{addr} v{expected} v{timeout}", ty.prefix()),
        Inst::MemoryNotify { addr, count } => format!("atomic.notify v{addr} v{count}"),
        Inst::AtomicFence { order } => format!("atomic.fence{}", ord_suffix(*order)),

        // ----- §17 SIMD (D58) — lane shape carried by the op, bytes printed little-endian. -----
        Inst::ConstV128(bytes) => format!("v128.const{}", byte_list(bytes)),
        Inst::V128Load {
            addr,
            offset,
            align,
        } => {
            format!("v128.load v{addr}{}", memarg(*offset, *align))
        }
        Inst::V128Store {
            addr,
            value,
            offset,
            align,
        } => format!("v128.store v{addr} v{value}{}", memarg(*offset, *align)),
        Inst::Splat { shape, a } => format!("{}.splat v{a}", shape.name()),
        Inst::ExtractLane {
            shape,
            lane,
            signed,
            a,
        } => format!(
            "{}.extract_lane{} {lane} v{a}",
            shape.name(),
            lane_sign_suffix(*shape, *signed)
        ),
        Inst::ReplaceLane { shape, lane, a, b } => {
            format!("{}.replace_lane {lane} v{a} v{b}", shape.name())
        }
        Inst::VIntBin { shape, op, a, b } => format!("{}.{} v{a} v{b}", shape.name(), op.name()),
        Inst::VFloatBin { shape, op, a, b } => format!("{}.{} v{a} v{b}", shape.name(), op.name()),
        Inst::VFloatUn { shape, op, a } => format!("{}.{} v{a}", shape.name(), op.name()),
        Inst::VBitBin { op, a, b } => format!("v128.{} v{a} v{b}", op.name()),
        Inst::VNot { a } => format!("v128.not v{a}"),
        Inst::Bitselect { a, b, mask } => format!("v128.bitselect v{a} v{b} v{mask}"),
        Inst::Shuffle { lanes, a, b } => format!("i8x16.shuffle{} v{a} v{b}", byte_list(lanes)),
        Inst::Swizzle { a, b } => format!("i8x16.swizzle v{a} v{b}"),
        Inst::SimdWidthBytes => "simd.width_bytes".to_string(),
    }
}

/// Render 16 bytes as ` b0 b1 ... b15` (decimal, leading space). Used by `v128.const`
/// (little-endian value bytes) and `i8x16.shuffle` (byte indices).
fn byte_list(bytes: &[u8; 16]) -> String {
    let mut s = String::new();
    for b in bytes {
        s.push(' ');
        s.push_str(&b.to_string());
    }
    s
}

/// The `_s`/`_u` suffix on a narrow-integer `extract_lane` (`i8x16`/`i16x8`); empty for
/// the wider shapes where extraction is unambiguous.
fn lane_sign_suffix(shape: VShape, signed: bool) -> &'static str {
    match shape {
        VShape::I8x16 | VShape::I16x8 => {
            if signed {
                "_s"
            } else {
                "_u"
            }
        }
        _ => "",
    }
}

/// The `.<order>` text suffix for an atomic op, empty for the default `seqcst` (so seq-cst atomics
/// print exactly as before this surface existed).
fn ord_suffix(order: Ordering) -> String {
    match order {
        Ordering::SeqCst => String::new(),
        o => format!(".{}", o.name()),
    }
}

/// Strip a trailing `.<order>` token off an atomic mnemonic tail, defaulting to `seqcst`. E.g.
/// `"rmw.add.relaxed"` → `("rmw.add", Relaxed)`, `"load"` → `("load", SeqCst)`. (Ordering names never
/// collide with op names, so this is unambiguous.)
fn split_order(rest: &str) -> (&str, Ordering) {
    for o in Ordering::ALL {
        if o == Ordering::SeqCst {
            continue;
        }
        if let Some(base) = rest.strip_suffix(o.name()) {
            if let Some(base) = base.strip_suffix('.') {
                return (base, o);
            }
        }
    }
    (rest, Ordering::SeqCst)
}

/// Render the optional `offset=`/`align=` suffix, omitting zero defaults.
fn memarg(offset: u64, align: u8) -> String {
    let mut s = String::new();
    if offset != 0 {
        let _ = write!(s, " offset={offset}");
    }
    if align != 0 {
        let _ = write!(s, " align={align}");
    }
    s
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
        Terminator::ReturnCall { func, args } => format!("return_call {func}{}", arglist(args)),
        Terminator::ReturnCallIndirect { ty, idx, args } => format!(
            "return_call_indirect ({}) -> ({}) v{idx}{}",
            types(&ty.params),
            types(&ty.results),
            arglist(args)
        ),
        Terminator::Unreachable => "unreachable".to_string(),
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
    Float(f64),
    /// A byte string `"..."` (data-segment bytes), with `\\`, `\"`, `\n`, `\t`, `\r`, `\0`,
    /// and `\xHH` hex escapes — so arbitrary (non-UTF-8) bytes are representable.
    Str(Vec<u8>),
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
                    let (tok, ni) = lex_number(bytes, i)?;
                    toks.push(tok);
                    i = ni;
                }
            }
            b'0'..=b'9' => {
                let (tok, ni) = lex_number(bytes, i)?;
                toks.push(tok);
                i = ni;
            }
            b'"' => {
                let (tok, ni) = lex_string(bytes, i)?;
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

/// Lex an integer or float literal. A `.` or exponent makes it a float.
fn lex_number(bytes: &[u8], start: usize) -> Result<(Tok, usize), ParseError> {
    let mut i = start;
    if bytes[i] == b'-' {
        i += 1;
    }
    let mut has_digit = false;
    let mut is_float = false;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
        has_digit = true;
    }
    if i < bytes.len() && bytes[i] == b'.' {
        is_float = true;
        i += 1;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
            has_digit = true;
        }
    }
    if i < bytes.len() && (bytes[i] | 0x20) == b'e' {
        is_float = true;
        i += 1;
        if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
            i += 1;
        }
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
    }
    if !has_digit {
        return err("expected digits in number");
    }
    let s = std::str::from_utf8(&bytes[start..i]).unwrap_or("");
    if is_float {
        let v: f64 = s
            .parse()
            .map_err(|_| ParseError(format!("invalid float: {s}")))?;
        Ok((Tok::Float(v), i))
    } else {
        let v: i64 = s
            .parse()
            .map_err(|_| ParseError(format!("integer out of range: {s}")))?;
        Ok((Tok::Int(v), i))
    }
}

/// Lex a byte string `"..."` starting at `bytes[start] == '"'`. Supports `\\`, `\"`, `\n`,
/// `\t`, `\r`, `\0`, and `\xHH` (two hex digits) escapes; every other byte is taken verbatim.
/// Returns the [`Tok::Str`] and the index just past the closing quote.
fn lex_string(bytes: &[u8], start: usize) -> Result<(Tok, usize), ParseError> {
    let mut i = start + 1; // past the opening quote
    let mut out = Vec::new();
    while i < bytes.len() {
        match bytes[i] {
            b'"' => return Ok((Tok::Str(out), i + 1)),
            b'\\' => {
                i += 1;
                let e = *bytes
                    .get(i)
                    .ok_or_else(|| ParseError("unterminated escape in string".into()))?;
                match e {
                    b'\\' => out.push(b'\\'),
                    b'"' => out.push(b'"'),
                    b'n' => out.push(b'\n'),
                    b't' => out.push(b'\t'),
                    b'r' => out.push(b'\r'),
                    b'0' => out.push(0),
                    b'x' => {
                        let hi = bytes.get(i + 1).copied().and_then(hex_val);
                        let lo = bytes.get(i + 2).copied().and_then(hex_val);
                        match (hi, lo) {
                            (Some(h), Some(l)) => {
                                out.push(h * 16 + l);
                                i += 2;
                            }
                            _ => return Err(ParseError("invalid \\xHH escape".into())),
                        }
                    }
                    _ => {
                        return Err(ParseError(format!(
                            "unknown string escape: \\{}",
                            e as char
                        )))
                    }
                }
                i += 1;
            }
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    Err(ParseError("unterminated string".into()))
}

fn hex_val(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
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
    // Calls may forward-reference functions defined later, so we need every
    // function's result arity before parsing bodies (it determines how many value
    // indices a `call` binds). A cheap header-only prescan supplies it.
    let fn_results = prescan_fn_results(&toks)?;
    let mut p = Parser {
        toks: &toks,
        pos: 0,
        fn_results,
        imports: Vec::new(),
    };
    let mut funcs = Vec::new();
    let mut memory = None;
    let mut data: Vec<Data> = Vec::new();
    while !p.at_end() {
        match p.peek() {
            // Module-level `memory <size_log2>` declaration.
            Some(Tok::Ident(s)) if s == "memory" => {
                p.next()?;
                let n = p.parse_int()?;
                let size_log2 = u8::try_from(n)
                    .map_err(|_| ParseError(format!("memory size_log2 out of range: {n}")))?;
                memory = Some(Memory { size_log2 });
            }
            // §7 named import: `import <idx> "<name>" (params) -> (results)`. Indices are
            // dense and in declaration order (they are what `call.import` references).
            Some(Tok::Ident(s)) if s == "import" => {
                p.next()?;
                let n = p.parse_int()?;
                if n < 0 || n as usize != p.imports.len() {
                    return err("import indices must be dense and in declaration order");
                }
                let name = String::from_utf8(p.parse_str()?)
                    .map_err(|_| ParseError("import name is not valid UTF-8".into()))?;
                let params = p.parse_type_list()?;
                p.expect(&Tok::Arrow)?;
                let results = p.parse_type_list()?;
                p.imports.push(Import {
                    name,
                    sig: FuncType { params, results },
                });
            }
            // Module-level `data [ro] <offset> "<bytes>"` segment (§3a / D40).
            Some(Tok::Ident(s)) if s == "data" => {
                p.next()?;
                let readonly = matches!(p.peek(), Some(Tok::Ident(k)) if k == "ro");
                if readonly {
                    p.next()?;
                }
                let n = p.parse_int()?;
                let offset = u64::try_from(n)
                    .map_err(|_| ParseError(format!("negative data offset: {n}")))?;
                let bytes = p.parse_str()?;
                data.push(Data {
                    offset,
                    readonly,
                    bytes,
                });
            }
            _ => funcs.push(p.parse_func()?),
        }
    }
    Ok(Module {
        funcs,
        memory,
        data,
        imports: std::mem::take(&mut p.imports),
    })
}

/// Header-only pass: each function's result count, indexed by function order.
/// Skips `memory` decls and function bodies (brace-matched).
fn prescan_fn_results(toks: &[Tok]) -> Result<Vec<usize>, ParseError> {
    let mut p = Parser {
        toks,
        pos: 0,
        fn_results: Vec::new(),
        imports: Vec::new(),
    };
    let mut out = Vec::new();
    while !p.at_end() {
        match p.peek() {
            Some(Tok::Ident(s)) if s == "memory" => {
                p.next()?;
                p.parse_int()?;
            }
            // §7 `import <idx> "<name>" (params) -> (results)` — skip in the header prescan.
            Some(Tok::Ident(s)) if s == "import" => {
                p.next()?;
                p.parse_int()?;
                p.parse_str()?;
                p.parse_type_list()?;
                p.expect(&Tok::Arrow)?;
                p.parse_type_list()?;
            }
            Some(Tok::Ident(s)) if s == "data" => {
                // `data [ro] <offset> "<bytes>"` — skip past it in the header prescan.
                p.next()?;
                if matches!(p.peek(), Some(Tok::Ident(k)) if k == "ro") {
                    p.next()?;
                }
                p.parse_int()?;
                p.parse_str()?;
            }
            Some(Tok::Ident(s)) if s == "func" => {
                p.next()?;
                let _params = p.parse_type_list()?;
                p.expect(&Tok::Arrow)?;
                out.push(p.parse_type_list()?.len());
                p.expect(&Tok::LBrace)?;
                let mut depth = 1usize;
                while depth > 0 {
                    match p.next()? {
                        Tok::LBrace => depth += 1,
                        Tok::RBrace => depth -= 1,
                        _ => {}
                    }
                }
            }
            _ => return err("expected `func` or `memory`"),
        }
    }
    Ok(out)
}

struct Parser<'a> {
    toks: &'a [Tok],
    pos: usize,
    /// Result arity of each function (by index), from the prescan.
    fn_results: Vec<usize>,
    /// §7 named imports declared at module top, in order; a `call.import <idx>` recovers its
    /// op signature from here (imports always precede the functions that reference them).
    imports: Vec<Import>,
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
    ReturnCall {
        func: u32,
        args: Vec<u32>,
    },
    ReturnCallIndirect {
        ty: FuncType,
        idx: u32,
        args: Vec<u32>,
    },
    Unreachable,
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
                PTerm::ReturnCall { func, args } => Terminator::ReturnCall { func, args },
                PTerm::ReturnCallIndirect { ty, idx, args } => {
                    Terminator::ReturnCallIndirect { ty, idx, args }
                }
                PTerm::Unreachable => Terminator::Unreachable,
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
            if matches!(
                kw.as_str(),
                "br" | "br_if"
                    | "br_table"
                    | "return"
                    | "return_call"
                    | "return_call_indirect"
                    | "unreachable"
            ) {
                let term = self.parse_term(&names)?;
                return Ok(PBlock {
                    label,
                    params,
                    insts,
                    term,
                });
            }

            // A value-producing instruction is `name(, name)* = opcode operands`; a
            // no-result instruction (`store`, void `call`) is just `opcode operands`.
            // The binding LHS (idents/commas ending in `=`) is what tells them apart.
            let lhs = self.try_binding_lhs();
            let inst = self.parse_inst(&names)?;
            let n = inst.result_count(&self.fn_results);
            match lhs {
                Some(lhs) => {
                    if lhs.len() != n {
                        return err(format!(
                            "instruction produces {n} result(s) but {} name(s) bound",
                            lhs.len()
                        ));
                    }
                    for name in lhs {
                        if names.insert(name.clone(), next_idx).is_some() {
                            return err(format!("duplicate value name `{name}`"));
                        }
                        next_idx += 1;
                    }
                }
                None if n == 0 => {}
                None => return err("expected `name =` for a value-producing instruction"),
            }
            insts.push(inst);
        }
    }

    /// If the upcoming tokens are a binding LHS — `ident (, ident)* =` — consume them
    /// and return the names; otherwise consume nothing and return `None`.
    fn try_binding_lhs(&mut self) -> Option<Vec<String>> {
        let start = self.pos;
        let mut names = Vec::new();
        loop {
            match self.toks.get(self.pos) {
                Some(Tok::Ident(s)) => {
                    names.push(s.clone());
                    self.pos += 1;
                }
                _ => {
                    self.pos = start;
                    return None;
                }
            }
            match self.toks.get(self.pos) {
                Some(Tok::Comma) => self.pos += 1,
                Some(Tok::Equals) => {
                    self.pos += 1;
                    return Some(names);
                }
                _ => {
                    self.pos = start;
                    return None;
                }
            }
        }
    }

    /// Parse a §12 atomic op given its `ty` and the mnemonic tail after `<ty>.atomic.`
    /// (`load` / `store` / `cmpxchg` / `rmw.<op>`). Atomics carry an `offset=` memarg but no
    /// `align` (the access is naturally aligned by definition).
    fn parse_atomic(
        &mut self,
        ty: IntTy,
        rest: &str,
        names: &HashMap<String, u32>,
    ) -> Result<Inst, ParseError> {
        // `wait` carries no memory ordering; everything else may end in a `.<order>` suffix.
        if rest == "wait" {
            let addr = self.value(names)?;
            let expected = self.value(names)?;
            let timeout = self.value(names)?;
            return Ok(Inst::MemoryWait {
                ty,
                addr,
                expected,
                timeout,
            });
        }
        let (base, order) = split_order(rest);
        match base {
            "load" => {
                let addr = self.value(names)?;
                let (offset, _) = self.parse_memarg()?;
                Ok(Inst::AtomicLoad {
                    ty,
                    addr,
                    offset,
                    order,
                })
            }
            "store" => {
                let addr = self.value(names)?;
                let value = self.value(names)?;
                let (offset, _) = self.parse_memarg()?;
                Ok(Inst::AtomicStore {
                    ty,
                    addr,
                    value,
                    offset,
                    order,
                })
            }
            "cmpxchg" => {
                let addr = self.value(names)?;
                let expected = self.value(names)?;
                let replacement = self.value(names)?;
                let (offset, _) = self.parse_memarg()?;
                Ok(Inst::AtomicCmpxchg {
                    ty,
                    addr,
                    expected,
                    replacement,
                    offset,
                    order,
                })
            }
            _ => {
                let opname = base.strip_prefix("rmw.").ok_or_else(|| {
                    ParseError(format!("unknown atomic op: {}.atomic.{rest}", ty.prefix()))
                })?;
                let op = AtomicRmwOp::from_name(opname)
                    .ok_or_else(|| ParseError(format!("unknown atomic rmw op: {opname}")))?;
                let addr = self.value(names)?;
                let value = self.value(names)?;
                let (offset, _) = self.parse_memarg()?;
                Ok(Inst::AtomicRmw {
                    ty,
                    op,
                    order,
                    addr,
                    value,
                    offset,
                })
            }
        }
    }

    fn parse_inst(&mut self, names: &HashMap<String, u32>) -> Result<Inst, ParseError> {
        let op = self.ident()?;

        // Ops whose full name is matched directly (no `prefix.suffix` split).
        if op == "select" {
            let cond = self.value(names)?;
            let a = self.value(names)?;
            let b = self.value(names)?;
            return Ok(Inst::Select { cond, a, b });
        }
        if op == "call" {
            let n = self.parse_int()?;
            let func = u32::try_from(n)
                .map_err(|_| ParseError(format!("function index out of range: {n}")))?;
            let args = self.parse_value_list(names)?;
            return Ok(Inst::Call { func, args });
        }
        // §7 named import call, two forms:
        //   indexed:     `call.import <idx> v<handle> (args)`          — sig from the `import` decl
        //   name-inline: `call.import "name" (params)->(results) v<h> (args)` — interned on the fly
        // The name-inline form lets a streaming frontend (chibicc) emit a capability call per site
        // with no pre-collected import table; the parser interns the name into `imports`.
        if op == "call.import" {
            let (import, sig) = if matches!(self.peek(), Some(Tok::Str(_))) {
                let name = String::from_utf8(self.parse_str()?)
                    .map_err(|_| ParseError("call.import name is not valid UTF-8".into()))?;
                let params = self.parse_type_list()?;
                self.expect(&Tok::Arrow)?;
                let results = self.parse_type_list()?;
                let sig = FuncType { params, results };
                // Intern: reuse an existing import of the same (name, sig), else append.
                let idx = self
                    .imports
                    .iter()
                    .position(|imp| imp.name == name && imp.sig == sig)
                    .unwrap_or_else(|| {
                        self.imports.push(Import {
                            name,
                            sig: sig.clone(),
                        });
                        self.imports.len() - 1
                    });
                (idx as u32, sig)
            } else {
                let n = self.parse_int()?;
                let import = u32::try_from(n)
                    .map_err(|_| ParseError(format!("import index out of range: {n}")))?;
                let sig = self
                    .imports
                    .get(import as usize)
                    .map(|imp| imp.sig.clone())
                    .ok_or_else(|| {
                        ParseError(format!("call.import references undeclared import {import}"))
                    })?;
                (import, sig)
            };
            let handle = self.value(names)?;
            let args = self.parse_value_list(names)?;
            return Ok(Inst::CallImport {
                import,
                sig,
                handle,
                args,
            });
        }
        if op == "ref.func" {
            let n = self.parse_int()?;
            let func = u32::try_from(n)
                .map_err(|_| ParseError(format!("function index out of range: {n}")))?;
            return Ok(Inst::RefFunc { func });
        }
        if op == "call_indirect" {
            let params = self.parse_type_list()?;
            self.expect(&Tok::Arrow)?;
            let results = self.parse_type_list()?;
            let idx = self.value(names)?;
            let args = self.parse_value_list(names)?;
            return Ok(Inst::CallIndirect {
                ty: FuncType { params, results },
                idx,
                args,
            });
        }
        if op == "cap.call" {
            let type_id = u32::try_from(self.parse_int()?)
                .map_err(|_| ParseError("cap.call type_id out of range".into()))?;
            let op_index = u32::try_from(self.parse_int()?)
                .map_err(|_| ParseError("cap.call op index out of range".into()))?;
            let params = self.parse_type_list()?;
            self.expect(&Tok::Arrow)?;
            let results = self.parse_type_list()?;
            let handle = self.value(names)?;
            let args = self.parse_value_list(names)?;
            return Ok(Inst::CapCall {
                type_id,
                op: op_index,
                sig: FuncType { params, results },
                handle,
                args,
            });
        }
        if let Some(cv) = ConvOp::from_name(&op) {
            return Ok(Inst::Convert {
                op: cv,
                a: self.value(names)?,
            });
        }
        if let Some(o) = FToI::from_name(&op) {
            return Ok(Inst::FToISat {
                op: o,
                a: self.value(names)?,
            });
        }
        if let Some(o) = FToI::from_trap_name(&op) {
            return Ok(Inst::FToITrap {
                op: o,
                a: self.value(names)?,
            });
        }
        if op == "ptr.add" {
            let a = self.value(names)?;
            let b = self.value(names)?;
            return Ok(Inst::PtrAdd { a, b });
        }
        // §12 fibers (stack switching).
        if op == "cont.new" {
            let func = self.value(names)?;
            let sp = self.value(names)?;
            return Ok(Inst::ContNew { func, sp });
        }
        if op == "cont.resume" {
            let k = self.value(names)?;
            let arg = self.value(names)?;
            return Ok(Inst::ContResume { k, arg });
        }
        if op == "suspend" {
            return Ok(Inst::Suspend {
                value: self.value(names)?,
            });
        }
        // §12 real threads.
        if op == "thread.spawn" {
            let n = self.parse_int()?;
            let func = u32::try_from(n)
                .map_err(|_| ParseError(format!("function index out of range: {n}")))?;
            let sp = self.value(names)?;
            let arg = self.value(names)?;
            return Ok(Inst::ThreadSpawn { func, sp, arg });
        }
        if op == "thread.join" {
            return Ok(Inst::ThreadJoin {
                handle: self.value(names)?,
            });
        }
        if op == "atomic.notify" {
            let addr = self.value(names)?;
            let count = self.value(names)?;
            return Ok(Inst::MemoryNotify { addr, count });
        }
        if let Some(tail) = op.strip_prefix("atomic.fence") {
            let order = if tail.is_empty() {
                Ordering::SeqCst
            } else {
                Ordering::from_name(tail.strip_prefix('.').unwrap_or(tail))
                    .ok_or_else(|| ParseError(format!("unknown fence ordering: {op}")))?
            };
            return Ok(Inst::AtomicFence { order });
        }
        if op == "ptr.from_int" || op == "ptr.to_int" {
            return Ok(Inst::PtrCast {
                to_int: op == "ptr.to_int",
                a: self.value(names)?,
            });
        }
        if let Some(o) = IToF::from_name(&op) {
            return Ok(Inst::IToFConv {
                op: o,
                a: self.value(names)?,
            });
        }
        if let Some(o) = CastOp::from_name(&op) {
            return Ok(Inst::Cast {
                op: o,
                a: self.value(names)?,
            });
        }
        if let Some(o) = LoadOp::from_name(&op) {
            let addr = self.value(names)?;
            let (offset, align) = self.parse_memarg()?;
            return Ok(Inst::Load {
                op: o,
                addr,
                offset,
                align,
            });
        }
        if let Some(o) = StoreOp::from_name(&op) {
            let addr = self.value(names)?;
            let value = self.value(names)?;
            let (offset, align) = self.parse_memarg()?;
            return Ok(Inst::Store {
                op: o,
                addr,
                value,
                offset,
                align,
            });
        }
        // §12 atomics: `<ty>.atomic.<load|store|cmpxchg|rmw.<op>>`.
        if let Some(rest) = op.strip_prefix("i32.atomic.") {
            return self.parse_atomic(IntTy::I32, rest, names);
        }
        if let Some(rest) = op.strip_prefix("i64.atomic.") {
            return self.parse_atomic(IntTy::I64, rest, names);
        }

        // ----- §17 SIMD (D58) -----
        if op == "v128.const" {
            let bytes = self.parse_byte16()?;
            return Ok(Inst::ConstV128(bytes));
        }
        if op == "v128.load" {
            let addr = self.value(names)?;
            let (offset, align) = self.parse_memarg()?;
            return Ok(Inst::V128Load {
                addr,
                offset,
                align,
            });
        }
        if op == "v128.store" {
            let addr = self.value(names)?;
            let value = self.value(names)?;
            let (offset, align) = self.parse_memarg()?;
            return Ok(Inst::V128Store {
                addr,
                value,
                offset,
                align,
            });
        }
        if op == "v128.not" {
            return Ok(Inst::VNot {
                a: self.value(names)?,
            });
        }
        if op == "v128.bitselect" {
            let a = self.value(names)?;
            let b = self.value(names)?;
            let mask = self.value(names)?;
            return Ok(Inst::Bitselect { a, b, mask });
        }
        if let Some(s) = op.strip_prefix("v128.") {
            if let Some(o) = VBitBinOp::from_name(s) {
                let a = self.value(names)?;
                let b = self.value(names)?;
                return Ok(Inst::VBitBin { op: o, a, b });
            }
        }
        if op == "i8x16.shuffle" {
            let lanes = self.parse_byte16()?;
            let a = self.value(names)?;
            let b = self.value(names)?;
            return Ok(Inst::Shuffle { lanes, a, b });
        }
        if op == "i8x16.swizzle" {
            let a = self.value(names)?;
            let b = self.value(names)?;
            return Ok(Inst::Swizzle { a, b });
        }
        if op == "simd.width_bytes" {
            return Ok(Inst::SimdWidthBytes);
        }
        if let Some((sh, suffix)) = op
            .split_once('.')
            .and_then(|(p, s)| VShape::from_name(p).map(|sh| (sh, s)))
        {
            return self.parse_shape_inst(sh, suffix, &op, names);
        }

        let (prefix, suffix) = op
            .split_once('.')
            .ok_or_else(|| ParseError(format!("unknown opcode `{op}`")))?;
        match prefix {
            "i32" => self.parse_int_inst(IntTy::I32, suffix, &op, names),
            "i64" => self.parse_int_inst(IntTy::I64, suffix, &op, names),
            "f32" => self.parse_float_inst(FloatTy::F32, suffix, &op, names),
            "f64" => self.parse_float_inst(FloatTy::F64, suffix, &op, names),
            _ => err(format!("unknown opcode `{op}`")),
        }
    }

    /// Parse a `<shape>.<suffix>` SIMD op (splat/extract_lane/replace_lane and the
    /// lane-wise int/float arithmetic). The whole-vector bitwise ops, `v128.const`,
    /// load/store, shuffle/swizzle are matched by full name in [`Self::parse_inst`].
    fn parse_shape_inst(
        &mut self,
        shape: VShape,
        suffix: &str,
        op: &str,
        names: &HashMap<String, u32>,
    ) -> Result<Inst, ParseError> {
        if suffix == "splat" {
            return Ok(Inst::Splat {
                shape,
                a: self.value(names)?,
            });
        }
        // `extract_lane[_s|_u] <lane> v<a>` — the sign suffix is only meaningful for narrow
        // integer shapes; accept (and ignore) it elsewhere only if absent.
        if let Some(rest) = suffix.strip_prefix("extract_lane") {
            let signed = match rest {
                "" => true, // wide shapes: extraction is exact; `signed` is unused
                "_s" => true,
                "_u" => false,
                _ => return err(format!("unknown opcode `{op}`")),
            };
            let lane = u8::try_from(self.parse_int()?)
                .map_err(|_| ParseError(format!("lane index out of range in `{op}`")))?;
            let a = self.value(names)?;
            return Ok(Inst::ExtractLane {
                shape,
                lane,
                signed,
                a,
            });
        }
        if suffix == "replace_lane" {
            let lane = u8::try_from(self.parse_int()?)
                .map_err(|_| ParseError(format!("lane index out of range in `{op}`")))?;
            let a = self.value(names)?;
            let b = self.value(names)?;
            return Ok(Inst::ReplaceLane { shape, lane, a, b });
        }
        // Dispatch the lane-arithmetic suffix by shape category — `add`/`sub`/`mul` name both an
        // integer and a float op, so the shape decides which (a float op on an int shape, or vice
        // versa, is then rejected at verify, not silently mis-parsed).
        if shape.is_float() {
            if let Some(o) = VFloatBinOp::from_name(suffix) {
                let a = self.value(names)?;
                let b = self.value(names)?;
                return Ok(Inst::VFloatBin { shape, op: o, a, b });
            }
            if let Some(o) = VFloatUnOp::from_name(suffix) {
                return Ok(Inst::VFloatUn {
                    shape,
                    op: o,
                    a: self.value(names)?,
                });
            }
        } else if let Some(o) = VIntBinOp::from_name(suffix) {
            let a = self.value(names)?;
            let b = self.value(names)?;
            return Ok(Inst::VIntBin { shape, op: o, a, b });
        }
        err(format!("unknown opcode `{op}`"))
    }

    /// Parse exactly 16 byte (`0..=255`) integer tokens into a `[u8; 16]` — the operand
    /// of `v128.const` (value bytes) and `i8x16.shuffle` (byte indices).
    fn parse_byte16(&mut self) -> Result<[u8; 16], ParseError> {
        let mut bytes = [0u8; 16];
        for slot in &mut bytes {
            let v = self.parse_int()?;
            *slot = u8::try_from(v).map_err(|_| ParseError(format!("byte out of range: {v}")))?;
        }
        Ok(bytes)
    }

    fn parse_int_inst(
        &mut self,
        ty: IntTy,
        suffix: &str,
        op: &str,
        names: &HashMap<String, u32>,
    ) -> Result<Inst, ParseError> {
        if suffix == "const" {
            let v = self.parse_int()?;
            return Ok(match ty {
                // An `i32.const` is a 32-bit pattern: accept both the signed-`i32` range and the
                // unsigned-`u32` range (e.g. `4294967295` = `0xFFFFFFFF` = `-1`), as a C frontend
                // emits unsigned constants like `0xFFFFFFFF`/`UINT32_MAX` by value.
                IntTy::I32 => Inst::ConstI32(
                    i32::try_from(v)
                        .or_else(|_| u32::try_from(v).map(|u| u as i32))
                        .map_err(|_| ParseError(format!("i32 const out of range: {v}")))?,
                ),
                IntTy::I64 => Inst::ConstI64(v),
            });
        }
        if suffix == "eqz" {
            return Ok(Inst::Eqz {
                ty,
                a: self.value(names)?,
            });
        }
        if let Some(o) = IntUnOp::from_name(suffix) {
            return Ok(Inst::IntUn {
                ty,
                op: o,
                a: self.value(names)?,
            });
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

    fn parse_float_inst(
        &mut self,
        ty: FloatTy,
        suffix: &str,
        op: &str,
        names: &HashMap<String, u32>,
    ) -> Result<Inst, ParseError> {
        if suffix == "const" {
            let v = self.parse_float()?;
            return Ok(match ty {
                FloatTy::F32 => Inst::ConstF32((v as f32).to_bits()),
                FloatTy::F64 => Inst::ConstF64(v.to_bits()),
            });
        }
        if let Some(o) = FBinOp::from_name(suffix) {
            let a = self.value(names)?;
            let b = self.value(names)?;
            return Ok(Inst::FBin { ty, op: o, a, b });
        }
        if let Some(o) = FUnOp::from_name(suffix) {
            return Ok(Inst::FUn {
                ty,
                op: o,
                a: self.value(names)?,
            });
        }
        if let Some(o) = FCmpOp::from_name(suffix) {
            let a = self.value(names)?;
            let b = self.value(names)?;
            return Ok(Inst::FCmp { ty, op: o, a, b });
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
            "unreachable" => PTerm::Unreachable,
            "return_call" => {
                let n = self.parse_int()?;
                let func = u32::try_from(n)
                    .map_err(|_| ParseError(format!("function index out of range: {n}")))?;
                let args = self.parse_value_list(names)?;
                PTerm::ReturnCall { func, args }
            }
            "return_call_indirect" => {
                let params = self.parse_type_list()?;
                self.expect(&Tok::Arrow)?;
                let results = self.parse_type_list()?;
                let idx = self.value(names)?;
                let args = self.parse_value_list(names)?;
                PTerm::ReturnCallIndirect {
                    ty: FuncType { params, results },
                    idx,
                    args,
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
        let args = self.parse_value_list(names)?;
        Ok((label, args))
    }

    /// Parse a parenthesized, comma-separated value list `(v, v, ...)`.
    fn parse_value_list(&mut self, names: &HashMap<String, u32>) -> Result<Vec<u32>, ParseError> {
        let mut args = Vec::new();
        self.expect(&Tok::LParen)?;
        while self.peek() != Some(&Tok::RParen) {
            if !args.is_empty() {
                self.expect(&Tok::Comma)?;
            }
            args.push(self.value(names)?);
        }
        self.expect(&Tok::RParen)?;
        Ok(args)
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

    /// Parse a byte-string literal (data-segment bytes).
    fn parse_str(&mut self) -> Result<Vec<u8>, ParseError> {
        match self.next()? {
            Tok::Str(b) => Ok(b.clone()),
            other => err(format!("expected a string, found {other:?}")),
        }
    }

    /// Parse the optional `offset=<int>` / `align=<int>` suffix of a memory op
    /// (either order, both optional; defaults 0).
    fn parse_memarg(&mut self) -> Result<(u64, u8), ParseError> {
        let mut offset = 0u64;
        let mut align = 0u8;
        while let Some(Tok::Ident(s)) = self.peek() {
            let key = s.clone();
            match key.as_str() {
                "offset" => {
                    self.next()?;
                    self.expect(&Tok::Equals)?;
                    let v = self.parse_int()?;
                    offset = u64::try_from(v)
                        .map_err(|_| ParseError(format!("offset out of range: {v}")))?;
                }
                "align" => {
                    self.next()?;
                    self.expect(&Tok::Equals)?;
                    let v = self.parse_int()?;
                    align = u8::try_from(v)
                        .map_err(|_| ParseError(format!("align out of range: {v}")))?;
                }
                _ => break,
            }
        }
        Ok((offset, align))
    }

    /// A float literal — accepts an integer token too (e.g. `f64.const 2`).
    fn parse_float(&mut self) -> Result<f64, ParseError> {
        match self.next()? {
            Tok::Float(v) => Ok(*v),
            Tok::Int(v) => Ok(*v as f64),
            other => err(format!("expected number, found {other:?}")),
        }
    }
}

#[cfg(test)]
mod import_text_tests {
    use super::*;
    use svm_ir::{Inst, ResolvedCap};

    const SRC: &str = "\
memory 16
import 0 \"write\" (i64, i64) -> (i64)
import 1 \"exit\" (i32) -> ()

func (i32) -> () {
block0(v0: i32):
  v1 = i64.const 0
  v2 = i64.const 3
  v3 = call.import 0 v0 (v1, v2)
  v4 = i32.const 0
  call.import 1 v0 (v4)
  return
}
";

    #[test]
    fn imports_round_trip() {
        let m = parse_module(SRC).expect("parse");
        assert_eq!(m.imports.len(), 2);
        assert_eq!(m.imports[0].name, "write");
        assert_eq!(m.imports[1].name, "exit");
        // The body carries two CallImports, sigs recovered from the import table.
        let insts = &m.funcs[0].blocks[0].insts;
        let imports: Vec<_> = insts
            .iter()
            .filter_map(|i| match i {
                Inst::CallImport { import, handle, .. } => Some((*import, *handle)),
                _ => None,
            })
            .collect();
        assert_eq!(imports, vec![(0, 0), (1, 0)]);
        // Print → re-parse is identity.
        let printed = print_module(&m);
        let m2 = parse_module(&printed).expect("reparse");
        assert_eq!(m, m2, "import syntax must round-trip");
    }

    #[test]
    fn resolves_to_capcalls_and_clears_imports() {
        let m = parse_module(SRC).expect("parse");
        let r = svm_ir::resolve_imports(&m, |n| match n {
            "write" => Some(ResolvedCap { type_id: 0, op: 1 }),
            "exit" => Some(ResolvedCap { type_id: 1, op: 0 }),
            _ => None,
        })
        .expect("resolve");
        assert!(r.imports.is_empty());
        let insts = &r.funcs[0].blocks[0].insts;
        assert!(!insts.iter().any(|i| matches!(i, Inst::CallImport { .. })));
        // write → cap.call 0 1, exit → cap.call 1 0.
        let caps: Vec<_> = insts
            .iter()
            .filter_map(|i| match i {
                Inst::CapCall { type_id, op, .. } => Some((*type_id, *op)),
                _ => None,
            })
            .collect();
        assert_eq!(caps, vec![(0, 1), (1, 0)]);
    }
}

#[cfg(test)]
mod import_inline_tests {
    use super::*;
    use svm_ir::Inst;

    // Name-inline `call.import "name" (sig) v<h> (args)` needs no `import` declaration: the
    // parser interns the name. Two sites with the same name+sig share one import index.
    #[test]
    fn name_inline_interns_imports() {
        let src = "\
func (i32) -> () {
block0(v0: i32):
  v1 = i64.const 0
  v2 = i64.const 3
  v3 = call.import \"write\" (i64, i64) -> (i64) v0 (v1, v2)
  v4 = call.import \"write\" (i64, i64) -> (i64) v0 (v1, v2)
  v5 = i32.const 0
  call.import \"exit\" (i32) -> () v0 (v5)
  return
}
";
        let m = parse_module(src).expect("parse name-inline imports");
        // Two distinct names → two interned imports; the repeated "write" reuses index 0.
        assert_eq!(m.imports.len(), 2);
        assert_eq!(m.imports[0].name, "write");
        assert_eq!(m.imports[1].name, "exit");
        let idxs: Vec<u32> = m.funcs[0].blocks[0]
            .insts
            .iter()
            .filter_map(|i| match i {
                Inst::CallImport { import, .. } => Some(*import),
                _ => None,
            })
            .collect();
        assert_eq!(idxs, vec![0, 0, 1], "repeated write shares index 0");
        // Canonical print → re-parse is identity (printer emits the indexed form + decls).
        assert_eq!(parse_module(&print_module(&m)).unwrap(), m);
    }
}
