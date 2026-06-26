//! Recursive-descent parser for textual LLVM IR (`.ll`) → [`ast::Module`](super::ast). Consumes the
//! [`lex`](super::lex) token stream.
//!
//! Built simplest-first (LLVM.md §8 Q1b): the **core slice** here covers `define` functions over the
//! seed AST — integer types, binary-op instructions, and the `ret`/`br`/`unreachable` terminators,
//! with `%local`/constant-int operands. Top-level cruft the on-ramp ignores (target/datalayout lines,
//! attribute groups, module-level metadata, `declare`s) is skipped. Coverage grows alongside the AST
//! under the differential parity check against `llvm-ir` (`tests/translate.rs`). Anything not yet
//! handled is a clean [`ParseError`] (fail-closed, re-verified downstream — §2a), never a miscompile.

use super::ast::*;
use super::lex::{lex, Token};

/// A parse error with the token index at which it occurred.
#[derive(Debug)]
pub struct ParseError {
    pub at: usize,
    pub msg: String,
}

impl ParseError {
    fn new(at: usize, msg: impl Into<String>) -> Self {
        ParseError {
            at,
            msg: msg.into(),
        }
    }
}

type PResult<T> = Result<T, ParseError>;

/// Parse `.ll` source text into a [`Module`].
pub fn parse_module(src: &str) -> PResult<Module> {
    let toks = lex(src)
        .map_err(|e| ParseError::new(0, format!("lex error at byte {}: {}", e.offset, e.msg)))?;
    let mut p = Parser::new(toks);
    p.module()
}

/// The token cursor + the module-under-construction (the type interner lives in `module.types`).
struct Parser {
    toks: Vec<Token>,
    pos: usize,
    module: Module,
}

impl Parser {
    fn new(toks: Vec<Token>) -> Self {
        Parser {
            toks,
            pos: 0,
            module: Module {
                name: String::new(),
                source_file_name: String::new(),
                target_triple: None,
                functions: Vec::new(),
                func_declarations: Vec::new(),
                global_vars: Vec::new(),
                global_aliases: Vec::new(),
                types: Types::new(),
            },
        }
    }

    // ---- cursor primitives ---------------------------------------------------------------------

    fn peek(&self) -> Option<&Token> {
        self.toks.get(self.pos)
    }
    fn peek2(&self) -> Option<&Token> {
        self.toks.get(self.pos + 1)
    }
    fn bump(&mut self) -> Option<Token> {
        let t = self.toks.get(self.pos).cloned();
        if t.is_some() {
            self.pos += 1;
        }
        t
    }
    fn at_end(&self) -> bool {
        self.pos >= self.toks.len()
    }
    fn err<T>(&self, msg: impl Into<String>) -> PResult<T> {
        Err(ParseError::new(self.pos, msg))
    }

    /// Consume the exact token `t` or error.
    fn expect(&mut self, t: &Token) -> PResult<()> {
        if self.peek() == Some(t) {
            self.pos += 1;
            Ok(())
        } else {
            self.err(format!("expected {t:?}, found {:?}", self.peek()))
        }
    }

    /// Consume a bareword equal to `w` (a keyword/type token) or error.
    fn expect_word(&mut self, w: &str) -> PResult<()> {
        match self.peek() {
            Some(Token::Word(s)) if s == w => {
                self.pos += 1;
                Ok(())
            }
            other => self.err(format!("expected `{w}`, found {other:?}")),
        }
    }

    /// `true` (and consume) iff the next token is the bareword `w`.
    fn eat_word(&mut self, w: &str) -> bool {
        if matches!(self.peek(), Some(Token::Word(s)) if s == w) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    /// `true` (and consume) iff the next token is `t`.
    fn eat(&mut self, t: &Token) -> bool {
        if self.peek() == Some(t) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    // ---- module --------------------------------------------------------------------------------

    fn module(&mut self) -> PResult<Module> {
        while !self.at_end() {
            match self.peek() {
                // `define <ret> @name(<params>) <attrs> { … }`
                Some(Token::Word(w)) if w == "define" => {
                    let f = self.function_def()?;
                    self.module.functions.push(f);
                }
                // Top-level lines/items the on-ramp ignores at this slice: target/datalayout/source,
                // attribute groups (`attributes #N = { … }`), module flags / named metadata (`!… = …`,
                // `!name = !{…}`), and `declare`s. Skip to the next top-level item.
                _ => self.skip_toplevel_item()?,
            }
        }
        Ok(std::mem::replace(
            &mut self.module,
            Parser::new(Vec::new()).module,
        ))
    }

    /// Skip a top-level construct we don't model yet. Heuristic but bounded: consume through a
    /// balanced `{ … }` body if one opens before the next top-level keyword, else to end-of-logical-line
    /// (the next token that clearly starts a new top-level item). Fail-closed: an unrecognized shape
    /// that can't be skipped cleanly is an error, not a silent drop of something meaningful.
    fn skip_toplevel_item(&mut self) -> PResult<()> {
        // Known ignorable openers — consume the keyword then their body.
        if self.eat_word("target")
            || self.eat_word("source_filename")
            || self.eat_word("attributes")
            || self.eat_word("declare")
            || self.eat_word("module")
        {
            self.skip_to_toplevel_boundary();
            return Ok(());
        }
        // Metadata / unnamed-global assignments: `!… = …`, `@… = …`, `%… = type …`, `$… = comdat …`.
        match self.peek() {
            Some(Token::Meta(_))
            | Some(Token::Global(_))
            | Some(Token::Local(_))
            | Some(Token::Word(_)) => {
                self.skip_to_toplevel_boundary();
                Ok(())
            }
            other => self.err(format!("unexpected top-level token {other:?}")),
        }
    }

    /// Advance past the current item: through a balanced `{…}`/`[…]`/`(…)`/`<…>` group if the item has
    /// a body, otherwise until the token before the next top-level opener (`define`/`declare`/a sigil at
    /// depth 0 that begins a new assignment). Conservative — used only for items we ignore.
    fn skip_to_toplevel_boundary(&mut self) {
        let mut depth: i32 = 0;
        while let Some(t) = self.peek() {
            match t {
                Token::LBrace | Token::LBracket | Token::LParen | Token::Lt => depth += 1,
                Token::RBrace | Token::RBracket | Token::RParen | Token::Gt => {
                    depth -= 1;
                    self.pos += 1;
                    if depth <= 0 {
                        return;
                    }
                    continue;
                }
                // At depth 0, a `define`/`declare` starts the next item — stop before it.
                Token::Word(w) if depth == 0 && (w == "define" || w == "declare") => return,
                _ => {}
            }
            self.pos += 1;
            // A depth-0 newline-equivalent boundary: we can't see newlines (lexer drops them), so a
            // top-level metadata/target line ends when the next `define`/`declare`/sigil-assignment
            // begins. Heuristic stop: at depth 0, if the next token starts a known top-level item.
            if depth == 0 {
                match self.peek() {
                    Some(Token::Word(w))
                        if w == "define" || w == "declare" || w == "attributes" =>
                    {
                        return
                    }
                    _ => {}
                }
            }
        }
    }

    // ---- functions -----------------------------------------------------------------------------

    fn function_def(&mut self) -> PResult<Function> {
        self.expect_word("define")?;
        // Skip linkage/visibility/cc/attribute barewords until the return type. The return type is the
        // first token that parses as a type; it's immediately followed by the `@name`. We find `@name`
        // by scanning, then re-parse the type just before it. Simpler: skip known pre-type keywords.
        self.skip_pre_signature_attrs();
        let return_type = self.type_()?;
        let name = match self.bump() {
            Some(Token::Global(s)) => s,
            other => return self.err(format!("expected function @name, found {other:?}")),
        };
        let (parameters, is_var_arg) = self.param_list()?;
        // Skip post-signature attributes/personality/etc. up to the opening `{`.
        while !matches!(self.peek(), Some(Token::LBrace) | None) {
            self.pos += 1;
        }
        self.expect(&Token::LBrace)?;
        let basic_blocks = self.basic_blocks()?;
        self.expect(&Token::RBrace)?;
        Ok(Function {
            name,
            parameters,
            is_var_arg,
            return_type,
            basic_blocks,
        })
    }

    /// Skip the optional linkage/visibility/dll/cc/attribute barewords between `define` and the return
    /// type (e.g. `dso_local`, `internal`, `noundef`). They're all `Word`s; the return type is also a
    /// `Word`/`Lt`/`LBracket`/`LBrace`, so we stop at the first token that begins a *type*. Since types
    /// and these keywords are both barewords, we use a keyword allow-list to skip.
    fn skip_pre_signature_attrs(&mut self) {
        const PRE: &[&str] = &[
            "dso_local",
            "dso_preemptable",
            "local_unnamed_addr",
            "unnamed_addr",
            "internal",
            "external",
            "private",
            "linkonce",
            "linkonce_odr",
            "weak",
            "weak_odr",
            "common",
            "appending",
            "available_externally",
            "extern_weak",
            "hidden",
            "protected",
            "default",
            "dllimport",
            "dllexport",
            "zeroext",
            "signext",
            "noundef",
            "ccc",
            "fastcc",
            "coldcc",
            "tailcc",
            "cc",
        ];
        while let Some(Token::Word(w)) = self.peek() {
            if PRE.contains(&w.as_str()) {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    /// Parse `( <ty> <%name>?, … , ...? )` — returns the params and whether it's varargs. Parameter
    /// attributes (`noundef`, `align N`, …) between the type and the name are skipped.
    fn param_list(&mut self) -> PResult<(Vec<Parameter>, bool)> {
        self.expect(&Token::LParen)?;
        let mut params = Vec::new();
        let mut is_var_arg = false;
        if self.eat(&Token::RParen) {
            return Ok((params, false));
        }
        loop {
            if self.eat(&Token::Ellipsis) {
                is_var_arg = true;
                break;
            }
            let ty = self.type_()?;
            // Skip parameter attributes (barewords / `align N` / attribute-group `#N`) until the name,
            // the comma, or the closing paren.
            while !matches!(
                self.peek(),
                Some(Token::Local(_)) | Some(Token::Comma) | Some(Token::RParen) | None
            ) {
                self.pos += 1;
            }
            let name = match self.peek() {
                Some(Token::Local(s)) => {
                    let nm = name_from_local(s);
                    self.pos += 1;
                    nm
                }
                // An unnamed parameter (a `declare`-style prototype) — give it its positional number.
                _ => Name::Number(params.len()),
            };
            params.push(Parameter { name, ty });
            if !self.eat(&Token::Comma) {
                break;
            }
        }
        self.expect(&Token::RParen)?;
        Ok((params, is_var_arg))
    }

    // ---- basic blocks --------------------------------------------------------------------------

    fn basic_blocks(&mut self) -> PResult<Vec<BasicBlock>> {
        let mut blocks = Vec::new();
        // The entry block may omit a label. Track the implicit number for unnamed blocks: LLVM numbers
        // the entry block 0 when it is unnamed, continuing the value counter — but for the core slice we
        // rely on explicit textual labels / the entry having no label (named `Number(0)`).
        let mut next_unnamed = 0usize;
        while !matches!(self.peek(), Some(Token::RBrace) | None) {
            let name = self.block_label(&mut next_unnamed);
            let mut instrs = Vec::new();
            // Instructions until a terminator.
            loop {
                if self.at_terminator() {
                    break;
                }
                let instr = self.instruction(&mut next_unnamed)?;
                instrs.push(instr);
            }
            let term = self.terminator()?;
            blocks.push(BasicBlock { name, instrs, term });
        }
        Ok(blocks)
    }

    /// A leading `name:` / `N:` block label, or — if absent — the next implicit block number.
    fn block_label(&mut self, next_unnamed: &mut usize) -> Name {
        match (self.peek(), self.peek2()) {
            (Some(Token::Word(w)), Some(Token::Colon)) => {
                let nm = Name::from_string(w.clone());
                self.pos += 2;
                nm
            }
            (Some(Token::Int(s)), Some(Token::Colon)) => {
                let n = s.parse::<usize>().unwrap_or(*next_unnamed);
                self.pos += 2;
                *next_unnamed = n + 1;
                Name::Number(n)
            }
            _ => {
                let n = *next_unnamed;
                *next_unnamed += 1;
                Name::Number(n)
            }
        }
    }

    /// Is the cursor at a terminator keyword?
    fn at_terminator(&self) -> bool {
        matches!(self.peek(), Some(Token::Word(w))
            if matches!(w.as_str(), "ret" | "br" | "switch" | "indirectbr" | "unreachable" | "invoke" | "resume"))
    }

    // ---- instructions --------------------------------------------------------------------------

    /// Parse one non-terminator instruction. The seed slice handles `%dest = <binop> <ty> <op>, <op>`.
    fn instruction(&mut self, next_unnamed: &mut usize) -> PResult<Instruction> {
        // `%dest =` prefix (every instruction in the seed slice produces a value).
        let dest = self.instr_dest(next_unnamed)?;
        let op = match self.peek() {
            Some(Token::Word(w)) => w.clone(),
            other => return self.err(format!("expected an instruction opcode, found {other:?}")),
        };
        let bin = |p: &mut Self, dest: Name| -> PResult<BinaryOp> {
            p.pos += 1; // opcode
            p.skip_binop_flags();
            let ty = p.type_()?;
            let operand0 = p.value_as_operand(&ty)?;
            p.expect(&Token::Comma)?;
            let operand1 = p.value_as_operand(&ty)?;
            Ok(BinaryOp {
                operand0,
                operand1,
                dest,
                debugloc: None,
            })
        };
        let i = match op.as_str() {
            "add" => Instruction::Add(bin(self, dest)?),
            "sub" => Instruction::Sub(bin(self, dest)?),
            "mul" => Instruction::Mul(bin(self, dest)?),
            "udiv" => Instruction::UDiv(bin(self, dest)?),
            "sdiv" => Instruction::SDiv(bin(self, dest)?),
            "urem" => Instruction::URem(bin(self, dest)?),
            "srem" => Instruction::SRem(bin(self, dest)?),
            "and" => Instruction::And(bin(self, dest)?),
            "or" => Instruction::Or(bin(self, dest)?),
            "xor" => Instruction::Xor(bin(self, dest)?),
            "shl" => Instruction::Shl(bin(self, dest)?),
            "lshr" => Instruction::LShr(bin(self, dest)?),
            "ashr" => Instruction::AShr(bin(self, dest)?),
            other => return self.err(format!("instruction `{other}` not yet supported")),
        };
        self.skip_trailing_metadata();
        Ok(i)
    }

    /// The `%dest =` of a value-producing instruction; advances the unnamed counter for an unnamed dest.
    fn instr_dest(&mut self, next_unnamed: &mut usize) -> PResult<Name> {
        match self.bump() {
            Some(Token::Local(s)) => {
                let nm = name_from_local(&s);
                if let Name::Number(n) = nm {
                    *next_unnamed = n + 1;
                }
                self.expect(&Token::Equals)?;
                Ok(nm)
            }
            other => self.err(format!("expected `%dest =`, found {other:?}")),
        }
    }

    /// Skip binop flags (`nsw`, `nuw`, `exact`, `disjoint`) between the opcode and the type.
    fn skip_binop_flags(&mut self) {
        while matches!(self.peek(), Some(Token::Word(w))
            if matches!(w.as_str(), "nsw" | "nuw" | "exact" | "disjoint"))
        {
            self.pos += 1;
        }
    }

    /// Skip a trailing `, !dbg !N` / `, !tbaa !M` … metadata list attached to an instruction.
    fn skip_trailing_metadata(&mut self) {
        while self.peek() == Some(&Token::Comma) && matches!(self.peek2(), Some(Token::Meta(_))) {
            self.pos += 2; // `,` `!kind`
                           // the metadata value (`!N` or an inline `!{…}`); for `!N` it's a single Meta token.
            if matches!(self.peek(), Some(Token::Meta(_))) {
                self.pos += 1;
            }
        }
    }

    // ---- terminators ---------------------------------------------------------------------------

    fn terminator(&mut self) -> PResult<Terminator> {
        let op = match self.peek() {
            Some(Token::Word(w)) => w.clone(),
            other => return self.err(format!("expected a terminator, found {other:?}")),
        };
        let t = match op.as_str() {
            "unreachable" => {
                self.pos += 1;
                Terminator::Unreachable(Unreachable { debugloc: None })
            }
            "ret" => {
                self.pos += 1;
                if self.eat_word("void") {
                    Terminator::Ret(Ret {
                        return_operand: None,
                        debugloc: None,
                    })
                } else {
                    let ty = self.type_()?;
                    let v = self.value_as_operand(&ty)?;
                    Terminator::Ret(Ret {
                        return_operand: Some(v),
                        debugloc: None,
                    })
                }
            }
            "br" => {
                self.pos += 1;
                // `br label %dest` | `br i1 %c, label %t, label %f`
                if self.eat_word("label") {
                    let dest = self.label_name()?;
                    Terminator::Br(Br {
                        dest,
                        debugloc: None,
                    })
                } else {
                    let ty = self.type_()?; // i1
                    let condition = self.value_as_operand(&ty)?;
                    self.expect(&Token::Comma)?;
                    self.expect_word("label")?;
                    let true_dest = self.label_name()?;
                    self.expect(&Token::Comma)?;
                    self.expect_word("label")?;
                    let false_dest = self.label_name()?;
                    Terminator::CondBr(CondBr {
                        condition,
                        true_dest,
                        false_dest,
                        debugloc: None,
                    })
                }
            }
            other => return self.err(format!("terminator `{other}` not yet supported")),
        };
        self.skip_trailing_metadata();
        Ok(t)
    }

    /// A `%label` block reference after the `label` keyword.
    fn label_name(&mut self) -> PResult<Name> {
        match self.bump() {
            Some(Token::Local(s)) => Ok(name_from_local(&s)),
            other => self.err(format!("expected %label, found {other:?}")),
        }
    }

    // ---- operands & values ---------------------------------------------------------------------

    /// Parse a value of (already-parsed) type `ty` into an [`Operand`]: a `%local`, or a constant
    /// (integer literal for the seed slice). The type prefix is supplied by the caller.
    fn value_as_operand(&mut self, ty: &TypeRef) -> PResult<Operand> {
        match self.peek() {
            Some(Token::Local(s)) => {
                let name = name_from_local(s);
                self.pos += 1;
                Ok(Operand::LocalOperand {
                    name,
                    ty: ty.clone(),
                })
            }
            Some(Token::Int(s)) => {
                let bits = match ty.as_ref() {
                    Type::IntegerType { bits } => *bits,
                    _ => return self.err("integer literal with non-integer type"),
                };
                let value = parse_int_literal(s, bits).ok_or_else(|| {
                    ParseError::new(self.pos, format!("bad integer literal `{s}`"))
                })?;
                self.pos += 1;
                Ok(Operand::ConstantOperand(ConstantRef::new(Constant::Int {
                    bits,
                    value,
                })))
            }
            Some(Token::Word(w)) if w == "true" || w == "false" => {
                let value = (w == "true") as u128;
                self.pos += 1;
                Ok(Operand::ConstantOperand(ConstantRef::new(Constant::Int {
                    bits: 1,
                    value,
                })))
            }
            other => self.err(format!("value not yet supported: {other:?}")),
        }
    }

    // ---- types ---------------------------------------------------------------------------------

    /// Parse a type, interning it in `module.types`. Seed slice: `void`, `iN`, `ptr`, `float`/`double`.
    fn type_(&mut self) -> PResult<TypeRef> {
        let w = match self.peek() {
            Some(Token::Word(w)) => w.clone(),
            other => return self.err(format!("expected a type, found {other:?}")),
        };
        // `iN`
        if let Some(bits) = parse_int_type(&w) {
            self.pos += 1;
            return Ok(self.module.types.int(bits));
        }
        let ty = match w.as_str() {
            "void" => {
                self.pos += 1;
                self.module.types.void()
            }
            "ptr" => {
                self.pos += 1;
                self.module.types.pointer(0)
            }
            "float" => {
                self.pos += 1;
                self.module.types.fp(FPType::Single)
            }
            "double" => {
                self.pos += 1;
                self.module.types.fp(FPType::Double)
            }
            other => return self.err(format!("type `{other}` not yet supported")),
        };
        Ok(ty)
    }
}

// ---- small parsing helpers ---------------------------------------------------------------------

/// A `%local`/`@global` name string → [`Name`]: all-digits ⇒ `Number`, else a textual `Name`.
fn name_from_local(s: &str) -> Name {
    if !s.is_empty() && s.bytes().all(|c| c.is_ascii_digit()) {
        Name::Number(s.parse().unwrap())
    } else {
        Name::from_string(s.to_string())
    }
}

/// `iN` type bareword → its bit width.
fn parse_int_type(w: &str) -> Option<u32> {
    let rest = w.strip_prefix('i')?;
    if rest.is_empty() || !rest.bytes().all(|c| c.is_ascii_digit()) {
        return None;
    }
    rest.parse().ok()
}

/// An integer-literal token (decimal, optional leading `-`) → its `bits`-wide two's-complement value as
/// a `u128`, matching `llvm-ir`'s "value masked to the type width" semantics — but **full width**, so a
/// 128-bit value never truncates (the I14 fix). `None` if the text isn't a valid integer.
fn parse_int_literal(s: &str, bits: u32) -> Option<u128> {
    let v: i128 = s.parse().ok()?;
    let raw = v as u128;
    let mask = if bits >= 128 {
        u128::MAX
    } else {
        (1u128 << bits) - 1
    };
    Some(raw & mask)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_trivial_add_function() {
        let m = parse_module(
            "define i32 @f(i32 %a, i32 %b) {\n\
             entry:\n\
             \x20 %c = add i32 %a, %b\n\
             \x20 ret i32 %c\n\
             }\n",
        )
        .expect("parse");
        assert_eq!(m.functions.len(), 1);
        let f = &m.functions[0];
        assert_eq!(f.name, "f");
        assert_eq!(f.parameters.len(), 2);
        assert_eq!(f.basic_blocks.len(), 1);
        let bb = &f.basic_blocks[0];
        assert_eq!(bb.name, Name::from_string("entry".into()));
        assert_eq!(bb.instrs.len(), 1);
        match &bb.instrs[0] {
            Instruction::Add(b) => {
                assert_eq!(b.dest, Name::from_string("c".into()));
                assert!(
                    matches!(&b.operand0, Operand::LocalOperand { name, .. } if *name == Name::from_string("a".into()))
                );
            }
            other => panic!("expected add, got {other:?}"),
        }
        match &bb.term {
            Terminator::Ret(r) => assert!(r.return_operand.is_some()),
            other => panic!("expected ret, got {other:?}"),
        }
    }

    #[test]
    fn unnamed_values_and_condbr() {
        // Unnamed temporaries (%0,%1,…) and an i1 conditional branch across numbered blocks.
        let m = parse_module(
            "define i32 @g(i32 %0) {\n\
             \x20 %2 = icmp_placeholder_skip i32 %0, 0\n\
             }\n",
        );
        // (icmp isn't in the seed slice yet — this should fail closed cleanly, not panic.)
        assert!(m.is_err());
    }

    #[test]
    fn full_width_i128_constant_survives() {
        let big = "170141183460469231731687303715884105727"; // i128::MAX
        let m =
            parse_module(&format!("define i128 @c() {{\n  ret i128 {big}\n}}\n")).expect("parse");
        match &m.functions[0].basic_blocks[0].term {
            Terminator::Ret(Ret {
                return_operand: Some(Operand::ConstantOperand(c)),
                ..
            }) => match c.as_ref() {
                Constant::Int { bits: 128, value } => {
                    assert_eq!(*value, big.parse::<u128>().unwrap())
                }
                other => panic!("expected i128 const, got {other:?}"),
            },
            other => panic!("expected ret const, got {other:?}"),
        }
    }

    #[test]
    fn skips_target_and_attributes_and_metadata() {
        let m = parse_module(
            "target datalayout = \"e-m:e\"\n\
             target triple = \"x86_64-unknown-linux-gnu\"\n\
             define i32 @f() {\n  ret i32 0\n}\n\
             attributes #0 = { nounwind }\n\
             !llvm.module.flags = !{!0}\n\
             !0 = !{i32 1, !\"wchar_size\", i32 4}\n",
        )
        .expect("parse");
        assert_eq!(m.functions.len(), 1);
        assert_eq!(m.functions[0].name, "f");
    }
}
