//! Recursive-descent parser for textual LLVM IR (`.ll`) → [`ast::Module`](super::ast). Consumes the
//! [`lex`](super::lex) token stream.
//!
//! Built simplest-first (LLVM.md §8 Q1b), growing under the differential parity check against the
//! bitcode reader (`assert_ll_parity`, `tests/translate.rs`). Covered so far: `define` functions with
//! integer/float types; the instruction set — integer + float **binops**, **conversions**
//! (`trunc`/`zext`/`sext`/`fptrunc`/…/`bitcast`), **`icmp`/`fcmp`**, **`select`**, **`fneg`/`freeze`**,
//! **`phi`**, and **`call`** (direct `@f`/indirect `%fp`, incl. result-less `void` calls and
//! intrinsics, reconstructing the callee's function type) — over `%local`/constant-int operands; the
//! `ret`/`br`/`condbr`/`unreachable` terminators; and **multi-block** CFGs with LLVM's implicit slot
//! numbering (so an unlabeled entry and the phi/branch refs into it resolve to the same `Name`s the
//! bitcode reader assigns). Top-level cruft the on-ramp ignores (target/datalayout lines, attribute
//! groups, module-level metadata, `declare`s) is skipped. Not yet handled (the growth frontier): memory
//! (`getelementptr`/`load`/`store`/`alloca`), globals, `switch`, aggregates, vectors, and
//! float/non-trivial constants. Anything unhandled is a clean [`ParseError`] (fail-closed, re-verified
//! downstream — §2a), never a miscompile.

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

/// A call/invoke argument list: each operand paired with its parameter attributes (always dropped —
/// the translator reads only the operand, matching the bitcode reader's `from_llvm_ir::cvt_args`).
type CallArgs = Vec<(Operand, Vec<ParameterAttribute>)>;

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
        let basic_blocks = self.basic_blocks(parameters.len())?;
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

    fn basic_blocks(&mut self, start_unnamed: usize) -> PResult<Vec<BasicBlock>> {
        let mut blocks = Vec::new();
        // LLVM's implicit slot counter is shared across (unnamed) parameters, blocks, and value-producing
        // instructions, in textual order. So an unlabeled entry block takes the number *after* the
        // parameters — `start_unnamed` is the parameter count. Getting this right is what makes an
        // unlabeled entry referenced as `%N` in a `phi`/`br` resolve to the same `Name` the bitcode
        // reader assigns (e.g. 2 params ⇒ entry is `%2`, first instruction `%3`).
        let mut next_unnamed = start_unnamed;
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

    /// Parse one non-terminator instruction.
    fn instruction(&mut self, next_unnamed: &mut usize) -> PResult<Instruction> {
        // A `call` is handled first: it may have *no* `%dest =` (a `void` call), and an optional
        // `tail`/`musttail`/`notail` marker sits where an opcode otherwise would — both break the
        // "every instruction starts `%dest = <opcode>`" shape the rest of this function assumes.
        if self.at_call() {
            return self.call_inst(next_unnamed).map(Instruction::Call);
        }
        // `%dest =` prefix (every other instruction modeled here produces a value).
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
            "fadd" => Instruction::FAdd(bin(self, dest)?),
            "fsub" => Instruction::FSub(bin(self, dest)?),
            "fmul" => Instruction::FMul(bin(self, dest)?),
            "fdiv" => Instruction::FDiv(bin(self, dest)?),
            "frem" => Instruction::FRem(bin(self, dest)?),
            "trunc" => Instruction::Trunc(self.conv_inst(dest)?),
            "zext" => Instruction::ZExt(self.conv_inst(dest)?),
            "sext" => Instruction::SExt(self.conv_inst(dest)?),
            "fptrunc" => Instruction::FPTrunc(self.conv_inst(dest)?),
            "fpext" => Instruction::FPExt(self.conv_inst(dest)?),
            "fptoui" => Instruction::FPToUI(self.conv_inst(dest)?),
            "fptosi" => Instruction::FPToSI(self.conv_inst(dest)?),
            "uitofp" => Instruction::UIToFP(self.conv_inst(dest)?),
            "sitofp" => Instruction::SIToFP(self.conv_inst(dest)?),
            "ptrtoint" => Instruction::PtrToInt(self.conv_inst(dest)?),
            "inttoptr" => Instruction::IntToPtr(self.conv_inst(dest)?),
            "bitcast" => Instruction::BitCast(self.conv_inst(dest)?),
            "addrspacecast" => Instruction::AddrSpaceCast(self.conv_inst(dest)?),
            "icmp" => Instruction::ICmp(self.icmp_inst(dest)?),
            "fcmp" => Instruction::FCmp(self.fcmp_inst(dest)?),
            "select" => Instruction::Select(self.select_inst(dest)?),
            "fneg" => Instruction::FNeg(self.fneg_inst(dest)?),
            "freeze" => Instruction::Freeze(self.freeze_inst(dest)?),
            "phi" => Instruction::Phi(self.phi_inst(dest)?),
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

    // ---- conversions / compares / select -------------------------------------------------------

    /// A conversion (`trunc`/`zext`/`sext`/`fptrunc`/…/`bitcast`): `<op> [flags] <srcty> <val> to <dstty>`.
    fn conv_inst(&mut self, dest: Name) -> PResult<UnaryOp> {
        self.pos += 1; // opcode
        self.skip_conv_flags();
        let srcty = self.type_()?;
        let operand = self.value_as_operand(&srcty)?;
        self.expect_word("to")?;
        let to_type = self.type_()?;
        Ok(UnaryOp {
            operand,
            to_type,
            dest,
            debugloc: None,
        })
    }

    /// `icmp <pred> <ty> <op0>, <op1>`.
    fn icmp_inst(&mut self, dest: Name) -> PResult<ICmp> {
        self.pos += 1; // `icmp`
        let predicate = self.int_predicate()?;
        let ty = self.type_()?;
        let operand0 = self.value_as_operand(&ty)?;
        self.expect(&Token::Comma)?;
        let operand1 = self.value_as_operand(&ty)?;
        Ok(ICmp {
            predicate,
            operand0,
            operand1,
            dest,
            debugloc: None,
        })
    }

    /// `fcmp [fast-math flags] <pred> <ty> <op0>, <op1>`.
    fn fcmp_inst(&mut self, dest: Name) -> PResult<FCmp> {
        self.pos += 1; // `fcmp`
        self.skip_fast_math_flags();
        let predicate = self.fp_predicate()?;
        let ty = self.type_()?;
        let operand0 = self.value_as_operand(&ty)?;
        self.expect(&Token::Comma)?;
        let operand1 = self.value_as_operand(&ty)?;
        Ok(FCmp {
            predicate,
            operand0,
            operand1,
            dest,
            debugloc: None,
        })
    }

    /// `select [fast-math flags] <condty> <cond>, <ty> <a>, <ty> <b>`.
    fn select_inst(&mut self, dest: Name) -> PResult<Select> {
        self.pos += 1; // `select`
        self.skip_fast_math_flags();
        let cty = self.type_()?;
        let condition = self.value_as_operand(&cty)?;
        self.expect(&Token::Comma)?;
        let tty = self.type_()?;
        let true_value = self.value_as_operand(&tty)?;
        self.expect(&Token::Comma)?;
        let fty = self.type_()?;
        let false_value = self.value_as_operand(&fty)?;
        Ok(Select {
            condition,
            true_value,
            false_value,
            dest,
            debugloc: None,
        })
    }

    /// `fneg [fast-math flags] <ty> <val>` — result type is the operand type.
    fn fneg_inst(&mut self, dest: Name) -> PResult<UnaryOp> {
        self.pos += 1; // `fneg`
        self.skip_fast_math_flags();
        let ty = self.type_()?;
        let operand = self.value_as_operand(&ty)?;
        Ok(UnaryOp {
            operand,
            to_type: ty,
            dest,
            debugloc: None,
        })
    }

    /// `phi <ty> [ <val0>, %<blk0> ], [ <val1>, %<blk1> ], …` — each incoming value is paired with the
    /// predecessor block it arrives from. The block refs are names the *terminators* of those blocks
    /// also use, so correct implicit block numbering (params counted, see [`Self::basic_blocks`]) is
    /// what makes the pairs resolve identically to the bitcode reader.
    fn phi_inst(&mut self, dest: Name) -> PResult<Phi> {
        self.pos += 1; // `phi`
        let to_type = self.type_()?;
        let mut incoming_values = Vec::new();
        loop {
            self.expect(&Token::LBracket)?;
            let val = self.value_as_operand(&to_type)?;
            self.expect(&Token::Comma)?;
            let block = self.label_name()?;
            self.expect(&Token::RBracket)?;
            incoming_values.push((val, block));
            if !self.eat(&Token::Comma) {
                break;
            }
        }
        Ok(Phi {
            incoming_values,
            dest,
            to_type,
            debugloc: None,
        })
    }

    /// `freeze <ty> <val>` — result type is the operand type.
    fn freeze_inst(&mut self, dest: Name) -> PResult<UnaryOp> {
        self.pos += 1; // `freeze`
        let ty = self.type_()?;
        let operand = self.value_as_operand(&ty)?;
        Ok(UnaryOp {
            operand,
            to_type: ty,
            dest,
            debugloc: None,
        })
    }

    // ---- calls ---------------------------------------------------------------------------------

    /// Does the cursor begin a `call`? Looks past an optional `%dest =` to the
    /// `call`/`tail`/`musttail`/`notail` keyword — the lookahead `instruction()` needs to route a
    /// (possibly result-less) call before assuming the leading `%dest =`.
    fn at_call(&self) -> bool {
        let mut i = self.pos;
        if matches!(self.toks.get(i), Some(Token::Local(_)))
            && matches!(self.toks.get(i + 1), Some(Token::Equals))
        {
            i += 2;
        }
        matches!(self.toks.get(i), Some(Token::Word(w))
            if matches!(w.as_str(), "call" | "tail" | "musttail" | "notail"))
    }

    /// `[%dest =] [tail|musttail|notail] call [fmf] [cconv/ret-attrs] <retty>|<fnty> <callee>(<args>)
    /// [#N] [, !meta]`. The callee `@g` becomes a `GlobalReference` whose `ty` is the reconstructed
    /// function type (return type + the arg types, or the explicit `<ret> (<params>)` signature for a
    /// vararg call) — the same shape the bitcode reader carries, so a direct call reaches parity.
    fn call_inst(&mut self, next_unnamed: &mut usize) -> PResult<Call> {
        let dest = if matches!(
            (self.peek(), self.peek2()),
            (Some(Token::Local(_)), Some(Token::Equals))
        ) {
            Some(self.instr_dest(next_unnamed)?)
        } else {
            None
        };
        let is_tail_call = match self.peek() {
            Some(Token::Word(w)) if matches!(w.as_str(), "tail" | "musttail") => {
                self.pos += 1;
                true
            }
            Some(Token::Word(w)) if w == "notail" => {
                self.pos += 1;
                false
            }
            _ => false,
        };
        self.expect_word("call")?;
        self.skip_fast_math_flags();
        self.skip_pre_signature_attrs(); // calling convention + return attributes
        let ret_ty = self.type_()?;
        // An explicit function-pointer type (`<ret> (<params>[, ...])`) — present for vararg/indirect
        // calls. Otherwise the function type is reconstructed from the return type + the argument types.
        let explicit_fnty = if self.peek() == Some(&Token::LParen) {
            let (param_types, is_var_arg) = self.fn_type_params()?;
            Some(TypeRef::new(Type::FuncType {
                result_type: ret_ty.clone(),
                param_types,
                is_var_arg,
            }))
        } else {
            None
        };
        // The callee name (a global `@f` or an indirect `%fp`); the operand is built once the function
        // type is known (after the arguments, in the reconstructed case).
        enum Callee {
            Global(Name),
            Reg(Name),
        }
        let callee = match self.bump() {
            Some(Token::Global(s)) => Callee::Global(name_from_local(&s)),
            Some(Token::Local(s)) => Callee::Reg(name_from_local(&s)),
            other => return self.err(format!("expected a call callee, found {other:?}")),
        };
        let (arguments, arg_types) = self.call_arg_list()?;
        let function_ty = explicit_fnty.unwrap_or_else(|| {
            TypeRef::new(Type::FuncType {
                result_type: ret_ty,
                param_types: arg_types,
                is_var_arg: false,
            })
        });
        let function = match callee {
            Callee::Global(name) => Either::Right(Operand::ConstantOperand(ConstantRef::new(
                Constant::GlobalReference {
                    name,
                    ty: function_ty.clone(),
                },
            ))),
            Callee::Reg(name) => Either::Right(Operand::LocalOperand {
                name,
                ty: self.module.types.pointer(0),
            }),
        };
        // Trailing attribute-group refs (`#4`) and `, !dbg`-style metadata.
        while matches!(self.peek(), Some(Token::Word(w)) if w.starts_with('#')) {
            self.pos += 1;
        }
        self.skip_trailing_metadata();
        Ok(Call {
            function,
            function_ty,
            arguments,
            dest,
            is_tail_call,
            debugloc: None,
        })
    }

    /// A call argument list `( <ty> [attrs] <val>, … )` — returns the operands (attributes dropped, as
    /// the translator only reads the operand) alongside the parsed types (to reconstruct the fn type).
    fn call_arg_list(&mut self) -> PResult<(CallArgs, Vec<TypeRef>)> {
        self.expect(&Token::LParen)?;
        let mut args = Vec::new();
        let mut types = Vec::new();
        if self.eat(&Token::RParen) {
            return Ok((args, types));
        }
        loop {
            let ty = self.type_()?;
            self.skip_arg_attrs();
            let val = self.value_as_operand(&ty)?;
            types.push(ty);
            args.push((val, Vec::new()));
            if !self.eat(&Token::Comma) {
                break;
            }
        }
        self.expect(&Token::RParen)?;
        Ok((args, types))
    }

    /// The `( <ty>, … [, ...] )` of an explicit function-pointer type in a call.
    fn fn_type_params(&mut self) -> PResult<(Vec<TypeRef>, bool)> {
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
            params.push(self.type_()?);
            self.skip_arg_attrs();
            if !self.eat(&Token::Comma) {
                break;
            }
        }
        self.expect(&Token::RParen)?;
        Ok((params, is_var_arg))
    }

    /// Skip the parameter attributes between an argument's type and its value (`noundef`, `nonnull`,
    /// `align N`, `byval(<ty>)`, `dereferenceable(N)`, …). A value word (`null`/`true`/…) stops it.
    fn skip_arg_attrs(&mut self) {
        const ATTRS: &[&str] = &[
            "noundef",
            "nonnull",
            "signext",
            "zeroext",
            "inreg",
            "returned",
            "nest",
            "sret",
            "byval",
            "byref",
            "preallocated",
            "inalloca",
            "noalias",
            "nocapture",
            "readonly",
            "readnone",
            "writeonly",
            "immarg",
            "nofree",
            "align",
            "dereferenceable",
            "dereferenceable_or_null",
        ];
        while let Some(Token::Word(w)) = self.peek() {
            if !ATTRS.contains(&w.as_str()) {
                break;
            }
            let is_align = w == "align";
            self.pos += 1; // the attribute word
            if self.peek() == Some(&Token::LParen) {
                // A parenthesized payload — `byval(<ty>)`, `dereferenceable(N)`, … — skipped balanced.
                let mut depth = 0usize;
                loop {
                    match self.bump() {
                        Some(Token::LParen) => depth += 1,
                        Some(Token::RParen) => {
                            depth -= 1;
                            if depth == 0 {
                                break;
                            }
                        }
                        None => break,
                        _ => {}
                    }
                }
            } else if is_align && matches!(self.peek(), Some(Token::Int(_))) {
                self.pos += 1; // `align N`
            }
        }
    }

    /// Skip conversion flags (`nneg` on `zext`, `nuw`/`nsw` on `trunc`).
    fn skip_conv_flags(&mut self) {
        while matches!(self.peek(), Some(Token::Word(w))
            if matches!(w.as_str(), "nneg" | "nuw" | "nsw"))
        {
            self.pos += 1;
        }
    }

    /// Skip fast-math flags (`nnan`/`ninf`/`nsz`/`arcp`/`contract`/`afn`/`reassoc`/`fast`) on a float op.
    fn skip_fast_math_flags(&mut self) {
        while matches!(self.peek(), Some(Token::Word(w)) if matches!(w.as_str(),
            "nnan" | "ninf" | "nsz" | "arcp" | "contract" | "afn" | "reassoc" | "fast"))
        {
            self.pos += 1;
        }
    }

    fn int_predicate(&mut self) -> PResult<IntPredicate> {
        let p = match self.peek() {
            Some(Token::Word(w)) => match w.as_str() {
                "eq" => IntPredicate::EQ,
                "ne" => IntPredicate::NE,
                "ugt" => IntPredicate::UGT,
                "uge" => IntPredicate::UGE,
                "ult" => IntPredicate::ULT,
                "ule" => IntPredicate::ULE,
                "sgt" => IntPredicate::SGT,
                "sge" => IntPredicate::SGE,
                "slt" => IntPredicate::SLT,
                "sle" => IntPredicate::SLE,
                other => return self.err(format!("unknown icmp predicate `{other}`")),
            },
            other => return self.err(format!("expected an icmp predicate, found {other:?}")),
        };
        self.pos += 1;
        Ok(p)
    }

    fn fp_predicate(&mut self) -> PResult<FPPredicate> {
        let p = match self.peek() {
            Some(Token::Word(w)) => match w.as_str() {
                "false" => FPPredicate::False,
                "oeq" => FPPredicate::OEQ,
                "ogt" => FPPredicate::OGT,
                "oge" => FPPredicate::OGE,
                "olt" => FPPredicate::OLT,
                "ole" => FPPredicate::OLE,
                "one" => FPPredicate::ONE,
                "ord" => FPPredicate::ORD,
                "uno" => FPPredicate::UNO,
                "ueq" => FPPredicate::UEQ,
                "ugt" => FPPredicate::UGT,
                "uge" => FPPredicate::UGE,
                "ult" => FPPredicate::ULT,
                "ule" => FPPredicate::ULE,
                "une" => FPPredicate::UNE,
                "true" => FPPredicate::True,
                other => return self.err(format!("unknown fcmp predicate `{other}`")),
            },
            other => return self.err(format!("expected an fcmp predicate, found {other:?}")),
        };
        self.pos += 1;
        Ok(p)
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
