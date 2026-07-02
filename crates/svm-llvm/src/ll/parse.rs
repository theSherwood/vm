//! Recursive-descent parser for textual LLVM IR (`.ll`) → [`ast::Module`](super::ast). Consumes the
//! [`lex`](super::lex) token stream.
//!
//! Built simplest-first (LLVM.md §8 Q1b), growing under the differential parity check against the
//! bitcode reader (`assert_ll_parity`, `tests/translate.rs`). Covered so far: `define` functions with
//! integer/float types; the instruction set — integer + float **binops**, **conversions**
//! (`trunc`/`zext`/`sext`/`fptrunc`/…/`bitcast`), **`icmp`/`fcmp`**, **`select`**, **`fneg`/`freeze`**,
//! **`phi`**, and **`call`** (direct `@f`/indirect `%fp`, incl. result-less `void` calls and
//! intrinsics, reconstructing the callee's function type) — over `%local`/constant-int operands; the
//! `ret`/`br`/`condbr`/`unreachable` terminators; **memory** (`alloca`/`load`/`store`/`getelementptr`,
//! incl. the second result-less shape `store` and opaque-pointer element types); and **multi-block**
//! CFGs with LLVM's implicit slot numbering (so an unlabeled entry and the phi/branch refs into it
//! resolve to the same `Name`s the bitcode reader assigns); and **module-level global variables**
//! (`@g = … global|constant <ty> <init>`, incl. `[N x T]` array types, array/byte-string/`zeroinitializer`
//! initializers, and `@g` operand references resolved to `GlobalReference`s via a `@name → pointee-type`
//! symbol table); and **named struct types** (`%s = type { … }` definitions + `%s`/`{…}` references in
//! `type_()`, so struct GEP/field access resolves); and **float constants** (`float`/`double` literals
//! in decimal or `0x` hex-image form, decoded to the exact bits the bitcode reader carries). Top-level
//! cruft the on-ramp ignores (target/datalayout lines, attribute groups, module-level metadata,
//! `declare`s) is skipped; the `switch` terminator's constant→label jump table is parsed too. **SIMD
//! vectors** are covered: `<N x T>` types, `<T …>` vector constants, `extractelement`/`insertelement`/
//! `shufflevector` (the mask canonicalized to a `Constant::Vector` of `i32` indices, as the bitcode
//! reader does), and vector binops/reductions; and **constant-expressions** (`getelementptr`/
//! `ptrtoint`/`bitcast`/`add`/… folded inside a constant, e.g. a global initialized to an offset into
//! another global); and **C++ exception handling** (`invoke`/`resume` terminators, `landingpad`,
//! `extractvalue`/`insertvalue` — the `personality` clause is skipped); and **atomics**
//! (`atomicrmw`/`cmpxchg`/`fence` + the `load atomic`/`store atomic` variants — syncscope + memory
//! ordering). Not yet handled (the growth frontier): scalable-vector-specific ops, `blockaddress`/
//! `indirectbr`, and literal-aggregate constants. Anything unhandled is a clean [`ParseError`]
//! (fail-closed, re-verified downstream — §2a), never a
//! miscompile.

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
    /// `@name` → its *pointee* type: a global variable's content type, or a function's type. A
    /// `@g`/`@f` operand tokenizes without its type (opaque pointers), but the `GlobalReference` the
    /// bitcode reader builds carries this pointee type, and the translator reads it (e.g. as a GEP's
    /// base pointee). clang emits globals + functions before their uses, so populating this as we parse
    /// resolves every reference; a forward/unknown reference is a clean error (fail-closed).
    symbols: std::collections::BTreeMap<String, TypeRef>,
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
            symbols: std::collections::BTreeMap::new(),
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
                // `@name = [linkage/attrs] global|constant <ty> [<init>] [, align N]`
                Some(Token::Global(_)) if self.at_global_var() => self.global_def()?,
                // `%name = type { … }` — a named struct type definition.
                Some(Token::Local(_)) if self.at_type_def() => self.type_def()?,
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
                        if w == "define"
                            || w == "declare"
                            || w == "attributes"
                            || w == "target"
                            || w == "source_filename" =>
                    {
                        return
                    }
                    // A depth-0 `@name`/`%name` begins the next top-level item (a global/alias/function,
                    // or a `%name = type …` definition). Stop before it so an unbraced line
                    // (`target … = "…"`) doesn't swallow the items that follow it. (We've already
                    // advanced ≥1 token this iteration, so this can't spin.)
                    Some(Token::Global(_)) | Some(Token::Local(_)) => return,
                    _ => {}
                }
            }
        }
    }

    // ---- named struct types --------------------------------------------------------------------

    /// Is the cursor at a named type definition (`%name = type …`)?
    fn at_type_def(&self) -> bool {
        matches!(self.toks.get(self.pos), Some(Token::Local(_)))
            && self.toks.get(self.pos + 1) == Some(&Token::Equals)
            && matches!(self.toks.get(self.pos + 2), Some(Token::Word(w)) if w == "type")
    }

    /// `%name = type { … }` | `%name = type opaque` — register the definition so `type_()` references
    /// (`%name`) and the translator's `named_struct_def` lookups resolve, matching the bitcode reader's
    /// `all_struct_names` registration.
    fn type_def(&mut self) -> PResult<()> {
        let name = match self.bump() {
            Some(Token::Local(s)) => s,
            other => return self.err(format!("expected a named type %name, found {other:?}")),
        };
        self.expect(&Token::Equals)?;
        self.expect_word("type")?;
        let def = if self.eat_word("opaque") {
            NamedStructDef::Opaque
        } else {
            NamedStructDef::Defined(self.type_()?)
        };
        self.module.types.add_named_struct_def(name, def);
        Ok(())
    }

    /// `[ < ] { <ty>, … } [ > ]` — a literal struct type body (`is_packed` set for the `<{…}>` form).
    fn struct_type(&mut self, is_packed: bool) -> PResult<TypeRef> {
        self.expect(&Token::LBrace)?;
        let mut element_types = Vec::new();
        if self.peek() != Some(&Token::RBrace) {
            loop {
                element_types.push(self.type_()?);
                if !self.eat(&Token::Comma) {
                    break;
                }
            }
        }
        self.expect(&Token::RBrace)?;
        Ok(self.module.types.struct_of(element_types, is_packed))
    }

    // ---- global variables ----------------------------------------------------------------------

    /// Is the cursor at a global *variable* definition (`@g = … global|constant …`)? Distinguishes it
    /// from an alias/ifunc (also `@g = …`) by scanning the linkage/attribute barewords for the
    /// `global`/`constant` keyword, skipping an `addrspace(N)` group.
    fn at_global_var(&self) -> bool {
        let mut i = self.pos;
        if !matches!(self.toks.get(i), Some(Token::Global(_))) {
            return false;
        }
        i += 1;
        if self.toks.get(i) != Some(&Token::Equals) {
            return false;
        }
        i += 1;
        let mut guard = 0;
        while guard < 32 {
            guard += 1;
            match self.toks.get(i) {
                Some(Token::Word(w)) if w == "global" || w == "constant" => return true,
                Some(Token::Word(w)) if w == "alias" || w == "ifunc" => return false,
                Some(Token::Word(_)) => i += 1,
                Some(Token::LParen) => {
                    // `addrspace(N)` — skip the balanced group.
                    let mut depth = 0usize;
                    loop {
                        match self.toks.get(i) {
                            Some(Token::LParen) => {
                                depth += 1;
                                i += 1;
                            }
                            Some(Token::RParen) => {
                                depth -= 1;
                                i += 1;
                                if depth == 0 {
                                    break;
                                }
                            }
                            None => return false,
                            _ => i += 1,
                        }
                    }
                }
                _ => return false,
            }
        }
        false
    }

    /// `@name = [linkage/visibility/attrs] [addrspace(N)] global|constant <ty> [<init>] [, align N]
    /// [, …]`. The `GlobalVariable.ty` is the *pointer* type (as the bitcode reader records
    /// `LLVMTypeOf`); the written `<ty>` is the content type — kept in `symbols` (for `@name` operands)
    /// and used to parse the initializer. Pushes into `module.global_vars`.
    fn global_def(&mut self) -> PResult<()> {
        let name = match self.bump() {
            Some(Token::Global(s)) => s,
            other => return self.err(format!("expected a global @name, found {other:?}")),
        };
        self.expect(&Token::Equals)?;
        let mut addr_space: AddrSpace = 0;
        let is_constant = loop {
            match self.peek() {
                Some(Token::Word(w)) if w == "global" => {
                    self.pos += 1;
                    break false;
                }
                Some(Token::Word(w)) if w == "constant" => {
                    self.pos += 1;
                    break true;
                }
                Some(Token::Word(w)) if w == "addrspace" => {
                    self.pos += 1;
                    if self.peek() == Some(&Token::LParen) {
                        self.pos += 1;
                        addr_space = self.int_lit_u32()?;
                        self.expect(&Token::RParen)?;
                    }
                }
                Some(Token::Word(_)) => self.pos += 1, // linkage / visibility / unnamed_addr / …
                other => return self.err(format!("expected `global`/`constant`, found {other:?}")),
            }
        };
        let content_ty = self.type_()?;
        self.symbols.insert(name.clone(), content_ty.clone());
        // The initializer (absent for an `external` declaration — then the next token is `,`/a new item).
        let initializer = if self.at_constant_start() {
            Some(self.constant(&content_ty)?)
        } else {
            None
        };
        let mut alignment = 0u32;
        while self.peek() == Some(&Token::Comma) {
            self.pos += 1; // `,`
            if self.eat_word("align") {
                alignment = self.int_lit_u32()?;
            } else {
                // section/comdat/metadata — skip this trailing clause to the next top-level item.
                self.skip_to_toplevel_boundary();
                break;
            }
        }
        self.module.global_vars.push(GlobalVariable {
            name: name_from_local(&name),
            ty: self.module.types.pointer(addr_space),
            initializer,
            is_constant,
            alignment,
        });
        Ok(())
    }

    /// Does the cursor begin a constant (an initializer value), as opposed to a `,`/end-of-item?
    fn at_constant_start(&self) -> bool {
        match self.peek() {
            Some(Token::Int(_))
            | Some(Token::Float(_))
            | Some(Token::Str(_))
            | Some(Token::LBracket)
            | Some(Token::LBrace)
            | Some(Token::Lt)
            | Some(Token::Global(_)) => true,
            Some(Token::Word(w)) => {
                matches!(
                    w.as_str(),
                    "true"
                        | "false"
                        | "zeroinitializer"
                        | "null"
                        | "undef"
                        | "poison"
                        // constant-expression openers
                        | "getelementptr"
                        | "trunc"
                        | "zext"
                        | "sext"
                        | "ptrtoint"
                        | "inttoptr"
                        | "bitcast"
                        | "addrspacecast"
                        | "add"
                        | "sub"
                        | "mul"
                ) || (w == "c" && matches!(self.peek2(), Some(Token::Str(_))))
            }
            _ => false,
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
        // Record `@name` → its function type, so a later `@name` operand (an address-of, not a direct
        // call) resolves to a `GlobalReference` carrying that pointee type.
        self.symbols.insert(
            name.clone(),
            TypeRef::new(Type::FuncType {
                result_type: return_type.clone(),
                param_types: parameters.iter().map(|p| p.ty.clone()).collect(),
                is_var_arg,
            }),
        );
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
            "nonnull",
            "noalias",
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
            let term = self.terminator(&mut next_unnamed)?;
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

    /// Is the cursor at a terminator? Looks past an optional `%result =` (an `invoke`/`callbr` produces a
    /// value yet ends the block), so `basic_blocks` routes it to `terminator()` rather than `instruction()`.
    fn at_terminator(&self) -> bool {
        let mut i = self.pos;
        if matches!(self.toks.get(i), Some(Token::Local(_)))
            && self.toks.get(i + 1) == Some(&Token::Equals)
        {
            i += 2;
        }
        matches!(self.toks.get(i), Some(Token::Word(w))
            if matches!(w.as_str(), "ret" | "br" | "switch" | "indirectbr" | "unreachable" | "invoke" | "resume" | "callbr"))
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
        // `store`/`fence` are the other result-less instructions (no `%dest =`).
        if matches!(self.peek(), Some(Token::Word(w)) if w == "store") {
            return self.store_inst().map(Instruction::Store);
        }
        if matches!(self.peek(), Some(Token::Word(w)) if w == "fence") {
            return self.fence_inst().map(Instruction::Fence);
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
            "alloca" => Instruction::Alloca(self.alloca_inst(dest)?),
            "load" => Instruction::Load(self.load_inst(dest)?),
            "getelementptr" => Instruction::GetElementPtr(self.gep_inst(dest)?),
            "extractelement" => Instruction::ExtractElement(self.extractelement_inst(dest)?),
            "insertelement" => Instruction::InsertElement(self.insertelement_inst(dest)?),
            "shufflevector" => Instruction::ShuffleVector(self.shufflevector_inst(dest)?),
            "extractvalue" => Instruction::ExtractValue(self.extractvalue_inst(dest)?),
            "insertvalue" => Instruction::InsertValue(self.insertvalue_inst(dest)?),
            "landingpad" => Instruction::LandingPad(self.landingpad_inst(dest)?),
            "atomicrmw" => Instruction::AtomicRMW(self.atomicrmw_inst(dest)?),
            "cmpxchg" => Instruction::CmpXchg(self.cmpxchg_inst(dest)?),
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

    fn terminator(&mut self, next_unnamed: &mut usize) -> PResult<Terminator> {
        // An `invoke` (value-producing terminator) carries a `%result =` prefix.
        let result = if matches!(
            (self.peek(), self.peek2()),
            (Some(Token::Local(_)), Some(Token::Equals))
        ) {
            Some(self.instr_dest(next_unnamed)?)
        } else {
            None
        };
        let op = match self.peek() {
            Some(Token::Word(w)) => w.clone(),
            other => return self.err(format!("expected a terminator, found {other:?}")),
        };
        let t = match op.as_str() {
            "unreachable" => {
                self.pos += 1;
                Terminator::Unreachable(Unreachable { debugloc: None })
            }
            // `[%r =] invoke [attrs] <retty>|<fnty> <callee>(<args>) to label %ok unwind label %lpad`.
            "invoke" => {
                self.pos += 1; // `invoke`
                let (function, function_ty, arguments) = self.call_signature()?;
                self.expect_word("to")?;
                self.expect_word("label")?;
                let return_label = self.label_name()?;
                self.expect_word("unwind")?;
                self.expect_word("label")?;
                let exception_label = self.label_name()?;
                // A `void` invoke has no `%r =`; it still occupies a value slot.
                let result = result.unwrap_or_else(|| {
                    let n = *next_unnamed;
                    *next_unnamed += 1;
                    Name::Number(n)
                });
                Terminator::Invoke(Invoke {
                    function,
                    function_ty,
                    arguments,
                    result,
                    return_label,
                    exception_label,
                    debugloc: None,
                })
            }
            // `resume <ty> <val>` — re-raise the in-flight exception.
            "resume" => {
                self.pos += 1;
                let ty = self.type_()?;
                let operand = self.value_as_operand(&ty)?;
                Terminator::Resume(Resume {
                    operand,
                    debugloc: None,
                })
            }
            // `switch <ty> <v>, label %default [ <cty> <c>, label %l … ]` — the case entries inside the
            // brackets are whitespace-separated (no commas between them).
            "switch" => {
                self.pos += 1; // `switch`
                let ty = self.type_()?;
                let operand = self.value_as_operand(&ty)?;
                self.expect(&Token::Comma)?;
                self.expect_word("label")?;
                let default_dest = self.label_name()?;
                self.expect(&Token::LBracket)?;
                let mut dests = Vec::new();
                while self.peek() != Some(&Token::RBracket) && self.peek().is_some() {
                    let cty = self.type_()?;
                    let case = self.constant(&cty)?;
                    self.expect(&Token::Comma)?;
                    self.expect_word("label")?;
                    dests.push((case, self.label_name()?));
                }
                self.expect(&Token::RBracket)?;
                Terminator::Switch(Switch {
                    operand,
                    dests,
                    default_dest,
                    debugloc: None,
                })
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

    /// Parse a value of (already-parsed) type `ty` into an [`Operand`]: a `%local`, or (everything else)
    /// a constant — delegated to [`Self::constant`], so int/float/bool/`poison`/`undef`/`zeroinitializer`/
    /// `null`/`@g`-ref/aggregate/vector all reach operand position uniformly.
    fn value_as_operand(&mut self, ty: &TypeRef) -> PResult<Operand> {
        if let Some(Token::Local(s)) = self.peek() {
            let name = name_from_local(s);
            self.pos += 1;
            return Ok(Operand::LocalOperand {
                name,
                ty: ty.clone(),
            });
        }
        Ok(Operand::ConstantOperand(self.constant(ty)?))
    }

    /// Decode a float literal `s` of floating type `ty` into a [`Float`]. LLVM prints these as decimal
    /// (`5.000000e-01`) or a `0x…` hex bit image; matching the bitcode reader (which reads the value as a
    /// `double` via `LLVMConstRealGetDouble`, casting to `f32` for `float`), we decode to an `f64` and
    /// cast. `half`/`bfloat`/`fp128`/`x86_fp80`/`ppc_fp128` are payload-free variants in the AST.
    fn float_lit(&self, ty: &TypeRef, s: &str) -> PResult<Float> {
        let fpt = match ty.as_ref() {
            Type::FPType(fpt) => *fpt,
            _ => return self.err("float literal with non-floating type"),
        };
        Ok(match fpt {
            FPType::Half => Float::Half,
            FPType::BFloat => Float::BFloat,
            FPType::Single => Float::Single(
                parse_fp_double(s)
                    .ok_or_else(|| ParseError::new(self.pos, format!("bad float literal `{s}`")))?
                    as f32,
            ),
            FPType::Double => Float::Double(
                parse_fp_double(s)
                    .ok_or_else(|| ParseError::new(self.pos, format!("bad float literal `{s}`")))?,
            ),
            FPType::FP128 => Float::Quadruple,
            FPType::X86_FP80 => Float::X86_FP80,
            FPType::PPC_FP128 => Float::PPC_FP128,
        })
    }

    /// A `@name` global/function reference → a `GlobalReference` constant (pointee type from `symbols`).
    fn global_ref(&mut self) -> PResult<ConstantRef> {
        let s = match self.bump() {
            Some(Token::Global(s)) => s,
            other => return self.err(format!("expected @global, found {other:?}")),
        };
        let ty = self.symbols.get(&s).cloned().ok_or_else(|| {
            ParseError::new(self.pos, format!("reference to undeclared global `@{s}`"))
        })?;
        Ok(ConstantRef::new(Constant::GlobalReference {
            name: name_from_local(&s),
            ty,
        }))
    }

    /// A constant-expression `getelementptr [inbounds] ( <srcty>, ptr <addr>, <ity> <idx>, … )`. The
    /// source element type is parsed but dropped (`ConstGetElementPtr` doesn't carry it, matching the
    /// bitcode reader's shim).
    fn const_gep(&mut self) -> PResult<Constant> {
        self.pos += 1; // `getelementptr`
        let in_bounds = self.eat_word("inbounds");
        while matches!(self.peek(), Some(Token::Word(w)) if matches!(w.as_str(), "nuw" | "nusw")) {
            self.pos += 1;
        }
        self.expect(&Token::LParen)?;
        let _src = self.type_()?;
        self.expect(&Token::Comma)?;
        let addr_ty = self.type_()?;
        let address = self.constant(&addr_ty)?;
        let mut indices = Vec::new();
        while self.eat(&Token::Comma) {
            let ity = self.type_()?;
            indices.push(self.constant(&ity)?);
        }
        self.expect(&Token::RParen)?;
        Ok(Constant::GetElementPtr(ConstGetElementPtr {
            address,
            indices,
            in_bounds,
        }))
    }

    /// A constant-expression conversion `<op> ( <srcty> <const> to <dstty> )`.
    fn const_conv(&mut self) -> PResult<ConstUnaryOp> {
        self.pos += 1; // opcode
        self.expect(&Token::LParen)?;
        let srcty = self.type_()?;
        let operand = self.constant(&srcty)?;
        self.expect_word("to")?;
        let to_type = self.type_()?;
        self.expect(&Token::RParen)?;
        Ok(ConstUnaryOp { operand, to_type })
    }

    /// A constant-expression binary op `<op> ( <ty> <c0>, <ty> <c1> )`.
    fn const_binop(&mut self) -> PResult<ConstBinaryOp> {
        self.pos += 1; // opcode
        while matches!(self.peek(), Some(Token::Word(w)) if matches!(w.as_str(), "nuw" | "nsw")) {
            self.pos += 1;
        }
        self.expect(&Token::LParen)?;
        let t0 = self.type_()?;
        let operand0 = self.constant(&t0)?;
        self.expect(&Token::Comma)?;
        let t1 = self.type_()?;
        let operand1 = self.constant(&t1)?;
        self.expect(&Token::RParen)?;
        Ok(ConstBinaryOp { operand0, operand1 })
    }

    /// Parse a constant of (already-parsed) type `ty` — global initializers and other constant
    /// positions. Covers int/`true`/`false`, `zeroinitializer`/`null`/`undef`/`poison`, `@g` references,
    /// `c"…"` byte strings, and `[ <ty> <c>, … ]` arrays. (Float constants are a later slice.)
    fn constant(&mut self, ty: &TypeRef) -> PResult<ConstantRef> {
        let c = match self.peek() {
            Some(Token::Int(s)) => {
                let s = s.clone();
                let bits = match ty.as_ref() {
                    Type::IntegerType { bits } => *bits,
                    _ => return self.err("integer constant with non-integer type"),
                };
                self.pos += 1;
                let value = parse_int_literal(&s, bits).ok_or_else(|| {
                    ParseError::new(self.pos, format!("bad integer literal `{s}`"))
                })?;
                Constant::Int { bits, value }
            }
            Some(Token::Word(w)) if w == "true" || w == "false" => {
                let value = (w == "true") as u128;
                self.pos += 1;
                Constant::Int { bits: 1, value }
            }
            Some(Token::Float(s)) => {
                let s = s.clone();
                let f = self.float_lit(ty, &s)?;
                self.pos += 1;
                Constant::Float(f)
            }
            Some(Token::Word(w)) if w == "zeroinitializer" => {
                self.pos += 1;
                Constant::AggregateZero(ty.clone())
            }
            Some(Token::Word(w)) if w == "null" => {
                self.pos += 1;
                Constant::Null(ty.clone())
            }
            Some(Token::Word(w)) if w == "undef" => {
                self.pos += 1;
                Constant::Undef(ty.clone())
            }
            Some(Token::Word(w)) if w == "poison" => {
                self.pos += 1;
                Constant::Poison(ty.clone())
            }
            // Constant-expressions: `<op> ( … )` folded at link time (a global initialized to an offset
            // into another global, a function-pointer table entry cast, etc.).
            Some(Token::Word(w)) if w == "getelementptr" => self.const_gep()?,
            Some(Token::Word(w))
                if matches!(
                    w.as_str(),
                    "trunc"
                        | "zext"
                        | "sext"
                        | "ptrtoint"
                        | "inttoptr"
                        | "bitcast"
                        | "addrspacecast"
                ) =>
            {
                let op = w.clone();
                let u = self.const_conv()?;
                match op.as_str() {
                    "trunc" => Constant::Trunc(u),
                    "zext" => Constant::ZExt(u),
                    "sext" => Constant::SExt(u),
                    "ptrtoint" => Constant::PtrToInt(u),
                    "inttoptr" => Constant::IntToPtr(u),
                    "bitcast" => Constant::BitCast(u),
                    _ => Constant::AddrSpaceCast(u),
                }
            }
            Some(Token::Word(w)) if matches!(w.as_str(), "add" | "sub" | "mul") => {
                let op = w.clone();
                let b = self.const_binop()?;
                match op.as_str() {
                    "add" => Constant::Add(b),
                    "sub" => Constant::Sub(b),
                    _ => Constant::Mul(b),
                }
            }
            Some(Token::Global(_)) => return self.global_ref(),
            // `c"…"` — a byte string, i.e. an array of `i8` constants (escapes decoded to raw bytes).
            // The `c` prefix lexes as its own `Word` before the `Str` body.
            Some(Token::Word(w)) if w == "c" && matches!(self.peek2(), Some(Token::Str(_))) => {
                self.pos += 1; // `c`
                let s = match self.bump() {
                    Some(Token::Str(s)) => s,
                    _ => unreachable!("peek2 checked Str"),
                };
                let bytes = super::lex::unescape(&s);
                let element_type = self.module.types.int(8);
                let elements = bytes
                    .into_iter()
                    .map(|b| {
                        ConstantRef::new(Constant::Int {
                            bits: 8,
                            value: b as u128,
                        })
                    })
                    .collect();
                Constant::Array {
                    element_type,
                    elements,
                }
            }
            // `[ <ty> <c>, … ]` — an array constant; the element type comes from the array type.
            Some(Token::LBracket) => {
                let element_type = match ty.as_ref() {
                    Type::ArrayType { element_type, .. } => element_type.clone(),
                    _ => return self.err("array constant with non-array type"),
                };
                self.pos += 1; // `[`
                let mut elements = Vec::new();
                if self.peek() != Some(&Token::RBracket) {
                    loop {
                        let ety = self.type_()?;
                        elements.push(self.constant(&ety)?);
                        if !self.eat(&Token::Comma) {
                            break;
                        }
                    }
                }
                self.expect(&Token::RBracket)?;
                Constant::Array {
                    element_type,
                    elements,
                }
            }
            // `< <ty> <c>, … >` — a vector constant.
            Some(Token::Lt) => {
                self.pos += 1; // `<`
                let mut elements = Vec::new();
                if self.peek() != Some(&Token::Gt) {
                    loop {
                        let ety = self.type_()?;
                        elements.push(self.constant(&ety)?);
                        if !self.eat(&Token::Comma) {
                            break;
                        }
                    }
                }
                self.expect(&Token::Gt)?;
                Constant::Vector(elements)
            }
            other => return self.err(format!("constant not yet supported: {other:?}")),
        };
        Ok(ConstantRef::new(c))
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

    // ---- vector element ops --------------------------------------------------------------------

    /// `extractelement <vty> <vec>, <ity> <idx>`.
    fn extractelement_inst(&mut self, dest: Name) -> PResult<ExtractElement> {
        self.pos += 1; // `extractelement`
        let vty = self.type_()?;
        let vector = self.value_as_operand(&vty)?;
        self.expect(&Token::Comma)?;
        let ity = self.type_()?;
        let index = self.value_as_operand(&ity)?;
        Ok(ExtractElement {
            vector,
            index,
            dest,
            debugloc: None,
        })
    }

    /// `insertelement <vty> <vec>, <ety> <elt>, <ity> <idx>`.
    fn insertelement_inst(&mut self, dest: Name) -> PResult<InsertElement> {
        self.pos += 1; // `insertelement`
        let vty = self.type_()?;
        let vector = self.value_as_operand(&vty)?;
        self.expect(&Token::Comma)?;
        let ety = self.type_()?;
        let element = self.value_as_operand(&ety)?;
        self.expect(&Token::Comma)?;
        let ity = self.type_()?;
        let index = self.value_as_operand(&ity)?;
        Ok(InsertElement {
            vector,
            element,
            index,
            dest,
            debugloc: None,
        })
    }

    /// `shufflevector <ty> <v0>, <ty> <v1>, <mty> <mask>` — the mask is a constant (index vector or
    /// `zeroinitializer`).
    fn shufflevector_inst(&mut self, dest: Name) -> PResult<ShuffleVector> {
        self.pos += 1; // `shufflevector`
        let t0 = self.type_()?;
        let operand0 = self.value_as_operand(&t0)?;
        self.expect(&Token::Comma)?;
        let t1 = self.type_()?;
        let operand1 = self.value_as_operand(&t1)?;
        self.expect(&Token::Comma)?;
        let mty = self.type_()?;
        let mask = self.shuffle_mask(&mty)?;
        Ok(ShuffleVector {
            operand0,
            operand1,
            dest,
            mask,
            debugloc: None,
        })
    }

    /// The `shufflevector` mask. The bitcode reader canonicalizes it (via `LLVMGetMaskValue`) to a
    /// `Constant::Vector` of `i32` indices — an `undef`/`poison` lane becoming `Constant::Undef(i32)` —
    /// regardless of whether the text writes `zeroinitializer`, `poison`, or an explicit `<i32 …>`
    /// vector. We reproduce that exact form so a splat's mask reaches parity.
    fn shuffle_mask(&mut self, mty: &TypeRef) -> PResult<ConstantRef> {
        let num = match mty.as_ref() {
            Type::VectorType { num_elements, .. } => *num_elements,
            _ => return self.err("shufflevector mask is not a vector type"),
        };
        let i32ty = self.module.types.int(32);
        let zero = || ConstantRef::new(Constant::Int { bits: 32, value: 0 });
        let undef = || ConstantRef::new(Constant::Undef(i32ty.clone()));
        let elements: Vec<ConstantRef> = match self.peek() {
            Some(Token::Word(w)) if w == "zeroinitializer" => {
                self.pos += 1;
                (0..num).map(|_| zero()).collect()
            }
            Some(Token::Word(w)) if w == "undef" || w == "poison" => {
                self.pos += 1;
                (0..num).map(|_| undef()).collect()
            }
            Some(Token::Lt) => {
                self.pos += 1; // `<`
                let mut v = Vec::new();
                if self.peek() != Some(&Token::Gt) {
                    loop {
                        let _ety = self.type_()?; // `i32`
                        match self.peek() {
                            Some(Token::Word(w)) if w == "undef" || w == "poison" => {
                                self.pos += 1;
                                v.push(undef());
                            }
                            Some(Token::Int(s)) => {
                                let s = s.clone();
                                self.pos += 1;
                                let value = parse_int_literal(&s, 32).ok_or_else(|| {
                                    ParseError::new(self.pos, format!("bad mask index `{s}`"))
                                })?;
                                v.push(ConstantRef::new(Constant::Int { bits: 32, value }));
                            }
                            other => {
                                return self.err(format!("bad shuffle-mask element {other:?}"))
                            }
                        }
                        if !self.eat(&Token::Comma) {
                            break;
                        }
                    }
                }
                self.expect(&Token::Gt)?;
                v
            }
            other => return self.err(format!("unsupported shuffle mask {other:?}")),
        };
        Ok(ConstantRef::new(Constant::Vector(elements)))
    }

    // ---- aggregate value ops + landing pad -----------------------------------------------------

    /// `extractvalue <ty> <agg>, <idx>[, <idx>]…` — the indices are plain integer literals.
    fn extractvalue_inst(&mut self, dest: Name) -> PResult<ExtractValue> {
        self.pos += 1; // `extractvalue`
        let aty = self.type_()?;
        let aggregate = self.value_as_operand(&aty)?;
        let indices = self.value_index_list()?;
        Ok(ExtractValue {
            aggregate,
            indices,
            dest,
            debugloc: None,
        })
    }

    /// `insertvalue <ty> <agg>, <ety> <elt>, <idx>[, <idx>]…`.
    fn insertvalue_inst(&mut self, dest: Name) -> PResult<InsertValue> {
        self.pos += 1; // `insertvalue`
        let aty = self.type_()?;
        let aggregate = self.value_as_operand(&aty)?;
        self.expect(&Token::Comma)?;
        let ety = self.type_()?;
        let element = self.value_as_operand(&ety)?;
        let indices = self.value_index_list()?;
        Ok(InsertValue {
            aggregate,
            element,
            indices,
            dest,
            debugloc: None,
        })
    }

    /// The trailing `, <idx>, <idx>, …` integer index list of `extractvalue`/`insertvalue` (a `, !meta`
    /// tail is left for [`Self::skip_trailing_metadata`]).
    fn value_index_list(&mut self) -> PResult<Vec<u32>> {
        let mut indices = Vec::new();
        while self.peek() == Some(&Token::Comma) && !matches!(self.peek2(), Some(Token::Meta(_))) {
            self.pos += 1; // `,`
            indices.push(self.int_lit_u32()?);
        }
        self.skip_trailing_metadata();
        Ok(indices)
    }

    /// `landingpad <ty> [cleanup] [catch <ty> <c>]… [filter <ty> <c>]…`. Each `catch`/`filter` becomes an
    /// (opaque) clause marker — the translator reads only the clause count + the `cleanup` flag, matching
    /// the bitcode reader's `LandingPadClause {}` mapping.
    fn landingpad_inst(&mut self, dest: Name) -> PResult<LandingPad> {
        self.pos += 1; // `landingpad`
        let result_type = self.type_()?;
        let mut cleanup = false;
        let mut clauses = Vec::new();
        loop {
            match self.peek() {
                Some(Token::Word(w)) if w == "cleanup" => {
                    self.pos += 1;
                    cleanup = true;
                }
                Some(Token::Word(w)) if w == "catch" || w == "filter" => {
                    self.pos += 1;
                    let cty = self.type_()?;
                    let _clause = self.constant(&cty)?;
                    clauses.push(LandingPadClause {});
                }
                _ => break,
            }
        }
        Ok(LandingPad {
            result_type,
            clauses,
            dest,
            cleanup,
            debugloc: None,
        })
    }

    // ---- atomics -------------------------------------------------------------------------------

    /// `atomicrmw [volatile] <op> ptr <addr>, <ty> <val> [syncscope("…")] <ordering> [, align N]`.
    fn atomicrmw_inst(&mut self, dest: Name) -> PResult<AtomicRMW> {
        self.pos += 1; // `atomicrmw`
        let volatile = self.eat_word("volatile");
        let operation = self.rmw_op()?;
        let addr_ty = self.type_()?;
        let address = self.value_as_operand(&addr_ty)?;
        self.expect(&Token::Comma)?;
        let vty = self.type_()?;
        let value = self.value_as_operand(&vty)?;
        let atomicity = self.atomicity()?;
        let _align = self.opt_align()?; // AtomicRMW carries no alignment field
        self.skip_trailing_metadata();
        Ok(AtomicRMW {
            operation,
            address,
            value,
            dest,
            volatile,
            atomicity,
            debugloc: None,
        })
    }

    /// `cmpxchg [weak] [volatile] ptr <addr>, <ty> <expected>, <ty> <replacement> [syncscope("…")]
    /// <success-ordering> <failure-ordering> [, align N]`.
    fn cmpxchg_inst(&mut self, dest: Name) -> PResult<CmpXchg> {
        self.pos += 1; // `cmpxchg`
        let weak = self.eat_word("weak");
        let volatile = self.eat_word("volatile");
        let addr_ty = self.type_()?;
        let address = self.value_as_operand(&addr_ty)?;
        self.expect(&Token::Comma)?;
        let ety = self.type_()?;
        let expected = self.value_as_operand(&ety)?;
        self.expect(&Token::Comma)?;
        let rty = self.type_()?;
        let replacement = self.value_as_operand(&rty)?;
        let atomicity = self.atomicity()?; // syncscope + success ordering
        let failure_memory_ordering = self.mem_ordering()?;
        let _align = self.opt_align()?;
        self.skip_trailing_metadata();
        Ok(CmpXchg {
            address,
            expected,
            replacement,
            dest,
            volatile,
            atomicity,
            failure_memory_ordering,
            weak,
            debugloc: None,
        })
    }

    /// `fence [syncscope("…")] <ordering>` — result-less.
    fn fence_inst(&mut self) -> PResult<Fence> {
        self.pos += 1; // `fence`
        let atomicity = self.atomicity()?;
        self.skip_trailing_metadata();
        Ok(Fence {
            atomicity,
            debugloc: None,
        })
    }

    /// An optional `syncscope("…")` followed by a memory ordering — the atomic annotation on
    /// `atomicrmw`/`cmpxchg`/`fence`/`load atomic`/`store atomic`. No `syncscope` ⇒ system scope.
    fn atomicity(&mut self) -> PResult<Atomicity> {
        let synch_scope = if self.eat_word("syncscope") {
            self.expect(&Token::LParen)?;
            let s = match self.bump() {
                Some(Token::Str(s)) => s,
                other => return self.err(format!("expected a syncscope string, found {other:?}")),
            };
            self.expect(&Token::RParen)?;
            if s == "singlethread" {
                SynchronizationScope::SingleThread
            } else {
                SynchronizationScope::System
            }
        } else {
            SynchronizationScope::System
        };
        let mem_ordering = self.mem_ordering()?;
        Ok(Atomicity {
            synch_scope,
            mem_ordering,
        })
    }

    fn mem_ordering(&mut self) -> PResult<MemoryOrdering> {
        let o = match self.peek() {
            Some(Token::Word(w)) => match w.as_str() {
                "unordered" => MemoryOrdering::Unordered,
                "monotonic" => MemoryOrdering::Monotonic,
                "acquire" => MemoryOrdering::Acquire,
                "release" => MemoryOrdering::Release,
                "acq_rel" => MemoryOrdering::AcquireRelease,
                "seq_cst" => MemoryOrdering::SequentiallyConsistent,
                other => return self.err(format!("unknown memory ordering `{other}`")),
            },
            other => return self.err(format!("expected a memory ordering, found {other:?}")),
        };
        self.pos += 1;
        Ok(o)
    }

    fn rmw_op(&mut self) -> PResult<RMWBinOp> {
        let op = match self.peek() {
            Some(Token::Word(w)) => match w.as_str() {
                "xchg" => RMWBinOp::Xchg,
                "add" => RMWBinOp::Add,
                "sub" => RMWBinOp::Sub,
                "and" => RMWBinOp::And,
                "nand" => RMWBinOp::Nand,
                "or" => RMWBinOp::Or,
                "xor" => RMWBinOp::Xor,
                "max" => RMWBinOp::Max,
                "min" => RMWBinOp::Min,
                "umax" => RMWBinOp::UMax,
                "umin" => RMWBinOp::UMin,
                "fadd" => RMWBinOp::FAdd,
                "fsub" => RMWBinOp::FSub,
                "fmax" => RMWBinOp::FMax,
                "fmin" => RMWBinOp::FMin,
                other => return self.err(format!("unknown atomicrmw operation `{other}`")),
            },
            other => return self.err(format!("expected an atomicrmw operation, found {other:?}")),
        };
        self.pos += 1;
        Ok(op)
    }

    // ---- memory --------------------------------------------------------------------------------

    /// `alloca [inalloca] <ty> [, <cty> <count>] [, align N] [, addrspace(N)]`. The array size defaults
    /// to `i32 1` — the operand the bitcode reader materializes when the count is implicit.
    fn alloca_inst(&mut self, dest: Name) -> PResult<Alloca> {
        self.pos += 1; // `alloca`
        self.eat_word("inalloca");
        let allocated_type = self.type_()?;
        let mut num_elements =
            Operand::ConstantOperand(ConstantRef::new(Constant::Int { bits: 32, value: 1 }));
        let mut alignment = 0u32;
        loop {
            if self.peek() != Some(&Token::Comma) || matches!(self.peek2(), Some(Token::Meta(_))) {
                break;
            }
            self.pos += 1; // `,`
            if self.eat_word("align") {
                alignment = self.int_lit_u32()?;
            } else if self.eat_word("addrspace") {
                self.skip_balanced_parens();
            } else {
                let cty = self.type_()?;
                num_elements = self.value_as_operand(&cty)?;
            }
        }
        self.skip_trailing_metadata();
        Ok(Alloca {
            allocated_type,
            num_elements,
            dest,
            alignment,
            debugloc: None,
        })
    }

    /// `load [atomic] [volatile] <ty>, ptr <addr> [syncscope("…")] <ordering>? [, align N] [, !meta]`.
    fn load_inst(&mut self, dest: Name) -> PResult<Load> {
        self.pos += 1; // `load`
        let atomic = self.eat_word("atomic");
        let volatile = self.eat_word("volatile");
        let loaded_ty = self.type_()?;
        self.expect(&Token::Comma)?;
        let addr_ty = self.type_()?; // `ptr`
        let address = self.value_as_operand(&addr_ty)?;
        // An atomic load carries `[syncscope] <ordering>` (before `, align`).
        let atomicity = if atomic {
            Some(self.atomicity()?)
        } else {
            None
        };
        let alignment = self.opt_align()?;
        self.skip_trailing_metadata();
        Ok(Load {
            address,
            dest,
            loaded_ty,
            volatile,
            atomicity,
            alignment,
            debugloc: None,
        })
    }

    /// `store [atomic] [volatile] <ty> <val>, ptr <addr> [syncscope("…")] <ordering>? [, align N]` —
    /// result-less.
    fn store_inst(&mut self) -> PResult<Store> {
        self.pos += 1; // `store`
        let atomic = self.eat_word("atomic");
        let volatile = self.eat_word("volatile");
        let vty = self.type_()?;
        let value = self.value_as_operand(&vty)?;
        self.expect(&Token::Comma)?;
        let addr_ty = self.type_()?; // `ptr`
        let address = self.value_as_operand(&addr_ty)?;
        let atomicity = if atomic {
            Some(self.atomicity()?)
        } else {
            None
        };
        let alignment = self.opt_align()?;
        self.skip_trailing_metadata();
        Ok(Store {
            address,
            value,
            volatile,
            atomicity,
            alignment,
            debugloc: None,
        })
    }

    /// `getelementptr [inbounds] [nuw|nusw] <srcty>, ptr <addr> [, <ity> <idx>]… [, !meta]`.
    fn gep_inst(&mut self, dest: Name) -> PResult<GetElementPtr> {
        self.pos += 1; // `getelementptr`
        let in_bounds = self.eat_word("inbounds");
        while matches!(self.peek(), Some(Token::Word(w)) if matches!(w.as_str(), "nuw" | "nusw")) {
            self.pos += 1;
        }
        let source_element_type = self.type_()?;
        self.expect(&Token::Comma)?;
        let addr_ty = self.type_()?; // `ptr`
        let address = self.value_as_operand(&addr_ty)?;
        let mut indices = Vec::new();
        loop {
            if self.peek() != Some(&Token::Comma) || matches!(self.peek2(), Some(Token::Meta(_))) {
                break;
            }
            self.pos += 1; // `,`
            let ity = self.type_()?;
            indices.push(self.value_as_operand(&ity)?);
        }
        self.skip_trailing_metadata();
        Ok(GetElementPtr {
            address,
            indices,
            dest,
            in_bounds,
            source_element_type,
            debugloc: None,
        })
    }

    /// `, align N` after a load/store — the alignment, or `0` if absent (a trailing `, !meta` is left
    /// for [`Self::skip_trailing_metadata`]).
    fn opt_align(&mut self) -> PResult<u32> {
        if self.peek() == Some(&Token::Comma)
            && matches!(self.peek2(), Some(Token::Word(w)) if w == "align")
        {
            self.pos += 2; // `,` `align`
            self.int_lit_u32()
        } else {
            Ok(0)
        }
    }

    /// Consume an integer literal as a `u32` (alignments, small counts).
    fn int_lit_u32(&mut self) -> PResult<u32> {
        match self.bump() {
            Some(Token::Int(s)) => s
                .parse::<u32>()
                .map_err(|_| ParseError::new(self.pos, format!("bad integer `{s}`"))),
            other => self.err(format!("expected an integer, found {other:?}")),
        }
    }

    /// Consume an integer literal as a `usize` (array/vector element counts).
    fn int_lit_usize(&mut self) -> PResult<usize> {
        match self.bump() {
            Some(Token::Int(s)) => s
                .parse::<usize>()
                .map_err(|_| ParseError::new(self.pos, format!("bad integer `{s}`"))),
            other => self.err(format!("expected an integer, found {other:?}")),
        }
    }

    /// Skip a balanced `( … )` group starting at the current `(`.
    fn skip_balanced_parens(&mut self) {
        if self.peek() != Some(&Token::LParen) {
            return;
        }
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
        let (function, function_ty, arguments) = self.call_signature()?;
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

    /// The shared `call`/`invoke` body *after* the opcode keyword: `[fmf] [cconv/ret-attrs]
    /// <retty>|<fnty> <callee>(<args>) [#N]` → the callee operand (a `GlobalReference` for `@f`, an
    /// opaque-ptr local for an indirect `%fp`), the reconstructed function type, and the arguments.
    fn call_signature(&mut self) -> PResult<(Either<InlineAssembly, Operand>, TypeRef, CallArgs)> {
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
        // Trailing attribute-group refs (`#4`).
        while matches!(self.peek(), Some(Token::Word(w)) if w.starts_with('#')) {
            self.pos += 1;
        }
        Ok((function, function_ty, arguments))
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
        // `[N x T]` array type.
        if self.peek() == Some(&Token::LBracket) {
            self.pos += 1;
            let num_elements = self.int_lit_usize()?;
            self.expect_word("x")?;
            let element_type = self.type_()?;
            self.expect(&Token::RBracket)?;
            return Ok(self.module.types.array_of(element_type, num_elements));
        }
        // `%name` — a reference to a named struct type (its definition is registered separately).
        if let Some(Token::Local(s)) = self.peek() {
            let name = s.clone();
            self.pos += 1;
            return Ok(self.module.types.named_struct(name));
        }
        // `{ <ty>, … }` — a literal struct type.
        if self.peek() == Some(&Token::LBrace) {
            return self.struct_type(false);
        }
        // `<[vscale x] N x T>` vector, or `<{ … }>` packed struct.
        if self.peek() == Some(&Token::Lt) {
            self.pos += 1;
            if self.peek() == Some(&Token::LBrace) {
                let t = self.struct_type(true)?;
                self.expect(&Token::Gt)?;
                return Ok(t);
            }
            let scalable = self.eat_word("vscale");
            if scalable {
                self.expect_word("x")?;
            }
            let num_elements = self.int_lit_usize()?;
            self.expect_word("x")?;
            let element_type = self.type_()?;
            self.expect(&Token::Gt)?;
            return Ok(self
                .module
                .types
                .vector_of(element_type, num_elements, scalable));
        }
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

/// A `float`/`double` literal token → its value as an `f64`. LLVM emits either decimal notation
/// (`5.000000e-01`, `-1.5e10`) or a `0x` + 16-hex-digit image of the **`double`** bit pattern (used even
/// for `float`, where the value must be exactly representable — the caller casts to `f32`). The wide-FP
/// hex prefixes (`0xK`/`0xL`/`0xM`/`0xH`/`0xR`) never reach here: those types are payload-free AST
/// variants. `None` if the text is neither form.
fn parse_fp_double(s: &str) -> Option<f64> {
    if let Some(hex) = s.strip_prefix("0x") {
        // Only the plain 16-hex-digit (double-image) form has an all-hex body.
        return u64::from_str_radix(hex, 16).ok().map(f64::from_bits);
    }
    s.parse::<f64>().ok()
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
